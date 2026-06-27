"""Yuan Tool · Panorama 节点

复刻自 ComfyUI-Panorama-Stickers，仅包含两个节点：
- YuanPanoramaPreview  : 交互式 ERP 全景预览（360°/180°），支持视频批次
- YuanPanoramaSeamPrep : 为接缝修复准备 ERP 图像

节点分类: "Yuan Tool/图像"
"""

import base64
import io
import logging
import math
import uuid
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)

# ComfyUI 内部模块（可选导入，便于在测试环境独立运行）
try:
    import folder_paths
except Exception:  # pragma: no cover
    folder_paths = None

try:
    import nodes as _comfy_nodes
except Exception:  # pragma: no cover
    _comfy_nodes = None

# 视频编码依赖（可选）。缺失时仅放弃 mp4 预览，不影响图像预览。
try:
    import av
except Exception:  # pragma: no cover
    av = None


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def _normalize_Coverage(value):
    text = str(value or "360").strip()
    return 180 if text in ("180", 180, 180.0) else 360


def _decode_data_url_image(data_url: str):
    """将前端截取的 base64 data URL 解码为 [1,H,W,C] 的 torch.Tensor（0..1）。"""
    text = str(data_url or "").strip()
    if not text:
        return None
    if "," in text and text.lower().startswith("data:image/"):
        text = text.split(",", 1)[1]
    try:
        payload = base64.b64decode(text, validate=False)
        with Image.open(io.BytesIO(payload)) as image:
            image = image.convert("RGB")
            array = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0).contiguous()
    except Exception:
        return None


def _finite_float(value, default=0.0):
    try:
        f = float(value)
        if not math.isfinite(f):
            return float(default)
        return f
    except Exception:
        return float(default)


def _audio_has_waveform(audio) -> bool:
    if audio is None or not hasattr(audio, "get"):
        return False
    try:
        return audio.get("waveform") is not None and int(audio.get("sample_rate") or 0) > 0
    except Exception:
        return False


def _resolve_temp_root() -> Path:
    if folder_paths is not None:
        try:
            return Path(folder_paths.get_temp_directory())
        except Exception:
            pass
    return Path(tempfile.gettempdir()) / "comfyui_yuan_panorama"


def _ensure_temp_root() -> Path:
    root = _resolve_temp_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _entry_from_path(path: Path, media_type: str) -> dict:
    root = _ensure_temp_root().resolve()
    target = Path(path).resolve()
    try:
        rel = target.relative_to(root)
        subfolder = str(rel.parent).replace("\\", "/")
    except Exception:
        subfolder = ""
    return {
        "filename": target.name,
        "subfolder": subfolder,
        "type": "temp",
        "storage": "temp",
        "format": media_type,
    }


def _first_image_tensor(image):
    """取批次第一帧，用于静态预览图。"""
    if image is None or not hasattr(image, "shape"):
        return image
    try:
        if len(image.shape) >= 4 and int(image.shape[0]) > 1:
            return image[:1]
    except Exception:
        return image
    return image


def _save_preview_images(images, key: str = "pano_input_images") -> dict:
    """复用 ComfyUI 内置 PreviewImage 将图像落盘，返回前端可用的 UI 条目。"""
    if _comfy_nodes is None or images is None:
        return {}
    try:
        res = _comfy_nodes.PreviewImage().save_images(images)
        if isinstance(res, dict) and "ui" in res and "images" in res["ui"]:
            return {key: res["ui"]["images"]}
    except Exception:
        log.exception("[YuanPanorama] 保存预览图失败")
    return {}


# --------------------------------------------------------------------------- #
# 视频编码（best-effort，依赖 PyAV）
# --------------------------------------------------------------------------- #
def _pad_to_even(frames: torch.Tensor):
    h = int(frames.shape[-3])
    w = int(frames.shape[-2])
    ph = h + (h % 2)
    pw = w + (w % 2)
    if ph == h and pw == w:
        return frames, h, w
    pad_h = ph - h
    pad_w = pw - w
    padded = F.pad(frames, (0, 0, 0, pad_w, 0, pad_h))
    return padded, ph, pw


def _extract_audio(audio, frame_count: int, frame_rate):
    """从 ComfyUI AUDIO 字典提取并对齐到目标帧数的波形。"""
    if not _audio_has_waveform(audio):
        return None, 0, "mono"
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate") or 0)
    if waveform is None or sample_rate <= 0:
        return None, 0, "mono"
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    waveform = waveform.detach().cpu().float().contiguous()
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    elif waveform.ndim == 3 and int(waveform.shape[0]) > 0:
        waveform = waveform[0]
    elif waveform.ndim != 2:
        return None, 0, "mono"

    target = max(0, int(round((float(frame_count) / float(frame_rate)) * sample_rate)))
    current = int(waveform.shape[-1])
    if current > target:
        waveform = waveform[..., :target]
    elif current < target:
        pad = torch.zeros((int(waveform.shape[0]), target - current), dtype=waveform.dtype)
        waveform = torch.cat([waveform, pad], dim=-1)
    waveform = waveform.contiguous()

    channels = int(waveform.shape[0])
    layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(channels)
    if layout is None:
        if channels >= 2:
            left = waveform[0::2].mean(dim=0, keepdim=True)
            right = waveform[1::2].mean(dim=0, keepdim=True)
            waveform = torch.cat([left, right], dim=0)
            layout = "stereo"
        else:
            layout = "mono"
    return waveform.numpy().astype(np.float32), sample_rate, layout


def _encode_frames_to_mp4(frames, fps: float, audio=None) -> Path:
    if av is None:
        raise RuntimeError("PyAV 不可用，无法编码 mp4 预览")
    fps_value = max(1e-3, float(fps or 24.0))
    output_path = _ensure_temp_root() / f"yuan_pano_{uuid.uuid4().hex[:12]}.mp4"
    frame_rate = Fraction(round(fps_value * 1000), 1000)

    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(np.asarray(frames))
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    frames = frames.detach().cpu().float()
    frames, height, width = _pad_to_even(frames)
    frame_count = int(frames.shape[0])

    waveform_np, sample_rate, layout = _extract_audio(audio, frame_count, float(frame_rate))

    last_error = None
    for codec in ("h264_nvenc", "h264"):
        container = av.open(str(output_path), mode="w")
        try:
            vs = container.add_stream(codec, rate=frame_rate)
            vs.width = int(width)
            vs.height = int(height)
            vs.pix_fmt = "yuv420p"
            if codec == "h264_nvenc":
                vs.options = {"preset": "p4"}
            astream = None
            if waveform_np is not None and sample_rate > 0:
                astream = container.add_stream("aac", rate=sample_rate, layout=layout)
            for i in range(frame_count):
                f = frames[i].numpy()
                if f.shape[-1] < 3:
                    f = np.repeat(f[..., :1], 3, axis=-1)
                elif f.shape[-1] > 3:
                    f = f[..., :3]
                fu8 = np.clip(f * 255.0, 0, 255).astype(np.uint8)
                vframe = av.VideoFrame.from_ndarray(fu8, format="rgb24")
                if codec == "h264_nvenc":
                    vframe = vframe.reformat(format="yuv420p")
                for pkt in vs.encode(vframe):
                    container.mux(pkt)
            for pkt in vs.encode():
                container.mux(pkt)
            if astream is not None:
                aframe = av.AudioFrame.from_ndarray(waveform_np, format="fltp", layout=layout)
                aframe.sample_rate = sample_rate
                aframe.pts = 0
                for pkt in astream.encode(aframe):
                    container.mux(pkt)
                for pkt in astream.encode(None):
                    container.mux(pkt)
            container.close()
            return output_path
        except Exception as ex:
            last_error = ex
            try:
                container.close()
            except Exception:
                pass
            try:
                output_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("未找到可用的视频编码器")


def _make_video_ui_payload(mp4_path: Path, fps: float, frame_count: int) -> dict:
    entry = _entry_from_path(Path(mp4_path), "video/mp4")
    return {
        "pano_videos": [entry],
        "pano_video_meta": [{
            "fps": float(fps or 24.0),
            "frames": int(frame_count),
            "duration": float(frame_count) / max(1e-6, float(fps or 24.0)),
            "has_audio": False,
        }],
    }


# --------------------------------------------------------------------------- #
# 节点 1：全景预览
# --------------------------------------------------------------------------- #
class YuanPanoramaPreview:
    """交互式预览 ERP 全景图（360°/180°），支持视频批次输入。

    后端职责：将输入 ERP 图像落盘为预览图（pano_input_images），
    若为视频批次则额外编码 mp4 预览（pano_videos / pano_video_meta）。
    前端通过 WebGL 读取这些 UI 条目并以球面投影交互渲染。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ERP_image": ("IMAGE",),
                "Coverage": (["360", "180"], {"default": "360"}),
                "output_current_view": ("BOOLEAN", {"default": True, "display_name": "全景模式"}),
                "view_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "display_name": "裁剪宽度"}),
                "view_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "display_name": "裁剪高度"}),
            },
            "optional": {
                "current_view_data": ("STRING", {"default": "", "multiline": False, "hidden": True, "display": "hidden"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出",)
    OUTPUT_TOOLTIPS = ("当前节点预览画面：开启全景模式输出完整 ERP 全景图；关闭则按 view_width/view_height 输出 3D 裁剪截图。",)
    FUNCTION = "execute"
    CATEGORY = "Yuan Tool/图像"
    OUTPUT_NODE = True
    DESCRIPTION = "交互式预览 ERP 全景图（360°/180°），支持视频批次输入；可输出当前 3D 裁剪画面。"

    def execute(self, ERP_image, Coverage="360", output_current_view=False, view_width=1024, view_height=512, current_view_data=""):
        ui_ret = {}
        warnings = []
        fps_value = 24.0
        # Coverage 由前端读取 widget 决定，这里仅确保合法
        _normalize_Coverage(Coverage)
        # output_current_view = True  → 全景模式（输出完整 ERP 全景图）
        # output_current_view = False → 裁剪模式（输出当前 3D 裁剪截图）
        panorama_mode = bool(output_current_view)
        # 裁剪分辨率：限制到合法范围
        view_w = max(64, min(8192, int(view_width or 1024)))
        view_h = max(64, min(8192, int(view_height or 512)))
        current_view_data = str(current_view_data or "")

        if ERP_image is not None:
            ui_ret.update(_save_preview_images(_first_image_tensor(ERP_image), key="pano_input_images"))

            frame_count = 1
            try:
                if hasattr(ERP_image, "ndim") and ERP_image.ndim == 4:
                    frame_count = int(ERP_image.shape[0])
            except Exception:
                frame_count = 1

            if frame_count > 1:
                try:
                    mp4_path = _encode_frames_to_mp4(ERP_image, fps_value)
                    payload = _make_video_ui_payload(mp4_path, fps_value, frame_count)
                    ui_ret.update(payload)
                except Exception as ex:
                    warnings.append(f"视频预览编码失败: {ex}")

        if warnings:
            ui_ret["pano_preview_warnings"] = warnings

        # 全景模式输出完整 ERP 全景图；裁剪模式输出前端截取的当前 3D 裁剪画面
        output_image = ERP_image if ERP_image is not None else torch.zeros(
            (1, 1, 1, 3), dtype=torch.float32,
        )
        if not panorama_mode:
            view_image = _decode_data_url_image(current_view_data)
            if view_image is not None:
                output_image = view_image
        return {"ui": ui_ret, "result": (output_image,)}


# --------------------------------------------------------------------------- #
# 节点 2：全景接缝
# --------------------------------------------------------------------------- #
class YuanPanoramaSeamPrep:
    """为接缝修复（seam-focused inpainting）准备 ERP 图像。

    输入 IMAGE 形状 [B,H,W,C]（0..1）。
    输出:
      - image        [B,H,W,C]  平移接缝到中心后的图像
      - mask         [B,H,W]    接缝带掩码
      - mask_blurred [B,H,W]    高斯模糊后的掩码

    seam_center_offset_px 为正时接缝带右移，为负时左移。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seam_width_px": ("INT", {"default": 64, "min": 1, "max": 2048, "step": 1}),
                "seam_center_offset_px": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "mask_blur_px": ("INT", {"default": 10, "min": 0, "max": 256, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "mask_blurred")
    FUNCTION = "execute"
    CATEGORY = "Yuan Tool/图像"
    DESCRIPTION = (
        "为接缝修复准备 ERP 图像。输入 IMAGE 形状 [B,H,W,C]，"
        "输出 image [B,H,W,C]、mask [B,H,W]、mask_blurred [B,H,W]。"
        "seam_center_offset_px 为正时接缝带右移，为负时左移。"
    )

    @staticmethod
    def _gaussian_kernel_1d(radius: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        radius = max(0, int(radius))
        if radius <= 0:
            return torch.ones((1,), dtype=dtype, device=device)
        sigma = max(0.5, float(radius) / 3.0)
        coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
        kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel = kernel / torch.clamp(kernel.sum(), min=torch.finfo(dtype).eps)
        return kernel

    @classmethod
    def _blur_mask(cls, mask: torch.Tensor, blur_px: int) -> torch.Tensor:
        radius = max(0, int(blur_px))
        if radius <= 0:
            return mask
        batch, height, width = mask.shape
        kernel = cls._gaussian_kernel_1d(radius, mask.dtype, mask.device)
        kernel_x = kernel.view(1, 1, 1, -1)
        kernel_y = kernel.view(1, 1, -1, 1)
        work = mask.contiguous().unsqueeze(1)
        work = F.pad(work, (radius, radius, 0, 0), mode="replicate")
        work = F.conv2d(work, kernel_x.expand(1, 1, 1, kernel.numel()), groups=1)
        work = F.pad(work, (0, 0, radius, radius), mode="replicate")
        work = F.conv2d(work, kernel_y.expand(1, 1, kernel.numel(), 1), groups=1)
        work = work.view(batch, height, width)
        return work.clamp(0.0, 1.0)

    def execute(self, image, seam_width_px=64, seam_center_offset_px=0, mask_blur_px=0):
        if image is None or not hasattr(image, "shape"):
            empty_img = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 1, 1), dtype=torch.float32)
            return (empty_img, empty_mask, empty_mask)

        img = image.contiguous().to(dtype=torch.float32)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.ndim != 4:
            raise ValueError("YuanPanoramaSeamPrep 期望输入 IMAGE 形状为 [B,H,W,C]。")

        batch, height, width, channels = img.shape
        if width < 1 or height < 1:
            empty_img = torch.zeros(
                (max(batch, 1), max(height, 1), max(width, 1), max(channels, 3)),
                dtype=img.dtype, device=img.device,
            )
            empty_mask = torch.zeros(
                (max(batch, 1), max(height, 1), max(width, 1)),
                dtype=img.dtype, device=img.device,
            )
            return (empty_img, empty_mask, empty_mask)

        seam_width_px = max(1, int(seam_width_px))
        seam_center_offset_px = int(seam_center_offset_px)
        mask_blur_px = max(0, int(mask_blur_px))

        # 将接缝平移到图像中心：水平拼接后再裁剪
        doubled = torch.cat((img, img), dim=2)
        start_x = int(width // 2 - seam_center_offset_px)
        start_x = max(0, min(start_x, width))
        prepared = doubled[:, :, start_x:start_x + width, :].contiguous().clamp(0.0, 1.0)

        # 生成接缝带掩码
        center_x = float(width) * 0.5 + float(seam_center_offset_px)
        half_width = float(seam_width_px) * 0.5
        x = torch.arange(width, dtype=img.dtype, device=img.device)
        band = ((x >= (center_x - half_width)) & (x < (center_x + half_width))).to(dtype=img.dtype)
        mask = band.view(1, 1, width).expand(batch, height, width).contiguous()
        mask_blurred = self._blur_mask(mask, mask_blur_px)

        return (prepared, mask, mask_blurred)


NODE_CLASS_MAPPINGS = {
    "YuanPanoramaPreview": YuanPanoramaPreview,
    "YuanPanoramaSeamPrep": YuanPanoramaSeamPrep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanPanoramaPreview": "全景预览",
    "YuanPanoramaSeamPrep": "全景接缝",
}
