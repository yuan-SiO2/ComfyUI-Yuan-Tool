"""H3 放大 (3D) 节点：纯 3D 卷积在 latent 空间放大 Minimax H3 视频（联合 AV latent 只放大视频流）。"""

import os
import glob

import torch

import folder_paths

from .h3_upscale_net import (
    LATENT_UPSCALE_FOLDER,
    LATENTS_MEAN,
    LATENTS_STD,
    LatentResizer3D,
    detect_arch,
)

# 缩放方式选项
UPSCALE_BY = "按倍数缩放"
UPSCALE_TARGET = "目标尺寸"


def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


# 模型加载（纯 3D 版本）
MODEL_CACHE = {}


def get_models_dir():
    return folder_paths.get_folder_paths(LATENT_UPSCALE_FOLDER)[0]


def scan_models():
    files = []
    model_dir = get_models_dir()
    for ext in ("*.pth", "*.safetensors"):
        files.extend(glob.glob(os.path.join(model_dir, ext)))
    # H3/MiniMax 权重排在前面，避免默认选中同目录下的 LTX 放大模型
    names = sorted(
        (os.path.basename(f) for f in files),
        key=lambda n: (0 if ("h3" in n.lower() or "minimax" in n.lower()) else 1, n),
    )
    return names if names else [f"(请将模型放入: {model_dir})"]


def _load_raw_sd(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path, device='cpu')
    else:
        sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    # 移除可能的前缀（如果有）并处理 FP8 格式
    sd = {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v
          for k, v in sd.items()}
    return sd


def _extract_upscaler_sd(sd):
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd


def _detect_dtype(sd):
    """从 state_dict 自动检测模型精度（跳过整数/布尔等非浮点张量）。"""
    for v in sd.values():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            return v.dtype
    return torch.float32


def load_model(name, device):
    if name in MODEL_CACHE:
        return MODEL_CACHE[name]

    path = os.path.join(get_models_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    raw_sd = _load_raw_sd(path)
    up_sd = _extract_upscaler_sd(raw_sd)
    if "conv_in.weight" not in up_sd:
        raise ValueError(
            f"{name} 不是 H3 LatentResizer3D 权重（缺少 conv_in.weight）。"
            "请选择 H3/MiniMax 的 latent upscaler，而不是 LTX 等其他模型的 upscaler。"
        )

    cfg = detect_arch(up_sd)

    model = LatentResizer3D(
        in_channels=cfg["in_channels"],
        in_blocks=cfg["in_blocks"],
        out_blocks=cfg["out_blocks"],
        channels=cfg["channels"],
        dropout=cfg["dropout"],
        attn=cfg["attn"],           # 强制 False
        temporal_every=cfg["temporal_every"],
        temporal_kernel=cfg["temporal_kernel"],
    )
    model.load_state_dict(up_sd, strict=True)
    # 自动检测模型精度：模型是什么精度就以什么精度推理
    dtype = _detect_dtype(up_sd)
    model = model.to(device=device, dtype=dtype).eval()

    MODEL_CACHE[name] = model

    return model


# ComfyUI 节点
class Yuan_H3Upscale3D:
    """H3 放大 (3D) 节点：latent 空间放大 H3 视频，仅放大空间分辨率，时间维度不变。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "display_name": "潜空间",
                    "tooltip": "Minimax H3 latent，可为纯视频 (B,C,T,H,W) 或联合 AV latent（NestedTensor，含音频流）。",
                }),
                "model_name": (scan_models(), {
                    "display_name": "放大模型",
                    "tooltip": "latent_upscale_models 目录下的 H3 latent 放大模型权重。",
                }),
                "resize_type": ([UPSCALE_BY, UPSCALE_TARGET], {
                    "default": UPSCALE_BY,
                    "display_name": "缩放方式",
                    "tooltip": "选择按倍数缩放或缩放到精确的目标尺寸。",
                }),
                "scale": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                    "display_name": "放大倍数",
                    "tooltip": "空间放大倍数（1.0~4.0）。1.0 原样返回；小于 1.0 会报错（仅支持放大）。结果尺寸归到最近的偶数 latent（像素对齐 32 的倍数）。仅在「按倍数缩放」模式下生效。",
                }),
                "width": ("INT", {
                    "default": 1920, "min": 64, "max": 8192, "step": 8,
                    "display_name": "目标宽度",
                    "tooltip": "目标宽度（像素），自动对齐到 32 的倍数（latent 偶数），可安全送入 H3 二采。仅支持放大：目标小于当前尺寸会报错。仅在「目标尺寸」模式下生效。",
                }),
                "height": ("INT", {
                    "default": 1080, "min": 64, "max": 8192, "step": 8,
                    "display_name": "目标高度",
                    "tooltip": "目标高度（像素），自动对齐到 32 的倍数（latent 偶数），可安全送入 H3 二采。仅支持放大：目标小于当前尺寸会报错。仅在「目标尺寸」模式下生效。",
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    OUTPUT_TOOLTIPS = ("放大后的 H3 latent，可直接送 VAE 解码或二次采样重绘。",)
    FUNCTION = "run"
    CATEGORY = "Yuan Tool/放大"
    DESCRIPTION = ("H3 放大：在 latent 空间用神经网络放大 Minimax H3 视频，跳过"
                   "「解码→放大→再编码」往返。纯 3D 卷积，时空一致性更好；"
                   "仅支持放大（scale ≥ 1.0 或目标 ≥ 当前尺寸）。")

    def run(self, latent, model_name, resize_type, scale, width, height):
        if model_name.startswith('('):
            raise ValueError("请将模型文件放入 latent_upscale_models 目录")

        samples = latent["samples"]
        # MiniMax H3 latent 为 NestedTensor((video [B,C,T,H,W], audio [B,32,2,T40]))
        # 仅放大 video 流，其余流 (audio 等) 原样保留
        is_nested = getattr(samples, "is_nested", False)
        if is_nested:
            streams = list(samples.unbind())
            probe = streams[0]
            if probe.ndim != 5:
                raise ValueError(f"MiniMax H3 video latent 应为 5D (B,C,T,H,W)，实际为 {tuple(probe.shape)}")
        else:
            probe = samples

        # 目标尺寸模式需要当前空间尺寸来判断是否无需放大 / 是否缩水
        if probe.ndim == 4:
            probe_5d = probe.unsqueeze(2)
        else:
            probe_5d = probe
        cur_t, cur_h, cur_w = probe_5d.shape[2], probe_5d.shape[3], probe_5d.shape[4]

        if resize_type == UPSCALE_BY:
            if abs(scale - 1.0) < 1e-6:
                return (latent,)
            if scale < 1.0:
                raise ValueError("仅支持放大 (scale >= 1.0)")
            # 归到最近的偶数 latent（等价像素对齐 32 的倍数），保证结果可安全
            # 送入 H3 采样器做 2×2 patch 化；偶数输入下结果不会小于当前尺寸
            t_h = max(2, int(round(cur_h * scale / 2.0)) * 2)
            t_w = max(2, int(round(cur_w * scale / 2.0)) * 2)
            if t_h == cur_h and t_w == cur_w:
                return (latent,)
            target_size = (cur_t, t_h, t_w)
            # 嵌入需要单一缩放标量：取两个轴向实际比值的平均作为整体缩放提示
            eff_scale = (t_h / cur_h + t_w / cur_w) / 2.0
        else:
            # 目标尺寸（像素）对齐到 32 的倍数 → latent 尺寸（偶数）
            t_w = max(2, round(int(width) / 32.0) * 32) // 16
            t_h = max(2, round(int(height) / 32.0) * 32) // 16
            if t_h < cur_h or t_w < cur_w:
                raise ValueError(
                    f"目标尺寸小于当前尺寸，仅支持放大（当前 latent {cur_w}x{cur_h}，"
                    f"目标 {t_w}x{t_h}，像素 {t_w*16}x{t_h*16}）")
            if t_h == cur_h and t_w == cur_w:
                return (latent,)
            target_size = (cur_t, t_h, t_w)
            # 嵌入需要单一缩放标量：取两个轴向比值的平均作为整体缩放提示
            eff_scale = (t_h / cur_h + t_w / cur_w) / 2.0

        # 自动检测设备：优先 cuda，无 GPU 时回退 cpu
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_name, dev)

        s = probe.clone()
        orig_dtype = s.dtype
        # 确保是 5D (B, C, T, H, W)
        if len(s.shape) == 4:
            s = s.unsqueeze(2)  # (B, C, 1, H, W)

        # 推理精度跟随模型自动检测的精度
        compute_dtype = next(model.parameters()).dtype
        s = s.to(dev, compute_dtype)

        # 归一化
        norm_mean, norm_std = _make_norm_tensors(dev, compute_dtype)
        s = (s - norm_mean) / norm_std

        with torch.no_grad():
            # 节点为单次前向，不做时间维分块
            out = model(s, scale=eff_scale, target_size=target_size, enable_chunking=False)

        # 反归一化
        out = out * norm_std + norm_mean

        # 还原维度
        if not is_nested and len(samples.shape) == 4:
            out = out.squeeze(2)

        out = out.cpu().to(orig_dtype)

        if dev.type == "cuda":
            torch.cuda.empty_cache()

        if is_nested:
            import comfy.nested_tensor
            streams[0] = out
            return ({"samples": comfy.nested_tensor.NestedTensor(streams)},)
        return ({"samples": out},)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "Yuan_H3Upscale3D": Yuan_H3Upscale3D,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_H3Upscale3D": "H3 放大",
}
