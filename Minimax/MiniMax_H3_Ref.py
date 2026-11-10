"""MiniMax-H3 参考嵌入：把参考条件拆成「预编码」与「生成」两个节点。

原生「参考转视频」把提示词 / 分辨率 / 时长和参考素材放在同一个节点里，而 ComfyUI 的缓存
粒度是节点：改一句提示词、改个时长，整条像素链路都会重跑——lanczos 缩放上百帧，再做一次
完整的 3D 视频 VAE 编码（15 秒 24fps ≈ 360 帧，单条参考的视频潜变量就有数百 MB）。

这里按 WanVideoWrapper 的 Animate Embeds 思路拆成两步：

    MiniMax-H3 参考嵌入  →  参考嵌入（MINIMAXH3_REFS）  →  MiniMax-H3 参考转视频（嵌入）

- 上游只依赖参考素材本身，ComfyUI 会缓存它的输出：之后改提示词 / 分辨率 / 时长 / 种子
  都不再触碰像素与视频 VAE；
- 可选的内容寻址磁盘缓存（按参考素材内容做指纹），重启、新工作流里也能直接复用；
- 参考进入模型的顺序与原生完全一致，序号 1-based 按类型计数：
  <Picture i> / <Video k>（其音轨的 <Audio j> 紧挨在 <Video k> 之前）/ <Audio j>。

几何换算、音频编码、空音频判定、AV 潜空间构造都直接复用 MiniMax_H3 里的实现（与原生逐行一致），
本文件只负责「预编码 / 复用 / 按目标时长裁剪」这一段。

下游 DiT 侧仍然是原生那条路：minimax_refs 写进 CONDITIONING，由 comfy/model_base.py 的
extra_conds 取走，在 PackedLayout 里排到 text 之后、目标 video/audio 之前，作为「每一步重新
注入、从不被去噪」的条件行。
"""

import hashlib
import math
import os

import torch

import comfy.model_management
import folder_paths
import nodes
import node_helpers

# 复用 MiniMax_H3 的几何/音频/潜空间helper，避免同一套参数在两处漂移
from .MiniMax_H3 import (
    CANVAS_MULTIPLE,
    FPS,
    REF_AUDIO_PORTS,
    REF_IMAGE_PORTS,
    REF_IMAGE_SHORT_EDGE,
    REF_VIDEO_PORTS,
    _empty_av_latent,
    _encode_ref_audio,
    _is_empty_audio,
    _resize,
    adapt_canvas,
    video_latent_t,
)

REFS_TYPE = "MINIMAXH3_REFS"
BUNDLE_VERSION = 1

CACHE_SUBDIR = "minimax_h3_ref_cache"
CACHE_DIR_ENV = "MINIMAX_H3_REF_CACHE_DIR"

# 界面中文选项 ↔ 内部值（保存的工作流里存的是界面值，这里统一在入口还原）
REF_SIZE_MATCH = "匹配"
REF_SIZE_MAX = "最大"
CACHE_DISK = "磁盘"
CACHE_OFF = "关闭"


# --------------------------------------------------------------------------- 磁盘缓存


def _cache_dir():
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        root = override
    else:
        models_dir = getattr(folder_paths, "models_dir", None) or folder_paths.get_output_directory()
        root = os.path.join(models_dir, CACHE_SUBDIR)
    os.makedirs(root, exist_ok=True)
    return root


def _hash_tensor(digest, tensor):
    if not isinstance(tensor, torch.Tensor):
        digest.update(b"\x00none")
        return
    t = tensor.detach().cpu()
    if t.dtype != torch.float32:  # bf16/fp16/uint8 的 numpy 视图不可靠，统一成 fp32 再哈希
        t = t.float()
    digest.update(repr((tuple(t.shape), str(t.dtype))).encode())
    digest.update(t.contiguous().numpy().tobytes())


def _hash_audio(digest, audio):
    if not isinstance(audio, dict) or not isinstance(audio.get("waveform"), torch.Tensor):
        digest.update(b"\x00noaudio")
        return
    digest.update(repr(audio.get("sample_rate")).encode())
    _hash_tensor(digest, audio["waveform"])


def _fingerprint(vae):
    """VAE 指纹：结构 + 关键属性。换了同结构的不同权重时，请填「缓存标记」把它区分开。"""
    if vae is None:
        return "none"
    first_stage = getattr(vae, "first_stage_model", None)
    return "|".join(str(part) for part in (
        type(first_stage).__name__ if first_stage is not None else "",
        getattr(vae, "latent_channels", ""),
        getattr(vae, "downscale_ratio_spacial", ""),
        getattr(vae, "audio_sample_rate", ""),
        getattr(vae, "downscale_ratio", ""),
    ))


def _cache_key(ref_image_size, match_width, match_height, vae, audio_vae,
               ref_images, videos, audios, cache_id):
    digest = hashlib.sha256()
    digest.update(("v%d|%s|%dx%d|%s|%s|%s" % (
        BUNDLE_VERSION, ref_image_size, match_width, match_height,
        _fingerprint(vae), _fingerprint(audio_vae), cache_id or "")).encode())
    _hash_tensor(digest, ref_images[:REF_IMAGE_PORTS] if ref_images is not None else None)
    for frames, soundtrack in videos:
        _hash_tensor(digest, frames)
        _hash_audio(digest, soundtrack)
    for audio in audios:
        _hash_audio(digest, audio)
    return digest.hexdigest()


def _cache_path(key):
    return os.path.join(_cache_dir(), key + ".pt")


def _cache_load(key):
    try:
        path = _cache_path(key)
        if not os.path.isfile(path):
            return None
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # 目录不可写、文件损坏、torch 版本没有 weights_only……
        return None
    if not isinstance(bundle, dict) or bundle.get("version") != BUNDLE_VERSION:
        return None
    return bundle


def _cache_store(key, bundle):
    tmp = None
    try:
        path = _cache_path(key)
        tmp = path + ".tmp"
        torch.save(bundle, tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _bundle_to_device(bundle):
    """磁盘缓存读回的是 CPU 张量；条件潜变量放到计算设备，避免采样时每步 H2D 拷贝。"""
    device = comfy.model_management.intermediate_device()
    for block in bundle.get("ref_blocks", ()):
        for field in ("latent", "audio_latent"):
            value = block.get(field)
            if isinstance(value, torch.Tensor):
                block[field] = value.to(device)
    return bundle


# --------------------------------------------------------------------------- 参考帧压缩存储


def _pack_frames(frames, chunk=8):
    """参考帧按 uint8 存进嵌入包：视频源本身就是 8bit，内存/磁盘体积降到 1/4。

    分块转换，避免一次多开一整份 float32 临时张量。
    """
    frames = frames.detach()
    if frames.shape[0] <= chunk:
        return (frames.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    parts = [(frames[start:start + chunk].float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
             for start in range(0, frames.shape[0], chunk)]
    return torch.cat(parts, dim=0)


def _unpack_frames(frames):
    return frames.to(torch.float32) / 255.0


def _copy_item(item):
    out = dict(item)
    if "data" in out:
        out["data"] = _unpack_frames(out["data"])
    if "timestamps" in out:
        out["timestamps"] = list(out["timestamps"])
    return out


# --------------------------------------------------------------------------- 参考编码


def _encode_images(items, blocks, ref_images, match_mode, match_width, match_height, vae):
    """参考图像：保持宽高比缩放（仅缩小）后编码；没有视频 VAE 时只进文本编码器。"""
    if ref_images is None:
        return
    width, height = match_width, match_height
    for img in ref_images[:REF_IMAGE_PORTS]:
        img = img.unsqueeze(0)  # 单张 [1, H, W, C]
        h, w = img.shape[1], img.shape[2]
        if match_mode == "match":
            scale = min(1.0, math.sqrt((width * height) / (w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img, tw, th, "disabled")
        item = {"type": "image", "data": _pack_frames(resized)}
        items.append(item)
        if vae is None:
            continue
        z = vae.encode(resized)
        item["block"] = len(blocks)
        blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                       "latent": z.detach()})


def _encode_videos(items, blocks, videos, vae, audio_vae):
    """参考视频：对齐到 17k+5 网格后整段编码；Qwen 以 2fps 带时间戳观看同一条片段。"""
    for frames, soundtrack in videos:
        if frames is None:
            continue
        vh, vw = frames.shape[1], frames.shape[2]
        cw, ch = adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            # 不放大：素材比画布小时按原尺寸取整
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(frames, cw, ch, "disabled")
        n = resized.shape[0]
        if n < 5:
            raise ValueError("MiniMax-H3 参考视频至少需要 5 帧（24fps 下约 0.2 秒）")
        while n % 17 != 5:
            n -= 1
        resized = resized[:n]
        has_soundtrack = not _is_empty_audio(soundtrack, audio_vae)
        if has_soundtrack:
            # 音轨获得独立的 <Audio j> 标记，紧挨在 <Video k> 之前
            items.append({"type": "audio"})
        sample_idx = list(range(0, n, FPS // 2))
        items.append({"type": "video", "data": _pack_frames(resized[sample_idx]),
                      "timestamps": [step / 2.0 for step in range(len(sample_idx))]})
        if vae is None:
            continue
        z = vae.encode(resized)
        audio_latent, ref_audio_t = (None, 0)
        if has_soundtrack and audio_vae is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
        items[-1]["block"] = len(blocks)
        blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                       "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                       "video_frames": n, "ref_audio_t": ref_audio_t,
                       "latent": z.detach(), "audio_latent": audio_latent})


def _encode_audios(items, blocks, audios, audio_vae):
    """独立参考音频：只占一个 <Audio j> 标记；没有音频 VAE 时只进文本编码器。"""
    for audio in audios:
        if _is_empty_audio(audio, audio_vae):
            continue
        items.append({"type": "audio"})
        if audio_vae is None:
            continue
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
        items[-1]["block"] = len(blocks)
        blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                       "audio_latent": audio_latent.detach()})


def _build_bundle(match_mode, match_width, match_height, vae, audio_vae, ref_images, videos, audios):
    items, blocks = [], []
    _encode_images(items, blocks, ref_images, match_mode, match_width, match_height, vae)
    _encode_videos(items, blocks, videos, vae, audio_vae)
    _encode_audios(items, blocks, audios, audio_vae)
    summary = {
        "items": len(items),
        "blocks": [block["kind"] for block in blocks],
        "condition_rows": sum(
            # 条件行的数量级（每帧按 2x2 patch 折算），用于日志判断好坏
            (block.get("latent_t", 1) * (block.get("latent_h", 1) // 2) * (block.get("latent_w", 1) // 2))
            for block in blocks if "latent" in block),
    }
    return {"version": BUNDLE_VERSION, "ref_items": items, "ref_blocks": blocks, "info": summary}


# --------------------------------------------------------------------------- 复用裁剪


def _trim_items(items, blocks, frame_count, fps=2):
    """参考视频比目标时长更长时，按原生行为裁到目标长度（视频侧裁剪，音轨保持完整）。

    原生在编码前就做 frames[:frame_count] + 17k+5 对齐；这里对已缓存的 2fps 帧和潜变量做
    等价的前缀裁剪（因果 3D VAE 下前缀精确位同）。
    """
    removed = 0
    for item in items:
        block_index = item.get("block")
        if block_index is None or item.get("type") != "video":
            continue
        block = blocks[block_index]
        total = block.get("video_frames")
        if total is None or total <= frame_count:
            continue
        n = frame_count
        while n % 17 != 5:
            n -= 1
        if n < 5 or n >= total:
            continue
        keep = len(range(0, n, FPS // fps))
        item["data"] = item["data"][:keep]
        item["timestamps"] = item["timestamps"][:keep]
        block["latent"] = block["latent"][:, :, :video_latent_t(n)]
        block["latent_t"] = block["latent"].shape[2]
        block["video_frames"] = n
        removed += total - n
    return removed


# --------------------------------------------------------------------------- 节点


class YuanMiniMaxH3RefEmbeds:
    """参考视频/图像/音频的一次性预编码，输出可跨提示词/分辨率/时长复用的参考嵌入。"""

    RETURN_TYPES = (REFS_TYPE,)
    RETURN_NAMES = ("参考嵌入",)
    OUTPUT_TOOLTIPS = ("预编码参考嵌入（视觉帧 + 视频/音频潜变量），可反复复用",)
    FUNCTION = "encode"
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("MiniMax-H3 参考素材预编码：缩放 + 视频/音频 VAE 编码只做一次，"
                   "下游改提示词/分辨率/时长不再重跑像素链路。")

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "vae": ("VAE", {"display_name": "视频VAE",
                            "tooltip": "把参考图像/视频编码成条件潜变量；不连接时参考只进入文本编码器（与原生一致）"}),
            "audio_vae": ("VAE", {"display_name": "音频VAE",
                                  "tooltip": "编码参考音轨；不连接时音轨只占一个 <Audio j> 标记"}),
            "ref_images": ("IMAGE", {"display_name": "参考图像",
                                     "tooltip": "参考图像（可接一批多张，最多 %d 张，超出自动切断）" % REF_IMAGE_PORTS}),
            "cache_id": ("STRING", {"default": "", "display_name": "缓存标记",
                                    "tooltip": "缓存指纹的补充标记：换了结构相同的另一份 VAE 权重时填一下（例如 VAE 文件名），避免复用旧缓存"}),
        }
        for i in range(1, REF_VIDEO_PORTS + 1):
            optional["ref_video_%d" % i] = (
                "IMAGE", {"display_name": "参考视频%d" % i,
                          "tooltip": "第 %d 路参考视频帧（24fps，2-15 秒；超出目标时长时下游按目标长度裁剪）" % i})
        for i in range(1, REF_VIDEO_PORTS + 1):
            optional["ref_video_audio_%d" % i] = (
                "AUDIO", {"display_name": "参考视频音频%d" % i,
                          "tooltip": "与参考视频%d 配对的音轨（有音轨时该视频会在提示词里多占一个 <Audio j> 标记）" % i})
        for i in range(1, REF_AUDIO_PORTS + 1):
            optional["ref_audio_%d" % i] = (
                "AUDIO", {"display_name": "参考音频%d" % i, "tooltip": "第 %d 路独立参考音频" % i})
        return {
            "required": {
                "ref_image_size": ([REF_SIZE_MATCH, REF_SIZE_MAX], {
                    "default": REF_SIZE_MATCH, "display_name": "参考图尺寸",
                    "tooltip": "参考图缩放策略：'匹配'缩到生成画面的像素面积（仅缩小、保宽高比）；"
                               "'最大'用参考管线的 2048px 短边，身份保真更好但每一步都更慢"}),
                "match_width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                    "display_name": "参考宽度",
                    "tooltip": "「匹配」模式下参考图按这个宽度折算目标像素面积：请与下游生成的宽度保持一致"}),
                "match_height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                    "display_name": "参考高度",
                    "tooltip": "「匹配」模式下参考图按这个高度折算目标像素面积：请与下游生成的高度保持一致"}),
                "cache_mode": ([CACHE_DISK, CACHE_OFF], {
                    "default": CACHE_DISK, "display_name": "缓存模式",
                    "tooltip": "'磁盘'：按参考素材内容做指纹缓存到磁盘，重启/新工作流也能直接复用；"
                               "'关闭'：只用 ComfyUI 的内存缓存"}),
            },
            "optional": optional,
        }

    def encode(self, ref_image_size, match_width, match_height, cache_mode,
               vae=None, audio_vae=None, ref_images=None, cache_id="", **kwargs):
        # 下拉框是中文选项，这里还原成内部处理用的英文 key
        match_mode = "max" if ref_image_size in (REF_SIZE_MAX, "max") else "match"
        disk_cache = cache_mode not in (CACHE_OFF, "off")
        videos = [(kwargs.get("ref_video_%d" % i), kwargs.get("ref_video_audio_%d" % i))
                  for i in range(1, REF_VIDEO_PORTS + 1)]
        audios = [kwargs.get("ref_audio_%d" % i) for i in range(1, REF_AUDIO_PORTS + 1)]

        key = None
        if disk_cache:
            key = _cache_key(match_mode, match_width, match_height, vae, audio_vae,
                             ref_images, videos, audios, cache_id)
            cached = _cache_load(key)
            if cached is not None:
                return (_bundle_to_device(cached),)

        bundle = _build_bundle(match_mode, match_width, match_height, vae, audio_vae,
                               ref_images, videos, audios)
        if key is not None:
            _cache_store(key, bundle)
        return (_bundle_to_device(bundle),)


class YuanMiniMaxH3RefToVideo:
    """用预编码参考 + 提示词生成条件与 AV 潜变量，等价于原生参考转视频，但不碰像素。"""

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("正向", "潜空间")
    OUTPUT_TOOLTIPS = ("正向条件（含参考潜变量 minimax_refs）", "视频+音频联合潜空间")
    FUNCTION = "generate"
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("MiniMax-H3 参考转视频（嵌入版）：<Picture i> / <Video k> / <Audio j> 引用参考内容，"
                   "参考素材来自「参考嵌入」节点的预编码输出。")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"display_name": "CLIP", "tooltip": "Qwen3-VL 文本/视觉编码器"}),
                "refs": (REFS_TYPE, {"display_name": "参考嵌入",
                                     "tooltip": "「MiniMax-H3 参考嵌入」节点输出的预编码参考"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "display_name": "提示词",
                    "tooltip": "视频内容的提示词；用 <Picture i> / <Video k> / <Audio j> 引用参考"
                               "（序号从 1 开始，按类型分别计数）"}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                    "display_name": "宽度", "tooltip": "视频宽度，需为 32 的倍数"}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                    "display_name": "高度", "tooltip": "视频高度，需为 32 的倍数"}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                    "display_name": "时长（帧）",
                    "tooltip": "24fps 视频帧数，自动向上对齐到 17k+5 帧网格（124≈5 秒；训练区间约 124-362）"}),
            },
        }

    def generate(self, clip, refs, prompt, width, height, length):
        latent, frame_count = _empty_av_latent(width, height, length)
        bundle = _prepare(refs)
        items = [_copy_item(item) for item in bundle["ref_items"]]
        blocks = [dict(block) for block in bundle["ref_blocks"]]

        _trim_items(items, blocks, frame_count)

        # 送进分词器的只有呈现信息（type / data / timestamps），块下标是本插件内部用的
        presented = [{key: value for key, value in item.items() if key != "block"} for item in items]
        tokens = clip.tokenize(prompt, minimax_ref_items=presented)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": blocks})
        return (cond, latent)


def _prepare(refs):
    if not isinstance(refs, dict) or "ref_items" not in refs:
        raise ValueError("「参考嵌入」输入无效：请连接「MiniMax-H3 参考嵌入」节点的输出")
    if refs.get("version") != BUNDLE_VERSION:
        raise ValueError("参考嵌入版本不匹配（%s != %d），请重新运行「MiniMax-H3 参考嵌入」节点"
                         % (refs.get("version"), BUNDLE_VERSION))
    return refs


NODE_CLASS_MAPPINGS = {
    "Yuan_MiniMaxH3RefEmbeds": YuanMiniMaxH3RefEmbeds,
    "Yuan_MiniMaxH3RefToVideo": YuanMiniMaxH3RefToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_MiniMaxH3RefEmbeds": "MiniMax-H3 参考嵌入",
    "Yuan_MiniMaxH3RefToVideo": "MiniMax-H3 参考转视频（嵌入）",
}
