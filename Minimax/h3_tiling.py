"""H3 高分辨率模型评估的实验性空间分块。

把高分辨率阶段的单次前向拆成多个空间块分别计算，再加权拼接。
"""

from functools import partial
import inspect
import math

import torch

import comfy.ldm.common_dit
from comfy.ldm.minimax.model import PackedLayout
import comfy.model_base
import comfy.patcher_extension
import comfy.sampler_helpers


def _regions(length, tile_count=2):
    """按 2×2 patch 对齐把长度维切成若干带重叠的区域。"""
    patches = (length + 1) // 2
    tile_count = max(1, min(tile_count, max(1, patches // 2)))
    if patches < 4 or tile_count <= 1:
        return [(0, length)]
    overlap = min(4, max(1, patches // 8))
    regions = []
    for index in range(tile_count):
        start = max(0, index * patches // tile_count - overlap) * 2
        end = min(patches, (index + 1) * patches // tile_count + overlap) * 2
        end = min(end, length)
        regions.append((start, end))
    regions[-1] = (regions[-1][0], length)
    return regions


def _packed_layout(signature, payload):
    options = {"keyframes": payload.get("keyframes"), "refs": payload.get("refs")}
    if "frame_count" in inspect.signature(PackedLayout).parameters:
        options["frame_count"] = payload.get("frame_count")
    return PackedLayout(*signature, **options)


def _tile_payload(payload, context, video, audio, axis, start, end):
    """为单个空间块构造 minimax payload：裁切关键帧并重算布局。"""
    height, width = video.shape[-2:]
    padded_height, padded_width = (height + 1) // 2 * 2, (width + 1) // 2 * 2
    full_layout = payload.get("layout")
    signature = (context.shape[1], video.shape[2], padded_height, padded_width, audio.shape[-1])
    if full_layout is None or full_layout.signature != signature:
        full_layout = _packed_layout(signature, payload)
    tiled = payload.copy()
    if payload.get("keyframes"):
        keyframes = []
        for keyframe in payload["keyframes"]:
            latent = keyframe.get("latent")
            if latent is None:
                # 纯音频锚点没有视频 latent，不占空间维，原样沿用
                keyframes.append(keyframe)
                continue
            if latent.shape[-2:] != (height, width):
                raise ValueError("H3 渐进式采样器：分块模式下 H3 关键帧必须与目标 latent 的高宽一致")
            region = latent.narrow(axis, start, end - start)
            keyframes.append({**keyframe, "latent": comfy.ldm.common_dit.pad_to_patch_size(
                region, (1, 2, 2)).contiguous()})
        tiled["keyframes"] = keyframes
        # 参考条件不参与裁切，其行数由自身尺寸决定、不随分块变化，直接沿用
        # 只把带 latent 的关键帧计入 cond_video_latents，与核心口径保持一致
        tiled["cond_video_latents"] = [kf["latent"] for kf in keyframes if kf.get("latent") is not None] + [
            ref["latent"] for ref in (payload.get("refs") or []) if "latent" in ref]
    tile_height = end - start if axis == 3 else height
    tile_width = end - start if axis == 4 else width
    layout = _packed_layout((context.shape[1], video.shape[2], (tile_height + 1) // 2 * 2,
                             (tile_width + 1) // 2 * 2, audio.shape[-1]), tiled)
    # 从完整布局裁出本块的位置编码，保持全图坐标一致
    for (source_start, source_end, kind), (target_start, target_end, _) in zip(full_layout.segments, layout.segments):
        positions = full_layout.position_ids[source_start:source_end]
        if kind in ("cond", "video"):
            positions = positions.reshape(-1, padded_height // 2, padded_width // 2, 3)
            positions = positions.narrow(axis - 2, start // 2, (end - start + 1) // 2).reshape(-1, 3)
        layout.position_ids[target_start:target_end].copy_(positions)
    tiled["layout"] = layout
    return tiled


def _blend_window(length, left_overlap, right_overlap):
    """升余弦交叉淡化窗口，重叠区权重一阶连续。"""
    window = torch.ones(length, dtype=torch.float32)
    for overlap, tail in ((left_overlap, False), (right_overlap, True)):
        n = max(0, min(overlap, length // 2))
        if n == 0:
            continue
        ramp = (torch.arange(n, dtype=torch.float32) + 0.5) / n
        fade = 0.5 - 0.5 * torch.cos(ramp * math.pi)
        if tail:
            window[-n:] *= fade.flip(0)
        else:
            window[:n] *= fade
    return window


def _tiled_forward(executor, streams, timestep, context, transformer_options, minimax_payload=None,
                   n_tiles=2, plan=None, **kwargs):
    """替换 DIFFUSION_MODEL 前向：逐块调用原前向，再做带权融合。"""
    video, audio = streams
    # 沿空间 patch 数较多的方向分块
    axis = 3 if (video.shape[3] + 1) // 2 >= (video.shape[4] + 1) // 2 else 4
    length = video.shape[axis]
    regions = _regions(length, plan['tiles'] if plan is not None else n_tiles)
    if len(regions) == 1:
        return executor(streams, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    if kwargs.get("control") is not None:
        raise ValueError("H3 渐进式采样器：H3 高分辨率分块不支持 ControlNet")
    # 累加缓冲放在 CPU，避免上一块激活常驻显存
    video_output = torch.zeros(video.shape, dtype=torch.float32, device="cpu")
    audio_output = None
    audio_tiles = 0
    weights = torch.zeros(length, dtype=torch.float32, device="cpu")
    window_shape = [1] * video.ndim
    for index, (start, end) in enumerate(regions):
        left_overlap = min(end, regions[index - 1][1]) - start if index > 0 else 0
        right_overlap = end - regions[index + 1][0] if index + 1 < len(regions) else 0
        try:
            payload = _tile_payload(minimax_payload or {}, context, video, audio, axis, start, end)
        except Exception:
            # 构造失败时退回整帧前向，避免整次采样中断
            return executor(streams, timestep, context, transformer_options,
                            minimax_payload=minimax_payload, **kwargs)
        tile = video.narrow(axis, start, end - start).contiguous()
        predicted_video, predicted_audio = executor(
            [tile, audio], timestep, context, transformer_options.copy(), minimax_payload=payload, **kwargs)
        # 重叠区升余弦交叉淡化
        window = _blend_window(end - start, left_overlap, right_overlap)
        weights[start:end].add_(window)
        window_shape[axis] = end - start
        video_output.narrow(axis, start, end - start).addcmul_(
            predicted_video.float().cpu(), window.view(window_shape))
        # 每块输入相同音频，取平均更稳
        predicted_audio = predicted_audio.float().cpu()
        audio_output = predicted_audio if audio_output is None else audio_output.add_(predicted_audio)
        audio_tiles += 1
        del tile, payload, predicted_video, predicted_audio, window
    window_shape[axis] = length
    video_output.div_(weights.view(window_shape))
    return [video_output.to(device=video.device, dtype=video.dtype),
            (audio_output / audio_tiles).to(audio.dtype)]


def _condition_elements(condition, tile_height, tile_width, channels):
    """估算单条 conditioning 在该块尺寸下占用的元素数量。"""
    text = condition.get("cross_attn")
    elements = text.shape[-2] * channels * 4 if text is not None else 0
    for keyframe in condition.get("minimax_keyframes") or []:
        latent = keyframe.get("latent")
        if latent is None:
            # 纯音频锚点不占视频空间
            continue
        shape = latent.shape
        elements += shape[1] * shape[2] * tile_height * tile_width
    for reference in condition.get("minimax_refs") or []:
        latent = reference.get("latent")
        if latent is not None:
            shape = latent.shape
            elements += shape[1] * shape[2] * ((shape[3] + 1) // 2 * 2) * ((shape[4] + 1) // 2 * 2)
        audio = reference.get("audio_latent")
        if audio is not None:
            elements += math.prod(audio.shape[1:])
    return elements


def _budget(model, noise_shape, conds, latent_shapes, regions, axis):
    """构造内存估算输入，交给 ComfyUI 估算最低/首选预算。"""
    video_shape, audio_shape = latent_shapes
    full_elements = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    tile_shape = list(video_shape)
    tile_shape[axis] = max(end - start for start, end in regions)
    tile_shape[3] = (tile_shape[3] + 1) // 2 * 2
    tile_shape[4] = (tile_shape[4] + 1) // 2 * 2
    tile_elements = math.prod(tile_shape[1:]) + math.prod(audio_shape[1:])
    condition_elements = max((_condition_elements(condition, tile_shape[3], tile_shape[4], video_shape[1])
                              for group in conds.values() for condition in (group or [])), default=0)
    buffer_bytes = full_elements * 4 * 8
    bytes_per_element = model.model.memory_required((1, 1, 1))
    budget_elements = tile_elements + condition_elements + math.ceil(buffer_bytes / bytes_per_element)
    budget_shape = (noise_shape[0], 1, budget_elements)
    preferred, minimum = comfy.sampler_helpers.estimate_memory(model, budget_shape, conds)
    return budget_shape, tuple(tile_shape), buffer_bytes, preferred, minimum


def _prepare_tiled_sampling(executor, model, noise_shape, conds, model_options=None,
                            force_full_load=False, force_offload=False, *, latent_shapes, plan=None):
    """替换 PREPARE_SAMPLING：按可用工作区挑选 1–8 块并改写内存预算。"""
    video_shape, audio_shape = latent_shapes
    axis = 3 if (video_shape[3] + 1) // 2 >= (video_shape[4] + 1) // 2 else 4
    full_elements = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    if tuple(noise_shape) != (video_shape[0], 1, full_elements):
        raise ValueError("H3 渐进式采样器：分块内存规划收到的 latent 形状与采样输入不一致")
    count = 2
    if plan is not None:
        available = _available_workspace(model)
        for count in range(1, 9):
            regions = _regions(video_shape[axis], count)
            _, _, _, _, minimum = _budget(model, noise_shape, conds, latent_shapes, regions, axis)
            if minimum <= available:
                break
        count = len(regions)
        plan['tiles'] = count
    regions = _regions(video_shape[axis], count)
    if len(regions) == 1 or force_offload:
        return executor(model, noise_shape, conds, model_options=model_options,
                        force_full_load=force_full_load, force_offload=force_offload)
    budget_shape = _budget(model, noise_shape, conds, latent_shapes, regions, axis)[0]
    return executor(model, budget_shape, conds, model_options=model_options,
                    force_full_load=force_full_load, force_offload=force_offload)


def _available_workspace(model):
    """估算当前设备可用于分块工作区的显存。"""
    manager = comfy.model_management
    free = manager.get_free_memory(model.load_device)
    reclaimable = 0
    seen = set()
    for loaded in manager.loaded_models():
        patcher = loaded if callable(getattr(loaded, "loaded_size", None)) else loaded.model
        load_device = getattr(patcher, "load_device", None)
        identity = id(patcher.model)
        if load_device == model.load_device and identity not in seen:
            seen.add(identity)
            size_fn = getattr(patcher, "loaded_size", None)
            if callable(size_fn):
                reclaimable += size_fn()
    pool = min(manager.get_total_memory(model.load_device), free + reclaimable)
    weights = min(model.model_size(), pool * manager.MIN_WEIGHT_MEMORY_RATIO)
    available = max(0, pool - weights - manager.minimum_inference_memory())
    return available


def tiled_model(model, latent_shapes):
    """克隆模型并挂上分块包装器；要求模型为 MiniMax H3 且同时具备视频/音频两路 latent。"""
    if not isinstance(model.model, comfy.model_base.MiniMaxH3):
        raise ValueError("H3 渐进式采样器：高分辨率分块需要 MiniMax H3 模型")
    if len(latent_shapes) != 2 or len(latent_shapes[0]) != 5 or len(latent_shapes[1]) != 4:
        raise ValueError("H3 渐进式采样器：高分辨率分块需要 H3 的视频与音频两路 latent 流")
    plan = {}
    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                 "yuan_h3_progressive_high_resolution_tiling", partial(_tiled_forward, plan=plan))
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
                                 "yuan_h3_progressive_high_resolution_tiling",
                                 partial(_prepare_tiled_sampling, latent_shapes=tuple(tuple(shape) for shape in latent_shapes), plan=plan))
    return patched
