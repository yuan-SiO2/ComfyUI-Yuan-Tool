"""MiniMax H3 的学习式 latent 提升器（3D 卷积上采样器）。

网络结构见 h3_upscale_net.py，权重放在 ComfyUI/models/latent_upscale_models/ 下。
"""

import os

import torch

import comfy.model_management
import comfy.model_patcher
import folder_paths

from .h3_upscale_net import (
    LATENT_UPSCALE_FOLDER,
    LATENTS_MEAN,
    LATENTS_STD,
    LatentResizer3D,
)

_FOLDER = LATENT_UPSCALE_FOLDER

_model_cache = {}


def list_upscaler_models():
    """扫描 latent_upscale_models 目录，返回全部 .pth/.safetensors 相对路径。"""
    try:
        paths = folder_paths.get_folder_paths(_FOLDER)
    except KeyError:
        return []
    names = []
    for p in paths:
        for root, _, files in os.walk(p):
            for f in files:
                if os.path.splitext(f)[1].lower() in (".pth", ".safetensors"):
                    names.append(os.path.relpath(os.path.join(root, f), p))
    return sorted(names)


def _detect_arch(sd):
    """从 state_dict 的键名和张量形状推断网络结构参数。"""
    import re
    cfg = {"in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
           "dropout": 0.1, "attn": False, "temporal_every": 2, "temporal_kernel": 5}
    if 'conv_in.weight' in sd:
        cfg["in_channels"] = sd['conv_in.weight'].shape[1]
        cfg["channels"] = sd['conv_in.weight'].shape[0]
    in_ids, out_ids, tin, tout = set(), set(), set(), set()
    for k in sd:
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m: out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m: tin.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m: tout.add(int(m.group(1)))
    if in_ids: cfg["in_blocks"] = len(in_ids)
    if out_ids: cfg["out_blocks"] = len(out_ids)
    if tin or tout:
        cfg["temporal_every"] = 2
        for k in sd:
            if k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0
    cfg["attn"] = False
    return cfg


def _normalize_checkpoint_dtype(state_dict):
    """把浮点权重统一到输入卷积的精度，兼容 FP8 checkpoint。"""
    weight = state_dict.get("conv_in.weight")
    if weight is None:
        raise ValueError("H3 渐进式采样器：提升器 checkpoint 缺少 conv_in.weight")
    dtype = weight.dtype
    if str(dtype).startswith("torch.float8_"):
        dtype = torch.bfloat16 if any(value.dtype == torch.bfloat16 for value in state_dict.values()) else torch.float16
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"H3 渐进式采样器：不支持的提升器权重精度 {dtype}")
    return {name: value.to(dtype=dtype) if value.is_floating_point() else value
            for name, value in state_dict.items()}


def _load_model(model_name, device):
    """按 (模型名, 设备) 缓存加载提升器，并用 ComfyUI 模型管理器托管驻留。"""
    key = (model_name, str(device))
    if key in _model_cache:
        return _model_cache[key]

    path = None
    for p in folder_paths.get_folder_paths(_FOLDER):
        candidate = os.path.join(p, model_name)
        if os.path.isfile(candidate):
            path = candidate
            break
    if path is None:
        raise FileNotFoundError(f"未找到 latent 提升器模型：{model_name}（请放到 ComfyUI/models/{_FOLDER}/ 下）")

    import comfy.utils
    sd = comfy.utils.load_torch_file(path)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    sd = _normalize_checkpoint_dtype(sd)

    cfg = _detect_arch(sd)
    with torch.device("meta"):
        model = LatentResizer3D(
            in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
            channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
            temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"],
        )
    model.load_state_dict(sd, strict=True, assign=True)
    model = model.eval().requires_grad_(False)
    patcher = comfy.model_patcher.CoreModelPatcher(
        model, load_device=device,
        offload_device=comfy.model_management.unet_offload_device())
    _model_cache[key] = patcher
    return patcher


def _inference_memory_required(model, z0_low, out_hw):
    """估算一次推理所需的显存工作区（含 8 倍安全系数）。"""
    H, W = out_hw
    T = z0_low.shape[2]
    temporal_window = model.temporal_window_budget(T)
    feature_elements = z0_low.shape[0] * model.conv_in.out_channels * temporal_window * H * W
    return feature_elements * model.conv_in.weight.element_size() * 8


def learned_latent_lift(z0_low, out_hw, model_name, device=None, force_unload=False):
    """把低分辨率干净端点提升到目标 latent 尺寸。

    z0_low：[B, 24, T, h, w] 的 H3 视频 latent，返回 [B, 24, T, H, W]。
    force_unload：提升结束后（含失败）把提升器从显存卸载。patcher 仍留在
    _model_cache 中，下次提升重新加载时无需再读文件。
    """
    H, W = out_hw
    if device is None:
        device = comfy.model_management.get_torch_device()
    h, w = z0_low.shape[-2], z0_low.shape[-1]
    scale = (H / h + W / w) / 2.0

    patcher = _load_model(model_name, device)
    model = patcher.model
    memory_required = _inference_memory_required(model, z0_low, (H, W))
    try:
        comfy.model_management.load_models_gpu([patcher], memory_required=memory_required)
        dtype = model.conv_in.weight.dtype
        # H3 VAE latent 均值/标准差归一化
        mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)

        x = z0_low.to(device=device, dtype=dtype)
        with torch.no_grad():
            x = (x - mean) / std
            out = model(x, scale=scale, target_size=(z0_low.shape[2], H, W))
            out = (out * std + mean).float().to(comfy.model_management.intermediate_device())
        return out
    finally:
        if force_unload:
            # 高分辨率阶段紧接其后且不会再用到提升器，卸载可省下一份常驻权重与工作区
            comfy.model_management.unload_model_and_clones(patcher, unload_additional_models=False)
            comfy.model_management.soft_empty_cache()
