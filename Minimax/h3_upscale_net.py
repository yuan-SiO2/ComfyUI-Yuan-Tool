"""MiniMax H3 latent 提升器的共享 3D 卷积网络。

由 h3_upscaler.py（渐进式采样器内部调用）与 Yuan_H3_Upscale.py（H3 放大节点）
共同导入，保证两侧网络结构与权重键名完全一致。

权重放在 ComfyUI/models/latent_upscale_models/ 下。
"""

import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths

from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD

LATENT_UPSCALE_FOLDER = "latent_upscale_models"

if LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, LATENT_UPSCALE_FOLDER),
    )


def _block_sequence(sd, prefix, total):
    """按 ModuleList 下标还原 in_blocks/out_blocks 的模块类型序列（R=残差块，T=时间卷积）。"""
    sequence = []
    for index in range(total):
        if f"{prefix}.{index}.dwconv.weight" in sd:
            sequence.append("T")
        elif any(key.startswith(f"{prefix}.{index}.in_layers.") for key in sd):
            sequence.append("R")
        else:
            sequence.append("?")
    return sequence


def _infer_temporal_every(sd, prefix, block_count):
    """用模块类型序列匹配候选 temporal_every。

    ModuleList 的下标不等于 block 下标（每个 block 后面可能插入 TemporalConv），
    因此不能用 dwconv 下标的间隔取 gcd；这里按构造顺序重建序列后逐候选比对。
    """
    if block_count <= 0:
        return 0
    max_index = -1
    for key in sd:
        match = re.match(rf"{prefix}\.(\d+)\.", key)
        if match:
            max_index = max(max_index, int(match.group(1)))
    if max_index < 0:
        return 0
    observed = _block_sequence(sd, prefix, max_index + 1)
    if "T" not in observed:
        return 0
    for candidate in range(1, block_count + 1):
        simulated = []
        for block in range(block_count):
            simulated.append("R")
            if block % candidate == 0:
                simulated.append("T")
        if simulated == observed:
            return candidate
    # 含未知模块（如 checkpoint 带 attention）时退回训练默认值
    return 2


def detect_arch(sd):
    """从 state_dict 的键名与张量形状推断网络结构参数。"""
    cfg = {
        "in_channels": 24,
        "in_blocks": 12,
        "out_blocks": 12,
        "channels": 512,
        "dropout": 0.1,
        "attn": False,
        "temporal_every": 2,
        "temporal_kernel": 5,
    }

    if "conv_in.weight" in sd:
        cfg["in_channels"] = sd["conv_in.weight"].shape[1]
        cfg["channels"] = sd["conv_in.weight"].shape[0]

    in_ids, out_ids = set(), set()
    for key in sd:
        match = re.match(r"in_blocks\.(\d+)\.in_layers\.", key)
        if match:
            in_ids.add(int(match.group(1)))
        match = re.match(r"out_blocks\.(\d+)\.in_layers\.", key)
        if match:
            out_ids.add(int(match.group(1)))

    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)

    cfg["temporal_every"] = _infer_temporal_every(sd, "in_blocks", cfg["in_blocks"])
    if cfg["temporal_every"] == 0:
        cfg["temporal_every"] = _infer_temporal_every(sd, "out_blocks", cfg["out_blocks"])
    if cfg["temporal_every"] > 0:
        for key in sd:
            if key.endswith("dwconv.weight"):
                cfg["temporal_kernel"] = sd[key].shape[2]
                break

    # 推理时强制关闭注意力，保证与推理实现一致
    cfg["attn"] = False
    return cfg


def normalization(channels):
    return nn.GroupNorm(32, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def _temporal_windows(length, chunk, overlap):
    for start in range(0, length, chunk):
        end = min(length, start + chunk)
        yield start, end, max(0, start - overlap), min(length, end + overlap)


class AttnBlock3D(nn.Module):
    """3D 自注意力块（可选）。"""

    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h).flatten(2).movedim(-1, 1).unsqueeze(1)
        k = self.k(h).flatten(2).movedim(-1, 1).unsqueeze(1)
        v = self.v(h).flatten(2).movedim(-1, 1).unsqueeze(1)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.squeeze(1).movedim(1, -1).reshape(x.shape)
        return x + self.proj_out(h)


class ResBlockEmb3D(nn.Module):
    """带尺度嵌入调制的 3D 残差块。"""

    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    """时间维深度可分离卷积（残差形式）。"""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


class LatentResizer3D(nn.Module):
    """3D 上采样主干，可选时间维分块。"""

    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(AttnBlock3D(channels))
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(AttnBlock3D(channels))
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def temporal_chunk_settings(self):
        """返回 (时间分块长度, 重叠长度)；无时间卷积时重叠为 0。"""
        for block in self.in_blocks:
            if isinstance(block, TemporalConv):
                return 32, block.dwconv.weight.shape[2]
        return 32, 0

    def temporal_window_budget(self, length):
        """估算一次前向实际会处理的最大时间窗口长度（含重叠与填充）。"""
        chunk, overlap = self.temporal_chunk_settings()
        return length if length <= chunk else min(length + 2 * overlap, chunk + 4 * overlap)

    def forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        B, C, T, H, W = x.shape

        chunk, overlap = self.temporal_chunk_settings()

        if not enable_chunking or T <= chunk:
            return self._forward_seg(x, scale, size)

        # 时间维按 replicate 填充，分块推理后用加权窗口拼接
        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode='replicate')

        out_full = torch.zeros(B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)

        for seg_start, seg_end, out_start, out_end in _temporal_windows(T, chunk, overlap):
            lo = out_start
            hi = out_end + 2 * overlap

            seg = x_padded[:, :, lo:hi].contiguous()
            seg_size = (hi - lo, size[-2], size[-1])
            seg_out = self._forward_seg(seg, scale, seg_size)

            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start

            # 两端线性淡入淡出，平滑重叠区
            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(1, blend_len + 1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(blend_len, 0, -1, device=x.device, dtype=x.dtype) / (blend_len + 1)

            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)

            del seg, seg_out, valid_out

        return out_full / weight_full.clamp(min=1e-8)

    def _forward_seg(self, x, scale, size):
        """对单个时间片段执行一次完整前向，并在中间层做空间插值。"""
        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        # 在瓶颈处一次性放大到目标空间尺寸
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)
