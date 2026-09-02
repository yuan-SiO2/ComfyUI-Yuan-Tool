"""H3 放大 (3D) 节点：纯 3D 卷积在 latent 空间放大 Minimax H3 视频（联合 AV latent 只放大视频流）。"""

import os
import re
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import folder_paths
import comfy.nested_tensor

# 模型文件夹注册
_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER)
    )

# 缩放方式选项
UPSCALE_BY = "按倍数缩放"
UPSCALE_TARGET = "目标尺寸"

# Minimax H3 归一化参数（24 通道）
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]


def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


# 3D 网络组件
def normalization(channels):
    return nn.GroupNorm(32, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = rearrange(self.q(h), "b c t h w -> b 1 (t h w) c")
        k = rearrange(self.k(h), "b c t h w -> b 1 (t h w) c")
        v = rearrange(self.v(h), "b c t h w -> b 1 (t h w) c")
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b 1 (t h w) c -> b c t h w", t=x.shape[2], h=x.shape[3], w=x.shape[4])
        return x + self.proj_out(h)


class ResBlockEmb3D(nn.Module):
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


# 纯 3D 主干网络
class LatentResizer3D(nn.Module):
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

    def forward(self, x, scale=None, target_size=None):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            # 计算目标大小 (T, H, W)
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        # 三线性插值
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


# 模型加载（纯 3D 版本）
MODEL_CACHE = {}


def get_models_dir():
    return folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)[0]


def scan_models():
    files = []
    model_dir = get_models_dir()
    for ext in ("*.pth", "*.safetensors"):
        files.extend(glob.glob(os.path.join(model_dir, ext)))
    names = sorted(os.path.basename(f) for f in files)
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


def _detect_arch(sd):
    """从 state_dict 推断模型结构参数。"""
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

    # 检测通道数
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    # 检测 in_blocks 和 out_blocks 数量
    in_ids = set()
    out_ids = set()
    temporal_in_indices = set()
    temporal_out_indices = set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m:
            out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m:
            temporal_in_indices.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m:
            temporal_out_indices.add(int(m.group(1)))

    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)

    # 检测 temporal 配置
    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2  # 训练默认
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                kernel_t = sd[k].shape[2]
                cfg["temporal_kernel"] = kernel_t
                break
    else:
        cfg["temporal_every"] = 0  # 无 temporal

    # 推理时为了性能和稳定性，强制 attn=False
    cfg["attn"] = False

    return cfg


def _detect_dtype(sd):
    """从 state_dict 自动检测模型精度（跳过整数/布尔等非浮点张量）。"""
    for v in sd.values():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            return v.dtype
    return torch.float32


def load_model(name, device):
    cache_key = f"{name}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    path = os.path.join(get_models_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    raw_sd = _load_raw_sd(path)
    up_sd = _extract_upscaler_sd(raw_sd)

    cfg = _detect_arch(up_sd)

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

    MODEL_CACHE[cache_key] = model

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
                    "tooltip": "Minimax H3 latent，可为纯视频 latent (B,C,T,H,W) 或含音频流的联合 AV latent（NestedTensor）。",
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
                    "tooltip": "空间放大倍数（1.0~4.0）。1.0 原样返回；小于 1.0 会报错（仅支持放大）。仅在「按倍数缩放」模式下生效。",
                }),
                "width": ("INT", {
                    "default": 1920, "min": 64, "max": 8192, "step": 8,
                    "display_name": "目标宽度",
                    "tooltip": "目标宽度（像素），自动对齐到 16 的倍数。仅支持放大：目标小于当前尺寸会报错。仅在「目标尺寸」模式下生效。",
                }),
                "height": ("INT", {
                    "default": 1080, "min": 64, "max": 8192, "step": 8,
                    "display_name": "目标高度",
                    "tooltip": "目标高度（像素），自动对齐到 16 的倍数。仅支持放大：目标小于当前尺寸会报错。仅在「目标尺寸」模式下生效。",
                }),
            },
            "optional": {
                "conditioning": ("CONDITIONING", {
                    "display_name": "条件 (Conditioning)",
                    "tooltip": "可选。如接入 Conditioning，会自动将其中的首尾帧关键帧及参考图 Latent 缩放到与当前目标尺寸一致，避免二采报错。",
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING")
    RETURN_NAMES = ("潜空间", "条件")
    OUTPUT_TOOLTIPS = ("放大后的 H3 latent，可直接送 VAE 解码或二次采样重绘。", "对齐当前放大尺寸后的 Conditioning（若未输入则原样返回或为 None）。")
    FUNCTION = "run"
    CATEGORY = "Yuan Tool/放大"
    DESCRIPTION = ("H3 放大：在 latent 空间用训练好的神经网络放大 Minimax H3 视频，"
                   "跳过「解码→像素放大→再编码」的慢速往返。纯 3D 卷积主干，"
                   "联合处理时空体、时间一致性更强。支持按倍数缩放或目标尺寸两种"
                   "缩放方式，仅支持放大（scale ≥ 1.0 或目标 ≥ 当前尺寸）。"
                   "设备（优先 CUDA）与推理精度（跟随模型权重）均自动检测。")

    def run(self, latent, model_name, resize_type, scale, width, height, conditioning=None):
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
                return (latent, conditioning)
            if scale < 1.0:
                raise ValueError("仅支持放大 (scale >= 1.0)")
            target_size = (cur_t, int(round(cur_h * scale)), int(round(cur_w * scale)))
            eff_scale = float(scale)
        else:
            # 目标尺寸（像素）对齐到 16 的倍数 → latent 尺寸
            t_w = max(16, round(int(width) / 16) * 16) // 16
            t_h = max(16, round(int(height) / 16) * 16) // 16
            if t_h < cur_h or t_w < cur_w:
                raise ValueError(
                    f"目标尺寸小于当前尺寸，仅支持放大（当前 latent {cur_w}x{cur_h}，"
                    f"目标 {t_w}x{t_h}，像素 {t_w*16}x{t_h*16}）")
            if t_h == cur_h and t_w == cur_w:
                return (latent, conditioning)
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
            out = model(s, scale=eff_scale, target_size=target_size)

        # 反归一化
        out = out * norm_std + norm_mean

        # 还原维度
        if not is_nested and len(samples.shape) == 4:
            out = out.squeeze(2)

        out = out.cpu().to(orig_dtype)

        if dev.type == "cuda":
            torch.cuda.empty_cache()

        if is_nested:
            streams[0] = out
            out_latent = {"samples": comfy.nested_tensor.NestedTensor(streams)}
        else:
            out_latent = {"samples": out}

        out_conditioning = conditioning
        if conditioning is not None:
            out_conditioning = []
            target_h, target_w = target_size[1], target_size[2]
            for t in conditioning:
                n = [t[0], t[1].copy()]
                # 处理 keyframes 中的条件 latent 尺寸匹配
                if "minimax_keyframes" in n[1]:
                    new_keyframes = []
                    for kf in n[1]["minimax_keyframes"]:
                        kf_copy = kf.copy()
                        if "latent" in kf_copy and isinstance(kf_copy["latent"], torch.Tensor):
                            z = kf_copy["latent"]
                            # z 形状为 [B, C, T, H, W] 或 [B, C, H, W]
                            if z.shape[-2] != target_h or z.shape[-1] != target_w:
                                z_5d = z if z.ndim == 5 else z.unsqueeze(2)
                                z_resized = F.interpolate(
                                    z_5d, size=(z_5d.shape[2], target_h, target_w),
                                    mode="trilinear", align_corners=False
                                )
                                kf_copy["latent"] = z_resized if z.ndim == 5 else z_resized.squeeze(2)
                        new_keyframes.append(kf_copy)
                    n[1]["minimax_keyframes"] = new_keyframes

                # 处理 refs 中的图片参考 latent 尺寸匹配
                if "minimax_refs" in n[1]:
                    new_refs = []
                    for ref in n[1]["minimax_refs"]:
                        ref_copy = ref.copy()
                        if ref_copy.get("kind") == "image" and "latent" in ref_copy and isinstance(ref_copy["latent"], torch.Tensor):
                            z = ref_copy["latent"]
                            if z.shape[-2] != target_h or z.shape[-1] != target_w:
                                z_5d = z if z.ndim == 5 else z.unsqueeze(2)
                                z_resized = F.interpolate(
                                    z_5d, size=(z_5d.shape[2], target_h, target_w),
                                    mode="trilinear", align_corners=False
                                )
                                ref_copy["latent"] = z_resized if z.ndim == 5 else z_resized.squeeze(2)
                                ref_copy["latent_h"] = target_h
                                ref_copy["latent_w"] = target_w
                        new_refs.append(ref_copy)
                    n[1]["minimax_refs"] = new_refs

                out_conditioning.append(n)

        return (out_latent, out_conditioning)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "Yuan_H3Upscale3D": Yuan_H3Upscale3D,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_H3Upscale3D": "H3 放大",
}
