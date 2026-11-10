"""SelfLift-zero：伪影感知一致性提升。

在过渡点把低分辨率干净端点提升到目标分辨率：直接提升与像素-VAE 重编码配对，
以两者残差作为伪影风险信号，只修正风险最高的一部分位置。
"""

import torch
import torch.nn.functional

import comfy.utils


def paired_lifts(z0_low, vae, out_hw, latent_mode="nearest", latent_lifter=None,
                 need_lat=True, need_pix=True):
    """构造直接 latent 提升与像素-VAE 重编码这一对结果。

    z0_low：[B, C, T, h, w] 视频或 [B, C, h, w] 图像，位于 VAE 原生 latent 空间的干净预测。
    latent_lifter：可选的学习式提升可调用对象 (z0_low, out_hw) -> latent。
    need_lat/need_pix 用于跳过会被权重直接丢弃的分支。
    返回 (z_lat, z_pix)，被跳过的分支为 None。
    """
    H, W = out_hw
    if z0_low.ndim == 4:  # 图像 latent
        z_lat = None
        if need_lat:
            z_lat = torch.nn.functional.interpolate(z0_low.float(), size=(H, W),
                                                    mode="nearest" if latent_mode == "nearest" else "bilinear")
        z_pix = None
        if need_pix:
            img = vae.decode(z0_low)  # [B, h*ratio, w*ratio, 3]
            ratio = img.shape[1] // z0_low.shape[-2]
            up = comfy.utils.common_upscale(img.movedim(-1, 1), W * ratio, H * ratio, "lanczos", "disabled").movedim(1, -1)
            del img
            z_pix = vae.encode(up).float()
            if z_lat is not None:
                z_lat = z_lat.to(z_pix.device)
        return z_lat, z_pix

    T = z0_low.shape[2]
    z_lat = None
    if need_lat:
        if latent_lifter is not None:
            z_lat = latent_lifter(z0_low, (H, W))
        else:
            if latent_mode == "bilinear":
                latent_mode = "trilinear"  # 5D 下换用同名插值方法
            z_lat = torch.nn.functional.interpolate(z0_low.float(), size=(T, H, W), mode=latent_mode)

    z_pix = None
    if need_pix:
        z_pix = _pixel_anchor_video(z0_low, vae, (H, W))
        if z_lat is not None:
            z_lat = z_lat.to(z_pix.device)
    return z_lat, z_pix


def _pixel_anchor_video(z0_low, vae, out_hw):
    """按批内每个视频独立构造像素锚点。"""
    if z0_low.shape[0] == 1:
        return _pixel_anchor_video_single(z0_low, vae, out_hw)
    return torch.cat([
        _pixel_anchor_video_single(sample, vae, out_hw)
        for sample in z0_low.split(1)
    ], dim=0)


def _pixel_anchor_video_single(z0_low, vae, out_hw):
    """视频像素锚点：解码低分辨率 → 上采样 → 重新编码。

    上采样按帧分块写入预分配缓冲，避免整段视频在 CPU 上物化多份副本。
    """
    H, W = out_hw
    frames = vae.decode(z0_low)
    if frames.ndim == 5:  # VAEDecode 约定：视频像素为帧批 [F, H, W, C]
        frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
    ratio = frames.shape[1] // z0_low.shape[-2]
    Hp, Wp = H * ratio, W * ratio
    work_dtype = vae.vae_dtype if vae.vae_dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    if vae.device.type == "cpu":
        work_dtype = torch.float32
    n = frames.shape[0]
    up = torch.empty((n, Hp, Wp, frames.shape[-1]), dtype=work_dtype)
    for i in range(0, n, 32):
        chunk = frames[i:i + 32].movedim(-1, 1).to(device=vae.device, dtype=work_dtype)
        chunk = torch.nn.functional.interpolate(chunk, size=(Hp, Wp), mode="bicubic", antialias=True)
        up[i:i + 32] = chunk.movedim(1, -1).to(up.device)
        del chunk
    del frames
    return vae.encode(up).float()  # 封装层会把帧批重新还原为时间维


def artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max):
    """把直接提升朝像素-VAE 锚点做选择性修正。"""
    if rho <= 0.0 or w_max <= 0.0:
        return z_lat
    if rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0:
        return z_pix
    delta = z_pix - z_lat
    s = delta.abs().mean(dim=1)  # 逐位置的不一致程度，[B, (T,) H, W]
    view = (-1,) + (1,) * (s.ndim - 1)
    flat = s.flatten(1)
    thr = torch.quantile(flat, 1.0 - rho, dim=1).view(view)
    mask = s >= thr
    s_min = s.masked_fill(~mask, float("inf")).flatten(1).amin(dim=1).view(view)
    s_max = s.masked_fill(~mask, float("-inf")).flatten(1).amax(dim=1).view(view)
    w = w_min + (w_max - w_min) * (s - s_min) / (s_max - s_min + 1e-8)
    w = torch.where(mask, w, torch.zeros_like(w)).unsqueeze(1)
    return z_lat + w * delta
