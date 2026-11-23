"""H3 渐进式采样器（MiniMax H3）。

在空间降采样的 latent 上跑前段采样步，把干净端点提升到目标分辨率，在过渡 sigma 处
重新加噪，再以全分辨率跑完剩余调度。采样只使用正条件、不做 CFG。
"""

import math
import os

import torch

import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy_extras.nodes_custom_sampler import Guider_Basic

from . import selflift
from . import h3_upscaler
from . import h3_tiling


def _upscaler_input():
    """构造高提升器下拉框选项；默认选中检测到的第一个 H3 模型。"""
    models = ["none"] + h3_upscaler.list_upscaler_models()
    h3_models = [name for name in models[1:] if "h3" in name.lower()]
    default = h3_models[0] if h3_models else "none"
    return (models, {"default": default, "display_name": "提升器模型",
                     "tooltip": "外部 H3 latent 提升器（models/latent_upscale_models）。默认选中检测到的第一个 H3 模型；选择 none 表示使用最近邻提升。"})


def _streams(samples):
    """把 NestedTensor 拆成多路流；普通张量视为单路。"""
    if samples.is_nested:
        return list(samples.unbind()), True
    return [samples], False


def _pack(streams, nested):
    """按输入是否为嵌套张量，把多路流重新打包。"""
    if nested:
        return comfy.nested_tensor.NestedTensor(streams)
    return streams[0]


# Sigma 加密的固定参数，不暴露给用户，只由开关决定是否启用。
SIGMA_REFINER_EXTRA_STEPS = 1
SIGMA_REFINER_START_AT_SIGMA = 0.70
SIGMA_REFINER_END_AT_SIGMA = 0.00


def _refine_sigmas(sigmas):
    """按固定参数加密 sigmas 的低噪尾部；未命中阈值时原样返回。"""
    sigmas_cpu = sigmas.detach().cpu()

    # 第一个 sigma <= start_at_sigma 的位置
    idx = -1
    for i, s in enumerate(sigmas_cpu):
        if s <= SIGMA_REFINER_START_AT_SIGMA:
            idx = i
            break
    if idx == -1 or idx >= len(sigmas_cpu) - 1:
        return sigmas

    head = sigmas_cpu[:idx]
    start = sigmas_cpu[idx].item()
    end = max(SIGMA_REFINER_END_AT_SIGMA, sigmas_cpu[-1].item())
    tail_len = len(sigmas_cpu) - idx + SIGMA_REFINER_EXTRA_STEPS

    # cosine 插值因子：趋近 0 时分布更密
    t = torch.linspace(0.0, 1.0, steps=tail_len)
    factor = (1.0 - torch.cos(t * math.pi)) / 2.0
    tail = start + (end - start) * factor
    return torch.cat([head, tail]).to(device=sigmas.device, dtype=sigmas.dtype)


def _validate_sampling(model_sampling, sampler):
    if not isinstance(model_sampling, comfy.model_sampling.CONST):
        raise ValueError("H3 渐进式采样器：需要 rectified-flow 模型")
    if not isinstance(sampler, comfy.samplers.KSAMPLER) or sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler:
        raise ValueError("H3 渐进式采样器：仅支持标准 Euler 采样器")
    if sampler.extra_options.get("s_churn", 0.0) != 0.0:
        raise ValueError("H3 渐进式采样器：需要 s_churn=0 的 Euler")


def _validate_schedule(sigmas, transition_step):
    if sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("H3 渐进式采样器：sigmas 必须是一维浮点张量")
    if not torch.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError("H3 渐进式采样器：sigmas 必须有限且非负")
    if sigmas.numel() < 2:
        return
    if not isinstance(transition_step, int) or not 1 <= transition_step <= sigmas.numel() - 2:
        raise ValueError(f"H3 渐进式采样器：过渡步数 {transition_step} 超出范围（共 {sigmas.numel() - 1} 步）")
    if (sigmas[1:] > sigmas[:-1]).any():
        raise ValueError("H3 渐进式采样器：sigmas 必须单调不增")
    if (sigmas[:-1] <= 0).any():
        raise ValueError("H3 渐进式采样器：只有最后一个 sigma 可以为 0")
    if sigmas[transition_step] >= 1:
        raise ValueError("H3 渐进式采样器：高分辨率起始 sigma 必须小于 1")


def _validate_latent_input(latent_image):
    """校验尺寸模板；允许携带 noise_mask 与掩码区域的初值（引导+重绘）。"""
    mask = latent_image.get("noise_mask")
    if mask is not None and getattr(mask, "is_nested", False):
        raise ValueError("H3 渐进式采样器：noise_mask 需为单路张量（视频流），音频流按全 1 "
                         "处理；请连接「H3 运动上下文」的「潜空间」输出")
    streams, _ = _streams(latent_image["samples"])
    if not streams or streams[0].ndim not in (4, 5):
        raise ValueError("H3 渐进式采样器：需要 4D 图像或 5D 视频 latent 尺寸模板")
    for stream in streams:
        if stream.ndim == 0 or any(size == 0 for size in stream.shape) or stream.shape[0] != streams[0].shape[0]:
            raise ValueError("H3 渐进式采样器：各路 latent 流都必须非空，且 batch 大小一致")
    if mask is None and any(bool(torch.count_nonzero(stream)) for stream in streams):
        raise ValueError("H3 渐进式采样器：潜空间带内容但没有 noise_mask。本节点只在掩码区域"
                         "使用初值，其余区域按全零重新生成；请连接「H3 运动上下文」的"
                         "「潜空间」输出，或改用不带内容的空 latent")


def _pin_into_resume(resume_latent, init_samples, denoise_mask):
    """把掩码区域（mask<1）的初值写回提升后的 latent，返回保留式重绘的基准。

    高分阶段需要同时成立两件事：起始 latent 带初值，且 KSamplerX0Inpaint 的混合
    目标（latent_image）在保留区就是该初值。这里按掩码硬替换，保留区因此精确等于
    初值，不再引入二次混合；其余区域保留分辨率提升后的生成结果。
    """
    resume_streams, nested = _streams(resume_latent)
    init_streams, _ = _streams(init_samples)
    if len(resume_streams) != len(init_streams):
        raise ValueError("H3 渐进式采样器：初值 latent 与目标 latent 的流数量不一致")
    pinned = []
    for resume, init in zip(resume_streams, init_streams):
        if tuple(resume.shape) != tuple(init.shape):
            raise ValueError(
                "H3 渐进式采样器：初值流形状 %s 与目标流 %s 不一致；引导+重绘要求两片段"
                "分辨率、时长与音频轨长度一致" % (tuple(init.shape), tuple(resume.shape)))
        if init.ndim != 5:
            pinned.append(resume)          # 音频等非视频流不参与保留
            continue
        # 掩码扩张到 latent 形状，并统一到 resume 的设备：初值张量可能仍在 CPU
        keep_mask = comfy.utils.reshape_mask(denoise_mask, init.shape)
        keep_mask = keep_mask.to(device=resume.device)
        keep = keep_mask < 1.0 - 1e-6
        init_dev = init.to(device=resume.device, dtype=resume.dtype)
        pinned.append(torch.where(keep, init_dev, resume))
    return _pack(pinned, nested)


def _euler_step(state, denoised, sigma, sigma_next):
    step = ((sigma_next - sigma) / sigma).to(device=state.device, dtype=state.dtype)
    return state + (state - denoised.to(state)) * step


def _resize_keyframes(cond, h, w):
    """关键帧条件的 latent 与生成网格同尺寸，这里缩放到低分辨率网格。

    逐帧做空间缩放，绝不在时间维上插值；缩放后补一次逐通道均值**与方差**匹配。
    只补均值会把锚点的对比度削平（bilinear 收缩方差），低分前缀因此拿到一个
    "均值对、锐度平"的锚点，高分收尾再把它锐回来，观感上就是锚点附近更容易
    出硬边和高光截断；方差回填让低分锚点保持原来的能量。
    """
    out = []
    for tensor, d in cond:
        kfs = d.get("minimax_keyframes")
        if kfs is None:
            out.append((tensor, d))
            continue
        d = d.copy()
        resized = []
        for kf in kfs:
            kf = dict(kf)
            lat = kf.get("latent")
            if lat is not None and (lat.shape[-2] != h or lat.shape[-1] != w):
                if lat.ndim == 5:
                    batch, channels, frames = lat.shape[:3]
                    resized_latent = torch.nn.functional.interpolate(
                        lat.float().permute(0, 2, 1, 3, 4).reshape(
                            batch * frames, channels, lat.shape[-2], lat.shape[-1]),
                        size=(h, w), mode="bilinear", align_corners=False
                    )
                    resized_latent = resized_latent.reshape(batch, frames, channels, h, w).permute(0, 2, 1, 3, 4)
                else:
                    resized_latent = torch.nn.functional.interpolate(
                        lat.float(), size=(h, w), mode="bilinear", align_corners=False
                    )
                source_mean = lat.float().mean(dim=(-2, -1), keepdim=True)
                source_std = lat.float().std(dim=(-2, -1), keepdim=True)
                resized_mean = resized_latent.mean(dim=(-2, -1), keepdim=True)
                resized_std = resized_latent.std(dim=(-2, -1), keepdim=True)
                scale = (source_std / resized_std.clamp_min(1e-4)).clamp(0.5, 2.0)
                kf["latent"] = ((resized_latent - resized_mean) * scale
                                + source_mean).to(lat)
            resized.append(kf)
        d["minimax_keyframes"] = resized
        out.append((tensor, d))
    return out


def _debug_dump(vae, latents):
    """YUAN_SELFLIFT_DEBUG=1 时把过渡中间结果解码成 PNG。"""
    if os.environ.get("YUAN_SELFLIFT_DEBUG", "0") != "1":
        return
    out_dir = os.path.join(os.path.dirname(__file__), "debug")
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image
    for name, lat in latents.items():
        if lat is None:
            continue
        img = vae.decode(lat)
        if img.ndim == 5:
            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
        frame = (img[0].float().cpu().numpy().clip(0.0, 1.0) * 255).round().astype("uint8")
        Image.fromarray(frame).save(os.path.join(out_dir, name + ".png"))


def _basic_guider_sample(model, noise, positive, sampler, sigmas, latent_image,
                         callback, disable_pbar, seed, denoise_mask=None):
    """用原生“基本引导器”（Guider_Basic）语义采样：只使用正条件、不做 CFG。

    denoise_mask 非空时走 ComfyUI 的保留式重绘（KSamplerX0Inpaint）：mask=1
    完全重绘、mask=0 完全保留 latent_image。
    """
    guider = Guider_Basic(model)
    guider.set_conds(positive)
    return guider.sample(noise, latent_image, sampler, sigmas, denoise_mask=denoise_mask,
                         callback=callback, disable_pbar=disable_pbar, seed=seed)


def progressive_sample(low_res_model, positive, vae, latent_image, sampler, sigmas, seed,
                       transition_step, lowres_scale, rho, w_min, w_max, latent_upsample, latent_lifter=None,
                       highres_tiling=False, h3_sigma_refiner=False, high_res_model=None):
    """渐进分辨率采样主体流程。

    低分辨率前缀 → 过渡（预测端点、成对提升、伪影修正、重新加噪）→ 全分辨率收尾，
    整个调度保持原有 NFE 次数。低分辨率前缀使用 low_res_model；分辨率提升后的收尾使用
    high_res_model，未接入时沿用 low_res_model（与 SelfLift 双模型语义一致）。
    """
    if h3_sigma_refiner:
        sigmas = _refine_sigmas(sigmas)
    _validate_schedule(sigmas, transition_step)
    if sigmas.numel() < 2:
        return latent_image
    if not 0.25 <= lowres_scale <= 1.0:
        raise ValueError("H3 渐进式采样器：低分辨率比例必须在 0.25 到 1 之间")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("H3 渐进式采样器：修正比例必须在 0 到 1 之间")
    if not 0.0 <= w_min <= w_max <= 1.0:
        raise ValueError("H3 渐进式采样器：权重必须满足 0 <= 修正强度下限 <= 修正强度上限 <= 1")
    _validate_latent_input(latent_image)

    model = low_res_model
    model_sampling = model.get_model_object("model_sampling")
    _validate_sampling(model_sampling, sampler)

    streams, nested = _streams(comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"], latent_image.get("downscale_ratio_spacial", None),
        latent_image.get("downscale_ratio_temporal", None)))
    # 高分辨率收尾阶段使用的模型：接入 high_res_model 时用高分辨率模型，未接入时沿用低分辨率模型
    hires_base = high_res_model if high_res_model is not None else model
    _validate_sampling(hires_base.get_model_object("model_sampling"), sampler)
    high_model = h3_tiling.tiled_model(hires_base, [tuple(stream.shape) for stream in streams]) if highres_tiling else hires_base
    video = streams[0].ndim == 5
    if video:
        b, c, t, H, W = streams[0].shape
    else:
        b, c, H, W = streams[0].shape
        t = None
    # 低分辨率空间尺寸按 2 取整，保证 latent 网格可被整除
    h = max(2, round(H * lowres_scale / 2) * 2)
    w = max(2, round(W * lowres_scale / 2) * 2)
    low_shape = (b, c, t, h, w) if video else (b, c, h, w)

    device = comfy.model_management.intermediate_device()
    # 低分辨率阶段从全零 latent 开始；音频流无空间维度，沿用零张量
    low_latent = _pack([torch.zeros(low_shape, device=device)] +
                       [torch.zeros_like(s) for s in streams[1:]], nested)
    del streams
    noise_low = comfy.sample.prepare_noise(low_latent, seed, latent_image.get("batch_index", None))

    total_steps = sigmas.shape[-1] - 1
    callback = latent_preview.prepare_callback(model, total_steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    # 关键帧条件 latent 与生成网格同尺寸，低分辨率阶段需同步缩放（4D/5D 均支持）
    positive_low = _resize_keyframes(positive, h, w)

    transition = {}
    low_evaluations = 0

    def callback_low(step, x0, x, total):
        nonlocal low_evaluations
        # 用本地计数，避免包装器带偏移的步号影响过渡位置
        step = low_evaluations
        low_evaluations += 1
        if low_evaluations > transition_step:
            raise RuntimeError("H3 渐进式采样器：低分辨率回调次数超过 Euler 调度长度")
        if step == transition_step - 1:
            # 记下过渡前的状态与干净预测，供过渡阶段复用
            transition["state"] = x
            transition["x0"] = x0
        result = callback(step, x0, x, total_steps)
        return result

    # 最后一次低分辨率评估的 Euler 更新会被丢弃，过渡后重建，从而保持 NFE 数量不变
    # 低分辨率前缀同样要固定：锚与掩码一起空间降采样到低分网格。若低分阶段自由生成，
    # 过渡提升后头部细节已经被抹平，高分阶段仅剩的几步补不回来（实测亮度阶跃 ~25%）。
    _low_anchor = low_latent
    _low_mask = None
    _mask_in = latent_image.get("noise_mask")
    if _mask_in is not None:
        _anchor_streams, _ = _streams(latent_image["samples"])
        _pin_v, _pin_rest = _anchor_streams[0], _anchor_streams[1:]
        _pin_low = torch.nn.functional.interpolate(
            _pin_v.float(), size=(t, h, w), mode="nearest").to(_pin_v.dtype)
        _low_anchor = _pack([_pin_low] + [torch.zeros_like(s) for s in _pin_rest], nested)
        _low_mask = torch.nn.functional.interpolate(
            _mask_in.float(), size=(t, h, w), mode="nearest")
        if os.environ.get("YUAN_H3_DBG", "0") == "1":
            print("[h3dbg] 低分阶段固定保留: anchor=%s mask=%s min=%.4f" % (
                tuple(_pin_low.shape), tuple(_low_mask.shape),
                float(_low_mask.min())), flush=True)
        del _pin_v, _pin_rest, _pin_low
    _basic_guider_sample(model, noise_low, positive_low, sampler, sigmas[:transition_step + 1],
                         _low_anchor, callback_low, disable_pbar, seed,
                         denoise_mask=_low_mask)
    if low_evaluations != transition_step:
        raise RuntimeError(f"H3 渐进式采样器：预期 {transition_step} 次低分辨率回调，实际 {low_evaluations} 次；请检查采样器包装器")
    low_streams, nested = _streams(transition.pop("state"))
    x0_streams, _ = _streams(transition.pop("x0"))
    sigma_k = sigmas[transition_step - 1]
    sigma_next = sigmas[transition_step]
    # 音频等辅助流不做空间提升，但仍按被复用的 Euler 边界推进
    auxiliary_next = [_euler_step(state.to(device), denoised.to(device), sigma_k, sigma_next)
                      for state, denoised in zip(low_streams[1:], x0_streams[1:])]
    latent_format = model.get_model_object("latent_format")
    z0_low_vae = latent_format.process_out(x0_streams[0].float()).to(device)
    del low_latent, noise_low, low_streams, x0_streams, positive_low

    # 伪影感知一致性提升；被权重丢弃的分支直接跳过
    need_pix = rho > 0.0 and w_max > 0.0
    need_lat = not (rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0)
    z_lat_vae, z_pix_vae = selflift.paired_lifts(
        z0_low_vae, vae, (H, W), latent_upsample, latent_lifter,
        need_lat=need_lat, need_pix=need_pix)
    z_lat = latent_format.process_in(z_lat_vae) if z_lat_vae is not None else None
    z_pix = latent_format.process_in(z_pix_vae) if z_pix_vae is not None else None
    z0_high = selflift.artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max)
    if os.environ.get("YUAN_SELFLIFT_DEBUG", "0") == "1":
        _debug_dump(vae, {
            "z0_low": z0_low_vae,
            "z_lat": z_lat_vae,
            "z_pix": z_pix_vae,
            "z0_high": latent_format.process_out(z0_high),
        })
    z0_high = z0_high.to(device)
    del z0_low_vae, z_lat_vae, z_pix_vae, z_lat, z_pix

    # 在被复用的 sigma 上对修正后的端点重新加噪，不调用 denoiser 直接走完该 Euler 区间
    video_noise = comfy.sample.prepare_noise(z0_high, (seed + 1) % (1 << 64),
                                             latent_image.get("batch_index", None)).to(z0_high)
    video_state = model_sampling.noise_scaling(sigma_k, video_noise, z0_high)
    next_streams = [_euler_step(video_state, z0_high, sigma_k, sigma_next)] + auxiliary_next
    del z0_high, video_noise, video_state, auxiliary_next
    resume_streams = [model_sampling.inverse_noise_scaling(sigma_next, s) for s in next_streams]
    resume_latent = model.model.process_latent_out(_pack(resume_streams, nested))
    resume_noise = _pack([torch.zeros_like(s) for s in resume_streams], nested)
    # 引导+重绘：只在分辨率提升后的高分阶段 固定 头部。低分前缀若也 pin，头部细节会先
    # 被空间下采样再提升回来，正是要避免的劣化，因此低分前缀保持自由生成。
    denoise_mask = latent_image.get("noise_mask")
    if denoise_mask is not None:
        resume_latent = _pin_into_resume(resume_latent, latent_image["samples"],
                                         denoise_mask)
    if os.environ.get("YUAN_H3_DBG", "0") == "1":
        _ss, _ = _streams(resume_latent)
        _ii, _ = _streams(latent_image["samples"])
        _m12 = (denoise_mask[0, 0, :12, 0, 0].tolist()
                if denoise_mask is not None and denoise_mask.ndim == 5 else None)
        print("[h3dbg] mask=%s | sigma0=%.3f last=%.3f hi_steps=%d | resume_v=%.3f "
              "init_v=%.3f(%s) | pin写入后 头10步最大差=%.6f | mask前12步=%s" % (
                  None if denoise_mask is None else tuple(denoise_mask.shape),
                  float(sigmas[transition_step]), float(sigmas[-1]),
                  int(len(sigmas) - transition_step),
                  float(_ss[0].float().mean()), float(_ii[0].float().mean()),
                  str(_ii[0].device),
                  float((_ss[0][:, :, :10].float().cpu()
                         - _ii[0][:, :, :10].float().cpu()).abs().max()),
                  [round(x, 3) for x in _m12] if _m12 else None), flush=True)
    del next_streams, resume_streams
    # 回收过渡阶段的缓存块，降低高分辨率阶段碎片化 OOM 的风险
    comfy.model_management.soft_empty_cache()

    high_evaluations = 0

    def callback_high(step, x0, x, total):
        nonlocal high_evaluations
        step = high_evaluations
        high_evaluations += 1
        if high_evaluations > total_steps - transition_step:
            raise RuntimeError("H3 渐进式采样器：高分辨率回调次数超过 Euler 调度长度")
        # 进度条步号累加低分辨率已完成的步数，保持整体连续
        result = callback(step + transition_step, x0, x, total_steps)
        return result

    out = _basic_guider_sample(high_model, resume_noise, positive, sampler, sigmas[transition_step:],
                               resume_latent, callback_high, disable_pbar, seed,
                               denoise_mask=denoise_mask)
    if os.environ.get("YUAN_H3_DBG", "0") == "1":
        _os, _ = _streams(out)
        _ii, _ = _streams(latent_image["samples"])
        print("[h3dbg] 采样后: out_head==pin头 最大差=%.5f | out_v=%.3f pin_v=%.3f" % (
            float((_os[0][:, :, :10].float().cpu()
                   - _ii[0][:, :, :10].float().cpu()).abs().max()),
            float(_os[0].float().mean()), float(_ii[0].float().mean())), flush=True)
    del resume_latent, resume_noise
    if high_evaluations != total_steps - transition_step:
        raise RuntimeError(f"H3 渐进式采样器：预期 {total_steps - transition_step} 次高分辨率回调，实际 {high_evaluations} 次；请检查采样器包装器")

    result = latent_image.copy()
    result["samples"] = out.to(device=comfy.model_management.intermediate_device(),
                                dtype=comfy.model_management.intermediate_dtype())
    return result


class Yuan_H3ProgressiveSampler:
    """MiniMax H3 音视频渐进分辨率采样节点。

    采样只使用正条件、不做 CFG，因此没有负条件输入端口与 cfg 设置。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "low_res_model": ("MODEL", {"display_name": "低分辨率模型",
                "tooltip": "低分辨率前缀使用的模型：前 transition_step 次去噪评估。未接入高分辨率模型时，本模型也负责分辨率提升后的收尾阶段。"}),
            "positive": ("CONDITIONING", {"display_name": "正向条件",
                "tooltip": "唯一条件输入。本节点按原生“基本引导器”语义采样：只使用正条件、不做 CFG，因此不需要负条件与 cfg。"}),
            "vae": ("VAE", {"display_name": "视频VAE",
                "tooltip": "在分辨率过渡点用于像素重编码锚点的视频 VAE。"}),
            "latent_image": ("LATENT", {"display_name": "潜空间",
                "tooltip": "目标分辨率 latent（如 Empty MiniMax H3 AV Latent），用于定义尺寸与时长。"
                           "支持「H3 运动上下文」的「潜空间」输出：带 noise_mask 时只在掩码区域"
                           "使用初值，其余区域自由生成（引导+重绘）；不带掩码时按全零模板处理。"
                           "关键帧请通过条件化传入。"}),
            "sampler": ("SAMPLER", {"display_name": "采样器",
                "tooltip": "仅支持标准 Euler；本节点会复用其在过渡步的预测，以保持原有 NFE 次数。"}),
            "sigmas": ("SIGMAS", {"display_name": "噪声调度"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                "display_name": "种子"}),
            "transition_step": ("INT", {"default": 6, "min": 1, "max": 10000, "display_name": "过渡步数",
                "tooltip": "在低分辨率执行的 denoiser 评估次数；H3 需要自行验证取值。"}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05,
                "display_name": "低分辨率比例", "tooltip": "低分辨率前缀的空间缩放比例。"}),
            "rho": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "display_name": "修正比例",
                "tooltip": "向像素-VAE 锚点修正的最高风险时空位置比例。默认 0 表示只走外部 latent 提升器并跳过 VAE 往返；若用提升器模型=none 跑 SelfLift-zero，建议从 0.6 附近开始。"}),
            "w_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                "display_name": "修正强度下限",
                "tooltip": "修正强度下限。H3 中广泛存在的最近邻提升误差可能需要 1.0。"}),
            "w_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "display_name": "修正强度上限",
                "tooltip": "修正强度上限。SelfLift-zero 诊断建议保持 1.0。"}),
            "upscaler_model": _upscaler_input(),
        }, "optional": {
            "high_res_model": ("MODEL", {"display_name": "高分辨率模型",
                "tooltip": "高分辨率阶段模型：分辨率提升后的收尾采样使用。不接时沿用低分辨率模型。两阶段必须使用兼容的架构、latent 格式、采样参数化与 VAE。"}),
            "highres_tiling": ("BOOLEAN", {"default": False, "label_on": "高分辨率分块：开启",
                "label_off": "高分辨率分块：关闭", "display_name": "高分辨率分块",
                "tooltip": "实验性：在高分辨率准备阶段根据可用显存选择 1–8 个空间分块。音频输入与参考条件保持完整，音频预测取各块平均。质量与速度可能变化。"}),
            "h3_sigma_refiner": ("BOOLEAN", {"default": False, "label_on": "Sigma 加密：开启",
                "label_off": "Sigma 加密：关闭", "display_name": "Sigma 加密",
                "tooltip": "开启后先对传入 sigmas 做低噪尾部加密，消除高速运动边缘的颗粒与闪烁；关闭则直接使用传入的 sigmas。参数固定为额外步数=1、起始 sigma=0.70、结束 sigma=0.00、间隔=cosine。"}),
            "upscaler_unload": ("BOOLEAN", {"default": True, "label_on": "放大后卸载：开启",
                "label_off": "放大后卸载：关闭", "display_name": "放大后卸载",
                "tooltip": "提升结束后立即把外部提升器从显存卸载，再进入高分辨率阶段。仅在反复复用提升器且显存宽裕时才需要关闭。"}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    OUTPUT_TOOLTIPS = ("渐进式采样得到的 H3 联合潜空间（视频流 + 音频流）。",)
    FUNCTION = "sample"
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("H3 渐进式采样器：先在空间降采样的 latent 上跑前段采样步，把干净端点提升到"
                   "目标分辨率并在过渡 sigma 处重新加噪，再以全分辨率跑完剩余调度。采样只使用"
                   "正条件、不做 CFG；过渡点可选像素-VAE 锚点修正，用于抑制提升引入的伪影。"
                   "支持双模型接入：低分辨率前缀用低分辨率模型，收尾阶段可另接高分辨率模型，"
                   "不接时沿用低分辨率模型。")

    def sample(self, low_res_model, positive, vae, latent_image, sampler, sigmas, seed,
               transition_step, lowres_scale, rho, w_min, w_max, upscaler_model,
               high_res_model=None, highres_tiling=False, h3_sigma_refiner=False, upscaler_unload=True):
        if rho == 0.0 and upscaler_model == "none":
            raise ValueError("H3 渐进式采样器：修正比例为 0 且提升器模型为 none 时，SelfLift-zero 修正与外部"
                             "提升都被关闭，等于什么都没做。请把修正比例设为大于 0，或选择一个外部 H3 提升器。")
        lifter = None
        if upscaler_model != "none":
            lifter = lambda z, hw: h3_upscaler.learned_latent_lift(
                z, hw, upscaler_model, force_unload=upscaler_unload)
        return (progressive_sample(low_res_model, positive, vae, latent_image, sampler, sigmas, seed,
                                   transition_step, lowres_scale, rho, w_min, w_max, "nearest",
                                   latent_lifter=lifter, highres_tiling=highres_tiling,
                                   h3_sigma_refiner=h3_sigma_refiner, high_res_model=high_res_model),)


NODE_CLASS_MAPPINGS = {
    "Yuan_H3ProgressiveSampler": Yuan_H3ProgressiveSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_H3ProgressiveSampler": "H3 渐进式采样器",
}
