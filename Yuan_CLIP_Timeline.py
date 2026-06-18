"""
Yuan CLIP Timeline - 视觉时间轴提示词编码节点
复刻自 ComfyUI-PromptRelay 的 Prompt Relay Encode (Timeline) 节点
集成 video latent / audio latent 自动生成方法（与 LTXV 空潜空间原理一致）
"""

import json
import logging
import math
import re
import types

import torch
import comfy.ldm.modules.attention
import comfy.model_management

log = logging.getLogger(__name__)

# text_input 时间格式解析：匹配行首的 "0-3s", "3-5秒", "5-7s" 等
_TIME_RANGE_PATTERN = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*[s秒]\s*[：:]?\s*')


# ==============================================================================
# prompt_relay.py 核心函数
# ==============================================================================

def build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, tokens_per_frame):
    """为视频交叉注意力构建高斯惩罚矩阵 [Lq, Lk]（整数帧索引）。"""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.long) // tokens_per_frame

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames.float()[:, None] - seg["midpoint"]).abs()
        strength = seg.get("strength", 1.0)
        cost = strength * (torch.relu(d - seg["window"]) ** 2) / (2 * seg["sigma"] ** 2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


def build_temporal_cost_scaled(q_token_idx, Lq, Lk, device, dtype, latent_frames):
    """为非整数帧映射的查询构建惩罚矩阵（例如 LTXAV 音频 token）。"""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.float32) * latent_frames / Lq

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames[:, None] - seg["midpoint"]).abs()
        sigma_a = seg.get("sigma_audio", seg["sigma"])
        window_a = seg.get("window_audio", seg["window"])
        strength_a = seg.get("strength_audio", 1.0)
        cost = strength_a * (torch.relu(d - window_a) ** 2) / (2 * sigma_a ** 2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


def create_mask_fn(q_token_idx, fallback_tokens_per_frame, latent_frames):
    """闭包：mask_fn(Lq, Lk, dtype, device, transformer_options) -> 附加掩码或 None。"""
    cache = {}
    max_token_idx = max(int(seg["local_token_idx"].max().item()) for seg in q_token_idx) + 1

    def mask_fn(Lq, Lk, dtype, device, transformer_options):
        if Lq == Lk:
            return None

        cond_or_uncond = transformer_options.get("cond_or_uncond", [])
        if 1 in cond_or_uncond and 0 not in cond_or_uncond:
            return None

        grid_sizes = transformer_options.get("grid_sizes", None)
        video_tpf = int(grid_sizes[1]) * int(grid_sizes[2]) if grid_sizes is not None else fallback_tokens_per_frame
        video_lq = latent_frames * video_tpf

        if Lk == video_lq or Lk < max_token_idx:
            return None

        mode = "video" if Lq == video_lq else "scaled"

        key = (Lq, Lk, mode, device)
        if key not in cache:
            if mode == "video":
                cost = build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, video_tpf)
            else:
                cost = build_temporal_cost_scaled(q_token_idx, Lq, Lk, device, dtype, latent_frames)
            log.info(
                "[Yuan CLIP Timeline] 构建惩罚矩阵 (%s): Lq=%d, Lk=%d, 非零=%d/%d",
                mode, Lq, Lk, (cost > 0).sum().item(), cost.numel(),
            )
            cache[key] = -cost

        return cache[key].to(dtype)

    return mask_fn


def build_segments(token_ranges, segment_lengths, epsilon=1e-3, relay_options=None):
    """为时间惩罚构建每段元数据。"""
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448

    opts = relay_options or {}
    v_strength = opts.get("video_strength", 1.0)
    v_window_scale = opts.get("video_window_scale", 1.0)
    a_epsilon = opts.get("audio_epsilon")
    a_strength = opts.get("audio_strength", 1.0)
    a_window_scale = opts.get("audio_window_scale", 1.0)

    if a_epsilon is not None and 0 < a_epsilon < 1:
        sigma_audio = 1.0 / math.log(1.0 / a_epsilon)
    else:
        sigma_audio = sigma

    if relay_options:
        log.info(
            "[Yuan CLIP Timeline] 高级选项 - 视频: strength=%.3f window_scale=%.3f | "
            "音频: epsilon=%s strength=%.3f window_scale=%.3f",
            v_strength, v_window_scale,
            f"{a_epsilon:.4f}" if a_epsilon is not None else "继承",
            a_strength, a_window_scale,
        )

    q_token_idx = []
    frame_cursor = 0

    for (tok_start, tok_end), L in zip(token_ranges, segment_lengths):
        if L <= 0:
            frame_cursor += L
            continue
        midpoint = (2 * frame_cursor + L) // 2
        base_window = max(L // 2 - 2, 0)
        q_token_idx.append({
            "local_token_idx": torch.arange(tok_start, tok_end),
            "midpoint": midpoint,
            "window": max(base_window * v_window_scale, 0.0),
            "sigma": sigma,
            "strength": v_strength,
            "window_audio": max(base_window * a_window_scale, 0.0),
            "sigma_audio": sigma_audio,
            "strength_audio": a_strength,
        })
        frame_cursor += L

    return q_token_idx


def get_raw_tokenizer(clip):
    """从 ComfyUI CLIP 对象中提取原始 SPiece/HF 分词器。"""
    tokenizer_wrapper = clip.tokenizer
    for attr_name in dir(tokenizer_wrapper):
        if attr_name.startswith("_"):
            continue
        inner = getattr(tokenizer_wrapper, attr_name, None)
        if inner is not None and hasattr(inner, "tokenizer"):
            return inner.tokenizer

    raise RuntimeError(
        f"无法从 CLIP 对象中找到原始分词器。"
        f"已知属性: {[a for a in dir(tokenizer_wrapper) if not a.startswith('_')]}"
    )


def map_token_indices(raw_tokenizer, global_prompt, local_prompts):
    """对全局提示词和空格前缀的本地提示词进行分词；返回 (完整提示词, 每段本地 token 范围)。"""
    prefixed_locals = [" " + lp for lp in local_prompts]
    full_prompt = global_prompt + "".join(prefixed_locals)
    has_eos = getattr(raw_tokenizer, "add_eos", False)
    eos_adj = 1 if has_eos else 0

    prev_len = len(raw_tokenizer(global_prompt)["input_ids"]) - eos_adj
    token_ranges = []
    built = global_prompt

    for plp in prefixed_locals:
        built += plp
        cur_len = len(raw_tokenizer(built)["input_ids"]) - eos_adj
        if cur_len <= prev_len:
            raise ValueError(f"本地提示词未产生任何 token: '{plp.strip()}'")
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len

    return full_prompt, token_ranges


def distribute_segment_lengths(num_segments, latent_frames, specified_lengths=None):
    """验证或自动分布段帧数，限制在 latent_frames 范围内。"""
    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError(
                f"segment_lengths 数量 ({len(specified_lengths)}) "
                f"必须与本地提示词数量 ({num_segments}) 一致"
            )
        lengths = specified_lengths
    else:
        step = -(-latent_frames // num_segments)  # 向上取整除法
        lengths = [step] * num_segments

    effective = []
    for L in lengths:
        effective.append(max(1, min(L, latent_frames)))
    return effective


# ==============================================================================
# patches.py 模型补丁函数
# ==============================================================================

def _masked_attention(q, k, v, heads, mask, transformer_options={}, **kwargs):
    """绕过 wrap_attn，直接调用 attention_pytorch 以保留掩码。"""
    return comfy.ldm.modules.attention.attention_pytorch(
        q, k, v, heads, mask=mask,
        _inside_attn_wrapper=True,
        transformer_options=transformer_options,
        **kwargs,
    )


def _wan_t2v_forward(self, mask_fn, x, context, transformer_options={}, **kwargs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)

    mask = mask_fn(q.shape[1], k.shape[1], q.dtype, q.device, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )
    return self.o(x)


def _wan_i2v_forward(self, mask_fn, x, context, context_img_len, transformer_options={}, **kwargs):
    context_img = context[:, :context_img_len]
    context_text = context[:, context_img_len:]

    q = self.norm_q(self.q(x))

    k_img = self.norm_k_img(self.k_img(context_img))
    v_img = self.v_img(context_img)
    img_x = comfy.ldm.modules.attention.optimized_attention(
        q, k_img, v_img, heads=self.num_heads, transformer_options=transformer_options,
    )

    k = self.norm_k(self.k(context_text))
    v = self.v(context_text)

    mask = mask_fn(q.shape[1], k.shape[1], q.dtype, q.device, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )

    return self.o(x + img_x)


def _make_masked_override(prev_override):
    """transformer_options 覆盖，将带掩码的注意力调用路由到 attention_pytorch。"""
    def override(func, *args, **kwargs):
        if kwargs.get("mask") is not None:
            return comfy.ldm.modules.attention.attention_pytorch(*args, **kwargs)
        if prev_override is not None:
            return prev_override(func, *args, **kwargs)
        return func(*args, **kwargs)
    return override


def _make_ltx_mask_wrapper(underlying, mask_fn):
    """包装 LTX 交叉注意力 forward，注入 PromptRelay 的附加掩码。"""
    def wrapped(_self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
        if context is not None:
            pr_mask = mask_fn(x.shape[1], context.shape[1], x.dtype, x.device, transformer_options)
            if pr_mask is not None:
                mask = pr_mask if mask is None else mask + pr_mask

        if mask is not None:
            prev = transformer_options.get("optimized_attention_override")
            transformer_options = {
                **transformer_options,
                "optimized_attention_override": _make_masked_override(prev),
            }

        return underlying(
            x, context=context, mask=mask, pe=pe, k_pe=k_pe,
            transformer_options=transformer_options,
        )

    wrapped._promptrelay_wrapper = True
    return wrapped


class _CrossAttnPatch:
    """描述符，将 (impl, mask_fn) 绑定为 Wan 交叉注意力模块的方法。"""

    def __init__(self, impl, mask_fn):
        self.impl = impl
        self.mask_fn = mask_fn

    def __get__(self, obj, objtype=None):
        impl, mask_fn = self.impl, self.mask_fn

        def wrapped(self_module, *args, **kwargs):
            return impl(self_module, mask_fn, *args, **kwargs)

        return types.MethodType(wrapped, obj)


def detect_model_type(model):
    """返回 (架构, patch_size, temporal_stride) 用于潜空间几何信息。"""
    diff_model = model.model.diffusion_model

    if hasattr(diff_model, "patch_size") and not hasattr(diff_model, "patchifier"):
        return "wan", tuple(diff_model.patch_size), 4

    if hasattr(diff_model, "patchifier"):
        return "ltx", (1, 1, 1), int(diff_model.vae_scale_factors[0])

    raise ValueError(
        f"不支持的模型类型: {type(diff_model).__name__}。"
        f"目前支持 Wan 和 LTX 模型。"
    )


def _check_unpatched(model_clone, key):
    if key in getattr(model_clone, "object_patches", {}):
        raise RuntimeError(
            f"Yuan CLIP Timeline: '{key}' 处的交叉注意力已被其他节点补丁。"
            "此架构不支持叠加 — 请移除冲突节点。"
        )


def apply_patches(model_clone, arch, mask_fn):
    diffusion_model = model_clone.get_model_object("diffusion_model")

    if arch == "wan":
        from comfy.ldm.wan.model import WanI2VCrossAttention
        for idx, block in enumerate(diffusion_model.blocks):
            key = f"diffusion_model.blocks.{idx}.cross_attn.forward"
            _check_unpatched(model_clone, key)
            cross_attn = block.cross_attn
            impl = _wan_i2v_forward if isinstance(cross_attn, WanI2VCrossAttention) else _wan_t2v_forward
            model_clone.add_object_patch(key, _CrossAttnPatch(impl, mask_fn).__get__(cross_attn, cross_attn.__class__))
        return

    if arch == "ltx":
        to = model_clone.model_options["transformer_options"]
        to["promptrelay_mask_fn"] = mask_fn

        for idx, block in enumerate(diffusion_model.transformer_blocks):
            for attr in ("attn2", "audio_attn2"):
                module = getattr(block, attr, None)
                if module is None:
                    continue
                key = f"diffusion_model.transformer_blocks.{idx}.{attr}.forward"
                underlying = model_clone.get_model_object(key)
                wrapper = _make_ltx_mask_wrapper(underlying, mask_fn)
                model_clone.add_object_patch(key, types.MethodType(wrapper, module))
        return

    raise ValueError(f"未知模型架构: {arch}")


# ==============================================================================
# nodes.py 编码函数
# ==============================================================================

def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    """使用最大余数法将像素空间段长度转换为整数潜空间长度。"""
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


# ==============================================================================
# LTXV 空潜空间自动生成（video / audio 原理一致：零张量 + type 标记，由采样器加噪去噪）
# ==============================================================================

def _auto_generate_latent(width, height, length_frames):
    """自动生成 LTXV 兼容的视频空潜空间张量。
    LTXV 时间压缩: latent_t = ((length - 1) // 8) + 1
    零张量不含 noise_mask，采样器将对整个潜空间加噪并去噪生成新内容。
    """
    w = max(32, (width // 32) * 32)
    h = max(32, (height // 32) * 32)
    latent_t = ((length_frames - 1) // 8) + 1
    samples = torch.zeros(
        [1, 128, latent_t, h // 32, w // 32],
        device=comfy.model_management.intermediate_device(),
    )
    log.info(
        "[Yuan CLIP Timeline] 自动生成视频潜空间: %dx%d, %d 像素帧 (%d 潜空间帧)",
        w, h, length_frames, latent_t,
    )
    return {"samples": samples}


def _auto_generate_audio_latent(audio_vae, length_frames, frame_rate):
    """自动生成 LTXV 兼容的音频空潜空间张量。
    与 video latent 使用相同的 length_frames 和 frame_rate，保证帧对齐。
    零张量不含 noise_mask，采样器将对整个音频潜空间加噪并去噪生成新音频。
    """
    inner = getattr(audio_vae, "first_stage_model", audio_vae)
    z_channels = audio_vae.latent_channels
    audio_freq = inner.latent_frequency_bins
    num_audio_latents = inner.num_of_latents_from_frames(length_frames, float(frame_rate))

    samples = torch.zeros(
        (1, z_channels, num_audio_latents, audio_freq),
        device=comfy.model_management.intermediate_device(),
    )
    log.info(
        "[Yuan CLIP Timeline] 自动生成音频潜空间: video=%d 像素帧 (fps=%.1f), audio=%d latents, ch=%d, freq=%d",
        length_frames, frame_rate, num_audio_latents, z_channels, audio_freq,
    )
    return {"samples": samples, "type": "audio"}


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(
                f"Yuan CLIP Timeline: '{name}' 为 None。"
                "可能原因：工作流 JSON 保存了空值、时间轴编辑器 Web 扩展未加载、或上游节点返回了 None。"
                "请将字段设为空字符串或修复上游连接。"
            )

    locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
    if not locals_list:
        raise ValueError("至少需要一个本地提示词（使用 | 分隔）")

    arch, patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("[Yuan CLIP Timeline] 全局: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[Yuan CLIP Timeline] 段 %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    log.info(
        "[Yuan CLIP Timeline] 潜空间: %d 帧, %d tokens/帧, 段: %s",
        latent_frames, tokens_per_frame, effective_lengths,
    )

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, None)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    return patched, conditioning


# ==============================================================================
# 主节点类
# ==============================================================================

class YuanCLIPTimeline:
    """可视化时间轴版本 — 段和长度来自节点 UI 中的可视化编辑器。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "要补丁的扩散模型"}),
                "clip": ("CLIP", {"tooltip": "用于编码提示词的 CLIP 模型"}),
                "audio_vae": ("VAE", {"tooltip": "Audio VAE，用于生成音频潜空间。video latent 与 audio latent 使用相同的帧数/帧率，保证对齐。"}),
                "global_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "贯穿整个视频的全局提示词。用于锚定持久的角色、物体和场景上下文。"
                }),
                "max_frames": ("INT", {
                    "default": 129, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "像素空间总帧数。仅用于编辑器的视觉缩放比例，实际帧数仍从潜空间读取。"
                }),
                "timeline_data": ("STRING", {
                    "default": "",
                    "tooltip": "时间轴编辑器的 JSON 状态（自动管理，请勿手动编辑）。"
                }),
                "local_prompts": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "由时间轴编辑器自动填充。"
                }),
                "segment_lengths": ("STRING", {
                    "default": "",
                    "tooltip": "由时间轴编辑器自动填充（像素空间帧数）。"
                }),
                "epsilon": ("FLOAT", {
                    "default": 1e-3, "min": 1e-6, "max": 0.99, "step": 1e-4,
                    "tooltip": "惩罚衰减参数。低于约 0.1 的值均产生锐利边界（论文默认 0.001）。"
                               "如需更柔和的过渡，尝试 0.5 或更高值。"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 0.1, "max": 240.0, "step": 0.1,
                    "tooltip": "每秒帧数 — 仅在 time_units 设为'seconds'时影响时间轴编辑器的显示。"
                }),
                "time_units": (["frames", "seconds"], {
                    "default": "frames",
                    "tooltip": "以帧或秒显示标尺、段范围、长度输入和总数。内部存储始终为像素空间帧。"
                }),
                "width": ("INT", {
                    "default": 768, "min": 32, "max": 8192, "step": 32,
                    "tooltip": "自动生成潜空间的目标宽度（未连接 latent 输入时生效）。"
                }),
                "height": ("INT", {
                    "default": 512, "min": 32, "max": 8192, "step": 32,
                    "tooltip": "自动生成潜空间的目标高度（未连接 latent 输入时生效）。"
                }),
            },
            "optional": {
                "latent": ("LATENT", {"tooltip": "潜空间视频 — 从形状读取尺寸。不连接时自动生成 LTXV 空潜空间。"}),
                "text_input": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "按行输入的提示词文本，支持两种模式：\n1. 时间格式（如 \"0-3s 提示词A\"），按指定秒数动态分配帧长\n2. 纯文本行，自动均分到各段落\n连接上游文本输出节点（如 Yuan TXT Splitter）可批量填充。"
                }),
                "prompt_lock": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "开启：预览模式：提示词只读不可编辑。\n关闭：各段落可自由编辑，不受 text_input 影响。"
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "LATENT")
    RETURN_NAMES = ("model", "positive", "video_latent", "audio_latent")
    FUNCTION = "encode_timeline"
    CATEGORY = "Yuan Tool/CLIP"

    # 调色板（与 JS 端 PALETTE 保持一致）
    _PALETTE = [
        "#4f8edc", "#e07b3a", "#5cb85c", "#d9534f", "#9b6cd6",
        "#a07060", "#e377c2", "#7f7f7f", "#c4c447", "#3fbac4",
    ]

    def encode_timeline(self, model, clip, audio_vae, global_prompt, max_frames, timeline_data,
                        local_prompts, segment_lengths, epsilon, fps=24.0, time_units="frames",
                        width=768, height=512, latent=None, text_input="", prompt_lock=True):
        # --- 处理 text_input：仅在锁定模式下按行智能分配到 local_prompts 和 timeline_data ---
        if prompt_lock and text_input and text_input.strip():
            lines_raw = text_input.split("\n")

            # 尝试按时间格式解析每行：如 "0-3s 提示词A" 或 "3-5秒 提示词B"
            parsed_time_lines = []
            non_empty_count = 0
            for line in lines_raw:
                stripped = line.strip()
                if not stripped:
                    continue
                non_empty_count += 1
                match = _TIME_RANGE_PATTERN.match(stripped)
                if match:
                    start_sec = float(match.group(1))
                    end_sec = float(match.group(2))
                    prompt_text = stripped[match.end():].strip()
                    if prompt_text:
                        parsed_time_lines.append({
                            "prompt": prompt_text,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "duration_sec": max(0.0, end_sec - start_sec),
                        })

            # 只有所有非空行都匹配时间格式时，才启用动态时长分配
            if parsed_time_lines and len(parsed_time_lines) == non_empty_count:
                # --- 动态时长分布：按时间格式分配帧数 ---
                lines_prompts = [p["prompt"] for p in parsed_time_lines]
                local_prompts = " | ".join(lines_prompts)
                log.info("[Yuan CLIP Timeline] text_input 提供 %d 行带时间格式的文本，按动态时长分配", len(parsed_time_lines))

                # 从时间段中读取最大结束时间，自动计算 max_frames
                max_end_sec = max(p["end_sec"] for p in parsed_time_lines)
                max_frames = int(max_end_sec * fps) + 1
                log.info("[Yuan CLIP Timeline] 最大结束时间 %.1f秒, fps %.1f, 自动计算 max_frames=%d",
                         max_end_sec, fps, max_frames)

                # 将秒数转换为帧数
                frame_allocations = []
                for p in parsed_time_lines:
                    frames = max(1, round(p["duration_sec"] * fps))
                    frame_allocations.append(frames)

                total_frames = sum(frame_allocations)

                if total_frames > max_frames:
                    # 时间轴已满：从末尾段落借用空间
                    excess = total_frames - max_frames
                    frame_allocations[-1] = max(1, frame_allocations[-1] - excess)
                    log.info("[Yuan CLIP Timeline] 总时长 (%d帧) 超出 max_frames (%d帧)，末尾段落缩减 %d 帧",
                             total_frames, max_frames, excess)
                elif total_frames < max_frames:
                    # 末尾段落未填满：填充剩余时间段
                    leftover = max_frames - total_frames
                    frame_allocations[-1] += leftover
                    log.info("[Yuan CLIP Timeline] 总时长 (%d帧) 未满 max_frames (%d帧)，末尾段落扩展 %d 帧",
                             total_frames, max_frames, leftover)

                # 构建 timeline_data
                new_segs = []
                for i, (prompt_text, flen) in enumerate(zip(lines_prompts, frame_allocations)):
                    new_segs.append({
                        "prompt": prompt_text,
                        "length": flen,
                        "color": self._PALETTE[i % len(self._PALETTE)],
                    })
                timeline_data = json.dumps({"segments": new_segs})

                # 同步更新 segment_lengths
                segment_lengths = ", ".join(str(s["length"]) for s in new_segs)

            else:
                # --- 无时间格式：按原先均分逻辑处理 ---
                lines = [line.strip() for line in lines_raw if line.strip()]
                if lines:
                    local_prompts = " | ".join(lines)
                    log.info("[Yuan CLIP Timeline] text_input 提供 %d 行文本，已均分到段落", len(lines))

                    try:
                        td = json.loads(timeline_data) if timeline_data and timeline_data.strip() else None
                    except (json.JSONDecodeError, ValueError):
                        td = None

                    if td and isinstance(td.get("segments"), list):
                        existing_segs = td["segments"]
                        for i, line in enumerate(lines):
                            if i < len(existing_segs):
                                existing_segs[i]["prompt"] = line
                            else:
                                remaining = max_frames - sum(s["length"] for s in existing_segs)
                                new_len = max(1, remaining // (len(lines) - len(existing_segs))) if len(lines) > len(existing_segs) else 1
                                existing_segs.append({
                                    "prompt": line,
                                    "length": new_len,
                                    "color": self._PALETTE[i % len(self._PALETTE)],
                                })
                        td["segments"] = existing_segs[:len(lines)]
                        timeline_data = json.dumps(td)
                    else:
                        base_len = max(1, max_frames // len(lines))
                        new_segs = []
                        for i, line in enumerate(lines):
                            new_segs.append({
                                "prompt": line,
                                "length": base_len,
                                "color": self._PALETTE[i % len(self._PALETTE)],
                            })
                        timeline_data = json.dumps({"segments": new_segs})

                    try:
                        td_final = json.loads(timeline_data)
                        segment_lengths = ", ".join(str(s["length"]) for s in td_final.get("segments", []))
                    except (json.JSONDecodeError, ValueError, KeyError):
                        pass

        # --- 自动生成 LTXV 潜空间（如果未连接 latent 输入） ---
        ltxv_length = max_frames + 1  # LTXV 约定: 像素帧 = 潜空间帧 * 8 + 1
        if latent is None:
            latent = _auto_generate_latent(width, height, ltxv_length)

        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon,
        )

        # --- 自动生成音频潜空间（原理同 video latent：零张量 + type="audio"，由采样器加噪去噪）---
        audio_latent = _auto_generate_audio_latent(audio_vae, ltxv_length, fps)

        return (patched, conditioning, latent, audio_latent)


# ==============================================================================
# 注册映射
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "YuanCLIPTimeline": YuanCLIPTimeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanCLIPTimeline": "Yuan CLIP 时间轴 (Timeline)",
}
