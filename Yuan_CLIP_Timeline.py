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
import torch.nn.functional as F
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


def build_segments(token_ranges, segment_lengths, epsilon=1e-3):
    """为时间惩罚构建每段元数据。"""
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448

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
            "window": max(base_window, 0.0),
            "sigma": sigma,
            "strength": 1.0,
            "window_audio": max(base_window, 0.0),
            "sigma_audio": sigma,
            "strength_audio": 1.0,
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


def _redistribute_to_total(lengths, target_total):
    """将长度列表重新分配到精确等于 target_total，使用最大余数法。"""
    if not lengths:
        return []
    total = sum(lengths)
    if total == target_total:
        return list(lengths)
    if total <= 0:
        return _distribute_evenly(len(lengths), target_total)
    exact = [L * target_total / total for L in lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1
    elif diff < 0:
        order = sorted(range(len(exact)), key=lambda i: exact[i] - int(exact[i]))
        for k in range(-diff):
            idx = order[k % len(order)]
            if result[idx] > 1:
                result[idx] -= 1
    return [max(1, L) for L in result]


def _distribute_evenly(num_segments, target_total):
    """最大余数法均分：确保总和精确等于 target_total。"""
    if num_segments <= 0 or target_total <= 0:
        return []
    base = target_total // num_segments
    remainder = target_total % num_segments
    return [max(1, base + (1 if i < remainder else 0)) for i in range(num_segments)]


def distribute_segment_lengths(num_segments, latent_frames, specified_lengths=None):
    """验证或自动分布段帧数，确保总和精确等于 latent_frames。

    无论 specified_lengths 来自何处（用户手动输入、_convert_to_latent_lengths 转换、
    时间轴编辑器），最终输出始终规范化到总和 = latent_frames，
    避免段长度溢出导致惩罚矩阵污染参考帧 tokens 区域。
    """
    if num_segments <= 0 or latent_frames <= 0:
        return []

    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError(
                f"segment_lengths 数量 ({len(specified_lengths)}) "
                f"必须与本地提示词数量 ({num_segments}) 一致"
            )
        clipped = [max(1, min(L, latent_frames)) for L in specified_lengths]
        total = sum(clipped)
        if total != latent_frames:
            log.warning(
                "[Yuan CLIP Timeline] segment_lengths 总和(%d) != latent_frames(%d)，自动重新规范化",
                total, latent_frames,
            )
            return _redistribute_to_total(clipped, latent_frames)
        return clipped

    return _distribute_evenly(num_segments, latent_frames)


# ==============================================================================
# patches.py 模型补丁函数
# ==============================================================================

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


def detect_model_type(model):
    """返回 (patch_size, temporal_stride) 用于 LTX 潜空间几何信息。"""
    diff_model = model.model.diffusion_model

    if hasattr(diff_model, "patchifier"):
        return (1, 1, 1), int(diff_model.vae_scale_factors[0])

    raise ValueError(
        f"不支持的模型类型: {type(diff_model).__name__}。"
        f"Yuan CLIP Timeline 仅支持 LTX 模型。"
    )


def apply_patches(model_clone, mask_fn, pointer_config=None):
    diffusion_model = model_clone.get_model_object("diffusion_model")

    to = model_clone.model_options["transformer_options"]
    to["promptrelay_mask_fn"] = mask_fn

    if pointer_config is not None:
        to["licon_msr_v3_relay_mask_fn"] = mask_fn
        to["licon_msr_v3_marker_token_indices"] = pointer_config["marker_token_indices"]

    for idx, block in enumerate(diffusion_model.transformer_blocks):
        for attr in ("attn2", "audio_attn2"):
            module = getattr(block, attr, None)
            if module is None:
                continue
            key = f"diffusion_model.transformer_blocks.{idx}.{attr}.forward"
            underlying = model_clone.get_model_object(key)
            if attr == "attn2" and pointer_config is not None:
                wrapper = _make_ltx_marker_relay_wrapper(underlying, mask_fn, pointer_config, idx)
            else:
                wrapper = _make_ltx_mask_wrapper(underlying, mask_fn)
            model_clone.add_object_patch(key, types.MethodType(wrapper, module))


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
# MSR Info / Marker Relay 辅助函数 (来自 Licon MSR V3)
# ==============================================================================

def _flatten_token_ids(raw):
    if isinstance(raw, dict):
        raw = raw.get("input_ids", [])
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().tolist()
    if raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    return list(raw or [])


def _token_count(raw_tokenizer, text):
    ids = _flatten_token_ids(raw_tokenizer(text))
    eos_adj = 1 if getattr(raw_tokenizer, "add_eos", False) else 0
    return max(0, len(ids) - eos_adj)


def _find_marker_phrase_token_indices(
    raw_tokenizer,
    prompt_text,
    markers,
    stop_markers=None,
    phrase_extend_tokens=12,
    stop_at_punctuation=True,
    stop_at_other_marker=True,
):
    if raw_tokenizer is None or not prompt_text:
        return []

    all_markers = [m for m in (stop_markers or markers) if m]
    prompt_folded = prompt_text.casefold()
    stop_chars = set(",，。.!！?？;；:：\n\r|")
    indices = []

    for marker in markers:
        marker_folded = marker.casefold()
        start = 0
        while True:
            char_start = prompt_folded.find(marker_folded, start)
            if char_start < 0:
                break

            char_stop = char_start + len(marker)
            scan = char_stop
            while scan < len(prompt_text):
                if stop_at_punctuation and prompt_text[scan] in stop_chars:
                    break
                if stop_at_other_marker:
                    tail = prompt_folded[scan:]
                    if any(tail.startswith(other.casefold()) for other in all_markers if other != marker):
                        break
                scan += 1

            char_stop = max(char_stop, scan)
            tok_start = _token_count(raw_tokenizer, prompt_text[:char_start])
            marker_tok_end = _token_count(raw_tokenizer, prompt_text[:char_start + len(marker)])
            phrase_tok_end = _token_count(raw_tokenizer, prompt_text[:char_stop])
            tok_end = max(marker_tok_end, phrase_tok_end)
            if phrase_extend_tokens > 0:
                tok_end = min(tok_end, tok_start + int(phrase_extend_tokens))
            if tok_end <= tok_start:
                tok_end = tok_start + 1
            indices.extend(range(tok_start, tok_end))
            start = char_start + len(marker)

    return sorted(set(i for i in indices if i >= 0))


def _map_subject_to_ref_latent_indices(
    subject, source_frame_count, latent_frame_count, reference_latent_count=None,
    max_ref_frames=None,
):
    """将主体映射到参考latent索引。

    当运行时 latent_frame_count 与 msr_info 中的 reference_latent_count 不一致时
    （例如 LTXV 对参考帧做了额外时间压缩），放弃使用 latent_start/latent_end
    （它们基于 reference_latent_count 计算，会截断或重叠），
    改用 frame_start/frame_end 按比例重新映射，避免图4特征丢失或与图3污染。

    max_ref_frames: 每个主体最多使用几个参考 latent 帧（取前 N 个）。
    避免参考帧过多导致 ref_summary 均值模糊，各主体统一取 2 帧以内保证公平。
    """
    if source_frame_count <= 0 or latent_frame_count <= 0:
        return []

    # 判断运行时latent帧数是否与msr_info中一致
    runtime_mismatch = (
        reference_latent_count is not None
        and reference_latent_count > 0
        and latent_frame_count != reference_latent_count
    )

    if not runtime_mismatch and "latent_start" in subject and "latent_end" in subject:
        ls = int(subject.get("latent_start", 0))
        le = int(subject.get("latent_end", ls))
        ls = max(0, min(latent_frame_count - 1, ls))
        le = max(ls, min(latent_frame_count - 1, le))
        indices = list(range(ls, le + 1))
    else:
        # 运行时不一致或没有latent_start/latent_end时，使用frame_start/frame_end重新计算
        fs = int(subject.get("frame_start", 0))
        fe = int(subject.get("frame_end", -1))
        fs = max(0, min(source_frame_count - 1, fs))
        fe = max(fs, min(source_frame_count - 1, fe))

        if latent_frame_count == source_frame_count:
            indices = list(range(fs, fe + 1))
        elif latent_frame_count == 1:
            indices = [0]
        else:
            stride = (source_frame_count - 1) / float(latent_frame_count - 1)
            indices = []
            for latent_idx in range(latent_frame_count):
                source_anchor = int(round(latent_idx * stride))
                if fs <= source_anchor <= fe:
                    indices.append(latent_idx)
            if not indices:
                center = (fs + fe) * 0.5
                nearest = int(round(center / stride))
                indices = [max(0, min(latent_frame_count - 1, nearest))]

    # 每个主体最多取 max_ref_frames 个参考 latent 帧（取前 N 个）
    if max_ref_frames is not None and len(indices) > max_ref_frames:
        indices = indices[:max_ref_frames]

    return indices


def _parse_block_filter(text, n_blocks):
    if not text or not text.strip():
        return None
    out = set()
    for raw in text.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                lo, hi = sorted((int(a.strip()), int(b.strip())))
                out.update(range(max(0, lo), min(n_blocks - 1, hi) + 1))
            except Exception:
                continue
        else:
            try:
                idx = int(part)
                if 0 <= idx < n_blocks:
                    out.add(idx)
            except Exception:
                continue
    return frozenset(out) if out else None


def _build_slot_ref_indices_from_target_latent(
    msr_info,
    slot,
    seq,
    device,
    target_latent_shape,
    spatial_patch_size=1,
    temporal_patch_size=1,
    max_ref_frames=None,
):
    if not isinstance(msr_info, dict) or target_latent_shape is None:
        return None

    reference_frame_count = int(msr_info.get("reference_frame_count") or 0)
    if reference_frame_count <= 0:
        return None

    reference_latent_count = int(msr_info.get("reference_latent_count") or 0)

    subjects = list(msr_info.get("subjects") or [])
    background = msr_info.get("background")
    if isinstance(background, dict):
        subjects.append(dict(background))

    subject = None
    for item in subjects:
        if isinstance(item, dict) and str(item.get("slot")) == str(slot):
            subject = item
            break
    if subject is None:
        return None

    _, _, target_frames, h_lat, w_lat = target_latent_shape
    tokens_per_frame = (
        max(1, h_lat // max(1, spatial_patch_size))
        * max(1, w_lat // max(1, spatial_patch_size))
    )
    if tokens_per_frame <= 0:
        return None

    target_count = (
        max(1, target_frames // max(1, temporal_patch_size))
        * tokens_per_frame
    )
    if target_count <= 0 or target_count >= seq:
        return None

    ref_count = seq - target_count
    if ref_count <= 0 or ref_count % tokens_per_frame != 0:
        log.warning(
            "[Yuan CLIP Timeline/MSR] slot=%s ref_count=%d 不合法 (seq=%d, target_count=%d, tpf=%d, ref%%tpf=%d)",
            slot, ref_count, seq, target_count, tokens_per_frame, ref_count % tokens_per_frame,
        )
        return None

    ref_latent_frames = ref_count // tokens_per_frame

    # 诊断：运行时参考latent帧数与msr_info中的不一致时，记录警告
    if reference_latent_count > 0 and ref_latent_frames != reference_latent_count:
        log.warning(
            "[Yuan CLIP Timeline/MSR] slot=%s 运行时参考latent帧数(%d) != msr_info.reference_latent_count(%d)，"
            "将使用frame_start/frame_end按比例重新映射（避免latent_start/latent_end截断导致主体缺失或污染）",
            slot, ref_latent_frames, reference_latent_count,
        )

    latent_indices = _map_subject_to_ref_latent_indices(
        subject,
        source_frame_count=reference_frame_count,
        latent_frame_count=ref_latent_frames,
        reference_latent_count=reference_latent_count,
        max_ref_frames=max_ref_frames,
    )
    ranges = []
    for latent_idx in latent_indices:
        start = target_count + latent_idx * tokens_per_frame
        stop = start + tokens_per_frame
        if target_count <= start < stop <= seq:
            ranges.append(torch.arange(start, stop, device=device, dtype=torch.long))

    if not ranges:
        log.warning(
            "[Yuan CLIP Timeline/MSR] slot=%s 无法构建ref索引 (ref_latent_frames=%d, latent_indices=%s, seq=%d, target_count=%d)",
            slot, ref_latent_frames, latent_indices, seq, target_count,
        )
        return None
    return torch.cat(ranges, dim=0) if len(ranges) > 1 else ranges[0]


def _positive_batch_mask(transformer_options, batch_size, device):
    cond_or_uncond = transformer_options.get("cond_or_uncond")
    if not cond_or_uncond or batch_size <= 0:
        return None

    cond_or_uncond = list(cond_or_uncond)
    group_count = len(cond_or_uncond)
    if group_count <= 0 or batch_size % group_count != 0:
        return None

    group_size = batch_size // group_count
    mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    for group_idx, value in enumerate(cond_or_uncond):
        if value == 0:
            start = group_idx * group_size
            mask[start:start + group_size] = True
    return mask


def _is_negative_only_call(transformer_options):
    cond_or_uncond = transformer_options.get("cond_or_uncond")
    return bool(cond_or_uncond) and all(value != 0 for value in cond_or_uncond)


def _relay_mask_for_positive_rows(relay_mask, transformer_options, batch_size, device, dtype):
    positive_mask = _positive_batch_mask(transformer_options, batch_size, device)
    if positive_mask is None:
        return relay_mask
    if not positive_mask.any():
        return None

    if relay_mask.dim() == 2:
        out = torch.zeros(
            batch_size,
            1,
            relay_mask.shape[-2],
            relay_mask.shape[-1],
            device=device,
            dtype=dtype,
        )
        out[positive_mask, 0, :, :] = relay_mask.to(device=device, dtype=dtype)
        return out

    view_shape = [batch_size] + [1] * (relay_mask.dim() - 1)
    return relay_mask * positive_mask.view(*view_shape).to(device=relay_mask.device, dtype=relay_mask.dtype)


def _make_ltx_marker_relay_wrapper(
    underlying,
    mask_fn,
    pointer_config,
    block_idx,
):
    def wrapped(_self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
        direct_supported = all(hasattr(_self, name) for name in ("to_q", "to_k", "to_v", "q_norm", "k_norm", "to_out"))

        if _is_negative_only_call(transformer_options):
            return underlying(
                x,
                context=context,
                mask=mask,
                pe=pe,
                k_pe=k_pe,
                transformer_options=transformer_options,
            )

        if not direct_supported:
            if context is not None:
                relay_mask = mask_fn(x.shape[1], context.shape[1], x.dtype, x.device, transformer_options)
                if relay_mask is not None:
                    relay_mask = _relay_mask_for_positive_rows(
                        relay_mask, transformer_options, x.shape[0], x.device, x.dtype
                    )
                if relay_mask is not None:
                    mask = relay_mask if mask is None else mask + relay_mask
            return underlying(
                x,
                context=context,
                mask=mask,
                pe=pe,
                k_pe=k_pe,
                transformer_options=transformer_options,
            )

        context = x if context is None else context
        q = _self.to_q(x)
        k = _self.to_k(context)
        v = _self.to_v(context)

        if context is not None:
            marker_token_indices = pointer_config["marker_token_indices"]
            msr_info = pointer_config["msr_info"]
            latent_shape = pointer_config["latent_shape"]
            pointer_blocks = pointer_config["pointer_blocks"]
            spatial_patch_size = pointer_config["spatial_patch_size"]
            temporal_patch_size = pointer_config["temporal_patch_size"]
            binding_strength = pointer_config["binding_strength"]
            preserve_text_strength = pointer_config["preserve_text_strength"]
            normalize_ref_summary = pointer_config["normalize_ref_summary"]
            max_ref_frames = pointer_config.get("max_ref_frames", 2)
            positive_mask = _positive_batch_mask(transformer_options, x.shape[0], x.device)
            positive_rows = None
            if positive_mask is not None:
                positive_rows = torch.where(positive_mask)[0]

            if pointer_blocks is None or block_idx in pointer_blocks:
                max_context_index = context.shape[1] - 1
                _slots_missing_logged = set()

                for slot, token_indices in marker_token_indices.items():
                    usable = [idx for idx in token_indices if idx <= max_context_index]
                    if not usable:
                        if slot not in _slots_missing_logged:
                            _slots_missing_logged.add(slot)
                            log.warning(
                                "[Yuan CLIP Timeline/MSR] block=%d slot=%s 所有标记token(%s)超出context范围(max=%d)，"
                                "该主体无法绑定——full_prompt token数超过实际context长度(%d, 由max_frames/段落数动态决定)导致截断，"
                                "请缩短global_prompt描述或减少主体数量",
                                block_idx, slot, token_indices[:5], max_context_index, context.shape[1],
                            )
                        continue

                    ref_indices = _build_slot_ref_indices_from_target_latent(
                        msr_info,
                        slot,
                        seq=x.shape[1],
                        device=x.device,
                        target_latent_shape=latent_shape,
                        spatial_patch_size=spatial_patch_size,
                        temporal_patch_size=temporal_patch_size,
                        max_ref_frames=max_ref_frames,
                    )
                    if ref_indices is None or ref_indices.numel() == 0:
                        if slot not in _slots_missing_logged:
                            _slots_missing_logged.add(slot)
                            log.warning(
                                "[Yuan CLIP Timeline/MSR] block=%d slot=%s 无法获取参考帧索引，该主体绑定被跳过",
                                block_idx, slot,
                            )
                        continue

                    ref_summary = x[:, ref_indices, :].mean(dim=1)
                    marker_tensor = torch.as_tensor(usable, device=k.device, dtype=torch.long)

                    ref_k = _self.to_k(ref_summary[:, None, :]).to(dtype=k.dtype, device=k.device)
                    ref_v = _self.to_v(ref_summary[:, None, :]).to(dtype=v.dtype, device=v.device)
                    if normalize_ref_summary:
                        marker_norm = k[:, marker_tensor, :].norm(dim=-1, keepdim=True).mean(dim=1)
                        ref_k = F.normalize(ref_k, dim=-1) * marker_norm.clamp_min(1e-6)
                        ref_v = F.normalize(ref_v, dim=-1) * marker_norm.to(dtype=ref_v.dtype, device=ref_v.device).clamp_min(1e-6)

                    if positive_rows is not None:
                        if positive_rows.numel() == 0:
                            continue
                        k[positive_rows[:, None], marker_tensor[None, :], :] = (
                            k[positive_rows[:, None], marker_tensor[None, :], :] * float(preserve_text_strength)
                            + ref_k[positive_rows] * float(binding_strength)
                        )
                        v[positive_rows[:, None], marker_tensor[None, :], :] = (
                            v[positive_rows[:, None], marker_tensor[None, :], :] * float(preserve_text_strength)
                            + ref_v[positive_rows] * float(binding_strength)
                        )
                    else:
                        k[:, marker_tensor, :] = (
                            k[:, marker_tensor, :] * float(preserve_text_strength)
                            + ref_k * float(binding_strength)
                        )
                        v[:, marker_tensor, :] = (
                            v[:, marker_tensor, :] * float(preserve_text_strength)
                            + ref_v * float(binding_strength)
                        )

        q = _self.q_norm(q)
        k = _self.k_norm(k)

        if pe is not None:
            try:
                from comfy.ldm.lightricks.model import apply_rotary_emb
                q = apply_rotary_emb(q, pe)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe)
            except Exception:
                pass

        if context is not None:
            relay_mask = mask_fn(x.shape[1], context.shape[1], x.dtype, x.device, transformer_options)
            if relay_mask is not None:
                relay_mask = _relay_mask_for_positive_rows(
                    relay_mask, transformer_options, x.shape[0], x.device, x.dtype
                )
            if relay_mask is not None:
                mask = relay_mask if mask is None else mask + relay_mask

        if mask is None:
            out = comfy.ldm.modules.attention.optimized_attention(
                q,
                k,
                v,
                _self.heads,
                attn_precision=getattr(_self, "attn_precision", None),
                transformer_options=transformer_options,
            )
        else:
            out = comfy.ldm.modules.attention.optimized_attention_masked(
                q,
                k,
                v,
                _self.heads,
                mask,
                attn_precision=getattr(_self, "attn_precision", None),
                transformer_options=transformer_options,
            )

        to_gate_logits = getattr(_self, "to_gate_logits", None)
        if to_gate_logits is not None:
            gate_logits = to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, _self.heads, _self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, _self.heads * _self.dim_head)

        return _self.to_out(out)

    return wrapped


def _parse_yuan_map_config(msr_info, config_str):
    role_order = []
    role_descriptions = {}
    background_role = None

    bg_keywords = ("背景", "场景", "bg", "background", "environment", "scene")

    lines = config_str.replace(",", "\n").split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(r'@(\S+?)\s*[=:：]\s*(.+)', line)
        if m:
            role_name = "@" + m.group(1).strip()
            desc = m.group(2).strip()
            role_order.append(role_name)
            role_descriptions[role_name] = desc
            role_lower = role_name.lower()
            if any(kw in role_lower for kw in [kw.lower() for kw in bg_keywords]):
                background_role = role_name

    subjects = msr_info.get("subjects") or []
    subject_slots = []
    for item in subjects:
        if isinstance(item, dict) and "slot" in item:
            subject_slots.append(str(item["slot"]))

    role_slots = {}
    subject_idx = 0
    for role_name in role_order:
        if role_name == background_role:
            continue
        if subject_idx < len(subject_slots):
            role_slots[subject_slots[subject_idx]] = role_name
            subject_idx += 1

    return role_descriptions, role_slots, background_role


def _split_local_prompts(local_prompts):
    if "|" in local_prompts:
        return [p.strip() for p in local_prompts.split("|") if p.strip()]
    return [p.strip() for p in local_prompts.splitlines() if p.strip()]


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

    patch_size, temporal_stride = detect_model_type(model)

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

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, mask_fn)

    return patched, conditioning


def _encode_relay_with_msr(model, clip, latent, msr_info, global_prompt, local_prompts, segment_lengths, epsilon, binding_strength=0.35):
    """基于 msr_info 的 @角色 Prompt Relay 编码（与 YuanNode 功能一致）。
    global_prompt 中使用 @图1=描述、@背景=描述 格式定义角色。"""
    if not isinstance(msr_info, dict):
        raise ValueError("Yuan CLIP Timeline: msr_info must be a dict")

    role_descriptions, role_slots, background_role = _parse_yuan_map_config(msr_info, global_prompt)
    if not role_descriptions and not background_role:
        raise ValueError("Yuan CLIP Timeline: global_prompt 中未找到有效的 @角色 定义")

    # 从 @角色定义构建全局提示词
    # role_descriptions 已包含 background_role（_parse_yuan_map_config 将 @背景 也加入 role_order），
    # 无需额外 append 背景描述，否则描述出现两次，浪费 CLIP token
    global_prompt_parts = []
    for role_name, desc in role_descriptions.items():
        global_prompt_parts.append(f"{role_name}是{desc}")
    global_prompt = "。".join(global_prompt_parts) + ("。" if global_prompt_parts else "")

    locals_list = _split_local_prompts(local_prompts.strip())
    if not locals_list:
        raise ValueError("Yuan CLIP Timeline: local_prompts 不能为空，使用 | 或换行分隔段落")

    patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_shape = tuple(samples.shape)
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    raw_tokenizer = get_raw_tokenizer(clip)

    # 构建 @角色 标记规格
    marker_specs = {}
    for slot, role_name in role_slots.items():
        marker_name = role_name.lstrip("@")
        markers = [role_name, marker_name]
        m = re.match(r'图\s*(\d+)', marker_name)
        if m:
            num = m.group(1)
            markers.extend([f"参考图{num}", f"pic{num}", f"图{num}"])
        seen = set()
        marker_specs[slot] = [m for m in markers if not (m in seen or seen.add(m))]
    if background_role:
        bg_marker_name = background_role.lstrip("@")
        markers = [background_role, bg_marker_name, "bg", "background", "背景", "场景"]
        seen = set()
        marker_specs["background"] = [m for m in markers if not (m in seen or seen.add(m))]

    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)
    global_token_count = token_ranges[0][0] if token_ranges else 0
    total_prompt_tokens = token_ranges[-1][1] if token_ranges else 0

    log.info("[Yuan CLIP Timeline/MSR] 全局: tokens [0:%d] (%d tokens)", global_token_count, global_token_count)
    for i, (s, e) in enumerate(token_ranges):
        log.info("[Yuan CLIP Timeline/MSR] 段 %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)
    log.info(
        "[Yuan CLIP Timeline/MSR] full_prompt 总token数=%d (global=%d + local=%d)",
        total_prompt_tokens, global_token_count, total_prompt_tokens - global_token_count,
    )

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)
    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    log.info(
        "[Yuan CLIP Timeline/MSR] 潜空间: %d 帧, %d tokens/帧, 段: %s",
        latent_frames, tokens_per_frame, effective_lengths,
    )

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon)
    relay_mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    # 查找标记 token 索引
    # 注意：CLIP 的 token 上限不是硬性 77，而是运行时 context.shape[1]（编码后的实际 token 数），
    # 该值由 max_frames / text_input 段落数动态决定（如 385/5=77）。
    # 实际的有效性过滤在注意力块回调中执行：usable = [idx for idx in token_indices if idx <= max_context_index]
    all_markers = [marker for markers in marker_specs.values() for marker in markers]
    marker_token_indices = {}
    for slot, markers in marker_specs.items():
        if not markers:
            continue

        # 1. 在 global prompt 中搜索标记（窄短语扩展=3，避免绑定到"@图1是描述"的长文字）
        global_indices = []
        if global_token_count > 0:
            global_indices = _find_marker_phrase_token_indices(
                raw_tokenizer, full_prompt, markers,
                stop_markers=all_markers,
                phrase_extend_tokens=3,
            )
            global_indices = [idx for idx in global_indices if 0 <= idx < global_token_count]

        # 2. 在 local prompt 中搜索标记（宽短语扩展=12，保持局部语义完整）
        local_indices = _find_marker_phrase_token_indices(
            raw_tokenizer, full_prompt, markers,
            stop_markers=all_markers,
            phrase_extend_tokens=12,
        )
        local_indices = [idx for idx in local_indices if idx >= global_token_count]

        # 3. 全局标记优先：global prompt 中的标记不在 q_token_idx 段中，
        #    不受 Prompt Relay 时间惩罚，提供跨段持续的全局主体绑定。
        #    标记 token 是否在 context 有效范围内，由运行时注意力块回调根据
        #    context.shape[1] 动态过滤（max_context_index），无需在此预过滤。
        if global_indices:
            marker_token_indices[slot] = global_indices
            log.info(
                "[Yuan CLIP Timeline/MSR] slot=%s 使用全局标记绑定 (%d tokens, 无时间惩罚, 全帧可见)",
                slot, len(global_indices),
            )
        elif local_indices:
            marker_token_indices[slot] = local_indices
            log.info(
                "[Yuan CLIP Timeline/MSR] slot=%s 使用局部标记绑定 (%d tokens, 受时间惩罚, 仅对应段可见)",
                slot, len(local_indices),
            )
        else:
            log.warning("[Yuan CLIP Timeline/MSR] slot=%s 未找到任何标记 token，该主体无法绑定", slot)

    log.info("[Yuan CLIP Timeline/MSR] 标记绑定: %s", {k: len(v) for k, v in marker_token_indices.items()})

    if msr_info:
        subjects = msr_info.get("subjects") or []
        bg = msr_info.get("background") or {}
        log.info(
            "[Yuan CLIP Timeline/MSR] msr_info: ref_frames=%s, ref_latents=%s, subjects=%s, bg=%s",
            msr_info.get("reference_frame_count"),
            msr_info.get("reference_latent_count"),
            [(s.get("slot"), s.get("frame_start"), s.get("frame_end"), s.get("latent_start"), s.get("latent_end")) for s in subjects],
            (bg.get("frame_start"), bg.get("frame_end"), bg.get("latent_start"), bg.get("latent_end")),
        )

    patched = model.clone()

    pointer_config = None
    if marker_token_indices:
        diffusion_model = patched.get_model_object("diffusion_model")
        n_blocks = len(diffusion_model.transformer_blocks)
        pointer_blocks = _parse_block_filter("8-47", n_blocks)
        log.info("[Yuan CLIP Timeline/MSR] 模型 blocks=%d, pointer_blocks=%s", n_blocks, sorted(pointer_blocks)[:5])

        # 动态计算每个主体最大参考latent帧数
        # max_ref_frames = 每段最少latent帧 / 被@主体数量（取整）
        # 例如：5段每段9帧，3个主体 → 9//3=3帧；4段每段12帧，4个主体 → 12//4=3帧
        num_at_subjects = len(marker_token_indices) - (1 if "background" in marker_token_indices else 0)
        per_segment_latent = min(effective_lengths) if effective_lengths else latent_frames
        max_ref_frames = max(1, per_segment_latent // max(1, num_at_subjects))
        log.info(
            "[Yuan CLIP Timeline/MSR] 动态max_ref_frames=%d (每段%d latents / %d 主体)",
            max_ref_frames, per_segment_latent, num_at_subjects,
        )

        pointer_config = {
            "marker_token_indices": marker_token_indices,
            "msr_info": msr_info,
            "latent_shape": latent_shape,
            "pointer_blocks": pointer_blocks,
            "spatial_patch_size": 1,
            "temporal_patch_size": 1,
            "binding_strength": binding_strength,
            "preserve_text_strength": 1.0,
            "normalize_ref_summary": True,
            "max_ref_frames": max_ref_frames,
        }

    apply_patches(patched, relay_mask_fn, pointer_config)

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
                    "tooltip": "贯穿整个视频的全局提示词。用于锚定持久的角色、物体和场景上下文。\n连接 msr_info 时，在此用 @图1=描述、@图2=描述、@背景=描述 定义角色，每行一条，按顺序对应 msr_info 的 subjects。"
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
                "msr_info": ("MSR_INFO", {
                    "tooltip": "连接多帧参考节点的 msr_info 输出。连接后启用 @角色标记绑定功能：\nglobal_prompt 中用 @图1=描述、@背景=描述 定义角色，\nlocal_prompts/text_input 中用 @图1、@图2 等引用角色，\n模型自动将标记 token 绑定到参考帧。"
                }),
                "binding_strength": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "标记绑定强度。控制 @角色 token 与参考帧的绑定程度。\n值越高绑定越强，但过高可能导致伪影。仅在 msr_info 连接时生效。"
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
                        width=768, height=512, latent=None, text_input="", prompt_lock=True,
                        msr_info=None, binding_strength=0.35):
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
                raw_max = int(max_end_sec * fps) + 1
                # 对齐 LTXV 时间步长 (8): 实际输出帧 = (max_frames//8)*8+1，确保 max_frames 与此一致
                max_frames = ((raw_max - 2) // 8 + 1) * 8 + 1
                log.info("[Yuan CLIP Timeline] 最大结束时间 %.1f秒, fps %.1f, 自动计算 max_frames=%d (原始值%d, 对齐 LTXV stride 8)",
                         max_end_sec, fps, max_frames, raw_max)

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
        # max_frames 已在动态分配时对齐 LTXV stride 8，ltxv_length 直接使用 max_frames
        ltxv_length = max_frames
        if latent is None:
            latent = _auto_generate_latent(width, height, ltxv_length)

        # --- 编码路径选择：msr_info 连接时使用 Marker Relay 编码，否则使用标准 Relay 编码 ---
        if msr_info is not None:
            log.info("[Yuan CLIP Timeline] 检测到 msr_info 输入，启用 @角色标记绑定 (Marker Relay)")
            patched, conditioning = _encode_relay_with_msr(
                model, clip, latent, msr_info, global_prompt, local_prompts, segment_lengths, epsilon, binding_strength,
            )
        else:
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
