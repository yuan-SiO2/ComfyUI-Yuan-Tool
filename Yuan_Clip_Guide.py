"""Yuan 引导注入节点：将图像/视频帧作为引导关键帧注入视频 latent。

支持图像引导、视频分段引导（IC-LoRA）与 Ref Guidance 参考特征引导。
"""

import os
import types
import numpy as np
from PIL import Image
import node_helpers
import torch
import comfy
import comfy.utils
import folder_paths
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords

# PyAV 为可选依赖（视频帧解码用），缺失时降级不阻断节点加载
try:
    import av
    _HAS_AV = True
except ImportError:
    av = None
    _HAS_AV = False

from .Yuan_CLIP_Timeline import GuideData, MotionGuideData, _resize_image
from .Yuan_common import resolve_media_path

# IC-LoRA 参数自定义类型 — 用字符串定义以保证所有 ComfyUI 版本兼容
ICLoRAParameters = "IC_LORA_PARAMETERS"


# ==============================================================================
# K/V 视觉特征注入：把参考帧视觉特征注入 @图X token 的 K/V
# 注入点：transformer_blocks.{idx}.attn2.forward；公式：k[marker] = k[marker]*(1-alpha) + ref_k*alpha
# 段外帧的注意力抑制由 Timeline 生成的 token 级 mask（promptrelay_mask_fn）负责
# ==============================================================================

def _positive_batch_mask(transformer_options, batch_size, device):
    """从 transformer_options 获取正向条件 batch mask，返回 (B,) bool tensor（True=正向行）。"""
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


def _ltxv_crossattn_forward_kv_injection(self, x, context, mask=None,
                                         transformer_options={}, **kwargs):
    """替换 attn2 forward：把预计算的参考帧视觉特征 ref_summary 投影为 ref_k/ref_v，注入到 @图X token 的 K/V（仅正向批次）。"""
    if mask is None:
        mask_provider = transformer_options.get("promptrelay_mask_fn")
        if mask_provider is not None:
            mask = mask_provider(x.shape[1], context.shape[1], x.dtype, x.device, transformer_options)

    marker_token_indices = getattr(self, "marker_token_indices", {})
    ref_alpha = getattr(self, "ref_alpha", 0.0)
    subject_ref_features = getattr(self, "subject_ref_features", None)

    q = self.q_norm(self.to_q(x))
    k = self.k_norm(self.to_k(context))
    v = self.to_v(context)

    # K/V 注入：仅对拥有独立参考特征的主体注入，绝不可从 x（去噪中的视频 token）取值，否则会把去噪噪声当作参考特征导致主体特征错乱
    if marker_token_indices and ref_alpha > 0.0 and subject_ref_features:
        # 正向批次 mask：只对正向条件做 K/V 注入，负向条件保持原样
        positive_mask = _positive_batch_mask(transformer_options, x.shape[0], x.device)
        positive_rows = None
        if positive_mask is not None:
            positive_rows = torch.where(positive_mask)[0]
            if positive_rows.numel() == 0:
                # 全是负向条件，跳过注入
                positive_rows = None

        # token 越界保护：marker token 索引可能超过 context 长度（CLIP 截断）
        max_context_index = context.shape[1] - 1

        # 段内全强度注入；段外帧对 @图X token 的注意力抑制由 token 级 mask 负责
        effective_alpha = ref_alpha

        for subject_num, token_indices in marker_token_indices.items():
            if subject_num not in subject_ref_features:
                continue
            # 越界过滤：只保留 context 范围内的 token
            usable = [idx for idx in token_indices if idx <= max_context_index]
            if not usable:
                continue

            # 用预计算的 ref_summary 独立编码，解耦物理坐标系错位问题
            ref_summary = subject_ref_features[subject_num].to(device=x.device, dtype=x.dtype)
            num_pos = len(positive_rows) if positive_rows is not None else x.shape[0]
            ref_summary = ref_summary.expand(num_pos, -1)  # [num_pos, inner_dim]

            # ref_k 必须经 k_norm（RMSNorm）与原始 K 尺度对齐，否则 attention 权重异常
            ref_k = self.k_norm(self.to_k(ref_summary[:, None, :])).to(dtype=k.dtype, device=k.device)  # (B', 1, C)
            ref_v = self.to_v(ref_summary[:, None, :]).to(dtype=v.dtype, device=v.device)  # (B', 1, C)

            # K/V 注入：k[marker] = k[marker]*(1-alpha) + ref_k*alpha
            marker_tensor = torch.as_tensor(usable, device=k.device, dtype=torch.long)
            ref_k_expanded = ref_k.expand(-1, len(usable), -1)  # (B', num_markers, C)
            ref_v_expanded = ref_v.expand(-1, len(usable), -1)  # (B', num_markers, C)
            if positive_rows is not None:
                k[positive_rows[:, None], marker_tensor[None, :], :] = (
                    k[positive_rows[:, None], marker_tensor[None, :], :] * (1.0 - effective_alpha)
                    + ref_k_expanded * effective_alpha
                )
                v[positive_rows[:, None], marker_tensor[None, :], :] = (
                    v[positive_rows[:, None], marker_tensor[None, :], :] * (1.0 - effective_alpha)
                    + ref_v_expanded * effective_alpha
                )
            else:
                k[:, marker_tensor, :] = (
                    k[:, marker_tensor, :] * (1.0 - effective_alpha) + ref_k_expanded * effective_alpha
                )
                v[:, marker_tensor, :] = (
                    v[:, marker_tensor, :] * (1.0 - effective_alpha) + ref_v_expanded * effective_alpha
                )

    if mask is None:
        out = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.heads,
            attn_precision=self.attn_precision,
            transformer_options=transformer_options,
        ).flatten(2)
    else:
        out = comfy.ldm.modules.attention.attention_pytorch(
            q, k, v, heads=self.heads, mask=mask,
            attn_precision=self.attn_precision,
            _inside_attn_wrapper=True,
            transformer_options=transformer_options,
        ).flatten(2)
    del q, k, v

    # gate（LTX2 零初始化门控，identity 初始化）
    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, self.heads, self.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)

    return self.to_out(out)


class _LTXVCrossAttentionRefPatch:
    """把 K/V 注入参数绑定到 attn2 模块，替换其 forward。"""

    def __init__(self, marker_token_indices, ref_alpha, subject_ref_features=None):
        self.marker_token_indices = marker_token_indices
        self.ref_alpha = ref_alpha
        self.subject_ref_features = subject_ref_features

    def __get__(self, obj, objtype=None):
        def wrapped_attention(self_module, *args, **kwargs):
            self_module.marker_token_indices = self.marker_token_indices
            self_module.ref_alpha = self.ref_alpha
            self_module.subject_ref_features = self.subject_ref_features
            return _ltxv_crossattn_forward_kv_injection(self_module, *args, **kwargs)
        return types.MethodType(wrapped_attention, obj)


# --- 辅助函数 ---

def _append_guide_attention_entry(positive, negative, pre_filter_count, latent_shape, strength=1.0):
    """向 positive 和 negative conditioning 各自追加一条 guide_attention_entry。"""
    new_entry = {
        "pre_filter_count": pre_filter_count,
        "strength": strength,
        "pixel_mask": None,
        "latent_shape": latent_shape,
    }
    results = []
    for cond in (positive, negative):
        existing = []
        for t in cond:
            found = t[1].get("guide_attention_entries", None)
            if found is not None:
                existing = found
                break
        entries = [*existing, new_entry]
        results.append(node_helpers.conditioning_set_values(cond, {"guide_attention_entries": entries}))
    return results[0], results[1]


def conditioning_get_any_value(conditioning, key, default=None):
    for t in conditioning:
        if key in t[1]:
            return t[1][key]
    return default


def get_noise_mask(latent):
    """从 latent 字典获取 noise_mask，没有则创建全 1 的默认 mask。"""
    noise_mask = latent.get("noise_mask", None)
    latent_image = latent["samples"]
    if noise_mask is None:
        batch_size, _, latent_length, _, _ = latent_image.shape
        noise_mask = torch.ones(
            (batch_size, 1, latent_length, 1, 1),
            dtype=torch.float32,
            device=latent_image.device,
        )
    else:
        noise_mask = noise_mask.clone()
    return noise_mask


def get_keyframe_idxs(cond, latent_shape=None):
    """从 conditioning 读取 keyframe_idxs 及关键帧数量。"""
    keyframe_idxs = conditioning_get_any_value(cond, "keyframe_idxs", None)
    if keyframe_idxs is None:
        return None, 0
    if latent_shape is not None and len(latent_shape) == 5:
        tokens_per_frame = latent_shape[-2] * latent_shape[-1]
        num_keyframes = keyframe_idxs.shape[2] // tokens_per_frame
        return keyframe_idxs, num_keyframes
    entries = conditioning_get_any_value(cond, "guide_attention_entries", None)
    if entries:
        num_keyframes = sum(e["latent_shape"][0] for e in entries)
        return keyframe_idxs, num_keyframes
    num_keyframes = torch.unique(keyframe_idxs[:, 0, :, 0]).shape[0]
    return keyframe_idxs, num_keyframes


class ResampleGuideFrames:
    """重采样引导帧序列，支持 nearest 和 linear 模式。"""

    def execute(self, images, source_fps, target_fps, target_num_frames, mode):
        if images is None:
            return images
        frames = images
        n = int(frames.shape[0])
        target_num_frames = int(target_num_frames)
        if n <= 1:
            if target_num_frames > 1 and n == 1:
                return frames.repeat(target_num_frames, 1, 1, 1)
            return frames
        source_fps = float(max(0.001, source_fps))
        target_fps = float(max(0.001, target_fps))
        if target_num_frames <= 0:
            duration = (n - 1) / source_fps
            target_num_frames = max(1, int(round(duration * target_fps)) + 1)
        if target_num_frames == n and abs(target_fps - source_fps) < 1e-6:
            return frames
        positions = torch.linspace(0, n - 1, target_num_frames, device=frames.device, dtype=torch.float32)
        if mode == "nearest":
            idx = torch.round(positions).long().clamp(0, n - 1)
            return frames.index_select(0, idx)
        idx0 = torch.floor(positions).long().clamp(0, n - 1)
        idx1 = torch.ceil(positions).long().clamp(0, n - 1)
        alpha = (positions - idx0.to(positions.dtype)).view(-1, 1, 1, 1)
        f0 = frames.index_select(0, idx0).to(torch.float32)
        f1 = frames.index_select(0, idx1).to(torch.float32)
        return (f0 * (1.0 - alpha) + f1 * alpha).to(frames.dtype)


def _resolve_input_video_path(video_file):
    """解析视频文件路径，支持绝对路径、相对路径和 annotated filepath。"""
    path = resolve_media_path(str(video_file))
    if path:
        return path
    raise FileNotFoundError(f"找不到运动引导视频文件：{video_file}")


def _load_motion_video_frames(video_file, trim_start_frames, length_frames, director_fps, resample_mode="nearest", frame_files=None):
    """加载视频文件的指定帧区间；frame_files 提供时为多图像拼接模式（均分帧数后拼接）。"""
    # 多图像拼接模式：frame_files 非空且有多张图时，按序加载拼接
    if frame_files and isinstance(frame_files, list) and len(frame_files) > 1:
        n_imgs = len(frame_files)
        # 最大余数法均分帧数
        base = length_frames // n_imgs
        rem = length_frames % n_imgs
        counts = [base + (1 if i < rem else 0) for i in range(n_imgs)]
        all_frames = []
        for i, fpath in enumerate(frame_files):
            try:
                img_path = _resolve_input_video_path(fpath)
            except FileNotFoundError:
                continue
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            ft = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]
            all_frames.append(ft.repeat(counts[i], 1, 1, 1))
        if not all_frames:
            raise FileNotFoundError(f"frameFiles 中的所有文件均无法加载")
        video_frames = torch.cat(all_frames, dim=0)
        return video_frames

    path = _resolve_input_video_path(video_file)

    # 静态图像：重复为 length_frames 帧
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        frame_tensor = torch.from_numpy(arr)
        video_frames = frame_tensor.unsqueeze(0).repeat(length_frames, 1, 1, 1)
        return video_frames

    target_fps = max(1.0, float(director_fps))
    start_s = max(0.0, float(trim_start_frames) / target_fps)
    dur_s = max(0.0, float(length_frames) / target_fps)
    end_s = start_s + dur_s if dur_s > 0 else None

    if not _HAS_AV:
        raise RuntimeError("运动引导视频解码需要 PyAV 库，请执行 `pip install av` 安装")
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    try:
        source_fps = float(stream.average_rate) if stream.average_rate else float(stream.base_rate)
    except Exception:
        source_fps = target_fps
    if source_fps <= 0:
        source_fps = target_fps

    if start_s > 0:
        try:
            if stream.time_base:
                seek_pts = int(max(0, start_s - 0.5) / float(stream.time_base))
            else:
                seek_pts = int(max(0, start_s - 0.5) * av.time_base)
            container.seek(seek_pts, stream=stream, backward=True)
        except Exception:
            pass

    frames = []
    decoded_count = 0
    for frame in container.decode(stream):
        if frame.time is not None:
            t = float(frame.time)
        elif frame.pts is not None and stream.time_base is not None:
            t = float(frame.pts * stream.time_base)
        else:
            t = float(decoded_count / source_fps)
        decoded_count += 1
        if t < start_s - 0.01:
            continue
        if end_s is not None and t >= end_s:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    if not frames:
        raise ValueError(f"未能解码运动引导分段的视频帧：{video_file}")

    frames_np = np.array(frames, dtype=np.float32) / 255.0
    images = torch.from_numpy(frames_np)
    target_count = max(1, int(round(float(length_frames))))
    images = ResampleGuideFrames().execute(images, source_fps, target_fps, target_count, resample_mode)
    return images


# --- 主节点类 ---

class YuanClipGuide:
    """Yuan 引导注入：将图像/视频引导帧经 VAE 编码后拼接到视频 latent 末尾，并用 keyframe_idxs 告知模型对应位置。"""

    PATCHIFIER = SymmetricPatchifier(1, start_end=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MODEL", {
                    "tooltip": "模型（来自 Yuan CLIP 时间轴的 模型 输出），透传到输出端口保持数据流契约统一。"
                }),
                "正向条件": ("CONDITIONING", {"tooltip": "正向条件（来自 Yuan CLIP 时间轴的 正向条件 输出）。"}),
                "负向条件": ("CONDITIONING", {"tooltip": "负向条件。"}),
                "VAE": ("VAE", {"tooltip": "用于编码引导帧的 VAE。"}),
                "潜空间": ("LATENT", {"tooltip": "视频 latent（来自 Yuan CLIP 时间轴的 视频潜空间 输出）。"}),
            },
            "optional": {
                "引导数据": (GuideData, {
                    "tooltip": "来自 Yuan CLIP 时间轴的引导数据，包含图像引导列表、插入帧索引、强度等。"
                }),
                "运动引导数据": (MotionGuideData, {
                    "tooltip": "来自 Yuan CLIP 时间轴的运动引导数据，包含 IC-LoRA 视频分段。"
                }),
                "IC_LORA参数": (ICLoRAParameters, {
                    "tooltip": "可选的 IC-LoRA 参数（如 reference_downscale_factor > 1 时需要）。"
                }),
                "图像注意力强度": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "图像引导的注意力强度，控制图像引导帧对生成结果的注意力影响。"
                }),
                "MSR_LORA": (["auto", "MSR2.5", "MSR2.3"], {
                    "default": "auto",
                    "tooltip": "MSR 适配模式：auto 按 LoRA 元数据自动检测（V1 触发、V2/普通不触发）；MSR2.5（LTX-2.5-Licon-MSR-V1）强制 MSR 适配（负偏移+槽位嵌入）；MSR2.3（LTX-2.3-Licon-MSR-V2）强制普通模式（正帧，无嵌入）。"
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("模型", "正向条件", "负向条件", "潜空间")
    FUNCTION = "execute"
    CATEGORY = "Yuan Tool/CLIP"

    @classmethod
    def encode(cls, vae, latent_width, latent_height, images, scale_factors, latent_downscale_factor=1):
        """将图像 VAE 编码为 latent，支持 IC-LoRA 低分辨率编码。"""
        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        images = images[:(images.shape[0] - 1) // time_scale_factor * time_scale_factor + 1]
        target_width = int(latent_width * width_scale_factor / latent_downscale_factor)
        target_height = int(latent_height * height_scale_factor / latent_downscale_factor)
        pixels = comfy.utils.common_upscale(
            images.movedim(-1, 1), target_width, target_height, "bilinear", crop="center"
        ).movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)
        return encode_pixels, t

    @classmethod
    def get_reference_downscale_factor(cls, iclora_parameters):
        if not iclora_parameters:
            return 1
        try:
            factor = max(1, round(float(iclora_parameters.get("reference_downscale_factor", 1))))
        except (TypeError, ValueError):
            factor = 1
        return factor

    @classmethod
    def _is_msr_model(cls, model):
        """检测模型是否加载了 MSR 类 LoRA：元数据声明 reference_slot_embedding_enabled/dim 且时间偏移为负时触发，元数据为空的普通 LoRA（如 MSR-V2）不触发。"""
        if model is None:
            return False
        try:
            meta = model.get_attachment("lora_metadata")
        except Exception:
            return False
        if not meta:
            return False
        try:
            offsets = str(meta.get("reference_slot_time_offsets", "pic1_based_negative_time"))
            if "negative" not in offsets:
                return False
            enabled_raw = meta.get("reference_slot_embedding_enabled", False)
            if isinstance(enabled_raw, bool):
                enabled = enabled_raw
            else:
                enabled = str(enabled_raw).strip().lower() in {"1", "true", "yes", "on"}
            dim_raw = meta.get("reference_slot_embedding_dim", 0)
            try:
                dim = int(float(str(dim_raw)))
            except (TypeError, ValueError):
                dim = 0
            return enabled or dim > 0
        except Exception:
            return False

    _msr_slot_state_cache = {}

    @classmethod
    def _load_msr_slot_state(cls, lora_path):
        """从 MSR LoRA 文件加载学习到的 reference_slot_embedding 槽位嵌入权重（带缓存）；元数据为空的普通 LoRA 返回 None。"""
        try:
            import safetensors
        except ImportError:
            return None
        if not lora_path or not os.path.isfile(lora_path):
            return None
        try:
            mtime = os.path.getmtime(lora_path)
        except OSError:
            mtime = 0.0
        cache_key = (lora_path, mtime)
        if cache_key in cls._msr_slot_state_cache:
            return cls._msr_slot_state_cache[cache_key]
        state = None
        try:
            with safetensors.safe_open(lora_path, framework="pt") as f:
                meta = f.metadata() or {}
                enabled_raw = meta.get("reference_slot_embedding_enabled", False)
                if isinstance(enabled_raw, bool):
                    enabled = enabled_raw
                else:
                    enabled = str(enabled_raw).strip().lower() in {"1", "true", "yes", "on"}
                if enabled:
                    offsets = str(meta.get("reference_slot_time_offsets", "pic1_based_negative_time"))
                    if "negative" in offsets:
                        prefixes = ("reference_slot_embedding.", "diffusion_model.reference_slot_embedding.")
                        state = {}
                        for key in f.keys():
                            for prefix in prefixes:
                                if key.startswith(prefix):
                                    state[key[len(prefix):]] = f.get_tensor(key).detach().cpu()
                                    break
                        required = {"frequencies", "net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"}
                        if not required.issubset(state):
                            state = None
        except Exception:
            state = None
        cls._msr_slot_state_cache[cache_key] = state
        return state

    @classmethod
    def _msr_slot_embedding(cls, slot_id, state, device, dtype):
        """MSR 的 Fourier-MLP slot embedding 计算（slot_id 从 1 起）。"""
        frequencies = state["frequencies"].to(device=device, dtype=torch.float32)
        slot_value = torch.tensor(float(slot_id), device=device, dtype=torch.float32)
        scaled = slot_value / 16.0
        phases = scaled * frequencies
        features = torch.cat((scaled.reshape(1), torch.sin(phases), torch.cos(phases)))
        weight0 = state["net.0.weight"].to(device=device, dtype=torch.float32)
        bias0 = state["net.0.bias"].to(device=device, dtype=torch.float32)
        hidden = torch.nn.functional.silu(torch.nn.functional.linear(features, weight0, bias0))
        weight2 = state["net.2.weight"].to(device=device, dtype=torch.float32)
        bias2 = state["net.2.bias"].to(device=device, dtype=torch.float32)
        embedding = torch.nn.functional.linear(hidden, weight2, bias2)
        return embedding.to(dtype=dtype)

    @classmethod
    def _resolve_msr_slot_state(cls, model, msr_lora):
        """按 MSR_LORA 选项解析槽位嵌入权重来源：MSR2.5 用 LTX-2.5-Licon-MSR-V1；MSR2.3 返回 None；auto 在 loras 目录中查找第一个元数据声明 MSR 的 LoRA。"""
        if model is None:
            return None
        keyword = None
        if msr_lora and msr_lora != "auto":
            if msr_lora == "MSR2.3":
                return None
            keyword = "LTX-2.5"
        try:
            names = folder_paths.get_filename_list("loras")
        except Exception:
            return None
        for name in names:
            if keyword is not None and (keyword not in name or "MSR" not in name):
                continue
            try:
                path = folder_paths.get_full_path("loras", name)
            except Exception:
                path = None
            if not path:
                continue
            state = cls._load_msr_slot_state(path)
            if state is not None:
                return state
        return None

    @classmethod
    def get_latent_index(cls, cond, latent_length, guide_length, frame_idx, scale_factors, latent_shape=None):
        """将像素帧索引转换为 latent 帧索引。"""
        time_scale_factor, _, _ = scale_factors
        _, num_keyframes = get_keyframe_idxs(cond, latent_shape)
        latent_count = latent_length - num_keyframes
        frame_idx = frame_idx if frame_idx >= 0 else max((latent_count - 1) * time_scale_factor + 1 + frame_idx, 0)
        if guide_length > 1 and frame_idx != 0:
            frame_idx = (frame_idx - 1) // time_scale_factor * time_scale_factor + 1
        latent_idx = (frame_idx + time_scale_factor - 1) // time_scale_factor
        return frame_idx, latent_idx

    @classmethod
    def add_keyframe_index(cls, cond, frame_idx, guiding_latent, scale_factors, latent_downscale_factor=1, causal_fix=None):
        """将引导 latent 的 patchify 坐标追加到 conditioning 的 keyframe_idxs 中。"""
        keyframe_idxs, _ = get_keyframe_idxs(cond)
        _, latent_coords = cls.PATCHIFIER.patchify(guiding_latent)
        if causal_fix is None:
            causal_fix = frame_idx == 0 or guiding_latent.shape[2] == 1
        pixel_coords = latent_to_pixel_coords(latent_coords, scale_factors, causal_fix=causal_fix)
        pixel_coords[:, 0] += frame_idx

        # IC-LoRA 小网格的 RoPE 端点修正
        spatial_end_offset = (latent_downscale_factor - 1) * torch.tensor(
            scale_factors[1:], device=pixel_coords.device,
        ).view(1, -1, 1, 1)
        pixel_coords[:, 1:, :, 1:] += spatial_end_offset.to(pixel_coords.dtype)

        if keyframe_idxs is None:
            keyframe_idxs = pixel_coords
        else:
            keyframe_idxs = torch.cat([keyframe_idxs, pixel_coords], dim=2)
        return node_helpers.conditioning_set_values(cond, {"keyframe_idxs": keyframe_idxs})

    @classmethod
    def append_keyframe(cls, positive, negative, frame_idx, latent_image, noise_mask,
                        guiding_latent, strength, scale_factors,
                        guide_mask=None, in_channels=128, latent_downscale_factor=1, causal_fix=None):
        """核心方法：拼接引导 latent 到主 latent 末尾，更新 noise_mask 和 keyframe_idxs。"""
        if latent_image.shape[1] != in_channels or guiding_latent.shape[1] != in_channels:
            raise ValueError("Adding guide to a combined AV latent is not supported.")

        positive = cls.add_keyframe_index(
            positive, frame_idx, guiding_latent, scale_factors, latent_downscale_factor, causal_fix=causal_fix
        )
        negative = cls.add_keyframe_index(
            negative, frame_idx, guiding_latent, scale_factors, latent_downscale_factor, causal_fix=causal_fix
        )

        if guide_mask is not None:
            target_h = max(noise_mask.shape[3], guide_mask.shape[3])
            target_w = max(noise_mask.shape[4], guide_mask.shape[4])
            if noise_mask.shape[3] == 1 or noise_mask.shape[4] == 1:
                noise_mask = noise_mask.expand(-1, -1, -1, target_h, target_w)
            if guide_mask.shape[3] == 1 or guide_mask.shape[4] == 1:
                guide_mask = guide_mask.expand(-1, -1, -1, target_h, target_w)
            mask = guide_mask - strength
        else:
            mask = torch.full(
                (noise_mask.shape[0], 1, guiding_latent.shape[2], noise_mask.shape[3], noise_mask.shape[4]),
                max(0.0, 1.0 - strength),
                dtype=noise_mask.dtype,
                device=noise_mask.device,
            )

        # 音视频混合 latent 的通道对齐
        if latent_image.shape[1] > guiding_latent.shape[1]:
            pad_len = latent_image.shape[1] - guiding_latent.shape[1]
            guiding_latent = torch.nn.functional.pad(
                guiding_latent, pad=(0, 0, 0, 0, 0, 0, 0, pad_len), value=0
            )

        latent_image = torch.cat([latent_image, guiding_latent], dim=2)
        noise_mask = torch.cat([noise_mask, mask], dim=2)
        return positive, negative, latent_image, noise_mask

    @classmethod
    def _encode_for_timeline(cls, vae, latent_width, latent_height, images, scale_factors,
                             latent_downscale_factor, resize_method="crop"):
        """Timeline 模式的 VAE 编码，支持多种 resize 方法。"""
        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        num_frames_to_keep = ((images.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        images = images[:num_frames_to_keep]
        target_width = int(latent_width * width_scale_factor / latent_downscale_factor)
        target_height = int(latent_height * height_scale_factor / latent_downscale_factor)
        target_width = max(8, target_width)
        target_height = max(8, target_height)
        # IC-LoRA 引导必须匹配精确的 latent 形状
        if resize_method == "maintain aspect ratio":
            resize_method = "pad"
        pixels = _resize_image(images, target_width, target_height, resize_method, divisible_by=1)
        encode_pixels = pixels[:, :, :, :3]
        guide_latent = vae.encode(encode_pixels)
        return pixels, guide_latent

    @classmethod
    def _dilate_latent_with_mask(cls, guide_latent, guide_mask, latent_downscale_factor):
        """IC-LoRA 空间扩张（保留原始 mask 值）。"""
        if latent_downscale_factor <= 1:
            return guide_latent, guide_mask
        scale = int(latent_downscale_factor)
        dilated_shape = guide_latent.shape[:3] + (
            guide_latent.shape[3] * scale,
            guide_latent.shape[4] * scale,
        )
        dilated_latent = torch.zeros(dilated_shape, device=guide_latent.device, dtype=guide_latent.dtype)
        dilated_latent[..., ::scale, ::scale] = guide_latent
        dilated_mask_shape = (
            guide_mask.shape[0], 1, guide_mask.shape[2],
            guide_mask.shape[3] * scale, guide_mask.shape[4] * scale,
        )
        dilated_mask = torch.full(dilated_mask_shape, -1.0, device=guide_latent.device, dtype=guide_latent.dtype)
        dilated_mask[..., ::scale, ::scale] = guide_mask
        return dilated_latent, dilated_mask

    @classmethod
    def _execute_timeline(cls, positive, negative, vae, latent, guide_data, motion_guide_data,
                          iclora_parameters=None, model=None, image_attention_strength=1.0,
                          msr_lora="auto"):
        """Timeline 批量引导模式：处理 guide_data 和 motion_guide_data。"""
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"].clone()
        noise_mask = get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        initial_latent_length = int(latent_length)

        latent_downscale_factor = cls.get_reference_downscale_factor(iclora_parameters)

        motion_segments = (motion_guide_data or {}).get("segments", []) if motion_guide_data else []
        images = guide_data.get("images", []) if guide_data else []
        insert_frames = guide_data.get("insert_frames", []) if guide_data else []
        strengths = guide_data.get("strengths", []) if guide_data else []

        active_resize_method = guide_data.get("resize_method") if guide_data else None
        if not active_resize_method:
            active_resize_method = (motion_guide_data or {}).get("resize_method") if motion_guide_data else None
        if not active_resize_method:
            active_resize_method = "crop"

        director_fps = float(
            (motion_guide_data or {}).get("frame_rate",
                                          guide_data.get("frame_rate", 24) if guide_data else 24)
        )
        segments = motion_segments
        time_scale_factor = scale_factors[0]

        # MSR_LORA：MSR2.5 强制启用（负偏移+槽位嵌入），MSR2.3 强制关闭，auto 按 LoRA 元数据自动检测
        if msr_lora == "MSR2.5":
            msr_mode = True
        elif msr_lora == "MSR2.3":
            msr_mode = False
        else:
            msr_mode = cls._is_msr_model(model)
        # MSR 模式加载槽位嵌入权重（加到参考图 latent 通道上恢复人物身份绑定）
        msr_slot_state = cls._resolve_msr_slot_state(model, msr_lora) if msr_mode else None

        # -----------------------------------------------------------------------
        # 标准 Timeline 引导模式：将图像分段和视频分段作为关键帧注入 latent
        # -----------------------------------------------------------------------
        if len(images) > 0 or len(segments) > 0:
            # A. 处理图像引导：主轨图按 insert_frames 正帧注入，不参与 MSR 参考槽位分配（槽位仅由 IC-LoRA 轨的 @图X 静态参考图独占）
            for idx, img_tensor in enumerate(images):
                f_idx = insert_frames[idx] if idx < len(insert_frames) else 0
                img_strength = float(strengths[idx] if idx < len(strengths) else 1.0)
                if img_strength <= 0.0:
                    continue

                _, H_img, W_img, _ = img_tensor.shape
                target_pix_w = int(latent_width * 32)
                target_pix_h = int(latent_height * 32)
                if target_pix_w != W_img or target_pix_h != H_img:
                    img_nchw = img_tensor.permute(0, 3, 1, 2)
                    img_resized = comfy.utils.common_upscale(img_nchw, target_pix_w, target_pix_h, "bicubic", "disabled")
                    img_tensor = img_resized.permute(0, 2, 3, 1)

                image_pixels, guide_latent = cls.encode(
                    vae, latent_width, latent_height, img_tensor, scale_factors, latent_downscale_factor
                )

                frame_idx, latent_idx = cls.get_latent_index(
                    positive, latent_length, len(image_pixels), int(f_idx), scale_factors
                )

                if latent_idx >= latent_length:
                    continue

                max_frames = latent_length - latent_idx
                if guide_latent.shape[2] > max_frames:
                    guide_latent = guide_latent[:, :, :max_frames]

                guide_orig_shape = list(guide_latent.shape[2:])

                # IC-LoRA 空间扩张：保证 LTXVCropGuides 按 keyframe_idxs 裁剪的帧数等于实际追加的 latent 帧数，避免尾部残留参考帧
                guide_mask = None
                if latent_downscale_factor > 1:
                    B_g, _, F_g, H_g, W_g = guide_latent.shape
                    guide_mask = torch.ones(
                        (B_g, 1, F_g, H_g, W_g), device=guide_latent.device, dtype=guide_latent.dtype
                    )
                    guide_latent, guide_mask = cls._dilate_latent_with_mask(
                        guide_latent, guide_mask, latent_downscale_factor
                    )

                tokens_added = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
                positive, negative, latent_image, noise_mask = cls.append_keyframe(
                    positive, negative, frame_idx, latent_image, noise_mask,
                    guide_latent, img_strength, scale_factors,
                    guide_mask=guide_mask,
                    latent_downscale_factor=latent_downscale_factor,
                )
                positive, negative = _append_guide_attention_entry(
                    positive, negative, tokens_added, guide_orig_shape, strength=float(image_attention_strength),
                )

            # B. 处理视频分段
            # MSR 模式：静态参考图段不合并，各槽位独立按负时间偏移追加；非 MSR 模式：自动合并相邻静态图像段（isStaticImage 且前段 end == 当前段 start）为合成视频序列
            merged_segments = []
            i_seg = 0
            while i_seg < len(segments):
                seg = segments[i_seg]
                is_static = bool(seg.get("isStaticImage", False))
                if is_static and not msr_mode:
                    # 收集连续相邻的静态图像段
                    batch = [seg]
                    j = i_seg + 1
                    while j < len(segments):
                        nxt = segments[j]
                        if (bool(nxt.get("isStaticImage", False))
                                and int(nxt.get("start", 0)) == int(seg.get("start", 0)) + int(seg.get("length", 0))):
                            batch.append(nxt)
                            seg = nxt  # 更新尾部，用于判断下一段是否继续相邻
                            j += 1
                            continue
                        break
                    if len(batch) > 1:
                        # 合并为一段：frameFiles 收集所有图，length 为总和，description 逐行合并（对应 @图X=描述）
                        merged_desc = "\n".join(
                            b.get("description", "") for b in batch if b.get("description")
                        )
                        merged_seg = {
                            "videoFile": batch[0].get("videoFile"),
                            "frameFiles": [b.get("videoFile") for b in batch],
                            "start": int(batch[0].get("start", 0)),
                            "length": sum(int(b.get("length", 0)) for b in batch),
                            "trimStart": 0.0,
                            "isStaticImage": True,
                            "resampleMode": batch[0].get("resampleMode", "nearest"),
                            "description": merged_desc,
                        }
                        merged_segments.append(merged_seg)
                    else:
                        merged_segments.append(seg)
                    i_seg = j
                else:
                    merged_segments.append(seg)
                    i_seg += 1

            # MSR 模式：静态参考图段按 @图X 编号（subjectNum）绑定槽位，num_slots 取槽位最大值
            _msr_slot_b = 0
            _msr_total_b = 0
            if msr_mode:
                _slot_nums = []
                for _seg in merged_segments:
                    if (bool(_seg.get("isStaticImage", False))
                            and float(_seg.get("videoStrength", 1.0)) > 0.0
                            and int(_seg.get("length", 1)) > 0):
                        _slot_nums.append(int(_seg.get("subjectNum", 0)))
                _valid_nums = [n for n in _slot_nums if n > 0]
                _msr_total_b = max(_valid_nums) if _valid_nums else len(_slot_nums)

            # 补帧对齐：合并段总帧数须满足 (N-1)%time_scale_factor==0，否则截断会丢帧
            for seg in merged_segments:
                if seg.get("isStaticImage"):
                    total_len = seg.get("length", 0)
                    if total_len > 0 and (total_len - 1) % time_scale_factor != 0:
                        seg["length"] = ((total_len - 1) // time_scale_factor + 1) * time_scale_factor + 1

            for seg in merged_segments:
                try:
                    video_file = seg.get("videoFile")
                    if not video_file:
                        continue

                    start_frame = int(seg.get("start", 0))
                    length_frames = int(seg.get("length", 1))
                    trim_start = int(seg.get("trimStart", 0))
                    video_strength = float(seg.get("videoStrength", 1.0))
                    video_attention_strength = 1.0

                    if length_frames <= 0 or video_strength <= 0.0:
                        continue

                    video_frames = _load_motion_video_frames(
                        video_file, trim_start, length_frames, director_fps,
                        seg.get("resampleMode", "nearest"),
                        frame_files=seg.get("frameFiles"),
                    )

                    num_frames_to_keep = ((video_frames.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
                    video_frames = video_frames[:num_frames_to_keep]

                    # MSR 模式静态参考图段：按原生 MSR 约定 frame_offset=-(num_slots-slot_index) 负时间偏移追加到视频起点前，
                    # 跳过正帧换算并 causal_fix=True，由 LTXVCropGuides 按 keyframe_idxs 裁剪
                    if msr_mode and bool(seg.get("isStaticImage", False)):
                        # 槽位编号 = @图X 编号（subjectNum），缺失时回退为排列顺序
                        slot_id = int(seg.get("subjectNum", 0))
                        if slot_id <= 0:
                            _msr_slot_b += 1
                            slot_id = _msr_slot_b
                        # 原生 MSR 约定：slot k 的时间偏移 = -(num_slots - k + 1)
                        frame_idx = -(_msr_total_b - slot_id + 1)
                        _, guide_latent = cls._encode_for_timeline(
                            vae, latent_width, latent_height, video_frames, scale_factors,
                            latent_downscale_factor, resize_method=active_resize_method,
                        )
                        # 注入原生 MSR 的学习槽位嵌入，恢复人物身份绑定
                        if msr_slot_state is not None:
                            emb = cls._msr_slot_embedding(slot_id, msr_slot_state, guide_latent.device, guide_latent.dtype)
                            if emb.numel() == guide_latent.shape[1]:
                                guide_latent = guide_latent + emb.view(1, -1, 1, 1, 1)
                        guide_orig_shape = list(guide_latent.shape[2:])
                        guide_mask = None
                        if latent_downscale_factor > 1:
                            B_g, _, F_g, H_g, W_g = guide_latent.shape
                            guide_mask = torch.ones(
                                (B_g, 1, F_g, H_g, W_g), device=guide_latent.device, dtype=guide_latent.dtype
                            )
                            guide_latent, guide_mask = cls._dilate_latent_with_mask(
                                guide_latent, guide_mask, latent_downscale_factor
                            )
                        tokens_added = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
                        positive, negative, latent_image, noise_mask = cls.append_keyframe(
                            positive, negative, frame_idx, latent_image, noise_mask,
                            guide_latent, video_strength, scale_factors,
                            guide_mask=guide_mask,
                            latent_downscale_factor=latent_downscale_factor,
                            causal_fix=True,
                        )
                        positive, negative = _append_guide_attention_entry(
                            positive, negative, tokens_added, guide_orig_shape,
                            strength=video_attention_strength,
                        )
                        continue

                    causal_fix = int(start_frame) == 0 or num_frames_to_keep == 1
                    encode_frames = video_frames if causal_fix else torch.cat([video_frames[:1], video_frames], dim=0)

                    _, guide_latent = cls._encode_for_timeline(
                        vae, latent_width, latent_height, encode_frames, scale_factors,
                        latent_downscale_factor, resize_method=active_resize_method,
                    )

                    if not causal_fix:
                        guide_latent = guide_latent[:, :, 1:, :, :]

                    frame_idx = start_frame
                    latent_idx = (frame_idx + time_scale_factor - 1) // time_scale_factor if frame_idx > 0 else 0

                    if latent_idx >= latent_length:
                        continue

                    if start_frame > 0 and guide_latent.shape[2] > 1:
                        guide_latent = guide_latent[:, :, 1:, :, :]
                        frame_idx += time_scale_factor
                        latent_idx += 1
                        if latent_idx >= latent_length:
                            continue

                    max_frames = latent_length - latent_idx
                    if guide_latent.shape[2] > max_frames:
                        guide_latent = guide_latent[:, :, :max_frames]

                    guide_orig_shape = list(guide_latent.shape[2:])

                    B_g, _, F_g, H_g, W_g = guide_latent.shape
                    guide_mask = torch.ones(
                        (B_g, 1, F_g, H_g, W_g), device=guide_latent.device, dtype=guide_latent.dtype
                    )

                    # 非首帧引导：前几帧使用渐变 mask 平滑过渡
                    if start_frame > 0:
                        ramp_steps = [0.25, 0.65]
                        for i, s in enumerate(ramp_steps):
                            if i < F_g:
                                guide_mask[:, :, i, :, :] = 1.0 + video_strength * (1.0 - s)

                    if latent_downscale_factor > 1:
                        guide_latent, guide_mask = cls._dilate_latent_with_mask(
                            guide_latent, guide_mask, latent_downscale_factor
                        )

                    tokens_added = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
                    positive, negative, latent_image, noise_mask = cls.append_keyframe(
                        positive, negative, frame_idx, latent_image, noise_mask,
                        guide_latent, video_strength, scale_factors,
                        guide_mask=guide_mask,
                        latent_downscale_factor=latent_downscale_factor,
                        causal_fix=causal_fix,
                    )
                    positive, negative = _append_guide_attention_entry(
                        positive, negative, tokens_added, guide_orig_shape,
                        strength=video_attention_strength,
                    )
                except Exception as e:
                    raise RuntimeError(f"Yuan 引导注入视频分段处理失败 {seg}: {e}") from e

        exact_crop_frames = max(0, int(latent_image.shape[2]) - initial_latent_length)
        positive = node_helpers.conditioning_set_values(positive, {"nghtdrp_guide_crop_latent_frames": exact_crop_frames})
        negative = node_helpers.conditioning_set_values(negative, {"nghtdrp_guide_crop_latent_frames": exact_crop_frames})

        # --- K/V 视觉特征注入：在启用且 model 可用时接管 attn2 forward ---
        ref_alpha = float(guide_data.get("ref_alpha", 0.0)) if guide_data else 0.0
        marker_token_indices = guide_data.get("marker_token_indices") if guide_data else None
        if ref_alpha > 0.0 and model is not None and marker_token_indices:
            # 为每个 @图X 参考图独立预计算视觉特征（不依赖 x 中参考帧的物理位置，解耦坐标系错位）；
            # 单个主体编码失败只跳过该主体，不中断整体执行
            subject_ref_features = {}
            try:
                diffusion_model = model.get_model_object("diffusion_model")
                patchify_proj = diffusion_model.patchify_proj
            except Exception:
                patchify_proj = None

            if patchify_proj is None:
                pass  # patchify_proj 缺失时禁用 @图X K/V 视觉特征注入，不打印
            else:
                for idx, seg in enumerate(segments):
                    raw_num = seg.get("subjectNum")
                    if isinstance(raw_num, int) and raw_num > 0:
                        subject_num = raw_num
                    else:
                        subject_num = idx + 1
                    if subject_num not in marker_token_indices:
                        continue
                    try:
                        video_file = seg.get("videoFile")
                        if not video_file:
                            continue
                        length_frames = int(seg.get("length", 1))
                        if length_frames <= 0:
                            continue
                        video_frames = _load_motion_video_frames(
                            video_file, 0, length_frames, director_fps,
                            seg.get("resampleMode", "nearest"),
                            frame_files=seg.get("frameFiles"),
                        )
                        _, guide_latent = cls._encode_for_timeline(
                            vae, latent_width, latent_height, video_frames, scale_factors,
                            latent_downscale_factor, resize_method=active_resize_method,
                        )
                        # patchify + patchify_proj → visual token [B, F*H*W, inner_dim]
                        patches, _ = cls.PATCHIFIER.patchify(guide_latent)
                        patches = patchify_proj(patches)
                        # 中心区域加权均值 → ref_summary [B, inner_dim]
                        F_g = guide_latent.shape[2]
                        H_g = guide_latent.shape[3]
                        W_g = guide_latent.shape[4]
                        weight_grid = torch.full((H_g, W_g), 0.3, device=patches.device, dtype=patches.dtype)
                        ch_start, ch_end = H_g // 4, H_g - H_g // 4
                        cw_start, cw_end = W_g // 4, W_g - W_g // 4
                        if ch_end > ch_start and cw_end > cw_start:
                            weight_grid[ch_start:ch_end, cw_start:cw_end] = 1.0
                        ref_weights = weight_grid.flatten().repeat(F_g)
                        ref_weights_expanded = ref_weights[None, :, None]
                        ref_summary = (patches * ref_weights_expanded).sum(dim=1) / ref_weights.sum().clamp_min(1e-6)
                        subject_ref_features[int(subject_num)] = ref_summary
                    except Exception:
                        # 单个主体编码失败时跳过，不中断整体执行
                        continue
            if subject_ref_features:
                model = cls._apply_ref_guidance_patch(
                    model, marker_token_indices, ref_alpha,
                    subject_ref_features=subject_ref_features,
                )

        return (model, positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

    @classmethod
    def _apply_ref_guidance_patch(cls, model, marker_token_indices, ref_alpha, subject_ref_features=None):
        """把 K/V 注入参数绑定到 attn2 并替换其 forward（仅视频分支，与 PromptRelay 兼容）。"""
        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")

        for idx, block in enumerate(diffusion_model.transformer_blocks):
            patched_attn2 = _LTXVCrossAttentionRefPatch(
                marker_token_indices, ref_alpha,
                subject_ref_features=subject_ref_features,
            ).__get__(block.attn2, block.__class__)
            model_clone.add_object_patch(
                f"diffusion_model.transformer_blocks.{idx}.attn2.forward", patched_attn2,
            )

        return model_clone

    def execute(self, 模型, 正向条件, 负向条件, VAE, 潜空间, 引导数据=None, 运动引导数据=None,
                IC_LORA参数=None, 图像注意力强度=1.0, MSR_LORA="auto"):
        """节点入口：Timeline 批量引导模式。"""
        return self._execute_timeline(
            正向条件, 负向条件, VAE, 潜空间, 引导数据, 运动引导数据,
            IC_LORA参数, 模型, 图像注意力强度, MSR_LORA,
        )


NODE_CLASS_MAPPINGS = {
    "YuanClipGuide": YuanClipGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanClipGuide": "Yuan 引导注入 (Guide)",
}
