"""MiniMax H3 节点：图生视频 / 参考图生视频 / 数字人，构建 AV 联合潜空间与任务条件。"""

import math

import torch
import torchaudio

import nodes
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers

try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except Exception:  # 旧版 ComfyUI 无 H3 模型模块时用相同数值兜底
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0


# 旧版核心（< v0.34.0）H3 修复反向移植；新版核心已含修复时自动跳过。
#   1) PackedLayout 支持首/末帧锚点并保留关键帧音频 latent；2) 关键帧/参考共存时条件与关键帧音频正确合并。
try:
    import inspect

    import comfy.ldm.minimax.model as _h3_model
    import comfy.model_base as _model_base

    if "frame_count" in inspect.signature(_h3_model.PackedLayout.__init__).parameters:
        # 与 Yuan_H3_Motion.py 共享的 ABI 标记（改名须两文件同步）
        BACKPORT_MARKER_LAYOUT = "_yuan_minimax_h3_v034_layout"
        BACKPORT_MARKER_PAYLOAD = "_yuan_minimax_h3_v034_extra_conds"

        def _h3_ref_t_span(blk):
            # 参考块在目标流之前占据的时间轴跨度（v0.34.0 同名函数）
            kind = blk["kind"]
            if kind == "image":
                return 1.0
            if kind == "audio":
                return float(blk["ref_audio_t"])
            if kind in ("video", "video_audio"):
                return max(float(blk["ref_audio_t"]), sum(_h3_model._video_t_spans(blk["latent_t"])))
            return 0.0

        def _h3_packed_layout_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                                   keyframes=None, refs=None, frame_count=None):
            # v0.34.0 PackedLayout 逻辑；frame_count 仅兼容旧调用点保留，不再使用。
            frame, w_grid = _h3_model._frame_grid(latent_h, latent_w)
            frame_rows = frame.shape[0]

            segments = [("text", text_len)]
            g = torch.zeros(text_len, 3, dtype=torch.float64)
            g[:, 0] = torch.arange(text_len, dtype=torch.float64)
            pos = [g]

            img_pos, img_update = [], []
            audio_pos, audio_update = [], []
            row = text_len

            target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
            # 参考块排在文本与目标流之间，目标时间轴起点在参考块跨度之后
            cursor = float(text_len)
            for blk in refs or ():
                cursor += _h3_ref_t_span(blk)

            if keyframes:
                # 锚点从目标时间轴起点计数：每像素帧 FRAME_RESCALE，每音频 latent 帧 1.0
                for kf in keyframes:
                    cond_t = cursor + _h3_model.FRAME_RESCALE * kf["resolved_frame_index"]
                    video_latent = kf.get("latent")
                    if video_latent is not None:
                        vt = video_latent.shape[2]
                        n = vt * frame_rows
                        segments.append(("cond", n))
                        pos.append(_h3_model._video_grid(vt, frame, cond_t))
                        img_pos.append(torch.arange(row, row + n))
                        img_update.append(torch.zeros(n, dtype=torch.bool))
                        row += n
                    audio_latent = kf.get("audio_latent")
                    if audio_latent is not None:
                        rt = audio_latent.shape[-1]
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_h3_model._audio_grid(cond_t, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2

            if refs:
                cursor = float(text_len)
                for blk in refs:
                    kind = blk["kind"]
                    if kind == "image":
                        r_frame, _ = _h3_model._frame_grid(blk["latent_h"], blk["latent_w"])
                        n = r_frame.shape[0]
                        g = torch.empty(n, 3, dtype=torch.float64)
                        g[:, 0] = cursor
                        g[:, 1:] = r_frame
                        segments.append(("ref_img", n))
                        pos.append(g)
                        img_pos.append(torch.arange(row, row + n))
                        img_update.append(torch.zeros(n, dtype=torch.bool))
                        row += n
                        cursor += 1.0
                    elif kind == "audio":
                        rt = blk["ref_audio_t"]
                        if rt > 0:
                            segments.append(("ref_audio", rt * 2))
                            pos.append(_h3_model._audio_grid(cursor, rt, *target_audio_w))
                            audio_pos.append(torch.arange(row, row + rt * 2))
                            audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                            row += rt * 2
                        cursor += float(rt)
                    elif kind in ("video", "video_audio"):
                        # 音轨行紧贴在视频行之前，两者共享同一游标起点
                        rt = blk["ref_audio_t"]
                        vt = blk["latent_t"]
                        r_frame, r_w_grid = _h3_model._frame_grid(blk["latent_h"], blk["latent_w"])
                        if rt > 0:
                            segments.append(("ref_audio", rt * 2))
                            pos.append(_h3_model._audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                            audio_pos.append(torch.arange(row, row + rt * 2))
                            audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                            row += rt * 2
                        n = vt * r_frame.shape[0]
                        segments.append(("ref_img", n))
                        pos.append(_h3_model._video_grid(vt, r_frame, cursor))
                        img_pos.append(torch.arange(row, row + n))
                        img_update.append(torch.zeros(n, dtype=torch.bool))
                        row += n
                        cursor += max(float(rt), sum(_h3_model._video_t_spans(vt)))

            # 目标音频与目标视频始终是最后两段
            segments.append(("audio", audio_t * 2))
            pos.append(_h3_model._audio_grid(cursor, audio_t, *target_audio_w))
            audio_pos.append(torch.arange(row, row + audio_t * 2))
            audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
            row += audio_t * 2

            n_video = latent_t * frame_rows
            segments.append(("video", n_video))
            pos.append(_h3_model._video_grid(latent_t, frame, cursor))
            img_pos.append(torch.arange(row, row + n_video))
            img_update.append(torch.ones(n_video, dtype=torch.bool))
            row += n_video

            self.seq_len = row
            self.position_ids = torch.cat(pos)
            self.img_pos = torch.cat(img_pos)
            self.img_update = torch.cat(img_update)
            self.audio_pos = torch.cat(audio_pos)
            self.audio_update = torch.cat(audio_update)
            self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
            seg_abs = []
            off = 0
            for kind, n in segments:
                seg_abs.append((off, off + n, kind))
                off += n
            self.segments = seg_abs

        _h3_model.PackedLayout.__init__ = _h3_packed_layout_init
        setattr(_h3_packed_layout_init, BACKPORT_MARKER_LAYOUT, True)

        # 旧核心 extra_conds 的缺陷：关键帧缺 latent 键直接 KeyError（纯音频引导）、
        # 关键帧与参考共存时参考列表覆盖关键帧列表、关键帧音频 latent 不收集
        _h3_orig_extra_conds = _model_base.MiniMaxH3.extra_conds

        class _H3KeyframeDict(dict):
            # 旧核心按 kf["latent"] 直接取键，纯音频关键帧会 KeyError；包装后缺失键
            # 返回 None，随后由本包装函数按 v0.34.0 规则重建条件 latent 列表
            def __missing__(self, key):
                return None

        def _h3_extra_conds(self, **kwargs):
            keyframes = kwargs.get("minimax_keyframes")
            if keyframes:
                kwargs["minimax_keyframes"] = [_H3KeyframeDict(kf) for kf in keyframes]
            out = _h3_orig_extra_conds(self, **kwargs)
            pc = out.get("minimax_payload")
            if pc is not None and keyframes:
                payload = pc.cond
                refs = kwargs.get("minimax_refs")
                payload["cond_video_latents"] = [kf["latent"] for kf in keyframes if kf.get("latent") is not None]
                audio_latents = [kf["audio_latent"] for kf in keyframes if kf.get("audio_latent") is not None]
                if refs:
                    payload["cond_video_latents"] += [r["latent"] for r in refs if "latent" in r]
                    audio_latents += [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]
                if audio_latents:
                    payload["cond_audio_latents"] = audio_latents
            return out

        _model_base.MiniMaxH3.extra_conds = _h3_extra_conds
        setattr(_h3_extra_conds, BACKPORT_MARKER_PAYLOAD, True)
except Exception:
    pass

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40

REF_IMAGE_PORTS = 9   # 参考图像最大数量（与原生 Autogrow max 一致）
REF_VIDEO_PORTS = 3   # 参考视频最大数量
REF_AUDIO_PORTS = 3   # 参考音频最大数量

MODE_IMAGE_TO_VIDEO = "图生视频"
MODE_REFERENCE = "参考图生视频"
MODE_GUIDE = "数字人"


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width, height):
    """768 短边画布，面积上限 768*1344，逐轴取整到 32 的倍数。"""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


def _is_empty_audio(audio, audio_vae=None):
    """空音频检测：未接入、无有效采样，或时长不足一个音频 latent 帧（800 采样≈25ms@32kHz）。

    分流节点在列表不足时会输出 1ms 静音占位；这类空音频不参与传递，等同未连接该端口。
    否则音频 VAE 编码前会把长度裁剪成压缩率的整数倍，不足一帧会得到 0 长度张量并崩溃。
    """
    if not isinstance(audio, dict):
        return True
    waveform = audio.get("waveform")
    if not isinstance(waveform, torch.Tensor) or waveform.numel() == 0 or waveform.shape[-1] == 0:
        return True
    samples = waveform.shape[-1]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    sr = audio.get("sample_rate") or vae_sr
    try:
        length = samples * float(vae_sr) / float(sr)
    except (TypeError, ValueError, ZeroDivisionError):
        length = float(samples)
    try:
        frame = audio_vae.spacial_compression_encode()
    except Exception:
        frame = getattr(audio_vae, "downscale_ratio", 800)
    if isinstance(frame, (list, tuple)):
        frame = frame[-1] if frame else 800
    try:
        frame = max(1, int(frame))
    except (TypeError, ValueError):
        frame = 800
    return length < frame


class YuanMiniMaxH3Video:
    """MiniMax-H3 视频生成（图生视频 / 参考图生视频 / 数字人），引导/参考 latent 写入 minimax_keyframes。"""

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("正向", "潜空间")
    OUTPUT_TOOLTIPS = ("正向条件（含关键帧/参考潜空间）", "视频+音频联合潜空间")
    FUNCTION = "execute"
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = "MiniMax-H3 视频生成：图生视频（首/尾帧关键帧）、参考图生视频（<Picture>/<Video>/<Audio> 参考）或数字人（引导图像/音频锚定到任意帧）。"

    @classmethod
    def INPUT_TYPES(cls):
        def image_port(display, tip, extra=None):
            cfg = {"optional": True, "display_name": display, "tooltip": tip}
            if extra:
                cfg.update(extra)
            return ("IMAGE", cfg)

        def audio_port(display, tip):
            return ("AUDIO", {"optional": True, "display_name": display, "tooltip": tip})

        optional = {
            # ---- 图生视频模式 ----
            "first_frame": image_port("首帧图像", "视频首帧图像，作为几何锚点"),
            "last_frame": image_port("尾帧图像", "视频尾帧图像"),
            # ---- 参考图生视频模式 ----
            "audio_vae": ("VAE", {"optional": True, "display_name": "音频VAE",
                                  "tooltip": "用于编码参考音频的音频 VAE 模型"}),
            "ref_image_size": (["匹配", "最大"], {"default": "匹配", "display_name": "参考图尺寸",
                "tooltip": "参考图像尺寸策略。'匹配'：将每张参考图（仅缩小、保持宽高比）缩放到生成画面的像素面积；'最大'：使用参考管线的 2048px 短边以获得最佳主体保真度。参考标记会贯穿每个采样步，'最大' 模式可能慢数倍。"}),
            "guide_frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999, "display_name": "锚定帧",
                "tooltip": "引导图像/音频锚定的帧位置（仅数字人模式）。负数从视频末尾倒数，如 -1 表示最后一帧"}),
            "ref_images": image_port(
                "参考图像", f"参考图像列表（可连接多张图像，最多 {REF_IMAGE_PORTS} 张，超出自动切断）"),
            # ---- 数字人模式（Add Guide）----
            "guide_image": image_port("引导图像",
                "锚定到指定帧的图像或多帧片段（仅数字人模式）。单帧图像直接锚定；多帧批次作为短片锚定，自动向下对齐到 17k+5 帧网格（5、22、39…），不足 5 帧只取首帧"),
            "guide_audio": audio_port("引导音频",
                "从锚定帧起对齐的语音/音轨（仅数字人模式），超出视频剩余时长自动截断"),
        }
        for i in range(1, REF_VIDEO_PORTS + 1):
            optional[f"ref_video_{i}"] = image_port(
                f"参考视频{i}", f"24fps 的参考视频帧（2-15 秒），第 {i} 路参考视频")
        for i in range(1, REF_AUDIO_PORTS + 1):
            optional[f"ref_video_audio_{i}"] = audio_port(
                f"参考视频音频{i}", f"与参考视频{i} 对应的音轨")
        for i in range(1, REF_AUDIO_PORTS + 1):
            optional[f"ref_audio_{i}"] = audio_port(
                f"参考音频{i}", f"独立的参考音频，第 {i} 路")

        return {
            "required": {
                "mode": ([MODE_IMAGE_TO_VIDEO, MODE_REFERENCE, MODE_GUIDE], {"default": MODE_IMAGE_TO_VIDEO,
                    "display_name": "模式", "tooltip": "生成模式：图生视频（首/尾帧关键帧）、参考图生视频（<Picture>/<Video>/<Audio> 参考）或数字人（引导图像/音频锚定到任意帧，复刻官方 Add Guide）"}),
                "clip": ("CLIP", {"display_name": "CLIP", "tooltip": "用于编码提示词的 CLIP 模型"}),
                "vae": ("VAE", {"display_name": "VAE", "tooltip": "用于编码关键帧/参考图像的 VAE 模型"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "display_name": "提示词",
                    "tooltip": "视频内容的提示词描述（参考模式下可用 <Picture i> / <Video k> / <Audio j> 引用参考内容）"}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                    "display_name": "宽度", "tooltip": "视频宽度，需为 32 的倍数"}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                    "display_name": "高度", "tooltip": "视频高度，需为 32 的倍数"}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                    "display_name": "时长（帧）",
                    "tooltip": "24fps 下的视频帧数，会自动向上对齐到模型的 17k+5 帧网格（124≈5秒；训练区间约 124-362，更长未经测试）"}),
            },
            "optional": optional,
        }

    def execute(self, mode, clip, vae, prompt, width, height, length,
                first_frame=None, last_frame=None,
                audio_vae=None, ref_image_size="match", ref_images=None,
                guide_frame_idx=0, guide_image=None, guide_audio=None,
                **kwargs):
        if mode == MODE_REFERENCE:
            return self._execute_reference(
                clip, vae, audio_vae, prompt, width, height, length,
                ref_image_size, ref_images, kwargs)
        if mode == MODE_GUIDE:
            # 数字人：纯引导锚定（不处理首/尾帧，对应端口在该模式下不显示）
            cond, latent = self._execute_image_to_video(clip, vae, prompt, width, height, length,
                                                        None, None)
            return (self._apply_guide(cond, latent, vae, audio_vae,
                                      guide_image, guide_audio, guide_frame_idx), latent)
        return self._execute_image_to_video(clip, vae, prompt, width, height, length,
                                            first_frame, last_frame)

    # ---- 图生视频（t2va / fl2va）----
    def _execute_image_to_video(self, clip, vae, prompt, width, height, length,
                                first_frame, last_frame):
        latent, frame_count = _empty_av_latent(width, height, length)

        images = []
        keyframes = []
        if first_frame is not None:
            # 几何锚点：直接拉伸到画布
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            # 跟随帧：保持宽高比的覆盖裁剪
            img = _resize(last_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)

        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            # minimax_frame_count：v0.34.0 核心已无消费者（PackedLayout 用
            # resolved_frame_index 原生支持末帧锚点），仅为旧核心兜底保留
            cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes,
                                                               "minimax_frame_count": frame_count})
        return (cond, latent)

    # ---- 数字人（Add Guide：任意帧锚定图像/音频引导）----
    def _apply_guide(self, cond, latent, vae, audio_vae, image, audio, frame_idx):
        """把引导图像/音频锚定到任意像素帧，追加进 minimax_keyframes。"""
        # 空音频（静音占位等）不参与传递，等同未连接该端口
        if _is_empty_audio(audio, audio_vae):
            audio = None
        if image is None and audio is None:
            raise ValueError("数字人模式需要至少连接引导图像（guide_image）或引导音频（guide_audio）")

        samples = latent["samples"]
        video = samples.tensors[0]
        height = video.shape[3] * 16
        width = video.shape[4] * 16
        # 由视频潜空间反推总帧数：首 token 1 帧，其后每 token 4 帧，5 个一组循环
        frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(video.shape[2]))

        # 引导图像：单帧直接锚定；多帧（>=5）作为短片，向下对齐到 17k+5 帧网格
        guide_frames = 1
        if image is not None:
            guide_frames = image.shape[0]
            if guide_frames < 5:
                guide_frames = 1
            else:
                while guide_frames % 17 != 5:
                    guide_frames -= 1

        resolved_frame_index = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        if resolved_frame_index < 0 or resolved_frame_index + guide_frames > frame_count:
            if guide_frames == 1:
                raise ValueError(f"锚定帧 {frame_idx} 超出视频的 {frame_count} 帧范围")
            raise ValueError(f"{guide_frames} 帧的引导片段在锚定帧 {frame_idx} 处放不进 {frame_count} 帧的视频")

        keyframe = {"resolved_frame_index": resolved_frame_index}
        if image is not None:
            # 保持宽高比的覆盖裁剪到画布
            frames = _resize(image[:guide_frames], width, height, "center")
            keyframe["latent"] = vae.encode(frames)

        if audio is not None:
            if audio_vae is None:
                raise ValueError("锚定引导音频需要连接音频 VAE（audio_vae）端口")
            audio_latent, audio_rt = _encode_ref_audio(audio_vae, audio)
            # 视频与音频共享同一时间轴：每个像素帧占 FRAME_RESCALE 个音频 latent 帧
            max_rt = math.floor(samples.tensors[1].shape[-1] - FRAME_RESCALE * resolved_frame_index)
            if max_rt < 1:
                raise ValueError(f"锚定帧 {frame_idx} 已越过视频音频轨的末尾")
            if audio_rt > max_rt:
                audio_latent = audio_latent[..., :max_rt].clone()
            keyframe["audio_latent"] = audio_latent

        # 写入 minimax_keyframes（数字人模式下引导是唯一关键帧）
        keyframes = list(cond[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)
        # 同样补写 minimax_frame_count（v0.34.0 核心无消费者，仅为旧核心兜底保留）
        return node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes,
                                                           "minimax_frame_count": frame_count})

    # ---- 参考图生视频（ref2va）----
    def _execute_reference(self, clip, vae, audio_vae, prompt, width, height, length,
                           ref_image_size, ref_images, kwargs):
        if audio_vae is None:
            raise ValueError("参考图生视频模式需要连接音频 VAE（audio_vae）端口")

        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items = []   # 供分词器按请求顺序呈现
        ref_blocks = []  # 供 DiT 负载，顺序一致
        # 参考图尺寸下拉框选项为中文（见 INPUT_TYPES），这里还原为内部处理用的英文 key
        _match_mode = "match" if ref_image_size in ("匹配", "match") else "max"

        # 1. 参考图像：ref_images 列表端口（batch 多图，最多 REF_IMAGE_PORTS 张，超出自动切断）
        if ref_images is not None:
            for img in ref_images[:REF_IMAGE_PORTS]:
                img = img.unsqueeze(0)  # 单张 [1, H, W, C]
                h, w = img.shape[1], img.shape[2]
                if _match_mode == "match":
                    # 保持宽高比的缩放（仅缩小）到生成画面的像素面积
                    scale = min(1.0, math.sqrt((width * height) / (w * h)))
                else:
                    scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
                tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                resized = _resize(img, tw, th, "disabled")
                z = vae.encode(resized)
                ref_items.append({"type": "image", "data": resized})
                ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        # 2. 参考视频：ref_video_1..N，配对的音轨 ref_video_audio_N 属于 ref_video_N
        for i in range(1, REF_VIDEO_PORTS + 1):
            video_frames = kwargs.get(f"ref_video_{i}")
            if video_frames is None:
                continue
            soundtrack = kwargs.get(f"ref_video_audio_{i}")
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 参考视频至少需要 5 帧（24fps 下约 0.2 秒）")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            # 空音频（静音占位等）不参与传递，等同未连接该端口，视频按无音轨处理
            if not _is_empty_audio(soundtrack, audio_vae):
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
                # 音轨获得独立的 <Audio j> 标记，在 <Video k> 之前输出
                ref_items.append({"type": "audio"})
            # Qwen 以 2fps 带时间戳观看视频
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({"type": "video", "data": qwen_frames,
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

        # 3. 独立参考音频：ref_audio_1..N
        for i in range(1, REF_AUDIO_PORTS + 1):
            audio = kwargs.get(f"ref_audio_{i}")
            # 空音频（静音占位等）不参与传递，等同未连接该端口
            if _is_empty_audio(audio, audio_vae):
                continue
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return (cond, latent)


NODE_CLASS_MAPPINGS = {
    "Yuan_MiniMaxH3Video": YuanMiniMaxH3Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_MiniMaxH3Video": "MiniMax-H3 视频生成",
}
