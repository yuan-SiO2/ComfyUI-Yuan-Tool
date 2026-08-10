"""MiniMax H3 节点复刻（汉化版）：AV 潜空间生成与任务条件构建（t2va / fl2va / ref2va）。

H3 packed-DiT 通过条件接收：
- Qwen3-VL-32B 的隐藏状态，带逐 token 模态标记（来自 MiniMax CLIP）
- 关键帧 / 参考条件潜空间，每个采样步重新注入（不做去噪）

潜空间为 NestedTensor 对（视频 [B,24,T,H/16,W/16]、音频 [B,32,2,T40]）；
采样在扁平打包数据上进行，可使用任何常规采样器（模型内部处理音频流的偏移调度）。

本文件复刻自 ComfyUI 原生 comfy_extras/nodes_minimax_h3.py 的：
- MiniMaxH3ImageToVideo（MiniMax H3 Image to Video）
- MiniMaxH3ReferenceToVideo（MiniMax H3 Reference to Video）

已合并为单个节点 Yuan_MiniMaxH3Video，通过"模式"下拉框切换：
- 图生视频：提示词 + 可选首帧/尾帧关键帧
- 参考图生视频：提示词 + <Picture i> / <Video k> / <Audio j> 参考内容

端口与说明均已汉化，node_id 加 "Yuan_" 前缀避免与原生节点冲突。
"""

import math

import torch
import torchaudio

import nodes
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers

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


class YuanMiniMaxH3Video:
    """MiniMax-H3 视频生成（图生视频 / 参考图生视频）。

    图生视频：提示词（+ 可选首帧/尾帧关键帧）生成正向条件与音视频联合潜空间。
    参考图生视频：提示词 + <Picture i> / <Video k> / <Audio j> 参考条件。

    参考内容按固定顺序进入：先图像，再视频（每个视频的音轨 <Audio j> 标记
    紧跟在对应 <Video k> 之前），最后是独立音频。每种类型的编号从 1 开始，
    提示词中可用 <Picture i> / <Video k> / <Audio j> 引用。
    """

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("正向", "潜空间")
    OUTPUT_TOOLTIPS = ("正向条件（含关键帧/参考潜空间）", "视频+音频联合潜空间")
    FUNCTION = "execute"
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = "MiniMax-H3 视频生成：图生视频（首/尾帧关键帧）或参考图生视频（<Picture>/<Video>/<Audio> 参考）。"

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
            # 参考图像列表端口：可连接多张图像（batch），最多 REF_IMAGE_PORTS 张，超出自动切断
            "ref_images": image_port(
                "参考图像", f"参考图像列表（可连接多张图像，最多 {REF_IMAGE_PORTS} 张，超出自动切断）"),
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
                "mode": ([MODE_IMAGE_TO_VIDEO, MODE_REFERENCE], {"default": MODE_IMAGE_TO_VIDEO,
                    "display_name": "模式", "tooltip": "生成模式：图生视频（首/尾帧关键帧）或参考图生视频（<Picture>/<Video>/<Audio> 参考）"}),
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
                **kwargs):
        if mode == MODE_REFERENCE:
            return self._execute_reference(
                clip, vae, audio_vae, prompt, width, height, length,
                ref_image_size, ref_images, kwargs)
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
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
            })
        return (cond, latent)

    # ---- 参考图生视频（ref2va）----
    def _execute_reference(self, clip, vae, audio_vae, prompt, width, height, length,
                           ref_image_size, ref_images, kwargs):
        if audio_vae is None:
            raise ValueError("参考图生视频模式需要连接音频 VAE（audio_vae）端口")

        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items = []   # 供分词器按请求顺序呈现
        ref_blocks = []  # 供 DiT 负载，顺序一致
        # 参考图尺寸下拉框选项已汉化（见 INPUT_TYPES），这里还原为内部处理用的英文 key
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
            if soundtrack is not None:
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
            if audio is None:
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
