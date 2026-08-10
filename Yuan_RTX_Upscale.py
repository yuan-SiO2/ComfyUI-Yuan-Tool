"""Yuan Tool · RTX 视频放大 (H3) 节点

复刻自 "C:\\My Xiangmu\\Yuan Tool" 的 RTX 视频超分辨率 (AV) 合并节点，
合并 RTX 视频超分辨率 + AV Decode Split + PT H3 Concat AV Latent 三者为一体。

支持两种输入模式（二选一，互斥）：
  - images 模式：直接对图像进行 RTX 超分辨率
  - av_latent 模式：先用 video_vae 解码 H3 联合 AV latent 为图像帧 + 分离后的
    audio_latent，再对图像帧进行 RTX 超分辨率

统一流程（无论哪种模式）：
  1. 得到图像帧（来自输入或解码）
  2. RTX 超分辨率 → upscaled_images
  3. 用 video_vae 对 upscaled_images 重新编码 → new_video_latent
  4. 合并 new_video_latent + audio_latent → av_latent 输出
     （images 模式下 audio_latent 为 None，av_latent 输出也为 None）

内置缓存机制：当输入指纹（张量 shape/dtype/device/version/data_ptr + 参数）未变时，
直接返回上次的两个输出，跳过耗时的 RTX 超分辨率与 VAE 编解码。

节点分类: "Yuan Tool/放大"
依赖: nvidia-vfx（pip install nvidia-vfx，需 NVIDIA RTX 显卡）
"""

import torch

try:
    import nvvfx
except ImportError:
    nvvfx = None

import comfy.nested_tensor

UPSCALE_BY = "按倍数缩放"
UPSCALE_TARGET = "目标尺寸"

QUALITY_OPTIONS = ["低", "中", "高", "超高"]

MAX_PIXELS = 1024 * 1024 * 16


def _nested_av_parts(av_latent):
    """将 MiniMax H3 联合 AV latent 拆分为视频与音频两个张量。

    H3 的 AV latent 是一个 NestedTensor，包含两个槽位：
      - video: 5 维张量 [1, 24, T, H/16, W/16]
      - audio: 4 维张量 [1, 32, 2, audio_t]
    """
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise ValueError("Expected a MiniMax H3 joint AV LATENT")
    samples = av_latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise ValueError("Expected a nested MiniMax H3 joint video/audio latent")
    parts = tuple(samples.unbind())
    if len(parts) != 2:
        raise ValueError(f"Expected exactly two AV latent parts, got {len(parts)}")
    video, audio = parts
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            "Unexpected MiniMax H3 AV latent layout: "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("MiniMax H3 currently supports batch size 1 only")
    return video, audio


def _decode_av_latent(av_latent, video_vae):
    """解码 latent（仅视频 VAE），兼容两种输入。

    支持的输入形式：
      1. H3 联合 AV latent：samples 为 NestedTensor，含 video + audio 两部分
         → 解码 video 部分为图像帧，并返回分离后的 video_latent / audio_latent
      2. 纯视频 latent：samples 为普通 torch.Tensor
         → 直接解码为图像帧，audio_latent 返回 None
    """
    samples = av_latent["samples"]

    # 判断是否为 H3 联合 AV latent（NestedTensor）
    if getattr(samples, "is_nested", False):
        # 联合 AV latent：拆分出 video + audio
        video, audio = _nested_av_parts(av_latent)
        images = video_vae.decode(video)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        video_latent = {key: value for key, value in av_latent.items() if key not in {"samples", "noise_mask"}}
        audio_latent = video_latent.copy()
        video_latent["samples"] = video
        audio_latent["samples"] = audio
        masks = av_latent.get("noise_mask")
        if getattr(masks, "is_nested", False):
            video_mask, audio_mask = masks.unbind()
            video_latent["noise_mask"] = video_mask
            audio_latent["noise_mask"] = audio_mask
        return images, video_latent, audio_latent

    # 纯视频 latent：直接解码，无 audio_latent
    images = video_vae.decode(samples)
    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])
    return images, None, None


class YuanRTXVideoUpscaleH3:
    """RTX 视频放大 (H3) 节点

    合并 RTX 视频超分辨率 + AV Decode Split + PT H3 Concat AV Latent 三者为一体。
    """

    # 类级缓存：{cache_key: (upscaled, av_latent_output)}
    _cache = {}
    _MAX_CACHE_ENTRIES = 8  # 防止内存无限增长

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resize_type": (
                    [UPSCALE_BY, UPSCALE_TARGET],
                    {
                        "default": UPSCALE_BY,
                        "display_name": "缩放方式",
                        "tooltip": "选择按倍数缩放或缩放到精确的目标尺寸。",
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.01,
                        "display_name": "缩放倍数",
                        "tooltip": "缩放倍数（例如 2.0 表示放大两倍），仅在「按倍数缩放」模式下生效。",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1920,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "display_name": "目标宽度",
                        "tooltip": "目标宽度（像素），仅在「目标尺寸」模式下生效。",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1080,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "display_name": "目标高度",
                        "tooltip": "目标高度（像素），仅在「目标尺寸」模式下生效。",
                    },
                ),
                "quality": (
                    QUALITY_OPTIONS,
                    {
                        "default": "超高",
                        "display_name": "质量",
                        "tooltip": "超分辨率质量等级。",
                    },
                ),
            },
            "optional": {
                "images": ("IMAGE", {"display_name": "图像", "tooltip": "图像输入（与 av_latent 二选一）。"}),
                "av_latent": ("LATENT", {"display_name": "AV Latent", "tooltip": "MiniMax H3 联合 AV latent 输入（与 images 二选一，需同时连接 video_vae）。"}),
                "video_vae": ("VAE", {"display_name": "视频 VAE", "tooltip": "视频 VAE（仅在 av_latent 模式下必填，用于解码和重新编码）。"}),
            },
        }

    CATEGORY = "Yuan Tool/放大"
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("放大图像", "AV Latent")
    OUTPUT_TOOLTIPS = (
        "超分辨率后的图像。",
        "合并后的 H3 联合 AV latent（仅 av_latent 模式有效，images 模式为空）。",
    )
    FUNCTION = "upscale"
    DESCRIPTION = (
        "RTX 视频放大 (H3)：合并 RTX 视频超分辨率 + AV Decode Split + PT H3 Concat AV Latent 三者为一体。\n"
        "支持图像直连放大，或接入 MiniMax H3 联合 AV latent（自动解码放大后重新合并音频）。\n"
        "依赖：pip install nvidia-vfx（需 NVIDIA RTX 显卡）。"
    )

    def upscale(self, resize_type, scale, width, height, quality,
                images=None, av_latent=None, video_vae=None):
        if nvvfx is None:
            raise ImportError("未安装 nvidia-vfx，请执行 pip install nvidia-vfx 后重试。")

        # 互斥校验：images 与 av_latent 只能接一个
        has_images = images is not None
        has_av = av_latent is not None
        if has_images and has_av:
            raise ValueError("images 和 av_latent 只能连接其中一个")
        if not has_images and not has_av:
            raise ValueError("必须连接 images 或 av_latent 中的一个")

        # 组装 resize 参数
        resize_params = {
            "resize_type": resize_type,
            "scale": scale,
            "width": width,
            "height": height,
        }

        # 计算缓存 key，命中则直接返回上次的两个输出
        cache_key = self._build_cache_key(images, av_latent, video_vae, resize_params, quality)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # av_latent 模式：先解码得到 frames + 分离后的 audio_latent
        audio_latent = None
        if has_av:
            if video_vae is None:
                raise ValueError("使用 av_latent 输入时必须连接 video_vae")
            images, _video_latent, audio_latent = _decode_av_latent(av_latent, video_vae)

        # 统一走 RTX 超分辨率
        upscaled = self._run_super_resolution(images, resize_params, quality)

        # 无 video_vae（images 模式且未接 vae）：只输出 upscaled_images
        if video_vae is None:
            result = (upscaled, None)
            self._store_cache(cache_key, result)
            return result

        # 用 video_vae 对超分辨率后的图像重新编码 → new_video_latent
        new_video_tensor = video_vae.encode(upscaled)

        # 无 audio_latent（纯视频 latent 输入 或 images 模式接了 vae）：输出普通视频 latent
        if audio_latent is None:
            result = (upscaled, {"samples": new_video_tensor})
            self._store_cache(cache_key, result)
            return result

        # 联合 AV latent 模式：合并 new_video_latent + audio_latent → av_latent
        audio_tensor = audio_latent["samples"]
        if not isinstance(new_video_tensor, torch.Tensor) or not isinstance(audio_tensor, torch.Tensor):
            raise ValueError("编码后的视频 latent 和音频 latent 必须是 torch.Tensor")
        if new_video_tensor.ndim != 5:
            raise ValueError(f"Video latent expects 5D tensor, got {new_video_tensor.ndim}D.")
        if audio_tensor.ndim != 4:
            raise ValueError(f"Audio latent expects 4D tensor, got {audio_tensor.ndim}D.")
        nested_latent = comfy.nested_tensor.NestedTensor([new_video_tensor, audio_tensor])
        merged_av_latent = {"samples": nested_latent}

        result = (upscaled, merged_av_latent)
        self._store_cache(cache_key, result)
        return result

    @staticmethod
    def _tensor_fingerprint(t):
        """张量指纹：shape + dtype + device + version + data_ptr。

        version 检测原地修改，data_ptr 检测内存地址变化。
        推理张量不跟踪 version，此时用 None 占位（仍可由 data_ptr 检测变化）。
        """
        try:
            version = t._version
        except RuntimeError:
            version = None
        return (
            tuple(t.shape),
            str(t.dtype),
            str(t.device),
            version,
            t.data_ptr(),
        )

    @classmethod
    def _build_cache_key(cls, images, av_latent, video_vae, resize_params, quality):
        """根据所有输入构建可哈希的缓存 key。"""
        # images 指纹
        images_fp = cls._tensor_fingerprint(images) if images is not None else None

        # av_latent 指纹（区分 NestedTensor 与普通 Tensor）
        av_latent_fp = None
        if av_latent is not None:
            samples = av_latent["samples"]
            if getattr(samples, "is_nested", False):
                # NestedTensor：取每个子张量的指纹
                parts = tuple(samples.unbind())
                av_latent_fp = ("nested", tuple(cls._tensor_fingerprint(p) for p in parts))
            else:
                av_latent_fp = ("plain", cls._tensor_fingerprint(samples))

        # video_vae 用 id 区分（同一对象 id 不变）
        vae_id = id(video_vae) if video_vae is not None else None

        # resize 参数转可哈希元组
        rt = resize_params["resize_type"]
        if rt == UPSCALE_BY:
            params_fp = ("scale", float(resize_params["scale"]))
        else:
            params_fp = ("target", int(resize_params["width"]), int(resize_params["height"]))

        return (images_fp, av_latent_fp, vae_id, params_fp, quality)

    @classmethod
    def _store_cache(cls, key, value):
        """存入缓存，超出上限时按 FIFO 清理最旧条目。"""
        if len(cls._cache) >= cls._MAX_CACHE_ENTRIES:
            # 弹出最早插入的 key（dict 保持插入顺序）
            oldest = next(iter(cls._cache))
            del cls._cache[oldest]
        cls._cache[key] = value

    @staticmethod
    def _run_super_resolution(images, resize_params, quality):
        b, h, w, c = images.shape

        selected_type = resize_params["resize_type"]
        if selected_type == UPSCALE_BY:
            scale = resize_params["scale"]
            output_width = int(w * scale)
            output_height = int(h * scale)
        elif selected_type == UPSCALE_TARGET:
            output_width = resize_params["width"]
            output_height = resize_params["height"]
        else:
            raise ValueError(f"不支持的缩放类型: {selected_type}")

        output_width = max(8, round(output_width / 8) * 8)
        output_height = max(8, round(output_height / 8) * 8)

        out_pixels = output_width * output_height
        batch_size = max(1, MAX_PIXELS // out_pixels)

        quality_mapping = {
            "低": nvvfx.effects.QualityLevel.LOW,
            "中": nvvfx.effects.QualityLevel.MEDIUM,
            "高": nvvfx.effects.QualityLevel.HIGH,
            "超高": nvvfx.effects.QualityLevel.ULTRA,
        }
        selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.HIGH)

        with nvvfx.VideoSuperRes(selected_quality) as sr:
            sr.output_width = output_width
            sr.output_height = output_height
            sr.load()

            out_tensor = torch.empty(
                (images.shape[0], output_height, output_width, c),
                device=images.device,
                dtype=images.dtype,
            )
            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i + batch_size]

                batch_cuda = batch.cuda().permute(0, 3, 1, 2).float().contiguous()

                for j in range(batch_cuda.shape[0]):
                    input_frame = batch_cuda[j]
                    dlpack_out = sr.run(input_frame).image
                    out_tensor[i + j: i + j + 1] = torch.from_dlpack(dlpack_out).movedim(0, -1).unsqueeze(0)

        return out_tensor


NODE_CLASS_MAPPINGS = {
    "Yuan_RTXVideoUpscaleH3": YuanRTXVideoUpscaleH3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_RTXVideoUpscaleH3": "RTX 视频放大 (H3)",
}
