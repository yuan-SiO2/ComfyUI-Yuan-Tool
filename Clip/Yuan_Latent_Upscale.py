"""缩放Latent（比例）节点：复刻原生 LatentUpscaleBy，额外兼容 H3 5D 与联合 AV latent。"""

import comfy.utils
import comfy.nested_tensor

UPSCALE_BY = "按倍数缩放"
UPSCALE_TARGET = "目标尺寸"


class YuanLatentUpscaleBy:
    """缩放Latent（比例）：按倍数或目标尺寸缩放空间分辨率，兼容常规 4D / H3 5D / 联合 AV latent。"""

    SEARCH_ALIASES = ["enlarge latent", "resize latent", "scale latent"]

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {
                    "display_name": "潜空间",
                    "tooltip": "待缩放 latent：常规 (B,C,H,W)、H3 视频 (B,C,T,H,W) 或 H3 联合 AV（仅缩放视频流）。",
                }),
                "upscale_method": (cls.upscale_methods, {
                    "display_name": "缩放方法",
                    "tooltip": "latent 插值方法（与原生节点一致）。",
                }),
                "resize_type": ([UPSCALE_BY, UPSCALE_TARGET], {
                    "default": UPSCALE_BY,
                    "display_name": "缩放方式",
                    "tooltip": "按倍数缩放（latent 尺寸 × 倍数）或缩放到目标尺寸（像素）。",
                }),
                "scale": ("FLOAT", {
                    "default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01,
                    "display_name": "缩放倍数",
                    "tooltip": "缩放倍数，作用于 latent 的 H×W（支持放大与缩小）。仅在「按倍数缩放」模式下生效。",
                }),
                "width": ("INT", {
                    "default": 1920, "min": 64, "max": 8192, "step": 8,
                    "display_name": "目标宽度",
                    "tooltip": "目标宽度（像素）：H3 latent 按 16 对齐、常规 latent 按 8 对齐换算。仅在「目标尺寸」模式下生效。",
                }),
                "height": ("INT", {
                    "default": 1080, "min": 64, "max": 8192, "step": 8,
                    "display_name": "目标高度",
                    "tooltip": "目标高度（像素）：H3 (5D) 对齐到 16 倍数再 ÷16；常规 (4D) 对齐到 8 倍数再 ÷8。仅「目标尺寸」模式生效。",
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    OUTPUT_TOOLTIPS = ("缩放后的 latent，键与其他字段保持不变。",)
    FUNCTION = "upscale"
    CATEGORY = "Yuan Tool/放大"
    DESCRIPTION = (
        "缩放Latent（比例）：复刻原生节点并扩展，兼容常规 4D、H3 纯视频 latent（5D，仅缩放空间维度）与 H3 联合 AV latent"
        "（仅缩放视频流、音频原样保留），可直接衔接「H3 放大」「RTX 视频放大 (H3)」。\n"
        "缩放方式：「按倍数缩放」（×倍，支持放大缩小）或「目标尺寸」（像素，H3 按 16 对齐、常规按 8 对齐换算）。"
    )

    def upscale(self, samples, upscale_method, resize_type, scale, width, height):
        s = samples.copy()
        tensor = s["samples"]

        # H3 联合 AV latent（NestedTensor）：仅缩放视频流，音频流原样保留
        if getattr(tensor, "is_nested", False):
            streams = list(tensor.unbind())
            streams[0] = self._upscale_tensor(streams[0], upscale_method, resize_type, scale, width, height)
            s["samples"] = comfy.nested_tensor.NestedTensor(streams)
            return (s,)

        s["samples"] = self._upscale_tensor(tensor, upscale_method, resize_type, scale, width, height)
        return (s,)

    @staticmethod
    def _upscale_tensor(t, upscale_method, resize_type, scale, width, height):
        """缩放单个 latent 张量（4D 常规 / 5D H3 视频），仅作用空间维度 H×W。"""
        if t.ndim not in (4, 5):
            raise ValueError(f"不支持的 latent 形状: {tuple(t.shape)}（期望 4D (B,C,H,W) 或 5D (B,C,T,H,W)）")

        cur_h, cur_w = t.shape[-2], t.shape[-1]
        # 空间压缩比：H3 视频 latent (5D) 为 16，常规 latent (4D) 为 8
        ratio = 16 if t.ndim == 5 else 8

        if resize_type == UPSCALE_BY:
            new_h = max(1, round(cur_h * scale))
            new_w = max(1, round(cur_w * scale))
        elif resize_type == UPSCALE_TARGET:
            # 目标尺寸（像素）对齐压缩比后换算为 latent 尺寸
            new_w = max(1, max(ratio, round(int(width) / ratio) * ratio) // ratio)
            new_h = max(1, max(ratio, round(int(height) / ratio) * ratio) // ratio)
        else:
            raise ValueError(f"不支持的缩放类型: {resize_type}")

        if new_h == cur_h and new_w == cur_w:
            return t

        # common_upscale 原生支持 4D/5D（仅缩放最后两个维度），时间维度保持不变
        return comfy.utils.common_upscale(t, new_w, new_h, upscale_method, "disabled")


NODE_CLASS_MAPPINGS = {
    "Yuan_LatentUpscaleBy": YuanLatentUpscaleBy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_LatentUpscaleBy": "缩放Latent（比例）",
}
