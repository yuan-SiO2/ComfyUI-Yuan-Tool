"""色彩匹配 节点：基于 color-matcher 库实现跨图像色彩迁移（校正/调色/光线色温统一）。"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

_COLOR_MATCH_METHODS = ['mkl', 'hm', 'reinhard', 'mvgd', 'hm-mvgd-hm', 'hm-mkl-hm']


class YuanColorMatch:
    """将参考图像（image_ref）的色彩分布迁移到目标图像（image_target），支持批量处理与多线程加速，strength 控制混合强度。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE", {"display_name": "参考图像", "tooltip": "提供色彩分布的参考图像（颜色来源）。"}),
                "image_target": ("IMAGE", {"display_name": "目标图像", "tooltip": "要被改色的目标图像或视频帧 batch（色彩去往）。"}),
                "method": (_COLOR_MATCH_METHODS, {"default": 'mkl', "display_name": "匹配方法", "tooltip": (
                    "匹配算法：mkl=线性化（推荐）；hm=直方图匹配；"
                    "reinhard=均值+方差；mvgd=多元高斯分布；"
                    "hm-mvgd-hm、hm-mkl-hm=复合。依赖需安装 color-matcher"
                )}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display_name": "强度", "tooltip": "匹配强度（0 完全保留原图，1 完全匹配，>1 过度迁移）"}),
                "multithread": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "关闭", "display_name": "多线程", "tooltip": "批量大于 1 时启用多线程加速"}),
            }
        }

    CATEGORY = "Yuan Tool/图像"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("匹配后图像",)
    OUTPUT_TOOLTIPS = ("色彩匹配结果图像（与输入 batch 尺寸一致，0..1 float）。",)
    FUNCTION = "colormatch"
    DESCRIPTION = (
        "色彩匹配：将参考图的色彩分布迁移到目标图，支持 mkl/hm/reinhard/mvgd 等方法，strength 控制混合强度。\n"
        "依赖：pip install color-matcher。"
    )

    def colormatch(self, image_ref, image_target, method, strength=1.0, multithread=True):
        # 强度为 0 时直接返回原图，跳过无意义的处理
        if strength == 0:
            return (image_target,)

        try:
            from color_matcher import ColorMatcher
        except ImportError as e:
            raise ImportError(
                "无法导入 color-matcher，请先安装依赖。手动安装：pip install color-matcher"
            ) from e

        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)
        ref_batch = image_ref.size(0)

        def process(i):
            """对第 i 帧执行色彩迁移；参考图不足时复用最后一帧。"""
            cm = ColorMatcher()
            target_i = image_target[0] if batch_size == 1 else image_target[i]
            ref_i = image_ref[0] if ref_batch == 1 else image_ref[min(i, ref_batch - 1)]
            target_np = target_i.numpy()
            ref_np = ref_i.numpy()
            try:
                # src 是待改色图，ref 是提供色彩分布的参考图
                image_result = cm.transfer(src=target_np, ref=ref_np, method=method)
                # 强度混合：在原目标图与匹配结果之间线性插值
                if strength != 1:
                    image_result = target_np + strength * (image_result - target_np)
                return torch.from_numpy(np.ascontiguousarray(image_result))
            except Exception:
                # 失败时回退到原图，不打印日志
                return torch.from_numpy(np.ascontiguousarray(target_np))

        # 多线程仅在批量大于 1 时启用
        if multithread and batch_size > 1:
            max_threads = min(os.cpu_count() or 1, batch_size)
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                out = list(executor.map(process, range(batch_size)))
        else:
            out = [process(i) for i in range(batch_size)]

        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "YuanColorMatch": YuanColorMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanColorMatch": "色彩匹配",
}
