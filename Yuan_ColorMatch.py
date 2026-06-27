"""Yuan Tool · 色彩匹配 节点

复刻自 ComfyUI-KJNodes 的 ColorMatch 节点，基于 color-matcher 库实现
跨图像的色彩迁移，可用于自动色彩校正、画面调色以及光线/色温统一。

节点分类: "Yuan Tool/图像"
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

log = logging.getLogger(__name__)


# 支持的色彩匹配方法及其中文说明（用于节点描述）
# 原始方法名保留为英文，因为 color-matcher 库按名称调用
_COLOR_MATCH_METHODS = ['mkl', 'hm', 'reinhard', 'mvgd', 'hm-mvgd-hm', 'hm-mkl-hm']


class YuanColorMatch:
    """色彩匹配节点

    将参考图像（image_ref）的色彩分布迁移到目标图像（image_target）上。
    支持批量处理与多线程加速；通过 strength 控制混合强度。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (_COLOR_MATCH_METHODS, {"default": 'mkl'}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "multithread": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "Yuan Tool/图像"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = (
        "色彩匹配：将参考图的色彩分布迁移到目标图。\n"
        "可用方法：\n"
        "  mkl         - Monge-Kantorovich 线性化（MKL）\n"
        "  hm          - 直方图匹配（HM）\n"
        "  reinhard    - Reinhard 等人的方法\n"
        "  mvgd        - 多元高斯分布迁移（MVGD）\n"
        "  hm-mvgd-hm  - HM-MVGD-HM 复合方法\n"
        "  hm-mkl-hm   - HM-MKL-HM 复合方法\n"
        "依赖：pip install color-matcher\n"
        "参考：https://github.com/hahnec/color-matcher/"
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

        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        def process(i):
            """对第 i 帧执行色彩迁移。"""
            cm = ColorMatcher()
            # 批次为 1 时直接使用整体数组，否则按帧索引
            image_target_np_i = images_target_np if batch_size == 1 else images_target[i].numpy()
            image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
            try:
                # 注意：src 是目标图（要被改色的图），ref 是参考图（提供色彩分布）
                image_result = cm.transfer(src=image_target_np_i, ref=image_ref_np_i, method=method)
                # 强度混合：在原目标图与匹配结果之间线性插值
                if strength != 1:
                    image_result = image_target_np_i + strength * (image_result - image_target_np_i)
                return torch.from_numpy(image_result)
            except Exception as e:
                log.warning("[YuanColorMatch] 第 %d 帧处理失败: %s", i, e)
                return torch.from_numpy(image_target_np_i)  # 失败时回退到原图

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
