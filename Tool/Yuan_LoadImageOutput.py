"""加载输出图节点：按「文件名前缀 + 编号」从 output 目录读取图像。

文件名规则同「保存图像」：{前缀}_{编号:05}_.png。
"""

import hashlib
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import node_helpers


class YuanLoadImageOutput:
    """按「文件名前缀 + 编号」从输出目录读取图像，输出图像与遮罩。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {
                    "default": "Mabuya/ComfyUI",
                    "display_name": "文件名前缀",
                    "tooltip": "输出目录下的文件名前缀，可含子目录。",
                }),
                "index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "display_name": "编号",
                    "tooltip": "文件名中 _00001_ 这一段的编号。",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "遮罩")
    OUTPUT_TOOLTIPS = ("读取到的图像。", "图像 alpha 通道的反相遮罩；无 alpha 时为全 0。")
    FUNCTION = "load_image"
    CATEGORY = "Yuan Tool/图像"
    DESCRIPTION = (
        "按「文件名前缀 + 编号」从 output 目录读取图像：编号 1 即读取 {前缀}_00001_.png。"
        "文件不存在时输出一张空的 64×64 图像，不中断工作流。"
    )

    @staticmethod
    def _resolve(filename_prefix, index):
        name = f"{filename_prefix}_{index:05}_.png"
        try:
            return folder_paths.get_annotated_filepath(name, folder_paths.get_output_directory())
        except ValueError:  # 路径越界（如含 ..）按文件不存在处理
            return None

    def load_image(self, filename_prefix, index):
        image_path = self._resolve(filename_prefix, index)

        # 文件不存在时输出空图，不中断工作流
        if not image_path or not os.path.exists(image_path):
            return (
                torch.zeros((1, 64, 64, 3), dtype=torch.float32),
                torch.zeros((1, 64, 64), dtype=torch.float32),
            )

        pil_image = node_helpers.pillow(Image.open, image_path)
        images = []
        masks = []
        size = None

        for frame in ImageSequence.Iterator(pil_image):
            frame = node_helpers.pillow(ImageOps.exif_transpose, frame)
            rgb = frame.convert("RGB")

            # 以第一帧尺寸为准，其余不同尺寸的帧跳过
            if size is None:
                size = rgb.size
            if rgb.size != size:
                continue

            array = np.array(rgb).astype(np.float32) / 255.0
            images.append(torch.from_numpy(array)[None])

            if "A" in frame.getbands():
                alpha = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                masks.append((1.0 - torch.from_numpy(alpha)).unsqueeze(0))
            else:
                masks.append(torch.zeros((1, 64, 64), dtype=torch.float32))

        return (torch.cat(images, dim=0), torch.cat(masks, dim=0))

    @classmethod
    def IS_CHANGED(cls, filename_prefix, index):
        path = cls._resolve(filename_prefix, index)
        if not path or not os.path.exists(path):
            return ""

        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()


NODE_CLASS_MAPPINGS = {
    "YuanLoadImageOutput": YuanLoadImageOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanLoadImageOutput": "加载输出图",
}
