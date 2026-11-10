"""图像网格拼接节点：把批量图像按 宽幅×高幅 拼成网格图，并按「文件名前缀 + 编号」保存到 output 目录。"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths


class YuanImageGridComposite:
    """把批量图像按「宽幅×高幅」拼成网格图，并保存到输出目录。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "display_name": "编号",
                    "tooltip": "写入文件名 _00001_ 这一段的编号；同一编号重复运行会覆盖原文件。",
                }),
                "images": ("IMAGE", {
                    "display_name": "图像",
                    "tooltip": "要拼接的批量图像。",
                }),
                "grid_width": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display_name": "宽幅",
                    "tooltip": "网格列数。",
                }),
                "grid_height": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display_name": "高幅",
                    "tooltip": "网格行数。",
                }),
                "filename_prefix": ("STRING", {
                    "default": "Mabuya/ComfyUI",
                    "display_name": "文件名前缀",
                    "tooltip": "输出目录下的文件名前缀，可含子目录与 %变量%（如 %year%-%month%-%day%）。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    OUTPUT_TOOLTIPS = ("拼接后的网格图。",)
    FUNCTION = "composite"
    CATEGORY = "Yuan Tool/图像"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "把批量图像按 宽幅×高幅 拼成网格图：图少于格子时等间隔铺开、空缺格填黑；"
        "图多于格子时等间隔抽取且首尾必留。图像统一缩放到首张尺寸；宽幅高幅均为 1 时输出最后一张。"
        "网格图按「文件名前缀 + 编号」保存到 output 目录：编号 1 即写入 …_00001_.png，"
        "同一编号重复运行覆盖原文件，不再自动累加。"
    )

    def composite(self, images, grid_width, grid_height, filename_prefix, index):
        count = len(images)
        cells = grid_width * grid_height

        # 1x1 输出最后一张
        if cells == 1:
            canvas = images[count - 1]
        else:
            # 格子与图片的等间隔对应关系
            if count == 1:
                pairs = [(0, 0)]
            elif count >= cells:
                step = (count - 1) / (cells - 1)
                pairs = [(cell, int(cell * step + 0.5)) for cell in range(cells)]
            else:
                step = (cells - 1) / (count - 1)
                pairs = [(int(idx * step + 0.5), idx) for idx in range(count)]

            cell_h, cell_w, channels = images[0].shape
            canvas = torch.zeros(
                (grid_height * cell_h, grid_width * cell_w, channels),
                dtype=images[0].dtype,
            )

            for cell, idx in pairs:
                image = images[idx]
                # 尺寸与首张不一致时缩放对齐
                if image.shape[0] != cell_h or image.shape[1] != cell_w:
                    image = F.interpolate(
                        image.permute(2, 0, 1).unsqueeze(0),
                        size=(cell_h, cell_w),
                        mode="bilinear",
                        antialias=True,
                    ).squeeze(0).permute(1, 2, 0)

                row, col = divmod(cell, grid_width)
                canvas[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w, :] = image

        self._save(canvas, filename_prefix, index)
        return (canvas.unsqueeze(0),)

    @staticmethod
    def _save(image, filename_prefix, index):
        # 落盘目录与「保存图像」一致：前缀可含子目录与 %变量%，子目录不存在时自动创建
        full_folder, filename, _, _, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            image.shape[1],
            image.shape[0],
        )
        file = f"{filename}_{index:05}_.png"
        array = np.clip(image.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(array).save(os.path.join(full_folder, file), compress_level=4)


NODE_CLASS_MAPPINGS = {
    "YuanImageGridComposite": YuanImageGridComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanImageGridComposite": "图像网格拼接",
}
