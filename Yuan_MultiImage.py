"""Yuan Tool · 加载批量图像 节点（多滑轨版）

复刻自 WhatDreamsCost-ComfyUI 的 MultiImageLoader，支持最多 20 个独立滑轨，
每滑轨单独加载批量图像并独立输出；使用 lanczos 插值将宽高向上取整到 16 的倍数。
前端通过 tracks_data（JSON 字符串）维护滑轨数据。

节点分类: "Yuan Tool/图像"
"""

import json
import os

import torch
import numpy as np
from PIL import Image, ImageOps

import folder_paths
import comfy.utils


def _default_track_name(index):
    """生成默认滑轨名称：0→批量图像A, 1→批量图像B, 25→批量图像Z, 26→批量图像AA..."""
    name = ""
    n = index
    do = True
    while do or n >= 0:
        name = chr(65 + (n % 26)) + name
        n = n // 26 - 1
        do = False
    return "批量图像" + name


class YuanMultiImage:
    """加载批量图像节点（多滑轨版）：tracks_data 维护多个滑轨，每滑轨独立加载并输出一路 IMAGE batch。"""

    MAX_TRACKS = 20

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tracks_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "display_name": "滑轨数据",
                    "tooltip": "JSON 格式滑轨数据，由前端自动维护。格式：[{\"name\":\"批量图像A\",\"paths\":[\"img1.png\"]}]",
                }),
            },
        }

    # 20 路输出，每路对应一个滑轨
    RETURN_TYPES = ("IMAGE",) * MAX_TRACKS
    RETURN_NAMES = tuple(_default_track_name(i) for i in range(MAX_TRACKS))
    OUTPUT_TOOLTIPS = tuple(f"滑轨 {i+1} 的合并 batch。" for i in range(MAX_TRACKS))
    FUNCTION = "load_images"
    CATEGORY = "Yuan Tool/图像"
    DESCRIPTION = (
        "加载批量图像（多滑轨版）：支持最多 20 个独立滑轨，每个滑轨单独加载批量图像。"
        "使用 lanczos 插值将宽高向上取整到 16 的倍数。每路输出对应一个滑轨的 batch。"
        "复刻自 WhatDreamsCost-ComfyUI 的 Multi Image Loader。"
    )

    def resize_image(self, image, multiple_of=16):
        """将图像宽高向上取整到 multiple_of 的倍数，使用 lanczos 插值缩放。"""
        _, oh, ow, _ = image.shape

        if multiple_of > 1:
            new_w = ((ow + multiple_of - 1) // multiple_of) * multiple_of
            new_h = ((oh + multiple_of - 1) // multiple_of) * multiple_of
        else:
            new_w = ow
            new_h = oh

        if new_w == ow and new_h == oh:
            return torch.clamp(image, 0, 1)

        outputs = image.permute(0, 3, 1, 2)
        outputs = comfy.utils.lanczos(outputs, new_w, new_h)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = torch.clamp(outputs, 0, 1)
        return outputs

    def _load_track_images(self, paths):
        """加载单个滑轨的所有图像，返回合并后的 batch 张量。"""
        track_images = []

        for path in paths:
            path = path.strip() if isinstance(path, str) else ""
            if not path:
                continue
            try:
                full_path = self._resolve_image_path(path)
                if not os.path.exists(full_path):
                    continue

                image = Image.open(full_path)
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")

                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                image_tensor = self.resize_image(image_tensor, 16)
                track_images.append(image_tensor)
            except Exception:
                pass

        if len(track_images) == 0:
            return torch.zeros((1, 64, 64, 3))

        # 检查所有图像尺寸是否一致
        first_shape = track_images[0].shape
        if all(t.shape == first_shape for t in track_images):
            return torch.cat(track_images, dim=0)

        # 尺寸不一致：以第一张尺寸为基准，对后续图像做中心裁切（防止失真）
        _, fh, fw, _ = first_shape
        normalized = [track_images[0]]
        for t in track_images[1:]:
            if t.shape != first_shape:
                t = self._center_crop(t, fw, fh)
            normalized.append(t)
        return torch.cat(normalized, dim=0)

    def _center_crop(self, image, target_w, target_h):
        """中心裁切到指定尺寸。图像小于目标尺寸时先按比例放大再裁切。"""
        _, src_h, src_w, _ = image.shape

        # 尺寸完全一致直接返回
        if src_w == target_w and src_h == target_h:
            return image

        # 若图像小于目标尺寸，先等比放大到能覆盖目标（短边对齐）
        if src_w < target_w or src_h < target_h:
            scale = max(target_w / src_w, target_h / src_h)
            new_w = max(target_w, round(src_w * scale))
            new_h = max(target_h, round(src_h * scale))
            outputs = image.permute(0, 3, 1, 2)
            outputs = comfy.utils.lanczos(outputs, new_w, new_h)
            outputs = outputs.permute(0, 2, 3, 1)
            image = torch.clamp(outputs, 0, 1)
            _, src_h, src_w, _ = image.shape

        # 中心裁切
        x1 = (src_w - target_w) // 2
        y1 = (src_h - target_h) // 2
        return image[:, y1:y1 + target_h, x1:x1 + target_w, :]

    def load_images(self, tracks_data):
        """解析 tracks_data JSON，为每个滑轨加载图像并返回 20 路 IMAGE。"""
        try:
            tracks = json.loads(tracks_data) if tracks_data and tracks_data.strip() else []
        except (json.JSONDecodeError, TypeError):
            tracks = []

        results = []
        for i in range(self.MAX_TRACKS):
            if i < len(tracks):
                track = tracks[i] if isinstance(tracks[i], dict) else {}
                paths = track.get("paths", [])
                if isinstance(paths, str):
                    paths = [p.strip() for p in paths.split("\n") if p.strip()]
                results.append(self._load_track_images(paths))
            else:
                results.append(torch.zeros((1, 64, 64, 3)))

        return tuple(results)

    @staticmethod
    def _resolve_image_path(path):
        """解析图像路径，支持 output:/input:/temp: 前缀及绝对/相对路径。"""
        if not path:
            return path
        if path.startswith("output:"):
            return os.path.join(folder_paths.get_output_directory(), path[len("output:"):])
        if path.startswith("input:"):
            return os.path.join(folder_paths.get_input_directory(), path[len("input:"):])
        if path.startswith("temp:"):
            return os.path.join(folder_paths.get_temp_directory(), path[len("temp:"):])
        if os.path.isabs(path) and os.path.exists(path):
            return path
        return os.path.join(folder_paths.get_input_directory(), path)


NODE_CLASS_MAPPINGS = {
    "YuanMultiImage": YuanMultiImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanMultiImage": "加载批量图像",
}
