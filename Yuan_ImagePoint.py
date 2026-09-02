"""图像点处理 节点：在内嵌画布上标注正面点/负面点/边界框（复刻自 YuanEditor，多帧可逐帧标注）。"""

import json
import random
import hashlib
import os

import numpy as np
import torch
from PIL import Image

import folder_paths


def tensor_to_pil(images: torch.Tensor):
    """将 [B,H,W,C] 的 0~1 float 张量转换为 PIL 图像列表。"""
    images = images.cpu()

    if images.dim() == 3:
        # Single image [H, W, C]
        images = images.unsqueeze(0)
    elif images.dim() == 2:
        # Grayscale [H, W]
        images = images.unsqueeze(0).unsqueeze(-1)

    if images.max() <= 1.0:
        images = images * 255.0
    images = images.clamp(0, 255).byte()

    pil_images = []
    for img in images:
        img_np = img.numpy()
        if img_np.shape[-1] == 1:
            pil_images.append(Image.fromarray(img_np.squeeze(-1), mode='L'))
        elif img_np.shape[-1] == 3:
            pil_images.append(Image.fromarray(img_np, mode='RGB'))
        elif img_np.shape[-1] == 4:
            pil_images.append(Image.fromarray(img_np, mode='RGBA'))
        else:
            raise ValueError(f"Unsupported channel count: {img_np.shape[-1]}")
    return pil_images


def pil_to_tensor(pil_images):
    """将 PIL 图像（单个或列表）转换为 [B,H,W,C] 的 0~1 float 张量。"""
    if isinstance(pil_images, Image.Image):
        pil_images = [pil_images]

    tensor_list = []
    for pil_img in pil_images:
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        tensor_list.append(torch.from_numpy(img_np))
    return torch.stack(tensor_list)


def _save_preview_images(images, filename_prefix):
    """将预览图像批次保存到 ComfyUI 临时目录，返回 [{filename, subfolder, type}] 条目列表。"""
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_temp_directory(), images[0].shape[1], images[0].shape[0]
    )
    results = []
    for batch_number, image_tensor in enumerate(images):
        img = tensor_to_pil(image_tensor.unsqueeze(0))[0]
        file = f"{filename.replace('%batch_num%', str(batch_number))}_{counter:05}_.png"
        img.save(os.path.join(full_output_folder, file), compress_level=4)
        results.append({"filename": file, "subfolder": subfolder, "type": "temp"})
        counter += 1
    return results


class Yuan_ImagePoint:
    """图像点处理 - 在内嵌画布上为图像标注正面点/负面点/边界框的帧编辑器节点。"""

    state = {
        "last_images_hash": None,
        "cached_preview": None,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"display_name": "图像", "tooltip": "待标注的图像，可传入多帧（视频帧序列）"}),
                "info": ("STRING", {"default": "", "display_name": "标注信息", "tooltip": "编辑器内部状态（正面点/负面点/边界框/帧索引的 JSON），由前端标注后写入"}),
                "preview_rescale": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05, "display_name": "预览缩放", "tooltip": "预览图像缩放比例（坐标会换算回原始尺寸）"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "BBOX", "INT")
    RETURN_NAMES = ("正面点坐标", "负面点坐标", "边界框", "帧索引")
    OUTPUT_TOOLTIPS = (
        "正面点坐标 JSON 字符串（[{x, y}, ...]）",
        "负面点坐标 JSON 字符串（[{x, y}, ...]）",
        "边界框列表（每项为 [x1, y1, x2, y2]）",
        "当前标注的帧索引（从 0 开始）",
    )
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Yuan Tool/图像"

    DESCRIPTION = (
        "图像点处理（复刻自 YuanEditor / Frames Editor）\n"
        "- 在内嵌画布上为图像标注正面点、负面点与边界框\n"
        "- 点标注模式：左键添加正面点，右键添加负面点\n"
        "- 框标注模式：拖拽添加边界框\n"
        "- 多帧图像可在底部滑块逐帧切换标注\n"
        "- 输出坐标 JSON 与边界框，供下游节点使用"
    )

    def execute(self, images, info, preview_rescale=1.0):
        positive_coords = None
        negative_coords = None
        bboxes = None
        frame_index = 0

        # 计算换算回原始尺寸的比例因子
        needs_scaling = preview_rescale > 0 and preview_rescale < 1.0
        scale_factor = 1.0 / preview_rescale if needs_scaling else 1.0

        if info != '':
            try:
                info = json.loads(info)
            except json.JSONDecodeError:
                info = None

            if info is not None:
                positive_coords = info.get("positive_coords", None)
                negative_coords = info.get("negative_coords", None)
                box = info.get("bbox", None)
                frame_index = info.get("frame_index", 0)

                # 将坐标换算回原始尺寸
                if needs_scaling:
                    if positive_coords is not None:
                        positive_coords = [{"x": coord["x"] * scale_factor, "y": coord["y"] * scale_factor} for coord in positive_coords]

                    if negative_coords is not None:
                        negative_coords = [{"x": coord["x"] * scale_factor, "y": coord["y"] * scale_factor} for coord in negative_coords]

                # 处理边界框
                bboxes = []
                if box is not None and len(box) > 0:
                    for i in box:
                        if needs_scaling:
                            x = i['x'] * scale_factor
                            y = i['y'] * scale_factor
                            w = i['w'] * scale_factor
                            h = i['h'] * scale_factor
                        else:
                            x = i['x']
                            y = i['y']
                            w = i['w']
                            h = i['h']
                        bboxes.append([x, y, x + w, y + h])

                # 转成 JSON 字符串
                if positive_coords is not None:
                    positive_coords = json.dumps(positive_coords, ensure_ascii=False)
                if negative_coords is not None:
                    negative_coords = json.dumps(negative_coords, ensure_ascii=False)

        # 无标注时输出空值
        if positive_coords is None:
            positive_coords = ""
        if negative_coords is None:
            negative_coords = ""
        if bboxes is None:
            bboxes = []

        # 准备预览图像（按需缩小）
        preview_images = images
        if needs_scaling:
            _, height, width, _ = images.shape
            new_height = int(height * preview_rescale)
            new_width = int(width * preview_rescale)

            # 转成 PIL 缩放后转回张量
            pil_images = tensor_to_pil(images)
            resized_pil = [img.resize((new_width, new_height), Image.LANCZOS) for img in pil_images]
            preview_images = pil_to_tensor(resized_pil)

        # 计算预览图像张量的哈希
        images_hash = hashlib.md5(preview_images.cpu().numpy().tobytes()).hexdigest()
        rescale_hash = f"{images_hash}_{preview_rescale}"

        # 若图像未变化则复用缓存的预览
        if 'last_images_hash' in self.state and self.state['last_images_hash'] == rescale_hash:
            preview_str = self.state['cached_preview']
            is_init = False
        else:
            preview = _save_preview_images(
                preview_images,
                filename_prefix="ComfyUI_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5)),
            )
            preview_str = json.dumps(preview, ensure_ascii=False)
            # 缓存预览与哈希
            self.state['last_images_hash'] = rescale_hash
            self.state['cached_preview'] = preview_str
            is_init = True

        return {
            "ui": {"preview": [{"preview_str": preview_str, "is_init": is_init}]},
            "result": (positive_coords, negative_coords, bboxes, frame_index),
        }


NODE_CLASS_MAPPINGS = {
    "Yuan_ImagePoint": Yuan_ImagePoint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_ImagePoint": "图像点处理",
}
