"""Yuan Tool · 画布 节点

复刻自 ComfyUI-Yuan 的 Yuan_Canvas 节点，自包含的合成器（V3）：
- 接收最多 8 张图像，在内嵌的 fabric.js 编辑器中可视化放置、旋转、缩放
- 前端合成后的图像会回传后端，作为单个 IMAGE 输出

节点分类: "Yuan Tool/画布"
"""

import folder_paths
from PIL import Image, ImageOps
import numpy as np
import torch
from comfy_execution.graph import ExecutionBlocker
from server import PromptServer
import web
import nodes as comfy_nodes

MAX_RESOLUTION = comfy_nodes.MAX_RESOLUTION

# --- 辅助函数 ---

def _image_signature(tensor):
    """计算 tensor 的内容签名，用于 config 变更检测（替代 base64 字符串比较）。"""
    if tensor is None:
        return None
    try:
        return (
            str(tuple(tensor.shape)),
            round(float(tensor.sum().item() / max(1, tensor.numel())), 6),
        )
    except Exception:
        return None

def _images_batch_signature(tensor):
    """计算一个 batch 的图像内容签名列表。"""
    if tensor is None:
        return None
    try:
        batch_count = tensor.shape[0]
        return [_image_signature(tensor[i:i+1]) for i in range(batch_count)]
    except Exception:
        return None


def _save_images_batch_with_sig(images_tensor):
    """将 images batch 落盘为预览图，返回前端可用的 UI 条目列表（含 sig）。

    参考全景预览节点的 _save_preview_images 实现，使用 ComfyUI 内置的
    PreviewImage 节点落盘。每个 entry 携带 sig（图像内容签名），前端用 sig
    作为图像唯一标识，调换 batch 顺序时 transforms 仍能正确对应同一张图。
    """
    if images_tensor is None:
        return []
    try:
        res = comfy_nodes.PreviewImage().save_images(images_tensor)
        if not (isinstance(res, dict) and "ui" in res and "images" in res["ui"]):
            return []
        entries = res["ui"]["images"]
    except Exception:
        return []

    # 为每个 entry 附加 sig
    sigs = _images_batch_signature(images_tensor) or []
    for i, entry in enumerate(entries):
        if i < len(sigs) and sigs[i]:
            sig = sigs[i]
            entry["sig"] = f"{sig[0]}_{sig[1]}"
        else:
            entry["sig"] = f"img_{i}"
    return entries


routes = PromptServer.instance.routes


@routes.post('/compositor/done')
async def receivedDone(request):
    """前端合成完成后的回调端点（保留兼容，无实际处理）。"""
    return web.json_response({})


class Yuan_Canvas:
    """自包含的合成器（V3）节点。

    接收最多 8 张图像，在内嵌的 fabric.js 编辑器中可视化放置、旋转、缩放。
    前端合成后的图像会回传后端，作为单个 IMAGE 输出。
    """

    result = None
    configCache = None

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        fabricData = kwargs.get("fabricData")
        bg_image = kwargs.get("bg_image")
        images_tensor = kwargs.get("images")
        # 包含图像签名，确保上游图像变化时节点重新执行，前端重新从上游获取图像
        return (fabricData, _image_signature(bg_image), _images_batch_signature(images_tensor))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bg_image": ("IMAGE",),
                "fabricData": ("STRING", {"default": "{}"}),
                "imageName": ("STRING", {"default": "new.png"}),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 32}),
                "padding": ("INT", {"default": 100, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "Yuan Tool/画布"

    DESCRIPTION = (
        "画布合成节点（V3）- 自包含。\n"
        "- 将一组图像（batch）作为独立图层传入\n"
        "- 在内嵌编辑器中可视化放置、旋转、缩放\n"
        "- 缓冲区（padding）可用于暂存不想导出的素材\n"
        "- 配置变化时节点会暂停，便于你构建合成，随后继续执行"
    )

    def composite(self, **kwargs):
        node_id = kwargs.pop('node_id', None)

        imageName = kwargs.get('imageName', "new.png")
        fabricData = kwargs.get("fabricData")

        width = kwargs.get('width', 512)
        height = kwargs.get('height', 512)
        padding = kwargs.get('padding', 100)

        # 后端不处理图像数据，前端直接从上游节点获取图像（参考全景预览节点的 ERP_image 端口）。
        # 图像内容签名仅用于 config 变更检测和 IS_CHANGED。
        bg_image = kwargs.get('bg_image')
        images_tensor = kwargs.get('images')

        # 构建 config 字典（用于变更检测）
        # 用图像内容签名替代 base64 字符串，避免每次落盘 filename 变化导致误判
        config = {
            "node_id": node_id,
            "width": width,
            "height": height,
            "padding": padding,
            "bg_sig": _image_signature(bg_image),
            "names_sigs": _images_batch_signature(images_tensor),
        }

        configChanged = self.configCache != config
        self.configCache = config

        # 始终落盘 images batch，确保切换工作流再返回等场景前端能获取图像。
        # 前端根据画布状态决定是否重新下载：
        #   - configChanged=false 且画布已有图像：不重新下载，保持原位可拖动
        #   - configChanged=false 且画布为空：从 images_entries 重新加载
        #   - configChanged=true：clearInputImages 后重新加载
        images_entries = _save_images_batch_with_sig(images_tensor)
        bg_entries = _save_images_batch_with_sig(bg_image)

        ui = {
            "test": ("value",),
            "padding": [padding],
            "width": [width],
            "height": [height],
            "config_node_id": [node_id],
            "node_id": [node_id],
            "fabricData": [fabricData],
            "awaited": [self.result],
            "configChanged": [configChanged],
            "images_entries": images_entries,
            "bg_entries": bg_entries,
        }

        # 通知前端，相当于"已执行"
        detail = {"output": ui, "node": node_id}
        PromptServer.instance.send_sync("compositor_init", detail)

        imageExists = folder_paths.exists_annotated_filepath(imageName)
        # 当配置变化或尚无合成图像时阻塞执行
        if imageName == "new.png" or not imageExists or configChanged:
            blocker_result = tuple([ExecutionBlocker(None)] * len(self.RETURN_TYPES))
            return {
                "ui": ui,
                "result": blocker_result
            }
        else:
            # 加载已上传的合成图像并返回
            image_path = folder_paths.get_annotated_filepath(imageName)
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None, ]

            return {
                "ui": ui,
                "result": (image,)
            }


NODE_CLASS_MAPPINGS = {
    "Yuan_Canvas": Yuan_Canvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_Canvas": "Yuan_画布",
}
