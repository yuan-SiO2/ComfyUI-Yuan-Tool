import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None


class YuanTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 736, "min": 32, "max": 8192, "step": 32, "display_name": "宽度", "tooltip": "生成视频宽度（32 的倍数）"}),
                "height": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 32, "display_name": "高度", "tooltip": "生成视频高度（32 的倍数）"}),
                "frame_multiplier": ([1, 8, 16, 24, 32], {"default": 16, "display_name": "每图帧数", "tooltip": "每张图像持续的帧数；16 帧≈每图 0.67 秒。第一组会额外多 1 帧用于 VAE 8 帧分组对齐"}),
                "list_mode": ("BOOLEAN", {"default": False, "label_on": "列表模式", "label_off": "单帧模式", "display_name": "输入模式", "tooltip": "开启：通过单个 image_list 端口接收多张图像（最多 8 张，超出自动截断）；关闭：端口 1-8 逐张输入（1、2 常显，前置连满后依次出现后续端口）"}),
            },
            "optional": {
                "background": ("IMAGE", {"display_name": "背景", "tooltip": "视频背景图像（可留空），尾部附加 frame_multiplier 帧"}),
                "1": ("IMAGE", {"display_name": "图像1", "tooltip": "第 1 张主体图像"}),
                "2": ("IMAGE", {"display_name": "图像2", "tooltip": "第 2 张主体图像"}),
                "3": ("IMAGE", {"display_name": "图像3", "tooltip": "第 3 张主体图像（1、2 连满后显示）"}),
                "4": ("IMAGE", {"display_name": "图像4", "tooltip": "第 4 张主体图像（前 3 张连满后显示）"}),
                "5": ("IMAGE", {"display_name": "图像5", "tooltip": "第 5 张主体图像（前 4 张连满后显示）"}),
                "6": ("IMAGE", {"display_name": "图像6", "tooltip": "第 6 张主体图像（前 5 张连满后显示）"}),
                "7": ("IMAGE", {"display_name": "图像7", "tooltip": "第 7 张主体图像（前 6 张连满后显示）"}),
                "8": ("IMAGE", {"display_name": "图像8", "tooltip": "第 8 张主体图像（前 7 张连满后显示）"}),
                "image_list": ("IMAGE", {"display_name": "图像列表", "tooltip": "批量图像端口（列表模式下显示），最多 8 张，超出自动截断"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("视频帧",)
    OUTPUT_TOOLTIPS = ("按图像顺序拼接并乘以每图帧数生成的视频帧 batch（0..1 float），可直接送入 VAE 编码或下一个节点。",)
    FUNCTION = "create_video"
    CATEGORY = "Yuan Tool/图像"
    DESCRIPTION = (
        "多帧参考节点：将多张主体图像按顺序展开为视频帧（每图帧数固定），"
        "可选附加背景帧。支持端口 1-8 逐张输入（端口递进显示）或 image_list 批量输入（最多 8 张）。"
    )

    def create_video(self, width, height, frame_multiplier, list_mode, **kwargs):
        background = kwargs.get("background")

        subjects = []
        if list_mode:
            image_list = kwargs.get("image_list")
            if image_list is not None:
                if isinstance(image_list, torch.Tensor):
                    batch = image_list.shape[0] if image_list.ndim == 4 else 1
                    batch = min(batch, 8)
                    for i in range(batch):
                        img = image_list[i] if image_list.ndim == 4 else image_list
                        subjects.append(self._prepare_image(img, (width, height), preserve_full=True))
                elif isinstance(image_list, list):
                    for idx, img in enumerate(image_list[:8]):
                        subjects.append(self._prepare_image(img, (width, height), preserve_full=True))
        else:
            for name in ("1", "2", "3", "4", "5", "6", "7", "8"):
                image = kwargs.get(name)
                if image is not None:
                    subjects.append(self._prepare_image(image, (width, height), preserve_full=True))

        background_image = self._prepare_image(background, (width, height), preserve_full=False) if background is not None else None

        # 背景帧数：有背景图时与 frame_multiplier 一致，无背景时为 0
        bg_frame_count = frame_multiplier if background_image is not None else 0
        # slot1 多1帧用于 VAE 8帧分组对齐；frame_multiplier=1 时无对齐意义，不加
        first_slot_extra = 1 if frame_multiplier > 1 else 0
        frame_count = len(subjects) * frame_multiplier + first_slot_extra + bg_frame_count
        frames = self._expand_frames_with_info(
            subjects, background_image, frame_multiplier, frame_count
        )
        output = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)

        return (output,)

    @staticmethod
    def _tensor_to_rgb_array(image):
        if isinstance(image, torch.Tensor):
            # ComfyUI image tensor is [B, H, W, C] or [H, W, C]
            if image.ndim == 4:
                image = image[0]
            image = image.detach().cpu().numpy()

        image = np.asarray(image)
        # Ensure it's in 0-255 uint8 format
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

        # Handle grayscale or RGBA
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

        return np.ascontiguousarray(image)

    @staticmethod
    def _prepare_image(image, target_size, preserve_full=False):
        image_array = YuanTool._tensor_to_rgb_array(image)
        source_height, source_width = image_array.shape[:2]
        target_width, target_height = target_size

        if source_width == target_width and source_height == target_height:
            return np.ascontiguousarray(image_array)

        if preserve_full:
            scale = min(target_width / source_width, target_height / source_height)
            resized_width = max(1, min(target_width, round(source_width * scale)))
            resized_height = max(1, min(target_height, round(source_height * scale)))
            resized = YuanTool._resize(image_array, resized_width, resized_height)
            canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
            left = (target_width - resized_width) // 2
            top = (target_height - resized_height) // 2
            canvas[top:top + resized_height, left:left + resized_width] = resized
            return np.ascontiguousarray(canvas)

        scale = max(target_width / source_width, target_height / source_height)
        resized_width = max(target_width, round(source_width * scale))
        resized_height = max(target_height, round(source_height * scale))
        resized = YuanTool._resize(image_array, resized_width, resized_height)
        left = (resized_width - target_width) // 2
        top = (resized_height - target_height) // 2
        return np.ascontiguousarray(
            resized[top:top + target_height, left:left + target_width]
        )

    @staticmethod
    def _resize(image_array, width, height):
        if cv2 is not None:
            interpolation = (
                cv2.INTER_AREA
                if width < image_array.shape[1] or height < image_array.shape[0]
                else cv2.INTER_LANCZOS4
            )
            return cv2.resize(image_array, (width, height), interpolation=interpolation)

        chw = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()
        resized = F.interpolate(
            chw,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return np.ascontiguousarray(
            resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
        )

    @staticmethod
    def _expand_frames_with_info(subjects, background, frame_multiplier, frame_count):
        frames = []

        # slot1 多1帧: VAE 第一帧 (frame 0) 是独立帧 → latent 0,
        # 不参与8帧分组。slot1 吸收这个独立帧后, 后续 slot 的帧边界
        # 才能对齐 VAE 的8帧分组边界, 避免混合帧。
        # 例: frame_multiplier=16
        #   slot1: frames[0,16]  → latent 0(frame0) + latent 1(f1-8) + latent 2(f9-16) = 纯img1
        #   slot2: frames[17,32] → latent 3(f17-24) + latent 4(f25-32) = 纯img2
        #   slot3: frames[33,48] → latent 5(f33-40) + latent 6(f41-48) = 纯img3
        for index, image in enumerate(subjects):
            # slot1 多1帧用于 VAE 8帧分组对齐；frame_multiplier=1 时无对齐意义，不加
            extra = 1 if (index == 0 and frame_multiplier > 1) else 0
            repeats = frame_multiplier + extra
            frames.extend([image] * repeats)

        # 仅当提供了背景图时才添加背景帧（帧数与 frame_multiplier 一致）
        if background is not None:
            frames.extend([background] * frame_multiplier)

        if len(frames) > frame_count:
            frames = frames[:frame_count]
        elif len(frames) < frame_count:
            # 不足帧填充：有背景用背景，无背景用最后一个主体帧（避免空白）
            filler = background if background is not None else (subjects[-1] if subjects else None)
            if filler is not None:
                while len(frames) < frame_count:
                    frames.append(filler)

        return frames


class GetImage:

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("筛选图像",)
    OUTPUT_TOOLTIPS = ("按指定索引从批量图像中筛选出的帧（顺序保持）。",)
    FUNCTION = "indexedimagesfrombatch"
    CATEGORY = "Yuan Tool/图像"
    DESCRIPTION = "从批量中筛选一张或者多张图像。用逗号分隔索引（从 0 起），支持多行；无效索引会被自动忽略，全无效时回退到第 0 张。"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"display_name": "输入图像", "tooltip": "要筛选的图像批次（batch）"}),
                "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True, "display_name": "索引列表", "tooltip": "逗号分隔的索引（从 0 起），如 0,2,5；也支持多行。超出范围的索引会被忽略。"}),
            },
        }

    def indexedimagesfrombatch(self, images, indexes):
        batch_size = images.shape[0]
        index_list = [int(index.strip()) for index in indexes.split(',')]
        valid_indices = [i for i in index_list if 0 <= i < batch_size]
        if not valid_indices:
            valid_indices = [0]
        indices_tensor = torch.tensor(valid_indices, dtype=torch.long)
        chosen_images = images[indices_tensor]
        return (chosen_images,)


NODE_CLASS_MAPPINGS = {
    "YuanTool": YuanTool,
    "GetImage": GetImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanTool": "多帧参考",
    "GetImage": "筛选图像",
}
