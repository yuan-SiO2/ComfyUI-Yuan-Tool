import logging
import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

log = logging.getLogger(__name__)

MSR_INFO_VERSION = 2


def _estimate_ref_latent_frames(source_frame_count):
    if source_frame_count <= 1:
        return max(1, source_frame_count)
    return int(round((source_frame_count - 1) / 8.0)) + 1


def _frame_range_to_latent(frame_start, frame_end):
    """将帧范围映射到latent帧索引。

    LTXV VAE 时间压缩结构: 第一帧独立 → latent 0, 后续每8帧一组。
    - frame 0       → latent 0 (独立帧)
    - frames[1,8]   → latent 1
    - frames[9,16]  → latent 2
    - frames[17,24] → latent 3
    - ...
    公式: frame 0 → latent 0; frame N (N>0) → (N-1)//8 + 1
    """
    frame_start = int(frame_start)
    frame_end = int(frame_end)
    if frame_start <= 0:
        latent_start = 0
    else:
        latent_start = (frame_start - 1) // 8 + 1
    if frame_end <= 0:
        latent_end = 0
    else:
        latent_end = (frame_end - 1) // 8 + 1
    return latent_start, latent_end


class YuanTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 736, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 32}),
                "frame_multiplier": ([8, 16, 24, 32], {"default": 16}),
                "list_mode": ("BOOLEAN", {"default": False, "label_on": "true", "label_off": "false"}),
            },
            "optional": {
                "background": ("IMAGE",),
                "1": ("IMAGE",),
                "2": ("IMAGE",),
                "3": ("IMAGE",),
                "4": ("IMAGE",),
                "image_list": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MSR_INFO")
    RETURN_NAMES = ("output", "msr_info")
    FUNCTION = "create_video"
    CATEGORY = "Yuan Tool/图像"

    def create_video(self, width, height, frame_multiplier, list_mode, **kwargs):
        background = kwargs.get("background")

        subjects = []
        subject_slots = []
        if list_mode:
            image_list = kwargs.get("image_list")
            if image_list is not None:
                if isinstance(image_list, torch.Tensor):
                    batch = image_list.shape[0] if image_list.ndim == 4 else 1
                    batch = min(batch, 4)
                    for i in range(batch):
                        img = image_list[i] if image_list.ndim == 4 else image_list
                        subjects.append(self._prepare_image(img, (width, height), preserve_full=True))
                        subject_slots.append(str(i + 1))
                elif isinstance(image_list, list):
                    for idx, img in enumerate(image_list[:4]):
                        subjects.append(self._prepare_image(img, (width, height), preserve_full=True))
                        subject_slots.append(str(idx + 1))
        else:
            for name in ("1", "2", "3", "4"):
                image = kwargs.get(name)
                if image is not None:
                    subjects.append(self._prepare_image(image, (width, height), preserve_full=True))
                    subject_slots.append(name)

        background_image = self._prepare_image(background, (width, height), preserve_full=False) if background is not None else None

        # 背景帧数：有背景图时 +8 帧，无背景时为 0
        bg_frame_count = 8 if background_image is not None else 0
        frame_count = len(subjects) * frame_multiplier + 1 + bg_frame_count
        frames, subject_frame_ranges, background_frame_range = self._expand_frames_with_info(
            subjects, background_image, frame_multiplier, frame_count
        )
        output = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)

        latent_count = _estimate_ref_latent_frames(frame_count)

        subjects_info = []
        for slot, (start, end) in zip(subject_slots, subject_frame_ranges):
            latent_start, latent_end = _frame_range_to_latent(start, end)
            item = {
                "slot": slot,
                "role": "subject",
                "frame_start": start,
                "frame_end": end,
                "frame_count": end - start + 1,
                "latent_aligned": True,
                "latent_start": latent_start,
                "latent_end": latent_end,
                "latent_count": latent_end - latent_start + 1,
            }
            subjects_info.append(item)

        msr_info = {
            "version": MSR_INFO_VERSION,
            "token_order": "target_then_reference",
            "reference_frame_count": frame_count,
            "reference_latent_count": latent_count,
            "width": width,
            "height": height,
            "subjects": subjects_info,
        }

        # 仅当提供了背景图时才添加 background 信息
        if background_image is not None:
            bg_start, bg_end = background_frame_range
            bg_latent_start, bg_latent_end = _frame_range_to_latent(bg_start, bg_end)
            msr_info["background"] = {
                "slot": "background",
                "role": "background",
                "frame_start": bg_start,
                "frame_end": bg_end,
                "frame_count": bg_end - bg_start + 1,
                "latent_aligned": True,
                "latent_start": bg_latent_start,
                "latent_end": bg_latent_end,
                "latent_count": bg_latent_end - bg_latent_start + 1,
            }

        return (output, msr_info)

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
        subject_frame_ranges = []
        cursor = 0

        # slot1 多1帧: VAE 第一帧 (frame 0) 是独立帧 → latent 0,
        # 不参与8帧分组。slot1 吸收这个独立帧后, 后续 slot 的帧边界
        # 才能对齐 VAE 的8帧分组边界, 避免混合帧。
        # 例: frame_multiplier=16
        #   slot1: frames[0,16]  → latent 0(frame0) + latent 1(f1-8) + latent 2(f9-16) = 纯img1
        #   slot2: frames[17,32] → latent 3(f17-24) + latent 4(f25-32) = 纯img2
        #   slot3: frames[33,48] → latent 5(f33-40) + latent 6(f41-48) = 纯img3
        for index, image in enumerate(subjects):
            repeats = frame_multiplier + (1 if index == 0 else 0)
            start = cursor
            end = cursor + repeats - 1
            subject_frame_ranges.append((start, end))
            frames.extend([image] * repeats)
            cursor = end + 1

        background_frame_range = None
        # 仅当提供了背景图时才添加背景帧
        if background is not None:
            bg_start = cursor
            bg_end = cursor + 7
            frames.extend([background] * 8)
            cursor = bg_end + 1
            background_frame_range = (bg_start, bg_end)

        if len(frames) > frame_count:
            frames = frames[:frame_count]
            subject_frame_ranges = [
                (s, min(e, frame_count - 1))
                for s, e in subject_frame_ranges
                if s < frame_count
            ]
            if background_frame_range is not None:
                bg_start, bg_end = background_frame_range
                background_frame_range = (bg_start, min(bg_end, frame_count - 1))
        elif len(frames) < frame_count:
            # 不足帧填充：有背景用背景，无背景用最后一个主体帧（避免空白）
            filler = background if background is not None else (subjects[-1] if subjects else None)
            if filler is not None:
                while len(frames) < frame_count:
                    frames.append(filler)
            if background_frame_range is not None:
                bg_start, bg_end = background_frame_range
                background_frame_range = (bg_start, frame_count - 1)

        return frames, subject_frame_ranges, background_frame_range


class GetImage:

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "indexedimagesfrombatch"
    CATEGORY = "Yuan Tool/图像"
    DESCRIPTION = "从批量中筛选一张或者多张图像。"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
            },
        }

    def indexedimagesfrombatch(self, images, indexes):
        batch_size = images.shape[0]
        index_list = [int(index.strip()) for index in indexes.split(',')]
        valid_indices = [i for i in index_list if 0 <= i < batch_size]
        if not valid_indices:
            valid_indices = [0]
        log.info("[GetImage] 请求索引 %s，批次大小 %d，有效索引 %s", index_list, batch_size, valid_indices)
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
