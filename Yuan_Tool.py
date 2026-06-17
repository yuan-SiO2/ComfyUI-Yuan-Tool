import cv2
import numpy as np
import torch
from PIL import Image


class YuanTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "width": ("INT", {"default": 736, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 32}),
                "frame_multiplier": ([8, 10, 12, 16], {"default": 8}),
                "list_mode": ("BOOLEAN", {"default": False, "label_on": "true", "label_off": "false"}),
            },
            "optional": {
                "1": ("IMAGE",),
                "2": ("IMAGE",),
                "3": ("IMAGE",),
                "4": ("IMAGE",),
                "image_list": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "create_video"
    CATEGORY = "Yuan Tool/图像"

    def create_video(self, background, width, height, frame_multiplier, list_mode, **kwargs):
        if background is None:
            raise ValueError("Background image is required and cannot be empty.")

        images = []
        if list_mode:
            image_list = kwargs.get("image_list")
            if image_list is not None:
                if isinstance(image_list, torch.Tensor):
                    # ComfyUI images are [B, H, W, C]
                    batch = image_list.shape[0] if image_list.ndim == 4 else 1
                    # Limit to first 4 images as per design
                    batch = min(batch, 4)
                    for i in range(batch):
                        img = image_list[i] if image_list.ndim == 4 else image_list
                        images.append(self._prepare_image(img, (width, height)))
                elif isinstance(image_list, list):
                    for img in image_list[:4]:
                        images.append(self._prepare_image(img, (width, height)))
        else:
            for name in ("1", "2", "3", "4"):
                image = kwargs.get(name)
                if image is not None:
                    images.append(self._prepare_image(image, (width, height)))

        # Add background last
        images.append(self._prepare_image(background, (width, height)))

        frame_count = len(images) * frame_multiplier + 1
        frames = self._expand_frames(images, frame_count)
        # Convert list of numpy arrays [H, W, C] to torch tensor [B, H, W, C]
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
    def _prepare_image(image, target_size):
        image_array = YuanTool._tensor_to_rgb_array(image)
        # Resize to target width/height
        # Note: target_size is (width, height) for cv2.resize
        if image_array.shape[1] != target_size[0] or image_array.shape[0] != target_size[1]:
            image_array = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LANCZOS4)
        return np.ascontiguousarray(image_array)

    @staticmethod
    def _expand_frames(images, frame_count):
        if not images:
            return []
        base_count = frame_count // len(images)
        remainder = frame_count % len(images)
        frames = []
        for index, image in enumerate(images):
            repeats = base_count + (1 if index < remainder else 0)
            frames.extend([image] * repeats)
        
        # Ensure exactly frame_count frames
        if len(frames) > frame_count:
            frames = frames[:frame_count]
        elif len(frames) < frame_count:
            while len(frames) < frame_count:
                frames.append(images[-1])
        return frames


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
        index_list = [int(index.strip()) for index in indexes.split(',')]
        indices_tensor = torch.tensor(index_list, dtype=torch.long)
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
