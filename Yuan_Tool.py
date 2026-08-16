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
        list_inputs = {}
        for i in range(1, 9):
            prev = f"图像列表{i - 1}、图像列表{i - 2}" if i >= 3 else "常显"
            list_inputs[f"image_list_{i}"] = (
                "IMAGE",
                {
                    "display_name": f"图像列表{i}",
                    "tooltip": (
                        f"第 {i} 路批量图像（列表模式下使用），最多 8 张，超出自动截断。"
                        if i <= 2
                        else f"第 {i} 路批量图像（列表模式下使用），最多 8 张，超出自动截断。仅当 {prev} 都已接入后自动出现，最多 8 路。"
                    ),
                },
            )

        return {
            "required": {
                "width": ("INT", {"default": 736, "min": 32, "max": 8192, "step": 32, "display_name": "宽度", "tooltip": "生成视频宽度（32 的倍数）"}),
                "height": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 32, "display_name": "高度", "tooltip": "生成视频高度（32 的倍数）"}),
                "frame_multiplier": ([1, 8, 16, 24, 32], {"default": 16, "display_name": "每图帧数", "tooltip": "每张图像持续的帧数；16 帧≈每图 0.67 秒。第一组会额外多 1 帧用于 VAE 8 帧分组对齐"}),
                "list_mode": ("BOOLEAN", {"default": False, "label_on": "列表模式", "label_off": "单帧模式", "display_name": "输入模式", "tooltip": "开启：通过图像列表1..8端口多路批量输入（1、2常显，接满后依次出现后续端口，每路最多 8 张，总共最多 8 张）；关闭：端口 1-8 逐张输入（1、2 常显，前置连满后依次出现后续端口）"}),
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
                **list_inputs,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("视频帧",)
    OUTPUT_TOOLTIPS = ("按图像顺序拼接并乘以每图帧数生成的视频帧 batch（0..1 float），可直接送入 VAE 编码或下一个节点。",)
    FUNCTION = "create_video"
    CATEGORY = "Yuan Tool/图像"
    DESCRIPTION = (
        "多帧参考节点：将多张主体图像按顺序展开为视频帧（每图帧数固定），"
        "可选附加背景帧。支持端口 1-8 逐张输入（端口递进显示）或"
        "图像列表1..8 多路批量输入（每路最多 8 张、总共最多 8 张，端口递进显示）。"
        "图像列表端口若接收到上游「筛选图像」节点的 64×64 空兜底图，将视为该端口未接入。"
    )

    @staticmethod
    def _is_empty_fallback_image(image):
        """判断是否为「筛选图像」节点的空兜底图：形状 (1,64,64,3) 且全为 0。"""
        if image is None:
            return True
        t = image
        if isinstance(t, torch.Tensor):
            if t.ndim == 4 and t.shape[0] == 1 and t.shape[1] == 64 and t.shape[2] == 64 and t.shape[3] == 3:
                try:
                    return bool(torch.allclose(t.float(), torch.zeros_like(t.float()), atol=1e-6))
                except Exception:
                    return False
            return False
        if isinstance(t, (list, tuple)) and len(t) == 1:
            return YuanTool._is_empty_fallback_image(t[0])
        # 其他情况（非张量，或形状不符）按非空处理
        return False

    def _iter_tensor_images(self, image_list, limit_per_list, prepare, target_size):
        """从一个 image_list（tensor 或 list）中按上限取出最多 limit_per_list 张图像，经 prepare 后 yield。"""
        if image_list is None:
            return
        n = 0
        if isinstance(image_list, torch.Tensor):
            batch = image_list.shape[0] if image_list.ndim == 4 else 1
            take = min(batch, limit_per_list)
            for i in range(take):
                img = image_list[i] if image_list.ndim == 4 else image_list
                yield prepare(img, target_size, preserve_full=True)
                n += 1
                if n >= limit_per_list:
                    return
        elif isinstance(image_list, (list, tuple)):
            for img in image_list:
                if n >= limit_per_list:
                    return
                yield prepare(img, target_size, preserve_full=True)
                n += 1

    def create_video(self, width, height, frame_multiplier, list_mode, **kwargs):
        background = kwargs.get("background")
        prepare = self._prepare_image
        target_size = (width, height)

        subjects = []
        TOTAL_LIMIT = 8
        LIMIT_PER_LIST = 8

        if list_mode:
            # 按 1..8 顺序取每个 image_list 端口；过滤空兜底端口；收集至总上限 8 张
            for i in range(1, 9):
                if len(subjects) >= TOTAL_LIMIT:
                    break
                key = f"image_list_{i}"
                img_list = kwargs.get(key)
                if YuanTool._is_empty_fallback_image(img_list):
                    continue
                for prepared in self._iter_tensor_images(img_list, LIMIT_PER_LIST, prepare, target_size):
                    subjects.append(prepared)
                    if len(subjects) >= TOTAL_LIMIT:
                        break
        else:
            for name in ("1", "2", "3", "4", "5", "6", "7", "8"):
                image = kwargs.get(name)
                if image is None:
                    continue
                # 单图端口同样识别「筛选图像」的空兜底图，视为未接入
                if YuanTool._is_empty_fallback_image(image):
                    continue
                subjects.append(prepare(image, target_size, preserve_full=True))

        # 背景：空兜底图(筛选节点空输出)视为未接入——不产出背景帧，也不作为填充背景
        bg_is_empty = YuanTool._is_empty_fallback_image(background)
        background_image = None if bg_is_empty else (
            prepare(background, target_size, preserve_full=False) if background is not None else None
        )

        # 背景帧数与每图帧数一致；slot1 额外 1 帧用于 VAE 8 帧分组对齐（无主体或帧数为 1 时无意义）
        bg_frame_count = frame_multiplier if background_image is not None else 0
        first_slot_extra = 1 if (subjects and frame_multiplier > 1) else 0
        frame_count = len(subjects) * frame_multiplier + first_slot_extra + bg_frame_count
        frames = self._expand_frames_with_info(
            subjects, background_image, frame_multiplier, frame_count
        )

        # 有帧则堆叠输出；完全无有效输入时输出 1 张 64×64 全黑空图（与 GetImage 兜底一致）
        if frames:
            output = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
        else:
            output = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return (output,)

    @staticmethod
    def _tensor_to_rgb_array(image):
        """统一转成 0-255 的 uint8 RGB 数组（兼容张量/数组、灰度/RGBA）。"""
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                image = image[0]
            image = image.detach().cpu().numpy()

        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

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

        # slot1 额外 1 帧吸收 VAE 的独立首帧（frame 0 不参与 8 帧分组），使后续帧边界对齐 8 帧分组、避免混合帧
        for index, image in enumerate(subjects):
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
        batch_size = images.shape[0] if images is not None and images.ndim >= 1 else 0

        valid_indices = []
        if batch_size > 0:
            for token in indexes.split(','):
                token = token.strip()
                if not token:
                    continue
                try:
                    idx = int(token)
                except ValueError:
                    continue
                if 0 <= idx < batch_size:
                    valid_indices.append(idx)

        if valid_indices:
            indices_tensor = torch.tensor(valid_indices, dtype=torch.long)
            chosen_images = images[indices_tensor]
        else:
            chosen_images = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return (chosen_images,)


NODE_CLASS_MAPPINGS = {
    "YuanTool": YuanTool,
    "GetImage": GetImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanTool": "多帧参考",
    "GetImage": "筛选图像",
}
