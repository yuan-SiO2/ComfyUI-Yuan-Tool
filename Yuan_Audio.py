"""Yuan 音频列表/分流/加载（播放器裁剪分段）节点。"""

import json
import os
import re

import av
import torch

import folder_paths

_CATEGORY = "Yuan Tool/音频"

# 支持的音频/视频扩展名（视频提取音轨），与前端过滤一致
_AUDIO_EXTS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma",
    ".aiff", ".aif", ".mp4", ".m4v", ".webm", ".mov", ".avi", ".mkv",
}


def _f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """转为 float32 PCM。"""
    if wav.dtype.is_floating_point:
        return wav
    if wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    if wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"不支持的音频 dtype: {wav.dtype}")


def _load_audio_file(filepath: str):
    """PyAV 解码音频文件，返回 (waveform, sample_rate)，waveform 为 (channels, samples) float32。"""
    with av.open(filepath) as container:
        if not container.streams.audio:
            raise ValueError("文件中未找到音频流。")

        stream = container.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        for frame in container.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)

        if not frames:
            raise ValueError("未解码到音频帧。")

        wav = torch.cat(frames, dim=1)
        return _f32_pcm(wav), sample_rate


# 音频数量上限（与前端 MAX_AUDIOS 一致）
MAX_AUDIOS = 30


def _parse_filter_indexes(indexes: str, count: int):
    """解析筛选索引：逗号分隔、从 0 起。留空/全无效返回 None（表示全部）。"""
    if not indexes or not indexes.strip():
        return None
    valid = []
    for token in indexes.replace("\n", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            continue
        if 0 <= idx < count:
            valid.append(idx)
    return valid if valid else None


def _silence_audio() -> dict:
    """空音频占位：1ms 静音（16kHz 单声道）。"""
    return {"waveform": torch.zeros(1, 1, 16), "sample_rate": 16000}


class YuanAudioList:
    """Yuan 音频列表：从输入目录加载音频文件，按筛选索引输出 AUDIO 列表。"""

    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "音频列表：从输入目录加载音频文件（前端方块滑轨选择/上传，上限 30 个），"
        "按筛选索引输出一个音频列表。留空输出全部；填索引（从 0 起，逗号分隔）"
        "可单选或多选，如 0 或 0,2,5；无效索引会被忽略，全部无效时回退输出全部。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "音频文件列表数据（JSON），由前端方块滑轨自动维护，请勿手动编辑。",
                }),
                "筛选索引": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "按索引筛选输出（参考筛选图像）：从 0 起，逗号分隔，"
                        "如 0（单选）或 0,2,5（多选）；留空输出全部；"
                        "无效索引忽略，全部无效时回退全部。索引对应方块上的序号减 1。"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频列表",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("按筛选索引排列的音频列表；未筛选时为全部音频。",)
    FUNCTION = "execute"

    def execute(self, audio_list_data: str = "", 筛选索引: str = ""):
        # 解析前端滑轨维护的文件列表
        files = []
        if audio_list_data and audio_list_data.strip():
            try:
                data = json.loads(audio_list_data)
                if isinstance(data, dict) and isinstance(data.get("files"), list):
                    files = [f for f in data["files"] if isinstance(f, str) and f.strip()]
            except json.JSONDecodeError:
                pass
        files = files[:MAX_AUDIOS]

        # 先在文件级筛选（索引 = 方块序号 - 1，不受解码失败影响），再解码
        indexes = _parse_filter_indexes(筛选索引, len(files))
        if indexes is not None:
            files = [files[i] for i in indexes]

        # 解码文件（失败跳过）
        audios = []
        for name in files:
            try:
                filepath = folder_paths.get_annotated_filepath(name)
                waveform, sample_rate = _load_audio_file(filepath)
                audios.append({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})
            except Exception:
                continue

        return (audios,)


class YuanAudioSplit:
    """Yuan 音频分流：列表按索引顺序分发到多个输出端口，不足端口输出空音频。"""

    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "音频分流：接收一个音频列表（对接 Yuan 音频列表），"
        "按顺序分配到多个输出端口——索引 0 对应第 1 个输出端口，依此类推；"
        "列表中没有对应音频的端口输出空音频（1ms 静音）。"
    )
    # 声明 INPUT_IS_LIST：上游 OUTPUT_IS_LIST 的列表在此一次完整接收（不逐项批处理）
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频列表": ("AUDIO", {
                    "tooltip": "输入的音频列表（来自 Yuan 音频列表的输出）。",
                }),
                "输出数量": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": MAX_AUDIOS,
                    "tooltip": "输出端口数量（默认 2）。索引 0 → 音频1，索引 1 → 音频2，依此类推。",
                }),
            },
        }

    # 后端固定声明 MAX_AUDIOS 个输出，前端按「输出数量」修剪显示
    RETURN_TYPES = ("AUDIO",) * MAX_AUDIOS
    RETURN_NAMES = tuple(f"音频{i + 1}" for i in range(MAX_AUDIOS))
    OUTPUT_TOOLTIPS = tuple(
        f"列表中索引 {i} 的音频；无对应音频时输出空音频。" for i in range(MAX_AUDIOS)
    )
    FUNCTION = "execute"

    def execute(self, 音频列表=None, 输出数量=2):
        # INPUT_IS_LIST 下 widget 值可能被广播为列表，统一归一化
        if isinstance(输出数量, list):
            输出数量 = 输出数量[0] if 输出数量 else 2
        try:
            count = int(输出数量)
        except (TypeError, ValueError):
            count = 2
        count = max(1, min(count, MAX_AUDIOS))

        audios = 音频列表 if isinstance(音频列表, list) else ([音频列表] if 音频列表 else [])

        return tuple(
            audios[i] if i < len(audios) else _silence_audio()
            for i in range(count)
        )


def _parse_durations(text) -> list:
    """解析"分段时长"文本：支持逗号(中英文)/顿号/分号/空白分隔，如 "6,3,5"。"""
    if text is None:
        return []
    parts = re.split(r"[,，、;；\s]+", str(text).strip())
    out = []
    for p in parts:
        if not p:
            continue
        try:
            v = float(p)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out.append(v)
    return out


class YuanAudioLoad:
    """Yuan 加载音频：内嵌播放器 + 时间轴裁剪 + 智能分段。"""

    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "加载音频：内置音频预览、时间轴裁剪与播放控制。\n"
        "自定义裁切：用开始/结束时间(或拖动时间轴手柄)截取一段。\n"
        "智能分段：按'分段时长'列表(逗号分隔，如 6,3,5)把 [开始时间, 结束时间] "
        "窗口切分成若干分段，用小方格显示在时间轴上，配合'分段索引'选择输出的分段。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        try:
            files = folder_paths.get_filename_list("audio")
        except Exception:
            files = []

        if not files:
            input_dir = folder_paths.get_input_directory()
            if os.path.exists(input_dir):
                try:
                    files = sorted(folder_paths.filter_files_content_types(
                        [f for f in os.listdir(input_dir)
                         if os.path.isfile(os.path.join(input_dir, f))],
                        ["audio", "video"],
                    ))
                except Exception:
                    files = [
                        f for f in os.listdir(input_dir)
                        if os.path.isfile(os.path.join(input_dir, f))
                        and os.path.splitext(f)[1].lower() in _AUDIO_EXTS
                    ]

        if not files:
            files = ["none"]

        return {
            "required": {
                "音频": (files, {"audio_upload": True}),
                "开始时间": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
                "结束时间": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
                "时长": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
                "输出模式": (["自定义裁切", "智能分段"], {"default": "自定义裁切"}),
                "分段时长": ("STRING", {
                    "default": "",
                    "tooltip": "智能分段下每段的时长(秒)，用逗号分隔；例如 6,3,5 表示第1段6秒、第2段3秒、第3段5秒。留空时整个窗口作为 1 段。",
                }),
                "分段索引": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 1,
                    "tooltip": "智能分段模式下，选择要输出的分段（第 1 段为 0）。",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "INT")
    RETURN_NAMES = ("音频", "时长", "分段总数")
    OUTPUT_TOOLTIPS = ("裁剪/分段后的音频。", "输出音频的时长（秒）。", "智能分段模式下的分段总数（自定义裁切恒为 1）。")
    FUNCTION = "execute"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # 直接放行，避免文件不在下拉列表时 ComfyUI 报 "Value not in list"
        return True

    def execute(self, 音频, 开始时间, 结束时间, 时长, 输出模式="自定义裁切", 分段时长="6,3,5", 分段索引=0):
        try:
            audio_path = folder_paths.get_annotated_filepath(音频) if 音频 != "none" else ""
        except Exception:
            audio_path = ""

        # 文件缺失或解码失败时回退为 1 秒静音，避免工作流崩溃
        if 音频 == "none" or not audio_path or not os.path.exists(audio_path):
            sample_rate = 44100
            waveform = torch.zeros((2, 44100))
        else:
            try:
                waveform, sample_rate = _load_audio_file(audio_path)
            except Exception:
                sample_rate = 44100
                waveform = torch.zeros((2, 44100))

        file_duration = waveform.shape[1] / sample_rate

        # 有效裁切窗口 [win_start, win_end]（秒）
        win_start = min(max(0.0, float(开始时间 or 0)), file_duration)
        if 结束时间 and float(结束时间) > win_start:
            win_end = min(float(结束时间), file_duration)
        else:
            win_end = file_duration  # 0 表示文件末尾
        win_start = min(win_start, win_end)

        if 输出模式 == "智能分段":
            # 在窗口内按"分段时长"列表切分
            durations = _parse_durations(分段时长)
            bounds = []
            cur = win_start
            for d in durations:
                if cur >= win_end - 1e-6:
                    break
                cur = min(cur + d, win_end)
                bounds.append(cur)
            if not bounds:
                bounds = [win_end]  # 无有效时长输入 -> 整个窗口视为 1 段

            seg_count = len(bounds)
            sel_idx = max(0, min(int(分段索引 or 0), seg_count - 1))
            seg_start = win_start if sel_idx == 0 else bounds[sel_idx - 1]
            seg_end = bounds[sel_idx]
        else:
            # 自定义裁切：直接输出 [win_start, win_end]
            seg_count = 1
            seg_start = win_start
            seg_end = win_end

        # 秒 -> 采样帧并裁剪
        start_frame = int(seg_start * sample_rate)
        end_frame = int(seg_end * sample_rate)
        start_frame = min(start_frame, waveform.shape[1])
        end_frame = min(end_frame, waveform.shape[1])
        start_frame = min(start_frame, end_frame)

        trimmed_waveform = waveform[:, start_frame:end_frame]

        # 裁剪结果为空时补 1 个采样点，避免下游收到空张量
        if trimmed_waveform.shape[1] == 0:
            trimmed_waveform = torch.zeros((waveform.shape[0], 1))

        # ComfyUI 标准 AUDIO 类型: [batch, 声道, 时间]
        audio_output = {
            "waveform": trimmed_waveform.unsqueeze(0),
            "sample_rate": sample_rate,
        }

        final_duration = float(trimmed_waveform.shape[1] / sample_rate)

        return (audio_output, final_duration, seg_count)


NODE_CLASS_MAPPINGS = {
    "YuanAudioList": YuanAudioList,
    "YuanAudioSplit": YuanAudioSplit,
    "YuanAudioLoad": YuanAudioLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanAudioList": "Yuan 音频列表",
    "YuanAudioSplit": "Yuan 音频分流",
    "YuanAudioLoad": "Yuan 加载音频",
}


# 后端路由：列出输入目录中的音频文件（供前端"添加音频"面板使用）
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/yuan_tool/audio_files")
    async def _list_audio_files(request):
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        entries = os.listdir(input_dir)
        try:
            files = folder_paths.filter_files_content_types(entries, ["audio", "video"])
        except Exception:
            # filter_files_content_types 不可用时按扩展名兜底
            files = [
                f for f in entries
                if os.path.isfile(os.path.join(input_dir, f))
                and os.path.splitext(f)[1].lower() in _AUDIO_EXTS
            ]
        return web.json_response({"files": sorted(files)})
except Exception:
    # 独立测试环境（无 ComfyUI 服务端）下跳过路由注册
    pass
