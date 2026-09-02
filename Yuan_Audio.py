"""Yuan 音频列表（加载/筛选音频）与 Yuan 音频分流（列表按索引分发到多端口）节点。"""

import json
import os

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


NODE_CLASS_MAPPINGS = {
    "YuanAudioList": YuanAudioList,
    "YuanAudioSplit": YuanAudioSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanAudioList": "Yuan 音频列表",
    "YuanAudioSplit": "Yuan 音频分流",
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
