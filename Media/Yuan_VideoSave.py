# -*- coding: utf-8 -*-
"""Yuan 视频保存：把图像序列编码为视频或动图，可混入音频，并生成首帧元数据图片。

本文件派生自 VideoHelperSuite 的 Video Combine 节点，按 GPL-3.0 许可证发布。
"""

from __future__ import annotations

import datetime
import glob
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from string import Template

import numpy as np
import torch
from PIL import ExifTags, Image
from PIL.PngImagePlugin import PngInfo
from aiohttp import web
from server import PromptServer

import folder_paths
from comfy.utils import ProgressBar

LOGGER = logging.getLogger("Yuan.VideoSave")
ENCODE_ARGS = ("utf-8", "backslashreplace")

VIDEO_FORMAT_FOLDER = "Yuan_video_formats"
BASE_FORMAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_formats")

# 下拉框显示的中文格式名称；内部仍使用格式文件名。
FORMAT_LABELS = {
    "8bit-png": "PNG 序列（8 位）",
    "16bit-png": "PNG 序列（16 位）",
    "h264-mp4": "MP4 视频（H.264）",
    "h265-mp4": "MP4 视频（H.265）",
    "webm": "WebM 视频（VP9）",
    "av1-webm": "WebM 视频（AV1）",
    "ffmpeg-gif": "GIF 动图（FFmpeg）",
    "gifski": "GIF 动图（Gifski 高质量）",
    "ProRes": "MOV 视频（ProRes）",
    "ffv1-mkv": "MKV 视频（FFV1 无损）",
    "nvenc_h264-mp4": "MP4 视频（NVIDIA H.264）",
    "nvenc_hevc-mp4": "MP4 视频（NVIDIA HEVC）",
    "nvenc_av1-mp4": "MP4 视频（NVIDIA AV1）",
}

# 内置格式的排列顺序，未列出的自定义格式会追加在末尾。
FORMAT_ORDER = [
    "h264-mp4",
    "h265-mp4",
    "webm",
    "av1-webm",
    "ffmpeg-gif",
    "gifski",
    "ProRes",
    "ffv1-mkv",
    "8bit-png",
    "16bit-png",
    "nvenc_h264-mp4",
    "nvenc_hevc-mp4",
    "nvenc_av1-mp4",
]

# 动图格式不走 video_formats 目录，单独在这里登记。
BUILTIN_IMAGE_FORMATS = [
    {
        "id": "image/gif",
        "label": "GIF 动图（Pillow）",
        "extension": "gif",
        "widgets": [],
    },
    {
        "id": "image/webp",
        "label": "WebP 动图（Pillow）",
        "extension": "webp",
        "widgets": [["lossless", "BOOLEAN", {"default": True}]],
    },
]

# 格式配置里的英文参数名 -> 节点上的中文控件名。
FORMAT_WIDGET_INPUTS = {
    "crf": "画质",
    "pix_fmt": "像素格式",
    "save_metadata": "写入元数据",
    "trim_to_audio": "按音频长度裁剪",
    "dither": "抖动算法",
    "profile": "ProRes 档位",
    "bitrate": "码率",
    "megabit": "码率单位",
    "input_color_depth": "输入色深",
    "level": "FFV1 级别",
    "coder": "FFV1 编码器",
    "context": "FFV1 上下文",
    "gop_size": "GOP 帧间隔",
    "slices": "切片数",
    "slicecrc": "切片校验",
    "lossless": "无损 WebP",
    "quality": "Gifski 质量",
}

_WIDGET_DEFS = None
_FORMAT_RECORDS = None


def _widget_defs() -> dict:
    """返回中文控件名 -> (类型, 配置)；像素格式的选项取自全部格式配置的并集。"""
    global _WIDGET_DEFS
    if _WIDGET_DEFS is not None:
        return _WIDGET_DEFS

    defs = {
        "画质": (
            "INT",
            {
                "default": 19,
                "min": 0,
                "max": 100,
                "step": 1,
                "tooltip": "数值越小画质越高、文件越大；每个格式的默认值通常可提供视觉无损效果。",
            },
        ),
        "像素格式": (
            _pix_fmt_options(),
            {
                "default": "yuv420p",
                "tooltip": "输出的像素格式。带 Alpha 的格式可以保留透明通道，但兼容性可能降低。",
            },
        ),
        "写入元数据": (
            "BOOLEAN",
            {"default": True, "tooltip": "把工作流与提示词写入视频文件的元数据中。"},
        ),
        "按音频长度裁剪": (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": "开启后按音频长度裁剪视频；关闭时会用静音补齐到视频长度。",
            },
        ),
        "抖动算法": (
            [
                "bayer",
                "heckbert",
                "floyd_steinberg",
                "sierra2",
                "sierra2_4a",
                "sierra3",
                "burkes",
                "atkinson",
                "none",
            ],
            {
                "default": "sierra2_4a",
                "tooltip": "GIF 调色板转换时使用的抖动算法，影响渐变与噪点表现。",
            },
        ),
        "ProRes 档位": (
            ["lt", "standard", "hq", "4444", "4444xq"],
            {
                "default": "hq",
                "tooltip": "ProRes 编码档位，档位越高画质越好、文件越大。",
            },
        ),
        "码率": (
            "INT",
            {
                "default": 10,
                "min": 1,
                "max": 999,
                "step": 1,
                "tooltip": "视频码率。数值越高画质越好、文件越大。",
            },
        ),
        "码率单位": (
            "BOOLEAN",
            {"default": True, "tooltip": "开启时码率按 Mbps 计算，关闭时按 Kbps 计算。"},
        ),
        "输入色深": (
            ["8bit", "16bit"],
            {
                "default": "8bit",
                "tooltip": "选择 8 位或 16 位输入色深。16 位画质更高，但文件更大、编码更慢。",
            },
        ),
        "FFV1 级别": (
            ["0", "1", "3"],
            {"default": "3", "tooltip": "FFV1 编码级别，影响兼容性与压缩特性。"},
        ),
        "FFV1 编码器": (
            ["0", "1", "2"],
            {"default": "1", "tooltip": "FFV1 熵编码器类型。"},
        ),
        "FFV1 上下文": (
            ["0", "1"],
            {"default": "1", "tooltip": "FFV1 上下文模型，影响压缩率与兼容性。"},
        ),
        "GOP 帧间隔": (
            "INT",
            {
                "default": 1,
                "min": 1,
                "max": 300,
                "step": 1,
                "tooltip": "关键帧间隔。数值越小越容易定位，但文件可能更大。",
            },
        ),
        "切片数": (
            ["4", "6", "9", "12", "16", "20", "24", "30"],
            {"default": "16", "tooltip": "FFV1 切片数量，可提高多核编码效率。"},
        ),
        "切片校验": (
            ["0", "1"],
            {"default": "1", "tooltip": "为每个切片写入校验信息，增强容错能力。"},
        ),
        "无损 WebP": (
            "BOOLEAN",
            {"default": True, "tooltip": "开启后使用无损 WebP，画质最好但文件较大。"},
        ),
        "Gifski 质量": (
            "INT",
            {
                "default": 90,
                "min": 1,
                "max": 100,
                "step": 1,
                "tooltip": "Gifski 的编码质量，范围 1 到 100。",
            },
        ),
    }
    _WIDGET_DEFS = defs
    return _WIDGET_DEFS


def _default_widget_value(widget):
    if len(widget) > 2 and isinstance(widget[2], dict) and "default" in widget[2]:
        return widget[2]["default"]
    if isinstance(widget[1], list):
        return widget[1][0] if widget[1] else None
    return {"BOOLEAN": False, "INT": 0, "FLOAT": 0.0, "STRING": ""}.get(widget[1])


def _format_widget_values(kwargs: dict) -> dict:
    """把节点上的中文控件值转换成格式配置使用的英文参数名。"""
    defs = _widget_defs()
    values = {}
    for key, name in FORMAT_WIDGET_INPUTS.items():
        if name in kwargs:
            values[key] = kwargs[name]
        else:
            values[key] = _default_widget_value([name, *defs[name]])
    return values


def _pix_fmt_options() -> list:
    options = []
    for record in _format_records()[0]:
        for widget in record["widgets"]:
            if widget[0] == "pix_fmt" and isinstance(widget[1], list):
                for value in widget[1]:
                    if value not in options:
                        options.append(value)
    return options or ["yuv420p"]


def _escape_metadata(key: str, value) -> str:
    value = str(value)
    value = value.replace("\\", "\\\\")
    value = value.replace(";", "\\;")
    value = value.replace("#", "\\#")
    value = value.replace("=", "\\=")
    value = value.replace("\n", "\\\n")
    return f"{key}={value}"


def _write_metadata_file(video_metadata, directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    handle, metadata_path = tempfile.mkstemp(
        prefix="yuan_video_metadata_", suffix=".txt", dir=directory
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(";FFMETADATA1\n")
            for key, value in video_metadata.items():
                encoded = json.dumps(value, ensure_ascii=False)
                stream.write(_escape_metadata(key, encoded) + "\n")
    except Exception:
        try:
            os.remove(metadata_path)
        except OSError:
            pass
        raise
    return metadata_path


def _ffmpeg_suitability(path: str) -> int:
    try:
        version = subprocess.run(
            [path, "-version"], check=True, capture_output=True
        ).stdout.decode(*ENCODE_ARGS)
    except Exception:
        return 0
    score = 0
    for keyword, weight in (
        ("libvpx", 20),
        ("264", 10),
        ("265", 3),
        ("svtav1", 5),
        ("libopus", 1),
    ):
        if keyword in version:
            score += weight
    copyright_index = version.find("2000-2")
    if copyright_index >= 0:
        year = version[copyright_index + 6 : copyright_index + 9]
        if year.isnumeric():
            score += int(year)
    return score


@lru_cache(maxsize=1)
def _ffmpeg_path():
    forced = os.environ.get("YUAN_FFMPEG_PATH")
    if forced:
        return forced

    candidates = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        candidates.append(get_ffmpeg_exe())
    except Exception:
        pass

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        candidates.append(system_ffmpeg)

    for local_name in ("ffmpeg.exe", "ffmpeg"):
        local_path = os.path.abspath(local_name)
        if os.path.isfile(local_path):
            candidates.append(local_path)

    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        LOGGER.error(
            "未找到 FFmpeg，视频格式将无法使用。请安装 imageio-ffmpeg 或把 FFmpeg 加入系统 PATH。"
        )
        return None
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=_ffmpeg_suitability)


@lru_cache(maxsize=1)
def _gifski_path():
    forced = os.environ.get("YUAN_GIFSKI")
    if forced:
        return forced
    return shutil.which("gifski")


def _ensure_format_folder() -> None:
    # 用户自定义格式放在 custom_nodes/Yuan_video_formats（与包目录同级）
    custom_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Yuan_video_formats",
    )
    custom_paths = [custom_dir] if os.path.isdir(custom_dir) else []
    entry = folder_paths.folder_names_and_paths.get(VIDEO_FORMAT_FOLDER)
    if entry is None:
        folder_paths.folder_names_and_paths[VIDEO_FORMAT_FOLDER] = (custom_paths, {".json"})
        return
    paths, extensions = entry
    if not isinstance(paths, list):
        paths = list(paths)
        folder_paths.folder_names_and_paths[VIDEO_FORMAT_FOLDER] = (paths, extensions)
    for path in custom_paths:
        if path not in paths:
            paths.append(path)
    extensions.add(".json")


def _flatten_list(values):
    result = []
    for value in values:
        if isinstance(value, list):
            result.extend(value)
        else:
            result.append(value)
    return result


def _iterate_format(video_format, for_widgets=True):
    """遍历格式定义中的控件或 ffmpeg 参数。"""

    def indirector(container, index):
        if isinstance(container[index], list) and (
            not for_widgets
            or (len(container[index]) > 1 and not isinstance(container[index][1], dict))
        ):
            value = yield container[index]
            if value is not None:
                container[index] = value
                yield

    for key in video_format:
        if key == "extra_widgets":
            if for_widgets:
                yield from video_format["extra_widgets"]
        elif key.endswith("_pass"):
            for index in range(len(video_format[key])):
                yield from indirector(video_format[key], index)
            if not for_widgets:
                video_format[key] = _flatten_list(video_format[key])
        else:
            yield from indirector(video_format, key)


def _read_format_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        LOGGER.warning("读取视频格式文件失败：%s；%s", path, exc)
        return None


def _collect_format_files() -> dict:
    _ensure_format_folder()
    files = {}

    if os.path.isdir(BASE_FORMAT_DIR):
        for entry in os.scandir(BASE_FORMAT_DIR):
            if entry.is_file() and entry.name.endswith(".json"):
                files[os.path.splitext(entry.name)[0]] = entry.path

    # 自定义目录后读取，同名格式可覆盖内置配置
    try:
        for name in folder_paths.get_filename_list(VIDEO_FORMAT_FOLDER):
            path = folder_paths.get_full_path(VIDEO_FORMAT_FOLDER, name)
            if path:
                files[os.path.splitext(os.path.basename(name))[0]] = path
    except Exception as exc:
        LOGGER.warning("读取自定义视频格式目录失败：%s", exc)

    return files


def _build_format_records():
    """返回 (全部格式记录, 显示名到格式记录的映射)。"""
    records = []
    seen_labels = set()

    files = _collect_format_files()
    ordered_stems = [stem for stem in FORMAT_ORDER if stem in files]
    ordered_stems += sorted(
        (stem for stem in files if stem not in ordered_stems), key=str.lower
    )

    for stem in ordered_stems:
        path = files[stem]
        video_format = _read_format_file(path)
        if video_format is None:
            continue
        if "gifski_pass" in video_format and _gifski_path() is None:
            LOGGER.info("未检测到 Gifski，已隐藏格式：%s", stem)
            continue

        label = FORMAT_LABELS.get(stem, stem)
        if label in seen_labels:
            label = f"{label}（{stem}）"
        seen_labels.add(label)

        try:
            widgets = [list(widget) for widget in list(_iterate_format(video_format))]
        except Exception as exc:
            LOGGER.warning("解析视频格式失败：%s；%s", stem, exc)
            continue

        records.append(
            {
                "id": f"video/{stem}",
                "label": label,
                "kind": "video",
                "extension": video_format.get("extension", "mp4"),
                "widgets": widgets,
                "path": path,
                "stem": stem,
            }
        )

    # Pillow 动图格式排在视频格式之后，下拉框优先展示常用视频格式。
    for item in BUILTIN_IMAGE_FORMATS:
        record = dict(item)
        record["kind"] = "image"
        record["widgets"] = [list(widget) for widget in item["widgets"]]
        record["path"] = None
        records.append(record)
        seen_labels.add(record["label"])

    lookup = {}
    for record in records:
        lookup[record["label"]] = record
        lookup[record["id"]] = record
        if record.get("stem"):
            lookup[record["stem"]] = record
    return records, lookup


def _format_records():
    global _FORMAT_RECORDS
    if _FORMAT_RECORDS is None:
        _FORMAT_RECORDS = _build_format_records()
    return _FORMAT_RECORDS


def get_video_formats():
    records, _ = _format_records()
    return [record["label"] for record in records]


def _resolve_format(value: str):
    _, lookup = _format_records()
    return lookup.get(value)


def _sequence_frame_files(pattern: str) -> list:
    """把图像序列的 %03d 模板展开为实际帧文件，并按帧号（而非字典序）排序。"""
    # 先转义前缀里的 glob 元字符，再把占位符换成通配符，避免文件名含 [] 时匹配失败
    marker = "\x00"
    escaped = glob.escape(pattern.replace("%03d", marker)).replace(marker, "*")
    paths = glob.glob(escaped)

    def frame_index(path):
        match = re.search(r"(\d+)\.[^.\\/]+$", path)
        return int(match.group(1)) if match else 0

    return sorted(paths, key=frame_index)


def _translated_widgets(widgets) -> list:
    """把格式控件定义中的英文参数名换成节点上的中文控件名。"""
    result = []
    seen = set()
    for widget in widgets:
        if not isinstance(widget, (list, tuple)) or not widget:
            continue
        name = FORMAT_WIDGET_INPUTS.get(widget[0], widget[0])
        if name in seen:
            continue
        seen.add(name)
        entry = [name]
        if len(widget) > 1:
            entry.append(widget[1])
        if len(widget) > 2:
            entry.append(widget[2])
        result.append(entry)
    return result


def _format_preview_data():
    records, _ = _format_records()
    formats = {}
    for record in records:
        formats[record["label"]] = {
            "kind": record["kind"],
            "extension": record["extension"],
            "widgets": _translated_widgets(record["widgets"]),
        }
    default = next(
        (record["label"] for record in records if "H.264" in record["label"]),
        records[0]["label"] if records else "",
    )
    return {
        "default": default,
        "formats": formats,
        "widgets": list(FORMAT_WIDGET_INPUTS.values()),
    }


@PromptServer.instance.routes.get("/yuan_tool/video_save_formats")
async def yuan_video_save_formats(_request):
    """前端按当前输出格式显隐高级参数时读取的格式定义。"""
    return web.json_response(_format_preview_data())


def _as_bool(value, default=False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(value)


def apply_format_widgets(record, kwargs):
    video_format = _read_format_file(record["path"])
    if video_format is None:
        raise RuntimeError(f"无法读取输出格式配置：{record['label']}")

    for widget in _iterate_format(video_format):
        name = widget[0]
        if name not in kwargs:
            default = _default_widget_value(widget)
            kwargs[name] = default
            LOGGER.warning("输出格式参数“%s”缺失，已使用默认值：%s", name, default)

    iterator = _iterate_format(video_format, False)
    for value in iterator:
        while isinstance(value, list):
            if len(value) == 1:
                value = [Template(item).substitute(**kwargs) for item in value[0]]
                break
            if isinstance(value[1], dict):
                value = value[1][str(kwargs[value[0]])]
            elif len(value) > 3:
                value = Template(value[3]).substitute(val=kwargs[value[0]])
            else:
                value = str(kwargs[value[0]])
        iterator.send(value)
    return video_format


def _tensor_to_int(tensor, bits: int):
    array = tensor.detach().cpu().numpy()
    if bits == 16 and array.dtype == np.uint8:
        array = array.astype(np.uint16) * 257
    elif array.dtype not in (np.uint8, np.uint16):
        array = array.astype(np.float32)
        array = array * (2**bits - 1) + 0.5
        array = np.clip(array, 0, 2**bits - 1)
    return array


def tensor_to_bytes(tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"图像张量维度不正确，预期为 H×W×C，实际为 {tuple(array.shape)}。")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] not in (3, 4):
        raise ValueError(f"图像通道数不正确，仅支持 3 或 4 通道，实际为 {array.shape[-1]}。")
    if array.dtype == np.uint16:
        array = np.clip(array.astype(np.float32) / 257.0 + 0.5, 0, 255)
    elif array.dtype != np.uint8:
        array = np.clip(array.astype(np.float32) * 255.0 + 0.5, 0, 255)
    return array.astype(np.uint8)


def tensor_to_shorts(tensor) -> np.ndarray:
    array = _tensor_to_int(tensor, 16)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.clip(array, 0, 65535).astype(np.uint16)


def to_pingpong(frames):
    if not hasattr(frames, "__getitem__"):
        frames = list(frames)
    yield from frames
    for index in range(len(frames) - 2, 0, -1):
        yield frames[index]


def _merge_filter_args(args, filter_type="-vf"):
    try:
        start_index = args.index(filter_type) + 1
        index = start_index
        while True:
            index = args.index(filter_type, index)
            args[start_index] += "," + args[index + 1]
            args.pop(index)
            args.pop(index)
    except ValueError:
        pass


def ffmpeg_process(frame_data, args, video_format, video_metadata, file_path, env, pbar=None):
    """把原始视频帧写入 FFmpeg，必要时先写入工作流元数据。"""
    metadata_path = None
    final_args = list(args)
    save_metadata = video_format.get("save_metadata", False)
    if isinstance(save_metadata, str):
        save_metadata = save_metadata != "False"

    if save_metadata and video_metadata:
        try:
            metadata_path = _write_metadata_file(
                video_metadata, folder_paths.get_temp_directory()
            )
            metadata_args = ["-metadata", "creation_time=now"]
            if str(video_format.get("extension", "")).lower() in (
                "mp4",
                "mov",
                "m4v",
            ):
                metadata_args += ["-movflags", "use_metadata_tags"]
            final_args = (
                final_args[:1]
                + ["-i", metadata_path]
                + final_args[1:]
                + metadata_args
            )
        except Exception as exc:
            LOGGER.warning("写入视频元数据失败，将跳过元数据：%s", exc)
            metadata_path = None

    total_frames = 0
    process = None
    stderr = b""
    write_error = None
    existed = os.path.exists(file_path)
    try:
        process = subprocess.Popen(
            final_args + [file_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        for data in frame_data:
            try:
                process.stdin.write(data)
            except (BrokenPipeError, OSError) as exc:
                write_error = exc
                break
            total_frames += 1
            if pbar is not None:
                pbar.update(1)
        try:
            process.stdin.close()
        except Exception:
            pass
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if metadata_path and os.path.exists(metadata_path):
            try:
                os.remove(metadata_path)
            except OSError:
                pass

    error_text = stderr.decode(*ENCODE_ARGS) if stderr else ""
    if return_code != 0 or write_error is not None:
        # 仅清理本次新建的半成品文件
        if not existed and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        raise RuntimeError(
            "使用 FFmpeg 保存视频失败。\n"
            f"输出文件：{file_path}\n"
            f"返回码：{return_code}\n"
            f"FFmpeg 信息：\n{error_text[-4000:]}"
        )
    if error_text.strip():
        LOGGER.info("FFmpeg 输出信息：%s", error_text.strip()[-2000:])
    return total_frames


def gifski_process(frame_data, args, dimensions, frame_rate, video_format, file_path, env, pbar=None):
    gifski = _gifski_path()
    if gifski is None:
        raise RuntimeError("未找到 Gifski，无法使用该 GIF 输出格式。")

    gifski_args = (
        [gifski]
        + list(video_format.get("gifski_pass", []))
        + ["-W", str(int(dimensions[0])), "-H", str(int(dimensions[1]))]
        + ["-r", str(frame_rate)]
        + ["-q", "-o", file_path, "-"]
    )
    ffmpeg_args = (
        list(args)
        + list(video_format.get("main_pass", []))
        + ["-f", "yuv4mpegpipe", "-"]
    )
    _merge_filter_args(ffmpeg_args)

    total_frames = 0
    ffmpeg_stderr = b""
    gifski_stderr = b""
    existed = os.path.exists(file_path)
    with subprocess.Popen(
        ffmpeg_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    ) as ffmpeg_proc:
        with subprocess.Popen(
            gifski_args,
            stdin=ffmpeg_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        ) as gifski_proc:
            if ffmpeg_proc.stdout is not None:
                ffmpeg_proc.stdout.close()
            write_error = None
            try:
                for data in frame_data:
                    try:
                        ffmpeg_proc.stdin.write(data)
                    except (BrokenPipeError, OSError) as exc:
                        write_error = exc
                        break
                    total_frames += 1
                    if pbar is not None:
                        pbar.update(1)
                try:
                    ffmpeg_proc.stdin.close()
                except Exception:
                    pass
                ffmpeg_stderr = (
                    ffmpeg_proc.stderr.read() if ffmpeg_proc.stderr is not None else b""
                )
                gifski_stderr = (
                    gifski_proc.stderr.read() if gifski_proc.stderr is not None else b""
                )
                if gifski_proc.stdout is not None:
                    gifski_proc.stdout.read()
                ffmpeg_code = ffmpeg_proc.wait()
                gifski_code = gifski_proc.wait()
            finally:
                for process in (ffmpeg_proc, gifski_proc):
                    if process.poll() is None:
                        process.kill()
                        process.wait()

    ffmpeg_text = ffmpeg_stderr.decode(*ENCODE_ARGS) if ffmpeg_stderr else ""
    gifski_text = gifski_stderr.decode(*ENCODE_ARGS) if gifski_stderr else ""
    if ffmpeg_code != 0 or gifski_code != 0 or write_error is not None:
        if not existed and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        raise RuntimeError(
            "使用 FFmpeg + Gifski 生成 GIF 失败。\n"
            f"FFmpeg 返回码：{ffmpeg_code}\n"
            f"Gifski 返回码：{gifski_code}\n"
            f"FFmpeg 信息：\n{ffmpeg_text[-3000:]}\n"
            f"Gifski 信息：\n{gifski_text[-3000:]}"
        )
    return total_frames


class YuanVideoSave:
    DESCRIPTION = (
        "把图像序列保存为视频或动图，可把音频混入视频。\n"
        "支持 MP4、WebM、GIF、WebP、PNG 序列、ProRes、FFV1 等输出格式。\n"
        "帧率、文件名前缀与保存位置常显，循环、格式与编码参数可通过「展开高级设置」查看。\n"
        "「图像」端口为可选：未接入时本节点直接跳过，不生成文件也不报错。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        formats = get_video_formats()
        default_format = next(
            (name for name in formats if "H.264" in name), formats[0] if formats else ""
        )
        required = {
            "帧率": (
                "FLOAT",
                {
                    "default": 24,
                    "min": 1,
                    "max": 240,
                    "step": 1,
                    "tooltip": "输出视频或动图的帧率。包含音频时请与音频时长匹配，避免音画不同步。",
                },
            ),
            "文件名前缀": (
                "STRING",
                {
                    "default": "Yuan Video/ComfyUI",
                    "tooltip": "输出文件名前缀，可包含子目录。",
                },
            ),
            "保存到输出目录": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "开启时保存到 output 目录；关闭时保存到 temp 临时目录。",
                },
            ),
            "循环次数": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "视频额外重复播放的次数；0 表示不循环。GIF/WebP 中 0 表示无限循环。",
                },
            ),
            "输出格式": (
                formats,
                {
                    "default": default_format,
                    "tooltip": "选择输出容器与编码格式。切换格式后，下方会出现该格式专用的参数。",
                },
            ),
            "往返播放": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "先正序播放，再倒序播放，形成来回循环效果。",
                },
            ),
        }
        for name, (widget_type, config) in _widget_defs().items():
            required[name] = (widget_type, config)
        return {
            "required": required,
            "optional": {
                "图像": (
                    "IMAGE",
                    {
                        "tooltip": "可选。要保存为视频或动图的图像序列；未连接时不生成文件，也不会报错。"
                    },
                ),
                "音频": (
                    "AUDIO",
                    {"tooltip": "可选。连接到视频时会把音频混入最终文件。"},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文件列表",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    CATEGORY = "Yuan Tool/视频"
    FUNCTION = "save_video"

    def save_video(
        self,
        图像=None,
        帧率: float = 24,
        文件名前缀: str = "Yuan Video/ComfyUI",
        保存到输出目录: bool = True,
        循环次数: int = 0,
        输出格式: str = "MP4 视频（H.264）",
        往返播放: bool = False,
        音频=None,
        prompt=None,
        extra_pnginfo=None,
        **kwargs,
    ):
        empty_result = {"ui": {"yuan_video": []}, "result": ([""],)}
        if 图像 is None:
            return empty_result

        if torch.is_tensor(图像):
            if 图像.ndim == 4:
                frames = 图像
            elif 图像.ndim == 3:
                frames = 图像.unsqueeze(0)
            else:
                raise ValueError(
                    f"输入“图像”必须是 4 维张量（帧数×高×宽×通道），实际为 {tuple(图像.shape)}。"
                )
            num_frames = int(frames.shape[0])
        elif isinstance(图像, (list, tuple)):
            frames = list(图像)
            num_frames = len(frames)
        else:
            try:
                frames = list(图像)
                num_frames = len(frames)
            except TypeError as exc:
                raise TypeError("输入“图像”必须是图像张量或图像列表。") from exc

        if num_frames == 0:
            return empty_result

        try:
            帧率 = float(帧率)
        except (TypeError, ValueError) as exc:
            raise ValueError("帧率必须是大于 0 的有限数字。") from exc
        if not math.isfinite(帧率) or 帧率 <= 0:
            raise ValueError("帧率必须是大于 0 的有限数字。")
        循环次数 = max(0, min(100, int(循环次数)))
        保存到输出目录 = _as_bool(保存到输出目录, True)

        record = _resolve_format(输出格式)
        if record is None:
            available = "、".join(item["label"] for item in _format_records()[0])
            raise ValueError(f"未知的输出格式“{输出格式}”。可用格式：{available}")

        文件名前缀 = str(文件名前缀 or "Yuan Video/ComfyUI").strip() or "Yuan Video/ComfyUI"
        first_image = frames[0]
        if not torch.is_tensor(first_image):
            first_image = torch.as_tensor(first_image)
        if first_image.ndim == 4 and first_image.shape[0] == 1:
            first_image = first_image[0]
        if first_image.ndim != 3:
            raise ValueError(
                f"单帧图像维度不正确，预期为 高×宽×通道，实际为 {tuple(first_image.shape)}。"
            )
        image_height, image_width = int(first_image.shape[0]), int(first_image.shape[1])
        if image_width <= 0 or image_height <= 0:
            raise ValueError("图像尺寸必须大于 0。")
        has_alpha = int(first_image.shape[-1]) == 4

        pbar = ProgressBar(num_frames)
        output_dir = (
            folder_paths.get_output_directory()
            if 保存到输出目录
            else folder_paths.get_temp_directory()
        )
        full_output_folder, filename, _counter, subfolder, _filename = (
            folder_paths.get_save_image_path(文件名前缀, output_dir)
        )
        output_files = []
        media_files = []
        intermediate_files = []

        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
            video_metadata["prompt"] = prompt
        if extra_pnginfo:
            for key, value in extra_pnginfo.items():
                metadata.add_text(key, json.dumps(value, ensure_ascii=False))
                video_metadata[key] = value
        extra_options = {}
        if isinstance(extra_pnginfo, dict):
            workflow = extra_pnginfo.get("workflow")
            if isinstance(workflow, dict):
                extra_options = workflow.get("extra", {}) or {}
        metadata.add_text(
            "CreationTime", datetime.datetime.now().isoformat(" ")[:19]
        )

        max_counter = 0
        try:
            existing_files = os.listdir(full_output_folder)
        except OSError:
            existing_files = []
        matcher = re.compile(rf"{re.escape(filename)}_(\d+).*\.[^.]+$", re.IGNORECASE)
        for existing_file in existing_files:
            match = matcher.fullmatch(existing_file)
            if match:
                max_counter = max(max_counter, int(match.group(1)))
        counter = max_counter + 1

        metadata_image_enabled = _as_bool(
            extra_options.get("Yuan_MetadataImage", True),
            True,
        )
        first_image_file = f"{filename}_{counter:05}.png"
        first_image_path = os.path.join(full_output_folder, first_image_file)
        if metadata_image_enabled:
            Image.fromarray(tensor_to_bytes(first_image)).save(
                first_image_path, pnginfo=metadata, compress_level=4
            )
            output_files.append(first_image_path)

        preview_format = record["id"]
        preview_width, preview_height = image_width, image_height
        file = None

        if isinstance(音频, (list, tuple)):
            音频 = 音频[0] if 音频 else None

        if record["kind"] == "image":
            if 音频 is not None:
                LOGGER.warning("动图输出格式不支持音频，已忽略“音频”输入。")
            image_kwargs = {}
            if record["extension"] == "gif":
                image_kwargs["disposal"] = 2
            if record["extension"] == "webp":
                exif = Image.Exif()
                try:
                    exif[ExifTags.IFD.Exif] = {
                        36867: datetime.datetime.now().isoformat(" ")[:19]
                    }
                except Exception:
                    pass
                image_kwargs["exif"] = exif
                image_kwargs["lossless"] = _as_bool(kwargs.get("无损 WebP", True), True)

            file = f"{filename}_{counter:05}.{record['extension']}"
            file_path = os.path.join(full_output_folder, file)

            def image_frames(update_progress=True):
                for image in frames:
                    if update_progress:
                        pbar.update(1)
                    yield Image.fromarray(tensor_to_bytes(image))

            if 往返播放 and num_frames > 1:
                pil_frames = list(image_frames(False))
                pbar.total = max(1, len(pil_frames) * 2 - 2)

                def pingpong_images():
                    for image in to_pingpong(pil_frames):
                        pbar.update(1)
                        yield image

                image_iterator = pingpong_images()
            else:
                image_iterator = image_frames()
            first_pil = next(image_iterator)
            first_pil.save(
                file_path,
                format=record["extension"].upper(),
                save_all=True,
                append_images=image_iterator,
                duration=round(1000 / 帧率),
                loop=循环次数,
                compress_level=4,
                **image_kwargs,
            )
            output_files.append(file_path)
            media_files.append(file_path)

        else:
            ffmpeg = _ffmpeg_path()
            if ffmpeg is None:
                raise RuntimeError(
                    "保存视频需要 FFmpeg，但没有找到可用的 FFmpeg。\n"
                    "请安装 imageio-ffmpeg，或安装 FFmpeg 并加入系统 PATH，"
                    "也可以在环境变量 YUAN_FFMPEG_PATH 中指定 ffmpeg.exe 的完整路径。"
                )

            values = _format_widget_values(kwargs)
            values["has_alpha"] = has_alpha
            video_format = apply_format_widgets(record, values)
            dim_alignment = int(video_format.get("dim_alignment", 2) or 2)
            pad_x = -image_width % dim_alignment
            pad_y = -image_height % dim_alignment
            if pad_x or pad_y:
                padding = (
                    pad_x // 2,
                    pad_x - pad_x // 2,
                    pad_y // 2,
                    pad_y - pad_y // 2,
                )
                pad_function = torch.nn.ReplicationPad2d(padding)

                def pad_frame(image):
                    image = image.permute((2, 0, 1))
                    padded = pad_function(image.to(dtype=torch.float32))
                    return padded.permute((1, 2, 0))

                frames = map(pad_frame, frames)
                dimensions = (image_width + pad_x, image_height + pad_y)
                LOGGER.warning("图像尺寸不符合编码器要求，已自动补边到 %s×%s。", *dimensions)
            else:
                dimensions = (image_width, image_height)
            preview_width, preview_height = dimensions

            if 往返播放:
                if not hasattr(frames, "__getitem__"):
                    frames = list(frames)
                frames = to_pingpong(frames)
                if num_frames > 2:
                    num_frames += num_frames - 2
                    pbar.total = num_frames

            # 循环次数靠 FFmpeg 的 loop 滤镜；-filter_complex 与 -vf 互斥，
            # 复杂滤镜格式（如调色板 GIF）只能把 loop 插到首条链的输入标签之后
            loop_args = ["-vf", f"loop=loop={循环次数}:size={num_frames}"] if 循环次数 > 0 else []
            main_pass = video_format.get("main_pass") or []
            at = next((i for i, a in enumerate(main_pass) if a == "-filter_complex"), -1)
            if loop_args and at >= 0 and isinstance(main_pass[at + 1], str):
                graph = main_pass[at + 1]
                head = re.match(r"\s*\[[^\]]*\]", graph)
                if not head:
                    raise RuntimeError(
                        f"输出格式「{record['label']}」的滤镜链不以 [0:v] 之类标签开头，"
                        "无法套用「循环次数」；请把「循环次数」设为 0。"
                    )
                chain = f"loop=loop={循环次数}:size={num_frames}"
                main_pass[at + 1] = f"{graph[:head.end()]} {chain},{graph[head.end():]}"
                loop_args = []

            if video_format.get("input_color_depth", "8bit") == "16bit":
                frames = map(tensor_to_shorts, frames)
                input_pixel_format = "rgba64" if has_alpha else "rgb48"
            else:
                frames = map(tensor_to_bytes, frames)
                input_pixel_format = "rgba" if has_alpha else "rgb24"

            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)

            bitrate_arg = []
            bitrate = video_format.get("bitrate")
            if bitrate is not None:
                suffix = "M" if video_format.get("megabit") == "True" else "K"
                bitrate_arg = ["-b:v", f"{bitrate}{suffix}"]

            args = [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                input_pixel_format,
                "-color_range",
                "pc",
                "-colorspace",
                "rgb",
                "-color_primaries",
                "bt709",
                "-color_trc",
                video_format.get("fake_trc", "iec61966-2-1"),
                "-s",
                f"{dimensions[0]}x{dimensions[1]}",
                "-r",
                str(帧率),
                "-i",
                "-",
            ] + loop_args

            frame_bytes = map(lambda value: value.tobytes(), frames)
            env = os.environ.copy()
            if "environment" in video_format:
                env.update(video_format["environment"])

            if "pre_pass" in video_format:
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                joined_frames = b"".join(frame_bytes)
                input_index = args.index("-i") + 2
                pre_pass_args = args[:input_index] + list(video_format["pre_pass"])
                _merge_filter_args(pre_pass_args)
                try:
                    pre_pass_result = subprocess.run(
                        pre_pass_args,
                        input=joined_frames,
                        env=env,
                        capture_output=True,
                        check=True,
                    )
                    if pre_pass_result.stderr:
                        LOGGER.info(
                            "预编码信息：%s",
                            pre_pass_result.stderr.decode(*ENCODE_ARGS)[-2000:],
                        )
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        "FFmpeg 预编码失败：\n"
                        + exc.stderr.decode(*ENCODE_ARGS)[-4000:]
                    ) from exc
                frame_bytes = iter([joined_frames])

            if "inputs_main_pass" in video_format:
                input_index = args.index("-i") + 2
                args = (
                    args[:input_index]
                    + list(video_format["inputs_main_pass"])
                    + args[input_index:]
                )

            sequence_before = None
            if "%03d" in str(video_format.get("extension", "")):
                sequence_before = set(_sequence_frame_files(file_path))
            try:
                if "gifski_pass" in video_format:
                    total_frames_output = gifski_process(
                        frame_bytes,
                        args,
                        dimensions,
                        帧率,
                        video_format,
                        file_path,
                        env,
                        pbar,
                    )
                    preview_format = "image/gif"
                    音频 = None
                else:
                    args += list(video_format.get("main_pass", [])) + bitrate_arg
                    _merge_filter_args(args)
                    total_frames_output = ffmpeg_process(
                        frame_bytes,
                        args,
                        video_format,
                        video_metadata,
                        file_path,
                        env,
                        pbar,
                    )
            except Exception:
                # 图像序列失败时清掉本次新写出的半成品帧
                if sequence_before is not None:
                    for frame_path in _sequence_frame_files(file_path):
                        if frame_path in sequence_before:
                            continue
                        try:
                            os.remove(frame_path)
                        except OSError:
                            pass
                raise
            output_files.append(file_path)
            if "%03d" in str(video_format.get("extension", "")):
                # 图像序列没有单一文件，输出列表展开为实际帧文件
                media_files.extend(_sequence_frame_files(file_path) or [file_path])
            else:
                media_files.append(file_path)

            if "%03d" in str(video_format.get("extension", "")) and 音频 is not None:
                LOGGER.warning("图像序列格式无法混入音频，已忽略“音频”输入。")
                音频 = None

            audio_waveform = None
            a_waveform = None
            if isinstance(音频, dict):
                a_waveform = 音频.get("waveform")
            if torch.is_tensor(a_waveform) and a_waveform.numel() > 0:
                audio_waveform = a_waveform

            if audio_waveform is not None:
                output_file_with_audio = (
                    f"{filename}_{counter:05}-含音频.{video_format['extension']}"
                )
                output_file_with_audio_path = os.path.join(
                    full_output_folder, output_file_with_audio
                )
                audio_pass = video_format.get("audio_pass")
                if not audio_pass:
                    LOGGER.warning("所选视频格式没有专用音频编码器，已改用默认音频编码。")
                    audio_pass = (
                        ["-c:a", "aac"]
                        if video_format["extension"] in ("mp4", "mov")
                        else ["-c:a", "libopus"]
                    )

                if audio_waveform.ndim == 3:
                    audio_waveform = audio_waveform[0]
                if audio_waveform.ndim != 2:
                    raise ValueError("音频波形维度不正确，预期为 声道×采样点。")
                channels = int(audio_waveform.shape[0])
                sample_rate = int(音频.get("sample_rate", 44100) or 44100)
                if sample_rate <= 0:
                    sample_rate = 44100
                # FFmpeg 的 loop 滤镜会把输入重复 循环次数 次，音频也要按最终时长补齐
                min_audio_duration = total_frames_output * (循环次数 + 1) / 帧率 + 1
                if video_format.get("trim_to_audio", "False") != "False":
                    apad = []
                else:
                    apad = ["-af", f"apad=whole_dur={min_audio_duration}"]

                mux_args = [
                    ffmpeg,
                    "-v",
                    "error",
                    "-n",
                    "-i",
                    file_path,
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    str(channels),
                    "-f",
                    "f32le",
                    "-i",
                    "-",
                    "-c:v",
                    "copy",
                    "-map_metadata",
                    "0",
                ] + list(audio_pass) + apad + (
                    ["-movflags", "use_metadata_tags"]
                    if video_format["extension"] in ("mp4", "mov", "m4v")
                    else []
                ) + [
                    "-shortest",
                    output_file_with_audio_path,
                ]
                _merge_filter_args(mux_args, "-af")
                audio_data = (
                    audio_waveform.transpose(0, 1)
                    .detach()
                    .cpu()
                    .float()
                    .contiguous()
                    .numpy()
                    .tobytes()
                )
                mux_existed = os.path.exists(output_file_with_audio_path)
                try:
                    mux_result = subprocess.run(
                        mux_args,
                        input=audio_data,
                        env=env,
                        capture_output=True,
                        check=True,
                    )
                    if mux_result.stderr:
                        LOGGER.info(
                            "音频混流信息：%s",
                            mux_result.stderr.decode(*ENCODE_ARGS)[-2000:],
                        )
                except subprocess.CalledProcessError as exc:
                    if not mux_existed and os.path.exists(output_file_with_audio_path):
                        try:
                            os.remove(output_file_with_audio_path)
                        except OSError:
                            pass
                    raise RuntimeError(
                        "音频混流失败：\n"
                        + exc.stderr.decode(*ENCODE_ARGS)[-4000:]
                    ) from exc
                output_files.append(output_file_with_audio_path)
                media_files.append(output_file_with_audio_path)
                intermediate_files.append(file_path)
                file = output_file_with_audio

        keep_intermediate = _as_bool(
            extra_options.get("Yuan_KeepIntermediate", True),
            True,
        )
        if not keep_intermediate:
            for intermediate in intermediate_files:
                if os.path.exists(intermediate):
                    os.remove(intermediate)
            media_files = [path for path in media_files if path not in intermediate_files]

        preview_name = file.replace("%03d", "001") if "%03d" in file else file
        preview_path = os.path.join(full_output_folder, preview_name)
        if not os.path.exists(preview_path):
            candidates = [path for path in media_files if os.path.exists(path)]
            preview_path = candidates[0] if candidates else output_files[-1]
            preview_name = os.path.basename(preview_path)
        preview = {
            "filename": preview_name,
            "subfolder": subfolder,
            "type": "output" if 保存到输出目录 else "temp",
            "format": preview_format,
            "format_label": record["label"],
            "frame_rate": 帧率,
            "width": int(preview_width),
            "height": int(preview_height),
            "fullpath": os.path.abspath(preview_path),
        }
        if metadata_image_enabled and os.path.exists(first_image_path):
            preview["workflow"] = first_image_file

        return {
            "ui": {"yuan_video": [preview]},
            "result": (media_files or [""],),
        }


NODE_CLASS_MAPPINGS = {
    "Yuan_VideoSave": YuanVideoSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_VideoSave": "Yuan 视频保存",
}
