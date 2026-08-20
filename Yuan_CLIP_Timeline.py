"""
Yuan CLIP Timeline - 视觉时间轴提示词编码节点（PromptRelay Timeline 复刻，
集成 LTXV 视频/音频空潜空间自动生成）。
"""

import json
import math
import re
import types
import os
import base64
import time
import io as _io
import asyncio

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# PyAV 为可选依赖（音频提取/视频解码用），缺失时降级不阻断节点加载
try:
    import av
except ImportError:
    av = None

import comfy.ldm.modules.attention
import comfy.model_management
import folder_paths
from server import PromptServer
from aiohttp import web

# 自定义 socket 类型 — 用字符串定义以保证所有 ComfyUI 版本兼容
GuideData = "YUAN_CLIP_GUIDE_DATA"
MotionGuideData = "YUAN_CLIP_MOTION_GUIDE_DATA"

# text_input 时间格式解析：匹配行首的 "0-3s", "3-5秒", "5-7s" 等
_TIME_RANGE_PATTERN = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*[s秒]\s*[：:]?\s*')

# 多媒体引导上传子目录（独立于 LTX 导演，避免冲突）
_TIMELINE_UPLOAD_SUBDIR = "yuan_clip"


# ==============================================================================
# Windows 连接重置异常抑制（避免 PyAV/上传时 ConnectionResetError 刷屏）
# ==============================================================================
try:
    _loop = None
    try:
        _loop = asyncio.get_event_loop()
    except RuntimeError:
        try:
            _loop = asyncio.get_event_loop_policy().get_event_loop()
        except Exception:
            pass
    if _loop is not None:
        _old_handler = _loop.get_exception_handler()

        def _silence_connection_reset_handler(loop, context):
            exception = context.get('exception')
            if (isinstance(exception, (ConnectionResetError, ConnectionAbortedError)) or
                    (isinstance(exception, OSError) and getattr(exception, 'winerror', None) in (10054, 10053))):
                return
            if _old_handler:
                _old_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        _loop.set_exception_handler(_silence_connection_reset_handler)
except Exception:
    pass


# ==============================================================================
# 多媒体引导：HTTP endpoint（文件去重/上传/音频提取/打开目录）
# ==============================================================================

@PromptServer.instance.routes.get("/yuan_clip_timeline_check_file")
async def _yuan_clip_timeline_check_file(request):
    filename = request.query.get("filename", "")
    file_size = request.query.get("size", "")
    if not filename:
        return web.json_response({"exists": False})

    upload_dir = folder_paths.get_input_directory()
    temp_dir = os.path.join(upload_dir, _TIMELINE_UPLOAD_SUBDIR)

    possible_paths = [os.path.join(temp_dir, filename), os.path.join(upload_dir, filename)]
    found_path = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isfile(p):
            if file_size:
                try:
                    if os.path.getsize(p) == int(file_size):
                        found_path = p
                        break
                except ValueError:
                    found_path = p
                    break
            else:
                found_path = p
                break
    if found_path:
        rel_name = os.path.relpath(found_path, upload_dir).replace('\\', '/')
        return web.json_response({"exists": True, "name": rel_name})

    # 后缀模糊匹配
    base_name = os.path.basename(filename)
    suffix = f"_{base_name}"
    try:
        for search_dir in [temp_dir, upload_dir]:
            if os.path.exists(search_dir):
                for f_name in os.listdir(search_dir):
                    if f_name.endswith(suffix) or f_name == base_name:
                        pot_path = os.path.join(search_dir, f_name)
                        if os.path.isfile(pot_path):
                            if file_size:
                                try:
                                    if os.path.getsize(pot_path) == int(file_size):
                                        rel_name = os.path.relpath(pot_path, upload_dir).replace('\\', '/')
                                        return web.json_response({"exists": True, "name": rel_name})
                                except ValueError:
                                    pass
                            else:
                                rel_name = os.path.relpath(pot_path, upload_dir).replace('\\', '/')
                                return web.json_response({"exists": True, "name": rel_name})
    except Exception:
        pass
    return web.json_response({"exists": False})


def _read_and_write_file_chunk(file, file_path, mode):
    chunk_bytes = file.file.read()
    with open(file_path, mode) as f:
        f.write(chunk_bytes)


@PromptServer.instance.routes.post("/yuan_clip_timeline_upload_chunk")
async def _yuan_clip_timeline_upload_chunk(request):
    post = await request.post()
    file = post.get("file")
    filename = post.get("filename")
    chunk_index = int(post.get("chunk_index"))
    total_chunks = int(post.get("total_chunks"))

    upload_dir = os.path.join(folder_paths.get_input_directory(), _TIMELINE_UPLOAD_SUBDIR)
    os.makedirs(upload_dir, exist_ok=True)
    filename = os.path.basename(filename)
    file_path = os.path.join(upload_dir, filename)
    if not os.path.realpath(file_path).startswith(os.path.realpath(upload_dir)):
        return web.json_response({"error": "无效的文件名"}, status=400)

    mode = "ab" if chunk_index > 0 else "wb"
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _read_and_write_file_chunk, file, file_path, mode)

    if chunk_index == total_chunks - 1:
        audio_file, peaks = None, None
        try:
            audio_file, peaks = await loop.run_in_executor(None, _extract_audio_from_video, file_path)
        except Exception:
            pass
        return web.json_response({
            "name": f"{_TIMELINE_UPLOAD_SUBDIR}/{filename}",
            "audio_file": audio_file,
            "peaks": peaks,
        })
    return web.json_response({"status": "ok"})


# ==============================================================================
# 多媒体引导：音频提取与波形峰值
# ==============================================================================

def _read_wav_peaks(wav_path):
    import wave
    peaks = []
    with wave.open(wav_path, 'rb') as w:
        n_frames = w.getnframes()
        if n_frames > 0:
            frames_bytes = w.readframes(n_frames)
            samples = np.frombuffer(frames_bytes, dtype=np.int16)
            num_peaks = 200
            step = max(1, len(samples) // num_peaks)
            for i in range(num_peaks):
                chunk = samples[i * step: (i + 1) * step]
                if len(chunk) > 0:
                    peaks.append(float(np.max(np.abs(chunk)) / 32767.0))
                else:
                    peaks.append(0.0)
        else:
            peaks = [0.0] * 200
    return peaks


def _extract_audio_from_video(video_path):
    import wave
    try:
        base, _ = os.path.splitext(video_path)
        output_wav = base + "_extracted_audio.wav"
        if os.path.exists(output_wav) and os.path.getsize(output_wav) > 44:
            try:
                with wave.open(output_wav, 'rb') as w_check:
                    if w_check.getframerate() == 44100:
                        peaks = _read_wav_peaks(output_wav)
                        input_dir = folder_paths.get_input_directory()
                        rel_output = os.path.relpath(output_wav, input_dir).replace('\\', '/')
                        return rel_output, peaks
            except Exception:
                pass

        with av.open(video_path) as container:
            if not container.streams.audio:
                return None, None
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format='s16', layout='mono', rate=44100)
            audio_bytes = bytearray()
            for frame in container.decode(stream):
                for resampled_frame in resampler.resample(frame):
                    arr = resampled_frame.to_ndarray()
                    audio_bytes.extend(arr.tobytes())
            for resampled_frame in resampler.resample(None):
                arr = resampled_frame.to_ndarray()
                audio_bytes.extend(arr.tobytes())
            if not audio_bytes:
                return None, None
            with wave.open(output_wav, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(44100)
                w.writeframes(audio_bytes)

        peaks = _read_wav_peaks(output_wav)
        input_dir = folder_paths.get_input_directory()
        rel_output = os.path.relpath(output_wav, input_dir).replace('\\', '/')
        return rel_output, peaks
    except Exception:
        return None, None


def _get_audio_peaks(audio_path):
    _, ext = os.path.splitext(audio_path)
    if ext.lower() == ".wav":
        try:
            return _read_wav_peaks(audio_path)
        except Exception:
            pass
    try:
        with av.open(audio_path) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format='s16', layout='mono', rate=8000)
            audio_bytes = bytearray()
            for frame in container.decode(stream):
                for resampled_frame in resampler.resample(frame):
                    arr = resampled_frame.to_ndarray()
                    audio_bytes.extend(arr.tobytes())
            for resampled_frame in resampler.resample(None):
                arr = resampled_frame.to_ndarray()
                audio_bytes.extend(arr.tobytes())
            if not audio_bytes:
                return None
            peaks = []
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            num_peaks = 200
            step = max(1, len(samples) // num_peaks)
            for i in range(num_peaks):
                chunk = samples[i * step: (i + 1) * step]
                if len(chunk) > 0:
                    peaks.append(float(np.max(np.abs(chunk)) / 32767.0))
                else:
                    peaks.append(0.0)
            return peaks
    except Exception:
        return None


@PromptServer.instance.routes.get("/yuan_clip_timeline_get_audio")
async def _yuan_clip_timeline_get_audio(request):
    filename = request.query.get("filename")
    if not filename:
        return web.json_response({"error": "缺少文件名"}, status=400)

    upload_dir = folder_paths.get_input_directory()
    clean_filename = filename.replace('\\', '/')
    file_path = os.path.join(upload_dir, clean_filename)
    if not os.path.exists(file_path):
        basename = os.path.basename(clean_filename)
        temp_path = os.path.join(upload_dir, _TIMELINE_UPLOAD_SUBDIR, basename)
        if os.path.exists(temp_path):
            file_path = temp_path
        else:
            file_path = os.path.join(upload_dir, basename)

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return web.json_response({"error": "文件未找到"}, status=404)

    _, ext = os.path.splitext(file_path)
    is_audio = ext.lower() in [".wav", ".mp3", ".ogg", ".flac", ".m4a"]
    if is_audio:
        peaks = None
        try:
            peaks = _get_audio_peaks(file_path)
        except Exception:
            pass
        rel_path = os.path.relpath(file_path, upload_dir).replace('\\', '/')
        return web.json_response({"audio_file": rel_path, "peaks": peaks})

    audio_file, peaks = None, None
    try:
        loop = asyncio.get_event_loop()
        audio_file, peaks = await loop.run_in_executor(None, _extract_audio_from_video, file_path)
    except Exception:
        pass
    return web.json_response({"audio_file": audio_file, "peaks": peaks})


@PromptServer.instance.routes.get("/yuan_clip_timeline_get_seg_images")
async def _yuan_clip_timeline_get_seg_images(request):
    """返回最新一次 segment_images 端口分配的图像文件列表（按时间戳分组取最新一组）。"""
    upload_dir = os.path.join(folder_paths.get_input_directory(), _TIMELINE_UPLOAD_SUBDIR)
    files = []
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            if f.startswith("segimg_") and f.endswith(".png"):
                # 解析 segimg_{ts}_{i}.png
                body = f[len("segimg_"):-len(".png")]
                parts = body.split("_")
                if len(parts) == 2:
                    try:
                        ts = int(parts[0])
                        idx = int(parts[1])
                        files.append({"file": f, "ts": ts, "index": idx})
                    except ValueError:
                        pass
    if not files:
        return web.json_response({"files": []})
    # 取最新的 ts（同一批分配的文件共享时间戳）
    max_ts = max(f["ts"] for f in files)
    latest_files = [f for f in files if f["ts"] == max_ts]
    latest_files.sort(key=lambda f: f["index"])
    return web.json_response({
        "files": [{"file": f["file"], "index": f["index"]} for f in latest_files]
    })


@PromptServer.instance.routes.post("/yuan_clip_timeline_clear_seg_images")
async def _yuan_clip_timeline_clear_seg_images(request):
    """清理磁盘上的 segimg_*.png 文件（用户点击清空时调用，避免切换工作流后自动重载）。"""
    upload_dir = os.path.join(folder_paths.get_input_directory(), _TIMELINE_UPLOAD_SUBDIR)
    removed = 0
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            if f.startswith("segimg_") and f.endswith(".png"):
                try:
                    os.remove(os.path.join(upload_dir, f))
                    removed += 1
                except Exception:
                    pass
    return web.json_response({"success": True, "removed": removed})


# ==============================================================================
# 多媒体引导：图像/视频张量加载与处理
# ==============================================================================

def _resolve_image_path(ref: str) -> str:
    """解析图像/视频引用路径，支持 output:/input:/temp: 类型前缀（无前缀默认 input 目录）。"""
    if not ref:
        return ""
    ref = ref.strip()
    # 检测前缀 "output:" / "input:" / "temp:"
    if ":" in ref and not ref[1:3] == ":\\" and not ref.startswith("\\\\"):
        prefix, _, rest = ref.partition(":")
        prefix = prefix.lower()
        if prefix == "output":
            return os.path.join(folder_paths.get_output_directory(), rest)
        if prefix == "temp":
            return os.path.join(folder_paths.get_temp_directory(), rest)
        if prefix == "input":
            return os.path.join(folder_paths.get_input_directory(), rest)
    return os.path.join(folder_paths.get_input_directory(), ref)


def _load_image_tensor(seg: dict) -> torch.Tensor:
    """加载图像为 [1,H,W,3] float32（支持类型前缀，回退 base64，失败返回 512x512 黑色零张量）。"""
    if seg.get("imageFile"):
        file_path = _resolve_image_path(seg["imageFile"])
        if file_path and os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)
    b64_str = seg.get("imageB64", "")
    if not b64_str or b64_str.startswith("/view?"):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)


def _load_video_tensor(seg: dict, frame_rate: float) -> torch.Tensor:
    """从视频文件按 trim 参数提取帧序列，返回 [N,H,W,3] float32。"""
    file_path = _resolve_image_path(seg.get("imageFile", ""))
    if not file_path or not os.path.exists(file_path):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    trim_start_frames = float(seg.get("trimStart", 0))
    length_frames = float(seg.get("length", 1))
    start_sec = trim_start_frames / frame_rate
    frames = []
    try:
        with av.open(file_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if stream.time_base:
                seek_pts = int((max(0, start_sec - 0.5)) / float(stream.time_base))
            else:
                seek_pts = int((max(0, start_sec - 0.5)) * av.time_base)
            container.seek(seek_pts, stream=stream, backward=True)
            for frame in container.decode(stream):
                frame_time = frame.time
                if frame_time is None and frame.pts is not None and stream.time_base:
                    frame_time = float(frame.pts * stream.time_base)
                if frame_time is None:
                    frame_time = 0.0
                if frame_time < start_sec - 0.01:
                    continue
                frames.append(frame.to_ndarray(format='rgb24'))
                if len(frames) >= int(length_frames):
                    break
    except Exception:
        pass
    if not frames:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    frames_np = np.array(frames, dtype=np.float32) / 255.0
    return torch.from_numpy(frames_np)


def _snap_to_divisible(val, div):
    """将值对齐到 div 的整数倍，最小为 div。"""
    return max(div, (val // div) * div)


def _resize_image(tensor: torch.Tensor, target_w: int, target_h: int, method: str, divisible_by: int) -> torch.Tensor:
    """将 [N,H,W,3] 张量缩放到目标尺寸，并对齐到 divisible_by。"""
    tw = _snap_to_divisible(target_w, divisible_by)
    th = _snap_to_divisible(target_h, divisible_by)
    N, H, W, C = tensor.shape
    if H == th and W == tw:
        return tensor
    t_nchw = tensor.permute(0, 3, 1, 2)
    if method == "stretch to fit":
        resized = F.interpolate(t_nchw, size=(th, tw), mode="bilinear", align_corners=False)
    elif method == "maintain aspect ratio":
        ratio = min(tw / W, th / H)
        new_w = _snap_to_divisible(int(W * ratio), divisible_by)
        new_h = _snap_to_divisible(int(H * ratio), divisible_by)
        resized = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
    elif method in ("pad", "pad green"):
        ratio = min(tw / W, th / H)
        new_w = _snap_to_divisible(int(W * ratio), divisible_by)
        new_h = _snap_to_divisible(int(H * ratio), divisible_by)
        inner = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_l = (tw - new_w) // 2
        pad_t = (th - new_h) // 2
        if method == "pad green":
            resized = torch.zeros((N, C, th, tw), dtype=t_nchw.dtype, device=t_nchw.device)
            resized[:, 0, :, :] = 102 / 255.0
            resized[:, 1, :, :] = 1.0
            resized[:, 2, :, :] = 0.0
            resized[:, :, pad_t:pad_t + new_h, pad_l:pad_l + new_w] = inner
        else:
            resized = F.pad(inner, (pad_l, tw - new_w - pad_l, pad_t, th - new_h - pad_t), mode="constant", value=0)
    elif method == "crop":
        ratio = max(tw / W, th / H)
        new_w = int(W * ratio)
        new_h = int(H * ratio)
        inner = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        resized = inner[:, :, top:top + th, left:left + tw]
    else:
        resized = F.interpolate(t_nchw, size=(th, tw), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1)


def _compress_image(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    """对 [N,H,W,3] 张量施加 H.264 压缩伪影。crf=0 不压缩。"""
    if crf == 0:
        return tensor
    N, H, W, C = tensor.shape
    h = (H // 2) * 2
    w = (W // 2) * 2
    tensor_bytes = (tensor[:, :h, :w, :] * 255.0).byte().cpu().numpy()
    try:
        buf = _io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=24)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "ultrafast"}
        for i in range(N):
            frame = av.VideoFrame.from_ndarray(tensor_bytes[i], format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()
        buf.seek(0)
        container_r = av.open(buf, mode="r")
        decoded = [frame_r.to_ndarray(format="rgb24") for frame_r in container_r.decode(video=0)]
        container_r.close()
        if not decoded:
            return tensor
        decoded_np = np.stack(decoded).astype(np.float32) / 255.0
        out = tensor.clone()
        dec_N = min(N, len(decoded))
        out[:dec_N, :h, :w] = torch.from_numpy(decoded_np[:dec_N]).to(tensor.device, tensor.dtype)
        return out
    except Exception:
        return tensor


def _build_combined_audio(timeline_data_str: str, start_frame: int, duration_frames: int,
                          frame_rate: float, override_audio: bool = False) -> dict:
    """解析 timeline JSON，加载/裁剪音频并按全局时间轴对齐合成。"""
    target_sr = 44100
    total_samples = max(1, int(math.ceil(duration_frames / frame_rate * target_sr)))
    empty_audio = {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32), "sample_rate": target_sr}
    if not timeline_data_str:
        return empty_audio
    try:
        data = json.loads(timeline_data_str)
        if override_audio:
            audio_segs = data.get("motionSegments", [])
        else:
            audio_segs = data.get("audioSegments", [])
    except Exception:
        return empty_audio
    if not audio_segs:
        return empty_audio

    out_waveform = torch.zeros((2, total_samples), dtype=torch.float32)
    for seg in audio_segs:
        buffer = None
        file_key = "videoFile" if override_audio else "audioFile"
        if seg.get(file_key):
            file_path = _resolve_image_path(seg[file_key])
            if file_path and not os.path.exists(file_path):
                basename = os.path.basename(seg[file_key])
                fallback_path = os.path.join(folder_paths.get_input_directory(), _TIMELINE_UPLOAD_SUBDIR, basename)
                if os.path.exists(fallback_path):
                    file_path = fallback_path
            if file_path and os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    buffer = _io.BytesIO(f.read())
        if not override_audio and not buffer and seg.get("audioB64"):
            b64 = seg.get("audioB64")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                audio_bytes = base64.b64decode(b64)
                buffer = _io.BytesIO(audio_bytes)
            except Exception:
                pass
        if not buffer:
            continue
        try:
            clip_frames = []
            with av.open(buffer) as container:
                if not container.streams.audio:
                    continue
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(format='fltp', layout='stereo', rate=target_sr)
                for frame in container.decode(stream):
                    for resampled_frame in resampler.resample(frame):
                        arr = resampled_frame.to_ndarray()
                        clip_frames.append(torch.from_numpy(arr))
                for resampled_frame in resampler.resample(None):
                    arr = resampled_frame.to_ndarray()
                    clip_frames.append(torch.from_numpy(arr))
            if not clip_frames:
                continue
            waveform = torch.cat(clip_frames, dim=1)
            trim_start_frames = float(seg.get("trimStart", 0))
            length_frames = float(seg.get("length", 1))
            start_frames = float(seg.get("start", 0))
            if start_frames + length_frames <= start_frame:
                continue
            offset = max(0, start_frame - start_frames)
            trim_start_frames += offset
            length_frames = max(1, length_frames - offset)
            start_frames = max(0, start_frames - start_frame)
            start_sample_src = int(trim_start_frames / frame_rate * target_sr)
            length_samples = int(length_frames / frame_rate * target_sr)
            end_sample_src = start_sample_src + length_samples
            if start_sample_src < 0:
                start_sample_src = 0
            if end_sample_src > waveform.shape[1]:
                end_sample_src = waveform.shape[1]
            actual_length = end_sample_src - start_sample_src
            if actual_length <= 0:
                continue
            clip_waveform = waveform[:, start_sample_src:end_sample_src]
            start_sample_dst = int(start_frames / frame_rate * target_sr)
            if start_sample_dst >= out_waveform.shape[1]:
                continue
            end_sample_dst = start_sample_dst + actual_length
            if end_sample_dst > out_waveform.shape[1]:
                actual_length = out_waveform.shape[1] - start_sample_dst
                clip_waveform = clip_waveform[:, :actual_length]
                end_sample_dst = start_sample_dst + actual_length
            if actual_length <= 0:
                continue
            out_waveform[:, start_sample_dst:end_sample_dst] += clip_waveform
        except Exception:
            continue
    return {"waveform": out_waveform.unsqueeze(0), "sample_rate": target_sr}


# ==============================================================================
# 多媒体引导：guide_data / motion_guide_data 构建
# ==============================================================================

def _build_guide_data(tdata: dict, start_frame: int, duration_frames: int, frame_rate: float,
                      guide_strength_str: str, custom_width: int, custom_height: int,
                      resize_method: str, divisible_by: int, img_compression: int) -> tuple:
    """从 timeline_data 构建图像引导数据。返回 (guide_data, derived_w, derived_h)。"""
    guide_data = {"images": [], "insert_frames": [], "strengths": [], "frame_rate": frame_rate}
    derived_w, derived_h = custom_width, custom_height
    try:
        img_segs = [
            s for s in tdata.get("segments", [])
            if s.get("type", "image") in ("image", "video")
            and (s.get("imageFile") or s.get("imageB64"))
            and int(s.get("start", 0)) < start_frame + duration_frames
            and int(s.get("start", 0)) + int(s.get("length", 1)) > start_frame
        ]
        img_segs.sort(key=lambda s: s["start"])
        strengths = []
        if guide_strength_str and guide_strength_str.strip():
            strengths = [float(x.strip()) for x in guide_strength_str.split(",") if x.strip()]
        for idx, seg in enumerate(img_segs):
            seg_start = int(seg.get("start", 0))
            offset = max(0, start_frame - seg_start)
            if seg.get("type") == "video":
                if offset > 0:
                    seg["trimStart"] = float(seg.get("trimStart", 0)) + offset
                    seg["length"] = max(1, int(seg.get("length", 1)) - offset)
                tensor = _load_video_tensor(seg, float(frame_rate))
            else:
                tensor = _load_image_tensor(seg)
            src_h, src_w = tensor.shape[1], tensor.shape[2]

            if custom_width > 0 and custom_height > 0:
                tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by)
            elif custom_width > 0:
                tgt_w = _snap_to_divisible(custom_width, divisible_by)
                tgt_h = _snap_to_divisible(int(src_h * tgt_w / src_w), divisible_by)
                tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
            elif custom_height > 0:
                tgt_h = _snap_to_divisible(custom_height, divisible_by)
                tgt_w = _snap_to_divisible(int(src_w * tgt_h / src_h), divisible_by)
                tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
            else:
                tensor = _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)

            if img_compression > 0:
                tensor = _compress_image(tensor, img_compression)
            if idx == 0:
                derived_h = tensor.shape[1]
                derived_w = tensor.shape[2]
            if seg.get("isEndFrame"):
                insert_frame = max(0, seg_start + int(seg.get("length", 1)) - 1 - start_frame)
            else:
                insert_frame = max(0, seg_start - start_frame)
            strength = strengths[idx] if idx < len(strengths) else 1.0
            guide_data["images"].append(tensor)
            guide_data["insert_frames"].append(insert_frame)
            guide_data["strengths"].append(float(strength))

    except Exception:
        pass
    return guide_data, derived_w, derived_h


def _build_motion_guide_data(tdata: dict, start_frame: int, duration_frames: int,
                             frame_rate: float, resize_method: str,
                             use_custom_motion: bool) -> dict:
    """从 timeline_data 构建运动引导数据（IC-LoRA 视频分段）。"""
    motion_guide_data = {
        "segments": [], "frame_rate": float(frame_rate),
        "duration_frames": int(duration_frames), "resize_method": resize_method,
    }
    try:
        motion_segments = tdata.get("motionSegments", []) if use_custom_motion else []
        for seg in motion_segments:
            seg_start = int(seg.get("start", 0))
            length = int(seg.get("length", 1))
            if seg_start >= start_frame + duration_frames or seg_start + length <= start_frame:
                continue
            if not seg.get("videoFile"):
                continue
            offset = max(0, start_frame - seg_start)
            new_start = max(0, seg_start - start_frame)
            clipped_len = min(length - offset, duration_frames - new_start)
            if clipped_len <= 0:
                continue
            clean = dict(seg)
            clean["start"] = new_start
            clean["length"] = clipped_len
            clean["trimStart"] = float(seg.get("trimStart", 0)) + offset
            motion_guide_data["segments"].append(clean)
    except Exception:
        pass
    return motion_guide_data


# ==============================================================================
# prompt_relay.py 核心函数
# ==============================================================================

def build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, tokens_per_frame):
    """为视频交叉注意力构建高斯惩罚矩阵 [Lq, Lk]（整数帧索引）。"""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.long) // tokens_per_frame

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames.float()[:, None] - seg["midpoint"]).abs()
        strength = seg.get("strength", 1.0)
        if seg.get("suppress"):
            cost = strength * torch.exp(-(d**2) / (2 * seg["sigma"]**2))
        else:
            cost = strength * (torch.relu(d - seg["window"]) ** 2) / (2 * seg["sigma"] ** 2)
        offset[:, local] += cost.to(offset.dtype)

    return offset


def build_temporal_cost_scaled(q_token_idx, Lq, Lk, device, dtype, latent_frames):
    """为非整数帧映射的查询构建惩罚矩阵（例如 LTXAV 音频 token）。"""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.float32) * latent_frames / Lq

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames[:, None] - seg["midpoint"]).abs()
        sigma_a = seg.get("sigma_audio", seg["sigma"])
        strength_a = seg.get("strength_audio", 1.0)
        if seg.get("suppress"):
            cost = strength_a * torch.exp(-(d**2) / (2 * sigma_a**2))
        else:
            window_a = seg.get("window_audio", seg["window"])
            cost = strength_a * (torch.relu(d - window_a) ** 2) / (2 * sigma_a ** 2)
        offset[:, local] += cost.to(offset.dtype)

    return offset


def create_mask_fn(q_token_idx, fallback_tokens_per_frame, latent_frames):
    """闭包：mask_fn(Lq, Lk, dtype, device, transformer_options) -> 附加掩码或 None。"""
    cache = {}
    max_token_idx = max(int(seg["local_token_idx"].max().item()) for seg in q_token_idx) + 1

    def mask_fn(Lq, Lk, dtype, device, transformer_options):
        if Lq == Lk:
            return None

        cond_or_uncond = transformer_options.get("cond_or_uncond", [])
        if 1 in cond_or_uncond and 0 not in cond_or_uncond:
            return None

        grid_sizes = transformer_options.get("grid_sizes", None)
        video_tpf = int(grid_sizes[1]) * int(grid_sizes[2]) if grid_sizes is not None else fallback_tokens_per_frame
        video_lq = latent_frames * video_tpf

        if Lk == video_lq or Lk < max_token_idx:
            return None

        mode = "video" if Lq == video_lq else "scaled"

        key = (Lq, Lk, mode, device)
        if key not in cache:
            if mode == "video":
                cost = build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, video_tpf)
            else:
                cost = build_temporal_cost_scaled(q_token_idx, Lq, Lk, device, dtype, latent_frames)
            cache[key] = -cost

        return cache[key].to(dtype)

    return mask_fn


def build_segments(token_ranges, segment_lengths, epsilon=1e-3,
                   marker_token_indices=None, marker_segment_refs=None, ref_tau=5.0):
    """为时间惩罚构建每段元数据。

    @图X token 级 mask 抑制 + K/V 全强度注入：
    - @图X token 活跃帧范围 = 引用它的主轨 local 段帧范围（段内全可见 cost=0，段外高斯衰减）
    - K/V 注入保持全强度（effective_alpha = ref_alpha）
    """
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448

    q_token_idx = []
    frame_cursor = 0
    seg_frame_ranges = []

    for (tok_start, tok_end), L in zip(token_ranges, segment_lengths):
        if L <= 0:
            frame_cursor += L
            seg_frame_ranges.append((frame_cursor, frame_cursor))
            continue
        seg_midpoint = (2 * frame_cursor + L) // 2
        base_window = max(L // 2 - 1, 0)
        q_token_idx.append({
            "local_token_idx": torch.arange(tok_start, tok_end),
            "midpoint": seg_midpoint,
            "window": float(max(base_window, 0)),
            "sigma": sigma,
            "strength": 1.0,
            "window_audio": float(max(base_window, 0)),
            "sigma_audio": sigma,
            "strength_audio": 1.0,
        })
        seg_frame_ranges.append((frame_cursor, frame_cursor + L))
        frame_cursor += L

    # --- @图X token 级 mask 抑制 ---
    if marker_token_indices and marker_segment_refs:
        sigma_suppress = max(1.0, ref_tau)
        for subject_num, token_indices in marker_token_indices.items():
            ref_seg_indices = marker_segment_refs.get(subject_num, [])
            if not ref_seg_indices:
                continue
            for seg_idx in ref_seg_indices:
                if seg_idx >= len(seg_frame_ranges):
                    continue
                seg_start_frame, seg_end_frame = seg_frame_ranges[seg_idx]
                L_seg = seg_end_frame - seg_start_frame
                if L_seg <= 0:
                    continue
                seg_midpoint = (seg_start_frame + seg_end_frame) // 2
                base_window = max(L_seg // 2, 1)
                q_token_idx.append({
                    "local_token_idx": torch.tensor(token_indices, dtype=torch.long),
                    "midpoint": seg_midpoint,
                    "window": float(base_window),
                    "sigma": sigma_suppress,
                    "strength": 1.0,
                    "suppress": True,
                    "window_audio": float(base_window),
                    "sigma_audio": sigma_suppress,
                    "strength_audio": 1.0,
                })

    return q_token_idx


def get_raw_tokenizer(clip):
    """从 ComfyUI CLIP 对象中提取原始 SPiece/HF 分词器。"""
    tokenizer_wrapper = clip.tokenizer
    for attr_name in dir(tokenizer_wrapper):
        if attr_name.startswith("_"):
            continue
        inner = getattr(tokenizer_wrapper, attr_name, None)
        if inner is not None and hasattr(inner, "tokenizer"):
            return inner.tokenizer

    raise RuntimeError(
        f"无法从 CLIP 对象中找到原始分词器。"
        f"已知属性: {[a for a in dir(tokenizer_wrapper) if not a.startswith('_')]}"
    )


def _flatten_token_ids(raw):
    """展平 CLIP tokenize 返回的 token ids 为一维列表。"""
    if isinstance(raw, dict):
        raw = raw.get("input_ids", [])
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().tolist()
    if raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    return list(raw or [])


def _token_count(raw_tokenizer, text):
    """计算文本的 CLIP token 数量（减去 EOS）。"""
    ids = _flatten_token_ids(raw_tokenizer(text))
    eos_adj = 1 if getattr(raw_tokenizer, "add_eos", False) else 0
    return max(0, len(ids) - eos_adj)


def _find_marker_phrase_token_indices(
    raw_tokenizer,
    prompt_text,
    markers,
    stop_markers=None,
    stop_at_punctuation=True,
    stop_at_other_marker=True,
    stop_at_equals=True,
):
    """返回 marker token 索引列表（离散）。
    stop_at_equals=True 时只定位 @图X 标记本身，不扩展到描述文本（避免 K/V 注入覆盖描述语义）。
    """
    if raw_tokenizer is None or not prompt_text or not markers:
        return []
    prompt_folded = prompt_text.casefold()
    stop_chars = set(",，.。")
    if stop_at_equals:
        stop_chars.add("=")
    ranges = []
    sorted_markers = sorted(set(m for m in markers if m), key=len, reverse=True)
    covered_char_ranges = []
    all_markers = [m for m in (stop_markers or markers) if m]

    for marker in sorted_markers:
        marker_folded = marker.casefold()
        marker_len = len(marker)
        start = 0
        while True:
            char_start = prompt_folded.find(marker_folded, start)
            if char_start < 0:
                break
            # 数字边界：@图1 不得匹配 @图10/@图11 等前缀，
            # 否则不同主体的 marker token 范围重叠，K/V 注入会互相覆盖（特征被覆盖/人物错乱）
            if char_start + marker_len < len(prompt_text) and prompt_text[char_start + marker_len].isdigit():
                start = char_start + marker_len
                continue
            if any(c_start <= char_start < c_end for c_start, c_end in covered_char_ranges):
                start = char_start + marker_len
                continue
            char_stop = char_start + marker_len
            if not stop_at_equals:
                # 仅在非 @图X 场景下扩展到后面的短语
                scan = char_stop
                while scan < len(prompt_text):
                    if stop_at_punctuation and prompt_text[scan] in stop_chars:
                        break
                    if stop_at_other_marker:
                        tail = prompt_folded[scan:]
                        if any(tail.startswith(other.casefold()) for other in all_markers if other != marker):
                            break
                    scan += 1
                char_stop = max(char_stop, scan)
            covered_char_ranges.append((char_start, char_stop))
            tok_start = _token_count(raw_tokenizer, prompt_text[:char_start])
            marker_tok_end = _token_count(raw_tokenizer, prompt_text[:char_start + marker_len])
            phrase_tok_end = _token_count(raw_tokenizer, prompt_text[:char_stop])
            tok_end = max(marker_tok_end, phrase_tok_end)
            if tok_end <= tok_start:
                tok_end = tok_start + 1
            ranges.append((tok_start, tok_end))
            start = char_start + marker_len

    indices = []
    for tok_start, tok_end in ranges:
        indices.extend(range(tok_start, tok_end))
    return sorted(set(i for i in indices if i >= 0))


def map_token_indices(raw_tokenizer, global_prompt, local_prompts):
    """对全局提示词和空格前缀的本地提示词进行分词；返回 (完整提示词, 每段本地 token 范围)。"""
    prefixed_locals = [" " + lp for lp in local_prompts]
    full_prompt = global_prompt + "".join(prefixed_locals)
    has_eos = getattr(raw_tokenizer, "add_eos", False)
    eos_adj = 1 if has_eos else 0

    prev_len = len(raw_tokenizer(global_prompt)["input_ids"]) - eos_adj
    token_ranges = []
    built = global_prompt

    for plp in prefixed_locals:
        built += plp
        cur_len = len(raw_tokenizer(built)["input_ids"]) - eos_adj
        if cur_len <= prev_len:
            raise ValueError(f"本地提示词未产生任何 token: '{plp.strip()}'")
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len

    return full_prompt, token_ranges


def _redistribute_to_total(lengths, target_total):
    """将长度列表重新分配到精确等于 target_total，使用最大余数法。"""
    if not lengths:
        return []
    total = sum(lengths)
    if total == target_total:
        return list(lengths)
    if total <= 0:
        return _distribute_evenly(len(lengths), target_total)
    exact = [L * target_total / total for L in lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1
    elif diff < 0:
        order = sorted(range(len(exact)), key=lambda i: exact[i] - int(exact[i]))
        for k in range(-diff):
            idx = order[k % len(order)]
            if result[idx] > 1:
                result[idx] -= 1
    return [max(1, L) for L in result]


def _distribute_evenly(num_segments, target_total):
    """最大余数法均分：确保总和精确等于 target_total。"""
    if num_segments <= 0 or target_total <= 0:
        return []
    base = target_total // num_segments
    remainder = target_total % num_segments
    return [max(1, base + (1 if i < remainder else 0)) for i in range(num_segments)]


def distribute_segment_lengths(num_segments, latent_frames, specified_lengths=None):
    """验证或自动分布段帧数，确保总和精确等于 latent_frames（避免段长度溢出污染参考帧 tokens 区域）。"""
    if num_segments <= 0 or latent_frames <= 0:
        return []

    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError(
                f"segment_lengths 数量 ({len(specified_lengths)}) "
                f"必须与本地提示词数量 ({num_segments}) 一致"
            )
        clipped = [max(1, min(L, latent_frames)) for L in specified_lengths]
        total = sum(clipped)
        if total != latent_frames:
            return _redistribute_to_total(clipped, latent_frames)
        return clipped

    return _distribute_evenly(num_segments, latent_frames)


# ==============================================================================
# patches.py 模型补丁函数
# ==============================================================================

def _make_masked_override(prev_override):
    """transformer_options 覆盖，将带掩码的注意力调用路由到 attention_pytorch。"""
    def override(func, *args, **kwargs):
        if kwargs.get("mask") is not None:
            return comfy.ldm.modules.attention.attention_pytorch(*args, **kwargs)
        if prev_override is not None:
            return prev_override(func, *args, **kwargs)
        return func(*args, **kwargs)
    return override


def _make_ltx_mask_wrapper(underlying, mask_fn):
    """包装 LTX 交叉注意力 forward，注入 PromptRelay 的附加掩码。"""
    def wrapped(_self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
        if context is not None:
            pr_mask = mask_fn(x.shape[1], context.shape[1], x.dtype, x.device, transformer_options)
            if pr_mask is not None:
                mask = pr_mask if mask is None else mask + pr_mask

        if mask is not None:
            prev = transformer_options.get("optimized_attention_override")
            transformer_options = {
                **transformer_options,
                "optimized_attention_override": _make_masked_override(prev),
            }

        return underlying(
            x, context=context, mask=mask, pe=pe, k_pe=k_pe,
            transformer_options=transformer_options,
        )

    return wrapped


def detect_model_type(model):
    """返回 (patch_size, temporal_stride) 用于 LTX 潜空间几何信息。"""
    diff_model = model.model.diffusion_model

    if hasattr(diff_model, "patchifier"):
        return (1, 1, 1), int(diff_model.vae_scale_factors[0])

    raise ValueError(
        f"不支持的模型类型: {type(diff_model).__name__}。"
        f"Yuan CLIP Timeline 仅支持 LTX 模型。"
    )


def apply_patches(model_clone, mask_fn):
    diffusion_model = model_clone.get_model_object("diffusion_model")
    to = model_clone.model_options["transformer_options"]
    to["promptrelay_mask_fn"] = mask_fn
    for idx, block in enumerate(diffusion_model.transformer_blocks):
        for attr in ("attn2", "audio_attn2"):
            module = getattr(block, attr, None)
            if module is None:
                continue
            key = f"diffusion_model.transformer_blocks.{idx}.{attr}.forward"
            underlying = model_clone.get_model_object(key)
            wrapper = _make_ltx_mask_wrapper(underlying, mask_fn)
            model_clone.add_object_patch(key, types.MethodType(wrapper, module))


# ==============================================================================
# nodes.py 编码函数
# ==============================================================================

def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    """使用最大余数法将像素空间段长度转换为整数潜空间长度。"""
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


# ==============================================================================
# LTXV 空潜空间自动生成（video / audio 原理一致：零张量 + type 标记，由采样器加噪去噪）
# ==============================================================================

def _auto_generate_latent(width, height, length_frames):
    """自动生成 LTXV 兼容视频空潜空间（LTXV 时间压缩: latent_t = ((length-1)//8)+1；
    零张量不含 noise_mask，采样器对整体加噪去噪生成新内容）。"""
    w = max(32, (width // 32) * 32)
    h = max(32, (height // 32) * 32)
    latent_t = ((length_frames - 1) // 8) + 1
    samples = torch.zeros(
        [1, 128, latent_t, h // 32, w // 32],
        device=comfy.model_management.intermediate_device(),
    )
    return {"samples": samples}


def _auto_generate_audio_latent(audio_vae, length_frames, frame_rate):
    """自动生成 LTXV 兼容音频空潜空间（与 video latent 使用相同帧数/帧率，保证帧对齐）。"""
    inner = getattr(audio_vae, "first_stage_model", audio_vae)
    z_channels = audio_vae.latent_channels
    audio_freq = inner.latent_frequency_bins
    num_audio_latents = inner.num_of_latents_from_frames(length_frames, float(frame_rate))

    samples = torch.zeros(
        (1, z_channels, num_audio_latents, audio_freq),
        device=comfy.model_management.intermediate_device(),
    )
    return {"samples": samples, "type": "audio"}


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon,
                  marker_tags=None, marker_segment_refs=None, ref_tau=5.0):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(
                f"Yuan CLIP Timeline: '{name}' 为 None。"
                "可能原因：工作流 JSON 保存了空值、时间轴编辑器 Web 扩展未加载、或上游节点返回了 None。"
                "请将字段设为空字符串或修复上游连接。"
            )

    locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
    if not locals_list:
        raise ValueError("至少需要一个本地提示词（使用 | 分隔）")

    patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    # 在 full_prompt 中定位所有 @图X token（global + local）
    # local 段保留了 @图X 标记（keep_marker=True），K/V 注入能直接作用于 local 段中的 @图X token
    marker_token_indices = {}
    if marker_tags:
        try:
            indices = _find_marker_phrase_token_indices(raw_tokenizer, full_prompt, marker_tags)
            if indices:
                # 按主体编号分组
                for tag in marker_tags:
                    num = int(tag[2:])
                    tag_indices = _find_marker_phrase_token_indices(raw_tokenizer, full_prompt, [tag])
                    if tag_indices:
                        marker_token_indices[num] = tag_indices
        except Exception:
            pass

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon,
                                 marker_token_indices=marker_token_indices,
                                 marker_segment_refs=marker_segment_refs,
                                 ref_tau=ref_tau)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, mask_fn)

    return patched, conditioning, effective_lengths, marker_token_indices


# ==============================================================================
# @图X 角色解析：从 global_prompt 提取 @图X=描述 定义
# ==============================================================================

_MSR_CHAR_PATTERN = re.compile(r'^@图(\d+)\s*[=：:]\s*(.+)')

# @图X=描述 行保留在 global_prompt 中，作为角色定义 + token 标记。
# K/V 视觉特征注入：@图X 在文本中的 token 位置被标记，注入对应主体参考帧的 K/V。


def _parse_msr_characters(global_prompt: str) -> tuple:
    """解析 global_prompt 中的 @图X=描述 行，返回 (char_map, char_list)。"""
    char_map = {}
    char_list = []
    if not global_prompt:
        return char_map, char_list

    for line in global_prompt.split("\n"):
        m = _MSR_CHAR_PATTERN.match(line.strip())
        if m:
            tag = f"@图{m.group(1)}"
            desc = m.group(2).strip()
            char_map[tag] = desc
            char_list.append({"tag": tag})

    return char_map, char_list


def _generate_short_alias(desc: str, max_chars: int = 10) -> str:
    """从角色描述中提取简短别名（前 max_chars 字符、标点处截断），用于后续 @图X 引用节省 CLIP token。"""
    short = desc[:max_chars]
    for sep in ['，', '、', '。', '；', '：', ',']:
        idx = short.rfind(sep)
        if idx > max_chars // 2:
            short = short[:idx]
            break
    return short.strip()


def _escape_repl(text: str) -> str:
    """转义 re.sub 替换字符串中的反斜杠，避免描述文本含 \\ 时被当作组引用。"""
    return text.replace('\\', '\\\\')


def _apply_msr_replacements(text: str, char_map: dict, force_short: bool = False,
                            seen_tags: set = None, keep_marker: bool = False) -> str:
    """将文本中的 @图X 引用替换为角色描述。

    - 按数字倒序 + 负向断言 @图X(?！[0-9]) 避免子串误匹配（@图1 不匹配 @图10）。
    - 首次出现用完整描述，后续用简短别名节省 token；force_short 时全部用别名；
      seen_tags 中已有的标签视为已出现（本函数会更新它）。
    - keep_marker=True 时保留 @图X 标记并追加描述（如 "@图1在喝奶茶"→"@图1女孩在喝奶茶"），
      使 K/V 注入能直接作用于 local 段中的 @图X token。
    """
    if not text or not char_map:
        return text
    if seen_tags is None:
        seen_tags = set()
    modified = text
    sorted_items = sorted(char_map.items(),
                         key=lambda kv: int(kv[0][2:]),
                         reverse=True)
    for tag, desc in sorted_items:
        short_alias = _generate_short_alias(desc)
        num = tag[2:]
        pattern = re.compile(rf'@图{num}(?!\d)')
        if keep_marker:
            # 保留 @图X 标记，在后面添加描述
            if force_short or tag in seen_tags:
                replacement = f"{tag}{short_alias}"
            else:
                replacement = f"{tag}{desc}"
                seen_tags.add(tag)
            modified = pattern.sub(_escape_repl(replacement), modified)
        elif force_short:
            modified = pattern.sub(_escape_repl(short_alias), modified)
        elif tag in seen_tags:
            # 该标签已在之前的文本中出现过，全部用简短别名
            modified = pattern.sub(_escape_repl(short_alias), modified)
        else:
            # 第一次出现用完整描述，后续用简短别名
            modified = pattern.sub(_escape_repl(desc), modified, count=1)
            modified = pattern.sub(_escape_repl(short_alias), modified)
            seen_tags.add(tag)
    return modified


# ==============================================================================
# 主节点类
# ==============================================================================

class YuanCLIPTimeline:
    """可视化时间轴版本 — 段和长度来自节点 UI 中的可视化编辑器。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MODEL", {"tooltip": "要补丁的扩散模型"}),
                "CLIP模型": ("CLIP", {"tooltip": "用于编码提示词的 CLIP 模型"}),
                "音频VAE": ("VAE", {"tooltip": "Audio VAE，用于生成音频潜空间。video latent 与 audio latent 使用相同的帧数/帧率，保证对齐。"}),
                "全局提示词": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "贯穿整个视频的全局提示词。用于锚定持久的角色、物体和场景上下文。"
                               "@图X=描述 格式的行作为角色定义，同时在 CLIP token 中标记 @图X 位置，"
                               "由 K/V 视觉特征注入机制把对应 motionSegments 参考帧的视觉特征注入到这些 token 的 K/V。"
                               "示例：\n场景描述\n@图1=角色描述\n@图2=另一角色描述"
                }),
                "最大帧数": ("INT", {
                    "default": 129, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "像素空间总帧数。仅用于编辑器的视觉缩放比例，实际帧数仍从潜空间读取。"
                }),
                "时间轴数据": ("STRING", {
                    "default": "",
                    "tooltip": "时间轴编辑器的 JSON 状态（自动管理，请勿手动编辑）。"
                }),
                "段落提示词": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "由时间轴编辑器自动填充。"
                }),
                "段落长度": ("STRING", {
                    "default": "",
                    "tooltip": "由时间轴编辑器自动填充（像素空间帧数）。"
                }),
                "衰减参数": ("FLOAT", {
                    "default": 1e-3, "min": 1e-6, "max": 0.99, "step": 1e-4,
                    "tooltip": "惩罚衰减参数。低于约 0.1 的值均产生锐利边界（论文默认 0.001）。"
                               "如需更柔和的过渡，尝试 0.5 或更高值。"
                }),
                "帧率": ("FLOAT", {
                    "default": 24.0, "min": 0.1, "max": 240.0, "step": 0.1,
                    "tooltip": "每秒帧数 — 仅在 时间单位 设为'seconds'时影响时间轴编辑器的显示。"
                }),
                "时间单位": (["frames", "seconds"], {
                    "default": "frames",
                    "tooltip": "以帧或秒显示标尺、段范围、长度输入和总数。内部存储始终为像素空间帧。"
                }),
                "宽度": ("INT", {
                    "default": 768, "min": 32, "max": 8192, "step": 32,
                    "tooltip": "自动生成潜空间的目标宽度（未连接 潜空间 输入时生效）。"
                }),
                "高度": ("INT", {
                    "default": 512, "min": 32, "max": 8192, "step": 32,
                    "tooltip": "自动生成潜空间的目标高度（未连接 潜空间 输入时生效）。"
                }),
            },
            "optional": {
                "潜空间": ("LATENT", {"tooltip": "潜空间视频 — 从形状读取尺寸。不连接时自动生成 LTXV 空潜空间。"}),
                "文本输入": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "按行输入的提示词文本，支持两种模式：\n1. 时间格式（如 \"0-3s 提示词A\"），按指定秒数动态分配帧长\n2. 纯文本行，自动均分到各段落\n连接上游文本输出节点（如 Yuan TXT Splitter）可批量填充。"
                }),
                "提示词锁定": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "开启：预览模式：提示词只读不可编辑。\n关闭：各段落可自由编辑，不受 文本输入 影响。"
                }),
                # --- 多媒体引导（与下游 Yuan 引导注入节点配合） ---
                "引导强度": ("STRING", {
                    "default": "",
                    "tooltip": "从时间轴编辑器自动填充（图像分段的引导强度，逗号分隔）。"
                }),
                "起始帧": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "时间轴生成的起始帧。用于引导分段在生成区间内的裁剪偏移。"
                }),
                "自定义宽度": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "图像引导缩放的目标宽度。设为 0 则跟随 宽度。"
                }),
                "自定义高度": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "图像引导缩放的目标高度。设为 0 则跟随 高度。"
                }),
                "缩放方式": (["maintain aspect ratio", "stretch to fit", "pad", "pad green", "crop"], {
                    "default": "maintain aspect ratio",
                    "tooltip": "图像引导分段缩放至目标尺寸的方式。"
                }),
                "整除数": ("INT", {
                    "default": 32, "min": 1, "max": 256, "step": 1,
                    "tooltip": "将输出图像尺寸对齐到可被该数整除（如 LTX 为 32）。"
                }),
                "图像压缩": ("INT", {
                    "default": 0, "min": 0, "max": 100, "step": 1,
                    "tooltip": "对每张引导图像应用的 H.264 CRF 压缩。0 = 不压缩，值越高伪影越多。"
                }),
                "使用自定义音频": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "开启则使用时间轴音频，关闭则从零生成音频。"
                }),
                "使用自定义运动": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "开启则使用时间轴运动引导（IC-LoRA 视频分段），关闭则忽略。"
                }),
                "覆盖音频": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用 IC-LoRA 视频的音频，而非使用音轨。"
                }),
                "段落图像": ("IMAGE", {
                    "tooltip": "段落引导图像输入。支持单张或多张图像（batch）。\n"
                               "按段落数量自动分配到对应段落：第1张→第1段、第2张→第2段……\n"
                               "多出段落数量的图像将被截取，不作参考。\n"
                               "连接后会覆盖编辑器中已上传的引导图。"
                }),
                "运动图像": ("IMAGE", {
                    "tooltip": "IC-LoRA 轨道图像/视频输入。支持单张或多张图像（batch）。\n"
                               "每张图像自动创建为一个静态段，帧数由 运动图像帧数 控制。\n"
                               "连接后会覆盖编辑器中已有的运动段。"
                }),
                "运动图像帧数": ("INT", {
                    "default": 16, "min": 8, "max": 32, "step": 8,
                    "tooltip": "每张运动图像的帧数（8/16/24/32）。"
                }),
                "音频输入": ("AUDIO", {
                    "tooltip": "音频输入端口。连接上游音频节点（如 LoadAudio）后，"
                               "音频会自动作为一条音频段添加到时间轴音频轨道，按原始时长分配帧数。\n"
                               "连接后会覆盖编辑器中已上传的音频段（仅在锁定状态下生效）。"
                }),
                # --- Ref Guidance：参考 cond 引导（在下游 Yuan 引导注入节点接管 attn2）---
                "参考强度": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "K/V 视觉特征注入强度：把 motionSegments 参考帧的视觉特征注入到对应 @图X token 的 K/V。"
                               "k[marker] = k[marker]*(1-alpha) + ref_k*alpha。0=不注入。"
                               "建议 0.3-0.5，确保主体视觉特征一致性。"
                }),
                "参考阈值": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.001,
                    "tooltip": "段内主体抑制：基于 motionSegments 帧范围，段外帧的 @图X token K/V 注入强度衰减系数。"
                               "越大越宽松（段外衰减弱），越小越严格（段外强抑制）。"
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "LATENT", GuideData, MotionGuideData, "FLOAT", "AUDIO")
    RETURN_NAMES = ("模型", "正向条件", "视频潜空间", "音频潜空间",
                     "引导数据", "运动引导数据", "帧率", "音频")
    FUNCTION = "encode_timeline"
    CATEGORY = "Yuan Tool/CLIP"

    # 调色板（与 JS 端 PALETTE 保持一致）
    _PALETTE = [
        "#4f8edc", "#e07b3a", "#5cb85c", "#d9534f", "#9b6cd6",
        "#a07060", "#e377c2", "#7f7f7f", "#c4c447", "#3fbac4",
    ]

    def encode_timeline(self, 模型, CLIP模型, 音频VAE, 全局提示词, 最大帧数, 时间轴数据,
                        段落提示词, 段落长度, 衰减参数, 帧率=24.0, 时间单位="frames",
                        宽度=768, 高度=512, 潜空间=None, 文本输入="", 提示词锁定=True,
                        引导强度="", 起始帧=0,
                        自定义宽度=0, 自定义高度=0, 缩放方式="maintain aspect ratio",
                        整除数=32, 图像压缩=0, 使用自定义音频=False,
                        使用自定义运动=True, 覆盖音频=False,
                        段落图像=None,
                        运动图像=None,
                        运动图像帧数=16,
                        音频输入=None,
                        参考强度=0.0,
                        参考阈值=5.0):
        # 中文参数名 → 英文别名（保持函数体内代码不变）
        model = 模型
        clip = CLIP模型
        audio_vae = 音频VAE
        global_prompt = 全局提示词
        max_frames = 最大帧数
        timeline_data = 时间轴数据
        local_prompts = 段落提示词
        segment_lengths = 段落长度
        epsilon = 衰减参数
        fps = 帧率
        width = 宽度
        height = 高度
        latent = 潜空间
        text_input = 文本输入
        prompt_lock = 提示词锁定
        guide_strength = 引导强度
        start_frame = 起始帧
        custom_width = 自定义宽度
        custom_height = 自定义高度
        resize_method = 缩放方式
        divisible_by = 整除数
        img_compression = 图像压缩
        use_custom_audio = 使用自定义音频
        use_custom_motion = 使用自定义运动
        override_audio = 覆盖音频
        segment_images = 段落图像
        motion_images = 运动图像
        motion_image_frames = 运动图像帧数
        audio_input = 音频输入
        ref_alpha = 参考强度
        ref_tau = 参考阈值

        # --- 处理 text_input：仅在锁定模式下按行智能分配到 local_prompts 和 timeline_data ---
        if prompt_lock and text_input and text_input.strip():
            lines_raw = text_input.split("\n")

            # 尝试按时间格式解析每行：如 "0-3s 提示词A" 或 "3-5秒 提示词B"
            parsed_time_lines = []
            non_empty_count = 0
            for line in lines_raw:
                stripped = line.strip()
                if not stripped:
                    continue
                non_empty_count += 1
                match = _TIME_RANGE_PATTERN.match(stripped)
                if match:
                    start_sec = float(match.group(1))
                    end_sec = float(match.group(2))
                    prompt_text = stripped[match.end():].strip()
                    if prompt_text:
                        parsed_time_lines.append({
                            "prompt": prompt_text,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "duration_sec": max(0.0, end_sec - start_sec),
                        })

            # 只有所有非空行都匹配时间格式时，才启用动态时长分配
            if parsed_time_lines and len(parsed_time_lines) == non_empty_count:
                # --- 动态时长分布：按时间格式分配帧数 ---
                lines_prompts = [p["prompt"] for p in parsed_time_lines]
                local_prompts = " | ".join(lines_prompts)

                # 按最大结束时间计算 max_frames，并对齐 LTXV 时间步长 (8)
                max_end_sec = max(p["end_sec"] for p in parsed_time_lines)
                raw_max = int(max_end_sec * fps) + 1
                max_frames = ((raw_max - 2) // 8 + 1) * 8 + 1

                # 将秒数转换为帧数
                frame_allocations = []
                for p in parsed_time_lines:
                    frames = max(1, round(p["duration_sec"] * fps))
                    frame_allocations.append(frames)

                total_frames = sum(frame_allocations)

                if total_frames > max_frames:
                    # 时间轴已满：从末尾段落借用空间
                    excess = total_frames - max_frames
                    frame_allocations[-1] = max(1, frame_allocations[-1] - excess)
                elif total_frames < max_frames:
                    # 末尾段落未填满：填充剩余时间段
                    leftover = max_frames - total_frames
                    frame_allocations[-1] += leftover

                # 构建 timeline_data
                new_segs = []
                for i, (prompt_text, flen) in enumerate(zip(lines_prompts, frame_allocations)):
                    new_segs.append({
                        "prompt": prompt_text,
                        "length": flen,
                        "color": self._PALETTE[i % len(self._PALETTE)],
                    })
                timeline_data = json.dumps({"segments": new_segs})

                # 同步更新 segment_lengths
                segment_lengths = ", ".join(str(s["length"]) for s in new_segs)

            else:
                # --- 无时间格式：按均分逻辑处理 ---
                lines = [line.strip() for line in lines_raw if line.strip()]
                if lines:
                    local_prompts = " | ".join(lines)

                    try:
                        td = json.loads(timeline_data) if timeline_data and timeline_data.strip() else None
                    except (json.JSONDecodeError, ValueError):
                        td = None

                    if td and isinstance(td.get("segments"), list):
                        existing_segs = td["segments"]
                        for i, line in enumerate(lines):
                            if i < len(existing_segs):
                                existing_segs[i]["prompt"] = line
                            else:
                                remaining = max_frames - sum(s["length"] for s in existing_segs)
                                new_len = max(1, remaining // (len(lines) - len(existing_segs))) if len(lines) > len(existing_segs) else 1
                                existing_segs.append({
                                    "prompt": line,
                                    "length": new_len,
                                    "color": self._PALETTE[i % len(self._PALETTE)],
                                })
                        td["segments"] = existing_segs[:len(lines)]
                        timeline_data = json.dumps(td)
                    else:
                        base_len = max(1, max_frames // len(lines))
                        new_segs = []
                        for i, line in enumerate(lines):
                            new_segs.append({
                                "prompt": line,
                                "length": base_len,
                                "color": self._PALETTE[i % len(self._PALETTE)],
                            })
                        timeline_data = json.dumps({"segments": new_segs})

                    try:
                        td_final = json.loads(timeline_data)
                        segment_lengths = ", ".join(str(s["length"]) for s in td_final.get("segments", []))
                    except (json.JSONDecodeError, ValueError, KeyError):
                        pass

        # --- 解析 timeline_data ---
        try:
            tdata = json.loads(timeline_data) if timeline_data else {}
        except Exception:
            tdata = {}

        # global_prompt 为空时从 timeline_data 全局提示词面板回填
        if not global_prompt:
            global_prompt = tdata.get("global_prompt", "")

        # --- 处理 segment_images 端口：图像 batch 按段序分配（第1张→第1段……），锁定状态下拒绝上游数据 ---
        if segment_images is not None and prompt_lock:
            try:
                segments = tdata.get("segments", [])
                if segments:
                    # 确保 tensor 为 4D [B,H,W,3]
                    if segment_images.dim() == 3:
                        segment_images = segment_images.unsqueeze(0)
                    batch_size = segment_images.shape[0]
                    # 段落数量与图像数量取最小值，多出的图像忽略
                    alloc_count = min(len(segments), batch_size)
                    upload_dir = os.path.join(folder_paths.get_input_directory(), _TIMELINE_UPLOAD_SUBDIR)
                    os.makedirs(upload_dir, exist_ok=True)
                    ts = int(time.time() * 1000)
                    for i in range(alloc_count):
                        img_tensor = segment_images[i].clamp(0, 1).cpu()
                        arr = (img_tensor.numpy() * 255.0).astype(np.uint8)
                        pil_img = Image.fromarray(arr, mode="RGB")
                        fname = f"segimg_{ts}_{i}.png"
                        fpath = os.path.join(upload_dir, fname)
                        pil_img.save(fpath, format="PNG")
                        segments[i]["imageFile"] = f"{_TIMELINE_UPLOAD_SUBDIR}/{fname}"
                        segments[i]["imageB64"] = ""  # 清空 base64，统一使用文件路径
                        segments[i]["type"] = "image"
                    # 重新序列化 timeline_data 以保持下游一致性
                    timeline_data = json.dumps(tdata)
                    # 通过 WebSocket 通知前端刷新时间轴 UI（比 executed 事件更可靠）
                    try:
                        PromptServer.instance.send_sync("yuan_clip_seg_images_updated", {
                            "files": [{"file": f"segimg_{ts}_{i}.png", "index": i} for i in range(alloc_count)],
                            "timestamp": ts,
                        })
                    except Exception:
                        pass
            except Exception:
                pass

        # --- 处理 motion_images 端口：图像/视频合并为 IC-LoRA 轨道段，锁定状态下拒绝上游数据 ---
        if motion_images is not None and prompt_lock:
            try:
                # 每张运动图像的帧数由 运动图像帧数 参数控制（8/16/24/32）
                seg_frame_len = max(8, min(32, int(motion_image_frames)))
                # 确保 tensor 为 4D [B,H,W,3]
                if motion_images.dim() == 3:
                    motion_images = motion_images.unsqueeze(0)
                batch_size = motion_images.shape[0]
                upload_dir = os.path.join(folder_paths.get_input_directory(), _TIMELINE_UPLOAD_SUBDIR)
                os.makedirs(upload_dir, exist_ok=True)
                ts = int(time.time() * 1000)
                new_motion_segs = []

                # 保存所有图像
                saved_refs = []
                for i in range(batch_size):
                    img_tensor = motion_images[i].clamp(0, 1).cpu()
                    arr = (img_tensor.numpy() * 255.0).astype(np.uint8)
                    pil_img = Image.fromarray(arr, mode="RGB")
                    fname = f"motion_seg_{ts}_{i}.png"
                    fpath = os.path.join(upload_dir, fname)
                    pil_img.save(fpath, format="PNG")
                    saved_refs.append(f"{_TIMELINE_UPLOAD_SUBDIR}/{fname}")

                # 每张图像创建一个独立的运动段，连续排列
                # 前端显示为一段段，后端 Guide 节点 B 分支会自动合并相邻静态图像段为合成视频序列处理
                # 描述字段按顺序对应 global_prompt 中的 @图X=描述 行
                char_map, char_list = _parse_msr_characters(global_prompt)
                current_start = 0
                for i, ref in enumerate(saved_refs):
                    seg_len = min(seg_frame_len, max_frames - current_start)
                    if seg_len <= 0:
                        break
                    # 描述：按顺序对应 @图X=描述；无对应角色定义时为空
                    char_desc = ""
                    if i < len(char_list):
                        tag = char_list[i].get("tag")
                        if tag and tag in char_map:
                            char_desc = char_map[tag]
                    seg = {
                        "videoFile": ref,
                        "frameFiles": [ref],
                        "start": current_start,
                        "length": seg_len,
                        "trimStart": 0.0,
                        "isStaticImage": True,
                        "fileName": f"motion_seg_{ts}_{i}.png",
                        "description": char_desc,
                        "subjectNum": i + 1,
                    }
                    new_motion_segs.append(seg)
                    current_start += seg_len

                # 替换 motionSegments（连接端口后覆盖已有运动段）
                tdata["motionSegments"] = new_motion_segs

                timeline_data = json.dumps(tdata)
                # 通过 WebSocket 通知前端刷新
                try:
                    PromptServer.instance.send_sync("yuan_clip_motion_images_updated", {
                        "files": [{"file": f"motion_seg_{ts}_{i}.png", "index": i} for i in range(batch_size)],
                        "count": batch_size,
                        "frame_len": seg_frame_len,
                    })
                except Exception:
                    pass
            except Exception:
                pass

        # --- 处理 音频输入 端口：上游 AUDIO 数据保存为音频文件并添加到音频轨道，锁定状态下拒绝上游数据 ---
        if audio_input is not None and prompt_lock:
            try:
                waveform = audio_input.get("waveform") if isinstance(audio_input, dict) else None
                sample_rate = audio_input.get("sample_rate") if isinstance(audio_input, dict) else None
                if waveform is not None and sample_rate:
                    # waveform: [B, C, N] 或 [C, N]，取第 0 batch
                    if waveform.dim() == 3:
                        wav_2d = waveform[0]
                    elif waveform.dim() == 2:
                        wav_2d = waveform
                    else:
                        wav_2d = None
                    if wav_2d is not None:
                        # 转为单声道 float32 [-1,1]，再转 int16 保存为 wav
                        if wav_2d.shape[0] > 1:
                            mono = wav_2d.mean(dim=0, keepdim=True)
                        else:
                            mono = wav_2d
                        mono_np = mono.cpu().numpy()
                        # 限制到 [-1,1] 后转 int16
                        int16_np = np.clip(mono_np, -1.0, 1.0)
                        int16_np = (int16_np * 32767.0).astype(np.int16)
                        import wave
                        upload_dir = os.path.join(folder_paths.get_input_directory(), _TIMELINE_UPLOAD_SUBDIR)
                        os.makedirs(upload_dir, exist_ok=True)
                        ts = int(time.time() * 1000)
                        fname = f"audio_in_{ts}.wav"
                        fpath = os.path.join(upload_dir, fname)
                        with wave.open(fpath, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(int(sample_rate))
                            wf.writeframes(int16_np.tobytes())
                        # 计算音频时长对应帧数（按当前帧率）
                        duration_sec = int16_np.shape[0] / float(sample_rate)
                        desired_len = max(1, int(math.ceil(duration_sec * fps)))
                        # 物理碰撞分配位置：与前端 _findFreeSlot 思路一致，从 0 开始找第一个能放下 desired_len 的空隙
                        existing_audio = tdata.get("audioSegments", [])
                        max_f = int(max_frames)
                        start_pos = 0
                        sorted_segs = sorted(existing_audio, key=lambda s: s.get("start", 0))
                        cursor = 0
                        for s in sorted_segs:
                            s_start = int(s.get("start", 0))
                            s_len = int(s.get("length", 0))
                            gap = s_start - cursor
                            if gap >= desired_len:
                                start_pos = cursor
                                break
                            cursor = max(cursor, s_start + s_len)
                        else:
                            start_pos = cursor
                        seg_len = min(desired_len, max(1, max_f - start_pos))
                        if seg_len > 0:
                            new_audio_seg = {
                                "audioFile": f"{_TIMELINE_UPLOAD_SUBDIR}/{fname}",
                                "audioB64": "",
                                "start": start_pos,
                                "length": seg_len,
                                "trimStart": 0.0,
                                "fileName": fname,
                            }
                            existing_audio.append(new_audio_seg)
                            tdata["audioSegments"] = existing_audio
                            timeline_data = json.dumps(tdata)
                            try:
                                PromptServer.instance.send_sync("yuan_clip_audio_input_updated", {
                                    "file": fname,
                                    "audioFile": f"{_TIMELINE_UPLOAD_SUBDIR}/{fname}",
                                    "start": start_pos,
                                    "length": seg_len,
                                    "trimStart": 0.0,
                                    "fileName": fname,
                                })
                            except Exception:
                                pass
            except Exception:
                pass

        # --- @图X=描述 角色解析 + marker token 定位 ---
        # K/V 注入：定位 @图X 在 full_prompt（global+local）中的位置，注入对应主体参考帧的 K/V
        marker_segment_refs = {}   # {subject_num(int): [seg_idx, ...]} 每个 @图X 被哪些 local 段引用
        marker_tags = []           # [@图1, @图2, ...] 用于在 full_prompt 中定位所有 @图X token
        if global_prompt and "@图" in global_prompt:
            char_map, char_list = _parse_msr_characters(global_prompt)
            if char_list:
                # 构建 markers 列表（用于后续在 full_prompt 中定位所有 @图X token）
                marker_tags = [item["tag"] for item in char_list]
                # 在替换前，记录每个 local 段引用了哪些 @图X（用于 token 级 mask 抑制）
                if local_prompts and marker_tags:
                    local_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
                    for seg_idx, local_text in enumerate(local_list):
                        for tag in marker_tags:
                            num = int(tag[2:])
                            # 数字边界：@图1 不匹配 @图10，避免段帧范围误绑定导致主体错位
                            if re.search(rf'@图{num}(?!\d)', local_text):
                                marker_segment_refs.setdefault(num, []).append(seg_idx)
                # 替换 local_prompts 中的 @图X 引用，保留 @图X 标记并添加描述
                # 这样 K/V 注入能直接作用于 local 段中的 @图X token，关联视觉特征与行为描述
                if local_prompts:
                    modified = _apply_msr_replacements(local_prompts, char_map, keep_marker=True)
                    if modified != local_prompts:
                        local_prompts = modified
                # 替换 motionSegments 描述中的 @图X 引用
                motion_segs = tdata.get("motionSegments", []) if tdata else []
                if motion_segs:
                    seg_changed = False
                    seen_tags = set()
                    for seg in motion_segs:
                        desc = seg.get("description", "")
                        if desc and "@图" in desc:
                            new_desc = _apply_msr_replacements(desc, char_map, seen_tags=seen_tags)
                            if new_desc != desc:
                                seg["description"] = new_desc
                                seg_changed = True
                    if seg_changed:
                        timeline_data = json.dumps(tdata)

        # --- 构建 guide_data（图像/视频引导），同时推导输出尺寸 ---
        cw = custom_width if custom_width > 0 else width
        ch = custom_height if custom_height > 0 else height
        guide_data, derived_w, derived_h = _build_guide_data(
            tdata, start_frame, max_frames, float(fps), guide_strength,
            cw, ch, resize_method, divisible_by, img_compression,
        )

        # （参考图像已通过 frameFiles 合并段统一走 IC-LoRA 视频路径）

        # --- 自动生成 LTXV 潜空间（未连接 latent 输入时；max_frames 已对齐 stride 8，直接使用） ---
        ltxv_length = max_frames
        if latent is None:
            # 优先使用引导图像推导的尺寸，否则使用 width/height
            gen_w = derived_w if derived_w > 0 else width
            gen_h = derived_h if derived_h > 0 else height
            latent = _auto_generate_latent(gen_w, gen_h, ltxv_length)

        patched, conditioning, effective_lengths, marker_token_indices = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon,
            marker_tags=marker_tags if marker_tags else None,
            marker_segment_refs=marker_segment_refs if marker_segment_refs else None,
            ref_tau=ref_tau,
        )

        # --- 音频潜空间（自动生成） ---
        audio_latent = _auto_generate_audio_latent(audio_vae, ltxv_length, fps)

        # --- 合成时间轴音频（供下游音频修复/替换使用） ---
        if use_custom_audio or override_audio:
            audio_out = _build_combined_audio(timeline_data, start_frame, ltxv_length, float(fps), override_audio=override_audio)
        else:
            total_samples = max(1, int(math.ceil(ltxv_length / float(fps) * 44100)))
            audio_out = {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32), "sample_rate": 44100}

        # --- 构建 motion_guide_data（IC-LoRA 视频分段） ---
        motion_guide_data = _build_motion_guide_data(
            tdata, start_frame, max_frames, float(fps), resize_method, use_custom_motion,
        )

        guide_data["resize_method"] = resize_method

        # --- K/V 视觉特征注入：marker_token_indices + ref_alpha ---
        # 由下游 Yuan 引导注入节点把 motionSegments 对应主体的参考帧视觉特征注入到 @图X token 的 K/V；
        # 段外注意力抑制由 build_segments 生成的 token 级 mask（promptrelay_mask_fn）负责，ref_tau 已在此生效。
        if marker_token_indices and ref_alpha > 0.0:
            guide_data["marker_token_indices"] = marker_token_indices
            guide_data["ref_alpha"] = float(ref_alpha)
        else:
            guide_data["ref_alpha"] = 0.0

        return (patched, conditioning, latent, audio_latent, guide_data, motion_guide_data,
                float(fps), audio_out)


# ==============================================================================
# 注册映射
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "YuanCLIPTimeline": YuanCLIPTimeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanCLIPTimeline": "Yuan CLIP 时间轴 (Timeline)",
}
