"""Yuan Tool · Yuan 加载视频 UI 节点

复刻自 Yuan-TV 插件的「加载视频 UI」节点（类/路由/前端扩展名均做隔离，
可与源插件共存）。支持从 input 目录或本地路径加载视频，提供内置预览、
时间轴裁剪、镜头智能分段与裁剪框交互 UI。

节点分类: "Yuan Tool/视频"
"""

import os
import gc
import asyncio
import json
import math
import threading
import tempfile
import torch
import numpy as np
import folder_paths
import av
from server import PromptServer
from aiohttp import web
import comfy.utils


# ====================================================================
# 内存诊断辅助：记录当前 Python 进程 RSS，用于定位"切换分段内存叠加"来源
# ====================================================================
def _log_ram(tag=""):
    try:
        import psutil
        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        print(f"[YuanVideoUI] {tag} 进程RSS={rss_mb:.0f}MB")
    except Exception:
        pass

# 自定义 API 路由：从系统任意路径读取视频文件，供前端预览
@PromptServer.instance.routes.get("/yuan_tool/video_custom_view")
async def yuan_custom_view(request):
    file_path = request.query.get("filename", "")
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return web.FileResponse(file_path)
    return web.Response(status=404, text="File not found")


# 自定义 API 路由：上传前查重（仅检查 input 目录根路径，与 WhatDreamsCost 完全隔离）
@PromptServer.instance.routes.get("/yuan_tool/video_check_file")
async def yuan_check_file(request):
    filename = os.path.basename(request.query.get("filename", ""))
    file_size = request.query.get("size", "")
    if not filename:
        return web.json_response({"exists": False})
    upload_dir = folder_paths.get_input_directory()
    candidate = os.path.join(upload_dir, filename)
    if os.path.isfile(candidate):
        if file_size:
            try:
                if os.path.getsize(candidate) == int(file_size):
                    return web.json_response({"exists": True, "name": filename})
            except ValueError:
                return web.json_response({"exists": True, "name": filename})
        else:
            return web.json_response({"exists": True, "name": filename})
    return web.json_response({"exists": False})


def _read_and_write_file_chunk(file, file_path, mode):
    chunk_bytes = file.file.read()
    with open(file_path, mode) as f:
        f.write(chunk_bytes)


# 自定义 API 路由：分块上传，绕过 413 Payload Too Large 错误
@PromptServer.instance.routes.post("/yuan_tool/video_upload_chunk")
async def yuan_upload_chunk(request):
    post = await request.post()
    file = post.get("file")
    filename = post.get("filename")
    chunk_index = int(post.get("chunk_index"))
    total_chunks = int(post.get("total_chunks"))

    upload_dir = folder_paths.get_input_directory()
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)

    # 如果不是第一个分块则追加，否则新建文件
    mode = "ab" if chunk_index > 0 else "wb"

    # 将阻塞的磁盘读写操作放到线程池执行
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _read_and_write_file_chunk, file, file_path, mode)

    if chunk_index == total_chunks - 1:
        return web.json_response({"name": filename})
    return web.json_response({"status": "ok"})


# 智能分段检测（供 /yuan_video_ui_detect_segments API 在线程池中调用）
def _detect_segments_sync(video_path, fps, start_time, end_time):
    """同步执行分段检测 + 缩略图提取，供 API 在线程池中运行，避免阻塞事件循环。"""
    import io, base64
    actual_start_time = max(0.0, start_time)
    video_duration = 0.0
    try:
        with av.open(video_path) as c:
            vs = c.streams.video[0] if len(c.streams.video) > 0 else None
            if vs and vs.duration and vs.time_base:
                video_duration = float(vs.duration * vs.time_base)
    except Exception as e:
        print(f"[YuanVideoUI] detect_segments open failed: {e}")
    actual_end_time = end_time if (end_time > 0 and end_time > actual_start_time) else video_duration
    if actual_end_time <= 0:
        actual_end_time = float('inf')

    bounds, seg_count = _build_segment_bounds(video_path, actual_start_time, actual_end_time, fps)
    if seg_count <= 1:
        return {"segments": [], "total": 0}

    # 打开容器，逐段 seek 提取首帧缩略图
    container = None
    vstream = None
    try:
        container = av.open(video_path)
        vstream = container.streams.video[0] if len(container.streams.video) > 0 else None
        if vstream:
            vstream.thread_type = "AUTO"
    except Exception as e:
        print(f"[YuanVideoUI] thumb open failed: {e}")

    segs = []
    for i in range(seg_count):
        s0, s1 = bounds[i], bounds[i + 1]
        seg_start = actual_start_time + s0 / fps
        seg_end = actual_start_time + s1 / fps
        thumb = ""
        if container is not None and vstream is not None:
            try:
                if vstream.time_base:
                    seek_pts = int(seg_start / float(vstream.time_base))
                else:
                    seek_pts = int(seg_start * av.time_base)
                container.seek(seek_pts, stream=vstream, backward=True)
                got = None
                for f in container.decode(vstream):
                    t = f.time if f.time is not None else 0.0
                    if t >= seg_start:
                        got = f
                        break
                if got is not None:
                    got = got.reformat(format="rgb24")
                    img = got.to_image()
                    img.thumbnail((160, 160))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=70)
                    thumb = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as e:
                print(f"[YuanVideoUI] thumb for seg {i} failed: {e}")
        segs.append({
            "index": i,
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "duration": round(max(0.0, seg_end - seg_start), 3),
            "frames": max(0, s1 - s0),
            "thumb": thumb,
        })
    if container is not None:
        container.close()

    return {"segments": segs, "total": seg_count}


@PromptServer.instance.routes.get("/yuan_tool/video_detect_segments")
async def yuan_detect_segments(request):
    video = request.query.get("filename", "")
    if not video:
        return web.json_response({"error": "no filename"}, status=400)
    try:
        fps = float(request.query.get("fps", "24") or 24)
        start_time = float(request.query.get("start_time", "0") or 0)
        end_time = float(request.query.get("end_time", "0") or 0)
    except (TypeError, ValueError):
        fps, start_time, end_time = 24.0, 0.0, 0.0

    # 与节点一致的路径解析
    video_path = video
    if not os.path.exists(video_path):
        annotated = folder_paths.get_annotated_filepath(video)
        if os.path.exists(annotated):
            video_path = annotated
        else:
            candidate = os.path.join(folder_paths.get_input_directory(), video)
            if os.path.exists(candidate):
                video_path = candidate
            else:
                return web.json_response({"error": f"video not found: {video}"}, status=404)

    # 完整结果磁盘缓存：同一视频 + 同一参数直接返回，不重新解码/提取缩略图
    only_cached = request.query.get("cached", "0") == "1"
    result_key = None
    try:
        st = os.stat(video_path)
        result_key = f"{os.path.abspath(video_path)}|{st.st_size}|{int(st.st_mtime)}|{fps}|{start_time}|{end_time}"
    except OSError:
        pass

    if result_key is not None:
        with _scene_cut_lock:
            result_cache = _load_result_cache()
        entry = result_cache.get(result_key)
        if entry and "segments" in entry:
            resp_entry = dict(entry)
            resp_entry["cached"] = True
            return web.json_response(resp_entry)

    if only_cached:
        # 前端切智能模式自动恢复：无缓存则静默返回，避免触发完整检测
        return web.json_response({"cached": False, "segments": [], "total": 0})

    # 耗时检测/缩略图放到线程池，避免阻塞服务器事件循环
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _detect_segments_sync, video_path, fps, start_time, end_time)

    # 只缓存有效结果，避免把失败/空结果存进磁盘
    if result_key is not None and result.get("total", 0) > 0:
        with _scene_cut_lock:
            result_cache = _load_result_cache()
            result_cache[result_key] = result
            if len(result_cache) > _SEGMENTS_RESULT_CACHE_MAX:
                for old_k in list(result_cache)[:len(result_cache) - _SEGMENTS_RESULT_CACHE_MAX]:
                    result_cache.pop(old_k, None)
            _save_result_cache(result_cache)
    return web.json_response(result)


def _detect_shot_frames(video_path):
    """使用 PySceneDetect 检测源视频的镜头切割帧（含 0 与末尾帧）。

    返回排序后的源视频帧号列表；PySceneDetect 不可用或检测失败时返回 None。
    """
    try:
        from scenedetect import AdaptiveDetector, detect
    except ImportError:
        print('[YuanVideoUI] 未检测到 PySceneDetect，智能分段不可用。请安装: pip install "scenedetect>=0.6.4,<0.8"')
        return None
    try:
        detector = AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=15)
        scenes = detect(video_path, detector, show_progress=False, start_in_scene=True)
    except Exception as e:
        print(f"[YuanVideoUI] 智能分割检测失败: {e}")
        return None
    if not scenes:
        return None
    cuts = [int(scenes[0][0].get_frames())]
    for start_tc, _end_tc in scenes[1:]:
        cuts.append(int(start_tc.get_frames()))
    end_frame = int(scenes[-1][1].get_frames())
    if end_frame not in cuts:
        cuts.append(end_frame)
    return sorted(set(cuts))


# ====================================================================
# 镜头检测缓存：PySceneDetect 需要整段解码，非常慢。
# 只缓存「切割点元数据」（每段起止位置），不缓存任何视频帧/缩略图数据。
# 1) 内存缓存：同一次运行内复用；
# 2) 磁盘缓存（JSON）：重启 ComfyUI 后，同一视频源直接读取切割点，
#    无需再次整段解码检测，也就没有大内存占用。
# ====================================================================
_scene_cut_cache = {}
_SCENE_CUT_CACHE_MAX = 16
_scene_cut_lock = threading.Lock()
_SEGMENT_DISK_CACHE_MAX = 64
_SEGMENTS_RESULT_CACHE_MAX = 8


def _segment_cache_dir():
    """缓存目录：input 目录下子目录（只存分割元数据 JSON，不含视频帧，启动不会被清理）。"""
    try:
        cache_dir = os.path.join(folder_paths.get_input_directory(), "yuan_tool_video_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    except Exception:
        try:
            return folder_paths.get_temp_directory()
        except Exception:
            return tempfile.gettempdir()


def _segment_disk_cache_path():
    return os.path.join(_segment_cache_dir(), "yuan_video_ui_segment_cache.json")


def _segment_result_cache_path():
    return os.path.join(_segment_cache_dir(), "yuan_video_ui_segments_result_cache.json")


def _load_disk_cache():
    try:
        with open(_segment_disk_cache_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_disk_cache(cache):
    try:
        with open(_segment_disk_cache_path(), "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError as e:
        print(f"[YuanVideoUI] save segment cache failed: {e}")


def _load_result_cache():
    """完整分段结果缓存（含缩略图），键 = 视频+参数；命中后无需重新解码/提取缩略图。"""
    try:
        with open(_segment_result_cache_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_result_cache(cache):
    try:
        with open(_segment_result_cache_path(), "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError as e:
        print(f"[YuanVideoUI] save segments result cache failed: {e}")


def _get_scene_cuts(video_path):
    """带缓存（内存 + 磁盘）的镜头切割检测。缓存键含文件大小与修改时间，文件变化自动失效。"""
    try:
        st = os.stat(video_path)
        key = (os.path.abspath(video_path), st.st_size, int(st.st_mtime))
    except OSError:
        key = (os.path.abspath(video_path), None, None)
        st = None

    with _scene_cut_lock:
        if key in _scene_cut_cache:
            return _scene_cut_cache[key]

        # 尝试磁盘缓存：同视频源直接读取切割点，避免再次整段解码
        if st is not None:
            disk_key = f"{key[0]}|{key[1]}|{key[2]}"
            disk_cache = _load_disk_cache()
            entry = disk_cache.get(disk_key)
            if entry and "cuts" in entry:
                cuts = list(entry["cuts"])
                _scene_cut_cache[key] = cuts
                return cuts

        cuts = _detect_shot_frames(video_path)
        cuts = cuts or []  # 检测失败或不可用 -> 视为无分段
        _scene_cut_cache[key] = cuts
        if len(_scene_cut_cache) > _SCENE_CUT_CACHE_MAX:
            _scene_cut_cache.pop(next(iter(_scene_cut_cache)))

        # 写入磁盘：只保存切割点元数据
        if st is not None:
            disk_key = f"{key[0]}|{key[1]}|{key[2]}"
            disk_cache = _load_disk_cache()
            disk_cache[disk_key] = {"cuts": cuts}
            if len(disk_cache) > _SEGMENT_DISK_CACHE_MAX:
                for old_k in list(disk_cache)[:len(disk_cache) - _SEGMENT_DISK_CACHE_MAX]:
                    disk_cache.pop(old_k, None)
            _save_disk_cache(disk_cache)

    return _scene_cut_cache[key]


def _build_segment_bounds(video_path, actual_start_time, actual_end_time, out_fps):
    """与节点内一致的智能分段边界计算（基于缓存镜头检测）。

    返回 (segment_bounds, seg_count)。segment_bounds 是输出帧坐标系下的边界列表。
    """
    end = actual_end_time
    if end == float('inf'):
        try:
            with av.open(video_path) as c:
                vs = c.streams.video[0] if len(c.streams.video) > 0 else None
                if vs and vs.duration and vs.time_base:
                    end = float(vs.duration * vs.time_base)
        except Exception:
            end = 0
    total_out = int(round((end - actual_start_time) * out_fps)) if end > actual_start_time else 0
    total_out = max(total_out, 1)
    if total_out <= 1:
        return [0, 1], 1

    bounds = [0, total_out]
    cuts = _get_scene_cuts(video_path)
    if cuts and len(cuts) >= 2:
        native_fps = 24.0
        try:
            with av.open(video_path) as c:
                vs = c.streams.video[0] if len(c.streams.video) > 0 else None
                if vs and vs.average_rate:
                    native_fps = float(vs.average_rate)
        except Exception:
            pass
        if native_fps <= 0:
            native_fps = 24.0
        for c_ in cuts:
            cut_time = float(c_) / native_fps
            # 仅保留修剪范围内的镜头边界（两端各留 0.1 秒缓冲）
            if actual_start_time + 0.1 < cut_time < end - 0.1:
                # +1 帧：切割点所在帧仍是上一分镜的最后 1 帧画面，
                # 在新段的起始帧之前裁切，避免遗留上一个分镜的画面
                fb = int(round((cut_time - actual_start_time) * out_fps)) + 1
                bounds.append(max(1, min(total_out - 1, fb)))
        bounds = sorted(set(bounds))
        # 合并过近的边界，避免产生极短片段
        min_gap = 4
        merged = [bounds[0]]
        for b in bounds[1:-1]:
            if b - merged[-1] < min_gap:
                continue
            merged.append(b)
        if bounds[-1] - merged[-1] < min_gap and len(merged) > 1:
            merged.pop()
        merged.append(bounds[-1])
        bounds = merged
    if len(bounds) < 2:
        bounds = [0, total_out]

    # ====================================================================
    # 长片段细分：镜头边界之间若仍有超过 15 秒的片段，则自动均分。
    # 段数 n = ceil(片段时长 / 15)：>15s 对半 2 段，>30s 均分 3 段，
    # >45s 均分 4 段，依此类推。无镜头检测时整段同样参与细分。
    # ====================================================================
    max_seg_sec = 15.0
    refined = [bounds[0]]
    for i in range(1, len(bounds)):
        s0 = bounds[i - 1]
        s1 = bounds[i]
        seg_frames = s1 - s0
        if seg_frames > 0 and out_fps > 0:
            n = int(math.ceil(seg_frames / (out_fps * max_seg_sec)))
            if n > 1:
                for k in range(1, n):
                    split = s0 + int(round(seg_frames * k / n))
                    if split > refined[-1] and split < s1:
                        refined.append(split)
        refined.append(s1)
    bounds = refined

    return bounds, max(1, len(bounds) - 1)


class YuanVideoUI:
    DESCRIPTION = (
        "加载视频 UI：内置视频预览与时间轴裁剪工具。支持从 input 目录或本地路径加载视频，"
        "按时间/帧两种显示模式裁剪起止范围，可选按镜头智能分段输出指定分段；"
        "输出提取到的图像序列、音频、帧数与分段总数。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            try:
                files = sorted(folder_paths.filter_files_content_types(all_files, ["video"]))
            except:
                video_extensions = ('.mp4', '.webm', '.mkv', '.avi', '.mov', '.m4v', '.flv', '.wmv')
                files = sorted([f for f in all_files if f.lower().endswith(video_extensions)])

        if not files or len(files) == 0:
            files = ["无"]

        return {
            "required": {
                "视频": (files,),
                "开始时间": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
                "结束时间": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
                "时长": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
                "开始帧": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1}),
                "结束帧": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1}),
                "时长帧数": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1}),
                "最长边": ("INT", {"default": 1536, "min": 0, "max": 100000, "step": 8, "tooltip": "输出视频的最长边像素，等比缩放另一条边；0 表示保持原始尺寸。"}),
                "帧率": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1, "tooltip": "强制以指定帧率提取视频帧。"}),
                "显示模式": (["秒", "帧"], {"default": "秒"}),
                "输出模式": (["自定义裁剪输出", "智能分段输出"], {"default": "自定义裁剪输出"}),
                "分段索引": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1, "tooltip": "智能分段模式下，选择输出的分段（第 1 段为 0）。"}),
                "裁剪X": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "裁剪Y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "裁剪宽度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "裁剪高度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "INT")
    RETURN_NAMES = ("图像", "音频", "帧数", "分段总数")
    FUNCTION = "load_video"
    CATEGORY = "Yuan Tool/视频"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def load_video(self, 视频, 帧率, 显示模式, 开始时间, 结束时间, 时长, 开始帧, 结束帧, 时长帧数, 最长边=1536, 裁剪X=0.0, 裁剪Y=0.0, 裁剪宽度=1.0, 裁剪高度=1.0, 输出模式="自定义裁剪输出", 分段索引=0, **kwargs):
        _log_ram("[YuanVideoUI] 执行前")
        if not 视频:
            # 未加载视频时返回空默认值
            empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            empty_audio = {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": 44100}
            return (empty_image, empty_audio, 0, 1)

        # 1. 优先尝试绝对路径，其次使用 ComfyUI 标准路径解析
        video_path = 视频  # 先按绝对/精确路径处理
        if not os.path.exists(video_path):
            video_path_annotated = folder_paths.get_annotated_filepath(视频)
            if os.path.exists(video_path_annotated):
                video_path = video_path_annotated
            else:
                video_path_input = os.path.join(folder_paths.get_input_directory(), 视频)
                if os.path.exists(video_path_input):
                    video_path = video_path_input
                else:
                    raise FileNotFoundError(f"视频文件未找到: {视频}")

        # 打开容器以读取流和元数据
        container = av.open(video_path)

        # 确定视频流和时长
        video_stream = container.streams.video[0] if len(container.streams.video) > 0 else None
        video_duration = 0
        if video_stream and video_stream.duration and video_stream.time_base:
            video_duration = float(video_stream.duration * video_stream.time_base)

        orig_w = video_stream.codec_context.width if video_stream else 512
        orig_h = video_stream.codec_context.height if video_stream else 512

        # 确定正确的色彩空间和颜色范围，避免 PyAV 转换时产生色偏
        try:
            from av.video.reformatter import Colorspace, ColorRange
            # 改进回退启发式：同时检查两个维度（例如 720x1280 竖屏视频也是高清）
            fallback_cs = Colorspace.ITU709 if max(orig_w, orig_h) >= 720 else Colorspace.ITU601
            fallback_cr = ColorRange.MPEG
            dst_range = ColorRange.JPEG # RGB 应始终为全范围
        except ImportError:
            fallback_cs = "itu709" if max(orig_w, orig_h) >= 720 else "itu601"
            fallback_cr = "mpeg"
            dst_range = "jpeg"

        src_colorspace = fallback_cs
        src_color_range = fallback_cr

        if video_stream and video_stream.codec_context:
            cc = video_stream.codec_context

            c_space = getattr(cc, 'colorspace', getattr(cc, 'color_space', None))
            if c_space and hasattr(c_space, 'name') and c_space.name != "UNSPECIFIED":
                src_colorspace = c_space
            elif c_space and isinstance(c_space, str) and "unspecified" not in c_space.lower():
                src_colorspace = c_space

            c_range = getattr(cc, 'color_range', None)
            if c_range and hasattr(c_range, 'name') and c_range.name != "UNSPECIFIED":
                src_color_range = c_range
            elif c_range and isinstance(c_range, str) and "unspecified" not in c_range.lower():
                src_color_range = c_range

        # 根据交互 UI 计算手动裁剪
        manual_crop_left = int(orig_w * 裁剪X)
        manual_crop_top = int(orig_h * 裁剪Y)
        manual_crop_right = orig_w - int(orig_w * (裁剪X + 裁剪宽度))
        manual_crop_bottom = orig_h - int(orig_h * (裁剪Y + 裁剪高度))

        # 确保裁剪不超出图像范围
        manual_crop_left = max(0, min(manual_crop_left, orig_w - 1))
        manual_crop_top = max(0, min(manual_crop_top, orig_h - 1))
        manual_crop_right = max(0, min(manual_crop_right, orig_w - manual_crop_left - 1))
        manual_crop_bottom = max(0, min(manual_crop_bottom, orig_h - manual_crop_top - 1))

        # 手动裁剪后，新的原始尺寸为：
        cropped_orig_w = orig_w - manual_crop_left - manual_crop_right
        cropped_orig_h = orig_h - manual_crop_top - manual_crop_bottom

        # 最长边等比缩放：以裁剪后原始尺寸的最长边为基准，等比缩放另一条边；
        # 值为 0 表示保持原始尺寸不缩放。
        scale_w, scale_h = cropped_orig_w, cropped_orig_h
        pad_left = pad_right = pad_top = pad_bottom = 0
        crop_left = crop_right = crop_top = crop_bottom = 0
        if 最长边 > 0 and cropped_orig_w > 0 and cropped_orig_h > 0:
            ratio = 最长边 / max(cropped_orig_w, cropped_orig_h)
            scale_w = max(2, int(round(cropped_orig_w * ratio)))
            scale_h = max(2, int(round(cropped_orig_h * ratio)))
            scale_w = scale_w - (scale_w % 2)
            scale_h = scale_h - (scale_h % 2)

        # 根据前端模式确定精确的起止范围
        if 显示模式 == "帧":
            fr = float(帧率) if 帧率 > 0 else 24.0
            actual_start_time = float(开始帧) / fr
            actual_end_time = float(结束帧) / fr if (结束帧 > 0 and 结束帧 > 开始帧) else video_duration
        else:
            actual_start_time = 开始时间
            actual_end_time = 结束时间 if (结束时间 > 0 and 结束时间 > 开始时间) else video_duration

        if actual_end_time <= 0:
            actual_end_time = float('inf') # 时长未知时的回退值

        # ====================================================================
        # 4. 智能分段：基于缓存镜头检测计算分段边界，
        #    并确定本次只解码的起止范围（切换分段时不再整段重新提取）
        # ====================================================================
        out_fps = float(帧率) if 帧率 > 0 else 24.0
        seg_count = 1
        segment_bounds = None
        decode_start_time = actual_start_time
        decode_end_time = actual_end_time
        if 输出模式 == "智能分段输出":
            segment_bounds, seg_count = _build_segment_bounds(video_path, actual_start_time, actual_end_time, out_fps)
            selected_idx = max(0, min(int(分段索引), seg_count - 1))
            s0, s1 = segment_bounds[selected_idx], segment_bounds[selected_idx + 1]
            if s1 <= s0:
                s1 = s0 + 1
            decode_start_time = actual_start_time + s0 / out_fps
            decode_end_time = actual_start_time + s1 / out_fps
            # 目标帧数与检测接口 frames = s1 - s0 口径一致，
            # 解码按目标帧数精确输出，避免浮点时间边界误差导致少 1 帧
            target_frames = max(0, s1 - s0)
        else:
            target_frames = None

        # 2. 提取视频帧 (PyAV)
        frames = []
        image_tensor = None
        frames_loaded = 0

        if video_stream:
            video_stream.thread_type = "AUTO" # 启用多线程解码

            # 高效回退到最近的关键帧
            if video_stream.time_base:
                seek_pts = int(decode_start_time / float(video_stream.time_base))
            else:
                seek_pts = int(decode_start_time * av.time_base)

            container.seek(seek_pts, stream=video_stream, backward=True)

            # 自定义采样以强制指定帧率
            frame_interval = 1.0 / float(帧率) if 帧率 > 0 else 1.0/24.0
            expected_target_time = decode_start_time

            # 预计算期望帧数
            alloc_end_time = decode_end_time if decode_end_time != float('inf') else video_duration
            expected_frames = 0
            if alloc_end_time > 0:
                duration_to_extract = alloc_end_time - decode_start_time
                if duration_to_extract > 0:
                    expected_frames = int(np.ceil(duration_to_extract / frame_interval)) + 2

            # ====================================================================
            # 内存保护：估算输出张量大小，避免整段解码把内存拔高到数 GB
            # （1080p float32 单帧约 25MB，240 帧即约 6GB；ComfyUI 还会缓存该输出）
            # ====================================================================
            _FRAME_BYTES = 4  # float32
            _MEMORY_HARD_LIMIT_GB = 8.0   # 超过则直接报错，防止内存暴涨
            _MEMORY_WARN_GB = 1.5         # 超过则打印明显警告
            if expected_frames > 0 and scale_w > 0 and scale_h > 0:
                est_mb = expected_frames * scale_w * scale_h * 3 * _FRAME_BYTES / (1024 * 1024)
                if est_mb > _MEMORY_HARD_LIMIT_GB * 1024:
                    raise RuntimeError(
                        f"[YuanVideoUI] 内存保护：当前范围将解码约 {expected_frames} 帧，"
                        f"输出图像预计占用约 {est_mb / 1024:.1f} GB 内存，内存占用过大已阻止运行。\n"
                        f"请缩小【开始时间~结束时间】范围，或降低【帧率】/【最长边】，"
                        f"或改用“智能分段输出”模式（只解码所选分段）。"
                    )
                elif est_mb > _MEMORY_WARN_GB * 1024:
                    print(f"\n[YuanVideoUI] ⚠ 警告：当前范围解码约 {expected_frames} 帧，"
                          f"输出图像预计占用约 {est_mb / 1024:.1f} GB 内存。\n"
                          f"[YuanVideoUI]    建议缩小时间范围、降低帧率，或用“智能分段输出”模式。\n")

            pbar = comfy.utils.ProgressBar(expected_frames) if expected_frames > 0 else None

            # 复用 resize 目标缓冲，避免每帧重新分配大块内存（减少 Windows 堆碎片/内存叠加）
            _resize_dst = None
            if scale_w != cropped_orig_w or scale_h != cropped_orig_h:
                import cv2
                _resize_dst = np.zeros((scale_h, scale_w, 3), dtype=np.uint8)

            for frame in container.decode(video_stream):
                frame_time = frame.time
                if frame_time is None:
                    frame_time = float(frame.pts * float(video_stream.time_base)) if frame.pts and video_stream.time_base else 0.0

                if frame_time < decode_start_time:
                    continue

                # 加一点缓冲（一个间隔）确保边界判断正确
                if frame_time > decode_end_time + frame_interval:
                    break

                # 强制正确的色彩空间和范围转换，修复 PyAV 色偏。
                # 省略 dst_colorspace 让 swscale 对 RGB 输出使用默认值
                # （传入它可能导致 YUV 矩阵应用错误）。
                try:
                    frame = frame.reformat(
                        format="rgb24",
                        src_colorspace=src_colorspace,
                        src_color_range=src_color_range,
                        dst_color_range=dst_range
                    )
                    frame_rgb = frame.to_ndarray(format='rgb24')
                except Exception as e:
                    # 回退：如果显式色彩转换失败，使用 PyAV 默认转换
                    print(f"[YuanVideoUI] Color reformat failed, using default: {e}")
                    frame_rgb = frame.to_ndarray(format='rgb24')

                # 先应用交互式裁剪
                if manual_crop_left > 0 or manual_crop_top > 0 or manual_crop_right > 0 or manual_crop_bottom > 0:
                    frame_rgb = frame_rgb[manual_crop_top:orig_h-manual_crop_bottom, manual_crop_left:orig_w-manual_crop_right, :]

                # 缩放到目标尺寸
                if scale_w != cropped_orig_w or scale_h != cropped_orig_h:
                    import cv2
                    cv2.resize(frame_rgb, (scale_w, scale_h), interpolation=cv2.INTER_AREA, dst=_resize_dst)
                    frame_rgb = _resize_dst

                if crop_left > 0 or crop_top > 0 or crop_right > 0 or crop_bottom > 0:
                    frame_rgb = frame_rgb[crop_top:scale_h-crop_bottom, crop_left:scale_w-crop_right, :]
                if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                    frame_rgb = np.pad(frame_rgb, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

                # 基于时间戳精确复制或跳过帧，以满足强制帧率。
                # FIX: 对 actual_end_time 使用严格小于 (<)，避免在时长切片边界多取一帧！
                # 智能分段模式（target_frames 已设置）按目标帧数精确输出，与检测接口 frames 一致；
                # 其余模式维持原浮点时间边界判断。
                while expected_target_time <= frame_time and (
                    (target_frames is not None and frames_loaded < target_frames)
                    or (target_frames is None and expected_target_time < decode_end_time - 1e-5)
                ):
                    if image_tensor is None and expected_frames > 0:
                        # 第一帧：分配张量
                        height, width = frame_rgb.shape[:2]
                        alloc_frames = expected_frames + 50 # 增加缓冲避免重新分配
                        try:
                            image_tensor = torch.zeros((alloc_frames, height, width, 3), dtype=torch.float32)
                        except Exception as e:
                            print(f"[YuanVideoUI] Pre-allocation failed, falling back to list: {e}")
                            expected_frames = 0 # 禁用预分配

                    if image_tensor is not None:
                        # 检查边界（以防万一）
                        if frames_loaded >= image_tensor.shape[0]:
                            # 预估不足时扩展张量
                            extension = torch.zeros((50, image_tensor.shape[1], image_tensor.shape[2], 3), dtype=torch.float32)
                            image_tensor = torch.cat((image_tensor, extension), dim=0)

                        # 以最小内存拷贝直接写入张量
                        image_tensor[frames_loaded] = torch.from_numpy(frame_rgb).float().div_(255.0)
                        frames_loaded += 1
                    else:
                        # 预分配失败时的列表回退
                        frames.append(frame_rgb)

                    if pbar:
                        pbar.update(1)

                    expected_target_time += frame_interval

        # 转换为 ComfyUI 标准图像格式 [N, H, W, C]，float32，范围 0.0-1.0
        if image_tensor is not None:
            if frames_loaded > 0:
                image_tensor = image_tensor[:frames_loaded]
            else:
                image_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        elif len(frames) > 0:
            frames_np = np.array(frames, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(frames_np)
        else:
            # 空切片的回退
            image_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)

        # 3. 提取音频 (PyAV)
        audio_dict = {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": 44100} # 默认空音频

        if len(container.streams.audio) > 0:
            try:
                audio_stream = container.streams.audio[0]
                audio_stream.thread_type = "AUTO"
                sample_rate = getattr(audio_stream, 'rate', 44100) or 44100

                # 必须针对音频流重新 seek
                if audio_stream.time_base:
                    seek_pts = int(decode_start_time / float(audio_stream.time_base))
                else:
                    seek_pts = int(decode_start_time * av.time_base)

                container.seek(seek_pts, stream=audio_stream, backward=True)

                # 重采样为标准浮点平面格式 (fltp)
                resampler = av.AudioResampler(format='fltp')

                audio_data = []
                first_frame_time = None

                for frame in container.decode(audio_stream):
                    frame_time = frame.time
                    if frame_time is None:
                        frame_time = float(frame.pts * float(audio_stream.time_base)) if frame.pts and audio_stream.time_base else 0.0

                    # 留出 1 秒缓冲确保能抓到结束帧
                    if frame_time > decode_end_time + 1.0:
                        break

                    if first_frame_time is None:
                        first_frame_time = frame_time

                    resampled_frames = resampler.resample(frame)
                    for r_frame in resampled_frames:
                        audio_data.append(r_frame.to_ndarray())

                if audio_data:
                    # 沿采样轴水平拼接所有帧
                    waveform_np = np.concatenate(audio_data, axis=1)
                    waveform = torch.from_numpy(waveform_np).float()

                    if first_frame_time is None:
                        first_frame_time = 0.0

                    # 计算精确切片点以精确裁剪
                    offset_sec = max(0.0, decode_start_time - first_frame_time)
                    start_sample = int(offset_sec * sample_rate)

                    duration_sec_audio = decode_end_time - decode_start_time
                    end_sample = start_sample + int(duration_sec_audio * sample_rate)

                    # 正确裁剪数组边界
                    if end_sample > start_sample:
                        waveform = waveform[:, start_sample:end_sample]
                    else:
                        waveform = waveform[:, start_sample:]

                    # 扩展为 ComfyUI 音频标准 [batch_size, channels, samples]
                    waveform = waveform.unsqueeze(0)
                    audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

            except Exception as e:
                # 优雅捕获异常，不影响主流程执行
                print(f"[YuanVideoUI] Audio track extraction skipped or failed: {e}")

        # 始终关闭容器以释放系统内存锁
        container.close()

        # 返回前主动回收解码/转换产生的临时大块内存，缓解"切换分段内存叠加"
        try:
            del _resize_dst
        except Exception:
            pass
        gc.collect()
        _log_ram("[YuanVideoUI] 执行后")

        # ====================================================================
        # 5. 输出组装：智能分段模式下，解码范围即为所选分段，直接输出
        # ====================================================================
        output_seg_count = seg_count

        if 输出模式 == "智能分段输出":
            # 解码范围就是所选分段，无需再切片
            final_duration_sec = float(max(0.0, decode_end_time - decode_start_time))
            frame_count = int(image_tensor.shape[0])
        else:
            # 自定义裁剪输出：保持原有行为
            final_duration_sec = float(max(0.0, actual_end_time - actual_start_time))
            frame_count = image_tensor.shape[0] if (frames_loaded > 0 or len(frames) > 0) else 0
            if frame_count == 0 and final_duration_sec > 0:
                # 仅当 PyAV 完全无法解码有效片段时估算
                calc_fr = float(帧率) if 帧率 > 0 else 24.0
                frame_count = int(np.floor(final_duration_sec * calc_fr))

        return (image_tensor, audio_dict, frame_count, output_seg_count)


NODE_CLASS_MAPPINGS = {
    "Yuan_VideoUI": YuanVideoUI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_VideoUI": "加载视频 UI",
}

