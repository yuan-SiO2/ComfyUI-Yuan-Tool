"""Yuan 工具公共函数：供各模块复用的通用后端工具。

集中放置被多个函数/多个后端文件重复使用的逻辑，避免散落复制：
- 分块上传（video / H3 潜空间上传共用）
- 磁盘 JSON 缓存读写
- 媒体路径解析
- PyAV 容器视频时长获取
- 输入目录文件枚举过滤
"""

import os
import asyncio
import inspect
import json

import av
from aiohttp import web

import folder_paths


def load_json_cache(path):
    """读取磁盘 JSON 缓存；不存在或解析失败返回空 dict。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_json_cache(path, cache):
    """写入磁盘 JSON 缓存；写失败静默忽略。"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError:
        pass


def resolve_media_path(name):
    """解析媒体路径：绝对路径 -> annotated -> input 目录，返回第一个存在的绝对路径。

    均不存在或入参为空时返回 None。
    """
    if not name:
        return None
    if os.path.exists(name):
        return name
    annotated = folder_paths.get_annotated_filepath(name)
    if os.path.exists(annotated):
        return annotated
    candidate = os.path.join(folder_paths.get_input_directory(), name)
    if os.path.exists(candidate):
        return candidate
    return None


def video_duration(path):
    """PyAV 打开容器读取视频时长（秒）；不可用或失败返回 0.0。"""
    try:
        with av.open(path) as c:
            vs = c.streams.video[0] if len(c.streams.video) > 0 else None
            if vs and vs.duration and vs.time_base:
                return float(vs.duration * vs.time_base)
    except Exception:
        pass
    return 0.0


def list_input_files(content_types, exts):
    """列出 input 目录文件并按内容类型过滤；filter_files_content_types 不可用时按扩展名兜底。"""
    input_dir = folder_paths.get_input_directory()
    if not os.path.exists(input_dir):
        return []
    all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    try:
        return sorted(folder_paths.filter_files_content_types(all_files, content_types))
    except Exception:
        lower_exts = set(e.lower() for e in exts)
        return sorted(f for f in all_files if os.path.splitext(f)[1].lower() in lower_exts)


async def handle_chunk_upload(request, upload_dir, normalize_name=None, validate=None, response_name=None, on_complete=None):
    """通用分块上传（aiohttp 路由）。

    解析 multipart 表单（file/filename/chunk_index/total_chunks），按 chunk_index
    以追加/新建模式把块写入 upload_dir，返回统一的 JSON 响应，写盘放到线程池避免阻塞事件循环。

    参数：
        upload_dir      目标目录（自动创建）。
        normalize_name  由原始 filename -> 存储文件名；默认 os.path.basename。
        validate        校验函数，入参为写盘绝对路径，返回错误 web.Response 或 None（None 通过）。
        response_name   末块完成时返回给前端的文件名（子目录前缀场景）；默认取存储文件名。
        on_complete     可选，末块写入后回调（可 async），入参为写盘绝对路径，返回 dict 并入响应。
    """
    post = await request.post()
    file = post.get("file")
    filename = post.get("filename") or ""
    chunk_index = int(post.get("chunk_index"))
    total_chunks = int(post.get("total_chunks"))

    os.makedirs(upload_dir, exist_ok=True)
    stored = normalize_name(filename) if normalize_name else os.path.basename(filename)
    file_path = os.path.join(upload_dir, stored)
    if validate:
        err = validate(file_path)
        if err is not None:
            return err

    mode = "ab" if chunk_index > 0 else "wb"

    def _write():
        chunk_bytes = file.file.read()
        with open(file_path, mode) as f:
            f.write(chunk_bytes)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _write)

    if chunk_index == total_chunks - 1:
        resp_name = response_name(stored) if response_name else stored
        payload = {"name": resp_name}
        if on_complete:
            extra = on_complete(file_path)
            if inspect.isawaitable(extra):
                extra = await extra
            if extra:
                payload.update(extra)
        return web.json_response(payload)
    return web.json_response({"status": "ok"})