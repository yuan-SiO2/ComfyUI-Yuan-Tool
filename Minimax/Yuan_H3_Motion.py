"""H3 运动上下文/裁剪节点：支持「视频图像」与「潜空间」两种衔接模式。

视频图像模式（默认，对齐官方 Add Guide for MiniMax H3 数据路径）：
「上下文」把上一片段尾部媒体编码为关键帧（minimax_keyframes）固定到本片段开头；
「裁剪」把 AV 潜空间解码后从头部裁掉被固定窗口覆盖的帧并保存尾段供下一片段衔接。
像素域裁切无潜空间 VRF 相位约束，任意帧数均可干净裁掉。

潜空间模式：上下文与裁剪都直接在 H3 的 AV 潜空间上切片，跳过「解码→重编码」往返，
不损失质量；代价是分辨率必须一致、窗口须为整 VRF 组（17 的倍数）。潜空间模式需要
布局/载荷两补丁（解除仅首末帧锚点限制、让关键帧与引用共存），两补丁带 ABI 标记
门控、先自测再安装，失败则拒绝运行。

两条数据路径由「衔接模式」开关切换，端口/参数/输出槽随模式显隐（见 web/js）。
"""

import hashlib
import math
import os
import re

import numpy as np
import folder_paths
import node_helpers
import torch
import torchaudio

import comfy.utils
import comfy.ldm.minimax.model as mm
import comfy.model_base as model_base
from comfy.nested_tensor import NestedTensor
from server import PromptServer
from aiohttp import web

from ..Yuan_common import handle_chunk_upload

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
    from safetensors import safe_open as _st_safe_open
except ImportError:  # ComfyUI 总是自带 safetensors，此处仅是双保险
    _st_load = _st_save = _st_safe_open = None


# ============================================================================
# H3 常量与潜空间工具函数
# ============================================================================

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24  # H3 原生帧率；音频潜空间按 40Hz 采样，故 FRAME_RESCALE 为 5/3
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0

# 潜空间模式可干净裁剪的整组窗口长度。一个 VRF 组 = 5 个潜空间步覆盖 17 个像素帧
# (1+4+4+4+4)，整组窗口即 17m 帧 = 5m 步。非整组窗口虽可切片，但裁剪会整步移除：
# 剩余步数若非 5 的倍数，潜空间起点落在周期中间，VAE 解码器按错误帧数读第一个
# token，画面闪烁。故可选项均为 17 的倍数，其他值（如过期的已存值）向下吸附到
# 最近的整组，保证固定段与裁剪一致；5 作为退化子组片段的下限。
VIDEO_RUN_GRID = (68, 51, 34, 17, 5)

#   ANCHOR_MODE "head" 把固定段放在片段开头（由裁剪节点移除）；"before"
#               放在负时间轴上无需裁剪，但坐标与文本行碰撞，会削弱锚点
#               并使输出变暗。
#   AUDIO_MODE  "timeline" 把固定音频放在本片段时间轴上让模型续写；"ref"
#               是 stock 放置方式，模型只会模仿：相似音乐而非同录音，
#               且衔接处有滴答声。
ANCHOR_MODE = "head"
AUDIO_MODE = "timeline"


def _pixel_frames(latent_t):
    """latent_t 个潜空间步覆盖的像素帧数。"""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _streams_from_latent(latent):
    """解包 H3 AV 潜空间为所含各流；NestedTensor 须用 unbind()——samples[0]
    会把索引广播进两流并剥掉批维，得不到单条流。"""
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_motion_context: expected a MiniMax H3 AV latent (a nested "
            "video/audio pair), got %r" % type(samples))
    if not parts:
        raise ValueError("h3_motion_context: AV latent contains no streams")
    return parts


# ============================================================================
# 共享 ABI 标记（潜空间模式）
# ============================================================================

# 关键帧真实位置在 keyframe dict 中的键，由布局补丁读取
MC_KEY = "motion_context_index"
# 音频引用尾端目标帧在 ref dict 中的键，由布局补丁读取
MC_AUDIO_KEY = "motion_context_audio_end_frame"


# ============================================================================
# 布局补丁
# 解除 MiniMax H3 仅首/末帧关键帧锚点的限制
# ============================================================================

# 本包装器上的标记：另一份内置本补丁的拷贝可识别并退出而非再包装一层。
# 与 MC_KEY/MC_AUDIO_KEY 同属共享 ABI，改名须所有拷贝同步。
PATCH_MARKER_LAYOUT = "_h3_motion_context_layout_patch"

# MiniMax_H3.py 反向移植包装器（旧核心）上的标记：布局补丁在其上叠加、
# 载荷补丁视为已应用。与 PATCH_MARKER_* 同属共享 ABI，改名须两文件同步。
BACKPORT_MARKER_LAYOUT = "_yuan_minimax_h3_v034_layout"
BACKPORT_MARKER_PAYLOAD = "_yuan_minimax_h3_v034_extra_conds"

_layout_orig_init = None
_layout_applied = False

REF_SEGMENT_KINDS = ("ref_img", "ref_audio")


def _target_origin(layout):
    """目标片段起始坐标，直接从已构建的布局上读出。

    参考块从 text_len 起的游标布局，目标行取游标终值作为原点；关键帧
    坐标按 text_len 直接计算且从不补偿，因此必须加上该项，否则参考会
    使锚点相对目标片段整体后移。从布局读回而非重算游标，上游改动时
    无需同步。
    """
    a, b, kind = layout.segments[-1]
    if kind != "video" or b <= a:
        raise RuntimeError(
            "h3_motion_context: expected the target video rows to be the "
            "last layout segment, found %r spanning %d rows. Upstream "
            "layout change; refusing to rewrite positions." % (kind, b - a))
    return float(layout.position_ids[a, 0])


def _expected_ref_segments(blk):
    """一个参考块按发射顺序应产生的段类型。

    镜像 stock 构造器的分支：image→ref_img；audio→窗口为空时无段；
    video_audio→音频行紧邻视频行之前，故为 ref_audio 再 ref_img。
    """
    kind = blk.get("kind")
    if kind == "image":
        return ("ref_img",)
    if kind == "audio":
        return ("ref_audio",) if int(blk.get("ref_audio_t", 0)) > 0 else ()
    if kind in ("video", "video_audio"):
        if int(blk.get("ref_audio_t", 0)) > 0:
            return ("ref_audio", "ref_img")
        return ("ref_img",)
    raise RuntimeError(
        "h3_motion_context: unknown reference kind %r; cannot tell which "
        "layout rows belong to it." % (kind,))


def _ref_segment_map(layout, refs):
    """返回 {块序号: {段类型: (起, 止)}}，即每个参考块实际产出的行。

    布局已发布段表，参考块按列表顺序发射段，因此直接按序配对即可，
    无需重算 stock 的游标算术，也无需排除落在范围内的关键帧行。
    """
    ref_segs = [(a, b, k) for a, b, k in layout.segments
                if k in REF_SEGMENT_KINDS]
    want = [(i, k) for i, blk in enumerate(refs or [])
            for k in _expected_ref_segments(blk)]
    if len(want) != len(ref_segs):
        raise RuntimeError(
            "h3_motion_context: %d reference blocks should have produced %d "
            "layout segments, the layout has %d. Upstream layout change; "
            "refusing to move rows." % (len(refs or []), len(want),
                                        len(ref_segs)))
    out = {}
    for (i, kind), (a, b, got) in zip(want, ref_segs):
        if got != kind:
            raise RuntimeError(
                "h3_motion_context: reference block %d (%r) should have "
                "emitted a %s segment, the layout has %s. Upstream layout "
                "change; refusing to move rows."
                % (i, refs[i].get("kind"), kind, got))
        out.setdefault(i, {})[kind] = (a, b)
    return out


def _cond_t(text_len, latent_t, frame_count, p):
    """锚定在像素帧 p 的关键帧时间坐标。"""
    if p == 0:
        return float(text_len)
    return float(text_len) + mm.FRAME_RESCALE * float(p)


def _fixup(layout, text_len, latent_t, frame_count, keyframes, refs=None):
    """把条件行时间坐标重写为通用位置公式。

    参考块对锚点的补偿取自目标实际落点（_target_origin）。
    """
    offset = _target_origin(layout) - float(text_len)
    if offset and any(kf.get(MC_KEY) is None for kf in keyframes):
        # 无 MC_KEY 的关键帧保持 stock 原样（不获参考补偿），与 MC 关键帧
        # 混用会相对滑动；当前无路径产生此情况，先拒绝以免出错。
        raise RuntimeError(
            "h3_motion_context: stock and motion-context keyframes mixed in "
            "one graph alongside a ref; their coordinates would disagree. "
            "Give every keyframe a %s entry or remove the refs." % MC_KEY)
    cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    if len(cond_spans) != len(keyframes):
        raise RuntimeError(
            "h3_motion_context: expected %d cond segments, layout has %d. "
            "Refusing to rewrite positions."
            % (len(keyframes), len(cond_spans)))
    for (a, b), kf in zip(cond_spans, keyframes):
        p = kf.get(MC_KEY)
        if p is None:
            continue
        layout.position_ids[a:b, 0] = _cond_t(text_len, latent_t, frame_count, p) + offset


def _fixup_audio(layout, text_len, refs):
    """把被标记的音频引用行平移到目标时间轴上。

    引用与关键帧的行机制相同：模型把坐标落在目标之前某区间的引用读作
    "另一段待模仿的剪辑"而非"本片段的延续"。因此音频引用仍按 stock
    原样构建（行、潜空间、payload 全不动），只平移其时间坐标使窗口尾端
    落在 MC_AUDIO_KEY（与固定视频尾端同一时刻）。平移而非逐行赋值：
    新 = 旧 + 偏移保持 stock 构建的块内结构（声道主序）不变。窗口长于
    视频时向空区溢出而不占用文本行，不会产生 before 模式的碰撞；其他
    参考块不受影响（Ref2VA 图可同时携带图自身的图像/视频/音频引用）。
    """
    marked = [i for i, r in enumerate(refs or [])
              if r.get(MC_AUDIO_KEY) is not None]
    if len(marked) != 1:
        raise RuntimeError(
            "h3_motion_context: audio timeline placement needs exactly one "
            "reference marked with %s; the layout has %d references and %d "
            "marked. If this appeared during startup, check for more than "
            "one H3 Motion Context folder in custom_nodes."
            % (MC_AUDIO_KEY, len(refs or []), len(marked)))
    idx = marked[0]
    blk = refs[idx]
    if blk.get("kind") != "audio":
        raise RuntimeError(
            "h3_motion_context: %s set on a %r ref; only audio refs can be "
            "moved onto the timeline." % (MC_AUDIO_KEY, blk.get("kind")))
    rt = int(blk.get("ref_audio_t", 0))
    if rt <= 0:
        return

    seg = _ref_segment_map(layout, refs).get(idx, {}).get("ref_audio")
    if seg is None:
        raise RuntimeError(
            "h3_motion_context: the marked audio reference produced no "
            "ref_audio segment. Upstream layout change; refusing to move "
            "rows.")
    a, b = seg
    if b - a != 2 * rt:
        # stock 每立体声声道恰好产出 rt 行。用精确计数而非容差：
        # 一旦改变，平移所保持的块内结构也随之改变。
        raise RuntimeError(
            "h3_motion_context: the marked audio reference has %d rows for "
            "%d latent steps, expected %d (stereo, channel-major). Upstream "
            "layout change; refusing to move rows." % (b - a, rt, 2 * rt))

    target_origin = _target_origin(layout)
    slot_start = float(layout.position_ids[a, 0])
    end_frame = float(blk[MC_AUDIO_KEY])
    # 窗口尾端位于目标时刻 FRAME_RESCALE*end_frame，宽度 rt 步
    desired_start = target_origin + mm.FRAME_RESCALE * end_frame - float(rt)
    layout.position_ids[a:b, 0] = (layout.position_ids[a:b, 0]
                                   + (desired_start - slot_start))


def _patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                  keyframes=None, refs=None, frame_count=None):
    _layout_orig_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                      keyframes=keyframes, refs=refs)
    has_mc_kf = bool(keyframes) and any(
        kf.get(MC_KEY) is not None for kf in keyframes)
    has_mc_audio = bool(refs) and any(
        r.get(MC_AUDIO_KEY) is not None for r in refs)
    if has_mc_kf:
        _fixup(self, text_len, latent_t, frame_count, keyframes, refs)
    if has_mc_audio:
        _fixup_audio(self, text_len, refs)
    # 两者皆未标记：stock 图，保持原样


def _layout_self_test():
    """提交前自测：重写必须逐位复现 stock 位置。

    用 stock 与本文机制各构建一次 stock 已支持的两端锚点，位置张量须
    完全相等；再覆盖 stock 无对应物的部分：内部锚点、参考补偿、音频
    平移、以及多参考 Ref2VA 布局中的同一次平移。ComfyUI 若改动位置
    数学或段表则失败，补丁不予安装。
    """
    text_len, latent_t, lh, lw, audio_t = 7, 7, 22, 38, 16
    frame_count = sum(mm.FRAME_PER_TOKEN[k % 5] for k in range(latent_t))

    def build(keyframes=None, refs=None, fix=False, move=False):
        lay = mm.PackedLayout.__new__(mm.PackedLayout)
        _layout_orig_init(lay, text_len, latent_t, lh, lw, audio_t,
                          keyframes=keyframes, refs=refs)
        if fix:
            _fixup(lay, text_len, latent_t, frame_count, keyframes, refs)
        if move:
            _fixup_audio(lay, text_len, refs)
        return lay

    def cond_ts(lay):
        return [float(lay.position_ids[a, 0])
                for a, _, k in lay.segments if k == "cond"]

    # 1. stock 支持的两端锚点必须逐位一致
    dummy_lat = torch.zeros(1, 24, 1, lh // 2, lw // 2)
    stock_kf = [{"resolved_frame_index": 0, "latent": dummy_lat},
                {"resolved_frame_index": frame_count - 1, "latent": dummy_lat}]
    ours_kf = [{"resolved_frame_index": 0, MC_KEY: 0, "latent": dummy_lat},
               {"resolved_frame_index": 0, MC_KEY: frame_count - 1, "latent": dummy_lat}]
    a = build(keyframes=stock_kf)
    b = build(keyframes=ours_kf, fix=True)
    if a.position_ids.shape != b.position_ids.shape:
        raise RuntimeError("position_ids shape mismatch in self-test")
    if not torch.equal(a.position_ids, b.position_ids):
        bad = (a.position_ids != b.position_ids).any(dim=1).nonzero().flatten()
        raise RuntimeError("position mismatch at rows %s" % bad[:8].tolist())

    # 2. 连续锚点须在两端点界定的区间内严格递增
    run = [{"resolved_frame_index": 0, MC_KEY: i, "latent": dummy_lat} for i in range(4)]
    c = build(keyframes=run, fix=True)
    ts = cond_ts(c)
    if len(ts) != len(run):
        raise RuntimeError("expected %d cond segments, got %d" % (len(run), len(ts)))
    if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
        raise RuntimeError("consecutive anchors not strictly increasing: %s" % ts)
    t_last = float(text_len) + mm.FRAME_RESCALE * (frame_count - 1)
    if not (ts[0] == float(text_len) and ts[-1] < t_last):
        raise RuntimeError("run %s escapes the [%.4f, %.4f] span"
                           % (ts, float(text_len), t_last))

    # 3. 加入参考不得使锚点相对目标移动：以目标行自身为基准，
    #    有/无参考时锚点到尾端的间距必须一致
    ref = [{"kind": "audio", "ref_audio_t": 8}]
    d = build(keyframes=run, refs=ref, fix=True)
    ts_ref = cond_ts(d)
    if len(ts_ref) != len(ts):
        raise RuntimeError("cond segment count changed when a ref was added")
    tol = 1e-3
    gap = float(c.position_ids[:, 0].max()) - ts[0]
    gap_ref = float(d.position_ids[:, 0].max()) - ts_ref[0]
    if abs(gap - gap_ref) > tol:
        raise RuntimeError(
            "ref compensation off by %.6f: anchor-to-target gap %.6f without "
            "ref, %.6f with. The target origin read back from the layout no "
            "longer matches its cursor arithmetic." % (gap_ref - gap, gap, gap_ref))
    shifts = [y - x for x, y in zip(ts, ts_ref)]
    if any(abs(sh - shifts[0]) > tol for sh in shifts):
        raise RuntimeError("ref shifted anchors unevenly: %s" % shifts)

    # 4. 音频平移：仅被标记块的行整体平移同一量，其余行逐位不变
    end_frame, rt = 4, 8
    ref_mc = [{"kind": "audio", "ref_audio_t": rt, MC_AUDIO_KEY: end_frame}]
    e = build(keyframes=run, refs=ref_mc, fix=True, move=True)
    _check_move(d, e, ref_mc, 0, "single-ref")

    # 5. 同一平移在 Ref2VA 布局中：图自身的图像/视频/音频引用须原样
    #    通过，被标记块故意放在列表中间（定位不依赖其位置）
    r_lh, r_lw, r_vt = 8, 12, 3
    others = [
        {"kind": "image", "latent_h": r_lh, "latent_w": r_lw},
        {"kind": "video_audio", "latent_h": r_lh, "latent_w": r_lw,
         "latent_t": r_vt, "ref_audio_t": 5},
        {"kind": "audio", "ref_audio_t": 3},
    ]
    marked = {"kind": "audio", "ref_audio_t": rt, MC_AUDIO_KEY: end_frame}
    plain = {"kind": "audio", "ref_audio_t": rt}
    multi_plain = others[:2] + [plain] + others[2:]
    multi_marked = others[:2] + [marked] + others[2:]
    f = build(keyframes=run, refs=multi_plain, fix=True)
    g = build(keyframes=run, refs=multi_marked, fix=True, move=True)
    _check_move(f, g, multi_marked, 2, "multi-ref")

    # 6. 段表须与实际布局一致：不重算游标，而是校验平移依赖的结构
    #    性质——参考块按列表顺序出现、行位于目标之前、块间互不重叠
    smap = _ref_segment_map(f, multi_plain)
    prev_hi = float(text_len) - 1e-9
    origin = _target_origin(f)
    for i in range(len(multi_plain)):
        spans = smap.get(i)
        if not spans:
            continue
        rows = [r for a0, b0 in spans.values() for r in range(a0, b0)]
        lo = min(float(f.position_ids[r, 0]) for r in rows)
        hi = max(float(f.position_ids[r, 0]) for r in rows)
        if lo < prev_hi - 1e-9:
            raise RuntimeError(
                "reference block %d starts at %.6f, before block %d ended "
                "at %.6f. Reference blocks are not laid out in list order."
                % (i, lo, i - 1, prev_hi))
        if hi >= origin - 1e-9:
            raise RuntimeError(
                "reference block %d reaches %.6f, at or past the target "
                "origin %.6f. Reference rows should sit before the target."
                % (i, hi, origin))
        prev_hi = hi


def _check_move(before, after, refs, idx, label):
    """仅被标记块的行沿时间轴整体平移、且平移量一致。"""
    if after.position_ids.shape != before.position_ids.shape:
        raise RuntimeError("%s: audio move changed the layout shape" % label)
    if not torch.equal(before.position_ids[:, 1:], after.position_ids[:, 1:]):
        raise RuntimeError(
            "%s: audio move touched a non-time coordinate column" % label)
    a, b = _ref_segment_map(before, refs)[idx]["ref_audio"]
    expect_moved = set(range(a, b))
    tb, ta = before.position_ids[:, 0], after.position_ids[:, 0]
    moved = set(i for i in range(len(tb)) if float(tb[i]) != float(ta[i]))
    if not moved:
        raise RuntimeError("%s: audio move moved no rows" % label)
    if moved != expect_moved:
        raise RuntimeError(
            "%s: audio move touched the wrong rows: %d moved, %d expected, "
            "e.g. %s" % (label, len(moved), len(expect_moved),
                         sorted(moved ^ expect_moved)[:8]))
    deltas = [float(ta[i]) - float(tb[i]) for i in sorted(moved)]
    if any(abs(dd - deltas[0]) > 1e-9 for dd in deltas):
        raise RuntimeError("%s: audio rows shifted non-uniformly: %s"
                           % (label, deltas[:4]))
    # 平移量本身故意不断言：它取决于参考游标前进多少，属 stock 行为，
    # 在此固定等于把那份算术复制回来。必须成立的是窗口尾端位置：
    # 在目标时间轴上为 FRAME_RESCALE * end_frame 相对目标原点处。
    blk = refs[idx]
    rt = int(blk["ref_audio_t"])
    want_end = (_target_origin(after)
                + mm.FRAME_RESCALE * float(blk[MC_AUDIO_KEY]))
    got_end = float(after.position_ids[a, 0]) + float(rt)
    if abs(got_end - want_end) > 1e-9:
        raise RuntimeError(
            "%s: audio window ends at %.6f, should end at %.6f"
            % (label, got_end, want_end))


setattr(_patched_init, PATCH_MARKER_LAYOUT, True)


def _layout_already_patched():
    """构造函数当前由谁接管？返回 None/"same"/"other"/"backport"/"foreign"。

    多个包会内置本补丁，后加载者若把先加载者的包装器当作原始实现去包装
    就会套上多层（各拷贝用已装好的版本自测，新旧行为互相校验会误拒绝）。
    按可信度递减做四项检查：带标记的拷贝（匹配版本，静默退出）；同插件
    反向移植包装器（MiniMax_H3.py 在旧核心上的 v0.34.0 布局，布局补丁在
    其上叠加）；仅同名的包装器是旧拷贝或分支（先加载者决定支持范围，退出
    并说明）；其余占着构造函数位置的是别的包在补同一处（按 __module__
    归属判断，无法同时持有；被 __wrapped__ 隐藏的无法识别）。
    """
    cls = getattr(mm, "PackedLayout", None)
    init = getattr(cls, "__init__", None)
    if init is None:
        return None
    if getattr(init, PATCH_MARKER_LAYOUT, False):
        return "same"
    if getattr(init, BACKPORT_MARKER_LAYOUT, False):
        return "backport"
    if getattr(init, "__name__", "") == "_patched_init":
        return "other"
    if hasattr(init, "__wrapped__"):
        return "foreign"
    home = getattr(cls, "__module__", None)
    where = getattr(init, "__module__", None)
    if home and where and where != home:
        return "foreign"
    return None


_layout_fail_reason = None


def _apply_layout_patch():
    global _layout_orig_init, _layout_applied, _layout_fail_reason
    if _layout_applied:
        return True
    who = _layout_already_patched()
    if who == "foreign":
        _layout_fail_reason = (
            "PackedLayout.__init__ is already wrapped by a different pack "
            "from another module; refusing to stack a second wrapper.")
        return False
    if who == "backport":
        # 同插件反向移植（MiniMax_H3.py 在旧核心上已替换构造函数）：以它为
        # 基础叠加本文锚点修复。自测对反向移植基础同样有效（复现两端锚点），
        # 继续走到下方公共安装流程。
        pass
    elif who:
        # 补丁已生效（非本份），调用方节点运行前会检查 is_applied()
        _layout_applied = True
        return True
    if not hasattr(mm, "PackedLayout") or not hasattr(mm, "FRAME_RESCALE"):
        _layout_fail_reason = (
            "this ComfyUI lacks comfy.ldm.minimax.model.PackedLayout or "
            "FRAME_RESCALE; its H3 backend predates the layout machinery "
            "this node needs.")
        return False
    _layout_orig_init = mm.PackedLayout.__init__
    try:
        _layout_self_test()
    except Exception as e:
        _layout_orig_init = None
        _layout_fail_reason = "self-test failed: %s" % e
        return False
    mm.PackedLayout.__init__ = _patched_init
    _layout_applied = True
    return True


def _layout_patch_applied():
    return _layout_applied


# ============================================================================
# 载荷补丁
# 让关键帧和引用可以共存
# ============================================================================

# 本包装器上的标记：另一份内置本补丁的拷贝可识别并退出而非再包装一层。
# 是每个内置此补丁的包共享的 ABI。
PATCH_MARKER_PAYLOAD = "_h3_motion_context_payload_patch"

_payload_orig_extra_conds = None
_payload_applied = False


def _patched_extra_conds(self, **kwargs):
    out = _payload_orig_extra_conds(self, **kwargs)

    keyframes = kwargs.get("minimax_keyframes", None)
    refs = kwargs.get("minimax_refs", None)
    if not keyframes or not refs:
        return out  # 只有一种机制在起作用，stock 行为即正确
    if not (any(MC_KEY in kf for kf in keyframes)
            or any(MC_AUDIO_KEY in r for r in refs)):
        # 与本包无关：布局补丁同款门控，不动 payload 可保持两补丁一致，
        # 无关图与 stock 逐位一致
        return out

    cond = out.get("minimax_payload", None)
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        return out

    kf_video = [kf["latent"] for kf in keyframes if "latent" in kf]
    ref_video = [r["latent"] for r in refs if "latent" in r]
    payload["cond_video_latents"] = kf_video + ref_video
    payload["cond_audio_latents"] = [r["audio_latent"] for r in refs
                                     if r.get("audio_latent") is not None]
    # 仅在确实拿到 frame_count 时才写入：本包装器对所有同时含关键帧和
    # 引用的图生效，缺 minimax_frame_count 时原值可能已有效，覆盖为
    # None 会破坏下游末帧锚点分支
    fc = kwargs.get("minimax_frame_count", None)
    if fc is not None:
        payload["frame_count"] = fc
    return out


setattr(_patched_extra_conds, PATCH_MARKER_PAYLOAD, True)


def _payload_already_patched(cls):
    """extra_conds 当前由谁接管？返回 None/"same"/"backport"/"other"/"foreign"。

    检测逻辑与 _layout_already_patched 相同：标记只识别新到能设置它的
    拷贝，同插件反向移植 extra_conds（旧核心）已实现关键帧/引用共存合并
    视为已应用，仅同名的包装器视为旧拷贝或分支（后加载者退出），其他
    包装者是别的包在补同一处则拒绝叠包。
    """
    fn = getattr(cls, "extra_conds", None)
    if fn is None:
        return None
    if getattr(fn, PATCH_MARKER_PAYLOAD, False):
        return "same"
    if getattr(fn, BACKPORT_MARKER_PAYLOAD, False):
        return "backport"
    if getattr(fn, "__name__", "") == "_patched_extra_conds":
        return "other"
    if hasattr(fn, "__wrapped__"):
        return "foreign"
    home = getattr(cls, "__module__", None)
    where = getattr(fn, "__module__", None)
    if home and where and where != home:
        return "foreign"
    return None


_payload_fail_reason = None


def _apply_payload_patch():
    global _payload_orig_extra_conds, _payload_applied, _payload_fail_reason
    if _payload_applied:
        return True
    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        _payload_fail_reason = (
            "model_base.MiniMaxH3.extra_conds was not found; this ComfyUI "
            "predates the H3 extra_conds mechanism this node needs.")
        return False
    who = _payload_already_patched(cls)
    if who == "foreign":
        _payload_fail_reason = (
            "MiniMaxH3.extra_conds is already wrapped by a different pack "
            "from another module; refusing to stack a second wrapper.")
        return False
    if who == "backport":
        # 同插件反向移植 extra_conds（旧核心）已实现关键帧/引用共存合并，
        # 视为已应用；反向移植布局不消费 payload 的 frame_count，无需补写。
        _payload_applied = True
        return True
    if who:
        # 补丁已生效（非本份），调用方节点运行前会检查 is_applied()
        _payload_applied = True
        return True
    _payload_orig_extra_conds = cls.extra_conds
    cls.extra_conds = _patched_extra_conds
    _payload_applied = True
    return True


def _payload_patch_applied():
    return _payload_applied


# ============================================================================
# 补丁安装入口
# ============================================================================

def _ensure_layout_patch():
    """首次运行节点时安装布局补丁（仅一次）。

    导入时打补丁会让本包装器进入本机每个 H3 图的路径；首次使用时安装
    则安装本包不改动任何东西，直到真正衔接片段。代价是自测失败在首次
    渲染而非启动日志中出现，但信息相同，且仍是拒绝而非渲染错误结果。
    """
    if _layout_patch_applied():
        return
    if not _apply_layout_patch():
        raise RuntimeError(
            "h3_motion_context: the layout patch could not be applied, so "
            "interior anchors would be rejected by ComfyUI. Reason: %s"
            % (_layout_fail_reason or "unknown"))


def _ensure_payload_patch():
    """安装载荷补丁（仅一次），仅在固定音频（引用与关键帧须共存）时到达。"""
    if _payload_patch_applied():
        return
    if not _apply_payload_patch():
        raise RuntimeError(
            "h3_motion_context: the payload patch could not be applied. "
            "Without it the audio ref would overwrite the pinned video "
            "latents and the motion context would be lost. Reason: %s"
            % (_payload_fail_reason or "unknown"))


def _snap_guide_frames(n):
    """向下吸附到合法引导长度（17k+5：5/22/39/56…，同官方 Add Guide 多帧批处理）；小于 5 帧返回 0 由调用方报错。"""
    n = int(n)
    if n < 5:
        return 0
    while n % 17 != 5:
        n -= 1
    return n


def _resize_guide(image, width, height):
    """把引导帧缩放到目标分辨率（与官方 Add Guide _resize(..., "center") 逐位一致：
    取前 3 通道、lanczos、保宽高比的居中裁剪）。"""
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos",
                                         "center")
    return samples.movedim(1, -1)


def _encode_guide_color_neutral(vae, guide, width, height, frames):
    """把引导帧编码为关键帧 latent，并做「往返偏色闭环补偿」。

    H3 视频 VAE 的 编码→解码 不是恒等映射：实测每过一遍往返各通道整体偏移
    约 2~3/255，且逐段线性累积（锚定行被采样器当作 σ=0 真值，整段画面跟着
    偏移），链条越长越发黄。这里先把引导帧过一遍往返量出偏移 b 再用 x − b 生成
    关键帧——模型渲染出的锚定画面即等于上一段尾部的真实色调，切断「每段一次
    往返」的累积回路。

    只补偿**逐通道均值**（DC）：往返偏差主体就是 DC，而按像素全量相减
    （x − (往返 − x) = 2x − 往返）等价于对引导做强度 1.0 的 USM——实测把引导的
    高频能量抬到 3.4 倍、过冲光晕 2.3 倍，模型照抄这份「脆」引导后就在衔接处
    留下锐化质感台阶，故此处只取均值（均值不改变任何梯度，锐化副作用为零，
    偏色补偿效果不变）。往返结果帧数/分辨率不符时（罕见）放弃补偿，退回 stock
    行为。"""
    latent = vae.encode(guide.contiguous())
    rendered = vae.decode(latent)
    if rendered.ndim == 5:
        rendered = rendered[0]
    if (rendered.ndim == 4 and int(rendered.shape[0]) == frames
            and tuple(rendered.shape[1:3]) == (height, width)):
        bias = (rendered.to(guide) - guide).mean(dim=(0, 1, 2), keepdim=True)
        guide = (guide - bias).clamp(0.0, 1.0)
        latent = vae.encode(guide.contiguous())
    return latent


def _tail_audio_latent(audio_vae, waveform, sample_rate, frames):
    """取波形尾部 frames 帧对应样本，按官方 Add Guide _encode_ref_audio 同路径编码。

    样本数向下对齐 hop（800 样本/步≈25ms@32kHz），使包装器的编码前裁剪恰好
    无操作、窗口尾端不发生亚步偏移。"""
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if waveform.ndim == 2:
        waveform = waveform[None]
    if int(sample_rate) != vae_sr:
        waveform = torchaudio.functional.resample(
            waveform, int(sample_rate), vae_sr)
    try:
        hop = int(audio_vae.downscale_ratio)
    except (TypeError, ValueError, AttributeError):
        hop = 800
    if hop < 1:
        hop = 800
    need = int(round(frames / float(FPS) * vae_sr))
    need -= need % hop
    length = int(waveform.shape[-1])
    if need > length:
        need = length - (length % hop)
    if need < hop:
        raise ValueError(
            "h3_motion_context: 音频上下文窗口为空（可用音频不足 %d 样本）。"
            % hop)
    z = audio_vae.encode(waveform[:1, ..., length - need:].movedim(1, -1))
    if z.ndim == 3:
        z = z.unsqueeze(0)
    return z


# ============================================================================
# 潜空间模式：直接从 AV 潜空间切片（跳过解码/重编码）
# ============================================================================

def _video_from_latent(latent):
    """从 H3 AV 潜空间中取出视频流。"""
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:  # 未批量化 [C,T,H,W]
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("h3_motion_context: expected video latent [B,C,T,H,W], "
                         "got shape %s" % (tuple(video.shape),))
    return video


def _steps_for_frames(n):
    """从周期位置 0 起恰好覆盖 n 个像素帧所需的潜空间步数。

    无整数步数覆盖 n 时返回 None。视频 VAE 步长按 1,4,4,4,4 像素帧交替，
    只有特定总数可达：1,5,9,13,17,18,…；本节点提供的 17/34/51/68
    恰好落在 5/10/15/20 步上。
    """
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def _video_tail_from_latent(latent, n):
    """直接从生成的 H3 潜空间切出视频尾部 n 个像素帧，跳过 h264 解码与
    VAE 编码。返回 (blocks, offsets, covered)，形状与编码路径产物一致，
    下游无需改动。

    窗口不必从周期位置 0 起：offsets 是每个块在窗口内的真实帧起点（读自
    源潜空间中的实际位置），从组边界起得 1,4,4,4,4 累积位，中途起得
    4,4,4,1,4…，保证固定潜空间与写入的位置始终一致。节点只提供整组，
    其尾部窗口恰落在周期中间，因此这点很关键。
    """
    video = _video_from_latent(latent)
    total = int(video.shape[2])
    steps = _steps_for_frames(n)
    if steps is None:
        raise ValueError(
            "h3_motion_context: a %d frame window is not a whole number of "
            "latent steps, so it cannot be sliced from a latent. Use 17, 34, "
            "51 or 68, or unwire context_latent to encode pixels." % n)
    if steps > total:
        raise ValueError(
            "h3_motion_context: asked for %d latent steps, context_latent "
            "has %d." % (steps, total))
    start = total - steps
    covered = _pixel_frames(steps)
    if covered != n:
        raise RuntimeError(
            "h3_motion_context: %d steps cover %d frames, expected %d."
            % (steps, covered, n))
    base = _pixel_frames(start)
    offsets = [_pixel_frames(start + k) - base for k in range(steps)]
    blocks = [video[:1, :, start + k:start + k + 1].clone()
              for k in range(steps)]
    return blocks, offsets, covered


def _audio_tail_from_latent(latent, a_frames):
    """直接从生成的 H3 潜空间切出末尾 a_frames 帧对应的音频步，
    跳过解码→重编码往返。

    返回 (tail [1,C,2,rt], rt, overhang)：rt 为 40Hz 潜空间步数，
    overhang 是片段音频网格超出最后一个像素帧的分数步。H3 把音频网格
    向上取整（124 帧要 206.67 步，布局分配 207），末步超出末帧约
    overhang/40 s。调用方不用 overhang 平移放置：窗口尾端改与裁剪节点
    的精确音频切点对齐，固定窗口始终落在被裁头部内（见 apply）。
    """
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "h3_motion_context: context_latent has no audio stream. Wire the "
            "sampler output of an H3 AV graph, not a video-only latent.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:  # 未批量化 [C,2,T]
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError("h3_motion_context: expected audio latent [B,C,2,T], "
                         "got shape %s" % (tuple(audio.shape),))
    total_t = int(audio.shape[-1])
    frames = _pixel_frames(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    if not (0.0 <= overhang < 1.0):
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        rt = total_t
    if rt < 1:
        raise ValueError("h3_motion_context: audio window is empty")
    tail = audio[:1, ..., total_t - rt:].clone()
    return tail, rt, float(overhang)


def _silence_audio_latent(audio_vae, audio_t):
    """把真实的数字静音（零波形）编码为恰好 audio_t 步的音频潜空间。

    潜空间直接填零不是静音：音频 VAE 编码器带偏置，零潜空间解码出来是
    非静音内容，必须把真正的零波形送进编码器。波形取 audio_t * hop 个
    样本（32kHz、每步 800 样本），编码器无需补零，输出步数精确等于
    audio_t。encode 走 VAE 包装器通道置后约定（与 VAEEncodeAudio 相同：
    movedim(1,-1) 后由包装器转回 [B,2,L]）。
    """
    hop = int(getattr(audio_vae, "downscale_ratio", 800))
    samples = int(audio_t) * hop
    waveform = torch.zeros(1, 2, samples)  # [B, 声道, 样本] 立体声数字静音
    z = audio_vae.encode(waveform.movedim(1, -1))
    if z.ndim == 3:
        z = z.unsqueeze(0)
    if int(z.shape[-1]) != int(audio_t):
        raise ValueError(
            "h3_motion_context: 静音编码得到 %d 步音频潜空间，期望 %d 步。"
            "audio_vae 端口连接的是 H3 音频 VAE 吗？"
            % (int(z.shape[-1]), int(audio_t)))
    return z[:1].clone()


# ============================================================================
# 上下文媒体标记与加载（自动索引 / 手动上传共用）
# ============================================================================

# 空标记键：片段序号为 0 / 文件未找到时，加载输出带此标记的空媒体，
# 运动上下文据此直通、不裁头
CONTEXT_EMPTY_MARKER = "_h3_motion_context_empty"
# 原因键：空标记的原因（first_clip / file_not_found）
CONTEXT_EMPTY_REASON = "_h3_motion_context_empty_reason"
# 细节键：原因细节（如未找到的片段序号）
CONTEXT_EMPTY_REASON_DETAIL = "_h3_motion_context_empty_reason_detail"
# 正常加载时携带的片段序号键，用于生成"已关联片段 N"提示
CONTEXT_CLIP_INDEX_KEY = "_h3_motion_context_clip_index"

# 尾段媒体文件名约定：存储位置目录下 clip_%05d.mp4（片段序号 2 → clip_00002.mp4）
CLIP_FILE_PREFIX = "clip"
CLIP_FILE_EXT = ".mp4"
# 加载上下文媒体时解码帧的最长边上限：超过该值先等比缩到最长边=1536 再参与
# 引导，降低解码量与后续 VAE 编码开销；未超限原样返回（字节不变），不破坏
# 链条内小分辨率尾段的逐字节精确。固定默认，不暴露参数。
CONTEXT_LOAD_MAX_SIDE = 1536


def _load_media_file(path, clip_index=None):
    """加载「H3 运动裁剪」保存的尾段媒体：.mp4 经 av 解码为像素帧+波形；
    旧版 .safetensors（uint8 帧+波形）仍兼容读取。

    返回 {"pixels": [t,H,W,3] float[0,1], "waveform": [1,2,L]（无音轨为 None）,
    "sample_rate": int}。
    """
    lower = (path or "").lower()
    if lower.endswith(".mp4"):
        return _read_mp4(path, clip_index=clip_index)
    return _read_st_media(path, clip_index=clip_index)


def _cap_longest_side(frames_u8):
    """把 uint8 视频帧等比缩到最长边 ≤ CONTEXT_LOAD_MAX_SIDE。

    仅当超上限时才缩放，未超限原样返回（字节不变）——链条内小分辨率尾段
    保持逐字节精确。支持单帧 [H,W,3] 与帧序列 [T,H,W,3]；解码循环里逐帧
    调用可在堆叠成大张量前先降内存。"""
    h, w = int(frames_u8.shape[-3]), int(frames_u8.shape[-2])
    if max(h, w) <= CONTEXT_LOAD_MAX_SIDE:
        return frames_u8
    ratio = CONTEXT_LOAD_MAX_SIDE / float(max(h, w))
    nh = max(int(h * ratio + 0.5), 2)
    nw = max(int(w * ratio + 0.5), 2)
    batched = frames_u8.ndim == 4
    x = frames_u8 if batched else frames_u8[None]
    x = x.float().div_(255.0).movedim(-1, 1)  # [*,3,H,W]
    y = comfy.utils.common_upscale(x, nw, nh, "area", "disabled")
    out = (y.movedim(1, -1).mul_(255.0).clamp_(0.0, 255.0)
           .round_().to(torch.uint8))
    return out if batched else out[0]


def _read_st_media(path, clip_index=None):
    """读取旧版 .safetensors 格式的尾段媒体（uint8 帧 + 音频波形）。"""
    if _st_load is None:
        raise RuntimeError("h3_motion_context: safetensors is not "
                           "available; cannot load context media.")
    data = _st_load(path)
    if "video" not in data or "audio" not in data:
        raise ValueError(
            "h3_motion_context: %s 不是「H3 运动裁剪」保存的上下文媒体文件"
            "（缺少 video/audio 数据）。" % path)
    video = data["video"]
    # 与主路径一致：4D [T,H,W,3] 即帧序列本身，只有 5D [B,T,H,W,3] 才去批维
    if video.ndim == 5:
        video = video[0]
    if video.ndim == 3:
        video = video[None]
    if video.dtype != torch.uint8:
        raise ValueError(
            "h3_motion_context: %s 是旧版潜空间格式（latent）。请用新版"
            "「H3 运动裁剪」重新生成分段媒体文件。" % path)
    video = _cap_longest_side(video)  # 解码即统一缩放（超上限才缩）
    pixels = video.float().div_(255.0)
    wave = data["audio"]
    if wave.ndim == 3:
        wave = wave[:1]
    elif wave.ndim == 2:
        wave = wave[None]
    sample_rate = 32000
    if _st_safe_open is not None:
        try:
            with _st_safe_open(path, framework="pt", device="cpu") as f:
                meta = f.metadata() or {}
                sample_rate = int(meta.get("sample_rate", 32000))
                if clip_index is None and meta.get("clip_index"):
                    clip_index = int(meta["clip_index"])
        except Exception:
            pass
    out = {"pixels": pixels, "waveform": wave, "sample_rate": sample_rate}
    if clip_index is not None:
        out[CONTEXT_CLIP_INDEX_KEY] = clip_index
    return out


def _read_mp4(path, clip_index=None):
    """把「H3 运动裁剪」保存的 .mp4（音视频一体）解码为像素帧与波形。

    视频帧 → uint8 → [0,1] float 的 [t,H,W,3]；音频重采样为 float32 立体声
    [2,L]（保留容器原始采样率，由调用方后续转 32kHz）；无音轨时 waveform 为 None。"""
    import av
    video_frames = []
    audio_parts = []
    sample_rate = 32000
    container = av.open(path)
    try:
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if video_stream is None:
            raise ValueError("h3_motion_context: %s 内没有视频轨。" % path)
        resampler = None
        if audio_stream is not None:
            sample_rate = int(audio_stream.codec_context.sample_rate or 48000)
            resampler = av.audio.resampler.AudioResampler(
                format="fltp", layout="stereo", rate=sample_rate)
        # 视频/音频须在同一次 decode 中交错取帧：先取完单流会把文件读到
        # EOF，后续再 decode 另一流将拿不到任何帧（av 18 实测返回空）
        targets = [s for s in (video_stream, audio_stream) if s is not None]
        for frame in container.decode(*targets):
            if isinstance(frame, av.VideoFrame):
                arr = frame.to_ndarray(format="rgb24")  # [H,W,3] uint8
                # 解码即统一缩放：逐帧等比缩到最长边 ≤ 1536（超上限才缩，
                # 在堆叠成大张量前先降内存），再参与引导
                video_frames.append(_cap_longest_side(
                    torch.from_numpy(np.ascontiguousarray(arr))))
            elif resampler is not None:
                for rf in resampler.resample(frame):
                    # to_ndarray 返回 [声道,样本]，且会按实际样本裁剪——
                    # 直接用 planes buffer 会连带对齐填充读到多余样本
                    nd = rf.to_ndarray()
                    audio_parts.append(torch.from_numpy(
                        np.ascontiguousarray(nd)))
        if resampler is not None:
            for rf in resampler.resample(None):  # 冲刷重采样器尾部
                nd = rf.to_ndarray()
                audio_parts.append(torch.from_numpy(
                    np.ascontiguousarray(nd)))
    finally:
        container.close()
    if not video_frames:
        raise ValueError("h3_motion_context: %s 未解码到任何视频帧。" % path)
    pixels = torch.stack(video_frames, 0).float().div_(255.0)
    waveform = None
    if audio_parts:
        wave = torch.cat(audio_parts, 1)  # [声道, L]
        if wave.shape[0] == 1:  # 单声道提升为立体声
            wave = wave.repeat(2, 1)
        elif wave.shape[0] > 2:
            wave = wave[:2]
        waveform = wave.unsqueeze(0)  # [1,2,L]
    out = {"pixels": pixels, "waveform": waveform, "sample_rate": sample_rate}
    if clip_index is not None:
        out[CONTEXT_CLIP_INDEX_KEY] = clip_index
    return out


def _load_context_media(存储位置, 片段序号=1):
    """按 存储位置+片段序号 加载本地尾段媒体：序号 0（首片段）返回 first_clip 空标记；
    >0 按 存储位置/clip_%05d.mp4 加载，未找到时返回 file_not_found 空标记。
    两种空标记调用方均直通、不裁头。"""
    try:
        idx = int(片段序号)
    except (TypeError, ValueError):
        raise ValueError("h3_motion_context: 片段序号必须是整数，得到 %r"
                         % (片段序号,))
    if idx == 0:
        return {CONTEXT_EMPTY_MARKER: True,
                CONTEXT_EMPTY_REASON: "first_clip"}
    try:
        path = _clip_file_path(存储位置, idx)
    except FileNotFoundError:
        # 兜底：主「存储位置」目录缺失（参数错位）时，扫描 output 目录实际保存位置
        path = _find_clip_in_output(idx)
        if path is None:
            return {CONTEXT_EMPTY_MARKER: True,
                    CONTEXT_EMPTY_REASON: "file_not_found",
                    CONTEXT_EMPTY_REASON_DETAIL: str(idx)}
    return _load_media_file(path, clip_index=idx)


def _context_latent_fingerprint(存储位置, 片段序号, 手动上传, 潜空间模式=False):
    """IS_CHANGED 缓存指纹：手动上传用文件指纹；片段序号常量 0 输出确定性 0；
    否则对存储目录全部尾段媒体做综合指纹（内容变→指纹变→下游重跑；同内容→命中）；
    异常统一返回 NaN 保守重跑。潜空间模式下改为对 latent_*.safetensors 做指纹，
    并用对应的目录兜底扫描。"""
    if (手动上传 or "").strip():
        try:
            path = _resolve_manual_media_path(手动上传)
            st = os.stat(path)
            return "manual:%s:%s:%s" % (path, st.st_mtime_ns, st.st_size)
        except Exception:
            return float("NaN")
    try:
        if int(片段序号) == 0:
            return 0
    except (TypeError, ValueError):
        pass
    try:
        fp = _dir_fingerprint(_build_latent_load_path(存储位置)
                              if 潜空间模式 else _build_load_path(存储位置))
    except Exception:
        return float("NaN")
    # 与加载时的兜底扫描保持一致：主存储位置目录不存在（参数错位）时，
    # 改扫 output 目录下实际保存的文件做指纹，避免缓存漏跑/误缓存
    if fp.startswith("missing"):
        alt = (_find_latent_in_output(片段序号) if 潜空间模式
               else _find_clip_in_output(片段序号))
        if alt:
            try:
                st = os.stat(alt)
                return "found:%s:%d:%d" % (alt, st.st_mtime_ns, st.st_size)
            except OSError:
                return float("NaN")
    return fp


# ============================================================================
# 潜空间模式：本地 latent 存取（存储位置目录下 latent_%05d_.safetensors）
# ============================================================================

def _load_context_latent(存储位置, 片段序号=1, 手动上传=""):
    """上下文潜空间端口未连接时，按 存储位置+片段序号+手动上传 加载本地潜空间。

    - 手动上传非空：优先加载该文件，完全忽略存储位置与片段序号
      （即使序号为 0 也照样加载）；
    - 片段序号 0：链条第一个片段（无前序上下文），不读文件，输出带空标记；
    - 片段序号 >0：按 存储位置/latent_0000N_.safetensors 加载，文件未找到
      时输出带 file_not_found 空标记的潜空间，直通、不裁头。
    """
    if _st_load is None:
        raise RuntimeError("h3_motion_context: safetensors is not "
                           "available; cannot load latents.")
    manual_path = _resolve_manual_media_path(手动上传)
    if manual_path is not None:
        data = _st_load(manual_path)
        if "video" not in data or "audio" not in data:
            raise ValueError(
                "h3_motion_context: %s is not an h3_motion_context latent "
                "(missing video/audio streams). Was it saved by the stock "
                "Save Latent node instead?" % manual_path)
        return {"samples": [data["video"], data["audio"]]}
    try:
        idx = int(片段序号)
    except (TypeError, ValueError):
        raise ValueError("h3_motion_context: 片段序号必须是整数，得到 %r"
                         % (片段序号,))
    if idx == 0:
        return {"samples": [], CONTEXT_EMPTY_MARKER: True,
                CONTEXT_EMPTY_REASON: "first_clip"}
    try:
        path = _resolve_latent_path(_build_latent_load_path(存储位置), idx)
    except FileNotFoundError:
        # 主「存储位置」解析失败：兜底扫描 output 目录。可能「存储位置」
        # 参数因工作流错位而错误（如被保存为 1 而非 H3-Mubu），但文件
        # 实际由「H3 运动裁剪」保存在输出目录的某个子目录中，扫描即可命中。
        path = _find_latent_in_output(idx)
        if path is None:
            return {"samples": [], CONTEXT_EMPTY_MARKER: True,
                    CONTEXT_EMPTY_REASON: "file_not_found",
                    CONTEXT_EMPTY_REASON_DETAIL: str(idx)}
    data = _st_load(path)
    if "video" not in data or "audio" not in data:
        raise ValueError(
            "h3_motion_context: %s is not an h3_motion_context latent "
            "(missing video/audio streams). Was it saved by the stock "
            "Save Latent node instead?" % path)
    return {"samples": [data["video"], data["audio"]],
            CONTEXT_CLIP_INDEX_KEY: idx}


def _save_av_latent(latent, 存储位置, 片段序号):
    """将 AV 潜空间保存为 {存储位置}/latent_%05d_.safetensors。

    重新生成同一片段会覆盖自身的废弃文件，不会堆叠新文件。
    """
    if _st_save is None:
        raise RuntimeError("h3_motion_context: safetensors is not "
                           "available; cannot save latents.")
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "h3_motion_context: latent has no audio stream; wire the "
            "sampler output of an H3 AV graph.")
    video = parts[0].cpu().contiguous()
    audio = parts[1].cpu().contiguous()
    # 文件前缀固定为 latent，用户只能设置存储目录
    full_prefix = os.path.join(存储位置, "latent")
    folder, filename, _, _, _ = folder_paths.get_save_image_path(
        full_prefix, folder_paths.get_output_directory())
    os.makedirs(folder, exist_ok=True)
    # 片段序号 2 -> latent_00002_.safetensors
    path = os.path.join(folder, "%s_%05d_.safetensors"
                        % (filename, int(片段序号)))
    _st_save({"video": video, "audio": audio}, path,
             metadata={"format": "h3_motion_context_av_v1"})


def _tail_portion_latent(delivered, tail_len):
    """从交付潜空间（已裁头）的尾部切出至多 tail_len 帧的尾段潜空间。

    保存到本地的只取交付部分的「后段」而非整段：下一片段的运动上下文
    只需要尾部窗口，文件更小、目录指纹更快。切点对齐整 VRF 组边界
    （帧数 ≡ 0 mod 17），切出的尾段自身时序相位锚定在步 0，可被
    「H3 运动上下文」正常取尾部窗口；交付帧数不足时退化为
    保存整个交付潜空间。
    """
    video, audio = _streams_from_latent(delivered)[:2]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    total = _pixel_frames(int(video.shape[2]))
    m = max(0, int(tail_len))
    m -= m % 17
    cut = total - m if 0 < m < total else 0
    cut -= cut % 17  # 切点吸附整组边界（尾段长度随之微调，仍为可达帧数）
    if cut <= 0:
        return delivered
    # cut 为 17 的倍数 → 恰有整数步覆盖，切片后尾段相位锚定在步 0
    video_t = video[:, :, _steps_for_frames(cut):].clone()
    audio_t = audio[..., int(round(cut * FRAME_RESCALE)):].clone()
    # 与交付潜空间同款尾部对齐：音频步数收敛到尾段帧数 × 5/3（向上取整）
    want = int(math.ceil((total - cut) * FRAME_RESCALE))
    if int(audio_t.shape[-1]) > want:
        audio_t = audio_t[..., :want]
    samples = delivered["samples"]
    new_samples = (NestedTensor([video_t, audio_t])
                   if getattr(samples, "is_nested", False)
                   else [video_t, audio_t])
    out = dict(delivered)
    out["samples"] = new_samples
    return out


def _build_latent_load_path(存储位置):
    """把用户输入的存储位置转换为内部路径前缀（存储位置/latent）。

    用户只设置目录名（如 H3-Mubu），文件前缀 latent 内部固定、不可改。
    """
    loc = (存储位置 or "").strip().strip('"').strip("'")
    if not loc:
        loc = "H3-Mubu"
    return os.path.join(loc, "latent")


def _resolve_latent_path(path, clip_index=1):
    """把加载器的路径输入解析为具体文件。

    接受两种形式（绝对路径或相对 ComfyUI 输出文件夹）：
      1. 文件路径      直接加载该文件；
      2. 文件前缀      与保存节点同款（如 "H3-Mubu/latent"），
                       clip_index 选择 {prefix}_0000N_.safetensors，
                       使加载与保存可用完全相同的默认值。
    """
    p = (path or "").strip().strip('"').strip("'")
    if not p:
        p = "H3-Mubu/latent"
    candidates = [p, os.path.join(folder_paths.get_output_directory(), p)]
    for c in candidates:
        if os.path.isfile(c):
            return c
        # 按文件前缀解析：如 "H3-Mubu/latent" → 在 H3-Mubu/ 下找
        # latent_*.safetensors（与保存节点默认值一致）
        dir_part = os.path.dirname(c)
        prefix = os.path.basename(c)
        if dir_part and prefix and os.path.isdir(dir_part):
            return _resolve_prefix(dir_part, prefix, int(clip_index))
    raise FileNotFoundError(
        "h3_motion_context: %r is neither a file nor a file "
        "prefix (also tried relative to the ComfyUI output directory)." % p)


def _resolve_prefix(dir_part, prefix, idx):
    """按文件名前缀解析潜空间文件（与保存节点的 filename_prefix 一致）。

    例：prefix="latent", idx=2 → latent_00002_.safetensors；同时兼容
    云端导出带任意后缀的文件名（latent_00002_etaar_1786585381.safetensors）。
    """
    pat = re.compile(r"^%s_%05d(?:_[^.]*)?\.safetensors$"
                     % (re.escape(prefix), int(idx)))
    pat_clip = re.compile(r"^%s_clip%03d\.safetensors$"
                          % (re.escape(prefix), int(idx)))
    files = [os.path.join(dir_part, f) for f in os.listdir(dir_part)
             if pat.match(f) or pat_clip.match(f)]
    if not files:
        raise FileNotFoundError(
            "h3_motion_context: no saved latent for clip %d "
            "(no %s_%05d_.safetensors in %s)."
            % (idx, prefix, idx, dir_part))
    return max(files, key=os.path.getmtime)


def _find_latent_in_output(idx):
    """兜底搜索：主「存储位置」解析失败时，扫描 ComfyUI 输出目录下所有
    子目录，查找文件名匹配 latent_%05d_.safetensors 的文件。

    用于「H3 运动上下文」自动索引模式——该模式依赖「存储位置」参数，
    而工作流中该参数可能因节点合并/参数错位而错误（例如被保存为数字
    1 而非 H3-Mubu）。只要文件实际存在于输出目录任意子目录（即「H3
    运动裁剪」保存的位置），就能找到，消除"未找到片段 N 文件"。
    命中多个时取修改时间最新的一个。
    """
    try:
        idx_i = int(idx)
    except (TypeError, ValueError):
        return None
    target = "latent_%05d_" % idx_i
    out = folder_paths.get_output_directory()
    if not out or not os.path.isdir(out):
        return None
    try:
        subs = sorted(os.listdir(out))
    except OSError:
        return None
    for sub in subs:
        subp = os.path.join(out, sub)
        if not os.path.isdir(subp):
            continue
        try:
            files = [os.path.join(subp, f) for f in os.listdir(subp)
                     if f.startswith(target) and f.endswith(".safetensors")]
        except OSError:
            continue
        if files:
            return max(files, key=os.path.getmtime)
    return None


# ============================================================================
# H3 运动上下文：把上一片段尾部媒体固定为本片段的关键帧引导
# ============================================================================

class Yuan_H3MotionContext:
    """把上一片段尾部画面/音频固定为本片段开头。

    「衔接模式」决定数据路径：
    - 视频图像（默认）：以 resolved_frame_index=0 锚定，与官方 Add Guide
      同路径——画面经视频 VAE 编码、音频经音频 VAE 编码后追加进
      minimax_keyframes（可与官方引导节点自由混用）。画面编码前做往返
      偏色闭环补偿（按逐通道均值抵消视频 VAE 编码→解码的非恒等 DC 偏移，
      防链条逐段累积发黄，且不引入锐化）。
    - 潜空间：直接从上一片段的 H3 AV 潜空间尾部切片（跳过解码/重编码，
      无质量损失），内部锚点由布局补丁经 MC_KEY 承载。要求两片段分辨率
      一致、窗口为整 VRF 组（17 的倍数）。

    上下文来源由「模式」决定：上传 / 端口 / 自动索引（视频图像模式读
    .mp4，潜空间模式读 .safetensors）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "条件化": ("CONDITIONING", {
                    "tooltip": "正向条件化。本节点向其追加关键帧引导后输出，"
                               "可与官方 Add Guide 等 H3 条件节点串联。"}),
                "潜空间": ("LATENT", {
                    "tooltip": "本片段的 H3 AV 潜空间（采样器或空 latent 节点"
                               "输出）。仅读取形状（时长/分辨率/音频轨长度），"
                               "不修改其内容。"}),
                "衔接模式": (["视频图像", "潜空间"], {
                    "default": "视频图像",
                    "tooltip": "衔接数据路径二选一。视频图像——把上一片段"
                               "尾部的媒体文件（.mp4）解码后经视频 VAE 编码为"
                               "本片段开头的关键帧引导，与官方 Add Guide 同"
                               "路径，分辨率可不同（按本片段缩放）。潜空间"
                               "——直接从上一片段的 H3 AV 潜空间切片（.safetensors，"
                               "跳过解码/重编码，无质量损失），但要求两片段"
                               "分辨率一致、窗口须为整 VRF 组（17 的倍数）。"}),
                "模式": (["上传", "端口", "自动索引"], {
                    "default": "自动索引",
                    "tooltip": "上下文来源三选一：上传——仅用手动上传的媒体"
                               "文件；端口——仅用「上下文图像」「上下文音频」"
                               "（视频图像模式）或「上下文潜空间」（潜空间"
                               "模式）端口的连线（如直接连上一片段「H3 运动"
                               "裁剪」的输出）；自动索引——按 存储位置+"
                               "片段序号 自动加载本地保存的尾段媒体。"}),
                "启用上下文": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "总开关。关闭时条件化直通、不固定任何引导，"
                               "输出 \"0:尾段长度\"——本片段完全独立生成，"
                               "但「运动裁剪」的尾段仍按两窗口较大值保存，"
                               "供下一片段衔接。"}),
                "存储位置": ("STRING", {
                    "default": "H3-Mubu",
                    "tooltip": "自动索引模式下加载的目录名（ComfyUI 输出"
                               "文件夹下的子目录）。与「H3 运动裁剪」的"
                               "「存储位置」一致即可对应加载。"}),
                "片段序号": ("INT", {
                    "default": 1, "min": 0, "max": 9999,
                    "tooltip": "自动索引模式下加载的片段序号：设为上一片段"
                               "「H3 运动裁剪」保存时使用的相同序号即可对应"
                               "加载（clip_00002.mp4 这类文件）。0 表示"
                               "链条第一个片段，不加载、直通。"}),
                "上下文长度": (["5", "22", "39", "56"], {
                    "default": "5",
                    "tooltip": "固定到本片段开头的画面帧数，必须是 H3 引导"
                               "片段的合法长度（17k+5：5/22/39/56，与官方"
                               " Add Guide 的多帧引导一致），其他值向下吸附。"
                               "「运动裁剪」将从交付部分裁掉同等帧数，接缝处"
                               "画面从引导末端无缝续接。锚窗越长自由帧越易被"
                               "拉向上一段画面并逐段污染，默认 5 帧短锚接缝"
                               "依然无缝、污染最小；长锚仅强延续需求时用。"}),
                "音频上下文长度": (["0", "5", "22", "39", "56"], {
                    "default": "5",
                    "tooltip": "从上一片段尾部固定的音频时长（按帧数换算），"
                               "经音频 VAE 编码后随关键帧锚定在本片段开头，"
                               "交付音频从固定窗口末端无缝续接。0=不固定音频"
                               "（模型自由生成，固定的画面可能带出上一片段"
                               "场景的声音）。建议与「上下文长度」保持一致"
                               "（超出时按画面窗收窄），大于 0 时需连接 "
                               "audio_vae。"}),
                # —— 以下参数仅「潜空间」衔接模式生效，与上面两窗口互不影响 ——
                "潜空间上下文长度": (["17", "34", "51", "68"], {
                    "default": "34",
                    "tooltip": "仅「潜空间」模式生效：从前一片段潜空间延续的"
                               "画面帧数。每 17 帧是 H3 潜空间的一整个 VRF 组"
                               "（5 个潜空间步），只有整组长度才能从尾部切片"
                               "又被「运动裁剪」整组切回，接缝处像素与时间"
                               "完全对齐；非整组长度会打乱剩余潜空间的时序"
                               "相位、导致画面闪烁。更长窗口桥接力更强，但会"
                               "从交付片段头部扣除更多长度。"}),
                "潜空间音频长度": (["0", "17", "34", "51", "68"], {
                    "default": "17",
                    "tooltip": "仅「潜空间」模式生效：从前一片段潜空间尾部"
                               "取声音的帧数（按 24fps 画面换算，音频潜空间"
                               "按 40Hz 连续采样、无 VRF 分组），独立于画面"
                               "窗口。「运动裁剪」按画面/音频两窗口的较大值"
                               "整段移除。设为 0 时不取音频：连接 audio_vae "
                               "时改用真实编码的静音（潜空间填零不是静音），"
                               "避免固定的上一片段画面诱导模型带出上一片段"
                               "的声音；未连接时模型自由生成。"}),
                "衔接余量": (["0", "17", "34"], {
                    "default": "17",
                    "tooltip": "仅「潜空间」模式生效：在锚定窗口之外、可见"
                               "剪辑起点之前多留一段整组长度的「自由沉降区」，"
                               "模型在这段里不再被强制对齐上一片段尾帧，可"
                               "平滑过渡到全新内容；这段随「运动裁剪」整组"
                               "移除、不出现在交付剪辑里。设 0 时可见段紧贴"
                               "锚定窗（旧行为，贴边可能偏僵硬/抖动）。"}),
            },
            "optional": {
                "VAE": ("VAE", {
                    "tooltip": "H3 视频 VAE（仅「视频图像」模式使用）。把上一"
                               "片段尾帧编码为关键帧 latent，与官方 Add Guide "
                               "连接 image 的路径相同。"}),
                "audio_vae": ("VAE", {
                    "tooltip": "H3 音频 VAE。「视频图像」模式下「音频上下文"
                               "长度」大于 0 时必须连接，用于把上一片段尾部的"
                               "音频波形编码为关键帧音频潜空间；「潜空间」模式"
                               "下仅在「潜空间音频长度」为 0 时使用，把零波形"
                               "真实编码成静音潜空间。"}),
                "上下文图像": ("IMAGE", {
                    "tooltip": "仅「视频图像」模式 +「端口」模式：上一片段"
                               "尾部画面（如直接连上一片段「H3 运动裁剪」的"
                               "「图像」输出）。本节点取其尾部「上下文长度」"
                               "帧作引导。"}),
                "上下文音频": ("AUDIO", {
                    "tooltip": "仅「视频图像」模式 +「端口」模式：上一片段"
                               "尾部音频（如直接连上一片段「H3 运动裁剪」的"
                               "「音频」输出）。「音频上下文长度」大于 0 时"
                               "必须连接。"}),
                "上下文潜空间": ("LATENT", {
                    "tooltip": "仅「潜空间」模式 +「端口」模式：前一片段的"
                               "采样器输出潜空间（与连接到解码节点的相同）。"
                               "同时提供画面和声音，直接切片获取。必须与正在"
                               "生成的片段分辨率相同。"}),
                "手动上传": ("STRING", {
                    "default": "",
                    "tooltip": "上传模式下由「上传上下文」按钮写入的媒体文件"
                               "路径，也可手填（支持 input:/output:/temp: "
                               "前缀）：「视频图像」模式为 .mp4，「潜空间」"
                               "模式为 .safetensors。"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("条件化", "裁剪帧数")
    FUNCTION = "apply"
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("把上一片段尾部的画面与音频固定为本片段开头的引导，"
                   "支持两条数据路径：「视频图像」模式全面对齐官方 Add Guide "
                   "for MiniMax H3（图像经视频 VAE 编码、音频经音频 VAE 编码，"
                   "锚定在第 0 帧并追加进正向条件化）；「潜空间」模式直接从"
                   "上一片段 AV 潜空间尾部切片，跳过解码/重编码。上下文来源"
                   "支持 上传/端口/自动索引。裁剪帧数输出为字符串\"状态:长度\""
                   "（如 1:22），供「H3 运动裁剪」解析。")

    def apply(self, 条件化, 潜空间, 启用上下文=True, 模式="自动索引",
              存储位置="H3-Mubu", 片段序号=1, 上下文长度="5",
              音频上下文长度="5", 衔接模式="视频图像", 潜空间上下文长度="34",
              潜空间音频长度="17", 衔接余量="17", VAE=None, audio_vae=None,
              上下文图像=None, 上下文音频=None, 上下文潜空间=None,
              手动上传=""):
        if 衔接模式 == "潜空间":
            return self._apply_latent(
                条件化, 潜空间, 启用上下文, 模式, 存储位置, 片段序号,
                潜空间上下文长度, 潜空间音频长度, 衔接余量, audio_vae,
                上下文潜空间, 手动上传)
        # 引导窗口与音频窗口的请求值；无上下文/直通路径下，尾段保存长度
        # 仍按两窗口较大值输出，保证下一片段有可衔接的媒体文件
        g_req = _snap_guide_frames(int(上下文长度 or 0))
        a_req = int(音频上下文长度 or 0)
        idle_tail = max(g_req, a_req)
        if not 启用上下文:
            return {"result": (条件化, "0:%d" % idle_tail), "ui": {
                "h3_hint": "上下文已关闭，直通"}}
        # 上下文来源由「模式」决定：上传（只用上传文件）/ 端口（只用连线）/ 自动索引
        if 模式 == "上传":
            if not (手动上传 or "").strip():
                raise ValueError(
                    "h3_motion_context: 「上传」模式下需先通过「上传上下文」"
                    "按钮上传媒体文件（.mp4），或填写「手动上传」路径。")
            media = _load_media_file(_resolve_manual_media_path(手动上传),
                                     clip_index=None)
            hint = "已上传上下文文件"
        elif 模式 == "端口":
            if 上下文图像 is None:
                raise ValueError(
                    "h3_motion_context: 「端口」模式下「上下文图像」"
                    "端口必须连接。")
            waveform = None
            sample_rate = 32000
            if isinstance(上下文音频, dict):
                waveform = 上下文音频.get("waveform")
                sample_rate = int(上下文音频.get("sample_rate", 32000))
            media = {"pixels": 上下文图像, "waveform": waveform,
                     "sample_rate": sample_rate}
            hint = "已关联上下文"
        else:  # 自动索引
            media = _load_context_media(存储位置, 片段序号)
            if media.get(CONTEXT_EMPTY_MARKER):
                reason = media.get(CONTEXT_EMPTY_REASON, "first_clip")
                if reason == "file_not_found":
                    detail = media.get(CONTEXT_EMPTY_REASON_DETAIL, "?")
                    return {"result": (条件化, "0:%d" % idle_tail), "ui": {
                        "h3_hint": "未找到片段 %s 文件" % detail}}
                return {"result": (条件化, "0:%d" % idle_tail), "ui": {
                    "h3_hint": "片段\"0\"，直通"}}
            hint = "已关联片段 %s 文件" % media.get(
                CONTEXT_CLIP_INDEX_KEY, "?")

        # 仅读取本片段形状：按「通道数 24」从 AV 两流中识别视频流（不假设顺序），
        # 兼容 NestedTensor 与普通 (video, audio) 元组
        parts = _streams_from_latent(潜空间)
        video = None
        for pt in parts:
            v = pt if pt.ndim != 4 else pt.unsqueeze(0)
            if v.ndim == 5 and v.shape[1] == 24:
                video = v
                break
        if video is None:
            raise ValueError(
                "h3_motion_context: 未在 AV 潜空间中找到 24 通道视频流，流形状="
                "%s。请连接 MiniMax H3 采样器/空潜空间节点的输出。"
                % ([tuple(t.shape) for t in parts],))
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)
        track_steps = None
        for pt in parts:
            if pt.ndim >= 1 and pt.shape[0] == 1 and pt.ndim >= 3 and pt is not video:
                try:
                    track_steps = int(pt.shape[-1])
                except (TypeError, ValueError):
                    pass

        pixels = media["pixels"]
        # 媒体像素约定 [T,H,W,3]；仅去掉真实的批维（5D [B,T,H,W,3]）。
        # 4D 就是帧序列本身，不能截断——早期把 4D 误当批维截成首帧，正是
        # 「裁剪保存的尾段加载后只剩 1 帧/引导断裂」的根因之一。
        if pixels.ndim == 5:
            pixels = pixels[0]
        elif pixels.ndim == 3:  # 单帧 [H,W,3] 补帧轴，交由下方帧数校验兜底
            pixels = pixels[None]
        available = int(pixels.shape[0])
        # 引导长度：min(请求,可用) 后吸附 17k+5（官方对多帧引导同款向下裁剪）
        g = _snap_guide_frames(min(g_req, available))
        if g < 5:
            raise ValueError(
                "h3_motion_context: 上下文媒体仅有 %d 帧画面，至少需要 5 帧"
                "才能固定引导。" % available)
        if g > frame_count:
            raise ValueError(
                "h3_motion_context: 引导片段 %d 帧超出本片段总长 %d 帧。"
                % (g, frame_count))

        # 引导帧：取尾部 g 帧，按官方 Add Guide 同款 center 裁剪缩放到
        # 本片段分辨率，经视频 VAE 编码为关键帧 latent（编码时做逐通道均值
        # 往返偏色闭环补偿，切断链条逐段累积的发黄漂移）
        guide = _resize_guide(pixels[available - g:], width, height)
        if width % 16 or height % 16:
            raise ValueError(
                "h3_motion_context: 本片段分辨率 %dx%d 不是 16 的倍数，无法"
                "编码引导。请调整工作流的目标分辨率。"
                % (width, height))
        if VAE is None:
            raise ValueError(
                "h3_motion_context: 「视频图像」衔接模式需要连接 VAE（H3 视频"
                "VAE）以编码引导画面；若要跳过解码/重编码请把「衔接模式」"
                "改为「潜空间」。")
        try:
            keyframe = {"resolved_frame_index": 0,
                        "latent": _encode_guide_color_neutral(
                            VAE, guide, width, height, g)}
        except RuntimeError as e:
            raise RuntimeError(
                "h3_motion_context: VAE 编码引导片段失败。\n"
                "  引导帧形状 guide=%s（g=%d 帧, 目标分辨率 %dx%d）\n"
                "  本片段视频潜空间形状=%s（像素 %dx%d, 共 %d 帧）\n"
                "  原始错误：%s\n"
                "请检查「H3 运动上下文」的「潜空间」是否接自 H3 采样器输出、"
                "上下文的画面分辨率是否与片段一致（不一致时按本片段分辨率"
                "缩放）。"
                % (tuple(guide.shape), g, width, height,
                   tuple(video.shape), width, height, frame_count, e))

        # 音频引导：取尾部音频窗（帧数换算样本）经音频 VAE 编码，按官方规则裁到本片段音频轨剩余长度（frame_idx=0→全轨可用）
        a = 0
        waveform = media.get("waveform")
        if a_req > 0:
            if waveform is None or int(waveform.shape[-1]) < 1:
                raise ValueError(
                    "h3_motion_context: 「音频上下文长度」大于 0，但当前"
                    "上下文%s无音频可用。「端口」模式需连接「上下文音频」，"
                    "或把「音频上下文长度」设为 0。"
                    % ("" if 模式 == "端口" else "文件"))
            if audio_vae is None:
                raise ValueError(
                    "h3_motion_context: 「音频上下文长度」大于 0 时需连接 "
                    "audio_vae（H3 音频 VAE）。")
            if waveform.ndim == 2:
                waveform = waveform[None]
            sr = int(media.get("sample_rate") or 32000)
            avail_frames = int(waveform.shape[-1]) / float(sr) * FPS
            # 音频锚窗上限收窄到画面锚窗 g：音频引导不应覆盖出画面引导之外
            # （否则该区段只有音频、无画面可对应，且下方裁剪量 max(g,a) 会
            # 白白多裁 a-g 帧新画面）
            a = min(a_req, int(avail_frames), g)
            if a < 1:
                raise ValueError(
                    "h3_motion_context: 上下文音频可用时长不足 1 帧，无法固定"
                    "音频引导。")
            z = _tail_audio_latent(audio_vae, waveform, sr, a)
            if track_steps is not None and int(z.shape[-1]) > track_steps:
                z = z[..., :track_steps].clone()
            keyframe["audio_latent"] = z

        # 裁剪量：覆盖画面引导窗与音频窗的较大值——音频窗比画面窗长时只裁
        # 画面窗会让固定音频泄漏进交付部分
        cut = max(g, a)
        if cut >= frame_count:
            raise ValueError(
                "h3_motion_context: 裁剪量 %d 帧达到/超过本片段总长 %d 帧。"
                "请减小「上下文长度」或「音频上下文长度」。"
                % (cut, frame_count))

        # 追加进正向条件化（与官方 Add Guide 相同：读取已有列表、追加、
        # 整表写回，可与官方引导节点自由混用）
        keyframes = list(条件化[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)
        out = node_helpers.conditioning_set_values(
            条件化, {"minimax_keyframes": keyframes})
        if a > 0:
            hint += "（含音频）"
        return {"result": (out, "1:%d" % cut), "ui": {
            "h3_hint": hint}}

    def _apply_latent(self, 条件化, 潜空间, 启用上下文, 模式, 存储位置, 片段序号,
                      上下文长度, 音频上下文长度, 衔接余量, audio_vae,
                      上下文潜空间, 手动上传):
        """潜空间衔接：直接在上一片段的 AV 潜空间上切片。

        与「视频图像」路径的关键差别：不经过「解码→重编码」，因此不损失质量，
        代价是分辨率必须与本片段一致、窗口必须是整 VRF 组（17 的倍数）。把
        锚点放在片段内部（而非仅首/末帧）需要布局补丁；让关键帧与引用共存需要
        载荷补丁——两补丁都带 ABI 标记门控、先自测再安装，失败即拒绝运行。"""
        if not 启用上下文:
            # 直通：尾段保存长度固定一个整组（17 帧）
            return {"result": (条件化, "0:17"), "ui": {
                "h3_hint": "上下文已关闭，直通"}}
        # 上下文来源由「模式」决定：上传（只用上传文件）/ 端口（只用连线）/ 自动索引
        if 模式 == "上传":
            if not (手动上传 or "").strip():
                raise ValueError(
                    "h3_motion_context: 「上传」模式下需先通过「上传上下文」"
                    "按钮上传 .safetensors 潜空间文件，或填写「手动上传」路径。")
            上下文潜空间 = _load_context_latent(存储位置, 片段序号, 手动上传)
        elif 模式 == "端口":
            if 上下文潜空间 is None:
                raise ValueError(
                    "h3_motion_context: 「端口」模式下「上下文潜空间」"
                    "端口必须连接。")
        else:  # 自动索引
            上下文潜空间 = _load_context_latent(存储位置, 片段序号, "")
        # 第一个片段：上下文潜空间带空标记（片段序号 0 / 文件未找到），
        # 无前序上下文，条件化直通、裁 0，并在节点下方显示提示
        if isinstance(上下文潜空间, dict) and 上下文潜空间.get(
                CONTEXT_EMPTY_MARKER):
            reason = 上下文潜空间.get(CONTEXT_EMPTY_REASON, "first_clip")
            if reason == "file_not_found":
                detail = 上下文潜空间.get(CONTEXT_EMPTY_REASON_DETAIL, "?")
                return {"result": (条件化, "0:17"), "ui": {
                    "h3_hint": "未找到片段 %s 文件" % detail}}
            return {"result": (条件化, "0:17"), "ui": {
                "h3_hint": "片段\"0\"，直通"}}

        video = _video_from_latent(潜空间)
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)

        # 潜空间不可缩放：直接从上片段 AV latent 切片视频尾部（跳过解码/编码）
        src_video = _video_from_latent(上下文潜空间)
        if (int(src_video.shape[4]) * 16 != width
                or int(src_video.shape[3]) * 16 != height):
            # 分辨率不一致时无法在同一 latent 网格上拼接前段画面/声音，
            # 跳过上下文，本次作为独立片段直通（效果等同重启链条）
            return {"result": (条件化, "0:17"), "ui": {
                "h3_hint": "分辨率不一致，已跳过"}}
        _ensure_layout_patch()
        if int(src_video.shape[1]) != int(video.shape[1]):
            raise ValueError(
                "h3_motion_context: 上下文潜空间有 %d 个通道，本片段有 %d 个。"
                "它并非同一模型的 H3 视频潜空间。"
                % (int(src_video.shape[1]), int(video.shape[1])))

        available = _pixel_frames(int(src_video.shape[2]))
        n = min(int(上下文长度), available)
        if n < 1:
            raise ValueError("h3_motion_context: 上下文潜空间没有可锚定的帧。")
        # 切片前对齐到整组网格，使切出的帧正好是整 VRF 组覆盖的帧
        n = next(g for g in VIDEO_RUN_GRID if g <= n)

        # 衔接余量：在锚定窗之外、可见剪起点之前多留一段整组长度的沉降区，
        # 让模型不被强制对齐上片段尾帧、平滑过渡到全新内容
        margin = int(衔接余量 or 0)
        # 裁剪量取画面窗、音频窗、画面窗+衔接余量 的较大值并吸附整组：
        # 音频窗口可比画面窗口长，只裁画面窗会让固定音频泄漏进交付部分
        trim_frames = next(g for g in VIDEO_RUN_GRID
                           if g <= max(n, int(音频上下文长度), n + margin))

        if n >= frame_count:
            raise ValueError(
                "h3_motion_context: 要把 %d 帧锚定进一个只有 %d 帧的片段，"
                "锚定窗必须远小于片段总长。" % (n, frame_count))
        if trim_frames >= frame_count:
            raise ValueError(
                "h3_motion_context: 衔接余量使裁剪量 %d 帧达到/超过本片段总长 "
                "%d 帧。请减小「衔接余量」或「潜空间上下文长度」。"
                % (trim_frames, frame_count))
        if _steps_for_frames(n) is None:
            # 节点提供的窗口都是完整步数，到达这里说明网格变了
            raise RuntimeError(
                "h3_motion_context: %d 帧窗口不是整数个潜空间步。"
                "VIDEO_RUN_GRID 已与 VAE 不匹配；拒绝渲染以避免接缝错位。" % n)

        blocks, offsets, covered = _video_tail_from_latent(上下文潜空间, n)
        span = covered
        indices = ([o - span for o in offsets] if ANCHOR_MODE == "before"
                   else list(offsets))
        keyframes = [{"resolved_frame_index": 0, MC_KEY: p, "latent": blk}
                     for p, blk in zip(indices, blocks)]
        values = {"minimax_keyframes": keyframes,
                  "minimax_frame_count": frame_count}

        # 音频上下文：大于 0 时切上下文潜空间尾部声音；为 0 且连接音频 VAE 时
        # 固定真实编码的静音（潜空间填零不是静音）。H3 是联合音视频模型：
        # 不装音频引用时，固定的上一片段画面会诱导模型为头部配出上一片段的
        # 声音并延续进交付部分（污染）；静音窗口把音频上下文锚定为「上一片段
        # 以无声结尾」，交付部分只跟随本片段提示词生成全新音频
        _ensure_payload_patch()
        audio_ref = None
        audio_silent = False
        if int(音频上下文长度) > 0:
            audio_latent, ref_audio_t, _overhang = _audio_tail_from_latent(
                上下文潜空间, int(音频上下文长度))
            audio_ref = {"kind": "audio", "ref_audio_t": ref_audio_t,
                         "audio_latent": audio_latent}
        elif audio_vae is not None:
            ref_audio_t = int(round(span * FRAME_RESCALE))
            audio_ref = {"kind": "audio", "ref_audio_t": ref_audio_t,
                         "audio_latent": _silence_audio_latent(
                             audio_vae, ref_audio_t)}
            audio_silent = True
        if audio_ref is not None:
            if AUDIO_MODE == "timeline":
                # 音频窗口尾端与「运动裁剪」的音频裁剪量用同一表达式，
                # 使固定窗口整体落在被裁头部内：既不泄漏（固定内容越过交付
                # 边界）也不误删（新生成音频被当作固定段）
                end_frame = float(trim_frames if ANCHOR_MODE == "head" else 0)
                end_coord = int(round(end_frame * FRAME_RESCALE))
                audio_ref[MC_AUDIO_KEY] = end_coord / FRAME_RESCALE
            # APPEND 而非赋值：Ref2VA 条件化可能已携带图自身的引用块，
            # 赋值会替换全部；用第二次调用让关键帧值先落位
            out = node_helpers.conditioning_set_values(条件化, values)
            out = node_helpers.conditioning_set_values(
                out, {"minimax_refs": [audio_ref]}, append=True)
        else:
            # 音频上下文为 0 且未连接音频 VAE：不安装音频引用，模型自由生成
            out = node_helpers.conditioning_set_values(条件化, values)

        trim = trim_frames if ANCHOR_MODE == "head" else 0
        trim_str = ("1:%d" % trim) if trim > 0 else "0:17"
        clip_idx = (上下文潜空间.get(CONTEXT_CLIP_INDEX_KEY)
                    if isinstance(上下文潜空间, dict) else None)
        hint = (("已关联片段 %s 文件" % clip_idx) if clip_idx is not None
                else "已关联上下文")
        if audio_silent:
            hint += "，音频上下文=静音"
        return {"result": (out, trim_str), "ui": {"h3_hint": hint}}

    @classmethod
    def IS_CHANGED(cls, 模式="自动索引", 存储位置="H3-Mubu", 片段序号=1,
                   手动上传="", 衔接模式="视频图像", **kwargs):
        # IS_CHANGED 只能拿 widget 值、无法感知端口连线：端口模式返回 0、按输入数据
        # 变化正常重跑；上传/自动索引模式按本地文件指纹判定（潜空间模式扫
        # latent_*.safetensors，视频图像模式扫 clip_*.mp4）。**kwargs 兼容其余端口值。
        if 模式 == "端口":
            return 0
        return _context_latent_fingerprint(存储位置, 片段序号, 手动上传,
                                           潜空间模式=(衔接模式 == "潜空间"))


# ============================================================================
# H3 运动裁剪：解码 AV 潜空间为图像+音频，裁头输出并保存尾段媒体
# ============================================================================

class Yuan_H3MotionContextTrim:
    """两种衔接模式下的 AV 潜空间两段式裁切，输出槽随模式切换。

    两模式共用第一次裁切的语义：按「裁剪帧数」"状态:长度"从头裁掉被固定窗口
    覆盖的部分，再按「保存到本地」开关把交付部分尾部的一段存盘，供下一片段
    「H3 运动上下文」加载衔接（可端口直连，也可按 存储位置+片段序号 自动索引）。

    「视频图像」模式：把 AV 潜空间解码为图像+音频后处理——像素域裁切无 VRF
    相位约束，任意帧数均可干净裁掉，音频按 24fps→采样率同步换算；尾段以无损
    mp4（clip_%05d.mp4）保存，输出 IMAGE+AUDIO。解码→重编码即官方 Add Guide
    的条件来源路径，天然重置采样器原始潜空间逐链累积的亮度漂移。

    「潜空间」模式：不解码，直接在 AV 潜空间上裁切——头部整组裁掉（帧数向下
    吸附到 17 的倍数，避免打乱剩余潜空间的 VRF 相位导致闪烁），音频按
    40Hz/24fps = 5/3 同步裁步，尾段以 safetensors（latent_%05d_.safetensors）
    保存，输出 LATENT（直连下一片段「H3 运动上下文」的「上下文潜空间」）。
    跳过解码/重编码无质量损失，但要求两片段分辨率一致。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "潜空间": ("LATENT", {
                    "tooltip": "H3 采样器的 AV 潜空间（同时含视频流与音频"
                               "流）。「视频图像」模式解码后裁切，「潜空间」"
                               "模式直接在其上裁切。"}),
                # 「衔接模式」放在第一个 widget，与「H3 运动上下文」的同名参数位置一致
                "衔接模式": (["视频图像", "潜空间"], {
                    "default": "视频图像",
                    "tooltip": "衔接数据路径二选一，与「H3 运动上下文」的"
                               "同名参数保持一致。视频图像——解码为图像+音频"
                               "后再裁切，输出 IMAGE+AUDIO，尾段存 .mp4，"
                               "分辨率可与下一片段不同；潜空间——直接裁切 AV "
                               "潜空间，输出 LATENT，尾段存 .safetensors，"
                               "无质量损失但要求两片段分辨率一致。"}),
                "片段序号": ("INT", {
                    "default": 1, "min": 1, "max": 9999,
                    "tooltip": "本片段在链条中的序号。「视频图像」模式设为2"
                               "保存到 clip_00002.mp4，「潜空间」模式保存到"
                               "latent_00002_.safetensors，重复生成覆盖原"
                               "文件；下一片段「H3 运动上下文」的「片段序号」"
                               "设相同值即可对应加载。"}),
                "保存到本地": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "尾段保存总开关。开启按裁剪帧数长度保存尾段"
                               "（1:22→22帧，0:22→22帧，未启用也保存供"
                               "衔接）；关闭仅输出、不生成文件，不影响输出"
                               "端口。「裁剪帧数」端口未接入时不裁切也不"
                               "保存，本开关不生效。"}),
                "存储位置": ("STRING", {
                    "default": "H3-Mubu",
                    "tooltip": "保存在 ComfyUI 输出文件夹下的子目录名。"
                               "下一片段「H3 运动上下文」使用相同的存储位置"
                               "即可对应加载。"}),
            },
            "optional": {
                # forceInput 输入在前端不会生成同名 widget（只建数据端口），端口未接入时
                # 该字段根本不会出现在执行提示里——声明为 required 会被判定「缺必填输入」
                # 直接报错。故放 optional：未接入时后端拿到 None，按「未接入」处理（不裁
                # 头、不保存尾段、不报错）。放在 optional 首位，使数据端口序号保持为
                # 潜空间(0)/裁剪帧数(1)/VAE(2)/audio_vae(3)，旧工作流的连线不错位。
                "裁剪帧数": ("STRING", {
                    "forceInput": True,
                    "tooltip": "强制输入端口（不可改），连接「H3 运动上下文」"
                               "的裁剪帧数输出：1:22=启用上下文（头部裁22帧、"
                               "尾段保存22帧）；0:22=未启用/首片段（不裁头、"
                               "尾段仍保存22帧供衔接）。也兼容纯数字（按启用"
                               "语义：裁头与尾段等长）。未接入时不裁头、不"
                               "保存尾段（等于不保存本地），也不会报错。"}),
                "VAE": ("VAE", {
                    "tooltip": "H3 视频 VAE（仅「视频图像」模式使用）。把 AV "
                               "潜空间的视频流解码为像素帧。"}),
                "audio_vae": ("VAE", {
                    "tooltip": "H3 音频 VAE（仅「视频图像」模式使用）。把 AV "
                               "潜空间的音频流解码为波形。"}),
            },
        }

    # 输出槽类型/顺序由前端按「衔接模式」重建（视频图像：IMAGE+AUDIO；
    # 潜空间：LATENT）。此处声明为 AnyType 通配，避免静态声明与实际输出槽
    # 不一致；隐含模式（未加载前端脚本）下第 0 槽即潜空间输出。
    RETURN_TYPES = ("*", "*")
    RETURN_NAMES = ("图像", "音频")
    FUNCTION = "trim"
    OUTPUT_NODE = True
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("按「H3 运动上下文」输出的裁剪帧数字符串\"状态:长度\"对 H3 "
                   "采样器的 AV 潜空间做两段式裁切，尾段存盘供下一片段衔接。"
                   "「视频图像」模式解码为图像+音频后按像素裁切，输出"
                   "图像/音频；「潜空间」模式直接按整 VRF 组裁切，输出潜空间。")

    def trim(self, 潜空间, 裁剪帧数=None, 片段序号=1, 保存到本地=True,
             存储位置="H3-Mubu", 衔接模式="视频图像", VAE=None, audio_vae=None):
        if 衔接模式 == "潜空间":
            return (self._trim_latent(潜空间, 裁剪帧数, 片段序号,
                                      保存到本地, 存储位置),)
        if VAE is None:
            raise ValueError(
                "h3_motion_context: 「视频图像」衔接模式需要连接 VAE（H3 视频"
                "VAE）以解码 AV 潜空间；若要直接裁切潜空间请把「衔接模式」"
                "改为「潜空间」。")
        if audio_vae is None:
            raise ValueError(
                "h3_motion_context: 「视频图像」衔接模式需要连接 audio_vae"
                "（H3 音频 VAE）以解码音频流；若要直接裁切潜空间请把"
                "「衔接模式」改为「潜空间」。")
        # 解析裁剪帧数字符串"状态:长度"："1:22"→裁头22帧且尾段存22帧；"0:22"→不裁头、
        # 尾段仍存22帧供衔接；兼容纯数字手填输入（按启用语义：n=tail=该值）。
        # 端口未接入：forceInput 输入由前端直接建数据端口、没有同名 widget，未连线时
        # 该字段不会出现在执行提示里，后端收到 None → 归一化为空串，落到下面的纯数字
        # 分支得 n=tail=0：不裁头、不保存尾段（等于不保存本地），也不会报错。
        s = "" if 裁剪帧数 is None else str(裁剪帧数).strip()
        if ":" in s:
            state_str, _, len_str = s.partition(":")
            try:
                state = int(state_str or "1")
            except ValueError:
                state = 1
            try:
                maxlen = int(len_str or "0")
            except ValueError:
                maxlen = 0
            if state:
                n, tail = maxlen, maxlen
            else:
                n, tail = 0, maxlen
        else:
            try:
                n = int(float(s or 0))
            except ValueError:
                n = 0
            tail = n
        n = max(0, n)
        tail = max(0, tail)
        # 像素域裁切：任意帧数均可，无需吸附整组（无 VRF 相位约束）
        parts = _streams_from_latent(潜空间)
        if len(parts) < 2:
            raise ValueError(
                "h3_motion_context: 裁剪需要含视频和音频两流的 AV 潜空间，"
                "得到 %d 个流。请连接 H3 采样器的输出。"
                % len(parts))
        video, audio = parts[0], parts[1]
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)
        # 解码：视频流 → [T,H,W,3] 像素帧；音频流 → [1,2,L] 波形
        # （VAE 包装器 decode 返回 [B,L,2]，movedim 回 ComfyUI AUDIO
        # 约定的 [B,声道,样本]）
        pixels = VAE.decode(video)
        if pixels.ndim == 5:
            pixels = pixels[0]
        total = int(pixels.shape[0])
        if n >= total:
            raise ValueError(
                "h3_motion_context: asked to trim %d frames from a %d frame "
                "clip" % (n, total))
        sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        wave = audio_vae.decode(audio)
        if wave.ndim == 3:
            wave = wave.movedim(1, -1)  # [1,L,2] → [1,2,L]
        elif wave.ndim == 2:
            wave = wave[None].movedim(1, -1)  # [L,2] → [1,2,L]
        audio_cut = int(round(n / float(FPS) * sample_rate))
        audio_cut = min(audio_cut, int(wave.shape[-1]))
        delivered_pixels = pixels[n:]
        delivered_wave = wave[..., audio_cut:]
        # 第二次裁切（受「保存到本地」开关控制）：在交付部分尾部按解析长度再切后段存盘，
        # 供下一片段「H3 运动上下文」取尾部窗口；交付帧数不足时保存整个交付部分。
        if 保存到本地 and tail > 0:
            t_frames = min(tail, total - n)
            tail_pixels = delivered_pixels[delivered_pixels.shape[0] - t_frames:]
            tail_samples = int(round(t_frames / float(FPS) * sample_rate))
            tail_samples = min(tail_samples, int(delivered_wave.shape[-1]))
            tail_wave = (delivered_wave[..., delivered_wave.shape[-1] - tail_samples:]
                         if tail_samples > 0 else delivered_wave[..., :0])
            _save_av_media(tail_pixels, tail_wave, sample_rate,
                           存储位置, 片段序号)
        return (delivered_pixels,
                {"waveform": delivered_wave, "sample_rate": sample_rate})

    def _trim_latent(self, 潜空间, 裁剪帧数=None, 片段序号=1,
                     保存到本地=True, 存储位置="H3-Mubu"):
        # 「潜空间」模式：不解码，直接在 AV 潜空间上裁切。解析裁剪帧数字符串
        # "状态:长度"："1:34"=启用上下文 → 头部裁 34 帧、尾段保存 34 帧；
        # "0:17"=未启用/无上下文 → 不裁头、尾段仍保存 17 帧供衔接。
        # 兼容纯数字输入（手填）→ 按启用语义：n=tail=该值
        # 端口未接入（后端收到 None）→ 归一化为空串：n=tail=0，不裁头、不保存尾段，
        # 也不会报错。
        s = "" if 裁剪帧数 is None else str(裁剪帧数).strip()
        if ":" in s:
            state_str, _, len_str = s.partition(":")
            try:
                state = int(state_str or "1")
            except ValueError:
                state = 1
            try:
                maxlen = int(len_str or "0")
            except ValueError:
                maxlen = 0
            if state:
                n, tail = maxlen, maxlen
            else:
                n, tail = 0, 17
        else:
            try:
                n = int(float(s or 0))
            except ValueError:
                n = 0
            tail = n
        n = max(0, n)
        # 向下吸附到最近的整组（17 的倍数）：非整组裁切会打乱剩余潜空间的
        # VRF 相位导致闪烁；小于 17 时吸附为 0（不裁剪，原样直通）
        n -= n % 17
        tail = max(0, tail)
        tail -= tail % 17
        parts = _streams_from_latent(潜空间)
        if len(parts) < 2:
            raise ValueError(
                "h3_motion_context: 裁剪需要含视频和音频两流的 AV 潜空间，"
                "得到 %d 个流。请连接 H3 采样器的输出。"
                % len(parts))
        video, audio = parts[0], parts[1]
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)
        total = _pixel_frames(int(video.shape[2]))
        if n >= total:
            raise ValueError(
                "h3_motion_context: asked to trim %d frames from a %d frame clip"
                % (n, total))
        # 前 n 帧覆盖的视频 latent 步数（每步覆盖 1/4/4/4/4 帧交替）
        k, removed = 0, 0
        while removed < n:
            removed += FRAME_PER_TOKEN[k % 5]
            k += 1
        video = video[:, :, k:].clone()
        # 音频流按 40Hz/24fps = 5/3 同步裁掉头部步数
        audio_cut = int(round(removed * FRAME_RESCALE))
        audio = audio[..., audio_cut:].clone()
        # 尾部对齐固定开启：把音频潜空间截断到恰好等于剩余帧数 × 5/3 步，
        # 消除 H3 音频网格向上取整在每个衔接处累积的约 8ms 额外声音
        rem_frames = total - removed
        want = int(math.ceil(rem_frames * FRAME_RESCALE))
        have = int(audio.shape[-1])
        if have > want:
            audio = audio[..., :want]
        # 保持输入的流容器类型：NestedTensor 或普通 list
        samples = 潜空间["samples"]
        new_samples = (NestedTensor([video, audio])
                       if getattr(samples, "is_nested", False)
                       else [video, audio])
        out = dict(潜空间)
        out["samples"] = new_samples
        # 第二次裁切（保存与否仅由「保存到本地」开关控制）：在第一次裁头
        # 结果的尾部按解析出的长度再切一段（保留后段）保存到本地，供下一
        # 片段「H3 运动上下文」取尾部窗口（连线传入或本地自动加载）。未启用
        # 上下文（"0:17"）时裁 0 帧但尾段仍保存 17 帧，保证链条任何配置下
        # 下一片段都有可衔接的上下文文件
        if 保存到本地 and tail > 0:
            _save_av_latent(_tail_portion_latent(out, tail),
                            存储位置, 片段序号)
        return out


def _save_av_media(pixels, waveform, sample_rate, 存储位置, 片段序号):
    """把尾段媒体（像素帧+音频波形）混流存为 {输出}/{存储位置}/clip_%05d.mp4。

    视频用 libx264 无损模式（yuv444p + crf 0）作为引导中间格式：画面是下一片段
    VAE 重编码的引导源，任何有损伪影都会随链条逐段累积（画面污染）；无损保存把
    该累积归零。音频仍 aac。重生成同一片段覆盖自身、不堆叠；文件名与「H3 运动
    上下文」自动索引加载约定一致（读取端对旧 yuv420 文件同样可解码）。
    """
    import av
    # pixels 约定 [T,H,W,3]；仅当仍带 [B,...] 批维（5D）时才塌缩首帧轴
    if pixels.ndim == 5:
        pixels = pixels[0]
    if waveform is not None and waveform.ndim == 2:
        waveform = waveform[None]
    has_audio = waveform is not None and int(waveform.shape[-1]) > 0
    loc = (存储位置 or "").strip().strip('"').strip("'") or "H3-Mubu"
    clip_dir = os.path.join(folder_paths.get_output_directory(), loc)
    os.makedirs(clip_dir, exist_ok=True)
    path = os.path.join(clip_dir, "%s_%05d%s" % (CLIP_FILE_PREFIX,
                                                 int(片段序号),
                                                 CLIP_FILE_EXT))
    # 原子写：先写同目录临时文件，编码成功后再 os.replace 覆盖目标——
    # 读取方永远不会读到半截文件；失败时旧文件保留
    tmp_path = "%s.tmp%d" % (path, os.getpid())
    try:
        os.remove(tmp_path)  # 清理上一次异常遗留的临时文件
    except OSError:
        pass
    h, w = int(pixels.shape[1]), int(pixels.shape[2])
    video_u8 = (pixels.clamp(0.0, 1.0).mul(255.0).round_()
                .to(torch.uint8).cpu().numpy())
    # 写入模式靠文件扩展名推断封装格式；tmp 后缀 .tmp<pid> 无法识别，
    # 必须显式指定 format="mp4"
    container = av.open(tmp_path, mode="w", format="mp4")
    try:
        # 两个流须先于任何写包建好（av 18 在已有时间戳后再加流会报
        # "Cannot rebase to zero time"）；先建后按序编码并无冲突
        vstream = container.add_stream("libx264", rate=FPS)
        vstream.width = w
        vstream.height = h
        # 无损引导中间格式：h264 无损仅支持 4:4:4（yuv444p）+ qp 0。
        # crf 16/yuv420p 的量化与色度抽样伪影会随多段再生逐段累积成画面污染，
        # 故引导源用 crf 0 无损（文件较大，但仅尾段短窗、且不对外分发）
        vstream.pix_fmt = "yuv444p"
        vstream.options = {"crf": "0", "preset": "slow"}
        astream = None
        if has_audio:
            astream = container.add_stream("aac", rate=int(sample_rate))
            astream.layout = "stereo"
        for i in range(video_u8.shape[0]):
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(video_u8[i]), format="rgb24")
            for pkt in vstream.encode(frame):
                container.mux(pkt)
        for pkt in vstream.encode():
            container.mux(pkt)
        if astream is not None:
            wave = waveform[0]
            if wave.shape[0] == 1:
                wave = wave.repeat(2, 1)
            elif wave.shape[0] > 2:
                wave = wave[:2]
            wave = wave.to(torch.float32).cpu()
            total = int(wave.shape[-1])
            chunk = int(sample_rate) // 10  # 0.1s 一块，控内存
            n = 0
            while n < total:
                seg = wave[:, n:n + chunk]
                aframe = av.AudioFrame(format="fltp", layout="stereo",
                                       samples=int(seg.shape[-1]))
                aframe.sample_rate = int(sample_rate)
                for ch in range(2):
                    aframe.planes[ch].update(
                        np.ascontiguousarray(seg[ch].numpy()))
                for pkt in astream.encode(aframe):
                    container.mux(pkt)
                n += chunk
            for pkt in astream.encode():
                container.mux(pkt)
    finally:
        container.close()
    # 封装全部成功且容器已关闭落盘后，才将临时文件原子替换为正式文件；
    # 中途任何异常都会在上方上抛（跳过此行），旧文件保持完整可用
    os.replace(tmp_path, path)


# ============================================================================
# 存储位置/文件名解析（约定：存储位置目录下 clip_%05d.mp4）
# ============================================================================

def _clip_file_path(存储位置, idx):
    """返回 clip_%05d.mp4 绝对路径（不存在抛 FileNotFoundError，由调用方兜底扫描 output 目录）。"""
    loc = (存储位置 or "").strip().strip('"').strip("'") or "H3-Mubu"
    path = os.path.join(folder_paths.get_output_directory(), loc,
                        "%s_%05d%s" % (CLIP_FILE_PREFIX, int(idx),
                                       CLIP_FILE_EXT))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "h3_motion_context: no clip media for index %d (%s)."
            % (int(idx), path))
    return path


def _find_clip_in_output(idx):
    """兜底搜索：主「存储位置」解析失败时（工作流参数可能错位），扫描 output 目录
    全部子目录找 clip_%05d.mp4，命中多个取 mtime 最新的。"""
    try:
        idx_i = int(idx)
    except (TypeError, ValueError):
        return None
    target = "%s_%05d%s" % (CLIP_FILE_PREFIX, idx_i, CLIP_FILE_EXT)
    out = folder_paths.get_output_directory()
    if not out or not os.path.isdir(out):
        return None
    try:
        subs = sorted(os.listdir(out))
    except OSError:
        return None
    for sub in subs:
        subp = os.path.join(out, sub)
        if not os.path.isdir(subp):
            continue
        try:
            files = [os.path.join(subp, f) for f in os.listdir(subp)
                     if f == target]
        except OSError:
            continue
        if files:
            return max(files, key=os.path.getmtime)
    return None


def _build_load_path(存储位置):
    """存储位置参数 → 目录级指纹前缀（用户只设目录名，clip 前缀内部固定、不可改）。"""
    loc = (存储位置 or "").strip().strip('"').strip("'") or "H3-Mubu"
    return os.path.join(loc, CLIP_FILE_PREFIX)


def _dir_fingerprint(prefix_path):
    """目录级综合指纹：对 prefix_path 所在目录下所有 clip_*.mp4/.safetensors 按
    「文件名+mtime+size」哈希（仅读元数据），任一文件增/删/改都改变指纹。

    IS_CHANGED 专用：链接输入拿不到真实片段序号、无法定位单文件，故对整目录做指纹
    ——保存节点每次覆盖写同一路径（mtime 必变）→ 指纹变 → 下游重跑；同内容重试→命中。
    目录不存在返回确定性 "missing"（可缓存），首次保存出现文件后自然触发重跑。
    """
    p = (prefix_path or "").strip().strip('"').strip("'")
    if not p:
        p = "H3-Mubu/clip"
    h = hashlib.sha256()
    h.update(p.encode("utf-8"))
    # 候选目录顺序与 _clip_file_path 一致：先 output 目录下的原路径
    for c in (os.path.join(folder_paths.get_output_directory(), p), p):
        dir_part = os.path.dirname(c)
        prefix = os.path.basename(c)
        if dir_part and prefix and os.path.isdir(dir_part):
            files = sorted(f for f in os.listdir(dir_part)
                           if f.startswith(prefix)
                           and (f.endswith(CLIP_FILE_EXT)
                                or f.endswith(".safetensors")))
            for fname in files:
                h.update(fname.encode("utf-8"))
                try:
                    st = os.stat(os.path.join(dir_part, fname))
                    h.update(("%d:%d;" % (st.st_mtime_ns, st.st_size))
                             .encode("utf-8"))
                except OSError:
                    pass
            return "%s:%s" % (p, h.hexdigest())
    return "missing:%s" % p


# ============================================================================
# 手动上传上下文媒体：分块上传 .mp4（兼容旧版 .safetensors）到
# input/h3_motion_latent/
# ============================================================================

_MANUAL_UPLOAD_SUBDIR = "h3_motion_latent"


@PromptServer.instance.routes.post("/yuan_h3_motion_upload_latent")
async def _yuan_h3_motion_upload_latent(request):
    """接收「H3 运动上下文」节点手动上传的媒体文件（分块追加写入）。"""

    def _normalize(name):
        return os.path.basename(name)

    def _validate(file_path):
        upload_dir = os.path.join(folder_paths.get_input_directory(),
                                  _MANUAL_UPLOAD_SUBDIR)
        if not os.path.basename(file_path).lower().endswith(
                (".mp4", ".safetensors")):
            return web.json_response(
                {"error": "仅支持 .mp4 上下文媒体文件（兼容旧版 .safetensors）"},
                status=400)
        if not os.path.realpath(file_path).startswith(os.path.realpath(upload_dir)):
            return web.json_response({"error": "无效的文件名"}, status=400)
        return None

    def _response_name(stored):
        return "%s/%s" % (_MANUAL_UPLOAD_SUBDIR, stored)

    upload_dir = os.path.join(folder_paths.get_input_directory(),
                              _MANUAL_UPLOAD_SUBDIR)
    return await handle_chunk_upload(request, upload_dir,
                                     normalize_name=_normalize,
                                     validate=_validate,
                                     response_name=_response_name)


def _resolve_manual_media_path(手动上传):
    """解析「手动上传」媒体路径为绝对路径（空输入返回 None）。

    支持 input:/output:/temp: 前缀与绝对路径；无前缀时依次在 input、output 目录查找。
    """
    p = (手动上传 or "").strip().strip('"').strip("'")
    if not p:
        return None
    candidates = []
    matched = False
    for prefix, base in (
        ("input:", folder_paths.get_input_directory()),
        ("output:", folder_paths.get_output_directory()),
        ("temp:", folder_paths.get_temp_directory()),
    ):
        if p.startswith(prefix):
            candidates.append(os.path.join(base, p[len(prefix):].lstrip("/\\")))
            matched = True
            break
    if not matched:
        if os.path.isabs(p):
            candidates.append(p)
        else:
            candidates.append(os.path.join(
                folder_paths.get_input_directory(), p))
            candidates.append(os.path.join(
                folder_paths.get_output_directory(), p))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "h3_motion_context: 手动上传的媒体文件未找到：%r"
        "（支持 input:/output:/temp: 前缀、绝对路径；无前缀时在 input 与"
        " output 目录下查找）。" % p)


NODE_CLASS_MAPPINGS = {
    "Yuan_H3MotionContext": Yuan_H3MotionContext,
    "Yuan_H3MotionContextTrim": Yuan_H3MotionContextTrim,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_H3MotionContext": "H3 运动上下文",
    "Yuan_H3MotionContextTrim": "H3 运动裁剪",
}
