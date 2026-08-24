"""H3 运动上下文相关节点。

包含 3 个节点：H3 加载潜空间 / H3 运动上下文 / H3 运动裁剪。片段衔接时直接从上一片段潜空间切片视频与音频尾部，
跳过解码/重编码。布局补丁解除仅首/末帧关键帧锚点限制，载荷补丁让固定
视频与固定音频共存；两补丁均在节点首次运行时安装（避免影响无关 H3 图），
带 ABI 标记门控并先自测再提交，失败则拒绝运行并说明原因。
"""

import asyncio
import hashlib
import math
import os
import re

import folder_paths
import node_helpers
import torch

from server import PromptServer
from aiohttp import web

from comfy.nested_tensor import NestedTensor
import comfy.ldm.minimax.model as mm
import comfy.model_base as model_base

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:  # ComfyUI 总是自带 safetensors，此处仅是双保险
    _st_load = _st_save = None


# ============================================================================
# 共享 ABI 标记
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
    """锚定在像素帧 p 的关键帧时间坐标。

    首/末两端复用 stock 的精确表达式：与一般公式数学等价，但 stock 是
    逐步累加 latent_t 浮点值、一般公式只做一次乘法，末位位元（约7e-15）
    不同。逐位一致保证已存在的首/末帧图在补丁后构建出字节相同的坐标，
    自测才能保持严格。
    """
    if p == 0:
        return float(text_len)
    if frame_count is not None and p == frame_count - 1:
        return float(text_len) + sum(mm._video_t_spans(latent_t)) - mm.FRAME_RESCALE
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
                      keyframes=keyframes, refs=refs, frame_count=frame_count)
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
                          keyframes=keyframes, refs=refs, frame_count=frame_count)
        if fix:
            _fixup(lay, text_len, latent_t, frame_count, keyframes, refs)
        if move:
            _fixup_audio(lay, text_len, refs)
        return lay

    def cond_ts(lay):
        return [float(lay.position_ids[a, 0])
                for a, _, k in lay.segments if k == "cond"]

    # 1. stock 支持的两端锚点必须逐位一致
    stock_kf = [{"resolved_frame_index": 0},
                {"resolved_frame_index": frame_count - 1}]
    ours_kf = [{"resolved_frame_index": 0, MC_KEY: 0},
               {"resolved_frame_index": 0, MC_KEY: frame_count - 1}]
    a = build(keyframes=stock_kf)
    b = build(keyframes=ours_kf, fix=True)
    if a.position_ids.shape != b.position_ids.shape:
        raise RuntimeError("position_ids shape mismatch in self-test")
    if not torch.equal(a.position_ids, b.position_ids):
        bad = (a.position_ids != b.position_ids).any(dim=1).nonzero().flatten()
        raise RuntimeError("position mismatch at rows %s" % bad[:8].tolist())

    # 2. 连续锚点须在两端点界定的区间内严格递增
    run = [{"resolved_frame_index": 0, MC_KEY: i} for i in range(4)]
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
    """本文件的另一份拷贝是否已包装过该构造函数？返回 None/"same"/"other"。

    多个包会内置本补丁，后加载者若把先加载者的包装器当作原始实现去包装
    就会套上多层（各拷贝用已装好的版本自测，新旧行为互相校验会误拒绝）。
    按可信度递减做三项检查：带标记的拷贝（匹配版本，静默退出）；仅同名
    的包装器是旧拷贝或分支（先加载者决定支持范围，退出并说明）；其余占着
    构造函数位置的是别的包在补同一处（按 __module__ 归属判断，无法同时
    持有；被 __wrapped__ 隐藏的无法识别）。
    """
    cls = getattr(mm, "PackedLayout", None)
    init = getattr(cls, "__init__", None)
    if init is None:
        return None
    if getattr(init, PATCH_MARKER_LAYOUT, False):
        return "same"
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
    if who:
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
    """另一份拷贝是否已包装过 extra_conds？返回 None/"same"/"other"/"foreign"。

    检测逻辑与 _layout_already_patched 相同：标记只识别新到能设置它的
    拷贝，仅同名的包装器视为旧拷贝或分支（后加载者退出），其他包装者
    是别的包在补同一处则拒绝叠包。
    """
    fn = getattr(cls, "extra_conds", None)
    if fn is None:
        return None
    if getattr(fn, PATCH_MARKER_PAYLOAD, False):
        return "same"
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


# ============================================================================
# H3 潜空间常量与工具函数
# ============================================================================

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24  # H3 原生帧率；音频潜空间按 40Hz 采样，故 FRAME_RESCALE 为 5/3
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0

# 可干净裁剪的整组窗口长度。一个 VRF 组 = 5 个潜空间步覆盖 17 个像素帧
# (1+4+4+4+4)，整组窗口即 17m 帧 = 5m 步。非整组窗口虽可切片，但裁剪
# 会整步移除：剩余步数若非 5 的倍数，潜空间起点落在周期中间，VAE 解码
# 器按错误帧数读第一个 token，画面闪烁。故可选项均为 17 的倍数，其他值
# （如过期的已存值）向下吸附到最近的整组，保证固定段与裁剪一致；5 作为
# 退化子组片段的下限。
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
    """把 H3 AV 潜空间解包成所含的流。

    NestedTensor.__getitem__ 会把索引广播进每个内含张量，samples[0]
    会剥掉两流的批维；unbind() 才返回这一对。
    """
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
# 节点定义
# ============================================================================

class Yuan_H3MotionContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "条件化": ("CONDITIONING",),
                "潜空间": ("LATENT",),
                "上下文潜空间": ("LATENT", {
                    "tooltip": "前一片段的采样器输出潜空间（与你连接到解码"
                               "节点的相同）。同时提供画面和声音，直接切片"
                               "获取，跳过链条中每次链接都会损失少许质量的"
                               "解码和重编码过程。必须与正在生成的片段分辨率"
                               "相同。需要分离/合并 AV latent 时可使用"
                               "「RTX 视频放大 (H3)」节点。"
                               "链条的第一个片段无前序上下文时，将「H3 加载"
                               "潜空间」节点的片段序号设为 0，本端口接收其空"
                               "标记后条件化直通、输出 \"0:17\"（不裁头）。"}),
                "启用上下文": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "总开关。关闭时条件化直通输出：不固定画面"
                               "窗口、不安装音频引用（不打包任何上下文），"
                               "裁剪帧数输出 \"0:17\"——「运动裁剪」不裁头、"
                               "但尾段仍按 17 帧保存，适合生成独立片段或"
                               "排查衔接问题。开启时上下文长度/音频上下文"
                               "长度等参数才生效（输出 \"1:较大值\"）。"}),
                "上下文长度": (["17", "34", "51", "68"], {
                    "default": "17",
                    "tooltip": "从前一片段延续的画面帧数。每 17 帧是 H3 潜空间"
                               "的一整个 VRF 组（5 个潜空间步），只有整组长度"
                               "才能从尾部切片、又被「运动裁剪」整组切回，"
                               "衔接处像素与时间完全对齐；非整组长度会在裁剪后"
                               "打乱剩余潜空间的时序相位，导致画面闪烁。"
                               "17 帧仅勉强流畅，34 帧近乎无缝。更长的窗口"
                               "固定更多运动，但会从交付片段的头部扣除。"}),
                "音频上下文长度": (["0", "17", "34", "51", "68"], {
                    "default": "17",
                    "tooltip": "从上下文潜空间取尾部声音的帧数，独立于画面"
                               "窗口。窗口尾端与裁剪边界对齐，「运动裁剪」按"
                               "画面/音频两窗口的较大值整段移除。音频潜空间"
                               "按 40Hz 连续采样，"
                               "没有 VRF 分组，帧数按 24fps 画面换算"
                               "（40/24 ≈ 5/3）。设为 0 时不从上下文潜空间"
                               "取音频：连接音频 VAE 时固定窗口的内容改为"
                               "真实编码的静音（与画面窗口等长、恰好被裁掉），"
                               "交付部分生成不受上一片段声音污染的全新音频；"
                               "未连接音频 VAE 时不安装音频引用，模型自由"
                               "生成（固定的画面可能带出上一片段的声音）。"}),
            },
            "optional": {
                "audio_vae": ("VAE", {
                    "tooltip": "音频 VAE（H3 音频 VAE）。仅在音频上下文长度"
                               "为 0 时使用：零波形经它真实编码成静音潜空间"
                               "（潜空间直接填零不是静音），作为固定音频窗口"
                               "的内容——窗口与画面窗口等长、对齐被裁头部，"
                               "随「运动裁剪」整段移除，交付部分的音频由"
                               "模型跟随本片段提示词全新生成。音频上下文长度"
                               "大于 0 时不参与（音频直接从上下文潜空间切片）。"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("条件化", "裁剪帧数")
    FUNCTION = "apply"
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("将上一片段尾部的连续帧作为不可去噪的条件行固定下来，"
                   "让模型读取真实运动而非从单帧静态图中猜测。画面和声音"
                   "都直接从上一片段的潜空间中切片获取，跳过每次链接都会"
                   "损失少许质量的解码和重编码过程。裁剪帧数输出为字符串"
                   "\"状态:长度\"（如 1:34），供「运动裁剪」节点解析。")

    def apply(self, 条件化, 潜空间, 上下文潜空间, 启用上下文=True, 上下文长度="17",
              音频上下文长度="17", audio_vae=None):
        # 「上下文长度」与「音频上下文长度」都只针对上下文潜空间输入端口：
        # 前者取其尾部画面，后者取其尾部声音。两者都只固定到本片段头部、
        # 由「运动裁剪」整组移除，交付部分的画面与音频由模型全新生成。
        # 裁剪帧数输出为字符串"状态:长度"：状态 1=启用上下文（长度为画面/
        # 音频两固定窗口较大值并吸附整组），0=未启用或无上下文（裁 0 帧，
        # 但「运动裁剪」的尾段保存仍按 17 帧执行，供下一片段衔接）。
        # 总开关关闭：条件化直通、不打包任何上下文、输出 "0:17"，
        # 等效于本片段完全独立生成
        if not 启用上下文:
            return {"result": (条件化, "0:17"), "ui": {
                "h3_hint": "上下文已关闭，直通"}}
        # 第一个片段：上下文潜空间是「H3 加载潜空间」片段序号 0 的空标记或
        # 文件未找到的回退空标记。无前序上下文，条件化直通、裁 0，
        # 并通过 ui.h3_hint 在节点下方显示提示
        if isinstance(上下文潜空间, dict) and 上下文潜空间.get(
                Yuan_H3MotionContextLoadLatent.EMPTY_MARKER):
            reason = 上下文潜空间.get(
                Yuan_H3MotionContextLoadLatent.EMPTY_REASON, "first_clip")
            if reason == "file_not_found":
                detail = 上下文潜空间.get(
                    Yuan_H3MotionContextLoadLatent.EMPTY_REASON_DETAIL, "?")
                return {"result": (条件化, "0:17"), "ui": {
                    "h3_hint": "未找到片段 %s 文件" % detail}}
            return {"result": (条件化, "0:17"), "ui": {
                "h3_hint": "片段\"0\"，直通"}}
        anchor_mode = ANCHOR_MODE
        audio_mode = AUDIO_MODE
        上下文长度 = int(上下文长度)
        音频上下文长度 = int(音频上下文长度)

        video = _video_from_latent(潜空间)
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)

        # 潜空间不可缩放：直接从上片段 AV latent 切片视频尾部（跳过解码/编码）
        src_video = _video_from_latent(上下文潜空间)
        src_w = int(src_video.shape[4]) * 16
        src_h = int(src_video.shape[3]) * 16
        if src_w != width or src_h != height:
            # 分辨率不一致时无法在同一 latent 网格上拼接前段画面/声音，
            # 跳过上下文，本次作为独立片段直通（效果等同重启链条）
            return {"result": (条件化, "0:17"), "ui": {
                "h3_hint": "分辨率不一致，已跳过"}}
        _ensure_layout_patch()
        if int(src_video.shape[1]) != int(video.shape[1]):
            raise ValueError(
                "h3_motion_context: context_latent has %d channels, "
                "this clip has %d. That is not an H3 video latent from "
                "the same model."
                % (int(src_video.shape[1]), int(video.shape[1])))
        available = _pixel_frames(int(src_video.shape[2]))

        n = min(int(上下文长度), available)
        if n < 1:
            raise ValueError("h3_motion_context: no frames available to pin")

        # 在切片前对齐到整组网格，使切片的帧正好是整 VRF 组覆盖的帧
        run = next(g for g in VIDEO_RUN_GRID if g <= n)
        n = run

        # 裁剪量取画面/音频两固定窗口的较大值并吸附整组：音频窗口可比
        # 画面窗口长（如 17/68），只裁画面窗口会让固定音频泄漏进交付
        # 部分；按较大值整段裁掉，两流共用同一时间切点，不做分离差异
        # 裁切后合并
        trim_frames = next(
            g for g in VIDEO_RUN_GRID if g <= max(n, 音频上下文长度))

        if n >= frame_count:
            raise ValueError(
                "h3_motion_context: asked to pin %d frames into a %d frame clip. "
                "The pinned run must be a small fraction of the timeline."
                % (n, frame_count))

        if _steps_for_frames(n) is None:
            # 节点提供的窗口都是完整步数，到达这里说明网格变了
            raise RuntimeError(
                "h3_motion_context: a %d frame window is not a whole number "
                "of latent steps. VIDEO_RUN_GRID no longer matches the "
                "VAE; refusing rather than rendering a shifted join." % n)

        # 从 latent 切片视频尾部
        blocks, offsets, covered = _video_tail_from_latent(上下文潜空间, n)
        span = covered

        if anchor_mode == "before":
            indices = [o - span for o in offsets]
        else:
            indices = list(offsets)

        keyframes = []
        for p, blk in zip(indices, blocks):
            keyframes.append({
                # stock 仅接受 0 或 frame_count-1，真实位置经 MC_KEY
                # 携带，由布局补丁应用
                "resolved_frame_index": 0,
                MC_KEY: p,
                "latent": blk,
            })

        values = {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        }

        # 音频上下文：与画面窗口一样只处理上下文潜空间输入——大于 0 时
        # 切其尾部声音；为 0 时不从上下文潜空间取音频，连接音频 VAE 时
        # 固定窗口的内容改为真实编码的静音（潜空间填零不是静音）。静音
        # 窗口长度与画面窗口一致（round(span*5/3) 步，恰等于「运动裁剪」
        # 的音频裁剪量），对齐被裁头部并随裁剪整段移除。H3 是联合音视频
        # 模型：不装音频引用时，固定的画面（上一片段的场景）会诱导模型
        # 为头部配出上一片段的声音并延续进交付部分（污染）；静音窗口把
        # 音频上下文锚定为"上一片段以无声结尾"，模型衔接的是静音→全新
        # 内容，交付部分生成只跟随本片段提示词的全新音频
        _ensure_payload_patch()
        a_frames = int(音频上下文长度)
        audio_ref = None
        audio_silent = False
        if a_frames > 0:
            audio_latent, ref_audio_t, _overhang = _audio_tail_from_latent(
                上下文潜空间, a_frames)
            audio_ref = {
                "kind": "audio",
                "ref_audio_t": ref_audio_t,
                "audio_latent": audio_latent,
            }
        elif audio_vae is not None:
            audio_ref = {
                "kind": "audio",
                "ref_audio_t": int(round(span * FRAME_RESCALE)),
                "audio_latent": _silence_audio_latent(
                    audio_vae, int(round(span * FRAME_RESCALE))),
            }
            audio_silent = True
        if audio_ref is not None:
            if audio_mode == "timeline":
                # 音频窗口尾端与「运动裁剪」的音频裁剪量用同一表达式
                # (round(trim_frames*FRAME_RESCALE))，固定窗口整体落在
                # 被裁头部内：
                # 既不泄漏（固定内容越过交付边界）也不误删（新生成音频被
                # 当作固定段）。不用 overhang 外推结束点：overhang 是音频
                # 网格向上取整的分数余量（<1 步），先加再取整会把窗口推过
                # 裁剪边界最多 1 步（25ms），且是否越界取决于该片段的
                # overhang 是否 ≥0.5——124 帧直出 (0.33) 对齐、107 帧裁剪后
                # (0.67) 越界，表现为部分片段长度处衔接有可听错位。
                end_frame = float(trim_frames if anchor_mode == "head" else 0)
                end_coord = int(round(end_frame * FRAME_RESCALE))
                end_frame = end_coord / FRAME_RESCALE
                audio_ref[MC_AUDIO_KEY] = end_frame
            # APPEND 而非赋值：Ref2VA 条件化可能已携带图自身的引用块，
            # 赋值会替换全部；用第二次调用让关键帧值先落位
            out = node_helpers.conditioning_set_values(条件化, values)
            out = node_helpers.conditioning_set_values(
                out, {"minimax_refs": [audio_ref]}, append=True)
        else:
            # 音频上下文为 0 且未连接音频 VAE：不安装音频引用，模型为
            # 当前片段自由生成音频（固定的画面可能带出上一片段的声音）
            out = node_helpers.conditioning_set_values(条件化, values)

        trim = trim_frames if anchor_mode == "head" else 0
        # 输出字符串"状态:长度"：1=启用上下文（长度即两固定窗口较大值，
        # 头部裁切与尾段保存同长）；0=无上下文可裁（长度固定 17，
        # 仅作尾段保存长度）
        trim_str = ("1:%d" % trim) if trim > 0 else "0:17"
        # 提示已关联的片段序号（来自加载节点）
        clip_idx = (上下文潜空间.get(Yuan_H3MotionContextLoadLatent.CLIP_INDEX_KEY)
                    if isinstance(上下文潜空间, dict) else None)
        hint_text = ("已关联片段 %s 文件" % clip_idx) if clip_idx is not None else "已关联上下文"
        if audio_silent:
            hint_text += "，音频上下文=静音"
        return {"result": (out, trim_str), "ui": {
            "h3_hint": hint_text}}


class Yuan_H3MotionContextTrim:
    """直接在 H3 的 AV 潜空间上两段式裁切，并可选保存尾段。

    第一次裁切：按「裁剪帧数」字符串（「H3 运动上下文」自动输出的
    "状态:长度"，如 1:34=启用上下文、0:17=未启用）从头部整段裁掉多余
    部分——视频流裁掉覆盖前 n 帧的 latent 步，音频流按 40Hz/24fps =
    5/3 同步裁掉对应步数，两流共用同一时间切点，不做分离裁切后合并；
    输出裁头后的交付潜空间，不经过解码/转码。状态为 0 时裁 0 帧（直通）。
    第二次裁切：在交付潜空间的尾部按字符串中的长度再切一段（状态 1 时
    =裁剪长度；状态 0 时固定 17 帧），保留后段保存到本地（是否保存由
    「保存到本地」开关控制），供下一次运行加载衔接。音视频分离取窗是
    「H3 加载潜空间 → H3 运动上下文」的事情，本节点只做整段切片，
    潜空间输出以本节点的裁剪结果为主。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "潜空间": ("LATENT", {
                    "tooltip": "H3 采样器的 AV 潜空间（同时含视频流与音频流），"
                               "直接在其上裁切。"}),
                "裁剪帧数": ("STRING", {
                    "default": "0:17",
                    "forceInput": True,
                    "tooltip": "强制输入端口（不可修改）：连接「H3 运动"
                               "上下文」的裁剪帧数输出（字符串\"状态:长度\"）："
                               "\"1:34\"=启用上下文，头部裁34帧、尾段保存34帧；"
                               "\"0:17\"=未启用或首片段，不裁头、尾段仍保存"
                               "17帧。未连线时按默认值 \"0:17\" 处理。"}),
                "片段序号": ("INT", {
                    "default": 1, "min": 1, "max": 9999,
                    "tooltip": "本片段在链条中的序号。设为 2 时保存到"
                               "latent_00002_.safetensors，重复生成会覆盖原文件。"
                               "「加载潜空间」节点设相同序号即可对应加载。"}),
                "保存到本地": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "尾段保存总开关。开启时按裁剪帧数字符串中的"
                               "长度保存尾段（\"1:34\"→34 帧，\"0:17\"→17 帧，"
                               "未启用上下文也保存，供下一片段衔接）；关闭时"
                               "仅输出不生成文件。不影响潜空间输出端口。"}),
                "存储位置": ("STRING", {
                    "default": "H3-Mubu",
                    "tooltip": "保存在 ComfyUI 输出文件夹下的子目录名。"
                               "「加载潜空间」节点使用相同的存储位置即可"
                               "对应加载。"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    FUNCTION = "trim"
    OUTPUT_NODE = True
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("两段式潜空间裁切：按「H3 运动上下文」输出的裁剪帧数"
                   "字符串\"状态:长度\"自动判断——状态 1 时头部裁掉长度帧、"
                   "尾段保存同长度；状态 0（未启用/首片段）时不裁头、尾段"
                   "仍保存 17 帧。是否保存由「保存到本地」开关单独控制。")

    def trim(self, 潜空间, 裁剪帧数, 片段序号=1, 保存到本地=True, 存储位置="H3-Mubu"):
        # 解析「H3 运动上下文」输出的裁剪帧数字符串"状态:长度"：
        #   "1:34"=启用上下文 → 头部裁 34 帧，尾段保存 34 帧
        #   "0:17"=未启用/无上下文 → 不裁头，尾段仍保存 17 帧供衔接
        # 兼容纯数字输入（手填）→ 按启用语义：n=tail=该值
        s = str(裁剪帧数).strip()
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
                "得到 %d 个流。请连接 H3 采样器（或加载潜空间）的输出。"
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
        # 片段「加载潜空间 → 运动上下文」取尾部窗口。未启用上下文（"0:17"）
        # 时裁 0 帧但尾段仍保存 17 帧，保证链条任何配置下下一片段都有可
        # 衔接的上下文文件。音视频分离取窗由那边负责，本节点只做整段切片
        if 保存到本地 and tail > 0:
            _save_av_latent(_tail_portion_latent(out, tail),
                            存储位置, 片段序号)
        return (out,)


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
    「加载潜空间 → 运动上下文」正常取尾部窗口；交付帧数不足时退化为
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


def _build_load_path(存储位置):
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


def _dir_fingerprint(prefix_path):
    """目录级综合指纹：对 prefix_path 所在目录下所有 prefix_*.safetensors
    按「文件名 + mtime + size」计算指纹（仅读元数据、不读文件内容），
    任一文件新增/覆盖/删除都会改变指纹。

    用于「加载潜空间」的 IS_CHANGED：该阶段链接输入拿不到真实值，
    片段序号来自 GetNode/表达式链路时不可用，因此对
    整个存储目录做指纹——保存节点每次覆盖写入同一路径（mtime 必变）
    → 指纹变化 → 下游重跑；同一片段重试（内容不变）→ 缓存命中。
    目录不存在时返回确定性 "missing" 标记（可缓存），首次保存出现
    文件后指纹变化自然触发重跑。
    """
    p = (prefix_path or "").strip().strip('"').strip("'")
    if not p:
        p = "H3-Mubu/latent"
    h = hashlib.sha256()
    h.update(p.encode("utf-8"))
    # 候选目录顺序与 _resolve_latent_path 一致：先原路径，再 output 目录
    for c in (p, os.path.join(folder_paths.get_output_directory(), p)):
        dir_part = os.path.dirname(c)
        prefix = os.path.basename(c)
        if dir_part and os.path.isdir(dir_part):
            files = sorted(f for f in os.listdir(dir_part)
                           if f.startswith(prefix) and f.endswith(".safetensors"))
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
# 手动上传潜空间：分块上传 .safetensors 到 input/h3_motion_latent/
# ============================================================================

_MANUAL_UPLOAD_SUBDIR = "h3_motion_latent"


def _read_and_write_latent_chunk(file, file_path, mode):
    chunk_bytes = file.file.read()
    with open(file_path, mode) as f:
        f.write(chunk_bytes)


@PromptServer.instance.routes.post("/yuan_h3_motion_upload_latent")
async def _yuan_h3_motion_upload_latent(request):
    """接收「H3 加载潜空间」节点手动上传的潜空间文件（分块追加写入）。"""
    post = await request.post()
    file = post.get("file")
    filename = post.get("filename")
    chunk_index = int(post.get("chunk_index"))
    total_chunks = int(post.get("total_chunks"))

    upload_dir = os.path.join(folder_paths.get_input_directory(),
                              _MANUAL_UPLOAD_SUBDIR)
    os.makedirs(upload_dir, exist_ok=True)
    filename = os.path.basename(filename)
    if not filename.lower().endswith(".safetensors"):
        return web.json_response({"error": "仅支持 .safetensors 文件"}, status=400)
    file_path = os.path.join(upload_dir, filename)
    if not os.path.realpath(file_path).startswith(os.path.realpath(upload_dir)):
        return web.json_response({"error": "无效的文件名"}, status=400)

    mode = "ab" if chunk_index > 0 else "wb"
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _read_and_write_latent_chunk,
                               file, file_path, mode)

    if chunk_index == total_chunks - 1:
        return web.json_response(
            {"name": "%s/%s" % (_MANUAL_UPLOAD_SUBDIR, filename)})
    return web.json_response({"status": "ok"})


def _resolve_manual_latent_path(手动上传):
    """解析「手动上传」输入的潜空间文件路径，返回绝对路径；空输入返回 None。

    支持 input:/output:/temp: 前缀与绝对路径；无前缀时依次在 input、
    output 目录下查找（上传端点默认存 input/h3_motion_latent/）。
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
        "h3_motion_context: 手动上传的潜空间文件未找到：%r"
        "（支持 input:/output:/temp: 前缀、绝对路径；无前缀时在 input 与"
        " output 目录下查找）。" % p)


class Yuan_H3MotionContextLoadLatent:
    """为 context_latent 输入加载已保存的 H3 AV 潜空间。

    片段序号即"要延续的片段"：设为 2 加载 latent_00002_.safetensors
    （与裁剪节点的保存命名一致）；重掷某片段会覆盖其自身保存，废弃文件
    不堆积。输出仅用于运动上下文节点的「上下文潜空间」输入，不是可解码
    的潜空间，勿接入 VAE 解码。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "存储位置": ("STRING", {
                    "default": "H3-Mubu",
                    "tooltip": "与「H3 运动裁剪」节点相同的存储位置"
                               "（ComfyUI 输出文件夹下的子目录名）。"}),
                "片段序号": ("INT", {
                    "default": 1, "min": 0, "max": 9999,
                    "tooltip": "要加载的片段序号。设为 2 时加载"
                               "latent_00002_.safetensors，与「H3 运动裁剪」"
                               "节点的命名规则一致。设为 0 时表示链条的第一个"
                               "片段（无前序上下文）：不读取本地文件，输出空"
                               "标记，运动上下文节点识别后条件化直通、裁剪帧"
                               "数为 0。手动上传非空时本参数被忽略。"}),
                "手动上传": ("STRING", {
                    "default": "",
                    "tooltip": "通过「上传潜空间」按钮上传 .safetensors 潜空间"
                               "文件，或手动填写路径（支持 input:/output:/temp: "
                               "前缀、绝对路径；无前缀时先查 input 再查 output "
                               "目录）。非空时优先加载该文件，完全忽略存储位置"
                               "与片段序号（序号为 0 也照样加载）；留空时按"
                               "存储位置+片段序号正常加载。"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("加载由「H3 运动裁剪」节点保存的潜空间，"
                   "仅用于运动上下文节点的「上下文潜空间」输入。"
                   "片段序号设为 0 时表示第一个片段，不读取文件，"
                   "运动上下文节点将条件化直通。")

    # 标记键：片段序号 0 时输出带此标记的空 latent，上下文节点据此直通
    EMPTY_MARKER = "_h3_motion_context_empty"
    # 原因键：空标记的原因（first_clip / file_not_found），供上下文节点生成提示
    EMPTY_REASON = "_h3_motion_context_empty_reason"
    # 细节键：原因细节（如未找到的片段序号），供上下文节点生成提示
    EMPTY_REASON_DETAIL = "_h3_motion_context_empty_reason_detail"
    # 正常加载时携带的片段序号键，供上下文节点生成"已关联片段 N"提示
    CLIP_INDEX_KEY = "_h3_motion_context_clip_index"

    @classmethod
    def IS_CHANGED(cls, 存储位置, 片段序号=1, 手动上传=""):
        # 手动上传非空：以该文件的 mtime+size 为指纹，忽略存储位置与序号。
        # 文件未找到等异常返回 NaN：不缓存，保守每次重跑（load 会大声报错）。
        if (手动上传 or "").strip():
            try:
                path = _resolve_manual_latent_path(手动上传)
                st = os.stat(path)
                return "manual:%s:%s:%s" % (path, st.st_mtime_ns, st.st_size)
            except Exception:
                return float("NaN")
        # 片段序号显式为常量 0 = 第一个片段：输出确定性空标记，无需读文件。
        # IS_CHANGED 阶段序号来自 GetNode/表达式链路时拿不到真实值，
        # 故对存储目录下全部 latent 文件做综合指纹：内容变化 → 指纹变化
        # → 下游重跑；同一片段重试内容不变 → 缓存命中。
        try:
            if int(片段序号) == 0:
                return 0
        except (TypeError, ValueError):
            pass  # 链接输入拿不到序号，走目录级指纹
        try:
            return _dir_fingerprint(_build_load_path(存储位置))
        except Exception:
            return float("NaN")  # 意外失败：永不缓存，保守重跑

    def load(self, 存储位置, 片段序号=1, 手动上传=""):
        # 手动上传非空：优先加载该文件，完全忽略存储位置与片段序号
        # （即使序号为 0 也照样加载，不以第一片段处理）
        if _st_load is None:
            raise RuntimeError("h3_motion_context: safetensors is not "
                               "available; cannot load latents.")
        manual_path = _resolve_manual_latent_path(手动上传)
        if manual_path is not None:
            data = _st_load(manual_path)
            if "video" not in data or "audio" not in data:
                raise ValueError(
                    "h3_motion_context: %s is not an h3_motion_context latent "
                    "(missing video/audio streams). Was it saved by the stock "
                    "Save Latent node instead?" % manual_path)
            # 手动加载无片段序号，不携带 CLIP_INDEX_KEY，
            # 上下文节点提示为"已关联上下文"
            return ({"samples": [data["video"], data["audio"]]},)
        # 片段序号 0 = 第一个片段：无前序上下文，不读文件，输出带空标记的
        # 空 latent，上下文节点识别后条件化直通、裁剪 0
        try:
            idx = int(片段序号)
        except (TypeError, ValueError):
            raise ValueError("h3_motion_context: 片段序号必须是整数，得到 %r"
                             % (片段序号,))
        if idx == 0:
            return ({"samples": [], self.EMPTY_MARKER: True,
                     self.EMPTY_REASON: "first_clip"},)
        try:
            path = _resolve_latent_path(_build_load_path(存储位置), idx)
        except FileNotFoundError as e:
            # 文件未找到：按第一片段处理，输出带原因的空标记，
            # 上下文节点识别后直通、裁剪 0 并显示提示
            return ({"samples": [], self.EMPTY_MARKER: True,
                     self.EMPTY_REASON: "file_not_found",
                     self.EMPTY_REASON_DETAIL: str(idx)},)
        data = _st_load(path)
        if "video" not in data or "audio" not in data:
            raise ValueError(
                "h3_motion_context: %s is not an h3_motion_context latent "
                "(missing video/audio streams). Was it saved by the stock "
                "Save Latent node instead?" % path)
        # 输出普通 list 而非 NestedTensor：仅本仓库的 context_latent 输入
        # 接受它，因此不可能被误当成可解码潜空间——接错会大声失败
        # 携带片段序号，供上下文节点生成"已关联片段 N"提示
        return ({"samples": [data["video"], data["audio"]],
                 self.CLIP_INDEX_KEY: int(片段序号)},)


NODE_CLASS_MAPPINGS = {
    "Yuan_H3MotionContext": Yuan_H3MotionContext,
    "Yuan_H3MotionContextTrim": Yuan_H3MotionContextTrim,
    "Yuan_H3MotionContextLoadLatent": Yuan_H3MotionContextLoadLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_H3MotionContext": "H3 运动上下文",
    "Yuan_H3MotionContextTrim": "H3 运动裁剪",
    "Yuan_H3MotionContextLoadLatent": "H3 加载潜空间",
}
