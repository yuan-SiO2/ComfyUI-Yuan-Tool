"""H3 运动上下文相关节点（Yuan Tool 本地化版）.

复刻自 "C:\\My Xiangmu\\Yuan Tool" 的 nodes.py，包含 4 个节点：
  - H3 加载潜空间 (Yuan_H3MotionContextLoadLatent)
  - H3 运动上下文 (Yuan_H3MotionContext)
  - H3 保存潜空间 (Yuan_H3MotionContextSaveLatent)
  - H3 运动裁剪 (Yuan_H3MotionContextTrim)

节点分类: "Yuan Tool/MiniMax"

MiniMax H3 的片段衔接：固定前一片段尾部的画面和声音，让下一个片段
真正地延续它。视频和音频直接从上一片段的潜空间切片获取，跳过每次
链接都会损失少许质量的解码和重编码过程。

布局补丁（patch_layout）解除仅首/末帧关键帧锚点的限制，将固定的音频
移动到片段自身的时间轴上，并在引用偏移布局时保持一切对齐。

载荷补丁（patch_payload）阻止引用分支覆盖关键帧条件潜空间，使固定
视频和固定音频可以同时使用。

两个补丁在运动上下文节点首次运行时才安装，而非导入时安装。ComfyUI
在启动时会导入 custom_nodes 中的每个文件夹，若在导入时打补丁，会将本
包的包装器置于本机上每个 H3 图的路径上。首次使用时安装意味着安装此
包不会改变任何东西，直到你真正衔接一个片段。

每个补丁也基于本包自身的标记进行门控，因此即使安装后，它们对无关的
H3 图也保持与原版逐位一致。

每个补丁在提交前都会对实时 ComfyUI 代码进行自测。如果测试失败，节点
会拒绝运行并说明原因，因此上游变更会产生清晰的错误，而非静默地渲染
出错误的结果。
"""

import logging
import os

import folder_paths
import node_helpers
import torch

import comfy.ldm.minimax.model as mm
import comfy.model_base as model_base

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:  # ComfyUI always ships safetensors; belt and braces
    _st_load = _st_save = None

_LOG = logging.getLogger("h3_motion_context")


# ============================================================================
# 共享 ABI 标记
# ============================================================================

# 关键帧真实位置在 keyframe dict 中的键，由布局补丁读取
MC_KEY = "motion_context_index"
# 音频引用尾端目标帧在 ref dict 中的键，由布局补丁读取
MC_AUDIO_KEY = "motion_context_audio_end_frame"


# ============================================================================
# 布局补丁 (原 patch_layout.py)
# 解除 MiniMax H3 仅首/末帧关键帧锚点的限制
# ============================================================================

# Marker set on our wrapper so a second copy of this file, vendored into
# another pack, can recognise it and stand down instead of wrapping it.
# Shared ABI across every pack that vendors this patch, exactly like
# MC_KEY and MC_AUDIO_KEY: rename it in all of them or in none.
PATCH_MARKER_LAYOUT = "_h3_motion_context_layout_patch"

_layout_orig_init = None
_layout_applied = False

REF_SEGMENT_KINDS = ("ref_img", "ref_audio")


def _target_origin(layout):
    """The coordinate the target clip starts at, read off the built layout.

    Stock lays reference blocks out from a cursor that starts at text_len,
    and the target rows take the cursor's final value as their origin.
    Keyframe coordinates are computed from text_len directly and never
    compensated, so without this term any reference slides the anchors
    backwards relative to the clip they are anchoring.

    Earlier versions recomputed that cursor with a local copy of stock's
    per-kind advance arithmetic. Reading it back out of the layout instead
    means there is nothing to keep in sync: if upstream changes how a
    reference kind advances the cursor, the number here changes with it.
    The target video segment is always last and always has at least one
    latent step, and _video_grid puts its first row exactly on the cursor.
    """
    a, b, kind = layout.segments[-1]
    if kind != "video" or b <= a:
        raise RuntimeError(
            "h3_motion_context: expected the target video rows to be the "
            "last layout segment, found %r spanning %d rows. Upstream "
            "layout change; refusing to rewrite positions." % (kind, b - a))
    return float(layout.position_ids[a, 0])


def _expected_ref_segments(blk):
    """The segment kinds one reference block emits, in emission order.

    Mirrors the branches of the stock constructor:
      image         one ref_img
      audio         one ref_audio, or nothing at all when the window is
                    empty (stock skips the segment but still advances)
      video         the block's audio rows pack immediately before its
      video_audio   video rows, so ref_audio then ref_img
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
    """Which rows each reference block actually produced.

    Returns {block_index: {segment_kind: (start, stop)}}.

    This is the whole point of the multi-reference support. The rows of
    one reference block could be found by working out the coordinate span
    it ought to occupy and selecting everything inside it, but that means
    duplicating stock's cursor arithmetic and then hoping nothing else
    shares the range. Stock keyframe rows genuinely do land inside it,
    which is why the coordinate approach needed an explicit exclusion for
    them, and a second reference of the same kind would need another.

    The layout already publishes a segment table, and reference blocks
    emit their segments in list order, so the mapping is exact. Nothing
    is inferred from coordinates and nothing has to be excluded.
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
    """Time coordinate for a keyframe anchored at pixel frame p.

    The endpoints reuse stock's exact expressions rather than the general
    formula. They are mathematically identical, but stock accumulates
    latent_t float additions where the general form does one multiply, and
    those differ in the last bits (about 7e-15). Matching stock bit for bit
    means an existing first/last graph builds byte-identical positions
    after this patch is applied, and lets the self-test stay strict.
    """
    if p == 0:
        return float(text_len)
    if frame_count is not None and p == frame_count - 1:
        return float(text_len) + sum(mm._video_t_spans(latent_t)) - mm.FRAME_RESCALE
    return float(text_len) + mm.FRAME_RESCALE * float(p)


def _fixup(layout, text_len, latent_t, frame_count, keyframes, refs=None):
    """Rewrite cond-row time coordinates to the general position formula.

    `refs` is accepted but no longer read for arithmetic: the compensation
    a reference block owes the anchors is now taken from where the target
    actually landed, not recomputed from the block list.
    """
    offset = _target_origin(layout) - float(text_len)
    if offset and any(kf.get(MC_KEY) is None for kf in keyframes):
        # keyframes without MC_KEY are left exactly as stock built them,
        # which means they do NOT get the reference compensation. Mixing
        # them with MC keyframes under a reference would slide the stock
        # anchors relative to ours and to the target. Nothing produces
        # this today; refuse loudly in case something ever does.
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
    """Move the marked audio ref's rows onto the target timeline.

    References and keyframes carry identical row machinery; what makes the
    model read a reference as "a separate clip to imitate" rather than
    "this clip, continued" is that its coordinates sit in a span before
    the target. That distinction decided continuation vs reproduction for
    video, and seam analysis showed the audio reference producing
    phase-unlocked imitation. So: keep the audio on the reference path for
    construction and payload (rows built, latents filled, all stock code
    untouched) and TRANSLATE its time coordinates so the window END lands
    at target frame MC_AUDIO_KEY, the same instant the pinned video ends.

    Translation, not per-row assignment: new = old + shift preserves
    whatever intra-block structure stock built. Stock lays an audio
    reference out channel-major, the same rt coordinates once per stereo
    channel, and a uniform shift keeps that intact without this code
    having to know about it.

    The block keeps its place in the cursor, so the coordinates it vacates
    are left empty. An audio window longer than the video window therefore
    spills backwards into empty space rather than onto the text rows, so
    the collision that made `before` mode fail for video does not arise.

    Other reference blocks are untouched. A Ref2VA graph can carry its own
    image, video and audio references and this moves only the one block
    the node marked, wherever in the list it sits.
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
        # stock emits exactly rt rows per stereo channel. An exact count
        # rather than a tolerance: if this ever changes, the intra-block
        # structure a translation is preserving has changed with it.
        raise RuntimeError(
            "h3_motion_context: the marked audio reference has %d rows for "
            "%d latent steps, expected %d (stereo, channel-major). Upstream "
            "layout change; refusing to move rows." % (b - a, rt, 2 * rt))

    target_origin = _target_origin(layout)
    slot_start = float(layout.position_ids[a, 0])
    end_frame = float(blk[MC_AUDIO_KEY])
    # window end at target time FRAME_RESCALE * end_frame, width rt steps
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
    # neither marked: stock graph, leave it exactly as built


def _layout_self_test():
    """Prove the rewrite reproduces stock positions before committing.

    Builds the two anchors stock code already supports, once the stock way
    and once through our mechanism, and requires the position tensors to
    match exactly. Then exercises the parts stock has no equivalent of:
    interior anchors, reference compensation, the audio move, and the
    same audio move inside a multi-reference Ref2VA layout. If ComfyUI
    changes the position maths or the segment table underneath us this
    fails and the patch is not applied.
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

    # 1. the two anchors stock supports must come out bit-identical
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

    # 2. a consecutive run lands on strictly increasing coordinates inside
    # the span the two endpoints define
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

    # 3. adding a reference must not move the anchors relative to the
    # target. Stock cond rows cannot be the reference here: stock computes
    # them from text_len and never compensates, which is the very bug the
    # compensation exists to fix. The ground truth is the target rows
    # themselves, so the anchor-to-end gap must be identical with and
    # without the reference.
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

    # 4. the audio move: exactly the marked block's rows shift, all by one
    # uniform amount, every other row bit-identical
    end_frame, rt = 4, 8
    ref_mc = [{"kind": "audio", "ref_audio_t": rt, MC_AUDIO_KEY: end_frame}]
    e = build(keyframes=run, refs=ref_mc, fix=True, move=True)
    _check_move(d, e, ref_mc, 0, "single-ref")

    # 5. the same move inside a Ref2VA layout: image, video and audio
    # references of the graph's own must come through untouched. The
    # marked block sits in the MIDDLE of the list, not at the end, because
    # nothing about locating it by segment depends on its position.
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

    # 6. the segment map must agree with how the layout is actually laid
    # out. Not by recomputing the cursor, which is the duplication this
    # rewrite exists to remove, but by checking the structural properties
    # the move depends on: reference blocks appear in list order, their
    # rows sit before the target, and no block's rows overlap another's.
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
    """Only the marked block's rows moved, uniformly, on the time axis."""
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
    # The size of the shift is deliberately NOT asserted. It depends on
    # how far the reference cursor advanced, which is stock's business,
    # and pinning it here would put a copy of that arithmetic back in.
    # What must hold is where the window ENDS: on the target timeline,
    # FRAME_RESCALE * end_frame past the target origin.
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
    """Has another copy of this file already wrapped the constructor?

    Returns None, "same" or "other".

    Two copies of this patch in one ComfyUI is normal enough: several
    packs vendor it, and forks of this repo carry their own. Whichever
    loads second would otherwise capture the first's wrapper as its
    original and wrap a wrapper. That is worse than it sounds, because
    each copy self-tests through whatever is already installed, so a copy
    with newer tests gets checked against older behaviour and refuses
    over a limitation that no longer exists.

    Three checks, in decreasing confidence.

    The marker is set by copies new enough to set it. That is a matching
    version and we stand down quietly.

    A wrapper merely NAMED like ours is an older copy of this code, or a
    fork. We stand down and say so, because whichever one loaded first is
    the one deciding what the patch supports.

    Anything else sitting where the stock constructor should be is a
    DIFFERENT pack patching the same thing. Several H3 packs lift the
    same first/last restriction independently, and they cannot both own
    the constructor. Detected by comparing where the function was defined
    against where the class was: stock's __init__ comes from the same
    module as PackedLayout itself, a wrapper comes from somewhere else.
    functools.wraps copies __module__ across, so __wrapped__ is checked
    too. A wrapper that hides both is indistinguishable from stock and
    nothing can be done about that.
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


def _apply_layout_patch():
    global _layout_orig_init, _layout_applied
    if _layout_applied:
        return True
    who = _layout_already_patched()
    if who == "foreign":
        _LOG.warning(
            "h3_motion_context: another pack has already patched H3's "
            "layout (PackedLayout.__init__ now comes from %r). Several "
            "ComfyUI packs lift the same first/last keyframe restriction "
            "independently and they cannot both own it, so this one is "
            "refusing rather than wrapping a wrapper and producing joins "
            "neither pack intended. Disable one of them and restart.",
            getattr(getattr(getattr(mm, "PackedLayout", None), "__init__",
                            None), "__module__", "?"))
        return False
    if who:
        # report success: the patch IS active, just not ours, and the
        # calling pack's nodes check is_applied() before they will run
        _layout_applied = True
        if who == "same":
            _LOG.info("h3_motion_context: interior keyframe anchors already "
                      "enabled by another pack, standing down")
        else:
            _LOG.warning(
                "h3_motion_context: the H3 layout patch is already installed "
                "by a DIFFERENT copy of this code (another version, or a "
                "fork). Standing down; that copy decides what the patch "
                "supports, so features added since it may be unavailable. "
                "If you have more than one H3 Motion Context folder in "
                "custom_nodes, keep one and remove the rest. Renaming a "
                "folder does not stop ComfyUI loading it.")
        return True
    if not hasattr(mm, "PackedLayout") or not hasattr(mm, "FRAME_RESCALE"):
        _LOG.warning("h3_motion_context: MiniMax H3 model module missing expected "
                     "attributes, patch not applied")
        return False
    _layout_orig_init = mm.PackedLayout.__init__
    try:
        _layout_self_test()
    except Exception as exc:
        _layout_orig_init = None
        _LOG.warning("h3_motion_context: self-test failed (%s), patch not "
                     "applied. Interior keyframe anchors unavailable.", exc)
        _LOG.warning(
            "h3_motion_context: if you have more than one H3 Motion Context "
            "folder in custom_nodes (a fork, a backup, a manual clone "
            "alongside a Manager install), that is the usual cause: each "
            "copy self-tests against whichever one loaded first. Keep one "
            "and remove the rest. Renaming a folder does not stop ComfyUI "
            "loading it. Otherwise this is an upstream ComfyUI change and "
            "the message above says what moved.")
        return False
    mm.PackedLayout.__init__ = _patched_init
    _layout_applied = True
    _LOG.info("h3_motion_context: interior keyframe anchors enabled")
    return True


def _layout_patch_applied():
    return _layout_applied


# ============================================================================
# 载荷补丁 (原 patch_payload.py)
# 让关键帧和引用可以共存
# ============================================================================

# Marker set on our wrapper so a second copy of this file, vendored into
# another pack, can recognise it and stand down instead of wrapping it.
# Shared ABI across every pack that vendors this patch.
PATCH_MARKER_PAYLOAD = "_h3_motion_context_payload_patch"

_payload_orig_extra_conds = None
_payload_applied = False


def _patched_extra_conds(self, **kwargs):
    out = _payload_orig_extra_conds(self, **kwargs)

    keyframes = kwargs.get("minimax_keyframes", None)
    refs = kwargs.get("minimax_refs", None)
    if not keyframes or not refs:
        return out  # only one mechanism in play, stock behaviour is correct
    if not (any(MC_KEY in kf for kf in keyframes)
            or any(MC_AUDIO_KEY in r for r in refs)):
        # nothing here came from this pack. The layout patch is gated the
        # same way, so leaving the payload alone keeps the two consistent
        # and leaves unrelated graphs bit-identical to stock.
        return out

    cond = out.get("minimax_payload", None)
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        _LOG.warning("h3_motion_context: could not reach the H3 payload, "
                     "keyframe latents may have been overwritten by refs")
        return out

    kf_video = [kf["latent"] for kf in keyframes if "latent" in kf]
    ref_video = [r["latent"] for r in refs if "latent" in r]
    payload["cond_video_latents"] = kf_video + ref_video
    payload["cond_audio_latents"] = [r["audio_latent"] for r in refs
                                     if r.get("audio_latent") is not None]
    # only write frame_count when we actually have one. This wrapper fires
    # for ANY graph combining keyframes and refs, not just ours; a graph
    # that reaches here without minimax_frame_count may have a valid value
    # already set by the original, and overwriting it with None would break
    # the last-frame anchor branch downstream.
    fc = kwargs.get("minimax_frame_count", None)
    if fc is not None:
        payload["frame_count"] = fc
    return out


setattr(_patched_extra_conds, PATCH_MARKER_PAYLOAD, True)


def _payload_already_patched(cls):
    """Has another copy of this file already wrapped extra_conds?

    Returns None, "same", "other" or "foreign". The marker only
    recognises copies new enough to set it, so a wrapper merely NAMED
    like ours counts as another copy: an older version, or a fork, and
    the second one in stands down. Anything else wrapping extra_conds is
    a different pack solving the same problem its own way, and this one
    refuses rather than stacking on it. See patch_layout's version for
    how the detection works and what it cannot see.
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


def _apply_payload_patch():
    global _payload_orig_extra_conds, _payload_applied
    if _payload_applied:
        return True
    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        _LOG.warning("h3_motion_context: MiniMaxH3.extra_conds not found, "
                     "keyframes and refs cannot be combined")
        return False
    who = _payload_already_patched(cls)
    if who == "foreign":
        _LOG.warning(
            "h3_motion_context: another pack has already patched "
            "MiniMaxH3.extra_conds (it now comes from %r). Both packs are "
            "solving the same keyframe/ref collision and they cannot both "
            "own it, so this one is refusing. Disable one of them and "
            "restart.",
            getattr(getattr(cls, "extra_conds", None), "__module__", "?"))
        return False
    if who:
        # report success: the patch IS active, just not ours, and the
        # calling pack's nodes check is_applied() before they will run
        _payload_applied = True
        if who == "same":
            _LOG.info("h3_motion_context: keyframe/ref coexistence already "
                      "enabled by another pack, standing down")
        else:
            _LOG.warning(
                "h3_motion_context: the H3 payload patch is already "
                "installed by a DIFFERENT copy of this code (another "
                "version, or a fork). Standing down. If you have more than "
                "one H3 Motion Context folder in custom_nodes, keep one and "
                "remove the rest.")
        return True
    _payload_orig_extra_conds = cls.extra_conds
    cls.extra_conds = _patched_extra_conds
    _payload_applied = True
    _LOG.info("h3_motion_context: keyframe/ref coexistence enabled")
    return True


def _payload_patch_applied():
    return _payload_applied


# ============================================================================
# 补丁安装入口
# ============================================================================

def _ensure_layout_patch():
    """Install the layout patch, once, the first time a node runs.

    ComfyUI imports every folder in custom_nodes at startup, so patching
    at import time would put this pack's wrappers in the path of every H3
    graph on the machine, including graphs that never go near these
    nodes. Installing on first use instead means the pack sitting in
    custom_nodes changes nothing at all until you actually chain a clip.

    The cost is that a self-test failure shows up on the first render
    rather than in the startup log. The message is the same either way,
    and it still refuses rather than rendering something wrong.
    """
    if _layout_patch_applied():
        return
    if not _apply_layout_patch():
        raise RuntimeError(
            "h3_motion_context: the layout patch could not be applied, so "
            "interior anchors would be rejected by ComfyUI. The reason was "
            "logged just above this error.")


def _ensure_payload_patch():
    """Install the payload patch, once, before anything needs it.

    Only reached when audio is being pinned, which is the only case where
    a ref and the keyframes have to coexist.
    """
    if _payload_patch_applied():
        return
    if not _apply_payload_patch():
        raise RuntimeError(
            "h3_motion_context: the payload patch could not be applied. "
            "Without it the audio ref would overwrite the pinned video "
            "latents and the motion context would be lost. The reason was "
            "logged just above this error.")


# ============================================================================
# H3 潜空间常量与工具函数
# ============================================================================

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24  # H3's native rate; audio latents run at 40 Hz, hence FRAME_RESCALE 5/3
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0

# Run lengths the video VAE's downscale formula max(1, (n - 5) // 17 * 5 + 2)
# actually distinguishes. Anything between two grid points encodes to the same
# number of latent steps as the lower one, but the steps then cover the FIRST
# `covered` frames of the input rather than the last: encoding 10 frames yields
# the same 2 steps as encoding 5, representing frames [-10..-6] of the source
# clip instead of [-5..-1]. The pinned run would end five frames early and the
# delivered clip would continue from the wrong instant. So off-grid requests
# are snapped DOWN before slicing, keeping content and coverage in agreement.
# The grid is 17m+5 and continues upward; the node only offers up to 56,
# but the snap-down logic knows the higher points so an out-of-range
# request lands on the nearest real one instead of being clamped to 39.
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)

# Settings that used to be widgets. Each had exactly one right answer, so
# offering the wrong one was noise. The losing branches are still in the
# code below: change a constant here to reproduce the failure they cause.
#
#   ANCHOR_MODE   "head" pins the run at the start of the clip, where the
#                 Trim node removes it. "before" places it at negative
#                 time so nothing needs trimming, but the coordinates
#                 collide with the text rows, which weakens the anchors
#                 and darkens the output.
#   AUDIO_MODE    "timeline" puts the pinned audio on this clip's own
#                 timeline so the model continues it. "ref" is the stock
#                 placement, which the model imitates instead: similar
#                 music, not the same recording, plus a tick at the join.
ANCHOR_MODE = "head"
AUDIO_MODE = "timeline"


def _pixel_frames(latent_t):
    """Pixel frames covered by latent_t latent steps."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _step_offsets(latent_t):
    """Pixel-frame index at which each latent step begins."""
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _streams_from_latent(latent):
    """Unpack an H3 AV latent into its contained streams.

    NestedTensor.__getitem__ broadcasts the index into every contained
    tensor rather than selecting one, so samples[0] would strip the batch
    dimension off both streams. unbind() returns the pair.
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
    """Pull the video stream out of an H3 AV latent."""
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:  # unbatched [C,T,H,W]
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("h3_motion_context: expected video latent [B,C,T,H,W], "
                         "got shape %s" % (tuple(video.shape),))
    return video


def _steps_for_frames(n):
    """Latent steps covering exactly n pixel frames from cycle position 0.

    Returns None when no whole number of steps covers n. The video VAE's
    steps alternate 1, 4, 4, 4, 4 pixel frames, so only certain totals are
    reachable: 1, 5, 9, ... and of the windows this node offers, 5, 22 and
    39 land on 2, 7 and 12 steps. The 1-frame window does not, because the
    last step of a clip spans 4 frames, not 1.
    """
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def _video_tail_from_latent(latent, n):
    """Slice the last n pixel frames of video straight out of a generated
    H3 latent, skipping the h264 decode and the VAE encode.

    Returns (blocks, offsets, covered) in the same shape the encode path
    produces, so everything downstream is unchanged.

    This is only sound because the tail window always starts at cycle
    position 0. A clip is 17g+5 frames, which is 5g+2 latent steps; the
    windows are 2, 7 and 12 steps; and 5g+2 minus any of those is a
    multiple of 5. So the sliced run has the same 1, 4, 4, 4, 4 phase as a
    freshly encoded one and _step_offsets applies unchanged. Asserted
    below rather than assumed, because if it ever stopped holding the
    pinned content would silently disagree with the positions written for
    it and the join would land at the wrong instant.
    """
    video = _video_from_latent(latent)
    total = int(video.shape[2])
    steps = _steps_for_frames(n)
    if steps is None:
        raise ValueError(
            "h3_motion_context: a %d frame window is not a whole number of "
            "latent steps, so it cannot be sliced from a latent. Use 5, 22 "
            "or 39, or unwire context_latent to encode pixels." % n)
    if steps > total:
        raise ValueError(
            "h3_motion_context: asked for %d latent steps, context_latent "
            "has %d." % (steps, total))
    start = total - steps
    if start % 5 != 0:
        raise RuntimeError(
            "h3_motion_context: the %d step tail of a %d step latent starts "
            "at cycle position %d, not 0, so its frame spans would not match "
            "the positions written for them. Clip lengths are meant to make "
            "this impossible; refusing rather than rendering a shifted join."
            % (steps, total, start % 5))
    covered = _pixel_frames(steps)
    if covered != n:
        raise RuntimeError(
            "h3_motion_context: %d steps cover %d frames, expected %d."
            % (steps, covered, n))
    blocks = [video[:1, :, start + k:start + k + 1].clone()
              for k in range(steps)]
    return blocks, _step_offsets(steps), covered


def _audio_tail_from_latent(latent, a_frames):
    """Slice the last `a_frames` worth of audio steps straight out of a
    generated H3 latent, skipping the decode -> re-encode round trip.

    Returns (tail latent [1, C, 2, rt], rt, overhang) where rt counts
    40 Hz latent steps and overhang is the fraction of a step by which the
    clip's audio grid extends past its last pixel frame. H3 rounds the
    audio grid UP (124 frames want 206.67 steps, the layout allocates
    207), so the latent's final step reaches ~overhang/40 s beyond the
    last frame. The decoded-audio path never sees this because match_tail
    cuts it; on this path the caller compensates the placement with it,
    so the pinned content lands exactly where its samples actually sit.
    """
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "h3_motion_context: context_latent has no audio stream. Wire the "
            "sampler output of an H3 AV graph, not a video-only latent.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:  # unbatched [C,2,T]
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError("h3_motion_context: expected audio latent [B,C,2,T], "
                         "got shape %s" % (tuple(audio.shape),))
    total_t = int(audio.shape[-1])
    frames = _pixel_frames(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    if not (0.0 <= overhang < 1.0):
        _LOG.warning(
            "h3_motion_context: context_latent audio grid is unexpected "
            "(%d steps for %d frames); assuming no overhang.", total_t, frames)
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        _LOG.warning("h3_motion_context: asked for %d audio steps, the latent "
                     "has %d. Pinning all of it.", rt, total_t)
        rt = total_t
    if rt < 1:
        raise ValueError("h3_motion_context: audio window is empty")
    tail = audio[:1, ..., total_t - rt:].clone()
    return tail, rt, float(overhang)


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
                               "标记后条件化直通、裁剪帧数为 0。"}),
                "上下文长度": (["5", "22", "39", "56"], {
                    "default": "22",
                    "tooltip": "从前一片段延续的画面帧数。只有这些长度是完整的"
                               "潜空间步数，因此只提供这些选项。5 帧仅勉强流畅，"
                               "22 帧近乎无缝。更长的窗口固定更多运动，但会从"
                               "交付片段的头部扣除，因此 56 帧会浪费 2.3 秒的"
                               "渲染在你要丢弃的帧上。"}),
                "音频上下文长度": (["5", "22", "39", "56"], {
                    "default": "22",
                    "tooltip": "尾部音频的固定帧数，独立于画面窗口。"
                               "该窗口与固定的视频尾端对齐，"
                               "因此 22 帧对应 22 帧画面窗口时完全重叠；更长"
                               "的窗口会进一步延伸到已腾空的坐标空间"
                               "（未测试）。"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("条件化", "裁剪帧数")
    FUNCTION = "apply"
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("将上一片段尾部的连续帧作为不可去噪的条件行固定下来，"
                   "让模型读取真实运动而非从单帧静态图中猜测。画面和声音"
                   "都直接从上一片段的潜空间中切片获取，跳过每次链接都会"
                   "损失少许质量的解码和重编码过程。")

    def apply(self, 条件化, 潜空间, 上下文潜空间, 上下文长度, 音频上下文长度="22"):
        # 第一个片段：上下文潜空间来自「H3 加载潜空间」片段序号 0 的空标记，
        # 或文件未找到时的回退空标记。无前序上下文，条件化直通、裁剪 0，
        # 不安装补丁、不做任何切片，并通过 ui.h3_hint 在节点下方显示简短提示
        if isinstance(上下文潜空间, dict) and 上下文潜空间.get(
                Yuan_H3MotionContextLoadLatent.EMPTY_MARKER):
            reason = 上下文潜空间.get(
                Yuan_H3MotionContextLoadLatent.EMPTY_REASON, "first_clip")
            if reason == "file_not_found":
                detail = 上下文潜空间.get(
                    Yuan_H3MotionContextLoadLatent.EMPTY_REASON_DETAIL, "?")
                return {"result": (条件化, 0), "ui": {
                    "h3_hint": "未找到片段 %s 文件" % detail}}
            return {"result": (条件化, 0), "ui": {
                "h3_hint": "片段\"0\"，直通"}}
        anchor_mode = ANCHOR_MODE
        audio_mode = AUDIO_MODE
        上下文长度 = int(上下文长度)
        音频上下文长度 = int(音频上下文长度)
        _ensure_layout_patch()

        video = _video_from_latent(潜空间)
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)

        # latent 无法缩放：直接从上一片段的 AV latent 切片视频尾部，
        # 跳过 h264 解码和 VAE 编码，块与模型产出的完全一致
        src_video = _video_from_latent(上下文潜空间)
        src_w = int(src_video.shape[4]) * 16
        src_h = int(src_video.shape[3]) * 16
        if src_w != width or src_h != height:
            # latent 无法缩放，分辨率变化在链条中几乎总是错误，直接报错
            raise ValueError(
                "h3_motion_context: context_latent is %dx%d but this "
                "clip is %dx%d. A latent cannot be resized, so the "
                "previous clip has to be regenerated at this "
                "resolution, or the chain restarted here."
                % (src_w, src_h, width, height))
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
        if n < 上下文长度:
            _LOG.warning("h3_motion_context: only %d frames available, pinning %d",
                         available, n)

        # 在切片前对齐到 VAE 网格，使切片的帧正好是潜空间步数覆盖的帧
        run = next(g for g in VIDEO_RUN_GRID if g <= n)
        if run != n:
            _LOG.warning(
                "h3_motion_context: %d frames is off the VAE grid; pinning "
                "the last %d instead (usable runs: 1, 5, 22, 39)", n, run)
        n = run

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
                # stock code 仅接受 0 或 frame_count-1，真实位置通过
                # MC_KEY 携带，由布局补丁应用
                "resolved_frame_index": 0,
                MC_KEY: p,
                "latent": blk,
            })

        values = {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        }

        # 从 latent 切片音频尾部
        _ensure_payload_patch()
        # 音频窗口独立于视频窗口：音频条件行占行但不占交付帧
        a_frames = int(音频上下文长度)
        audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(
            上下文潜空间, a_frames)
        ref = {
            "kind": "audio",
            "ref_audio_t": ref_audio_t,
            "audio_latent": audio_latent,
        }
        if audio_mode == "timeline":
            # 将音频窗口尾端对齐到固定视频：两者都是片段 A 的尾部，
            # 必须在新时间轴的同一时刻结束。latent 路径上切片内容
            # 会超出 A 最后帧 overhang 个步（H3 音频网格向上取整），
            # 所以结束坐标移动那么多
            end_frame = float(span if anchor_mode == "head" else 0)
            end_frame += overhang / FRAME_RESCALE
            # 将窗口对齐到目标自身的音频网格
            end_coord = round(FRAME_RESCALE * end_frame)
            end_frame = end_coord / FRAME_RESCALE
            ref[MC_AUDIO_KEY] = end_frame
        # APPEND 而非赋值：Ref2VA 条件化已携带图自身的引用块，
        # 赋值会替换全部。用第二次调用让关键帧值先落位
        audio_ref = ref

        out = node_helpers.conditioning_set_values(条件化, values)
        out = node_helpers.conditioning_set_values(
            out, {"minimax_refs": [audio_ref]}, append=True)

        trim = span if anchor_mode == "head" else 0
        # 正常切片成功：提示已关联的片段序号（来自加载节点）
        clip_idx = (上下文潜空间.get(Yuan_H3MotionContextLoadLatent.CLIP_INDEX_KEY)
                    if isinstance(上下文潜空间, dict) else None)
        hint_text = ("已关联片段 %s 文件" % clip_idx) if clip_idx is not None else "已关联上下文"
        return {"result": (out, trim), "ui": {
            "h3_hint": hint_text}}


class Yuan_H3MotionContextTrim:
    """Drop the pinned head off a decoded clip, picture and sound together.

    The pinned frames occupy the start of the delivered timeline, so they
    have to come off before concatenating. Trimming only the images would
    leave the audio a full trim_frames longer than the video, and muxing
    those puts the whole soundtrack ahead of the picture by trim_frames/24
    seconds. At 5 frames that is 208ms, silent on ambience but squarely
    offbeat on anything with a pulse.

    So this takes both streams and removes the same span from each: whole
    frames from the images, the matching number of samples from the
    waveform. Wire trim_frames from the motion context node so the count
    follows whatever the encoder actually produced.

    The tail needs the same treatment for a different reason. H3's audio
    latent runs at 40 Hz against 24 fps picture, and FRAME_RESCALE is 5/3,
    so a 124 frame clip wants 206.67 audio steps and the layout rounds up
    to 207. Every clip therefore ships about 8.3 ms more sound than
    picture. Concatenate two and the second seam is out by 16.7 ms, three
    and it is 25 ms, and the error grows without bound down a chain. It
    reads as a faint dampening at the first join and a short click at
    later ones. Truncating the tail to exactly frames/fps stops it
    accumulating.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "裁剪帧数": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "音频": ("AUDIO", {
                    "tooltip": "同一片段的已解码音频。按匹配的时长裁剪以保持"
                               "声画同步。静音片段可不连接。"}),
                "帧率": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "用于将裁剪帧数转换为音频时长的帧率。必须与你"
                               "输入到 Create Video 节点的帧率一致。"}),
                "尾部对齐": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "截断音频使其时长精确等于 帧数/帧率。H3 的"
                               "音频网格向上取整，因此每个片段携带约 8ms 额外"
                               "声音，在链条的每个衔接处累积。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("图像", "音频")
    FUNCTION = "trim"
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("从已解码的 H3 片段中移除头部固定的帧，同时按相同时长"
                   "裁剪画面和声音。")

    def trim(self, 图像, 裁剪帧数, 音频=None, 帧率=24.0, 尾部对齐=True):
        n = max(0, int(裁剪帧数))
        total = int(图像.shape[0])
        if n >= total:
            raise ValueError(
                "h3_motion_context: asked to trim %d frames from a %d frame clip"
                % (n, total))
        out_images = 图像[n:] if n else 图像

        out_audio = 音频
        if 音频 is not None:
            waveform = 音频["waveform"]
            sr = int(音频["sample_rate"])
            seconds = n / float(帧率)
            cut = int(round(seconds * sr))
            length = int(waveform.shape[-1])
            if cut >= length:
                raise ValueError(
                    "h3_motion_context: trimming %.3fs from %.3fs of audio would "
                    "leave nothing. Check that fps matches the clip."
                    % (seconds, length / sr))
            waveform = waveform[..., cut:]

            if 尾部对齐:
                frames_left = total - n
                want = int(round(frames_left / float(帧率) * sr))
                have = int(waveform.shape[-1])
                if have > want:
                    over = have - want
                    waveform = waveform[..., :want]
                    _LOG.info("h3_motion_context: tail trimmed %d samples "
                              "(%.2fms) so audio matches %d frames exactly",
                              over, over / sr * 1000.0, frames_left)
                elif have < want:
                    _LOG.warning("h3_motion_context: audio is %.2fms shorter than "
                                 "%d frames; leaving the tail alone",
                                 (want - have) / sr * 1000.0, frames_left)

            out_audio = {"waveform": waveform, "sample_rate": sr}
            _LOG.info("h3_motion_context: %d frames / %.4fs picture, %.4fs sound, "
                      "drift %.2fms",
                      total - n, (total - n) / float(帧率),
                      int(waveform.shape[-1]) / sr,
                      abs((total - n) / float(帧率) - int(waveform.shape[-1]) / sr) * 1000.0)
        elif n:
            _LOG.info("h3_motion_context: trimmed %d leading frames, %d remain. "
                      "No audio wired; if this clip has sound, mux it through "
                      "this node or it will run %.3fs ahead of the picture.",
                      n, total - n, n / float(帧率))

        return (out_images, out_audio)


def _build_load_path(存储位置):
    """把用户输入的存储位置转换为内部路径前缀（存储位置/latent）。

    用户只需设置目录名（如 H3-Mubu），文件前缀 latent 在内部固定，
    用户不可见、不可改。
    """
    loc = (存储位置 or "").strip().strip('"').strip("'")
    if not loc:
        loc = "H3-Mubu"
    return os.path.join(loc, "latent")


def _resolve_latent_path(path, clip_index=1):
    """Turn the loader's path input into a concrete file.

    Accepts three forms (absolute or relative to ComfyUI's output folder):

      1. A file path       that exact file is loaded.
      2. A file prefix     same as Save's filename_prefix (e.g.
                           "H3-Mubu/latent"). clip_index picks
                           {prefix}_0000N_.safetensors.
                           This lets Load and Save use the SAME default
                           value, so wiring them is intuitive.
    """
    p = (path or "").strip().strip('"').strip("'")
    if not p:
        p = "H3-Mubu/latent"
    candidates = [p, os.path.join(folder_paths.get_output_directory(), p)]
    for c in candidates:
        if os.path.isfile(c):
            return c
        # 尝试作为文件前缀
        # 例如 "H3-Mubu/latent" → 在 H3-Mubu/ 目录中查找 latent_*.safetensors
        # 这样 Load 和 Save 可以使用完全相同的默认值
        dir_part = os.path.dirname(c)
        prefix = os.path.basename(c)
        if dir_part and prefix and os.path.isdir(dir_part):
            return _resolve_prefix(dir_part, prefix, int(clip_index))
    raise FileNotFoundError(
        "h3_motion_context: %r is neither a file nor a file "
        "prefix (also tried relative to the ComfyUI output directory)." % p)


def _resolve_prefix(dir_part, prefix, idx):
    """Resolve a latent file by filename prefix (matches Save's filename_prefix).

    e.g. prefix="latent", idx=2  ->  latent_00002_.safetensors
    """
    endings = ("%s_%05d_.safetensors" % (prefix, idx),
               "%s_%05d.safetensors" % (prefix, idx),  # 兼容旧版无下划线
               "%s_clip%03d.safetensors" % (prefix, idx))  # 更早版本
    files = [os.path.join(dir_part, f) for f in os.listdir(dir_part)
             if f.endswith(endings)]
    if not files:
        raise FileNotFoundError(
            "h3_motion_context: no saved latent for clip %d "
            "(no %s_%05d_.safetensors in %s)."
            % (idx, prefix, idx, dir_part))
    return max(files, key=os.path.getmtime)


class Yuan_H3MotionContextSaveLatent:
    """Save an H3 AV latent to disk so the NEXT run can load it.

    Wiring the sampler's output straight into context_latent is a cycle:
    the sampler would be consuming its own result. The latent that motion
    context needs is the PREVIOUS clip's, which lives in the previous run
    -- so it has to cross runs through disk, the same way the frames and
    audio already do. Stock Save/Load Latent can't serialise H3's nested
    video/audio pair; this saves the two streams side by side.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "潜空间": ("LATENT", {
                    "tooltip": "采样器的输出潜空间（与你连接到解码节点的"
                               "相同）。"}),
                "存储位置": ("STRING", {
                    "default": "H3-Mubu",
                    "tooltip": "保存在 ComfyUI 输出文件夹下的子目录名。"
                               "「加载潜空间」节点使用相同的存储位置即可"
                               "对应加载。"}),
                "片段序号": ("INT", {
                    "default": 1, "min": 1, "max": 9999,
                    "tooltip": "本片段在链条中的序号。设为 2 时保存到"
                               "latent_00002_.safetensors，重复生成会覆盖原文件。"
                               "「加载潜空间」节点设相同序号即可对应加载。"}),
                "保存到本地": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "开启时将潜空间保存到本地文件，关闭时仅直通"
                               "输出不生成文件。不影响潜空间输出端口。"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("潜空间",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("保存采样器的 AV 潜空间到磁盘，同时将输入潜空间原样直通输出，"
                   "使下一次运行的运动上下文节点可通过配套的「加载潜空间」节点"
                   "从中固定音频，本片段的潜空间也可继续向下游传输。")

    def save(self, 潜空间, 存储位置, 片段序号=1, 保存到本地=True):
        if 保存到本地:
            if _st_save is None:
                raise RuntimeError("h3_motion_context: safetensors is not "
                                   "available; cannot save latents.")
            parts = _streams_from_latent(潜空间)
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
            os.makedirs(folder, exist_ok=True)  # 确保子目录存在
            # 片段序号 2 -> latent_00002_.safetensors
            # 重新生成会覆盖自身的废弃文件，不会堆叠新文件
            path = os.path.join(folder, "%s_%05d_.safetensors"
                                % (filename, int(片段序号)))
            _st_save({"video": video, "audio": audio}, path,
                     metadata={"format": "h3_motion_context_av_v1"})
            _LOG.info("h3_motion_context: saved AV latent to %s (video %s, "
                      "audio %s)", path, tuple(video.shape), tuple(audio.shape))
        # 输入潜空间原样直通输出，无论是否保存到本地都不影响输出端口
        return (潜空间,)


class Yuan_H3MotionContextLoadLatent:
    """Load a saved H3 AV latent for the context_latent input.

    clip_index means exactly what it says: set it to the clip you want to
    CONTINUE FROM, and that clip's slot is loaded. E.g. clip_index=2 loads
    latent_00002_.safetensors, the same file Save node with clip_index=2
    writes to. Re-rolling a clip overwrites its own save, so rejects
    never accumulate.

    The output is ONLY for the Motion Context node's context_latent input.
    It is not a decodable latent -- do not wire it into VAE decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "存储位置": ("STRING", {
                    "default": "H3-Mubu",
                    "tooltip": "与「保存潜空间」节点相同的存储位置"
                               "（ComfyUI 输出文件夹下的子目录名）。"}),
                "片段序号": ("INT", {
                    "default": 1, "min": 0, "max": 9999,
                    "tooltip": "要加载的片段序号。设为 2 时加载"
                               "latent_00002_.safetensors，与「保存潜空间」"
                               "节点的命名规则一致。设为 0 时表示链条的第一个"
                               "片段（无前序上下文）：不读取本地文件，输出空"
                               "标记，运动上下文节点识别后条件化直通、裁剪帧"
                               "数为 0。"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("加载由「H3 保存潜空间」节点保存的潜空间，"
                   "仅用于运动上下文节点的「上下文潜空间」输入。"
                   "片段序号设为 0 时表示第一个片段，不读取文件，"
                   "运动上下文节点将条件化直通。")

    # 标记键：片段序号 0 时输出带此标记的空 latent，运动上下文节点据此直通
    EMPTY_MARKER = "_h3_motion_context_empty"
    # 原因键：空标记的具体原因（first_clip / file_not_found），供运动上下文节点生成提示词
    EMPTY_REASON = "_h3_motion_context_empty_reason"
    # 细节键：原因对应的细节（如未找到的片段序号），供运动上下文节点生成提示词
    EMPTY_REASON_DETAIL = "_h3_motion_context_empty_reason_detail"
    # 正常加载时携带的片段序号键，供运动上下文节点生成"已关联片段 N"提示
    CLIP_INDEX_KEY = "_h3_motion_context_clip_index"

    @classmethod
    def IS_CHANGED(cls, 存储位置, 片段序号=1):
        # 片段序号 0 = 第一个片段，输出确定性的空标记，无需读取文件
        if int(片段序号) == 0:
            return 0
        # the path string stays constant while the file behind it changes
        # (newest save, or an overwritten slot), so cache on the resolved
        # file identity instead -- otherwise ComfyUI would happily serve
        # a stale latent forever
        try:
            p = _resolve_latent_path(_build_load_path(存储位置), 片段序号)
            return "%s:%d" % (p, os.stat(p).st_mtime_ns)
        except Exception:
            return float("NaN")  # unresolvable: never cache

    def load(self, 存储位置, 片段序号=1):
        # 片段序号 0 = 第一个片段：无前序上下文，不读取本地文件，
        # 输出带空标记的空 latent，运动上下文节点识别后条件化直通、裁剪 0
        if int(片段序号) == 0:
            return ({"samples": [], self.EMPTY_MARKER: True,
                     self.EMPTY_REASON: "first_clip"},)
        if _st_load is None:
            raise RuntimeError("h3_motion_context: safetensors is not "
                               "available; cannot load latents.")
        try:
            path = _resolve_latent_path(_build_load_path(存储位置), 片段序号)
        except FileNotFoundError as e:
            # 文件未找到：按第一片段处理，输出空标记并携带原因，
            # 运动上下文节点识别后条件化直通、裁剪 0，并在节点下方显示提示
            return ({"samples": [], self.EMPTY_MARKER: True,
                     self.EMPTY_REASON: "file_not_found",
                     self.EMPTY_REASON_DETAIL: str(int(片段序号))},)
        data = _st_load(path)
        if "video" not in data or "audio" not in data:
            raise ValueError(
                "h3_motion_context: %s is not an h3_motion_context latent "
                "(missing video/audio streams). Was it saved by the stock "
                "Save Latent node instead?" % path)
        _LOG.info("h3_motion_context: loaded AV latent from %s", path)
        # a plain list, not a NestedTensor: only this repo's context_latent
        # input accepts it, which is the point -- it cannot be mistaken
        # for a decodable latent without failing loudly downstream
        # 携带片段序号，供运动上下文节点生成"已关联片段 N"提示
        return ({"samples": [data["video"], data["audio"]],
                 self.CLIP_INDEX_KEY: int(片段序号)},)


NODE_CLASS_MAPPINGS = {
    "Yuan_H3MotionContext": Yuan_H3MotionContext,
    "Yuan_H3MotionContextTrim": Yuan_H3MotionContextTrim,
    "Yuan_H3MotionContextSaveLatent": Yuan_H3MotionContextSaveLatent,
    "Yuan_H3MotionContextLoadLatent": Yuan_H3MotionContextLoadLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_H3MotionContext": "H3 运动上下文",
    "Yuan_H3MotionContextTrim": "H3 运动裁剪",
    "Yuan_H3MotionContextSaveLatent": "H3 保存潜空间",
    "Yuan_H3MotionContextLoadLatent": "H3 加载潜空间",
}
