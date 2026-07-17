const { app } = window.comfyAPI.app;

const PALETTE = [
  "#4f8edc", "#e07b3a", "#5cb85c", "#d9534f", "#9b6cd6",
  "#a07060", "#e377c2", "#7f7f7f", "#c4c447", "#3fbac4",
];

// text_input 时间格式解析：匹配行首的 "0-3s", "3-5秒", "5-7s" 等
const TIME_RANGE_PATTERN = /^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*[s秒]\s*[：:]?\s*/;

const RULER_HEIGHT = 22;
const BLOCK_HEIGHT = 64;
const CANVAS_HEIGHT = RULER_HEIGHT + BLOCK_HEIGHT;
const HANDLE_HIT_PX = 6;
const REORDER_THRESHOLD_PX = 6;
const MIN_SEGMENT_LENGTH = 1;
const HIDDEN_WIDGET_NAMES = ["timeline_data", "local_prompts", "segment_lengths"];

function hideWidget(w) {
  if (!w) return;
  w.type = "hidden";
  w.hidden = true;
  w.computeSize = () => [0, -4];
}

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

function pickColor(existingColors) {
  for (const c of PALETTE) if (!existingColors.has(c)) return c;
  const idx = existingColors.size;
  const hue = (idx * 137.508) % 360;
  return `hsl(${hue.toFixed(0)}, 55%, 55%)`;
}

function defaultTimeline(maxFrames) {
  const half = Math.max(MIN_SEGMENT_LENGTH, Math.floor(maxFrames / 2));
  return {
    segments: [
      { prompt: "", length: half, color: PALETTE[0] },
      { prompt: "", length: Math.max(MIN_SEGMENT_LENGTH, maxFrames - half), color: PALETTE[1] },
    ],
  };
}

function parseInitial(jsonStr, maxFrames) {
  if (!jsonStr) return defaultTimeline(maxFrames);
  try {
    const obj = JSON.parse(jsonStr);
    if (Array.isArray(obj?.segments) && obj.segments.length > 0) {
      return {
        segments: obj.segments.map((s, i) => ({
          prompt: typeof s.prompt === "string" ? s.prompt : "",
          length: Math.max(MIN_SEGMENT_LENGTH, parseInt(s.length, 10) || MIN_SEGMENT_LENGTH),
          color: typeof s.color === "string" ? s.color : PALETTE[i % PALETTE.length],
        })),
      };
    }
  } catch (_) {}
  return defaultTimeline(maxFrames);
}

class TimelineEditor {
  constructor(node, container) {
    this.node = node;
    this.container = container;
    this.maxFramesWidget = node.widgets.find(w => w.name === "max_frames");
    this.fpsWidget = node.widgets.find(w => w.name === "fps");
    this.timeUnitsWidget = node.widgets.find(w => w.name === "time_units");
    this.timelineDataWidget = node.widgets.find(w => w.name === "timeline_data");
    this.localPromptsWidget = node.widgets.find(w => w.name === "local_prompts");
    this.segmentLengthsWidget = node.widgets.find(w => w.name === "segment_lengths");
    this.textInputWidget = node.widgets.find(w => w.name === "text_input");
    this.promptLockWidget = node.widgets.find(w => w.name === "prompt_lock");
    this.promptLocked = this.promptLockWidget?.value !== false;  // 默认 true（锁定）

    // --- 初始化时间轴：锁定模式下优先从 text_input 构建，否则从 timeline_data 解析 ---
    const textVal = this.promptLocked ? this._getTextInputValue() : null;
    if (textVal) {
      const lines = textVal.split("\n").map(l => l.trim()).filter(l => l.length > 0);
      if (lines.length > 0) {
        this.timeline = this._buildTimelineFromLines(lines);
      } else {
        this.timeline = parseInitial(this.timelineDataWidget?.value, this.getMaxFrames());
      }
    } else {
      this.timeline = parseInitial(this.timelineDataWidget?.value, this.getMaxFrames());
    }

    this.selectedIndex = 0;
    this.hoverIndex = -1;
    this.hoverHandle = -1;
    this.dragHandle = -1;
    this.dragStart = null;
    this.reorder = null;
    this._settling = false;
    this._inputBaseline = null;
    this._textCommitTimer = null;
    this._displayedX = new Map();
    this._targetX = new Map();
    this._animRaf = null;

    this.buildDOM();
    this.bindEvents();
    this.syncWidgetsFromTimeline();
    this.updateUIFromSelection();
    this.render();
  }

  getMaxFrames() {
    return Math.max(1, parseInt(this.maxFramesWidget?.value, 10) || 1);
  }

  getFps() {
    const v = parseFloat(this.fpsWidget?.value);
    return Number.isFinite(v) && v > 0 ? v : 24;
  }

  isSecondsMode() {
    return this.timeUnitsWidget?.value === "seconds";
  }

  formatTime(frames) {
    if (!this.isSecondsMode()) return String(frames);
    const s = frames / this.getFps();
    return `${s.toFixed(2).replace(/\.?0+$/, "")}s`;
  }

  formatLength(frames) {
    return this.isSecondsMode() ? this.formatTime(frames) : `${frames}f`;
  }

  buildDOM() {
    this.container.innerHTML = "";
    this.container.style.cssText = `
      display: flex; flex-direction: column; gap: 6px;
      padding: 6px 8px; box-sizing: border-box;
      font-family: sans-serif; font-size: 11px; color: #ddd;
      width: 100%; height: 100%;
    `;

    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText = `
      width: 100%; height: ${CANVAS_HEIGHT}px;
      display: block; background: #1a1a1a; border-radius: 4px;
      cursor: default;
    `;
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");

    this.textarea = document.createElement("textarea");
    this.textarea.placeholder = "点击上方段落以编辑提示词…";
    this.textarea.style.cssText = `
      width: 100%; min-height: 60px; flex: 1 1 auto;
      box-sizing: border-box; resize: none;
      background: #2a2a2a; color: #eee; border: 1px solid #444;
      border-radius: 4px; padding: 6px; font-family: inherit; font-size: 12px;
    `;
    this.textarea.readOnly = this.promptLocked;
    this.container.appendChild(this.textarea);

    const row = document.createElement("div");
    row.style.cssText = "display: flex; gap: 6px; align-items: center;";

    const lengthLabel = document.createElement("label");
    lengthLabel.style.cssText = "display: flex; align-items: center; gap: 4px;";
    lengthLabel.textContent = "长度:";
    this.lengthInput = document.createElement("input");
    this.lengthInput.type = "number";
    this.lengthInput.style.cssText = `
      width: 70px; background: #2a2a2a; color: #eee;
      border: 1px solid #444; border-radius: 3px; padding: 2px 4px;
    `;
    lengthLabel.appendChild(this.lengthInput);
    row.appendChild(lengthLabel);

    this.totalLabel = document.createElement("span");
    this.totalLabel.style.cssText = "color: #888; margin-left: 4px;";
    row.appendChild(this.totalLabel);

    const spacer = document.createElement("div");
    spacer.style.flex = "1";
    row.appendChild(spacer);

    this.addBtn = this.makeButton(
      "+ 添加",
      "添加新段落。若时间轴已满，将从末尾段落借用空间。",
    );
    this.updateBtn = this.makeButton(
      "更新",
      "无视锁定状态，从 text_input 重新读取段落提示词并自动更新 max_frames。",
    );
    this.distributeBtn = this.makeButton(
      "均分",
      "将所有段落设为相同长度，使总和恰好填满最大帧数。",
    );
    this.deleteBtn = this.makeButton(
      "删除",
      "删除当前选中的段落。仅剩一个段落时禁用。",
    );
    row.appendChild(this.addBtn);
    row.appendChild(this.updateBtn);
    row.appendChild(this.distributeBtn);
    row.appendChild(this.deleteBtn);

    // 初始锁定状态下禁用操作按钮
    if (this.promptLocked) {
      this.addBtn.disabled = true;
      this.distributeBtn.disabled = true;
      this.deleteBtn.disabled = true;
    }

    this.container.appendChild(row);
  }

  makeButton(label, tooltip) {
    const b = document.createElement("button");
    b.textContent = label;
    if (tooltip) b.title = tooltip;
    b.style.cssText = `
      background: #3a3a3a; color: #eee; border: 1px solid #555;
      border-radius: 3px; padding: 3px 10px; cursor: pointer; font-size: 11px;
    `;
    b.addEventListener("mouseenter", () => b.style.background = "#4a4a4a");
    b.addEventListener("mouseleave", () => b.style.background = "#3a3a3a");
    return b;
  }

  bindEvents() {
    this.canvas.addEventListener("pointerdown", e => { e.stopPropagation(); this.onPointerDown(e); });
    this.canvas.addEventListener("pointermove", e => { e.stopPropagation(); this.onPointerMove(e); });
    this.canvas.addEventListener("pointerup", e => { e.stopPropagation(); this.onPointerUp(e); });
    this.canvas.addEventListener("contextmenu", e => { e.preventDefault(); e.stopPropagation(); });
    this.canvas.addEventListener("wheel", e => e.stopPropagation(), { passive: true });
    this.canvas.addEventListener("pointerleave", () => {
      if (this.dragHandle < 0) {
        this.hoverIndex = -1;
        this.hoverHandle = -1;
        this.canvas.style.cursor = "default";
        this.render();
      }
    });
    this.textarea.addEventListener("wheel", e => e.stopPropagation(), { passive: true });
    this.textarea.addEventListener("pointerdown", e => e.stopPropagation());
    this.lengthInput.addEventListener("pointerdown", e => e.stopPropagation());
    this.lengthInput.addEventListener("wheel", e => e.stopPropagation(), { passive: true });

    this.textarea.addEventListener("input", () => {
      if (this.promptLocked) return;  // 锁定模式下不允许编辑
      const seg = this.timeline.segments[this.selectedIndex];
      if (!seg) return;
      seg.prompt = this.textarea.value;
      if (this.localPromptsWidget) {
        this.localPromptsWidget.value = this.timeline.segments.map(s => s.prompt).join(" | ");
      }
      this.render();
      if (this._textCommitTimer) clearTimeout(this._textCommitTimer);
      this._textCommitTimer = setTimeout(() => {
        this._textCommitTimer = null;
        this.commit();
      }, 120);
    });
    this.textarea.addEventListener("blur", () => {
      if (this._textCommitTimer) {
        clearTimeout(this._textCommitTimer);
        this._textCommitTimer = null;
        this.commit();
      }
    });
    this.lengthInput.addEventListener("focus", () => { this._inputBaseline = null; });
    this.lengthInput.addEventListener("blur", () => { this._inputBaseline = null; });
    this.lengthInput.addEventListener("input", () => {
      const idx = this.selectedIndex;
      const seg = this.timeline.segments[idx];
      if (!seg) return;
      const raw = parseFloat(this.lengthInput.value);
      if (!Number.isFinite(raw)) return;
      const frames = Math.max(
        MIN_SEGMENT_LENGTH,
        Math.round(this.isSecondsMode() ? raw * this.getFps() : raw),
      );
      if (!this._inputBaseline) {
        this._inputBaseline = this.timeline.segments.map(s => s.length);
      }
      this._setLengthShifting(idx, frames, this._inputBaseline);
      this.commit();
      this.render();
      this.updateTotalLabel();
    });

    this.addBtn.addEventListener("click", () => { if (!this.promptLocked) this.addSegment(); });
    this.updateBtn.addEventListener("click", () => { this.syncFromTextInput(); });
    this.distributeBtn.addEventListener("click", () => { if (!this.promptLocked) this.distributeEvenly(); });
    this.deleteBtn.addEventListener("click", () => { if (!this.promptLocked) this.deleteSelected(); });

    if (this.maxFramesWidget) {
      const prev = this.maxFramesWidget.callback;
      this.maxFramesWidget.callback = (...args) => {
        prev?.apply(this.maxFramesWidget, args);
        this.trimToFit();
        this.commit();
        this.updateUIFromSelection();
        this.render();
      };
    }
    for (const w of [this.fpsWidget, this.timeUnitsWidget]) {
      if (!w) continue;
      const prev = w.callback;
      w.callback = (...args) => {
        prev?.apply(w, args);
        this.updateUIFromSelection();
        this.render();
      };
    }

    // text_input 变化时智能分配到各段落（仅锁定模式下自动同步）
    if (this.textInputWidget) {
      const prevTI = this.textInputWidget.callback;
      this.textInputWidget.callback = (...args) => {
        prevTI?.apply(this.textInputWidget, args);
        if (this.promptLocked) this.syncFromTextInput();
      };
    }

    // prompt_lock 开关切换只读/编辑模式
    if (this.promptLockWidget) {
      const prevPL = this.promptLockWidget.callback;
      this.promptLockWidget.callback = (...args) => {
        prevPL?.apply(this.promptLockWidget, args);
        this._updateLockState();
        if (this.promptLocked) {
          // 切换为锁定时，从 text_input 重新同步
          this.syncFromTextInput();
        }
      };
    }

    this.resizeObserver = new ResizeObserver(() => this.resizeCanvas());
    this.resizeObserver.observe(this.container);
    this.resizeCanvas();
  }

  resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(50, Math.floor(this.canvas.offsetWidth));
    this.canvas.width = w * dpr;
    this.canvas.height = CANVAS_HEIGHT * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._cssWidth = w;
    this.render();
  }

  // ─── 布局 ───

  pxPerFrame() {
    return this._cssWidth / this.getMaxFrames();
  }

  segmentRects(order) {
    const segs = this.timeline.segments;
    const ord = order ?? this._getEffectiveOrder();
    const ppf = this.pxPerFrame();
    const rects = [];
    let cursor = 0;
    for (let visualPos = 0; visualPos < ord.length; visualPos++) {
      const idx = ord[visualPos];
      const len = segs[idx].length;
      rects.push({ index: idx, visualPos, x: cursor * ppf, w: len * ppf, frameStart: cursor, frameEnd: cursor + len });
      cursor += len;
    }
    return rects;
  }

  _getEffectiveOrder() {
    const n = this.timeline.segments.length;
    const natural = Array.from({ length: n }, (_, i) => i);
    if (!this.reorder?.active || this.reorder.targetIdx === this.reorder.sourceIdx) return natural;
    const order = natural.filter(i => i !== this.reorder.sourceIdx);
    order.splice(this.reorder.targetIdx, 0, this.reorder.sourceIdx);
    return order;
  }

  _computeReorderTarget() {
    const ppf = this.pxPerFrame();
    const sourceIdx = this.reorder.sourceIdx;
    const sourceLen = this.timeline.segments[sourceIdx].length;
    const blockCenterFrame =
      (this.reorder.cursorX - this.reorder.dragOffsetPx) / ppf + sourceLen / 2;

    const others = [];
    for (let i = 0; i < this.timeline.segments.length; i++) {
      if (i !== sourceIdx) others.push(this.timeline.segments[i].length);
    }

    let bestTarget = 0;
    let bestDist = Infinity;
    let cum = 0;
    for (let target = 0; target <= others.length; target++) {
      const sourceCenter = cum + sourceLen / 2;
      const dist = Math.abs(blockCenterFrame - sourceCenter);
      if (dist < bestDist) { bestDist = dist; bestTarget = target; }
      if (target < others.length) cum += others[target];
    }
    return bestTarget;
  }

  hitBoundary(mx) {
    const rects = this.segmentRects();
    for (let i = 0; i < rects.length; i++) {
      const right = rects[i].x + rects[i].w;
      if (Math.abs(mx - right) <= HANDLE_HIT_PX) return i;
    }
    return -1;
  }

  hitBlock(mx, my) {
    if (my < RULER_HEIGHT) return -1;
    const rects = this.segmentRects();
    for (const r of rects) {
      if (mx >= r.x && mx < r.x + r.w) return r.index;
    }
    return -1;
  }

  // ─── 指针事件 ───

  onPointerDown(e) {
    const { x, y } = this.localPos(e);
    this._settling = false;
    const handle = this.hitBoundary(x);
    if (handle >= 0) {
      this.dragHandle = handle;
      const segs = this.timeline.segments;
      this.dragStart = {
        x,
        initialLengths: segs.map(s => s.length),
      };
      this.canvas.setPointerCapture(e.pointerId);
      return;
    }
    const block = this.hitBlock(x, y);
    if (block >= 0) {
      this.selectedIndex = block;
      this.updateUIFromSelection();
      this.render();
      const sourceX = this._displayedX.get(block) ?? 0;
      this.reorder = {
        sourceIdx: block, targetIdx: block,
        startX: x, startY: y,
        cursorX: x,
        dragOffsetPx: x - sourceX,
        active: false,
      };
      try { this.canvas.setPointerCapture(e.pointerId); } catch (_) {}
    }
  }

  onPointerMove(e) {
    const { x, y } = this.localPos(e);
    if (this.dragHandle >= 0) {
      const ppf = this.pxPerFrame();
      const dxFrames = Math.round((x - this.dragStart.x) / ppf);
      const handle = this.dragHandle;
      const initial = this.dragStart.initialLengths;
      this._setLengthShifting(handle, initial[handle] + dxFrames, initial);

      const segs = this.timeline.segments;
      this.commit();
      if (segs[this.selectedIndex]) this.lengthInput.value = this.lengthInputValueFor(segs[this.selectedIndex].length);
      this.updateTotalLabel();
      this.render();
      return;
    }

    if (this.reorder) {
      const dx = x - this.reorder.startX;
      const dy = y - this.reorder.startY;
      if (!this.reorder.active && Math.hypot(dx, dy) > REORDER_THRESHOLD_PX) {
        this.reorder.active = true;
        this.canvas.style.cursor = "grabbing";
      }
      if (this.reorder.active) {
        this.reorder.cursorX = x;
        this.reorder.targetIdx = this._computeReorderTarget();
        this.render();
        return;
      }
    }

    const handle = this.hitBoundary(x);
    const block = handle >= 0 ? -1 : this.hitBlock(x, y);
    if (handle !== this.hoverHandle || block !== this.hoverIndex) {
      this.hoverHandle = handle;
      this.hoverIndex = block;
      this.canvas.style.cursor = handle >= 0 ? "ew-resize" : (block >= 0 ? "pointer" : "default");
      this.render();
    }
  }

  onPointerUp(e) {
    if (this.dragHandle >= 0) {
      try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      this.dragHandle = -1;
      this.dragStart = null;
    }
    if (this.reorder) {
      try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      if (this.reorder.active) {
        this.canvas.style.cursor = "default";
        const { sourceIdx, targetIdx } = this.reorder;
        if (sourceIdx !== targetIdx) {
          const effective = this._getEffectiveOrder();
          const oldDisplayed = new Map(this._displayedX);
          const seg = this.timeline.segments.splice(sourceIdx, 1)[0];
          this.timeline.segments.splice(targetIdx, 0, seg);
          this._displayedX = new Map();
          for (let newIdx = 0; newIdx < effective.length; newIdx++) {
            const oldIdx = effective[newIdx];
            if (oldDisplayed.has(oldIdx)) this._displayedX.set(newIdx, oldDisplayed.get(oldIdx));
          }
          this.selectedIndex = targetIdx;
          this.commit();
          this.updateUIFromSelection();
          this._settling = true;
        }
      }
      this.reorder = null;
      this.render();
    }
  }

  localPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const sx = (rect.width / this.canvas.offsetWidth) || 1;
    const sy = (rect.height / this.canvas.offsetHeight) || 1;
    return {
      x: (e.clientX - rect.left) / sx,
      y: (e.clientY - rect.top) / sy,
    };
  }

  // ─── 变更操作 ───

  addSegment() {
    const max = this.getMaxFrames();
    const n = this.timeline.segments.length;
    if (max < (n + 1) * MIN_SEGMENT_LENGTH) return;

    const desired = Math.max(MIN_SEGMENT_LENGTH, Math.floor(max / (n + 1)));
    const newIdx = n;
    const usedColors = new Set(this.timeline.segments.map(s => s.color));
    this.timeline.segments.push({ prompt: "", length: desired, color: pickColor(usedColors) });
    this.trimToFit(newIdx);

    let total = this.timeline.segments.reduce((a, s) => a + s.length, 0);
    if (total > max) {
      this.timeline.segments[newIdx].length -= (total - max);
    }

    this.selectedIndex = newIdx;
    this.commit();
    this.updateUIFromSelection();
    this.updateTotalLabel();
    this.render();
  }

  maxLengthFor(idx) {
    const max = this.getMaxFrames();
    let others = 0;
    for (let i = 0; i < this.timeline.segments.length; i++) {
      if (i !== idx) others += this.timeline.segments[i].length;
    }
    return Math.max(MIN_SEGMENT_LENGTH, max - others);
  }

  _setLengthShifting(idx, newLen, baseline) {
    const segs = this.timeline.segments;
    const max = this.getMaxFrames();
    for (let i = 0; i < segs.length; i++) segs[i].length = baseline[i];
    segs[idx].length = Math.max(MIN_SEGMENT_LENGTH, newLen);
    let total = segs.reduce((a, s) => a + s.length, 0);
    for (let i = idx + 1; i < segs.length && total > max; i++) {
      const reducible = segs[i].length - MIN_SEGMENT_LENGTH;
      const take = Math.min(reducible, total - max);
      segs[i].length -= take;
      total -= take;
    }
    if (total > max) segs[idx].length -= (total - max);
  }

  trimToFit(protectIndex = -1) {
    const max = this.getMaxFrames();
    let total = this.timeline.segments.reduce((a, s) => a + s.length, 0);
    for (let i = this.timeline.segments.length - 1; i >= 0 && total > max; i--) {
      if (i === protectIndex) continue;
      const seg = this.timeline.segments[i];
      const reducible = seg.length - MIN_SEGMENT_LENGTH;
      const take = Math.min(reducible, total - max);
      seg.length -= take;
      total -= take;
    }
  }

  distributeEvenly() {
    const max = this.getMaxFrames();
    const n = this.timeline.segments.length;
    if (n === 0) return;
    const base = Math.max(MIN_SEGMENT_LENGTH, Math.floor(max / n));
    const remainder = Math.max(0, max - base * n);
    for (let i = 0; i < n; i++) {
      this.timeline.segments[i].length = base + (i < remainder ? 1 : 0);
    }
    this.commit();
    this.updateUIFromSelection();
    this.render();
  }

  deleteSelected() {
    if (this.timeline.segments.length <= 1) return;
    this.timeline.segments.splice(this.selectedIndex, 1);
    this.selectedIndex = clamp(this.selectedIndex, 0, this.timeline.segments.length - 1);
    this.commit();
    this.updateUIFromSelection();
    this.updateTotalLabel();
    this.render();
  }

  // ─── 持久化 ───

  commit() {
    this.syncWidgetsFromTimeline();
    this.node.graph?.setDirtyCanvas?.(true, true);
  }

  syncWidgetsFromTimeline() {
    const segs = this.timeline.segments;
    if (this.timelineDataWidget) this.timelineDataWidget.value = JSON.stringify(this.timeline);
    if (this.localPromptsWidget) this.localPromptsWidget.value = segs.map(s => s.prompt).join(" | ");
    if (this.segmentLengthsWidget) this.segmentLengthsWidget.value = segs.map(s => s.length).join(", ");
  }

  // ─── text_input 智能分配 ───

  _buildTimelineFromLines(lines) {
    const max = this.getMaxFrames();

    // 尝试按时间格式解析每行：如 "0-3s 提示词A" 或 "3-5秒 提示词B"
    const parsed = [];
    for (const line of lines) {
      const match = line.match(TIME_RANGE_PATTERN);
      if (match) {
        const startSec = parseFloat(match[1]);
        const endSec = parseFloat(match[2]);
        const prompt = line.substring(match[0].length).trim();
        if (prompt) {
          parsed.push({
            prompt,
            startSec,
            endSec,
            durationSec: Math.max(0, endSec - startSec),
          });
        }
      }
    }

    // 只有所有行都匹配时间格式时，才启用动态时长分配
    if (parsed.length > 0 && parsed.length === lines.length) {
      // --- 动态时长分布：按时间格式分配帧数 ---
      const fps = this.getFps();

      // 从时间段中读取最大结束时间，自动计算 max_frames
      const maxEndSec = Math.max(...parsed.map(p => p.endSec));
      const rawMax = Math.floor(maxEndSec * fps) + 1;
      // 对齐 LTXV 时间步长 (8): 实际输出帧 = (max//8)*8+1，确保 max 与此一致
      const max = (Math.floor((rawMax - 2) / 8) + 1) * 8 + 1;

      // 同步 max_frames 到 UI 数值显示
      if (this.maxFramesWidget) {
        this.maxFramesWidget.value = max;
      }

      const frameAllocations = parsed.map(p =>
        Math.max(MIN_SEGMENT_LENGTH, Math.round(p.durationSec * fps)),
      );
      const totalFrames = frameAllocations.reduce((a, b) => a + b, 0);

      if (totalFrames > max) {
        // 时间轴已满：从末尾段落借用空间
        const excess = totalFrames - max;
        frameAllocations[frameAllocations.length - 1] = Math.max(
          MIN_SEGMENT_LENGTH,
          frameAllocations[frameAllocations.length - 1] - excess,
        );
      } else if (totalFrames < max) {
        // 末尾段落未填满：填充剩余时间段
        frameAllocations[frameAllocations.length - 1] += (max - totalFrames);
      }

      const usedColors = new Set();
      const segments = parsed.map((p, i) => {
        const color = PALETTE[i % PALETTE.length];
        usedColors.add(color);
        return { prompt: p.prompt, length: frameAllocations[i], color };
      });
      return { segments };
    }

    // --- 无时间格式：按原先均分逻辑处理 ---
    const baseLen = Math.max(MIN_SEGMENT_LENGTH, Math.floor(max / lines.length));
    const usedColors = new Set();
    const segments = lines.map((line, i) => {
      const color = PALETTE[i % PALETTE.length];
      usedColors.add(color);
      return { prompt: line, length: baseLen, color };
    });
    // 修正最后一段长度使总和恰好等于 max
    let total = segments.reduce((a, s) => a + s.length, 0);
    if (total !== max && segments.length > 0) {
      segments[segments.length - 1].length += (max - total);
      if (segments[segments.length - 1].length < MIN_SEGMENT_LENGTH) {
        segments[segments.length - 1].length = MIN_SEGMENT_LENGTH;
      }
    }
    return { segments };
  }

  // ─── 从上游连接节点读取文本值 ───

  _readConnectedTextInput() {
    const node = this.node;
    if (!node || !node.graph) return null;
    // 查找 text_input 输入槽位
    const inputIdx = node.findInputSlot ? node.findInputSlot("text_input") : -1;
    if (inputIdx < 0) return null;
    const linkId = node.inputs[inputIdx]?.link;
    if (linkId == null) return null;

    const link = node.graph.links[linkId];
    if (!link) return null;
    const srcNode = node.graph.getNodeById(link.origin_id);
    if (!srcNode) return null;

    // 尝试从源节点获取文本：优先 STRING 类型 widget，其次按名称推断
    const candidates = [];
    for (const w of (srcNode.widgets || [])) {
      if (w.type === "customtext" || w.name === "string" || (w.name && /^(text|prompt|string|multiline)$/i.test(w.name))) {
        candidates.push(w);
      }
    }
    // 若没匹配到，选取所有非 hidden 的文本型 widget
    if (candidates.length === 0) {
      for (const w of (srcNode.widgets || [])) {
        if (!w.hidden && typeof w.value === "string" && w.value.trim()) {
          candidates.push(w);
        }
      }
    }
    if (candidates.length === 0) return null;

    // 优先用名为 "string" 的 widget，否则合并所有候选内容
    const stringW = candidates.find(w => w.name === "string");
    if (stringW && stringW.value?.trim()) return stringW.value;

    const combined = candidates.map(w => w.value).filter(v => typeof v === "string" && v.trim()).join("\n");
    return combined || null;
  }

  // ─── 锁定状态切换 ───

  _updateLockState() {
    this.promptLocked = this.promptLockWidget?.value !== false;
    if (this.textarea) {
      this.textarea.readOnly = this.promptLocked;
      this.textarea.placeholder = this.promptLocked
        ? "提示词已锁定 — 切换「prompt_lock」为 false 以自由编辑"
        : "点击上方段落以编辑提示词…";
    }
    // 锁定模式下禁用添加/均分/删除按钮
    for (const btn of [this.addBtn, this.distributeBtn, this.deleteBtn]) {
      if (btn) btn.disabled = this.promptLocked;
    }
  }

  // ─── 获取 text_input 的可用文本（widget 优先，连接源兜底） ───

  _getTextInputValue() {
    // widget 已有内容（用户手动输入/粘贴）
    const widgetVal = this.textInputWidget?.value?.trim();
    if (widgetVal) return widgetVal;
    // 否则尝试从上游连接节点读取
    return this._readConnectedTextInput();
  }

  // ─── text_input 智能分配 ───

  syncFromTextInput() {
    const raw = this._getTextInputValue();
    if (!raw || !raw.trim()) {
      // text_input 为空时重置为默认时间轴（清空编辑器旧内容）
      this.timeline = defaultTimeline(this.getMaxFrames());
      this.selectedIndex = 0;
      this.commit();
      this.updateUIFromSelection();
      this.render();
      return;
    }
    const lines = raw.split("\n").map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length === 0) return;

    // 直接用行文本重建整个时间轴
    this.timeline = this._buildTimelineFromLines(lines);

    this.selectedIndex = clamp(this.selectedIndex, 0, this.timeline.segments.length - 1);
    this.commit();
    this.updateUIFromSelection();
    this.render();
  }

  // ─── UI 同步 ───

  lengthInputValueFor(frames) {
    if (!this.isSecondsMode()) return String(frames);
    return (frames / this.getFps()).toFixed(3).replace(/\.?0+$/, "");
  }

  updateUIFromSelection() {
    const seg = this.timeline.segments[this.selectedIndex];
    if (!seg) {
      this.textarea.value = "";
      this.lengthInput.value = "";
    } else {
      if (this.textarea.value !== seg.prompt) this.textarea.value = seg.prompt;
      this.lengthInput.value = this.lengthInputValueFor(seg.length);
    }
    this.lengthInput.step = this.isSecondsMode() ? (1 / this.getFps()).toFixed(4) : "1";
    this.lengthInput.min = this.isSecondsMode() ? (MIN_SEGMENT_LENGTH / this.getFps()).toFixed(4) : MIN_SEGMENT_LENGTH;
    this._inputBaseline = null;
    this.updateTotalLabel();
  }

  updateTotalLabel() {
    const total = this.timeline.segments.reduce((a, s) => a + s.length, 0);
    const max = this.getMaxFrames();
    if (this.isSecondsMode()) {
      const fps = this.getFps();
      const fmt = (f) => (f / fps).toFixed(2).replace(/\.?0+$/, "");
      this.totalLabel.textContent = `总计: ${fmt(total)} / ${fmt(max)} 秒 @ ${fps}fps`;
    } else {
      this.totalLabel.textContent = `总计: ${total} / ${max} 帧`;
    }
  }

  // ─── 渲染 ───

  render() {
    const rects = this.segmentRects();
    this._targetX = new Map();
    for (const r of rects) this._targetX.set(r.index, r.x);

    if (this.reorder?.active) {
      const sourceIdx = this.reorder.sourceIdx;
      const sourcePos = this.reorder.cursorX - this.reorder.dragOffsetPx;
      this._targetX.set(sourceIdx, sourcePos);
      this._displayedX.set(sourceIdx, sourcePos);
      this._kickAnim();
    } else if (this._settling) {
      this._kickAnim();
    } else {
      for (const [idx, target] of this._targetX) this._displayedX.set(idx, target);
      this._draw();
    }
  }

  _kickAnim() {
    if (this._animRaf) return;
    this._animRaf = requestAnimationFrame(() => this._tick());
  }

  _tick() {
    this._animRaf = null;
    let needsMore = false;
    const speed = 0.15;
    for (const [idx, target] of this._targetX) {
      const cur = this._displayedX.get(idx);
      if (cur === undefined) { this._displayedX.set(idx, target); continue; }
      const diff = target - cur;
      if (Math.abs(diff) < 0.3) {
        this._displayedX.set(idx, target);
      } else {
        this._displayedX.set(idx, cur + diff * speed);
        needsMore = true;
      }
    }
    this._draw();
    if (needsMore) {
      this._animRaf = requestAnimationFrame(() => this._tick());
    } else if (!this.reorder?.active) {
      this._settling = false;
    }
  }

  _draw() {
    const ctx = this.ctx;
    const w = this._cssWidth;
    ctx.clearRect(0, 0, w, CANVAS_HEIGHT);
    this.drawRuler(ctx, w);
    this.drawSegments(ctx, w);
  }

  drawRuler(ctx, w) {
    const max = this.getMaxFrames();
    ctx.fillStyle = "#222";
    ctx.fillRect(0, 0, w, RULER_HEIGHT);

    const ppf = this.pxPerFrame();
    const targetLabelSpacing = 60;

    let step;
    if (this.isSecondsMode()) {
      const fps = this.getFps();
      const target = targetLabelSpacing / (ppf * fps);
      const nice = [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
      let chosen = nice[nice.length - 1];
      for (const s of nice) { if (s >= target) { chosen = s; break; } }
      step = Math.max(1, Math.round(chosen * fps));
    } else {
      step = Math.max(1, Math.round(targetLabelSpacing / ppf));
      const niceSteps = [1, 2, 4, 5, 8, 10, 16, 20, 25, 50, 100];
      for (const s of niceSteps) { if (s >= step) { step = s; break; } }
    }

    ctx.strokeStyle = "#444";
    ctx.fillStyle = "#aaa";
    ctx.font = "10px sans-serif";
    ctx.textBaseline = "top";
    ctx.lineWidth = 1;

    for (let f = 0; f <= max; f += step) {
      const x = Math.floor(f * ppf) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, RULER_HEIGHT - 6);
      ctx.lineTo(x, RULER_HEIGHT);
      ctx.stroke();
      ctx.fillText(this.formatTime(f), x + 2, 2);
    }
    const xMax = Math.floor(max * ppf) - 0.5;
    ctx.strokeStyle = "#666";
    ctx.beginPath();
    ctx.moveTo(xMax, 0);
    ctx.lineTo(xMax, RULER_HEIGHT);
    ctx.stroke();
  }

  drawSegments(ctx, w) {
    const rects = this.segmentRects();
    const blockY = RULER_HEIGHT + 2;
    const blockH = BLOCK_HEIGHT - 4;

    ctx.fillStyle = "#101010";
    ctx.fillRect(0, blockY, w, blockH);

    const dragIdx = this.reorder?.active ? this.reorder.sourceIdx : -1;
    const rendered = [...rects].sort((a, b) => (a.index === dragIdx ? 1 : 0) - (b.index === dragIdx ? 1 : 0));

    for (const r of rendered) {
      const seg = this.timeline.segments[r.index];
      const color = seg.color || PALETTE[r.index % PALETTE.length];
      const isSelected = r.index === this.selectedIndex;
      const isHover = r.index === this.hoverIndex;
      const isDragging = r.index === dragIdx;

      const drawX = Math.floor(this._displayedX.get(r.index) ?? r.x);
      const drawW = Math.max(2, Math.floor(r.w));

      ctx.fillStyle = color;
      ctx.globalAlpha = isDragging ? 0.9 : (isSelected ? 1.0 : (isHover ? 0.9 : 0.75));
      ctx.fillRect(drawX, blockY, drawW, blockH);
      ctx.globalAlpha = 1.0;

      ctx.strokeStyle = isDragging ? "#ffd54f" : (isSelected ? "#fff" : "rgba(0,0,0,0.4)");
      ctx.lineWidth = isDragging || isSelected ? 2 : 1;
      ctx.strokeRect(drawX + 0.5, blockY + 0.5, drawW - 1, blockH - 1);

      ctx.fillStyle = "#fff";
      ctx.font = "11px sans-serif";
      ctx.textBaseline = "top";
      const label = seg.prompt || `(段落 ${r.index + 1})`;
      const [line1, line2] = this.wrapTwoLines(ctx, label, drawW - 8);
      ctx.fillText(line1, drawX + 4, blockY + 4);
      if (line2) ctx.fillText(line2, drawX + 4, blockY + 18);

      ctx.fillStyle = "rgba(255,255,255,0.75)";
      ctx.font = "10px monospace";
      const range = `${this.formatTime(r.frameStart)}–${this.formatTime(r.frameEnd)} (${this.formatLength(seg.length)})`;
      const rangeTrunc = this.truncateText(ctx, range, drawW - 8);
      ctx.fillText(rangeTrunc, drawX + 4, blockY + blockH - 14);
    }

    if (!this.reorder?.active) {
      for (let i = 0; i < rects.length; i++) {
        const r = rects[i];
        const drawX = this._displayedX.get(r.index) ?? r.x;
        const right = Math.floor(drawX + r.w);
        const isHover = i === this.hoverHandle || i === this.dragHandle;
        ctx.fillStyle = isHover ? "#fff" : "rgba(255,255,255,0.4)";
        ctx.fillRect(right - 1, blockY + 4, 2, blockH - 8);
      }
    }
  }

  truncateText(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) return text;
    let lo = 0, hi = text.length;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (ctx.measureText(text.slice(0, mid) + "…").width <= maxWidth) lo = mid;
      else hi = mid - 1;
    }
    return lo > 0 ? text.slice(0, lo) + "…" : "";
  }

  wrapTwoLines(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) return [text, ""];

    const tokens = text.split(/(\s+)/);
    let line1 = "";
    let consumed = 0;
    for (let i = 0; i < tokens.length; i++) {
      const candidate = line1 + tokens[i];
      if (ctx.measureText(candidate).width > maxWidth) break;
      line1 = candidate;
      consumed = i + 1;
    }

    if (!line1.trim()) return [this.truncateText(ctx, text, maxWidth), ""];

    let line2 = tokens.slice(consumed).join("").trim();
    if (!line2) return [line1.trimEnd(), ""];
    if (ctx.measureText(line2).width > maxWidth) {
      line2 = this.truncateText(ctx, line2, maxWidth);
    }
    return [line1.trimEnd(), line2];
  }

  destroy() {
    this.resizeObserver?.disconnect();
    if (this._animRaf) cancelAnimationFrame(this._animRaf);
    if (this._textCommitTimer) {
      clearTimeout(this._textCommitTimer);
      this._textCommitTimer = null;
      try { this.commit(); } catch (_) {}
    }
  }
}

// 工作流兼容：旧版本保存的工作流缺少 fps/time_units，恢复为默认值
const APPENDED_WIDGET_DEFAULTS = [["fps", 24.0], ["time_units", "frames"]];

app.registerExtension({
  name: "YuanTool.CLIPTimeline",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "YuanCLIPTimeline") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);

      for (const name of HIDDEN_WIDGET_NAMES) {
        hideWidget(this.widgets.find(w => w.name === name));
      }

      const container = document.createElement("div");
      this._timelineWidget = this.addDOMWidget("yuan_clip_timeline", "YuanCLIPTimeline", container, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 220,
        getHeight: () => 220,
      });

      const self = this;
      setTimeout(() => {
        try {
          self._timelineEditor = new TimelineEditor(self, container);
        } catch (err) {
          console.error("[Yuan CLIP Timeline] 时间轴编辑器初始化失败:", err);
        }
      }, 0);

      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        this._timelineEditor?.destroy();
        return onRemoved?.apply(this, arguments);
      };

      const onConfigure = this.onConfigure;
      this.onConfigure = function (info) {
        const out = onConfigure?.apply(this, arguments);
        for (const [name, def] of APPENDED_WIDGET_DEFAULTS) {
          const w = this.widgets.find(x => x.name === name);
          if (w && (w.value == null || w.value === "")) w.value = def;
        }
        setTimeout(() => {
          if (this._timelineEditor) {
            // 同步锁定状态（onConfigure 可能已重设 widget 值）
            this._timelineEditor._updateLockState();
            // 锁定模式下优先从 text_input 构建段落，否则从 timeline_data 解析
            const textVal = this._timelineEditor.promptLocked
              ? this._timelineEditor._getTextInputValue()
              : null;
            if (textVal) {
              const lines = textVal.split("\n").map(l => l.trim()).filter(l => l.length > 0);
              if (lines.length > 0) {
                this._timelineEditor.timeline = this._timelineEditor._buildTimelineFromLines(lines);
              } else {
                this._timelineEditor.timeline = parseInitial(
                  this._timelineEditor.timelineDataWidget?.value,
                  this._timelineEditor.getMaxFrames(),
                );
              }
            } else {
              this._timelineEditor.timeline = parseInitial(
                this._timelineEditor.timelineDataWidget?.value,
                this._timelineEditor.getMaxFrames(),
              );
            }
            this._timelineEditor.selectedIndex = clamp(
              this._timelineEditor.selectedIndex, 0,
              this._timelineEditor.timeline.segments.length - 1,
            );
            this._timelineEditor.updateUIFromSelection();
            this._timelineEditor.render();
          }
        }, 10);
        return out;
      };

      // 监听 text_input 端口连接/断开，自动触发同步
      const origOnConnectionsChange = this.onConnectionsChange;
      this.onConnectionsChange = function (type, slot, isConnect, linkInfo, ioSlot) {
        if (origOnConnectionsChange) {
          origOnConnectionsChange.apply(this, arguments);
        }
        if (ioSlot?.name === "text_input") {
          setTimeout(() => {
            if (this._timelineEditor && this._timelineEditor.promptLocked) {
              this._timelineEditor.syncFromTextInput();
            }
          }, isConnect ? 150 : 50);
        }
      };

      return r;
    };
  },
});

// ============================================================================
// @ 标记自动补全扩展（复刻自 ComfyUI-AtSignRef）
// 在 global_prompt / text_input / 时间轴编辑器中输入 @ 时弹出角色选择菜单
// ============================================================================
(function() {
    const LOG = "[Timeline@]";

    // ── 获取 ComfyApp graph ──
    function getGraph() {
        const app = window.comfyAPI?.app?.app;
        if (app?.graph) return app.graph;
        if (app?.canvas?.graph) return app.canvas.graph;
        if (window.app?.graph) return window.app.graph;
        if (window.canvas?.graph) return window.canvas.graph;
        return null;
    }

    // ── 找到所有 YuanCLIPTimeline 节点 ──
    function findTimelineNodes() {
        const graph = getGraph();
        if (!graph?._nodes) return [];
        const nodes = graph._nodes;
        if (!Array.isArray(nodes)) return [];
        return nodes.filter(n => n && n.type === 'YuanCLIPTimeline');
    }

    // ── 找到与 textarea 关联的 Timeline 节点 ──
    function findNodeByTextarea(textEl) {
        try {
            const nodes = findTimelineNodes();
            for (const node of nodes) {
                // 策略1: 匹配 widget 的 inputEl（global_prompt、text_input 等标准 widget）
                if (node.widgets) {
                    for (const w of node.widgets) {
                        if (w.inputEl === textEl || w.element === textEl || w.canvas === textEl) {
                            return node;
                        }
                    }
                }
                // 策略2: 匹配时间轴编辑器的自定义 textarea
                if (node._timelineEditor && node._timelineEditor.textarea === textEl) {
                    return node;
                }
            }
            // 策略3: 只有一个 Timeline 节点时直接使用
            if (nodes.length === 1) return nodes[0];
        } catch (e) {
            console.error(LOG, "findNodeByTextarea error:", e);
        }
        return null;
    }

    // ── 从 global_prompt 解析 @标记和描述 ──
    // 与后端 _parse_yuan_map_config 保持一致的格式
    // 支持：@图1=描述  /  @图1:描述  /  @图1：描述
    function parseMarkersFromPrompt(promptText) {
        if (!promptText) return [];
        const markers = [];
        const lines = promptText.replace(/,/g, '\n').split('\n');
        for (const line of lines) {
            const trimmed = line.trim();
            const m = trimmed.match(/^@(\S+?)\s*[=:：]\s*(.+)/);
            if (m) {
                const name = '@' + m[1].trim();
                const desc = m[2].trim();
                if (!markers.find(x => x.name === name)) {
                    markers.push({ name, desc });
                }
            }
        }
        return markers;
    }

    // ── 构建并显示弹窗 ──
    function showPopup(textEl, node) {
        const existing = document.querySelector('.timeline-atsign-popup');
        if (existing) existing.remove();

        const gpWidget = node.widgets?.find(w => w.name === 'global_prompt');
        const promptText = gpWidget?.value || '';
        const markers = parseMarkersFromPrompt(promptText);

        if (markers.length === 0) return;

        const rect = textEl.getBoundingClientRect();
        const overlay = document.createElement('div');
        overlay.className = 'timeline-atsign-popup';
        overlay.style.cssText = `
            position: fixed;
            background: #2a2a2a;
            border: 1px solid #555;
            border-radius: 8px;
            padding: 6px;
            z-index: 99999;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            max-width: 500px;
            left: ${Math.max(10, Math.min(rect.left, window.innerWidth - 510))}px;
            top: ${rect.bottom + 4}px;
            min-width: 120px;
        `;

        markers.forEach((marker) => {
            const item = document.createElement('div');
            item.style.cssText = `
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                padding: 8px 12px;
                border-radius: 6px;
                transition: background 0.15s;
                background: #1e1e1e;
                border: 1px solid #444;
                min-width: 64px;
                min-height: 40px;
            `;
            item.onmouseenter = () => { item.style.background = '#444'; item.style.borderColor = '#777'; };
            item.onmouseleave = () => { item.style.background = '#1e1e1e'; item.style.borderColor = '#444'; };

            const nameEl = document.createElement('div');
            nameEl.textContent = marker.name;
            nameEl.style.cssText = `
                font-size: 14px;
                font-weight: bold;
                color: #e8a850;
                text-align: center;
            `;
            item.appendChild(nameEl);

            if (marker.desc) {
                const descEl = document.createElement('div');
                descEl.textContent = marker.desc;
                descEl.style.cssText = `
                    font-size: 10px;
                    color: #999;
                    margin-top: 2px;
                    text-align: center;
                    max-width: 120px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                `;
                item.appendChild(descEl);
            }

            item.onclick = function() {
                const val = textEl.value;
                const cursorPos = textEl.selectionStart;
                let insertText = marker.name + ' ';
                let start = cursorPos;

                if (cursorPos > 0 && val[cursorPos - 1] === '@') {
                    start = cursorPos - 1;
                } else if (cursorPos > 0) {
                    const beforeText = val.substring(0, cursorPos);
                    const lastAt = beforeText.lastIndexOf('@');
                    if (lastAt >= 0) {
                        const afterAt = beforeText.substring(lastAt + 1);
                        if (!afterAt.includes(' ') && !afterAt.includes('\n')) {
                            start = lastAt;
                        }
                    }
                }

                textEl.setRangeText(insertText, start, cursorPos, 'end');
                textEl.focus();
                textEl.dispatchEvent(new Event('input', { bubbles: true }));
                if (overlay.parentNode) overlay.remove();
            };

            overlay.appendChild(item);
        });

        document.body.appendChild(overlay);

        const closeHandler = function(e) {
            if (!overlay.contains(e.target) && e.target !== textEl) {
                if (overlay.parentNode) overlay.remove();
                document.removeEventListener('click', closeHandler);
                document.removeEventListener('focusin', closeHandler);
            }
        };
        setTimeout(() => {
            document.addEventListener('click', closeHandler);
            document.addEventListener('focusin', closeHandler);
        }, 10);
    }

    // ── 绑定 textarea @事件 ──
    function bindTextarea(textEl) {
        if (textEl.dataset.timelineAtInited) return;
        textEl.dataset.timelineAtInited = '1';

        function handleAt() {
            const node = findNodeByTextarea(textEl);
            if (!node) return;
            showPopup(textEl, node);
        }

        textEl.addEventListener('input', function() {
            const val = this.value;
            const cursorPos = this.selectionStart || 0;
            if (cursorPos > 0 && val[cursorPos - 1] === '@') handleAt();
        });

        textEl.addEventListener('click', function() {
            const val = this.value;
            const cursorPos = this.selectionStart || 0;
            if (cursorPos > 0 && val[cursorPos - 1] === '@') handleAt();
        });

        textEl.addEventListener('keyup', function(e) {
            if (e.key.startsWith('Arrow')) {
                const val = this.value;
                const cursorPos = this.selectionStart || 0;
                if (cursorPos > 0 && val[cursorPos - 1] === '@') handleAt();
            }
        });
    }

    // ── 扫描并绑定所有 textarea ──
    function scanTextareas() {
        const textareas = document.querySelectorAll('textarea');
        textareas.forEach(bindTextarea);
    }

    // ── 初始化 ──
    function init() {
        console.log(LOG, "Initializing @ mention for Yuan CLIP Timeline...");
        scanTextareas();
        setInterval(scanTextareas, 2000);

        if (window.MutationObserver) {
            const observer = new MutationObserver(() => {
                const all = document.querySelectorAll('textarea');
                const inited = document.querySelectorAll('textarea[data-timeline-at-inited]');
                if (inited.length < all.length) scanTextareas();
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }

        console.log(LOG, "Ready.");
    }

    // ── 等待 ComfyUI 就绪 ──
    function waitForReady(retries) {
        if (retries <= 0) {
            console.warn(LOG, "Timeout, init anyway");
            init();
            return;
        }
        const graph = getGraph();
        const textareas = document.querySelectorAll('textarea').length;
        if (graph && graph._nodes && textareas > 0) {
            init();
        } else {
            setTimeout(() => waitForReady(retries - 1), 1000);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => waitForReady(90));
    } else {
        waitForReady(90);
    }
})();
