// ============================================================================
// Yuan CLIPTimeline 多轨道可视化时间轴编辑器
// 单类 TimelineEditor + 单 canvas 多轨道 + CSS 注入
// 节点类型：YuanCLIPTimeline
// ============================================================================

const { app } = window.comfyAPI.app;

// ── 调色板 ──
const PALETTE = [
  "#4f8edc", "#e07b3a", "#5cb85c", "#d9534f", "#9b6cd6",
  "#a07060", "#e377c2", "#7f7f7f", "#c4c447", "#3fbac4",
];

// text_input 时间格式解析：匹配行首的 "0-3s", "3-5秒", "5-7s" 等
const TIME_RANGE_PATTERN = /^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*[s秒]\s*[：:]?\s*/;

// ── 维度常量 ──
const RULER_HEIGHT = 26;            // 标尺高度（容纳更清晰的时间刻度文字）
const BLOCK_HEIGHT = 88;            // 主轨高度（容纳更大缩略图和段标签）
const MOTION_TRACK_HEIGHT = 72;     // IC-LoRA 运动引导轨道高度
const AUDIO_TRACK_HEIGHT = 54;      // 音频轨道高度（容纳更清晰波形）
const CANVAS_HEIGHT = RULER_HEIGHT + BLOCK_HEIGHT + MOTION_TRACK_HEIGHT + AUDIO_TRACK_HEIGHT;
const CANVAS_MIN_WIDTH = 520;       // 画布最小宽度，过窄时允许横向滚动
const HANDLE_HIT_PX = 6;
const REORDER_THRESHOLD_PX = 6;
const MIN_SEGMENT_LENGTH = 1;
const CHUNK_SIZE = 50 * 1024 * 1024; // 50MB 分片

// ── 由编辑器自动管理的 widget（对用户隐藏） ──
const HIDDEN_WIDGET_NAMES = [
  "时间轴数据", "段落提示词", "段落长度", "引导强度",
  "起始帧", "自定义宽度", "自定义高度", "缩放方式",
  "整除数", "图像压缩", "使用自定义音频", "使用自定义运动",
  "覆盖音频", "时间单位", "提示词锁定",
  "运动图像帧数",
  "段落图像", "运动图像",   // 抑制默认 IMAGE 预览 widget，由编辑器自行管理
];

// 工作流兼容：旧版本缺少 帧率/时间单位 时恢复默认值
const APPENDED_WIDGET_DEFAULTS = [["帧率", 24.0], ["时间单位", "frames"]];

// ── CSS 样式（注入到 <style id="yuan-clip-timeline-styles">） ──
const STYLES = `
.yuan-clip-tl-container {
  display: flex; flex-direction: column; gap: 6px;
  padding: 8px 10px; box-sizing: border-box;
  font-family: sans-serif; font-size: 11px; color: #ddd;
  width: 100%; min-width: 320px;
}
.yuan-clip-tl-toolbar {
  display: flex; flex-wrap: wrap; gap: 4px; align-items: center;
  padding: 6px 8px; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 5px;
}
.yuan-clip-tl-toolbar .btn {
  background: #3a3a3a; color: #eee; border: 1px solid #555;
  border-radius: 4px; padding: 4px 12px; cursor: pointer; font-size: 11px;
  white-space: nowrap; line-height: 1.4; transition: background 0.15s, border-color 0.15s;
}
.yuan-clip-tl-toolbar .btn:hover { background: #4a4a4a; border-color: #6a6a6a; }
.yuan-clip-tl-toolbar .btn:disabled { opacity: 0.35; cursor: not-allowed; }
.yuan-clip-tl-toolbar .btn.active { background: #4f8edc; border-color: #6fa0e8; color: #fff; }
.yuan-clip-tl-toolbar .btn.danger { background: #5a3030; border-color: #7a4040; }
.yuan-clip-tl-toolbar .btn.danger:hover { background: #6a3838; }
.yuan-clip-tl-toolbar .btn.success { background: #2d4a2d; border-color: #3d5a3d; }
.yuan-clip-tl-toolbar .btn.success:hover { background: #355435; }
.yuan-clip-tl-toolbar .separator { width: 1px; height: 22px; background: #3a3a3a; margin: 0 4px; }
.yuan-clip-tl-toolbar .spacer { flex: 1; min-width: 8px; }
.yuan-clip-tl-total-label {
  color: #9a9a9a; font-size: 11px; padding: 2px 6px; white-space: nowrap;
}
.yuan-clip-tl-canvas-wrap {
  width: 100%; overflow-x: auto; overflow-y: hidden;
  background: #141414; border: 1px solid #2a2a2a; border-radius: 5px;
}
.yuan-clip-tl-canvas {
  display: block; background: #141414;
  min-width: 520px; width: 100%;
  cursor: default;
}
.yuan-clip-tl-properties {
  display: flex; flex-direction: column; gap: 8px;
  padding: 10px; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 5px;
  min-height: 60px;
}
.yuan-clip-tl-prop-section {
  display: flex; flex-direction: column; gap: 8px;
  padding: 8px 10px; background: #222; border-radius: 4px;
  border: 1px solid #2e2e2e;
}
.yuan-clip-tl-prop-title {
  font-size: 12px; font-weight: bold; color: #e8a850;
  padding-bottom: 4px; border-bottom: 1px solid #2e2e2e;
  display: flex; align-items: center; gap: 8px;
}
.yuan-clip-tl-prop-title .badge {
  background: #333; color: #aaa; font-size: 9px; padding: 1px 6px; border-radius: 8px;
  font-weight: normal;
}
.yuan-clip-tl-field-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
  gap: 8px 10px; align-items: end;
}
.yuan-clip-tl-properties .field {
  display: flex; flex-direction: column; gap: 3px; min-width: 0;
}
.yuan-clip-tl-properties .field.full { grid-column: 1 / -1; }
.yuan-clip-tl-properties label {
  font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 0.3px;
}
.yuan-clip-tl-properties input[type="text"],
.yuan-clip-tl-properties input[type="number"],
.yuan-clip-tl-properties select {
  background: #2a2a2a; color: #eee; border: 1px solid #444;
  border-radius: 3px; padding: 4px 6px; font-family: inherit; font-size: 11px;
  min-width: 0; width: 100%; box-sizing: border-box;
}
.yuan-clip-tl-properties textarea {
  background: #2a2a2a; color: #eee; border: 1px solid #444;
  border-radius: 3px; padding: 5px 7px; font-family: inherit; font-size: 12px;
  min-height: 54px; resize: vertical; width: 100%; box-sizing: border-box; line-height: 1.4;
}
.yuan-clip-tl-properties input[type="color"] {
  width: 100%; height: 26px; border: 1px solid #444; border-radius: 3px;
  background: #2a2a2a; padding: 0; cursor: pointer; box-sizing: border-box;
}
.yuan-clip-tl-properties input[type="range"] {
  width: 100%; cursor: pointer; accent-color: #4f8edc;
}
.yuan-clip-tl-properties input:disabled,
.yuan-clip-tl-properties textarea:disabled,
.yuan-clip-tl-properties select:disabled {
  opacity: 0.55; cursor: not-allowed;
}
.yuan-clip-tl-btn-row { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; }
.yuan-clip-tl-mini-btn {
  background: #3a3a3a; color: #eee; border: 1px solid #555;
  border-radius: 3px; padding: 4px 10px; cursor: pointer; font-size: 11px;
  white-space: nowrap; transition: background 0.15s;
}
.yuan-clip-tl-mini-btn:hover { background: #4a4a4a; }
.yuan-clip-tl-mini-btn.danger { background: #5a3030; border-color: #7a4040; }
.yuan-clip-tl-mini-btn.danger:hover { background: #6a3838; }
.yuan-clip-tl-strength-row { display: flex; gap: 6px; align-items: center; }
.yuan-clip-tl-strength-val {
  font-size: 10px; color: #aaa; min-width: 32px; text-align: right;
  font-variant-numeric: tabular-nums;
}
.yuan-clip-tl-checkbox-label {
  display: flex; align-items: center; gap: 5px; cursor: pointer; font-size: 11px; color: #ddd;
}
.yuan-clip-tl-file-name {
  font-size: 10px; color: #888; word-break: break-all; padding: 2px 4px;
  background: #1e1e1e; border-radius: 2px;
}
.yuan-clip-tl-image-preview {
  max-width: 100%; max-height: 110px; border: 1px solid #444; border-radius: 3px;
  display: block; margin-top: 4px; object-fit: contain;
}
.yuan-clip-tl-upload-progress {
  font-size: 11px; color: #e8a850; padding: 5px 10px;
  background: #2a2018; border: 1px solid #4a3a28; border-radius: 3px;
}
.yuan-clip-tl-context-menu {
  position: fixed; background: #2a2a2a; border: 1px solid #555;
  border-radius: 4px; padding: 4px; z-index: 99999;
  box-shadow: 0 4px 12px rgba(0,0,0,0.5); min-width: 130px;
}
.yuan-clip-tl-context-menu div.item {
  padding: 5px 12px; cursor: pointer; border-radius: 3px;
  font-size: 11px; color: #ddd;
}
.yuan-clip-tl-context-menu div.item:hover { background: #444; }
`;

// ── 工具函数 ──

function hideWidget(w) {
  if (!w) return;
  w.type = "hidden";
  w.hidden = true;
  w.computeSize = () => [0, -4];
}

// ── V3 (Nodes 2.0) 端口占位处理 ──
// V3 下被编辑器隐藏管理的参数会显示为空端口占位把节点拉长，此函数移除这些输入端口（保留真实可连接的 IMAGE 端口）。

function isV3Node(node) {
  if (node._yuanIsV3Checked) return !!node._yuanIsV3;
  node._yuanIsV3Checked = true;
  try {
    const el = node._timelineWidget?.element ||
      node.widgets?.find(w => w.element)?.element;
    let n = el;
    while (n) {
      if ((n.tagName && n.tagName.toLowerCase().includes('comfy-node')) ||
          (n.classList && n.classList.contains('comfy-node'))) {
        node._yuanIsV3 = true;
        break;
      }
      n = n.parentElement || (n.getRootNode ? n.getRootNode().host : null);
    }
  } catch (_) {}
  return !!node._yuanIsV3;
}

function hideManagedWidgetPorts(node) {
  if (!node || !Array.isArray(node.inputs) || !isV3Node(node)) return;
  for (let i = node.inputs.length - 1; i >= 0; i--) {
    const inp = node.inputs[i];
    if (!inp) continue;
    // 段落图像/运动图像 是真实图像输入端口（保留可连接），
    // 其余由编辑器隐藏管理的参数端口在 V3 下移除
    if (HIDDEN_WIDGET_NAMES.includes(inp.name) &&
        inp.name !== "段落图像" && inp.name !== "运动图像") {
      if (inp.link != null) continue; // 已有连线时保留端口，避免破坏连接
      try { node.removeInput(i); } catch (_) {}
    }
  }
}

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

function pickColor(existingColors) {
  for (const c of PALETTE) if (!existingColors.has(c)) return c;
  const idx = existingColors.size;
  const hue = (idx * 137.508) % 360;
  return `hsl(${hue.toFixed(0)}, 55%, 55%)`;
}

function injectStyles() {
  if (document.getElementById("yuan-clip-timeline-styles")) return;
  const style = document.createElement("style");
  style.id = "yuan-clip-timeline-styles";
  style.textContent = STYLES;
  document.head.appendChild(style);
}

// ── 默认时间轴（两段均分） ──
function defaultTimeline(maxFrames) {
  const half = Math.max(MIN_SEGMENT_LENGTH, Math.floor(maxFrames / 2));
  const mkSeg = (start, len, color) => ({
    start, length: len, prompt: "", color,
    type: "image", imageFile: "", imageB64: "",
    isEndFrame: false, strength: 1.0, trimStart: 0.0,
  });
  return {
    segments: [
      mkSeg(0, half, PALETTE[0]),
      mkSeg(half, Math.max(MIN_SEGMENT_LENGTH, maxFrames - half), PALETTE[1]),
    ],
    motionSegments: [],
    audioSegments: [],
    global_prompt: "",
  };
}

// ── 从 JSON 字符串解析时间轴（含 motion/audio 字段） ──
function parseInitial(jsonStr, maxFrames) {
  if (!jsonStr) return defaultTimeline(maxFrames);
  try {
    const obj = JSON.parse(jsonStr);
    const timeline = {
      segments: [],
      motionSegments: [],
      audioSegments: [],
      global_prompt: "",
    };

    if (Array.isArray(obj?.segments)) {
      timeline.segments = obj.segments.map((s, i) => ({
        start: parseInt(s.start, 10) || 0,
        length: Math.max(MIN_SEGMENT_LENGTH, parseInt(s.length, 10) || MIN_SEGMENT_LENGTH),
        prompt: typeof s.prompt === "string" ? s.prompt : "",
        color: typeof s.color === "string" ? s.color : PALETTE[i % PALETTE.length],
        type: s.type === "video" ? "video" : "image",
        imageFile: typeof s.imageFile === "string" ? s.imageFile : "",
        imageB64: typeof s.imageB64 === "string" ? s.imageB64 : "",
        isEndFrame: !!s.isEndFrame,
        strength: Number.isFinite(s.strength) ? s.strength : 1.0,
        trimStart: Number.isFinite(s.trimStart) ? s.trimStart : 0.0,
      }));
    }
    if (Array.isArray(obj?.motionSegments)) {
      timeline.motionSegments = obj.motionSegments.map(m => ({
        videoFile: typeof m.videoFile === "string" ? m.videoFile : "",
        frameFiles: Array.isArray(m.frameFiles) ? m.frameFiles : [],
        start: parseInt(m.start, 10) || 0,
        length: Math.max(MIN_SEGMENT_LENGTH, parseInt(m.length, 10) || MIN_SEGMENT_LENGTH),
        trimStart: Number.isFinite(m.trimStart) ? m.trimStart : 0.0,
        isStaticImage: !!m.isStaticImage,
        fileName: typeof m.fileName === "string" ? m.fileName : "",
        description: typeof m.description === "string" ? m.description : "",
        subjectNum: Number.isInteger(m.subjectNum) ? m.subjectNum : 0,
      }));
    }
    if (Array.isArray(obj?.audioSegments)) {
      timeline.audioSegments = obj.audioSegments.map(a => ({
        audioFile: typeof a.audioFile === "string" ? a.audioFile : "",
        audioB64: typeof a.audioB64 === "string" ? a.audioB64 : "",
        start: parseInt(a.start, 10) || 0,
        length: Math.max(MIN_SEGMENT_LENGTH, parseInt(a.length, 10) || MIN_SEGMENT_LENGTH),
        trimStart: Number.isFinite(a.trimStart) ? a.trimStart : 0.0,
        fileName: typeof a.fileName === "string" ? a.fileName : "",
      }));
    }
    if (typeof obj?.global_prompt === "string") timeline.global_prompt = obj.global_prompt;

    if (timeline.segments.length > 0) return timeline;
  } catch (_) {}
  return defaultTimeline(maxFrames);
}

// ============================================================================
// TimelineEditor — 集中式多轨道时间轴编辑器
// ============================================================================

class TimelineEditor {
  constructor(node, container) {
    this.node = node;
    this.container = container;

    // ── 绑定 widget 引用 ──
    const find = (name) => node.widgets?.find(w => w.name === name);
    this.maxFramesWidget = find("最大帧数");
    this.fpsWidget = find("帧率");
    this.timeUnitsWidget = find("时间单位");
    this.timelineDataWidget = find("时间轴数据");
    this.localPromptsWidget = find("段落提示词");
    this.segmentLengthsWidget = find("段落长度");
    this.guideStrengthWidget = find("引导强度");
    this.textInputWidget = find("文本输入");
    this.promptLockWidget = find("提示词锁定");
    this.motionImageFramesWidget = find("运动图像帧数");
    this.globalPromptWidget = find("全局提示词");
    this.useCustomAudioWidget = find("使用自定义音频");
    this.useCustomMotionWidget = find("使用自定义运动");
    this.widthWidget = find("宽度");
    this.heightWidget = find("高度");

    this.promptLocked = this.promptLockWidget?.value !== false; // 默认锁定
    this.motionImageFrames = parseInt(this.motionImageFramesWidget?.value) || 16;  // 默认 16 帧

    // ── 初始化时间轴 ──
    const textVal = this.promptLocked ? this._getTextInputValue() : null;
    if (textVal) {
      const lines = textVal.split("\n").map(l => l.trim()).filter(l => l.length > 0);
      this.timeline = lines.length > 0
        ? this._buildTimelineFromLines(lines)
        : parseInitial(this.timelineDataWidget?.value, this.getMaxFrames());
    } else {
      this.timeline = parseInitial(this.timelineDataWidget?.value, this.getMaxFrames());
    }

    // ── 状态 ──
    this.selectedIndices = new Set([0]);
    this.selectedMotionIndices = new Set();
    this.selectedAudioIndices = new Set();
    this.hoverIndex = -1;
    this.hoverHandle = -1;
    this.dragHandle = -1;
    this.dragStart = null;
    this.reorder = null;
    this.dragMotion = null;
    this.dragAudio = null;
    this._textCommitTimer = null;
    this._cssWidth = 0;
    this.audioPeaksCache = new Map(); // filename → peaks 数组（内存缓存，不序列化）
    this._imageCache = new Map();     // segIndex → HTMLImageElement（引导图缩略图缓存）
    // Playhead 播放头状态
    this.playheadFrame = 0;           // 当前播放头帧位置
    this.dragPlayhead = false;        // 是否正在拖拽播放头
    this.snapEnabled = true;          // 吸附到段边缘
    // 视频缩略图序列缓存：fileKey → [{time, img}, ...]
    this._videoThumbCache = new Map();
    this._videoThumbPromises = new Map(); // 防止并发重复提取
    // 音频波形缓冲区缓存：fileKey → AudioBuffer（decodeAudioData 解码结果）
    this._audioBufferCache = new Map();
    this._audioBufferPromises = new Map();
    // IC-LoRA 静态图像缓存：fileKey → HTMLImageElement
    this._motionImageCache = new Map();
    this._motionImagePromises = new Map();
    // 剪贴板（复制/粘贴段用）
    this._clipboard = null;
    // 当前激活轨道：segment/motion/audio，决定键盘快捷键和复制/删除作用对象
    this._activeTrack = "segment";
    // 框选状态
    this._rubberBand = null; // { startX, startY, currentX, currentY, track }
    this._pointerDown = false; // 鼠标左键是否按下（框选只有按着左键拖拽才启动）

    this.buildDOM();
    this.bindEvents();
    this.syncWidgetsFromTimeline();
    this._preloadAllImages();
    this._preloadAllVideoThumbs();
    this._preloadAllAudioBuffers();
    this.updateUIFromSelection();
    this.render();

    // 异步加载已有音频段的 peaks
    this._loadAudioPeaks();
  }

  // ── 多选辅助方法 ──

  // 获取某轨道的第一个选中索引，无可选时返回 -1
  _firstSelected(track) {
    const s = this._getSelectionSet(track);
    return s.size > 0 ? Math.min(...s) : -1;
  }

  // 获取某轨道的选中 Set
  _getSelectionSet(track) {
    if (track === "motion") return this.selectedMotionIndices;
    if (track === "audio") return this.selectedAudioIndices;
    return this.selectedIndices;
  }

  // 判断某轨道的指定索引是否被选中
  _isSelected(track, index) {
    return this._getSelectionSet(track).has(index);
  }

  // 点击某个段时的选中逻辑：按住 Ctrl 时 toggle，否则替换选中
  _selectIndex(track, index, ctrlKey) {
    const set = this._getSelectionSet(track);
    if (ctrlKey) {
      if (set.has(index)) set.delete(index); else set.add(index);
    } else {
      set.clear();
      set.add(index);
    }
  }

  // ── 基础属性访问 ──

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

  lengthInputValueFor(frames) {
    if (!this.isSecondsMode()) return String(frames);
    return (frames / this.getFps()).toFixed(3).replace(/\.?0+$/, "");
  }

  // ── DOM 构建 ──

  buildDOM() {
    this.container.innerHTML = "";
    this.container.className = "yuan-clip-tl-container";

    // 工具栏（单行，分组：段编辑 | 设置 | 媒体上传 | 总计）
    this.toolbar = document.createElement("div");
    this.toolbar.className = "yuan-clip-tl-toolbar";

    this.addBtn = this.makeToolbarButton("＋ 段", "添加新段落");
    this.deleteBtn = this.makeToolbarButton("－ 段", "删除选中段落", "danger");
    this.splitBtn = this.makeToolbarButton("拆分", "拆分选中段落");
    this.clearBtn = this.makeToolbarButton("清空", "重置为默认时间轴");
    this.sep1 = this.makeSeparator();
    this.timeUnitsBtn = this.makeToolbarButton("⏱ 帧数", "切换时间单位（帧/秒）");
    this.promptLockBtn = this.makeToolbarButton("🔒 锁定", "切换 prompt_lock（锁定/解锁提示词编辑）");
    this.motionImageFramesBtn = this.makeToolbarButton("16帧", "每张运动图像的帧数（8/16/24/32）");
    this.sep2 = this.makeSeparator();
    this.addMotionBtn = this.makeToolbarButton("🎬 运动视频", "上传运动引导视频", "success");
    this.addAudioBtn = this.makeToolbarButton("🎵 音频", "上传音频文件", "success");
    this.spacer = document.createElement("div");
    this.spacer.className = "spacer";

    for (const b of [this.addBtn, this.deleteBtn, this.splitBtn, this.clearBtn, this.sep1,
      this.timeUnitsBtn, this.promptLockBtn, this.motionImageFramesBtn, this.sep2,
      this.addMotionBtn, this.addAudioBtn,
      this.spacer]) {
      this.toolbar.appendChild(b);
    }

    this.totalLabel = document.createElement("span");
    this.totalLabel.className = "yuan-clip-tl-total-label";
    this.toolbar.appendChild(this.totalLabel);

    this.container.appendChild(this.toolbar);

    // Canvas（包裹在 wrap 中以支持横向滚动）
    this.canvasWrap = document.createElement("div");
    this.canvasWrap.className = "yuan-clip-tl-canvas-wrap";
    this.canvas = document.createElement("canvas");
    this.canvas.className = "yuan-clip-tl-canvas";
    this.canvas.style.height = CANVAS_HEIGHT + "px";
    this.canvasWrap.appendChild(this.canvas);
    this.container.appendChild(this.canvasWrap);
    this.ctx = this.canvas.getContext("2d");

    // 属性面板
    this.propertiesPanel = document.createElement("div");
    this.propertiesPanel.className = "yuan-clip-tl-properties";
    this.container.appendChild(this.propertiesPanel);

    // 上传进度提示
    this.uploadProgressEl = document.createElement("div");
    this.uploadProgressEl.className = "yuan-clip-tl-upload-progress";
    this.uploadProgressEl.style.display = "none";
    this.container.appendChild(this.uploadProgressEl);

    this._updateToolbarState();
  }

  makeToolbarButton(label, tooltip, variant) {
    const b = document.createElement("button");
    b.className = "btn" + (variant ? " " + variant : "");
    b.textContent = label;
    if (tooltip) b.title = tooltip;
    return b;
  }

  makeSeparator() {
    const s = document.createElement("div");
    s.className = "separator";
    return s;
  }

  _updateToolbarState() {
    this.timeUnitsBtn.textContent = this.isSecondsMode() ? "⏱ 秒数" : "⏱ 帧数";
    const locked = this.promptLocked;
    this.promptLockBtn.textContent = locked ? "🔒 锁定" : "🔓 解锁";
    this.promptLockBtn.classList.toggle("active", locked);
    // 运动图像帧数按钮
    this.motionImageFramesBtn.textContent = this.motionImageFrames + "帧";
    this.addBtn.disabled = locked;
    this.deleteBtn.disabled = locked || this.timeline.segments.length <= 1;
    this.splitBtn.disabled = locked;
    this.clearBtn.disabled = locked;
  }

  // ── 事件绑定 ──

  bindEvents() {
    this.canvas.addEventListener("pointerdown", e => { e.stopPropagation(); this.handleCanvasClick(e); });
    this.canvas.addEventListener("pointermove", e => { e.stopPropagation(); this.handleCanvasDrag(e); });
    this.canvas.addEventListener("pointerup", e => { e.stopPropagation(); this.handleCanvasRelease(e); });
    this.canvas.addEventListener("contextmenu", e => {
      e.preventDefault(); e.stopPropagation();
      this._handleContextMenu(e);
    });
    this.canvas.addEventListener("wheel", e => e.stopPropagation(), { passive: true });
    this.canvas.addEventListener("pointerleave", () => {
      if (this.dragHandle < 0 && !this.reorder && !this.dragMotion && !this.dragAudio) {
        this.hoverIndex = -1;
        this.hoverHandle = -1;
        this.canvas.style.cursor = "default";
        this.render();
      }
    });

    // 鼠标进出容器追踪：键盘快捷键仅在鼠标悬停于编辑器上时生效
    this._mouseOver = false;
    this.container.addEventListener("pointerenter", () => { this._mouseOver = true; });
    this.container.addEventListener("pointerleave", () => { this._mouseOver = false; });

    // 键盘快捷键：Ctrl+C 复制、Ctrl+V 粘贴、Delete/Backspace 删除
    document.addEventListener("keydown", (e) => {
      if (!this._mouseOver) return;
      // 焦点在输入框/文本域/选择框时不拦截，避免影响属性面板编辑
      const ae = document.activeElement;
      const tag = ae?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || ae?.isContentEditable) return;

      const ctrl = e.ctrlKey || e.metaKey;
      if (ctrl && (e.key === "c" || e.key === "C")) {
        e.preventDefault(); e.stopPropagation();
        this._copySegment();
      } else if (ctrl && (e.key === "v" || e.key === "V")) {
        e.preventDefault(); e.stopPropagation();
        if (!this.promptLocked) this._pasteSegment();
      } else if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault(); e.stopPropagation();
        if (!this.promptLocked) this._deleteSelectedAny();
      }
    });

    this.addBtn.addEventListener("click", () => { if (!this.promptLocked) this.addSegment(); });
    this.deleteBtn.addEventListener("click", () => { if (!this.promptLocked) this.deleteSelected(); });
    this.splitBtn.addEventListener("click", () => { if (!this.promptLocked) this.splitSelected(); });
    this.clearBtn.addEventListener("click", () => { if (!this.promptLocked) this.clearAll(); });
    this.timeUnitsBtn.addEventListener("click", () => this.toggleTimeUnits());
    this.promptLockBtn.addEventListener("click", () => this.togglePromptLock());
    this.motionImageFramesBtn.addEventListener("click", () => this.cycleMotionImageFrames());
    this.addMotionBtn.addEventListener("click", () => { if (!this.promptLocked) this.uploadMotionVideo(); });
    this.addAudioBtn.addEventListener("click", () => { if (!this.promptLocked) this.uploadAudioFile(); });

    // max_frames 变化时按比例调整段长度
    if (this.maxFramesWidget) {
      const prev = this.maxFramesWidget.callback;
      this.maxFramesWidget.callback = (...args) => {
        prev?.apply(this.maxFramesWidget, args);
        this.handleMaxFramesChange();
      };
    }
    // fps / time_units 变化时重绘
    for (const w of [this.fpsWidget, this.timeUnitsWidget]) {
      if (!w) continue;
      const prev = w.callback;
      w.callback = (...args) => {
        prev?.apply(w, args);
        this._updateToolbarState();
        this.updateUIFromSelection();
        this.render();
      };
    }

    // text_input 变化时同步（仅锁定模式）
    if (this.textInputWidget) {
      const prevTI = this.textInputWidget.callback;
      this.textInputWidget.callback = (...args) => {
        prevTI?.apply(this.textInputWidget, args);
        if (this.promptLocked) this.syncFromTextInput();
      };
    }

    // prompt_lock 切换：仅同步 UI 状态（实际逻辑由 togglePromptLock 处理，避免重复执行）
    if (this.promptLockWidget) {
      const prevPL = this.promptLockWidget.callback;
      this.promptLockWidget.callback = (...args) => {
        prevPL?.apply(this.promptLockWidget, args);
        this._updateLockState();
        this._updateToolbarState();
        this.updateUIFromSelection();
        this.render();
      };
    }

    // global_prompt 变化时同步到 timeline，同时尝试读取上游连接文本
    if (this.globalPromptWidget) {
      const prevGP = this.globalPromptWidget.callback;
      this.globalPromptWidget.callback = (...args) => {
        prevGP?.apply(this.globalPromptWidget, args);
        // 优先使用 widget 值，若为空则尝试从上游连接节点读取
        const val = this.globalPromptWidget.value || "";
        if (!val.trim()) {
          const upstream = this._readConnectedGlobalPrompt();
          if (upstream) {
            this.timeline.global_prompt = upstream;
          } else {
            this.timeline.global_prompt = val;
          }
        } else {
          this.timeline.global_prompt = val;
        }
        this.commitChanges();
      };
    }

    this.resizeObserver = new ResizeObserver(() => this.resizeCanvas());
    this.resizeObserver.observe(this.container);
    this.resizeCanvas();
  }

  resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    // 取 wrap 的宽度（无 wrap 则取 canvas 父级），保证最小宽度避免挤压
    const host = this.canvasWrap?.parentElement || this.canvas.parentElement;
    const avail = Math.max(CANVAS_MIN_WIDTH, Math.floor(host?.clientWidth || this.canvas.offsetWidth));
    const w = Math.max(CANVAS_MIN_WIDTH, Math.floor(this.canvas.offsetWidth || avail));
    this.canvas.width = w * dpr;
    this.canvas.height = CANVAS_HEIGHT * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._cssWidth = w;
    this.render();
  }

  // ── 引导图缩略图缓存 ──

  _segImageKey(idx) {
    const seg = this.timeline.segments[idx];
    if (!seg) return null;
    if (seg.imageB64) return "b64:" + seg.imageB64.slice(0, 32);
    if (seg.imageFile) return "file:" + seg.imageFile;
    return null;
  }

  _buildImageUrl(file) {
    if (!file) return null;
    // 支持类型前缀：output:subfolder/filename 或 input:subfolder/filename（默认 input）
    let type = 'input';
    let path = file.replace(/\\/g, '/');
    const colonIdx = path.indexOf(':');
    if (colonIdx > 0 && colonIdx < 10) {
      const prefix = path.slice(0, colonIdx);
      if (prefix === 'output' || prefix === 'input' || prefix === 'temp') {
        type = prefix;
        path = path.slice(colonIdx + 1);
      }
    }
    const parts = path.split('/');
    const filename = parts.pop();
    const subfolder = parts.join('/');
    let url = `/view?filename=${encodeURIComponent(filename)}&type=${type}`;
    if (subfolder) url += `&subfolder=${encodeURIComponent(subfolder)}`;
    return url;
  }

  _loadImageForSeg(idx) {
    const seg = this.timeline.segments[idx];
    if (!seg || (!seg.imageFile && !seg.imageB64)) {
      this._imageCache.delete(idx);
      return;
    }
    const key = this._segImageKey(idx);
    const cached = this._imageCache.get(idx);
    if (cached && cached._key === key) return; // 已缓存且 key 一致
    const src = seg.imageB64 || this._buildImageUrl(seg.imageFile);
    if (!src) { this._imageCache.delete(idx); return; }
    const img = new Image();
    img._key = key;
    img.onload = () => this.render();
    img.onerror = () => {
      this._imageCache.delete(idx);
      this.render();
    };
    img.src = src;
    this._imageCache.set(idx, img);
  }

  // 加载 IC-LoRA 轨道的静态图像（isStaticImage=true 的运动段）
  _loadMotionImage(seg) {
    if (!seg || !seg.isStaticImage || !seg.videoFile) return;
    const fileKey = seg.videoFile;
    if (this._motionImageCache.has(fileKey)) return;
    if (this._motionImagePromises.has(fileKey)) return;
    const src = this._buildImageUrl(fileKey);
    if (!src) return;
    this._motionImagePromises.set(fileKey, true);
    const img = new Image();
    img.onload = () => {
      this._motionImageCache.set(fileKey, img);
      this._motionImagePromises.delete(fileKey);
      this.render();
    };
    img.onerror = () => {
      this._motionImagePromises.delete(fileKey);
    };
    img.src = src;
  }

  _preloadAllImages() {
    // 清理失效缓存
    const validKeys = new Set();
    for (let i = 0; i < this.timeline.segments.length; i++) {
      const k = this._segImageKey(i);
      if (k) validKeys.add(i);
    }
    for (const k of [...this._imageCache.keys()]) {
      if (!validKeys.has(k)) this._imageCache.delete(k);
    }
    for (const i of validKeys) this._loadImageForSeg(i);
  }

  // ── 段落图像 端口支持：实时读取上游节点图像（类似 文本输入 实时映射） ──

  hasSegmentImagesConnected() {
    const input = this.node.inputs?.find(i => i.name === "段落图像");
    return !!(input && input.link != null);
  }

  // 获取 segment_images 连接的上游节点
  _getUpstreamNode() {
    const node = this.node;
    if (!node || !node.graph) return null;
    const inputIdx = node.findInputSlot ? node.findInputSlot("段落图像") : -1;
    if (inputIdx < 0) return null;
    const linkId = node.inputs[inputIdx]?.link;
    if (linkId == null) return null;
    const link = node.graph.links[linkId];
    if (!link) return null;
    return node.graph.getNodeById(link.origin_id);
  }

  // 从上游节点实时获取图像列表（nodeOutputs 优先，回退到 srcNode.imgs / widget）
  _getUpstreamSegmentImages() {
    const srcNode = this._getUpstreamNode();
    if (!srcNode) return null;

    const comfyApi = app.api || window.comfyAPI?.api?.api || window.api || window.comfyAPI?.api;

    // 方式1：从 ComfyUI nodeOutputs 获取（上游节点执行后的输出缓存）
    // 支持多张图像输出（batch 预览，每张图像作为独立条目）
    const nodeOutputs = comfyApi?.nodeOutputs || {};
    const outputs = nodeOutputs[String(srcNode.id)] || nodeOutputs[srcNode.id];
    if (outputs?.images?.length > 0) {
      return outputs.images.map(img => ({
        filename: img.filename,
        subfolder: img.subfolder || "",
        type: img.type || "output",
      }));
    }

    // 方式2：从 srcNode.imgs 获取（节点已加载的渲染缩略图，无需执行）
    if (srcNode.imgs?.length) {
      return srcNode.imgs.map((img, i) => {
        let src = null;
        if (typeof img === 'string') src = img;
        else if (img instanceof HTMLImageElement) src = img.currentSrc || img.src;
        else if (img?.src) src = img.src;
        if (src) {
          // 从 URL 提取 filename（格式: /view?filename=xxx&type=...）
          const url = new URL(src, window.location.origin);
          const fname = url.searchParams.get('filename') || '';
          const ftype = url.searchParams.get('type') || 'input';
          const fsub = url.searchParams.get('subfolder') || '';
          if (fname) return { filename: fname, subfolder: fsub, type: ftype };
        }
        return null;
      }).filter(Boolean);
    }

    // 方式3：从上游节点 widget 读取（无需执行，实时映射）
    // 候选 widget 名：image, images, image_paths, file, files, dirpath
    const widgetNames = ["image", "images", "image_paths", "file", "files", "dirpath"];
    const imgWidgets = (srcNode.widgets || []).filter(w =>
      widgetNames.includes(w.name) || /^(image|img|file|path|folder|directory)/i.test(w.name)
    );
    for (const w of imgWidgets) {
      const val = w.value;
      if (!val) continue;
      // 字符串：单行或多行路径
      if (typeof val === "string") {
        const trimmed = val.trim();
        if (!trimmed) continue;
        if (trimmed.includes("\n")) {
          const lines = trimmed.split("\n").map(l => l.trim()).filter(l => l.length > 0);
          if (lines.length > 0) {
            return lines.map(line => {
              const norm = line.replace(/\\/g, '/');
              const parts = norm.split('/');
              const filename = parts.pop();
              const subfolder = parts.join('/');
              return { filename, subfolder, type: "input" };
            });
          }
        }
        const norm = trimmed.replace(/\\/g, '/');
        const parts = norm.split('/');
        const filename = parts.pop();
        const subfolder = parts.join('/');
        return [{ filename, subfolder, type: "input" }];
      }
      // 对象：{ filename, subfolder, type } 格式
      if (typeof val === "object" && val.filename) {
        return [{
          filename: val.filename,
          subfolder: val.subfolder || "",
          type: val.type || "input",
        }];
      }
      // 数组：多个文件对象
      if (Array.isArray(val) && val.length > 0) {
        return val.map(v => {
          if (typeof v === "string") {
            const norm = v.replace(/\\/g, '/');
            const parts = norm.split('/');
            const fn = parts.pop();
            return { filename: fn, subfolder: parts.join('/'), type: "input" };
          }
          if (typeof v === "object" && v.filename) {
            return { filename: v.filename, subfolder: v.subfolder || "", type: v.type || "input" };
          }
          return null;
        }).filter(Boolean);
      }
    }

    return null;
  }

  // 将上游图像实时分配到时间段段落（第1张→第1段、第2张→第2段……）
  _applyUpstreamSegmentImages() {
    // 锁定状态下拒绝上游数据，完全以时间轴编辑器内的数据为准
    if (this.promptLocked) return false;
    if (!this.hasSegmentImagesConnected()) return false;
    const images = this._getUpstreamSegmentImages();
    if (!images || images.length === 0) return false;

    const segs = this.timeline.segments;
    let changed = false;
    for (let i = 0; i < segs.length && i < images.length; i++) {
      const img = images[i];
      // 构建带类型前缀的 imageFile 引用
      const ref = (img.type && img.type !== "input")
        ? `${img.type}:${img.subfolder ? img.subfolder + "/" : ""}${img.filename}`
        : (img.subfolder ? `${img.subfolder}/${img.filename}` : img.filename);
      const seg = segs[i];
      if (seg.imageFile !== ref || seg.imageB64 !== "") {
        seg.imageFile = ref;
        seg.imageB64 = "";
        seg.type = "image";
        changed = true;
      }
    }
    if (changed) {
      this._imageCache.clear();
      this._preloadAllImages();
      this.commitChanges();
      this.updateUIFromSelection();
      this.render();
    }
    return changed;
  }

  // 上游节点 ID（用于 executed 事件匹配）
  _getUpstreamNodeId() {
    const srcNode = this._getUpstreamNode();
    return srcNode ? String(srcNode.id) : null;
  }

  // ── 运动图像 端口支持：实时读取上游节点图像分配到 IC-LoRA 轨道 ──

  hasMotionImagesConnected() {
    const input = this.node.inputs?.find(i => i.name === "运动图像");
    return !!(input && input.link != null);
  }

  _getUpstreamMotionNode() {
    const node = this.node;
    if (!node || !node.graph) return null;
    const inputIdx = node.findInputSlot ? node.findInputSlot("运动图像") : -1;
    if (inputIdx < 0) return null;
    const linkId = node.inputs[inputIdx]?.link;
    if (linkId == null) return null;
    const link = node.graph.links[linkId];
    if (!link) return null;
    return node.graph.getNodeById(link.origin_id);
  }

  _getUpstreamMotionImages() {
    const srcNode = this._getUpstreamMotionNode();
    if (!srcNode) return null;

    const comfyApi = app.api || window.comfyAPI?.api?.api || window.api || window.comfyAPI?.api;

    // 方式1：从 ComfyUI nodeOutputs 获取
    const nodeOutputs = comfyApi?.nodeOutputs || {};
    const outputs = nodeOutputs[String(srcNode.id)] || nodeOutputs[srcNode.id];
    if (outputs?.images?.length > 0) {
      return outputs.images.map(img => ({
        filename: img.filename,
        subfolder: img.subfolder || "",
        type: img.type || "output",
      }));
    }

    // 方式2：从 srcNode.imgs 获取（节点已加载的渲染缩略图，无需执行）
    if (srcNode.imgs?.length) {
      return srcNode.imgs.map((img, i) => {
        let src = null;
        if (typeof img === 'string') src = img;
        else if (img instanceof HTMLImageElement) src = img.currentSrc || img.src;
        else if (img?.src) src = img.src;
        if (src) {
          const url = new URL(src, window.location.origin);
          const fname = url.searchParams.get('filename') || '';
          const ftype = url.searchParams.get('type') || 'input';
          const fsub = url.searchParams.get('subfolder') || '';
          if (fname) return { filename: fname, subfolder: fsub, type: ftype };
        }
        return null;
      }).filter(Boolean);
    }

    // 方式3：从上游节点 widget 读取
    const widgetNames = ["image", "images", "image_paths", "file", "files", "dirpath"];
    const imgWidgets = (srcNode.widgets || []).filter(w =>
      widgetNames.includes(w.name) || /^(image|img|file|path|folder|directory)/i.test(w.name)
    );
    for (const w of imgWidgets) {
      const val = w.value;
      if (!val) continue;
      if (typeof val === "string") {
        const trimmed = val.trim();
        if (!trimmed) continue;
        if (trimmed.includes("\n")) {
          const lines = trimmed.split("\n").map(l => l.trim()).filter(l => l.length > 0);
          if (lines.length > 0) {
            return lines.map(line => {
              const norm = line.replace(/\\/g, '/');
              const parts = norm.split('/');
              const filename = parts.pop();
              const subfolder = parts.join('/');
              return { filename, subfolder, type: "input" };
            });
          }
        }
        const norm = trimmed.replace(/\\/g, '/');
        const parts = norm.split('/');
        const filename = parts.pop();
        const subfolder = parts.join('/');
        return [{ filename, subfolder, type: "input" }];
      }
      if (typeof val === "object" && val.filename) {
        return [{ filename: val.filename, subfolder: val.subfolder || "", type: val.type || "input" }];
      }
      if (Array.isArray(val) && val.length > 0) {
        return val.map(v => {
          if (typeof v === "string") {
            const norm = v.replace(/\\/g, '/');
            const parts = norm.split('/');
            const fn = parts.pop();
            return { filename: fn, subfolder: parts.join('/'), type: "input" };
          }
          if (typeof v === "object" && v.filename) {
            return { filename: v.filename, subfolder: v.subfolder || "", type: v.type || "input" };
          }
          return null;
        }).filter(Boolean);
      }
    }

    return null;
  }

  // 将上游图像分配到 IC-LoRA 轨道：多张图像合并为单个段，携带 frameFiles 走 IC-LoRA 视频路径
  _applyUpstreamMotionImages() {
    // 锁定状态下拒绝上游数据，完全以时间轴编辑器内的数据为准
    if (this.promptLocked) return false;
    if (!this.hasMotionImagesConnected()) return false;
    const images = this._getUpstreamMotionImages();
    if (!images || images.length === 0) return false;

    const max = this.getMaxFrames();
    const frameLen = this.motionImageFrames;
    const newMotionSegs = [];

    // 解析 global_prompt 中的 @图X=描述 行，按顺序对应到每个运动段
    const chars = this._parseCharacterLines();

    // 每张图像创建一个独立的运动段，连续排列
    // 前端显示为一段段，后端 Guide 节点 B 分支会自动合并相邻静态图像段为合成视频序列处理
    let currentStart = 0;
    for (let i = 0; i < images.length; i++) {
      const img = images[i];
      const ref = (img.type && img.type !== "input")
        ? `${img.type}:${img.subfolder ? img.subfolder + "/" : ""}${img.filename}`
        : (img.subfolder ? `${img.subfolder}/${img.filename}` : img.filename);

      const segLen = Math.min(frameLen, max - currentStart);
      if (segLen <= 0) break;

      // 描述字段：按顺序对应 @图X=描述；无对应角色定义时为空
      const charDesc = (i < chars.length) ? chars[i].desc : "";
      // 显式绑定 @图X 编号（第 i 张图对应 @图(i+1)），后端据此反查，不再依赖数组下标
      const seg = {
        videoFile: ref,
        frameFiles: [ref],
        start: currentStart,
        length: segLen,
        trimStart: 0.0,
        isStaticImage: true,
        fileName: img.filename,
        description: charDesc,
        subjectNum: i + 1,
      };
      newMotionSegs.push(seg);
      this._loadMotionImage(seg);
      currentStart += segLen;
    }

    // 替换整个 motionSegments 列表（链接端口时覆盖已有段）
    this.timeline.motionSegments = newMotionSegs;
    this.selectedMotionIndices.clear();
    this.commitChanges();
    this.updateUIFromSelection();
    this.render();
    return true;
  }

  // 从 global_prompt 中提取非 @图X=描述 行的剩余文本（在属性面板下方显示）
  _getRemainingGlobalText() {
    const gp = this.globalPromptWidget?.value || "";
    const lines = gp.split("\n");
    const pattern = /^@图(\d+)\s*[=：:]\s*(.+)/;
    const remaining = [];
    for (const line of lines) {
      if (!line.trim().match(pattern)) {
        remaining.push(line);
      }
    }
    return remaining.join("\n").trim();
  }

  // 将编辑后的剩余全局文本同步回 global_prompt（保留 @图X=描述 行，替换其余行）
  // 同时更新所有 IC-运动段的描述字段（合并剩余文本 + 对应 @图X 描述）
  _syncRemainingTextToGlobalPrompt(newRemainingText) {
    if (this.promptLocked) return;
    const gp = this.globalPromptWidget?.value || "";
    const lines = gp.split("\n");
    const pattern = /^@图(\d+)\s*[=：:]\s*(.+)/;
    // 保留 @图X=描述 行
    const charLines = [];
    for (const line of lines) {
      if (line.trim().match(pattern)) {
        charLines.push(line);
      }
    }
    // 组合：@图X=描述 行 + 剩余文本
    const newLines = [...charLines];
    if (newRemainingText && newRemainingText.trim()) {
      newLines.push(newRemainingText.trim());
    }
    const newGlobalPrompt = newLines.join("\n");
    // 写回 global_prompt widget
    if (this.globalPromptWidget) {
      this.globalPromptWidget.value = newGlobalPrompt;
    }
    // 同步到 timeline.global_prompt
    this.timeline.global_prompt = newGlobalPrompt;
  }

  _getMotionUpstreamNodeId() {
    const srcNode = this._getUpstreamMotionNode();
    return srcNode ? String(srcNode.id) : null;
  }

  // ── 音频输入 端口支持：从上游节点读取 AUDIO 输出并刷新时间轴音频轨道 ──

  hasAudioInputConnected() {
    const input = this.node.inputs?.find(i => i.name === "音频输入");
    return !!(input && input.link != null);
  }

  _getUpstreamAudioInputNode() {
    const node = this.node;
    if (!node || !node.graph) return null;
    const inputIdx = node.findInputSlot ? node.findInputSlot("音频输入") : -1;
    if (inputIdx < 0) return null;
    const linkId = node.inputs[inputIdx]?.link;
    if (linkId == null) return null;
    const link = node.graph.links[linkId];
    if (!link) return null;
    return node.graph.getNodeById(link.origin_id);
  }

  _getUpstreamAudioInputId() {
    const srcNode = this._getUpstreamAudioInputNode();
    return srcNode ? String(srcNode.id) : null;
  }

  // 从上游节点获取音频文件信息（两层回退策略）
  _getUpstreamAudioInput() {
    const srcNode = this._getUpstreamAudioInputNode();
    if (!srcNode) return null;

    const comfyApi = app.api || window.comfyAPI?.api?.api || window.api || window.comfyAPI?.api;

    // 方式1：从 ComfyUI nodeOutputs 获取（上游已执行）
    const nodeOutputs = comfyApi?.nodeOutputs || {};
    const outputs = nodeOutputs[String(srcNode.id)] || nodeOutputs[srcNode.id];
    if (outputs?.audio?.length > 0) {
      const a = outputs.audio[0];
      const ref = (a.type && a.type !== "input")
        ? `${a.type}:${a.subfolder ? a.subfolder + "/" : ""}${a.filename}`
        : (a.subfolder ? `${a.subfolder}/${a.filename}` : a.filename);
      return { filename: a.filename, subfolder: a.subfolder || "", type: a.type || "input", ref };
    }

    // 方式2：从上游节点 widget 读取（LoadAudio 等）
    const widgetNames = ["audio", "file", "filename", "path", "audio_file", "audio_path"];
    const audioWidgets = (srcNode.widgets || []).filter(w =>
      widgetNames.includes(w.name) || /^(audio|file|sound|wave)/i.test(w.name)
    );
    for (const w of audioWidgets) {
      const val = w.value;
      if (!val || typeof val !== "string") continue;
      const trimmed = val.trim();
      if (!trimmed) continue;
      const norm = trimmed.replace(/\\/g, '/');
      const parts = norm.split('/');
      const filename = parts.pop();
      const subfolder = parts.join('/');
      const ref = subfolder ? `${subfolder}/${filename}` : filename;
      return { filename, subfolder, type: "input", ref };
    }

    return null;
  }

  // 应用上游音频输入到时间轴音频轨道
  // detail 由后端 send_sync 提供（执行后推送）；无 detail 时前端主动从上游获取
  _applyUpstreamAudioInput(detail) {
    // 锁定状态下拒绝上游数据，完全以时间轴编辑器内的数据为准
    if (this.promptLocked) return false;
    if (!this.hasAudioInputConnected()) return false;

    // 场景1：后端 send_sync 推送的音频段（节点执行后）
    if (detail && detail.audioFile) {
      const existing = this.timeline.audioSegments;
      for (const seg of existing) {
        if (seg.audioFile === detail.audioFile) {
          seg.start = detail.start;
          seg.length = detail.length;
          seg.trimStart = detail.trimStart || 0.0;
          seg.fileName = detail.fileName;
          this.commitChanges();
          this._preloadAllAudioBuffers();
          this.updateUIFromSelection();
          this.render();
          return true;
        }
      }
      const newSeg = {
        audioFile: detail.audioFile,
        audioB64: "",
        start: detail.start,
        length: detail.length,
        trimStart: detail.trimStart || 0.0,
        fileName: detail.fileName,
      };
      existing.push(newSeg);
      this.selectedAudioIndices = new Set([existing.length - 1]);
      this.commitChanges();
      this._preloadAllAudioBuffers();
      this.updateUIFromSelection();
      this.render();
      return true;
    }

    // 场景2：前端主动从上游获取（连接时/上游执行完成时，参考运动图像同步创建段）
    const audioInfo = this._getUpstreamAudioInput();
    if (!audioInfo) return false;

    const max = this.getMaxFrames();
    const fps = this.getFps();

    // 检查是否已存在相同 ref 的段（避免重复添加）
    const existing = this.timeline.audioSegments;
    for (const seg of existing) {
      if (seg.audioFile === audioInfo.ref) {
        // 已存在，仅刷新预加载
        this._preloadAllAudioBuffers();
        this.render();
        return true;
      }
    }

    // 同步创建段（使用默认长度，后续异步更新真实长度）——参考运动图像的同步创建模式
    const defaultLen = Math.max(MIN_SEGMENT_LENGTH, Math.floor(max / 4));
    const slot = this._findFreeSlot(existing, defaultLen, max);
    const { start, length: segLen } = slot;
    if (segLen <= 0) return false;

    const newSeg = {
      audioFile: audioInfo.ref,
      audioB64: "",
      start,
      length: segLen,
      trimStart: 0.0,
      fileName: audioInfo.filename,
    };
    existing.push(newSeg);
    this.selectedAudioIndices = new Set([existing.length - 1]);
    this.commitChanges();
    this._preloadAllAudioBuffers();
    this.updateUIFromSelection();
    this.render();

    // 异步更新真实长度（解码音频获取时长后更新段长度）
    this._decodeAudioBuffer(audioInfo.ref).then(audioBuf => {
      if (audioBuf) {
        const dur = audioBuf.duration || 1;
        const realLen = Math.max(MIN_SEGMENT_LENGTH, Math.ceil(dur * fps));
        newSeg.length = Math.min(realLen, max - newSeg.start);
        this.commitChanges();
        this.render();
      }
    });

    return true;
  }

  // ── 视频缩略图序列 ──

  _videoFileKey(seg) {
    return seg.imageFile || seg.videoFile || "";
  }

  // 为视频段提取多帧缩略图，返回 Promise<thumbnails[]>
  async _extractVideoThumbnails(seg) {
    const fileKey = this._videoFileKey(seg);
    if (!fileKey) return [];
    // 缓存命中
    if (this._videoThumbCache.has(fileKey)) {
      return this._videoThumbCache.get(fileKey);
    }
    // 防止并发重复提取
    if (this._videoThumbPromises.has(fileKey)) {
      return this._videoThumbPromises.get(fileKey);
    }
    const promise = (async () => {
      const url = this._buildImageUrl(fileKey);
      if (!url) return [];
      const video = document.createElement("video");
      video.muted = true;
      video.preload = "auto";
      video.playsInline = true;
      video.src = url;
      // 等待元数据加载
      const metaOk = await new Promise((resolve) => {
        const onLoaded = () => { cleanup(); resolve(true); };
        const onError = () => { cleanup(); resolve(false); };
        const cleanup = () => {
          video.removeEventListener("loadedmetadata", onLoaded);
          video.removeEventListener("error", onError);
        };
        video.addEventListener("loadedmetadata", onLoaded);
        video.addEventListener("error", onError);
        setTimeout(() => { cleanup(); resolve(false); }, 15000);
      });
      if (!metaOk) return [];
      const duration = video.duration;
      if (!duration || !isFinite(duration)) {
        return [];
      }
      // 等待第一帧可绘制
      await new Promise((resolve) => {
        if (video.readyState >= 2) return resolve();
        const onReady = () => { cleanup(); resolve(); };
        const cleanup = () => video.removeEventListener("loadeddata", onReady);
        video.addEventListener("loadeddata", onReady);
        setTimeout(() => { cleanup(); resolve(); }, 5000);
      });
      // 帧数：根据时长动态决定，限制 6~12 帧
      const numFrames = Math.max(6, Math.min(12, Math.ceil(duration * 2)));
      const thumbnails = [];
      const offCanvas = document.createElement("canvas");
      const targetH = 80; // 缩略图高度
      for (let i = 0; i < numFrames; i++) {
        const t = (i / numFrames) * duration;
        try {
          await new Promise((resolve) => {
            const onSeeked = () => { video.removeEventListener("seeked", onSeeked); resolve(); };
            video.addEventListener("seeked", onSeeked);
            video.currentTime = t;
            setTimeout(() => { video.removeEventListener("seeked", onSeeked); resolve(); }, 2000);
          });
          const vw = video.videoWidth, vh = video.videoHeight;
          if (!vw || !vh) continue;
          const tw = Math.round(targetH * vw / vh);
          offCanvas.width = tw;
          offCanvas.height = targetH;
          const octx = offCanvas.getContext("2d");
          octx.drawImage(video, 0, 0, tw, targetH);
          const img = new Image();
          img.src = offCanvas.toDataURL("image/jpeg", 0.7);
          await new Promise(r => { img.onload = r; img.onerror = r; });
          thumbnails.push({ time: t, img });
        } catch (_) { continue; }
      }
      this._videoThumbCache.set(fileKey, thumbnails);
      return thumbnails;
    })();
    this._videoThumbPromises.set(fileKey, promise);
    try {
      const result = await promise;
      return result;
    } finally {
      this._videoThumbPromises.delete(fileKey);
    }
  }

  // 预加载所有视频段的缩略图（主轨道 + IC-LoRA 运动轨道）
  async _preloadAllVideoThumbs() {
    for (let i = 0; i < this.timeline.segments.length; i++) {
      const seg = this.timeline.segments[i];
      if (seg.type === "video" && this._videoFileKey(seg)) {
        this._extractVideoThumbnails(seg).then(() => this.render());
      }
    }
    // IC-LoRA 运动段也预加载缩略图
    for (const seg of this.timeline.motionSegments) {
      if (this._videoFileKey(seg)) {
        this._extractVideoThumbnails(seg).then(() => this.render());
      }
    }
  }

  // 绘制视频缩略图序列（水平铺满段宽度）
  _drawVideoThumbnails(ctx, seg, dx, dy, dw, dh) {
    const fileKey = this._videoFileKey(seg);
    const thumbs = this._videoThumbCache.get(fileKey);
    if (!thumbs || thumbs.length === 0) return false;
    ctx.save();
    ctx.beginPath();
    ctx.rect(dx, dy, dw, dh);
    ctx.clip();
    const n = thumbs.length;
    const thumbW = dw / n;
    for (let i = 0; i < n; i++) {
      const img = thumbs[i].img;
      if (img && img.complete && img.naturalWidth > 0) {
        // cover 模式填充每个缩略图格子
        this._drawImageCover(ctx, img, dx + i * thumbW, dy, thumbW + 1, dh);
      }
    }
    ctx.restore();
    return true;
  }

  // ── 音频波形：decodeAudioData 解码完整波形 ──

  _audioFileKey(seg) {
    return seg.audioFile || seg.fileName || "";
  }

  // 用 AudioContext.decodeAudioData 解码音频文件，返回 AudioBuffer
  async _decodeAudioBuffer(fileKey) {
    if (!fileKey) return null;
    if (this._audioBufferCache.has(fileKey)) {
      return this._audioBufferCache.get(fileKey);
    }
    if (this._audioBufferPromises.has(fileKey)) {
      return this._audioBufferPromises.get(fileKey);
    }
    const promise = (async () => {
      const url = this._buildImageUrl(fileKey);
      if (!url) return null;
      try {
        const resp = await fetch(url);
        const arrayBuffer = await resp.arrayBuffer();
        const AC = window.AudioContext || window.webkitAudioContext;
        if (!AC) return null;
        const audioCtx = new AC();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        audioCtx.close?.();
        this._audioBufferCache.set(fileKey, audioBuffer);
        return audioBuffer;
      } catch (e) {
        return null;
      }
    })();
    this._audioBufferPromises.set(fileKey, promise);
    try {
      return await promise;
    } finally {
      this._audioBufferPromises.delete(fileKey);
    }
  }

  // 预加载所有音频段的波形缓冲区
  async _preloadAllAudioBuffers() {
    for (const seg of this.timeline.audioSegments) {
      const key = this._audioFileKey(seg);
      if (key) {
        this._decodeAudioBuffer(key).then(() => this.render());
      }
    }
  }

  // 绘制真实波形（从 AudioBuffer 采样）
  _drawAudioWaveform(ctx, seg, dx, dy, dw, dh) {
    const fileKey = this._audioFileKey(seg);
    const buf = this._audioBufferCache.get(fileKey);
    if (!buf) return false;
    const channelData = buf.getChannelData(0);
    const samples = channelData.length;
    if (samples === 0) return false;
    const midY = dy + dh / 2;
    const maxH = dh - 8;
    // 按像素采样：每个像素对应一个峰值
    const numBars = Math.max(1, Math.floor(dw));
    ctx.fillStyle = "#6fa0e8";
    for (let i = 0; i < numBars; i++) {
      const startIdx = Math.floor(i * samples / numBars);
      const endIdx = Math.floor((i + 1) * samples / numBars);
      let peak = 0;
      for (let j = startIdx; j < endIdx && j < samples; j++) {
        const v = Math.abs(channelData[j]);
        if (v > peak) peak = v;
      }
      const barH = Math.max(1, peak * maxH);
      const barX = dx + i;
      ctx.fillRect(barX, midY - barH / 2, 1, barH);
    }
    return true;
  }

  _drawImageCover(ctx, img, dx, dy, dw, dh) {
    // cover 模式：保持比例填充目标区域，超出部分裁剪
    const iw = img.naturalWidth, ih = img.naturalHeight;
    if (!iw || !ih) return;
    const sRatio = dw / dh;
    const iRatio = iw / ih;
    let sx, sy, sw, sh;
    if (iRatio > sRatio) {
      // 图像更宽，按高度裁剪两侧
      sh = ih;
      sw = ih * sRatio;
      sx = (iw - sw) / 2;
      sy = 0;
    } else {
      sw = iw;
      sh = iw / sRatio;
      sx = 0;
      sy = (ih - sh) / 2;
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(dx, dy, dw, dh);
    ctx.clip();
    ctx.drawImage(img, sx, sy, sw, sh, dx, dy, dw, dh);
    ctx.restore();
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

  // ── 布局计算 ──

  pxPerFrame() {
    return this._cssWidth / this.getMaxFrames();
  }

  getTrackAtY(y) {
    if (y < RULER_HEIGHT) return "ruler";
    const segEnd = RULER_HEIGHT + BLOCK_HEIGHT;
    if (y < segEnd) return "segment";
    const motionEnd = segEnd + MOTION_TRACK_HEIGHT;
    if (y < motionEnd) return "motion";
    return "audio";
  }

  segmentRects() {
    const segs = this.timeline.segments;
    const ppf = this.pxPerFrame();
    const rects = [];
    let cursor = 0;
    for (let i = 0; i < segs.length; i++) {
      const len = segs[i].length;
      rects.push({
        index: i, x: cursor * ppf, w: len * ppf,
        frameStart: cursor, frameEnd: cursor + len,
      });
      cursor += len;
    }
    return rects;
  }

  hitBoundary(mx) {
    const rects = this.segmentRects();
    for (let i = 0; i < rects.length - 1; i++) {
      const right = rects[i].x + rects[i].w;
      if (Math.abs(mx - right) <= HANDLE_HIT_PX) return i;
    }
    return -1;
  }

  hitBlock(mx, my) {
    if (my < RULER_HEIGHT || my > RULER_HEIGHT + BLOCK_HEIGHT) return -1;
    const rects = this.segmentRects();
    for (const r of rects) {
      if (mx >= r.x && mx < r.x + r.w) return r.index;
    }
    return -1;
  }

  // ── 指针事件 ──

  handleCanvasClick(e) {
    this._pointerDown = true;
    const { x, y } = this.localPos(e);
    const track = this.getTrackAtY(y);

    // Playhead 拖拽：ruler 区域内命中播放头把手
    if (this.hitPlayhead(x, y)) {
      this.dragPlayhead = true;
      try { this.canvas.setPointerCapture(e.pointerId); } catch (_) {}
      this.canvas.style.cursor = "grabbing";
      return;
    }

    // 点击 ruler 空白处：移动播放头到该位置
    if (track === "ruler") {
      const ppf = this.pxPerFrame();
      this.setPlayhead(x / ppf);
      this.dragPlayhead = true;
      try { this.canvas.setPointerCapture(e.pointerId); } catch (_) {}
      this.canvas.style.cursor = "grabbing";
      return;
    }

    if (track === "segment") {
      this._handleSegmentDown(x, y, e);
    } else if (track === "motion") {
      this._handleMotionDown(x, y, e);
    } else {
      this._handleAudioDown(x, y, e);
    }
  }

  _handleSegmentDown(x, y, e) {
    // 边界拖拽（调整 length）
    const handle = this.hitBoundary(x);
    if (handle >= 0 && !this.promptLocked) {
      this.dragHandle = handle;
      this.dragStart = {
        x,
        initialLengths: this.timeline.segments.map(s => s.length),
      };
      try { this.canvas.setPointerCapture(e.pointerId); } catch (_) {}
      return;
    }
    // 段选择
    const block = this.hitBlock(x, y);
    if (block >= 0) {
      this._selectIndex("segment", block, e.ctrlKey || e.metaKey);
      this.selectedMotionIndices.clear();
      this.selectedAudioIndices.clear();
      this._activeTrack = "segment";
      this.updateUIFromSelection();
      this.render();
      if (!this.promptLocked) {
        // 记录初始段长度和拖拽起始帧偏移（光标在段内的位置）
        const ppf = this.pxPerFrame();
        const segStartFrame = this.timeline.segments[block].start;
        const cursorFrame = x / ppf;
        this.reorder = {
          sourceIdx: block,
          startX: x,
          startY: y,
          cursorX: x,
          active: false,
          initialLengths: this.timeline.segments.map(s => s.length),
          initialStart: segStartFrame,
          grabOffset: cursorFrame - segStartFrame, // 光标在段内的帧偏移
        };
        try { this.canvas.setPointerCapture(e.pointerId); } catch (_) {}
      }
    }
  }

  _handleMotionDown(x, y, e) {
    const ppf = this.pxPerFrame();
    const frame = x / ppf;
    for (let i = 0; i < this.timeline.motionSegments.length; i++) {
      const seg = this.timeline.motionSegments[i];
      if (frame >= seg.start && frame < seg.start + seg.length) {
        this._selectIndex("motion", i, e.ctrlKey || e.metaKey);
        this.selectedIndices.clear();
        this.selectedAudioIndices.clear();
        this._activeTrack = "motion";
        if (!this.promptLocked) {
          this.dragMotion = { index: i, startX: x, initialStart: seg.start, active: false };
          try { this.canvas.setPointerCapture(e.pointerId); } catch (_) {}
        }
        this.updateUIFromSelection();
        this.render();
        return;
      }
    }
    // 空白处点击：保持选中（与主轨行为一致）
  }

  _handleAudioDown(x, y, e) {
    const ppf = this.pxPerFrame();
    const frame = x / ppf;
    for (let i = 0; i < this.timeline.audioSegments.length; i++) {
      const seg = this.timeline.audioSegments[i];
      if (frame >= seg.start && frame < seg.start + seg.length) {
        this._selectIndex("audio", i, e.ctrlKey || e.metaKey);
        this.selectedIndices.clear();
        this.selectedMotionIndices.clear();
        this._activeTrack = "audio";
        if (!this.promptLocked) {
          this.dragAudio = { index: i, startX: x, initialStart: seg.start, active: false };
          try { this.canvas.setPointerCapture(e.pointerId); } catch (_) {}
        }
        this.updateUIFromSelection();
        this.render();
        return;
      }
    }
    // 空白处点击：保持选中（与主轨行为一致）
  }

  handleCanvasDrag(e) {
    const { x, y } = this.localPos(e);

    // Playhead 拖拽
    if (this.dragPlayhead) {
      const ppf = this.pxPerFrame();
      this.setPlayhead(x / ppf);
      return;
    }

    // 段边界拖拽
    if (this.dragHandle >= 0) {
      const ppf = this.pxPerFrame();
      const dxFrames = Math.round((x - this.dragStart.x) / ppf);
      const handle = this.dragHandle;
      const initial = this.dragStart.initialLengths;
      this._setLengthShifting(handle, initial[handle] + dxFrames, initial);
      this.commitChanges();
      this.updateUIFromSelection();
      this.render();
      return;
    }

    // 段重排拖拽（物理碰撞：实时推开相邻段）
    if (this.reorder) {
      const dx = x - this.reorder.startX;
      const dy = y - (this.reorder.startY != null ? this.reorder.startY : y);
      if (!this.reorder.active) {
        // 垂直移动优先 → 用户意图框选，取消 reorder
        if (Math.abs(dy) > REORDER_THRESHOLD_PX && Math.abs(dy) > Math.abs(dx)) {
          try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) {}
          this.canvas.style.cursor = "default";
          this.reorder = null;
        } else if (Math.abs(dx) > REORDER_THRESHOLD_PX) {
          this.reorder.active = true;
          this.canvas.style.cursor = "grabbing";
        }
      }
      if (this.reorder && this.reorder.active) {
        this.reorder.cursorX = x;
        this._applyCenterDragPhysics();
        this.render();
        return;
      }
      // reorder 仍 inactive 时，不 return，继续往下走到框选检测
    }

    // 运动段拖拽
    if (this.dragMotion) {
      const dx = x - this.dragMotion.startX;
      if (!this.dragMotion.active && Math.abs(dx) > REORDER_THRESHOLD_PX) {
        this.dragMotion.active = true;
        this.canvas.style.cursor = "grabbing";
      }
      if (this.dragMotion.active) {
        const ppf = this.pxPerFrame();
        const dxFrames = Math.round(dx / ppf);
        const seg = this.timeline.motionSegments[this.dragMotion.index];
        if (seg) {
          const rawStart = this.dragMotion.initialStart + dxFrames;
          seg.start = this._resolveNoOverlap(
            this.timeline.motionSegments, this.dragMotion.index,
            rawStart, seg.length, this.getMaxFrames(),
            this.dragMotion.initialStart
          );
          this.render();
        }
      }
      return;
    }

    // 音频段拖拽
    if (this.dragAudio) {
      const dx = x - this.dragAudio.startX;
      if (!this.dragAudio.active && Math.abs(dx) > REORDER_THRESHOLD_PX) {
        this.dragAudio.active = true;
        this.canvas.style.cursor = "grabbing";
      }
      if (this.dragAudio.active) {
        const ppf = this.pxPerFrame();
        const dxFrames = Math.round(dx / ppf);
        const seg = this.timeline.audioSegments[this.dragAudio.index];
        if (seg) {
          const rawStart = this.dragAudio.initialStart + dxFrames;
          seg.start = this._resolveNoOverlap(
            this.timeline.audioSegments, this.dragAudio.index,
            rawStart, seg.length, this.getMaxFrames(),
            this.dragAudio.initialStart
          );
          this.render();
        }
      }
      return;
    }

    // 框选（rubber band selection）
    if (this._rubberBand) {
      this._rubberBand.currentX = x;
      this._rubberBand.currentY = y;
      this.render();
      return;
    }

    // 检查是否可在轨道区域内启动框选（需按着鼠标左键 + 不在 ruler 区域）
    const trackDrag = this.getTrackAtY(y);
    if (this._pointerDown && trackDrag !== "ruler" && !this.dragPlayhead && this.dragHandle < 0 && !this.reorder && !this.dragMotion && !this.dragAudio) {
      this._rubberBand = { startX: x, startY: y, currentX: x, currentY: y, track: trackDrag };
      this.render();
      return;
    }

    // 悬停检测
    if (this.hitPlayhead(x, y)) {
      this.canvas.style.cursor = "grab";
      return;
    }
    const handle = this.hitBoundary(x);
    const block = handle >= 0 ? -1 : this.hitBlock(x, y);
    if (handle !== this.hoverHandle || block !== this.hoverIndex) {
      this.hoverHandle = handle;
      this.hoverIndex = block;
      this.canvas.style.cursor = handle >= 0
        ? "ew-resize"
        : (block >= 0 ? "pointer" : "default");
      this.render();
    }
  }

  handleCanvasRelease(e) {
    this._pointerDown = false;
    let captured = false;

    if (this.dragPlayhead) {
      try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      this.dragPlayhead = false;
      this.canvas.style.cursor = "default";
      captured = true;
    }

    if (this.dragHandle >= 0) {
      try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      this.dragHandle = -1;
      this.dragStart = null;
      captured = true;
    }

    if (this.reorder) {
      try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      if (this.reorder.active) {
        this.canvas.style.cursor = "default";
        this._commitReorder();
      }
      this.reorder = null;
      captured = true;
    }

    if (this.dragMotion) {
      try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      if (this.dragMotion.active) {
        this.canvas.style.cursor = "default";
        this.commitChanges();
      }
      this.dragMotion = null;
      captured = true;
    }

    if (this.dragAudio) {
      try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      if (this.dragAudio.active) {
        this.canvas.style.cursor = "default";
        this.commitChanges();
      }
      this.dragAudio = null;
      captured = true;
    }

    if (this._rubberBand) {
      this._finishRubberBand();
      this._rubberBand = null;
      this.render();
      captured = true;
    }

    if (captured) this.render();
  }

  // 物理碰撞拖拽：拖拽源段时实时向相邻段借用/归还帧，保持总长不变
  _applyCenterDragPhysics() {
    const segs = this.timeline.segments;
    const n = segs.length;
    if (n === 0) return;
    const sourceIdx = this.reorder.sourceIdx;
    const ppf = this.pxPerFrame();
    const cursorFrame = this.reorder.cursorX / ppf;
    const grabOffset = this.reorder.grabOffset;
    const initialLengths = this.reorder.initialLengths;
    // 源段期望的新起始帧（保持抓取偏移），且不超出 [0, totalLen - sourceLen]；总长不变
    let desiredStart = Math.max(0, cursorFrame - grabOffset);
    const sourceLen = initialLengths[sourceIdx];
    const totalLen = initialLengths.reduce((a, b) => a + b, 0);
    desiredStart = Math.max(0, Math.min(totalLen - sourceLen, desiredStart));
    // 简化策略：保持段顺序，源段起始帧变为 desiredStart，前后段按初始比例伸缩
    const frontTotal = desiredStart;
    const backTotal = totalLen - desiredStart - sourceLen;
    let frontInitialTotal = 0;
    for (let i = 0; i < sourceIdx; i++) frontInitialTotal += initialLengths[i];
    let backInitialTotal = 0;
    for (let i = sourceIdx + 1; i < n; i++) backInitialTotal += initialLengths[i];
    // 分配前段长度（按比例缩放，保留最小长度）
    let frontAllocated = 0;
    for (let i = 0; i < sourceIdx; i++) {
      const ratio = frontInitialTotal > 0 ? initialLengths[i] / frontInitialTotal : 1 / sourceIdx;
      let len = Math.max(MIN_SEGMENT_LENGTH, Math.round(frontTotal * ratio));
      segs[i].length = len;
      frontAllocated += len;
    }
    // 分配后段长度
    let backAllocated = 0;
    const backCount = n - sourceIdx - 1;
    for (let i = sourceIdx + 1; i < n; i++) {
      const ratio = backInitialTotal > 0 ? initialLengths[i] / backInitialTotal : 1 / backCount;
      let len = Math.max(MIN_SEGMENT_LENGTH, Math.round(backTotal * ratio));
      segs[i].length = len;
      backAllocated += len;
    }
    // 修正源段长度以吸收舍入误差，保持总长精确
    segs[sourceIdx].length = totalLen - frontAllocated - backAllocated;
    segs[sourceIdx].length = Math.max(MIN_SEGMENT_LENGTH, segs[sourceIdx].length);
    let cursor = 0;
    for (const seg of segs) { seg.start = cursor; cursor += seg.length; }
    // 源段中心跨越相邻段中心时交换顺序（实现重排）
    const sourceCenter = segs[sourceIdx].start + segs[sourceIdx].length / 2;
    if (sourceIdx > 0) {
      const prevCenter = segs[sourceIdx - 1].start + segs[sourceIdx - 1].length / 2;
      if (sourceCenter < prevCenter) {
        // 与前一段交换
        const tmp = segs[sourceIdx - 1];
        segs[sourceIdx - 1] = segs[sourceIdx];
        segs[sourceIdx] = tmp;
        this.reorder.sourceIdx = sourceIdx - 1;
        this.selectedIndices = new Set([sourceIdx - 1]);
        let c = 0;
        for (const seg of segs) { seg.start = c; c += seg.length; }
      }
    }
    if (this.reorder.sourceIdx < n - 1) {
      const nextIdx = this.reorder.sourceIdx + 1;
      const sourceC = segs[this.reorder.sourceIdx].start + segs[this.reorder.sourceIdx].length / 2;
      const nextCenter = segs[nextIdx].start + segs[nextIdx].length / 2;
      if (sourceC > nextCenter) {
        const tmp = segs[nextIdx];
        segs[nextIdx] = segs[this.reorder.sourceIdx];
        segs[this.reorder.sourceIdx] = tmp;
        this.reorder.sourceIdx = nextIdx;
        this.selectedIndices = new Set([nextIdx]);
        let c = 0;
        for (const seg of segs) { seg.start = c; c += seg.length; }
      }
    }
  }

  _commitReorder() {
    // 物理拖拽已在拖动过程中实时调整段长度和顺序，此处仅需提交
    this.commitChanges();
    this.updateUIFromSelection();
  }

  _handleContextMenu(e) {
    const { x, y } = this.localPos(e);
    const track = this.getTrackAtY(y);
    const ppf = this.pxPerFrame();
    const frame = x / ppf;

    if (track === "segment") {
      this._activeTrack = "segment";
      const block = this.hitBlock(x, y);
      // 空白处也弹出菜单（添加段/粘贴段）；命中段时选中该段
      if (block >= 0) {
        this.selectedIndices = new Set([block]);
        this.selectedMotionIndices.clear();
        this.selectedAudioIndices.clear();
      }
      this.updateUIFromSelection();
      this.render();
      this._showSegmentContextMenu(e.clientX, e.clientY, block);
    } else if (track === "motion") {
      this._activeTrack = "motion";
      // 查找点击的运动段
      let hitIdx = -1;
      for (let i = 0; i < this.timeline.motionSegments.length; i++) {
        const seg = this.timeline.motionSegments[i];
        if (frame >= seg.start && frame < seg.start + seg.length) { hitIdx = i; break; }
      }
      if (hitIdx >= 0) {
        this.selectedMotionIndices = new Set([hitIdx]);
        this.selectedIndices.clear();
        this.selectedAudioIndices.clear();
      }
      this.updateUIFromSelection();
      this.render();
      this._showMotionContextMenu(e.clientX, e.clientY, hitIdx);
    } else if (track === "audio") {
      this._activeTrack = "audio";
      // 查找点击的音频段
      let hitIdx = -1;
      for (let i = 0; i < this.timeline.audioSegments.length; i++) {
        const seg = this.timeline.audioSegments[i];
        if (frame >= seg.start && frame < seg.start + seg.length) { hitIdx = i; break; }
      }
      if (hitIdx >= 0) {
        this.selectedAudioIndices = new Set([hitIdx]);
        this.selectedIndices.clear();
        this.selectedMotionIndices.clear();
      }
      this.updateUIFromSelection();
      this.render();
      this._showAudioContextMenu(e.clientX, e.clientY, hitIdx);
    }
  }

  _createContextMenu(clientX, clientY) {
    const existing = document.querySelector(".yuan-clip-tl-context-menu");
    if (existing) existing.remove();
    const menu = document.createElement("div");
    menu.className = "yuan-clip-tl-context-menu";
    menu.style.left = clientX + "px";
    menu.style.top = clientY + "px";
    const addItem = (label, callback) => {
      const item = document.createElement("div");
      item.className = "item";
      item.textContent = label;
      item.onclick = () => { callback(); menu.remove(); };
      menu.appendChild(item);
    };
    const attach = () => {
      document.body.appendChild(menu);
      const close = (ev) => {
        if (!menu.contains(ev.target)) {
          menu.remove();
          document.removeEventListener("click", close);
        }
      };
      setTimeout(() => document.addEventListener("click", close), 10);
    };
    return { menu, addItem, attach };
  }

  _showSegmentContextMenu(clientX, clientY, segIndex) {
    const { addItem, attach } = this._createContextMenu(clientX, clientY);
    if (segIndex >= 0) {
      // 点击已有段：编辑操作
      addItem("拆分段", () => { if (!this.promptLocked) this.splitSelected(); });
      addItem("删除段", () => { if (!this.promptLocked) this.deleteSelected(); });
      addItem("复制段", () => this._copySegment());
      addItem("上传引导图", () => this.uploadImageForSegment(segIndex));
      addItem("清除引导图", () => {
        const seg = this.timeline.segments[segIndex];
        if (seg) {
          seg.imageFile = ""; seg.imageB64 = "";
          this._imageCache.delete(segIndex);
          this.commitChanges(); this.updateUIFromSelection(); this.render();
        }
      });
    } else {
      // 点击空白处：添加/粘贴
      addItem("添加段", () => { if (!this.promptLocked) this.addSegment(); });
    }
    if (this._clipboard && this._clipboard.type === "segment") {
      addItem("粘贴段", () => { if (!this.promptLocked) this._pasteSegment(); });
    }
    attach();
  }

  _showMotionContextMenu(clientX, clientY, motionIdx) {
    const { addItem, attach } = this._createContextMenu(clientX, clientY);
    if (motionIdx >= 0) {
      // 点击已有运动段：替换/删除/复制
      addItem("替换运动视频", () => { if (!this.promptLocked) this._replaceMotionVideo(motionIdx); });
      addItem("删除运动段", () => { if (!this.promptLocked) this._deleteMotionSegment(motionIdx); });
      addItem("复制段", () => this._copySegment());
    } else {
      // 点击空白处：添加
      addItem("添加运动视频", () => { if (!this.promptLocked) this.uploadMotionVideo(); });
    }
    if (this._clipboard && this._clipboard.type === "motion") {
      addItem("粘贴段", () => { if (!this.promptLocked) this._pasteSegment(); });
    }
    attach();
  }

  _showAudioContextMenu(clientX, clientY, audioIdx) {
    const { addItem, attach } = this._createContextMenu(clientX, clientY);
    if (audioIdx >= 0) {
      // 点击已有音频段：替换/删除/复制
      addItem("替换音频", () => { if (!this.promptLocked) this._replaceAudioFile(audioIdx); });
      addItem("删除音频段", () => { if (!this.promptLocked) this._deleteAudioSegment(audioIdx); });
      addItem("复制段", () => this._copySegment());
    } else {
      // 点击空白处：添加
      addItem("添加音频", () => { if (!this.promptLocked) this.uploadAudioFile(); });
    }
    if (this._clipboard && this._clipboard.type === "audio") {
      addItem("粘贴段", () => { if (!this.promptLocked) this._pasteSegment(); });
    }
    attach();
  }

  // ── 复制/粘贴段（三轨道通用，剪贴板存类型+深拷贝数据） ──

  _copySegment() {
    // 按 _activeTrack 决定复制来源（由右键点击轨道时设置）
    if (this._activeTrack === "segment") {
      const idx = this._firstSelected("segment");
      if (idx < 0 || idx >= this.timeline.segments.length) return;
      const seg = this.timeline.segments[idx];
      this._clipboard = { type: "segment", data: JSON.parse(JSON.stringify(seg)) };
    } else if (this._activeTrack === "motion") {
      const idx = this._firstSelected("motion");
      if (idx < 0 || idx >= this.timeline.motionSegments.length) return;
      const seg = this.timeline.motionSegments[idx];
      this._clipboard = { type: "motion", data: JSON.parse(JSON.stringify(seg)) };
    } else if (this._activeTrack === "audio") {
      const idx = this._firstSelected("audio");
      if (idx < 0 || idx >= this.timeline.audioSegments.length) return;
      const seg = this.timeline.audioSegments[idx];
      this._clipboard = { type: "audio", data: JSON.parse(JSON.stringify(seg)) };
    }
  }

  _pasteSegment() {
    if (!this._clipboard) return;
    const max = this.getMaxFrames();
    const { type, data } = this._clipboard;
    const copy = JSON.parse(JSON.stringify(data));
    if (type === "segment") {
      if (this.promptLocked) return;
      const slot = this._findFreeSlot(this.timeline.segments, copy.length, max);
      copy.start = slot.start;
      copy.length = slot.length;
      const usedColors = new Set(this.timeline.segments.map(s => s.color));
      copy.color = pickColor(usedColors);
      this.timeline.segments.push(copy);
      this.selectedIndices = new Set([this.timeline.segments.length - 1]);
      this.selectedMotionIndices.clear();
      this.selectedAudioIndices.clear();
      this._activeTrack = "segment";
    } else if (type === "motion") {
      if (this.promptLocked) return;
      const slot = this._findFreeSlot(this.timeline.motionSegments, copy.length, max);
      copy.start = slot.start;
      copy.length = slot.length;
      // 粘贴时重新分配 subjectNum，避免与原段冲突导致 K/V 注入帧范围覆盖
      copy.subjectNum = this.timeline.motionSegments.reduce((mx, s) => Math.max(mx, s.subjectNum || 0), 0) + 1;
      this.timeline.motionSegments.push(copy);
      this.selectedMotionIndices = new Set([this.timeline.motionSegments.length - 1]);
      this.selectedIndices.clear();
      this.selectedAudioIndices.clear();
      this._activeTrack = "motion";
    } else if (type === "audio") {
      if (this.promptLocked) return;
      const slot = this._findFreeSlot(this.timeline.audioSegments, copy.length, max);
      copy.start = slot.start;
      copy.length = slot.length;
      this.timeline.audioSegments.push(copy);
      this.selectedAudioIndices = new Set([this.timeline.audioSegments.length - 1]);
      this.selectedIndices.clear();
      this.selectedMotionIndices.clear();
      this._activeTrack = "audio";
    }
    this.commitChanges();
    this.updateUIFromSelection();
    this.render();
  }

  // 删除当前选中段（三轨道通用，供 Delete 键调用）
  _deleteSelectedAny() {
    if (this._activeTrack === "segment") {
      if (this.timeline.segments.length <= 1) return;
      if (this.promptLocked) return;
      const toDelete = [...this.selectedIndices].sort((a,b) => b-a);
      if (toDelete.length === 0) return;
      for (const idx of toDelete) {
        this.timeline.segments.splice(idx, 1);
      }
      this.selectedIndices.clear();
      const first = clamp(this._firstSelected("segment"), 0, Math.max(0, this.timeline.segments.length - 1));
      if (this.timeline.segments.length > 0) this.selectedIndices.add(first);
    } else if (this._activeTrack === "motion") {
      if (this.promptLocked) return;
      const toDelete = [...this.selectedMotionIndices].sort((a,b) => b-a);
      for (const idx of toDelete) this.timeline.motionSegments.splice(idx, 1);
      this.selectedMotionIndices.clear();
    } else if (this._activeTrack === "audio") {
      if (this.promptLocked) return;
      const toDelete = [...this.selectedAudioIndices].sort((a,b) => b-a);
      for (const idx of toDelete) this.timeline.audioSegments.splice(idx, 1);
      this.selectedAudioIndices.clear();
    } else {
      return;
    }
    this.commitChanges();
    this.updateUIFromSelection();
    this.render();
  }

  // ── 运动段/音频段编辑操作 ──

  _deleteMotionSegment(idx) {
    if (this.promptLocked) return;
    if (idx < 0 || idx >= this.timeline.motionSegments.length) return;
    this.timeline.motionSegments.splice(idx, 1);
    this.selectedMotionIndices.clear();
    this.commitChanges();
    this.updateUIFromSelection();
    this.render();
  }

  _replaceMotionVideo(idx) {
    if (this.promptLocked) return;
    if (idx < 0 || idx >= this.timeline.motionSegments.length) return;
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*";
    input.addEventListener("change", async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        this.showUploadProgress("替换运动视频中…");
        const result = await this.uploadFileChunked(file, (done, total) => {
          this.showUploadProgress(`上传视频 ${done}/${total}…`);
        });
        if (result?.name) {
          const seg = this.timeline.motionSegments[idx];
          seg.videoFile = result.name;
          seg.trimStart = 0.0;
          // 清除旧缩略图缓存
          this._videoThumbCache.delete(seg.videoFile);
          this.commitChanges();
          this.updateUIFromSelection();
          this.render();
          // 重新提取缩略图
          this._extractVideoThumbnails(seg).then(() => this.render());
        }
        this.hideUploadProgress();
      } catch (err) {
        this.showUploadProgress("替换失败: " + err.message);
        setTimeout(() => this.hideUploadProgress(), 3000);
      }
    });
    input.click();
  }

  _deleteAudioSegment(idx) {
    if (this.promptLocked) return;
    if (idx < 0 || idx >= this.timeline.audioSegments.length) return;
    this.timeline.audioSegments.splice(idx, 1);
    this.selectedAudioIndices.clear();
    this.commitChanges();
    this.updateUIFromSelection();
    this.render();
  }

  _replaceAudioFile(idx) {
    if (this.promptLocked) return;
    if (idx < 0 || idx >= this.timeline.audioSegments.length) return;
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "audio/*";
    input.addEventListener("change", async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        this.showUploadProgress("替换音频中…");
        const result = await this.uploadFileChunked(file, (done, total) => {
          this.showUploadProgress(`上传音频 ${done}/${total}…`);
        });
        if (result?.name || result?.audio_file) {
          const audioName = result.audio_file || result.name;
          // 获取音频波形
          let peaks = result.peaks || null;
          if (!peaks) {
            try {
              const audioRes = await fetch(`/yuan_clip_timeline_get_audio?filename=${encodeURIComponent(audioName)}`);
              const audioData = await audioRes.json();
              peaks = audioData.peaks || [];
            } catch (_) { peaks = []; }
          }
          const seg = this.timeline.audioSegments[idx];
          seg.audioFile = audioName;
          seg.audioB64 = "";
          seg.fileName = file.name;
          seg.trimStart = 0.0;
          // 清除旧波形缓存
          this._audioBufferCache.delete(audioName);
          this.audioPeaksCache.set(audioName, peaks || []);
          this.commitChanges();
          this.updateUIFromSelection();
          this.render();
          // 重新解码波形
          this._decodeAudioBuffer(audioName).then(() => this.render());
        }
        this.hideUploadProgress();
      } catch (err) {
        this.showUploadProgress("替换失败: " + err.message);
        setTimeout(() => this.hideUploadProgress(), 3000);
      }
    });
    input.click();
  }

  // ── 物理碰撞分配：在已有段中找第一个能放下新长度的空位（参考 LTX Director 算法） ──
  // 返回 { start, length }，length 可能被截断以适应总长度
  _findFreeSlot(segments, desiredLength, max) {
    if (!Array.isArray(segments)) segments = [];
    const sorted = [...segments].filter(Boolean)
      .map(s => ({ start: s.start || 0, length: s.length || 0 }))
      .sort((a, b) => a.start - b.start);

    // 期望长度不超过总长度
    let newLength = Math.max(MIN_SEGMENT_LENGTH, Math.min(desiredLength, max));
    let newStart = 0;

    // 遍历已有段找第一个能放下 newLength 的空隙
    for (let i = 0; i < sorted.length; i++) {
      const s = sorted[i];
      if (newStart + newLength <= s.start) break;
      newStart = Math.max(newStart, s.start + s.length);
    }

    // 若超出总长度，截断长度并紧贴末尾
    if (newStart + newLength > max) {
      // 尝试从 0 开始找更小的空隙（可能前面有空隙但不够大，这里仍优先末尾）
      if (newStart >= max) {
        // 完全没空间，回退到 0 并截断
        newStart = 0;
        newLength = Math.max(MIN_SEGMENT_LENGTH, Math.min(desiredLength, max));
      } else {
        newLength = Math.max(MIN_SEGMENT_LENGTH, max - newStart);
      }
    }
    return { start: newStart, length: newLength };
  }

  // 拖动段时的碰撞避让：基于"缝隙法"保证绝不与其他段重叠
  // 计算其他段之间的所有可用缝隙，将拖动段放到最近的可容纳缝隙中
  _resolveNoOverlap(segments, dragIdx, newStart, segLength, max, initialStart) {
    if (!Number.isFinite(newStart)) newStart = 0;
    if (segLength >= max) return 0;
    // 边界限制
    newStart = Math.max(0, Math.min(max - segLength, newStart));

    // 收集其他段的区间并按 start 排序
    const others = [];
    for (let i = 0; i < segments.length; i++) {
      if (i === dragIdx) continue;
      const o = segments[i];
      const s = o.start || 0;
      others.push({ start: s, end: s + (o.length || 0) });
    }
    if (others.length === 0) return newStart;
    others.sort((a, b) => a.start - b.start);

    // 计算可用缝隙：[0, first.start]、段之间、[last.end, max]
    const gaps = [];
    if (others[0].start > 0) gaps.push({ start: 0, end: others[0].start });
    for (let i = 0; i < others.length - 1; i++) {
      const gs = others[i].end;
      const ge = others[i + 1].start;
      if (ge > gs) gaps.push({ start: gs, end: ge });
    }
    const lastEnd = others[others.length - 1].end;
    if (max > lastEnd) gaps.push({ start: lastEnd, end: max });

    // 在每个能容纳拖动段的缝隙中，可放置范围是 [gap.start, gap.end - segLength]
    // 选择将 newStart 限制后距离最近的缝隙
    let bestPos = initialStart;
    let bestDist = Infinity;
    for (const gap of gaps) {
      if (gap.end - gap.start < segLength) continue; // 缝隙太小
      const lo = gap.start;
      const hi = gap.end - segLength;
      const clamped = Math.max(lo, Math.min(hi, newStart));
      const dist = Math.abs(clamped - newStart);
      if (dist < bestDist) {
        bestDist = dist;
        bestPos = clamped;
      }
    }
    return bestPos;
  }

  // ── 变更操作 ──

  addSegment() {
    const max = this.getMaxFrames();
    const n = this.timeline.segments.length;
    if (max < (n + 1) * MIN_SEGMENT_LENGTH) return;
    const desired = Math.max(MIN_SEGMENT_LENGTH, Math.floor(max / (n + 1)));
    const usedColors = new Set(this.timeline.segments.map(s => s.color));
    const newSeg = {
      start: 0, length: desired, prompt: "", color: pickColor(usedColors),
      type: "image", imageFile: "", imageB64: "",
      isEndFrame: false, strength: 1.0, trimStart: 0.0,
    };
    this.timeline.segments.push(newSeg);
    this.trimToFit(n);
    let total = this.timeline.segments.reduce((a, s) => a + s.length, 0);
    if (total > max) this.timeline.segments[n].length -= (total - max);
    this.selectedIndices = new Set([n]);
    this.commitChanges();
    this.updateUIFromSelection();
    this.render();
  }

  deleteSelected() {
    if (this.timeline.segments.length <= 1) return;
    const toDelete = [...this.selectedIndices].sort((a,b) => b-a);
    for (const idx of toDelete) {
      if (idx >= 0 && idx < this.timeline.segments.length) {
        this.timeline.segments.splice(idx, 1);
      }
    }
    this.selectedIndices.clear();
    const first = clamp(this._firstSelected("segment"), 0, Math.max(0, this.timeline.segments.length - 1));
    if (this.timeline.segments.length > 0) this.selectedIndices.add(first);
    this.commitChanges();
    this.updateUIFromSelection();
    this.render();
  }

  splitSelected() {
    const idx = this._firstSelected("segment");
    const seg = this.timeline.segments[idx];
    if (!seg || seg.length < 2) return;
    const half = Math.floor(seg.length / 2);
    const usedColors = new Set(this.timeline.segments.map(s => s.color));
    const newSeg = {
      start: seg.start + half,
      length: seg.length - half,
      prompt: seg.prompt,
      color: pickColor(usedColors),
      type: seg.type,
      imageFile: "",
      imageB64: "",
      isEndFrame: false,
      strength: 1.0,
      trimStart: 0.0,
    };
    seg.length = half;
    this.timeline.segments.splice(idx + 1, 0, newSeg);
    this.commitChanges();
    this.updateUIFromSelection();
    this.render();
  }

  clearAll() {
    this.timeline = defaultTimeline(this.getMaxFrames());
    this.selectedIndices = new Set([0]);
    this.selectedMotionIndices.clear();
    this.selectedAudioIndices.clear();
    this.audioPeaksCache.clear();
    this._imageCache.clear();
    // 清理磁盘上残留的 segimg_*.png 文件
    fetch("/yuan_clip_timeline_clear_seg_images", { method: "POST" }).catch(() => {});
    this.commitChanges();
    this.updateUIFromSelection();
    this.render();
  }

  toggleTimeUnits() {
    if (!this.timeUnitsWidget) return;
    this.timeUnitsWidget.value = this.isSecondsMode() ? "frames" : "seconds";
    this.timeUnitsWidget.callback?.(this.timeUnitsWidget.value);
    this._updateToolbarState();
    this.updateUIFromSelection();
    this.render();
  }

  togglePromptLock() {
    if (!this.promptLockWidget) return;
    this.promptLockWidget.value = !this.promptLocked;
    this.promptLockWidget.callback?.(this.promptLockWidget.value);
    this._updateLockState();
    if (this.promptLocked) {
      // 锁定状态下拒绝上游数据，仅同步 text_input 到时间轴
      this.syncFromTextInput();
    } else {
      // 解锁状态下获取上游数据（覆盖时间轴）
      this._syncGlobalPromptFromUpstream();
      if (this.hasSegmentImagesConnected()) this._applyUpstreamSegmentImages();
      if (this.hasMotionImagesConnected()) this._applyUpstreamMotionImages();
    }
    this._updateToolbarState();
    this.updateUIFromSelection();
    this.render();
  }

  // ── @图X 角色解析（从 global_prompt 解析 @图X=描述 行）──

  // 返回 [{tag: "@图1", desc: "描述", index: 0}, ...]
  _parseCharacterLines() {
    const gp = this.globalPromptWidget?.value || "";
    const lines = gp.split("\n");
    const chars = [];
    const pattern = /^@图(\d+)\s*[=：:]\s*(.+)/;
    for (const line of lines) {
      const m = line.trim().match(pattern);
      if (m) {
        chars.push({ tag: `@图${m[1]}`, index: parseInt(m[1], 10) - 1, desc: m[2].trim() });
      }
    }
    return chars;
  }

  cycleMotionImageFrames() {
    const options = [8, 16, 24, 32];
    const idx = options.indexOf(this.motionImageFrames);
    this.motionImageFrames = options[(idx + 1) % options.length];
    if (this.motionImageFramesWidget) {
      this.motionImageFramesWidget.value = this.motionImageFrames;
      this.motionImageFramesWidget.callback?.(this.motionImageFrames);
    }
    this._updateToolbarState();
    // 直接更新现有 motion 段的长度，不重新调用 _applyUpstreamMotionImages
    // 这样可以保留选中状态，避免属性面板丢失
    const segs = this.timeline.motionSegments;
    if (segs.length > 0) {
      const max = this.getMaxFrames();
      let cursor = 0;
      for (const seg of segs) {
        seg.start = cursor;
        seg.length = Math.min(this.motionImageFrames, max - cursor);
        if (seg.length <= 0) seg.length = Math.min(this.motionImageFrames, max);
        cursor += seg.length;
      }
      this.commitChanges();
      this.render();
    }
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

  handleMaxFramesChange() {
    const max = this.getMaxFrames();
    const segs = this.timeline.segments;
    if (segs.length === 0) return;
    const currentTotal = segs.reduce((a, s) => a + s.length, 0);
    if (currentTotal === 0) {
      this.distributeEvenly();
      return;
    }
    // 按比例分配（最大余数法）
    const ratios = segs.map(s => s.length / currentTotal);
    const newLengths = segs.map((_, i) => Math.max(MIN_SEGMENT_LENGTH, Math.round(ratios[i] * max)));
    let total = newLengths.reduce((a, b) => a + b, 0);
    // 修正溢出
    while (total > max) {
      let maxIdx = -1, maxVal = MIN_SEGMENT_LENGTH;
      for (let i = 0; i < newLengths.length; i++) {
        if (newLengths[i] > maxVal) { maxVal = newLengths[i]; maxIdx = i; }
      }
      if (maxIdx < 0) break;
      newLengths[maxIdx]--; total--;
    }
    // 修正不足
    while (total < max) {
      let minIdx = 0, minVal = Infinity;
      for (let i = 0; i < newLengths.length; i++) {
        if (newLengths[i] < minVal) { minVal = newLengths[i]; minIdx = i; }
      }
      newLengths[minIdx]++; total++;
    }
    for (let i = 0; i < segs.length; i++) segs[i].length = newLengths[i];
    this.commitChanges();
    this.updateUIFromSelection();
    this.render();
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
    this.commitChanges();
    this.updateUIFromSelection();
    this.render();
  }

  // ── 序列化与 widget 同步 ──

  commitChanges() {
    this.syncWidgetsFromTimeline();
    this.node.graph?.setDirtyCanvas?.(true, true);
  }

  syncWidgetsFromTimeline() {
    const tl = this.timeline;
    // 重新计算段 start（绝对位置 = 前面段 length 之和）
    let cursor = 0;
    for (const seg of tl.segments) {
      seg.start = cursor;
      cursor += seg.length;
    }
    // 同步 global_prompt
    if (this.globalPromptWidget) tl.global_prompt = this.globalPromptWidget.value || "";
    // 序列化到 timeline_data
    if (this.timelineDataWidget) {
      this.timelineDataWidget.value = JSON.stringify(tl);
    }
    // 派生 widget
    const segs = tl.segments;
    if (this.localPromptsWidget) {
      this.localPromptsWidget.value = segs.map(s => s.prompt).join(" | ");
    }
    if (this.segmentLengthsWidget) {
      this.segmentLengthsWidget.value = segs.map(s => s.length).join(", ");
    }
    if (this.guideStrengthWidget) {
      this.guideStrengthWidget.value = segs
        .filter(s => s.imageFile || s.imageB64)
        .map(s => (Number.isFinite(s.strength) ? s.strength : 1.0).toFixed(2))
        .join(", ");
    }
    // 根据 motion/audio 段自动设置开关
    if (this.useCustomMotionWidget) {
      this.useCustomMotionWidget.value = tl.motionSegments.length > 0;
    }
    if (this.useCustomAudioWidget) {
      this.useCustomAudioWidget.value = tl.audioSegments.length > 0;
    }
    this._updateToolbarState();
  }

  // ── text_input 智能分配 ──

  // 获取应保留的 motionSegments/audioSegments：优先从现有 timeline，其次从 timelineDataWidget 解析
  // 避免 _buildTimelineFromLines 重建 segments 时清空 motion/audio 轨道（切换工作流丢失 bug 的根因）
  _preservedMotionAudio() {
    let motionSegments = [];
    let audioSegments = [];
    // 优先从现有 timeline 保留
    if (this.timeline?.motionSegments?.length > 0) {
      motionSegments = this.timeline.motionSegments;
    } else if (this.timelineDataWidget?.value) {
      const parsed = parseInitial(this.timelineDataWidget.value, this.getMaxFrames());
      motionSegments = parsed.motionSegments || [];
    }
    if (this.timeline?.audioSegments?.length > 0) {
      audioSegments = this.timeline.audioSegments;
    } else if (this.timelineDataWidget?.value) {
      const parsed = parseInitial(this.timelineDataWidget.value, this.getMaxFrames());
      audioSegments = parsed.audioSegments || [];
    }
    return { motionSegments, audioSegments };
  }

  _buildTimelineFromLines(lines) {
    const max = this.getMaxFrames();
    // 保留现有的 motion/audio 轨道数据，不被 segments 重建清空
    const { motionSegments, audioSegments } = this._preservedMotionAudio();
    // 保留现有的引导图（imageFile/imageB64/type），按段索引回填，避免 segment_images 分配的图像丢失
    const existingSegs = this.timeline?.segments || [];
    const preserveImg = (i) => {
      const ex = existingSegs[i];
      if (!ex) return { imageFile: "", imageB64: "", type: "image" };
      return {
        imageFile: ex.imageFile || "",
        imageB64: ex.imageB64 || "",
        type: ex.imageFile || ex.imageB64 ? ex.type : "image",
      };
    };
    // 尝试按时间格式解析
    const parsed = [];
    for (const line of lines) {
      const match = line.match(TIME_RANGE_PATTERN);
      if (match) {
        const startSec = parseFloat(match[1]);
        const endSec = parseFloat(match[2]);
        const prompt = line.substring(match[0].length).trim();
        if (prompt) {
          parsed.push({ prompt, startSec, endSec, durationSec: Math.max(0, endSec - startSec) });
        }
      }
    }
    if (parsed.length > 0 && parsed.length === lines.length) {
      const fps = this.getFps();
      const maxEndSec = Math.max(...parsed.map(p => p.endSec));
      const rawMax = Math.floor(maxEndSec * fps) + 1;
      const newMax = (Math.floor((rawMax - 2) / 8) + 1) * 8 + 1;
      if (this.maxFramesWidget) this.maxFramesWidget.value = newMax;
      const frameAlloc = parsed.map(p => Math.max(MIN_SEGMENT_LENGTH, Math.round(p.durationSec * fps)));
      const totalFrames = frameAlloc.reduce((a, b) => a + b, 0);
      const targetMax = this.getMaxFrames();
      if (totalFrames > targetMax) {
        const excess = totalFrames - targetMax;
        frameAlloc[frameAlloc.length - 1] = Math.max(MIN_SEGMENT_LENGTH, frameAlloc[frameAlloc.length - 1] - excess);
      } else if (totalFrames < targetMax) {
        frameAlloc[frameAlloc.length - 1] += (targetMax - totalFrames);
      }
      const segments = parsed.map((p, i) => ({
        start: 0, length: frameAlloc[i], prompt: p.prompt, color: PALETTE[i % PALETTE.length],
        ...preserveImg(i), isEndFrame: false, strength: 1.0, trimStart: 0.0,
      }));
      // 计算 start
      let c = 0;
      for (const s of segments) { s.start = c; c += s.length; }
      return {
        segments,
        motionSegments, audioSegments,
        global_prompt: "",
      };
    }
    // 无时间格式：均分
    const baseLen = Math.max(MIN_SEGMENT_LENGTH, Math.floor(max / lines.length));
    const segments = lines.map((line, i) => ({
      start: 0, length: baseLen, prompt: line, color: PALETTE[i % PALETTE.length],
      ...preserveImg(i), isEndFrame: false, strength: 1.0, trimStart: 0.0,
    }));
    let total = segments.reduce((a, s) => a + s.length, 0);
    if (total !== max && segments.length > 0) {
      segments[segments.length - 1].length = Math.max(MIN_SEGMENT_LENGTH, segments[segments.length - 1].length + (max - total));
    }
    let c = 0;
    for (const s of segments) { s.start = c; c += s.length; }
    return {
      segments,
      motionSegments, audioSegments,
      global_prompt: "",
    };
  }

  _readConnectedTextInput() {
    const node = this.node;
    if (!node || !node.graph) return null;
    const inputIdx = node.findInputSlot ? node.findInputSlot("文本输入") : -1;
    if (inputIdx < 0) return null;
    const linkId = node.inputs[inputIdx]?.link;
    if (linkId == null) return null;
    const link = node.graph.links[linkId];
    if (!link) return null;
    const srcNode = node.graph.getNodeById(link.origin_id);
    if (!srcNode) return null;
    const candidates = [];
    for (const w of (srcNode.widgets || [])) {
      if (w.type === "customtext" || w.name === "string" || (w.name && /^(text|prompt|string|multiline)$/i.test(w.name))) {
        candidates.push(w);
      }
    }
    if (candidates.length === 0) {
      for (const w of (srcNode.widgets || [])) {
        if (!w.hidden && typeof w.value === "string" && w.value.trim()) candidates.push(w);
      }
    }
    if (candidates.length === 0) return null;
    const stringW = candidates.find(w => w.name === "string");
    if (stringW && stringW.value?.trim()) return stringW.value;
    return candidates.map(w => w.value).filter(v => typeof v === "string" && v.trim()).join("\n") || null;
  }

  _getTextInputValue() {
    const widgetVal = this.textInputWidget?.value?.trim();
    if (widgetVal) return widgetVal;
    return this._readConnectedTextInput();
  }

  // ── 全局提示词 上游连接读取（与 文本输入 保持一致） ──

  _readConnectedGlobalPrompt() {
    const node = this.node;
    if (!node || !node.graph) return null;
    const inputIdx = node.findInputSlot ? node.findInputSlot("全局提示词") : -1;
    if (inputIdx < 0) return null;
    const linkId = node.inputs[inputIdx]?.link;
    if (linkId == null || linkId === -1) return null;
    const link = node.graph.links[linkId];
    if (!link) return null;
    const srcNode = node.graph.getNodeById(link.origin_id);
    if (!srcNode) return null;
    const candidates = [];
    for (const w of (srcNode.widgets || [])) {
      if (w.type === "customtext" || w.name === "string" || (w.name && /^(text|prompt|string|multiline|global_prompt)$/i.test(w.name))) {
        candidates.push(w);
      }
    }
    if (candidates.length === 0) {
      for (const w of (srcNode.widgets || [])) {
        if (!w.hidden && typeof w.value === "string" && w.value.trim()) candidates.push(w);
      }
    }
    if (candidates.length === 0) return null;
    const stringW = candidates.find(w => w.name === "string");
    if (stringW && stringW.value?.trim()) return stringW.value;
    return candidates.map(w => w.value).filter(v => typeof v === "string" && v.trim()).join("\n") || null;
  }

  _syncGlobalPromptFromUpstream() {
    // 锁定状态下拒绝上游数据，完全以时间轴编辑器内的数据为准
    if (this.promptLocked) return;
    const upstream = this._readConnectedGlobalPrompt();
    if (upstream && upstream.trim()) {
      this.timeline.global_prompt = upstream;
      // 同时更新 widget 值（如果 widget 存在且未被隐藏）
      if (this.globalPromptWidget && !this.globalPromptWidget.hidden) {
        this.globalPromptWidget.value = upstream;
      }
      this.commitChanges();
      // 上游文本变化后，重新应用到 IC-LoRA 轨道（更新段描述）
      if (this.hasMotionImagesConnected()) {
        this._applyUpstreamMotionImages();
      }
    }
  }

  _updateLockState() {
    this.promptLocked = this.promptLockWidget?.value !== false;
  }

  syncFromTextInput() {
    const raw = this._getTextInputValue();
    if (!raw || !raw.trim()) {
      // 空文本：从 timeline_data 持久化值恢复（保留 segment_images 分配的 imageFile），
      // 无持久化值时才回退到默认时间轴；保留现有的 motion/audio 轨道数据
      const { motionSegments, audioSegments } = this._preservedMotionAudio();
      const base = this.timelineDataWidget?.value
        ? parseInitial(this.timelineDataWidget.value, this.getMaxFrames())
        : defaultTimeline(this.getMaxFrames());
      this.timeline = {
        ...base,
        motionSegments, audioSegments,
      };
      this.selectedIndices = new Set([0]);
      this.commitChanges();
      this.updateUIFromSelection();
      this.render();
      return;
    }
    const lines = raw.split("\n").map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length === 0) return;
    this.timeline = this._buildTimelineFromLines(lines);
    this.selectedIndices = new Set([0]);
    this.commitChanges();
    this.updateUIFromSelection();
    this.render();
  }

  _restoreFromWidgets() {
    const textVal = this.promptLocked ? this._getTextInputValue() : null;
    if (textVal) {
      const lines = textVal.split("\n").map(l => l.trim()).filter(l => l.length > 0);
      this.timeline = lines.length > 0
        ? this._buildTimelineFromLines(lines)
        : parseInitial(this.timelineDataWidget?.value, this.getMaxFrames());
    } else {
      this.timeline = parseInitial(this.timelineDataWidget?.value, this.getMaxFrames());
    }
    this.selectedIndices = new Set([0]);
    this.selectedMotionIndices.clear();
    this.selectedAudioIndices.clear();
    this._preloadAllImages();
  }

  // ── 属性面板更新 ──

  updateUIFromSelection() {
    const panel = this.propertiesPanel;
    panel.innerHTML = "";

    // 选中运动段时显示运动属性
    const motionIdx = this._firstSelected("motion");
    if (motionIdx >= 0 && this.timeline.motionSegments[motionIdx]) {
      panel.appendChild(this._buildMotionProps(motionIdx));
      this._updateTotalLabel();
      return;
    }
    // 选中音频段时显示音频属性
    const audioIdx = this._firstSelected("audio");
    if (audioIdx >= 0 && this.timeline.audioSegments[audioIdx]) {
      panel.appendChild(this._buildAudioProps(audioIdx));
      this._updateTotalLabel();
      return;
    }

    const segIdx = this._firstSelected("segment");
    const seg = this.timeline.segments[segIdx];
    if (seg) {
      panel.appendChild(this._buildSegmentProps(segIdx, seg));
    }

    this._updateTotalLabel();
  }

  _buildSegmentProps(idx, seg) {
    const section = document.createElement("div");
    section.className = "yuan-clip-tl-prop-section";
    const title = document.createElement("div");
    title.className = "yuan-clip-tl-prop-title";
    title.textContent = `段 ${idx + 1} / ${this.timeline.segments.length}`;
    if (seg.imageFile || seg.imageB64) {
      const badge = document.createElement("span");
      badge.className = "badge"; badge.textContent = "引导图";
      title.appendChild(badge);
    }
    section.appendChild(title);

    // 字段网格容器
    const grid = document.createElement("div");
    grid.className = "yuan-clip-tl-field-grid";
    section.appendChild(grid);

    const mkField = (labelText, contentEl, full = false) => {
      const f = document.createElement("div");
      f.className = "field" + (full ? " full" : "");
      const l = document.createElement("label");
      l.textContent = labelText;
      f.appendChild(l);
      f.appendChild(contentEl);
      grid.appendChild(f);
    };

    // 提示词（独占一行）
    const promptTextarea = document.createElement("textarea");
    promptTextarea.value = seg.prompt || "";
    promptTextarea.readOnly = this.promptLocked;
    promptTextarea.placeholder = this.promptLocked ? "提示词已锁定 — 切换 prompt_lock 为 false 以编辑" : "输入提示词…";
    promptTextarea.addEventListener("input", () => {
      if (this.promptLocked) return;
      seg.prompt = promptTextarea.value;
      if (this._textCommitTimer) clearTimeout(this._textCommitTimer);
      this._textCommitTimer = setTimeout(() => {
        this._textCommitTimer = null;
        this.commitChanges();
      }, 120);
    });
    promptTextarea.addEventListener("blur", () => {
      if (this._textCommitTimer) { clearTimeout(this._textCommitTimer); this._textCommitTimer = null; this.commitChanges(); }
    });
    promptTextarea.addEventListener("pointerdown", e => e.stopPropagation());
    promptTextarea.addEventListener("wheel", e => e.stopPropagation(), { passive: true });
    mkField("提示词", promptTextarea, true);

    // 长度
    const lenInput = document.createElement("input");
    lenInput.type = "number";
    lenInput.value = this.lengthInputValueFor(seg.length);
    lenInput.step = this.isSecondsMode() ? (1 / this.getFps()).toFixed(4) : "1";
    lenInput.min = this.isSecondsMode() ? (MIN_SEGMENT_LENGTH / this.getFps()).toFixed(4) : MIN_SEGMENT_LENGTH;
    lenInput.disabled = this.promptLocked;
    lenInput.addEventListener("input", () => {
      if (this.promptLocked) return;
      const raw = parseFloat(lenInput.value);
      if (!Number.isFinite(raw)) return;
      const frames = Math.max(MIN_SEGMENT_LENGTH, Math.round(this.isSecondsMode() ? raw * this.getFps() : raw));
      this._setLengthShifting(idx, frames, this.timeline.segments.map(s => s.length));
      this.commitChanges();
      this.render();
    });
    lenInput.addEventListener("pointerdown", e => e.stopPropagation());
    mkField("长度", lenInput);

    // 颜色
    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = seg.color || "#4f8edc";
    colorInput.disabled = this.promptLocked;
    if (!this.promptLocked) {
      colorInput.addEventListener("input", () => {
        seg.color = colorInput.value;
        this.commitChanges();
        this.render();
      });
    }
    mkField("颜色", colorInput);

    // 类型
    const typeSelect = document.createElement("select");
    for (const t of ["image", "video"]) {
      const opt = document.createElement("option");
      opt.value = t; opt.textContent = t;
      if (seg.type === t) opt.selected = true;
      typeSelect.appendChild(opt);
    }
    typeSelect.disabled = this.promptLocked;
    if (!this.promptLocked) {
      typeSelect.addEventListener("change", () => {
        seg.type = typeSelect.value;
        this.commitChanges();
        this.updateUIFromSelection();
      });
    }
    mkField("类型", typeSelect);

    // 强度
    const strengthRow = document.createElement("div");
    strengthRow.className = "yuan-clip-tl-strength-row";
    const strengthInput = document.createElement("input");
    strengthInput.type = "range";
    strengthInput.min = "0"; strengthInput.max = "2"; strengthInput.step = "0.05";
    strengthInput.value = Number.isFinite(seg.strength) ? seg.strength : 1.0;
    strengthInput.disabled = this.promptLocked;
    const strengthVal = document.createElement("span");
    strengthVal.className = "yuan-clip-tl-strength-val";
    strengthVal.textContent = parseFloat(strengthInput.value).toFixed(2);
    if (!this.promptLocked) {
      strengthInput.addEventListener("input", () => {
        seg.strength = parseFloat(strengthInput.value);
        strengthVal.textContent = seg.strength.toFixed(2);
        this.commitChanges();
      });
    }
    strengthRow.appendChild(strengthInput);
    strengthRow.appendChild(strengthVal);
    mkField("强度", strengthRow);

    // isEndFrame 复选框
    const endLabel = document.createElement("label");
    endLabel.className = "yuan-clip-tl-checkbox-label";
    const endCheckbox = document.createElement("input");
    endCheckbox.type = "checkbox";
    endCheckbox.checked = !!seg.isEndFrame;
    endCheckbox.disabled = this.promptLocked;
    if (!this.promptLocked) {
      endCheckbox.addEventListener("change", () => {
        seg.isEndFrame = endCheckbox.checked;
        this.commitChanges();
      });
    }
    endLabel.appendChild(endCheckbox);
    endLabel.appendChild(document.createTextNode("末尾帧"));
    mkField("插入位置", endLabel);

    // trimStart（type=video 时）
    if (seg.type === "video") {
      const trimInput = document.createElement("input");
      trimInput.type = "number";
      trimInput.step = "0.1"; trimInput.min = "0";
      trimInput.value = Number.isFinite(seg.trimStart) ? seg.trimStart : 0;
      trimInput.disabled = this.promptLocked;
      if (!this.promptLocked) {
        trimInput.addEventListener("input", () => {
          seg.trimStart = parseFloat(trimInput.value) || 0;
          this.commitChanges();
        });
      }
      mkField("裁剪起始", trimInput);
    }

    // 引导图像（仅显示预览）
    const imgWrap = document.createElement("div");
    if (seg.imageFile || seg.imageB64) {
      const imgPreview = document.createElement("img");
      imgPreview.className = "yuan-clip-tl-image-preview";
      imgPreview.src = seg.imageB64 || this._buildImageUrl(seg.imageFile);
      imgWrap.appendChild(imgPreview);
    }
    mkField("引导图像", imgWrap, true);

    return section;
  }

  _buildMotionProps(idx) {
    const seg = this.timeline.motionSegments[idx];
    const section = document.createElement("div");
    section.className = "yuan-clip-tl-prop-section";
    const title = document.createElement("div");
    title.className = "yuan-clip-tl-prop-title";
    // 显示合并段帧数信息
    const charTag = seg.frameFiles ? ` (${seg.frameFiles.length}张合并)` : "";
    title.textContent = `运动段 ${idx + 1}${charTag}`;
    // 静态图像时显示 badge
    if (seg.isStaticImage) {
      const badge = document.createElement("span");
      badge.className = "badge"; badge.textContent = "静态图";
      title.appendChild(badge);
    }
    section.appendChild(title);

    const grid = document.createElement("div");
    grid.className = "yuan-clip-tl-field-grid";
    section.appendChild(grid);

    const addField = (labelText, el, full = false) => {
      const f = document.createElement("div");
      f.className = "field" + (full ? " full" : "");
      const l = document.createElement("label");
      l.textContent = labelText;
      f.appendChild(l);
      f.appendChild(el);
      grid.appendChild(f);
    };

    // 描述（独占一行，参考主轨段落提示词编辑方法）
    const descTextarea = document.createElement("textarea");
    descTextarea.value = seg.description || "";
    descTextarea.readOnly = this.promptLocked;
    descTextarea.placeholder = this.promptLocked ? "提示词已锁定 — 切换 prompt_lock 为 false 以编辑" : "输入运动段描述（可包含 @图X 引用）…";
    if (!this.promptLocked) {
      descTextarea.addEventListener("input", () => {
        seg.description = descTextarea.value;
        if (this._textCommitTimer) clearTimeout(this._textCommitTimer);
        this._textCommitTimer = setTimeout(() => {
          this._textCommitTimer = null;
          this.commitChanges();
        }, 120);
      });
      descTextarea.addEventListener("blur", () => {
        if (this._textCommitTimer) { clearTimeout(this._textCommitTimer); this._textCommitTimer = null; this.commitChanges(); }
      });
    }
    descTextarea.addEventListener("pointerdown", e => e.stopPropagation());
    descTextarea.addEventListener("wheel", e => e.stopPropagation(), { passive: true });
    addField("描述", descTextarea, true);

    // 剩余全局文本（非 @图X=描述 行）：在描述下方显示，所有 IC 分段共享同一份
    // 解锁状态下可编辑，编辑后同步到所有 IC-运动段描述及 global_prompt（保留 @图X=描述 行）
    const remainingText = this._getRemainingGlobalText();
    if (remainingText || !this.promptLocked) {
      const remTextarea = document.createElement("textarea");
      remTextarea.value = remainingText;
      remTextarea.readOnly = this.promptLocked;
      remTextarea.style.opacity = this.promptLocked ? "0.85" : "1.0";
      remTextarea.style.background = this.promptLocked ? "#1a1a1a" : "#2a2a2a";
      remTextarea.placeholder = "（无剩余全局文本）";
      if (!this.promptLocked) {
        remTextarea.addEventListener("input", () => {
          this._syncRemainingTextToGlobalPrompt(remTextarea.value);
          if (this._textCommitTimer) clearTimeout(this._textCommitTimer);
          this._textCommitTimer = setTimeout(() => {
            this._textCommitTimer = null;
            this.commitChanges();
            // 同步后刷新所有 IC-运动段属性面板（保持显示一致）
            this.updateUIFromSelection();
          }, 200);
        });
        remTextarea.addEventListener("blur", () => {
          if (this._textCommitTimer) {
            clearTimeout(this._textCommitTimer);
            this._textCommitTimer = null;
            this.commitChanges();
            this.updateUIFromSelection();
          }
        });
      }
      remTextarea.addEventListener("pointerdown", e => e.stopPropagation());
      remTextarea.addEventListener("wheel", e => e.stopPropagation(), { passive: true });
      addField("剩余全局文本（合并执行）", remTextarea, true);
    }

    // 起始帧
    const startInput = document.createElement("input");
    startInput.type = "number"; startInput.min = "0";
    startInput.value = seg.start;
    startInput.disabled = this.promptLocked;
    if (!this.promptLocked) {
      startInput.addEventListener("input", () => {
        seg.start = Math.max(0, parseInt(startInput.value, 10) || 0);
        this.commitChanges(); this.render();
      });
    }
    addField("起始帧", startInput);

    // 长度
    const lenInput = document.createElement("input");
    lenInput.type = "number"; lenInput.min = String(MIN_SEGMENT_LENGTH);
    lenInput.value = seg.length;
    lenInput.disabled = this.promptLocked;
    if (!this.promptLocked) {
      lenInput.addEventListener("input", () => {
        seg.length = Math.max(MIN_SEGMENT_LENGTH, parseInt(lenInput.value, 10) || MIN_SEGMENT_LENGTH);
        this.commitChanges(); this.render();
      });
    }
    addField("长度", lenInput);

    // trimStart
    const trimInput = document.createElement("input");
    trimInput.type = "number"; trimInput.step = "0.1"; trimInput.min = "0";
    trimInput.value = seg.trimStart || 0;
    trimInput.disabled = this.promptLocked;
    if (!this.promptLocked) {
      trimInput.addEventListener("input", () => {
        seg.trimStart = parseFloat(trimInput.value) || 0;
        this.commitChanges();
      });
    }
    addField("裁剪起始", trimInput);

    return section;
  }

  _buildAudioProps(idx) {
    const seg = this.timeline.audioSegments[idx];
    const section = document.createElement("div");
    section.className = "yuan-clip-tl-prop-section";
    const title = document.createElement("div");
    title.className = "yuan-clip-tl-prop-title";
    title.textContent = `音频段 ${idx + 1}`;
    section.appendChild(title);

    const grid = document.createElement("div");
    grid.className = "yuan-clip-tl-field-grid";
    section.appendChild(grid);

    const addField = (labelText, el, full = false) => {
      const f = document.createElement("div");
      f.className = "field" + (full ? " full" : "");
      const l = document.createElement("label");
      l.textContent = labelText;
      f.appendChild(l);
      f.appendChild(el);
      grid.appendChild(f);
    };

    // 音频播放控件
    if (seg.audioFile || seg.audioB64) {
      const audioSrc = seg.audioB64 || this._buildImageUrl(seg.audioFile);
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "metadata";
      audio.style.width = "100%";
      audio.src = audioSrc;
      addField("试听", audio, true);
    }

    const startInput = document.createElement("input");
    startInput.type = "number"; startInput.min = "0";
    startInput.value = seg.start;
    startInput.disabled = this.promptLocked;
    if (!this.promptLocked) {
      startInput.addEventListener("input", () => {
        seg.start = Math.max(0, parseInt(startInput.value, 10) || 0);
        this.commitChanges(); this.render();
      });
    }
    addField("起始帧", startInput);

    const lenInput = document.createElement("input");
    lenInput.type = "number"; lenInput.min = String(MIN_SEGMENT_LENGTH);
    lenInput.value = seg.length;
    lenInput.disabled = this.promptLocked;
    if (!this.promptLocked) {
      lenInput.addEventListener("input", () => {
        seg.length = Math.max(MIN_SEGMENT_LENGTH, parseInt(lenInput.value, 10) || MIN_SEGMENT_LENGTH);
        this.commitChanges(); this.render();
      });
    }
    addField("长度", lenInput);

    const trimInput = document.createElement("input");
    trimInput.type = "number"; trimInput.step = "0.1"; trimInput.min = "0";
    trimInput.value = seg.trimStart || 0;
    trimInput.disabled = this.promptLocked;
    if (!this.promptLocked) {
      trimInput.addEventListener("input", () => {
        seg.trimStart = parseFloat(trimInput.value) || 0;
        this.commitChanges();
      });
    }
    addField("裁剪起始", trimInput);

    return section;
  }

  _updateTotalLabel() {
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

  // ── 渲染 ──

  render() {
    this._draw();
  }

  _draw() {
    const ctx = this.ctx;
    const w = this._cssWidth;
    if (!w) return;
    ctx.clearRect(0, 0, w, CANVAS_HEIGHT);
    this.drawRuler(ctx, w);
    this.drawSegments(ctx, w);
    this.drawMotionTrack(ctx, w);
    this.drawAudioTrack(ctx, w);
    this.drawPlayhead(ctx, w);
    this.drawRubberBand(ctx);
  }

  drawRubberBand(ctx) {
    if (!this._rubberBand) return;
    const { startX, startY, currentX, currentY } = this._rubberBand;
    const x = Math.min(startX, currentX);
    const y = Math.min(startY, currentY);
    const w = Math.abs(currentX - startX);
    const h = Math.abs(currentY - startY);
    ctx.strokeStyle = "#4f8edc";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 2]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(79,142,220,0.08)";
    ctx.fillRect(x, y, w, h);
  }

  _finishRubberBand() {
    const { startX, startY, currentX, currentY, track } = this._rubberBand;
    const rx = Math.min(startX, currentX);
    const ry = Math.min(startY, currentY);
    const rw = Math.abs(currentX - startX);
    const rh = Math.abs(currentY - startY);
    if (rw < 4 && rh < 4) return; // 太小忽略

    const ppf = this.pxPerFrame();
    const frameStart = rx / ppf;
    const frameEnd = (rx + rw) / ppf;

    // 查找轨道区域内的所有段
    let segments, set;
    if (track === "segment") {
      segments = this.timeline.segments;
      set = this.selectedIndices;
    } else if (track === "motion") {
      segments = this.timeline.motionSegments;
      set = this.selectedMotionIndices;
    } else if (track === "audio") {
      segments = this.timeline.audioSegments;
      set = this.selectedAudioIndices;
    } else return;

    // 先收集命中的段索引，未命中任何段时保持之前的选中状态（避免丢失属性面板）
    const hits = new Set();
    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      const segEnd = seg.start + seg.length;
      // 判断段是否与框选矩形相交
      if (seg.start < frameEnd && segEnd > frameStart) {
        hits.add(i);
      }
    }
    if (hits.size === 0) return; // 空白区域：保持之前的选中状态

    // 命中段：更新当前轨道选中，并清除其他轨道选中
    set.clear();
    hits.forEach(i => set.add(i));
    if (track === "segment") {
      this.selectedMotionIndices.clear();
      this.selectedAudioIndices.clear();
    } else if (track === "motion") {
      this.selectedIndices.clear();
      this.selectedAudioIndices.clear();
    } else if (track === "audio") {
      this.selectedIndices.clear();
      this.selectedMotionIndices.clear();
    }
    this._activeTrack = track;
    this.updateUIFromSelection();
  }

  // ── Playhead 播放头 ──

  drawPlayhead(ctx, w) {
    const ppf = this.pxPerFrame();
    const x = Math.floor(this.playheadFrame * ppf) + 0.5;
    // 竖线贯穿所有轨道
    ctx.strokeStyle = "#ff4444";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, CANVAS_HEIGHT);
    ctx.stroke();
    // 顶部把手（小三角形）
    ctx.fillStyle = "#ff4444";
    ctx.beginPath();
    ctx.moveTo(x - 5, 0);
    ctx.lineTo(x + 5, 0);
    ctx.lineTo(x, 6);
    ctx.closePath();
    ctx.fill();
  }

  // 吸附 playhead 到段边缘
  getSnappedPlayhead(frame) {
    if (!this.snapEnabled) return frame;
    const snapPx = 8; // 像素吸附阈值
    const ppf = this.pxPerFrame();
    const snapFrames = snapPx / ppf;
    // 候选吸附点：0、各段边界、maxFrames
    const candidates = [0, this.getMaxFrames()];
    let cursor = 0;
    for (const seg of this.timeline.segments) {
      cursor += seg.length;
      candidates.push(cursor);
    }
    let best = frame;
    let bestDist = snapFrames;
    for (const c of candidates) {
      const d = Math.abs(c - frame);
      if (d < bestDist) { bestDist = d; best = c; }
    }
    return Math.max(0, Math.min(this.getMaxFrames(), best));
  }

  hitPlayhead(mx, my) {
    if (my > RULER_HEIGHT + 6) return false;
    const ppf = this.pxPerFrame();
    const phX = this.playheadFrame * ppf;
    return Math.abs(mx - phX) <= 8;
  }

  setPlayhead(frame) {
    this.playheadFrame = Math.max(0, Math.min(this.getMaxFrames(), this.getSnappedPlayhead(frame)));
    this.render();
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

    for (const r of rects) {
      const seg = this.timeline.segments[r.index];
      const color = seg.color || PALETTE[r.index % PALETTE.length];
      const isSelected = this._isSelected("segment", r.index) && this.selectedMotionIndices.size === 0 && this.selectedAudioIndices.size === 0;
      const isHover = r.index === this.hoverIndex;
      const isDragging = this.reorder?.active && this.reorder.sourceIdx === r.index;

      const drawX = Math.floor(r.x);
      const drawW = Math.max(2, Math.floor(r.w));

      // 边框使用段颜色，填充使用轨道背景色（与空白区域一致，便于区分图像边界）
      ctx.strokeStyle = isDragging ? "#ffd54f" : color;
      ctx.lineWidth = isDragging || isSelected ? 3 : 2;
      ctx.globalAlpha = isDragging || isSelected ? 1.0 : (isHover ? 1.0 : 0.7);
      ctx.strokeRect(drawX + 0.5, blockY + 0.5, drawW - 1, blockH - 1);
      ctx.globalAlpha = 1.0;

      // 引导图缩略图：视频段优先绘制缩略图序列，图像段绘制单张封面
      if (seg.type === "video" && this._videoFileKey(seg)) {
        const drew = this._drawVideoThumbnails(ctx, seg, drawX, blockY, drawW, blockH);
        if (drew) {
          // 半透明遮罩，保证文字可读
          ctx.fillStyle = "rgba(0,0,0,0.45)";
          ctx.fillRect(drawX, blockY, drawW, blockH);
        } else if (seg.imageFile || seg.imageB64) {
          // 缩略图仍在提取，显示占位
          ctx.fillStyle = "#5cb85c";
          ctx.font = "10px sans-serif";
          ctx.textBaseline = "top";
          ctx.fillText("⏳", drawX + drawW - 14, blockY + 2);
        }
      } else {
        const img = this._imageCache.get(r.index);
        if (img && img.complete && img.naturalWidth > 0) {
          this._drawImageCover(ctx, img, drawX, blockY, drawW, blockH);
          // 半透明遮罩，保证文字可读
          ctx.fillStyle = "rgba(0,0,0,0.45)";
          ctx.fillRect(drawX, blockY, drawW, blockH);
        } else if (seg.imageFile || seg.imageB64) {
          // 图像仍在加载，显示占位图标
          ctx.fillStyle = "#e8a850";
          ctx.font = "10px sans-serif";
          ctx.textBaseline = "top";
          ctx.fillText("🖼", drawX + drawW - 14, blockY + 2);
        }
      }
      if (seg.type === "video") {
        ctx.fillStyle = "#5cb85c";
        ctx.font = "10px sans-serif";
        ctx.textBaseline = "top";
        ctx.fillText("▶", drawX + drawW - 26, blockY + 2);
      }

      // 提示词文字
      ctx.fillStyle = "#fff";
      ctx.font = "11px sans-serif";
      ctx.textBaseline = "top";
      const label = seg.prompt || `(段 ${r.index + 1})`;
      const [line1, line2] = this.wrapTwoLines(ctx, label, drawW - 8);
      ctx.fillText(line1, drawX + 4, blockY + 4);
      if (line2) ctx.fillText(line2, drawX + 4, blockY + 18);

      // 帧范围
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.font = "10px monospace";
      const range = `${this.formatTime(r.frameStart)}–${this.formatTime(r.frameEnd)} (${this.formatLength(seg.length)})`;
      ctx.fillText(this.truncateText(ctx, range, drawW - 8), drawX + 4, blockY + blockH - 14);
    }

    // 边界手柄
    if (!this.promptLocked && !this.reorder?.active) {
      for (let i = 0; i < rects.length - 1; i++) {
        const r = rects[i];
        const right = Math.floor(r.x + r.w);
        const isHover = i === this.hoverHandle || i === this.dragHandle;
        ctx.fillStyle = isHover ? "#fff" : "rgba(255,255,255,0.4)";
        ctx.fillRect(right - 1, blockY + 4, 2, blockH - 8);
      }
    }
  }

  drawMotionTrack(ctx, w) {
    const trackY = RULER_HEIGHT + BLOCK_HEIGHT;
    const trackH = MOTION_TRACK_HEIGHT;

    ctx.fillStyle = "#100a0a";
    ctx.fillRect(0, trackY, w, trackH);

    ctx.fillStyle = "#666";
    ctx.font = "10px sans-serif";
    ctx.textBaseline = "top";
    ctx.fillText("IC-LoRA", 4, trackY + 2);

    const ppf = this.pxPerFrame();
    for (let i = 0; i < this.timeline.motionSegments.length; i++) {
      const seg = this.timeline.motionSegments[i];
      const x = seg.start * ppf;
      const segW = Math.max(2, seg.length * ppf);
      const isSelected = this._isSelected("motion", i);

      // 段背景使用轨道背景色（与空白区域一致，便于区分图像/视频边界）

      // 视频缩略图序列（优先于文件名文字）
      const drewThumb = this._drawVideoThumbnails(ctx, seg, x, trackY + 2, segW, trackH - 4);
      // 静态图像：缩略图未命中时尝试加载并绘制单张图像
      let drewImg = false;
      if (!drewThumb && seg.isStaticImage) {
        this._loadMotionImage(seg);
        const img = this._motionImageCache.get(seg.videoFile);
        if (img && img.complete && img.naturalWidth > 0) {
          this._drawImageCover(ctx, img, x, trackY + 2, segW, trackH - 4);
          drewImg = true;
        }
      }
      if (drewThumb || drewImg) {
        // 半透明遮罩保证文字可读
        ctx.fillStyle = "rgba(0,0,0,0.4)";
        ctx.fillRect(x, trackY + 2, segW, trackH - 4);
      }

      // 段边框
      ctx.strokeStyle = isSelected ? "#e8a850" : "#805030";
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(x + 0.5, trackY + 2.5, Math.max(1, segW - 1), trackH - 5);

      // 顶部 "IC-LoRA" 徽章
      ctx.fillStyle = "#9b6cd6";
      ctx.fillRect(x + 2, trackY + 2, 50, 12);
      ctx.fillStyle = "#fff";
      ctx.font = "9px sans-serif";
      ctx.textBaseline = "top";
      ctx.fillText("IC-LoRA", x + 5, trackY + 4);

      // 底部信息：显示段描述（截断）
      const descText = seg.description || "";
      if (descText && segW > 60) {
        ctx.fillStyle = "#ccc";
        ctx.font = "9px sans-serif";
        ctx.textBaseline = "top";
        ctx.fillText(this.truncateText(ctx, descText, segW - 8), x + 4, trackY + trackH - 14);
      }
    }
  }

  drawAudioTrack(ctx, w) {
    const trackY = RULER_HEIGHT + BLOCK_HEIGHT + MOTION_TRACK_HEIGHT;
    const trackH = AUDIO_TRACK_HEIGHT;

    ctx.fillStyle = "#0d0d0d";
    ctx.fillRect(0, trackY, w, trackH);

    ctx.fillStyle = "#666";
    ctx.font = "10px sans-serif";
    ctx.textBaseline = "top";
    ctx.fillText("Audio", 4, trackY + 2);

    const ppf = this.pxPerFrame();
    for (let i = 0; i < this.timeline.audioSegments.length; i++) {
      const seg = this.timeline.audioSegments[i];
      const x = seg.start * ppf;
      const segW = Math.max(2, seg.length * ppf);
      const isSelected = this._isSelected("audio", i);

      // 段背景使用轨道背景色（与空白区域一致，便于区分波形边界）
      ctx.strokeStyle = isSelected ? "#6fa0e8" : "#3a6080";
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(x + 0.5, trackY + 2.5, Math.max(1, segW - 1), trackH - 5);

      // 波形：优先使用 decodeAudioData 解码的完整波形，回退到 peaks
      const drewWave = this._drawAudioWaveform(ctx, seg, x, trackY + 2, segW, trackH - 4);
      if (!drewWave) {
        const peaksKey = seg.audioFile || seg.fileName;
        const peaks = this.audioPeaksCache.get(peaksKey);
        if (peaks && peaks.length > 0) {
          ctx.fillStyle = "#6fa0e8";
          const midY = trackY + trackH / 2;
          const barCount = Math.min(peaks.length, Math.max(1, Math.floor(segW / 2)));
          for (let j = 0; j < barCount; j++) {
            const peakIdx = Math.floor(j * peaks.length / barCount);
            const peak = Math.abs(peaks[peakIdx] || 0);
            const barH = Math.max(1, peak * (trackH - 8));
            const barX = x + (j / barCount) * segW;
            const barW = Math.max(1, segW / barCount - 1);
            ctx.fillRect(barX, midY - barH / 2, barW, barH);
          }
        } else {
          ctx.fillStyle = "#5577aa";
          ctx.font = "10px sans-serif";
          const label = seg.fileName || "audio";
          ctx.fillText(this.truncateText(ctx, label, segW - 8), x + 4, trackY + trackH / 2 - 5);
        }
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
    if (ctx.measureText(line2).width > maxWidth) line2 = this.truncateText(ctx, line2, maxWidth);
    return [line1.trimEnd(), line2];
  }

  // ── 文件上传（分片） ──

  showUploadProgress(msg) {
    this.uploadProgressEl.textContent = msg;
    this.uploadProgressEl.style.display = "block";
  }

  hideUploadProgress() {
    this.uploadProgressEl.style.display = "none";
  }

  async uploadFileChunked(file, onProgress) {
    const filename = file.name;
    const size = file.size;

    // 文件去重检查
    try {
      const checkUrl = `/yuan_clip_timeline_check_file?filename=${encodeURIComponent(filename)}&size=${size}`;
      const checkRes = await fetch(checkUrl);
      const checkData = await checkRes.json();
      if (checkData.exists && checkData.name) {
        return { name: checkData.name, audio_file: null, peaks: null };
      }
    } catch (_) {}

    // 分片上传
    const totalChunks = Math.max(1, Math.ceil(size / CHUNK_SIZE));
    let lastResult = null;
    for (let i = 0; i < totalChunks; i++) {
      const start = i * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, size);
      const chunk = file.slice(start, end);
      const formData = new FormData();
      formData.append("file", chunk, filename);
      formData.append("filename", filename);
      formData.append("chunk_index", i);
      formData.append("total_chunks", totalChunks);
      const res = await fetch("/yuan_clip_timeline_upload_chunk", { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Upload chunk ${i + 1}/${totalChunks} failed: ${res.status}`);
      const data = await res.json();
      if (onProgress) onProgress(i + 1, totalChunks);
      lastResult = data;
    }
    return lastResult;
  }

  async uploadImageForSegment(idx) {
    const seg = this.timeline.segments[idx];
    if (!seg) return;
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.addEventListener("change", async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        this.showUploadProgress("上传图像中…");
        const result = await this.uploadFileChunked(file, (done, total) => {
          this.showUploadProgress(`上传图像 ${done}/${total}…`);
        });
        if (result?.name) {
          seg.imageFile = result.name;
          seg.imageB64 = "";
          this.commitChanges();
          this._loadImageForSeg(idx);
          this.updateUIFromSelection();
          this.render();
        }
        this.hideUploadProgress();
      } catch (err) {
        this.showUploadProgress("上传失败: " + err.message);
        setTimeout(() => this.hideUploadProgress(), 3000);
      }
    });
    input.click();
  }

  async uploadAudioFile() {
    if (this.promptLocked) return;
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "audio/*,video/*";
    input.addEventListener("change", async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        this.showUploadProgress("上传音频中…");
        const max = this.getMaxFrames();

        // 读取音频时长计算帧数（完整长度）
        const desiredLen = await new Promise(resolve => {
          const reader = new FileReader();
          reader.onload = async () => {
            try {
              const arrayBuf = reader.result;
              const AC = window.AudioContext || window.webkitAudioContext;
              if (!AC) { resolve(Math.max(MIN_SEGMENT_LENGTH, Math.floor(max / 4))); return; }
              const ac = new AC();
              const audioBuf = await ac.decodeAudioData(arrayBuf);
              ac.close();
              const dur = audioBuf.duration || 1;
              resolve(Math.max(MIN_SEGMENT_LENGTH, Math.ceil(dur * this.getFps())));
            } catch (e) {
              resolve(Math.max(MIN_SEGMENT_LENGTH, Math.floor(max / 4)));
            }
          };
          reader.onerror = () => resolve(Math.max(MIN_SEGMENT_LENGTH, Math.floor(max / 4)));
          reader.readAsArrayBuffer(file);
        });

        // 物理碰撞分配位置（超总长度自动截取）
        const slot = this._findFreeSlot(this.timeline.audioSegments, desiredLen, max);
        const { start, length: segLen } = slot;

        // 后台上传文件
        const result = await this.uploadFileChunked(file, (done, total) => {
          this.showUploadProgress(`上传音频 ${done}/${total}…`);
        });
        if (!(result?.name || result?.audio_file)) { this.hideUploadProgress(); return; }
        const audioName = result.audio_file || result.name;
        // 获取音频波形
        let peaks = result.peaks || null;
        if (!peaks) {
          try {
            const audioRes = await fetch(`/yuan_clip_timeline_get_audio?filename=${encodeURIComponent(audioName)}`);
            const audioData = await audioRes.json();
            peaks = audioData.peaks || [];
          } catch (_) { peaks = []; }
        }
        this.timeline.audioSegments.push({
          audioFile: audioName,
          audioB64: "",
          start,
          length: segLen,
          trimStart: 0.0,
          fileName: file.name,
        });
        this.audioPeaksCache.set(audioName, peaks || []);
        this.selectedAudioIndices = new Set([this.timeline.audioSegments.length - 1]);
        this.commitChanges();
        this.updateUIFromSelection();
        this.render();
        this.hideUploadProgress();
      } catch (err) {
        this.showUploadProgress("上传失败: " + err.message);
        setTimeout(() => this.hideUploadProgress(), 3000);
      }
    });
    input.click();
  }

  async uploadMotionVideo() {
    if (this.promptLocked) return;
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*,image/*";
    input.addEventListener("change", async () => {
      const file = input.files?.[0];
      if (!file) return;
      const nameLower = file.name.toLowerCase();
      const isImage = file.type.startsWith("image/") || nameLower.match(/\.(jpg|jpeg|png|webp|gif|bmp)$/i);
      const isVideo = file.type.startsWith("video/") || nameLower.match(/\.(mp4|webm|mkv|avi|mov|m4v|flv|wmv)$/i);
      if (!isImage && !isVideo) return;
      try {
        this.showUploadProgress(isImage ? "上传图像中…" : "上传视频中…");
        const max = this.getMaxFrames();

        // 先确定段长度：图像默认16帧，视频取完整时长（帧数）
        let desiredLen;
        if (isImage) {
          desiredLen = 16;
        } else {
          // 读取视频时长计算帧数
          desiredLen = await new Promise(resolve => {
            const blobUrl = URL.createObjectURL(file);
            const vid = document.createElement("video");
            vid.preload = "metadata";
            vid.muted = true;
            vid.onloadedmetadata = () => {
              const dur = vid.duration || 1;
              URL.revokeObjectURL(blobUrl);
              resolve(Math.max(MIN_SEGMENT_LENGTH, Math.ceil(dur * this.getFps())));
            };
            vid.onerror = () => { URL.revokeObjectURL(blobUrl); resolve(Math.max(MIN_SEGMENT_LENGTH, Math.floor(max / 4))); };
            vid.src = blobUrl;
          });
        }

        // 物理碰撞分配位置（超总长度自动截取）
        const slot = this._findFreeSlot(this.timeline.motionSegments, desiredLen, max);
        const { start, length: segLen } = slot;

        // 后台上传文件
        this.showUploadProgress(isImage ? "上传图像中…" : "上传视频中…");
        const result = await this.uploadFileChunked(file, (done, total) => {
          this.showUploadProgress(`上传 ${done}/${total}…`);
        });
        if (!result?.name) { this.hideUploadProgress(); return; }

        const newSeg = {
          videoFile: result.name,
          start,
          length: segLen,
          trimStart: 0.0,
          isStaticImage: isImage,
          fileName: file.name,
          // 显式绑定 @图X 编号：取当前 motionSegments 中最大 subjectNum + 1，无则 = 1
          subjectNum: this.timeline.motionSegments.reduce((mx, s) => Math.max(mx, s.subjectNum || 0), 0) + 1,
        };
        this.timeline.motionSegments.push(newSeg);
        this.selectedMotionIndices = new Set([this.timeline.motionSegments.length - 1]);
        this.commitChanges();
        this.updateUIFromSelection();
        this.render();

        // 视频触发缩略图提取
        if (!isImage) {
          this._extractVideoThumbnails(newSeg).then(() => this.render());
        }
        this.hideUploadProgress();
      } catch (err) {
        this.showUploadProgress("上传失败: " + err.message);
        setTimeout(() => this.hideUploadProgress(), 3000);
      }
    });
    input.click();
  }

  async _loadAudioPeaks() {
    for (const seg of this.timeline.audioSegments) {
      const key = seg.audioFile || seg.fileName;
      if (key && !this.audioPeaksCache.has(key)) {
        try {
          const res = await fetch(`/yuan_clip_timeline_get_audio?filename=${encodeURIComponent(key)}`);
          const data = await res.json();
          if (data.peaks) this.audioPeaksCache.set(key, data.peaks);
        } catch (_) {}
      }
    }
    this.render();
  }

  destroy() {
    this.resizeObserver?.disconnect();
    if (this._textCommitTimer) {
      clearTimeout(this._textCommitTimer);
      this._textCommitTimer = null;
      try { this.commitChanges(); } catch (_) {}
    }
  }
}

// ============================================================================
// 扩展注册
// ============================================================================

app.registerExtension({
  name: "Yuan.Tool.CLIP timeline",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "YuanCLIPTimeline") return;

    // 节点创建：设置尺寸 + 创建编辑器（在 onNodeCreated 原型方法中，确保可靠触发）
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      if (!this.size || this.size[0] < 1375) {
        this.size = [1375, Math.max(this.size?.[1] || 0, 460)];
      }
      // 延迟创建编辑器，确保 widgets 已初始化
      const node = this;
      if (node._timelineEditorInit) return r;
      node._timelineEditorInit = true;

      setTimeout(() => {
        try {
          // 注入 CSS
          injectStyles();

          // 隐藏由编辑器管理的 widget
          for (const name of HIDDEN_WIDGET_NAMES) {
            hideWidget(node.widgets?.find(w => w.name === name));
          }

          // 创建容器 DOM
          const container = document.createElement("div");
          container.className = "yuan-clip-tl-container";
          // 固定高度，不使用自动调节
          const FIXED_HEIGHT = 460;
          node._timelineWidget = node.addDOMWidget("yuan_clip_timeline", "YuanCLIPTimeline", container, {
            serialize: false,
            hideOnZoom: false,
            getMinHeight: () => FIXED_HEIGHT,
            getHeight: () => FIXED_HEIGHT,
          });

          // ── V3 (Nodes 2.0) 尺寸适配：使 DOM 渲染尺寸与 V1 设计保持一致 ──
          (function adaptTimelineV3() {
            try {
              // 查找 V3 前端的 comfy-node 祖先元素（V1 前端不存在，直接跳过）
              let v3NodeEl = null;
              let el = container.parentElement;
              while (el) {
                if ((el.tagName && el.tagName.toLowerCase().includes('comfy-node')) ||
                    (el.classList && el.classList.contains('comfy-node'))) {
                  v3NodeEl = el;
                  break;
                }
                el = el.parentElement || (el.getRootNode ? el.getRootNode().host : null);
              }
              if (!v3NodeEl) return;

              // 移除被编辑器隐藏管理参数的空端口占位（避免节点被拉长）
              node._yuanIsV3 = true;
              hideManagedWidgetPorts(node);

              // 与 V1 一致的最小尺寸：宽=画布520+容器内边距，高=固定设计高度
              const MIN_W = 540;
              const MIN_H = FIXED_HEIGHT;

              node.min_size = [MIN_W, MIN_H];
              if (node.size[0] < MIN_W) node.size[0] = MIN_W;
              if (node.size[1] < MIN_H) node.size[1] = MIN_H;

              // 在 V3 DOM 元素上强制最小尺寸
              v3NodeEl.style.removeProperty("min-width");
              v3NodeEl.style.setProperty("min-width", MIN_W + "px", "important");
              v3NodeEl.style.setProperty("min-height", MIN_H + "px", "important");

              // 容器保底高度，与 V1 的固定 widget 高度一致
              container.style.minHeight = MIN_H + "px";

              // V3 专用：告知布局系统该 DOM widget 的最小尺寸
              const widget = node._timelineWidget;
              if (widget) {
                const prev = typeof widget.computeLayoutSize === "function"
                  ? widget.computeLayoutSize.bind(widget) : null;
                widget.computeLayoutSize = (targetNode) => {
                  const p = prev ? (prev(targetNode) || {}) : {};
                  return {
                    ...p,
                    minWidth: Math.max(MIN_W, Number(p.minWidth || 0)),
                    minHeight: Math.max(MIN_H, Number(p.minHeight || 0)),
                  };
                };
              }

              // 用户缩放节点时保持最小尺寸
              const origSetSize = node.setSize;
              node.setSize = function (size) {
                if (Array.isArray(size)) {
                  size[0] = Math.max(size[0], MIN_W);
                  size[1] = Math.max(size[1], MIN_H);
                }
                if (origSetSize) origSetSize.call(this, size); else this.size = size;
              };
            } catch (_) {}
          })();

          // 实例化编辑器
          node._timelineEditor = new TimelineEditor(node, container);

          // 节点创建时若 segment_images / motion_images / 音频输入 端口已连接，延迟尝试实时加载（覆盖加载工作流场景）
          setTimeout(() => {
            if (node._timelineEditor?.hasSegmentImagesConnected()) {
              node._timelineEditor._applyUpstreamSegmentImages();
            }
            if (node._timelineEditor?.hasMotionImagesConnected()) {
              node._timelineEditor._applyUpstreamMotionImages();
            }
            if (node._timelineEditor?.hasAudioInputConnected()) {
              node._timelineEditor._applyUpstreamAudioInput();
            }
          }, 300);

          // 监听执行完成：上游节点执行后实时刷新 segment_images / motion_images 图像
          const comfyApi = app.api || window.comfyAPI?.api?.api || window.api || window.comfyAPI?.api;
          if (comfyApi?.addEventListener) {
            // executed 事件：上游节点或自身节点执行完成时触发实时加载
            const onExecuted = (e) => {
              const evNodeId = e?.detail?.node_id;
              if (evNodeId == null) return;
              const editor = node._timelineEditor;
              if (!editor) return;
              // 自身节点执行完成 → 后端已处理，通过 send_sync 更新
              // 上游节点执行完成 → 实时从 nodeOutputs 获取图像/音频
              const upstreamSegId = editor._getUpstreamNodeId?.();
              const upstreamMotionId = editor._getMotionUpstreamNodeId?.();
              const upstreamAudioId = editor._getUpstreamAudioInputId?.();
              if (String(evNodeId) === String(node.id)) {
                setTimeout(() => {
                  editor._applyUpstreamSegmentImages();
                  editor._applyUpstreamMotionImages();
                  editor._applyUpstreamAudioInput();
                }, 100);
              } else if (upstreamSegId && String(evNodeId) === upstreamSegId) {
                setTimeout(() => editor._applyUpstreamSegmentImages(), 100);
              } else if (upstreamMotionId && String(evNodeId) === upstreamMotionId) {
                setTimeout(() => editor._applyUpstreamMotionImages(), 100);
              } else if (upstreamAudioId && String(evNodeId) === upstreamAudioId) {
                setTimeout(() => editor._applyUpstreamAudioInput(), 100);
              }
            };
            comfyApi.addEventListener("executed", onExecuted);
            node._segImgExecutedHandler = onExecuted;

            // 自定义 WebSocket 事件（后端 send_sync：执行时后端保存 segimg 后通知）
            const onSegImagesUpdated = (e) => {
              if (!node._timelineEditor?.hasSegmentImagesConnected()) return;
              setTimeout(() => node._timelineEditor?._applyUpstreamSegmentImages(), 100);
            };
            comfyApi.addEventListener("yuan_clip_seg_images_updated", onSegImagesUpdated);
            node._segImagesUpdatedHandler = onSegImagesUpdated;

            // motion_images 的 WebSocket 事件
            const onMotionImagesUpdated = (e) => {
              if (!node._timelineEditor?.hasMotionImagesConnected()) return;
              setTimeout(() => node._timelineEditor?._applyUpstreamMotionImages(), 100);
            };
            comfyApi.addEventListener("yuan_clip_motion_images_updated", onMotionImagesUpdated);
            node._motionImagesUpdatedHandler = onMotionImagesUpdated;

            // 音频输入端口的 WebSocket 事件（后端 send_sync：保存音频文件后通知）
            const onAudioInputUpdated = (e) => {
              const detail = e?.detail || e;
              setTimeout(() => node._timelineEditor?._applyUpstreamAudioInput(detail), 100);
            };
            comfyApi.addEventListener("yuan_clip_audio_input_updated", onAudioInputUpdated);
            node._audioInputUpdatedHandler = onAudioInputUpdated;
          }
        } catch (err) {
          // 时间轴编辑器初始化失败时静默处理
        }
      }, 0);

      return r;
    };

    // 工作流恢复
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const out = onConfigure?.apply(this, arguments);
      for (const [name, def] of APPENDED_WIDGET_DEFAULTS) {
        const w = this.widgets?.find(x => x.name === name);
        if (w && (w.value == null || w.value === "")) w.value = def;
      }
      // 工作流恢复后重新隐藏由编辑器管理的 widget（防止 onConfigure 后 widget 重新显示）
      for (const name of HIDDEN_WIDGET_NAMES) {
        hideWidget(this.widgets?.find(w => w.name === name));
      }
      // V3 下再次移除被隐藏管理参数的空端口占位（加载工作流后端口可能重建）
      hideManagedWidgetPorts(this);
      setTimeout(() => {
        if (this._timelineEditor) {
          this._timelineEditor._updateLockState();
          this._timelineEditor._restoreFromWidgets();
          this._timelineEditor.updateUIFromSelection();
          this._timelineEditor.render();
        }
      }, 10);
      return out;
    };

    // 节点删除清理
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this._timelineEditor?.destroy();
      // 清理事件监听
      const comfyApi = app.api || window.comfyAPI?.api?.api || window.api || window.comfyAPI?.api;
      if (this._segImgExecutedHandler) {
        comfyApi?.removeEventListener?.("executed", this._segImgExecutedHandler);
        this._segImgExecutedHandler = null;
      }
      if (this._segImagesUpdatedHandler) {
        comfyApi?.removeEventListener?.("yuan_clip_seg_images_updated", this._segImagesUpdatedHandler);
        this._segImagesUpdatedHandler = null;
      }
      if (this._motionImagesUpdatedHandler) {
        comfyApi?.removeEventListener?.("yuan_clip_motion_images_updated", this._motionImagesUpdatedHandler);
        this._motionImagesUpdatedHandler = null;
      }
      if (this._audioInputUpdatedHandler) {
        comfyApi?.removeEventListener?.("yuan_clip_audio_input_updated", this._audioInputUpdatedHandler);
        this._audioInputUpdatedHandler = null;
      }
      return onRemoved?.apply(this, arguments);
    };

    // 连接变化同步
    const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, slot, isConnect, linkInfo, ioSlot) {
      if (origOnConnectionsChange) origOnConnectionsChange.apply(this, arguments);
      if (ioSlot?.name === "文本输入") {
        setTimeout(() => {
          if (this._timelineEditor && this._timelineEditor.promptLocked) {
            this._timelineEditor.syncFromTextInput();
          }
        }, isConnect ? 150 : 50);
      }
      // 全局提示词 连接时，同步上游文本节点内容
      if (ioSlot?.name === "全局提示词") {
        setTimeout(() => {
          if (this._timelineEditor) {
            this._timelineEditor._syncGlobalPromptFromUpstream();
          }
        }, isConnect ? 150 : 50);
      }
      // 段落图像 连接时，实时读取上游节点图像（类似 文本输入 实时映射）
      if (ioSlot?.name === "段落图像" && isConnect) {
        setTimeout(() => this._timelineEditor?._applyUpstreamSegmentImages(), 100);
      }
      // 运动图像 连接时，实时读取上游节点图像分配到 IC-LoRA 轨道
      if (ioSlot?.name === "运动图像" && isConnect) {
        setTimeout(() => this._timelineEditor?._applyUpstreamMotionImages(), 100);
      }
      // 音频输入 连接时，主动从上游读取音频并添加到音频轨道（参考运动图像）
      if (ioSlot?.name === "音频输入" && isConnect) {
        setTimeout(() => this._timelineEditor?._applyUpstreamAudioInput(), 100);
      }
      // ref_images 连接时无需特殊处理，@ 弹窗会自动从端口获取缩略图
    };
  },
});

// ============================================================================
// @ 标记自动补全扩展（复刻自 ComfyUI-Yuan-Tool 注入了@图像和视觉模型）
// 在 global_prompt / text_input / 时间轴编辑器中输入 @ 时弹出角色选择菜单
// ============================================================================
(function() {
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
            }
            // 策略2: 只有一个 Timeline 节点时直接使用
            if (nodes.length === 1) return nodes[0];
        } catch (e) {
            // 静默处理
        }
        return null;
    }

    // ── 从 global_prompt 解析 @标记和描述 ──
    // 支持：@图1=描述  /  @图1:描述  /  @图1：描述
    function parseMarkersFromPrompt(promptText) {
        if (!promptText) return [];
        const markers = [];
        // 与后端 _MSR_CHAR_PATTERN 严格对齐：仅识别 @图数字=描述，避免前端识别 @角色A=xxx
        // 而后端不替换导致的字面量泄漏。按真实换行解析，不再把逗号转成换行（修复多逗号描述被截断）。
        const lines = promptText.split('\n');
        for (const line of lines) {
            const trimmed = line.trim();
            const m = trimmed.match(/^@图(\d+)\s*[=：:]\s*(.+)/);
            if (m) {
                const name = '@图' + m[1];
                const desc = m[2].trim();
                if (!markers.find(x => x.name === name)) {
                    markers.push({ name, desc, index: parseInt(m[1], 10) - 1 });
                }
            }
        }
        return markers;
    }

    // ── 构建图像 view URL（与 _buildImageUrl 逻辑一致）──
    function buildImageViewUrl(file) {
        if (!file) return null;
        let type = 'input';
        let path = file.replace(/\\/g, '/');
        const colonIdx = path.indexOf(':');
        if (colonIdx > 0 && colonIdx < 10) {
            const prefix = path.slice(0, colonIdx);
            if (prefix === 'output' || prefix === 'input' || prefix === 'temp') {
                type = prefix;
                path = path.slice(colonIdx + 1);
            }
        }
        const parts = path.split('/');
        const filename = parts.pop();
        const subfolder = parts.join('/');
        let url = `/view?filename=${encodeURIComponent(filename)}&type=${type}`;
        if (subfolder) url += `&subfolder=${encodeURIComponent(subfolder)}`;
        return url;
    }

    // ── 从 motion_images 输入端 / TimelineEditor 缓存获取缩略图 ──
    function getRefThumbnails(node) {
        const editor = node._timelineEditor;

        // 策略1: 从 TimelineEditor 的 motionSegments 缓存获取（WebSocket 加载的图像）
        if (editor && editor.timeline && Array.isArray(editor.timeline.motionSegments) && editor.timeline.motionSegments.length > 0) {
            const segs = editor.timeline.motionSegments;
            const results = [];
            for (let i = 0; i < segs.length; i++) {
                const seg = segs[i];
                // 静态图像：从 _motionImageCache 获取 Image 对象
                if (seg.isStaticImage && seg.videoFile) {
                    const cachedImg = editor._motionImageCache?.get(seg.videoFile);
                    if (cachedImg && cachedImg.complete && cachedImg.naturalWidth > 0) {
                        results.push({ src: cachedImg.src, index: i });
                    } else {
                        // 缓存未命中时用 view URL，触发异步加载
                        const url = buildImageViewUrl(seg.videoFile);
                        if (url) results.push({ src: url, index: i });
                    }
                }
                // 视频文件或 frameFiles：用第一帧
                else if (seg.videoFile) {
                    const url = buildImageViewUrl(seg.videoFile);
                    if (url) results.push({ src: url, index: i });
                }
                else if (seg.frameFiles && Array.isArray(seg.frameFiles) && seg.frameFiles.length > 0) {
                    const url = buildImageViewUrl(seg.frameFiles[0]);
                    if (url) results.push({ src: url, index: i });
                }
            }
            if (results.length > 0) return results;
        }

        // 策略1.5: 从 TimelineEditor 的主轨段引导图像获取（segment_images 上传的图像）
        if (editor && editor.timeline && Array.isArray(editor.timeline.segments) && editor.timeline.segments.length > 0) {
            const segs = editor.timeline.segments;
            const results = [];
            for (let i = 0; i < segs.length; i++) {
                const seg = segs[i];
                if (seg.imageFile) {
                    // 从 _imageCache 获取已加载的 Image 对象
                    const cachedImg = editor._imageCache?.get(i);
                    if (cachedImg && cachedImg.complete && cachedImg.naturalWidth > 0) {
                        results.push({ src: cachedImg.src, index: i });
                    } else {
                        // 缓存未命中时用 view URL
                        const url = buildImageViewUrl(seg.imageFile);
                        if (url) results.push({ src: url, index: i });
                    }
                } else if (seg.imageB64) {
                    results.push({ src: seg.imageB64, index: i });
                }
            }
            if (results.length > 0) return results;
        }

        // 策略2: 从 LiteGraph 链路追踪上游源节点
        const graph = getGraph();
        if (!graph || !node?.inputs) return [];
        
        let inputIdx = node.inputs.findIndex(inp => inp?.name === '运动图像');
        if (inputIdx < 0) {
            inputIdx = node.inputs.findIndex(inp => inp?.name === 'ref_images');
        }
        if (inputIdx < 0) return [];
        
        const linkId = node.inputs[inputIdx]?.link;
        if (linkId == null || linkId === -1) return [];

        let linkData;
        try { linkData = graph.links[linkId]; } catch(e) {}
        if (!linkData) { try { linkData = graph._links?.get(linkId); } catch(e2) {} }
        if (!linkData) return [];
        const srcNode = graph._nodes?.find(n => n?.id === linkData.origin_id) ||
                        graph.getNodeById?.(linkData.origin_id);
        if (!srcNode) return [];

        // srcNode.imgs（节点已渲染的缩略图）
        if (srcNode.imgs?.length) {
            return srcNode.imgs.map((img, i) => {
                let src = null;
                if (typeof img === 'string') src = img;
                else if (img instanceof HTMLImageElement) src = img.currentSrc || img.src;
                else if (img?.src) src = img.src;
                else if (img?.currentSrc) src = img.currentSrc;
                else if (img?.url) src = img.url;
                return { src, index: i };
            });
        }

        // LoadImage / MultiImageLoader 节点
        if (srcNode.type === 'LoadImage' || srcNode.type?.includes('LoadImage') || srcNode.type?.includes('MultiImage')) {
            const imgW = srcNode.widgets?.find(w => w.name === 'image') || 
                         srcNode.widgets?.find(w => w.name === 'file') ||
                         srcNode.widgets?.find(w => w.name === 'image_paths');
            if (imgW?.value) {
                const fname = typeof imgW.value === 'string' ? imgW.value.trim() : (imgW.value.filename || imgW.value[0] || '');
                if (fname) {
                    return [{ src: `/view?filename=${encodeURIComponent(fname)}&type=input&subfolder=&t=${Date.now()}`, index: 0 }];
                }
            }
        }

        return [];
    }

    // ── 绘制缩略图到 canvas ──
    function drawThumbnail(canvas, src, fallbackLabel) {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        if (src) {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => { try { ctx.drawImage(img, 0, 0, canvas.width, canvas.height); } catch(e) {} };
            img.onerror = () => {
                ctx.fillStyle = '#555';
                ctx.font = '10px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(fallbackLabel, canvas.width/2, canvas.height/2 + 4);
            };
            img.src = src;
        } else {
            ctx.fillStyle = '#555';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(fallbackLabel, canvas.width/2, canvas.height/2 + 4);
        }
    }

    // ── 从连接的源节点读取 全局提示词 文本 ──
    function readConnectedGlobalPrompt(node) {
        if (!node || !node.graph) return null;
        const inputIdx = node.findInputSlot ? node.findInputSlot("全局提示词") : -1;
        if (inputIdx < 0) return null;
        const linkId = node.inputs[inputIdx]?.link;
        if (linkId == null) return null;

        const link = node.graph.links[linkId];
        if (!link) return null;
        const srcNode = node.graph.getNodeById(link.origin_id);
        if (!srcNode) return null;

        const candidates = [];
        for (const w of (srcNode.widgets || [])) {
            if (w.type === "customtext" || w.name === "string" || (w.name && /^(text|prompt|string|multiline|global_prompt)$/i.test(w.name))) {
                candidates.push(w);
            }
        }
        if (candidates.length === 0) {
            for (const w of (srcNode.widgets || [])) {
                if (!w.hidden && typeof w.value === "string" && w.value.trim()) {
                    candidates.push(w);
                }
            }
        }
        if (candidates.length === 0) return null;

        const stringW = candidates.find(w => w.name === "string");
        if (stringW && stringW.value?.trim()) return stringW.value;

        const combined = candidates.map(w => w.value).filter(v => typeof v === "string" && v.trim()).join("\n");
        return combined || null;
    }

    // ── 构建并显示弹窗 ──
    function showPopup(textEl, node) {
        const existing = document.querySelector('.timeline-atsign-popup');
        if (existing) existing.remove();

        const gpWidget = node.widgets?.find(w => w.name === '全局提示词');
        // 优先读取 widget 值；外接连接时 widget 被隐藏，回退到从连接的源节点读取
        let promptText = gpWidget?.value || '';
        if (!promptText.trim()) {
            const connected = readConnectedGlobalPrompt(node);
            if (connected) promptText = connected;
        }
        let markers = parseMarkersFromPrompt(promptText);

        // markers 为空时，从已上传的运动图像缩略图自动生成 @图X 标记
        // 这样用户即使没写 @图X=描述 行，输入 @ 也能看到已上传的图像供选择
        if (markers.length === 0) {
            const thumbs = getRefThumbnails(node);
            if (thumbs.length === 0) return;
            markers = thumbs.map((t, i) => ({
                name: '@图' + (i + 1),
                desc: '',
                index: i,
                auto: true
            }));
        }


        // 获取缩略图，按 @图X 数字匹配
        const thumbs = getRefThumbnails(node);
        const thumbMap = {};
        for (const t of thumbs) {
            thumbMap[`@图${t.index + 1}`] = t;
        }
        const hasThumbs = Object.keys(thumbMap).length > 0;

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
                padding: 6px 10px;
                border-radius: 6px;
                transition: background 0.15s;
                background: #1e1e1e;
                border: 1px solid #444;
                min-width: 64px;
                min-height: 40px;
            `;
            item.onmouseenter = () => { item.style.background = '#444'; item.style.borderColor = '#777'; };
            item.onmouseleave = () => { item.style.background = '#1e1e1e'; item.style.borderColor = '#444'; };

            // ── 缩略图（如果 motion_images / ref_images 已连接）──
            const t = thumbMap[marker.name];
            if (hasThumbs) {
                const canvas = document.createElement('canvas');
                canvas.width = 64;
                canvas.height = 64;
                canvas.style.cssText = `
                    width: 64px;
                    height: 64px;
                    border-radius: 4px;
                    border: 1px solid #555;
                    background: #1a1a1a;
                    display: block;
                    margin-bottom: 4px;
                `;
                drawThumbnail(canvas, t?.src || null, marker.name);
                item.appendChild(canvas);
            }

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
            // 全局提示词是定义 @图X=描述 的地方，不弹出 @ 选择菜单
            // 只有 IC-LoRA 轨段的描述 textarea（引用 @ 的地方）才弹窗
            const gpWidget = node.widgets?.find(w => w.name === '全局提示词');
            if (gpWidget && (gpWidget.inputEl === textEl || gpWidget.element === textEl)) return;
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
        scanTextareas();

        // 新 textarea 出现时由 MutationObserver 响应式触发，不再使用 setInterval 轮询
        if (window.MutationObserver) {
            const observer = new MutationObserver(() => {
                const all = document.querySelectorAll('textarea');
                const inited = document.querySelectorAll('textarea[data-timeline-at-inited]');
                if (inited.length < all.length) scanTextareas();
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }
    }

    // ── 等待 ComfyUI 就绪 ──
    function waitForReady(retries) {
        if (retries <= 0) {
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
