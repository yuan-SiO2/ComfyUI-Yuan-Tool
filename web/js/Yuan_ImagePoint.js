/**
 * Yuan Tool · 图像点处理 前端
 * 复刻自 ComfyUI-YuanEditor 的 YuanEditor（原版为 ComfyUI-Easy-Sam3 的 FramesEditor）。
 * 为 Yuan_ImagePoint 节点提供内嵌画布：点标注模式（左键正面点/右键负面点）、
 * 框标注模式（拖拽边界框）、多帧图像底部滑块逐帧切换、撤销/重做/清空。
 * 辅助函数均为模块作用域，不会与原版产生全局冲突。
 */
import { getApi, findComfyNodeEl, enforceV3MinSize } from "./Yuan_Common.js";

const { app } = window.comfyAPI.app;

const api = getApi();

const getRealURL = obj => {
    const url = `/view?filename=${encodeURIComponent(obj.filename)}&type=${obj.type}&subfolder=${obj.subfolder}&rand=${Math.random()}`
    return api ? api.apiURL(url) : url
}
const makeUUID = _ =>{
  let dt = new Date().getTime()
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = ((dt + Math.random() * 16) % 16) | 0
    dt = Math.floor(dt / 16)
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
  return uuid
}
const chainCallback = (object, property, callback) => {
  if (object == undefined) {
    // 理论上不应发生
    console.error("Tried to add callback to non-existant object")
    return;
  }
  if (property in object) {
    const callback_orig = object[property]
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r
    };
  } else {
    object[property] = callback;
  }
}


app.registerExtension({
    name: "Comfy.Yuan.ImagePoint",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const nodeName = nodeData.name;
        if (nodeName === "Yuan_ImagePoint"){
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; height: 100%; background: #0f1011; overflow: hidden; box-sizing: border-box;border-radius:4px; margin: 0; padding: 0; display: flex; flex-direction: column;";

                // 工具栏
                const toolbar = document.createElement("div");
                toolbar.style.cssText = "flex: 0 0 32px; width: 100%; background: #222; display: flex; align-items: center; justify-content: space-between; padding: 0 4px; box-sizing: border-box; border-bottom: 1px solid #333; z-index: 10;";

                // 左侧（撤销/重做/清空）
                const leftGroup = document.createElement("div");
                leftGroup.style.display = "flex";
                leftGroup.style.gap = "4px";

                const createBtn = (iconSvg, title, onClick, isActive = false) => {
                    const btn = document.createElement("div");
                    btn.style.cssText = `width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; cursor: pointer; border-radius: 4px; color: ${isActive ? '#fff' : '#ccc'}; background-color: ${isActive ? '#444' : 'transparent'};`;
                    btn.innerHTML = iconSvg;
                    btn.title = title;
                    btn.onmouseover = () => { if (!btn.classList.contains("active")) btn.style.backgroundColor = "#333"; };
                    btn.onmouseout = () => { if (!btn.classList.contains("active")) btn.style.backgroundColor = "transparent"; };
                    btn.onclick = onClick;
                    return btn;
                };

                const undoIcon = `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z"/></svg>`;
                const redoIcon = `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M18.4 10.6C16.55 9 14.15 8 11.5 8c-4.65 0-8.58 3.03-9.96 7.22L3.9 16c1.05-3.19 4.05-5.5 7.6-5.5 1.95 0 3.73.72 5.12 1.88L13 16h9V7l-3.6 3.6z"/></svg>`;
                const resetIcon = `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>`;
                const pointIcon = `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>`;
                const boxIcon = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2" stroke-dasharray="4 4"/></svg>`;

                const undoBtn = createBtn(undoIcon, "撤销", () => this.undo());
                const redoBtn = createBtn(redoIcon, "重做", () => this.redo());
                const resetBtn = createBtn(resetIcon, "清空全部", () => {
                    const { positivePoints, negativePoints, bboxes } = this.canvasWidget;
                    if (positivePoints.length === 0 && negativePoints.length === 0 && bboxes.length === 0) return;
                    this.canvasWidget.positivePoints = [];
                    this.canvasWidget.negativePoints = [];
                    this.canvasWidget.bboxes = [];
                    this.canvasWidget.selectedBox = null;
                    this.canvasWidget.history = [];
                    this.canvasWidget.historyIndex = -1;
                    this.redrawCanvas();
                    this.updateUndoRedoUI();
                    this.updateWidgetValue();
                });

                leftGroup.appendChild(undoBtn);
                leftGroup.appendChild(redoBtn);
                leftGroup.appendChild(resetBtn);

                // 右侧（标注模式）
                const rightGroup = document.createElement("div");
                rightGroup.style.display = "flex";
                rightGroup.style.gap = "4px";

                let pointBtn, boxBtn;

                const setMode = (mode) => {
                    this.canvasWidget.mode = mode;
                    pointBtn.style.backgroundColor = mode === 'point' ? '#444' : 'transparent';
                    pointBtn.classList.toggle("active", mode === 'point');
                    pointBtn.style.color = mode === 'point' ? '#fff' : '#ccc';

                    boxBtn.style.backgroundColor = mode === 'box' ? '#444' : 'transparent';
                    boxBtn.classList.toggle("active", mode === 'box');
                    boxBtn.style.color = mode === 'box' ? '#fff' : '#ccc';
                };

                pointBtn = createBtn(pointIcon, "点标注模式", () => setMode('point'), true);
                pointBtn.classList.add("active");
                boxBtn = createBtn(boxIcon, "框标注模式", () => setMode('box'), false);

                rightGroup.appendChild(pointBtn);
                rightGroup.appendChild(boxBtn);

                toolbar.appendChild(leftGroup);
                toolbar.appendChild(rightGroup);
                container.appendChild(toolbar);

                // 画布容器
                const canvasWrapper = document.createElement("div");
                canvasWrapper.style.cssText = "flex: 1; width: 100%; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; background: #0f1011;";
                container.appendChild(canvasWrapper);

                // 图像与标注画布
                const canvas = document.createElement("canvas");
                canvas.width = 512;
                canvas.height = 512;
                // 使用 max-width/max-height 而非 100% 宽高，防止溢出
                canvas.style.cssText = "display: block; max-width: 100%; max-height: 100%; object-fit: contain; cursor: crosshair; margin: 0 auto;";
                canvasWrapper.appendChild(canvas);

                const ctx = canvas.getContext("2d");

                // 多帧切换条 UI
                const tracker = document.createElement("div");
                tracker.style.cssText = "flex: 0 0 32px; width: 100%; background: #222; display: none; align-items: center; justify-content: space-between; padding: 0 8px; box-sizing: border-box; border-top: 1px solid #333; gap: 2px;";

                const frameInfo = document.createElement("div");
                frameInfo.style.cssText = "color: #ccc; font-family: monospace; font-size: 12px; min-width: 40px; text-align: center; user-select: none;";
                frameInfo.innerText = "0/0";

                const slider = document.createElement("input");
                slider.type = "range";
                slider.min = "0";
                slider.max = "0";
                slider.value = "0";
                slider.step = "1";
                slider.style.cssText = "flex: 1; height: 4px; cursor: pointer;";

                tracker.appendChild(frameInfo);
                tracker.appendChild(slider);
                container.appendChild(tracker);

                slider.addEventListener("input", (e) => {
                    const frameIndex = parseInt(e.target.value);
                    this.canvasWidget.frameIndex = frameIndex;
                    this.canvasWidget.frameInfo.innerText = `${frameIndex + 1}/${this.canvasWidget.previewFrames.length}`;
                    this.updateWidgetValue();

                    const img = new Image();
                    img.onload = () => {
                        this.canvasWidget.image = img;
                        canvas.width = img.width;
                        canvas.height = img.height;
                        this.redrawCanvas();
                    };
                    img.src = getRealURL(this.canvasWidget.previewFrames[frameIndex]);
                });

                // 保存编辑器状态
                this.canvasWidget = {
                    canvas: canvas,
                    ctx: ctx,
                    container: container,
                    tracker: tracker,
                    slider: slider,
                    frameInfo: frameInfo,
                    image: null,
                    positivePoints: [],
                    negativePoints: [],
                    bboxes: [],
                    hoverBBox: null,
                    hoveredPoint: null,
                    mode: 'point', // 'point' | 'box'
                    history: [],
                    historyIndex: -1,
                    isDrawingBox: false,
                    currentBox: null,
                    selectedBox: null,   // 当前选中的边界框索引（用于移动/缩放）
                    dragMode: null,      // 拖拽类型：null | 'move' | 'resize-tl/tr/bl/br'
                    dragStart: null,     // 拖拽起始信息 {x, y, origBox}
                    frameIndex: 0,
                    previewFrames: [],
                };

                // 作为 DOM widget 挂载
                const widget = this.addDOMWidget("canvas", "points_editor", container, );
                this.uuid = makeUUID();
                // 保存 widget 引用供更新
                this.canvasWidget.domWidget = widget;

                // 获取 info widget
                const infoWidget = this.widgets.find(w=> w.name == 'info')
                setTimeout(_=>{
                    if(infoWidget && infoWidget.value) {
                        try {
                            const info = JSON.parse(infoWidget.value);
                            // 恢复正面/负面点
                            if (Array.isArray(info.positive_coords)) {
                                this.canvasWidget.positivePoints = info.positive_coords;
                            }
                            if (Array.isArray(info.negative_coords)) {
                                this.canvasWidget.negativePoints = info.negative_coords;
                            }
                            // 恢复边界框
                            if (Array.isArray(info.bbox)) {
                                this.canvasWidget.bboxes = info.bbox;
                            }
                            // 恢复帧索引与滑块
                            if (typeof info.frame_index === 'number' && this.canvasWidget.slider) {
                                this.canvasWidget.frameIndex = info.frame_index;
                                this.canvasWidget.slider.value = info.frame_index;
                                this.canvasWidget.frameInfo.innerText = `${info.frame_index + 1}/${this.canvasWidget.previewFrames.length}`;
                            }
                            this.redrawCanvas();
                        } catch (e) {
                            // 忽略解析错误
                        }
                    }
                },1)

                // V3 (Nodes 2.0) 尺寸适配：告知布局系统 DOM widget 行的最小尺寸
                if (typeof widget.computeLayoutSize === "function") {
                    const prevCLS = widget.computeLayoutSize.bind(widget);
                    widget.computeLayoutSize = (targetNode) => {
                        const p = prevCLS(targetNode) || {};
                        const w = Math.max(400, this.size ? this.size[0] : 400);
                        const h = Math.max(245, (this.size ? this.size[1] : 500) - 150);
                        return { ...p, minWidth: Math.max(w, Number(p.minWidth || 0)), minHeight: Math.max(h, Number(p.minHeight || 0)) };
                    };
                }

                // 使 widget 随节点尺寸变化 - 覆盖 computeSize
                widget.computeSize = (width) => {
                    // 返回固定最小高度，避免无限增长循环；实际高度由 onResize 处理
                    const nodeHeight = this.size ? this.size[1] : 500;
                    const widgetHeight = Math.max(245, nodeHeight - 150);
                    return [width, widgetHeight];
                };

                // V3 (Nodes 2.0) 尺寸适配：在 comfy-node 元素上同步节点最小尺寸
                const applyV3MinSize = () => {
                    try {
                        const el = findComfyNodeEl(container.parentElement);
                        if (!el || !this.size) return;
                        const minW = Math.max(400, this.size[0] || 400);
                        const minH = Math.max(400, this.size[1] || 500);
                        this.min_size = [minW, minH];
                        enforceV3MinSize(el, minW, minH);
                    } catch (_) {}
                };

                // 节点尺寸变化时动态更新容器高度
                chainCallback(this, "onResize", function(size) {
                    // 容器高度与 widget 一致（偏移 150 覆盖标题与边距）
                    const containerHeight = Math.max(245, size[1] - 150);
                    container.style.height = containerHeight + "px";
                    applyV3MinSize();
                });

                // draw 时同步处理尺寸变化
                chainCallback(this, "onDrawForeground", function(ctx) {
                    const containerHeight = Math.max(245, this.size[1] - 150);
                    if (container.style.height !== containerHeight + "px") {
                        container.style.height = containerHeight + "px";
                    }
                    applyV3MinSize();
                });

                // 处理图像输入变化
                chainCallback(this, "onExecuted", function(message) {
                    if (message.preview && message.preview[0]) {
                        const {preview_str, is_init} = message.preview[0];
                        const previewData = JSON.parse(preview_str);
                        this.canvasWidget.previewFrames = previewData;
                        // 更新多帧切换条
                        if(is_init){
                            if(this.canvasWidget.frameIndex>=previewData.length-1){
                                this.canvasWidget.frameIndex = 0;
                                this.restoreState({ positivePoints: [], negativePoints: [], bboxes: [] });
                                this.updateWidgetValue();
                                this.canvasWidget.history = [];
                                this.canvasWidget.historyIndex = -1;
                                this.updateUndoRedoUI();
                            }
                        }
                        if (previewData.length > 1) {
                            this.canvasWidget.tracker.style.display = "flex";
                            slider.max = previewData.length - 1;
                            slider.value = this.canvasWidget.frameIndex;
                            this.canvasWidget.slider = slider;
                            this.canvasWidget.frameInfo.innerText = `${this.canvasWidget.frameIndex + 1}/${previewData.length}`;
                        } else {
                            this.canvasWidget.tracker.style.display = "none";
                        }

                        const img = new Image();
                        img.onload = () => {
                            this.canvasWidget.image = img;
                            canvas.width = img.width;
                            canvas.height = img.height;
                            this.redrawCanvas();
                        };

                        if(previewData?.length>0){
                            if (this.canvasWidget.frameIndex >= previewData.length) {
                                this.canvasWidget.frameIndex = 0;
                                slider.value = 0;
                                this.canvasWidget.slider = slider;
                            }
                            img.src = getRealURL(previewData[this.canvasWidget.frameIndex]);
                        }

                    }
                });

                // 历史管理
                this.addToHistory = () => {
                    const { positivePoints, negativePoints, bboxes, history, historyIndex } = this.canvasWidget;
                    // 若处于历史中间，清除未来的历史
                    if (historyIndex < history.length - 1) {
                        this.canvasWidget.history = history.slice(0, historyIndex + 1);
                    }
                    // 压入新状态
                    const state = {
                        positivePoints: JSON.parse(JSON.stringify(positivePoints)),
                        negativePoints: JSON.parse(JSON.stringify(negativePoints)),
                        bboxes: JSON.parse(JSON.stringify(bboxes))
                    };
                    this.canvasWidget.history.push(state);
                    this.canvasWidget.historyIndex++;
                    this.updateUndoRedoUI();
                    this.updateWidgetValue();
                };

                this.undo = () => {
                    const { history, historyIndex } = this.canvasWidget;
                    if (historyIndex > 0) {
                        this.canvasWidget.historyIndex--;
                        const state = history[this.canvasWidget.historyIndex];
                        this.restoreState(state);
                    } else if (historyIndex === 0) {
                        // 撤销到空状态
                        this.canvasWidget.historyIndex--;
                        this.restoreState({ positivePoints: [], negativePoints: [], bboxes: [] });
                    }
                    this.updateUndoRedoUI();
                };

                this.redo = () => {
                    const { history, historyIndex } = this.canvasWidget;
                    if (historyIndex < history.length - 1) {
                        this.canvasWidget.historyIndex++;
                        const state = history[this.canvasWidget.historyIndex];
                        this.restoreState(state);
                    }
                    this.updateUndoRedoUI();
                };

                this.restoreState = (state) => {
                    this.canvasWidget.positivePoints = JSON.parse(JSON.stringify(state.positivePoints));
                    this.canvasWidget.negativePoints = JSON.parse(JSON.stringify(state.negativePoints));
                    this.canvasWidget.bboxes = JSON.parse(JSON.stringify(state.bboxes));
                    this.canvasWidget.selectedBox = null;
                    this.canvasWidget.dragMode = null;
                    this.canvasWidget.dragStart = null;
                    this.redrawCanvas();
                    this.updateWidgetValue();
                };

                this.updateUndoRedoUI = () => {
                    const { historyIndex, history, positivePoints, negativePoints, bboxes } = this.canvasWidget;
                    undoBtn.style.color = historyIndex >= 0 ? '#ccc' : '#555';
                    undoBtn.style.cursor = historyIndex >= 0 ? 'pointer' : 'default';
                    redoBtn.style.color = historyIndex < history.length - 1 ? '#ccc' : '#555';
                    redoBtn.style.cursor = historyIndex < history.length - 1 ? 'pointer' : 'default';

                    const hasContent = positivePoints.length > 0 || negativePoints.length > 0 || bboxes.length > 0;
                    resetBtn.style.color = hasContent ? '#ccc' : '#555';
                    resetBtn.style.cursor = hasContent ? 'pointer' : 'default';
                };

                this.updateWidgetValue = () => {
                    const { positivePoints, negativePoints, bboxes, image, frameIndex } = this.canvasWidget;
                    const info_widget = this.widgets.find(w=> w.name == 'info')
                    info_widget.value = image ? JSON.stringify({
                        positive_coords: positivePoints,
                        negative_coords: negativePoints,
                        bbox: bboxes,
                        frame_index: frameIndex
                    }) : '';
                }

                // 事件监听
                const getCoords = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    return {
                        x: (e.clientX - rect.left) * scaleX,
                        y: (e.clientY - rect.top) * scaleY
                    };
                };

                // 命中检测辅助（框内 / 缩放手柄）
                const pointInBox = (p, b, margin = 0) => {
                    return p.x >= b.x - margin && p.x <= b.x + b.w + margin &&
                           p.y >= b.y - margin && p.y <= b.y + b.h + margin;
                };
                const hitResizeHandle = (p, b, handleSize = 10) => {
                    const corners = [
                        { name: 'tl', cx: b.x, cy: b.y },
                        { name: 'tr', cx: b.x + b.w, cy: b.y },
                        { name: 'bl', cx: b.x, cy: b.y + b.h },
                        { name: 'br', cx: b.x + b.w, cy: b.y + b.h },
                    ];
                    for (const c of corners) {
                        if (Math.abs(p.x - c.cx) <= handleSize && Math.abs(p.y - c.cy) <= handleSize) return c.name;
                    }
                    return null;
                };

                canvas.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    const coords = getCoords(e);
                    const { mode, image } = this.canvasWidget;
                    if(!image) return
                    if (mode === 'point') {
                        if (e.button === 0) { // 左键：正面点
                            this.canvasWidget.positivePoints.push(coords);
                        } else if (e.button === 2) { // 右键：负面点
                            this.canvasWidget.negativePoints.push(coords);
                        }
                        this.addToHistory();
                        this.redrawCanvas();
                    } else if (mode === 'box') {
                        if (e.button !== 0) return; // 仅左键
                        const { bboxes, selectedBox } = this.canvasWidget;
                        // 1) 先检测选中框的缩放手柄
                        if (selectedBox != null && bboxes[selectedBox]) {
                            const handle = hitResizeHandle(coords, bboxes[selectedBox]);
                            if (handle) {
                                this.canvasWidget.dragMode = 'resize-' + handle;
                                this.canvasWidget.dragStart = { x: coords.x, y: coords.y, origBox: {...bboxes[selectedBox]} };
                                return;
                            }
                        }
                        // 2) 检测是否点击已有框（从最上层开始）→ 选中并移动
                        let hitIndex = -1;
                        for (let i = bboxes.length - 1; i >= 0; i--) {
                            if (pointInBox(coords, bboxes[i], 4)) { hitIndex = i; break; }
                        }
                        if (hitIndex >= 0) {
                            this.canvasWidget.selectedBox = hitIndex;
                            this.canvasWidget.dragMode = 'move';
                            this.canvasWidget.dragStart = { x: coords.x, y: coords.y, origBox: {...bboxes[hitIndex]} };
                            this.redrawCanvas();
                            return;
                        }
                        // 3) 点击空白 → 取消选中并开始绘制新框
                        this.canvasWidget.selectedBox = null;
                        this.canvasWidget.isDrawingBox = true;
                        this.canvasWidget.currentBox = { x: coords.x, y: coords.y, w: 0, h: 0 };
                        this.redrawCanvas();
                    }
                });

                canvas.addEventListener('mousemove', (e) => {
                    const coords = getCoords(e);
                    const { mode, isDrawingBox, currentBox, dragMode, dragStart, image, bboxes, selectedBox } = this.canvasWidget;
                    if(!image) return
                    if (dragMode === 'move' && dragStart && selectedBox != null && bboxes[selectedBox]) {
                        // 移动选中框
                        const box = bboxes[selectedBox];
                        box.x = dragStart.origBox.x + (coords.x - dragStart.x);
                        box.y = dragStart.origBox.y + (coords.y - dragStart.y);
                        this.redrawCanvas();
                    } else if (dragMode && dragMode.startsWith('resize-') && dragStart && selectedBox != null && bboxes[selectedBox]) {
                        // 缩放选中框（按手柄拖拽方向）
                        const box = bboxes[selectedBox];
                        const orig = dragStart.origBox;
                        const dx = coords.x - dragStart.x;
                        const dy = coords.y - dragStart.y;
                        const right = orig.x + orig.w;
                        const bottom = orig.y + orig.h;
                        if (dragMode === 'resize-tl') {
                            box.x = Math.min(orig.x + dx, right - 1);
                            box.y = Math.min(orig.y + dy, bottom - 1);
                            box.w = right - box.x;
                            box.h = bottom - box.y;
                        } else if (dragMode === 'resize-tr') {
                            box.x = orig.x;
                            box.w = Math.max(1, right + dx - orig.x);
                            box.y = Math.min(orig.y + dy, bottom - 1);
                            box.h = bottom - box.y;
                        } else if (dragMode === 'resize-bl') {
                            box.x = Math.min(orig.x + dx, right - 1);
                            box.w = right - box.x;
                            box.y = orig.y;
                            box.h = Math.max(1, bottom + dy - orig.y);
                        } else if (dragMode === 'resize-br') {
                            box.w = Math.max(1, right + dx - orig.x);
                            box.h = Math.max(1, bottom + dy - orig.y);
                        }
                        this.redrawCanvas();
                    } else if (mode === 'box' && isDrawingBox && currentBox) {
                        currentBox.w = coords.x - currentBox.x;
                        currentBox.h = coords.y - currentBox.y;
                        this.redrawCanvas();
                    } else {
                        // 悬停反馈：命中选中框手柄→缩放光标，命中其他框→移动光标
                        let cursor = 'crosshair';
                        if (selectedBox != null && bboxes[selectedBox]) {
                            const h = hitResizeHandle(coords, bboxes[selectedBox]);
                            if (h) {
                                cursor = (h === 'tl' || h === 'br') ? 'nwse-resize' : 'nesw-resize';
                            }
                        }
                        if (cursor === 'crosshair') {
                            for (let i = bboxes.length - 1; i >= 0; i--) {
                                if (pointInBox(coords, bboxes[i], 4)) { cursor = 'move'; break; }
                            }
                        }
                        if (canvas.style.cursor !== cursor) canvas.style.cursor = cursor;
                    }
                });

                // 结束当前交互（绘制/移动/缩放），幂等，可安全重复调用
                const finishInteraction = () => {
                    const { isDrawingBox, currentBox, dragMode, dragStart } = this.canvasWidget;
                    if (dragMode && dragStart) {
                        // 完成移动/缩放 → 记录历史
                        this.canvasWidget.dragMode = null;
                        this.canvasWidget.dragStart = null;
                        this.addToHistory();
                        this.redrawCanvas();
                    } else if (isDrawingBox && currentBox) {
                        // 规范化框（w/h 可能为负）
                        const box = {
                            x: Math.min(currentBox.x, currentBox.x + currentBox.w),
                            y: Math.min(currentBox.y, currentBox.y + currentBox.h),
                            w: Math.abs(currentBox.w),
                            h: Math.abs(currentBox.h)
                        };
                        // 仅有尺寸的框才添加
                        if (box.w > 5 && box.h > 5) {
                            this.canvasWidget.bboxes.push(box);
                            this.canvasWidget.selectedBox = this.canvasWidget.bboxes.length - 1;
                            this.addToHistory();
                        }
                        this.canvasWidget.isDrawingBox = false;
                        this.canvasWidget.currentBox = null;
                        this.redrawCanvas();
                    }
                };

                canvas.addEventListener('mouseup', (e) => {
                    const { image } = this.canvasWidget;
                    if(!image) return
                    finishInteraction();
                });

                // 画布外释放鼠标也能结束拖拽/绘制（防止框拖到边缘后"卡住"）
                const docMouseUpHandler = (e) => {
                    const { image, dragMode, dragStart, isDrawingBox, currentBox } = this.canvasWidget;
                    if (!image) return;
                    if (!dragMode && !(isDrawingBox && currentBox)) return;
                    finishInteraction();
                };
                document.addEventListener('mouseup', docMouseUpHandler);

                // 节点移除时清理挂在 document 上的监听
                const origOnRemoved = this.onRemoved;
                this.onRemoved = function () {
                    document.removeEventListener('mouseup', docMouseUpHandler);
                    if (origOnRemoved) origOnRemoved.apply(this, arguments);
                };

                canvas.addEventListener('contextmenu', (e) => {
                    e.preventDefault();
                });

                // 绘制初始占位
                this.redrawCanvas();

                // 设置初始节点尺寸
                const nodeWidth = Math.max(400, this.size[0] || 400);
                const nodeHeight = 530; // 初始高度：画布(400) + 空间(80)
                this.setSize([nodeWidth, nodeHeight]);

                // 设置初始容器高度
                container.style.height = "400px";

                this.updateUndoRedoUI();
                applyV3MinSize();
            });

            // 辅助：重绘画布
            nodeType.prototype.redrawCanvas = function() {
                const {canvas, ctx, image, positivePoints, negativePoints, bboxes, currentBox, selectedBox, hoveredPoint, mode} = this.canvasWidget;

                // 清空
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // pointSize：随画布显示尺寸缩放（考虑缩放）
                let pointSize = Math.max(4, Math.min(canvas.width, canvas.height) * 0.016);

                // 绘制图像
                if (image) {
                    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                } else {
                    const desc = [
                        "从您自己的图像或视频开始",
                        "左键点击：添加正面点",
                        "右键点击：添加负面点",
                        "拖动框：添加边界框",
                    ]
                    // 占位提示
                    ctx.fillStyle = "transparent";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = "#ddd";
                    ctx.font = "34px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("🖼️", canvas.width / 2, canvas.height / 2 - 50);
                    ctx.font = "24px sans-serif";
                    desc.map((text,index)=>{
                        ctx.fillText(text, canvas.width / 2, canvas.height / 2 + (index * 40));
                    })
                }

                // 绘制边界框
                ctx.strokeStyle = "#00f";
                ctx.lineWidth = 2;
                for (const box of bboxes) {
                    ctx.strokeRect(box.x, box.y, box.w, box.h);
                    // 半透明填充
                    ctx.fillStyle = "rgba(0, 0, 255, 0.1)";
                    ctx.fillRect(box.x, box.y, box.w, box.h);
                }

                // 绘制选中框（高亮 + 四角缩放手柄）
                if (selectedBox != null) {
                    const selBox = bboxes[selectedBox];
                    if (selBox) {
                        ctx.strokeStyle = "#ffd400";
                        ctx.lineWidth = 2;
                        ctx.strokeRect(selBox.x, selBox.y, selBox.w, selBox.h);
                        const hs = Math.max(5, Math.min(canvas.width, canvas.height) * 0.008);
                        const corners = [
                            [selBox.x, selBox.y],
                            [selBox.x + selBox.w, selBox.y],
                            [selBox.x, selBox.y + selBox.h],
                            [selBox.x + selBox.w, selBox.y + selBox.h],
                        ];
                        ctx.fillStyle = "#ffd400";
                        for (const [hx, hy] of corners) {
                            ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
                        }
                    }
                }

                // 绘制当前框
                if (currentBox) {
                    ctx.strokeStyle = "#0ff";
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    ctx.strokeRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);
                    ctx.setLineDash([]);
                }

                // 绘制正面点（绿色）
                ctx.strokeStyle = "#139613";
                ctx.fillStyle = "#139613";
                for (const point of positivePoints) {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, pointSize, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }

                // 绘制负面点（红色）
                ctx.strokeStyle = "#8A1616";
                ctx.fillStyle = "#8A1616";
                for (const point of negativePoints) {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, pointSize, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            };


        }
    }
})
