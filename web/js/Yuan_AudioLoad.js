/**
 * Yuan Tool · 加载音频前端：内嵌播放器 + 时间轴裁剪 + 智能分段
 * （复刻自 Yuan-TV「加载音频 UI」节点，适配本插件 V2/V3 双前端：
 *   V3 下移除隐藏参数的占位输入端口；Vue 控件不走原生 callback，用 onDrawForeground 轮询同步）。
 */
import { getApi, findComfyNodeEl, enforceV3MinSize, removeV3PlaceholderInput } from "./Yuan_Common.js";

const { app } = window.comfyAPI.app;

const api = getApi();

// 最小尺寸与 DOM 固定高度（节点尺寸决定 DOM 高度，不做内容实测）
const NODE_MIN_W = 475;
const NODE_MIN_H = 310;
const DOM_H = 220;

// 前端隐藏管理、由自绘 UI 驱动的参数（V3 下移除其占位输入端口）
const MANAGED_PARAMS = ["开始时间", "结束时间", "时长", "输出模式", "分段时长", "分段索引"];

// 音频/视频扩展名（与后端一致；上传过滤用扩展名判断，避免空 MIME 类型被拒）
const AUDIO_EXTS = [
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma",
    ".aiff", ".aif", ".mp4", ".m4v", ".webm", ".mov", ".avi", ".mkv",
];
function isAudioFileName(name) {
    const dot = name.lastIndexOf(".");
    if (dot < 0) return false;
    return AUDIO_EXTS.indexOf(name.slice(dot).toLowerCase()) >= 0;
}

function chainCallback(object, property, callback) {
    if (object == undefined) return;
    if (property in object) {
        const orig = object[property];
        object[property] = function () {
            const r = orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r;
        };
    } else {
        object[property] = callback;
    }
}

function hideWidget(w) {
    if (!w) return;
    w.hidden = true;
    if (!w.options) w.options = {};
    w.options.hidden = true;
    if (!window.LiteGraph || !window.LiteGraph.vueNodesMode) {
        w.computeSize = () => [0, -4];
        if (!w._hiddenDrawHooked) {
            w._origDraw = w.hasOwnProperty("draw") ? w.draw : undefined;
            w._hiddenDrawHooked = true;
        }
        w.draw = () => { };
    }
    if (w.element) w.element.style.display = "none";
}

function showWidget(w) {
    if (!w) return;
    w.hidden = false;
    if (w.options) w.options.hidden = false;
    if (!window.LiteGraph || !window.LiteGraph.vueNodesMode) {
        delete w.computeSize;
        if (w._hiddenDrawHooked) {
            if (w._origDraw !== undefined) {
                w.draw = w._origDraw;
            } else {
                delete w.draw;
            }
            delete w._hiddenDrawHooked;
        }
    }
    if (w.element) w.element.style.display = "";
}

app.registerExtension({
    name: "Yuan-Tool.AudioLoad",
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "YuanAudioLoad") return;

        // 工作流加载：恢复控件显隐与模式视觉（不改节点尺寸）
        chainCallback(nodeType.prototype, "onConfigure", function () {
            setTimeout(() => {
                try {
                    if (this.syncLayoutToNode) this.syncLayoutToNode();
                } catch (e) { console.warn("[YuanAudioLoad] onConfigure layout error:", e); }
            }, 0);
            // 等 UI 绑定完成后恢复模式按钮高亮与小方格状态
            setTimeout(() => {
                try {
                    if (this.syncModeVisual) this.syncModeVisual();
                } catch (e) { console.warn("[YuanAudioLoad] onConfigure visual error:", e); }
            }, 120);
            // V3：工作流加载后端口由核心重建，延迟移除占位端口
            [200, 800, 2000].forEach((ms) => setTimeout(() => {
                if (this._yuanAudioRemovePorts) this._yuanAudioRemovePorts(this);
            }, ms));
        });

        // 节点尺寸变化时同步 DOM 高度（仅 V2；V3 高度由布局系统管理）
        chainCallback(nodeType.prototype, "onResize", function (size) {
            if (this.isYuanAudioV3 && this.isYuanAudioV3()) return;
            if (this.syncLayoutToNode) this.syncLayoutToNode();
            if (this.domWidget && this.domWidget.element) {
                this.domWidget.element.style.margin = "0";
                let yOffset = this.domWidget.last_y;
                if (!yOffset) {
                    yOffset = 30; // LiteGraph 标题高兜底
                    if (this.widgets) {
                        for (const w of this.widgets) {
                            if (w === this.domWidget) break;
                            yOffset += (w.computeSize ? w.computeSize()[1] : 20) + 4;
                        }
                    }
                }
                const remainingHeight = size[1] - yOffset - 18;
                this.domWidget.element.style.height = `${Math.max(DOM_H, remainingHeight)}px`;
            }
        });

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            const node = this;

            // 初始化标记（true 期间切换音频不重置裁剪区间，避免工作流载入误重置）
            node._initializing = true;
            node._should_reset_trim = false;

            // --- V3 检测（向上找 comfy-node 祖先，含 shadow DOM host） ---
            let v3NodeElement = null;
            function checkIsV3() {
                if (v3NodeElement) return true;
                const mount = (node.domWidget && node.domWidget.element)
                    ? node.domWidget.element.parentElement : null;
                const el = findComfyNodeEl(mount);
                if (el) { v3NodeElement = el; return true; }
                return false;
            }
            node.isYuanAudioV3 = checkIsV3;

            // V3：移除被隐藏管理参数的空占位端口（已有连线的保留不删）。
            // 以输入端口名签名做变化检测：端口列表未变时直接跳过，
            // 避免每帧调用 removeInput 触发重排（工作流切换时节点"闪一下"的根源之一）
            function removePlaceholderPorts(n, force) {
                if (!checkIsV3() || !n || !Array.isArray(n.inputs)) return;
                const sig = n.inputs.map(i => i.name).join("|");
                if (!force && sig === n._yuanAudioPortSig) return;
                for (const name of MANAGED_PARAMS) removeV3PlaceholderInput(n, name);
                n._yuanAudioPortSig = n.inputs.map(i => i.name).join("|");
            }
            node._yuanAudioRemovePorts = removePlaceholderPorts;

            // --- 通用上传处理（拖拽/选择文件，扩展名过滤） ---
            const handleFileUpload = async (file) => {
                if (!isAudioFileName(file.name)) return false;
                try {
                    const body = new FormData();
                    body.append("image", file); // /upload/image 端点按字段名取文件，不限扩展名
                    body.append("type", "input");

                    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                    if (resp.status === 200) {
                        const data = await resp.json();
                        let name = data.name;
                        if (data.subfolder) name = data.subfolder + "/" + name; // 修复：带上子目录
                        const audioWidget = node.widgets && node.widgets.find(w => w.name === "音频");
                        if (audioWidget) {
                            // 手动上传后总是重置裁剪区间
                            node._should_reset_trim = true;
                            audioWidget.value = name;
                            if (audioWidget.options && audioWidget.options.values && !audioWidget.options.values.includes(name)) {
                                audioWidget.options.values.push(name);
                            }
                            if (audioWidget.callback) {
                                audioWidget.callback(name);
                            }
                            if (app.graph) app.graph.setDirtyCanvas(true, false);
                        }
                    }
                } catch (err) {
                    console.error("[YuanAudioLoad] 上传音频出错:", err);
                }
                return true;
            };

            node.onDragOver = function (e) {
                if (e.dataTransfer && e.dataTransfer.types && e.dataTransfer.types.includes("Files")) {
                    e.preventDefault();
                    return true;
                }
                return false;
            };
            node.onDragDrop = function (e) {
                if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    const file = e.dataTransfer.files[0];
                    if (isAudioFileName(file.name)) {
                        handleFileUpload(file);
                        return true;
                    }
                }
                return false;
            };

            // --- 自绘 UI 主容器 ---
            const container = document.createElement("div");
            const defaultBg = "rgba(30, 30, 30, 0.9)";
            Object.assign(container.style, {
                display: "flex",
                flexDirection: "column",
                gap: "10px",
                width: "100%",
                padding: "10px",
                boxSizing: "border-box",
                background: defaultBg,
                borderRadius: "6px",
                color: "white",
                fontFamily: "sans-serif",
                marginTop: "0",
                flexShrink: "0",
                transition: "background 0.2s"
            });

            // 上传按钮：点击弹出文件选择，复用 handleFileUpload
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.accept = "audio/*,video/*," + AUDIO_EXTS.join(",");
            fileInput.style.display = "none";

            const uploadBtn = document.createElement("button");
            uploadBtn.type = "button";
            uploadBtn.textContent = "选择音频上传";
            Object.assign(uploadBtn.style, {
                fontSize: "11px",
                color: "#7dd3fc",
                background: "rgba(56,189,248,0.12)",
                border: "1px solid rgba(56,189,248,0.35)",
                padding: "2px 8px",
                borderRadius: "4px",
                cursor: "pointer",
                whiteSpace: "nowrap",
                flexShrink: "0",
                fontFamily: "inherit",
                outline: "none"
            });
            uploadBtn.onmouseenter = () => { uploadBtn.style.background = "rgba(56,189,248,0.28)"; };
            uploadBtn.onmouseleave = () => { uploadBtn.style.background = "rgba(56,189,248,0.12)"; };
            uploadBtn.onclick = (e) => { e.preventDefault(); e.stopPropagation(); fileInput.click(); };
            fileInput.addEventListener("change", (e) => {
                const f = e.target.files && e.target.files[0];
                if (f) handleFileUpload(f);
                fileInput.value = "";
            });

            // 时间输入框：「开始-结束 时长」（单位秒）
            const timeInputWrap = document.createElement("div");
            Object.assign(timeInputWrap.style, {
                display: "flex",
                alignItems: "center",
                gap: "2px",
                flexShrink: "0"
            });

            const mkTimeBox = (tip) => {
                const inp = document.createElement("input");
                inp.type = "number";
                inp.step = "0.01";
                inp.min = "0";
                inp.title = tip;
                Object.assign(inp.style, {
                    width: "56px",
                    height: "22px",
                    boxSizing: "border-box",
                    background: "rgba(0, 0, 0, 0.4)",
                    color: "#38bdf8",
                    border: "1px solid rgba(56, 189, 248, 0.3)",
                    borderRadius: "4px",
                    textAlign: "center",
                    fontSize: "12px",
                    fontWeight: "bold",
                    outline: "none",
                    padding: "0 2px"
                });
                inp.addEventListener("keydown", (e) => e.stopPropagation());
                return inp;
            };
            const startTimeBox = mkTimeBox("开始时间（秒）");
            const endTimeBox = mkTimeBox("结束时间（秒）");
            const durationBox = mkTimeBox("时长（秒）");

            const timeDash = document.createElement("span");
            timeDash.textContent = "-";
            Object.assign(timeDash.style, {
                color: "rgba(255, 255, 255, 0.4)",
                fontSize: "12px",
                fontWeight: "bold",
                padding: "0 1px"
            });
            const timeGap = document.createElement("span");
            timeGap.style.width = "10px";

            timeInputWrap.appendChild(startTimeBox);
            timeInputWrap.appendChild(timeDash);
            timeInputWrap.appendChild(endTimeBox);
            timeInputWrap.appendChild(timeGap);
            timeInputWrap.appendChild(durationBox);

            // 顶栏：左侧上传按钮，右侧时间输入框
            const playerTop = document.createElement("div");
            Object.assign(playerTop.style, {
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "0 2px",
                marginBottom: "-4px"
            });
            playerTop.appendChild(uploadBtn);
            playerTop.appendChild(timeInputWrap);
            container.appendChild(playerTop);
            container.appendChild(fileInput);

            const audioEl = document.createElement("audio");
            audioEl.controls = true;
            audioEl.style.width = "100%";
            audioEl.style.height = "40px";
            audioEl.style.outline = "none";
            container.appendChild(audioEl);

            // 输出模式切换行：自定义裁切 / 智能分段
            const modeRow = document.createElement("div");
            Object.assign(modeRow.style, {
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "8px"
            });

            const modeToggleBox = document.createElement("div");
            Object.assign(modeToggleBox.style, {
                display: "flex",
                alignItems: "center",
                background: "rgba(0, 0, 0, 0.35)",
                border: "1px solid rgba(56, 189, 248, 0.3)",
                borderRadius: "4px",
                overflow: "hidden",
                flexShrink: "0",
                cursor: "pointer"
            });

            const createModeBtn = (label) => {
                const b = document.createElement("span");
                b.textContent = label;
                Object.assign(b.style, {
                    fontSize: "11px",
                    padding: "3px 12px",
                    whiteSpace: "nowrap",
                    userSelect: "none",
                    transition: "background 0.15s, color 0.15s"
                });
                return b;
            };
            const segCustomBtn = createModeBtn("自定义裁切");
            const segSmartBtn = createModeBtn("智能分段");
            const modeDivider = document.createElement("span");
            modeDivider.style.cssText = "width:1px;height:12px;background:rgba(56,189,248,0.25);flex-shrink:0;";
            modeToggleBox.appendChild(segCustomBtn);
            modeToggleBox.appendChild(modeDivider);
            modeToggleBox.appendChild(segSmartBtn);

            const segStatus = document.createElement("span");
            Object.assign(segStatus.style, {
                fontSize: "11px",
                color: "#aaa",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
                minWidth: "0",
                textAlign: "right"
            });
            segStatus.textContent = "已裁剪: 0.0秒";

            modeRow.appendChild(modeToggleBox);
            modeRow.appendChild(segStatus);
            container.appendChild(modeRow);

            const trimArea = document.createElement("div");
            Object.assign(trimArea.style, {
                display: "flex",
                flexDirection: "column",
                gap: "6px",
                background: "rgba(0, 0, 0, 0.35)",
                padding: "12px",
                borderRadius: "6px",
                border: "1px solid rgba(255, 255, 255, 0.05)"
            });

            const timeRuler = document.createElement("div");
            Object.assign(timeRuler.style, {
                position: "relative",
                width: "100%",
                height: "22px",
                fontSize: "10px",
                color: "#aaa",
                pointerEvents: "none",
                userSelect: "none"
            });
            trimArea.appendChild(timeRuler);

            const sliderBox = document.createElement("div");
            Object.assign(sliderBox.style, {
                position: "relative",
                width: "100%",
                height: "24px",
                background: "#111",
                borderRadius: "4px",
                cursor: "pointer",
                userSelect: "none",
                boxShadow: "inset 0 1px 3px rgba(0,0,0,0.5)"
            });

            const fill = document.createElement("div");
            Object.assign(fill.style, {
                position: "absolute",
                height: "100%",
                background: "rgba(14, 165, 233, 0.35)",
                pointerEvents: "none"
            });
            sliderBox.appendChild(fill);

            const createHandle = (color) => {
                const h = document.createElement("div");
                Object.assign(h.style, {
                    position: "absolute",
                    top: "0",
                    width: "8px",
                    height: "100%",
                    background: color,
                    transform: "translateX(-50%)",
                    pointerEvents: "none",
                    boxShadow: "0 0 4px rgba(0,0,0,0.8)",
                    borderRadius: "2px"
                });
                return h;
            };

            const startHandle = createHandle("#38bdf8");
            const endHandle = createHandle("#38bdf8");
            sliderBox.appendChild(startHandle);
            sliderBox.appendChild(endHandle);

            // 智能分段的小方格层（自定义裁切模式下隐藏）
            const segBlocksLayer = document.createElement("div");
            Object.assign(segBlocksLayer.style, {
                position: "absolute",
                left: "0", top: "0", width: "100%", height: "100%",
                display: "none",
                pointerEvents: "none",
                zIndex: "3"
            });
            sliderBox.appendChild(segBlocksLayer);

            trimArea.appendChild(sliderBox);

            container.appendChild(trimArea);

            // 尺寸计算前先同步行显隐，避免把隐藏的原生行计入节点高度
            (function () {
                if (!node.widgets) return;
                const findW = (n) => node.widgets.find(w => w.name === n);
                const smart = !!(findW("输出模式") && findW("输出模式").value === "智能分段");
                hideWidget(findW("开始时间"));
                hideWidget(findW("结束时间"));
                hideWidget(findW("时长"));
                hideWidget(findW("输出模式"));
                if (smart) {
                    showWidget(findW("分段时长"));
                    showWidget(findW("分段索引"));
                } else {
                    hideWidget(findW("分段时长"));
                    hideWidget(findW("分段索引"));
                }
            })();

            // 把容器挂到节点 UI 上
            node.domWidget = node.addDOMWidget("audio_ui", "yuan_audio_ui", container);

            // 同步 DOM 宽度到节点宽度（仅 V2；V3 由布局系统管理宽度）
            node.syncLayoutToNode = function () {
                if (checkIsV3()) {
                    container.style.width = "100%";
                    container.style.maxWidth = "100%";
                    container.style.boxSizing = "border-box";
                    return;
                }
                const nodeWidth = this.size?.[0] || NODE_MIN_W;
                const targetWidth = Math.max(10, nodeWidth - 30);
                container.style.width = `${targetWidth}px`;
                container.style.maxWidth = `${targetWidth}px`;
                container.style.boxSizing = "border-box";
            };

            // V2：DOM widget 固定包围盒（宽度跟随节点，高度固定常量）
            node.domWidget.computeSize = function (width) {
                const nodeWidth = node.size?.[0] || width || NODE_MIN_W;
                return [Math.max(10, nodeWidth - 30), DOM_H];
            };

            // V3：DOM 行最小尺寸（必须返回 widget 自身尺寸，不能返回整节点尺寸）
            if (typeof node.domWidget.computeLayoutSize === "function") {
                const prevCLS = node.domWidget.computeLayoutSize.bind(node.domWidget);
                node.domWidget.computeLayoutSize = (targetNode) => {
                    const p = prevCLS(targetNode) || {};
                    return {
                        ...p,
                        minWidth: Math.max(NODE_MIN_W - 30, Number(p.minWidth || 0)),
                        minHeight: Math.max(DOM_H, Number(p.minHeight || 0)),
                    };
                };
            }

            // V3：在 comfy-node 元素上同步最小尺寸。
            // 缓存上次写入值：尺寸为常量，只在首次（或元素更换后）写一次样式，
            // 避免每帧 removeProperty/setProperty 与 V3 布局系统争抢引发闪动
            let _v3MinApplied = null;
            function applyV3MinSize() {
                if (!checkIsV3()) return;
                try {
                    node.min_size = [NODE_MIN_W, NODE_MIN_H];
                    if (_v3MinApplied !== v3NodeElement) {
                        enforceV3MinSize(v3NodeElement, NODE_MIN_W, NODE_MIN_H);
                        _v3MinApplied = v3NodeElement;
                    }
                } catch (_) {}
            }

            // 仅兜底最小尺寸，不做内容实测；V2 DOM 高度由 onResize/onDrawForeground 同步
            requestAnimationFrame(() => {
                if (node.size[0] < NODE_MIN_W) node.size[0] = NODE_MIN_W;
                if (node.size[1] < NODE_MIN_H) node.size[1] = NODE_MIN_H;
                node.syncLayoutToNode();
                applyV3MinSize();
                if (!checkIsV3() && node.onResize) node.onResize(node.size);
            });

            // 把节点数据动态绑定到 UI
            setTimeout(() => {
                if (!node.widgets) return;
                const audioWidget = node.widgets.find(w => w.name === "音频");
                const startWidget = node.widgets.find(w => w.name === "开始时间");
                const endWidget = node.widgets.find(w => w.name === "结束时间");
                const durationWidget = node.widgets.find(w => w.name === "时长");
                const outputModeWidget = node.widgets.find(w => w.name === "输出模式");
                const segmentIndexWidget = node.widgets.find(w => w.name === "分段索引");
                const durationsWidget = node.widgets.find(w => w.name === "分段时长");

                let isSmart = !!(outputModeWidget && outputModeWidget.value === "智能分段");

                // 开始/结束/时长已挪到上方时间框，隐藏原生控件行
                hideWidget(startWidget);
                hideWidget(endWidget);
                hideWidget(durationWidget);

                let duration = 0;
                let dragging = null;
                let dragOffset = 0;
                let dragSelectionWidth = 0;
                let isUpdatingDuration = false; // 防止死循环
                let playBounds = null;
                let segRange = null;

                // 挂钩原生"时长"控件
                if (durationWidget) {
                    const origCallback = durationWidget.callback;
                    durationWidget.callback = function (v) {
                        // 智能分段下"时长"由选中分段决定，忽略手动修改
                        if (isSmart) {
                            renderSegBlocks();
                            return;
                        }
                        if (!duration || isUpdatingDuration) {
                            if (origCallback) origCallback.apply(this, arguments);
                            return;
                        }

                        isUpdatingDuration = true;
                        let d = parseFloat(v) || 0;
                        if (d < 0) d = 0;
                        if (d > duration) d = duration;

                        let s = startWidget ? parseFloat(startWidget.value) || 0 : 0;
                        let newStart = s;
                        let newEnd = s + d;

                        // 超出音频末尾时开始时间往回平移
                        if (newEnd > duration) {
                            newEnd = duration;
                            newStart = duration - d;
                        }

                        if (startWidget) startWidget.value = parseFloat(newStart.toFixed(2));
                        if (endWidget) endWidget.value = parseFloat(newEnd.toFixed(2));

                        updateUI(true);
                        if (app.graph) app.graph.setDirtyCanvas(true, false);

                        if (origCallback) origCallback.apply(this, arguments);
                        isUpdatingDuration = false;
                    };
                }

                const updateAudio = () => {
                    if (!audioWidget || !audioWidget.value || audioWidget.value === "none") {
                        audioEl.removeAttribute("src");
                        return;
                    }
                    let filename = audioWidget.value;
                    let subfolder = "";
                    if (filename.includes("/") || filename.includes("\\")) {
                        const sep = filename.includes("/") ? "/" : "\\";
                        const parts = filename.split(sep);
                        filename = parts.pop();
                        subfolder = parts.join("/");
                    }
                    audioEl.src = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=${encodeURIComponent(subfolder)}`);
                };

                if (audioWidget) {
                    const origAudioCb = audioWidget.callback;
                    audioWidget.callback = function () {
                        // 手动切换下拉选项时标记需要重置裁剪区间
                        if (!node._initializing) {
                            node._should_reset_trim = true;
                        }
                        updateAudio();
                        if (origAudioCb) origAudioCb.apply(this, arguments);
                    };
                    node._lastAudioValue = audioWidget.value;
                    updateAudio();
                }

                container.ondragover = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    container.style.background = "rgba(14, 165, 233, 0.2)";
                };
                container.ondragleave = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    container.style.background = defaultBg;
                };
                container.ondrop = async (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    container.style.background = defaultBg;
                    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                        handleFileUpload(e.dataTransfer.files[0]);
                    }
                };

                const formatTime = (secs) => {
                    if (secs < 60) return secs.toFixed(1) + "秒";
                    const m = Math.floor(secs / 60);
                    const s = (secs % 60).toFixed(1);
                    return `${m}:${s.padStart(4, '0')}`;
                };

                const updateRuler = () => {
                    timeRuler.innerHTML = '';
                    if (!duration) return;
                    const numMajorTicks = 5;
                    const subTicks = 4;
                    const totalTicks = (numMajorTicks - 1) * subTicks;
                    for (let i = 0; i <= totalTicks; i++) {
                        const pct = i / totalTicks;
                        const t = duration * pct;
                        const isMajor = i % subTicks === 0;
                        const tickWrapper = document.createElement("div");
                        Object.assign(tickWrapper.style, {
                            position: "absolute", left: `${pct * 100}%`, top: "0",
                            display: "flex", flexDirection: "column", alignItems: "center", transform: "translateX(-50%)"
                        });
                        if (i === 0) { tickWrapper.style.transform = "none"; tickWrapper.style.alignItems = "flex-start"; }
                        if (i === totalTicks) { tickWrapper.style.transform = "translateX(-100%)"; tickWrapper.style.alignItems = "flex-end"; }
                        const line = document.createElement("div");
                        Object.assign(line.style, {
                            width: isMajor ? "2px" : "1px", height: isMajor ? "6px" : "4px",
                            background: isMajor ? "#aaa" : "#555", marginBottom: "2px", borderRadius: "1px"
                        });
                        tickWrapper.appendChild(line);
                        if (isMajor) {
                            const label = document.createElement("div");
                            label.textContent = formatTime(t);
                            tickWrapper.appendChild(label);
                        }
                        timeRuler.appendChild(tickWrapper);
                    }
                };

                const updateUI = (syncPlayer = false) => {
                    if (!duration) return;
                    if (isSmart) {
                        // 智能分段：时间轴切换为小方格分段视图
                        renderSegBlocks();
                        if (syncPlayer && audioEl.readyState >= 1) { audioEl.currentTime = effectiveWindow().s; }
                        return;
                    }
                    const win = effectiveWindow();
                    const sPct = (win.s / duration) * 100;
                    const ePct = (win.e / duration) * 100;
                    startHandle.style.left = `${sPct}%`;
                    endHandle.style.left = `${ePct}%`;
                    fill.style.left = `${sPct}%`;
                    fill.style.width = `${ePct - sPct}%`;

                    // 同步原生"时长"控件
                    const currentDur = parseFloat((win.e - win.s).toFixed(2));
                    segStatus.textContent = `已裁剪: ${currentDur}秒`;
                    if (durationWidget && durationWidget.value !== currentDur) {
                        isUpdatingDuration = true;
                        durationWidget.value = currentDur;
                        isUpdatingDuration = false;
                    }
                    syncTimeInputs();

                    if (syncPlayer && audioEl.readyState >= 1) { audioEl.currentTime = win.s; }
                };

                // 按"分段时长"(如 6,3,5) 把 [开始,结束] 窗口切成小方格
                const parseDurationList = (text) => {
                    if (text == null) return [];
                    const out = [];
                    String(text).split(/[,，、;；\s]+/).forEach((p) => {
                        if (!p) return;
                        const v = parseFloat(p);
                        if (isFinite(v) && v > 0) out.push(v);
                    });
                    return out;
                };

                const effectiveWindow = () => {
                    let s = startWidget ? (parseFloat(startWidget.value) || 0) : 0;
                    let e = endWidget ? (parseFloat(endWidget.value) || 0) : 0;
                    if (e <= s) e = duration; // 0 或 <=开始 => 取到文件末尾
                    s = Math.max(0, Math.min(s, duration));
                    e = Math.max(s, Math.min(e, duration));
                    return { s, e };
                };

                // 与后端保持一致的分段边界算法
                const computeSegBounds = (s, e, durs) => {
                    const bounds = [];
                    let cur = s;
                    for (const d of durs) {
                        if (cur >= e - 1e-6) break;
                        cur = Math.min(cur + d, e);
                        bounds.push(cur);
                    }
                    if (!bounds.length) bounds = [e];
                    return bounds;
                };

                const segStartOf = (bounds, i, s) => (i === 0 ? s : bounds[i - 1]);

                const getPlayBounds = () => {
                    if (isSmart && playBounds && playBounds.e > playBounds.s) return playBounds;
                    return effectiveWindow();
                };

                // 模式切换：按钮高亮 + 时间轴显示（长条/小方格）+ 原生控件显隐
                const applyModeVisual = () => {
                    if (isSmart) {
                        segCustomBtn.style.background = "transparent";
                        segCustomBtn.style.color = "rgba(255,255,255,0.45)";
                        segSmartBtn.style.background = "rgba(37,126,235,0.85)";
                        segSmartBtn.style.color = "#fff";
                        segStatus.style.color = "#7dd3fc";
                    } else {
                        segCustomBtn.style.background = "rgba(56,189,248,0.85)";
                        segCustomBtn.style.color = "#fff";
                        segSmartBtn.style.background = "transparent";
                        segSmartBtn.style.color = "rgba(255,255,255,0.45)";
                        segStatus.style.color = "#aaa";
                    }
                    fill.style.display = isSmart ? "none" : "block";
                    startHandle.style.display = isSmart ? "none" : "block";
                    endHandle.style.display = isSmart ? "none" : "block";
                    segBlocksLayer.style.display = isSmart ? "block" : "none";
                    syncNativeRows();
                    setTimeBoxReadonly(isSmart);
                };

                const syncNativeRows = () => {
                    // 输出模式由按钮驱动，隐藏原生下拉框
                    hideWidget(outputModeWidget);
                    if (isSmart) {
                        showWidget(durationsWidget);
                        showWidget(segmentIndexWidget);
                    } else {
                        hideWidget(durationsWidget);
                        hideWidget(segmentIndexWidget);
                    }
                    // V3：移除隐藏参数的空占位端口
                    removePlaceholderPorts(node, true);
                    // 只兜底最小包围盒，不压缩、不改动节点尺寸；
                    // 仅在实际发生变化时才触发 onResize/重绘，避免工作流加载时无谓闪动
                    let sizeChanged = false;
                    if (node.size[0] < NODE_MIN_W) { node.size[0] = NODE_MIN_W; sizeChanged = true; }
                    if (node.size[1] < NODE_MIN_H) { node.size[1] = NODE_MIN_H; sizeChanged = true; }
                    if (!checkIsV3() && sizeChanged && node.onResize) node.onResize(node.size);
                    if (app.graph) app.graph.setDirtyCanvas(true, true);
                };

                // 在时间轴上绘制分段小方格，并联动"分段索引"
                const renderSegBlocks = () => {
                    segBlocksLayer.innerHTML = "";
                    if (!isSmart || !duration) return;
                    const win = effectiveWindow();
                    // 留空时整个窗口作为 1 段（与后端兜底逻辑一致），显示完整长方块
                    const durs = parseDurationList(durationsWidget ? durationsWidget.value : "");
                    const bounds = computeSegBounds(win.s, win.e, durs);
                    const count = bounds.length;
                    let selIdx = segmentIndexWidget ? (parseInt(segmentIndexWidget.value) || 0) : 0;
                    if (selIdx < 0) selIdx = 0;
                    if (selIdx >= count) selIdx = count - 1;
                    if (segmentIndexWidget && (parseInt(segmentIndexWidget.value) || 0) !== selIdx) {
                        segmentIndexWidget.value = selIdx;
                    }

                    for (let i = 0; i < count; i++) {
                        const a = segStartOf(bounds, i, win.s);
                        const b = bounds[i];
                        const leftPct = Math.max(0, (a / duration) * 100);
                        const rightPct = Math.min(100, (b / duration) * 100);
                        const widthPct = Math.max(rightPct - leftPct, 0.6);
                        const isSel = i === selIdx;
                        const block = document.createElement("div");
                        const hue = 195 + ((i % 5) * 9);
                        Object.assign(block.style, {
                            position: "absolute",
                            left: `${leftPct}%`,
                            width: `${widthPct}%`,
                            top: "6%",
                            height: "88%",
                            background: isSel ? "rgba(56,189,248,0.9)" : `hsla(${hue}, 90%, 60%, 0.4)`,
                            border: isSel ? "1px solid #fff" : "1px solid rgba(255,255,255,0.5)",
                            borderRadius: "2px",
                            boxSizing: "border-box",
                            pointerEvents: "auto",
                            cursor: "pointer",
                            boxShadow: isSel ? "0 0 6px rgba(56,189,248,0.9)" : "none"
                        });
                        block.title = `分段 ${i + 1}/${count} · ${(b - a).toFixed(1)}秒（点击选择输出）`;
                        block.onpointerdown = (ev) => ev.stopPropagation();
                        block.onclick = () => {
                            if (segmentIndexWidget) segmentIndexWidget.value = i;
                            renderSegBlocks();
                            if (app.graph) app.graph.setDirtyCanvas(true, false);
                            // 暂停状态下跳转到该分段起点预览
                            if (audioEl.readyState >= 1 && audioEl.paused && duration > 0) {
                                audioEl.currentTime = Math.min(a + 1e-6, Math.max(0, duration - 0.01));
                            }
                        };
                        segBlocksLayer.appendChild(block);
                    }

                    const a = segStartOf(bounds, selIdx, win.s);
                    const b = bounds[selIdx];
                    segRange = { s: a, e: b };
                    playBounds = { s: a, e: b };
                    segStatus.textContent = `分段 ${selIdx + 1}/${count} · ${(b - a).toFixed(1)}秒 · 共${count}段`;
                    const segDur = parseFloat((b - a).toFixed(2));
                    if (durationWidget && durationWidget.value !== segDur) {
                        isUpdatingDuration = true;
                        durationWidget.value = segDur;
                        isUpdatingDuration = false;
                    }
                    syncTimeInputs();
                };

                // 「开始-结束 时长」时间框
                const syncTimeInputs = () => {
                    if (!duration) {
                        startTimeBox.value = startWidget ? (parseFloat(startWidget.value) || 0) : 0;
                        endTimeBox.value = endWidget ? (parseFloat(endWidget.value) || 0) : 0;
                        durationBox.value = 0;
                        return;
                    }
                    let a, b, dur;
                    if (isSmart && segRange && segRange.e > segRange.s) {
                        // 智能分段：展示当前选中分段的起止范围（只读）
                        a = segRange.s;
                        b = segRange.e;
                        dur = b - a;
                    } else {
                        const win = effectiveWindow();
                        a = win.s;
                        b = win.e;
                        dur = b - a;
                    }
                    startTimeBox.value = parseFloat(a.toFixed(2));
                    endTimeBox.value = parseFloat(b.toFixed(2));
                    durationBox.value = parseFloat(dur.toFixed(2));
                };

                // 智能分段时时间框只读（范围由小方格 / 分段索引决定）
                const setTimeBoxReadonly = (ro) => {
                    const tips = { start: "开始时间（秒）", end: "结束时间（秒）", duration: "时长（秒）" };
                    const map = { start: startTimeBox, end: endTimeBox, duration: durationBox };
                    Object.keys(map).forEach((role) => {
                        const box = map[role];
                        box.readOnly = ro;
                        box.style.opacity = ro ? "0.55" : "1";
                        box.style.cursor = ro ? "not-allowed" : "text";
                        box.title = ro ? "智能分段下由小方格 / 分段索引决定" : tips[role];
                    });
                };

                const applyTimeBoxChange = (role) => {
                    if (!duration || isSmart) return; // 未加载音频或智能分段下不接收手动修改
                    let s = startWidget ? (parseFloat(startWidget.value) || 0) : 0;
                    let e = endWidget ? (parseFloat(endWidget.value) || 0) : 0;
                    if (e <= s) e = duration;
                    const box = role === "start" ? startTimeBox : role === "end" ? endTimeBox : durationBox;
                    const n = parseFloat(box.value);
                    if (isNaN(n) || n < 0) { syncTimeInputs(); return; }

                    if (role === "duration") {
                        // 结束 = 开始 + 时长（可自动回推开始）
                        const d = Math.min(n, duration);
                        let newStart = s;
                        let newEnd = s + d;
                        if (newEnd > duration) {
                            newEnd = duration;
                            newStart = duration - d;
                        }
                        if (startWidget) startWidget.value = parseFloat(newStart.toFixed(2));
                        if (endWidget) endWidget.value = parseFloat(newEnd.toFixed(2));
                    } else if (role === "start") {
                        if (n > duration) { syncTimeInputs(); return; }
                        const newEnd = n >= e ? duration : e;
                        if (startWidget) startWidget.value = parseFloat(Math.min(n, duration).toFixed(2));
                        if (endWidget && newEnd !== e) endWidget.value = parseFloat(newEnd.toFixed(2));
                    } else {
                        // end
                        if (n <= s || n > duration) { syncTimeInputs(); return; }
                        if (endWidget) endWidget.value = parseFloat(n.toFixed(2));
                    }
                    updateUI(true);
                    if (app.graph) app.graph.setDirtyCanvas(true, false);
                };

                startTimeBox.addEventListener("change", () => applyTimeBoxChange("start"));
                endTimeBox.addEventListener("change", () => applyTimeBoxChange("end"));
                durationBox.addEventListener("change", () => applyTimeBoxChange("duration"));

                audioEl.onloadedmetadata = () => {
                    duration = audioEl.duration;

                    // 切换新音频 -> 重置裁剪区间
                    if (node._should_reset_trim) {
                        if (startWidget) startWidget.value = 0;
                        if (endWidget) endWidget.value = parseFloat(duration.toFixed(2));
                        if (segmentIndexWidget) segmentIndexWidget.value = 0;
                        playBounds = null;
                        node._should_reset_trim = false;
                    } else {
                        // 初始加载或保存值越界时做默认钳制
                        let e = endWidget ? parseFloat(endWidget.value) || 0 : 0;
                        if (endWidget && (e === 0 || e > duration)) {
                            endWidget.value = parseFloat(duration.toFixed(2));
                        }
                    }

                    updateRuler();
                    updateUI();
                    if (app.graph) app.graph.setDirtyCanvas(true, false);
                };

                // 播放越过有效窗口时自动停止/回到起点
                audioEl.ontimeupdate = () => {
                    if (dragging || !duration) return;
                    const pb = getPlayBounds();
                    if (audioEl.currentTime >= pb.e) { audioEl.pause(); audioEl.currentTime = pb.s; }
                };

                audioEl.onplay = () => {
                    if (!duration) return;
                    const pb = getPlayBounds();
                    if (audioEl.currentTime < pb.s || audioEl.currentTime >= pb.e) { audioEl.currentTime = pb.s; }
                };

                [startWidget, endWidget].forEach(w => {
                    if (w) {
                        const orig = w.callback;
                        w.callback = function () { updateUI(true); if (orig) orig.apply(this, arguments); };
                    }
                });

                // 分段时长 / 分段索引变化时刷新小方格（V2 callback 路径）
                if (durationsWidget) {
                    const origDur = durationsWidget.callback;
                    durationsWidget.callback = function () {
                        if (origDur) origDur.apply(this, arguments);
                        if (isSmart) { renderSegBlocks(); if (app.graph) app.graph.setDirtyCanvas(true, false); }
                    };
                }
                if (segmentIndexWidget) {
                    const origIdx = segmentIndexWidget.callback;
                    segmentIndexWidget.callback = function () {
                        if (origIdx) origIdx.apply(this, arguments);
                        if (isSmart) { renderSegBlocks(); if (app.graph) app.graph.setDirtyCanvas(true, false); }
                    };
                }

                sliderBox.onpointerdown = (e) => {
                    if (!duration || isSmart) return; // 智能分段下窗口由数值控件调整
                    const rect = sliderBox.getBoundingClientRect();
                    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                    const val = (x / rect.width) * duration;
                    let s = startWidget ? parseFloat(startWidget.value) || 0 : 0;
                    let e_val = endWidget ? parseFloat(endWidget.value) || duration : duration;

                    // 手柄容差区（两侧各约 10px），优先"拖动手柄"而非"整体拖动"
                    const handleTolerance = (10 / rect.width) * duration;

                    if (val > s + handleTolerance && val < e_val - handleTolerance) {
                        dragging = 'center';
                        dragOffset = val - s;
                        dragSelectionWidth = e_val - s;
                    } else if (Math.abs(val - s) < Math.abs(val - e_val)) {
                        dragging = 'start';
                        if (startWidget) startWidget.value = parseFloat(Math.min(val, e_val).toFixed(2));
                    } else {
                        dragging = 'end';
                        if (endWidget) endWidget.value = parseFloat(Math.max(val, s).toFixed(2));
                    }
                    updateUI(true);
                    if (app.graph) app.graph.setDirtyCanvas(true, false);
                    try { sliderBox.setPointerCapture(e.pointerId); } catch (_) {}
                };

                sliderBox.onpointermove = (e) => {
                    if (!dragging || !duration) return;
                    const rect = sliderBox.getBoundingClientRect();
                    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                    const val = (x / rect.width) * duration;
                    if (dragging === 'start') {
                        let e_val = endWidget ? parseFloat(endWidget.value) || duration : duration;
                        if (startWidget) startWidget.value = parseFloat(Math.min(val, e_val).toFixed(2));
                    } else if (dragging === 'end') {
                        const s = startWidget ? parseFloat(startWidget.value) || 0 : 0;
                        if (endWidget) endWidget.value = parseFloat(Math.max(val, s).toFixed(2));
                    } else if (dragging === 'center') {
                        let newStart = val - dragOffset;
                        let newEnd = newStart + dragSelectionWidth;

                        // 边界钳制
                        if (newStart < 0) {
                            newStart = 0;
                            newEnd = dragSelectionWidth;
                        } else if (newEnd > duration) {
                            newEnd = duration;
                            newStart = duration - dragSelectionWidth;
                        }

                        if (startWidget) startWidget.value = parseFloat(newStart.toFixed(2));
                        if (endWidget) endWidget.value = parseFloat(newEnd.toFixed(2));
                    }
                    updateUI(true);
                    if (app.graph) app.graph.setDirtyCanvas(true, false);
                };

                sliderBox.onpointerup = (e) => { dragging = null; try { sliderBox.releasePointerCapture(e.pointerId); } catch (_) {} };

                // 输出模式切换
                function setSmartMode(smart) {
                    if (isSmart === smart) {
                        applyModeVisual();
                        return;
                    }
                    isSmart = smart;
                    if (outputModeWidget) outputModeWidget.value = smart ? "智能分段" : "自定义裁切";
                    node._lastModeValue = outputModeWidget ? outputModeWidget.value : null;
                    applyModeVisual();
                    if (smart) {
                        segStatus.textContent = "正在切分…";
                        renderSegBlocks();
                    } else {
                        segStatus.textContent = "";
                        playBounds = null;
                        updateUI();
                    }
                    node.syncLayoutToNode();
                    applyV3MinSize();
                    if (app.graph) app.graph.setDirtyCanvas(true, false);
                }
                segCustomBtn.onclick = () => setSmartMode(false);
                segSmartBtn.onclick = () => setSmartMode(true);

                // 初始状态应用（默认自定义裁切；若工作流保存为智能分段则恢复）
                applyModeVisual();
                if (isSmart) {
                    renderSegBlocks();
                }

                // 工作流加载/重连时由 onConfigure 调用，恢复按钮与小方格状态
                node.syncModeVisual = function () {
                    if (!node.widgets) return;
                    const savedSmart = !!(outputModeWidget && outputModeWidget.value === "智能分段");
                    if (savedSmart !== isSmart) {
                        isSmart = savedSmart;
                    }
                    applyModeVisual();
                    if (isSmart) {
                        renderSegBlocks();
                    } else {
                        segStatus.textContent = "";
                        updateUI();
                    }
                    if (app.graph) app.graph.setDirtyCanvas(true, false);
                };

                // V3：Vue 控件不走原生 callback，每帧轻量比对值，变化即同步
                node.pollWidgetValues = function () {
                    if (!node.widgets) return;
                    if (audioWidget && audioWidget.value !== node._lastAudioValue) {
                        const changed = node._lastAudioValue !== undefined && !node._initializing;
                        node._lastAudioValue = audioWidget.value;
                        if (changed) node._should_reset_trim = true;
                        updateAudio();
                    }
                    const modeVal = outputModeWidget ? outputModeWidget.value : null;
                    if (modeVal !== node._lastModeValue) {
                        node._lastModeValue = modeVal;
                        const smart = modeVal === "智能分段";
                        if (smart !== isSmart) {
                            isSmart = smart;
                            applyModeVisual();
                            if (smart) renderSegBlocks();
                            else { segStatus.textContent = ""; playBounds = null; updateUI(); }
                        }
                    }
                    if (isSmart) {
                        const durVal = durationsWidget ? durationsWidget.value : null;
                        if (durVal !== node._lastSegDurValue) {
                            node._lastSegDurValue = durVal;
                            renderSegBlocks();
                        }
                        const idxVal = segmentIndexWidget ? segmentIndexWidget.value : null;
                        if (idxVal !== node._lastSegIdxValue) {
                            node._lastSegIdxValue = idxVal;
                            renderSegBlocks();
                        }
                    }
                    removePlaceholderPorts(node);
                };
                node._lastModeValue = outputModeWidget ? outputModeWidget.value : null;
                node._lastSegDurValue = durationsWidget ? durationsWidget.value : null;
                node._lastSegIdxValue = segmentIndexWidget ? segmentIndexWidget.value : null;

                // 退出初始化阶段
                setTimeout(() => { node._initializing = false; }, 500);

                node.syncLayoutToNode();
                applyV3MinSize();
            }, 100);

            // V2：每帧把 DOM 高度拉回节点剩余高度，消除工作流重载导致的漂移；
            // 同时承载 V3 轮询（Vue 控件值变化同步）与 V3 最小尺寸兜底
            chainCallback(node, "onDrawForeground", function () {
                this.pollWidgetValues && this.pollWidgetValues();
                if (checkIsV3()) {
                    applyV3MinSize();
                    return;
                }
                if (this.domWidget && this.domWidget.element && this.domWidget.last_y) {
                    const remainingHeight = this.size[1] - this.domWidget.last_y - 18;
                    const currentHeight = parseFloat(this.domWidget.element.style.height);
                    const targetHeight = Math.max(DOM_H, remainingHeight);
                    if (isNaN(currentHeight) || Math.abs(currentHeight - targetHeight) > 1) {
                        this.domWidget.element.style.height = `${targetHeight}px`;
                    }
                }
            });
        });
    }
});
