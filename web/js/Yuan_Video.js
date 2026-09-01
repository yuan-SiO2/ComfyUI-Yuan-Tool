/**
 * Yuan Tool · 加载视频 UI 前端
 *
 * 复刻自 Yuan-TV 插件的「加载视频 UI」节点（节点键名/路由/扩展注册名均做隔离，
 * 可与源插件共存）。为 Yuan_VideoUI 节点提供内置视频预览、时间轴裁剪、
 * 镜头智能分段与裁剪框交互 UI。
 */
const { app } = window.comfyAPI.app;

/** 从 window.comfyAPI 获取 api 实例 */
function getApi() {
    try {
        const c = window.comfyAPI;
        if (c && c.api) {
            if (c.api.api && typeof c.api.api.apiURL === "function") return c.api.api;
            if (typeof c.api.apiURL === "function") return c.api.api;
        }
    } catch (_) {}
    return null;
}
const api = getApi();

function hideWidget(w) {
    if (!w) return;
    w.hidden = true;
    if (!w.options) w.options = {};
    w.options.hidden = true;

    if (!window.LiteGraph || !window.LiteGraph.vueNodesMode) {
        w.computeSize = () => [0, -4];
        if (!w._hiddenDrawHooked) {
            w._origDraw = w.hasOwnProperty('draw') ? w.draw : undefined;
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
    name: "Comfy.YuanTool.VideoUI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Yuan_VideoUI") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onConfigure = nodeType.prototype.onConfigure;
            const onResize = nodeType.prototype.onResize;
            const onDrawForeground = nodeType.prototype.onDrawForeground;

            // Hook into workflow loading to instantly restore the video UI
            nodeType.prototype.onConfigure = function (info) {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }

                // 恢复保存的显示模式对应的输入框（不换算/不覆盖数值）
                if (this.toggleWidgetVisibility) this.toggleWidgetVisibility();
                if (this.syncToggleVisual) this.syncToggleVisual();
                if (this.syncOutputToggleVisual) this.syncOutputToggleVisual();

                if (this.widgets) {
                    const videoWidget = this.widgets.find(w => w.name === "视频");
                    if (videoWidget && videoWidget.value && this.updatePreview) {
                        this.updatePreview(videoWidget.value);
                    }
                }

                if (this.syncLayoutToNode) {
                    setTimeout(() => {
                        this.syncLayoutToNode();
                    }, 0);
                }
            };

            // Continuous frame-accurate check to guarantee exact height alignment 
            // even on initial graph load when the workflow reloads!
            nodeType.prototype.onDrawForeground = function (ctx) {
                if (onDrawForeground) onDrawForeground.apply(this, arguments);

                if (this.domWidget && this.domWidget.element && this.domWidget.last_y) {
                    const remainingHeight = this.size[1] - this.domWidget.last_y - 18;
                    const currentHeight = parseFloat(this.domWidget.element.style.height);
                    const targetHeight = Math.max(150, remainingHeight);

                    // Only update DOM if the height has drifted by more than 1 pixel
                    if (isNaN(currentHeight) || Math.abs(currentHeight - targetHeight) > 1) {
                        this.domWidget.element.style.height = `${targetHeight}px`;
                    }
                }
            };

            // Allow the node to scale nicely when resized by the user
            nodeType.prototype.onResize = function (size) {
                if (onResize) onResize.apply(this, arguments);
                if (this.syncLayoutToNode) {
                    this.syncLayoutToNode();
                }
                if (this.domWidget && this.domWidget.element) {
                    this.domWidget.element.style.margin = "0";

                    // Fallback calc if last_y isn't ready
                    let yOffset = this.domWidget.last_y;
                    if (!yOffset) {
                        yOffset = 30; // Default LiteGraph Title Height
                        if (this.widgets) {
                            for (let w of this.widgets) {
                                if (w === this.domWidget) break;
                                yOffset += (w.computeSize ? w.computeSize()[1] : 20) + 4;
                            }
                        }
                    }

                    const remainingHeight = size[1] - yOffset - 18;
                    this.domWidget.element.style.height = `${Math.max(150, remainingHeight)}px`;
                }
            };

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this;

                node._initializing = true;
                node._should_reset_trim = false;

                // --- THE CORE FIX FOR COMFYUI V2 ---
                Object.defineProperty(node, 'imgs', {
                    get: function() { return undefined; },
                    set: function(val) { /* Ignore attempts by ComfyUI to set an image preview */ },
                    configurable: true
                });

                // Find the core widgets
                const videoWidget = this.widgets.find((w) => w.name === "视频");
                const frameRateWidget = this.widgets.find((w) => w.name === "帧率");
                const displayModeWidget = this.widgets.find((w) => w.name === "显示模式");

                const startTimeWidget = this.widgets.find((w) => w.name === "开始时间");
                const endTimeWidget = this.widgets.find((w) => w.name === "结束时间");
                const durationWidget = this.widgets.find((w) => w.name === "时长");

                const startFrameWidget = this.widgets.find((w) => w.name === "开始帧");
                const endFrameWidget = this.widgets.find((w) => w.name === "结束帧");
                const durationFramesWidget = this.widgets.find((w) => w.name === "时长帧数");

                const cropXWidget = this.widgets.find((w) => w.name === "裁剪X");
                const cropYWidget = this.widgets.find((w) => w.name === "裁剪Y");
                const cropWWidget = this.widgets.find((w) => w.name === "裁剪宽度");
                const cropHWidget = this.widgets.find((w) => w.name === "裁剪高度");

                const outputModeWidget = this.widgets.find((w) => w.name === "输出模式");
                const segmentIndexWidget = this.widgets.find((w) => w.name === "分段索引");

                // ====================================================================
                // WIDGET HIDING & SYNC ENGINE
                // ====================================================================
                let isSyncing = false;

                node.toggleWidgetVisibility = function () {
                    // 时间/帧 六 widget 全部迁移到 UI 时间轴左侧输入框显示，节点面板始终隐藏
                    hideWidget(startTimeWidget);
                    hideWidget(endTimeWidget);
                    hideWidget(durationWidget);
                    hideWidget(startFrameWidget);
                    hideWidget(endFrameWidget);
                    hideWidget(durationFramesWidget);
                    hideWidget(displayModeWidget); // driven by UI

                    hideWidget(cropXWidget);
                    hideWidget(cropYWidget);
                    hideWidget(cropWWidget);
                    hideWidget(cropHWidget);

                    // 智能分段模式下显示分段索引，自定义裁剪模式下隐藏
                    const isSmart = outputModeWidget && outputModeWidget.value === "智能分段输出";
                    if (isSmart) {
                        showWidget(segmentIndexWidget);
                    } else {
                        hideWidget(segmentIndexWidget);
                    }
                    hideWidget(outputModeWidget); // driven by UI

                    if (app.graph) {
                        app.graph.setDirtyCanvas(true, true);
                    }

                    // Allow the node to calculate its required min size, but DO NOT overwrite
                    // the current user-defined width/height unless it's strictly smaller than the minimum.
                    const minSize = node.computeSize();
                    node.size[0] = Math.max(node.size[0], minSize[0]);
                    node.size[1] = Math.max(node.size[1], minSize[1]);

                    if (node.onResize) node.onResize(node.size);
                    app.graph.setDirtyCanvas(true, true);
                };

                node.syncFramesFromTime = function () {
                    if (isSyncing || !frameRateWidget) return;
                    isSyncing = true;
                    const fr = frameRateWidget.value || 24;
                    if (startTimeWidget && startFrameWidget) startFrameWidget.value = Math.round(startTimeWidget.value * fr);
                    if (endTimeWidget && endFrameWidget) endFrameWidget.value = Math.round(endTimeWidget.value * fr);
                    if (durationWidget && durationFramesWidget) durationFramesWidget.value = Math.round(durationWidget.value * fr);
                    isSyncing = false;
                };

                node.syncTimeFromFrames = function () {
                    if (isSyncing || !frameRateWidget) return;
                    isSyncing = true;
                    const fr = frameRateWidget.value || 24;
                    if (startTimeWidget && startFrameWidget) startTimeWidget.value = parseFloat((startFrameWidget.value / fr).toFixed(3));
                    if (endTimeWidget && endFrameWidget) endTimeWidget.value = parseFloat((endFrameWidget.value / fr).toFixed(3));
                    if (durationWidget && durationFramesWidget) durationFramesWidget.value = parseFloat((durationFramesWidget.value / fr).toFixed(3));
                    isSyncing = false;
                };

                // Bind standard input callbacks to synchronize automatically
                function bindWidget(w, isFrame, isFrameRate = false) {
                    if (!w) return;
                    const orig = w.callback;
                    w.callback = function () {
                        if (orig) orig.apply(this, arguments);
                        if (isFrame) node.syncTimeFromFrames();
                        else node.syncFramesFromTime();

                        // Always force a ruler update if framerate changes so the timeline marks match the new rate
                        if (duration === 0 || isFrameRate) updateRuler();
                        updateUI(true);
                    };
                }

                bindWidget(startTimeWidget, false);
                bindWidget(endTimeWidget, false);
                bindWidget(startFrameWidget, true);
                bindWidget(endFrameWidget, true);
                bindWidget(frameRateWidget, false, true); // Triggers re-sync of frames from time AND updates ruler

                // Bind update function to the node so onConfigure can access it
                node.updatePreview = function (filename) {
                    if (!filename || filename === "无") {
                        return;
                    }
                    let url;

                    // Check if absolute path (Starts with C:\ or /)
                    if (filename.match(/^[a-zA-Z]:\\/) || filename.startsWith('/')) {
                        url = api.apiURL(`/yuan_tool/video_custom_view?filename=${encodeURIComponent(filename)}`);
                    } else {
                        url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input`);
                    }

                    if (videoPreview) videoPreview.src = url;
                };

                if (videoWidget) {
                    const originalCallback = videoWidget.callback;
                    videoWidget.callback = function () {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        if (!node._initializing) {
                            node._should_reset_trim = true;
                        }
                        if (node.updatePreview) node.updatePreview(this.value);
                    };
                }

                // Initialize widget visibility right away
                if (displayModeWidget && !displayModeWidget.value) displayModeWidget.value = "秒";
                node.toggleWidgetVisibility();

                // ====================================================================
                // CHOOSE FILE BUTTON (Native ComfyUI Widget, placed below duration)
                // ====================================================================
                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.accept = "video/*";
                fileInput.style.display = "none";
                document.body.appendChild(fileInput);

                const btnWidget = this.addWidget("button", "选择视频上传", null, () => {
                    fileInput.click();
                });

                // Define robust upload logic
                const uploadFile = async (file) => {
                    try {
                        if (errorMsg) errorMsg.style.display = "none";

                        // Fast Path: If desktop environment exposes absolute file path, skip upload entirely!
                        if (file.path) {
                            if (videoWidget.options && videoWidget.options.values && !videoWidget.options.values.includes(file.path)) {
                                videoWidget.options.values.push(file.path);
                            }
                            videoWidget.value = file.path;
                            node._should_reset_trim = true;
                            node.updatePreview(file.path);
                            node.syncFramesFromTime();
                            return;
                        }

                        // First check if the file already exists on the server to de-duplicate
                        const safeFileName = file.name.replace(/[^a-zA-Z0-9.\-_]/g, '_');
                        try {
                            const checkResp = await api.fetchApi(`/yuan_tool/video_check_file?filename=${encodeURIComponent(safeFileName)}&size=${file.size}`);
                            if (checkResp.status === 200) {
                                const checkResult = await checkResp.json();
                                if (checkResult.exists) {
                                    console.log(`[YuanVideoUI] File already exists: ${checkResult.name}. Reusing existing file.`);
                                    if (videoWidget.options && videoWidget.options.values && !videoWidget.options.values.includes(checkResult.name)) {
                                        videoWidget.options.values.push(checkResult.name);
                                    }
                                    videoWidget.value = checkResult.name;
                                    node._should_reset_trim = true;
                                    node.updatePreview(checkResult.name);
                                    node.syncFramesFromTime();
                                    return;
                                }
                            }
                        } catch (e) {
                            console.warn("[YuanVideoUI] Failed to check for existing file, proceeding with upload", e);
                        }

                        btnWidget.name = "上传中...";
                        node.setDirtyCanvas(true, false);

                        const CHUNK_SIZE = 10 * 1024 * 1024; // 10MB chunks

                        if (file.size > CHUNK_SIZE) {
                            const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
                            const safeName = Date.now() + "_" + safeFileName;

                            for (let i = 0; i < totalChunks; i++) {
                                btnWidget.name = `上传中... ${Math.round((i / totalChunks) * 100)}%`;
                                node.setDirtyCanvas(true, false);

                                const chunk = file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE);

                                const formData = new FormData();
                                formData.append("file", chunk);
                                formData.append("filename", safeName);
                                formData.append("chunk_index", i);
                                formData.append("total_chunks", totalChunks);

                                const resp = await api.fetchApi("/yuan_tool/video_upload_chunk", {
                                    method: "POST",
                                    body: formData,
                                });

                                if (resp.status !== 200) {
                                    throw new Error("Chunk upload failed");
                                }

                                if (i === totalChunks - 1) {
                                    const data = await resp.json();
                                    if (videoWidget.options && videoWidget.options.values && !videoWidget.options.values.includes(data.name)) {
                                        videoWidget.options.values.push(data.name);
                                    }
                                    videoWidget.value = data.name;
                                    node._should_reset_trim = true;
                                    node.updatePreview(data.name);
                                    node.syncFramesFromTime();
                                }
                            }
                        } else {
                            // Standard upload for small files
                            const body = new FormData();
                            body.append("image", file);

                            const resp = await api.fetchApi("/upload/image", {
                                method: "POST",
                                body: body,
                              // No subfolder param -> goes to input root!
                            });

                            if (resp.status === 413) {
                                throw new Error("File too large. Make sure python backend has the chunking update.");
                            }

                            if (resp.status === 200) {
                                const data = await resp.json();
                                if (videoWidget.options && videoWidget.options.values && !videoWidget.options.values.includes(data.name)) {
                                    videoWidget.options.values.push(data.name);
                                }
                                videoWidget.value = data.name;
                                node._should_reset_trim = true;
                                node.updatePreview(data.name);
                                node.syncFramesFromTime();
                            } else {
                                throw new Error(`Upload failed: ${resp.statusText}`);
                            }
                        }
                    } catch (error) {
                        console.error("Upload failed", error);
                        if (errorMsg) {
                            errorMsg.textContent = "上传失败，请查看控制台。";
                            errorMsg.style.display = "block";
                        }
                    } finally {
                        btnWidget.name = "选择视频上传";
                        node.setDirtyCanvas(true, false);
                        fileInput.value = ""; // reset input
                    }
                };

                fileInput.addEventListener("change", (e) => {
                    if (e.target.files.length) {
                        uploadFile(e.target.files[0]);
                    }
                });

                // Attach drag & drop directly onto the LiteGraph node canvas frame
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
                        if (file.type.startsWith('video/') || file.name.toLowerCase().match(/\.(mp4|webm|mkv|avi|mov|m4v|flv|wmv)$/)) {
                            uploadFile(file);
                            return true;
                        }
                    }
                    return false;
                };

                node.onDropFile = function (file) {
                    // Check MIME type or common video file extensions to ensure all videos are caught
                    if (file.type.startsWith('video/') || file.name.toLowerCase().match(/\.(mp4|webm|mkv|avi|mov|m4v|flv|wmv)$/)) {
                        uploadFile(file);
                        return true;
                    }
                    return false;
                };

                // Clean up DOM elements strictly tied to this node instance
                const originalOnRemove = node.onRemoved;
                node.onRemoved = function () {
                    if (fileInput && fileInput.parentNode) fileInput.parentNode.removeChild(fileInput);
                    if (originalOnRemove) originalOnRemove.apply(this, arguments);
                };

                // ====================================================================
                // UI CONTAINER (Preview & Timeline Editor)
                // ====================================================================
                const container = document.createElement("div");
                const defaultBg = "rgba(30, 30, 30, 0.9)";
                Object.assign(container.style, {
                    display: "flex",
                    flexDirection: "column",
                    gap: "10px",
                    width: "100%",
                    margin: "0",
                    padding: "10px",
                    boxSizing: "border-box",
                    background: defaultBg,
                    borderRadius: "6px",
                    color: "white",
                    fontFamily: "sans-serif",
                    marginTop: "8px",
                    flexShrink: "0",
                    transition: "background 0.2s"
                });

                const errorMsg = document.createElement("div");
                Object.assign(errorMsg.style, {
                    color: "#ff6b6b",
                    fontSize: "12px",
                    display: "none",
                    marginBottom: "4px",
                    flexShrink: "0",
                    boxSizing: "border-box"
                });
                container.appendChild(errorMsg);

                // Top Bar: Display Mode Toggle & Trimmed Length
                const playerTop = document.createElement("div");
                Object.assign(playerTop.style, {
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "0 2px",
                    marginBottom: "-4px",
                    flexShrink: "0",
                    boxSizing: "border-box",
                    flexWrap: "wrap", // Prevent squishing/overflow by letting it wrap gracefully
                    gap: "6px",
                    position: "relative"
                });

                // Toggle Container UI
                const toggleWrapper = document.createElement("div");
                Object.assign(toggleWrapper.style, {
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    background: "rgba(0, 0, 0, 0.2)",
                    padding: "0 8px",
                    borderRadius: "4px",
                    height: "22px",
                    boxSizing: "border-box"
                });

                // Segmented pill control
                const segmentedToggle = document.createElement("div");
                Object.assign(segmentedToggle.style, {
                    display: "flex",
                    alignItems: "center",
                    background: "rgba(0, 0, 0, 0.35)",
                    border: "1px solid rgba(56, 189, 248, 0.3)",
                    borderRadius: "4px",
                    overflow: "hidden",
                    height: "18px",
                    flexShrink: "0",
                    cursor: "pointer"
                });

                const createSegBtn = (label) => {
                    const btn = document.createElement("span");
                    btn.textContent = label;
                    Object.assign(btn.style, {
                        fontSize: "11px",
                        fontWeight: "bold",
                        padding: "0 8px",
                        lineHeight: "18px",
                        color: "rgba(255,255,255,0.45)",
                        background: "transparent",
                        transition: "background 0.2s, color 0.2s",
                        userSelect: "none",
                        whiteSpace: "nowrap"
                    });
                    return btn;
                };

                const segTime = createSegBtn("时间");
                const segDivider = document.createElement("span");
                segDivider.style.cssText = "width:1px;height:12px;background:rgba(56,189,248,0.25);flex-shrink:0;";
                const segFrames = createSegBtn("帧");

                segmentedToggle.appendChild(segTime);
                segmentedToggle.appendChild(segDivider);
                segmentedToggle.appendChild(segFrames);

                const applySegmentState = (frames) => {
                    if (frames) {
                        segTime.style.background = "transparent";
                        segTime.style.color = "rgba(255,255,255,0.45)";
                        segFrames.style.background = "rgba(37,126,235,0.85)";
                        segFrames.style.color = "#fff";
                    } else {
                        segTime.style.background = "rgba(56,189,248,0.85)";
                        segTime.style.color = "#fff";
                        segFrames.style.background = "transparent";
                        segFrames.style.color = "rgba(255,255,255,0.45)";
                    }
                };

                // Keep a reference so the init block below can call it
                let isFramesMode = false;
                applySegmentState(false); // Default: Time is active

                // 统一模式切换入口：UI 按钮与节点面板"显示模式"下拉共用。
                // 切换只改变 UI 输入框的标签描述与显示值（时间组 <-> 帧组），
                // 不换算/不覆盖数值——两组各自独立，"是什么就是什么"。
                const applyDisplayMode = (nextFramesMode) => {
                    isFramesMode = nextFramesMode;
                    applySegmentState(isFramesMode);

                    if (displayModeWidget) displayModeWidget.value = isFramesMode ? "帧" : "秒";

                    node.toggleWidgetVisibility();
                    updateTimeInputMode();
                    updateRuler();
                    updateUI(true);
                };

                const doToggle = () => applyDisplayMode(!isFramesMode);

                segmentedToggle.onclick = doToggle;

                // Expose the switch activation so the init requestAnimationFrame below can call it
                const switchBox = { onclick: doToggle };

                // 节点面板"显示模式"下拉直接切换时，同样触发同步
                if (displayModeWidget) {
                    const origDmCb = displayModeWidget.callback;
                    displayModeWidget.callback = function () {
                        if (origDmCb) origDmCb.apply(this, arguments);
                        applyDisplayMode(this.value === "帧");
                    };
                }

                // Allow onConfigure (workflow reload) to re-sync the visual highlight
                node.syncToggleVisual = function () {
                    const savedIsFrames = displayModeWidget && displayModeWidget.value === "帧";
                    isFramesMode = savedIsFrames;
                    applySegmentState(savedIsFrames);
                    // 同步刷新 UI 输入框的标签描述与显示值（时间组/帧组各自独立）
                    updateTimeInputMode();
                };

                toggleWrapper.appendChild(segmentedToggle);

                const leftContainer = document.createElement("div");
                Object.assign(leftContainer.style, {
                    flex: "1 1 0%",
                    display: "flex",
                    justifyContent: "flex-start",
                    minWidth: "max-content"
                });
                leftContainer.appendChild(toggleWrapper);

                // ---------- 输出模式切换按钮（自定义裁剪输出 / 智能分段输出） ----------
                const outputToggleWrapper = document.createElement("div");
                Object.assign(outputToggleWrapper.style, {
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    background: "rgba(0, 0, 0, 0.2)",
                    padding: "0 8px",
                    borderRadius: "4px",
                    height: "22px",
                    boxSizing: "border-box",
                    marginLeft: "8px"
                });

                const outputSegmentedToggle = document.createElement("div");
                Object.assign(outputSegmentedToggle.style, {
                    display: "flex",
                    alignItems: "center",
                    background: "rgba(0, 0, 0, 0.35)",
                    border: "1px solid rgba(56, 189, 248, 0.3)",
                    borderRadius: "4px",
                    overflow: "hidden",
                    height: "18px",
                    flexShrink: "0",
                    cursor: "pointer"
                });

                const segCustom = createSegBtn("自定义裁剪");
                const segDivider2 = document.createElement("span");
                segDivider2.style.cssText = "width:1px;height:12px;background:rgba(56,189,248,0.25);flex-shrink:0;";
                const segSmart = createSegBtn("智能分段");

                outputSegmentedToggle.appendChild(segCustom);
                outputSegmentedToggle.appendChild(segDivider2);
                outputSegmentedToggle.appendChild(segSmart);

                const applyOutputSegState = (smart) => {
                    if (smart) {
                        segCustom.style.background = "transparent";
                        segCustom.style.color = "rgba(255,255,255,0.45)";
                        segSmart.style.background = "rgba(37,126,235,0.85)";
                        segSmart.style.color = "#fff";
                    } else {
                        segCustom.style.background = "rgba(56,189,248,0.85)";
                        segCustom.style.color = "#fff";
                        segSmart.style.background = "transparent";
                        segSmart.style.color = "rgba(255,255,255,0.45)";
                    }
                };

                let isSmartMode = false;
                applyOutputSegState(false); // 默认：自定义裁剪输出

                // 检测分段按钮：位于“输出模式”右侧，仅智能分段模式下显示
                const detectBtn = document.createElement("button");
                detectBtn.textContent = "检测分段";
                Object.assign(detectBtn.style, {
                    display: "none",
                    background: "rgba(56, 189, 248, 0.85)",
                    color: "#fff", border: "none", borderRadius: "4px",
                    padding: "0 10px", height: "18px", fontSize: "12px", fontWeight: "bold",
                    cursor: "pointer", flexShrink: "0"
                });

                const doOutputToggle = () => {
                    isSmartMode = !isSmartMode;
                    applyOutputSegState(isSmartMode);

                    if (outputModeWidget) outputModeWidget.value = isSmartMode ? "智能分段输出" : "自定义裁剪输出";
                    detectBtn.style.display = isSmartMode ? "block" : "none";
                    node.toggleWidgetVisibility();
                    app.graph.setDirtyCanvas(true, false);
                    // 切到智能分段时自动恢复已保存的分段（后端缓存命中则秒回，不会重新分割）
                    if (isSmartMode && node.tryRestoreCachedSegments) node.tryRestoreCachedSegments();
                    // 时间轴显示：自定义裁剪 = 长条，智能分段 = 小方块
                    applyOutputTimelineVisual(isSmartMode);
                };

                outputSegmentedToggle.onclick = doOutputToggle;

                const outputSwitchBox = { onclick: doOutputToggle };

                // 工作流加载时恢复输出模式的高亮状态
                node.syncOutputToggleVisual = function () {
                    const savedIsSmart = outputModeWidget && outputModeWidget.value === "智能分段输出";
                    isSmartMode = savedIsSmart;
                    applyOutputSegState(savedIsSmart);
                    detectBtn.style.display = savedIsSmart ? "block" : "none";
                    // 工作流加载后自动恢复已保存的分段（仅查缓存，不会重新分割）
                    if (savedIsSmart && node.tryRestoreCachedSegments) node.tryRestoreCachedSegments();
                    // 时间轴显示：自定义裁剪 = 长条，智能分段 = 小方块
                    applyOutputTimelineVisual(savedIsSmart);
                };

                // 直接通过原生下拉框切换输出模式时，也同步按钮高亮与检测按钮显隐
                if (outputModeWidget) {
                    const origOutCb = outputModeWidget.callback;
                    outputModeWidget.callback = function () {
                        if (origOutCb) origOutCb.apply(this, arguments);
                        node.syncOutputToggleVisual();
                    };
                }

                outputToggleWrapper.appendChild(outputSegmentedToggle);
                outputToggleWrapper.appendChild(detectBtn);
                leftContainer.appendChild(outputToggleWrapper);

                playerTop.appendChild(leftContainer);

                const cropBtn = document.createElement("button");
                cropBtn.textContent = "裁剪";
                Object.assign(cropBtn.style, {
                    background: "rgba(255, 255, 255, 0.1)",
                    color: "white",
                    border: "none",
                    borderRadius: "4px",
                    padding: "0 8px",
                    height: "22px",
                    fontSize: "12px",
                    fontWeight: "bold",
                    cursor: "pointer"
                });

                let isCropVisible = false;

                const cropUIContainer = document.createElement("div");
                Object.assign(cropUIContainer.style, {
                    display: "flex", alignItems: "center", gap: "6px", zIndex: "11"
                });

                const cropDims = document.createElement("span");
                Object.assign(cropDims.style, {
                    fontSize: "12px", color: "#38bdf8", fontWeight: "bold",
                    display: "none", padding: "0 6px", pointerEvents: "none"
                });

                const cropEditContainer = document.createElement("div");
                Object.assign(cropEditContainer.style, {
                    display: "none", alignItems: "center", gap: "4px"
                });

                const arSelect = document.createElement("select");
                Object.assign(arSelect.style, {
                    background: "#222", color: "#fff", border: "1px solid #555",
                    borderRadius: "3px", fontSize: "12px", padding: "2px", outline: "none",
                    cursor: "pointer"
                });
                const ratios = [
                    { name: "自由", val: 0 },
                    { name: "原始", val: -1 },
                    { name: "1:1", val: 1 },
                    { name: "4:5", val: 4 / 5 },
                    { name: "5:4", val: 5 / 4 },
                    { name: "16:9", val: 16 / 9 },
                    { name: "9:16", val: 9 / 16 },
                    { name: "4:3", val: 4 / 3 },
                    { name: "3:4", val: 3 / 4 },
                    { name: "3:2", val: 3 / 2 },
                    { name: "2:3", val: 2 / 3 },
                    { name: "2:1", val: 2 },
                    { name: "1:2", val: 1 / 2 }
                ];
                ratios.forEach(r => {
                    const opt = document.createElement("option");
                    opt.textContent = r.name;
                    opt.value = r.val;
                    arSelect.appendChild(opt);
                });

                const wInput = document.createElement("input");
                const hInput = document.createElement("input");
                const inputStyle = {
                    width: "40px", background: "rgba(0,0,0,0.5)", color: "#38bdf8",
                    border: "1px solid #555", borderRadius: "3px", fontSize: "12px",
                    textAlign: "center", padding: "2px", outline: "none"
                };
                Object.assign(wInput.style, inputStyle);
                Object.assign(hInput.style, inputStyle);
                wInput.type = "text";
                hInput.type = "text";

                const xSpan = document.createElement("span");
                xSpan.textContent = "x";
                xSpan.style.color = "#888";
                xSpan.style.fontSize = "12px";

                cropEditContainer.appendChild(arSelect);
                cropEditContainer.appendChild(wInput);
                cropEditContainer.appendChild(xSpan);
                cropEditContainer.appendChild(hInput);

                cropUIContainer.appendChild(cropDims);
                cropUIContainer.appendChild(cropEditContainer);
                playerTop.appendChild(cropUIContainer);

                const rightContainer = document.createElement("div");
                Object.assign(rightContainer.style, {
                    flex: "1 1 0%",
                    display: "flex",
                    justifyContent: "flex-end",
                    gap: "6px",
                    minWidth: "max-content"
                });
                // 时间输入框：样式为「口-口 口」，对应「开始时间-结束时间 时长」。
                // 只显示输入框，不显示文字标签；悬停输入框可查看说明（开始时间/结束时间/时长，单位秒）。
                const timeInputWrap = document.createElement("div");
                Object.assign(timeInputWrap.style, {
                    display: "flex",
                    alignItems: "center",
                    gap: "2px",
                    minWidth: "max-content"
                });

                const mkTimeBox = (tip) => {
                    const inp = document.createElement("input");
                    inp.type = "number";
                    inp.step = "0.01";
                    inp.min = "0";
                    inp.title = tip;
                    Object.assign(inp.style, {
                        width: "58px",
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
                timeGap.style.width = "8px";

                // 框旁边不显示文字标签，只显示框；说明放在鼠标悬停提示（title）里
                timeInputWrap.appendChild(startTimeBox);
                timeInputWrap.appendChild(timeDash);
                timeInputWrap.appendChild(endTimeBox);
                timeInputWrap.appendChild(timeGap);
                timeInputWrap.appendChild(durationBox);

                // widget 值变化时回写输入框（按当前显示模式取对应组的值）
                node.syncTimeInputs = function () {
                    const isFrames = isFramesMode;
                    if (isFrames) {
                        if (startFrameWidget) startTimeBox.value = parseFloat(startFrameWidget.value) || 0;
                        if (endFrameWidget) endTimeBox.value = parseFloat(endFrameWidget.value) || 0;
                        if (durationFramesWidget) durationBox.value = parseFloat(durationFramesWidget.value) || 0;
                    } else {
                        if (startTimeWidget) startTimeBox.value = parseFloat(startTimeWidget.value) || 0;
                        if (endTimeWidget) endTimeBox.value = parseFloat(endTimeWidget.value) || 0;
                        if (durationWidget) durationBox.value = parseFloat(durationWidget.value) || 0;
                    }
                };

                // 切换时间/帧：更新输入框的悬停说明与显示值（不换算，各自独立）
                const updateTimeInputMode = () => {
                    const isFrames = isFramesMode;
                    startTimeBox.title = isFrames ? "开始帧" : "开始时间（秒）";
                    endTimeBox.title = isFrames ? "结束帧" : "结束时间（秒）";
                    durationBox.title = isFrames ? "时长帧数" : "时长（秒）";
                    node.syncTimeInputs();
                };

                // 输入框变化时写入当前显示模式对应的 widget，不换算另一组。
                // 时长输入框：仅在【当前组内】调整结束值（结束=开始+时长），绝不跨组换算。
                const applyTimeBoxChange = (role) => {
                    const isFrames = isFramesMode;
                    const fr = frameRateWidget ? (frameRateWidget.value || 24) : 24;
                    const pick = (r) => isFrames
                        ? (r === "start" ? startFrameWidget : r === "end" ? endFrameWidget : durationFramesWidget)
                        : (r === "start" ? startTimeWidget : r === "end" ? endTimeWidget : durationWidget);
                    const startW = pick("start");
                    const endW = pick("end");
                    const durW = pick("duration");
                    const box = role === "start" ? startTimeBox : role === "end" ? endTimeBox : durationBox;
                    if (!durW) return;
                    const n = parseFloat(box.value);
                    if (isNaN(n) || n < 0) { box.value = durW.value; return; }

                    if (role === "duration") {
                        // 时长语义：结束 = 开始 + 时长（仅当前组内，可自动回推开始）
                        const activeScale = isFrames ? Math.round(getActiveDuration() * fr) : getActiveDuration();
                        const d = Math.min(n, activeScale);
                        const s0 = startW ? parseFloat(startW.value) || 0 : 0;
                        let newStart = s0;
                        let newEnd = s0 + d;
                        if (newEnd > activeScale) {
                            newEnd = activeScale;
                            newStart = activeScale - d;
                        }
                        if (startW) startW.value = newStart;
                        if (endW) endW.value = newEnd;
                        if (durW) durW.value = d;
                    } else {
                        const w = role === "start" ? startW : endW;
                        if (!w) return;
                        if (parseFloat(w.value) === n) return;
                        w.value = n;
                    }

                    if (duration === 0) updateRuler();
                    updateUI(true);
                    if (app.graph) app.graph.setDirtyCanvas(true, false);
                };
                startTimeBox.addEventListener("change", () => applyTimeBoxChange("start"));
                endTimeBox.addEventListener("change", () => applyTimeBoxChange("end"));
                durationBox.addEventListener("change", () => applyTimeBoxChange("duration"));

                rightContainer.appendChild(timeInputWrap);
                rightContainer.appendChild(cropBtn);
                playerTop.appendChild(rightContainer);

                let currentAspectRatio = 0;

                const handleManualDimensionInput = (isWidth) => {
                    const vw = videoPreview.videoWidth;
                    const vh = videoPreview.videoHeight;
                    if (!vw || !vh) return;

                    let newW = parseInt(wInput.value) || Math.round((cropWWidget ? parseFloat(cropWWidget.value) || 1 : 1) * vw);
                    let newH = parseInt(hInput.value) || Math.round((cropHWidget ? parseFloat(cropHWidget.value) || 1 : 1) * vh);

                    if (currentAspectRatio > 0) {
                        if (isWidth) {
                            newH = Math.round(newW / currentAspectRatio);
                        } else {
                            newW = Math.round(newH * currentAspectRatio);
                        }
                    }

                    newW = Math.max(1, Math.min(newW, vw));
                    newH = Math.max(1, Math.min(newH, vh));

                    let cw_val = newW / vw;
                    let ch_val = newH / vh;

                    let cx = cropXWidget ? parseFloat(cropXWidget.value) || 0 : 0;
                    let cy = cropYWidget ? parseFloat(cropYWidget.value) || 0 : 0;

                    if (cx + cw_val > 1) cx = 1 - cw_val;
                    if (cy + ch_val > 1) cy = 1 - ch_val;

                    if (cropXWidget) cropXWidget.value = parseFloat(cx.toFixed(3));
                    if (cropYWidget) cropYWidget.value = parseFloat(cy.toFixed(3));
                    if (cropWWidget) cropWWidget.value = parseFloat(cw_val.toFixed(3));
                    if (cropHWidget) cropHWidget.value = parseFloat(ch_val.toFixed(3));

                    updateCropUI();
                    app.graph.setDirtyCanvas(true, false);
                };

                wInput.addEventListener("change", () => handleManualDimensionInput(true));
                hInput.addEventListener("change", () => handleManualDimensionInput(false));
                wInput.addEventListener("keydown", (e) => { if (e.key === "Enter") handleManualDimensionInput(true); });
                hInput.addEventListener("keydown", (e) => { if (e.key === "Enter") handleManualDimensionInput(false); });

                arSelect.onchange = () => {
                    currentAspectRatio = parseFloat(arSelect.value);
                    if (currentAspectRatio === -1 && videoPreview.videoWidth) {
                        currentAspectRatio = videoPreview.videoWidth / videoPreview.videoHeight;
                    }
                    if (currentAspectRatio > 0 && videoPreview.videoWidth) {
                        const vw = videoPreview.videoWidth;
                        const vh = videoPreview.videoHeight;
                        let cw_val = cropWWidget ? parseFloat(cropWWidget.value) || 1 : 1;
                        let cx = cropXWidget ? parseFloat(cropXWidget.value) || 0 : 0;
                        let cy = cropYWidget ? parseFloat(cropYWidget.value) || 0 : 0;

                        const actualW = cw_val * vw;
                        let actualH = actualW / currentAspectRatio;
                        let ch_val = actualH / vh;

                        if (ch_val > 1) {
                            ch_val = 1;
                            const newActualW = vh * currentAspectRatio;
                            cw_val = newActualW / vw;
                        }
                        if (cy + ch_val > 1) cy = 1 - ch_val;
                        if (cx + cw_val > 1) cx = 1 - cw_val;

                        if (cropXWidget) cropXWidget.value = parseFloat(cx.toFixed(3));
                        if (cropYWidget) cropYWidget.value = parseFloat(cy.toFixed(3));
                        if (cropWWidget) cropWWidget.value = parseFloat(cw_val.toFixed(3));
                        if (cropHWidget) cropHWidget.value = parseFloat(ch_val.toFixed(3));

                        updateCropUI();
                        app.graph.setDirtyCanvas(true, false);
                    }
                };

                cropBtn.onclick = () => {
                    isCropVisible = !isCropVisible;
                    cropBtn.style.background = isCropVisible ? "#38bdf8" : "rgba(255, 255, 255, 0.1)";
                    cropBtn.style.color = isCropVisible ? "black" : "white";
                    if (isCropVisible) {
                        cropBox.style.display = "block";
                        cropEditContainer.style.display = "flex";
                        cropDims.style.display = "none";
                    } else {
                        cropBox.style.display = "none";
                        cropEditContainer.style.display = "none";
                        // updateCropUI handles cropDims visibility when off
                    }
                    if (isCropVisible) {
                        videoPreview.pause();
                        videoPreview.controls = false;
                    } else {
                        videoPreview.controls = true;
                    }
                    updateCropUI();
                };

                container.appendChild(playerTop);

                // Video Preview Area (Native Controls)
                const videoWrapper = document.createElement("div");
                Object.assign(videoWrapper.style, {
                    position: "relative",
                    width: "100%",
                    flexGrow: "1",
                    minHeight: "0px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "#000",
                    borderRadius: "4px",
                    overflow: "hidden"
                });

                const videoPreview = document.createElement("video");
                Object.assign(videoPreview.style, {
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                    outline: "none",
                    boxSizing: "border-box"
                });
                videoPreview.controls = true;
                videoPreview.controlsList = "nodownload nofullscreen noremoteplayback";
                // 只加载元数据，不让浏览器预载整段源视频（否则硬件解码整片，占用 GPU/内存）
                videoPreview.preload = "metadata";
                // 静音：避免音频解码开销，也允许浏览器以更低开销渲染
                videoPreview.muted = true;
                videoWrapper.appendChild(videoPreview);

                const cropBox = document.createElement("div");
                Object.assign(cropBox.style, {
                    position: "absolute",
                    border: "2px dashed #38bdf8",
                    display: "none",
                    pointerEvents: "auto",
                    cursor: "move",
                    boxSizing: "border-box",
                    boxShadow: "0 0 0 9999px rgba(0, 0, 0, 0.5)",
                    zIndex: "10",
                    overflow: "hidden"
                });

                // 3x3 Grid lines
                for (let i = 1; i <= 2; i++) {
                    const vLine = document.createElement("div");
                    Object.assign(vLine.style, {
                        position: "absolute", left: `${i * 33.33}%`, top: "0", bottom: "0",
                        borderLeft: "1px dashed rgba(255,255,255,0.3)", pointerEvents: "none"
                    });
                    const hLine = document.createElement("div");
                    Object.assign(hLine.style, {
                        position: "absolute", top: `${i * 33.33}%`, left: "0", right: "0",
                        borderTop: "1px dashed rgba(255,255,255,0.3)", pointerEvents: "none"
                    });
                    cropBox.appendChild(vLine);
                    cropBox.appendChild(hLine);
                }

                const createCropHandle = (cursor, pos, borders) => {
                    const h = document.createElement("div");
                    Object.assign(h.style, {
                        position: "absolute",
                        width: "20px",
                        height: "20px",
                        background: "transparent",
                        cursor: cursor,
                        pointerEvents: "auto",
                        ...borders,
                        ...pos
                    });
                    return h;
                };

                const tlHandle = createCropHandle("nwse-resize", { top: "-3px", left: "-3px" }, { borderTop: "6px solid #38bdf8", borderLeft: "6px solid #38bdf8" });
                const trHandle = createCropHandle("nesw-resize", { top: "-3px", right: "-3px" }, { borderTop: "6px solid #38bdf8", borderRight: "6px solid #38bdf8" });
                const blHandle = createCropHandle("nesw-resize", { bottom: "-3px", left: "-3px" }, { borderBottom: "6px solid #38bdf8", borderLeft: "6px solid #38bdf8" });
                const brHandle = createCropHandle("nwse-resize", { bottom: "-3px", right: "-3px" }, { borderBottom: "6px solid #38bdf8", borderRight: "6px solid #38bdf8" });

                const tmHandle = createCropHandle("ns-resize", { top: "-3px", left: "50%", transform: "translateX(-50%)" }, { borderTop: "6px solid #38bdf8", width: "16px", height: "10px" });
                const bmHandle = createCropHandle("ns-resize", { bottom: "-3px", left: "50%", transform: "translateX(-50%)" }, { borderBottom: "6px solid #38bdf8", width: "16px", height: "10px" });
                const lmHandle = createCropHandle("ew-resize", { top: "50%", left: "-3px", transform: "translateY(-50%)" }, { borderLeft: "6px solid #38bdf8", width: "10px", height: "16px" });
                const rmHandle = createCropHandle("ew-resize", { top: "50%", right: "-3px", transform: "translateY(-50%)" }, { borderRight: "6px solid #38bdf8", width: "10px", height: "16px" });

                const handles = [tlHandle, trHandle, blHandle, brHandle, tmHandle, bmHandle, lmHandle, rmHandle];
                handles.forEach(h => cropBox.appendChild(h));
                videoWrapper.appendChild(cropBox);

                container.appendChild(videoWrapper);

                // Trim Area (Time Ruler & Slider)
                const trimArea = document.createElement("div");
                Object.assign(trimArea.style, {
                    display: "flex",
                    flexDirection: "column",
                    gap: "6px",
                    background: "rgba(0, 0, 0, 0.35)",
                    padding: "12px",
                    borderRadius: "6px",
                    border: "1px solid rgba(255, 255, 255, 0.05)",
                    flexShrink: "0", // Prevent timeline from squishing when shrinking node
                    boxSizing: "border-box"
                });

                const timeRuler = document.createElement("div");
                Object.assign(timeRuler.style, {
                    position: "relative",
                    width: "100%",
                    height: "22px",
                    fontSize: "11px",
                    color: "#aaa",
                    pointerEvents: "none",
                    userSelect: "none",
                    boxSizing: "border-box"
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
                    boxShadow: "inset 0 1px 3px rgba(0,0,0,0.5)",
                    boxSizing: "border-box"
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

                // 分段小方块图层：叠加在裁剪时间轴上，按时间位置显示检测到的分段
                const segBlocksLayer = document.createElement("div");
                Object.assign(segBlocksLayer.style, {
                    position: "absolute",
                    left: "0", top: "0",
                    width: "100%", height: "100%",
                    pointerEvents: "none", // 空白区域不拦截时间轴拖动；方块自身可点击
                    boxSizing: "border-box",
                    overflow: "hidden",
                    borderRadius: "4px"
                });
                sliderBox.appendChild(segBlocksLayer);
                trimArea.appendChild(sliderBox);

                container.appendChild(trimArea);

                // 输出模式切换时的时间轴显示：自定义裁剪 = 长条（高亮+手柄），智能分段 = 小方块
                const applyOutputTimelineVisual = (smart) => {
                    fill.style.display = smart ? "none" : "block";
                    startHandle.style.display = smart ? "none" : "block";
                    endHandle.style.display = smart ? "none" : "block";
                    segBlocksLayer.style.display = smart ? "block" : "none";
                };
                applyOutputTimelineVisual(false); // 默认：自定义裁剪输出，显示长条

                node.syncLayoutToNode = function() {
                    const nodeWidth = this.size?.[0] || 690;
                    const targetWidth = Math.max(10, nodeWidth - 30);
                    if (container) {
                        container.style.width = `${targetWidth}px`;
                        container.style.maxWidth = `${targetWidth}px`;
                        container.style.boxSizing = "border-box";
                    }
                };

                // ====================================================================
                // 智能分段：点击「检测分段」调用后端一次性检测镜头边界，
                // 分段结果直接绘制在下方裁剪时间轴上（小方块），无需重新运行整个流程。
                // ====================================================================
                let segmentsCacheKey = "";
                let segmentsCacheData = null;
                // 当前在时间轴上选中的分段（自定义裁剪模式下控制预览定位与播放范围）
                let selectedSegment = null;

                const fmtSegTime = (sec) => {
                    sec = Math.max(0, sec || 0);
                    const m = Math.floor(sec / 60);
                    const s = Math.floor(sec % 60);
                    return `${m}:${s.toString().padStart(2, "0")}`;
                };

                // 在裁剪时间轴上按当前裁切范围绘制分段小方块
                const renderSegBlocks = () => {
                    segBlocksLayer.innerHTML = "";
                    if (!segmentsCacheData || !segmentsCacheData.segments || segmentsCacheData.segments.length === 0) {
                        return;
                    }
                    const activeDur = getActiveDuration();
                    let s = startTimeWidget ? (parseFloat(startTimeWidget.value) || 0) : 0;
                    let e = endTimeWidget ? (parseFloat(endTimeWidget.value) || activeDur) : activeDur;
                    if (e === 0 || e > activeDur) e = activeDur;
                    if (s > e) s = e;

                    const selIdx = segmentIndexWidget ? (parseInt(segmentIndexWidget.value) || 0) : 0;

                    segmentsCacheData.segments.forEach((seg, i) => {
                        if (seg.end <= s || seg.start >= e) return; // 完全在裁切范围外，跳过
                        // 按绝对时间定位（与裁切手柄/fill 同一坐标系：整条时间轴 = 全片时长）
                        const leftPct = Math.max(0, (seg.start / activeDur) * 100);
                        const rightPct = Math.min(100, (seg.end / activeDur) * 100);
                        const widthPct = Math.max(rightPct - leftPct, 1.2); // 最小宽度保证可点击

                        const isSel = (isSmartMode && seg.index === selIdx) ||
                            (selectedSegment && Math.abs(selectedSegment.start - seg.start) < 0.01);

                        const block = document.createElement("div");
                        const hue = 195 + (i * 12) % 40; // 每个分段色相略不同，便于区分
                        Object.assign(block.style, {
                            position: "absolute",
                            left: `${leftPct}%`,
                            width: `${widthPct}%`,
                            top: "18%", height: "64%",
                            background: `hsla(${hue}, 90%, 55%, 0.35)`,
                            border: isSel ? "1px solid #fff" : "1px solid rgba(255,255,255,0.45)",
                            borderRadius: "2px",
                            boxSizing: "border-box",
                            pointerEvents: "auto",
                            cursor: "pointer",
                            boxShadow: isSel ? "0 0 6px rgba(56,189,248,0.9)" : "none",
                            zIndex: "2"
                        });
                        block.title = `#${seg.index} ${fmtSegTime(seg.start)}–${fmtSegTime(seg.end)}`;
                        // 防止点击方块时触发时间轴的拖动
                        block.onpointerdown = (ev) => ev.stopPropagation();
                        block.onclick = () => {
                            selectedSegment = { start: seg.start, end: seg.end, index: seg.index };
                            if (isSmartMode && segmentIndexWidget) segmentIndexWidget.value = seg.index;
                            renderSegBlocks();
                            node.setDirtyCanvas(true, false);
                            // 同步预览画面到该分段的起始帧：seg.start 已由后端对齐为「图像」输出首帧的精确 PTS。
                            // 加 1e-6 秒的帧内微小偏移，避免 seek 恰好落在帧边界上被浏览器吸附到相邻帧。
                            if (videoPreview && duration > 0) {
                                videoPreview.currentTime = Math.min(seg.start + 1e-6, Math.max(0, duration - 0.01));
                            }
                        };
                        segBlocksLayer.appendChild(block);
                    });
                };

                let _detectFlashTimer = null;
                const flashDetectBtn = (msg) => {
                    detectBtn.textContent = msg;
                    if (_detectFlashTimer) clearTimeout(_detectFlashTimer);
                    _detectFlashTimer = setTimeout(() => {
                        detectBtn.textContent = "检测分段";
                    }, 1600);
                };

                // 智能分段模式恢复：仅查询后端已保存的分段结果，命中则直接渲染，不触发重新分割
                node.tryRestoreCachedSegments = async function () {
                    const videoVal = videoWidget ? videoWidget.value : "";
                    if (!videoVal) return;
                    if (segmentsCacheData && segmentsCacheData.segments && segmentsCacheData.segments.length > 0) return;
                    const fr = frameRateWidget ? (frameRateWidget.value || 24) : 24;
                    const st = startTimeWidget ? (parseFloat(startTimeWidget.value) || 0) : 0;
                    const et = endTimeWidget ? (parseFloat(endTimeWidget.value) || 0) : 0;
                    try {
                        const url = `/yuan_tool/video_detect_segments?filename=${encodeURIComponent(videoVal)}&fps=${fr}&start_time=${st}&end_time=${et}&cached=1`;
                        const resp = await api.fetchApi(url);
                        if (resp.status !== 200) return;
                        const data = await resp.json();
                        if (data.segments && data.segments.length > 0) {
                            segmentsCacheKey = `${videoVal}|${fr}|${st}|${et}`;
                            segmentsCacheData = data;
                            renderSegBlocks();
                            node.setDirtyCanvas(true, false);
                        }
                    } catch (e) {
                        // 静默失败，用户仍可手动点击“检测分段”
                    }
                };

                detectBtn.onclick = async () => {
                    const videoVal = videoWidget ? videoWidget.value : "";
                    if (!videoVal) {
                        flashDetectBtn("请先选择视频");
                        return;
                    }
                    const fr = frameRateWidget ? (frameRateWidget.value || 24) : 24;
                    const st = startTimeWidget ? (parseFloat(startTimeWidget.value) || 0) : 0;
                    const et = endTimeWidget ? (parseFloat(endTimeWidget.value) || 0) : 0;
                    const key = `${videoVal}|${fr}|${st}|${et}`;
                    if (segmentsCacheKey === key && segmentsCacheData) {
                        renderSegBlocks();
                        return;
                    }
                    detectBtn.disabled = true;
                    detectBtn.textContent = "检测中...";
                    try {
                        const url = `/yuan_tool/video_detect_segments?filename=${encodeURIComponent(videoVal)}&fps=${fr}&start_time=${st}&end_time=${et}`;
                        const resp = await api.fetchApi(url);
                        if (resp.status !== 200) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
                        const data = await resp.json();
                        segmentsCacheKey = key;
                        segmentsCacheData = data;
                        renderSegBlocks();
                    } catch (err) {
                        console.error("[YuanVideoUI] 分段检测失败", err);
                        flashDetectBtn("检测失败");
                    } finally {
                        detectBtn.disabled = false;
                        detectBtn.textContent = "检测分段";
                    }
                };

                // 分段索引变化时同步时间轴高亮
                if (segmentIndexWidget) {
                    const origSegCb = segmentIndexWidget.callback;
                    segmentIndexWidget.callback = function () {
                        if (origSegCb) origSegCb.apply(this, arguments);
                        if (segmentsCacheData && segmentsCacheData.segments) {
                            renderSegBlocks();
                        }
                    };
                }

                // Delay DOM Widget creation to ensure it is added after all standard widgets
                setTimeout(() => {
                    // Add HTML widget to LiteGraph
                    node.domWidget = node.addDOMWidget("VideoUI", "div", container);

                    // Fixed: Return a solid minimum required bounding box.
                    // Bumped horizontal from 200px to 360px. This natively stops LiteGraph 
                    // from letting the node be squished too thin, completely preventing overlap.
                    node.domWidget.computeSize = function (width) {
                        const nodeWidth = node.size?.[0] || width || 690;
                        return [Math.max(10, nodeWidth - 30), 250];
                    };

                    // Applies the default creation bounds natively, increased default height
                    // to match the widgets required height out of the box.
                    requestAnimationFrame(() => {
                        if (node.size[0] < 690) {
                            node.size[0] = 690;
                        }

                        // INCREASE DEFAULT HEIGHT HERE:
                        // Change the 620 below to adjust the starting height of the node
                        if (node.size[1] < 740) {
                            node.size[1] = 740;
                        }

                        // Trigger manual resize call so the vertical math applies instantly
                        node.syncLayoutToNode();
                        if (node.onResize) node.onResize(node.size);

                        // Sync visual toggle to initial data
                        if (displayModeWidget && displayModeWidget.value === "帧") {
                            isFramesMode = false; // prime for click
                            switchBox.onclick();
                        }

                        // 同步输出模式切换按钮到初始数据
                        if (outputModeWidget && outputModeWidget.value === "智能分段输出") {
                            isSmartMode = false; // prime for click
                            outputSwitchBox.onclick();
                        }

                        // 初始化 UI 输入框的标签描述（时间/帧）与显示值
                        updateTimeInputMode();

                        app.graph.setDirtyCanvas(true, true);
                    });
                }, 100);

                // ====================================================================
                // LOGIC & SYNCING
                // ====================================================================
                let duration = 0;
                let dragging = null;
                let dragOffset = 0;
                let dragSelectionWidth = 0;
                let isUpdatingDuration = false;

                // Crop logic
                let cropDragging = null;
                let dragStartX = 0;
                let dragStartY = 0;
                let dragStartCropX = 0;
                let dragStartCropY = 0;
                let dragStartCropW = 1;
                let dragStartCropH = 1;

                const updateCropUI = () => {
                    const vw = videoPreview.videoWidth;
                    const vh = videoPreview.videoHeight;

                    let cx = cropXWidget ? parseFloat(cropXWidget.value) || 0 : 0;
                    let cy = cropYWidget ? parseFloat(cropYWidget.value) || 0 : 0;
                    let cw_val = cropWWidget ? parseFloat(cropWWidget.value) || 1 : 1;
                    let ch_val = cropHWidget ? parseFloat(cropHWidget.value) || 1 : 1;

                    const actualW = vw ? Math.round(cw_val * vw) : 0;
                    const actualH = vh ? Math.round(ch_val * vh) : 0;

                    if (!isCropVisible || !vw) {
                        cropBox.style.display = "none";
                        cropEditContainer.style.display = "none";
                        if (cw_val < 0.999 || ch_val < 0.999 || cx > 0.001 || cy > 0.001) {
                            cropDims.textContent = `裁剪: ${actualW}x${actualH}`;
                            cropDims.style.display = "inline-block";
                        } else {
                            cropDims.style.display = "none";
                        }
                        return;
                    }

                    cropDims.style.display = "none";
                    cropEditContainer.style.display = "flex";
                    cropBox.style.display = "block";

                    if (document.activeElement !== wInput) wInput.value = actualW;
                    if (document.activeElement !== hInput) hInput.value = actualH;

                    const cw = videoPreview.clientWidth;
                    const ch = videoPreview.clientHeight;

                    const ratio = Math.min(cw / vw, ch / vh);
                    const renderedW = vw * ratio;
                    const renderedH = vh * ratio;
                    const xOffset = (cw - renderedW) / 2;
                    const yOffset = (ch - renderedH) / 2;

                    cropBox.style.left = `${xOffset + cx * renderedW}px`;
                    cropBox.style.top = `${yOffset + cy * renderedH}px`;
                    cropBox.style.width = `${cw_val * renderedW}px`;
                    cropBox.style.height = `${ch_val * renderedH}px`;
                };

                const onCropPointerDown = (e, handle) => {
                    if (!isCropVisible) return;
                    e.preventDefault();
                    e.stopPropagation();
                    cropDragging = handle;
                    e.target.setPointerCapture(e.pointerId);

                    dragStartX = e.clientX;
                    dragStartY = e.clientY;
                    dragStartCropX = cropXWidget ? parseFloat(cropXWidget.value) || 0 : 0;
                    dragStartCropY = cropYWidget ? parseFloat(cropYWidget.value) || 0 : 0;
                    dragStartCropW = cropWWidget ? parseFloat(cropWWidget.value) || 1 : 1;
                    dragStartCropH = cropHWidget ? parseFloat(cropHWidget.value) || 1 : 1;

                    e.target.addEventListener("pointermove", onCropPointerMove);
                    e.target.addEventListener("pointerup", onCropPointerUp);
                };

                const onCropPointerMove = (e) => {
                    if (!cropDragging) return;
                    e.preventDefault();

                    const vw = videoPreview.videoWidth;
                    const vh = videoPreview.videoHeight;
                    const cw = videoPreview.clientWidth;
                    const ch = videoPreview.clientHeight;

                    const ratio = Math.min(cw / vw, ch / vh);
                    const renderedW = vw * ratio;
                    const renderedH = vh * ratio;

                    const dx = (e.clientX - dragStartX) / renderedW;
                    const dy = (e.clientY - dragStartY) / renderedH;

                    let new_cw = dragStartCropW;
                    let new_ch = dragStartCropH;
                    let new_cx = dragStartCropX;
                    let new_cy = dragStartCropY;

                    if (cropDragging === "tl") {
                        new_cw = dragStartCropW - dx;
                        new_ch = dragStartCropH - dy;
                    } else if (cropDragging === "tr") {
                        new_cw = dragStartCropW + dx;
                        new_ch = dragStartCropH - dy;
                    } else if (cropDragging === "bl") {
                        new_cw = dragStartCropW - dx;
                        new_ch = dragStartCropH + dy;
                    } else if (cropDragging === "br") {
                        new_cw = dragStartCropW + dx;
                        new_ch = dragStartCropH + dy;
                    } else if (cropDragging === "tm") {
                        new_ch = dragStartCropH - dy;
                    } else if (cropDragging === "bm") {
                        new_ch = dragStartCropH + dy;
                    } else if (cropDragging === "lm") {
                        new_cw = dragStartCropW - dx;
                    } else if (cropDragging === "rm") {
                        new_cw = dragStartCropW + dx;
                    }

                    if (currentAspectRatio > 0 && cropDragging !== "center") {
                        const R = currentAspectRatio * (vh / vw);
                        if (["tm", "bm"].includes(cropDragging)) {
                            new_cw = new_ch * R;
                            new_cx = dragStartCropX + (dragStartCropW - new_cw) / 2;
                        } else if (["lm", "rm"].includes(cropDragging)) {
                            new_ch = new_cw / R;
                            new_cy = dragStartCropY + (dragStartCropH - new_ch) / 2;
                        } else {
                            new_ch = new_cw / R;
                        }
                    }

                    if (cropDragging === "tl") {
                        new_cx = dragStartCropX + dragStartCropW - new_cw;
                        new_cy = dragStartCropY + dragStartCropH - new_ch;
                    } else if (cropDragging === "tr") {
                        new_cx = dragStartCropX;
                        new_cy = dragStartCropY + dragStartCropH - new_ch;
                    } else if (cropDragging === "bl") {
                        new_cx = dragStartCropX + dragStartCropW - new_cw;
                        new_cy = dragStartCropY;
                    } else if (cropDragging === "br") {
                        new_cx = dragStartCropX;
                        new_cy = dragStartCropY;
                    } else if (cropDragging === "tm") {
                        new_cy = dragStartCropY + dragStartCropH - new_ch;
                        if (!(currentAspectRatio > 0)) new_cx = dragStartCropX;
                    } else if (cropDragging === "bm") {
                        new_cy = dragStartCropY;
                        if (!(currentAspectRatio > 0)) new_cx = dragStartCropX;
                    } else if (cropDragging === "lm") {
                        new_cx = dragStartCropX + dragStartCropW - new_cw;
                        if (!(currentAspectRatio > 0)) new_cy = dragStartCropY;
                    } else if (cropDragging === "rm") {
                        new_cx = dragStartCropX;
                        if (!(currentAspectRatio > 0)) new_cy = dragStartCropY;
                    } else if (cropDragging === "center") {
                        new_cx = dragStartCropX + dx;
                        new_cy = dragStartCropY + dy;
                    }

                    if (new_cw < 0.02) {
                        new_cw = 0.02;
                        if (currentAspectRatio > 0) new_ch = new_cw / (currentAspectRatio * (vh / vw));
                    }
                    if (new_ch < 0.02) {
                        new_ch = 0.02;
                        if (currentAspectRatio > 0) new_cw = new_ch * (currentAspectRatio * (vh / vw));
                    }

                    if (cropDragging === "center") {
                        new_cx = Math.max(0, Math.min(new_cx, 1 - new_cw));
                        new_cy = Math.max(0, Math.min(new_cy, 1 - new_ch));
                    } else {
                        if (new_cx < 0) {
                            if (["tl", "bl", "lm"].includes(cropDragging)) { new_cw += new_cx; new_cx = 0; }
                        }
                        if (new_cy < 0) {
                            if (["tl", "tr", "tm"].includes(cropDragging)) { new_ch += new_cy; new_cy = 0; }
                        }
                        if (new_cx + new_cw > 1) {
                            if (["tr", "br", "rm"].includes(cropDragging)) new_cw = 1 - new_cx;
                        }
                        if (new_cy + new_ch > 1) {
                            if (["bl", "br", "bm"].includes(cropDragging)) new_ch = 1 - new_cy;
                        }

                        if (currentAspectRatio > 0) {
                            const R = currentAspectRatio * (vh / vw);
                            if (new_cw / new_ch > R + 0.001) {
                                new_cw = new_ch * R;
                                if (["tl", "bl", "lm"].includes(cropDragging)) new_cx = dragStartCropX + dragStartCropW - new_cw;
                            } else if (new_cw / new_ch < R - 0.001) {
                                new_ch = new_cw / R;
                                if (["tl", "tr", "tm"].includes(cropDragging)) new_cy = dragStartCropY + dragStartCropH - new_ch;
                            }
                        }
                    }

                    if (cropXWidget) cropXWidget.value = parseFloat(new_cx.toFixed(3));
                    if (cropYWidget) cropYWidget.value = parseFloat(new_cy.toFixed(3));
                    if (cropWWidget) cropWWidget.value = parseFloat(new_cw.toFixed(3));
                    if (cropHWidget) cropHWidget.value = parseFloat(new_ch.toFixed(3));

                    updateCropUI();
                    app.graph.setDirtyCanvas(true, false);
                };

                const onCropPointerUp = (e) => {
                    cropDragging = null;
                    e.target.releasePointerCapture(e.pointerId);
                    e.target.removeEventListener("pointermove", onCropPointerMove);
                    e.target.removeEventListener("pointerup", onCropPointerUp);
                };

                cropBox.onpointerdown = (e) => {
                    if (e.target === cropBox) onCropPointerDown(e, "center");
                };
                tlHandle.onpointerdown = (e) => onCropPointerDown(e, "tl");
                trHandle.onpointerdown = (e) => onCropPointerDown(e, "tr");
                blHandle.onpointerdown = (e) => onCropPointerDown(e, "bl");
                brHandle.onpointerdown = (e) => onCropPointerDown(e, "br");
                tmHandle.onpointerdown = (e) => onCropPointerDown(e, "tm");
                bmHandle.onpointerdown = (e) => onCropPointerDown(e, "bm");
                lmHandle.onpointerdown = (e) => onCropPointerDown(e, "lm");
                rmHandle.onpointerdown = (e) => onCropPointerDown(e, "rm");

                // Add a resize observer to the video wrapper so crop handles stay pinned
                const resizeObserver = new ResizeObserver(() => {
                    if (isCropVisible) updateCropUI();
                });
                resizeObserver.observe(videoWrapper);

                // Ensure we clean up observer
                const oldOnRemoved = node.onRemoved;
                node.onRemoved = function () {
                    resizeObserver.disconnect();
                    if (oldOnRemoved) oldOnRemoved.apply(this, arguments);
                }

                // Smart helper to ensure timeline displays correctly even with no video loaded
                const getActiveDuration = () => {
                    if (duration > 0) return duration;
                    let e = endTimeWidget ? parseFloat(endTimeWidget.value) || 0 : 0;
                    let s = startTimeWidget ? parseFloat(startTimeWidget.value) || 0 : 0;
                    let maxVal = Math.max(e, s);
                    return maxVal > 0 ? Math.max(maxVal, 1.0) : 1.0; // Default to 1.0 if completely empty
                };

                // Time Duration Hook
                if (durationWidget) {
                    const origCallback = durationWidget.callback;
                    durationWidget.callback = function (v) {
                        if (isUpdatingDuration) {
                            if (origCallback) origCallback.apply(this, arguments);
                            return;
                        }

                        isUpdatingDuration = true;
                        const activeDur = getActiveDuration();
                        let d = parseFloat(v) || 0;
                        if (d < 0) d = 0;
                        if (d > activeDur) d = activeDur;

                        let s = startTimeWidget ? parseFloat(startTimeWidget.value) || 0 : 0;
                        let newStart = s;
                        let newEnd = s + d;

                        if (newEnd > activeDur) {
                            newEnd = activeDur;
                            newStart = activeDur - d;
                        }

                        if (startTimeWidget) startTimeWidget.value = parseFloat(newStart.toFixed(2));
                        if (endTimeWidget) endTimeWidget.value = parseFloat(newEnd.toFixed(2));
                        node.syncFramesFromTime();

                        if (duration === 0) updateRuler();
                        updateUI(true);
                        app.graph.setDirtyCanvas(true, false);

                        if (origCallback) origCallback.apply(this, arguments);
                        isUpdatingDuration = false;
                    };
                }

                // Frame Duration Hook
                if (durationFramesWidget) {
                    const origCallback = durationFramesWidget.callback;
                    durationFramesWidget.callback = function (v) {
                        if (isUpdatingDuration || !frameRateWidget) {
                            if (origCallback) origCallback.apply(this, arguments);
                            return;
                        }

                        isUpdatingDuration = true;
                        const fr = frameRateWidget.value || 24;
                        const activeDurFrames = Math.round(getActiveDuration() * fr);

                        let d = parseInt(v) || 0;
                        if (d < 0) d = 0;
                        if (d > activeDurFrames) d = activeDurFrames;

                        let s = startFrameWidget ? parseInt(startFrameWidget.value) || 0 : 0;
                        let newStart = s;
                        let newEnd = s + d;

                        if (newEnd > activeDurFrames) {
                            newEnd = activeDurFrames;
                            newStart = activeDurFrames - d;
                        }

                        if (startFrameWidget) startFrameWidget.value = newStart;
                        if (endFrameWidget) endFrameWidget.value = newEnd;
                        node.syncTimeFromFrames();

                        if (duration === 0) updateRuler();
                        updateUI(true);
                        app.graph.setDirtyCanvas(true, false);

                        if (origCallback) origCallback.apply(this, arguments);
                        isUpdatingDuration = false;
                    };
                }

                // Standard Video Player Format HH:MM:SS (only shows hours if it's over an hour long)
                const formatTime = (secs) => {
                    const h = Math.floor(secs / 3600);
                    const m = Math.floor((secs % 3600) / 60);
                    const s = Math.floor(secs % 60);
                    const mStr = m.toString().padStart(2, '0');
                    const sStr = s.toString().padStart(2, '0');

                    if (h > 0) {
                        return `${h}:${mStr}:${sStr}`;
                    } else {
                        return `${m}:${sStr}`;
                    }
                };

                const updateRuler = () => {
                    timeRuler.innerHTML = '';
                    const activeDur = getActiveDuration();
                    const numMajorTicks = 5;
                    const subTicks = 4;
                    const totalTicks = (numMajorTicks - 1) * subTicks;

                    const isFrames = displayModeWidget && displayModeWidget.value === "帧";
                    const fr = frameRateWidget ? frameRateWidget.value : 24;

                    for (let i = 0; i <= totalTicks; i++) {
                        const pct = i / totalTicks;
                        const t = activeDur * pct;
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
                            if (isFrames) {
                                label.textContent = Math.round(t * fr);
                            } else {
                                label.textContent = formatTime(t);
                            }
                            tickWrapper.appendChild(label);
                        }
                        timeRuler.appendChild(tickWrapper);
                    }
                };

                function updateUI(syncPlayer = false) {
                    const activeDur = getActiveDuration();
                    const fr = frameRateWidget ? (frameRateWidget.value || 24) : 24;
                    const isFrames = isFramesMode;
                    // 当前显示模式对应的刻度范围：时间模式用秒，帧模式用帧
                    const activeScale = isFrames ? Math.round(activeDur * fr) : activeDur;

                    // 时间输入框（裁剪左侧）与 widget 值保持同步
                    if (node.syncTimeInputs) node.syncTimeInputs();

                    // 只读当前显示模式对应的起止值，不跨组换算（两组各自独立）
                    let s, e;
                    if (isFrames) {
                        s = startFrameWidget ? parseInt(startFrameWidget.value) || 0 : 0;
                        e = endFrameWidget ? parseInt(endFrameWidget.value) || 0 : 0;
                    } else {
                        s = startTimeWidget ? parseFloat(startTimeWidget.value) || 0 : 0;
                        e = endTimeWidget ? parseFloat(endTimeWidget.value) || 0 : 0;
                    }

                    let visualEnd = e;
                    if (visualEnd === 0 || visualEnd > activeScale) visualEnd = activeScale;
                    if (s > visualEnd) s = visualEnd;

                    let pStart = (s / activeScale) * 100;
                    let pEnd = (visualEnd / activeScale) * 100;

                    pStart = Math.max(0, Math.min(pStart, 100));
                    pEnd = Math.max(0, Math.min(pEnd, 100));

                    startHandle.style.left = `${pStart}%`;
                    endHandle.style.left = `${pEnd}%`;

                    fill.style.left = `${pStart}%`;
                    fill.style.width = `${pEnd - pStart}%`;

                    // 只同步当前显示模式对应的时长 widget（帧模式 -> 时长帧数，时间模式 -> 时长），
                    // 由当前组的起止值计算，绝不跨组换算/覆盖另一组的值。
                    if (duration > 0 && !isUpdatingDuration) {
                        isUpdatingDuration = true;
                        if (isFrames) {
                            const durF = Math.round(visualEnd - s);
                            if (durationFramesWidget && durationFramesWidget.value !== durF) {
                                durationFramesWidget.value = durF;
                            }
                        } else {
                            const durT = parseFloat((visualEnd - s).toFixed(2));
                            if (durationWidget && durationWidget.value !== durT) {
                                durationWidget.value = durT;
                            }
                        }
                        isUpdatingDuration = false;
                    }

                    if (syncPlayer && duration > 0) {
                        videoPreview.currentTime = isFrames ? s / fr : s;
                    }

                    // 裁切范围变化时，重绘时间轴上的分段小方块
                    if (segmentsCacheData && segmentsCacheData.segments) {
                        renderSegBlocks();
                    }
                }

                // Force draw default empty state on creation
                setTimeout(() => {
                    updateRuler();
                    updateUI();
                }, 50);

                videoPreview.onloadedmetadata = () => {
                    duration = videoPreview.duration;
                    if (node._should_reset_trim) {
                        if (startTimeWidget) startTimeWidget.value = 0;
                        if (endTimeWidget) endTimeWidget.value = duration;
                        node._should_reset_trim = false;
                        node.syncFramesFromTime();
                    } else {
                        if (endTimeWidget && (endTimeWidget.value === 0 || endTimeWidget.value > duration)) {
                            endTimeWidget.value = duration;
                            node.syncFramesFromTime();
                        }
                    }
                    updateRuler();
                    updateUI();
                    updateCropUI();
                };

                // 播放范围控制：选中分段时只播该分段（播完即停），否则按裁切范围循环
                videoPreview.ontimeupdate = () => {
                    if (!duration || dragging) return;

                    // 选中了分段小方块：仅在 [start, end] 内播放，播到结束即停止。
                    // 段尾暂停由 rVFC/interval 快速监控主处理（见 _pauseAtSegEnd），
                    // timeupdate 事件频率低（约 250ms），这里仅作极端情况兜底。
                    if (selectedSegment && selectedSegment.end > 0) {
                        if (videoPreview.currentTime >= selectedSegment.end - 0.001) {
                            _pauseAtSegEnd();
                            return;
                        }
                        if (videoPreview.currentTime < selectedSegment.start) {
                            videoPreview.currentTime = selectedSegment.start;
                        }
                        return;
                    }

                    let s = startTimeWidget ? parseFloat(startTimeWidget.value) || 0 : 0;
                    let e = endTimeWidget ? parseFloat(endTimeWidget.value) || duration : duration;
                    if (e === 0) e = duration;

                    if (videoPreview.currentTime >= e && e > 0) {
                        videoPreview.currentTime = s;
                    } else if (videoPreview.currentTime < s) {
                        videoPreview.currentTime = s;
                    }
                };

                // 若用户在分段范围之外开始播放，则取消分段选择；已播完再播放则从头重播
                videoPreview.addEventListener('play', () => {
                    if (!selectedSegment || !duration) return;
                    const ct = videoPreview.currentTime;
                    if (ct < selectedSegment.start - 0.05 || ct > selectedSegment.end + 0.05) {
                        selectedSegment = null;
                        if (segmentsCacheData && segmentsCacheData.segments) renderSegBlocks();
                    } else if (ct >= selectedSegment.end - 0.05) {
                        videoPreview.currentTime = selectedSegment.start;
                    }
                });

                // ====================================================================
                // 段尾精准停帧监控
                // 背景：HTMLMediaElement 的 timeupdate 事件频率由浏览器决定（约每 250ms 一次）。
                // 播放越过分段段尾后，浏览器会先渲染出下一分镜的首帧画面，timeupdate 才触发，
                // 此时 pause + 拉回只是"事后补救"，用户已经看到了下一分镜画面。
                // 方案：
                //   1) requestVideoFrameCallback（Chrome/Edge/Firefox）：每呈现一帧回调一次，
                //      用该帧精确 PTS(mediaTime) 判断——输出末帧（end-1e-6）一呈现立即暂停，
                //      下一分镜首帧根本不会被提交渲染，零闪现。
                //   2) setInterval 8ms 兜底：读 currentTime >= end 即暂停（支持 rVFC 时是双保险）。
                //   3) ontimeupdate 兜底：后台节流等极端情况最后补救。
                // 说明：end = 输出末帧精确PTS + 1e-6，浏览器显示 PTS<=currentTime 的帧，暂停后
                // 停留帧仍是输出末帧；所有函数幂等，重复触发不会产生重复 pause/seek 抖动。
                // ====================================================================
                const _pauseAtSegEnd = () => {
                    if (!selectedSegment || selectedSegment.end <= 0 || duration <= 0) return;
                    if (!videoPreview.paused) videoPreview.pause();
                    // 吸附到段尾边界；仅当播放头明显偏离时才 seek，避免重复 seek 造成画面抖动
                    const cur = videoPreview.currentTime;
                    if (Math.abs(cur - selectedSegment.end) > 1e-4) {
                        videoPreview.currentTime = selectedSegment.end;
                    }
                };
                const _checkSegEndByFrame = (frameTime) => {
                    if (!selectedSegment || selectedSegment.end <= 0 || duration <= 0 || videoPreview.paused) return false;
                    // 1ms 容差：覆盖 mediaTime 与后端 PTS 的浮点换算误差；
                    // 容差小于任何实际视频帧间隔，只会在输出末帧呈现的瞬间触发，不会提前误停
                    if (frameTime >= selectedSegment.end - 0.001) {
                        _pauseAtSegEnd();
                        return true;
                    }
                    return false;
                };
                const _checkSegEndByTime = () => {
                    if (!selectedSegment || selectedSegment.end <= 0 || duration <= 0 || videoPreview.paused) return false;
                    if (videoPreview.currentTime >= selectedSegment.end) {
                        _pauseAtSegEnd();
                        return true;
                    }
                    return false;
                };
                const _rvfcSegLoop = (now, meta) => {
                    if (!videoPreview.requestVideoFrameCallback) return;
                    if (!videoPreview.paused) {
                        _checkSegEndByFrame(meta.mediaTime);
                    }
                    // 暂停时 rVFC 不再回调；恢复播放后自动续上，因此始终续订
                    videoPreview.requestVideoFrameCallback(_rvfcSegLoop);
                };
                if (videoPreview.requestVideoFrameCallback) {
                    videoPreview.requestVideoFrameCallback(_rvfcSegLoop);
                }
                const _segEndPollTimer = setInterval(_checkSegEndByTime, 8);

                // --- Timeline Drag Logic (Primary state runs in Seconds format to lock playback natively) ---
                sliderBox.onpointerdown = (e) => {
                    selectedSegment = null; // 拖动裁切点时取消分段选择
                    const activeDur = getActiveDuration();
                    const rect = sliderBox.getBoundingClientRect();
                    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                    const val = (x / rect.width) * activeDur;

                    let s = startTimeWidget ? parseFloat(startTimeWidget.value) || 0 : 0;
                    let e_val = endTimeWidget ? parseFloat(endTimeWidget.value) || activeDur : activeDur;
                    if (e_val === 0) e_val = activeDur;

                    const handleTolerance = (10 / rect.width) * activeDur;

                    if (val > s + handleTolerance && val < e_val - handleTolerance) {
                        dragging = 'center';
                        dragOffset = val - s;
                        dragSelectionWidth = e_val - s;
                    } else if (Math.abs(val - s) < Math.abs(val - e_val)) {
                        dragging = 'start';
                        if (startTimeWidget) startTimeWidget.value = parseFloat(Math.min(val, e_val).toFixed(2));
                        if (duration > 0) videoPreview.currentTime = startTimeWidget.value;
                    } else {
                        dragging = 'end';
                        if (endTimeWidget) endTimeWidget.value = parseFloat(Math.max(val, s).toFixed(2));
                        if (duration > 0) videoPreview.currentTime = endTimeWidget.value;
                    }

                    node.syncFramesFromTime();
                    updateUI();
                    app.graph.setDirtyCanvas(true, false);
                    sliderBox.setPointerCapture(e.pointerId);
                };

                sliderBox.onpointermove = (e) => {
                    if (!dragging) return;
                    const activeDur = getActiveDuration();
                    const rect = sliderBox.getBoundingClientRect();
                    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                    const val = (x / rect.width) * activeDur;

                    if (dragging === 'start') {
                        let e_val = endTimeWidget ? parseFloat(endTimeWidget.value) || activeDur : activeDur;
                        if (e_val === 0) e_val = activeDur;
                        if (startTimeWidget) startTimeWidget.value = parseFloat(Math.min(val, e_val).toFixed(2));
                        if (duration > 0) videoPreview.currentTime = startTimeWidget.value;
                    } else if (dragging === 'end') {
                        const s = startTimeWidget ? parseFloat(startTimeWidget.value) || 0 : 0;
                        if (endTimeWidget) endTimeWidget.value = parseFloat(Math.max(val, s).toFixed(2));
                        if (duration > 0) videoPreview.currentTime = endTimeWidget.value;
                    } else if (dragging === 'center') {
                        let newStart = val - dragOffset;
                        let newEnd = newStart + dragSelectionWidth;

                        if (newStart < 0) {
                            newStart = 0;
                            newEnd = dragSelectionWidth;
                        } else if (newEnd > activeDur) {
                            newEnd = activeDur;
                            newStart = activeDur - dragSelectionWidth;
                        }

                        if (startTimeWidget) startTimeWidget.value = parseFloat(newStart.toFixed(2));
                        if (endTimeWidget) endTimeWidget.value = parseFloat(newEnd.toFixed(2));
                        if (duration > 0) videoPreview.currentTime = startTimeWidget.value;
                    }

                    node.syncFramesFromTime();
                    updateUI();
                    app.graph.setDirtyCanvas(true, false);
                };

                sliderBox.onpointerup = (e) => {
                    dragging = null;
                    sliderBox.releasePointerCapture(e.pointerId);
                };

                // --- Improved Global Drag & Drop for Node Inner Content ---
                let dragCounter = 0;
                container.addEventListener("dragenter", (e) => {
                    e.preventDefault();
                    dragCounter++;
                    if (dragCounter === 1) {
                        container.style.outline = "2px dashed #38bdf8";
                        container.style.outlineOffset = "-2px";
                        container.style.background = "rgba(14, 165, 233, 0.1)";
                    }
                });

                container.addEventListener("dragover", (e) => {
                    e.preventDefault();
                });

                container.addEventListener("dragleave", (e) => {
                    e.preventDefault();
                    dragCounter--;
                    if (dragCounter === 0) {
                        container.style.outline = "none";
                        container.style.background = defaultBg;
                    }
                });

                container.addEventListener("drop", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dragCounter = 0;
                    container.style.outline = "none";
                    container.style.background = defaultBg;
                    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                        const file = e.dataTransfer.files[0];
                        if (file.type.startsWith('video/') || file.name.toLowerCase().match(/\.(mp4|webm|mkv|avi|mov|m4v|flv|wmv)$/)) {
                            uploadFile(file);
                        }
                    }
                });

                if (videoWidget && videoWidget.value) {
                    node.updatePreview(videoWidget.value);
                }

                setTimeout(() => { node._initializing = false; }, 500);

                return r;
            };
        }
    },
});
