/**
 * Yuan Tool · 加载批量图像 前端（多滑轨版）
 *
 * 功能：
 *  - 支持最多 20 个独立滑轨，每个滑轨可单独加载批量图像
 *  - 每个滑轨可自定义命名（默认 批量图像A、批量图像B...）
 *  - 每个滑轨有独立的 multi_output 输出端口，通过 getConnectionPos 重写
 *    将端口定位到滑轨右侧（而非节点顶部）
 *  - 滑轨内 Gallery 横向排列，支持滚动条与滚轮预览
 *  - 默认显示 1 个滑轨，新增滑轨在下方添加
 *  - 状态持久化到工作流（tracks_data widget）
 *  - 兼容 V1 / V3 前端
 */
(function () {
    "use strict";

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

    // --- 尺寸常量 ---
    const MAX_TRACKS = 20;
    const THUMB_SIZE = 80;
    const THUMB_GAP = 6;
    const TRACK_HEADER_HEIGHT = 26;
    const TRACK_GALLERY_HEIGHT = 96;
    const TRACK_GAP = 8;
    const TRACK_TOTAL_HEIGHT = TRACK_HEADER_HEIGHT + TRACK_GALLERY_HEIGHT + TRACK_GAP;
    const ADD_BTN_HEIGHT = 28;
    const CONTAINER_PADDING = 8;

    // --- 注入全局样式 ---
    if (!document.getElementById('yuan-multiimage-style')) {
        const style = document.createElement('style');
        style.id = 'yuan-multiimage-style';
        style.textContent = `
            .yuan-mi-gallery::-webkit-scrollbar { height: 6px; }
            .yuan-mi-gallery::-webkit-scrollbar-track { background: #1a1a1a; border-radius: 3px; }
            .yuan-mi-gallery::-webkit-scrollbar-thumb { background: #555; border-radius: 3px; }
            .yuan-mi-gallery::-webkit-scrollbar-thumb:hover { background: #777; }
            .yuan-mi-gallery { scrollbar-width: thin; scrollbar-color: #555 #1a1a1a; }
            .yuan-mi-name-input {
                background: #1a1a2e; color: #ddd; border: 1px solid #3a3a5a;
                border-radius: 3px; padding: 2px 6px; font-size: 10px;
                flex-grow: 1; min-width: 60px; outline: none;
            }
            .yuan-mi-name-input:focus { border-color: #5a7fb8; }
            .yuan-mi-btn {
                border: 1px solid #444; border-radius: 3px; padding: 2px 6px;
                font-size: 10px; cursor: pointer; color: white; white-space: nowrap;
                transition: background 0.15s;
            }
            /* --- 原图预览窗口 --- */
            .yuan-mi-preview-overlay {
                position: fixed; inset: 0; z-index: 99999;
                background: rgba(0, 0, 0, 0.82);
                display: flex; flex-direction: column;
                align-items: center; justify-content: center;
                font-family: system-ui, sans-serif;
            }
            .yuan-mi-preview-toolbar {
                display: flex; align-items: center; gap: 8px;
                padding: 8px 12px; background: #22222e;
                border: 1px solid #3a3a4a; border-radius: 6px;
                margin-bottom: 8px; color: #ddd; font-size: 12px;
                max-width: 92vw; box-sizing: border-box;
            }
            .yuan-mi-preview-btn {
                background: #3a3f4b; border: 1px solid #5a5f6b; color: #fff;
                border-radius: 4px; padding: 3px 10px; font-size: 12px;
                cursor: pointer; white-space: nowrap; user-select: none;
                flex-shrink: 0;
            }
            .yuan-mi-preview-btn:hover { background: #4a4f5b; }
            .yuan-mi-preview-close { background: #5a2a2a; border-color: #7f3a3a; }
            .yuan-mi-preview-close:hover { background: #7a3a3a; }
            .yuan-mi-preview-name {
                max-width: 260px; overflow: hidden; text-overflow: ellipsis;
                white-space: nowrap; color: #aaa;
                direction: rtl; text-align: left; flex-shrink: 1;
            }
            .yuan-mi-preview-viewport {
                position: relative; width: 92vw; height: calc(100vh - 120px);
                min-height: 200px; overflow: hidden;
                border: 1px solid #3a3a4a; border-radius: 4px;
                background: #101014; cursor: grab; touch-action: none;
            }
            .yuan-mi-preview-viewport.dragging { cursor: grabbing; }
            .yuan-mi-preview-img {
                position: absolute; top: 0; left: 0;
                transform-origin: 0 0; will-change: transform;
                user-select: none; -webkit-user-drag: none;
                max-width: none; max-height: none;
            }
            .yuan-mi-preview-hint {
                position: absolute; bottom: 8px; left: 50%;
                transform: translateX(-50%); color: #999;
                font-size: 11px; pointer-events: none;
                background: rgba(0, 0, 0, 0.5); padding: 2px 8px;
                border-radius: 3px; white-space: nowrap;
            }
        `;
        document.head.appendChild(style);
    }

    /** 生成默认滑轨名称：0→A, 1→B, 25→Z, 26→AA... */
    function getDefaultTrackName(index) {
        let name = "";
        let n = index;
        do {
            name = String.fromCharCode(65 + (n % 26)) + name;
            n = Math.floor(n / 26) - 1;
        } while (n >= 0);
        return "批量图像" + name;
    }

    app.registerExtension({
        name: "ComfyUI-Yuan-Tool.MultiImage",
        nodeCreated(node) {
            if (node.comfyClass !== "YuanMultiImage") return;

            // --- 滑轨数据模型 ---
            let tracks = [];
            /** 每个滑轨的 Y 中心位置（相对于节点左上角），供 getConnectionPos 使用 */
            let trackYPositions = [];
            /** 标记是否已通过 onConfigure 恢复数据（防止初始化竞态删端口） */
            let hasBeenConfigured = false;

            // --- V3 前端检测 ---
            let v3NodeElement = null;
            function checkIsV3() {
                if (v3NodeElement) return true;
                let el = container.parentElement;
                while (el) {
                    if ((el.tagName && el.tagName.toLowerCase().includes('comfy-node')) ||
                        (el.classList && el.classList.contains('comfy-node'))) {
                        v3NodeElement = el;
                        return true;
                    }
                    el = el.parentElement || (el.getRootNode ? el.getRootNode().host : null);
                }
                return false;
            }

            // --- 1. UI 主容器（动态高度）---
            const container = document.createElement("div");
            container.style.cssText = `
                width: 100%;
                background: #1e1e2a;
                border: 1px solid #353545;
                border-radius: 4px;
                margin-top: 5px;
                padding: ${CONTAINER_PADDING}px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                gap: 0;
                pointer-events: auto;
                overflow: hidden;
            `;

            // 滑轨列表容器
            const tracksContainer = document.createElement("div");
            tracksContainer.style.cssText = "display: flex; flex-direction: column; gap: 0;";
            container.appendChild(tracksContainer);

            // 添加滑轨按钮
            const addTrackBtn = document.createElement("button");
            addTrackBtn.className = "yuan-mi-btn";
            addTrackBtn.innerText = "+ 添加滑轨";
            addTrackBtn.style.cssText = `
                background: #2d5a2d; border: 1px solid #4a7f4a;
                margin-top: 6px; width: 100%; padding: 4px;
            `;
            addTrackBtn.onmouseenter = () => { addTrackBtn.style.background = "#3a7f3a"; };
            addTrackBtn.onmouseleave = () => { addTrackBtn.style.background = "#2d5a2d"; };
            addTrackBtn.onclick = () => {
                if (tracks.length >= MAX_TRACKS) return;
                tracks.push({
                    name: getDefaultTrackName(tracks.length),
                    paths: [],
                });
                renderAllTracks();
                serializeTracks();
                updateLayout();
            };
            container.appendChild(addTrackBtn);

            // 隐藏的文件上传 input
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.multiple = true;
            fileInput.accept = "image/*";
            fileInput.style.display = "none";
            container.appendChild(fileInput);

            // 当前活跃滑轨索引（用于文件上传目标）
            let activeTrackIndex = 0;
            fileInput.onchange = (e) => {
                // 必须先将 FileList 转为数组，否则 e.target.value="" 会清空 FileList，
                // 导致 async handleFiles 中 await 后续迭代拿不到文件（只能上传第一张）
                const files = Array.from(e.target.files);
                if (files.length > 0) {
                    handleFiles(files, activeTrackIndex);
                }
                e.target.value = "";
            };

            // 将容器作为 DOM widget
            const galleryWidget = node.addDOMWidget("Gallery", "html_gallery", container, { serialize: false });

            // --- 2. 隐藏 tracks_data widget ---
            const dataWidget = node.widgets.find(w => w.name === "tracks_data");
            if (dataWidget) {
                Object.defineProperty(dataWidget, 'hidden', { get: () => true, set: () => {} });
                Object.defineProperty(dataWidget, 'type', { get: () => "hidden", set: () => {} });
                dataWidget.computeSize = function () { return [0, 0]; };

                const hideInterval = setInterval(() => {
                    if (dataWidget.element) dataWidget.element.style.display = "none";
                }, 50);
                dataWidget._hideTimer = hideInterval;
                setTimeout(() => clearInterval(hideInterval), 1000);
            }

            // V3 (Nodes 2.0)：widget 与输入端口共存，
            // 移除 tracks_data 的占位端口（避免节点被端口拉长）
            if (checkIsV3()) {
                const slot = node.findInputSlot("tracks_data");
                if (slot >= 0 && !node.inputs[slot].link) {
                    node.removeInput(slot);
                }
            }

            // --- 3. 序列化 / 反序列化 ---
            function serializeTracks() {
                if (!dataWidget) return;
                const val = JSON.stringify(tracks);
                const tempCb = dataWidget.callback;
                dataWidget.callback = null;
                dataWidget.value = val;
                dataWidget.callback = tempCb;
                // 标记画布脏，确保工作流保存时写入最新 tracks_data
                if (app.graph) app.graph.setDirtyCanvas(true, true);
            }

            function deserializeTracks() {
                if (!dataWidget) {
                    tracks = [createDefaultTrack()];
                    return;
                }
                try {
                    const data = dataWidget.value;
                    const parsed = data ? JSON.parse(data) : [];
                    if (Array.isArray(parsed) && parsed.length > 0) {
                        tracks = parsed.map((t, i) => ({
                            name: (t && typeof t.name === "string") ? t.name : getDefaultTrackName(i),
                            paths: (t && Array.isArray(t.paths)) ? t.paths : [],
                        }));
                    } else {
                        tracks = [createDefaultTrack()];
                    }
                } catch (e) {
                    tracks = [createDefaultTrack()];
                }
            }

            function createDefaultTrack() {
                return { name: getDefaultTrackName(0), paths: [] };
            }

            // --- 4. 输出端口管理 ---
            /** slot 编号与滑轨索引 1:1 映射（所有滑轨始终输出） */
            function slotToTrackIndex(slotNumber) {
                return slotNumber < tracks.length ? slotNumber : -1;
            }

            /**
             * 同步输出端口与 tracks 数组（严格 1:1 对应）
             * @param {number} deletedIndex - 删除滑轨时传入被删除的索引；添加/重命名时不传
             */
            function syncOutputs(deletedIndex) {
                if (!node.outputs) return;
                const targetLen = tracks.length;

                // 删除：精确删除对应 slot，LiteGraph 自动重映射后续连接的 origin_slot
                if (deletedIndex !== undefined && deletedIndex >= 0 && deletedIndex < node.outputs.length) {
                    node.removeOutput(deletedIndex);
                }

                // 增删差异：端口多于 tracks 时从末尾删，少于时从末尾加
                while (node.outputs.length > targetLen) {
                    node.removeOutput(node.outputs.length - 1);
                }
                while (node.outputs.length < targetLen) {
                    node.addOutput(tracks[node.outputs.length].name, "IMAGE");
                }

                // 同步所有端口名称和类型（重命名场景）
                for (let i = 0; i < targetLen; i++) {
                    if (node.outputs[i]) {
                        node.outputs[i].name = tracks[i].name;
                        node.outputs[i].type = "IMAGE";
                    }
                }
            }

            /** 更新输出端口名称（不重建，保持连接） */
            function updateOutputName(trackIndex, newName) {
                if (node.outputs[trackIndex]) {
                    node.outputs[trackIndex].name = newName;
                }
            }

            // --- 5. getConnectionPos 重写：输出端口定位到滑轨右侧 ---
            const origGetConnectionPos = node.getConnectionPos
                ? node.getConnectionPos.bind(node)
                : null;

            node.getConnectionPos = function (isInput, slotNumber, out) {
                out = out || new Float32Array(2);
                if (!isInput) {
                    const trackIdx = slotToTrackIndex(slotNumber);
                    if (trackIdx >= 0 && trackYPositions[trackIdx] !== undefined) {
                        out[0] = this.size[0]; // 右边缘
                        out[1] = trackYPositions[trackIdx];
                        return out;
                    }
                }
                if (origGetConnectionPos) {
                    return origGetConnectionPos(isInput, slotNumber, out);
                }
                out[0] = isInput ? 0 : this.size[0];
                out[1] = 10 + slotNumber * 20;
                return out;
            };

            /** 计算并更新每个滑轨的 Y 中心位置（相对于节点左上角） */
            function updateTrackYPositions() {
                const widgetY = galleryWidget.last_y || 0;
                trackYPositions = tracks.map((track, i) => {
                    const trackStart = widgetY + CONTAINER_PADDING + i * TRACK_TOTAL_HEIGHT;
                    return trackStart + (TRACK_HEADER_HEIGHT + TRACK_GALLERY_HEIGHT) / 2;
                });
            }

            // --- 6. 原图预览（点击缩略图弹出，可放大缩小 / 平移）---
            let previewOverlay = null;
            let previewImg = null;
            let previewZoom = 1;
            let previewFitZoom = 1;
            let previewTx = 0;
            let previewTy = 0;

            function openPreview(path) {
                closePreview();

                const overlay = document.createElement("div");
                overlay.className = "yuan-mi-preview-overlay";

                // 顶部工具栏：缩放按钮 + 文件名 + 关闭
                const toolbar = document.createElement("div");
                toolbar.className = "yuan-mi-preview-toolbar";

                const zoomOutBtn = document.createElement("button");
                zoomOutBtn.className = "yuan-mi-preview-btn";
                zoomOutBtn.innerText = "−";

                const zoomLabel = document.createElement("span");
                zoomLabel.style.minWidth = "44px";
                zoomLabel.style.textAlign = "center";
                zoomLabel.style.flexShrink = "0";
                zoomLabel.innerText = "100%";

                const zoomInBtn = document.createElement("button");
                zoomInBtn.className = "yuan-mi-preview-btn";
                zoomInBtn.innerText = "+";

                const resetBtn = document.createElement("button");
                resetBtn.className = "yuan-mi-preview-btn";
                resetBtn.innerText = "适应窗口";

                const nameSpan = document.createElement("span");
                nameSpan.className = "yuan-mi-preview-name";
                nameSpan.innerText = path;

                const closeBtn = document.createElement("button");
                closeBtn.className = "yuan-mi-preview-btn yuan-mi-preview-close";
                closeBtn.innerText = "✕ 关闭";

                toolbar.append(zoomOutBtn, zoomLabel, zoomInBtn, resetBtn, nameSpan, closeBtn);

                // 视口区域
                const viewport = document.createElement("div");
                viewport.className = "yuan-mi-preview-viewport";

                const img = document.createElement("img");
                img.className = "yuan-mi-preview-img";
                img.src = "/api/view?filename=" + encodeURIComponent(path) + "&type=input";
                img.alt = path;

                const hint = document.createElement("div");
                hint.className = "yuan-mi-preview-hint";
                hint.innerText = "滚轮缩放 · 拖拽平移 · 双击适应窗口 · ESC 关闭";

                viewport.appendChild(img);
                viewport.appendChild(hint);
                overlay.appendChild(toolbar);
                overlay.appendChild(viewport);
                document.body.appendChild(overlay);

                previewOverlay = overlay;
                previewImg = img;
                previewZoom = 1;
                previewFitZoom = 1;
                previewTx = 0;
                previewTy = 0;

                function fitToWindow() {
                    const vw = viewport.clientWidth;
                    const vh = viewport.clientHeight;
                    const iw = img.naturalWidth || vw;
                    const ih = img.naturalHeight || vh;
                    previewFitZoom = Math.min(vw / iw, vh / ih, 1);
                    previewZoom = previewFitZoom;
                    previewTx = (vw - iw * previewZoom) / 2;
                    previewTy = (vh - ih * previewZoom) / 2;
                    updateTransform();
                }

                function updateTransform() {
                    img.style.transform =
                        "translate(" + previewTx + "px," + previewTy + "px) scale(" + previewZoom + ")";
                    zoomLabel.innerText = Math.round(previewZoom * 100) + "%";
                }

                /** 以屏幕坐标 (clientX, clientY) 为锚点缩放 */
                function zoomAt(clientX, clientY, factor) {
                    const rect = viewport.getBoundingClientRect();
                    const mx = clientX - rect.left;
                    const my = clientY - rect.top;
                    const px = (mx - previewTx) / previewZoom;
                    const py = (my - previewTy) / previewZoom;
                    const newZoom = Math.min(Math.max(previewZoom * factor, previewFitZoom * 0.5), 8);
                    if (newZoom === previewZoom) return;
                    previewZoom = newZoom;
                    previewTx = mx - px * previewZoom;
                    previewTy = my - py * previewZoom;
                    updateTransform();
                }

                // 按钮
                zoomInBtn.onclick = () => {
                    zoomAt(viewport.clientWidth / 2, viewport.clientHeight / 2, 1.25);
                };
                zoomOutBtn.onclick = () => {
                    zoomAt(viewport.clientWidth / 2, viewport.clientHeight / 2, 0.8);
                };
                resetBtn.onclick = () => fitToWindow();
                closeBtn.onclick = () => closePreview();

                // 滚轮缩放
                viewport.addEventListener("wheel", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    zoomAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.12 : 1 / 1.12);
                }, { passive: false });

                // 拖拽平移
                let panStart = null;
                viewport.addEventListener("mousedown", (e) => {
                    if (e.button !== 0) return;
                    e.preventDefault();
                    e.stopPropagation();
                    panStart = { x: e.clientX, y: e.clientY, tx: previewTx, ty: previewTy };
                    viewport.classList.add("dragging");
                });
                window.addEventListener("mousemove", onPanMove);
                window.addEventListener("mouseup", onPanUp, { once: true });
                function onPanMove(e) {
                    if (!panStart) return;
                    previewTx = panStart.tx + (e.clientX - panStart.x);
                    previewTy = panStart.ty + (e.clientY - panStart.y);
                    updateTransform();
                }
                function onPanUp() {
                    panStart = null;
                    viewport.classList.remove("dragging");
                    window.removeEventListener("mousemove", onPanMove);
                }

                // 双击适应窗口
                viewport.addEventListener("dblclick", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    fitToWindow();
                });

                // 点击遮罩空白处关闭
                overlay.addEventListener("mousedown", (e) => {
                    if (e.target === overlay) closePreview();
                });

                // ESC 关闭
                const escHandler = (e) => {
                    if (e.key === "Escape") closePreview();
                };
                document.addEventListener("keydown", escHandler);
                overlay._escHandler = escHandler;

                if (img.complete && img.naturalWidth > 0) {
                    fitToWindow();
                } else {
                    img.onload = () => fitToWindow();
                }
            }

            function closePreview() {
                if (!previewOverlay) return;
                if (previewOverlay._escHandler) {
                    document.removeEventListener("keydown", previewOverlay._escHandler);
                }
                previewOverlay.remove();
                previewOverlay = null;
                previewImg = null;
            }

            // --- 7. 滑轨 UI 渲染 ---
            /** 仅渲染 DOM（不触碰端口，避免破坏连接） */
            function renderTracksUI() {
                tracksContainer.innerHTML = "";
                tracks.forEach((track, trackIndex) => {
                    tracksContainer.appendChild(createTrackElement(track, trackIndex));
                });
                updateTrackYPositions();
                updateAddButtonState();
                if (app.graph) app.graph.setDirtyCanvas(true, true);
            }

            /** 渲染 DOM + 同步端口（用于新增/删除滑轨等用户操作） */
            function renderAllTracks() {
                renderTracksUI();
                syncOutputs();
            }

            function updateAddButtonState() {
                addTrackBtn.disabled = tracks.length >= MAX_TRACKS;
                addTrackBtn.style.opacity = tracks.length >= MAX_TRACKS ? "0.4" : "1";
                addTrackBtn.style.cursor = tracks.length >= MAX_TRACKS ? "not-allowed" : "pointer";
            }

            function createTrackElement(track, trackIndex) {
                const trackDiv = document.createElement("div");
                trackDiv.style.cssText = `
                    display: flex; flex-direction: column;
                    margin-bottom: ${TRACK_GAP}px;
                    border: 1px solid #2a2a3a; border-radius: 4px;
                    overflow: hidden; background: #252530;
                `;

                // --- 滑轨头部：名称 + 按钮 ---
                const header = document.createElement("div");
                header.style.cssText = `
                    display: flex; align-items: center; gap: 4px;
                    padding: 3px 6px; height: ${TRACK_HEADER_HEIGHT}px;
                    background: #2a2a38; box-sizing: border-box;
                `;

                // 名称输入框
                const nameInput = document.createElement("input");
                nameInput.className = "yuan-mi-name-input";
                nameInput.type = "text";
                nameInput.value = track.name;
                nameInput.addEventListener("input", () => {
                    tracks[trackIndex].name = nameInput.value || getDefaultTrackName(trackIndex);
                    updateOutputName(trackIndex, tracks[trackIndex].name);
                });
                nameInput.addEventListener("blur", () => {
                    serializeTracks();
                    if (app.graph) app.graph.setDirtyCanvas(true, true);
                });
                // 阻止 LiteGraph 键盘事件
                nameInput.addEventListener("keydown", (e) => e.stopPropagation());

                // 上传按钮
                const uploadBtn = document.createElement("button");
                uploadBtn.className = "yuan-mi-btn";
                uploadBtn.innerText = "上传";
                uploadBtn.style.background = "#3a3f4b";
                uploadBtn.style.borderColor = "#5a5f6b";
                uploadBtn.onmouseenter = () => { uploadBtn.style.background = "#4a4f5b"; };
                uploadBtn.onmouseleave = () => { uploadBtn.style.background = "#3a3f4b"; };
                uploadBtn.onclick = (e) => {
                    e.stopPropagation();
                    activeTrackIndex = trackIndex;
                    fileInput.click();
                };

                // 清空按钮
                const clearBtn = document.createElement("button");
                clearBtn.className = "yuan-mi-btn";
                clearBtn.innerText = "清空";
                clearBtn.style.background = "#5a3a2a";
                clearBtn.style.borderColor = "#7f5a3a";
                clearBtn.onmouseenter = () => { clearBtn.style.background = "#7a5a3a"; };
                clearBtn.onmouseleave = () => { clearBtn.style.background = "#5a3a2a"; };
                clearBtn.onclick = (e) => {
                    e.stopPropagation();
                    tracks[trackIndex].paths = [];
                    renderTrackGallery(trackIndex);
                    serializeTracks();
                };

                // 删除滑轨按钮
                const deleteBtn = document.createElement("button");
                deleteBtn.className = "yuan-mi-btn";
                deleteBtn.innerText = "删除";
                deleteBtn.style.background = "#5a2a2a";
                deleteBtn.style.borderColor = "#7f3a3a";
                deleteBtn.onmouseenter = () => { deleteBtn.style.background = "#7a3a3a"; };
                deleteBtn.onmouseleave = () => { deleteBtn.style.background = "#5a2a2a"; };
                deleteBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (tracks.length <= 1) return;
                    tracks.splice(trackIndex, 1);
                    // 精确删除对应 slot，让 LiteGraph 自动重映射后续连接
                    syncOutputs(trackIndex);
                    renderTracksUI();
                    serializeTracks();
                    updateLayout(true);
                };
                // 仅 1 个滑轨时禁用删除
                deleteBtn.disabled = tracks.length <= 1;
                deleteBtn.style.opacity = tracks.length <= 1 ? "0.4" : "1";
                deleteBtn.style.cursor = tracks.length <= 1 ? "not-allowed" : "pointer";

                header.appendChild(nameInput);
                header.appendChild(uploadBtn);
                header.appendChild(clearBtn);
                header.appendChild(deleteBtn);
                trackDiv.appendChild(header);

                // --- Gallery 区域 ---
                const galleryWrapper = document.createElement("div");
                galleryWrapper.style.cssText = `
                    position: relative; width: 100%;
                    height: ${TRACK_GALLERY_HEIGHT}px;
                    background: #1a1a20; overflow: hidden;
                `;

                const gallery = document.createElement("div");
                gallery.className = "yuan-mi-gallery";
                gallery.style.cssText = `
                    display: flex; flex-direction: row; gap: ${THUMB_GAP}px;
                    height: 100%; width: 100%;
                    overflow-x: auto; overflow-y: hidden;
                    align-items: center; padding: 2px; box-sizing: border-box;
                `;

                // 滚轮：垂直→横向
                gallery.addEventListener("wheel", (e) => {
                    if (gallery.scrollWidth > gallery.clientWidth && e.deltaY !== 0) {
                        e.preventDefault();
                        gallery.scrollLeft += e.deltaY;
                    }
                }, { passive: false });

                // 拖拽上传
                gallery.ondragover = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    gallery.style.background = "#2a2a30";
                };
                gallery.ondragleave = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    gallery.style.background = "#1a1a20";
                };
                gallery.ondrop = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    gallery.style.background = "#1a1a20";
                    // FileList 转数组，防止事件结束后 DataTransfer 被清空
                    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith("image/"));
                    if (files.length > 0) {
                        handleFiles(files, trackIndex);
                    }
                };

                // 点击空白区域上传
                gallery.onclick = (e) => {
                    if (e.target === gallery) {
                        activeTrackIndex = trackIndex;
                        fileInput.click();
                    }
                };

                galleryWrapper.appendChild(gallery);
                trackDiv.appendChild(galleryWrapper);

                // 渲染缩略图
                renderTrackGallery(trackIndex, gallery);

                return trackDiv;
            }

            /** 渲染单个滑轨的缩略图（可指定 gallery 元素，否则查找） */
            function renderTrackGallery(trackIndex, galleryEl) {
                if (!galleryEl) {
                    galleryEl = tracksContainer.querySelectorAll(".yuan-mi-gallery")[trackIndex];
                }
                if (!galleryEl) return;

                const track = tracks[trackIndex];
                if (!track) return;

                galleryEl.innerHTML = "";
                let draggedNode = null;
                let lastSwapTime = 0;

                track.paths.forEach((path, imgIndex) => {
                    const item = document.createElement("div");
                    item.dataset.path = path;
                    item.draggable = true;
                    item.style.cssText = `
                        position: relative; width: ${THUMB_SIZE}px; height: ${THUMB_SIZE}px;
                        flex-shrink: 0; background: #000; border-radius: 3px;
                        border: 1px solid #444; overflow: hidden; cursor: grab;
                        display: flex; align-items: center; justify-content: center;
                    `;

                    const img = document.createElement("img");
                    img.src = "/api/view?filename=" + encodeURIComponent(path) + "&type=input";
                    img.style.cssText = "max-width:100%;max-height:100%;object-fit:contain;pointer-events:auto;display:block;";
                    img.draggable = false;

                    const del = document.createElement("div");
                    del.style.cssText = `
                        position:absolute;top:0;right:0;background:#cc2222;color:white;
                        width:16px;height:16px;display:flex;align-items:center;justify-content:center;
                        font-size:12px;cursor:pointer;z-index:10;border-bottom-left-radius:3px;
                    `;
                    del.innerHTML = '<svg width="8" height="8" viewBox="0 0 10 10"><path d="M1 1L9 9M9 1L1 9" stroke="white" stroke-width="2" stroke-linecap="round"/></svg>';
                    del.onclick = (e) => {
                        e.stopPropagation();
                        tracks[trackIndex].paths.splice(imgIndex, 1);
                        renderTrackGallery(trackIndex, galleryEl);
                        serializeTracks();
                    };

                    const numBadge = document.createElement("div");
                    numBadge.style.cssText = `
                        position:absolute;bottom:0;left:0;background:rgba(0,0,0,0.75);color:#fff;
                        padding:1px 4px;font-size:10px;font-weight:bold;border-top-right-radius:3px;
                        pointer-events:none;z-index:5;
                    `;
                    numBadge.innerText = (imgIndex + 1).toString();

                    item.addEventListener("contextmenu", (e) => e.stopPropagation());

                    // 点击缩略图 → 原图预览（记录按下位置，区分拖拽）
                    let pressX = 0;
                    let pressY = 0;
                    item.addEventListener("mousedown", (e) => {
                        if (e.button !== 0) return;
                        pressX = e.clientX;
                        pressY = e.clientY;
                    });
                    item.addEventListener("click", (e) => {
                        // 发生位移视为拖拽，不触发预览
                        if (Math.abs(e.clientX - pressX) > 4 || Math.abs(e.clientY - pressY) > 4) return;
                        // 点击删除按钮时不触发预览（del.onclick 已 stopPropagation，此处双保险）
                        if (e.target === del || del.contains(e.target)) return;
                        e.stopPropagation();
                        openPreview(path);
                    });

                    // 拖拽排序
                    item.ondragstart = (e) => {
                        draggedNode = item;
                        e.dataTransfer.setData("text/plain", path);
                        e.dataTransfer.effectAllowed = "move";
                        setTimeout(() => {
                            if (draggedNode === item) {
                                item.style.opacity = "0.3";
                            }
                        }, 0);
                    };
                    item.ondragend = () => {
                        if (draggedNode) draggedNode.style.opacity = "1";
                        draggedNode = null;
                        // 保存新顺序
                        const newPaths = Array.from(galleryEl.children).map(n => n.dataset.path);
                        const current = tracks[trackIndex].paths.join("\n");
                        if (newPaths.join("\n") !== current) {
                            tracks[trackIndex].paths = newPaths;
                            serializeTracks();
                        }
                    };
                    item.ondragover = (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        if (!draggedNode || draggedNode === item) return;
                        if (Date.now() - lastSwapTime < 50) return;
                        const rect = item.getBoundingClientRect();
                        const midX = rect.left + rect.width / 2;
                        if (e.clientX < midX) {
                            if (item.previousSibling !== draggedNode) {
                                galleryEl.insertBefore(draggedNode, item);
                                lastSwapTime = Date.now();
                            }
                        } else {
                            if (item.nextSibling !== draggedNode) {
                                galleryEl.insertBefore(draggedNode, item.nextSibling);
                                lastSwapTime = Date.now();
                            }
                        }
                    };
                    item.ondrop = (e) => { e.preventDefault(); e.stopPropagation(); };

                    item.appendChild(img);
                    item.appendChild(del);
                    item.appendChild(numBadge);
                    galleryEl.appendChild(item);
                });
            }

            // --- 7. 文件上传 ---
            async function handleFiles(files, trackIndex) {
                if (trackIndex < 0 || trackIndex >= tracks.length) return;
                const uploaded = [];
                for (const file of files) {
                    if (!file.type.startsWith("image/")) continue;
                    const body = new FormData();
                    body.append("image", file);
                    try {
                        const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                        if (resp.status === 200) {
                            const data = await resp.json();
                            let name = data.name;
                            if (data.subfolder) name = data.subfolder + "/" + name;
                            uploaded.push(name);
                        }
                    } catch (e) { console.error("上传错误", e); }
                }
                if (uploaded.length > 0) {
                    tracks[trackIndex].paths = tracks[trackIndex].paths.concat(uploaded);
                    renderTrackGallery(trackIndex);
                    serializeTracks();
                }
            }

            // --- 8. 布局管理 ---
            function getContainerHeight() {
                return CONTAINER_PADDING * 2 + tracks.length * TRACK_TOTAL_HEIGHT + ADD_BTN_HEIGHT + 6;
            }

            galleryWidget.computeSize = function (width) {
                const nodeWidth = node.size?.[0] || width || 220;
                return [Math.max(10, nodeWidth - 30), getContainerHeight()];
            };

            function getAbsoluteMinHeight() {
                const galleryY = galleryWidget.last_y || 40;
                return galleryY + getContainerHeight() + 5;
            }

            // 最小宽度需容纳轨道头部内容：header padding(12) + 名称输入框(60)
            // + 上传/清空/删除按钮(34*3=102) + 间距(16) = 190，加容器 padding(16)
            // 和右侧端口区(30) ≈ 236，取 240 确保按钮不被裁剪
            function getMinW() { return 240; }

            let v3EventsAttached = false;
            function enforceV3CSS() {
                const isV3 = checkIsV3();
                if (isV3 && v3NodeElement) {
                    v3NodeElement.style.removeProperty("min-width");
                    v3NodeElement.style.setProperty("min-height", getAbsoluteMinHeight() + "px", "important");
                    if (!v3EventsAttached) {
                        v3EventsAttached = true;
                        v3NodeElement.addEventListener("dragover", (e) => e.preventDefault());
                        v3NodeElement.addEventListener("drop", (e) => {
                            if (e.dataTransfer && e.dataTransfer.files) {
                                const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith("image/"));
                                if (files.length > 0) {
                                    e.preventDefault();
                                    e.stopPropagation();
                                    handleFiles(files, activeTrackIndex);
                                }
                            }
                        });
                    }
                }
            }

            let isLayouting = false;
            function updateLayout(forceShrink) {
                if (isLayouting) return;
                isLayouting = true;
                const minW = getMinW();
                const absMinH = getAbsoluteMinHeight();
                node.min_size = [minW, absMinH];
                enforceV3CSS();

                let targetW = Math.max(node.size[0], minW);
                let targetH = forceShrink ? absMinH : Math.max(node.size[1], absMinH);
                if (node.size[0] !== targetW || node.size[1] !== targetH) {
                    node.setSize([targetW, targetH]);
                    if (app.graph) app.graph.setDirtyCanvas(true, true);
                }
                container.style.height = getContainerHeight() + "px";
                updateTrackYPositions();
                isLayouting = false;
            }

            // 重写尺寸方法
            const origOnResize = node.onResize;
            node.onResize = function (size) {
                size[0] = Math.max(size[0], getMinW());
                size[1] = Math.max(size[1], getAbsoluteMinHeight());
                if (origOnResize) origOnResize.call(this, size);
                if (isLayouting) return;
                node.min_size = [getMinW(), getAbsoluteMinHeight()];
                enforceV3CSS();
                container.style.height = getContainerHeight() + "px";
                updateTrackYPositions();
            };

            const origComputeSize = node.computeSize;
            node.computeSize = function (out) {
                let res = origComputeSize ? origComputeSize.apply(this, arguments) : [getMinW(), 200];
                res[0] = Math.max(res[0], getMinW());
                res[1] = Math.max(res[1], getAbsoluteMinHeight());
                node.min_size = [getMinW(), getAbsoluteMinHeight()];
                enforceV3CSS();
                return res;
            };

            const origSetSize = node.setSize;
            node.setSize = function (size) {
                size[0] = Math.max(size[0], getMinW());
                size[1] = Math.max(size[1], getAbsoluteMinHeight());
                if (origSetSize) origSetSize.call(this, size); else this.size = size;
                enforceV3CSS();
            };

            // --- 9. 序列化 / 反序列化钩子 ---
            // tracks_data widget 由 ComfyUI 自动序列化/恢复，无需额外 onSerialize
            /** 防止 onConfigure/onAdded/setTimeout 三处初始化重复执行 */
            let uiInitialized = false;
            function ensureInitialized() {
                if (uiInitialized) return;
                uiInitialized = true;
                if (hasBeenConfigured) {
                    // 已通过 onConfigure 恢复数据和端口，仅渲染 UI
                    renderTracksUI();
                } else {
                    // 新建节点：初始化默认滑轨和端口
                    renderAllTracks();
                    serializeTracks();
                }
                updateLayout(true);
                updateTrackYPositions();
                if (app.graph) app.graph.setDirtyCanvas(true, true);
            }

            const origOnConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                const out = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                // 标记已配置，阻止 onAdded/初始化中的竞态 syncOutputs
                hasBeenConfigured = true;

                // 同步恢复 tracks 数据（origOnConfigure 后 dataWidget.value 已从工作流恢复）
                deserializeTracks();
                if (tracks.length === 0) tracks = [createDefaultTrack()];

                // 同步端口：此时 LiteGraph 已恢复 node.outputs[i].links，
                // onConfigure 时端口数量与 tracks 一致，只同步名称不删端口
                syncOutputs();

                // 异步渲染 UI（等 DOM 就绪）
                requestAnimationFrame(ensureInitialized);
                return out;
            };

            // --- 10. 拖拽 / 粘贴 ---
            const origOnDragDrop = node.onDragDrop;
            node.onDragDrop = function (e) {
                if (e.dataTransfer && e.dataTransfer.files) {
                    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith("image/"));
                    if (files.length > 0) {
                        e.preventDefault();
                        handleFiles(files, activeTrackIndex);
                        return true;
                    }
                }
                if (origOnDragDrop) return origOnDragDrop.apply(this, arguments);
                return false;
            };

            const origOnDragOver = node.onDragOver;
            node.onDragOver = function (e) {
                if (e.dataTransfer && e.dataTransfer.items) {
                    const hasImage = Array.from(e.dataTransfer.items).some(
                        f => f.kind === "file" && f.type.startsWith("image/")
                    );
                    if (hasImage) { e.preventDefault(); return true; }
                }
                if (origOnDragOver) return origOnDragOver.apply(this, arguments);
                return false;
            };

            const pasteHandler = (e) => {
                if (app.canvas.selected_nodes && app.canvas.selected_nodes[node.id]) {
                    const items = e.clipboardData?.items;
                    if (!items) return;
                    const files = [];
                    for (let i = 0; i < items.length; i++) {
                        if (items[i].kind === "file" && items[i].type.startsWith("image/")) {
                            files.push(items[i].getAsFile());
                        }
                    }
                    if (files.length > 0) {
                        e.preventDefault();
                        e.stopImmediatePropagation();
                        handleFiles(files, activeTrackIndex);
                    }
                }
            };
            document.addEventListener("paste", pasteHandler, { capture: true });

            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                document.removeEventListener("paste", pasteHandler, { capture: true });
                // 清理隐藏 widget 的轮询定时器（最长 1 秒自清理，此处提前终止）
                if (dataWidget && dataWidget._hideTimer) clearInterval(dataWidget._hideTimer);
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };

            // --- 11. 初始化 ---
            // 先尝试从 widget 恢复，否则创建默认滑轨
            if (dataWidget && dataWidget.value && dataWidget.value.trim()) {
                deserializeTracks();
            } else {
                tracks = [createDefaultTrack()];
            }

            // 新建节点立即设置足够宽度，避免轨道头部按钮被裁剪
            if (node.size) {
                node.size[0] = Math.max(node.size[0] || 0, getMinW());
            }

            // 首次添加到图时初始化
            const origOnAdded = node.onAdded;
            node.onAdded = function () {
                if (origOnAdded) origOnAdded.apply(this, arguments);
                requestAnimationFrame(ensureInitialized);
            };

            // 延迟初始化兜底（确保 DOM 就绪）
            setTimeout(ensureInitialized, 50);

            // 一次性修正：DOM 渲染后 last_y 已稳定，按实际偏移重新计算节点高度
            // （新建节点首次渲染时 last_y=0，估算高度与实际可能有偏差）
            setTimeout(() => {
                updateLayout(true);
                updateTrackYPositions();
            }, 300);
        }
    });
})();
