/**
 * Yuan Tool · 音频节点前端
 *
 * - Yuan 音频列表：每个音频一个固定方块（序号/文件名/时长/删除）；点击播放/暂停，
 *   全局互斥（同一时刻仅一个播放）；滚轮上下滑动横向滚动滑轨（仅滚动，不触发播放）；
 *   支持上传 / 从输入目录选择 / 拖拽排序；状态持久化到 audio_list_data。
 * - Yuan 音频分流：按「输出数量」widget 动态修剪输出端口（默认 2，上限 30）。
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
    const MAX_AUDIOS = 30; // 音频数量上限（与后端一致）
    const BLOCK_SIZE = 84;
    const BLOCK_GAP = 6;
    const GALLERY_HEIGHT = 96;
    const BTN_ROW_HEIGHT = 28;
    const CONTAINER_PADDING = 8;

    // --- 音频/视频扩展名（视频提取音轨，与后端一致） ---
    const AUDIO_EXTS = [
        ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma",
        ".aiff", ".aif", ".mp4", ".m4v", ".webm", ".mov", ".avi", ".mkv",
    ];
    function isAudioFileName(name) {
        const dot = name.lastIndexOf(".");
        if (dot < 0) return false;
        return AUDIO_EXTS.indexOf(name.slice(dot).toLowerCase()) >= 0;
    }

    // --- 工具函数 ---
    /** 构造 /api/view 播放地址，支持 "子目录/文件名" 形式 */
    function audioUrl(path) {
        const idx = path.lastIndexOf("/");
        let subfolder = "", filename = path;
        if (idx >= 0) { subfolder = path.slice(0, idx); filename = path.slice(idx + 1); }
        let url = "/api/view?filename=" + encodeURIComponent(filename) + "&type=input";
        if (subfolder) url += "&subfolder=" + encodeURIComponent(subfolder);
        return url;
    }

    function fmtDuration(sec) {
        if (!isFinite(sec) || sec < 0) return "";
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return m + ":" + (s < 10 ? "0" : "") + s;
    }

    // --- 注入全局样式 ---
    if (!document.getElementById("yuan-tool-audio-list-style")) {
        const style = document.createElement("style");
        style.id = "yuan-tool-audio-list-style";
        style.textContent = `
            .yuan-al-gallery::-webkit-scrollbar { height: 6px; }
            .yuan-al-gallery::-webkit-scrollbar-track { background: #1a1a1a; border-radius: 3px; }
            .yuan-al-gallery::-webkit-scrollbar-thumb { background: #555; border-radius: 3px; }
            .yuan-al-gallery::-webkit-scrollbar-thumb:hover { background: #777; }
            .yuan-al-gallery { scrollbar-width: thin; scrollbar-color: #555 #1a1a1a; }
            .yuan-al-btn {
                border: 1px solid #444; border-radius: 3px; padding: 2px 6px;
                font-size: 10px; cursor: pointer; color: white; white-space: nowrap;
                transition: background 0.15s; flex: 1;
            }
            /* --- 音频小方块 --- */
            .yuan-al-block {
                position: relative; width: ${BLOCK_SIZE}px; height: ${BLOCK_SIZE}px;
                flex-shrink: 0; background: linear-gradient(160deg, #23232e, #1a1a24);
                border: 1px solid #3a3a4a; border-radius: 5px; overflow: hidden;
                cursor: pointer; user-select: none;
                display: flex; flex-direction: column; align-items: center;
                justify-content: flex-start; padding-top: 16px; box-sizing: border-box;
                transition: border-color 0.15s, box-shadow 0.15s;
            }
            .yuan-al-block:hover { border-color: #6a8fbf; }
            .yuan-al-block.playing {
                border-color: #4caf50;
                box-shadow: 0 0 6px rgba(76, 175, 80, 0.55);
                background: linear-gradient(160deg, #1e3020, #182018);
            }
            .yuan-al-num {
                position: absolute; top: 0; left: 0;
                background: rgba(0, 0, 0, 0.75); color: #fff;
                padding: 1px 5px; font-size: 10px; font-weight: bold;
                border-bottom-right-radius: 4px; pointer-events: none;
            }
            .yuan-al-del {
                position: absolute; top: 0; right: 0;
                background: #cc2222; color: white;
                width: 15px; height: 15px; display: flex; align-items: center;
                justify-content: center; font-size: 11px; cursor: pointer;
                z-index: 10; border-bottom-left-radius: 4px;
            }
            .yuan-al-del:hover { background: #ee3333; }
            .yuan-al-icon {
                font-size: 18px; color: #8ab4d8; opacity: 0.45;
                line-height: 1; pointer-events: none; margin-top: 2px;
                transition: opacity 0.15s;
            }
            .yuan-al-block:hover .yuan-al-icon { opacity: 0.95; }
            .yuan-al-block.playing .yuan-al-icon { display: none; }
            /* 播放中的均衡器动画 */
            .yuan-al-eq {
                display: none; align-items: flex-end; gap: 2px;
                height: 16px; margin-top: 2px; pointer-events: none;
            }
            .yuan-al-block.playing .yuan-al-eq { display: flex; }
            .yuan-al-eq span {
                width: 3px; background: #4caf50; border-radius: 1px;
                animation: yuanAlEq 0.9s ease-in-out infinite;
            }
            .yuan-al-eq span:nth-child(2) { animation-delay: 0.25s; }
            .yuan-al-eq span:nth-child(3) { animation-delay: 0.5s; }
            .yuan-al-eq span:nth-child(4) { animation-delay: 0.12s; }
            @keyframes yuanAlEq {
                0%, 100% { height: 4px; }
                50% { height: 16px; }
            }
            .yuan-al-name {
                max-width: 94%; font-size: 9px; color: #ccc; text-align: center;
                margin-top: 5px; line-height: 1.2; overflow: hidden;
                text-overflow: ellipsis; display: -webkit-box;
                -webkit-line-clamp: 2; -webkit-box-orient: vertical;
                pointer-events: none; word-break: break-all;
            }
            .yuan-al-dur {
                font-size: 9px; color: #888; margin-top: auto; margin-bottom: 4px;
                pointer-events: none;
            }
            /* --- 添加音频文件选择面板 --- */
            .yuan-al-picker-overlay {
                position: fixed; inset: 0; z-index: 99999;
                background: rgba(0, 0, 0, 0.6);
                display: flex; align-items: center; justify-content: center;
                font-family: system-ui, sans-serif;
            }
            .yuan-al-picker {
                width: 360px; max-width: 90vw; max-height: 70vh;
                background: #1e1e28; border: 1px solid #3a3a4a; border-radius: 8px;
                display: flex; flex-direction: column; overflow: hidden;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            }
            .yuan-al-picker-header {
                display: flex; align-items: center; justify-content: space-between;
                padding: 10px 12px; background: #262634; border-bottom: 1px solid #3a3a4a;
                color: #eee; font-size: 13px;
            }
            .yuan-al-picker-close {
                background: #5a2a2a; border: 1px solid #7f3a3a; color: #fff;
                border-radius: 4px; padding: 2px 8px; font-size: 12px; cursor: pointer;
            }
            .yuan-al-picker-close:hover { background: #7a3a3a; }
            .yuan-al-picker-search {
                margin: 8px 10px; padding: 5px 8px; background: #14141c;
                border: 1px solid #3a3a4a; border-radius: 4px; color: #ddd;
                font-size: 12px; outline: none;
            }
            .yuan-al-picker-search:focus { border-color: #5a7fb8; }
            .yuan-al-picker-list {
                flex: 1; overflow-y: auto; padding: 0 10px 8px;
            }
            .yuan-al-picker-item {
                display: flex; align-items: center; justify-content: space-between;
                gap: 8px; padding: 6px 8px; margin-bottom: 3px;
                background: #23232e; border: 1px solid #323240; border-radius: 4px;
                color: #ccc; font-size: 12px; cursor: pointer;
            }
            .yuan-al-picker-item:hover { border-color: #5a7fb8; background: #2a2a38; }
            .yuan-al-picker-item .yuan-al-pi-name {
                flex: 1; overflow: hidden; text-overflow: ellipsis;
                white-space: nowrap; direction: rtl; text-align: left;
            }
            .yuan-al-picker-item .yuan-al-pi-tag {
                flex-shrink: 0; font-size: 10px; padding: 1px 6px;
                border-radius: 3px; background: #2d5a2d; color: #bfe8bf;
            }
            .yuan-al-picker-item.selected { border-color: #4caf50; }
            .yuan-al-picker-item.selected .yuan-al-pi-tag { background: #4caf50; color: #fff; }
            .yuan-al-picker-empty {
                text-align: center; color: #777; font-size: 12px; padding: 20px 0;
            }
            .yuan-al-picker-footer {
                padding: 8px 12px; background: #262634; border-top: 1px solid #3a3a4a;
                display: flex; justify-content: flex-end;
            }
            .yuan-al-picker-done {
                background: #2d5a2d; border: 1px solid #4a7f4a; color: #fff;
                border-radius: 4px; padding: 4px 16px; font-size: 12px; cursor: pointer;
            }
            .yuan-al-picker-done:hover { background: #3a7f3a; }
        `;
        document.head.appendChild(style);
    }

    // --- 全局互斥播放器：整个页面同一时刻只允许一个音频播放 ---
    let globalPlayer = null; // { audio: Audio, blockEl: HTMLElement }

    function setBlockPlaying(blockEl, playing) {
        if (!blockEl) return;
        blockEl.classList.toggle("playing", !!playing);
    }

    /** 停止全局当前播放并复位方块视觉 */
    function stopGlobalPlayback() {
        if (!globalPlayer) return;
        const { audio, blockEl } = globalPlayer;
        try { audio.pause(); } catch (_) {}
        try { audio.src = ""; } catch (_) {}
        setBlockPlaying(blockEl, false);
        globalPlayer = null;
    }

    /**
     * 播放/暂停切换（互斥）：
     * - 该方块正在播放 → 暂停；
     * - 否则先停止全局其他播放（自动关闭另一个），再播放该方块。
     */
    function togglePlayback(blockEl, src) {
        if (globalPlayer && globalPlayer.blockEl === blockEl) {
            stopGlobalPlayback();
            return;
        }
        stopGlobalPlayback(); // 互斥：只允许一个播放
        const audio = new Audio(src);
        audio.preload = "auto";
        globalPlayer = { audio, blockEl };
        setBlockPlaying(blockEl, true);
        audio.onended = () => { if (globalPlayer && globalPlayer.audio === audio) stopGlobalPlayback(); };
        audio.onerror = () => { if (globalPlayer && globalPlayer.audio === audio) stopGlobalPlayback(); };
        const p = audio.play();
        if (p && p.catch) p.catch(() => stopGlobalPlayback());
    }

    app.registerExtension({
        name: "Yuan-Tool.AudioList",
        nodeCreated(node) {
            // --- Yuan 音频分流：按「输出数量」修剪输出端口 + 动态调节尺寸 ---
            if (node.comfyClass === "YuanAudioSplit") {
                const w = node.widgets?.find(x => x.name === "输出数量");
                if (!w) return;

                const syncOutputs = () => {
                    const n = Math.max(1, Math.min(MAX_AUDIOS, parseInt(w.value) || 2));
                    while (node.outputs.length > n) node.removeOutput(node.outputs.length - 1);
                    while (node.outputs.length < n) {
                        node.addOutput("音频" + (node.outputs.length + 1), "AUDIO");
                    }
                    // 根据端口数量重算节点高度（后端声明 30 端口，初始会渲染过高）
                    if (node.computeSize && node.setSize) {
                        const sz = node.computeSize();
                        const width = Math.max(node.size?.[0] || 0, sz[0]);
                        node.setSize([width, sz[1]]);
                    }
                    if (app.graph) app.graph.setDirtyCanvas(true, true);
                };

                const origCallback = w.callback;
                w.callback = function (...args) {
                    const out = origCallback ? origCallback.apply(this, args) : undefined;
                    syncOutputs();
                    return out;
                };

                // 工作流加载后按 widget 值同步（保存的 outputs 数量可能不一致）
                const origOnConfigure = node.onConfigure;
                node.onConfigure = function (info) {
                    const out = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                    syncOutputs();
                    return out;
                };

                // 节点加入画布后 widget 布局就绪，再校准一次尺寸
                const origOnAdded = node.onAdded;
                node.onAdded = function () {
                    if (origOnAdded) origOnAdded.apply(this, arguments);
                    requestAnimationFrame(syncOutputs);
                };

                syncOutputs();
                return;
            }

            if (node.comfyClass !== "YuanAudioList") return;

            // --- 数据模型：音频文件相对路径列表（input 目录） ---
            let audioFiles = [];
            let pickerOverlay = null;
            // 拖拽排序状态（所有方块共享，定义在方块闭包内会导致交换失效）
            let draggedNode = null;
            let lastSwapTime = 0;
            // 本节点创建的方块集合（onRemoved 时判断播放归属，避免误停其他节点）
            const myBlocks = new WeakSet();

            // --- V3 前端检测 ---
            let v3NodeElement = null;
            function checkIsV3() {
                if (v3NodeElement) return true;
                let el = container.parentElement;
                while (el) {
                    if ((el.tagName && el.tagName.toLowerCase().includes("comfy-node")) ||
                        (el.classList && el.classList.contains("comfy-node"))) {
                        v3NodeElement = el;
                        return true;
                    }
                    el = el.parentElement || (el.getRootNode ? el.getRootNode().host : null);
                }
                return false;
            }

            // --- 1. UI 主容器 ---
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

            // 音频方块滑轨
            const gallery = document.createElement("div");
            gallery.className = "yuan-al-gallery";
            gallery.style.cssText = `
                display: flex; flex-direction: row; gap: ${BLOCK_GAP}px;
                height: ${GALLERY_HEIGHT}px; width: 100%;
                overflow-x: auto; overflow-y: hidden;
                align-items: center; padding: 2px; box-sizing: border-box;
                background: #1a1a20; border-radius: 3px;
            `;
            container.appendChild(gallery);

            // 按钮行
            const btnRow = document.createElement("div");
            btnRow.style.cssText = `
                display: flex; gap: 6px; margin-top: 6px; height: ${BTN_ROW_HEIGHT}px;
            `;

            const addBtn = document.createElement("button");
            addBtn.className = "yuan-al-btn";
            addBtn.innerText = "+ 添加音频";
            addBtn.style.background = "#2d4a5a";
            addBtn.style.borderColor = "#4a7f8f";
            addBtn.onmouseenter = () => { addBtn.style.background = "#3a6a7f"; };
            addBtn.onmouseleave = () => { addBtn.style.background = "#2d4a5a"; };
            addBtn.onclick = (e) => { e.stopPropagation(); openFilePicker(); };

            const uploadBtn = document.createElement("button");
            uploadBtn.className = "yuan-al-btn";
            uploadBtn.innerText = "上传";
            uploadBtn.style.background = "#3a3f4b";
            uploadBtn.style.borderColor = "#5a5f6b";
            uploadBtn.onmouseenter = () => { uploadBtn.style.background = "#4a4f5b"; };
            uploadBtn.onmouseleave = () => { uploadBtn.style.background = "#3a3f4b"; };
            uploadBtn.onclick = (e) => { e.stopPropagation(); fileInput.click(); };

            const clearBtn = document.createElement("button");
            clearBtn.className = "yuan-al-btn";
            clearBtn.innerText = "清空";
            clearBtn.style.background = "#5a3a2a";
            clearBtn.style.borderColor = "#7f5a3a";
            clearBtn.onmouseenter = () => { clearBtn.style.background = "#7a5a3a"; };
            clearBtn.onmouseleave = () => { clearBtn.style.background = "#5a3a2a"; };
            clearBtn.onclick = (e) => {
                e.stopPropagation();
                stopGlobalPlayback();
                audioFiles = [];
                renderGallery();
                serializeFiles();
            };

            btnRow.append(addBtn, uploadBtn, clearBtn);
            container.appendChild(btnRow);

            // 隐藏的文件上传 input
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.multiple = true;
            fileInput.accept = "audio/*,video/*," + AUDIO_EXTS.join(",");
            fileInput.style.display = "none";
            container.appendChild(fileInput);
            fileInput.onchange = (e) => {
                // FileList 先转数组，防止 value 重置后异步迭代丢失文件
                const files = Array.from(e.target.files);
                if (files.length > 0) handleFiles(files);
                e.target.value = "";
            };

            const galleryWidget = node.addDOMWidget("AudioGallery", "yuan_audio_gallery", container, { serialize: false });

            // --- 2. 隐藏 audio_list_data widget ---
            const dataWidget = node.widgets.find(w => w.name === "audio_list_data");
            if (dataWidget) {
                Object.defineProperty(dataWidget, "hidden", { get: () => true, set: () => {} });
                Object.defineProperty(dataWidget, "type", { get: () => "hidden", set: () => {} });
                dataWidget.computeSize = function () { return [0, 0]; };

                const hideInterval = setInterval(() => {
                    if (dataWidget.element) dataWidget.element.style.display = "none";
                }, 50);
                dataWidget._hideTimer = hideInterval;
                setTimeout(() => clearInterval(hideInterval), 1000);
            }

            // V3：移除 audio_list_data 占位输入端口（避免节点被拉长）
            if (checkIsV3()) {
                const slot = node.findInputSlot("audio_list_data");
                if (slot >= 0 && !node.inputs[slot].link) {
                    node.removeInput(slot);
                }
            }

            // --- 3. 序列化 / 反序列化 ---
            function serializeFiles() {
                if (!dataWidget) return;
                const val = JSON.stringify({ files: audioFiles });
                const tempCb = dataWidget.callback;
                dataWidget.callback = null;
                dataWidget.value = val;
                dataWidget.callback = tempCb;
                if (app.graph) app.graph.setDirtyCanvas(true, true);
            }

            function deserializeFiles() {
                if (!dataWidget) return;
                try {
                    const data = dataWidget.value;
                    const parsed = data ? JSON.parse(data) : null;
                    if (parsed && Array.isArray(parsed.files)) {
                        audioFiles = parsed.files.filter(f => typeof f === "string" && f);
                    }
                } catch (e) {
                    audioFiles = [];
                }
            }

            // --- 4. 音频方块渲染 ---
            function renderGallery() {
                // 记录本节点正在播放的音频路径（重渲染会替换 DOM，需恢复视觉状态）
                let playingPath = null;
                if (globalPlayer && myBlocks.has(globalPlayer.blockEl) && globalPlayer.blockEl.dataset) {
                    playingPath = globalPlayer.blockEl.dataset.path;
                }

                gallery.innerHTML = "";
                if (audioFiles.length === 0) {
                    const empty = document.createElement("div");
                    empty.style.cssText = `
                        width: 100%; text-align: center; color: #666;
                        font-size: 11px; pointer-events: none;
                    `;
                    empty.innerText = "点击「+ 添加音频」选择文件，或拖拽 / 上传音频";
                    gallery.appendChild(empty);
                    return;
                }
                audioFiles.forEach((path, index) => {
                    gallery.appendChild(createAudioBlock(path, index));
                });

                // 恢复播放视觉：音频仍在播，把 blockEl 指向新方块（旧元素已成孤儿）
                if (playingPath !== null && globalPlayer) {
                    const newBlock = Array.from(gallery.querySelectorAll(".yuan-al-block"))
                        .find(b => b.dataset.path === playingPath);
                    if (newBlock) {
                        globalPlayer.blockEl = newBlock;
                        setBlockPlaying(newBlock, true);
                    }
                }
                if (app.graph) app.graph.setDirtyCanvas(true, true);
            }

            function createAudioBlock(path, index) {
                const block = document.createElement("div");
                block.className = "yuan-al-block";
                block.dataset.path = path;
                myBlocks.add(block);

                // 序号徽标
                const num = document.createElement("div");
                num.className = "yuan-al-num";
                num.innerText = (index + 1).toString();

                // 删除按钮
                const del = document.createElement("div");
                del.className = "yuan-al-del";
                del.innerHTML = '<svg width="8" height="8" viewBox="0 0 10 10"><path d="M1 1L9 9M9 1L1 9" stroke="white" stroke-width="2" stroke-linecap="round"/></svg>';
                del.onclick = (e) => {
                    e.stopPropagation();
                    if (globalPlayer && globalPlayer.blockEl === block) stopGlobalPlayback();
                    // 按路径定位当前索引（拖拽排序后闭包 index 已过期，splice 会删错）
                    const i = audioFiles.indexOf(path);
                    if (i !== -1) audioFiles.splice(i, 1);
                    renderGallery();
                    serializeFiles();
                };

                // 播放图标 + 均衡器动画
                const icon = document.createElement("div");
                icon.className = "yuan-al-icon";
                icon.innerText = "▶";
                const eq = document.createElement("div");
                eq.className = "yuan-al-eq";
                eq.innerHTML = "<span></span><span></span><span></span><span></span>";

                // 文件名
                const name = document.createElement("div");
                name.className = "yuan-al-name";
                name.innerText = path.split("/").pop();
                name.title = path;

                // 时长
                const dur = document.createElement("div");
                dur.className = "yuan-al-dur";
                dur.innerText = "···";

                block.append(num, del, icon, eq, name, dur);

                // 异步加载时长（仅读元数据）
                loadDuration(path, dur);

                // 点击播放/暂停（区分拖拽位移）
                let pressX = 0, pressY = 0;
                block.addEventListener("mousedown", (e) => {
                    if (e.button !== 0) return;
                    pressX = e.clientX; pressY = e.clientY;
                });
                block.addEventListener("click", (e) => {
                    if (Math.abs(e.clientX - pressX) > 4 || Math.abs(e.clientY - pressY) > 4) return;
                    if (e.target === del || del.contains(e.target)) return;
                    e.stopPropagation();
                    togglePlayback(block, audioUrl(path));
                });
                block.addEventListener("contextmenu", (e) => e.stopPropagation());

                // 拖拽排序
                block.draggable = true;
                block.ondragstart = (e) => {
                    draggedNode = block;
                    e.dataTransfer.setData("text/plain", path);
                    e.dataTransfer.effectAllowed = "move";
                    setTimeout(() => { if (draggedNode === block) block.style.opacity = "0.3"; }, 0);
                };
                block.ondragend = () => {
                    if (draggedNode) draggedNode.style.opacity = "1";
                    draggedNode = null;
                    // 保存新顺序并重渲染（刷新序号徽标）
                    const newPaths = Array.from(gallery.querySelectorAll(".yuan-al-block")).map(n => n.dataset.path);
                    if (newPaths.join("\n") !== audioFiles.join("\n")) {
                        audioFiles = newPaths;
                        serializeFiles();
                        renderGallery();
                    }
                };
                block.ondragover = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (!draggedNode || draggedNode === block) return;
                    if (Date.now() - lastSwapTime < 50) return;
                    const rect = block.getBoundingClientRect();
                    const midX = rect.left + rect.width / 2;
                    if (e.clientX < midX) {
                        if (block.previousSibling !== draggedNode) {
                            gallery.insertBefore(draggedNode, block);
                            lastSwapTime = Date.now();
                        }
                    } else {
                        if (block.nextSibling !== draggedNode) {
                            gallery.insertBefore(draggedNode, block.nextSibling);
                            lastSwapTime = Date.now();
                        }
                    }
                };
                block.ondrop = (e) => { e.preventDefault(); e.stopPropagation(); };

                return block;
            }

            /** 读取音频元数据获取时长显示 */
            function loadDuration(path, durEl) {
                try {
                    const probe = new Audio();
                    probe.preload = "metadata";
                    probe.src = audioUrl(path);
                    probe.onloadedmetadata = () => {
                        if (durEl && durEl.isConnected) durEl.innerText = fmtDuration(probe.duration);
                    };
                } catch (_) {}
            }

            // --- 5. 滚轮：上下滑动横向滚动滑轨（不触发播放） ---
            gallery.addEventListener("wheel", (e) => {
                if (gallery.scrollWidth > gallery.clientWidth && e.deltaY !== 0) {
                    e.preventDefault();
                    gallery.scrollLeft += e.deltaY;
                }
            }, { passive: false });

            // --- 6. 拖拽上传 / 点击空白上传 ---
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
                const files = Array.from(e.dataTransfer.files).filter(f => isAudioFileName(f.name));
                if (files.length > 0) handleFiles(files);
            };
            gallery.onclick = (e) => {
                if (e.target === gallery && audioFiles.length === 0) fileInput.click();
            };

            // --- 7. 上传到 input 目录 ---
            async function handleFiles(files) {
                const uploaded = [];
                for (const file of files) {
                    if (!isAudioFileName(file.name)) continue;
                    const body = new FormData();
                    body.append("image", file); // /upload/image 端点按字段名取文件，不限扩展名
                    body.append("type", "input");
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
                    audioFiles = audioFiles.concat(uploaded).slice(0, MAX_AUDIOS);
                    renderGallery();
                    serializeFiles();
                }
            }

            // --- 8. 从输入目录添加音频（复刻原生加载音频的文件列表） ---
            async function openFilePicker() {
                let files = [];
                try {
                    const resp = await api.fetchApi("/yuan_tool/audio_files");
                    if (resp && resp.ok) {
                        const data = await resp.json();
                        if (Array.isArray(data.files)) files = data.files;
                    }
                } catch (e) {
                    console.error("获取音频文件列表失败", e);
                }

                closeFilePicker();

                const overlay = document.createElement("div");
                overlay.className = "yuan-al-picker-overlay";

                const panel = document.createElement("div");
                panel.className = "yuan-al-picker";

                // 头部
                const header = document.createElement("div");
                header.className = "yuan-al-picker-header";
                const title = document.createElement("span");
                function updatePickerTitle() {
                    title.innerText = "从输入目录添加音频（已选 " + audioFiles.length + "/" + MAX_AUDIOS + "，目录共 " + files.length + "）";
                }
                updatePickerTitle();
                const closeBtn = document.createElement("button");
                closeBtn.className = "yuan-al-picker-close";
                closeBtn.innerText = "✕ 关闭";
                closeBtn.onclick = () => closeFilePicker();
                header.append(title, closeBtn);

                // 搜索框
                const search = document.createElement("input");
                search.className = "yuan-al-picker-search";
                search.type = "text";
                search.placeholder = "搜索文件名…";

                // 文件列表
                const list = document.createElement("div");
                list.className = "yuan-al-picker-list";

                function renderPickerList(keyword) {
                    list.innerHTML = "";
                    const kw = (keyword || "").toLowerCase();
                    const filtered = files.filter(f => f.toLowerCase().includes(kw));
                    if (filtered.length === 0) {
                        const empty = document.createElement("div");
                        empty.className = "yuan-al-picker-empty";
                        empty.innerText = "没有匹配的音频文件，可先上传到输入目录";
                        list.appendChild(empty);
                        return;
                    }
                    filtered.forEach(f => {
                        const item = document.createElement("div");
                        item.className = "yuan-al-picker-item";
                        const nameSpan = document.createElement("span");
                        nameSpan.className = "yuan-al-pi-name";
                        nameSpan.innerText = f;
                        nameSpan.title = f;
                        const tag = document.createElement("span");
                        tag.className = "yuan-al-pi-tag";
                        const selected = audioFiles.includes(f);
                        if (selected) item.classList.add("selected");
                        tag.innerText = selected ? "✓ 已添加" : "+ 添加";
                        item.append(nameSpan, tag);
                        item.onclick = () => {
                            const idx = audioFiles.indexOf(f);
                            if (idx >= 0) {
                                audioFiles.splice(idx, 1);
                            } else {
                                if (audioFiles.length >= MAX_AUDIOS) return; // 上限 30，不可再添加
                                audioFiles.push(f);
                            }
                            updatePickerTitle();
                            renderGallery();
                            serializeFiles();
                            renderPickerList(search.value);
                        };
                        list.appendChild(item);
                    });
                }
                search.addEventListener("input", () => renderPickerList(search.value));
                search.addEventListener("keydown", (e) => e.stopPropagation());

                // 底部
                const footer = document.createElement("div");
                footer.className = "yuan-al-picker-footer";
                const doneBtn = document.createElement("button");
                doneBtn.className = "yuan-al-picker-done";
                doneBtn.innerText = "完成";
                doneBtn.onclick = () => closeFilePicker();
                footer.appendChild(doneBtn);

                panel.append(header, search, list, footer);
                overlay.appendChild(panel);
                overlay.addEventListener("mousedown", (e) => {
                    if (e.target === overlay) closeFilePicker();
                });
                document.body.appendChild(overlay);
                pickerOverlay = overlay;

                const escHandler = (e) => { if (e.key === "Escape") closeFilePicker(); };
                overlay._escHandler = escHandler;
                document.addEventListener("keydown", escHandler);

                renderPickerList("");
                search.focus();
            }

            function closeFilePicker() {
                if (!pickerOverlay) return;
                if (pickerOverlay._escHandler) {
                    document.removeEventListener("keydown", pickerOverlay._escHandler);
                }
                pickerOverlay.remove();
                pickerOverlay = null;
            }

            // --- 9. 布局管理 ---
            // 容器高度固定（方块滑轨横向滚动，不随文件数变化）。
            // 节点尺寸完全交给 LiteGraph / V3 处理，允许用户自由拉伸：
            // - 不重写 onResize / computeSize / setSize（反复强制最小高度会导致拉伸异常）；
            // - min_size 仅初始化时设置一次。
            const MIN_W = 240;

            function getContainerHeight() {
                return CONTAINER_PADDING * 2 + GALLERY_HEIGHT + 6 + BTN_ROW_HEIGHT;
            }

            galleryWidget.computeSize = function (width) {
                const nodeWidth = node.size?.[0] || width || 220;
                return [Math.max(10, nodeWidth - 30), getContainerHeight()];
            };

            // V3 前端：仅绑定拖拽上传事件，不强制 min-height（避免拉伸被锁死）
            let v3EventsAttached = false;
            function attachV3Events() {
                if (!v3NodeElement) checkIsV3(); // 确保 DOM 挂载后完成检测
                if (v3NodeElement && !v3EventsAttached) {
                    v3EventsAttached = true;
                    v3NodeElement.addEventListener("dragover", (e) => e.preventDefault());
                    v3NodeElement.addEventListener("drop", (e) => {
                        if (e.dataTransfer && e.dataTransfer.files) {
                            const files = Array.from(e.dataTransfer.files).filter(f => isAudioFileName(f.name));
                            if (files.length > 0) {
                                e.preventDefault();
                                e.stopPropagation();
                                handleFiles(files);
                            }
                        }
                    });
                }
            }

            /** 初始化时一次性保证最小尺寸；之后不干预用户拉伸 */
            function updateLayout() {
                attachV3Events();
                container.style.height = getContainerHeight() + "px";
                if (node.size && node.size[0] < MIN_W) {
                    node.size[0] = MIN_W;
                }
                if (app.graph) app.graph.setDirtyCanvas(true, true);
            }

            // --- 10. 初始化钩子 ---
            let uiInitialized = false;
            function ensureInitialized() {
                if (uiInitialized) return;
                uiInitialized = true;
                renderGallery();
                serializeFiles();
                updateLayout();
                if (app.graph) app.graph.setDirtyCanvas(true, true);
            }

            const origOnConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                const out = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                deserializeFiles();
                requestAnimationFrame(ensureInitialized);
                return out;
            };

            // 拖拽到节点任意位置上传
            const origOnDragDrop = node.onDragDrop;
            node.onDragDrop = function (e) {
                if (e.dataTransfer && e.dataTransfer.files) {
                    const files = Array.from(e.dataTransfer.files).filter(f => isAudioFileName(f.name));
                    if (files.length > 0) {
                        e.preventDefault();
                        handleFiles(files);
                        return true;
                    }
                }
                if (origOnDragDrop) return origOnDragDrop.apply(this, arguments);
                return false;
            };

            const origOnDragOver = node.onDragOver;
            node.onDragOver = function (e) {
                if (e.dataTransfer && e.dataTransfer.items) {
                    const hasAudio = Array.from(e.dataTransfer.items).some(
                        f => f.kind === "file" && isAudioFileName(f.name)
                    );
                    if (hasAudio) { e.preventDefault(); return true; }
                }
                if (origOnDragOver) return origOnDragOver.apply(this, arguments);
                return false;
            };

            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                // 仅停止本节点方块的播放（WeakSet 判断归属，避免误停其他节点）
                if (globalPlayer && myBlocks.has(globalPlayer.blockEl)) {
                    stopGlobalPlayback();
                }
                closeFilePicker();
                if (dataWidget && dataWidget._hideTimer) clearInterval(dataWidget._hideTimer);
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };

            // 初始恢复 / 新建
            if (dataWidget && dataWidget.value && dataWidget.value.trim()) {
                deserializeFiles();
            }

            const origOnAdded = node.onAdded;
            node.onAdded = function () {
                if (origOnAdded) origOnAdded.apply(this, arguments);
                requestAnimationFrame(ensureInitialized);
            };

            setTimeout(ensureInitialized, 50);
            // 兜底：DOM 完全挂载后再绑定 V3 拖拽事件
            setTimeout(() => { attachV3Events(); }, 300);
        }
    });
})();
