/**
 * Yuan Tool · 画布 前端
 *
 * 为 Yuan_Canvas 节点提供基于 fabric.js 的内嵌合成编辑器：
 *  - 接收 bg_image 与 images（batch）作为图层
 *  - 可视化放置、旋转、缩放、锁定、隐藏、调整层级
 *  - 合成结果上传后端，作为节点 IMAGE 输出
 *  - 支持位置信息持久化，切换工作流后可恢复
 */
import { fabric } from "./fabric.js";

const { app } = window.comfyAPI.app;

/** 从 window.comfyAPI 获取 api 实例（兼容不同 ComfyUI 版本） */
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


// 图层面板使用的 SVG 图标路径（参考 comfyui_pano_stickers）
const LAYER_ICONS = {
    bring_front: "<path stroke='none' d='M0 0h24v24H0z' fill='none' /><path d='M12 4l-8 4l8 4l8 -4l-8 -4' /><path d='M8 14l-4 2l8 4l8 -4l-4 -2' /><path d='M8 10l-4 2l8 4l8 -4l-4 -2' />",
    send_back: "<path stroke='none' d='M0 0h24v24H0z' fill='none' /><path d='M4 8l8 4l8 -4l-8 -4l-8 4' /><path d='M12 16l-4 -2l-4 2l8 4l8 -4l-4 -2l-4 2' /><path d='M8 10l-4 2l4 2m8 0l4 -2l-4 -2' />",
    eye: "<path stroke='none' d='M0 0h24v24H0z' fill='none' /><path d='M10 12a2 2 0 1 0 4 0a2 2 0 0 0 -4 0' /><path d='M21 12c-2.4 4 -5.4 6 -9 6c-3.6 0 -6.6 -2 -9 -6c2.4 -4 5.4 -6 9 -6c3.6 0 6.6 2 9 6' />",
    eye_dashed: "<path stroke='none' d='M0 0h24v24H0z' fill='none' /><path d='M10.585 10.587a2 2 0 0 0 2.829 2.828' /><path d='M16.681 16.673a8.717 8.717 0 0 1 -4.681 1.327c-3.6 0 -6.6 -2 -9 -6c1.272 -2.12 2.712 -3.678 4.32 -4.674m2.86 -1.146a9.055 9.055 0 0 1 1.82 -.18c3.6 0 6.6 2 9 6c-.666 1.11 -1.379 2.067 -2.138 2.87' /><path d='M3 3l18 18' />",
    lock_closed: "<path stroke='none' d='M0 0h24v24H0z' fill='none' /><path d='M5 13a2 2 0 0 1 2 -2h10a2 2 0 0 1 2 2v6a2 2 0 0 1 -2 2h-10a2 2 0 0 1 -2 -2v-6' /><path d='M11 16a1 1 0 1 0 2 0a1 1 0 0 0 -2 0' /><path d='M8 11v-4a4 4 0 1 1 8 0v4' />",
    lock_open: "<path stroke='none' d='M0 0h24v24H0z' fill='none' /><path d='M5 13a2 2 0 0 1 2 -2h10a2 2 0 0 1 2 2v6a2 2 0 0 1 -2 2h-10a2 2 0 0 1 -2 -2l0 -6' /><path d='M11 16a1 1 0 1 0 2 0a1 1 0 1 0 -2 0' /><path d='M8 11v-5a4 4 0 0 1 8 0' />",
};

function makeIconSvg(pathBody) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${pathBody}</svg>`;
}

/** 判断是否为 Yuan_Canvas 节点 */
function isYuan_Canvas(node) {
    return node.constructor.comfyClass == "Yuan_Canvas";
}

function getCompositorWidget(node, widgetName) {
    return node.widgets.find((w) => w.name === widgetName);
}

/**
 * 将 ComfyUI 图像 UI 条目（{filename, subfolder, type, storage}）转为 /view URL。
 * 参考自 ComfyUI-Yuan-Tool 的全景预览节点前端实现。
 */
function comfyImageEntryToUrl(entry) {
    if (!entry || typeof entry !== "object") return "";
    const filename = String(entry.filename || "").trim();
    if (!filename) return "";
    const params = new URLSearchParams();
    params.set("filename", filename);
    const viewType = String(
        entry.storage
        || (String(entry.type || "").trim().toLowerCase() === "comfy_image" ? "output" : entry.type)
        || "output"
    );
    params.set("type", viewType);
    if (entry.subfolder) params.set("subfolder", String(entry.subfolder));
    return api ? api.apiURL(`/view?${params.toString()}`) : `/view?${params.toString()}`;
}

/**
 * 从 ComfyUI 图像 UI 条目、字符串或数组中提取 /view URL。
 * 参考自 ComfyUI-Yuan-Tool 的全景预览节点前端实现。
 */
function imageSourceFromCandidate(candidate) {
    if (!candidate) return "";
    if (typeof candidate === "string") return String(candidate).trim();
    if (Array.isArray(candidate)) {
        if (candidate.length === 0) return "";
        if (candidate.length === 1) return imageSourceFromCandidate(candidate[0]);
        const filename = typeof candidate[0] === "string" ? String(candidate[0]).trim() : "";
        if (filename) {
            return comfyImageEntryToUrl({
                filename,
                subfolder: String(candidate[1] || "").trim(),
                type: String(candidate[2] || "output").trim() || "output",
            });
        }
        for (const e of candidate) {
            const s = imageSourceFromCandidate(e);
            if (s) return s;
        }
        return "";
    }
    if (typeof candidate?.src === "string" && candidate.src) return candidate.src;
    if (typeof candidate?.url === "string" && candidate.url) return candidate.url;
    return comfyImageEntryToUrl(candidate);
}

function lookupNodeOutputEntry(nodeId) {
    const store = app?.nodeOutputs;
    if (!store || nodeId == null) return null;
    const raw = String(nodeId);
    if (store instanceof Map) {
        return store.get(nodeId) || store.get(raw) || store.get(Number(raw)) || null;
    }
    return store[nodeId] || store[raw] || null;
}

/**
 * 从上游连接节点获取图像 URL 列表（支持 batch）。
 * 参考 ComfyUI-Yuan-Tool 的 getLinkedImageUrl 实现，不缓存图像信息。
 * 前端直接从上游节点获取图像，Yuan_画布节点只考虑图像的位置信息（transforms）。
 */
function getUpstreamImageUrls(node, inputName) {
    const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
    let linkId = null;
    const preferred = inputs.find((i) => String(i?.name || "") === String(inputName));
    if (preferred?.link != null) linkId = preferred.link;
    
    if (linkId == null) return [];

    const link = node?.graph?.links?.[linkId] || app?.graph?.links?.[linkId];
    if (!link) return [];
    const originId = Number(link.origin_id);
    if (!Number.isFinite(originId)) return [];

    const urls = [];

    // 1) 上游节点如果是加载类节点，优先从 widget 获取（避免切换工作流被缓存污染）
    const originNode = app?.graph?.getNodeById?.(originId);
    if (originNode) {
        const imageWidget = originNode?.widgets?.find?.((w) => String(w?.name || "").toLowerCase() === "image");
        const isLoader = originNode.type === "LoadImage" || originNode.comfyClass === "LoadImage" || String(originNode.type || "").toLowerCase().includes("load");

        if (isLoader && imageWidget && typeof imageWidget.value === "string") {
            const imageName = String(imageWidget.value).trim();
            if (imageName) {
                // 加载类节点的 widget 图像必定在 input 目录下（哪怕它的文件名叫 ComfyUI_temp_xxx）
                let type = "input";
                let file = imageName;
                let subfolder = "";
                let slashIdx = file.lastIndexOf("/");
                if (slashIdx > -1) {
                    subfolder = file.substring(0, slashIdx);
                    file = file.substring(slashIdx + 1);
                } else {
                    slashIdx = file.lastIndexOf("\\");
                    if (slashIdx > -1) {
                        subfolder = file.substring(0, slashIdx);
                        file = file.substring(slashIdx + 1);
                    }
                }
                const url = apiUrl(`/view?filename=${encodeURIComponent(file)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`);
                if (url) urls.push(url);
                return urls;
            }
        }
    }

    // 2) 上游节点的 imgs 数组（反映当前 UI 状态，可能包含 batch 多张图像）
    if (originNode) {
        const imgs = Array.isArray(originNode?.imgs) ? originNode.imgs : [];
        for (const c of imgs) {
            const s = imageSourceFromCandidate(c);
            if (s) urls.push(s);
        }
    }
    if (urls.length > 0) return urls;

    // 3) 上游节点的 nodeOutputs（上游为 OUTPUT_NODE 时含 batch 多张图像，但最易被缓存污染）
    const originOutputs = lookupNodeOutputEntry(originId);
    const candidateGroups = [
        originOutputs?.images,
        originOutputs?.ui?.images,
    ];
    for (const g of candidateGroups) {
        if (!Array.isArray(g)) continue;
        for (const c of g) {
            const s = imageSourceFromCandidate(c);
            if (s) urls.push(s);
        }
        if (urls.length > 0) return urls;
    }

    // 3) ComfyUI 内置缩略图（fallback）
    let nodeUrls = [];
    try {
        nodeUrls = typeof app?.getNodeImageUrls === "function" ? (app.getNodeImageUrls(originNode) || []) : [];
    } catch (_) { nodeUrls = []; }
    for (const c of nodeUrls) {
        const s = imageSourceFromCandidate(c);
        if (s) urls.push(s);
    }
    if (urls.length > 0) return urls;

    // 4) image widget（LoadImage 类节点）
    if (originNode) {
        const imageWidget = originNode?.widgets?.find?.((w) => String(w?.name || "").toLowerCase() === "image");
        const imageName = String(imageWidget?.value || "").trim();
        if (imageName) {
            const url = `/view?filename=${encodeURIComponent(imageName)}&type=input&subfolder=`;
            urls.push(api ? api.apiURL(url) : url);
        }
    }

    return urls;
}

/**
 * 计算图像内容签名（sig），用于作为图像唯一标识。
 * 参考后端 _image_signature 实现：sig = shape_均值。
 * 前端从已加载的 fabric.Image 计算 sig，作为 transforms 的 key，
 * 这样调换 batch 顺序时 transforms 仍能正确对应同一张图。
 */
function computeFabricImageSig(fabricImg) {
    try {
        const elem = fabricImg?.getElement?.() || fabricImg?._element;
        const w = elem?.naturalWidth || fabricImg?.width || 0;
        const h = elem?.naturalHeight || fabricImg?.height || 0;
        if (!w || !h) return null;
        // 使用小尺寸 canvas 计算平均像素值，避免大图性能问题
        const canvas = document.createElement("canvas");
        const scale = Math.min(1, 32 / Math.max(w, h));
        canvas.width = Math.max(1, Math.round(w * scale));
        canvas.height = Math.max(1, Math.round(h * scale));
        const ctx = canvas.getContext("2d");
        ctx.drawImage(elem, 0, 0, canvas.width, canvas.height);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        let sum = 0;
        let count = 0;
        for (let i = 0; i < data.length; i += 4) {
            sum += data[i] + data[i + 1] + data[i + 2];
            count += 3;
        }
        const mean = count > 0 ? (sum / count / 255) : 0;
        return `(1,${h},${w},3)_${mean.toFixed(6)}`;
    } catch (e) {
        // 跨域或读取失败时，使用尺寸作为 fallback 签名
        const w = fabricImg?.width || 0;
        const h = fabricImg?.height || 0;
        if (!w || !h) return null;
        return `(1,${h},${w},3)_0`;
    }
}

function handleTogglePreciseSelection(e, currentNode) {
    const optionValue = e.data.value;
    currentNode.compositorInstance.preciseSelection = optionValue;
    const c = currentNode.compositorInstance.fcanvas;
    c.getObjects().map(function (i) {
        return i.set('perPixelTargetFind', optionValue);
    });
}

function handleResetOldTransform(e, currentNode) {
    const optionValue = e.data.value;
    const instance = currentNode.compositorInstance;

    for (const sig in instance.inputImages) {
        instance.resetOldTransform(sig);
    }
}

function centerSelected(e, currentNode) {
    const optionValue = e.data.value;
    const instance = currentNode.compositorInstance;

    const c = instance.fcanvas;
    // 获取选中对象并居中
    instance.needsUpload = true;
    c.getActiveObjects().forEach((o)=>o.center());
    c.renderAll();
    instance.uploadIfNeeded(instance);
}

/**
 * 注册扩展，可介入生命周期方法。
 * 以下是文档中的调用顺序：
 *
 * api 事件
 * 0: "status"
 * 1: "graphChanged"
 * 2: "promptQueued"
 * 3: "graphCleared"
 * 4: "executed"
 * 5: "execution_start"
 * 6: "execution_cached"
 * 7: "executing"
 * 8: "reconnecting"
 * 9: "reconnected"
 * 10: "manager-terminal-feedback"
 * 11: "cm-api-try-install-customnode"
 * 12: "progress"
 * 13: "execution_error"
 * 14: "b_preview"
 * 15: "crystools.monitor"
 * 16: "configure"
 * 17: "compositor.images"
 *
 * -- 网页加载 --
 * invokeExtensionsAsync init
 * invokeExtensionsAsync addCustomNodeDefs
 * invokeExtensionsAsync getCustomWidgets
 * invokeExtensionsAsync beforeRegisterNodeDef    [多次触发]
 * invokeExtensionsAsync registerCustomNodes
 * invokeExtensionsAsync beforeConfigureGraph
 * invokeExtensionsAsync nodeCreated
 * invokeExtensions      loadedGraphNode
 * invokeExtensionsAsync afterConfigureGraph
 * invokeExtensionsAsync setup
 *
 * -- 加载工作流 --
 * invokeExtensionsAsync beforeConfigureGraph
 * invokeExtensionsAsync beforeRegisterNodeDef   [零次、一次或多次]
 * invokeExtensionsAsync nodeCreated             [多次触发]
 * invokeExtensions      loadedGraphNode         [多次触发]
 * invokeExtensionsAsync afterConfigureGraph
 *
 * -- 添加新节点 --
 * invokeExtensionsAsync nodeCreated
 *
 * 关于 node 是什么等更多信息：
 * https://docs.comfy.org/essentials/javascript_objects_and_hijacking
 */
app.registerExtension({
    name: "Comfy.Yuan_Canvas",

    async getCustomWidgets(app) {
        // 无自定义 widget
    },
    /**
     * 在启动流程末尾调用。
     * 适合添加事件监听（Comfy 事件或 DOM 事件）或全局菜单项。
     * 此时从消息中可拿到 nodeId（若传递了），但没有节点上下文，需自行查找。
     *
     * 捕获 UI 事件
     * 正如预期 - 在 DOM 中找到 UI 元素并 addEventListener。
     * setup() 是做这事的好地方，因为页面已完全加载。
     * 例如，检测对 'Queue' 按钮的点击：
     * ```
     *      function queue_button_pressed() { console.log("Queue button was pressed!") }
     *      document.getElementById("queue-button").addEventListener("click", queue_button_pressed);
     * ```
     */
    async setup(app) {
        Editor.addCompositorSettings();

        /**
         * @deprecated
         * 节点在 python 中处理后，最终从事件中得知连接的图像。
         * 它们以 base64 编码传递，或未连接时为 null，以及节点的唯一名称 id。
         * @param event
         */
        function imageMessageHandler(event) {
        }

        function hook(nodeId) {
            return app.graph.getNodeById(nodeId);
        }


        /** 任意消息示例
         * PromptServer.instance.send_sync("my.custom.message", {"node": node_id, "other_things": etc})
         * 在 api.ts 中搜索 "case 'executing'": 可找到所有发出的事件，或 "new CustomEvent('executing'"
         * 内置事件示例，这应是节点即将开始（在后端）处理时
         */
        function executingMessageHandler(event) {
            //console.log("executingMessageHandler", event, arguments);
            const current = app.graph.getNodeById(event.detail);

            // 这里可能太晚了，因为它已经在后台运行
            // if (current && current.type == "Yuan_Canvas") {
            //     const instance = current.compositorInstance;
            //     if (instance.captureOnQueue.value) {
            //         //instance.capture();
            //         instance.grabUploadAndSetOutput.bind(instance);
            //     }
            // }

            // }
        }

        /**
         * 处理 .py 执行期间发送的 progress 消息
         */
        function progressHandler() {
            // 需按 node id 过滤
            // console.log(arguments);
        }

        /** 当节点"返回"一个 ui 元素时，通常在处理末尾 */
        function executedMessageHandler(event, a, b) {
            //console.log("executedMessageHandler", event, a, b);

            // Litegraph 文档
            // https://github.com/jagenjo/litegraph.js/blob/master/guides/README.md
            // 也要关注连接到此 config 的东西...当心 GUI...

            const e = event.detail.output;
            const nodeId = event.detail.node;
            const node = Editor.hook(nodeId);
            if (!node || node.type != "Yuan_Canvas") {
                // console.log(node.type);
                return;
            }
            const instance = node.compositorInstance;
            // console.log("hasResult,awaitedResult", e.hasResult[0], e.awaited[0]);

            // 仅在 w/h/p 实际变化时才调整尺寸，否则点击"继续"
            // 每次都会不必要地重置节点尺寸
            const newW = e.width[0];
            const newH = e.height[0];
            const newP = e.padding[0];
            if (node.compositorInstance.w.value !== newW) {
                node.compositorInstance.w.value = newW;
                node.compositorInstance.onWidthChange(newW);
            }
            if (node.compositorInstance.h.value !== newH) {
                node.compositorInstance.h.value = newH;
                node.compositorInstance.onHeightChange(newH);
            }
            if (node.compositorInstance.p.value !== newP) {
                node.compositorInstance.p.value = newP;
                node.compositorInstance.onPaddingChange(newP);
            }
            // 保持节点自身的 config widget 与后端同步
            const wWidget = getCompositorWidget(node, "width");
            const hWidget = getCompositorWidget(node, "height");
            const pWidget = getCompositorWidget(node, "padding");
            if (wWidget) wWidget.value = newW;
            if (hWidget) hWidget.value = newH;
            if (pWidget) pWidget.value = newP;
            // node.compositorInstance.onCaptureOnQueueChange(e.captureOnQueue[0]);

            instance.configChanged = e.configChanged[0];

            // 画布上是否已有图像
            const hasImagesOnCanvas = Object.keys(instance.inputImages).length > 0;

            // configChanged 为 false 且画布上已有图像时（用户点击 continue 后图像内容未变），
            // 保持画布上的图像原位，不重新加载，立即可拖动。
            // configChanged 为 false 但画布上没有图像时（如切换工作流再返回），
            // 需要从上游重新加载图像。
            if (!instance.configChanged && hasImagesOnCanvas) {
                return;
            }

            // 在清空前，先把当前 inputImages 的 transforms 持久化到 fabricDataWidget。
            // 这样用户调整过的位置不会因 clearInputImages 而丢失，
            // 后续按 sig 恢复时能正确还原到最近一次的位置。
            try {
                const currentSerialized = Editor.serializeStuff(node);
                const currentParsed = JSON.parse(currentSerialized);
                const hasValid = (currentParsed.transforms && Object.values(currentParsed.transforms).some((t) => t != null));
                if (hasValid) {
                    node.fabricDataWidget.value = currentSerialized;
                }
            } catch (e) { /* 忽略 */ }

            const restore = Editor.deserializeStuff(node.fabricDataWidget.value);
            const shouldRestore = restore ?? false;

            // config 变化时清空旧的输入图层（数量或内容可能变化）；
            // config 未变化但画布为空时（切换工作流再返回）不清空（本来就是空的）
            if (instance.configChanged) {
                instance.clearInputImages();
            }

            // 从后端 UI 输出获取 bg_image，fallback 到上游节点获取
            let bgEntries = Array.isArray(e.bg_entries) ? e.bg_entries : [];
            bgEntries = bgEntries.filter((entry) => entry && entry.sig && imageSourceFromCandidate(entry));
            
            if (bgEntries.length > 0) {
                const bgUrl = imageSourceFromCandidate(bgEntries[0]);
                if (bgUrl) {
                    fabric.Image.fromURL(bgUrl, function (oImg) {
                        node.compositorInstance.setBgImage(oImg);
                    }, { crossOrigin: "anonymous" });
                }
            } else {
                const bgUrls = getUpstreamImageUrls(node, "bg_image");
                const bgUrl = bgUrls.length > 0 ? bgUrls[0] : null;
                if (bgUrl) {
                    fabric.Image.fromURL(bgUrl, function (oImg) {
                        node.compositorInstance.setBgImage(oImg);
                    }, { crossOrigin: "anonymous" });
                }
            }

            // 参考 ERP_image 端口：优先从后端 UI 输出（images_entries）获取图像（含 sig），
            // fallback 从上游节点获取（前端计算 sig）。
            // 后端使用 PreviewImage 落盘 images batch（在后台线程完成，不阻塞前端 UI），
            // 确保 batch 多张图像都能获取到。前端异步加载 URL，图像加载后立即可拖动。
            let imageEntries = Array.isArray(e.images_entries) ? e.images_entries : [];
            imageEntries = imageEntries.filter((entry) => entry && entry.sig && imageSourceFromCandidate(entry));

            if (imageEntries.length > 0) {
                // 从后端 UI 输出获取（含 sig）
                imageEntries.forEach((entry, index) => {
                    const url = imageSourceFromCandidate(entry);
                    if (!url) return;
                    const sig = entry.sig;
                    fabric.Image.fromURL(url, function (oImg) {
                        if (!oImg || !oImg.width) return;
                        node.compositorInstance.addOrReplaceImage(oImg, sig, nodeId, restore, shouldRestore, index);
                    }, { crossOrigin: "anonymous" });
                });
            } else {
                // fallback：后端未执行时从上游获取（前端计算 sig）
                const upstreamImageUrls = getUpstreamImageUrls(node, "images");
                upstreamImageUrls.forEach((url, index) => {
                    if (!url) return;
                    fabric.Image.fromURL(url, function (oImg) {
                        if (!oImg || !oImg.width) return;
                        const sig = computeFabricImageSig(oImg) || `upstream_${index}`;
                        node.compositorInstance.addOrReplaceImage(oImg, sig, nodeId, restore, shouldRestore, index);
                    }, { crossOrigin: "anonymous" });
                });
            }

            // 不在此处调用 uploadIfNeeded：
            // 1. 此时 fabric.Image.fromURL 异步加载尚未完成，canvas 为空，
            //    toDataURL 会导出空图像，且同步阻塞 UI 线程导致图像加载后无法立即拖动。
            // 2. composition 上传应在用户点击 continue 时进行（continue 方法中先上传再执行）。

        }

        /** 重要消息考量  https://docs.comfy.org/essentials/comms_messages */

        function configureHandler() {
            //console.log("configurehanlder", arguments);
        }

        function executionStartHandler() {
            //console.log("executionStartHandler", arguments);
        }

        function executionCachedHandler() {
            //console.log("executionCachedHandler", arguments);
        }

        function graphChangedHandler() {
            console.log("graphChangedHandler", arguments);
        }

        function changeWorkflowHandler() {
            //console.log("changeWorkflowHandler", arguments);
        }


        // change_workflow
        // node 和 widget 的全局 on_change

        api.addEventListener("compositor_init", executedMessageHandler);
        api.addEventListener("graphChanged", graphChangedHandler);
        api.addEventListener("change_workflow", changeWorkflowHandler);
        api.addEventListener("execution_start", executionStartHandler);
        api.addEventListener("execution_cached", executionCachedHandler);
        api.addEventListener("executing", executingMessageHandler);
        // 注意：不监听 "executed" 事件。
        // 后端 composite 方法已通过 send_sync("compositor_init", ...) 推送 UI 数据，
        // executedMessageHandler 会在 compositor_init 时执行一次。
        // 若再监听 executed，同一节点每次执行会触发两次 executedMessageHandler，
        // 第二次的 clearInputImages 会清空第一次刚恢复的图像并重新异步加载，
        // 复杂的异步时序会导致 transforms 丢失（位置信息清零）。
        /**
         * 测试 .py 执行期间收到的 "progress"
         */
        api.addEventListener("progress", progressHandler);
        /** ? */
        api.addEventListener("configure", configureHandler);


    },
    /**
     * 在 Comfy 网页加载（或重载）时调用。
     * 调用发生在 graph 对象创建之后，但任何节点注册或创建之前。
     * 可用于通过劫持 app 或 graph（LiteGraph 对象）的方法来修改 Comfy 核心行为。
     * 这在 Comfy Objects 中进一步讨论。
     */
    async init(args) {
        // console.log("init", args)
    },
    /**
     * 对每个节点类型（AddNode 菜单中可用的节点列表）调用一次，
     * 用于修改节点行为。
     *
     * async beforeRegisterNodeDef(nodeType, nodeData, app)
     * 传入的 nodeType 参数作为该类型所有待创建节点的模板。
     * 对 "nodeType.prototype" 的修改会应用于该类型的所有节点。
     * nodeData 封装了 Python 代码中定义的节点信息，
     * 如分类、输入、输出。
     * app 是主 Comfy app 对象的引用（反正你已经 import 了！）
     ```
     async beforeRegisterNodeDef(nodeType, nodeData, app) {

        if (nodeType.comfyClass == 'Compositor') {
          //  console.log("beforeRegisterNodeDef", nodeType, nodeData, app);

            const orig_nodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function () {
                // console.log("onNodeCreated", this);
                orig_nodeCreated?.apply(this, arguments)
                this.setSize([this.stuff.v.getWidth() + 100, this.stuff.v.getHeight() + 556])
            }

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                // console.log("onExecuted", this, message);
                const r = onExecuted?.apply?.(this, arguments)
                return r;
            }
        }
    },
     ```
     */
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // chainCallback(nodeType.prototype, "onNodeCreate, createdCallback(){
        // 这里就是节点本身，可添加 widget、隐藏它们、加按钮、给 widget 加属性
        // 这里他加了整个编辑器，或在节点有/无某些属性（如 points）时重置
        // onConfigure 和 onExecuted 也在这里
        //}
        // console.log("beforeRegisterNodeDef",nodeType,nodeData);
        // KJ 在这里初始化节点
        // 带一个 onNodeCreated 回调
        // nodeType.prototype.capture = ()=>{
        //     console.log(this,arguments);
        // }
    },
    /** loadedGraphNode，在 nodeCreated 之后
     *  ```
     if(node.type == "Compositor" && console.log("loadedGraphNode", node, app, node.stuff)){
         const ns = node.stuff;

         ns.safeArea.setHeight(ns.h.value);
         ns.safeArea.setWidth(ns.w.value);
         ns.safeArea.setLeft(ns.p.value);
         ns.safeArea.setTop(ns.p.value);

         ns.compositionBorder.setHeight(ns.h.value + ns.stuff.COMPOSITION_BORDER_SIZE*2);
         ns.compositionBorder.setWidth(ns.w.value  + ns.stuff.COMPOSITION_BORDER_SIZE*2);
         ns.compositionBorder.setLeft(ns.p.value - ns.stuff.COMPOSITION_BORDER_SIZE);
         ns.compositionBorder.setTop(ns.p.value - ns.stuff.COMPOSITION_BORDER_SIZE*2);
         ns.compositionBorder.set("strokeWidth", ns.stuff.COMPOSITION_BORDER_SIZE);
         ns.compositionBorder.set("stroke", ns.stuff.COMPOSITION_BORDER_COLOR);
         ns.compositionBorder.bringToFront()

         canvas.bringToFront(ns.compositionBorder);

         //console.log(v.getWidth(), v.getHeight(), value);
         ns.canvas.setHeight(ns.safeArea.getHeight() + (ns.p.value * 2));
         ns.canvas.setWidth(ns.safeArea.getWidth() + (ns.p.value * 2));
         ns.canvas.renderAll();
         ns.node.setSize(calculateNodeSize(v));

         ns.capture();
         }
     ```
     */
    async loadedGraphNode(node, app) {
        if (!isYuan_Canvas(node)) return;
        const instance = node.compositorInstance;
        if (!instance) return;

        // widget 值在 nodeCreated 时尚未从 workflow JSON 恢复，
        // initFabric 用默认 512x512 初始化了画布；此处值已恢复，需重新应用。
        const widthWidget = getCompositorWidget(node, "width");
        const heightWidget = getCompositorWidget(node, "height");
        const paddingWidget = getCompositorWidget(node, "padding");

        if (widthWidget) {
            instance.w.value = widthWidget.value;
            instance.onWidthChange(widthWidget.value);
        }
        if (heightWidget) {
            instance.h.value = heightWidget.value;
            instance.onHeightChange(heightWidget.value);
        }
        if (paddingWidget) {
            instance.p.value = paddingWidget.value;
            instance.onPaddingChange(paddingWidget.value);
        }

        // 参考 ERP_image 端口：切换工作流后直接从上游节点获取图像（不自动执行工作流）。
        // 上游若已执行过，nodeOutputs 中有图像；若未执行，等用户手动执行后由
        // executedMessageHandler 从上游重新获取。前端计算 sig 作为 transforms 的 key。
        const restore = Editor.deserializeStuff(node.fabricDataWidget.value);
        const shouldRestore = restore ?? false;

        // bg_image：单张图像，直接从上游获取
        const bgUrls = getUpstreamImageUrls(node, "bg_image");
        const bgUrl = bgUrls.length > 0 ? bgUrls[0] : null;
        if (bgUrl) {
            fabric.Image.fromURL(bgUrl, function (oImg) {
                if (oImg && oImg.width) {
                    instance.setBgImage(oImg);
                }
            }, { crossOrigin: "anonymous" });
        }

        // images：一张或多张图像（batch），直接从上游获取
        // 前端计算图像内容签名（sig）作为唯一标识，按 sig 恢复 transforms
        const upstreamImageUrls = getUpstreamImageUrls(node, "images");
        upstreamImageUrls.forEach((url, index) => {
            if (!url) return;
            fabric.Image.fromURL(url, function (oImg) {
                if (!oImg || !oImg.width) return;
                const sig = computeFabricImageSig(oImg) || `upstream_${index}`;
                instance.addOrReplaceImage(oImg, sig, node.id, restore, shouldRestore, index);
            }, { crossOrigin: "anonymous" });
        });

        // 重新盖上 firstRun 时间戳，让 IS_CHANGED 在下次执行时返回新值，
        // 用户手动执行时节点重新执行，从上游重新拉取图像并按 fabricDataWidget 中的 transforms 恢复
        if (node.fabricDataWidget) {
            const data = restore || {};
            data.firstRun = Date.now();
            node.fabricDataWidget.value = JSON.stringify(data);
        }
    },
    async afterConfigureGraph(args) {
        // 参考 ERP_image 端口：不自动执行工作流获取图像。
        // 图像由 loadedGraphNode 直接从上游获取，无需触发执行。
        // 用户需要手动执行工作流（点击 Queue Prompt 或 continue 按钮）来获取图像。
    },
    /**
     * 当节点的某个具体实例被创建时调用
     * （在 nodeType 末尾的 ComfyNode() 构造函数中）。
     * 在此钩子中可修改节点的单个实例。
     * 注：before register node def 更适合原型修改（？）
     * node 参考
     * https://docs.comfy.org/essentials/javascript_objects_and_hijacking
     *
     * 与 beforeRegisterNodeDef 原型中的 nodeCreated 事件不同（那是原型节点类实例）
     */
    async nodeCreated(node) {
        if (!isYuan_Canvas(node)) return;

        /** 我们的输出合成图像 */
        //node.compositionChangedWidget = getCompositorWidget(node, "compositionChanged");
        node.imageNameWidget = getCompositorWidget(node, "imageName");
        const originalCallback = node.imageNameWidget.callback;
        node.imageNameWidget.callback = () => {
            //debugger;
            //console.log("callback of imageNameWidget with ", arguments);
            originalCallback(arguments);
        }
        node.imageNameWidget.computeSize = () => [0, 0];
        // imageName.computeSize = () => [0, 0];
        hideWidgetForGood(node, node.imageNameWidget);

        node.fabricDataWidget = getCompositorWidget(node, "fabricData");
        node.fabricDataWidget.computeSize = () => [0, 0];
        hideWidgetForGood(node, node.fabricDataWidget);


        // 确保重载时 widget 会被重新执行
        const firstRun = Editor.deserializeStuff(node.fabricDataWidget.value);
        firstRun["firstRun"] = Date.now();
        node.fabricDataWidget.value = JSON.stringify(firstRun);

        const containerDiv = Editor.createCompositorContainerDiv(node)

        const c = document.createElement("canvas");
        c.id= "c_" + node.id;
        containerDiv.appendChild(c);



        node.editorWidget = node.addDOMWidget("test", "test", containerDiv, {
            //serialize: false,
            hideOnZoom: false,
        });
        const fc = new fabric.Canvas(c,{
            backgroundColor: 'transparent',
            selectionColor: 'transparent',
            selectionLineWidth: 1,
            preserveObjectStacking: true,
            altSelectionKey: "ctrlKey",
            altActionKey: "ctrlKey",
            centeredKey: "altKey",
        });


        /** 初始化合成器 GUI widget */
        const compositorInstance = new Editor(node, containerDiv);
        compositorInstance.initFabric(fc);

        // 绑定节点自身的 config widget，使编辑时画布实时更新
        const widthWidget = getCompositorWidget(node, "width");
        const heightWidget = getCompositorWidget(node, "height");
        const paddingWidget = getCompositorWidget(node, "padding");
        if (widthWidget) {
            compositorInstance.w.value = widthWidget.value;
            widthWidget.callback = () => {
                compositorInstance.w.value = widthWidget.value;
                compositorInstance.onWidthChange(widthWidget.value);
            };
        }
        if (heightWidget) {
            compositorInstance.h.value = heightWidget.value;
            heightWidget.callback = () => {
                compositorInstance.h.value = heightWidget.value;
                compositorInstance.onHeightChange(heightWidget.value);
            };
        }
        if (paddingWidget) {
            compositorInstance.p.value = paddingWidget.value;
            paddingWidget.callback = () => {
                compositorInstance.p.value = paddingWidget.value;
                compositorInstance.onPaddingChange(paddingWidget.value);
            };
        }
        // 立即将当前 widget 值应用到画布
        compositorInstance.onWidthChange(compositorInstance.w.value);
        compositorInstance.onHeightChange(compositorInstance.h.value);
        compositorInstance.onPaddingChange(compositorInstance.p.value);



        /**
         * grabUploadAndSetOutput 回调不能是 async，所以在 upload image 中传入 widget
         * addWidget(type, name, value, callback, options)
         */
        //node.capture = node.addWidget("button", "capture", "capture", compositorInstance.grabUploadAndSetOutput.bind(compositorInstance));
        node.continue = node.addWidget("button", "continue", "continue", compositorInstance.continue.bind(compositorInstance));

        node.onMouseOut = function (e, pos, canvas) {
            // console.log("mouseout")
            const original_onMouseDown = node.onMouseOut;
            return original_onMouseDown?.apply(this, arguments);
        }
        //node.compositorInstance = compositorInstance;
    },
});


//来自 melmass
function hideWidgetForGood(node, widget, suffix = '') {
    widget.origType = widget.type
    widget.origComputeSize = widget.computeSize
    widget.origSerializeValue = widget.serializeValue
    widget.computeSize = () => [0, -4] // -4 是因为 litegraph 会在 widget 间自动加间隙
    widget.type = "converted-widget" + suffix
    // widget.serializeValue = () => {
    //     // 若无输入连接，则阻止序列化该 widget
    //     const w = node.inputs?.find((i) => i.widget?.name === widget.name);
    //     if (w?.link == null) {
    //         return undefined;
    //     }
    //     return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
    // };

    // 隐藏关联的 widget，例如 seed+seedControl
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            hideWidgetForGood(node, w, ':' + widget.name)
        }
    }
}

/** 将在节点创建时通过 addDOMWidget 添加到节点 */
class Editor {
    id;
    canvasEl;
    /** fabric canvas */
    fcanvas;
    /** 传给 addDomWidget 的 dom 元素 */
    containerDiv;
    /** 上一张图像，可能只需要其哈希 */
    cblob;
    /** 当前 blob 中图像的哈希 */
    c1;
    /** 待检查的新 blob 的哈希 */
    c2;
    /** 若 c1 === c2 */
    sameHash;
    /** fcanvas 中选中的对象，用于操作事件 */
    selected;

    /** 设置 */
    CANVAS_BORDER_COLOR = "#00b300b0";
    COMPOSITION_BORDER_COLOR = "#00b300b0";
    COMPOSITION_BORDER_SIZE = 2;
    COMPOSITION_BACKGROUND_COLOR = "rgba(0,0,0,0.2)";

    compositionArea;
    compositionBorder;
    /** 背景图像图层（始终在最底层，compositionArea 之上） */
    bgImage;
    /** 浮动图层工具栏 DOM 元素（选中输入图像时显示） */
    layerToolbar;
    /** 当前选中的输入图像（工具栏操作的目标） */
    selectedLayerImage;
    preciseSelection = false;

    /** （widget）引用 / 配置参数 */
    p;
    w;
    h;

    /** capture on queue widget 值的引用 */
        //captureOnQueue;

    /** 以图像内容签名(sig)为 key 存储输入图像图层，调换 batch 顺序时 transforms 仍能正确对应 */
    inputImages = {};
    fabricDataWidget;
    needsUpload = false;

    configurationNode;

    static hook(nodeId) {
        return app.graph.getNodeById(nodeId);
    }

    static deserializeStuff(value) {
        try {
            return JSON.parse(value)


        } catch (e) {
            console.log("deserializeStuff", e, value);
            return undefined;
        }

    }

    /**
     * 序列化图像位置信息：每张图三个信息 = sig(唯一ID) + 起始点(x1,y1) + 对角点(x2,y2)。
     * 同时持久化 locked/hidden 状态，切换工作流后按 sig 恢复。
     * 不缓存图像像素，只保存位置和状态信息。
     */
    static serializeStuff(node) {
        const instance = node.compositorInstance;
        const result = {
            width: instance.w.value,
            height: instance.h.value,
            padding: instance.p.value,
            transforms: {},
            locked: {},
            hidden: {},
        };
        // transforms 以 sig 为 key，值为 {x1,y1,x2,y2} 两点坐标
        for (const sig in instance.inputImages) {
            try {
                const img = instance.inputImages[sig];
                result.transforms[sig] = instance.getOldTransform(sig);
                result.locked[sig] = !!img.locked;
                result.hidden[sig] = img.opacity === 0;
            } catch (e) {
                result.transforms[sig] = undefined;
                result.locked[sig] = false;
                result.hidden[sig] = false;
            }
        }
        return JSON.stringify(result);
    }


    static addCanvasBorderColorSetting() {
        app.extensionManager.setting.set({
            id: "Yuan_Canvas.Canvas.BORDER_COLOR",
            name: "Border Color",
            tooltip: "give an hex code with alpha, e.g.: #00b300b0, it's the area controlled by 'padding' size outside the  output that will not be exported but used for manipulation",
            type: "text",
            defaultValue: "#00b300b0",
            onChange: (newVal, oldVal) => {
                // console.log(newVal, this);
            },
        });
    }

    static addCompositionBorderColorSetting() {
        app.extensionManager.setting.set({
            id: "Yuan_Canvas.Composition.BORDER_COLOR",
            name: "Border Color (not rendered)",
            type: "text",
            tooltip: "give hex code with alpha eg.: #00b300b0, this will help identifying what is withing the output",
            defaultValue: "#00b300b0",
            onChange: (newVal, oldVal) => {
                // console.log(newVal, this);
            },
        });
    }

    static addCompositionBorderSizeSetting() {
        app.extensionManager.setting.set({
            id: "Yuan_Canvas.Composition.BORDER_SIZE",
            name: "Border Size",
            type: "slider",
            attrs: {
                min: 0,
                max: 2,
                step: 1
            },
            defaultValue: 2,
            tooltip: "Border size, 0 for invisible, overlayed and unselectable, not part of the node ouptut",

            onChange: (newVal, oldVal) => {
                // console.log(newVal, this);
            },
        });
    }

    static addCompositionBackgroundColorSetting() {
        app.extensionManager.setting.set({
            id: "Yuan_Canvas.Composition.BACKGROUND_COLOR",
            name: "Background Color - Output",
            type: "text",
            tooltip: "give hex code with alpha eg.: #00b300b0, this will help identifying what is withing the output",
            defaultValue: "rgba(0,0,0,0.2)",
            onChange: (newVal, oldVal) => {
                // console.log(newVal, this);
            },
        });
    }

    static addCompositorSettings() {
        Editor.addCanvasBorderColorSetting();
        Editor.addCompositionBorderColorSetting();
        Editor.addCompositionBorderSizeSetting();
        Editor.addCompositionBackgroundColorSetting();
    }

    getCompositorSettings() {
        // this.CANVAS_BORDER_COLOR = app.extensionManager.setting.get("Yuan_Canvas.Canvas.BORDER_COLOR");
        // this.COMPOSITION_BORDER_COLOR = app.extensionManager.setting.get("Yuan_Canvas.Composition.BORDER_COLOR");
        // this.COMPOSITION_BORDER_SIZE = app.extensionManager.setting.get("Yuan_Canvas.Composition.BORDER_SIZE");
        // this.COMPOSITION_BACKGROUND_COLOR = app.extensionManager.setting.get("Yuan_Canvas.Composition.BACKGROUND_COLOR");
    }

    static getRandomCompositorUniqueId() {
        const randomUniqueIds = new Uint32Array(10);
        const compositorId = 'c_' + self.crypto.getRandomValues(randomUniqueIds)[0] + '_' + self.crypto.getRandomValues(randomUniqueIds)[1];
        return compositorId;
    }

    static createCompositorContainerDiv() {
        const container = document.createElement("div");
        container.style.backgroundColor = "rgba(15,0,25,0.25)";
        container.style.textAlign = "center";
        // 需要设置 position: relative，浮动图层工具栏才能相对此容器定位
        container.style.position = "relative";
        return container;
    }

    // static CreateIframe() {
    //     const container = document.createElement("iframe");
    //     container.style.backgroundColor = "rgba(15,0,25,0.25)";
    //     container.style.zIndex = 3000;
    //     container.style.textAlign = "center";
    //     container.src = "https://comfy.org/";
    //     return container;
    // }

    static createCanvasElement() {
        const canvas = document.createElement("canvas");
        canvas.id = Editor.getRandomCompositorUniqueId();
        return canvas;
    }

    onHeightChange(value) {
        // console.log("h callback");
        this.fcanvas.setHeight(value + (this.p.value * 2));
        this.compositionArea.setHeight(value);
        this.compositionBorder.setHeight(value + this.COMPOSITION_BORDER_SIZE * 2);
        this.fitBgImage();
        this.enforceLayerOrder();
        this.syncContainerSize();
        this.node.setSize(this.calculateNodeSize())
        this.fcanvas.renderAll();
    }

    onWidthChange(value) {
        // console.log("h callback");
        this.fcanvas.setWidth(value + (this.p.value * 2));
        this.compositionArea.setWidth(value);
        this.compositionBorder.setWidth(value + this.COMPOSITION_BORDER_SIZE * 2);
        this.fitBgImage();
        this.enforceLayerOrder();
        this.syncContainerSize();
        this.node.setSize(this.calculateNodeSize());
        this.fcanvas.renderAll();
    }

    onPaddingChange(padding) {

        // console.log("p callback")
        // value 即 padding 值
        this.compositionArea.setHeight(this.h.value);
        this.compositionArea.setWidth(this.w.value);
        this.compositionArea.setLeft(padding);
        this.compositionArea.setTop(padding);

        this.compositionBorder.setHeight(this.h.value + this.COMPOSITION_BORDER_SIZE * 2);
        this.compositionBorder.setWidth(this.w.value + this.COMPOSITION_BORDER_SIZE * 2);
        this.compositionBorder.setLeft(padding - this.COMPOSITION_BORDER_SIZE);
        this.compositionBorder.setTop(padding - this.COMPOSITION_BORDER_SIZE);

        this.fcanvas.setHeight(this.compositionArea.getHeight() + (padding * 2));
        this.fcanvas.setWidth(this.compositionArea.getWidth() + (padding * 2));
        this.fitBgImage();
        this.enforceLayerOrder();
        this.syncContainerSize();
        this.fcanvas.renderAll();
        this.node.setSize(this.calculateNodeSize())

    }

    /**
     * 将 containerDiv 的 CSS 尺寸同步为 fabric canvas 尺寸，
     * 使 DOM widget 布局稳定（防止重新执行时节点尺寸闪烁）。
     */
    syncContainerSize() {
        if (!this.fcanvas || !this.containerDiv) return;
        const w = this.fcanvas.getWidth();
        const h = this.fcanvas.getHeight();
        this.containerDiv.style.width = w + "px";
        this.containerDiv.style.height = h + "px";
    }

    getOldTransform(sig) {
        const ref = this.inputImages[sig];
        // 用起始点(x1,y1)和对角点(x2,y2)确定图像位置和大小
        // 三个信息确定一张图：sig(唯一ID) + (x1,y1) + (x2,y2)
        const bounds = ref.getBoundingRect();
        return {
            x1: bounds.left,
            y1: bounds.top,
            x2: bounds.left + bounds.width,
            y2: bounds.top + bounds.height,
        };
    }

    /**
     * 用两点坐标 {x1,y1,x2,y2} 恢复图像位置和大小。
     * (x1,y1) 为左上角，(x2,y2) 为右下角，由此确定位置和缩放。
     */
    applyTransformFromPoints(theImage, points) {
        if (!points || points.x1 == null) return;
        let imgW = theImage.width || 1;
        let imgH = theImage.height || 1;
        if ((!imgW || !imgH) && typeof theImage.getOriginalSize === "function") {
            try {
                const orig = theImage.getOriginalSize();
                imgW = orig.width || 1;
                imgH = orig.height || 1;
            } catch (e) { /* 忽略 */ }
        }
        const scaleX = (points.x2 - points.x1) / imgW;
        const scaleY = (points.y2 - points.y1) / imgH;
        theImage.set({
            left: points.x1,
            top: points.y1,
            scaleX: isFinite(scaleX) && scaleX > 0 ? scaleX : 1,
            scaleY: isFinite(scaleY) && scaleY > 0 ? scaleY : 1,
            originX: "left",
            originY: "top",
            angle: 0,
            flipX: false,
            flipY: false,
        });
        theImage.setCoords();
    }

    resetOldTransform(sig) {
        const img = this.inputImages[sig];
        if (!img) return;
        img.left = 0;
        img.top = 0;
        img.scaleX = 1;
        img.scaleY = 1;
        img.angle = 0;
        img.flipX = false;
        img.flipY = false;
        img.originX = "left";
        img.originY = "top";
        img.skewY = 0;
        img.skewX = 0;
        img.perPixelTargetFind = this.preciseSelection;
        this.fcanvas.renderAll();
    }




    /**
     * 检查 sig 处的图像引用是否不为 null
     * 引用存储在 "inputImages" 中（以 sig 为 key）
     * @param sig
     * @return {boolean}
     */
    hasImageAtIndex(sig) {
        return this.inputImages[sig] != null;
    }

    /**
     * 清空所有输入图层（从画布移除并清空 inputImages 对象）。
     * 在后端每次推送新一批 images 之前调用，确保数量变化时不会有残留图层。
     */
    clearInputImages() {
        if (!this.fcanvas) {
            this.inputImages = {};
            return;
        }
        for (const sig in this.inputImages) {
            const img = this.inputImages[sig];
            if (img) {
                try { this.fcanvas.remove(img); } catch (e) { /* 忽略 */ }
            }
        }
        this.inputImages = {};
        // 隐藏可能存在的工具栏（选中对象已被移除）
        this.hideLayerToolbar();
    }

    /**
     * 返回第 index 张图层的显示名（从 1 开始计数）。
     * 现在仅用于日志/调试，存储 key 已改为整数 index。
     */
    imageNameAt(index) {
        return 'image' + (index + 1);
    }

    addImage(sig, theImage, index) {
        this.inputImages[sig] = theImage;
        // 根据画布内绿色边框（合成区域 w×h，不含 padding）的最短边等比缩放 image
        // image 最长边对齐绿色边框最短边，保持原比例，确保能放进绿色边框内
        const greenW = Number(this.w.value) || 1;
        const greenH = Number(this.h.value) || 1;
        const greenMin = Math.min(greenW, greenH);
        // 稳妥获取图片原始尺寸（优先 getOriginalSize，回退到 width/height）
        let imgW = theImage.width || 0;
        let imgH = theImage.height || 0;
        if ((!imgW || !imgH) && typeof theImage.getOriginalSize === "function") {
            try {
                const orig = theImage.getOriginalSize();
                imgW = orig.width;
                imgH = orig.height;
            } catch (e) { /* 忽略 */ }
        }
        imgW = imgW || 1;
        imgH = imgH || 1;
        const imgMax = Math.max(imgW, imgH);
        const scale = greenMin / imgMax;
        // 从左上角开始排列，每张图在 X 和 Y 轴各错位 30 像素，防止重叠
        // origin 统一为 left/top，与 transforms 的 {x1,y1,x2,y2} 语义一致
        const idx = (index != null) ? index : Object.keys(this.inputImages).length - 1;
        const offsetStep = 30;
        const posX = Number(this.p.value) + idx * offsetStep;
        const posY = Number(this.p.value) + idx * offsetStep;
        theImage.set({
            scaleX: scale,
            scaleY: scale,
            originX: "left",
            originY: "top",
            left: posX,
            top: posY,
            angle: 0,
            flipX: false,
            flipY: false,
        });
        theImage.setCoords();
        this.fcanvas.add(theImage);
    }

    replaceImage(sig, theImage) {
        const oldTransform = this.getOldTransform(sig);
        // 从画布移除旧图像
        this.fcanvas.remove(this.inputImages[sig]);
        // 用两点坐标恢复位置
        this.applyTransformFromPoints(theImage, oldTransform);
        this.fcanvas.add(theImage);
        this.inputImages[sig] = theImage;
    }

    addOrReplaceImage(theImage, sig, nodeId, r, shouldRestore, index) {
        const node = app.graph.getNodeById(nodeId);
        const instance = node.compositorInstance;
        if (instance.hasImageAtIndex(sig)) {
            instance.replaceImage(sig, theImage);
        } else {
            instance.addImage(sig, theImage, index);
        }
        // 恢复位置：用两点坐标 {x1,y1,x2,y2}，以及 locked/hidden 状态
        if (shouldRestore) {
            try {
                if (theImage) {
                    const restoreParams = r.transforms && r.transforms[sig];
                    if (restoreParams && restoreParams.x1 != null) {
                        instance.applyTransformFromPoints(theImage, restoreParams);
                    }
                    // 恢复 lock 状态（按 sig 对应）
                    const isLocked = !!(r.locked && r.locked[sig]);
                    theImage.locked = isLocked;
                    instance.applyLock(theImage);
                    // 恢复 hidden 状态（opacity=0 表示隐藏，但仍可选中）
                    const isHidden = !!(r.hidden && r.hidden[sig]);
                    theImage.set({ opacity: isHidden ? 0 : 1, visible: true });
                }
            } catch (e) {
                // console.log(e);
            }
        }

        // 无论发生什么，确保图层顺序被强制执行
        // (compositionArea < bgImage < images < compositionBorder)
        instance.enforceLayerOrder();
    }

    /**
     * 将 bg_image 放置为最底层的图像图层。
     * 背景图像不可选中、不响应事件，cover 缩放至
     * 合成区域 (w x h) 并定位在 (padding, padding)。
     * 它始终位于 compositionArea 之上、所有输入图像之下。
     */
    setBgImage(img) {
        if (!img) return;
        // 移除之前的背景图像（若有）
        if (this.bgImage) {
            this.fcanvas.remove(this.bgImage);
        }
        img.set({
            selectable: false,
            evented: false,
            hasControls: false,
            hasBorders: false,
            lockMovementX: true,
            lockMovementY: true,
            lockRotation: true,
            lockScalingX: true,
            lockScalingY: true,
            hoverCursor: 'default',
            // 自定义标记，便于后续识别此对象
            isBgImage: true,
        });
        this.bgImage = img;
        this.fcanvas.add(img);
        this.fitBgImage();
        this.enforceLayerOrder();
        this.fcanvas.renderAll();
    }

    /**
     * 将背景图像 cover 缩放并定位，精确填满
     * 合成区域 (w x h，偏移 padding, padding)。
     */
    fitBgImage() {
        if (!this.bgImage) return;
        const cw = Number(this.w.value) || 1;
        const ch = Number(this.h.value) || 1;
        // 稳妥获取 bg image 原始尺寸
        let iw = this.bgImage.width || 0;
        let ih = this.bgImage.height || 0;
        if ((!iw || !ih) && typeof this.bgImage.getOriginalSize === "function") {
            try {
                const orig = this.bgImage.getOriginalSize();
                iw = orig.width;
                ih = orig.height;
            } catch (e) { /* 忽略 */ }
        }
        iw = iw || 1;
        ih = ih || 1;
        // cover 缩放：取较大的缩放比，确保完全覆盖合成区域
        const scale = Math.max(cw / iw, ch / ih);
        this.bgImage.set({
            left: Number(this.p.value),
            top: Number(this.p.value),
            originX: 'left',
            originY: 'top',
            scaleX: scale,
            scaleY: scale,
            angle: 0,
            flipX: false,
            flipY: false,
        });
        this.bgImage.setCoords();
    }

    /**
     * 强制执行规范图层顺序：
     *   compositionArea (index 0) < bgImage < image1..N < compositionBorder (顶层)
     * 在任何结构性变更（添加/移除/重排/调整尺寸）后调用。
     */
    enforceLayerOrder() {
        if (!this.fcanvas) return;
        // 先把 bg 送到底，再把 compositionArea 置于其下
        if (this.bgImage) {
            this.fcanvas.sendToBack(this.bgImage);
        }
        if (this.compositionArea) {
            this.fcanvas.sendToBack(this.compositionArea);
        }
        // 边框始终在顶层
        if (this.compositionBorder) {
            this.fcanvas.bringToFront(this.compositionBorder);
        }
        this.fcanvas.renderAll();
    }

    /**
     * 构建浮动图层工具栏（4 个图标按钮，垂直布局）。
     * 定位在合成器容器右侧，仅在画布上选中输入图像（image1..N）时显示。
     * 按钮始终操作当前选中的图像。
     */
    createLayerToolbar() {
        const toolbar = document.createElement("div");
        toolbar.className = "Yuan_Canvas-layer-toolbar";
        toolbar.style.cssText = [
            "position: absolute",
            "right: 6px",
            "top: 50%",
            "transform: translateY(-50%)",
            "display: none",
            "flex-direction: column",
            "gap: 4px",
            "padding: 6px",
            "background: rgba(15,0,25,0.92)",
            "border: 1px solid rgba(255,255,255,0.2)",
            "border-radius: 6px",
            "z-index: 10",
            "pointer-events: auto",
            "box-shadow: 0 2px 8px rgba(0,0,0,0.4)",
        ].join(";");

        const makeBtn = (iconKey, title, onClick) => {
            const btn = document.createElement("button");
            btn.title = title;
            btn.type = "button";
            btn.dataset.action = iconKey;
            btn.style.cssText = [
                "background: transparent",
                "border: 0",
                "cursor: pointer",
                "padding: 4px",
                "color: #ddd",
                "display: inline-flex",
                "align-items: center",
                "justify-content: center",
                "line-height: 0",
                "width: 34px",
                "height: 34px",
                "border-radius: 4px",
            ].join(";");
            btn.innerHTML = makeIconSvg(LAYER_ICONS[iconKey]);
            btn.addEventListener("mouseenter", () => {
                btn.style.background = "rgba(255,255,255,0.12)";
                btn.style.color = "#fff";
            });
            btn.addEventListener("mouseleave", () => {
                btn.style.background = "transparent";
                btn.style.color = "#ddd";
            });
            btn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                onClick();
            });
            return btn;
        };

        toolbar.appendChild(makeBtn("bring_front", "Bring to Front", () => this.bringSelectedToFront()));
        toolbar.appendChild(makeBtn("send_back", "Send to Back", () => this.sendSelectedToBack()));
        // eye / eye_dashed 在 refreshLayerToolbar() 中动态切换
        toolbar.appendChild(makeBtn("eye", "Hide", () => this.toggleSelectedVisibility()));
        // lock_closed / lock_open 动态切换
        toolbar.appendChild(makeBtn("lock_open", "Lock", () => this.toggleSelectedLock()));

        return toolbar;
    }

    /**
     * 显示工具栏并绑定到当前选中的输入图像。
     * @param img 选中的输入图像（必须是 image1..N 之一）
     */
    showLayerToolbar(img) {
        if (!this.layerToolbar || !img) return;
        this.selectedLayerImage = img;
        this.layerToolbar.style.display = "flex";
        this.refreshLayerToolbar();
    }

    /** 隐藏工具栏并清除绑定的选中图像。 */
    hideLayerToolbar() {
        if (!this.layerToolbar) return;
        this.selectedLayerImage = undefined;
        this.layerToolbar.style.display = "none";
    }

    /**
     * 根据选中图像的当前可见性和锁定状态，
     * 刷新工具栏按钮的图标/标题。
     */
    refreshLayerToolbar() {
        if (!this.layerToolbar || !this.selectedLayerImage) return;
        const img = this.selectedLayerImage;
        const buttons = this.layerToolbar.querySelectorAll("button");
        buttons.forEach((btn) => {
            const action = btn.dataset.action;
            if (action === "eye") {
                const hidden = img.opacity === 0;
                btn.innerHTML = makeIconSvg(LAYER_ICONS[hidden ? "eye_dashed" : "eye"]);
                btn.title = hidden ? "Show" : "Hide";
            } else if (action === "lock_open" || action === "lock_closed") {
                const locked = !!img.locked;
                btn.innerHTML = makeIconSvg(LAYER_ICONS[locked ? "lock_closed" : "lock_open"]);
                btn.title = locked ? "Unlock" : "Lock";
                btn.dataset.action = locked ? "lock_closed" : "lock_open";
            }
        });
    }

    /**
     * 检查给定 fabric 对象是否为输入图像（image1..N）之一。
     * 用于决定选中时是否显示图层工具栏。
     */
    isInputImage(obj) {
        if (!obj) return false;
        for (const sig in this.inputImages) {
            if (this.inputImages[sig] === obj) return true;
        }
        return false;
    }

    /**
     * 将当前选中的输入图像上移一层。
     * 不会越过 compositionBorder（顶层装饰），仅在各 input image 之间调整。
     */
    bringSelectedToFront() {
        const img = this.selectedLayerImage;
        if (!img) return;
        // 使用 bringForward 逐层上移，而非 bringToFront 直接置顶
        this.fcanvas.bringForward(img);
        this.enforceLayerOrder();
        this.needsUpload = true;
    }

    /**
     * 将当前选中的输入图像下移一层。
     * 不会越过 bgImage（底层背景），仅在各 input image 之间调整。
     */
    sendSelectedToBack() {
        const img = this.selectedLayerImage;
        if (!img) return;
        // 使用 sendBackwards 逐层下移，而非 sendToBack 直接置底
        this.fcanvas.sendBackwards(img);
        this.enforceLayerOrder();
        this.needsUpload = true;
    }

    /** 切换当前选中输入图像的可见性。
     *  用 opacity=0 控制隐藏，保持 visible=true/selectable/evented，
     *  这样隐藏后选中框依旧可见可选中，用户可再次点击 eye 切换显示。
     */
    toggleSelectedVisibility() {
        const img = this.selectedLayerImage;
        if (!img) return;
        const isHidden = img.opacity === 0;
        img.set({ opacity: isHidden ? 1 : 0, visible: true });
        // 隐藏后保持选中，不 discardActiveObject，用户可点击 eye 切换显示
        this.refreshLayerToolbar();
        this.fcanvas.renderAll();
        this.needsUpload = true;
        this.persistState();
    }

    /** 切换当前选中输入图像的锁定状态。 */
    toggleSelectedLock() {
        const img = this.selectedLayerImage;
        if (!img) return;
        img.locked = !img.locked;
        this.applyLock(img);
        this.fcanvas.renderAll();
        this.refreshLayerToolbar();
        this.persistState();
    }

    /**
     * 立即将当前 transforms/locked/hidden 状态持久化到 fabricDataWidget。
     * 切换工作流时节点被销毁前，widget 值需为最新，否则状态会丢失。
     */
    persistState() {
        try {
            const serialized = Editor.serializeStuff(this.node);
            const parsed = JSON.parse(serialized);
            const hasValid = (parsed.transforms && Object.values(parsed.transforms).some((t) => t != null));
            if (hasValid) {
                this.node.fabricDataWidget.value = serialized;
            }
        } catch (e) { /* 忽略 */ }
    }

    /**
     * 根据自定义 `locked` 标志应用 fabric.js 锁定属性。
     * 参照 pano_stickers 的 Jd() 模式。
     */
    applyLock(img) {
        const locked = !!img.locked;
        img.set({
            lockMovementX: locked,
            lockMovementY: locked,
            lockRotation: locked,
            lockScalingX: locked,
            lockScalingY: locked,
            hasControls: !locked,
            // 保持可选，用户仍可点击选中（与 pano_stickers 行为一致）
            selectable: true,
            evented: true,
        });
    }


    /**
     * 初始化一个 fabricJs 实例。
     * Fabric 是引擎，使操作图像并提取最终合成图像成为可能。
     * init 参数: http://fabricjs.com/docs/fabric.Canvas.html
     * @param cavasEl 带 canvas 标签的 dom 元素
     * @return {fabric.Canvas}
     */
    static createFabricCanvas(id) {
        const canvasElement = document.getElementById(id);
        const fcanvas = new fabric.Canvas(canvasElement, {
            backgroundColor: 'transparent',
            selectionColor: 'transparent',
            selectionLineWidth: 1,
            preserveObjectStacking: true,
            altSelectionKey: "ctrlKey",
            altActionKey: "ctrlKey",
            centeredKey: "altKey",
        });

        return fcanvas;
    }

    /** 将 data URL 转为 blob */
    static dataURLToBlob = (dataURL) => {
        const parts = dataURL.split(',');
        const mime = parts[0].match(/:(.*?);/)[1];
        const binary = atob(parts[1]);
        const array = [];
        for (let i = 0; i < binary.length; i++) {
            array.push(binary.charCodeAt(i));
        }
        return new Blob([new Uint8Array(array)], {type: mime});
    }
    // #addDrawNodeHandler in code
    static uploadImage = (blob, imageNameWidget, node_id, setDone, callback) => {
        const node = app.graph.getNodeById(node_id);

        node.compositorInstance.compositionBorder.set("stroke", "orange");
        node.compositorInstance.fcanvas.renderAll();

        const UPLOAD_ENDPOINT = "/upload/image";
        //const name = `composition.png`;
        const name = `${+new Date()}.png`;
        const file = new File([blob], name);
        const body = new FormData();

        body.append("image", file);
        body.append("subfolder", "compositor");
        body.append("type", "temp");

        api.fetchApi(UPLOAD_ENDPOINT, {
            method: "POST",
            body,
        }).then((value) => {
            // debugger;
            const outputValue = `compositor/${name} [temp]`;
            //imageNameWidget.value = Math.random() > 0.5 ? outputValue : "mask.png"
            imageNameWidget.value = outputValue;

            const body = new FormData();
            body.append('filename', outputValue);
            body.append('node_id', node_id);
            body.append('overwrite', "true");


            node.compositorInstance.compositionBorder.set("stroke", node.compositorInstance.COMPOSITION_BORDER_COLOR);
            node.compositorInstance.fcanvas.renderAll();

            node.setDirtyCanvas(true, true);
            if (callback) callback()
            // 已弃用，不再需要
            if (setDone) api.fetchApi("/compositor/done", {method: "POST", body});

        }, () => {
            console.log("some error")
        });
    }

    /** 若内存中没有 blob，说明是首次运行 */
    hasNeverRun() {
        return this.cblob == undefined
    }

    /** 这个不能是 async，所以用 promise 解析和回调
     * @params setDone  **已弃用** 当 setDone 为 true 时，会为后端触发 /compositor/done 事件
     * @params callback  会传给 uploadImage，上传完成时调用
     * */
    grabUploadAndSetOutput(instance, setDone, callback) {
        // console.log("capture");
        // console.log("grap upload and set output")
        // 准备图像
        const img = new Image();
        // 通过 view api 加载已有图像，用于测试
        // api/view?filename=R.jpg&type=input&subfolder=&rand=0.6726800041773884
        // img.src = "/api/view?filename=R.jpg&type=input&subfolder=&rand=0.6726800041773884";
        this.fcanvas.discardActiveObject().renderAll();
        const data = this.fcanvas.toDataURL({
            format: 'jpeg',
            quality: 0.8,
            left: this.p.value,
            top: this.p.value,
            width: this.w.value,
            height: this.h.value
        });

        img.src = data;
        // 完成后，导出图像，用临时名上传以模拟合成
        // 并更新输出名值
        img.onload = (e) => {

            // 测试用 widget 调整 fcanvas 尺寸
            // cmp.setSize([1600/3,(1200/3)+123])
            // cmp.setDirtyCanvas(true, true)


            const blob = Editor.dataURLToBlob(data);

            if (this.hasNeverRun()) {
                Editor.uploadImage(blob, this.node.imageNameWidget, this.node.id, false, callback);
            } else {
                /**
                 * grabUploadAndSetOutput 回调不能是 async，所以这里无法等待结果和名称，
                 * 把 widget 也传给 upload image，这样我们就可以直接处理它。
                 * 不确定运行 "capture on queue" 时是否会出问题
                 */
                Editor.uploadImage(blob, this.node.imageNameWidget, this.node.id, setDone, callback);
            }

            this.cblob = blob;

            // 序列化 transforms（图像位置信息）
            const serialized = Editor.serializeStuff(this.node);
            // 有 transforms 时才保存（只保存图像位置信息，不保存图像数据）
            try {
                const parsed = JSON.parse(serialized);
                // transforms 是以 sig 为 key 的对象，检查是否有有效值
                const hasValid = (parsed.transforms && Object.values(parsed.transforms).some((t) => t != null));
                if (hasValid) {
                    this.node.fabricDataWidget.value = serialized;
                }
            } catch (e) { /* 忽略 */ }
        }
    }

    continue(setDone) {
        // 先上传当前 composition（含 transforms），上传完成后再执行工作流。
        // 这样后端执行时能拿到最新的 composition 图像。
        this.grabUploadAndSetOutput(this, setDone, () => {
            app.queuePrompt(0, 1);
        });
    }


    /**
     * 在 fabric canvas 中移动激活对象
     * @param direction [x,y] 坐标数组，范围 -1 +1，0 表示不移动
     * @param withShift
     */
    moveSelected(direction = [], withShift = false) {
        const STEP = withShift ? 10 : 1;
        const activeObject = this.fcanvas.getActiveObject();
        if (activeObject) {
            activeObject.set({
                left: activeObject.left + direction[0] * STEP,
                top: activeObject.top + direction[1] * STEP,
            });
            this.fcanvas.renderAll();
            instance.fcanvas.bringToFront(instance.compositionBorder);
            // console.log("selected objects are moved");
        }
    }

    /** 处理 fabric canvas 内的
     * - 选中
     * - 滚轮
     * - 键盘
     */
    setupfCanvasEvents(compositorInstance) {

        function isSubmit(key, ctl) {

            return key === 13 && ctl;
        }

        function isLeft(key) {
            return key === 37;
        }

        function isTop(key) {
            return key === 38;
        }

        function isRight(key) {
            return key === 39;
        }

        function isDown(key) {
            return key === 40;
        }

        function downDirection() {
            return [-1, 0];
        }

        function topDirection() {
            return [0, -1];
        }

        function rightDirection() {
            return [1, 0];
        }

        this.fcanvas.on('selection:created', function (opt) {
            this.selected = opt.selected;
            // 仅当选中输入图像（image1..N）时显示图层工具栏
            const sel = opt.selected && opt.selected[0];
            if (compositorInstance.isInputImage(sel)) {
                compositorInstance.showLayerToolbar(sel);
            } else {
                compositorInstance.hideLayerToolbar();
            }
        });

        this.fcanvas.on('selection:updated', function (opt) {
            this.selected = opt.selected;
            const sel = opt.selected && opt.selected[0];
            if (compositorInstance.isInputImage(sel)) {
                compositorInstance.showLayerToolbar(sel);
            } else {
                compositorInstance.hideLayerToolbar();
            }
        });

        this.fcanvas.on('selection:cleared', function (opt) {
            this.selected = undefined;
            compositorInstance.hideLayerToolbar();
        });

        this.fcanvas.on('mouse:out', function (opt) {
            // console.log("mouseout")
            // 移出编辑器，根据画布满载程度，此事件可能无法被拦截
            if (opt.target === null || opt.target === undefined || opt.target && opt.nextTarget === undefined) {
                compositorInstance.uploadIfNeeded(compositorInstance);
            }
        });

        this.fcanvas.on('object:modified', function (opt) {
            // console.log(this, compositorInstance);
            // 标记需要上传，这样鼠标移出时再上传并重置
            // mouse out 不可靠，有时不触发
            compositorInstance.needsUpload = true;
            compositorInstance.enforceLayerOrder();
            // 立即持久化位置/状态，切换工作流后可恢复
            compositorInstance.persistState();
        });

        this.fcanvas.on('mouse:wheel', function (opt) {
            //console.log(opt);
            try {
                if (opt.target.cacheKey !== this.selected[0].cacheKey) return;
                if (!this.selected) return

                const sign = Math.sign(opt.e.deltaY);
                opt.target.scaleX = opt.target.scaleX + (sign * 0.01);
                opt.target.scaleY = opt.target.scaleY + (sign * 0.01);
                opt.target.dirty = true;

                opt.e.preventDefault();
                opt.e.stopPropagation();
                //this.fcanvas.renderAll()
                this.renderAll()
            } catch (e) {
                return;
            }
        })

        fabric.util.addListener(document.body, 'keydown', function keydownHandler(options) {

            var key = options.which || options.keyCode; // 按键检测
            if (isLeft(key)) {
                this.moveSelected(downDirection(), options.shiftKey);
            } else if (isTop(key)) {
                this.moveSelected(topDirection(), options.shiftKey);
            } else if (isRight(key)) {
                this.moveSelected(rightDirection(), options.shiftKey);
            } else if (isDown(key)) {
                this.moveSelected([0, 1], options.shiftKey);
            } else if (isSubmit(key, options.ctrlKey)) {

                compositorInstance.uploadIfNeeded(compositorInstance);
            }
        }.bind(this));
    }

    uploadIfNeeded(compositorInstance,callback = ()=>{console.log("upload if needed")}) {

        if (compositorInstance.needsUpload) {
            compositorInstance.needsUpload = false;
            // 注意：不在此处序列化并覆盖 fabricDataWidget。
            // 因为调用 uploadIfNeeded 时 inputImages 可能已被 clearInputImages 清空
            // 且 fabric.Image.fromURL 异步加载尚未完成，此时 serializeStuff 会拿到空 inputImages，
            // 无条件覆盖会导致 fabricData 中的 transforms 被清空（位置信息丢失）。
            // grabUploadAndSetOutput 内部已有 serializeStuff + hasValid 保护，由它负责持久化。
            compositorInstance.grabUploadAndSetOutput(compositorInstance, false, callback)
        } else {
            console.log("no upload needed to be done");
        }
    }

    /**
     * 实际导出为输出的 WxH 尺寸区域
     */
    createCompositionArea() {
        //p, w, h, node
        return new fabric.Rect({
            left: this.p.value,
            top: this.p.value,
            fill: this.COMPOSITION_BACKGROUND_COLOR,
            width: this.w.value,
            height: this.h.value,
            selectable: false,
        });
    }

    /**
     * 一个非交互式矩形，内容透明，外围有彩色边框，
     * 从外圈框住合成区域，叠加在所有传入图像之上。
     * 尺寸和位置根据 width、height 和
     * COMPOSITION_BORDER_SIZE
     * COMPOSITION_BORDER_COLOR 计算
     */
    createCompositionBorder() {
        // p, w, h, node
        const compositionBorder = new fabric.Rect({
            left: this.p.value - this.COMPOSITION_BORDER_SIZE,
            top: this.p.value - this.COMPOSITION_BORDER_SIZE,
            fill: 'transparent',
            width: this.w.value + this.COMPOSITION_BORDER_SIZE * 2,
            height: this.h.value + this.COMPOSITION_BORDER_SIZE * 2,
            selectable: false,
            evented: false,
        });


        console.log("compositionBorder", compositionBorder, this.COMPOSITION_BORDER_COLOR, this.COMPOSITION_BORDER_SIZE);

        compositionBorder.set("strokeWidth", this.COMPOSITION_BORDER_SIZE);
        compositionBorder.set("stroke", this.COMPOSITION_BORDER_COLOR);
        compositionBorder.set("selectable", false);
        compositionBorder.set("evented", false);

        return compositionBorder;
    }

    calculateNodeSize() {
        const ch = this.fcanvas.getHeight();
        const cw = this.fcanvas.getWidth();
        // 余量：标题栏+端口+widgets+continue按钮+边框
        // 之前 +91 不够导致绿色边框底部被截断
        return [cw + 28, ch + 160];
    }

    initFabric(c) {


        this.getCompositorSettings()

        // wannabe widgets
        this.w = {
            value: 512, callback: (value, graphCanvas, node) => {

            }
        };
        this.h = {
            value: 512, callback: (value, graphCanvas, node) => {

            }
        };
        this.p = {
            value: 100, callback: (value, graphCanvas, node) => {

            }
        };

        const initialW = this.w.value + 2 * this.p.value;
        const initialH = this.h.value + 2 * this.p.value;
        this.containerDiv.width = initialW;
        this.containerDiv.height = initialH;
        this.containerDiv.style.width = initialW + "px";
        this.containerDiv.style.height = initialH + "px";

        if(!c) {
        this.canvasEl = Editor.createCanvasElement();



        //this.canvasEl.id = 'test'; // ditor.getRandomCompositorUniqueId();
        // this.canvasEl.id = Editor.getRandomCompositorUniqueId();

        this.containerDiv.appendChild(this.canvasEl);



        this.containerDiv.style.overflow = "hidden";
        this.canvasEl.width = this.w.value + 2 * this.p.value;
        this.canvasEl.height = this.h.value + 2 * this.p.value;


        this.fcanvas = Editor.createFabricCanvas(this.canvasEl);
    }else{
        this.containerDiv.style.overflow = "hidden";
        this.fcanvas = c;
        this.fcanvas.setWidth(this.w.value + 2 * this.p.value);
        this.fcanvas.setHeight(this.h.value + 2 * this.p.value);

    }

        console.log("this.fcanvas",this.fcanvas);

        this.compositionArea = this.createCompositionArea();
        this.compositionBorder = this.createCompositionBorder();

        this.fcanvas.add(this.compositionArea)
        this.fcanvas.add(this.compositionBorder)
        //this.fcanvas.bringToFront(this.compositionBorder);


        this.setupfCanvasEvents(this);

        // 构建浮动图层工具栏（选中输入图像时显示）
        this.layerToolbar = this.createLayerToolbar();
        this.containerDiv.appendChild(this.layerToolbar);

        // 将 containerDiv CSS 尺寸同步到 fcanvas，使 DOM widget 从一开始就稳定
        this.syncContainerSize();

        this.fcanvas.renderAll();

        this.node["compositorInstance"] = this;

        this.node.setSize(this.calculateNodeSize())
        this.node.setDirtyCanvas(true, true);

        // 使节点不可调整尺寸
        // this.node.resizable = false;
    }

    constructor(context, container) {
        this.node = context;
        this.containerDiv = container;
        this.node["compositorInstance"] = this;


        // 也在此获取设置
        // WIDGET 回调
        this.reference = context.widgets.find(w => w.name === "widgetName");
        // this.reference.callback = () => {
        //     this.someVariable = this.reference.value
        //     context.setSize...
        //     this.updateSomething()
        // }

        // 初始化库并设置事件
        // 定义回调和其他函数
    }
}


async function interrupt() {
    const response = await fetch('/interrupt', {
        method: 'POST',
        cache: 'no-cache',
        headers: {
            'Content-Type': 'text/html'
        },
    });
    return await response.json();
}
