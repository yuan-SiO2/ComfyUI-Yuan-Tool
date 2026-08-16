/**
 * Yuan Tool · 画布 前端：为 Yuan_Canvas 节点提供基于 fabric.js 的内嵌合成编辑器，
 * 接收 bg_image 与 images(batch) 作为图层，支持放置/旋转/缩放/锁定/层级调整，
 * 合成结果上传后端作为节点 IMAGE 输出，并持久化位置信息以在工作流切换后恢复。
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

/** 从上游连接节点获取图像 URL 列表（支持 batch，不缓存图像信息）。 */
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
                const url = `/view?filename=${encodeURIComponent(file)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`;
                const viewUrl = api ? api.apiURL(url) : url;
                if (viewUrl) urls.push(viewUrl);
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

    // 4) ComfyUI 内置缩略图（fallback）
    let nodeUrls = [];
    try {
        nodeUrls = typeof app?.getNodeImageUrls === "function" ? (app.getNodeImageUrls(originNode) || []) : [];
    } catch (_) { nodeUrls = []; }
    for (const c of nodeUrls) {
        const s = imageSourceFromCandidate(c);
        if (s) urls.push(s);
    }
    if (urls.length > 0) return urls;

    // 5) image widget（LoadImage 类节点）
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

/** 从已加载的 fabric.Image 计算内容签名（shape_均值），作为 transforms 的 key。 */
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

/** 注册扩展，可介入节点生命周期（api 事件与扩展钩子的详细调用顺序见 ComfyUI 文档）。 */
app.registerExtension({
    name: "Comfy.Yuan_Canvas",

    async getCustomWidgets(app) {
        // 无自定义 widget
    },
    /** 启动流程末尾调用，用于注册事件监听与全局 UI 操作。 */
    async setup(app) {
        Editor.addCompositorSettings();

        /** 当节点"返回"一个 ui 元素时，通常在处理末尾 */
        function executedMessageHandler(event, a, b) {
            const e = event.detail.output;
            const nodeId = event.detail.node;
            const node = Editor.hook(nodeId);
            if (!node || node.type != "Yuan_Canvas") {
                return;
            }
            const instance = node.compositorInstance;

            // 仅在 w/h/p 实际变化时才调整尺寸，避免每次执行不必要地重置节点尺寸
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

            instance.configChanged = e.configChanged[0];

            // 画布上是否已有图像
            const hasImagesOnCanvas = Object.keys(instance.inputImages).length > 0;

            // configChanged 为 false 且画布已有图像时保持原位不重载；
            // 画布为空时（如切换工作流再返回）需要从上游重新加载
            if (!instance.configChanged && hasImagesOnCanvas) {
                return;
            }

            // 清空前先把当前 transforms 持久化，避免 clearInputImages 导致位置丢失
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

            // config 变化时清空旧图层；未变化且画布为空时无需清空
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

            // 优先从后端 images_entries（含 sig）获取图像，fallback 从上游节点获取（前端计算 sig）
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

            // 不在执行回调时调用 uploadIfNeeded：此时图像异步加载未完成，导出会得到空图；
            // 上传应在用户点击 continue 时进行（continue 方法中先上传再执行）
        }

        // 注意：不监听 "executed" 事件。
        // 后端 composite 方法已通过 send_sync("compositor_init", ...) 推送 UI 数据，
        // executedMessageHandler 会在 compositor_init 时执行一次。
        // 若再监听 executed，同一节点每次执行会触发两次 executedMessageHandler，
        // 第二次的 clearInputImages 会清空第一次刚恢复的图像并重新异步加载，
        // 复杂的异步时序会导致 transforms 丢失（位置信息清零）。
        api.addEventListener("compositor_init", executedMessageHandler);
    },
    /** 网页加载（或重载）时调用，可劫持 app / graph（LiteGraph）修改 Comfy 核心行为。 */
    async init(args) {
    },
    /** 对每个节点类型（AddNode 菜单中的列表）调用一次，可修改 nodeType.prototype 影响该类型所有节点。 */
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
    },
    /** 在 nodeCreated 之后调用，此时 widget 值已从工作流 JSON 恢复。 */
    async loadedGraphNode(node, app) {
        if (!isYuan_Canvas(node)) return;
        const instance = node.compositorInstance;
        if (!instance) return;

        // widget 值已从工作流恢复，重新应用到画布（nodeCreated 时用的是默认值）
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

        // 切换工作流后直接从上游获取图像（不自动执行工作流）；前端计算 sig 作为 transforms 的 key
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

        // images：batch 多张图像，按 sig 恢复 transforms
        const upstreamImageUrls = getUpstreamImageUrls(node, "images");
        upstreamImageUrls.forEach((url, index) => {
            if (!url) return;
            fabric.Image.fromURL(url, function (oImg) {
                if (!oImg || !oImg.width) return;
                const sig = computeFabricImageSig(oImg) || `upstream_${index}`;
                instance.addOrReplaceImage(oImg, sig, node.id, restore, shouldRestore, index);
            }, { crossOrigin: "anonymous" });
        });

        // 刷新 firstRun 时间戳，保证下次手动执行时 IS_CHANGED 返回新值以重新拉取图像
        if (node.fabricDataWidget) {
            const data = restore || {};
            data.firstRun = Date.now();
            node.fabricDataWidget.value = JSON.stringify(data);
        }
    },
    async afterConfigureGraph(args) {
        // 不自动执行工作流获取图像：由 loadedGraphNode 从上游获取，用户手动执行
    },
    /** 节点的某个具体实例创建时调用，可修改单个节点实例。 */
    async nodeCreated(node) {
        if (!isYuan_Canvas(node)) return;

        node.imageNameWidget = getCompositorWidget(node, "imageName");
        const originalCallback = node.imageNameWidget.callback;
        node.imageNameWidget.callback = () => {
            originalCallback(arguments);
        }
        node.imageNameWidget.computeSize = () => [0, 0];
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
        // 将当前 widget 值应用到画布
        compositorInstance.onWidthChange(compositorInstance.w.value);
        compositorInstance.onHeightChange(compositorInstance.h.value);
        compositorInstance.onPaddingChange(compositorInstance.p.value);

        // ── V3 (Nodes 2.0) 尺寸适配：使 DOM 渲染尺寸与 V1 设计保持一致 ──
        // V3 专用：告知布局系统该 DOM widget 的最小尺寸（随画布尺寸变化）
        if (node.editorWidget) {
            const widget = node.editorWidget;
            const prevCLS = typeof widget.computeLayoutSize === "function"
                ? widget.computeLayoutSize.bind(widget) : null;
            widget.computeLayoutSize = (targetNode) => {
                const p = prevCLS ? (prevCLS(targetNode) || {}) : {};
                const size = compositorInstance.calculateNodeSize();
                return {
                    ...p,
                    minWidth: Math.max(size[0], Number(p.minWidth || 0)),
                    minHeight: Math.max(size[1], Number(p.minHeight || 0)),
                };
            };
            compositorInstance.enforceV3Size();
        }

        // V3 (Nodes 2.0)：widget 与输入端口共存，
        // 移除 fabricData / imageName 的占位端口（避免节点被端口拉长）
        if (compositorInstance.v3NodeElement) {
            for (const name of ["fabricData", "imageName"]) {
                const slot = node.findInputSlot(name);
                if (slot >= 0 && !node.inputs[slot].link) {
                    node.removeInput(slot);
                }
            }
        }

        // grabUploadAndSetOutput 回调不能是 async，故把 widget 传给 uploadImage 直接处理
        node.continue = node.addWidget("button", "continue", "continue", compositorInstance.continue.bind(compositorInstance));
    },
});

// 来自 melmass
function hideWidgetForGood(node, widget, suffix = '') {
    widget.origType = widget.type
    widget.origComputeSize = widget.computeSize
    widget.origSerializeValue = widget.serializeValue
    widget.computeSize = () => [0, -4] // -4 是因为 litegraph 会在 widget 间自动加间隙
    widget.type = "converted-widget" + suffix

    // 隐藏关联的 widget，例如 seed+seedControl
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            hideWidgetForGood(node, w, ':' + widget.name)
        }
    }
}

/** 将在节点创建时通过 addDOMWidget 添加到节点 */
class Editor {
    canvasEl;
    /** fabric canvas */
    fcanvas;
    /** 传给 addDomWidget 的 dom 元素 */
    containerDiv;
    /** 上一张图像，可能只需要其哈希 */
    cblob;
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

    /** （widget）引用 / 配置参数 */
    p;
    w;
    h;

    /** 以图像内容签名(sig)为 key 存储输入图像图层，调换 batch 顺序时 transforms 仍能正确对应 */
    inputImages = {};
    fabricDataWidget;
    needsUpload = false;

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

    /** 序列化图像位置（sig + 两点坐标）及 locked/hidden 状态，切换工作流后按 sig 恢复。 */
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

    static createCanvasElement() {
        const canvas = document.createElement("canvas");
        canvas.id = Editor.getRandomCompositorUniqueId();
        return canvas;
    }

    onHeightChange(value) {
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

    /** 将 containerDiv 的 CSS 尺寸同步为 fabric canvas 尺寸，稳定 DOM widget 布局。 */
    syncContainerSize() {
        if (!this.fcanvas || !this.containerDiv) return;
        const w = this.fcanvas.getWidth();
        const h = this.fcanvas.getHeight();
        this.containerDiv.style.width = w + "px";
        this.containerDiv.style.height = h + "px";
        this.enforceV3Size();
    }

    /**
     * V3 (Nodes 2.0) 尺寸适配：在 comfy-node 元素上同步节点最小尺寸，
     * 使 DOM 渲染尺寸与 V1 设计保持一致（V1 无 comfy-node 元素，直接跳过）。
     */
    enforceV3Size() {
        try {
            const size = this.calculateNodeSize();
            let el = this.containerDiv?.parentElement;
            while (el) {
                if ((el.tagName && el.tagName.toLowerCase().includes('comfy-node')) ||
                    (el.classList && el.classList.contains('comfy-node'))) break;
                el = el.parentElement || (el.getRootNode ? el.getRootNode().host : null);
            }
            if (!el || !this.node) return;
            this.v3NodeElement = el;
            this.node.min_size = size;
            el.style.removeProperty("min-width");
            el.style.setProperty("min-width", size[0] + "px", "important");
            el.style.setProperty("min-height", size[1] + "px", "important");
        } catch (_) {}
    }

    getOldTransform(sig) {
        const ref = this.inputImages[sig];
        // 用左上角(x1,y1)与右下角(x2,y2)两点确定图像的位置和大小
        const bounds = ref.getBoundingRect();
        return {
            x1: bounds.left,
            y1: bounds.top,
            x2: bounds.left + bounds.width,
            y2: bounds.top + bounds.height,
        };
    }

    /** 用两点坐标 {x1,y1,x2,y2}（左上角/右下角）恢复图像的位置和缩放。 */
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

    /** 检查 sig 处的图像引用是否不为 null（引用存储在 inputImages 中，以 sig 为 key）。 */
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

    addImage(sig, theImage, index) {
        this.inputImages[sig] = theImage;
        // 按合成区域（w×h，不含 padding）最短边等比缩放，确保图像能放进绿色边框内
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
        // 从左上角排列，每张图在 X/Y 轴各错位 30 像素，防止重叠
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
        // 按 sig 恢复位置（两点坐标）及 locked/hidden 状态
        if (shouldRestore) {
            try {
                if (theImage) {
                    const restoreParams = r.transforms && r.transforms[sig];
                    if (restoreParams && restoreParams.x1 != null) {
                        instance.applyTransformFromPoints(theImage, restoreParams);
                    }
                    // 恢复 lock 状态
                    const isLocked = !!(r.locked && r.locked[sig]);
                    theImage.locked = isLocked;
                    instance.applyLock(theImage);
                    // 恢复 hidden 状态（opacity=0 表示隐藏，但仍可选中）
                    const isHidden = !!(r.hidden && r.hidden[sig]);
                    theImage.set({ opacity: isHidden ? 0 : 1, visible: true });
                }
            } catch (e) {
            }
        }

        // 确保图层顺序正确 (compositionArea < bgImage < images < compositionBorder)
        instance.enforceLayerOrder();
    }

    /**
     * 将 bg_image 作为不可选中的最底层图层（compositionArea 之上、输入图像之下），
     * cover 铺满合成区域（w×h，偏移 padding）。
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

    /** 将背景图像 cover 缩放定位，精确填满合成区域（w×h，偏移 padding）。 */
    fitBgImage() {
        if (!this.bgImage) return;
        const cw = Number(this.w.value) || 1;
        const ch = Number(this.h.value) || 1;
        // 优先 getOriginalSize 获取原始尺寸，回退到 width/height
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
        // cover 缩放：取较大缩放比，确保完全覆盖合成区域
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
     * 强制执行规范图层顺序（compositionArea < bgImage < images < compositionBorder）。
     * 在任何结构性变更（添加/移除/重排/调整尺寸）后调用。
     */
    enforceLayerOrder() {
        if (!this.fcanvas) return;
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

    /** 构建浮动图层工具栏（4 个图标按钮，垂直布局，操作当前选中的输入图像）。 */
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

    /** 显示工具栏并绑定到当前选中的输入图像（必须是输入图像之一）。 */
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

    /** 根据选中图像的可见性和锁定状态，刷新工具栏按钮的图标/标题。 */
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

    /** 判断对象是否为输入图像之一（决定选中时是否显示工具栏）。 */
    isInputImage(obj) {
        if (!obj) return false;
        for (const sig in this.inputImages) {
            if (this.inputImages[sig] === obj) return true;
        }
        return false;
    }

    /** 将选中的输入图像上移一层（不越过 compositionBorder）。 */
    bringSelectedToFront() {
        const img = this.selectedLayerImage;
        if (!img) return;
        // 使用 bringForward 逐层上移，而非 bringToFront 直接置顶
        this.fcanvas.bringForward(img);
        this.enforceLayerOrder();
        this.needsUpload = true;
    }

    /** 将选中的输入图像下移一层（不越过 bgImage）。 */
    sendSelectedToBack() {
        const img = this.selectedLayerImage;
        if (!img) return;
        // 使用 sendBackwards 逐层下移，而非 sendToBack 直接置底
        this.fcanvas.sendBackwards(img);
        this.enforceLayerOrder();
        this.needsUpload = true;
    }

    /** 切换选中输入图像的可见性（opacity=0 隐藏，保持可见可选中以便再次切换）。 */
    toggleSelectedVisibility() {
        const img = this.selectedLayerImage;
        if (!img) return;
        const isHidden = img.opacity === 0;
        img.set({ opacity: isHidden ? 1 : 0, visible: true });
        // 隐藏后保持选中，用户可再次点击 eye 切换显示
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

    /** 将当前 transforms/locked/hidden 状态持久化到 fabricDataWidget（切换工作流前需为最新）。 */
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

    /** 根据自定义 `locked` 标志应用 fabric.js 锁定属性（参照 pano_stickers 模式）。 */
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


    /** 创建 fabric.Canvas 实例（透明背景、ctrlKey 组合键等统一配置）。 */
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
    static uploadImage = (blob, imageNameWidget, node_id, setDone, callback) => {
        const node = app.graph.getNodeById(node_id);

        node.compositorInstance.compositionBorder.set("stroke", "orange");
        node.compositorInstance.fcanvas.renderAll();

        const UPLOAD_ENDPOINT = "/upload/image";
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
            const outputValue = `compositor/${name} [temp]`;
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

    /** 不能是 async，故用 promise 解析和回调；setDone 已弃用，callback 为上传完成回调。 */
    grabUploadAndSetOutput(instance, setDone, callback) {
        // 导出合成区域（w×h，偏移 padding）为图像数据
        const img = new Image();
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
        // 导出后用临时名上传模拟合成，并更新输出名
        img.onload = (e) => {

            const blob = Editor.dataURLToBlob(data);

            if (this.hasNeverRun()) {
                Editor.uploadImage(blob, this.node.imageNameWidget, this.node.id, false, callback);
            } else {
                // 把 widget 传给 uploadImage，上传完成后直接更新输出名
                Editor.uploadImage(blob, this.node.imageNameWidget, this.node.id, setDone, callback);
            }

            this.cblob = blob;

            // 有有效 transforms 时才持久化（只保存位置信息，不保存图像数据）
            const serialized = Editor.serializeStuff(this.node);
            try {
                const parsed = JSON.parse(serialized);
                const hasValid = (parsed.transforms && Object.values(parsed.transforms).some((t) => t != null));
                if (hasValid) {
                    this.node.fabricDataWidget.value = serialized;
                }
            } catch (e) { /* 忽略 */ }
        }
    }

    continue(setDone) {
        // 先上传当前 composition（含 transforms），上传完成后再执行工作流
        this.grabUploadAndSetOutput(this, setDone, () => {
            app.queuePrompt(0, 1);
        });
    }


    /** 在 fabric canvas 中移动激活对象；direction 为 [-1,1] 方向向量，withShift 时步长 10。 */
    moveSelected(direction = [], withShift = false) {
        const STEP = withShift ? 10 : 1;
        const activeObject = this.fcanvas.getActiveObject();
        if (activeObject) {
            activeObject.set({
                left: activeObject.left + direction[0] * STEP,
                top: activeObject.top + direction[1] * STEP,
            });
            this.fcanvas.renderAll();
            this.fcanvas.bringToFront(this.compositionBorder);
        }
    }

    /** 处理 fabric canvas 内的选中、滚轮与键盘事件。 */
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
            // 移出编辑器，根据画布满载程度，此事件可能无法被拦截
            if (opt.target === null || opt.target === undefined || opt.target && opt.nextTarget === undefined) {
                compositorInstance.uploadIfNeeded(compositorInstance);
            }
        });

        this.fcanvas.on('object:modified', function (opt) {
            // 标记需要上传，这样鼠标移出时再上传并重置
            // mouse out 不可靠，有时不触发
            compositorInstance.needsUpload = true;
            compositorInstance.enforceLayerOrder();
            // 立即持久化位置/状态，切换工作流后可恢复
            compositorInstance.persistState();
        });

        this.fcanvas.on('mouse:wheel', function (opt) {
            try {
                if (opt.target.cacheKey !== this.selected[0].cacheKey) return;
                if (!this.selected) return

                const sign = Math.sign(opt.e.deltaY);
                opt.target.scaleX = opt.target.scaleX + (sign * 0.01);
                opt.target.scaleY = opt.target.scaleY + (sign * 0.01);
                opt.target.dirty = true;

                opt.e.preventDefault();
                opt.e.stopPropagation();
                this.renderAll()
            } catch (e) {
                return;
            }
        })

        fabric.util.addListener(document.body, 'keydown', function keydownHandler(options) {

            var key = options.which || options.keyCode;
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

    uploadIfNeeded(compositorInstance, callback) {
        if (compositorInstance.needsUpload) {
            compositorInstance.needsUpload = false;
            // 注意：不在此处序列化并覆盖 fabricDataWidget——
            // 此时 inputImages 可能已被清空、图像异步加载未完成，serializeStuff 会拿到空数据，
            // 无条件覆盖会把 fabricData 中的 transforms 清空（位置丢失）。
            // 持久化由 grabUploadAndSetOutput 内部的 serializeStuff + hasValid 保护负责。
            compositorInstance.grabUploadAndSetOutput(compositorInstance, false, callback)
        }
    }

    /** 实际导出为输出的 WxH 尺寸区域 */
    createCompositionArea() {
        return new fabric.Rect({
            left: this.p.value,
            top: this.p.value,
            fill: this.COMPOSITION_BACKGROUND_COLOR,
            width: this.w.value,
            height: this.h.value,
            selectable: false,
        });
    }

    /** 非交互式透明矩形，外围彩色边框从外圈框住合成区域，叠加在所有输入图像之上。 */
    createCompositionBorder() {
        const compositionBorder = new fabric.Rect({
            left: this.p.value - this.COMPOSITION_BORDER_SIZE,
            top: this.p.value - this.COMPOSITION_BORDER_SIZE,
            fill: 'transparent',
            width: this.w.value + this.COMPOSITION_BORDER_SIZE * 2,
            height: this.h.value + this.COMPOSITION_BORDER_SIZE * 2,
            selectable: false,
            evented: false,
        });

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

        if (!c) {
            this.canvasEl = Editor.createCanvasElement();
            this.containerDiv.appendChild(this.canvasEl);
            this.containerDiv.style.overflow = "hidden";
            this.canvasEl.width = this.w.value + 2 * this.p.value;
            this.canvasEl.height = this.h.value + 2 * this.p.value;
            this.fcanvas = Editor.createFabricCanvas(this.canvasEl);
        } else {
            this.containerDiv.style.overflow = "hidden";
            this.fcanvas = c;
            this.fcanvas.setWidth(this.w.value + 2 * this.p.value);
            this.fcanvas.setHeight(this.h.value + 2 * this.p.value);
        }

        this.compositionArea = this.createCompositionArea();
        this.compositionBorder = this.createCompositionBorder();

        this.fcanvas.add(this.compositionArea)
        this.fcanvas.add(this.compositionBorder)

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
    }

    constructor(context, container) {
        this.node = context;
        this.containerDiv = container;
        this.node["compositorInstance"] = this;
    }
}
