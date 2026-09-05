/** Yuan Tool · 前端共享工具库：供各节点前端导出的公共函数（ES module，无副作用） */

/** 从 window.comfyAPI 获取 api 实例（兼容不同 ComfyUI 版本）。 */
export function getApi() {
    try {
        const c = window.comfyAPI;
        if (c && c.api) {
            if (c.api.api && typeof c.api.api.apiURL === "function") return c.api.api;
            if (typeof c.api.apiURL === "function") return c.api.api;
        }
    } catch (_) {}
    return null;
}

/** 向上查找 V3 (Nodes 2.0) 前端的 comfy-node 祖先元素；V1 前端返回 null。 */
export function findComfyNodeEl(startEl) {
    let el = startEl;
    while (el) {
        if ((el.tagName && el.tagName.toLowerCase().includes("comfy-node")) ||
            (el.classList && el.classList.contains("comfy-node"))) {
            return el;
        }
        el = el.parentElement || (el.getRootNode ? el.getRootNode().host : null);
    }
    return null;
}

/** 在 V3 comfy-node 元素上强制最小尺寸；minW 传 null 时仅强制高度 */
export function enforceV3MinSize(comfyNodeEl, minW, minH) {
    if (!comfyNodeEl) return;
    comfyNodeEl.style.removeProperty("min-width");
    if (minW != null) {
        comfyNodeEl.style.setProperty("min-width", minW + "px", "important");
    }
    comfyNodeEl.style.setProperty("min-height", minH + "px", "important");
}

/** 移除 V3 下隐藏管理参数的空占位端口（避免节点被拉长）；已有连线的端口保留不删 */
export function removeV3PlaceholderInput(node, name) {
    if (!node || !Array.isArray(node.inputs)) return false;
    const slot = node.findInputSlot(name);
    if (slot >= 0 && !node.inputs[slot].link) {
        try { node.removeInput(slot); return true; } catch (_) {}
    }
    return false;
}

/** 将 ComfyUI 图像 UI 条目（{filename, subfolder, type, storage}）转为 /view URL；api 为空时返回相对路径。 */
export function comfyImageEntryToUrl(entry, api) {
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
    const q = `/view?${params.toString()}`;
    return (api && typeof api.apiURL === "function") ? api.apiURL(q) : q;
}

/** 从 ComfyUI 图像 UI 条目、字符串或数组中提取 /view URL（递归解析数组与嵌套结构）。 */
export function imageSourceFromCandidate(candidate, api) {
    if (!candidate) return "";
    if (typeof candidate === "string") return String(candidate).trim();
    if (Array.isArray(candidate)) {
        if (candidate.length === 0) return "";
        if (candidate.length === 1) return imageSourceFromCandidate(candidate[0], api);
        const filename = typeof candidate[0] === "string" ? String(candidate[0]).trim() : "";
        if (filename) {
            return comfyImageEntryToUrl({
                filename,
                subfolder: String(candidate[1] || "").trim(),
                type: String(candidate[2] || "output").trim() || "output",
            }, api);
        }
        for (const e of candidate) {
            const s = imageSourceFromCandidate(e, api);
            if (s) return s;
        }
        return "";
    }
    if (typeof candidate?.src === "string" && candidate.src) return candidate.src;
    if (typeof candidate?.url === "string" && candidate.url) return candidate.url;
    return comfyImageEntryToUrl(candidate, api);
}

/** 从 app.nodeOutputs 中查找指定节点的输出条目（兼容 Map/对象两种存储）。 */
export function lookupNodeOutputEntry(app, nodeId) {
    const store = app?.nodeOutputs;
    if (!store || nodeId == null) return null;
    const raw = String(nodeId);
    if (store instanceof Map) {
        return store.get(nodeId) || store.get(raw) || store.get(Number(raw)) || null;
    }
    return store[nodeId] || store[raw] || null;
}

/** 判断文件名是否为支持的音频/视频扩展名（含视频提取音轨）。 */
const _AUDIO_EXTS = [
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma",
    ".aiff", ".aif", ".mp4", ".m4v", ".webm", ".mov", ".avi", ".mkv",
];
export function isAudioFileName(name) {
    const dot = name.lastIndexOf(".");
    if (dot < 0) return false;
    return _AUDIO_EXTS.indexOf(name.slice(dot).toLowerCase()) >= 0;
}

/** 链式包装 ComfyUI 节点生命周期/绘制回调，保留原回调再追加新逻辑。 */
export function chainCallback(object, property, callback) {
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

/** 隐藏 ComfyUI widget（兼容 V2 原生绘制与 V3 Vue 模式）。 */
export function hideWidget(w) {
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

/** 恢复被 hideWidget 隐藏的 ComfyUI widget。 */
export function showWidget(w) {
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

/** 通用分块上传循环：把大文件切片逐块 POST 到 uploadUrl，返回最后一块响应的 JSON。

    与后端 handle_chunk_upload 对应。sendChunk(formData) 由调用方自定义
    （用 fetch 或 api.fetchApi、指定 URL、校验状态码），返回解析后的 JSON。
 */
export async function uploadChunked(file, {
    chunkSize = 10 * 1024 * 1024,
    filename,
    sendChunk,
    onProgress,
} = {}) {
    const name = filename != null ? filename : file.name;
    const totalChunks = Math.max(1, Math.ceil(file.size / chunkSize));
    let last = null;
    for (let i = 0; i < totalChunks; i++) {
        const blob = file.slice(i * chunkSize, Math.min((i + 1) * chunkSize, file.size));
        const formData = new FormData();
        formData.append("file", blob);
        formData.append("filename", name);
        formData.append("chunk_index", String(i));
        formData.append("total_chunks", String(totalChunks));
        last = await sendChunk(formData, i, totalChunks, name);
        if (onProgress) onProgress(i + 1, totalChunks, name);
    }
    return last;
}
