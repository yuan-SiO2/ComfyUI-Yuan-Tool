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
