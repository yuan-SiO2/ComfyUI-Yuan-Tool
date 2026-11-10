/** Yuan Tool · 前端共享工具库：供各节点前端导出的公共函数（ES module，无副作用） */

/** 从 window.comfyAPI 获取 api 实例（兼容不同 ComfyUI 版本）。 */
export function getApi() {
    try {
        const c = window.comfyAPI;
        if (c && c.api) {
            if (c.api.api && typeof c.api.api.apiURL === "function") return c.api.api;
            if (typeof c.api.apiURL === "function") return c.api;
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
    let slot = typeof node.findInputSlot === "function" ? node.findInputSlot(name) : -1;
    if (slot < 0) slot = node.inputs.findIndex((inp) => inp && inp.name === name);
    if (slot >= 0 && node.inputs[slot] && !node.inputs[slot].link) {
        try { node.removeInput(slot); return true; } catch (_) {}
    }
    return false;
}

/**
 * 设置端口（输入/输出槽）的显示名。
 * 前端绘制标签用的是 `localized_name || name`（Slot.draw），而后端 RETURN_NAMES 会把
 * localized_name 预填成声明名；只改 name 时画布/重载后仍显示旧名。
 */
export function setSlotName(slot, name) {
    if (!slot) return;
    slot.name = name;
    slot.localized_name = name;
}

/**
 * 前端是否为 V3（Nodes 2.0）渲染。
 * 先读 LiteGraph 的开关，再看文档里是否存在 comfy-node 元素——不能依赖「本节点的
 * 某个控件是否已挂载 DOM」，否则占位端口的清理会随挂载时机不同而静默失效。
 */
let v3ModeCache = null;
export function isV3NodesMode() {
    if (v3ModeCache !== null) return v3ModeCache;
    let v3 = false;
    try {
        if (window.LiteGraph && window.LiteGraph.vueNodesMode) v3 = true;
    } catch (_) {}
    if (!v3) {
        try { v3 = !!document.querySelector("comfy-node"); } catch (_) {}
    }
    v3ModeCache = v3;
    return v3;
}

/**
 * 清理"名字已知"的隐藏控件占位端口（V3 下用）。签名去重：端口列表未变时零开销。
 */
export function prunePlaceholderPorts(node, names) {
    if (!isV3NodesMode()) return;
    pruneNamedPorts(node, names);
}

/**
 * 清理"本节点所有隐藏控件"的占位端口：不依赖任何名字清单或前端版本判定。
 * 只要控件被本节点隐藏且端口未连线，就移除——新增格式参数也不会漏（杜绝"删一个
 * 又来一个"）。被用户显式转成输入的控件已不在 node.widgets 里，因此不会被误删。
 */
export function pruneHiddenWidgetPorts(node) {
    if (!node || !Array.isArray(node.inputs) || !Array.isArray(node.widgets)) return;
    const hidden = [];
    for (const w of node.widgets) {
        if (!w || !w.name) continue;
        if (w.hidden || (w.options && w.options.hidden)) hidden.push(w.name);
    }
    if (hidden.length) pruneNamedPorts(node, hidden);
}

/** 移除一批名字对应的未连线输入端口；以「名字 + 端口列表」签名去重。 */
function pruneNamedPorts(node, names) {
    if (!node || !Array.isArray(node.inputs) || !names || !names.length) return;
    const sig = () => `${names.join("|")}::${node.inputs.map((i) => i && i.name).join("|")}`;
    if (sig() === node._yuanPortPruneSig) return;
    for (const name of names) removeV3PlaceholderInput(node, name);
    node._yuanPortPruneSig = sig();
}


/** 将 ComfyUI 图像 UI 条目（{filename, subfolder, type, storage}）转为 /view URL；api 为空时返回相对路径。 */
function comfyImageEntryToUrl(entry, api) {
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

/** 加载工作流后按存档里的「端口名」把连线重新指到同名端口。
/** 按存档里的端口名把连线重新指到当前节点的同名端口；alias 为旧端口名映射。
 *  info 为存档原文（configure 入参），含 outputs/inputs 的端口名。 */
export function remapSavedPorts(node, info, alias = {}) {
    const graph = node?.graph;
    const links = graph?.links;
    if (!info || !links) return;
    // 按「存档槽位下标 → 端口名 → 现在同名端口的槽位」迁移该槽位上的所有连线。
    // 不依赖存档里的 links 字段（可能是 null），以链接仓库里的 origin_slot / target_slot 为准。
    const moves = [];
    for (const [i, o] of (info.outputs ?? []).entries()) {
        const to = node.outputs?.findIndex((slot) => slot?.name === (alias[o?.name] ?? o?.name)) ?? -1;
        if (to < 0 || to === i) continue;
        for (const link of links.values()) {
            if (link?.origin_id === node.id && link.origin_slot === i) {
                moves.push({ out: true, from: i, to, otherId: link.target_id, otherSlot: link.target_slot });
            }
        }
    }
    for (const [i, input] of (info.inputs ?? []).entries()) {
        const to = node.inputs?.findIndex((slot) => slot?.name === (alias[input?.name] ?? input?.name)) ?? -1;
        if (to < 0 || to === i) continue;
        for (const link of links.values()) {
            if (link?.target_id === node.id && link.target_slot === i) {
                moves.push({ out: false, from: i, to, otherId: link.origin_id, otherSlot: link.origin_slot });
            }
        }
    }
    if (!moves.length) return;
    // 先断开涉及的槽位（出线与进线分开处理，避免混用同一批槽位号）
    for (const m of moves) {
        if (m.out) node.disconnectOutput?.(m.from);
    }
    for (const m of moves) {
        if (!m.out) node.disconnectInput?.(m.from);
    }
    for (const m of moves) {
        const other = graph.getNodeById?.(m.otherId);
        if (!other) continue;
        if (m.out) node.connect(m.to, other, m.otherSlot);
        else other.connect(m.otherSlot, node, m.to);
    }
}

/** 隐藏 ComfyUI widget（兼容 V2 原生绘制与 V3 Vue 模式）。 */
export function hideWidget(w) {
    if (!w) return;
    w.hidden = true;
    if (!w.options) w.options = {};
    w.options.hidden = true;
    // 注意：不要写 w.visibility —— 它是只读 getter（内部 _visibility），
    // 直接赋值会在严格模式抛 TypeError 并中断工作流加载；hidden 的 setter 已经同步了它。
    // 旧画布渲染由 LiteGraph 自绘、不读 visibility：用零高度 + 空绘制达到同样效果
    if (!w._yuanHiddenHooked) {
        w._yuanHiddenHooked = true;
        w._yuanOrigDraw = Object.prototype.hasOwnProperty.call(w, "draw") ? w.draw : undefined;
        w._yuanOrigSize = w.computeSize;
    }
    w.computeSize = () => [0, -4];
    w.draw = () => {};
    if (w.element) w.element.style.display = "none";
}

/** 恢复被 hideWidget 隐藏的 ComfyUI widget。 */
export function showWidget(w) {
    if (!w) return;
    w.hidden = false;
    if (w.options) w.options.hidden = false;
    if (w._yuanHiddenHooked) {
        if (w._yuanOrigDraw !== undefined) w.draw = w._yuanOrigDraw;
        else delete w.draw;
        if (w._yuanOrigSize) w.computeSize = w._yuanOrigSize;
        else delete w.computeSize;
        delete w._yuanHiddenHooked;
        delete w._yuanOrigDraw;
        delete w._yuanOrigSize;
    }
    if (w.element) w.element.style.display = "";
}

/** 通用分块上传：逐块 POST 到后端 handle_chunk_upload，返回最后一块响应 JSON（sendChunk 由调用方自定义）。 */
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
