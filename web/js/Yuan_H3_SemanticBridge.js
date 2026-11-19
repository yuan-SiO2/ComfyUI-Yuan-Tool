/** Yuan Tool · H3 语义桥前端：模式切换时显隐参数。
 *
 * 手动模式显示手填参数（模型文件名/启用/强度/幅度匹配/作用范围/分块），
 * 自动模式只显示「模型（自动）」。移植自 ComfyUI-Yuan 的 web/semantic_bridge_mode.js。
 * 只切 widget 可见性（hidden/options.hidden），不长期覆盖 computeSize，避免与 Vue 布局打架。
 */
const { app } = window.comfyAPI.app;

const NODE_NAME = "Yuan_H3SemanticBridge";

const MODE_MANUAL = "手动模式";
const MODE_AUTO = "自动模式";
const MODES = [MODE_MANUAL, MODE_AUTO];

// 各模式可见的控件：手动用「模型（手动）」文件名，自动用「模型（自动）」名称。
const MODE_WIDGETS = {
    [MODE_MANUAL]: ["model_name", "enabled", "alpha", "magnitude_match", "token_scope", "chunk_tokens"],
    [MODE_AUTO]: ["model_preset"],
};
const ALL_MODE_WIDGETS = [...new Set([...MODE_WIDGETS[MODE_MANUAL], ...MODE_WIDGETS[MODE_AUTO]])];
// 被摘掉的端口补回时用；类型与后端 schema 一致。
const PORT_TYPES = {
    enabled: "BOOLEAN",
    alpha: "FLOAT",
    magnitude_match: "COMBO",
    token_scope: "COMBO",
    chunk_tokens: "INT",
};

function findWidget(node, name) {
    return node && node.widgets ? node.widgets.find((w) => w && w.name === name) : undefined;
}

function normalizeMode(widget) {
    if (!widget) return MODE_MANUAL;
    const value = String(widget.value == null ? "" : widget.value).trim();
    if (MODES.includes(value)) return value;
    widget.value = MODE_MANUAL;
    return MODE_MANUAL;
}

// 只动 visible 标记，不碰 computeSize；旧版本留下的收起补丁顺手清掉，避免"再也显示不出来"。
function setWidgetHidden(widget, hidden) {
    if (!widget) return;
    if (widget._yuanCollapsed) {
        delete widget.computeSize;
        delete widget._yuanCollapsed;
    }
    widget.hidden = hidden;
    if (widget.options) widget.options.hidden = hidden;
    if (widget.element) widget.element.style.display = hidden ? "none" : "";
    if (widget.inputEl) widget.inputEl.style.display = hidden ? "none" : "";
}

// 隐藏的控件可能留一个空端口占位：没连线的才摘，摘过就记下来。
function pruneWidgetPort(node, name) {
    if (!node || !Array.isArray(node.inputs) || typeof node.removeInput !== "function") return;
    const index = node.inputs.findIndex((input) => input && input.name === name && input.link == null);
    if (index === -1) return;
    node._yuanPrunedPorts = node._yuanPrunedPorts || {};
    node._yuanPrunedPorts[name] = true;
    node.removeInput(index);
}

function restoreWidgetPort(node, name) {
    const pruned = node && node._yuanPrunedPorts;
    if (!pruned || !pruned[name] || typeof node.addInput !== "function") return;
    delete pruned[name];
    if (Array.isArray(node.inputs) && node.inputs.some((input) => input && input.name === name)) return;
    node.addInput(name, PORT_TYPES[name] || "STRING", { optional: true });
    const input = Array.isArray(node.inputs) ? node.inputs.find((item) => item && item.name === name) : undefined;
    if (input) input.widget = { name };
}

function applyMode(node, mode) {
    const visible = new Set(MODE_WIDGETS[mode] || []);
    for (const name of ALL_MODE_WIDGETS) {
        const show = visible.has(name);
        setWidgetHidden(findWidget(node, name), !show);
        if (show) restoreWidgetPort(node, name);
        else pruneWidgetPort(node, name);
    }
    if (node && typeof node.setSize === "function" && typeof node.computeSize === "function") {
        const computed = node.computeSize();
        const width = node.size ? Math.max(node.size[0], computed[0]) : computed[0];
        node.setSize([width, node.computeSize()[1]]);
    }
    if (app && app.graph) app.graph.setDirtyCanvas(true, true);
}

function syncMode(node) {
    const widget = findWidget(node, "mode");
    const mode = normalizeMode(widget);
    node._yuanMode = mode;
    applyMode(node, mode);
    return mode;
}

app.registerExtension({
    name: "YuanTool.SemanticBridge.Mode",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            normalizeMode(findWidget(this, "mode"));
            // 加载期改可见性/尺寸会和 LGraph.configure 抢时序，推迟到本轮结束后。
            if (!this._yuanModeDefer) {
                this._yuanModeDefer = true;
                setTimeout(() => {
                    this._yuanModeDefer = false;
                    syncMode(this);
                }, 0);
            }
            return result;
        };

        // 部分前端版本的 V3 下拉不触发 callback：每次重绘前比一下模式值。
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const widget = findWidget(this, "mode");
            if (widget) {
                const mode = normalizeMode(widget);
                if (mode !== this._yuanMode) syncMode(this);
            }
            if (onDrawForeground) return onDrawForeground.apply(this, arguments);
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (this._yuanModeWired) return result;
            this._yuanModeWired = true;

            const widget = findWidget(this, "mode");
            if (widget) {
                const original = widget.callback;
                widget.callback = (...args) => {
                    const out = original ? original.apply(widget, args) : undefined;
                    syncMode(this);
                    return out;
                };
            }
            syncMode(this);
            return result;
        };
    },
});
