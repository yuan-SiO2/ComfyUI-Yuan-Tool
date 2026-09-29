// 「自定义选项」节点：编辑模式填写选项名称，选项模式按名称显示互斥开关（只能开启一个）。
// 部件顺序：[模式, index(隐藏), text(隐藏), option1, pick1, option2, pick2, …]，名称框与开关同后缀一一对应。

// app 实例可能挂在 comfyAPI.app.app 下
const comfyAPI = window.comfyAPI ?? {};
const app = comfyAPI.app?.app ?? comfyAPI.app;

const YUAN_CUSTOM_COMBO_TYPE = "YuanCustomCombo";
const MODE_EDIT = "编辑";
const INDEX_WIDGET_NAME = "index";
const TEXT_WIDGET_NAME = "text";
const OPTION_PREFIX = "option";
const PICK_PREFIX = "pick";
const OPTION_LABEL_PREFIX = "选项名称";

// 先执行原回调，再执行追加回调
function chainCallback(originalCallback, ...callbacks) {
    return function (...args) {
        if (typeof originalCallback === "function") originalCallback.call(this, ...args);
        for (const callback of callbacks) callback.call(this, ...args);
    };
}

function onYuanCustomComboCreated() {
    const node = this;
    const widgets = node.widgets;
    const modeWidget = widgets?.find((w) => w.name === "mode");
    if (!modeWidget) return;

    const optionWidgets = () => widgets.filter((w) => w.name.startsWith(OPTION_PREFIX));
    const pickWidgets = () => widgets.filter((w) => w.name.startsWith(PICK_PREFIX));
    const pickOf = (optionWidget) =>
        widgets.find((w) => w.name === PICK_PREFIX + optionWidget.name.slice(OPTION_PREFIX.length));
    const optionOf = (pickWidget) =>
        widgets.find((w) => w.name === OPTION_PREFIX + pickWidget.name.slice(PICK_PREFIX.length));
    const nameOf = (pickWidget) => {
        const optionWidget = optionOf(pickWidget);
        return optionWidget ? `${optionWidget.value ?? ""}` : "";
    };
    // 只有「编辑」显示名称框，其余值按选项模式处理
    const isEditMode = () => modeWidget.value === MODE_EDIT;

    // 画布渲染读 widget.hidden，Vue 渲染读 options.hidden
    const setHidden = (w, hidden) => {
        w.hidden = hidden;
        if (w.options) w.options.hidden = hidden;
    };

    // 只同步高度，保持当前宽度
    const syncHeight = () => {
        const size = node.computeSize?.();
        if (!Array.isArray(size)) return;
        node.size[1] = size[1];
        node.setDirtyCanvas?.(true, true);
    };

    // 多选一：只保留一个开启的开关，并把非布尔值纠正为 false
    const enforceSingle = () => {
        const picks = pickWidgets();
        for (const w of picks) if (typeof w.value !== "boolean") w.value = false;
        const opened = picks.filter((w) => w.value);
        for (let i = 1; i < opened.length; i++) opened[i].value = false;
    };

    // 当前开启项在已命名选项中的位置，未开启任何开关时为 -1
    const selectedIndex = () => {
        let index = 0;
        for (const optionWidget of optionWidgets()) {
            if (!`${optionWidget.value ?? ""}`) continue;
            if (pickOf(optionWidget)?.value) return index;
            index++;
        }
        return -1;
    };

    // 当前开启项的名称，未开启任何开关时为空字符串
    const selectedText = () => {
        for (const optionWidget of optionWidgets()) {
            const name = `${optionWidget.value ?? ""}`;
            if (!name) continue;
            if (pickOf(optionWidget)?.value) return name;
        }
        return "";
    };

    // 刷新模式可见性、名称框与开关的显示标签，并同步高度
    const refresh = () => {
        const edit = isEditMode();
        for (const w of widgets) {
            if (w.name.startsWith(OPTION_PREFIX)) {
                w.label = `${OPTION_LABEL_PREFIX}${w.name.slice(OPTION_PREFIX.length)}`;
                setHidden(w, !edit);
            } else if (w.name.startsWith(PICK_PREFIX)) {
                const label = nameOf(w);
                w.label = label || w.name;
                setHidden(w, edit || !label);
            }
        }
        enforceSingle();
        syncHeight();
    };

    // 追加一行「名称框 + 开关」
    function addOption() {
        if (!node.widgets) return;
        const seq = optionWidgets().length + 1;
        const optionWidget = node.addWidget("string", `${OPTION_PREFIX}${seq}`, "", () => {});
        if (!optionWidget) return;
        let localName = `${optionWidget.value ?? ""}`;

        Object.defineProperty(optionWidget, "value", {
            get() {
                return localName;
            },
            set(v) {
                localName = v;
                refresh();
                if (!node.widgets) return;
                const options = optionWidgets();
                const lastOption = options.at(-1);
                if (lastOption === this) {
                    // 末行输入内容时追加新行
                    if (v) addOption();
                    return;
                }
                // 清空倒数第二行且末行为空时，收起末尾一行
                if (v || options.at(-2) !== this || lastOption?.value) return;
                const lastPick = pickOf(lastOption);
                widgets.splice(widgets.indexOf(lastOption), 1);
                if (lastPick) widgets.splice(widgets.indexOf(lastPick), 1);
                syncHeight();
                this.callback(v);
            },
        });

        const pickWidget = node.addWidget(
            "toggle",
            `${PICK_PREFIX}${seq}`,
            false,
            (v) => {
                // 互斥：开启本开关时关闭其余开关
                if (v) {
                    for (const w of pickWidgets()) if (w !== pickWidget) w.value = false;
                }
                refresh();
            },
            { on: "Yes", off: "No" }
        );
        if (pickWidget) refresh();
    }

    // 隐藏的后端输入部件：取值随选项状态变化（索引 = 位置，文本 = 名称）
    const managedInputs = {
        [INDEX_WIDGET_NAME]: selectedIndex,
        [TEXT_WIDGET_NAME]: selectedText,
    };
    for (const [name, getValue] of Object.entries(managedInputs)) {
        const widget = widgets.find((w) => w.name === name);
        if (!widget) continue;
        Object.defineProperty(widget, "value", { get: getValue, set(_) {} });
        Object.defineProperty(widget, "hidden", { get: () => true, set: () => {} });
        Object.defineProperty(widget, "type", { get: () => "hidden", set: () => {} });
        if (!widget.options) widget.options = {};
        widget.options.hidden = true;
        widget.computeSize = () => [0, -4];
        widget.draw = () => undefined;
    }

    // V3 (Nodes 2.0) 下被隐藏的参数会显示为空端口占位并拉长节点，需主动移除这些端口；
    // 以输入端口名签名做变化检测，端口列表未变时跳过，避免每帧调用 removeInput 触发重排。
    const isV3Frontend = () => {
        if (node._yuanComboIsV3 !== undefined) return node._yuanComboIsV3;
        let isV3 = false;
        try {
            if (document.querySelector("comfy-node")) isV3 = true;
        } catch (_) {}
        if (!isV3) {
            let el = widgets.find((w) => w.element)?.element;
            while (el) {
                if ((el.tagName && el.tagName.toLowerCase().includes("comfy-node")) ||
                    (el.classList && el.classList.contains("comfy-node"))) {
                    isV3 = true;
                    break;
                }
                el = el.parentElement || (el.getRootNode ? el.getRootNode().host : null);
            }
        }
        if (isV3) node._yuanComboIsV3 = true;
        return isV3;
    };

    const pruneManagedPorts = () => {
        if (!isV3Frontend() || !Array.isArray(node.inputs)) return;
        const sig = node.inputs.map((i) => i.name).join("|");
        if (sig === node._yuanComboPortSig) return;
        for (const name of Object.keys(managedInputs)) {
            const slot = node.findInputSlot?.(name);
            if (slot >= 0 && !node.inputs[slot].link) {
                node.removeInput(slot);
            }
        }
        node._yuanComboPortSig = node.inputs.map((i) => i.name).join("|");
    };

    node.onDrawForeground = chainCallback(node.onDrawForeground, function () {
        pruneManagedPorts();
    });

    // 切换模式时刷新可见性
    modeWidget.callback = chainCallback(modeWidget.callback, () => refresh());

    // 工作流加载完成（值恢复后）刷新一次
    node.onAdded = chainCallback(node.onAdded, function () {
        refresh();
    });

    addOption();
    // 创建时定一次尺寸，之后只同步高度
    node.computeSize?.(node.size);
}

app.registerExtension({
    name: "Comfy.Yuan.CustomCombo",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== YUAN_CUSTOM_COMBO_TYPE) return;
        nodeType.prototype.onNodeCreated = chainCallback(
            nodeType.prototype.onNodeCreated,
            onYuanCustomComboCreated
        );
    },
});
