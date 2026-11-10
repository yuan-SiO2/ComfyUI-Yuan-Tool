/** JSON提取：按「模式」切换端口（分镜 8 个 / 档案 4 个）。
 *
 * 端口顺序与节点定义一致：类型校验按端口下标查 RETURN_TYPES，两种模式同一下标的类型必须相同。
 *   分镜模式：分镜类型(INT) / 分镜序列 / 角色索引 / 音色索引 / 道具索引 / 场景索引 / 索引时长(FLOAT) / 场景衔接(BOOLEAN)
 *   档案模式：前 4 个改名为 档案编码(INT) / 作品名 / 整体风格 / 档案，尾部 4 个删除
 * 载入工作流时按存档端口名迁移连线（含旧名 `场景上下文` → `场景衔接`）。
 */
import { chainCallback, hideWidget, remapSavedPorts, showWidget } from "./Yuan_Common.js";
const app = window.comfyAPI.app?.app ?? window.comfyAPI.app;
const NODE_NAME = "YUAN_TXTJsonExtractor";
const MODE_WIDGET = "模式";
const ARCHIVE_MODE = "档案模式";

const STORY_OUTPUTS = ["分镜类型", "分镜序列", "角色索引", "音色索引", "道具索引", "场景索引", "索引时长", "场景衔接"];
const ARCHIVE_OUTPUTS = ["档案编码", "作品名", "整体风格", "档案"];
const OUTPUT_TYPES = {
    分镜类型: "INT", 分镜序列: "STRING", 角色索引: "STRING", 音色索引: "STRING",
    道具索引: "STRING", 场景索引: "STRING", 索引时长: "FLOAT", 场景衔接: "BOOLEAN",
};
const OUTPUT_ALIAS = { 场景上下文: "场景衔接" };          // 旧端口名 → 现端口名
const HIDDEN_INPUTS = ["开关配置", "情节衔接文本"];        // 仅档案模式隐藏
const ARCHIVE_ONLY_WIDGETS = ["档案选择"];                // 仅档案模式显示

const sameOrder = (slots, wanted) => slots.length === wanted.length && wanted.every((n, i) => slots[i]?.name === n);

/** 按模式调整端口：删除尾部多余端口、补齐缺失端口、改名前 N 个端口。 */
function applyPorts(node, archive, dropLinks) {
    const wanted = archive ? ARCHIVE_OUTPUTS : STORY_OUTPUTS;
    if (sameOrder(node.outputs, wanted)) return;

    for (let i = node.outputs.length - 1; i >= wanted.length; i--) node.removeOutput(i);
    for (let i = node.outputs.length; i < wanted.length; i++) {
        node.addOutput(wanted[i], OUTPUT_TYPES[wanted[i]] || "*");
    }
    // 渲染用 label || localized_name || name，改名时三者一并更新
    for (let i = 0; i < wanted.length; i++) {
        const slot = node.outputs[i];
        if (!slot || slot.name === wanted[i]) continue;
        if (dropLinks) node.disconnectOutput?.(i);
        slot.name = wanted[i];
        slot.localized_name = undefined;
        slot.label = undefined;
    }
}

/** 按模式调整端口、输入可见性与控件显隐。 */
function applyMode(node, dropLinks) {
    if (!node?.outputs || !node?.inputs) return;
    const archive = node.widgets?.find((w) => w.name === MODE_WIDGET)?.value === ARCHIVE_MODE;
    applyPorts(node, archive, dropLinks);
    for (const name of HIDDEN_INPUTS) {
        const slot = node.findInputSlot?.(name) ?? -1;
        if (archive) {
            if (slot >= 0) node.removeInput(slot);
        } else if (slot < 0) {
            node.addInput(name, "*", { optional: true });
        }
    }
    for (const name of ARCHIVE_ONLY_WIDGETS) {
        const widget = node.widgets?.find((w) => w.name === name);
        if (widget) (archive ? showWidget : hideWidget)(widget);
    }
    const size = node.computeSize?.();
    if (Array.isArray(size)) node.setSize?.([node.size?.[0] || size[0], size[1]]);
}

app.registerExtension({
    name: "YuanTool.TXT.JsonExtractorMode",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const hasLinks = (node) => {
            for (const link of node.graph?.links?.values?.() ?? []) {
                if (link?.origin_id === node.id || link?.target_id === node.id) return true;
            }
            return false;
        };
        // 延后一拍执行；加载中的节点须等 configure 完成，否则端口名与存档不一致
        const schedule = (node, dropLinks = false) => {
            clearTimeout(node._yuanModeTimer);
            node._yuanModeTimer = setTimeout(() => {
                if (!node.graph) return;
                if (!node._yuanConfigured && hasLinks(node)) return;
                applyMode(node, dropLinks);
            }, 0);
        };

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            const widget = this.widgets?.find((w) => w.name === MODE_WIDGET);
            if (widget) chainCallback(widget, "callback", () => schedule(this, true));
            schedule(this);
        });
        // 下拉框也可能走 onWidgetChanged
        chainCallback(nodeType.prototype, "onWidgetChanged", function (name) {
            if (name === MODE_WIDGET) schedule(this, true);
        });

        // configure 的入参是存档原文（端口名完整），onConfigure 的入参已按定义重排
        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            if (info?.outputs || info?.inputs) {
                this._yuanSavedPorts = { outputs: info.outputs, inputs: info.inputs };
            }
            return origConfigure ? origConfigure.apply(this, arguments) : undefined;
        };
        chainCallback(nodeType.prototype, "onConfigure", function (info) {
            this._yuanConfigured = true;
            const saved = this._yuanSavedPorts ?? info;
            applyPorts(this, this.widgets?.find((w) => w.name === MODE_WIDGET)?.value === ARCHIVE_MODE, false);
            remapSavedPorts(this, saved, OUTPUT_ALIAS);
            schedule(this);
        });
    },
});
