// 「选项」节点（YuanPrimitive）：COMBO 输入展开为 Yes/No 互斥开关写回目标节点，非 COMBO 保持原生行为。
const { app } = window.comfyAPI.app;
import { setSlotName } from "./Yuan_Common.js";


const YUAN_PRIMITIVE_TYPE = "YuanPrimitive";
const YUAN_PRIMITIVE_TITLE = "选项";
const YUAN_PRIMITIVE_CATEGORY = "Yuan Tool/选项";
const YUAN_PRIMITIVE_OUTPUT_NAME = "连接到选项输入";
// 节点来源归属：前端给纯前端节点生成的 def 写死 python_module="custom_nodes.frontend_only"，
// 节点库的来源徽标/按模块分组就会显示 frontend_only。这里取本扩展自身 URL 里的包名
// （/extensions/<包名>/...），与后端节点的 "custom_nodes.<包名>" 保持一致。
const YUAN_PRIMITIVE_MODULE = "custom_nodes." + (
    (import.meta.url.match(/\/extensions\/([^/]+)\//) || [])[1] || "ComfyUI-Yuan-Tool"
);
const YUAN_PRIMITIVE_DESCRIPTION =
    "把输出端口连接到任意下拉选项输入端口（如「MiniMax-H3 视频生成」的「模式」），" +
    "即自动读回该输入的全部选项并展开为一组互斥开关（Yes/No），点选开关即可切换取值并写回目标节点。" +
    "非下拉输入（数值/文本等）保持原生 Primitive 行为。本节点为 Yuan Tool 前端节点，与官方 Primitive 相互独立。";

app.registerExtension({
    name: "Comfy.Yuan.Primitive.Options",
    registerCustomNodes() {
        const LiteGraph = window.LiteGraph;
        const { PrimitiveNode } = window.comfyAPI?.widgetInputs ?? {};
        if (!LiteGraph) {
            return;
        }

        // 原生类缺失时的最小基类（不依赖 comfyAPI.widgetInputs）
        const Base = PrimitiveNode ?? class {
            constructor() {
                this.outputs = [];
                this.widgets = null;
                this.serialize_widgets = true;
                this.isVirtualNode = true;
            }
            addOutput(name, type) {
                this.outputs.push({ name, type, links: null });
                return this.outputs.length - 1;
            }
            applyToGraph() {}
        };

        class YuanPrimitive extends Base {
            constructor(title) {
                super(title);
                // 强制实例标题，防止被父类构造改回 "Primitive"
                this.title = YUAN_PRIMITIVE_TITLE;
                this.serialize_widgets = true;
                this.isVirtualNode = true;
                if (!this.outputs?.length) this.addOutput(YUAN_PRIMITIVE_OUTPUT_NAME, "*");
            }

            // 从任意 spec 形态中提取 COMBO 选项数组；非 COMBO 返回 null
            _extractComboOptions(e) {
                if (!Array.isArray(e)) return null;
                // 形态1：[["选项A","选项B"], {...}]
                if (Array.isArray(e[0])) return e[0];
                // 形态2：["COMBO", {options: [...]}]
                const head = String(e[0] ?? "").toUpperCase();
                const opts = e[1]?.options;
                if (Array.isArray(opts) && (head === "COMBO" || head === "")) return opts;
                // 形态3：["STRING"|其他类型, {options: [...]}] —— 带 options 视为组合
                if (Array.isArray(opts)) return opts;
                return null;
            }

            // 覆盖原生 _createWidget：COMBO 输入 → 互斥开关按钮，绝不回落原生
            _createWidget(e, t, n, r) {
                const options = this._extractComboOptions(e);
                if (options) {
                    // 端口标签改为所连输入名，避免显示成 COMBO
                    if (this.outputs?.[0]) setSlotName(this.outputs[0], n);
                    const [w, h] = this.size;
                    const s = this._createComboToggles(options, t, n);
                    if (typeof this._finalizeWidget === "function") {
                        this._finalizeWidget(s, w, h, r);
                    } else {
                        // 内置基类无原生私有方法，手动恢复尺寸
                        this.size = [w, h];
                        this.setSize?.(this.computeSize?.() ?? this.size);
                        if (r) this.applyToGraph();
                    }
                    return;
                }
                // 非 COMBO（INT/FLOAT/STRING/BOOLEAN…）：仍走原生路径
                if (super._createWidget) return super._createWidget(e, t, n, r);
            }

            _createComboToggles(options, targetNode, inputName) {
                // 目标输入当前的选中值，作为默认打开的开关
                let current = null;
                if (targetNode?.widgets) {
                    const tw = targetNode.widgets.find((w) => w.name === inputName);
                    if (tw) current = tw.value;
                }
                let first = null;
                let anyOn = false;
                for (const opt of options) {
                    const label = String(opt);
                    const isOn =
                        label === String(current) ||
                        (Array.isArray(current) && current.map(String).includes(label));
                    if (isOn) anyOn = true;
                    const toggle = this.addWidget("toggle", label, isOn, (v) => {
                        if (v) {
                            // 互斥：打开当前开关时关闭其余所有开关
                            for (const w of this.widgets) {
                                if (w !== toggle && w.type === "toggle" && w.value) {
                                    w.value = false;
                                }
                            }
                        }
                        this.applyToGraph();
                    });
                    toggle.options = toggle.options || {};
                    toggle.options.on = "Yes";
                    toggle.options.off = "No";
                    if (!first) first = toggle;
                }
                // 当前值不在选项中时，默认打开第一个开关，保证始终有选中值（与原生 combo 默认取首个选项一致）
                if (first && !anyOn && options.length) first.value = true;
                // 选项为空时保留一个占位 widget（applyToGraph 依赖 widgets[0] 存在）
                if (!first) {
                    first = this.addWidget("toggle", "", false, () => {});
                }
                return first;
            }

            // 当前打开的开关所对应的选项
            _getSelectedOption() {
                if (!this.widgets) return null;
                for (const w of this.widgets) {
                    if (w.type === "toggle" && w.value) return w.name;
                }
                return null;
            }

            // 把选中值写入目标 widget 的内置兜底，
            // 兼容 V2 links 数组与 V3 links Map 两种容器
            _applyValueFallback(selected) {
                const graph = this.graph;
                if (!graph) return;
                const out = this.outputs?.[0];
                if (!out) return;
                const linkIds = [...(out.links ?? [])];
                for (const id of linkIds) {
                    const link = graph.links?.[id] ?? graph._links?.get?.(id);
                    if (!link) continue;
                    const target = graph.getNodeById?.(link.target_id);
                    if (!target) continue;
                    const inputName = target.inputs?.[link.target_slot]?.name;
                    const w = target.widgets?.find((w) => w.name === inputName);
                    if (!w) continue;
                    if (String(w.value) !== String(selected)) {
                        w.value = selected;
                        if (typeof w.callback === "function") w.callback(selected);
                    }
                }
            }

            // 覆盖原生 applyToGraph：把打开的开关对应的选项写回目标输入
            applyToGraph(links = []) {
                if (!this.graph) return;
                const selected = this._getSelectedOption();
                if (selected == null) return;
                const apply = window.comfyAPI?.widgetValuePropagation
                    ?.applyFirstWidgetValueToGraph;
                if (typeof apply === "function") {
                    apply(this, links, () => selected);
                } else {
                    // API 缺失时兜底，保证开关点击始终生效
                    this._applyValueFallback(selected);
                    for (const l of links) {
                        const link = typeof l === "number"
                            ? (this.graph.links?.[l] ?? this.graph._links?.get?.(l))
                            : l;
                        if (!link) continue;
                        const target = this.graph.getNodeById?.(link.target_id);
                        const inputName = target?.inputs?.[link.target_slot]?.name;
                        const w = target?.widgets?.find((w) => w.name === inputName);
                        if (w && String(w.value) !== String(selected)) {
                            w.value = selected;
                            if (typeof w.callback === "function") w.callback(selected);
                        }
                    }
                }
            }
        }

        // 重复注册（前端热重载）仅替换，无副作用
        LiteGraph.registerNodeType(
            YUAN_PRIMITIVE_TYPE,
            Object.assign(YuanPrimitive, {
                title: YUAN_PRIMITIVE_TITLE,
                // 静态 description：纯前端注册节点生成定义时读取
                // （updateVueAppNodeDefs: description = r.description ?? "Frontend only node for ..."）
                description: YUAN_PRIMITIVE_DESCRIPTION,
            })
        );
        YuanPrimitive.category = YUAN_PRIMITIVE_CATEGORY;
    },

    // 纯前端节点的定义由前端补全：display_name/description/python_module 都是默认值，
    // 注册进 Vue 应用前统一改成本插件的中文名与归属，避免节点库把它算成 frontend_only。
    beforeRegisterVueAppNodeDefs(defs) {
        if (!Array.isArray(defs)) return;
        const def = defs.find((d) => d && d.name === YUAN_PRIMITIVE_TYPE);
        if (!def) return;
        def.display_name = YUAN_PRIMITIVE_TITLE;
        def.description = YUAN_PRIMITIVE_DESCRIPTION;
        def.category = YUAN_PRIMITIVE_CATEGORY;
        def.python_module = YUAN_PRIMITIVE_MODULE;
    },
});
