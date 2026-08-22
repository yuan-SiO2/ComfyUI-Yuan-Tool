// 「选项」节点（YuanPrimitive）：基于原生 PrimitiveNode 的隔离实现，
// COMBO 输入展开为 Yes/No 互斥开关并写回目标节点，非 COMBO 输入保持原生行为；标题/端口名汉化。
const { app } = window.comfyAPI.app;

const YUAN_PRIMITIVE_TYPE = "YuanPrimitive";
const YUAN_PRIMITIVE_TITLE = "选项";
const YUAN_PRIMITIVE_OUTPUT_NAME = "连接到选项输入";
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
            console.warn("[YuanTool] 未获取到 LiteGraph，'选项' 节点未注册。");
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
                    if (this.outputs?.[0]) this.outputs[0].name = n;
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

            // 把 COMBO 选项展开为一组互斥 toggle 开关
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
                // 静态 description：前端为纯前端注册节点生成节点定义时读取
                // （updateVueAppNodeDefs: description = r.description ?? "Frontend only node for ..."）
                description: YUAN_PRIMITIVE_DESCRIPTION,
            })
        );
        YuanPrimitive.category = "Yuan Tool/选项";
        if (!PrimitiveNode) {
            console.warn("[YuanTool] 未获取到原生 PrimitiveNode，'选项' 节点已用内置基类注册（功能受限：连接后请手动点击开关写入）。");
        }
    },

    // 汉化 V3 节点库/搜索：前端为纯前端注册节点生成的定义默认
    // display_name=类型名（"YuanPrimitive"），注册进 Vue 应用前改写为中文
    beforeRegisterVueAppNodeDefs(defs) {
        if (!Array.isArray(defs)) return;
        const def = defs.find((d) => d && d.name === YUAN_PRIMITIVE_TYPE);
        if (def) {
            def.display_name = YUAN_PRIMITIVE_TITLE;
            def.description = YUAN_PRIMITIVE_DESCRIPTION;
        }
    },
});
