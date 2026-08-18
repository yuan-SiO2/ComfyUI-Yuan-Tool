// ==================== 复刻原生 Primitive 节点 → Yuan Tool/选项 ====================
// 在原生 PrimitiveNode 之上做完整隔离，适配各整合包前端版本（含 1.48.x）：
//   隔离1 注册名/标题：type="YuanPrimitive"，实例 title 强制设为 Yuan 的标题，
//          防止原生实例字段把标题覆盖回 "Primitive"、或被 V3 的
//          type==="PrimitiveNode" 判定误收进原生 primitive 集合；
//   隔离2 COMBO 判定增强：兼容全部 spec 形态（数组式 / "COMBO" 合并式 /
//          options 对象式），COMBO 一律生成 Yes/No 互斥开关，绝不回落
//          super._createWidget（那会退化为原生下拉+「生成后控制」样式）；
//   隔离3 写回兜底：comfyAPI.widgetValuePropagation.applyFirstWidgetValueToGraph
//          缺失或签名变化时，用内置实现直接把选中值写入目标 widget，
//          兼容 V2(graph.links[id]) 与 V3(graph._links.get(id)) 两种链接容器；
//   隔离4 原生类缺失兜底：老整合包无 window.comfyAPI.widgetInputs.PrimitiveNode
//          时注册最小基类，节点仍可用（仅失去原生生命周期，需手动连一次线）。
// 非 COMBO 输入仍走原生行为。
const { app } = window.comfyAPI.app;

const YUAN_PRIMITIVE_TYPE = "YuanPrimitive";
const YUAN_PRIMITIVE_TITLE = "选项 (Primitive)";

app.registerExtension({
    name: "Comfy.Yuan.Primitive.Options",
    registerCustomNodes() {
        const LiteGraph = window.LiteGraph;
        const { PrimitiveNode } = window.comfyAPI?.widgetInputs ?? {};
        if (!LiteGraph) {
            console.warn("[YuanTool] 未获取到 LiteGraph，'选项' 节点未注册。");
            return;
        }

        // 隔离4：原生类缺失时的最小基类（不依赖 comfyAPI.widgetInputs）
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
                // 隔离1：强制实例标题，防原生实例字段/父类构造把标题改回 "Primitive"
                this.title = YUAN_PRIMITIVE_TITLE;
                this.serialize_widgets = true;
                this.isVirtualNode = true;
                if (!this.outputs?.length) this.addOutput("connect to widget input", "*");
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
                        // 隔离4兜底：内置基类无原生私有方法，手动恢复尺寸
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

            // 隔离3：把选中值写入目标 widget 的内置兜底，
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
                    // 隔离3：API 缺失时兜底，保证开关点击始终生效
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

        // 与官方一致的注册姿势；重复注册（前端热重载）仅替换，无副作用
        LiteGraph.registerNodeType(
            YUAN_PRIMITIVE_TYPE,
            Object.assign(YuanPrimitive, { title: YUAN_PRIMITIVE_TITLE })
        );
        YuanPrimitive.category = "Yuan Tool/选项";
        if (!PrimitiveNode) {
            console.warn("[YuanTool] 未获取到原生 PrimitiveNode，'选项' 节点已用内置基类注册（功能受限：连接后请手动点击开关写入）。");
        }
    },
});
