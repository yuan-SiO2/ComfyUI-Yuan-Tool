// ==================== 复刻原生 Primitive 节点 → Yuan Tool/选项 ====================
// 子类继承原生 PrimitiveNode 的全部行为，仅改类型名与分类；
// 对 COMBO 输入：删除原生「生成后控制」控件，将全部选项展开为互斥开关按钮（Yes/No），
// 点击后写回目标输入；非 COMBO 输入仍走原生行为。
const { app } = window.comfyAPI.app;

app.registerExtension({
    name: "Comfy.Yuan.Primitive.Options",
    registerCustomNodes() {
        const LiteGraph = window.LiteGraph;
        const { PrimitiveNode } = window.comfyAPI.widgetInputs || {};
        if (!LiteGraph || !PrimitiveNode) {
            console.warn("[YuanTool] 未获取到原生 PrimitiveNode（window.comfyAPI.widgetInputs.PrimitiveNode），'选项' 节点未注册。");
            return;
        }

        class YuanPrimitive extends PrimitiveNode {
            // 覆盖原生 _createWidget：COMBO 输入 → 互斥开关按钮。
            // 兼容两种配置形态（核心 _onFirstConnection 传入 a[j] ?? o）：
            //   1) 原生定义形态 e = [["选项A","选项B",...], {...}]，e[0] 为选项数组
            //   2) 合并后的形态 e = ["COMBO", {options:[...]}], e[0] 为字符串 "COMBO"
            //      （该形态由 mergeIfValid 写入 input.widget[j]，j 为符号键不随工作流保存，
            //        连接时传入会导致回退到原生 Primitive 的 COMBO 下拉样式）
            _createWidget(e, t, n, r) {
                const isCombo = Array.isArray(e?.[0]) || e?.[0] === "COMBO";
                const options = Array.isArray(e?.[0])
                    ? e[0]
                    : Array.isArray(e?.[1]?.options)
                        ? e[1].options
                        : null;
                if (isCombo && options) {
                    // 端口标签改为所连输入名，避免显示成 COMBO
                    if (this.outputs?.[0]) this.outputs[0].name = n;
                    const [w, h] = this.size;
                    const s = this._createComboToggles(options, t, n);
                    this._finalizeWidget(s, w, h, r);
                    return;
                }
                return super._createWidget(e, t, n, r);
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

            // 覆盖原生 applyToGraph：把打开的开关对应的选项写回目标输入
            applyToGraph(links = []) {
                if (!this.graph) return;
                const apply = window.comfyAPI?.widgetValuePropagation?.applyFirstWidgetValueToGraph;
                if (!apply) return;
                const selected = this._getSelectedOption();
                if (selected == null) return;
                apply(this, links, () => selected);
            }
        }

        LiteGraph.registerNodeType(
            "YuanPrimitive",
            Object.assign(YuanPrimitive, { title: "选项 (Primitive)" })
        );
        YuanPrimitive.category = "Yuan Tool/选项";
    },
});
