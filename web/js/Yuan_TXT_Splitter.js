import { app } from "../../../scripts/app.js";

/** 文本处理节点：确保 widget 名称与后端汉化名称（"输入端口"/"输出段落"）严格匹配。 */
app.registerExtension({
    name: "YuanTool.TXTParagraphSplitter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "YUAN_TXTParagraphSplitter") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                // 1. 添加汉化按钮
                this.addWidget("button", "更新端口", null, () => {
                    this.updateInputPorts(true);
                    this.updateOutputPorts(true);
                });

                // 2. 初始尺寸与端口状态
                this.setSize([400, 300]);
                this.updateInputPorts(false);
                this.updateOutputPorts(false);
            };

            // 3. 尺寸调整辅助
            nodeType.prototype.resizeNode = function(updateFn) {
                const oldSize = this.computeSize()[1];
                updateFn();
                const newSize = this.computeSize()[1];
                if (this.size) {
                    this.setSize([this.size[0], this.size[1] + (newSize - oldSize)]);
                }
            };

            // 4. 更新动态输入端口 (any_X)，数量取「输入端口」widget 值
            nodeType.prototype.updateInputPorts = function(doResize = false) {
                if (!this.widgets) return;
                
                const inputCountWidget = this.widgets.find(w => w.name === "输入端口");
                if (!inputCountWidget) return;

                const updateFn = () => {
                    const targetCount = Math.max(1, inputCountWidget.value);
                    this.inputs = this.inputs || [];
                    let dynamicInputs = this.inputs.filter(i => i.name.startsWith("any_"));
                    let currentCount = dynamicInputs.length;

                    // 已有端口统一标 optional，避免实心圆点
                    for (const inp of dynamicInputs) {
                        if (!inp.extra) inp.extra = {};
                        inp.extra.optional = true;
                    }

                    if (targetCount > currentCount) {
                        // 新增端口同样 optional（对齐后端 INPUT_TYPES.optional）
                        for (let i = currentCount + 1; i <= targetCount; i++) {
                            this.addInput("any_" + i, "*", { optional: true });
                        }
                    } else if (targetCount < currentCount) {
                        // 减少端口 (从后往前删)
                        for (let i = this.inputs.length - 1; i >= 0; i--) {
                            const inputName = this.inputs[i].name;
                            if (inputName.startsWith("any_")) {
                                const idx = parseInt(inputName.split("_")[1]);
                                if (idx > targetCount) {
                                    this.removeInput(i);
                                }
                            }
                        }
                    }
                };

                if (doResize) {
                    this.resizeNode(updateFn);
                    if (this.setDirtyCanvas) {
                        this.setDirtyCanvas(true, true);
                    }
                } else {
                    updateFn();
                }
            };

            // 5. 更新动态输出端口 (段落X)，数量取「输出段落」widget 值
            nodeType.prototype.updateOutputPorts = function(doResize = false) {
                if (!this.widgets) return;
                
                const outputCountWidget = this.widgets.find(w => w.name === "输出段落");
                if (!outputCountWidget) return;

                const updateFn = () => {
                    const targetCount = Math.max(0, outputCountWidget.value);
                    this.outputs = this.outputs || [];
                    
                    // 基础输出始终存在
                    if (this.outputs.length < 1) this.addOutput("数", "INT");
                    if (this.outputs.length < 2) this.addOutput("总段", "STRING");
                    let dynamicOutputs = this.outputs.filter(o => o.name.startsWith("段落"));
                    let currentCount = dynamicOutputs.length;

                    if (targetCount > currentCount) {
                        // 增加端口
                        for (let i = currentCount + 1; i <= targetCount; i++) {
                            this.addOutput("段落" + i, "STRING");
                        }
                    } else if (targetCount < currentCount) {
                        // 减少端口 (从后往前删)
                        for (let i = this.outputs.length - 1; i >= 0; i--) {
                            const outputName = this.outputs[i].name;
                            if (outputName.startsWith("段落")) {
                                const idx = parseInt(outputName.replace("段落", ""));
                                if (idx > targetCount) {
                                    this.removeOutput(i);
                                }
                            }
                        }
                    }
                };

                if (doResize) {
                    this.resizeNode(updateFn);
                    if (this.setDirtyCanvas) {
                        this.setDirtyCanvas(true, true);
                    }
                } else {
                    updateFn();
                }
            };
        }
    }
});

// ==== 预览内容节点（Yuan-TV ShowText 复刻）：预览模式回填结果至「文本」框并只读，编辑模式直接输出 ====
const PREVIEW_MODE_WIDGET = "显示模式";
const PREVIEW_TEXT_WIDGET = "文本";

function applyReadonly(node) {
    const mode = node.widgets?.find((w) => w.name === PREVIEW_MODE_WIDGET);
    const txt = node.widgets?.find((w) => w.name === PREVIEW_TEXT_WIDGET);
    if (mode && txt && txt.inputEl) {
        txt.inputEl.readOnly = !!mode.value; // 预览只读，编辑可编辑
        txt.inputEl.style.opacity = mode.value ? 0.6 : 1;
    }
}

function setupPreview(node) {
    const mode = node.widgets?.find((w) => w.name === PREVIEW_MODE_WIDGET);
    applyReadonly(node);
    // 仅绑定一次回调，切换开关时更新只读状态
    if (mode && !mode.__yuantool_preview_bound) {
        mode.__yuantool_preview_bound = true;
        const old = mode.callback;
        mode.callback = function () {
            old?.apply(this, arguments);
            applyReadonly(node);
        };
    }
}

app.registerExtension({
    name: "YuanTool.TXT.PreviewContent",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "YUAN_TXTPreviewContent") return;

        // 创建后 widgets 已就绪即应用只读，避免预览模式下默认文本可编辑
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            setupPreview(this);
        };

        // 加载工作流配置后：按保存的模式恢复只读状态
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            setupPreview(this);
        };
        // 执行后：预览模式把结果写入「文本」框（上游为空文本时也要清空，否则会一直显示上一次的内容）
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            setupPreview(this);
            const mode = this.widgets?.find((w) => w.name === PREVIEW_MODE_WIDGET);
            const txt = this.widgets?.find((w) => w.name === PREVIEW_TEXT_WIDGET);
            if (!mode?.value || !txt) return;
            const value = Array.isArray(message?.text) ? String(message.text[0] ?? "") : "";
            if (txt.value !== value) {
                txt.value = value;
                app.graph.setDirtyCanvas(true, false);
            }
        };
    },
});