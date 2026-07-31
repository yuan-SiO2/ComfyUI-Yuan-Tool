import { app } from "../../../scripts/app.js";

/**
 * 文本处理 - 修复版
 * 确保 widget 名称与后端汉化后的名称 ("输入端口", "输出段落") 严格匹配。
 */
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

            // 4. 更新输入端口 (any_X) - 匹配汉化名称 "输入端口"
            nodeType.prototype.updateInputPorts = function(doResize = false) {
                if (!this.widgets) return;
                
                const inputCountWidget = this.widgets.find(w => w.name === "输入端口");
                if (!inputCountWidget) return;

                const updateFn = () => {
                    const targetCount = Math.max(1, inputCountWidget.value);
                    this.inputs = this.inputs || [];
                    
                    // 获取所有动态输入 (any_X)
                    let dynamicInputs = this.inputs.filter(i => i.name.startsWith("any_"));
                    let currentCount = dynamicInputs.length;

                    if (targetCount > currentCount) {
                        // 增加端口
                        for (let i = currentCount + 1; i <= targetCount; i++) {
                            this.addInput("any_" + i, "*");
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

            // 5. 更新输出端口 (段落X) - 匹配汉化名称 "输出段落"
            nodeType.prototype.updateOutputPorts = function(doResize = false) {
                if (!this.widgets) return;
                
                const outputCountWidget = this.widgets.find(w => w.name === "输出段落");
                if (!outputCountWidget) return;

                const updateFn = () => {
                    const targetCount = Math.max(0, outputCountWidget.value);
                    this.outputs = this.outputs || [];
                    
                    // 确保基础输出始终存在 (汉化)
                    if (this.outputs.length < 1) this.addOutput("数", "INT");
                    if (this.outputs.length < 2) this.addOutput("总段", "STRING");

                    // 获取所有动态输出 (段落X)
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