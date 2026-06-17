const { app } = window.comfyAPI.app;

app.registerExtension({
    name: "ComfyUI-Yuan-Tool",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "YuanTool") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const singleNames = ["1", "2", "3", "4"];
            const listName = "image_list";
            const imageType = "IMAGE";
            const bgName = "background";

            const self = this;

            const ensureBackgroundSolid = () => {
                const bgInput = self.inputs.find(inp => inp.name === bgName);
                if (bgInput) {
                    bgInput.removable = false;
                    bgInput.optional = false;
                }
            };

            const removeInputAndWidget = (name) => {
                const idx = self.inputs.findIndex(inp => inp.name === name);
                if (idx !== -1) self.removeInput(idx);
                const wIdx = self.widgets ? self.widgets.findIndex(w => w.name === name) : -1;
                if (wIdx !== -1) self.widgets.splice(wIdx, 1);
            };

            const addOptionalImageInput = (name) => {
                self.addInput(name, imageType, { shape: 7, optional: true });
                const inp = self.inputs.find(inp => inp.name === name);
                if (inp) inp.optional = true;
            };

            const syncPorts = (mode) => {
                const desiredNames = mode ? [listName] : [...singleNames];
                const allOptional = [...singleNames, listName];

                // 第一步：保存所有端口的连接信息（从 graph 层面，防止被 ComfyUI 内部清除）
                const savedLinks = [];
                const allPortNames = [...allOptional, bgName];
                for (let i = self.inputs.length - 1; i >= 0; i--) {
                    const inp = self.inputs[i];
                    const name = inp.name;
                    if (allPortNames.includes(name)) {
                        if (inp.link != null) {
                            const linkObj = app.graph.links[inp.link];
                            if (linkObj) {
                                savedLinks.push({
                                    name: name,
                                    origin_id: linkObj.origin_id,
                                    origin_slot: linkObj.origin_slot,
                                    type: linkObj.type,
                                });
                            }
                        }
                    }
                }

                // 第二步：删除不需要的可选端口
                for (let i = self.inputs.length - 1; i >= 0; i--) {
                    const inp = self.inputs[i];
                    const name = inp.name;
                    if (allOptional.includes(name) && !desiredNames.includes(name)) {
                        removeInputAndWidget(name);
                    }
                }

                // 第三步：添加缺少的可选端口
                for (const name of desiredNames) {
                    if (!self.inputs.find(inp => inp.name === name)) {
                        addOptionalImageInput(name);
                    }
                }

                // 第四步：确保背景端口存在且在最后
                const bgIdx = self.inputs.findIndex(inp => inp.name === bgName);
                if (bgIdx !== -1) {
                    if (bgIdx !== self.inputs.length - 1) {
                        self.removeInput(bgIdx);
                        self.addInput(bgName, imageType);
                    }
                } else {
                    self.addInput(bgName, imageType);
                }

                // 第五步：恢复所有保存的连接
                for (const saved of savedLinks) {
                    const inp = self.inputs.find(inp => inp.name === saved.name);
                    if (inp && inp.link == null) {
                        const originNode = app.graph.getNodeById(saved.origin_id);
                        if (originNode) {
                            const targetSlot = self.inputs.indexOf(inp);
                            try {
                                originNode.connect(saved.origin_slot, self.id, targetSlot);
                            } catch (_) {}
                        }
                    }
                }

                ensureBackgroundSolid();

                // 保持当前宽度不变，只更新高度
                const currentWidth = self.size ? self.size[0] : self.computeSize()[0];
                self.setSize([currentWidth, self.computeSize()[1]]);
                app.graph.setDirtyCanvas(true, true);
            };

            self._syncPorts = syncPorts;

            ensureBackgroundSolid();

            const modeWidget = this.widgets.find(w => w.name === "list_mode");
            if (modeWidget) {
                syncPorts(!!modeWidget.value);

                const origCallback = modeWidget.callback;
                modeWidget.callback = function (value) {
                    if (origCallback) origCallback.apply(this, arguments);
                    self._syncPorts(!!value);
                };
            }

            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (this._syncPorts) {
                const w = this.widgets.find(w => w.name === "list_mode");
                if (w) this._syncPorts(!!w.value);
            }
            return r;
        };
    },
});
