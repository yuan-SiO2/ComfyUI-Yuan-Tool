const { app } = window.comfyAPI.app;

// ==================== YuanTool（多帧参考节点，list_mode 动态端口）====================
function registerYuanTool(nodeType, portMeta) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

        const singleNames = ["1", "2", "3", "4", "5", "6", "7", "8"];
        const listNames = ["image_list_1", "image_list_2", "image_list_3", "image_list_4",
                           "image_list_5", "image_list_6", "image_list_7", "image_list_8"];
        const imageType = "IMAGE";
        const bgName = "background";

        const self = this;

        const removeInputAndWidget = (name) => {
            const idx = self.inputs.findIndex(inp => inp.name === name);
            if (idx !== -1) self.removeInput(idx);
            const wIdx = self.widgets ? self.widgets.findIndex(w => w.name === name) : -1;
            if (wIdx !== -1) self.widgets.splice(wIdx, 1);
        };

        const addOptionalImageInput = (name) => {
            const meta = portMeta[name] || {};
            const opts = { shape: 7, optional: true };
            if (meta.display_name) opts.display_name = meta.display_name;
            if (meta.tooltip) opts.tooltip = meta.tooltip;
            self.addInput(name, imageType, opts);
            const inp = self.inputs.find(inp => inp.name === name);
            if (inp) {
                inp.optional = true;
                if (meta.display_name) inp.label = meta.display_name;
                if (meta.tooltip) inp.tooltip = meta.tooltip;
            }
        };

        const isConnected = (name) => {
            const inp = self.inputs.find(inp => inp.name === name);
            return !!(inp && inp.link != null);
        };

        // 单帧端口可见性：1、2 总是显示；第 k 个（k>=3）在前面所有端口都已连接或自身已连接时显示
        const singleVisible = (idx) => {
            if (idx < 2) return true;
            if (isConnected(singleNames[idx])) return true;
            for (let i = 0; i < idx; i++) {
                if (!isConnected(singleNames[i])) return false;
            }
            return true;
        };

        // 列表端口可见性：1、2 总是显示；第 k 个（k>=3）在前 k-1 个列表端口全部连接或自身已连接时显示
        const listVisible = (idx) => {
            if (idx < 2) return true;
            if (isConnected(listNames[idx])) return true;
            for (let i = 0; i < idx; i++) {
                if (!isConnected(listNames[i])) return false;
            }
            return true;
        };

        const syncPortsReal = (mode) => {
            if (self._syncing) return;
            self._syncing = true;
            // 期望端口集合（有序）：
            // - 列表模式：按 listVisible 递进显示 image_list_1..8
            // - 单帧模式：按 singleVisible 递进显示 1..8
            const desiredNames = [];
            if (mode) {
                for (let i = 0; i < listNames.length; i++) {
                    if (listVisible(i)) desiredNames.push(listNames[i]);
                }
            } else {
                for (let i = 0; i < singleNames.length; i++) {
                    if (singleVisible(i)) desiredNames.push(singleNames[i]);
                }
            }
            const allOptional = [...singleNames, ...listNames];

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

            // 第二步：删除不需要的可选端口（含已弃用的老 image_list 名称，兼容旧工作流加载）
            for (let i = self.inputs.length - 1; i >= 0; i--) {
                const inp = self.inputs[i];
                const name = inp.name;
                const isOptionalOld = (allOptional.includes(name) || name === "image_list");
                if (isOptionalOld && !desiredNames.includes(name)) {
                    removeInputAndWidget(name);
                }
            }

            // 第三步：添加缺少的可选端口（按 desiredNames 顺序，保证排列整洁）
            for (const name of desiredNames) {
                if (!self.inputs.find(inp => inp.name === name)) {
                    addOptionalImageInput(name);
                }
            }

            // 第四步：确保背景端口存在且在最后（可选端口，空心圆点）
            // 注意：不能使用 removeInput 删除背景端口来调整顺序，
            // removeInput 会调用 disconnectInput 把已连接的线从 graph 中删掉，
            // 而切换工作流/切换 list_mode 时第 5 步的重连又会因节点 id 为字符串而静默失败，
            // 导致 background 端口的线丢失。这里改为在数组层面移动端口，
            // 保留端口上的连接，并同步修正 graph 中相关 link 的 target_slot。
            const bgIdx = self.inputs.findIndex(inp => inp.name === bgName);
            if (bgIdx !== -1) {
                const bgInput = self.inputs[bgIdx];
                bgInput.optional = true;
                bgInput.removable = true;
                bgInput.shape = 7;

                if (bgIdx !== self.inputs.length - 1) {
                    // 从数组中移除背景端口（不删除连接）
                    self.inputs.splice(bgIdx, 1);
                    // 移除后，bgIdx 及其之后端口的下标都前移了 1，同步修正对应 link 的 target_slot
                    for (let i = bgIdx; i < self.inputs.length; i++) {
                        const inp = self.inputs[i];
                        if (inp && inp.link != null) {
                            const l = app.graph.links[inp.link];
                            if (l) l.target_slot--;
                        }
                    }
                    // 追加到末尾
                    self.inputs.push(bgInput);
                    if (bgInput.link != null) {
                        const l = app.graph.links[bgInput.link];
                        if (l) l.target_slot = self.inputs.length - 1;
                    }
                }
            } else {
                addOptionalImageInput(bgName);
            }

            // 第五步：恢复所有保存的连接
            for (const saved of savedLinks) {
                const inp = self.inputs.find(inp => inp.name === saved.name);
                if (inp && inp.link == null) {
                    const originNode = app.graph.getNodeById(saved.origin_id);
                    if (originNode) {
                        const targetSlot = self.inputs.indexOf(inp);
                        try {
                            // 传节点对象而不是 self.id：
                            // 新版 ComfyUI 的节点 id 是字符串（toNodeId 返回 String），
                            // connect 只会解析 number 类型的 id，传 self.id 会静默返回 null 导致重连失败
                            originNode.connect(saved.origin_slot, self, targetSlot);
                        } catch (_) {}
                    }
                }
            }

            self._syncing = false;

            // 保持当前宽度不变，只更新高度
            const currentWidth = self.size ? self.size[0] : self.computeSize()[0];
            self.setSize([currentWidth, self.computeSize()[1]]);
            app.graph.setDirtyCanvas(true, true);
        };

        // 把 syncPortsReal 调度到下一 tick，确保 graph.links 已完整恢复。
        // 在 onNodeCreated / onConfigure / mode.callback / onConnectionsChange
        // 等生命周期中，links 可能尚未挂到 input.link 上，立即执行会
        // 误判递进端口、重排 slot 错位、savedLinks 为空，导致切换工作流时断开。
        const scheduleSync = (mode) => {
            clearTimeout(self._syncTimer);
            self._syncTimer = setTimeout(() => syncPortsReal(mode), 0);
        };

        self._syncPorts = scheduleSync;

        const modeWidget = this.widgets.find(w => w.name === "list_mode");
        if (modeWidget) {
            scheduleSync(!!modeWidget.value);

            const origCallback = modeWidget.callback;
            modeWidget.callback = function (value) {
                if (origCallback) origCallback.apply(this, arguments);
                scheduleSync(!!value);
            };
        }

        // 修正初始已创建端口的 label / tooltip（框架自动生成的端口可能未正确应用 display_name）
        for (const inp of self.inputs) {
            const meta = portMeta[inp.name];
            if (meta) {
                if (meta.display_name) inp.label = meta.display_name;
                if (meta.tooltip) inp.tooltip = meta.tooltip;
            }
        }

        return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, slot, connected, link_info, input_or_output) {
        const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
        // 输入端口（type===1）连接变化时刷新递进显示；_syncing 期间跳过避免递归
        if (!this._syncing && this._syncPorts && type === 1) {
            const w = this.widgets.find(w => w.name === "list_mode");
            this._syncPorts(!!(w && w.value));
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
}

// ==================== Yuan_MiniMaxH3Video（模式切换 + 递进显示动态端口）====================
// 与后端 MiniMax_H3.py 中的常量保持一致
const MINIMAX_MODE_I2V = "图生视频";
const MINIMAX_MODE_REF = "参考图生视频";

const MINIMAX_I2V_INPUTS = ["first_frame", "last_frame"];
const MINIMAX_REF_BASE_INPUTS = ["audio_vae", "ref_images"];  // ref_images：图像列表端口（多图 batch）
// 递进组：前一端口连接后才显示下一个，最多 3 个
const MINIMAX_REF_CHAINS = [
    ["ref_video_1", "ref_video_2", "ref_video_3"],
    ["ref_video_audio_1", "ref_video_audio_2", "ref_video_audio_3"],
    ["ref_audio_1", "ref_audio_2", "ref_audio_3"],
];
const MINIMAX_ALL_OPTIONAL = [
    ...MINIMAX_I2V_INPUTS,
    ...MINIMAX_REF_BASE_INPUTS,
    ...MINIMAX_REF_CHAINS.flat(),
];

const MINIMAX_PORT_TYPES = {
    first_frame: "IMAGE",
    last_frame: "IMAGE",
    audio_vae: "VAE",
    ref_images: "IMAGE",
    ref_video_1: "IMAGE", ref_video_2: "IMAGE", ref_video_3: "IMAGE",
    ref_video_audio_1: "AUDIO", ref_video_audio_2: "AUDIO", ref_video_audio_3: "AUDIO",
    ref_audio_1: "AUDIO", ref_audio_2: "AUDIO", ref_audio_3: "AUDIO",
};

function registerYuanMiniMaxH3Video(nodeType, portMeta) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
        const self = this;

        // 递进组端口可见性：第一个总是显示；第 k 个（k>=2）在前一个已连接或自身已连接时显示
        const chainVisible = (chain, idx) => {
            if (idx === 0) return true;
            const prevInp = self.inputs.find(inp => inp.name === chain[idx - 1]);
            if (prevInp && prevInp.link != null) return true;
            const curInp = self.inputs.find(inp => inp.name === chain[idx]);
            return !!(curInp && curInp.link != null);
        };

        const addOptionalInput = (name) => {
            const meta = portMeta[name] || {};
            const typ = MINIMAX_PORT_TYPES[name] || (meta.type || "*");
            const opts = { shape: 7, optional: true };
            if (meta.display_name) opts.display_name = meta.display_name;
            if (meta.tooltip) opts.tooltip = meta.tooltip;
            self.addInput(name, typ, opts);
            const inp = self.inputs.find(inp => inp.name === name);
            if (inp) {
                inp.optional = true;
                if (meta.display_name) inp.label = meta.display_name;
                if (meta.tooltip) inp.tooltip = meta.tooltip;
            }
        };

        const syncPortsReal = () => {
            if (self._syncing) return;
            self._syncing = true;
            try {
                const modeWidget = self.widgets.find(w => w.name === "mode");
                const isRef = !!(modeWidget && modeWidget.value === MINIMAX_MODE_REF);

                // 计算期望端口集合（有序）
                const desiredNames = [];
                if (isRef) {
                    desiredNames.push(...MINIMAX_REF_BASE_INPUTS);
                    for (const chain of MINIMAX_REF_CHAINS) {
                        for (let i = 0; i < chain.length; i++) {
                            if (chainVisible(chain, i)) desiredNames.push(chain[i]);
                        }
                    }
                } else {
                    desiredNames.push(...MINIMAX_I2V_INPUTS);
                }

                // 第一步：保存所有可选端口的连接信息
                const savedLinks = [];
                for (let i = self.inputs.length - 1; i >= 0; i--) {
                    const inp = self.inputs[i];
                    if (MINIMAX_ALL_OPTIONAL.includes(inp.name) && inp.link != null) {
                        const linkObj = app.graph.links[inp.link];
                        if (linkObj) {
                            savedLinks.push({
                                name: inp.name,
                                origin_id: linkObj.origin_id,
                                origin_slot: linkObj.origin_slot,
                            });
                        }
                    }
                }

                // 第二步：删除不需要的可选端口
                for (let i = self.inputs.length - 1; i >= 0; i--) {
                    const inp = self.inputs[i];
                    if (MINIMAX_ALL_OPTIONAL.includes(inp.name) && !desiredNames.includes(inp.name)) {
                        const idx = self.inputs.findIndex(x => x.name === inp.name);
                        if (idx !== -1) self.removeInput(idx);
                    }
                }

                // 第三步：添加缺少的可选端口（按 desiredNames 顺序，保证排列整洁）
                for (const name of desiredNames) {
                    if (!self.inputs.find(inp => inp.name === name)) {
                        addOptionalInput(name);
                    }
                }

                // 第四步：按 desiredNames 顺序整理可选端口（保险，保留连接并修正 target_slot）
                const required = self.inputs.filter(inp => !MINIMAX_ALL_OPTIONAL.includes(inp.name));
                const optional = [];
                for (const name of desiredNames) {
                    const inp = self.inputs.find(i => i.name === name);
                    if (inp) optional.push(inp);
                }
                const currentOptional = self.inputs.filter(inp => MINIMAX_ALL_OPTIONAL.includes(inp.name));
                const orderOk = currentOptional.length === optional.length &&
                    currentOptional.every((inp, i) => inp === optional[i]);
                if (!orderOk) {
                    self.inputs = [...required, ...optional];
                    optional.forEach((inp, idx) => {
                        if (inp.link != null) {
                            const l = app.graph.links[inp.link];
                            if (l) l.target_slot = required.length + idx;
                        }
                    });
                }

                // 第五步：恢复所有保存的连接
                for (const saved of savedLinks) {
                    const inp = self.inputs.find(inp => inp.name === saved.name);
                    if (inp && inp.link == null) {
                        const originNode = app.graph.getNodeById(saved.origin_id);
                        if (originNode) {
                            const targetSlot = self.inputs.indexOf(inp);
                            try {
                                // 传节点对象而不是 self.id（新版节点 id 是字符串，connect 只解析 number）
                                originNode.connect(saved.origin_slot, self, targetSlot);
                            } catch (_) {}
                        }
                    }
                }

                // 第六步：参考图尺寸 widget 显隐（只隐藏不删除，避免 widgets_values 索引错位）
                const sizeWidget = self.widgets.find(w => w.name === "ref_image_size");
                if (sizeWidget) sizeWidget.hidden = !isRef;

                // 第七步：保持当前宽度不变，只更新高度
                const currentWidth = self.size ? self.size[0] : self.computeSize()[0];
                self.setSize([currentWidth, self.computeSize()[1]]);
                app.graph.setDirtyCanvas(true, true);
            } finally {
                self._syncing = false;
            }
        };

        // 把 syncPortsReal 调度到下一 tick，确保 graph.links 已完整恢复（与 YuanTool 同源 BUG 修复）。
        // 在 onNodeCreated / onConfigure / mode.callback / onConnectionsChange
        // 等生命周期中，links 可能尚未挂到 input.link 上，立即执行会
        // 误判递进端口、重排 slot 错位、savedLinks 为空，导致切换工作流时断开。
        const scheduleSync = () => {
            clearTimeout(self._syncTimer);
            self._syncTimer = setTimeout(syncPortsReal, 0);
        };

        self._syncPorts = scheduleSync;

        const modeWidget = self.widgets.find(w => w.name === "mode");
        if (modeWidget) {
            scheduleSync();

            const origCallback = modeWidget.callback;
            modeWidget.callback = function () {
                if (origCallback) origCallback.apply(this, arguments);
                scheduleSync();
            };
        }

        // 修正初始已创建端口的 label / tooltip（框架自动生成的端口可能未正确应用 display_name）
        for (const inp of self.inputs) {
            const meta = portMeta[inp.name];
            if (meta) {
                if (meta.display_name) inp.label = meta.display_name;
                if (meta.tooltip) inp.tooltip = meta.tooltip;
            }
        }

        return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, slot, connected, link_info, input_or_output) {
        const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
        // 输入端口（type===1）连接变化时刷新递进显示；_syncing 期间跳过避免递归
        if (!this._syncing && this._syncPorts && type === 1) {
            this._syncPorts();
        }
        return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
        const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
        if (this._syncPorts) this._syncPorts();
        return r;
    };
}

app.registerExtension({
    name: "ComfyUI-Yuan-Tool",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // 从后端注册的 nodeData.input 提取每个可选端口的 display_name / tooltip 元数据
        // 前端动态 addInput 时会用这份元数据，避免删建后汉化名丢失。
        const buildPortMeta = () => {
            const meta = {};
            const input = nodeData.input || {};
            for (const sect of ["required", "optional"]) {
                const grp = input[sect] || {};
                for (const [name, def] of Object.entries(grp)) {
                    if (def && Array.isArray(def) && def.length >= 2 && def[1] && typeof def[1] === "object") {
                        const opts = def[1];
                        meta[name] = {
                            type: def[0],
                            display_name: opts.display_name,
                            tooltip: opts.tooltip,
                        };
                    }
                }
            }
            return meta;
        };
        if (nodeData.name === "YuanTool") {
            registerYuanTool(nodeType, buildPortMeta());
        } else if (nodeData.name === "Yuan_MiniMaxH3Video") {
            registerYuanMiniMaxH3Video(nodeType, buildPortMeta());
        }
    },
});
