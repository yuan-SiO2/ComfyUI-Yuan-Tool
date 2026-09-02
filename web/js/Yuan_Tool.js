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

            // 第四步：确保背景端口存在且在最后。
            // 不能用 removeInput 调整顺序（会删除已连接的线，且重连会因字符串 id 静默失败），
            // 改为数组层面移动端口并同步修正 link 的 target_slot。
            const bgIdx = self.inputs.findIndex(inp => inp.name === bgName);
            if (bgIdx !== -1) {
                const bgInput = self.inputs[bgIdx];
                bgInput.optional = true;
                bgInput.removable = true;
                bgInput.shape = 7;

                if (bgIdx !== self.inputs.length - 1) {
                    self.inputs.splice(bgIdx, 1);
                    // 下标前移 1 位，同步修正后续端口的 target_slot
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
                            // 传节点对象而非 self.id：新版节点 id 为字符串，connect 只解析 number，传 id 会静默失败
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

        // 调度到下一 tick 再同步：生命周期回调中 graph.links 可能尚未挂到 input.link 上，
        // 立即执行会误判递进端口、slot 错位，导致切换工作流时断开
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
const MINIMAX_MODE_REF = "参考图生视频";
const MINIMAX_MODE_GUIDE = "数字人";

const MINIMAX_I2V_INPUTS = ["first_frame", "last_frame"];
const MINIMAX_REF_BASE_INPUTS = ["audio_vae", "ref_images"];  // ref_images：图像列表端口（多图 batch）
// 数字人模式：仅引导端口（audio_vae 与参考模式共用），不显示首/尾帧
const MINIMAX_GUIDE_INPUTS = ["audio_vae", "guide_image", "guide_audio"];
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
    ...MINIMAX_GUIDE_INPUTS,
];

const MINIMAX_PORT_TYPES = {
    first_frame: "IMAGE",
    last_frame: "IMAGE",
    audio_vae: "VAE",
    ref_images: "IMAGE",
    guide_image: "IMAGE",
    guide_audio: "AUDIO",
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
                const isGuide = !!(modeWidget && modeWidget.value === MINIMAX_MODE_GUIDE);

                // 计算期望端口集合（有序）
                const desiredNames = [];
                if (isRef) {
                    desiredNames.push(...MINIMAX_REF_BASE_INPUTS);
                    for (const chain of MINIMAX_REF_CHAINS) {
                        for (let i = 0; i < chain.length; i++) {
                            if (chainVisible(chain, i)) desiredNames.push(chain[i]);
                        }
                    }
                } else if (isGuide) {
                    desiredNames.push(...MINIMAX_GUIDE_INPUTS);
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

                // 第六步：参考图尺寸 / 锚定帧 widget 显隐（只隐藏不删除，避免 widgets_values 索引错位）
                const sizeWidget = self.widgets.find(w => w.name === "ref_image_size");
                if (sizeWidget) sizeWidget.hidden = !isRef;
                const guideIdxWidget = self.widgets.find(w => w.name === "guide_frame_idx");
                if (guideIdxWidget) guideIdxWidget.hidden = !isGuide;

                // 第七步：保持当前宽度不变，只更新高度
                const currentWidth = self.size ? self.size[0] : self.computeSize()[0];
                self.setSize([currentWidth, self.computeSize()[1]]);
                app.graph.setDirtyCanvas(true, true);
            } finally {
                self._syncing = false;
            }
        };

        // 调度到下一 tick 再同步：links 可能尚未挂到 input.link 上，立即执行会误判端口
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

// ==================== resize_type 条件参数（RTX 视频放大 / H3 放大 / 缩放Latent（比例）通用）====================
// 缩放方式为「按倍数缩放」时只显示 scale，为「目标尺寸」时只显示 width/height
function registerResizeTypeConditionalWidgets(nodeType) {
    const UPSCALE_BY = "按倍数缩放";

    // 根据 resize_type 切换 scale / width / height 的显隐
    // 只隐藏不删除，避免 widgets_values 索引错位
    const syncResizeWidgets = (self) => {
        const resizeWidget = self.widgets.find(w => w.name === "resize_type");
        if (!resizeWidget) return;
        self._lastResizeTypeValue = resizeWidget.value;
        const isByScale = resizeWidget.value === UPSCALE_BY;
        const scaleWidget = self.widgets.find(w => w.name === "scale");
        const widthWidget = self.widgets.find(w => w.name === "width");
        const heightWidget = self.widgets.find(w => w.name === "height");
        if (scaleWidget) scaleWidget.hidden = !isByScale;
        if (widthWidget) widthWidget.hidden = isByScale;
        if (heightWidget) heightWidget.hidden = isByScale;
        // 保持当前宽度不变，只更新高度
        const currentWidth = self.size ? self.size[0] : self.computeSize()[0];
        self.setSize([currentWidth, self.computeSize()[1]]);
        app.graph.setDirtyCanvas(true, true);
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
        const self = this;
        const resizeWidget = self.widgets.find(w => w.name === "resize_type");
        if (resizeWidget) {
            syncResizeWidgets(self);
            const origCallback = resizeWidget.callback;
            resizeWidget.callback = function () {
                if (origCallback) origCallback.apply(this, arguments);
                syncResizeWidgets(self);
            };
        }
        return r;
    };

    // 工作流加载时同步一次显隐
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
        const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
        syncResizeWidgets(this);
        return r;
    };

    // V3 (Nodes 2.0)：Vue 下拉组件直接改 widget.value、不走原生 callback，
    // 每帧轻量比对 resize_type 值，变化即同步显隐（V2 下与 callback 路径幂等）
    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function () {
        const r = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;
        if (!this.widgets) return r;
        const resizeWidget = this.widgets.find(w => w.name === "resize_type");
        if (resizeWidget && resizeWidget.value !== this._lastResizeTypeValue) {
            syncResizeWidgets(this);
        }
        return r;
    };
}

// ==================== Yuan_H3MotionContext（H3 运动上下文，节点下方提示）====================

function yuanHideWidget(w) {
    if (!w) return;
    w.type = "hidden";
    w.hidden = true;
    w.computeSize = () => [0, -4];
}

// 确保某输入端口存在（用于模式切换重建被移除的端口）。
// widgetName 非空表示该端口绑定同名 widget（V3 下据此把参数渲染为 widget 而非可连端口）
function yuanEnsureInput(node, name, type, opts, widgetName) {
    if (node && node.inputs && node.inputs.find((i) => i.name === name)) return;
    node.addInput(name, type, opts || {});
    const inp = node.inputs && node.inputs.find((i) => i.name === name);
    if (inp) {
        inp.optional = !!(opts && opts.optional);
        if (widgetName) inp.widget = widgetName;
        if (opts && opts.label) inp.label = opts.label;
        if (opts && opts.tooltip) inp.tooltip = opts.tooltip;
    }
}

// V3 (Nodes 2.0) 检测：向上找 comfy-node 元素（含 shadow DOM host）
function yuanIsV3Node(node) {
    try {
        const w = node.widgets && node.widgets.find((x) => x.element);
        const el = w ? w.element : null;
        let n = el;
        while (n) {
            if ((n.tagName && n.tagName.toLowerCase().includes("comfy-node")) ||
                (n.classList && n.classList.contains("comfy-node"))) return true;
            n = n.parentElement || (n.getRootNode ? n.getRootNode().host : null);
        }
    } catch (_) {}
    return false;
}

// V3 下被前端隐藏的 widget 会显示为空端口占位把节点拉长，主动移除对应输入端口
// （保留真实可连接端口；已有连线的端口保留不删，避免破坏连接）
function yuanRemoveHiddenWidgetPorts(node, names) {
    if (!node || !Array.isArray(node.inputs) || !yuanIsV3Node(node)) return;
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const inp = node.inputs[i];
        if (inp && names.indexOf(inp.name) !== -1 && inp.link == null) {
            try { node.removeInput(i); } catch (_) {}
        }
    }
}

// 兼容 V2（数组）与 V3（Map）的 graph.links 取值
function yuanH3LinkObj(graph, id) {
    if (!graph || id == null) return null;
    const links = graph.links;
    if (!links) return null;
    if (typeof links.get === "function") return links.get(id) || null;
    return links[id] || null;
}

// 把「H3 运动上下文 → 裁剪帧数 → H3 运动裁剪」链路上裁剪节点的「存储位置」
// 同步到本节点的隐藏「存储位置」widget（只在值不同时写入并重绘）
function yuanH3SyncStorageFromTrim(node) {
    if (!node || !app || !app.graph || !Array.isArray(node.outputs)) return;
    const out = node.outputs[1]; // 索引 1 = 裁剪帧数
    if (!out || !Array.isArray(out.links) || out.links.length === 0) return;
    const graph = app.graph;
    for (const lid of out.links) {
        const l = yuanH3LinkObj(graph, lid);
        if (!l || typeof graph.getNodeById !== "function") continue;
        const trim = graph.getNodeById(l[3]);
        if (!trim || trim.type !== "Yuan_H3MotionContextTrim") continue;
        const src = trim.widgets && trim.widgets.find((w) => w.name === "存储位置");
        const dst = node.widgets && node.widgets.find((w) => w.name === "存储位置");
        if (!src || !dst || src.value === dst.value) continue;
        dst.value = src.value;
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
        break;
    }
}

function registerYuanH3MotionContext(nodeType) {
    const HINT_HEIGHT = 20; // 提示行预留高度
    const PORT_NAME = "上下文潜空间"; // 随模式显隐的真实数据端口
    const SEQ_NAME = "片段序号"; // 仅自动索引模式显示

    const origOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (data) {
        // data 即后端返回的 ui 字典，需在默认处理之前提取
        let hint = null;
        if (data && data.h3_hint != null) {
            const raw = data.h3_hint;
            hint = Array.isArray(raw) ? raw.join("") : String(raw);
        }
        if (origOnExecuted) origOnExecuted.apply(this, arguments);
        // 默认处理之后赋值，避免提示被覆盖
        this._h3Hint = hint;
        // 重新计算大小以容纳/移除提示行
        const curW = this.size ? this.size[0] : 200;
        this.setSize([curW, this.computeSize()[1]]);
        this.setDirtyCanvas(true, true);
    };

    // 让节点高度包含提示行（仅在存在提示时增高）
    const origComputeSize = nodeType.prototype.computeSize;
    nodeType.prototype.computeSize = function () {
        const size = origComputeSize ? origComputeSize.apply(this, arguments) : [200, 100];
        if (this._h3Hint) size[1] += HINT_HEIGHT;
        return size;
    };

    // 在节点底部绘制单行提示（不可编辑）。绘制前做一次「存储位置」轻量同步，
    // 并比对「模式」值（V3 下 Vue 下拉不走原生 callback，值变化即触发显隐同步）
    const origOnDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
        yuanH3SyncStorageFromTrim(this);
        const modeWidget = this.widgets && this.widgets.find((w) => w.name === "模式");
        if (modeWidget && this._syncModeReal && modeWidget.value !== this._lastModeValue) {
            this._syncModeReal();
        }
        if (origOnDrawForeground) origOnDrawForeground.apply(this, arguments);
        if (!this._h3Hint) return;
        ctx.save();
        ctx.fillStyle = "#cccccc";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        const y = this.size[1] - HINT_HEIGHT / 2;
        ctx.fillText(this._h3Hint, 10, y);
        ctx.restore();
    };

    // 工作流加载时清除提示（提示由运行时生成，不随工作流保存），
    // 并按「模式」重建显隐、移除被隐藏 widget 的空端口占位、同步存储位置
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
        const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
        this._h3Hint = null;
        if (this._syncModeReal) this._syncModeReal();
        yuanRemoveHiddenWidgetPorts(this, ["存储位置", "手动上传"]);
        yuanH3SyncStorageFromTrim(this);
        return r;
    };

    // 输入端口连接/断开时（尤其「上下文潜空间」在端口模式下的连线）重建显隐
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, slot, connected, link_info, input_or_output) {
        const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
        if (!this._syncing && this._syncModeReal && type === 1) this._syncModeReal();
        return r;
    };

    // 模式切换：按「上传/端口/自动索引」重建端口与按钮显隐；存储位置/手动上传始终隐藏
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
        if (this._yuanH3MotionBuilt) return r;
        this._yuanH3MotionBuilt = true;
        const self = this;

        const storageWidget = this.widgets && this.widgets.find((w) => w.name === "存储位置");
        const manualWidget = this.widgets && this.widgets.find((w) => w.name === "手动上传");
        const seqWidget = this.widgets && this.widgets.find((w) => w.name === "片段序号");
        const modeWidget = this.widgets && this.widgets.find((w) => w.name === "模式");

        // 「片段序号」serializeValue 兜底：归一化为合法整数，避免残留空串触发后端"输入值类型错误"
        if (seqWidget) {
            const seqNormalize = (v) => {
                const n = parseInt(String(v == null ? "" : v).trim(), 10);
                return isNaN(n) ? 1 : Math.max(0, Math.min(9999, n));
            };
            seqWidget.serializeValue = function (node, idx) {
                return seqNormalize(this.value);
            };
        }

        // 始终隐藏：存储位置（随裁剪同步）、手动上传（仅由按钮写入）
        yuanHideWidget(storageWidget);
        yuanHideWidget(manualWidget);
        yuanRemoveHiddenWidgetPorts(this, ["存储位置", "手动上传"]);

        // 上传按钮 input 须挂到 DOM 且不能用 display:none，否则部分浏览器会拦截 click() 弹框
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = ".safetensors";
        Object.assign(fileInput.style, {
            position: "absolute",
            width: "1px",
            height: "1px",
            opacity: "0",
            overflow: "hidden",
            pointerEvents: "none",
            zIndex: "-1",
        });
        document.body.appendChild(fileInput);
        fileInput.addEventListener("change", async () => {
            const file = fileInput.files && fileInput.files[0];
            if (!file) return;
            btn.name = "上传中…";
            try {
                const resp = await yuanH3LatentUploadFile(file, (done, total) => {
                    btn.name = `上传潜空间 ${done}/${total}`;
                });
                if (!resp || !resp.name) {
                    throw new Error((resp && resp.error) || "上传失败：服务器未返回文件名");
                }
                if (manualWidget) manualWidget.value = resp.name;
                // 上传的是上一片段潜空间：音频上下文置 0 防上一片段声音污染本片段
                const audioCtxWidget = self.widgets &&
                    self.widgets.find((w) => w.name === "音频上下文长度");
                if (audioCtxWidget && String(audioCtxWidget.value) !== "0") {
                    audioCtxWidget.value = "0";
                    if (typeof audioCtxWidget.callback === "function") {
                        try { audioCtxWidget.callback(); } catch (_) {}
                    }
                }
                self._yuanH3Uploaded = true;
                btn.name = "上传完毕";
                self.setDirtyCanvas(true, true);
            } catch (err) {
                btn.name = "上传失败";
            } finally {
                // 只有未成功上传时才把按钮复位；成功上传后保持「上传完毕」
                if (!self._yuanH3Uploaded) {
                    setTimeout(() => { btn.name = "上传潜空间"; }, 2000);
                }
            }
        });
        // 按钮触发：仅 widget 回调（与 Yuan_Video 的成熟按钮一致，回调可靠）
        const btnClicked = () => {
            fileInput.value = ""; // 允许重复选择同一文件时仍触发 change
            fileInput.click();
        };
        const btn = this.addWidget("button", "上传潜空间", null, btnClicked);
        // 防抖：V3 下 widget 回调 + 按钮 DOM 可能同时收到点击，去重防双弹框。
        // 但只有真正从回调里 click()，才是用户手势链内的调用，浏览器才放行弹框。
        let _btnLastClickMs = 0;
        const btnSafeClicked = () => {
            const now = Date.now();
            if (now - _btnLastClickMs < 500) return; // 双触发，防重复弹框
            _btnLastClickMs = now;
            btnClicked();
        };
        const origBtnCallback = btn.callback;
        btn.callback = btnSafeClicked;
        // 补充原生 DOM 监听（仅当 widget 回调确实不触发时才需要）
        (function bindBtnDom() {
            if (btn.element && !btn._domClickBound) {
                try {
                    btn.element.addEventListener("click", btnSafeClicked);
                    btn._domClickBound = true;
                    return;
                } catch (_) {}
            }
            if (!btn._domClickBound) setTimeout(bindBtnDom, 100);
        })();
        this._yuanH3UploadBtn = btn;

        // 模式显隐核心：按「模式」widget 值重建端口/按钮/参数可见性
        const syncModeReal = () => {
            if (self._syncing) return;
            self._syncing = true;
            try {
                const mode = modeWidget ? modeWidget.value : "自动索引";
                self._lastModeValue = mode;
                const isUpload = mode === "上传";
                const isPort = mode === "端口";
                const isAuto = mode === "自动索引";

                // 1) 上下文潜空间 端口显隐（离开时保存连接，返回时恢复）
                if (isPort) {
                    if (!(self.inputs && self.inputs.find((i) => i.name === PORT_NAME))) {
                        yuanEnsureInput(self, PORT_NAME, "LATENT", {
                            shape: 7, optional: true, label: "上下文潜空间",
                        });
                        const saved = self._yuanH3SavedPortLink;
                        self._yuanH3SavedPortLink = null;
                        if (saved && app && app.graph && typeof app.graph.getNodeById === "function") {
                            const inp = self.inputs.find((i) => i.name === PORT_NAME);
                            const origin = app.graph.getNodeById(saved.origin_id);
                            const ts = self.inputs.indexOf(inp);
                            if (origin) { try { origin.connect(saved.origin_slot, self, ts); } catch (_) {} }
                        }
                    }
                } else {
                    const inp = self.inputs && self.inputs.find((i) => i.name === PORT_NAME);
                    if (inp && inp.link != null) {
                        const l = yuanH3LinkObj(app.graph, inp.link);
                        if (l) self._yuanH3SavedPortLink = { origin_id: l[1], origin_slot: l[2] };
                    }
                    if (inp) {
                        const idx = self.inputs.findIndex((i) => i.name === PORT_NAME);
                        if (idx !== -1) self.removeInput(idx);
                    }
                }

                // 2) 片段序号 port+widget 显隐：仅自动索引显示。
                // 上传/端口模式下它完全无关（上传模式用手动文件、端口模式用
                // 端口潜空间），即使有连线也整端口移除，切回自动索引时恢复连接
                // （与「上下文潜空间」端口的保存/恢复逻辑一致）。
                if (isAuto) {
                    if (!(self.inputs && self.inputs.find((i) => i.name === SEQ_NAME))) {
                        yuanEnsureInput(self, SEQ_NAME, "INT",
                            { default: 1, min: 0, max: 9999 }, SEQ_NAME);
                        const saved = self._yuanH3SavedSeqLink;
                        self._yuanH3SavedSeqLink = null;
                        if (saved && app && app.graph && typeof app.graph.getNodeById === "function") {
                            const inp = self.inputs.find((i) => i.name === SEQ_NAME);
                            const ts = self.inputs.indexOf(inp);
                            const origin = app.graph.getNodeById(saved.origin_id);
                            if (origin) { try { origin.connect(saved.origin_slot, self, ts); } catch (_) {} }
                        }
                    }
                    if (seqWidget) seqWidget.hidden = false;
                } else {
                    // 端口移除前先把 widget 值归一化为合法整数：端口被移除后，
                    // 该参数由 widget 值参与 ComfyUI 类型校验，其残留的空串/
                    // 无效值会报"输入值类型错误"。上传/端口模式下后端完全忽略
                    // 此参数，归一化无任何副作用。
                    if (seqWidget) {
                        const n = parseInt(String(seqWidget.value).trim(), 10);
                        seqWidget.value = isNaN(n) ? 1 : Math.max(0, Math.min(9999, n));
                    }
                    const inp = self.inputs && self.inputs.find((i) => i.name === SEQ_NAME);
                    if (inp && inp.link != null) {
                        const l = yuanH3LinkObj(app.graph, inp.link);
                        if (l) self._yuanH3SavedSeqLink = { origin_id: l[1], origin_slot: l[2] };
                    }
                    if (inp) {
                        const idx = self.inputs.findIndex((i) => i.name === SEQ_NAME);
                        if (idx !== -1) self.removeInput(idx);
                    }
                    if (seqWidget) seqWidget.hidden = true;
                }

                // 3) 上传潜空间 按钮显隐：仅上传模式显示
                // 按钮隐藏须用 hidden+disabled（type="hidden"+computeSize 会在恢复后丢点击绑定）
                if (btn) {
                    btn.hidden = !isUpload;
                    btn.disabled = !isUpload;
                }

                // 4) 离开上传模式：清除手动上传值，重置按钮文本（覆盖/清除之前的 latent）
                if (!isUpload) {
                    self._yuanH3Uploaded = false;
                    if (manualWidget && manualWidget.value) manualWidget.value = "";
                    if (btn && btn.name && btn.name.indexOf("上传潜空间") === -1 && btn.name.indexOf("上传") !== -1) {
                        btn.name = "上传潜空间";
                    }
                }

                // 5) 移除被隐藏 widget 的空端口占位（V3）
                yuanRemoveHiddenWidgetPorts(self, ["存储位置", "手动上传"]);

                // 6) 保持当前宽度不变，只更新高度
                const curW = self.size ? self.size[0] : self.computeSize()[0];
                self.setSize([curW, self.computeSize()[1]]);
                if (app && app.graph) app.graph.setDirtyCanvas(true, true);
            } finally {
                self._syncing = false;
            }
        };
        this._syncModeReal = syncModeReal;

        // 初始按当前模式建立显隐；模式下拉 callback（V2）触发重建
        if (modeWidget) {
            syncModeReal();
            const origCallback = modeWidget.callback;
            modeWidget.callback = function () {
                if (origCallback) origCallback.apply(this, arguments);
                syncModeReal();
            };
        }

        // 初始同步一次「存储位置」（可能「裁剪帧数」尚未连线，连线后由 onDrawForeground 持续推进）
        yuanH3SyncStorageFromTrim(this);
        // 工作流加载时链路可能尚未建立（Trim 节点后配置），延迟重试几次确保
        // 「存储位置」从裁剪节点同步到位，避免因参数错位导致的"未找到片段 N 文件"
        if (!self._yuanH3SyncRetried) {
            self._yuanH3SyncRetried = true;
            const retry = () => {
                const stillAlive = self.widgets && self.widgets.length > 0 &&
                    self.widgets.some((w) => w.name === "存储位置");
                if (!stillAlive) return;
                yuanH3SyncStorageFromTrim(self);
            };
            [200, 800, 2000].forEach((ms) => setTimeout(retry, ms));
        }
        return r;
    };
}

// ==================== 潜空间手动上传（分块上传 .safetensors） ====================

async function yuanH3LatentUploadFile(file, onProgress) {
    // 分块上传潜空间文件到后端 /yuan_h3_motion_upload_latent
    // （避免单次请求超出服务端 body 上限），最后一块的响应携带 {"name": "..."}
    const CHUNK_SIZE = 4 * 1024 * 1024;
    const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
    let lastResp = null;
    for (let i = 0; i < totalChunks; i++) {
        const blob = file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE);
        const formData = new FormData();
        formData.append("file", blob);
        formData.append("filename", file.name);
        formData.append("chunk_index", String(i));
        formData.append("total_chunks", String(totalChunks));
        const res = await fetch("/yuan_h3_motion_upload_latent", { method: "POST", body: formData });
        if (!res.ok) throw new Error(`chunk ${i + 1}/${totalChunks} failed: ${res.status}`);
        if (i === totalChunks - 1) lastResp = await res.json();
        if (onProgress) onProgress(i + 1, totalChunks);
    }
    return lastResp;
}

app.registerExtension({
    name: "ComfyUI-Yuan-Tool",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // 提取后端注册的可选端口元数据（display_name/tooltip），动态建端口时避免汉化名丢失
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
        } else if (nodeData.name === "Yuan_RTXVideoUpscaleH3") {
            registerResizeTypeConditionalWidgets(nodeType);
        } else if (nodeData.name === "Yuan_H3Upscale3D") {
            registerResizeTypeConditionalWidgets(nodeType);
        } else if (nodeData.name === "Yuan_LatentUpscaleBy") {
            registerResizeTypeConditionalWidgets(nodeType);
        } else if (nodeData.name === "Yuan_H3MotionContext") {
            registerYuanH3MotionContext(nodeType);
        }
    },
});
