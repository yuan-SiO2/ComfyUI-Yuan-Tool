const { app } = window.comfyAPI.app;
import { uploadChunked } from "./Yuan_Common.js";

// ---------- YuanTool：多帧参考节点（list_mode 动态端口） ----------
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

        // 单帧端口递进：1、2 常显，第 k 个(k>=3)需前 k-1 个均已连接或自身已连接
        const singleVisible = (idx) => {
            if (idx < 2) return true;
            if (isConnected(singleNames[idx])) return true;
            for (let i = 0; i < idx; i++) {
                if (!isConnected(singleNames[i])) return false;
            }
            return true;
        };

        // 列表端口递进：1、2 常显，第 k 个(k>=3)需前 k-1 个列表端口全部连接或自身已连接
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
            // 期望端口集合（有序）：列表模式=listNames 递进组，单帧模式=singleNames 递进组
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

            // 第四步：背景端口置末。removeInput 会删已连线（重连还因字符串 id 静默失败），
            // 故数组内移动端口并同步修正 link 的 target_slot。
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

        // 下一 tick 再同步：生命周期回调中 graph.links 尚未挂到 input.link，立即执行会误判端口、slot 错位致切换工作流断开
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

// ---------- Yuan_MiniMaxH3Video：模式切换 + 递进显示动态端口（常量对齐后端 Minimax/MiniMax_H3.py） ----------
const MINIMAX_MODE_REF = "参考图生视频";
const MINIMAX_MODE_GUIDE = "数字人";

const MINIMAX_I2V_INPUTS = ["first_frame", "last_frame"];
const MINIMAX_REF_BASE_INPUTS = ["audio_vae", "ref_images"]; // ref_images：图像列表端口（多图 batch）
// 数字人模式：仅引导端口（audio_vae 与参考模式共用），不显示首/尾帧
const MINIMAX_GUIDE_INPUTS = ["audio_vae", "guide_image", "guide_audio"];
// 递进组：前一端口连接后才显示下一个
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

        // 下一 tick 再同步：links 可能尚未挂到 input.link，立即执行会误判端口
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

// ---------- resize_type 条件显隐（RTX 放大 / H3 放大 / Latent 比例缩放通用） ----------
function registerResizeTypeConditionalWidgets(nodeType) {
    const UPSCALE_BY = "按倍数缩放";

    // 按 resize_type 切换 scale/width/height 显隐：按倍数缩放⇔scale，目标尺寸⇔width/height（只隐藏不删，避免 widgets_values 索引错位）
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

    // V3 (Nodes 2.0)：Vue 下拉直接改 widget.value 不走原生 callback，每帧轻量比对值变化即同步显隐（V2 下与 callback 路径幂等）
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

// widget 值类型自愈：工作流按「widget 位置」还原，后端新增/挪动 widget 会让旧值整体错位，
// 类型不符时重置为默认值，避免序列化出 null/错类型值触发后端"输入值类型错误"
function yuanHealWidgetType(widget, kind, fallback) {
    if (!widget) return;
    const v = widget.value;
    if (kind === "boolean") {
        if (typeof v !== "boolean") widget.value = fallback;
    } else if (kind === "string") {
        if (typeof v !== "string") widget.value = fallback;
    } else if (kind === "int") {
        if (isNaN(parseInt(String(v == null ? "" : v).trim(), 10))) widget.value = fallback;
    }
}

// 确保输入端口存在（重建被移除的端口）；widgetName 非空表示该端口绑定同名 widget（V3 下据此渲染为 widget 而非可连端口）
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

// V3 下被隐藏的 widget 会留空端口占位拉长节点，主动移除对应输入端口（有连线的保留，避免破坏连接）
function yuanRemoveHiddenWidgetPorts(node, names) {
    if (!node || !Array.isArray(node.inputs) || !yuanIsV3Node(node)) return;
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const inp = node.inputs[i];
        if (inp && names.indexOf(inp.name) !== -1 && inp.link == null) {
            try { node.removeInput(i); } catch (_) {}
        }
    }
}

// 兼容 V2（数组）与 V3（Map）的 graph.links 取值，统一归一化为命名属性对象。
// 新版前端 links 是 Map<number, LLink> 的 Proxy，取到的是 LLink 实例（origin_id/origin_slot/target_id/target_slot），
// 按数组下标读 l[1]/l[2]/l[3] 会得到 undefined（连接信息静默丢失，模式切换后端口无法重连）。
function yuanH3LinkObj(graph, id) {
    if (!graph || id == null) return null;
    const links = graph.links;
    if (!links) return null;
    let l = links[id]; // V3 Proxy 会把 id 数值化；老版为普通对象
    if (l == null && typeof links.get === "function") {
        try { l = links.get(id); } catch (_) { l = null; }
    }
    if (!l) return null;
    if (Array.isArray(l)) {
        // [id, origin_id, origin_slot, target_id, target_slot, type]
        return { origin_id: l[1], origin_slot: l[2], target_id: l[3], target_slot: l[4], type: l[5] };
    }
    return {
        origin_id: l.origin_id,
        origin_slot: l.origin_slot,
        target_id: l.target_id,
        target_slot: l.target_slot,
        type: l.type,
    };
}

// 把裁剪链路上 Trim 节点的「存储位置」同步到本节点隐藏 widget（值不同才写入并重绘）
function yuanH3SyncStorageFromTrim(node) {
    if (!node || !app || !app.graph || !Array.isArray(node.outputs)) return;
    const out = node.outputs[1]; // 索引 1 = 裁剪帧数
    if (!out || !Array.isArray(out.links) || out.links.length === 0) return;
    const graph = app.graph;
    for (const lid of out.links) {
        const l = yuanH3LinkObj(graph, lid);
        if (!l || typeof graph.getNodeById !== "function") continue;
        const trim = graph.getNodeById(l.target_id);
        if (!trim || trim.type !== "Yuan_H3MotionContextTrim") continue;
        const src = trim.widgets && trim.widgets.find((w) => w.name === "存储位置");
        const dst = node.widgets && node.widgets.find((w) => w.name === "存储位置");
        if (!src || !dst || src.value === dst.value) continue;
        dst.value = src.value;
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
        break;
    }
}

function registerYuanH3MotionContext(nodeType, portMeta) {
    const HINT_HEIGHT = 20; // 提示行预留高度
    const PORT_IMG = "上下文图像"; // 随模式显隐的真实数据端口（视频图像 + 端口模式）
    const PORT_AUD = "上下文音频"; // 随模式显隐的真实数据端口（视频图像 + 端口模式）
    const PORT_LAT = "上下文潜空间"; // 随模式显隐的真实数据端口（潜空间 + 端口模式）
    const PORT_VAE = "VAE"; // 仅「视频图像」模式需要的解码端口
    const MODE_VALUES = ["上传", "端口", "自动索引"];
    const LINK_VIDEO = "视频图像"; // 「衔接模式」取值：解码→重编码的媒体路径
    const LINK_LATENT = "潜空间"; // 「衔接模式」取值：直接切片潜空间
    const LINK_VALUES = [LINK_VIDEO, LINK_LATENT];
    const LINK_LATENT_WIDGETS = ["潜空间上下文长度", "潜空间音频长度", "衔接余量"];
    const LINK_VIDEO_WIDGETS = ["上下文长度", "音频上下文长度"];

    // combo 自愈：历史错位曾致误删「片段序号」端口、GetNode 连线永久丢失；无效值一律重置为 fallback 并返回归一化结果
    const normalizeCombo = (widget, values, fallback) => {
        if (!widget) return fallback;
        const v = String(widget.value == null ? "" : widget.value).trim();
        if (values.indexOf(v) !== -1) return v;
        widget.value = fallback;
        return fallback;
    };

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

    // 节点底部绘提示；顺带轻量同步「存储位置」、比对「模式」/「衔接模式」值（V3 下拉不走原生 callback，值变即同步）
    const origOnDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
        yuanH3SyncStorageFromTrim(this);
        const modeWidget = this.widgets && this.widgets.find((w) => w.name === "模式");
        const linkWidget = this.widgets && this.widgets.find((w) => w.name === "衔接模式");
        const modeChanged = modeWidget && modeWidget.value !== this._lastModeValue;
        const linkChanged = linkWidget && linkWidget.value !== this._lastLinkValue;
        if ((modeChanged || linkChanged) && this._syncModeReal) {
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

    // 加载时清提示（运行时生成，不随工作流保存），并按「模式」重建显隐、清空占位、同步存储位置
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
        const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
        this._h3Hint = null;
        // combo 自愈：历史错位工作流（widget 值前移）会把别的参数值塞进 combo（如模式="H3-Mubu"），
        // 重置默认值防止模式误判与序列化固化错位
        const w = (name) => this.widgets && this.widgets.find((x) => x.name === name);
        normalizeCombo(w("模式"), MODE_VALUES, "自动索引");
        normalizeCombo(w("衔接模式"), LINK_VALUES, LINK_VIDEO);
        normalizeCombo(w("上下文长度"), ["5", "22", "39", "56"], "5");
        normalizeCombo(w("音频上下文长度"), ["0", "5", "22", "39", "56"], "5");
        normalizeCombo(w("潜空间上下文长度"), ["17", "34", "51", "68"], "34");
        normalizeCombo(w("潜空间音频长度"), ["0", "17", "34", "51", "68"], "17");
        normalizeCombo(w("衔接余量"), ["0", "17", "34"], "17");
        // 非 combo 参数同样自愈：widget 位置变动会让旧工作流的值整体错位（如布尔位拿到文件夹名）
        yuanHealWidgetType(w("启用上下文"), "boolean", true);
        yuanHealWidgetType(w("存储位置"), "string", "H3-Mubu");
        yuanHealWidgetType(w("片段序号"), "int", 1);
        yuanHealWidgetType(w("手动上传"), "string", "");
        if (this._syncModeReal) {
            // 同「运动裁剪」：加载期（LGraph.configure 进行中）增删端口会破坏刚建立的
            // 连线（其他节点的输入对象随后还会被替换），推迟到加载结束后再校正显隐。
            if (!this._yuanH3DeferSync) {
                this._yuanH3DeferSync = true;
                setTimeout(() => {
                    this._yuanH3DeferSync = false;
                    if (this._syncModeReal) this._syncModeReal();
                }, 0);
            }
        }
        yuanRemoveHiddenWidgetPorts(this, ["存储位置", "手动上传"]);
        yuanH3SyncStorageFromTrim(this);
        return r;
    };

    // 输入端口连接/断开时（尤其「上下文图像/上下文音频」在端口模式下的连线）重建显隐
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
        const linkWidget = this.widgets && this.widgets.find((w) => w.name === "衔接模式");

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
        fileInput.accept = ".mp4,.safetensors";
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
                const resp = await yuanH3ContextUploadFile(file, (done, total) => {
                    btn.name = `上传上下文 ${done}/${total}`;
                });
                if (!resp || !resp.name) {
                    throw new Error((resp && resp.error) || "上传失败：服务器未返回文件名");
                }
                if (manualWidget) manualWidget.value = resp.name;
                self._yuanH3Uploaded = true;
                btn.name = "上传完毕";
                self.setDirtyCanvas(true, true);
            } catch (err) {
                btn.name = "上传失败";
            } finally {
                // 只有未成功上传时才把按钮复位；成功上传后保持「上传完毕」
                if (!self._yuanH3Uploaded) {
                    setTimeout(() => { btn.name = "上传上下文"; }, 2000);
                }
            }
        });
        // 按钮触发：仅 widget 回调（与 Yuan_Video 的成熟按钮一致，回调可靠）
        const btnClicked = () => {
            fileInput.value = ""; // 允许重复选择同一文件时仍触发 change
            fileInput.click();
        };
        const btn = this.addWidget("button", "上传上下文", null, btnClicked);
        // 防抖：V3 下 widget 回调 + 按钮 DOM 可能双触发双弹框；且只有回调内 click() 才是浏览器放行弹框的用户手势链
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
                // 模式值自愈：无效值（历史错位遗留，如模式="H3-Mubu"）兜底为自动索引，防误判触发端口移除
                const mode = normalizeCombo(modeWidget, MODE_VALUES, "自动索引");
                self._lastModeValue = mode;
                const isUpload = mode === "上传";
                const isPort = mode === "端口";
                const isAuto = mode === "自动索引";
                // 衔接模式值自愈：旧工作流新增 widget 无值（或历史错位残留媒体文件名）时兜底
                const link = normalizeCombo(linkWidget, LINK_VALUES, LINK_VIDEO);
                self._lastLinkValue = link;
                const isLatentLink = link === LINK_LATENT;

                // 1) 端口显隐（离开时保存连接，返回时恢复）。哪些端口属于哪种衔接模式：
                //    上下文图像/上下文音频——仅「视频图像」；上下文潜空间——仅「潜空间」；
                //    VAE——仅「视频图像」（潜空间模式直接从 AV 潜空间切片，不需要视频 VAE）
                const portSync = (name, type, label, savedKey, visible) => {
                    if (visible) {
                        if (!(self.inputs && self.inputs.find((i) => i.name === name))) {
                            const meta = (portMeta && portMeta[name]) || {};
                            yuanEnsureInput(self, name, type, {
                                optional: true, label, tooltip: meta.tooltip,
                            });
                            const saved = self[savedKey];
                            self[savedKey] = null;
                            if (saved && app && app.graph && typeof app.graph.getNodeById === "function") {
                                const inp = self.inputs.find((i) => i.name === name);
                                const ts = self.inputs.indexOf(inp);
                                const origin = app.graph.getNodeById(saved.origin_id);
                                if (origin) { try { origin.connect(saved.origin_slot, self, ts); } catch (_) {} }
                            }
                        }
                    } else {
                        const inp = self.inputs && self.inputs.find((i) => i.name === name);
                        if (inp && inp.link != null) {
                            const l = yuanH3LinkObj(app.graph, inp.link);
                            if (l) self[savedKey] = { origin_id: l.origin_id, origin_slot: l.origin_slot };
                        }
                        if (inp) {
                            const idx = self.inputs.findIndex((i) => i.name === name);
                            if (idx !== -1) self.removeInput(idx);
                        }
                    }
                };
                portSync(PORT_IMG, "IMAGE", "上下文图像", "_yuanH3SavedImgLink", isPort && !isLatentLink);
                portSync(PORT_AUD, "AUDIO", "上下文音频", "_yuanH3SavedAudLink", isPort && !isLatentLink);
                portSync(PORT_LAT, "LATENT", "上下文潜空间", "_yuanH3SavedLatLink", isPort && isLatentLink);
                portSync(PORT_VAE, "VAE", "VAE", "_yuanH3SavedVaeLink", !isLatentLink);

                // 1b) 「衔接模式」专属参数显隐：两条数据路径各用各的参数，互不影响
                const setWidgetHidden = (name, hidden) => {
                    const wd = self.widgets && self.widgets.find((x) => x.name === name);
                    if (wd) wd.hidden = hidden;
                };
                for (const nm of LINK_VIDEO_WIDGETS) setWidgetHidden(nm, isLatentLink);
                for (const nm of LINK_LATENT_WIDGETS) setWidgetHidden(nm, !isLatentLink);

                // 2) 片段序号仅自动索引显示；端口常驻、只隐藏 widget——动态移除绑 widget 的端口
                // 曾致保存丢值错位、删掉接好的 GetNode 连线（实例级连接不随序列化，切换即永久丢失）。
                // 后端非自动索引模式下忽略此参数，常驻无副作用
                if (isAuto) {
                    if (seqWidget) seqWidget.hidden = false;
                } else {
                    // 隐藏前把 widget 值归一化为合法整数：残留空串会报"输入值类型错误"
                    if (seqWidget) {
                        const n = parseInt(String(seqWidget.value).trim(), 10);
                        seqWidget.value = isNaN(n) ? 1 : Math.max(0, Math.min(9999, n));
                        seqWidget.hidden = true;
                    }
                }

                // 3) 上传按钮仅上传模式显示；隐藏须用 hidden+disabled（type="hidden"+computeSize 恢复后丢点击绑定）
                if (btn) {
                    btn.hidden = !isUpload;
                    btn.disabled = !isUpload;
                }

                // 4) 离开上传模式：清除手动上传值，重置按钮文本（覆盖/清除之前的媒体）
                if (!isUpload) {
                    self._yuanH3Uploaded = false;
                    if (manualWidget && manualWidget.value) manualWidget.value = "";
                    if (btn && btn.name && btn.name.indexOf("上传上下文") === -1 && btn.name.indexOf("上传") !== -1) {
                        btn.name = "上传上下文";
                    }
                }

                // 5) 移除被隐藏 widget 的空端口占位（V3）
                yuanRemoveHiddenWidgetPorts(self, ["存储位置", "手动上传",
                    ...(isLatentLink ? LINK_VIDEO_WIDGETS : LINK_LATENT_WIDGETS)]);

                // 6) 保持当前宽度不变，只更新高度
                const curW = self.size ? self.size[0] : self.computeSize()[0];
                self.setSize([curW, self.computeSize()[1]]);
                if (app && app.graph) app.graph.setDirtyCanvas(true, true);
            } finally {
                self._syncing = false;
            }
        };
        this._syncModeReal = syncModeReal;

        // 初始按当前模式建立显隐；模式/衔接模式下拉 callback（V2）触发重建
        if (modeWidget || linkWidget) {
            syncModeReal();
        }
        if (modeWidget) {
            const origCallback = modeWidget.callback;
            modeWidget.callback = function () {
                if (origCallback) origCallback.apply(this, arguments);
                syncModeReal();
            };
        }
        if (linkWidget) {
            const origLinkCallback = linkWidget.callback;
            linkWidget.callback = function () {
                if (origLinkCallback) origLinkCallback.apply(this, arguments);
                syncModeReal();
            };
        }

        // 初始同步「存储位置」（未连线时由 onDrawForeground 持续推进）；加载时 Trim 链可能未建好，
        // 故延迟重试数次防参数错位导致"未找到片段 N 文件"
        yuanH3SyncStorageFromTrim(this);
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

// ==================== 上传上下文媒体（分块上传 .mp4，兼容旧版 .safetensors） ====================

async function yuanH3ContextUploadFile(file, onProgress) {
    // 分块上传到 /yuan_h3_motion_upload_latent（防单次请求超 body 上限），末块响应携带 {"name": "..."}
    const CHUNK_SIZE = 4 * 1024 * 1024;
    return uploadChunked(file, {
        chunkSize: CHUNK_SIZE,
        filename: file.name,
        sendChunk: async (formData) => {
            const res = await fetch("/yuan_h3_motion_upload_latent", { method: "POST", body: formData });
            if (!res.ok) throw new Error(`chunk ${res.status} failed`);
            return res.json();
        },
        onProgress,
    });
}

// ==================== Yuan_H3MotionContextTrim（H3 运动裁剪，输出槽随「衔接模式」切换）====================

function registerYuanH3MotionContextTrim(nodeType, portMeta) {
    const LINK_VIDEO = "视频图像"; // 「衔接模式」取值：解码为图像+音频后裁切
    const LINK_DECODE = "解码模式"; // 「衔接模式」取值：只解码为图像+音频，不裁切、不存盘
    const LINK_LATENT = "潜空间"; // 「衔接模式」取值：直接裁切 AV 潜空间
    const LINK_VALUES = [LINK_VIDEO, LINK_DECODE, LINK_LATENT];
    // 输出槽定义：两模式类型完全不同（IMAGE+AUDIO ↔ LATENT），切换时整体重建
    const OUT_VIDEO = [{ name: "图像", type: "IMAGE" }, { name: "音频", type: "AUDIO" }];
    const OUT_LATENT = [{ name: "潜空间", type: "LATENT" }];
    // 仅「视频图像」模式需要的解码端口（潜空间模式不解码）
    const DECODE_PORTS = [
        { name: "VAE", key: "_yuanH3TrimSavedVaeLink" },
        { name: "audio_vae", key: "_yuanH3TrimSavedAudioVaeLink" },
    ];
    // 仅裁切模式（视频图像/潜空间）需要：「解码模式」是纯解码，端口移除、参数隐藏
    const TRIM_PORT = "裁剪帧数";
    const TRIM_WIDGETS = ["片段序号", "保存到本地", "存储位置"];

    // combo 自愈：旧工作流无此 widget 值（位置还原得到 undefined）时兜底为默认值
    const normalizeCombo = (widget, values, fallback) => {
        if (!widget) return fallback;
        const v = String(widget.value == null ? "" : widget.value).trim();
        if (values.indexOf(v) !== -1) return v;
        widget.value = fallback;
        return fallback;
    };

    // 加载时自愈并按当前模式重建解码端口与输出槽
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
        const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
        const w = (name) => this.widgets && this.widgets.find((x) => x.name === name);
        normalizeCombo(w("衔接模式"), LINK_VALUES, LINK_VIDEO);
        // widget 位置变动会让旧工作流的值整体错位，按类型兜底
        yuanHealWidgetType(w("片段序号"), "int", 1);
        yuanHealWidgetType(w("保存到本地"), "boolean", true);
        yuanHealWidgetType(w("存储位置"), "string", "H3-Mubu");
        // 加载期不能动端口/输出槽：LGraph.configure 按 JSON 顺序逐个配置节点，此时
        // 其他节点可能尚未配置（输入对象随后会被替换）、图内连线也刚建立，此处增删
        // 端口或重建输出槽会留下悬空 link。推迟到本次加载结束后再按模式校正。
        if (this._syncTrimMode) {
            if (!this._yuanH3TrimDeferSync) {
                this._yuanH3TrimDeferSync = true;
                setTimeout(() => {
                    this._yuanH3TrimDeferSync = false;
                    if (this._syncTrimMode) this._syncTrimMode();
                }, 0);
            }
        }
        return r;
    };

    // V3 下拉不走原生 callback：值变即同步
    const origOnDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
        const linkWidget = this.widgets && this.widgets.find((x) => x.name === "衔接模式");
        if (linkWidget && this._syncTrimMode && linkWidget.value !== this._lastTrimModeValue) {
            this._syncTrimMode();
        }
        if (origOnDrawForeground) return origOnDrawForeground.apply(this, arguments);
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
        if (this._yuanH3TrimBuilt) return r;
        this._yuanH3TrimBuilt = true;
        const self = this;
        const linkWidget = this.widgets && this.widgets.find((w) => w.name === "衔接模式");

        // 输出槽按模式整体重建：槽数/槽序/类型都不同（0=图像+1=音频 vs 0=潜空间），
        // 旧连线语义已失效，须先断开再重建。但重建不能无条件丢弃连线：
        // 后端 RETURN_TYPES 是通配 "*"、RETURN_NAMES 固定两槽，加载工作流时前端会把
        // 存档输出与静态槽按位合并，类型恒为 "*"（与目标槽的 IMAGE/AUDIO 不等），
        // 于是每次加载都会触发重建——若无条件断开，输出连线在加载时被清空。
        // 故按「原始槽位 + 槽类型」记录目标，重建后类型兼容者自动重连，不兼容者
        // （如另一模式专属的音频输出）留待切回该模式时再恢复。
        const savedOutLinks = self._yuanH3SavedOutLinks || (self._yuanH3SavedOutLinks = []);
        const outTypeOk = (a, b) => !a || !b || a === "*" || b === "*" || a === b;
        const rebuildOutputs = (spec) => {
            const graph = self.graph || (app && app.graph);
            // 1) 记录现有输出连线（含槽位与槽类型）
            if (Array.isArray(self.outputs) && graph && typeof graph.getNodeById === "function") {
                self.outputs.forEach((o, i) => {
                    if (!o || !Array.isArray(o.links)) return;
                    for (const lid of o.links.slice()) {
                        const l = yuanH3LinkObj(graph, lid);
                        if (!l) continue;
                        const dup = savedOutLinks.some((r) => r.slot === i &&
                            r.target_id === l.target_id && r.target_slot === l.target_slot);
                        if (!dup) savedOutLinks.push({
                            slot: i, type: o.type,
                            target_id: l.target_id, target_slot: l.target_slot,
                        });
                    }
                });
            }
            // 2) 断开并重建
            if (Array.isArray(self.outputs)) {
                for (let i = self.outputs.length - 1; i >= 0; i--) {
                    try { if (self.disconnectOutput) self.disconnectOutput(i); } catch (_) {}
                }
                self.outputs.length = 0;
            }
            for (const s of spec) {
                if (self.addOutput) self.addOutput(s.name, s.type, {});
                const o = self.outputs && self.outputs[self.outputs.length - 1];
                if (o) {
                    o.name = s.name;
                    o.type = s.type;
                    if (!o.links) o.links = [];
                }
            }
            if (Array.isArray(self.outputs)) self.outputs.forEach((o, i) => {
                o.slot = i;
                o.slot_index = i;
            });
            // 3) 恢复类型兼容的连线；connect 自身还会按目标端口类型再校验一次
            if (!Array.isArray(self.outputs) || !graph ||
                typeof graph.getNodeById !== "function") return;
            for (let i = 0; i < self.outputs.length; i++) {
                const o = self.outputs[i];
                for (let k = savedOutLinks.length - 1; k >= 0; k--) {
                    const rec = savedOutLinks[k];
                    if (rec.slot !== i || !outTypeOk(rec.type, o.type)) continue;
                    const target = graph.getNodeById(rec.target_id);
                    const ti = target && Array.isArray(target.inputs) ? target.inputs[rec.target_slot] : null;
                    if (!ti) continue; // 目标端口已不存在，保留记录待其恢复
                    if (ti.link != null) { savedOutLinks.splice(k, 1); continue; } // 已被占用，记录作废
                    let made = null;
                    try { made = self.connect(i, target, rec.target_slot); } catch (_) {}
                    if (made) savedOutLinks.splice(k, 1);
                }
            }
        };

        // 模式切换：重建解码端口与输出槽（端口留空会被后端判定缺输入，必须真正增删）
        const syncTrimMode = () => {
            if (self._trimSyncing) return;
            self._trimSyncing = true;
            try {
                const mode = normalizeCombo(linkWidget, LINK_VALUES, LINK_VIDEO);
                self._lastTrimModeValue = mode;
                const isLatent = mode === LINK_LATENT;
                const isDecode = mode === LINK_DECODE;

                // 0) 「裁剪帧数」端口仅裁切模式需要：解码模式移除（离开时保存连接、返回时恢复）。
                //    端口按名字恢复，加载期前端按「名字」对齐存档连线，故端口次序变动不丢连线。
                {
                    const inp = self.inputs && self.inputs.find((i) => i.name === TRIM_PORT);
                    if (isDecode) {
                        if (inp && inp.link != null) {
                            const l = yuanH3LinkObj(app.graph, inp.link);
                            if (l) self._yuanH3TrimSavedTrimLink = { origin_id: l.origin_id, origin_slot: l.origin_slot };
                        }
                        if (inp) {
                            const idx = self.inputs.findIndex((i) => i.name === TRIM_PORT);
                            if (idx !== -1) self.removeInput(idx);
                        }
                    } else if (!inp) {
                        const meta = (portMeta && portMeta[TRIM_PORT]) || {};
                        yuanEnsureInput(self, TRIM_PORT, "STRING", {
                            optional: true, tooltip: meta.tooltip,
                        });
                        const saved = self._yuanH3TrimSavedTrimLink;
                        self._yuanH3TrimSavedTrimLink = null;
                        if (saved && app && app.graph && typeof app.graph.getNodeById === "function") {
                            const ni = self.inputs.find((i) => i.name === TRIM_PORT);
                            const ts = self.inputs.indexOf(ni);
                            const origin = app.graph.getNodeById(saved.origin_id);
                            if (origin) { try { origin.connect(saved.origin_slot, self, ts); } catch (_) {} }
                        }
                    }
                }

                // 0b) 解码模式隐藏「片段序号」「保存到本地」「存储位置」（widget 保留但不可见，
                //     值仍随工作流序列化，切回裁切模式即恢复显示）；V3 下另清掉其空端口占位
                for (const nm of TRIM_WIDGETS) {
                    const wd = self.widgets && self.widgets.find((x) => x.name === nm);
                    if (wd) wd.hidden = isDecode;
                }
                if (isDecode) yuanRemoveHiddenWidgetPorts(self, TRIM_WIDGETS);

                // 1) 解码端口仅「潜空间」模式移除（视频图像/解码模式都要解码，离开时保存连接、返回时恢复）
                for (const spec of DECODE_PORTS) {
                    const inp = self.inputs && self.inputs.find((i) => i.name === spec.name);
                    if (isLatent) {
                        if (!inp) continue;
                        if (inp.link != null) {
                            const l = yuanH3LinkObj(app.graph, inp.link);
                            if (l) self[spec.key] = { origin_id: l.origin_id, origin_slot: l.origin_slot };
                        }
                        const idx = self.inputs.findIndex((i) => i.name === spec.name);
                        if (idx !== -1) self.removeInput(idx);
                    } else if (!inp) {
                        const meta = (portMeta && portMeta[spec.name]) || {};
                        yuanEnsureInput(self, spec.name, "VAE", {
                            optional: true, tooltip: meta.tooltip,
                        });
                        const saved = self[spec.key];
                        self[spec.key] = null;
                        if (saved && app && app.graph && typeof app.graph.getNodeById === "function") {
                            const ni = self.inputs.find((i) => i.name === spec.name);
                            const ts = self.inputs.indexOf(ni);
                            const origin = app.graph.getNodeById(saved.origin_id);
                            if (origin) { try { origin.connect(saved.origin_slot, self, ts); } catch (_) {} }
                        }
                    }
                }

                // 2) 输出槽随模式重建（潜空间模式只留 1 个 LATENT 槽）
                const want = isLatent ? OUT_LATENT : OUT_VIDEO;
                const cur = self.outputs || [];
                const same = cur.length === want.length &&
                    cur.every((o, i) => o && o.name === want[i].name &&
                        (o.type === want[i].type || o.type === "*" || want[i].type === "*"));
                if (same) {
                    // 通配 "*" 只是加载时静态槽类型覆盖了存档类型，就地落实即可——不重建就不动连线
                    cur.forEach((o, i) => { o.type = want[i].type; o.name = want[i].name; });
                } else {
                    rebuildOutputs(want);
                }

                if (self.setSize && self.computeSize) {
                    const curW = self.size ? self.size[0] : self.computeSize()[0];
                    self.setSize([curW, self.computeSize()[1]]);
                }
                if (app && app.graph) app.graph.setDirtyCanvas(true, true);
            } finally {
                self._trimSyncing = false;
            }
        };
        this._syncTrimMode = syncTrimMode;

        // 初始按当前模式建立；衔接模式下拉 callback（V2）触发重建
        if (linkWidget) {
            syncTrimMode();
            const origCallback = linkWidget.callback;
            linkWidget.callback = function () {
                if (origCallback) origCallback.apply(this, arguments);
                syncTrimMode();
            };
        }
        return r;
    };
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
            registerYuanH3MotionContext(nodeType, buildPortMeta());
        } else if (nodeData.name === "Yuan_H3MotionContextTrim") {
            registerYuanH3MotionContextTrim(nodeType, buildPortMeta());
        }
    },
});
