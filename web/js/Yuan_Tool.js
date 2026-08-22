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

// ==================== resize_type 条件参数（RTX 视频放大 / H3 放大通用）====================
// 缩放方式为「按倍数缩放」时只显示 scale，为「目标尺寸」时只显示 width/height
function registerResizeTypeConditionalWidgets(nodeType) {
    const UPSCALE_BY = "按倍数缩放";

    // 根据 resize_type 切换 scale / width / height 的显隐
    // 只隐藏不删除，避免 widgets_values 索引错位
    const syncResizeWidgets = (self) => {
        const resizeWidget = self.widgets.find(w => w.name === "resize_type");
        if (!resizeWidget) return;
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
}

// ==================== Yuan_H3MotionContext（H3 运动上下文，节点下方提示）====================
function registerYuanH3MotionContext(nodeType) {
    const HINT_HEIGHT = 20; // 提示行预留高度

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

    // 在节点底部绘制单行提示（不可编辑）
    const origOnDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
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

    // 工作流加载时清除提示（提示由运行时生成，不随工作流保存）
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
        const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
        this._h3Hint = null;
        return r;
    };
}

// ==================== Yuan_H3MotionContextLoadLatent（H3 加载潜空间：手动上传按钮）====================

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

function registerYuanH3MotionContextLoadLatent(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
        const self = this;
        const manualWidget = this.widgets && this.widgets.find((w) => w.name === "手动上传");
        if (!manualWidget || this._yuanLatentUploadBtn) return r;
        const btn = this.addWidget("button", "上传潜空间", null, () => {
            const input = document.createElement("input");
            input.type = "file";
            input.accept = ".safetensors";
            input.onchange = async () => {
                const file = input.files && input.files[0];
                if (!file) return;
                const origLabel = btn.name;
                btn.name = "上传中…";
                try {
                    const resp = await yuanH3LatentUploadFile(file, (done, total) => {
                        btn.name = `上传潜空间 ${done}/${total}`;
                    });
                    if (!resp || !resp.name) {
                        throw new Error((resp && resp.error) || "上传失败：服务器未返回文件名");
                    }
                    manualWidget.value = resp.name;
                    self.setDirtyCanvas(true, true);
                    btn.name = "上传完成";
                } catch (err) {
                    console.error("[Yuan H3 加载潜空间] 上传失败:", err);
                    btn.name = "上传失败";
                } finally {
                    setTimeout(() => { btn.name = origLabel; }, 2000);
                }
            };
            input.click();
        });
        this._yuanLatentUploadBtn = btn;
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
        } else if (nodeData.name === "Yuan_H3MotionContext") {
            registerYuanH3MotionContext(nodeType);
        } else if (nodeData.name === "Yuan_H3MotionContextLoadLatent") {
            registerYuanH3MotionContextLoadLatent(nodeType);
        }
    },
});
