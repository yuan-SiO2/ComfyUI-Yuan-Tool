/**
 * Yuan Tool · Panorama 前端
 *
 * 为 YuanPanoramaPreview 节点提供自包含的 WebGL 球面投影全景查看器：
 *  - 拖拽旋转（yaw/pitch，带惯性）
 *  - 滚轮缩放（FOV）
 *  - 360° / 180° 覆盖支持
 *  - 图像与视频（mp4 批次）输入
 *  - 双击 / 空格 播放暂停视频
 *
 * YuanPanoramaSeamPrep 为普通图像/掩码节点，由 ComfyUI 默认渲染，无需特殊前端。
 */
(function () {
    "use strict";

    const { app } = window.comfyAPI.app;

    function getApi() {
        try {
            const c = window.comfyAPI;
            if (c && c.api) {
                if (c.api.api && typeof c.api.api.apiURL === "function") return c.api.api;
                if (typeof c.api.apiURL === "function") return c.api.api;
            }
        } catch (_) {}
        return null;
    }
    const api = getApi();
    function apiUrl(q) {
        return (api && typeof api.apiURL === "function") ? api.apiURL(q) : q;
    }

    const PREVIEW_NODE = "YuanPanoramaPreview";
    const PREVIEW_MIN_HEIGHT = 200;
    const PREVIEW_MIN_WIDTH = 240;

    const DEG2RAD = Math.PI / 180;
    const TWO_PI = Math.PI * 2;
    const PI = Math.PI;

    // 交互参数（复刻 GJJ 360 全景浏览器的操作手感）
    const DRAG_SENSITIVITY = 0.34;   // deg / px（≈ GJJ 的 0.006 rad/px，固定灵敏度）
    const WHEEL_ZOOM_OUT = 1.08;     // 滚轮放大方向（deltaY>0）的乘数
    const WHEEL_ZOOM_IN = 0.92;      // 滚轮缩小方向（deltaY<0）的乘数
    const FOV_MIN = 22.5;            // ≈ π/8 rad
    const FOV_MAX = 165.6;           // ≈ π*0.92 rad
    const FOV_DEFAULT = 82;          // ≈ π/2.2 rad（GJJ 默认 FOV）
    const PITCH_LIMIT = 88.3;        // ≈ π/2 - 0.03 rad（GJJ 极点安全余量）
    const SYNC_DEBOUNCE_MS = 150;    // 视图数据同步防抖
    const DPR_MAX = 1.5;             // 限制 DPR 以降低 GPU 负载

    // ------------------------------------------------------------------ //
    // 小工具
    // ------------------------------------------------------------------ //
    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

    function wrapYaw(deg) {
        deg = ((deg % 360) + 360) % 360;
        if (deg > 180) deg -= 360;
        return deg;
    }

    function CoverageFromNode(node) {
        const w = node?.widgets?.find?.((x) => x?.name === "Coverage");
        const v = String(w?.value ?? "360").trim();
        return v === "180" ? 180 : 360;
    }

    function setWidgetValue(node, name, value) {
        const widget = node?.widgets?.find?.((w) => w?.name === name);
        if (!widget) return;
        widget.value = value;
        const index = node.widgets.indexOf(widget);
        if (Array.isArray(node.widgets_values) && index >= 0) node.widgets_values[index] = value;
        try { widget.callback?.(value); } catch (_) {}
        // 仅刷新背景，避免在拖拽过程中触发整图重绘造成卡顿
        app.graph?.setDirtyCanvas?.(false, true);
    }

    function comfyImageEntryToUrl(entry) {
        if (!entry || typeof entry !== "object") return "";
        const filename = String(entry.filename || "").trim();
        if (!filename) return "";
        const params = new URLSearchParams();
        params.set("filename", filename);
        const viewType = String(
            entry.storage
            || (String(entry.type || "").trim().toLowerCase() === "comfy_image" ? "output" : entry.type)
            || "output"
        );
        params.set("type", viewType);
        if (entry.subfolder) params.set("subfolder", String(entry.subfolder));
        const q = `/view?${params.toString()}`;
        return apiUrl(q);
    }

    function imageSourceFromCandidate(candidate) {
        if (!candidate) return "";
        if (typeof candidate === "string") return String(candidate).trim();
        if (Array.isArray(candidate)) {
            if (candidate.length === 0) return "";
            if (candidate.length === 1) return imageSourceFromCandidate(candidate[0]);
            const filename = typeof candidate[0] === "string" ? String(candidate[0]).trim() : "";
            if (filename) {
                return comfyImageEntryToUrl({
                    filename,
                    subfolder: String(candidate[1] || "").trim(),
                    type: String(candidate[2] || "output").trim() || "output",
                });
            }
            for (const e of candidate) {
                const s = imageSourceFromCandidate(e);
                if (s) return s;
            }
            return "";
        }
        if (typeof candidate?.src === "string" && candidate.src) return candidate.src;
        if (typeof candidate?.url === "string" && candidate.url) return candidate.url;
        return comfyImageEntryToUrl(candidate);
    }

    function lookupNodeOutputEntry(nodeId) {
        const store = app?.nodeOutputs;
        if (!store || nodeId == null) return null;
        const raw = String(nodeId);
        if (store instanceof Map) {
            return store.get(nodeId) || store.get(raw) || store.get(Number(raw)) || null;
        }
        return store[nodeId] || store[raw] || null;
    }

    // 从节点自身输出中找视频
    function getSelfVideoUrl(node) {
        const outputs = lookupNodeOutputEntry(node?.id);
        const groups = [
            outputs?.ui?.pano_videos,
            outputs?.pano_videos,
        ];
        for (const g of groups) {
            if (!Array.isArray(g)) continue;
            for (const c of g) {
                const src = imageSourceFromCandidate(c);
                if (src && /\.mp4(\?|$)/i.test(src)) return src;
                if (src && String(c?.format || "").toLowerCase() === "video/mp4") return src;
            }
        }
        return "";
    }

    // 从节点自身输出或上游连接节点找图像
    function getLinkedImageUrl(node, imageInputName) {
        imageInputName = imageInputName || "ERP_image";

        const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
        let linkId = null;
        const preferred = inputs.find((i) => String(i?.name || "") === String(imageInputName));
        if (preferred?.link != null) linkId = preferred.link;
        if (linkId == null) {
            const anyImg = inputs.find((i) => String(i?.type || "").toUpperCase() === "IMAGE" && i?.link != null);
            if (anyImg?.link != null) linkId = anyImg.link;
        }

        let originId = null;
        if (linkId != null) {
            const link = node?.graph?.links?.[linkId] || app?.graph?.links?.[linkId];
            if (link) {
                const oid = Number(link.origin_id);
                if (Number.isFinite(oid)) {
                    originId = oid;
                    const originNode = app?.graph?.getNodeById?.(originId);

                    // 1) 优先检查当前图中的实际节点（避免跨工作流 ID 冲突导致的 nodeOutputs 污染）
                    if (originNode) {
                        const imageWidget = originNode?.widgets?.find?.((w) => String(w?.name || "").toLowerCase() === "image");
                        const isLoader = originNode.type === "LoadImage" || originNode.comfyClass === "LoadImage" || String(originNode.type || "").toLowerCase().includes("load");

                        const extractWidgetUrl = (widget) => {
                            const imageName = String(widget?.value || "").trim();
                            if (!imageName) return "";
                            // 加载类节点的 widget 图像必定在 input 目录下（哪怕它的文件名叫 ComfyUI_temp_xxx）
                            let type = "input";
                            let file = imageName;
                            let subfolder = "";
                            let slashIdx = file.lastIndexOf("/");
                            if (slashIdx > -1) {
                                subfolder = file.substring(0, slashIdx);
                                file = file.substring(slashIdx + 1);
                            } else {
                                slashIdx = file.lastIndexOf("\\");
                                if (slashIdx > -1) {
                                    subfolder = file.substring(0, slashIdx);
                                    file = file.substring(slashIdx + 1);
                                }
                            }
                            return apiUrl(`/view?filename=${encodeURIComponent(file)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`);
                        };

                        // 1.1) 对于图像加载类节点，优先通过 widget 提取。
                        // 因为加载节点未执行时，其 imgs 可能会被上一个工作流同 ID 节点的缓存污染，而 widget 始终保存着用户当前选择的图片。
                        if (isLoader && imageWidget && typeof imageWidget.value === "string") {
                            const url = extractWidgetUrl(imageWidget);
                            if (url) return url;
                        }

                        // 1.2) 其次使用节点自己显示的缩略图（反映当前节点执行后的 UI 状态）
                        const imgs = Array.isArray(originNode?.imgs) ? originNode.imgs : [];
                        for (const c of imgs) {
                            const s = imageSourceFromCandidate(c);
                            if (s) return s;
                        }

                        // 1.3) 对于非加载节点，但拥有 image widget 的情况（后备降级）
                        if (!isLoader && imageWidget && typeof imageWidget.value === "string") {
                            const url = extractWidgetUrl(imageWidget);
                            if (url) return url;
                        }

                        // 1.4) 最后才尝试 getNodeImageUrls（因为它会读取 nodeOutputs，可能被跨工作流的同 ID 节点污染）
                        let urls = [];
                        try {
                            urls = typeof app?.getNodeImageUrls === "function" ? (app.getNodeImageUrls(originNode) || []) : [];
                        } catch (_) { urls = []; }
                        for (const c of urls) {
                            const s = imageSourceFromCandidate(c);
                            if (s) return s;
                        }
                    }

                    // 2) 检查 originOutputs（可能是当前图执行结果，也可能是被污染的缓存）
                    const originOutputs = lookupNodeOutputEntry(oid);
                    if (originOutputs) {
                        const candidateGroups = [
                            originOutputs?.images,
                            originOutputs?.ui?.images,
                            originOutputs?.ui?.pano_input_images,
                            originOutputs?.pano_input_images,
                        ];
                        for (const g of candidateGroups) {
                            if (!Array.isArray(g)) continue;
                            for (const c of g) {
                                const s = imageSourceFromCandidate(c);
                                if (s) return s;
                            }
                        }
                    }
                }
            }
        }

        // 3) 自身预览图（后备：仅在无上游输出时使用）
        const selfOutput = lookupNodeOutputEntry(node?.id);
        const selfGroups = [
            selfOutput?.ui?.pano_input_images,
            selfOutput?.pano_input_images,
        ];
        for (const g of selfGroups) {
            if (!Array.isArray(g)) continue;
            for (const c of g) {
                const s = imageSourceFromCandidate(c);
                if (s) return s;
            }
        }

        return "";
    }

    // ------------------------------------------------------------------ //
    // WebGL 渲染器（球面投影 raycasting）
    // ------------------------------------------------------------------ //
    const VERT_SRC = `
        attribute vec2 aPos;
        varying vec2 vUv;
        void main() {
            vUv = aPos * 0.5 + 0.5;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
    `;

    const FRAG_SRC = `
        precision highp float;
        varying vec2 vUv;
        uniform sampler2D uTex;
        uniform float uYaw;      // radians
        uniform float uPitch;    // radians
        uniform float uFov;      // vertical fov radians
        uniform float uAspect;   // canvas w/h
        uniform float uCoverage; // 360.0 or 180.0
        uniform float uReady;
        uniform vec3 uBg;

        vec3 dirFromYawPitch(float yaw, float pitch) {
            float cp = cos(pitch);
            return vec3(cp * sin(yaw), sin(pitch), cp * cos(yaw));
        }

        void main() {
            if (uReady < 0.5) {
                gl_FragColor = vec4(uBg, 1.0);
                return;
            }
            vec3 forward = dirFromYawPitch(uYaw, uPitch);
            vec3 worldUp = vec3(0.0, 1.0, 0.0);
            vec3 right = cross(worldUp, forward);
            if (length(right) < 1e-6) right = vec3(1.0, 0.0, 0.0);
            right = normalize(right);
            vec3 up = normalize(cross(forward, right));

            vec2 ndc = vUv * 2.0 - 1.0;
            float tanHalf = tan(uFov * 0.5);
            vec3 ray = normalize(forward
                + right * (ndc.x * tanHalf * uAspect)
                + up * (ndc.y * tanHalf));

            float yaw = atan(ray.x, ray.z);
            float pitch = asin(clamp(ray.y, -1.0, 1.0));
            float u;
            if (uCoverage > 270.0) {
                u = (yaw / 6.28318530718) + 0.5;
                u = fract(u);
            } else {
                u = (yaw / 3.14159265359) + 0.5;
            }
            float v = (pitch / 3.14159265359) + 0.5;
            v = clamp(v, 0.0, 1.0);

            if (uCoverage <= 270.0 && (u < 0.0 || u > 1.0)) {
                gl_FragColor = vec4(uBg, 1.0);
                return;
            }
            gl_FragColor = texture2D(uTex, vec2(u, v));
        }
    `;

    function createShader(gl, type, src) {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
            const info = gl.getShaderInfoLog(sh);
            gl.deleteShader(sh);
            throw new Error("shader compile failed: " + info);
        }
        return sh;
    }

    function createProgram(gl) {
        const vs = createShader(gl, gl.VERTEX_SHADER, VERT_SRC);
        const fs = createShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            throw new Error("program link failed: " + gl.getProgramInfoLog(prog));
        }
        return prog;
    }

    class PanoRenderer {
        constructor(canvas) {
            this.canvas = canvas;
            const gl = canvas.getContext("webgl", { antialias: true, premultipliedAlpha: false })
                || canvas.getContext("experimental-webgl");
            if (!gl) throw new Error("WebGL 不可用");
            this.gl = gl;
            this.program = createProgram(gl);
            this.loc = {
                aPos: gl.getAttribLocation(this.program, "aPos"),
                uTex: gl.getUniformLocation(this.program, "uTex"),
                uYaw: gl.getUniformLocation(this.program, "uYaw"),
                uPitch: gl.getUniformLocation(this.program, "uPitch"),
                uFov: gl.getUniformLocation(this.program, "uFov"),
                uAspect: gl.getUniformLocation(this.program, "uAspect"),
                uCoverage: gl.getUniformLocation(this.program, "uCoverage"),
                uReady: gl.getUniformLocation(this.program, "uReady"),
                uBg: gl.getUniformLocation(this.program, "uBg"),
            };
            // 全屏 quad
            this.buffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
                -1, -1, 1, -1, -1, 1,
                -1, 1, 1, -1, 1, 1,
            ]), gl.STATIC_DRAW);

            this.texture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, this.texture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            this.hasTexture = false;
        }

        upload(media) {
            const gl = this.gl;
            if (!media) {
                this.hasTexture = false;
                return;
            }
            gl.bindTexture(gl.TEXTURE_2D, this.texture);
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            try {
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, media);
                this.hasTexture = true;
            } catch (e) {
                this.hasTexture = false;
            }
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        }

        render(view, CoverageDeg) {
            const gl = this.gl;
            const w = this.canvas.width;
            const h = this.canvas.height;
            if (w <= 0 || h <= 0) return;
            gl.viewport(0, 0, w, h);
            gl.clearColor(0.027, 0.027, 0.027, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.useProgram(this.program);
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
            gl.enableVertexAttribArray(this.loc.aPos);
            gl.vertexAttribPointer(this.loc.aPos, 2, gl.FLOAT, false, 0, 0);

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.texture);
            gl.uniform1i(this.loc.uTex, 0);

            gl.uniform1f(this.loc.uYaw, Number(view.yaw) * DEG2RAD);
            gl.uniform1f(this.loc.uPitch, Number(view.pitch) * DEG2RAD);
            gl.uniform1f(this.loc.uFov, Number(view.fov) * DEG2RAD);
            gl.uniform1f(this.loc.uAspect, w / Math.max(1, h));
            gl.uniform1f(this.loc.uCoverage, CoverageDeg);
            gl.uniform1f(this.loc.uReady, this.hasTexture ? 1.0 : 0.0);
            gl.uniform3f(this.loc.uBg, 0.027, 0.027, 0.027);

            gl.drawArrays(gl.TRIANGLES, 0, 6);
        }

        screenshot(media, view, CoverageDeg, outWidth = 0, outHeight = 0) {
            if (!media || !this.hasTexture) return "";
            const w = Math.max(1, Math.round(Number(outWidth) || 0));
            const h = Math.max(1, Math.round(Number(outHeight) || 0));
            const out = document.createElement("canvas");
            out.width = w;
            out.height = h;
            let renderer = null;
            try {
                renderer = new PanoRenderer(out);
                renderer.upload(media);
                renderer.render(view, CoverageDeg);
                return out.toDataURL("image/png");
            } catch (_) {
                return "";
            } finally {
                renderer?.dispose?.();
            }
        }

        dispose() {
            const gl = this.gl;
            try {
                if (this.buffer) gl.deleteBuffer(this.buffer);
                if (this.texture) gl.deleteTexture(this.texture);
                if (this.program) gl.deleteProgram(this.program);
            } catch (_) {}
            this.buffer = null;
            this.texture = null;
            this.program = null;
        }
    }

    // ------------------------------------------------------------------ //
    // 预览运行时
    // ------------------------------------------------------------------ //
    function isRenderableMediaReady(media) {
        if (!media) return false;
        if (media instanceof HTMLVideoElement) {
            return Number(media.videoWidth || 0) > 0 && Number(media.readyState || 0) >= 2;
        }
        return !!media.complete && Number(media.naturalWidth || media.width || 0) > 0;
    }

    class PreviewRuntime {
        constructor(node) {
            this.node = node;
            this.root = null;
            this.canvas = null;
            this.renderer = null;
            this.widget = null;
            this.resizeObserver = null;
            this.rafId = 0;
            this.img = null;
            this.imgSrc = "";
            this.mediaCleanup = null;
            this.videoPaused = false;
            this.view = { yaw: 0, pitch: 0, fov: FOV_DEFAULT };
            this.Coverage = CoverageFromNode(node);
            this.dragging = false;
            this.lastX = 0;
            this.lastY = 0;
            this.needsDraw = false;
            this.inTick = false;
            this.queued = false;
            this.outputCurrentView = false;
            this.syncTimer = 0;
            this.orig = {
                onExecuted: node.onExecuted,
                onConnectionsChange: node.onConnectionsChange,
                onRemoved: node.onRemoved,
                onResize: node.onResize,
                CoverageCallback: null,
                ocvCallback: null,
            };
            this.tick = this.tick.bind(this);
            this.onResizeDom = this.onResizeDom.bind(this);
        }

        attach() {
            this._hideHiddenWidgets();
            this._buildDom();
            this._installHooks();
            this.refreshMedia();
            this.requestDraw();
        }

        _hideHiddenWidgets() {
            // 隐藏 current_view_data 输入项，不在节点上显示
            const names = ["current_view_data"];
            for (const name of names) {
                const widget = this.node?.widgets?.find?.((w) => w?.name === name);
                if (!widget || widget.__yuanHidden) continue;
                widget.__yuanHidden = true;
                widget.hidden = true;
                widget.options ||= {};
                widget.options.hidden = true;
                widget.options.display = "hidden";
                widget.type = "hidden";
                widget.computeSize = () => [0, -4];
                widget.getHeight = () => 0;
                widget.draw = () => {};
                widget.mouse = () => false;
            }
        }

        _buildDom() {
            try {
                this.root = document.createElement("div");
                this.root.className = "yuan-pano-preview";
                this.root.setAttribute("data-capture-wheel", "true");
                this.root.setAttribute("tabindex", "0");
                this.root.style.cssText = [
                    "width:100%",
                    "height:100%",
                    `min-height:${PREVIEW_MIN_HEIGHT}px`,
                    "position:relative",
                    "display:block",
                    "overflow:hidden",
                    "background:#070707",
                    "border-radius:8px",
                    "border:1px solid rgba(63,63,70,1)",
                ].join(";");

                this.canvas = document.createElement("canvas");
                this.canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;cursor:grab;";
                this.root.appendChild(this.canvas);

                const widgetOptions = {
                    serialize: false,
                    hideOnZoom: false,
                    getValue() { return ""; },
                    setValue() {},
                    getMinHeight() { return PREVIEW_MIN_HEIGHT; },
                    getHeight() { return PREVIEW_MIN_HEIGHT; },
                    onRemove: () => this.teardown(),
                    afterResize: () => this.requestDraw(),
                };

                this.widget = this.node.addDOMWidget("preview", "preview", this.root, widgetOptions);
                if (this.widget) {
                    this.widget.serialize = false;
                    const prev = typeof this.widget.computeLayoutSize === "function"
                        ? this.widget.computeLayoutSize.bind(this.widget) : null;
                    this.widget.computeLayoutSize = (targetNode) => {
                        const p = prev ? (prev(targetNode) || {}) : {};
                        return {
                            ...p,
                            minHeight: Math.max(PREVIEW_MIN_HEIGHT, Number(p.minHeight || 0)),
                            minWidth: Math.max(PREVIEW_MIN_WIDTH, Number(p.minWidth || 0)),
                        };
                    };
                }

                try {
                    this.renderer = new PanoRenderer(this.canvas);
                } catch (e) {
                    this.renderer = null;
                }

                if (typeof ResizeObserver !== "undefined") {
                    this.resizeObserver = new ResizeObserver(() => this.onResizeDom());
                    this.resizeObserver.observe(this.root);
                }
                this._syncViewData();
                this._bindInput();
                this.onResizeDom();
            } catch (e) {
                // 降级：addDOMWidget 不可用时，挂一个静态 onDrawForeground
                this.root = null;
                this._installLegacyForeground();
            }
        }

        screenshotDataUrl() {
            if (!this.renderer || !isRenderableMediaReady(this.img)) return "";
            const wWidget = this.node?.widgets?.find?.((w) => w?.name === "view_width");
            const hWidget = this.node?.widgets?.find?.((w) => w?.name === "view_height");
            const w = Math.max(1, Math.round(Number(wWidget?.value) || 1024));
            const h = Math.max(1, Math.round(Number(hWidget?.value) || 512));
            const off = this._getOffscreenRenderer(w, h);
            if (!off) return "";
            if (this._screenshotMedia !== this.img) {
                off.renderer.upload(this.img);
                this._screenshotMedia = this.img;
            }
            off.renderer.render(this.view, this.Coverage);
            return off.canvas.toDataURL("image/png");
        }

        _getOffscreenRenderer(w, h) {
            if (!this._screenshotCanvas) {
                this._screenshotCanvas = document.createElement("canvas");
            }
            if (this._screenshotCanvas.width !== w || this._screenshotCanvas.height !== h) {
                this._screenshotCanvas.width = w;
                this._screenshotCanvas.height = h;
                // 尺寸变化时强制重新上传纹理
                this._screenshotMedia = null;
                if (this._screenshotRenderer) {
                    try { this._screenshotRenderer.dispose?.(); } catch (_) {}
                    this._screenshotRenderer = null;
                }
            }
            if (!this._screenshotRenderer) {
                try {
                    this._screenshotRenderer = new PanoRenderer(this._screenshotCanvas);
                } catch (e) {
                    this._screenshotRenderer = null;
                    return null;
                }
            }
            return { canvas: this._screenshotCanvas, renderer: this._screenshotRenderer };
        }

        _disposeScreenshotRenderer() {
            if (this._screenshotRenderer) {
                try { this._screenshotRenderer.dispose(); } catch (_) {}
                this._screenshotRenderer = null;
                this._screenshotCanvas = null;
                this._screenshotMedia = null;
            }
        }

        _syncViewData() {
            // output_current_view widget 即可见开关：True=全景模式，False=裁剪模式
            const widget = this.node?.widgets?.find?.((w) => w?.name === "output_current_view");
            this.outputCurrentView = Boolean(widget?.value);
            // 静默设置 current_view_data：不触发 widget.callback 和 setDirtyCanvas，
            // 避免大量 dataUrl 字符串引发 ComfyUI/LiteGraph 持续重绘或处理
            const dataWidget = this.node?.widgets?.find?.((w) => w?.name === "current_view_data");
            if (!dataWidget) return;
            const idx = this.node.widgets.indexOf(dataWidget);
            if (this.outputCurrentView) {
                // 全景模式：不需要截图数据
                dataWidget.value = "";
                if (Array.isArray(this.node.widgets_values) && idx >= 0) this.node.widgets_values[idx] = "";
                return;
            }
            // 裁剪模式：截取当前 3D 裁剪画面
            const dataUrl = this.screenshotDataUrl();
            if (!dataUrl) {
                // 如果当前无法截图（例如媒体尚未加载、或切换工作流时暂无图像），
                // 不要清空已有的 current_view_data，以免丢失先前保存的裁剪画面
                return;
            }
            dataWidget.value = dataUrl;
            if (Array.isArray(this.node.widgets_values) && idx >= 0) this.node.widgets_values[idx] = dataUrl;
        }

        scheduleViewSync() {
            if (this.outputCurrentView) return;
            window.clearTimeout(this.syncTimer);
            this.syncTimer = window.setTimeout(() => this._syncViewData(), SYNC_DEBOUNCE_MS);
        }

        _bindInput() {
            const canvas = this.canvas;
            const root = this.root;
            if (!canvas || !root) return;

            canvas.addEventListener("pointerdown", (ev) => {
                if (ev.button !== 0) return;
                // 释放离屏截图渲染器，避免拖拽期间存在双 WebGL 上下文导致 GPU 资源争用
                this._disposeScreenshotRenderer();
                // 取消待执行的视图同步（如模式切换后延迟截图），确保拖拽期间不触发截图
                window.clearTimeout(this.syncTimer);
                this.syncTimer = 0;
                root.focus?.({ preventScroll: true });
                canvas.setPointerCapture?.(ev.pointerId);
                canvas.style.cursor = "grabbing";
                this.dragging = true;
                this.lastX = ev.clientX;
                this.lastY = ev.clientY;
                ev.preventDefault();
                ev.stopPropagation();
            });
            canvas.addEventListener("pointermove", (ev) => {
                if (!this.dragging) return;
                const dx = ev.clientX - this.lastX;
                const dy = ev.clientY - this.lastY;
                this.lastX = ev.clientX;
                this.lastY = ev.clientY;
                // 复刻 GJJ：grab 约定，拖右 → 视角左移；拖下 → 视角上抬
                const dYaw = -dx * DRAG_SENSITIVITY;
                const dPitch = dy * DRAG_SENSITIVITY;
                this._applyDelta(dYaw, dPitch);
                this.requestDraw();
                // 拖拽中不同步视图数据：WebGL canvas 的 toDataURL 会引发 GPU→CPU 管线停顿，
                // 阻塞主线程导致掉帧卡顿。仅在 pointerup / wheel 时同步。
                ev.preventDefault();
                ev.stopPropagation();
            });
            const end = (ev) => {
                if (!this.dragging) return;
                this.dragging = false;
                canvas.releasePointerCapture?.(ev.pointerId);
                canvas.style.cursor = "grab";
                this.requestDraw();
                this.scheduleViewSync();
                ev.preventDefault();
                ev.stopPropagation();
            };
            canvas.addEventListener("pointerup", end);
            canvas.addEventListener("pointercancel", end);
            root.addEventListener("wheel", (ev) => {
                const delta = Number(ev.deltaY ?? ev.wheelDeltaY ?? 0);
                if (delta !== 0) {
                    // 复刻 GJJ：乘法缩放，deltaY>0 放大（×1.08），deltaY<0 缩小（×0.92）
                    const factor = delta > 0 ? WHEEL_ZOOM_OUT : WHEEL_ZOOM_IN;
                    this.view.fov = clamp(this.view.fov * factor, FOV_MIN, FOV_MAX);
                    this.requestDraw();
                    this.scheduleViewSync();
                }
                ev.preventDefault();
                ev.stopPropagation();
            }, { passive: false, capture: true });
            canvas.addEventListener("dblclick", (ev) => {
                this.togglePlayback();
                ev.preventDefault();
                ev.stopPropagation();
            });
            root.addEventListener("keydown", (ev) => {
                if (ev.key === " " || ev.key === "Spacebar") {
                    this.togglePlayback();
                    ev.preventDefault();
                    ev.stopPropagation();
                }
            });
        }

        _applyDelta(dYaw, dPitch) {
            // 复刻 GJJ：固定灵敏度，无动态缩放，直接 1:1 映射
            this.view.yaw = this._wrapYaw(this.view.yaw + dYaw);
            this.view.pitch = clamp(this.view.pitch + dPitch, -PITCH_LIMIT, PITCH_LIMIT);
        }

        _wrapYaw(deg) {
            if (this.Coverage === 180) {
                return clamp(deg, -90, 90);
            }
            return wrapYaw(deg);
        }

        togglePlayback() {
            if (!(this.img instanceof HTMLVideoElement) || !isRenderableMediaReady(this.img)) return;
            if (this.img.paused) {
                this.videoPaused = false;
                void this.img.play().catch(() => {});
            } else {
                this.videoPaused = true;
                this.img.pause();
            }
            this.requestDraw();
        }

        _installHooks() {
            const self = this;
            this.node.onExecuted = function () {
                const out = self.orig.onExecuted ? self.orig.onExecuted.apply(this, arguments) : undefined;
                self.Coverage = CoverageFromNode(self.node);
                self.refreshMedia();
                return out;
            };
            this.node.onConnectionsChange = function () {
                const out = self.orig.onConnectionsChange ? self.orig.onConnectionsChange.apply(this, arguments) : undefined;
                self.refreshMedia();
                return out;
            };
            this.node.onResize = function () {
                const out = self.orig.onResize ? self.orig.onResize.apply(this, arguments) : undefined;
                self.requestDraw();
                return out;
            };
            this.node.onRemoved = function () {
                const out = self.orig.onRemoved ? self.orig.onRemoved.apply(this, arguments) : undefined;
                self.teardown();
                return out;
            };
            const covWidget = this.node?.widgets?.find?.((w) => w?.name === "Coverage");
            if (covWidget) {
                this.orig.CoverageCallback = typeof covWidget.callback === "function"
                    ? covWidget.callback.bind(covWidget) : null;
                covWidget.callback = (...args) => {
                    const out = self.orig.CoverageCallback ? self.orig.CoverageCallback(...args) : undefined;
                    self.Coverage = CoverageFromNode(self.node);
                    self.view.yaw = self._wrapYaw(self.view.yaw);
                    self.requestDraw();
                    self.scheduleViewSync();
                    return out;
                };
            }
            // 监听 output_current_view widget：True=全景模式，False=裁剪模式
            const ocvWidget = this.node?.widgets?.find?.((w) => w?.name === "output_current_view");
            if (ocvWidget) {
                this.orig.ocvCallback = typeof ocvWidget.callback === "function"
                    ? ocvWidget.callback.bind(ocvWidget) : null;
                ocvWidget.callback = (...args) => {
                    const out = self.orig.ocvCallback ? self.orig.ocvCallback(...args) : undefined;
                    // 立即更新模式标志，但延迟截图（scheduleViewSync），
                    // 避免模式切换瞬间同步截图导致后续拖拽卡顿
                    const ocvW = self.node?.widgets?.find?.((w) => w?.name === "output_current_view");
                    self.outputCurrentView = Boolean(ocvW?.value);
                    self.scheduleViewSync();
                    return out;
                };
            }
        }

        _installLegacyForeground() {
            const self = this;
            this.node.onDrawForeground = function (ctx) {
                if (!ctx || this.flags?.collapsed) return;
                const w = Math.max(80, Number(this.size?.[0] || 0) - 20);
                const h = Math.max(60, Number(this.size?.[1] || 0) - 50);
                ctx.save();
                ctx.fillStyle = "#070707";
                ctx.fillRect(10, 40, w, h);
                ctx.fillStyle = "rgba(236,236,242,0.7)";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("全景预览（需要 DOM widget 支持）", 10 + w * 0.5, 40 + h * 0.5);
                ctx.restore();
            };
        }

        onResizeDom() {
            if (!this.root || !this.canvas) return;
            const rect = this.root.getBoundingClientRect();
            const dpr = Math.min(window.devicePixelRatio || 1, DPR_MAX);
            const w = Math.max(1, Math.round(rect.width * dpr));
            const h = Math.max(1, Math.round(rect.height * dpr));
            if (this.canvas.width !== w || this.canvas.height !== h) {
                this.canvas.width = w;
                this.canvas.height = h;
                this.requestDraw();
            }
        }

        refreshMedia() {
            if (!this.root) return;
            const nextVideo = getSelfVideoUrl(this.node);
            if (nextVideo) {
                if (nextVideo === this.imgSrc && this.img instanceof HTMLVideoElement) return;
                this.mediaCleanup?.();
                this.imgSrc = nextVideo;
                this.videoPaused = false;
                const video = document.createElement("video");
                video.muted = true;
                video.loop = true;
                video.playsInline = true;
                video.crossOrigin = "anonymous";
                const onReady = () => {
                    if (this.imgSrc !== nextVideo) return;
                    this.img = video;
                    if (this.renderer) this.renderer.upload(video);
                    if (!this.videoPaused) void video.play().catch(() => {});
                    this.requestDraw();
                    // 图像就绪后同步裁剪视图数据，避免切换工作流回来后
                    // current_view_data 为空导致后端输出原图
                    this.scheduleViewSync();
                };
                const onTick = () => this.requestDraw();
                video.addEventListener("loadedmetadata", onReady);
                video.addEventListener("canplay", onReady);
                video.addEventListener("timeupdate", onTick);
                video.addEventListener("play", onTick);
                video.addEventListener("pause", onTick);
                video.src = nextVideo;
                video.load();
                this.mediaCleanup = () => {
                    try { video.pause(); } catch (_) {}
                    video.removeEventListener("loadedmetadata", onReady);
                    video.removeEventListener("canplay", onReady);
                    video.removeEventListener("timeupdate", onTick);
                    video.removeEventListener("play", onTick);
                    video.removeEventListener("pause", onTick);
                };
                return;
            }

            const nextSrc = getLinkedImageUrl(this.node);
            if (!nextSrc) {
                this.mediaCleanup?.();
                this.mediaCleanup = null;
                this.img = null;
                this.imgSrc = "";
                if (this.renderer) this.renderer.hasTexture = false;
                this.requestDraw();
                return;
            }
            if (nextSrc === this.imgSrc && this.img) return;
            this.mediaCleanup?.();
            this.mediaCleanup = null;
            this.imgSrc = nextSrc;
            const image = new Image();
            image.crossOrigin = "anonymous";
            image.onload = () => {
                if (this.imgSrc !== nextSrc) return;
                this.img = image;
                if (this.renderer) this.renderer.upload(image);
                this.requestDraw();
                // 图像就绪后同步裁剪视图数据，避免切换工作流回来后
                // current_view_data 为空导致后端输出原图
                this.scheduleViewSync();
            };
            image.onerror = () => {
                if (this.imgSrc !== nextSrc) return;
                this.img = null;
                if (this.renderer) this.renderer.hasTexture = false;
                this.requestDraw();
            };
            image.src = nextSrc;
        }

        requestDraw() {
            this.needsDraw = true;
            if (this.inTick) {
                this.queued = true;
                return;
            }
            if (!this.rafId) this.rafId = requestAnimationFrame(this.tick);
        }

        tick(ts) {
            ts = ts || performance.now();
            this.rafId = 0;
            this.inTick = true;
            this.queued = false;
            this.needsDraw = false;

            // 视频帧更新纹理
            if (this.img instanceof HTMLVideoElement && isRenderableMediaReady(this.img) && !this.img.paused) {
                if (this.renderer) this.renderer.upload(this.img);
            }

            if (this.renderer) {
                this.renderer.render(this.view, this.Coverage);
            } else if (this.node?.setDirtyCanvas) {
                this.node.setDirtyCanvas(true, false);
            }

            this.inTick = false;
            // 复刻 GJJ：无惯性，仅在需要重绘或视频播放时持续 RAF
            const shouldContinue = this.needsDraw
                || this.queued
                || (this.img instanceof HTMLVideoElement && !this.img.paused && !this.img.ended);
            if (shouldContinue && !this.rafId) this.rafId = requestAnimationFrame(this.tick);
        }

        teardown() {
            if (this.node?._yuanPanoRuntime !== this) return;
            if (this.rafId) {
                cancelAnimationFrame(this.rafId);
                this.rafId = 0;
            }
            try { this.resizeObserver?.disconnect?.(); } catch (_) {}
            this.resizeObserver = null;
            this.mediaCleanup?.();
            this.mediaCleanup = null;
            window.clearTimeout(this.syncTimer);
            this.syncTimer = 0;
            this._disposeScreenshotRenderer();
            try { this.root?.remove?.(); } catch (_) {}
            if (Array.isArray(this.node?.widgets) && this.widget) {
                try {
                    this.node.widgets = this.node.widgets.filter((w) => w !== this.widget);
                } catch (_) {}
            }
            this.renderer?.dispose?.();
            this.renderer = null;
            this.node.onExecuted = this.orig.onExecuted;
            this.node.onConnectionsChange = this.orig.onConnectionsChange;
            this.node.onRemoved = this.orig.onRemoved;
            this.node.onResize = this.orig.onResize;
            const covWidget = this.node?.widgets?.find?.((w) => w?.name === "Coverage");
            if (covWidget) covWidget.callback = this.orig.CoverageCallback;
            const ocvWidget = this.node?.widgets?.find?.((w) => w?.name === "output_current_view");
            if (ocvWidget) ocvWidget.callback = this.orig.ocvCallback;
            this.node._yuanPanoRuntime = null;
        }
    }

    function attachPreview(node) {
        if (!node || node._yuanPanoRuntime) return;
        const runtime = new PreviewRuntime(node);
        node._yuanPanoRuntime = runtime;
        runtime.attach();
    }

    // ------------------------------------------------------------------ //
    // 注册扩展
    // ------------------------------------------------------------------ //
    app.registerExtension({
    name: "ComfyUI-Yuan-Tool.Panorama",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== PREVIEW_NODE) return;
        // 复刻 GJJ 360 全景生成器：节点自带 WebGL 预览，关闭默认输出预览避免重复
        nodeData.output_preview = false;
        if (Array.isArray(nodeData.outputs)) {
            for (const output of nodeData.outputs) output.preview = false;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                attachPreview(this);
                return r;
            };
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                if (this._yuanPanoRuntime) {
                    this._yuanPanoRuntime.Coverage = CoverageFromNode(this);
                    this._yuanPanoRuntime._syncViewData?.();
                    this._yuanPanoRuntime.refreshMedia?.();
                    this._yuanPanoRuntime.requestDraw?.();
                }
                return r;
            };
        },
    });
})();
