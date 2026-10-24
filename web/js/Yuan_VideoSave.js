/** Yuan Tool · 视频保存前端：高级设置折叠、格式参数显隐与播放预览窗口。 */
import { chainCallback, findComfyNodeEl, getApi, hideWidget, showWidget } from "./Yuan_Common.js";

const { app } = window.comfyAPI.app;
const api = getApi();

const NODE_NAME = "Yuan_VideoSave";

// 后端按格式配置生成的参数控件，由当前输出格式决定显隐
const FORMAT_WIDGET_NAMES = [
    "画质",
    "像素格式",
    "写入元数据",
    "按音频长度裁剪",
    "抖动算法",
    "ProRes 档位",
    "码率",
    "码率单位",
    "输入色深",
    "FFV1 级别",
    "FFV1 编码器",
    "FFV1 上下文",
    "GOP 帧间隔",
    "切片数",
    "切片校验",
    "无损 WebP",
    "Gifski 质量",
];

const ADVANCED_WIDGET_NAMES = ["循环次数", "输出格式", "往返播放"];

// 兼容早期独立版节点按英文参数名保存的 widgets_values（对象形式）
const LEGACY_WIDGET_ALIASES = {
    crf: "画质",
    pix_fmt: "像素格式",
    save_metadata: "写入元数据",
    trim_to_audio: "按音频长度裁剪",
    dither: "抖动算法",
    profile: "ProRes 档位",
    bitrate: "码率",
    megabit: "码率单位",
    input_color_depth: "输入色深",
    level: "FFV1 级别",
    coder: "FFV1 编码器",
    context: "FFV1 上下文",
    gop_size: "GOP 帧间隔",
    slices: "切片数",
    slicecrc: "切片校验",
    lossless: "无损 WebP",
    quality: "Gifski 质量",
};

const VIDEO_EXTENSIONS = new Set([
    "mp4", "webm", "mov", "m4v", "mkv", "avi", "flv", "wmv", "ogv",
]);

let formatRequest = null;

/** 读取后端格式定义（控件名、下拉选项与默认值），每个页面只请求一次。 */
function fetchFormats() {
    if (!formatRequest) {
        formatRequest = (async () => {
            try {
                const response = await api.fetchApi("/yuan_tool/video_save_formats");
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                return await response.json();
            } catch (error) {
                console.warn("[Yuan 视频保存] 读取输出格式定义失败：", error);
                formatRequest = null;
                return null;
            }
        })();
    }
    return formatRequest;
}

function extensionOf(filename) {
    const name = String(filename || "");
    const dot = name.lastIndexOf(".");
    return dot >= 0 ? name.slice(dot + 1).toLowerCase() : "";
}

function buildViewUrl(params, filename = params.filename) {
    const query = new URLSearchParams({
        filename,
        subfolder: params.subfolder || "",
        type: params.type || "output",
        t: String(Date.now()),
    });
    const relative = `/view?${query.toString()}`;
    return api && typeof api.apiURL === "function" ? api.apiURL(relative) : relative;
}

function formatFrameRate(value) {
    const fps = Number(value);
    if (!Number.isFinite(fps) || fps <= 0) {
        return "";
    }
    return Number.isInteger(fps) ? String(fps) : fps.toFixed(2);
}

function copyText(text) {
    if (!text) {
        return;
    }
    if (navigator.clipboard?.writeText) {
        navigator.clipboard.writeText(text).catch(() => {});
        return;
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    try {
        document.execCommand("copy");
    } finally {
        textarea.remove();
    }
}

function downloadUrl(url, filename) {
    if (!url) {
        return;
    }
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename || "Yuan Video";
    document.body.appendChild(anchor);
    anchor.click();
    requestAnimationFrame(() => anchor.remove());
}

/**
 * 判断指针是否在节点矩形内（画布坐标换算与加载视频节点一致）。
 */
function isPointerInsideNode(node, event) {
    if (!event || typeof event.clientX !== "number" || !node?.pos || !node?.size) {
        return false;
    }
    if (node.graph && app.graph && node.graph !== app.graph) {
        return false;
    }
    const canvas = app.canvas;
    const element = canvas?.canvas;
    if (!element) {
        return false;
    }
    const rect = element.getBoundingClientRect();
    const scale = canvas.ds?.scale || 1;
    const offset = canvas.ds?.offset || [0, 0];
    const x = (event.clientX - rect.left) / scale - (offset[0] || 0);
    const y = (event.clientY - rect.top) / scale - (offset[1] || 0);
    const titleHeight = globalThis.LiteGraph?.NODE_TITLE_HEIGHT || 30;
    return (
        x >= node.pos[0] && x <= node.pos[0] + node.size[0]
        && y >= node.pos[1] - titleHeight && y <= node.pos[1] + node.size[1]
    );
}

/**
 * 播放预览窗口：深色面板 + 顶部信息栏 + 黑色视频区。
 * 视频沿用加载视频节点的原生 <video controls> 播放方式；无法播放的格式回退到首帧图片。
 * 声音跟随鼠标：指针在节点内时播放声音，移出节点或窗口即静音。
 */
function createPlayer(node, scheduleFit) {
    const container = document.createElement("div");
    Object.assign(container.style, {
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        width: "100%",
        padding: "10px",
        boxSizing: "border-box",
        background: "rgba(30, 30, 30, 0.9)",
        borderRadius: "6px",
        color: "white",
        fontFamily: "sans-serif",
        flexShrink: "0",
    });

    const topBar = document.createElement("div");
    Object.assign(topBar.style, {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        height: "22px",
        flexShrink: "0",
    });

    const info = document.createElement("span");
    Object.assign(info.style, {
        fontSize: "12px",
        color: "#9ca3af",
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis",
        flex: "1 1 auto",
        minWidth: "0",
    });
    info.textContent = "暂无预览";
    topBar.append(info);

    const state = {
        params: null,
        url: null,
        aspect: 16 / 9,
        paused: false,
        inside: false,
        suppress: false,
    };
    let lastPointer = null;

    const stage = document.createElement("div");
    Object.assign(stage.style, {
        position: "relative",
        width: "100%",
        flexGrow: "1",
        minHeight: "90px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#000",
        borderRadius: "4px",
        overflow: "hidden",
    });

    const video = document.createElement("video");
    Object.assign(video.style, {
        width: "100%",
        height: "100%",
        objectFit: "contain",
        outline: "none",
        display: "none",
    });
    video.controls = true;
    video.controlsList = "nodownload nofullscreen noremoteplayback";
    video.preload = "metadata";
    video.playsInline = true;
    video.loop = true;
    video.muted = true;

    const image = document.createElement("img");
    Object.assign(image.style, {
        maxWidth: "100%",
        maxHeight: "100%",
        objectFit: "contain",
        display: "none",
    });
    image.alt = "视频预览";

    const empty = document.createElement("div");
    Object.assign(empty.style, {
        position: "absolute",
        left: "50%",
        top: "50%",
        transform: "translate(-50%, -50%)",
        color: "#aaaaaa",
        fontSize: "12px",
        textAlign: "center",
        pointerEvents: "none",
    });
    empty.textContent = "运行后在此显示预览";

    const caption = document.createElement("div");
    Object.assign(caption.style, {
        position: "absolute",
        left: "50%",
        bottom: "8px",
        transform: "translateX(-50%)",
        maxWidth: "90%",
        padding: "3px 8px",
        borderRadius: "3px",
        background: "rgba(0, 0, 0, 0.65)",
        color: "#d1d5db",
        fontSize: "12px",
        textAlign: "center",
        display: "none",
        pointerEvents: "none",
    });

    stage.append(video, image, empty, caption);
    container.append(topBar, stage);

    function infoText() {
        const params = state.params;
        if (!params) {
            return "暂无预览";
        }
        const parts = [params.format_label || params.format || "预览"];
        const frameRate = formatFrameRate(params.frame_rate);
        if (frameRate) {
            parts.push(`${frameRate} 帧/秒`);
        }
        if (params.width && params.height) {
            parts.push(`${params.width}×${params.height}`);
        }
        return parts.join(" · ");
    }

    function showCaption(message) {
        caption.textContent = message;
        caption.style.display = "block";
    }

    function updateInfo() {
        info.textContent = infoText();
        info.title = info.textContent;
    }

    // 声音跟随鼠标：指针在节点内时取消静音，移出节点或窗口时静音
    function updateAudibility() {
        if (!state.params) {
            return;
        }
        const muted = !state.inside;
        if (video.muted !== muted) {
            video.muted = muted;
        }
    }

    function persist(info = null) {
        const saved = {
            params: state.params ? { ...state.params } : null,
            paused: state.paused === true,
        };
        if (!node.properties) {
            node.properties = {};
        }
        node.properties.yuanVideoPreview = saved;
        // onSerialize 在 properties 克隆之后才回调，必须同步写回序列化对象本身
        if (info) {
            if (!info.properties) {
                info.properties = {};
            }
            info.properties.yuanVideoPreview = saved;
        }
    }

    function open() {
        if (state.url) {
            window.open(state.url, "_blank", "noopener");
        }
    }

    function download() {
        if (state.params && state.url) {
            downloadUrl(state.url, state.params.filename);
        }
    }

    function downloadWorkflow() {
        const params = state.params;
        if (params?.workflow) {
            downloadUrl(buildViewUrl(params, params.workflow), params.workflow);
        }
    }

    function setPreview(preview, force = false) {
        if (!preview || !preview.filename) {
            return;
        }
        const params = { ...preview };
        state.params = params;
        const width = Number(params.width);
        const height = Number(params.height);
        if (width > 0 && height > 0) {
            state.aspect = width / height;
        }
        if (!force) {
            state.paused = false;
        }
        if (lastPointer) {
            state.inside = isPointerInsideNode(node, lastPointer);
        }
        updateInfo();

        state.url = buildViewUrl(params);
        state.suppress = true;
        video.pause();
        video.removeAttribute("src");
        video.load();
        image.removeAttribute("src");
        video.style.display = "none";
        image.style.display = "none";
        empty.style.display = "none";
        caption.style.display = "none";

        if (VIDEO_EXTENSIONS.has(extensionOf(params.filename))) {
            video.style.display = "block";
            video.src = state.url;
            video.muted = true;
            video.load();
            if (!state.paused) {
                const attempt = video.play();
                if (attempt && typeof attempt.catch === "function") {
                    attempt.catch(() => {});
                }
            }
        } else {
            image.style.display = "block";
            image.src = state.url;
        }
        setTimeout(() => {
            state.suppress = false;
        }, 0);

        updateAudibility();
        scheduleFit();
    }

    video.addEventListener("loadedmetadata", () => {
        if (video.videoWidth > 0 && video.videoHeight > 0) {
            const aspect = video.videoWidth / video.videoHeight;
            if (Math.abs(aspect - state.aspect) > 0.01) {
                state.aspect = aspect;
                scheduleFit();
            }
        }
    });

    video.addEventListener("play", () => {
        if (!state.suppress) {
            state.paused = false;
        }
    });

    video.addEventListener("pause", () => {
        if (!state.suppress) {
            state.paused = true;
        }
    });

    video.addEventListener("volumechange", () => {
        // 鼠标不在节点内时保持静音（即使通过原生音量控件取消静音）
        if (state.params && !state.inside && !video.muted) {
            video.muted = true;
        }
    });

    video.addEventListener("error", () => {
        if (video.style.display === "none" || !video.getAttribute("src")) {
            return;
        }
        const params = state.params;
        if (params?.workflow) {
            video.style.display = "none";
            image.style.display = "block";
            image.src = buildViewUrl(params, params.workflow);
            showCaption("该格式无法在浏览器内播放，已显示首帧预览");
        } else {
            showCaption("浏览器无法播放此格式，请右键打开或下载。");
        }
    });

    image.addEventListener("load", () => {
        if (image.naturalWidth > 0 && image.naturalHeight > 0) {
            const params = state.params;
            if (!params?.width || !params?.height) {
                state.aspect = image.naturalWidth / image.naturalHeight;
                scheduleFit();
            }
        }
    });

    image.addEventListener("error", () => {
        if (!image.getAttribute("src")) {
            return;
        }
        showCaption("预览加载失败，请检查输出文件是否存在。");
    });

    // 用节点矩形判断指针是否在节点内，兼容 DOM 播放器与画布控件之间的移动
    const onPointerMove = (event) => {
        lastPointer = { clientX: event.clientX, clientY: event.clientY };
        if (!state.params) {
            return;
        }
        const inside = isPointerInsideNode(node, lastPointer);
        if (inside !== state.inside) {
            state.inside = inside;
            updateAudibility();
        }
    };
    const muteOutside = () => {
        if (state.inside) {
            state.inside = false;
            updateAudibility();
        }
    };
    const onPointerOut = (event) => {
        if (!event.relatedTarget) {
            muteOutside();
        }
    };
    document.addEventListener("pointermove", onPointerMove, true);
    document.addEventListener("mouseout", onPointerOut);
    window.addEventListener("blur", muteOutside);
    chainCallback(node, "onRemoved", function () {
        document.removeEventListener("pointermove", onPointerMove, true);
        document.removeEventListener("mouseout", onPointerOut);
        window.removeEventListener("blur", muteOutside);
    });

    const widget = node.addDOMWidget("播放预览", "div", container, { serialize: false });
    widget.computeSize = function (width) {
        const nodeWidth = node.size?.[0] || width || 360;
        const rowWidth = Math.max(10, nodeWidth - 30);
        const stageHeight = Math.max(90, (rowWidth - 20) / state.aspect);
        const height = Math.round(stageHeight + 52);
        container.style.height = `${height}px`;
        return [rowWidth, height];
    };
    widget.computeLayoutSize = () => ({ minWidth: 300, minHeight: 140 });

    const player = {
        container,
        state,
        infoText,
        setPreview,
        open,
        download,
        downloadWorkflow,
        persist,
    };
    node.yuanPlayer = player;
    node.setYuanPreview = setPreview;
    return player;
}

function applyWidgetDefinition(widget, definition) {
    const type = definition[1];
    const config = definition[2] || {};
    if (Array.isArray(type) && type.length > 0) {
        const values = type.slice();
        if (widget.options) {
            widget.options.values = values;
        }
        if (!values.includes(widget.value)) {
            widget.value = config.default !== undefined ? config.default : values[0];
        }
    } else if (widget.options && (type === "INT" || type === "FLOAT")) {
        for (const key of ["min", "max", "step"]) {
            if (config[key] !== undefined) {
                widget.options[key] = config[key];
            }
        }
    }
}

// 「图像」声明在可选输入中后会排到控件之后；这里把它移回端口区首行，保持与旧版一致的接线位置。
const IMAGE_INPUT_NAME = "图像";

function hoistImageInput(node) {
    const inputs = node?.inputs;
    if (!Array.isArray(inputs)) {
        return;
    }
    const index = inputs.findIndex((input) => input?.name === IMAGE_INPUT_NAME);
    if (index <= 0) {
        return;
    }
    const [imageInput] = inputs.splice(index, 1);
    inputs.unshift(imageInput);
    // 连线按槽位索引寻址，重排后把已保存连线的目标槽位同步到同一输入
    for (let slot = 0; slot < inputs.length; slot++) {
        const linkId = inputs[slot].link;
        if (linkId === null || linkId === undefined) {
            continue;
        }
        const link = node.graph?.links?.[linkId];
        if (link) {
            link.target_slot = slot;
        }
    }
    node.setDirtyCanvas?.(true, true);
}

/** 把一个节点上的高级设置与格式参数联动、预览窗口、状态恢复全部装好。 */
function addVideoSaveUI(node) {
    const widgets = {};
    for (const widget of node.widgets || []) {
        widgets[widget.name] = widget;
    }

    const formatWidget = widgets["输出格式"];
    const anchorWidget = widgets["保存到输出目录"] || widgets["文件名前缀"];
    if (!formatWidget || !anchorWidget) {
        return;
    }
    node.properties = node.properties || {};

    let advancedOpen = false;
    let formatsData = null;

    const toggle = node.addWidget("button", "展开高级设置", "展开高级设置", () => {
        advancedOpen = !advancedOpen;
        applyAdvanced();
    });
    if (toggle) {
        const index = node.widgets.indexOf(toggle);
        node.widgets.splice(index, 1);
        node.widgets.splice(node.widgets.indexOf(anchorWidget) + 1, 0, toggle);
        // 只排除提交参数，保留序列化占位，避免 widgets_values 出现空洞导致位置错位
        toggle.options = { ...(toggle.options || {}), serialize: false };
    }

    let fitTimer = null;
    function scheduleFit() {
        if (fitTimer) {
            return;
        }
        fitTimer = setTimeout(() => {
            fitTimer = null;
            const width = node.size?.[0];
            if (!width || !node.computeSize) {
                return;
            }
            const size = node.computeSize([width, -1]);
            if (!Array.isArray(size)) {
                return;
            }
            if (Math.abs((node.size[1] || 0) - size[1]) > 0.5) {
                node.size[1] = size[1];
            }
            node.setDirtyCanvas?.(true, true);
            node.graph?.setDirtyCanvas?.(true, true);
        }, 30);
    }

    function applyFormatWidgets() {
        const names = formatsData?.widgets?.length ? formatsData.widgets : FORMAT_WIDGET_NAMES;
        if (!formatsData) {
            for (const name of names) {
                const widget = widgets[name];
                if (widget) {
                    if (advancedOpen) {
                        showWidget(widget);
                    } else {
                        hideWidget(widget);
                    }
                }
            }
            return;
        }

        const definitions = new Map();
        const current = formatsData.formats?.[formatWidget.value];
        for (const definition of current?.widgets || []) {
            definitions.set(definition[0], definition);
        }
        for (const name of names) {
            const widget = widgets[name];
            if (!widget) {
                continue;
            }
            const definition = definitions.get(name);
            if (!advancedOpen || !definition) {
                hideWidget(widget);
                continue;
            }
            applyWidgetDefinition(widget, definition);
            showWidget(widget);
        }
    }

    function applyAdvanced() {
        const label = advancedOpen ? "收起高级设置" : "展开高级设置";
        if (toggle) {
            toggle.name = label;
            toggle.label = label;
            toggle.value = label;
        }
        for (const name of ADVANCED_WIDGET_NAMES) {
            const widget = widgets[name];
            if (!widget) {
                continue;
            }
            if (advancedOpen) {
                showWidget(widget);
            } else {
                hideWidget(widget);
            }
        }
        applyFormatWidgets();
        scheduleFit();
    }

    if (formatWidget) {
        chainCallback(formatWidget, "callback", () => {
            applyFormatWidgets();
            scheduleFit();
        });
    }

    function restoreLegacyValues(values) {
        if (!values || Array.isArray(values) || typeof values !== "object") {
            return null;
        }
        for (const [key, value] of Object.entries(values)) {
            if (key === "yuan_video_preview" || value === undefined) {
                continue;
            }
            const widget = widgets[LEGACY_WIDGET_ALIASES[key] || key];
            if (widget) {
                widget.value = value;
            }
        }
        return values["yuan_video_preview"] || null;
    }

    const player = createPlayer(node, scheduleFit);

    // 预览状态随工作流保存；播放/静音过程中不写 properties，避免频繁标记工作流已修改
    chainCallback(node, "onSerialize", (info) => player.persist(info));

    chainCallback(node, "onConfigure", function (info) {
        const legacyPreview = restoreLegacyValues(info?.widgets_values);
        advancedOpen = false;
        applyAdvanced();
        const saved = info?.properties?.yuanVideoPreview
            || (legacyPreview?.params?.filename ? legacyPreview : null);
        if (saved?.params?.filename) {
            player.state.paused = saved.paused === true;
            player.setPreview(saved.params, true);
        }
        scheduleFit();
    });

    // onExecuted 与 api 的 executed 事件可能同时到达，短时间内同一文件只加载一次
    let lastPreviewKey = "";
    let lastPreviewAt = 0;
    const applyExecuted = (output) => {
        const previews = output?.yuan_video;
        if (!Array.isArray(previews) || previews.length === 0) {
            return;
        }
        const preview = previews[0];
        const key = `${preview.type || ""}|${preview.subfolder || ""}|${preview.filename || ""}`;
        const now = Date.now();
        if (key === lastPreviewKey && now - lastPreviewAt < 500) {
            return;
        }
        lastPreviewKey = key;
        lastPreviewAt = now;
        player.setPreview(preview);
    };

    chainCallback(node, "onExecuted", applyExecuted);

    if (api && typeof api.addEventListener === "function") {
        const onExecuted = (event) => {
            const detail = event?.detail;
            if (String(detail?.node) !== String(node.id)) {
                return;
            }
            applyExecuted(detail?.output);
        };
        api.addEventListener("executed", onExecuted);
        chainCallback(node, "onRemoved", function () {
            api.removeEventListener?.("executed", onExecuted);
        });
    }

    if (toggle) {
        applyAdvanced();
    } else {
        applyFormatWidgets();
    }

    fetchFormats().then((data) => {
        formatsData = data;
        if (data) {
            applyFormatWidgets();
            scheduleFit();
        }
    });

    let lastWidth = node.size?.[0] || 0;
    chainCallback(node, "onResize", function (size) {
        const width = size?.[0] || node.size?.[0] || 0;
        if (Math.abs(width - lastWidth) <= 0.5) {
            return;
        }
        lastWidth = width;
        scheduleFit();
    });

    chainCallback(node, "onAdded", function () {
        if ((node.size?.[0] || 0) < 360) {
            node.size[0] = 360;
        }
        scheduleFit();
    });

    // V3（Nodes 2.0）会为隐藏的控件显示空端口并拉长节点，按隐藏状态移除这些占位端口。
    let portSignature = "";
    chainCallback(node, "onDrawForeground", function () {
        const mount = player.container.parentElement;
        if (!mount || !findComfyNodeEl(mount) || !Array.isArray(node.inputs)) {
            return;
        }
        const hidden = [];
        for (const name of [...ADVANCED_WIDGET_NAMES, ...FORMAT_WIDGET_NAMES]) {
            const widget = widgets[name];
            if (!widget) {
                continue;
            }
            if (widget.hidden || widget.options?.hidden) {
                hidden.push(name);
            }
        }
        const inputs = node.inputs.map((input) => input.name).join("|");
        const signature = `${hidden.join("|")}::${inputs}`;
        if (signature === portSignature) {
            return;
        }
        for (const name of hidden) {
            const slot = node.findInputSlot?.(name);
            if (slot >= 0 && node.inputs[slot] && !node.inputs[slot].link) {
                node.removeInput(slot);
            }
        }
        portSignature = `${hidden.join("|")}::${node.inputs.map((input) => input.name).join("|")}`;
    });

    if ((node.size?.[0] || 0) < 360) {
        node.size[0] = 360;
    }
    scheduleFit();
}

function addPreviewMenu(nodeType) {
    chainCallback(nodeType.prototype, "getExtraMenuOptions", function (_, options) {
        if (!Array.isArray(options)) {
            return;
        }
        const player = this.yuanPlayer;
        if (!player?.state?.params) {
            return;
        }
        const params = player.state.params;
        const items = [{ content: player.infoText(), disabled: true }];

        items.push({ content: "打开预览", callback: () => player.open() });
        items.push({ content: "保存预览", callback: () => player.download() });
        if (params.fullpath) {
            items.push({
                content: "复制输出文件路径",
                callback: () => copyText(params.fullpath),
            });
        }
        if (params.workflow) {
            items.push({
                content: "保存工作流首帧图片",
                callback: () => player.downloadWorkflow(),
            });
        }

        if (options.length > 0) {
            items.push(null);
        }
        options.unshift(...items);
    });
}

console.info("[Yuan 视频保存] 前端扩展已加载");

app.registerExtension({
    name: "Yuan.VideoSave",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) {
            return;
        }
        if (nodeType.prototype.__yuanVideoSaveInstalled) {
            return;
        }
        nodeType.prototype.__yuanVideoSaveInstalled = true;

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            addVideoSaveUI(this);
            hoistImageInput(this);
        });
        addPreviewMenu(nodeType);
    },
    afterConfigureGraph() {
        // 工作流反序列化后再排列一次：此时连线已就绪，可同步槽位
        for (const node of app.graph?._nodes ?? []) {
            if (node?.type === NODE_NAME) {
                hoistImageInput(node);
            }
        }
    },
});
