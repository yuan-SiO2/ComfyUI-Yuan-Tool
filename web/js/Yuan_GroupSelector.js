// 「组节点选项」节点（YuanGroupSelector）：复刻 rgthree 的 Fast Groups Bypasser（纯前端节点、无后端）。
// 自动收集当前工作流（含子图）里的全部组，为每个组画一行开关：
//   · 点击行内任意位置（行尾箭头以外）＝ 把该组内所有节点切到「旁路」或「启用」；
//   · 点击行尾箭头 ＝ 定位并缩放到该组。
//
// 代码隔离说明：本文件与 rgthree 完全独立，不 import / 不引用它的任何模块、全局变量或节点类型，
// 只使用 ComfyUI 前端公开的 LiteGraph API（onDrawForeground / onMouseDown 等），
// 因此可与 ComfyUI-rgthree 同时安装，互不影响。

/* ------------------------------ 常量 ------------------------------ */

const YUAN_GROUP_SELECTOR_TYPE = "YuanGroupSelector";
const YUAN_GROUP_SELECTOR_TITLE = "组节点选项";
const YUAN_GROUP_SELECTOR_CATEGORY = "Yuan Tool/选项";
// 旧版本的默认标题，载入老工作流时自动换成新名（用户自定义过的标题不动）
const YUAN_GROUP_SELECTOR_LEGACY_TITLE = "组节点选择";

// 节点来源归属：纯前端节点的 def 由前端补全，默认 python_module 是 custom_nodes.frontend_only，
// 这里取本扩展 URL 里的包名（/extensions/<包名>/...），与后端节点保持一致。
const YUAN_GROUP_SELECTOR_MODULE = "custom_nodes." + (
    (import.meta.url.match(/\/extensions\/([^/]+)\//) || [])[1] || "ComfyUI-Yuan-Tool"
);
const YUAN_GROUP_SELECTOR_DESCRIPTION =
    "自动收集当前工作流（含子图）中的全部组，为每个组生成一行开关：" +
    "点击行内任意位置即把组内所有节点切换为旁路 / 启用，点击行尾箭头可定位并缩放到该组。" +
    "支持按位置、名称或自定义字母排序，按颜色 / 标题过滤，以及「最多一个」「始终一个」互斥限制。";

const MODE_ALWAYS = 0; // LiteGraph.ALWAYS（正常执行）
const MODE_BYPASS = 4; // ComfyUI 的 bypass

// 节点属性名：写在 node.properties 上，随工作流一起保存。
// 键名与取值都用中文，这样「属性面板」里看到的就是中文说明，不需要额外的翻译层。
const PROP_MATCH_COLORS = "颜色过滤"; // 只看这些颜色的组（逗号分隔，颜色名或 #hex）
const PROP_MATCH_TITLE = "标题过滤"; // 只看标题匹配（支持正则）的组
const PROP_SHOW_NAV = "显示定位箭头"; // 是否显示行尾的定位箭头（开 / 关）
const PROP_SHOW_ALL_GRAPHS = "显示所有子图的分组"; // 是否包含子图里的组（开 / 关）
const PROP_SORT = "排序方式"; // 按位置 | 按名称 | 按自定义字母
const PROP_SORT_CUSTOM_ALPHA = "自定义排序字母"; // 自定义字母表
const PROP_RESTRICTION = "互斥限制"; // 默认（可多开）| 最多一个 | 始终一个

// 「显示 / 隐藏」类属性用中文开关值，属性面板里直接可读
const TOGGLE_ON = "开";
const TOGGLE_OFF = "关";

const SORT_POSITION = "按位置";
const SORT_ALPHA = "按名称";
const SORT_CUSTOM = "按自定义字母";

const RESTRICTION_DEFAULT = "默认（可多开）";
const RESTRICTION_MAX_ONE = "最多一个";
const RESTRICTION_ALWAYS_ONE = "始终一个";

// 旧版本（英文键名/取值）迁移到中文键名，避免已保存的工作流丢设置
const LEGACY_PROPERTY_KEYS = {
    matchColors: PROP_MATCH_COLORS,
    matchTitle: PROP_MATCH_TITLE,
    showNav: PROP_SHOW_NAV,
    showAllGraphs: PROP_SHOW_ALL_GRAPHS,
    sort: PROP_SORT,
    customSortAlphabet: PROP_SORT_CUSTOM_ALPHA,
    toggleRestriction: PROP_RESTRICTION,
};
const LEGACY_PROPERTY_VALUES = {
    position: SORT_POSITION,
    alphanumeric: SORT_ALPHA,
    "custom alphabet": SORT_CUSTOM,
    default: RESTRICTION_DEFAULT,
    "max one": RESTRICTION_MAX_ONE,
    "always one": RESTRICTION_ALWAYS_ONE,
};

// 轮询刷新间隔（毫秒）：组的新增/删除/改名、组内节点模式变化都靠它同步过来
const REFRESH_INTERVAL = 400;

// 行布局（单位：像素，坐标系原点在节点主体左上角）
// 尺寸整体按「开关直径 26px（原 13px 的 2 倍）」放大，其余间距同步放大，保持观感比例
const ROW_HEIGHT = 40; // 每行高度（原 22）
const ROW_PADDING = 6; // 主体上下留白
const ROW_MARGIN = 12; // 每行左右留白
const NAV_WIDTH = 40; // 行尾定位箭头热区宽度（含分隔线）
const TOGGLE_RADIUS = 13; // 开关圆点半径（原 6.5，直径翻倍）
const ROW_FONT_SIZE = 17; // 行内文字字号（原 13）
const MIN_WIDTH = 320; // 节点最小宽度（放大后留足标题空间）

/* ------------------------------ 基础工具 ------------------------------ */

/** 取当前 app 实例（不同版本可能挂在 comfyAPI.app.app 下）。 */
function getApp() {
    return window.comfyAPI?.app?.app ?? window.comfyAPI?.app ?? null;
}

/** 读节点属性，缺省时返回 fallback。 */
function prop(node, name, fallback) {
    const value = node?.properties?.[name];
    return value === undefined || value === null ? fallback : value;
}

/** 读「开 / 关」型属性（兼容旧版本写入的 true/false）。 */
function propOn(node, name) {
    const value = prop(node, name, TOGGLE_ON);
    return value !== false && `${value}` !== TOGGLE_OFF;
}

/** 画布是否处于低质量（缩小）绘制模式。 */
function isLowQuality() {
    const canvas = getApp()?.canvas;
    if (canvas?.low_quality !== undefined) return !!canvas.low_quality;
    return (canvas?.ds?.scale ?? 1) <= 0.5;
}

/** 文本太长时截断并加省略号（需要与绘制使用同一字体）。 */
function fitString(ctx, text, maxWidth) {
    const str = `${text ?? ""}`;
    if (!str || maxWidth <= 0) return "";
    if (ctx.measureText(str).width <= maxWidth) return str;
    const ellipsis = "…";
    const ellipsisWidth = ctx.measureText(ellipsis).width;
    if (maxWidth <= ellipsisWidth) return "";
    let low = 0;
    let high = str.length;
    while (low < high) {
        const mid = Math.ceil((low + high) / 2);
        if (ctx.measureText(str.slice(0, mid)).width + ellipsisWidth <= maxWidth) low = mid;
        else high = mid - 1;
    }
    return str.slice(0, low) + ellipsis;
}

/** 把颜色统一成 #rrggbb（支持 3 位缩写）；不是合法 hex 时返回空串。 */
function normalizeHex(color) {
    let value = `${color ?? ""}`.trim().toLocaleLowerCase().replace(/^#/, "");
    if (!/^([0-9a-f]{3}|[0-9a-f]{6})$/.test(value)) return "";
    if (value.length === 3) value = value.replace(/(.)(.)(.)/, "$1$1$2$2$3$3");
    return `#${value}`;
}

/** 把颜色名（red / pale_blue…）或 #hex 都转成组用的 #rrggbb。 */
function expandColor(color) {
    let value = `${color ?? ""}`.trim().toLocaleLowerCase();
    if (!value) return "";
    const named = window.LGraphCanvas?.node_colors?.[value];
    if (named?.groupcolor) value = named.groupcolor;
    return normalizeHex(value);
}

/** 圆角矩形（老浏览器无 roundRect 时退化为普通矩形）。 */
function roundRectPath(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
        ctx.roundRect(x, y, width, height, radius);
        return;
    }
    const r = Math.min(radius, width * 0.5, height * 0.5);
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

/* ------------------------------ 组相关 ------------------------------ */

/** 当前画布正在编辑的图。 */
function currentGraph() {
    const app = getApp();
    return app?.canvas?.getCurrentGraph?.() ?? app?.graph ?? null;
}

/** 组的坐标（兼容 _pos / pos 两种取法）。 */
function groupPos(group) {
    const pos = group?._pos ?? group?.pos ?? [0, 0];
    return [Number(pos[0]) || 0, Number(pos[1]) || 0];
}

/** 组的尺寸（兼容 _size / size 两种取法）。 */
function groupSize(group) {
    const size = group?._size ?? group?.size ?? [0, 0];
    return [Number(size[0]) || 0, Number(size[1]) || 0];
}

/** 收集当前工作流里所有组（含子图的组），按对象去重。 */
function collectGroups() {
    const graph = currentGraph();
    const groups = [];
    const seen = new Set();
    const push = (g) => {
        if (!g) return;
        for (const group of g._groups ?? g.groups ?? []) {
            if (!group || seen.has(group)) continue;
            seen.add(group);
            groups.push(group);
        }
    };
    push(graph);
    const subgraphs = graph?.subgraphs;
    if (subgraphs) {
        // 新版前端是 Map，老版是普通对象
        const values = typeof subgraphs.values === "function" ? [...subgraphs.values()] : Object.values(subgraphs);
        for (const sub of values) push(sub);
    }
    return groups;
}

/**
 * 组内节点（会先重算组的成员）。
 * group._children 里同时含节点、连线点（reroute）和嵌套的子组，这里只保留真正的节点；
 * 同时排除本节点自身，避免把「组节点选项」一起旁路掉。
 */
function groupInsideNodes(group) {
    if (!group) return [];
    try {
        group.recomputeInsideNodes?.();
    } catch (_) {
        /* 组的成员重算失败时退回已有的成员列表 */
    }
    const GroupClass = window.LGraphGroup;
    const children = group._children;
    const list = children && typeof children[Symbol.iterator] === "function"
        ? [...children]
        : [...(group.nodes ?? [])];
    return list.filter((node) => {
        if (!node || node.type === YUAN_GROUP_SELECTOR_TYPE) return false;
        if (typeof node.mode !== "number") return false;
        if (GroupClass && node instanceof GroupClass) return false;
        return true;
    });
}

/** 组里是否还有处于「正常执行」状态的节点。 */
function groupHasActiveNode(group) {
    return groupInsideNodes(group).some((node) => node.mode === MODE_ALWAYS);
}

/** 自定义字母表：含逗号按短语切分，否则按单字切分。 */
function parseAlphabet(value) {
    const str = `${value ?? ""}`.replace(/\n/g, "").toLocaleLowerCase().trim();
    if (!str) return [];
    const list = str.includes(",") ? str.split(",") : str.split("");
    return list.map((item) => item.trim()).filter(Boolean);
}

/** 组标题在自定义字母表里的位置（未匹配返回 -1）。 */
function alphabetIndex(title, alphabet) {
    const lower = `${title ?? ""}`.toLocaleLowerCase();
    for (let i = 0; i < alphabet.length; i++) {
        if (lower.startsWith(alphabet[i])) return i;
    }
    return -1;
}

/** 按属性排序组列表。 */
function sortGroups(node, groups) {
    const mode = `${prop(node, PROP_SORT, SORT_POSITION)}`;
    const list = [...groups];
    if (mode === SORT_ALPHA) {
        return list.sort((a, b) => `${a?.title ?? ""}`.localeCompare(`${b?.title ?? ""}`));
    }
    if (mode === SORT_CUSTOM) {
        const alphabet = parseAlphabet(prop(node, PROP_SORT_CUSTOM_ALPHA, ""));
        if (alphabet.length) {
            return list.sort((a, b) => {
                const aIndex = alphabetIndex(a?.title, alphabet);
                const bIndex = alphabetIndex(b?.title, alphabet);
                if (aIndex > -1 && bIndex > -1) {
                    return aIndex === bIndex
                        ? `${a?.title ?? ""}`.localeCompare(`${b?.title ?? ""}`)
                        : aIndex - bIndex;
                }
                if (aIndex > -1) return -1;
                if (bIndex > -1) return 1;
                return `${a?.title ?? ""}`.localeCompare(`${b?.title ?? ""}`);
            });
        }
    }
    // 默认按位置：先比 Y 再比 X，都按 30px 网格取整（同一行内按从左到右）
    return list.sort((a, b) => {
        const [ax, ay] = groupPos(a);
        const [bx, by] = groupPos(b);
        const ay30 = Math.floor(ay / 30);
        const by30 = Math.floor(by / 30);
        if (ay30 !== by30) return ay30 - by30;
        return Math.floor(ax / 30) - Math.floor(bx / 30);
    });
}

/** 组是否通过颜色 / 标题 / 图范围过滤。 */
function matchGroup(node, group) {
    const colors = `${prop(node, PROP_MATCH_COLORS, "")}`
        .split(",")
        .map(expandColor)
        .filter(Boolean);
    if (colors.length) {
        const color = expandColor(group?.color);
        if (!color || !colors.includes(color)) return false;
    }
    const matcher = `${prop(node, PROP_MATCH_TITLE, "")}`.trim();
    if (matcher) {
        try {
            if (!new RegExp(matcher, "i").test(`${group?.title ?? ""}`)) return false;
        } catch (_) {
            /* 正则非法时忽略标题过滤，避免整个列表消失 */
        }
    }
    if (!propOn(node, PROP_SHOW_ALL_GRAPHS)) {
        const graph = currentGraph();
        if (group?.graph && graph && group.graph !== graph) return false;
    }
    return true;
}

/** 定位并缩放到某个组。 */
function focusGroup(group) {
    const canvas = getApp()?.canvas;
    if (!canvas || !group) return;
    const [x, y] = groupPos(group);
    const [width, height] = groupSize(group);
    try {
        canvas.centerOnNode?.({ pos: [x, y], size: [width, height] });
        const canvasWidth = canvas.canvas?.width || 1;
        const canvasHeight = canvas.canvas?.height || 1;
        const zoomCurrent = canvas.ds?.scale || 1;
        const zoomX = canvasWidth / Math.max(width, 1) - 0.02;
        const zoomY = canvasHeight / Math.max(height, 1) - 0.02;
        canvas.setZoom?.(Math.min(zoomCurrent, zoomX, zoomY), [canvasWidth / 2, canvasHeight / 2]);
        canvas.setDirty?.(true, true);
    } catch (err) {
        console.error("[Yuan Tool] 组节点选项：定位分组失败", err);
    }
}

/* ------------------------------ 选组状态 ------------------------------ */

/** 视图中需要显示的组（排序 + 过滤）。 */
function visibleGroups(node) {
    return sortGroups(node, collectGroups()).filter((group) => matchGroup(node, group));
}

/** 取节点的行列表（每行是纯数据对象，不参与序列化）。 */
function rowsOf(node) {
    if (!Array.isArray(node.__yuanRows)) node.__yuanRows = [];
    return node.__yuanRows;
}
/** 「最多一个」或「始终一个」——都要求同一时刻只能有一行开启。 */
function isOnlyOneRestriction(restriction) {
    return restriction === RESTRICTION_MAX_ONE || restriction === RESTRICTION_ALWAYS_ONE || `${restriction}`.includes("one");
}

/** 旧默认标题「组节点选择」→ 新默认标题「组节点选项」（只替换完全相同的默认名）。 */
function migrateLegacyTitle(node) {
    if (node && node.title === YUAN_GROUP_SELECTOR_LEGACY_TITLE) node.title = YUAN_GROUP_SELECTOR_TITLE;
}

/** 把旧版英文属性的键名/取值迁移成中文（只做一次，已保存的工作流不会丢设置）。 */
function migrateLegacyProperties(node) {
    if (!node || node.__yuanMigrated) return;
    node.__yuanMigrated = true;
    const props = node.properties;
    if (!props) return;
    for (const [oldKey, newKey] of Object.entries(LEGACY_PROPERTY_KEYS)) {
        if (props[oldKey] === undefined) continue;
        if (props[newKey] === undefined) {
            const oldValue = props[oldKey];
            let newValue = oldValue;
            if (typeof oldValue === "boolean") {
                newValue = oldValue ? TOGGLE_ON : TOGGLE_OFF;
            } else if (typeof oldValue === "string" && LEGACY_PROPERTY_VALUES[oldValue]) {
                newValue = LEGACY_PROPERTY_VALUES[oldValue];
            }
            props[newKey] = newValue;
        }
        delete props[oldKey];
    }
}

/**
 * 切换某一行的状态。
 * @param {object} node 本节点
 * @param {object} row 行数据
 * @param {boolean|null} force 强制指定目标状态（null = 取反）
 * @param {boolean} skipOthers 跳过互斥限制（内部递归用）
 */
function setRowMode(node, row, force, skipOthers) {
    if (!node || !row?.group) return;
    const nodes = groupInsideNodes(row.group);
    const hasActive = nodes.some((item) => item.mode === MODE_ALWAYS);
    let value = force === null || force === undefined ? !hasActive : !!force;
    const restriction = `${prop(node, PROP_RESTRICTION, RESTRICTION_DEFAULT)}`;

    if (!skipOthers) {
        if (value && isOnlyOneRestriction(restriction)) {
            // 最多一个 / 始终一个：先把其它行关掉
            for (const other of rowsOf(node)) {
                if (other !== row) setRowMode(node, other, false, true);
            }
        } else if (!value && restriction === RESTRICTION_ALWAYS_ONE) {
            // 始终一个：如果其余行都关着，则本行保持开启，保证至少有一个
            value = rowsOf(node).every((other) => other === row || !other.toggled);
        }
    }

    const mode = value ? MODE_ALWAYS : MODE_BYPASS;
    for (const item of nodes) item.mode = mode;
    row.toggled = value;
    node.graph?.setDirtyCanvas?.(true, true);
    getApp()?.canvas?.setDirty?.(true, true);
}

/* ------------------------------ 尺寸与绘制 ------------------------------ */

/** 行数对应的主体高度。 */
function contentHeight(count) {
    return ROW_PADDING * 2 + Math.max(1, count) * ROW_HEIGHT;
}

/** 根据行数调整节点尺寸（首次只放大，之后随行数精确收缩）。 */
function arrangeSize(node, count) {
    if (!node?.size) return;
    const needHeight = contentHeight(count);
    const first = node.__yuanArranged !== true;
    const countChanged = node.__yuanRowCount !== count;
    const width = Math.max(node.size[0] ?? 0, MIN_WIDTH);
    let height = node.size[1] ?? needHeight;
    if (first) height = Math.max(height, needHeight);
    else if (countChanged) height = needHeight; // 行数变化时才重排高度，保留用户手动拉高的尺寸
    node.__yuanArranged = true;
    node.__yuanRowCount = count;
    if (width !== node.size[0] || height !== node.size[1]) node.setSize?.([width, height]);
}

/** 第 index 行的顶边 Y。 */
function rowTop(index) {
    return ROW_PADDING + index * ROW_HEIGHT;
}

/** 行尾的定位箭头（tipX 是箭头尖端，half 是箭头半高，其余尺寸按比例推）。 */
function drawNavArrow(ctx, tipX, midY, half) {
    const shaftHalf = Math.max(3, Math.round(half * 0.45));
    const shaftLength = Math.max(6, Math.round(half * 0.9));
    const back = tipX - half;
    ctx.beginPath();
    ctx.moveTo(tipX, midY);
    ctx.lineTo(back, midY + half);
    ctx.lineTo(back, midY + shaftHalf);
    ctx.lineTo(back - shaftLength, midY + shaftHalf);
    ctx.lineTo(back - shaftLength, midY - shaftHalf);
    ctx.lineTo(back, midY - shaftHalf);
    ctx.lineTo(back, midY - half);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
}

/** 画一行。 */
function drawRow(ctx, node, row, index) {
    const width = node.size[0];
    const top = rowTop(index);
    const lowQuality = isLowQuality();
    const outline = window.LiteGraph?.WIDGET_OUTLINE_COLOR ?? "#666";
    const background = window.LiteGraph?.WIDGET_BGCOLOR ?? "#222";
    const textColor = window.LiteGraph?.WIDGET_TEXT_COLOR ?? "#DDD";
    const secondaryColor = window.LiteGraph?.WIDGET_SECONDARY_TEXT_COLOR ?? "#999";
    const showNav = propOn(node, PROP_SHOW_NAV);

    const barY = top + 4;
    const barHeight = ROW_HEIGHT - 8;
    roundRectPath(ctx, ROW_MARGIN, barY, width - ROW_MARGIN * 2, barHeight, lowQuality ? 0 : barHeight * 0.5);
    ctx.fillStyle = background;
    ctx.fill();
    if (!lowQuality) {
        ctx.strokeStyle = outline;
        ctx.stroke();
    }

    const midY = top + ROW_HEIGHT * 0.5;
    let cursor = width - ROW_MARGIN;

    // 行尾：分隔线 + 定位箭头
    if (showNav) {
        const separatorX = cursor - NAV_WIDTH + 8;
        if (!lowQuality) {
            ctx.strokeStyle = outline;
            ctx.beginPath();
            ctx.moveTo(separatorX, barY + 3);
            ctx.lineTo(separatorX, barY + barHeight - 3);
            ctx.stroke();

            ctx.fillStyle = "#89A";
            ctx.strokeStyle = "#89A";
            ctx.lineJoin = "round";
            ctx.lineCap = "round";
            drawNavArrow(ctx, cursor - 10, midY, barHeight * 0.3);
        }
    }
    cursor -= showNav ? NAV_WIDTH : 0;

    // 开关圆点（直径 = TOGGLE_RADIUS * 2）
    const toggleX = cursor - TOGGLE_RADIUS - 8;
    ctx.fillStyle = row.toggled ? "#89A" : "#333";
    ctx.beginPath();
    ctx.arc(toggleX, midY, TOGGLE_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    if (!lowQuality) {
        ctx.strokeStyle = outline;
        ctx.stroke();
    }
    cursor = toggleX - TOGGLE_RADIUS;

    if (lowQuality) return;

    // 「开 / 关」文字
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillStyle = row.toggled ? textColor : secondaryColor;
    const stateText = row.toggled ? "开" : "关";
    ctx.fillText(stateText, cursor - 8, midY);
    cursor -= ctx.measureText(stateText).width + 16;

    // 组名
    ctx.textAlign = "left";
    ctx.fillStyle = row.toggled ? textColor : secondaryColor;
    const labelX = ROW_MARGIN + 14;
    const label = fitString(ctx, row.title || "（未命名分组）", cursor - labelX);
    ctx.fillText(label, labelX, midY);
}

/** 空状态提示行。 */
function drawEmptyHint(ctx, node) {
    if (isLowQuality()) return;
    const width = node.size[0];
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = window.LiteGraph?.WIDGET_SECONDARY_TEXT_COLOR ?? "#999";
    const labelX = ROW_MARGIN + 14;
    ctx.fillText(fitString(ctx, "当前工作流没有分组", width - labelX - ROW_MARGIN), labelX, ROW_PADDING + ROW_HEIGHT * 0.5);
}

/* ------------------------------ 刷新逻辑 ------------------------------ */

/** 同步本节点的行列表、尺寸与绘制。 */
function refresh(node) {
    if (!node || node.removed || !node.graph) return;
    migrateLegacyTitle(node);
    migrateLegacyProperties(node);
    const groups = visibleGroups(node);
    node.__yuanRows = groups.map((group) => ({
        group,
        title: `${group?.title ?? ""}`,
        toggled: groupHasActiveNode(group),
    }));

    // 内容没变化时不必重绘
    const signature = node.__yuanRows.map((row) => `${row.title}\u0001${row.toggled ? 1 : 0}`).join("\u0002");
    arrangeSize(node, node.__yuanRows.length);
    if (signature !== node.__yuanSignature) {
        node.__yuanSignature = signature;
        node.setDirtyCanvas?.(true, true);
    }
}

/* ------------------------------ 轮询服务 ------------------------------ */

const attachedNodes = new Set();
let refreshTimer = null;

function tickRefresh() {
    for (const node of [...attachedNodes]) {
        if (!node || node.removed || !node.graph) {
            attachedNodes.delete(node);
            continue;
        }
        try {
            refresh(node);
        } catch (err) {
            console.error("[Yuan Tool] 组节点选项：刷新失败", err);
        }
    }
    if (!attachedNodes.size) stopService();
}

function startService() {
    if (refreshTimer) return;
    refreshTimer = setInterval(tickRefresh, REFRESH_INTERVAL);
}

function stopService() {
    if (!refreshTimer) return;
    clearInterval(refreshTimer);
    refreshTimer = null;
}

function attachNode(node) {
    attachedNodes.add(node);
    startService();
    // 稍后再刷新：此时节点可能还没配置完（属性/尺寸还在恢复），立刻刷新会读到空数据
    setTimeout(() => {
        if (attachedNodes.has(node)) refresh(node);
    }, 8);
}

function detachNode(node) {
    attachedNodes.delete(node);
    if (!attachedNodes.size) stopService();
}

/* ------------------------------ 节点类 ------------------------------ */

function createGroupSelectorClass(Base) {
    class YuanGroupSelector extends Base {
        constructor(title) {
            super(title ?? YUAN_GROUP_SELECTOR_TITLE);
            this.title = YUAN_GROUP_SELECTOR_TITLE;
            this.isVirtualNode = true;
            this.serialize_widgets = false;
            this.resizable = true;

            this.properties = this.properties || {};
            const defaults = {
                [PROP_MATCH_COLORS]: "",
                [PROP_MATCH_TITLE]: "",
                [PROP_SHOW_NAV]: TOGGLE_ON,
                [PROP_SHOW_ALL_GRAPHS]: TOGGLE_ON,
                [PROP_SORT]: SORT_POSITION,
                [PROP_SORT_CUSTOM_ALPHA]: "",
                [PROP_RESTRICTION]: RESTRICTION_DEFAULT,
            };
            for (const [name, value] of Object.entries(defaults)) {
                if (this.properties[name] === undefined) this.properties[name] = value;
            }

            this.__yuanRows = [];
            this.size = [MIN_WIDTH, contentHeight(0)];
        }

        /** 尺寸跟随行数（高度不包含标题）。 */
        computeSize(out) {
            const size = out ?? [0, 0];
            size[0] = Math.max(this.size?.[0] ?? 0, MIN_WIDTH);
            size[1] = contentHeight(this.__yuanRows?.length ?? 0);
            return size;
        }

        onAdded(graph) {
            super.onAdded?.(graph);
            attachNode(this);
        }

        onRemoved() {
            detachNode(this);
            super.onRemoved?.();
        }

        onConfigure(info) {
            // 载入工作流后重新按行数整理一次尺寸
            migrateLegacyProperties(this);
            this.__yuanArranged = false;
            this.__yuanSignature = null;
            setTimeout(() => refresh(this), 0);
            super.onConfigure?.(info);
        }

        onDrawForeground(ctx) {
            if (this.flags?.collapsed) return;
            const rows = rowsOf(this);
            // 行超出主体高度时不画（用户手动拖得太小时以内容高度为准）
            const limit = Math.floor((this.size[1] - ROW_PADDING * 2) / ROW_HEIGHT);
            const count = Math.min(rows.length, Math.max(limit, 0));
            if (!rows.length) {
                drawEmptyHint(ctx, this);
                return;
            }
            ctx.save();
            ctx.font = `normal ${ROW_FONT_SIZE}px ${window.LiteGraph?.NODE_FONT ?? "sans-serif"}`;
            ctx.textBaseline = "middle";
            for (let i = 0; i < count; i++) drawRow(ctx, this, rows[i], i);
            ctx.restore();
        }

        onMouseDown(event, pos, canvas) {
            if (event && event.button !== undefined && event.button !== 0) return false;
            const rows = rowsOf(this);
            if (!rows.length || !pos) return false;
            const width = this.size[0];
            const localX = pos[0];
            const localY = pos[1];
            if (localX < ROW_MARGIN || localX > width - ROW_MARGIN) return false;
            if (localY < ROW_PADDING) return false;
            const index = Math.floor((localY - ROW_PADDING) / ROW_HEIGHT);
            if (index < 0 || index >= rows.length) return false;
            if (localY - ROW_PADDING - index * ROW_HEIGHT > ROW_HEIGHT) return false;

            const row = rows[index];
            const showNav = propOn(this, PROP_SHOW_NAV);
            if (showNav && localX >= width - ROW_MARGIN - NAV_WIDTH) {
                focusGroup(row.group);
            } else {
                setRowMode(this, row, null, false);
                refresh(this);
            }
            canvas?.setDirty?.(true, true);
            return true;
        }
    }

    // 属性面板的类型描述（前端会读 constructor["@属性名"]）：类型决定控件样式
    Object.assign(YuanGroupSelector, {
        [`@${PROP_MATCH_COLORS}`]: { type: "string" },
        [`@${PROP_MATCH_TITLE}`]: { type: "string" },
        [`@${PROP_SHOW_NAV}`]: { type: "combo", values: [TOGGLE_ON, TOGGLE_OFF] },
        [`@${PROP_SHOW_ALL_GRAPHS}`]: { type: "combo", values: [TOGGLE_ON, TOGGLE_OFF] },
        [`@${PROP_SORT}`]: {
            type: "combo",
            values: [SORT_POSITION, SORT_ALPHA, SORT_CUSTOM],
        },
        [`@${PROP_SORT_CUSTOM_ALPHA}`]: { type: "string" },
        [`@${PROP_RESTRICTION}`]: {
            type: "combo",
            values: [RESTRICTION_DEFAULT, RESTRICTION_MAX_ONE, RESTRICTION_ALWAYS_ONE],
        },
    });

    return YuanGroupSelector;
}
/* ------------------------------ 注册 ------------------------------ */

const app = getApp();

app?.registerExtension({
    name: "Comfy.Yuan.GroupSelector",
    registerCustomNodes() {
        const LiteGraph = window.LiteGraph;
        const Base = window.LGraphNode;
        if (!LiteGraph || !Base) {
            console.warn("[Yuan Tool] 组节点选项：LiteGraph 尚未就绪，跳过注册");
            return;
        }
        const YuanGroupSelector = createGroupSelectorClass(Base);
        LiteGraph.registerNodeType(
            YUAN_GROUP_SELECTOR_TYPE,
            Object.assign(YuanGroupSelector, {
                title: YUAN_GROUP_SELECTOR_TITLE,
                description: YUAN_GROUP_SELECTOR_DESCRIPTION,
            })
        );
        // registerNodeType 之后前端会把分类重置，这里再写一次
        YuanGroupSelector.category = YUAN_GROUP_SELECTOR_CATEGORY;
    },

    // 纯前端节点的 def 由前端补全，注册进节点库前统一改成本插件的中文名与归属
    beforeRegisterVueAppNodeDefs(defs) {
        if (!Array.isArray(defs)) return;
        const def = defs.find((item) => item && item.name === YUAN_GROUP_SELECTOR_TYPE);
        if (!def) return;
        def.display_name = YUAN_GROUP_SELECTOR_TITLE;
        def.description = YUAN_GROUP_SELECTOR_DESCRIPTION;
        def.category = YUAN_GROUP_SELECTOR_CATEGORY;
        def.python_module = YUAN_GROUP_SELECTOR_MODULE;
    },
});
