import json as _json
import os
import re

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont


# ==== 图像文本标签：本节点的字体目录 ====
FONTS_FOLDER = "yuan_tool_fonts"
DEFAULT_FONT = "default"

_font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
os.makedirs(_font_dir, exist_ok=True)
if FONTS_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(FONTS_FOLDER, _font_dir)


class AnyType(str):
    def __ne__(self, __value):
        return False


# ==== 台词保护：引号对常量与掩码构建（JSON提取 / 文本批量替换 共用） ====

# opening, closing, rank（秩：同秩才能配成一对，避免不同引号互配）
# 包含 ASCII 半角引号、中文弯引号（""/''）、中文直角引号「」/『』
QUOTE_PAIRS = [
    ('"',  '"',  1),
    ("'",  "'",  2),
    ("“",  "”",  3),
    ("‘",  "’",  4),
    ("「", "」", 5),
    ('『', "』", 6),
]

# 半角撇号作缩写/所有格（谢枕's、<Picture 1>'s、swordsman's）时不算引号边界；
# 仅当前一字符属于下列「引号起始上下文」（空白或左括号/引号/分隔标点）时，才把它当引号处理
APOSTROPHE_OPENER_CHARS = frozenset(" \t\r\n([{<“‘「『（【《，,:：;；、—")


def _build_quote_protect_mask(text):
    """返回与 text 等长的 bool 列表：True 表示该位置在引号对内部（台词保护，不替换）；False 表示可替换。"""
    n = len(text)
    mask = [False] * n
    if n == 0:
        return mask

    opening_map = {}
    closing_map = {}
    for op, cl, rk in QUOTE_PAIRS:
        opening_map.setdefault(op, []).append((cl, rk))
        closing_map.setdefault(cl, []).append((op, rk))

    # 栈：存 (rank, closing_char, start_idx)
    stack = []
    i = 0
    while i < n:
        ch = text[i]
        # 后接字母的半角撇号：若前一字符不是引号起始上下文，视为缩写/所有格（谢枕's、<Picture 1>'s），跳过不参与配对
        if ch == "'" and i + 1 < n and text[i + 1].isalpha() and i > 0 and text[i - 1] not in APOSTROPHE_OPENER_CHARS:
            i += 1
            continue
        # 尝试作为 opening：半角引号 " / ' 既可开也可关，同 rank 未闭合栈顶在则作 closing（优先关闭）
        if ch in opening_map:
            if ch in ('"', "'"):
                expected_rank = 1 if ch == '"' else 2
                if stack and stack[-1][0] == expected_rank:
                    _rank, _cl, start = stack.pop()
                    # 把 [start, i] 全部标记为台词内部（含引号自身）
                    for k in range(start, i + 1):
                        mask[k] = True
                    i += 1
                    continue
            (closing_char, rank) = opening_map[ch][0]
            stack.append((rank, closing_char, i))
            i += 1
            continue
        # 尝试作为 closing：匹配最近未闭合、同 rank 的 opening
        if ch in closing_map:
            found = None
            for si in range(len(stack) - 1, -1, -1):
                if stack[si][1] == ch:
                    found = si
                    break
            if found is not None:
                _rank, _cl, start = stack.pop(found)
                for k in range(start, i + 1):
                    mask[k] = True
                i += 1
                continue
        # 普通字符或找不到对应开引号的闭引号：跳过
        i += 1
    # 文本结束后仍留在栈中的未闭合开引号：不标记（按正常文本处理）
    return mask


# <d>...</d> 台词标签（与引号保护叠加，JSON提取 的档案匹配与名称替换共用）
DIALOGUE_TAG_RE = re.compile(r'<d>.*?</d>', re.DOTALL)


def _build_dialogue_protect_mask(text):
    """完整台词保护掩码：引号对内部 + <d>...</d> 标签内部（含标签自身）。"""
    mask = _build_quote_protect_mask(text)
    for m in DIALOGUE_TAG_RE.finditer(text):
        for k in range(m.start(), m.end()):
            mask[k] = True
    return mask


# ==== JSON提取：按端口名提取对应字段（内置分镜角色替换，分镜序列输出替换后的文本） ====

class YUAN_TXTJsonExtractor:
    # 输出端口（分镜模式顺序）：类型校验按端口下标查 RETURN_TYPES，两种模式同一下标的类型须相同：
    #   分镜模式 [0..7] 见下；档案模式 [0..3] = 档案编码(INT) / 作品名 / 整体风格 / 档案
    OUTPUT_NAMES = ("分镜类型", "分镜序列", "角色索引", "音色索引", "道具索引", "场景索引", "索引时长", "场景衔接")

    # 台词归属边界标点：说话者前必须是句首/句末标点/逗号/空白，或紧跟 </d> 之后
    # （「甲对乙说」的乙被"对"字挡住；「甲看向乙，乙说道」的乙由逗号放行，归属竞争离冒号最近者胜）
    BELONG_BOUNDARY = '。！？!?\n；;，,'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json": (AnyType("*"), {
                    "forceInput": True,
                    "tooltip": "JSON 字符串或对象；支持多对象拼接，自动逐段解析合并提取。"
                }),
                "索引": ("INT", {
                    "default": 1,
                    "min": 0,
                    "step": 1,
                    "tooltip": "分镜模式：选择「编号」分镜（1 起）。档案模式：选择「档案选择」里的第几条（0 起，0=第一条），「档案」端口只输出该条的全名与描述。"
                               "分镜序列输出：subject_definitions 定义段 + summary（固定前缀「[reference generation] A N秒 continuous, playful vlog-style sequence: 」+ 本镜「情节」原文，名称同样替换为 <Picture N>）+ detailed_description（首部两行风格头「目标视频为<整体风格>风格，角色一致性必须严格把控。画面质感：<画面质感>。」「场景：<场景>，<在场位置>。<运镜方式>」，空一行后接首段时间段 + At MM:SS.mmm 后续时间段）+ 环境音 + BGM（名称→<Picture N>、说话者→<Picture N><Audio M>；正文不生成 [Shot N]，写手手写的编号一律丢弃）；定义行固定为「<Picture N> is <名称> for the reference,<描述>.」（名称取第一个分隔符前的部分，角色保留「补充：」内容、其余条目截断补充，无描述时省略逗号段；情节衔接整句当名称；不再输出保留分析行）；索引输出各档案 0 基序号，索引时长取「类型」时长，分镜类型取该分镜「类型」编码（文戏 0 / 武戏 1 / MV 2 / 其他 3）。"
                }),
                "档案选择": (["角色档案", "音色档案", "道具档案", "场景档案"], {
                    "default": "角色档案",
                    "tooltip": "选择「档案」输出端口输出的档案类型：角色档案、音色档案、道具档案或场景档案。"
                }),
                "模式": (["分镜模式", "档案模式"], {
                    "default": "分镜模式",
                    "tooltip": "档案模式：输入只剩 JSON，输出只保留「作品名 / 整体风格 / 档案 / 档案编码」；"
                               "分镜模式：隐藏这四项，改显示「开关配置」「情节衔接文本」输入与其余输出端口。",
                }),
            },
            "optional": {
                "开关配置": (AnyType("*"), {
                    "forceInput": True,
                    "tooltip": "接「JSON提取开关」输出，传入七项开关；未接入时按默认值执行（情节衔接默认关闭，其余开启）。"
                }),
                "情节衔接文本": (AnyType("*"), {
                    "forceInput": True,
                    "tooltip": "接外部文本。情节衔接开启时，在分镜序列定义段中场景行之后新增一行（编号顺延，如场景为 <Picture 3> 则新增 <Picture 4>），内容为该文本原文；关闭或未接入时不新增。"
                }),
            },
        }

    RETURN_TYPES = ("INT", "STRING", "STRING", "STRING", "STRING", "STRING", "FLOAT", "BOOLEAN")
    RETURN_NAMES = OUTPUT_NAMES
    FUNCTION = "extract_json"
    CATEGORY = "Yuan Tool/文本"
    OUTPUT_NODE = True

    @staticmethod
    def _list_to_lines(val):
        """列表/元组：每元素转字符串后逐行拼接。"""
        if isinstance(val, (list, tuple)):
            lines = []
            for item in val:
                if item is None:
                    continue
                try:
                    s = str(item)
                except Exception:
                    continue
                if s:
                    lines.append(s)
            return "\n".join(lines)
        if val is None:
            return ""
        try:
            return str(val)
        except Exception:
            return ""

    @staticmethod
    def _strip_json_comments(text):
        """去掉字符串之外的 // 行注释与 /* */ 块注释（AI 生成 JSON 常带集界注释行，如 //**第01集…**///）。"""
        out = []
        i, n = 0, len(text)
        in_str = False
        esc = False
        while i < n:
            ch = text[i]
            if in_str:
                out.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                in_str = True
                out.append(ch)
                i += 1
                continue
            if ch == "/" and i + 1 < n:
                nxt = text[i + 1]
                if nxt == "/":
                    i += 2
                    while i < n and text[i] not in "\r\n":
                        i += 1
                    continue
                if nxt == "*":
                    end = text.find("*/", i + 2)
                    i = n if end == -1 else end + 2
                    continue
            out.append(ch)
            i += 1
        return "".join(out)

    @staticmethod
    def _parse_json_text(text):
        """解析 JSON 字符串：整体解析 → 去掉注释后整体解析 → raw_decode 逐段解析多个拼接对象并合并（同名键后者覆盖）。"""
        s = text.strip() if isinstance(text, str) else text
        if not s:
            return {}
        try:
            obj = _json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        # AI 生成的 JSON 可能夹整行 // 注释（集界标记），去掉注释后重新整体解析
        cleaned = YUAN_TXTJsonExtractor._strip_json_comments(s)
        if cleaned != s:
            try:
                obj = _json.loads(cleaned)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            # 兜底逐段解析同样要在去注释后的文本上进行，否则每个 { 都会被注释卡死、
            # 只剩碎片键（例如多文件拼接时顶层键 角色档案/分镜序列 全部丢失）
            s = cleaned
        # 多个 JSON 对象拼接：逐段 raw_decode 解析并合并
        decoder = _json.JSONDecoder()
        data = {}
        i = 0
        n = len(s)
        while i < n:
            while i < n and s[i] in " \t\r\n":
                i += 1
            if i >= n:
                break
            if s[i] not in "{[":
                i += 1
                continue
            try:
                obj, end = decoder.raw_decode(s, i)
            except Exception:
                i += 1
                continue
            if isinstance(obj, dict):
                data.update(obj)
            i = end
        return data

    # 名称分隔符：名称与描述之间可能是中/英文逗号、冒号、分号或顿号（JSON 由 AI 生成，标点不固定）
    NAME_SEPARATORS = ("，", ",", "：", ":", "；", ";", "、")
    # 名称别名分隔符：括号/方括号/冒号之后是别名或描述，例如「周明瑞（克莱恩）」「周明瑞: 描述」
    NAME_ALIAS_RE = re.compile(r'[（(【\[：:]')

    @classmethod
    def _before_first_comma(cls, s):
        """取名称部分：第一个中/英文逗号、冒号、分号或顿号之前的部分（无分隔符返回整串）。
        按最早出现的分隔符切分——档案描述里常含另一种分隔符，不能固定优先某一种。"""
        text = str(s) if s is not None else ""
        cut = -1
        for sep in cls.NAME_SEPARATORS:
            p = text.find(sep)
            if p != -1 and (cut == -1 or p < cut):
                cut = p
        return (text[:cut] if cut != -1 else text).strip()

    @classmethod
    def _extract_name(cls, entry):
        """从档案条目中提取名称（第一个逗号/冒号/分号/顿号前的部分）。"""
        return cls._before_first_comma(entry)

    @classmethod
    def _name_core(cls, name):
        """名称主干：去掉括号别名与尾部描述后的短名（「周明瑞（克莱恩）」→「周明瑞」）。"""
        base = cls._before_first_comma(name)
        return cls.NAME_ALIAS_RE.split(base, maxsplit=1)[0].strip()

    @classmethod
    def _format_archive_entry(cls, entries, index):
        """档案模式：取该档案第 index 条（0 基），输出「全名称 + 描述」两行。

        条目约定为「全名，描述」；描述前的分隔标点去掉，描述内的「；补充：」原样保留。
        越界或空档案返回空串。
        """
        if not isinstance(entries, (list, tuple)) or not entries:
            return ""
        try:
            i = int(index)
        except (TypeError, ValueError):
            i = 0
        if not 0 <= i < len(entries):
            return ""
        text = entries[i] if isinstance(entries[i], str) else str(entries[i])
        text = text.strip()
        name = cls._before_first_comma(text)
        desc = text[len(name):].lstrip(" \t，,：:；;、。")
        return "\n".join(part for part in (name, desc.strip()) if part)

    @staticmethod
    def _find_appearing_indices(text, char_names, prop_names):
        """在非台词区域（引号与 <d> 内不匹配）查找出现的角色和道具（最长匹配优先），返回按首次出现位置排序的 (角色索引列表, 道具索引列表)。"""
        if not text:
            return [], []

        mask = _build_dialogue_protect_mask(text)
        n = len(text)

        # 收集所有名称：(name, type, index, length)；带括号/方括号后缀的条目额外登记「括号前」的别名，
        # 例如档案「油纸伞（黑）」，正文只写「油纸伞」也能匹配（最长优先，全名仍优先命中）
        all_names = []
        for names, ntype in ((char_names, 'char'), (prop_names, 'prop')):
            for i, name in enumerate(names):
                if not name:
                    continue
                all_names.append((name, ntype, i, len(name)))
                alias = re.split(r'[（(【\[:：]', name, maxsplit=1)[0].strip()
                if len(alias) >= 2 and alias != name:
                    all_names.append((alias, ntype, i, len(alias)))

        consumed = [False] * n
        char_found = {}   # index -> first_pos
        prop_found = {}   # index -> first_pos

        for name, ntype, idx, m in all_names:
            if m == 0 or m > n:
                continue
            i = 0
            while i <= n - m:
                # 窗口内有保护或已消费位置则跳过
                blocked = False
                for k in range(m):
                    if mask[i + k] or consumed[i + k]:
                        blocked = True
                        break
                if not blocked and text[i:i + m] == name:
                    pos = i
                    if ntype == 'char':
                        if idx not in char_found:
                            char_found[idx] = pos
                    else:
                        if idx not in prop_found:
                            prop_found[idx] = pos
                    for k in range(m):
                        consumed[i + k] = True
                    i += m
                else:
                    i += 1

        char_indices = [idx for idx, _ in sorted(char_found.items(), key=lambda x: x[1])]
        prop_indices = [idx for idx, _ in sorted(prop_found.items(), key=lambda x: x[1])]
        return char_indices, prop_indices

    @staticmethod
    def _max_duration_seconds(text):
        """提取时间段文本中的最大结束时间（秒）：仅匹配行首的「开始-结束」时段标记（支持 s/秒/无单位、小数），避免误匹配其他数字。"""
        if not text:
            return 0.0
        pattern = r'(?:^|\n)\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:s|秒)?'
        matches = re.findall(pattern, text)
        if not matches:
            return 0.0
        # 取每段的结束时间（第二个数字），返回最大值
        end_times = [float(m[1]) for m in matches]
        return max(end_times) if end_times else 0.0

    @staticmethod
    def _type_duration_seconds(type_str):
        """从分镜「类型」字段提取锁定时长（秒）：取字符串中第一个数字；无数字或为空时返回 None。"""
        if not type_str:
            return None
        m = re.search(r'(\d+(?:\.\d+)?)', str(type_str))
        if not m:
            return None
        return float(m.group(1))


    @staticmethod
    def _shot_type_code(type_str):
        """分镜类型编码：文戏 0、武戏 1、MV 2、其他（含未命中）3。"""
        text = str(type_str or "")
        if "武戏" in text:
            return 1
        if "文戏" in text:
            return 0
        if "MV" in text.upper():
            return 2
        return 3

    # ==== 官方分段：时间码 MM:SS.mmm / 定义段 subject_definitions ====
    # 节点不再生成镜头编号：正文里残留的手写 [Shot N] 一律丢弃
    SHOT_MARK_RE = re.compile(r'^\s*\[Shot\s+\d+\]')
    OVERVIEW_RE = re.compile(r'^\s*(?:本镜\s*\d+(?:\.\d+)?\s*秒|This shot runs)', re.I)
    CONNECTOR_RE = re.compile(
        r'^\s*(the camera cuts to|the shot cuts to|the shot transitions to|'
        r'the shot changes to|the shot switches to)', re.I)

    SUPPLEMENT_RE = re.compile(r'(?:补充|supplement)\s*[：:]', re.IGNORECASE)
    REF_TAG_RE = re.compile(r'\s*<(?:Picture|Subject|Audio)\s+\d+>')

    @staticmethod
    def _fmt_ts(seconds):
        """秒 → 官方时间码 MM:SS.mmm。"""
        ms = int(round(max(0.0, float(seconds)) * 1000))
        return "%02d:%02d.%03d" % (ms // 60000, (ms % 60000) // 1000, ms % 1000)

    # 配速基准：中文口播 4.2 字/秒、英文 3.0 词/秒；动作单元 0.3 秒/个；基础 1.2 秒
    SPEECH_CPS_CN, SPEECH_CPS_EN, ACTION_UNIT_SEC, BEAT_BASE = 4.2, 3.0, 0.3, 1.2
    BEAT_MIN, BEAT_MAX = 1.5, 4.5      # 单段合理耗时区间（秒），避免 1 秒闪切或 6 秒僵镜

    @classmethod
    def _beat_seconds(cls, text):
        """一段的预估耗时（秒）：台词按字/词语速，动作按可数单元，静默给基础值；
        单段夹在 BEAT_MIN..BEAT_MAX。时间码由这里生成——写手只会按时长÷段数均分，均分就是机械感的来源。"""
        t = str(text or "")
        cn = en = 0
        for m in re.finditer(r'<d>(.*?)</d>', t, re.S):
            inner = re.sub(r'^\[[^\]]*\]', '', m.group(1))
            cn += len(re.findall(r'[\u4e00-\u9fff]', inner))
            en += len(re.findall(r"[A-Za-z]+", inner))
        speech = cn / cls.SPEECH_CPS_CN + en / cls.SPEECH_CPS_EN
        core = re.sub(r'<d>.*?</d>', '', t, flags=re.S)
        units = len([x for x in re.split(r'[，、；,;]', core) if len(x.strip()) >= 6])
        sec = cls.BEAT_BASE + speech + min(units, 8) * cls.ACTION_UNIT_SEC
        return min(max(sec, cls.BEAT_MIN), cls.BEAT_MAX)

    @classmethod
    def _distribute_times(cls, weights, duration):
        """把每段耗时缩放到镜长：先按比例缩放，余量均摊，再保证每段不低于 1.0 秒。"""
        n = len(weights)
        total = sum(weights) or 1.0
        dur = float(duration) if duration else total
        k = dur / total
        secs = [w * k for w in weights]
        delta = dur - sum(secs)
        secs = [max(1.0, x + delta / n) for x in secs]
        # 仍不足镜长时按比例补回（只补不削）
        gap = dur - sum(secs)
        if gap > 0:
            secs = [x + gap * x / sum(secs) for x in secs]
        stamps, acc = [], 0.0
        for x in secs[:-1]:
            stamps.append(acc)
            acc += x
        stamps.append(acc)
        return stamps

    @classmethod
    def _build_shot_blocks(cls, beats, duration):
        """官方分段，时间码由本节点规划（正文不带镜头编号，不生成 `[Shot N]`）：
        - 首段写纯正文（参考样例：`The sequence starts with a close-up…`）；
        - 其余节拍段 `At MM:SS.mmm, 承接语, 正文`。
        写手手写的 `[Shot N]` 一律丢弃，时间码按段内容耗时分配。
        「整体风格 / 画面质感 / 场景 / 在场位置 / 运镜方式」不进正文，由 `_build_style_header` 统一写在
        detailed_description 的风格头两行里（在场位置/运镜方式先按本函数同一套规矩清洗 `_clean_header_segment`）。"""
        raw = [str(b).strip() for b in beats if b is not None and str(b).strip()]
        if not raw:
            return []
        # 旧格式的「本镜 N 秒 / This shot runs N seconds」梗概段：只要首段是就迁走，
        # 不受同镜其它段是否已带官方标记影响（一份稿里新旧混排是常态）
        migrated = cls._migrate_legacy_beats(raw)
        if migrated:
            raw = migrated

        bodies = []
        for b in raw:
            body = cls.SHOT_MARK_RE.sub("", b).strip()      # 丢掉手写编号
            body = re.sub(r'^At\s+\d{1,2}:\d{2}\.\d{1,3},\s*', '', body).strip()
            if body:
                bodies.append(body)
        if not bodies:
            return []

        stamps = cls._distribute_times([cls._beat_seconds(b) for b in bodies], duration)
        out = [bodies[0]]
        for i, body in enumerate(bodies[1:], start=1):
            out.append("At %s, %s" % (cls._fmt_ts(stamps[i]), body))
        return out

    # 风格头三项（场景/在场位置/运镜方式）沿用「时间段」的替换规矩：丢手写 [Shot N]、
    # 去首部 At 时间码、压空白、去句尾标点；名称替换由 _replace_names 在调用前完成
    @classmethod
    def _clean_header_segment(cls, text):
        """风格头字段清洗：与时间段同一套规矩，句尾标点交给 _build_style_header 统一补。"""
        s = cls.SHOT_MARK_RE.sub("", str(text or "")).strip()
        s = re.sub(r'^At\s+\d{1,2}:\d{2}(?:\.\d{1,3})?\s*[,，]?\s*', '', s).strip()
        s = re.sub(r'[ \t]{2,}', " ", s).strip()
        return s.rstrip("。.；;，,、 \t")

    @classmethod
    def _build_style_header(cls, style, texture, scene, place, camera, english=False):
        """detailed_description 首部风格头（两行）：整体风格 + 画面质感 + 场景 + 在场位置 + 运镜方式。
        中文模板：`目标视频为<整体风格>风格，角色一致性必须严格把控。画面质感：<画面质感>。`
                  `场景：<场景>，<在场位置>。<运镜方式>`
        英文模板：`The target video is <整体风格>. Visual quality: <画面质感>.`
                  `Stage: <场景>, <在场位置>. <运镜方式>`
        缺哪项就省哪项（在场位置缺失时退回 `场景：<场景>。<运镜方式>`）；整体风格已带「风格」二字时不重复追加。
        场景/在场位置/运镜方式由调用方先按时间段规矩清洗、再做完角色/道具/场景名称替换。"""
        style = str(style or "").strip().rstrip("。.")
        texture = str(texture or "").strip().rstrip("。.")
        scene = str(scene or "").strip().rstrip("。.")
        place = str(place or "").strip().rstrip("。.")
        camera = str(camera or "").strip()
        line1, line2 = [], []
        if english:
            if style:
                line1.append("The target video is %s." % style)
            if texture:
                line1.append("Visual quality: %s." % texture)
            if scene and place:
                line2.append("Stage: %s, %s." % (scene, place))
            elif scene or place:
                line2.append("Stage: %s." % (scene or place))
            if camera:
                line2.append(camera if camera.endswith((".", "。")) else camera + ".")
        else:
            if style:
                line1.append("目标视频为%s%s，角色一致性必须严格把控。"
                             % (style, "" if style.endswith("风格") else "风格"))
            if texture:
                line1.append("画面质感：%s。" % texture)
            if scene and place:
                line2.append("场景：%s，%s。" % (scene, place))
            elif scene or place:
                line2.append("场景：%s。" % (scene or place))
            if camera:
                line2.append(camera if camera.endswith(("。", ".")) else camera + "。")
        sep = " " if english else ""
        return "\n".join(x for x in (sep.join(line1), sep.join(line2)) if x)

    @classmethod
    def _migrate_legacy_beats(cls, beats):
        """旧格式迁移：首段「本镜N秒，画面呈现…；站位；光影；气氛」只丢梗概子句，
        其余子句并进下一段，站位、光影与气氛信息不丢。"""
        first = str(beats[0])
        if not cls.OVERVIEW_RE.match(first):
            return list(beats)
        clauses = [c.strip() for c in re.split(r'[；;]', first) if c.strip()]
        rest = "；".join(clauses[1:])
        migrated = [str(b) for b in beats[1:] if str(b).strip()]
        if not migrated:
            return list(beats)          # 只有梗概段时保留原文，不整镜丢内容
        if rest:
            migrated[0] = migrated[0].rstrip("。 ") + "；" + rest
        return migrated

    @classmethod
    def _definition_text(cls, text, keep_supplement):
        """档案条目 → 定义描述：角色保留「补充」里的配装并去掉标记，其余截断补充。"""
        text = str(text).strip().rstrip("。.")
        m = cls.SUPPLEMENT_RE.search(text)
        if not m:
            return text
        head = text[:m.start()].rstrip(" ；;，,、")
        tail = text[m.end():].strip().rstrip("。.")
        if not keep_supplement or not tail:
            return head
        sep = "，" if re.search(r'[\u4e00-\u9fff]', head) else ", "
        return "%s%s%s" % (head, sep, tail)

    @staticmethod
    def _subject_no(tag):
        """<Picture N> → N（兼容旧稿的 <Subject N>，两者同号）。"""
        m = re.match(r'^<(?:Subject|Picture) (\d+)>$', str(tag or ""))
        return int(m.group(1)) if m else 0

    @classmethod
    def _definition_parts(cls, text, keep_supplement):
        """档案条目 → (名称, 描述)：先处理「补充：」，再从最早出现的分隔符拆成名称 + 描述。
        角色保留补充里的配装（「补充：」标记去掉），其余条目在「补充：」处截断。"""
        full = cls._definition_text(text, keep_supplement)
        name = cls._before_first_comma(full)
        desc = full[len(name):].lstrip("，,：:；;、 \t").strip().rstrip("。.；;，,、 \t")
        return name or str(text).strip(), desc

    @classmethod
    def _build_definitions(cls, entries, voice_lines, summary):
        """官方定义顺序：subject_definitions（定义行 + 声音引用行）→ summary。

        定义行固定为 `<Picture N> is <名称> for the reference,<描述>.`；没有描述时退回
        `<Picture N> is <名称> for the reference.`。不再输出保留分析行
        （`<Picture N> (appears throughout): fully_preserved - ...`）。
        情节衔接条目不拆名称/描述（整句原文当名称），避免把一句话按逗号切碎。
        """
        lines = []
        for i, (kind, t) in enumerate(entries):
            if kind == "衔接":
                name, desc = cls._definition_text(t, False), ""
            else:
                name, desc = cls._definition_parts(t, kind == "角色")
            lines.append("<Picture %d> is %s for the reference%s."
                         % (i + 1, name, ("," + desc) if desc else ""))
        lines.extend(voice_lines)
        parts = []
        if lines:
            parts.append("subject_definitions:\n" + "\n".join(lines))
        if summary:
            parts.append("summary:\n" + summary)
        return "\n\n".join(parts)

    @classmethod
    def _build_summary(cls, shot_type, plot):
        """official summary：固定片种前缀 + 本镜「情节」原文（不再合成类型/参考关系描述）。
        时长取「类型」字段里的秒数（如「文戏 10秒」→ 10-second）；无秒数时省略时长部分。"""
        m = re.search(r'(\d+)\s*秒', str(shot_type or ""))
        head = ("[reference generation] A %s-second continuous, playful vlog-style sequence: " % m.group(1)
                if m else "[reference generation] A continuous, playful vlog-style sequence: ")
        return head + str(plot or "").strip()

    @staticmethod
    def _replace_names(text, pairs, protect):
        """名称 → <Picture N>：pairs 按最长优先排好；台词保护开启时引号对与 <d> 内不替换。
        detailed_description 与 summary（「情节」）共用这一份替换规则。"""
        for name, tag in pairs:
            if protect:
                text = YUAN_TXTReplace._replace_with_protect(
                    text, name, tag, _build_dialogue_protect_mask(text))
            else:
                text = text.replace(name, tag)
        return text

    @classmethod
    def _clean_sound(cls, text):
        """声音字段：去掉参考标签与多余空白（官方要求纯声学 / 纯音乐描述，不引用参考）。"""
        s = cls.REF_TAG_RE.sub("", str(text or ""))
        return re.sub(r'[ \t]{2,}', " ", s).strip()

    @staticmethod
    def _extract_scene_prefix(title):
        """从分镜标题提取场景关键词：取「—」或「-」前部分。"""
        if not title:
            return ""
        prefix = title
        for sep in ("—", "-"):
            if sep in prefix:
                prefix = prefix.split(sep)[0]
                break
        return prefix.strip()

    @staticmethod
    def _match_scene_index(title, scenes):
        """根据分镜标题智能匹配场景档案索引：标题取「-」前部分、档案取「，」前部分，双向包含匹配，匹配不到返回 -1。"""
        if not title or not isinstance(scenes, list) or not scenes:
            return -1
        prefix = YUAN_TXTJsonExtractor._extract_scene_prefix(title)
        if not prefix:
            return -1
        for idx, scene in enumerate(scenes):
            scene_name = YUAN_TXTJsonExtractor._before_first_comma(scene)
            if not scene_name:
                continue
            if prefix in scene_name or scene_name in prefix:
                return idx
        return -1

    @classmethod
    def _match_name_index(cls, name, archive_names):
        """名称 → 档案 0 基序号：精确 → 主干（去括号别名）相等 → 双向包含（最长优先）；匹配不到返回 -1。

        AI 生成的 JSON 里同一角色的写法常不一致（全名/别名/带括号/带冒号描述），
        只做精确相等会让角色整条丢掉，因此逐级放宽。
        """
        target = cls._before_first_comma(name)
        if not target:
            return -1
        for i, archive_name in enumerate(archive_names):
            if archive_name and target == archive_name:
                return i
        target_core = cls._name_core(target)
        if len(target_core) >= 2:
            for i, archive_name in enumerate(archive_names):
                if archive_name and target_core == cls._name_core(archive_name) \
                        and len(cls._name_core(archive_name)) >= 2:
                    return i
        if len(target) < 2:
            return -1
        best, best_len = -1, 0
        for i, archive_name in enumerate(archive_names):
            if not archive_name:
                continue
            for cand in (archive_name, cls._name_core(archive_name)):
                if len(cand) < 2:
                    continue
                if (cand in target or target in cand) and len(cand) > best_len:
                    best, best_len = i, len(cand)
        return best

    @classmethod
    def _archive_indices_by_names(cls, appearing, archive_names):
        """按显式出现名称列表解析档案 0 基索引（去重、保持顺序）。

        非列表、或列表里一个名称也匹配不到时返回 None，让调用方回退到正文匹配
        （AI 偶尔会写空数组或写英文别名，此时不能把角色/道具整条丢掉）。
        """
        if not isinstance(appearing, list):
            return None
        result = []
        for entry in appearing:
            if entry is None:
                continue
            idx = cls._match_name_index(str(entry), archive_names)
            if idx >= 0 and idx not in result:
                result.append(idx)
        return result or None

    @staticmethod
    def _build_belong_pattern(name):
        """构建台词归属 pattern：`说话者(<Audio M>)?+引导句+冒号+<d>台词`；引导句允许逗号顿号但不跨句、不跨冒号。"""
        boundary = YUAN_TXTJsonExtractor.BELONG_BOUNDARY
        return re.compile(
            r'(?:(?<=</d>)|(?<![^\s' + boundary + r']))'  # 说话者前边界：句末标点/逗号/空白 或 </d> 之后
            r'(' + re.escape(name) + r')'        # 组1：说话者（名称或 <Picture N>）
            r'(?:<Audio \d+>)?'                  # 可选：已有的 <Audio M>（重入时跳过，替换时重写）
            r'([^。！？!?；;\n\r：:]*?)'          # 组2：引导句（允许逗号顿号，不得跨句/跨冒号）
            r'[：:]\s*'                          # 冒号+可选空白
            r'(?=<d>)'                           # 紧跟台词标签
        )

    @staticmethod
    def _parse_voice_indices(text):
        """解析逗号分隔的整数索引串（如「0,4,1」），容错中文逗号/空格/非法项。"""
        if not text:
            return []
        values = []
        for part in re.split(r'[,，]', str(text)):
            part = part.strip()
            if part and re.fullmatch(r'-?\d+', part):
                values.append(int(part))
        return values

    @staticmethod
    def _scan_speaking_order(text, pairs, protect_enabled):
        """扫描分镜序列，按说话角色首次说话的出现顺序分配 <Audio M>（最多3个）。

        识别 `名称+引导句+冒号+<d>` 与 `<Picture N>+引导句+冒号+<d>` 两种说话形式；
        同一台词多候选说话者时取离冒号最近者（起点相同取名称更长者）；
        标准规则未命中时从冒号往前追溯最近的未保护候选（引号内/<d>内跳过）；
        台词保护开启时被保护位置不算说话；同一角色只按首次出现分配一个编号。

        返回 (audio_by_tag, speaking_tags, winners)：
        - audio_by_tag：{<Picture N>: "<Audio M>"}，超过3个不分配
        - speaking_tags：按 <Audio 1/2/3> 分配顺序的标记列表，用于音色索引
        - winners：[(台词位置, 替换起点, 替换终点, 标记)]，替换区间覆盖说话者及已有 <Audio M>（重入重写）
        """
        mask = _build_dialogue_protect_mask(text) if protect_enabled else None
        # 待匹配的「说话者」形式：名称 + <Picture N> 标记（去重）
        forms = []
        for name, tag in pairs:
            forms.append((name, tag))
        for _name, tag in pairs:
            if (tag, tag) not in forms:
                forms.append((tag, tag))

        # 收集所有候选命中：(台词目标位置, 说话者起点, 说话者终点, 替换终点, 标记)
        hits = []
        for form, tag in forms:
            for m in YUAN_TXTJsonExtractor._build_belong_pattern(form).finditer(text):
                if mask is not None:
                    if any(mask[k] for k in range(m.start(1), m.end(1))):
                        continue
                hits.append((m.end(), m.start(1), m.end(1), m.start(2), tag))
        # 同一台词目标的候选者竞争：离冒号最近（起点最大）者胜，起点相同取名称更长者
        by_target = {}
        for end, s1, e1, s2, tag in hits:
            cur = by_target.get(end)
            if cur is None or (s1, e1) > (cur[1], cur[2]):
                by_target[end] = (end, s1, e1, s2, tag)

        # 兜底追溯：标准规则未命中的台词目标，从冒号往前追溯最近的未保护候选
        # 候选 pattern 含可选的已有 <Audio M>（重入时替换区间覆盖、重写编号）
        backtrack_pats = [
            (re.compile(re.escape(form) + r'(?:<Audio \d+>)?'), tag)
            for form, tag in forms
        ]
        for m in re.finditer(r'[：:]\s*(?=<d>)', text):
            target = m.end()
            if target in by_target:
                continue
            best = None  # (起点, 终点, 标记)
            for pat, tag in backtrack_pats:
                for bm in pat.finditer(text, 0, target):
                    s, e = bm.start(), bm.end()
                    if mask is not None and any(mask[k] for k in range(s, e)):
                        continue  # 引号内/<d>内的候选跳过，继续往前追溯
                    if best is None or (s, e) > (best[0], best[1]):
                        best = (s, e, tag)
            if best is not None:
                by_target[target] = (target, best[0], best[1], best[1], best[2])

        winners = sorted(by_target.values(), key=lambda x: x[0])

        audio_by_tag = {}
        speaking_tags = []
        next_num = 1
        for _end, _s1, _e1, _s2, tag in winners:
            if tag not in audio_by_tag:
                if next_num <= 3:
                    audio_by_tag[tag] = f"<Audio {next_num}>"
                    speaking_tags.append(tag)
                    next_num += 1
        return audio_by_tag, speaking_tags, winners

    # 七个选项开关的聚合键序（「JSON提取开关」子节点的输出与其一致）
    SWITCH_KEYS = ("角色开关", "音色开关", "道具开关", "场景开关", "BGM开关", "台词开关", "情节衔接开关")
    # 未接入子节点时的默认值（除情节衔接默认关闭外，其余默认开启）
    SWITCH_DEFAULTS = (True, True, True, True, True, True, False)

    @classmethod
    def _parse_switch_config(cls, 开关配置):
        """解析「开关配置」为七个布尔值元组：None→默认值；dict→按键名取值（缺失键取默认）；
        7元列表/元组→按 SWITCH_KEYS 顺序取值；其他类型容错回退默认值。"""
        if 开关配置 is None:
            return cls.SWITCH_DEFAULTS
        values = dict(zip(cls.SWITCH_KEYS, cls.SWITCH_DEFAULTS))
        if isinstance(开关配置, dict):
            for k in cls.SWITCH_KEYS:
                v = 开关配置.get(k)
                if isinstance(v, bool):
                    values[k] = v
        elif isinstance(开关配置, (list, tuple)) and len(开关配置) == len(cls.SWITCH_KEYS):
            for k, v in zip(cls.SWITCH_KEYS, 开关配置):
                if isinstance(v, bool):
                    values[k] = v
        return tuple(values[k] for k in cls.SWITCH_KEYS)

    def extract_json(self, json=None, 索引=1, 档案选择="角色档案", 开关配置=None, 情节衔接文本=None, 模式="分镜模式"):
        # 形参名与输入端口名一致；七个选项开关来自「开关配置」（未接入时取默认值）
        角色开关, 音色开关, 道具开关, 场景开关, BGM开关, 台词开关, 情节衔接开关 = self._parse_switch_config(开关配置)
        # 情节衔接文本：未接入时不新增行（空文本同样不新增）
        衔接文本 = self._list_to_lines(情节衔接文本).strip() if 情节衔接文本 is not None else ""
        data = json

        # 字符串自动解析为 dict（支持多个 JSON 对象拼接合并）
        if isinstance(data, str):
            data = self._parse_json_text(data)

        if not isinstance(data, dict):
            data = {}

        # 整体风格
        整体风格 = self._list_to_lines(data.get("整体风格", ""))
        # 各类档案数据（内部保留，角色/道具档案用于索引匹配）
        角色档案数据 = data.get("角色档案", [])
        音色档案数据 = data.get("音色档案", [])
        道具档案数据 = data.get("道具档案", [])
        场景档案数据 = data.get("场景档案", [])

        # 档案：按「档案选择」输出对应档案内容
        档案表 = {
            "角色档案": 角色档案数据,
            "音色档案": 音色档案数据,
            "道具档案": 道具档案数据,
            "场景档案": 场景档案数据,
        }
        档案条目 = 档案表.get(档案选择, 角色档案数据)
        # 档案模式：索引改指该档案里的第几条（0 基），只输出这一条（全名 + 描述两行）；
        # 分镜模式：索引是分镜编号，档案端口输出整份档案（该端口在分镜模式下本就隐藏）
        档案 = (self._format_archive_entry(档案条目, 索引) if 模式 == "档案模式"
                else self._list_to_lines(档案条目))

        # 档案编码：按「档案选择」输出对应编码（角色档案=0、音色档案=1、道具档案=2、场景档案=3）
        档案编码 = {
            "角色档案": 0,
            "音色档案": 1,
            "道具档案": 2,
            "场景档案": 3,
        }.get(档案选择, 0)

        # 分镜序列：按编号选取对应分镜（时间段/环境音/BGM）
        分镜序列数据 = data.get("分镜序列", [])
        分镜情节数据 = data.get("分镜情节", [])
        分镜序列文本 = ""
        时间段 = []
        环境音 = ""
        BGM = ""
        画面质感item = ""
        运镜方式item = ""
        在场位置item = ""
        matched_title = ""
        matched_type = ""
        情节条目 = None
        if isinstance(分镜情节数据, list):
            for item in 分镜情节数据:
                if isinstance(item, dict) and item.get("编号") == 索引:
                    情节条目 = item
                    break
        found_shot = 情节条目 is not None
        if isinstance(分镜序列数据, list):
            for item in 分镜序列数据:
                if isinstance(item, dict) and item.get("编号") == 索引:
                    时间段 = item.get("时间段", [])
                    分镜序列文本 = self._list_to_lines(时间段)
                    环境音 = self._list_to_lines(item.get("环境音", ""))
                    BGM = self._list_to_lines(item.get("BGM", ""))
                    画面质感item = self._list_to_lines(item.get("画面质感", ""))
                    运镜方式item = self._list_to_lines(item.get("运镜方式", ""))
                    在场位置item = self._list_to_lines(item.get("在场位置", ""))
                    matched_title = str(item.get("标题", ""))
                    matched_type = str(item.get("类型", ""))
                    found_shot = True
                    break
        # 仅命中分镜情节而未命中分镜序列时，标题/类型取自分镜情节条目
        if 情节条目 is not None and not matched_title:
            matched_title = str(情节条目.get("标题", ""))
        if 情节条目 is not None and not matched_type:
            matched_type = str(情节条目.get("类型", ""))
        # 「情节」只写进 summary：取当前分镜（编号=索引）情节条目的原文
        情节原文 = str(情节条目.get("情节") or "").strip() if 情节条目 is not None else ""
        # 分镜序列按官方分段输出（未命中分镜时时间段/环境音/BGM 留空，不报错；
        # BGM开关关闭时 non_diegetic_music 输出 N/A）
        # 镜头级风格字段：优先取「分镜情节」条目（阶段一新增在「出现场景」之后），
        # 其次退回「分镜序列」条目；都没有时为空，风格头里就省掉该项
        画面质感 = (str(情节条目.get("画面质感") or "").strip() if 情节条目 is not None else "") or 画面质感item
        在场位置 = (str(情节条目.get("在场位置") or "").strip() if 情节条目 is not None else "") or 在场位置item
        运镜方式 = (str(情节条目.get("运镜方式") or "").strip() if 情节条目 is not None else "") or 运镜方式item
        分镜序列整合 = "\n".join(
            self._build_shot_blocks(时间段, self._type_duration_seconds(matched_type) or 0.0))

        # 角色道具场景定义（内部变量）：条目按 角色→道具→场景 顺序，生成 subject_definitions 供前置与名称替换
        # 角色/道具/场景索引：匹配到的档案 0 基序号（角色/道具逗号分隔，场景单个），未匹配为空
        条目列表 = []
        场景名称 = ""
        if not found_shot:
            角色道具场景 = ""
            角色索引 = ""
            道具索引 = ""
            场景索引 = ""
            索引时长 = 0.0
            分镜类型 = 3
        else:
            # 角色、道具：优先取「分镜情节」条目的 出现角色/出现道具 显式列表；未提供时在分镜时间段文本中智能匹配
            char_names = [self._extract_name(e) for e in (角色档案数据 if isinstance(角色档案数据, list) else [])]
            prop_names = [self._extract_name(e) for e in (道具档案数据 if isinstance(道具档案数据, list) else [])]
            出现角色 = 情节条目.get("出现角色") if 情节条目 is not None else None
            出现道具 = 情节条目.get("出现道具") if 情节条目 is not None else None
            char_indices = self._archive_indices_by_names(出现角色, char_names)
            prop_indices = self._archive_indices_by_names(出现道具, prop_names)
            if char_indices is None or prop_indices is None:
                # 匹配范围：标题 + 时间段 + 情节文本 + 环境音（只拿时间段会漏掉「情节/标题里提到的道具」）
                情节文本 = str(情节条目.get("情节") or "") if 情节条目 is not None else ""
                查找文本 = "\n".join(x for x in (matched_title, 分镜序列文本, 情节文本, 环境音) if x)
                text_char, text_prop = self._find_appearing_indices(查找文本, char_names, prop_names)
                if char_indices is None:
                    char_indices = text_char
                if prop_indices is None:
                    prop_indices = text_prop
            角色描述列表 = []
            if isinstance(角色档案数据, list):
                for i in char_indices:
                    if 0 <= i < len(角色档案数据) and 角色档案数据[i] is not None:
                        角色描述列表.append(str(角色档案数据[i]))
            道具描述列表 = []
            if isinstance(道具档案数据, list):
                for i in prop_indices:
                    if 0 <= i < len(道具档案数据) and 道具档案数据[i] is not None:
                        道具描述列表.append(str(道具档案数据[i]))

            # 场景：优先取「分镜情节」条目的 出现场景；未提供时根据分镜标题智能匹配
            出现场景 = 情节条目.get("出现场景") if 情节条目 is not None else None
            scene_names = [self._extract_name(e) for e in (场景档案数据 if isinstance(场景档案数据, list) else [])]
            idx = -1
            if isinstance(出现场景, str) and 出现场景.strip() and scene_names:
                target = 出现场景.strip()
                for si, sname in enumerate(scene_names):
                    if target == sname or (sname and (target in sname or sname in target)):
                        idx = si
                        break
            if idx < 0:
                idx = self._match_scene_index(matched_title, 场景档案数据)
            场景描述 = ""
            if idx >= 0 and isinstance(场景档案数据, list) and idx < len(场景档案数据) and 场景档案数据[idx] is not None:
                场景描述 = str(场景档案数据[idx])
            # 风格头里的场景名：优先「出现场景」原文，其次匹配到的场景档案名（不受场景开关影响）
            场景名称 = (str(出现场景).strip() if isinstance(出现场景, str) and 出现场景.strip()
                        else self._before_first_comma(场景描述))

            # 角色/道具/场景索引：对应档案的 0 基序号，未匹配为空；由对应输出开关控制，关时输出空文本
            角色索引 = ",".join(str(i) for i in char_indices) if 角色开关 else ""
            道具索引 = ",".join(str(i) for i in prop_indices) if 道具开关 else ""
            场景索引 = (str(idx) if idx >= 0 else "") if 场景开关 else ""

            # 定义段条目：角色 → 道具 → 场景 →（可选）情节衔接；顺序即 <Picture N> 与参考图端口序
            条目列表 = []
            if 角色开关:
                条目列表.extend(("角色", d) for d in 角色描述列表)
            if 道具开关:
                条目列表.extend(("道具", d) for d in 道具描述列表)
            if 场景开关 and 场景描述:
                条目列表.append(("场景", 场景描述))
            衔接行 = 衔接文本 if (情节衔接开关 and 衔接文本) else ""
            if 衔接行:
                条目列表.append(("衔接", 衔接行))
            角色道具场景 = ""

            # 索引时长：优先取「类型」字段中的时长；未含数字时回退为时间段文本中的最大结束时间
            索引时长 = self._type_duration_seconds(matched_type)
            if 索引时长 is None:
                索引时长 = self._max_duration_seconds(分镜序列文本)

            # 分镜类型：取当前分镜（编号=索引）「类型」字段的编码（文戏 0 / 武戏 1 / MV 2 / 其他 3）
            分镜类型 = self._shot_type_code(matched_type)

        # 场景衔接：当前分镜（编号=索引）与上一个分镜（编号-1）的场景关键词是否相同（取「-」前部分比较）；索引为1（无前参考）或找不到上一编号则输出 False
        场景判断 = False
        if found_shot and isinstance(分镜序列数据, list):
            prev_title = ""
            prev_found = False
            for item in 分镜序列数据:
                if isinstance(item, dict) and item.get("编号") == 索引 - 1:
                    prev_title = str(item.get("标题", ""))
                    prev_found = True
                    break
            if prev_found:
                curr_prefix = self._extract_scene_prefix(matched_title)
                prev_prefix = self._extract_scene_prefix(prev_title)
                if curr_prefix and prev_prefix and curr_prefix == prev_prefix:
                    场景判断 = True

        # ==== 官方形态组装 ====
        # subject_definitions（含保留分析行）→ summary → detailed_description → 两个声音字段
        # 名称替换：名称 → <Picture N>（Subject 与 Picture 同号，顺序即参考图端口序）
        pairs = []
        for i, (_kind, text) in enumerate(条目列表):
            name = self._before_first_comma(text)
            if not name:
                continue
            tag = "<Picture %d>" % (i + 1)
            pairs.append((name, tag))
            # 别名也登记：档案写全名带括号/别名时，正文只写短名同样替换（同一编号）
            core = self._name_core(name)
            if len(core) >= 2 and core != name:
                pairs.append((core, tag))
        pairs.sort(key=lambda x: -len(x[0]))   # 最长匹配优先，避免短名误替换长名中的子串

        分镜序列输出 = 分镜序列整合 or ""
        音色索引 = ""
        voice_lines = []

        # 说话者：追加 (Sx)（S 号 = <Audio M> 号 = 本镜首次发声顺序）；<Audio M> 只写进定义段，
        # 不再插在句子主语中间；归属算法沿用 _scan_speaking_order（台词保护、离冒号最近者胜）
        if 音色开关 and pairs:
            audio_by_tag, speaking_tags, winners = self._scan_speaking_order(分镜序列输出, pairs, 台词开关)
            replaced_ranges = []
            for _end, s1, _e1, s2, tag in sorted(winners, key=lambda x: -x[0]):
                sid = re.sub(r'\D', '', audio_by_tag.get(tag, ""))
                if not sid:
                    continue
                if any(s1 < re_ and s2 > rs for rs, re_ in replaced_ranges):
                    continue
                分镜序列输出 = 分镜序列输出[:s1] + "%s (S%s)" % (tag, sid) + 分镜序列输出[s2:]
                replaced_ranges.append((s1, s2))

            # 声音引用行 + 音色索引：按发声顺序取 <Picture N> 对应的角色索引（0 基）
            indices = self._parse_voice_indices(角色索引)
            values = []
            for tag in speaking_tags:
                sid = re.sub(r'\D', '', audio_by_tag.get(tag, ""))
                if not sid:
                    continue
                subj_no = self._subject_no(tag)
                voice_lines.append("<Audio %s> is the voice-timbre reference for <Picture %d> (S%s)."
                                   % (sid, subj_no, sid))
                if indices and 1 <= subj_no <= len(indices):
                    values.append(str(indices[subj_no - 1]))
            if values:
                音色索引 = ",".join(values)

        # 名称替换（detailed_description 与 summary 里的「情节」都做同样替换）
        分镜序列输出 = self._replace_names(分镜序列输出, pairs, 台词开关)
        情节原文 = self._replace_names(情节原文, pairs, 台词开关)

        # 定义段：summary 取本镜「情节」原文（固定片种前缀）；不再生成保留分析行
        英文 = "英文" in self._list_to_lines(data.get("生成语言", ""))
        summary = self._build_summary(matched_type, 情节原文) if 情节原文 else ""
        definitions = self._build_definitions(条目列表, voice_lines, summary)

        # detailed_description：风格头（整体风格 + 画面质感 + 场景 + 在场位置 + 运镜方式）两行，
        # 空一行后接时间段正文；在场位置/运镜方式沿用「时间段」的替换规矩——先按正文规矩清洗，
        # 再做角色/道具/场景名称 → <Picture N>（台词保护同样生效）
        在场位置头 = self._replace_names(self._clean_header_segment(在场位置), pairs, 台词开关)
        运镜方式头 = self._replace_names(self._clean_header_segment(运镜方式), pairs, 台词开关)
        风格头 = self._build_style_header(整体风格, 画面质感, 场景名称, 在场位置头, 运镜方式头, 英文)
        段落 = []
        if definitions:
            段落.append(definitions.rstrip("\n"))
        if 分镜序列输出.strip():
            body = 分镜序列输出.strip()
            段落.append("detailed_description:\n" + ("%s\n\n%s" % (风格头, body) if 风格头 else body))
        段落.append("overall_soundscape:\n" + (self._clean_sound(环境音) or "N/A"))
        if BGM开关:
            段落.append("non_diegetic_music:\n" + (self._clean_sound(BGM) or "N/A"))
        else:
            段落.append("non_diegetic_music:\nN/A")
        分镜序列输出 = "\n\n".join(段落)

        # 两种模式都返回 8 项（= 定义端口数），顺序与定义一致：
        # 两种模式都返回 8 项（= 定义端口数），顺序与定义一致：
        #   分镜模式 = 分镜类型/分镜序列/角色索引/音色索引/道具索引/场景索引/索引时长/场景衔接
        #   档案模式 = 档案编码/作品名/整体风格/档案/空/空/0.0/False（前端把前四个端口改名为档案名）
        # 音色档案只用于配音、与画面风格无关，此时整体风格输出空文本。
        作品名输出 = self._list_to_lines(data.get("作品名", ""))
        if 模式 == "档案模式":
            return {"result": [档案编码, 作品名输出, "" if 档案选择 == "音色档案" else 整体风格, 档案,
                               "", "", 0.0, False]}
        return {"result": [分镜类型, 分镜序列输出, 角色索引, 音色索引, 道具索引, 场景索引, 索引时长, 场景判断]}

# ==== JSON提取开关（JSON提取 的子节点）：七个选项开关聚合为单个「开关配置」端口输出 ====

class YUAN_TXTJsonSwitch:
    # 输出接「JSON提取」的「开关配置」可选端口；未接入时 JSON提取 按默认值执行
    OUTPUT_NAMES = ("开关配置",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "角色开关": ("BOOLEAN", {
                    "default": True,
                    "label_on": "输出",
                    "label_off": "不输出",
                    "display_name": "角色输出",
                    "tooltip": "开=角色名称替换为 <Picture N> 并输出「角色索引」；关=不替换、索引为空。"
                }),
                "音色开关": ("BOOLEAN", {
                    "default": True,
                    "label_on": "输出",
                    "label_off": "不输出",
                    "display_name": "音色输出",
                    "tooltip": "开=说话者替换为 <Picture N><Audio M>（最多3个，按说话顺序编号）并输出「音色索引」；关=仅普通替换 <Picture N>、音色索引为空。",
                }),
                "道具开关": ("BOOLEAN", {
                    "default": True,
                    "label_on": "输出",
                    "label_off": "不输出",
                    "display_name": "道具输出",
                    "tooltip": "开=道具名称替换为 <Picture N> 并输出「道具索引」；关=不替换、索引为空。"
                }),
                "场景开关": ("BOOLEAN", {
                    "default": True,
                    "label_on": "输出",
                    "label_off": "不输出",
                    "display_name": "场景输出",
                    "tooltip": "开=场景名称替换为 <Picture N> 并输出「场景索引」；关=不替换、索引为空。"
                }),
                "BGM开关": ("BOOLEAN", {
                    "default": True,
                    "label_on": "输出",
                    "label_off": "N/A",
                    "display_name": "BGM输出",
                    "tooltip": "开=分镜序列正常输出 BGM；关=输出 N/A。"
                }),
                "台词开关": ("BOOLEAN", {
                    "default": True,
                    "label_on": "保护台词",
                    "label_off": "正常替换",
                    "display_name": "台词保护",
                    "tooltip": "开=引号对与 <d>...</d> 内的名称不替换、原文保留；关=整段正常替换。",
                }),
                "情节衔接开关": ("BOOLEAN", {
                    "default": False,
                    "label_on": "新增行",
                    "label_off": "不新增",
                    "display_name": "情节衔接",
                    "tooltip": "开=分镜序列定义段在场景行之后新增一行「<Picture N>：情节衔接文本」（编号顺延），场景描述行原样输出；关=不新增。"
                }),
            },
        }

    RETURN_TYPES = (AnyType("*"),)
    RETURN_NAMES = OUTPUT_NAMES
    FUNCTION = "get_switches"
    CATEGORY = "Yuan Tool/文本"

    def get_switches(self, 角色开关, 音色开关, 道具开关, 场景开关, BGM开关, 台词开关, 情节衔接开关):
        # 聚合为开关配置 dict，须包在元组中返回（裸 dict 会被 ComfyUI 当作 ui/result/expand 特殊返回，
        # 导致零输出、下游 IndexError）；键序与 YUAN_TXTJsonExtractor.SWITCH_KEYS 一致
        return ({
            "角色开关": 角色开关,
            "音色开关": 音色开关,
            "道具开关": 道具开关,
            "场景开关": 场景开关,
            "BGM开关": BGM开关,
            "台词开关": 台词开关,
            "情节衔接开关": 情节衔接开关,
        },)


# ==== 出场排序 ====

class YUAN_TXTAppearanceOrder:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "placeholder": "输入需要排序的文本...",
                    "tooltip": "输入要检查的原始文本。"
                }),
                "验证对象": ("STRING", {
                    "multiline": True,
                    "placeholder": "每行一个验证对象...",
                    "tooltip": "需要验证的对象列表，每行一个。"
                }),
                "分隔符": ("STRING", {
                    "multiline": False,
                    "default": ",",
                    "placeholder": "排序输出的分隔符...",
                    "tooltip": "输出排序结果时使用的分隔符。"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "appearance_order"
    CATEGORY = "Yuan Tool/文本"
    OUTPUT_NODE = True

    def appearance_order(self, text, 验证对象, 分隔符):
        # 按行解析验证对象，去空白、去空行、去重（保持首次出现顺序）
        targets = []
        seen = set()
        for line in 验证对象.split("\n"):
            name = line.strip()
            if name and name not in seen:
                targets.append(name)
                seen.add(name)

        if not targets or not text:
            return ("",)

        # 记录每个对象在文本中第一次出现的位置；未出现则跳过
        found = []  # (first_pos, name)
        for name in targets:
            pos = text.find(name)
            if pos >= 0:
                found.append((pos, name))

        # 按首次出现位置升序排序，输出对象名称（不重复）
        found.sort(key=lambda x: x[0])
        result = 分隔符.join(name for _, name in found)

        return (result,)


# ==== 格式转换 ====

class YUAN_TXTConvertAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "*": (AnyType("*"), {
                    "tooltip": "接受任何类型的输入。"
                }),
                "格式类型": (["string", "int", "float", "boolean"], {
                    "default": "string",
                    "tooltip": "选择要将输入转换成的目标类型。"
                }),
            }
        }

    RETURN_TYPES = (AnyType("*"),)
    RETURN_NAMES = ("输出",)
    FUNCTION = "convert_any"
    CATEGORY = "Yuan Tool/文本"
    OUTPUT_NODE = True

    def convert_any(self, **kwargs):
        anything = kwargs['*']
        output_type = kwargs['格式类型']
        if output_type == 'string':
            result = str(anything)
        elif output_type == 'int':
            result = int(anything)
        elif output_type == 'float':
            result = float(anything)
        elif output_type == 'boolean':
            result = bool(anything)
        else:
            result = anything
        return (result,)


# ==== 列表编号 ====

class YUAN_TXTListNumber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本": ("STRING", {
                    "multiline": True,
                    "placeholder": "输入需要编号的文本，每行一组...",
                    "tooltip": "输入待编号的文本列表。每行作为一组，从第一组到最后一组依次编号。"
                }),
                "起始编号": ("INT", {
                    "default": 1,
                    "min": 0,
                    "step": 1,
                    "tooltip": "编号起始值，从该数字开始递增编号。"
                }),
                "编号前缀": ("STRING", {
                    "default": "",
                    "placeholder": "编号前添加的文本，如\"第\"",
                    "tooltip": "每个编号前添加的自定义文本前缀。"
                }),
                "编号后缀": ("STRING", {
                    "default": "",
                    "placeholder": "编号后添加的文本，如\"项\"",
                    "tooltip": "每个编号后添加的自定义文本后缀。"
                }),
                "输出模式": (["列表", "合并文本"], {
                    "default": "列表",
                    "tooltip": "列表=逐条输出编号文本列表；合并文本=合并为单个字符串。"
                }),
                "合并间隔符": ("STRING", {
                    "default": "\\n",
                    "placeholder": "合并文本的分隔符，如\\n",
                    "tooltip": "仅在输出模式为\"合并文本\"时生效，用于分隔各条带编号的文本。"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("输出", "接续编号")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "number_list"
    CATEGORY = "Yuan Tool/文本"

    def number_list(self, 文本, 起始编号, 编号前缀, 编号后缀, 输出模式, 合并间隔符):
        if not 文本 or not 文本.strip():
            # OUTPUT_IS_LIST=True 必须返回长度 ≥1 的列表，否则空列表会中断下游执行；统一返回 [""] 保链路不断
            return ([""], 起始编号)

        lines = [line for line in 文本.split('\n') if line.strip()]
        count = len(lines)
        next_num = 起始编号 + count

        results = []
        for i, line in enumerate(lines):
            num = 起始编号 + i
            numbered = f"{编号前缀}{num}{编号后缀}{line}"
            results.append(numbered)

        if 输出模式 == "合并文本":
            separator = 合并间隔符.replace("\\n", "\n")
            merged = separator.join(results)
            return ([merged], next_num)

        return (results, next_num)


# ==== 文本批量替换 ====

class YUAN_TXTReplace:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "placeholder": "输入需要替换的文本...",
                    "tooltip": "输入要进行替换操作的原始文本。"
                }),
                "查找文本": ("STRING", {
                    "multiline": True,
                    "placeholder": "每行一个要查找的文本...",
                    "tooltip": "查找文本，每行一条，与「替换文本」按行逐行对应。"
                }),
                "替换文本": ("STRING", {
                    "multiline": True,
                    "placeholder": "每行一个要替换的文本...",
                    "tooltip": "替换文本，每行一条，与「查找文本」按行逐行对应。"
                }),
                "台词开关": ("BOOLEAN", {
                    "default": False,
                    "label_on": "保护台词",
                    "label_off": "正常替换",
                    "display_name": "台词保护",
                    "tooltip": "开=引号（\"…\"、'…'、“…”、'…'、「…」）内的内容不替换；关=整段正常批量替换。",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "replace_text"
    CATEGORY = "Yuan Tool/文本"
    OUTPUT_NODE = True

    @staticmethod
    def _replace_with_protect(text, find_str, replace_str, mask):
        """仅在 mask[i]==False 的位置允许替换 find_str -> replace_str；find_str 任一字符被保护则整段跳过。"""
        if not find_str:
            return text
        m = len(find_str)
        n = len(text)
        if m == 0 or m > n:
            return text
        out = []
        i = 0
        while i <= n - m:
            # 先快速判断窗口内是否存在任何被保护字符；无则再做字符串全等比较（避免含中文大窗口时重复切片）
            window_protected = False
            for k in range(m):
                if mask[i + k]:
                    window_protected = True
                    break
            if not window_protected and text[i:i + m] == find_str:
                out.append(replace_str)
                i += m
                continue
            out.append(text[i])
            i += 1
        # 末尾剩余字符
        while i < n:
            out.append(text[i])
            i += 1
        return "".join(out)

    def replace_text(self, text, 查找文本, 替换文本, 台词开关):
        # splitlines 同时兼容 LF/CRLF，避免 \r 残留在查找串里导致匹配失败
        find_lines = 查找文本.splitlines()
        replace_lines = 替换文本.splitlines()

        # 台词开关：开启时逐轮以当前文本重建保护掩码进行保护替换
        protect = bool(台词开关)

        # 配对查找/替换，过滤空查找串
        pairs = []
        count = min(len(find_lines), len(replace_lines))
        for i in range(count):
            find_str = find_lines[i]
            replace_str = replace_lines[i]
            if find_str:
                pairs.append((find_str, replace_str))

        # 按查找文本长度降序排序（最长匹配优先，避免短名误替换长名中的子串）
        pairs.sort(key=lambda x: -len(x[0]))

        result = text
        for find_str, replace_str in pairs:
            if protect:
                result = YUAN_TXTReplace._replace_with_protect(result, find_str, replace_str,
                                                                _build_quote_protect_mask(result))
            else:
                result = result.replace(find_str, replace_str)

        return (result,)


# ==== 文本处理（分段） ====

class YUAN_TXTParagraphSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "placeholder": "输入需要分割的文本...",
                    "tooltip": "待分割文本。接入 any_x 端口时，此文本为第1部分，其余端口内容按顺序拼接其后。"
                }),
                "输出模式": ("BOOLEAN", {
                    "default": False,
                    "label_on": "输出分段列表",
                    "label_off": "输出原始文本",
                    "tooltip": "输出原始文本=按全部分段规则合并为一段；输出分段列表=按分段方式分割后以列表输出。"
                }),
                "段落优化": ("BOOLEAN", {
                    "default": True,
                    "label_on": "去除首尾空格",
                    "label_off": "保留原始空格",
                    "tooltip": "去除首尾空格=清理首尾空白与换行；保留原始空格=完全保留原始格式与缩进。"
                }),
                "分段方式": (["端口", "空行", "序号", "段落", "标题", "数字", "地址", "手动"], {
                    "default": "空行",
                    "tooltip": "分割方式：端口/空行(双换行)/序号/段落(每行一段)/标题/数字/地址(Windows路径)/手动(|||)。"
                }),
                "输出段落": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "display": "number",
                    "tooltip": "右侧「段落x」输出端口数量，改后需点击「更新端口」生效。"
                }),
                "输入端口": ("INT", {
                    "default": 1,
                    "min": 1,
                    "step": 1,
                    "display": "number",
                    "tooltip": "左侧「any_x」输入端口数量（多文本源按顺序拼合分段），改后需点击「更新端口」生效。"}),
                "选取段落": ("STRING", {
                    "default": "-1",
                    "placeholder": "输入要选取的段落，用逗号分隔，如0,2,4；填 -1 输出所有；留空总段输出为空",
                    "tooltip": "选取输出哪些段落：逗号分隔的 0 基索引（如 0,2,4），填 -1 输出全部，留空不选（总段为空）。影响「总段」与「段落x」。",
                }),
            },
            "optional": {
                **{f"any_{i}": (AnyType("*"),) for i in range(1, 65)}
            }
        }

    MAX_OUTPUTS = 100
    RETURN_TYPES = ("INT", "STRING") + ("STRING",) * MAX_OUTPUTS
    RETURN_NAMES = ("数:", "总段:") + tuple(f"段落{i + 1}" for i in range(MAX_OUTPUTS))
    OUTPUT_IS_LIST = (False, True) + (False,) * MAX_OUTPUTS
    FUNCTION = "split_paragraphs"
    CATEGORY = "Yuan Tool/文本"
    OUTPUT_NODE = True

    def is_title_content(self, processed_line, 段落优化):
        line_stripped = processed_line.strip() if 段落优化 else processed_line
        if not line_stripped: return False
        if len(line_stripped) > 20: return False
        last_char = line_stripped[-1] if line_stripped else ''
        forbidden_punctuation = (
            ',', '，', '.', '。', '!', '！', '?', '？', ';', '；',
            '"', "'", '（', '）', '、', '…', '—')
        if last_char in forbidden_punctuation: return False
        bracket_patterns = [r'^【.+】$', r'^《.+》$', r'^<.+>$']
        for pattern in bracket_patterns:
            if re.match(pattern, line_stripped): return True
        num_title_pattern = r'^(?:[一二三四五六七八九十百千万]+、|\d+\. |[a-zA-Z]+\. )'
        if re.match(num_title_pattern, line_stripped): return True
        if last_char in (':', '：'): return len(line_stripped) > 1
        if not re.search(r'[^\u4e00-\u9fa5a-zA-Z0-9]', last_char): return True
        return False

    def _convert_to_str(self, val):
        """把任意 ComfyUI 输入统一转成纯字符串：None→""、容器逐元素换行拼接、bytes 先 utf-8 再 latin-1 解码、其余直接 str()（异常返回 ""）。"""
        if val is None:
            return ""
        if isinstance(val, bool):
            # bool 是 int 的子类，需要先判断
            return str(val)
        if isinstance(val, (int, float, str)):
            return str(val)
        if isinstance(val, bytes):
            try:
                return val.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    return val.decode("latin-1")
                except Exception:
                    return ""
        if isinstance(val, (list, tuple, set, frozenset)):
            parts = []
            for x in val:
                if x is None:
                    continue
                try:
                    s = str(x)
                except Exception:
                    continue
                if s:
                    parts.append(s)
            return "\n".join(parts)
        if isinstance(val, dict):
            try:
                return str(val)
            except Exception:
                return ""
        try:
            return str(val)
        except Exception:
            return ""

    def split_paragraphs(self, text, 分段方式, 段落优化, 输出模式, 输出段落, 选取段落, 输入端口,
                         **kwargs):
        input_count = 输入端口
        collected_texts = []
        for i in range(1, input_count + 1):
            key = f"any_{i}"
            val = kwargs.get(key, None)
            if val is not None:
                val_str = self._convert_to_str(val)
                if val_str.strip():
                    collected_texts.append(val_str)
        if collected_texts:
            if input_count >= 2:
                text = "\n\n\n".join(collected_texts)
            else:
                text = collected_texts[0]
        if not text:
            return (0, [""],) + ("",) * self.MAX_OUTPUTS

        if 分段方式 == "端口":
            # 段落优化开启：端口文本去掉空行（只保留有内容的行）；关闭：连空行一起原样保留，
            # 与输入一一对应（某端口里的空行留在该段自己的输出里，不与其他段合并）
            def port_text(val):
                s = self._convert_to_str(val)
                if not 段落优化:
                    return s
                return "\n".join(line.strip() for line in s.splitlines() if line.strip())

            if collected_texts:
                # 端口模式下每个 any_x 端口对应一个段落位置：即使端口为空也保留占位，
                # 保证选取段落索引与端口序号一一对应（如 any2 未接入时索引1应输出空文本）
                paras = []
                for i in range(1, input_count + 1):
                    val = kwargs.get(f"any_{i}", None)
                    paras.append("" if val is None else port_text(val))
            else:
                paras = [port_text(text)] if text else []
        elif 分段方式 == "空行":
            lines, paras, curr_para = text.split('\n'), [], []
            for line in lines:
                pl = line.strip() if 段落优化 else line
                if not pl:
                    if curr_para:
                        paras.append(' '.join(curr_para) if 段落优化 else '\n'.join(curr_para))
                        curr_para = []
                else:
                    curr_para.append(pl)
            if curr_para: paras.append(' '.join(curr_para) if 段落优化 else '\n'.join(curr_para))
        elif 分段方式 == "序号":
            lines = text.split('\n')
            paras, current_para = [], []
            p_standalone = r'(?:【\d+】|\*?[\u2460-\u24FF]|\*?[\u3200-\u32FF]|[•▪*])'
            p_counters = r'(?:\d+|[IVXLCDMivxlcdm]+|[A-Za-z]|[一二三四五六七八九十百千万]+|[壹贰叁肆伍陆柒捌玖拾]+)'
            p_seps = r'(?:[,，、.·:：\-\*•▪])'
            pattern = r'^\s*(?:' + p_standalone + r'|' + p_counters + p_seps + r')'
            for line in lines:
                processed_line = line.strip() if 段落优化 else line
                if re.match(pattern, processed_line):
                    if current_para:
                        paras.append(' '.join(current_para) if 段落优化 else '\n'.join(current_para))
                        current_para = []
                    current_para.append(processed_line)
                else:
                    if current_para or processed_line.strip(): current_para.append(processed_line)
            if current_para: paras.append(' '.join(current_para) if 段落优化 else '\n'.join(current_para))
        elif 分段方式 == "段落":
            # 每行一段：关闭优化时连空行一起保留（与输入逐行对应），开启时由下方统一去空行
            paras = [line.strip() if 段落优化 else line for line in text.split('\n')]
        elif 分段方式 == "标题":
            lines = text.split('\n')
            paras = []
            current_para = []
            line_info = []
            for line in lines:
                processed = line.strip() if 段落优化 else line
                is_blank = not processed.strip() if 段落优化 else not processed
                is_title = self.is_title_content(processed, 段落优化) and not is_blank
                line_info.append({'content': processed, 'is_blank': is_blank, 'is_title': is_title})
            n = len(line_info)
            i = 0
            while i < n and not line_info[i]['is_title'] and not line_info[i]['is_blank']:
                current_para.append(line_info[i]['content'])
                i += 1
            while i < n:
                while i < n and line_info[i]['is_blank']: i += 1
                if i >= n: break
                if line_info[i]['is_title']:
                    if current_para:
                        paras.append(' '.join(current_para) if 段落优化 else '\n'.join(current_para))
                        current_para = []
                    title_block = []
                    while i < n:
                        curr_info = line_info[i]
                        if curr_info['is_blank']:
                            i += 1
                            continue
                        if curr_info['is_title']:
                            title_block.append(curr_info['content'])
                            i += 1
                        else:
                            break
                    current_para.extend(title_block)
                    while i < n and not line_info[i]['is_title']:
                        if not line_info[i]['is_blank']: current_para.append(line_info[i]['content'])
                        i += 1
                else:
                    current_para.append(line_info[i]['content'])
                    i += 1
            if current_para: paras.append(' '.join(current_para) if 段落优化 else '\n'.join(current_para))
        elif 分段方式 == "数字":
            pattern = r'[ \t]*\d+(?:\.\d+)?[ \t]*'
            matches = re.findall(pattern, text)
            paras = []
            for m in matches:
                pl = m.strip() if 段落优化 else m
                if pl:
                    paras.append(pl)
        elif 分段方式 == "地址":
            pro_text = text.replace('\\\\', '\\')
            pattern = r'([a-zA-Z]:[\\/][^\"\'<>,;\[\]\n\r]+)'
            matches = re.findall(pattern, pro_text)
            paras = []
            for m in matches:
                clean_path = m.strip()
                if " object" in clean_path:
                    clean_path = clean_path.split(" object")[0].strip()
                clean_path = clean_path.rstrip('.')
                if clean_path and len(clean_path) > 3:
                    paras.append(clean_path)
        elif 分段方式 == "手动":
            raw_paras = text.split('|||')
            paras = []
            for m in raw_paras:
                pl = m.strip() if 段落优化 else m
                if pl:
                    paras.append(pl)

        sel = 选取段落.strip() if 选取段落 is not None else ""
        selected_indices = []  # 被选取段落的原始索引（段落x端口按原始索引一一对应）
        if sel == "-1":
            # -1：输出所有段落（默认行为）
            to = paras.copy()
            selected_indices = list(range(len(paras)))
        elif sel == "":
            # 留空：总段输出为空（不选取任何段落）
            to = []
        else:
            # 数字索引组合：按 0/1/2... 索引选取，支持 。,，./\ 等分隔
            to = []
            si = re.split(r'[。,，./\\]', sel)
            for i in si:
                try:
                    idx = int(i.strip())
                    if 0 <= idx < len(paras):
                        to.append(paras[idx])
                        selected_indices.append(idx)
                except (TypeError, ValueError):
                    continue

        # 段落优化是后处理：先按端口/行序号完成选取，再把空段落从结果里去掉。
        # 这样开启与关闭优化的选取顺序与内容一致（空段落/空端口只影响是否输出，不改变索引）
        if 段落优化:
            to = [p for p in to if p and p.strip()]

        if not 输出模式:
            to = ["\n".join(to)] if to else [""]

        cnt = len(to)
        # 「总段」是 OUTPUT_IS_LIST 端口：空列表会让下游直接不执行，用单个空串占位
        if 输出模式 and not to:
            to = [""]

        max_out = self.MAX_OUTPUTS
        po = [""] * max_out
        if 输出模式:
            # 分段列表模式：段落x端口按原始段落索引一一对应（选取段落=0→段落1、=1→段落2...，支持索引多端口）
            for idx in selected_indices:
                if idx < 输出段落:
                    po[idx] = paras[idx]
        else:
            # 原始文本模式：保持原行为，总段文本落到段落1端口
            for i in range(min(max_out, len(to), 输出段落)):
                po[i] = to[i]

        return (cnt, to,) + tuple(po)


# ==== 长度 ====

class YUAN_TXTLength:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (AnyType("*"), {
                    "tooltip": "要计算长度的文本。支持直接输入文本或从其他节点接入任意类型值；非字符串会自动转为字符串。"
                }),
                "长度模式": (["字符串", "段落", "空行", "列表"], {
                    "default": "字符串",
                    "tooltip":
                        "长度计数方式：字符串=字符总数；段落=行数；空行=按空行分段数；列表=列表元素数（非列表视为1）。空文本均输出 0。",
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("长度",)
    OUTPUT_TOOLTIPS = ("根据所选长度模式统计得到的数量。",)
    FUNCTION = "count_length"
    CATEGORY = "Yuan Tool/文本"
    DESCRIPTION = (
        "统计接入文本的长度：字符串长度 / 段落（行数） / 空行分段段数 / 列表元素个数。"
        "空文本（无内容）在任意模式下均输出 0。"
    )

    @staticmethod
    def _to_plain_text(val):
        """把任意 ComfyUI 输入统一转成纯字符串（list 用换行拼接，其他走 str）。"""
        if val is None:
            return ""
        if isinstance(val, list):
            return "\n".join("" if v is None else str(v) for v in val)
        return str(val)

    def count_length(self, text, 长度模式):
        # 列表模式：优先判断上游真实传入的是否就是 list（例如 文本处理-输出分段列表），不做 str(list) 干扰
        if 长度模式 == "列表":
            if isinstance(text, list):
                return (len(text),)
            # 上游不是列表（普通字符串/其他）：当做"一个整体"处理。空文本输出 0
            plain = self._to_plain_text(text)
            return (0 if plain == "" else 1,)

        # 其余三种模式：都基于纯文本内容
        plain = self._to_plain_text(text)

        if 长度模式 == "字符串":
            return (len(plain),)

        if 长度模式 == "段落":
            # 每一行（包括空行）算一个段落；空文本（无内容）输出 0
            if plain == "":
                return (0,)
            return (len(plain.splitlines()),)

        if 长度模式 == "空行":
            # 去掉首尾空白后，按空行（空白行）切分文本，统计非空片段数
            stripped = plain.strip()
            if stripped == "":
                return (0,)
            # 拆分为行，累计非空段；遇到连续空段则产生分隔
            blocks = []
            curr = []
            for line in stripped.splitlines():
                if line.strip() == "":
                    if curr:
                        blocks.append("\n".join(curr))
                        curr = []
                else:
                    curr.append(line)
            if curr:
                blocks.append("\n".join(curr))
            return (len(blocks),)

        # 兜底（未知模式按字符串）
        return (len(plain),)


class YUAN_TXTShotDurations:
    """分段时间提取：从分镜策划 JSON 的「分镜情节」中按编号顺序提取各分镜时长。"""

    @staticmethod
    def _fmt_seconds(v):
        """时长格式化：整数值去小数点（10 而非 10.0），小数保留原值。"""
        if v is None:
            return "0"
        if float(v).is_integer():
            return str(int(v))
        return str(float(v))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json": (AnyType("*"), {
                    "forceInput": True,
                    "tooltip": "分镜策划 JSON（字符串或对象，支持多对象拼接）。"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("时长列表", "时长序列", "分镜数量", "总时长")
    OUTPUT_TOOLTIPS = (
        "按编号 1-N 顺序提取的各分镜时长，逗号分隔（如 10,9,7,5,9,7），可直接接入「Yuan 加载音频」的分段时长。",
        "各分镜时长逐个输出（列表端口，按编号顺序）。",
        "分镜情节中的分镜条目数量。",
        "所有分镜时长之和（秒）。",
    )
    OUTPUT_IS_LIST = (False, True, False, False)
    FUNCTION = "extract_durations"
    CATEGORY = "Yuan Tool/文本"
    DESCRIPTION = (
        "从分镜策划 JSON 的「分镜情节」中按编号顺序（1-N）提取各分镜时长："
        "时长取自每个分镜「类型」字段中的第一个数字（如 MV：10秒 → 10、文戏：8s → 8），单位 s/秒/无单位均可。"
        "时长列表端口输出逗号分隔字符串（可直接接入 Yuan 加载音频的分段时长）；"
        "时长序列端口按列表逐个输出；无有效分镜时输出空列表文本与 0。"
    )

    def extract_durations(self, json=None):
        data = json
        if isinstance(data, str):
            data = YUAN_TXTJsonExtractor._parse_json_text(data)
        if not isinstance(data, dict):
            data = {}

        分镜情节数据 = data.get("分镜情节", [])
        # 按编号排序（编号缺失/非法的排最后，保持出现顺序）
        def _num(item):
            try:
                return int(item.get("编号"))
            except Exception:
                return float("inf")
        entries = [it for it in 分镜情节数据 if isinstance(it, dict)] if isinstance(分镜情节数据, list) else []
        entries.sort(key=_num)

        durations = []
        for item in entries:
            dur = YUAN_TXTJsonExtractor._type_duration_seconds(item.get("类型", ""))
            # 类型字段无数字时按 0 计，保持与分镜数量对齐
            durations.append(dur if dur is not None else 0.0)

        时长列表 = ",".join(self._fmt_seconds(d) for d in durations)
        总时长 = float(sum(durations))
        # OUTPUT_IS_LIST=True 必须返回长度 ≥1 的列表，否则空列表会中断下游执行；空时兜底 [0.0]
        时长序列 = durations if durations else [0.0]
        return (时长列表, 时长序列, len(entries), 总时长)


# ==== 预览内容 ====

class YUAN_TXTPreviewContent:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本": ("STRING", {"multiline": True, "default": ""}),
                "显示模式": ("BOOLEAN", {"default": True, "label_on": "预览", "label_off": "编辑"}),
            },
            "optional": {
                "输出内容": (AnyType("*"), {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("内容",)
    OUTPUT_TOOLTIPS = ("预览模式下为序列化后的文本，编辑模式下为输入文本。",)
    FUNCTION = "main"
    OUTPUT_NODE = True

    CATEGORY = "Yuan Tool/文本"
    SEARCH_ALIASES = ["预览", "显示", "编辑", "inspect", "debug", "show text"]
    DESCRIPTION = "预览/编辑任意内容。显示模式=预览：只读展示上游内容；显示模式=编辑：以文本内容输出字符串。"

    def main(self, 文本, 显示模式, 输出内容=None):
        if 显示模式 and 输出内容 is not None:
            value = self._serialize(输出内容)
        else:
            value = 文本

        return {"ui": {"text": (value,)}, "result": (value,)}

    def _serialize(self, source):
        # 限制张量 str() 回退时的输出长度
        torch.set_printoptions(edgeitems=6)
        if isinstance(source, str):
            return source
        if isinstance(source, (int, float, bool)):
            return str(source)
        try:
            return _json.dumps(source, indent=4, ensure_ascii=False)
        except Exception:
            return str(source)


# ==== 图像文本标签 ====

DIRECTION_UP = "上方"
DIRECTION_DOWN = "下方"
DIRECTION_LEFT = "左侧"
DIRECTION_RIGHT = "右侧"
DIRECTION_OVERLAY = "覆盖"


def _parse_color(color_string):
    """把取色器输出的颜色字符串解析为 RGB 或 RGBA 整数元组。"""
    try:
        return ImageColor.getrgb(str(color_string).strip())
    except ValueError:
        return (255, 255, 255)


class YUAN_TXTLabelColor:

    @classmethod
    def INPUT_TYPES(cls):
        fonts = folder_paths.get_filename_list(FONTS_FOLDER)
        # 只列出本节点 fonts 文件夹里的字体；目录为空时回退到内置字体
        font_choices = fonts if fonts else [DEFAULT_FONT]
        return {
            "required": {
                "image": ("IMAGE", {
                    "display_name": "图像",
                    "tooltip": "要添加标签的图像，支持批量。"
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Text",
                    "display_name": "文字",
                    "tooltip": "要绘制的文字，支持多行与自动换行。"
                }),
                "text_x": ("INT", {
                    "default": 10, "min": 0, "max": 4096, "step": 1,
                    "display_name": "文字横向偏移",
                    "tooltip": "文字距标签边缘的横向像素偏移。"
                }),
                "text_y": ("INT", {
                    "default": 2, "min": 0, "max": 4096, "step": 1,
                    "display_name": "文字纵向偏移",
                    "tooltip": "文字距标签顶部的纵向像素偏移。"
                }),
                "height": ("INT", {
                    "default": 48, "min": -1, "max": 4096, "step": 1,
                    "display_name": "标签高度",
                    "tooltip": "标签高度（像素）；-1 为按文字行数自动计算。"
                }),
                "font_size": ("INT", {
                    "default": 32, "min": 0, "max": 4096, "step": 1,
                    "display_name": "字号",
                    "tooltip": "文字字号（像素）。"
                }),
                "font_color": ("COLOR", {
                    "default": "#ffffff",
                    "display_name": "文字颜色",
                    "tooltip": "文字颜色（取色器选择）。"
                }),
                "label_color": ("COLOR", {
                    "default": "#000000",
                    "display_name": "标签颜色",
                    "tooltip": "标签底色；方向为“覆盖”时不生效。"
                }),
                "font": (font_choices, {
                    "default": font_choices[0],
                    "display_name": "字体",
                    "tooltip": "标签字体，仅列出本节点 fonts 文件夹中的字体。"
                }),
                "direction": ([DIRECTION_UP, DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_OVERLAY], {
                    "default": DIRECTION_UP,
                    "display_name": "方向",
                    "tooltip": "标签拼接位置；“覆盖”为直接叠加在原图上。"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    OUTPUT_TOOLTIPS = ("添加标签后的图像。",)
    FUNCTION = "add_label"
    CATEGORY = "Yuan Tool/文本"
    SEARCH_ALIASES = [
        "addlabel", "add label", "label", "text", "caption",
        "subtitle", "watermark", "标签", "文字标签", "图像文本标签", "添加标签", "取色器",
    ]
    DESCRIPTION = (
        "把文字绘制成标签，拼接在图像的上/下/左/右；“覆盖”为直接叠加在原图上。\n"
        "文字颜色与标签颜色用取色器选择，字体取自本节点 fonts 文件夹。"
    )

    def add_label(self, image, text, text_x, text_y, height, font_size, font_color,
                  label_color, font, direction):
        width = image.shape[2]
        channels = image.shape[3]
        # 与输入保持一致，alpha 图像仍保持 4 通道
        pil_mode = "RGBA" if channels == 4 else "RGB"

        font_path = None if font == DEFAULT_FONT else folder_paths.get_full_path(FONTS_FOLDER, font)

        # 解析取色器传入的颜色
        font_color_rgb = _parse_color(font_color)
        label_color_rgb = _parse_color(label_color)

        font_color_tuple = tuple(font_color_rgb[:3])
        label_color_tuple = tuple(label_color_rgb[:3])
        if pil_mode == "RGBA":
            font_color_tuple += (font_color_rgb[3] if len(font_color_rgb) > 3 else 255,)
            label_color_tuple += (label_color_rgb[3] if len(label_color_rgb) > 3 else 255,)

        horizontal = direction in (DIRECTION_LEFT, DIRECTION_RIGHT)
        # 左/右侧的标签会旋转 90°，因此要沿图像高度方向排版
        strip_length = image.shape[1] if horizontal else width

        def load_font():
            size = max(1, font_size)
            if font_path is None:
                try:
                    return ImageFont.load_default(size=size)
                except TypeError:
                    # Pillow < 10.1 的 load_default 不接受 size
                    return ImageFont.load_default()
            return ImageFont.truetype(font_path, size)

        def measure(label_font, content):
            """测量文本宽度，按 Pillow 版本依次尝试 getlength / getbbox / getsize。"""
            try:
                return label_font.getlength(content)
            except Exception:
                try:
                    return label_font.getbbox(content)[2]
                except Exception:
                    return label_font.getsize(content)[0]

        def wrap_line(label_font, text_line, max_width):
            """按宽度折行：中文逐字断行，拉丁文按词断行；超宽单词单独占一行。"""
            if max_width <= 0:
                return [text_line]
            tokens = re.findall(r"[A-Za-z0-9_\-.']+|[\u3000-\u9fff\uff00-\uffef]|.", text_line)
            wrapped = []
            current = ""
            for token in tokens:
                if token.isspace():
                    if current and not current.endswith(" "):
                        current += " "
                    continue
                candidate = current + token if current else token
                if not current or measure(label_font, candidate) <= max_width:
                    current = candidate
                else:
                    wrapped.append(current.rstrip())
                    current = token
            if current.strip():
                wrapped.append(current.rstrip())
            return wrapped or [""]

        def process_image(input_image, caption_text):
            label_font = load_font()
            lines = []
            max_width = strip_length - 2 * max(0, text_x)
            for text_line in caption_text.split('\n'):
                if text_line.strip() == "":
                    # 保留空行，以便支持连续换行
                    lines.append("")
                    continue
                lines.extend(wrap_line(label_font, text_line, max_width))

            if direction == DIRECTION_OVERLAY:
                pil_image = Image.fromarray((input_image.cpu().numpy() * 255).astype(np.uint8))
            elif height == -1:
                # 自动计算所需高度
                margin = 8
                required_height = (text_y + len(lines) * font_size) + margin
                pil_image = Image.new(pil_mode, (strip_length, required_height), label_color_tuple)
            else:
                pil_image = Image.new(pil_mode, (strip_length, height), label_color_tuple)

            draw = ImageDraw.Draw(pil_image)

            y_offset = text_y
            for line in lines:
                try:
                    draw.text((text_x, y_offset), line, font=label_font, fill=font_color_tuple, features=['-liga'])
                except Exception:
                    draw.text((text_x, y_offset), line, font=label_font, fill=font_color_tuple)
                y_offset += font_size

            return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

        processed_images = [process_image(img, text) for img in image]
        processed_batch = torch.cat(processed_images, dim=0)

        # 根据方向拼接
        if direction == DIRECTION_DOWN:
            combined_images = torch.cat((image, processed_batch), dim=1)
        elif direction == DIRECTION_LEFT:
            # 标签沿图像高度排版后顺时针旋转 90°，拼接到图像左侧
            processed_batch = torch.rot90(processed_batch, 3, (1, 2))
            combined_images = torch.cat((processed_batch, image), dim=2)
        elif direction == DIRECTION_RIGHT:
            # 逆时针旋转 90°，拼接到图像右侧（首行文字贴近图像）
            processed_batch = torch.rot90(processed_batch, 1, (1, 2))
            combined_images = torch.cat((image, processed_batch), dim=2)
        elif direction == DIRECTION_UP:
            combined_images = torch.cat((processed_batch, image), dim=1)
        else:
            combined_images = processed_batch

        return (combined_images,)


NODE_CLASS_MAPPINGS = {
    "YUAN_TXTLabelColor": YUAN_TXTLabelColor,
    "YUAN_TXTJsonExtractor": YUAN_TXTJsonExtractor,
    "YUAN_TXTJsonSwitch": YUAN_TXTJsonSwitch,
    "YUAN_TXTAppearanceOrder": YUAN_TXTAppearanceOrder,
    "YUAN_TXTConvertAny": YUAN_TXTConvertAny,
    "YUAN_TXTListNumber": YUAN_TXTListNumber,
    "YUAN_TXTReplace": YUAN_TXTReplace,
    "YUAN_TXTParagraphSplitter": YUAN_TXTParagraphSplitter,
    "YUAN_TXTLength": YUAN_TXTLength,
    "YUAN_TXTShotDurations": YUAN_TXTShotDurations,
    "YUAN_TXTPreviewContent": YUAN_TXTPreviewContent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YUAN_TXTLabelColor": "图像文本标签",
    "YUAN_TXTJsonExtractor": "JSON提取",
    "YUAN_TXTJsonSwitch": "JSON提取开关",
    "YUAN_TXTAppearanceOrder": "出场排序",
    "YUAN_TXTConvertAny": "格式转换",
    "YUAN_TXTListNumber": "列表编号",
    "YUAN_TXTReplace": "文本批量替换",
    "YUAN_TXTParagraphSplitter": "文本处理",
    "YUAN_TXTLength": "长度",
    "YUAN_TXTShotDurations": "分段时间提取",
    "YUAN_TXTPreviewContent": "预览内容",
}
