import re
import json as _json


class AnyType(str):
    def __ne__(self, __value):
        return False


# ==== 台词保护：引号对常量与掩码构建（JSON提取 / 文本批量替换 / 分镜角色替换 共用） ====

# opening, closing, rank（秩：同秩才能配成一对，避免不同引号互配）
# 包含 ASCII 半角引号、中文弯引号（""/''）、中文直角引号「」/『』
QUOTE_PAIRS = [
    ('"',  '"',  1),
    ("'",  "'",  2),
    ("“",  "”",  3),
    ("‘",  "’",  4),
    ("「", "」", 5),
    ("『", "』", 6),
]


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


# <d>...</d> 台词标签（与引号保护叠加，JSON提取 / 分镜角色替换 共用）
DIALOGUE_TAG_RE = re.compile(r'<d>.*?</d>', re.DOTALL)


def _build_dialogue_protect_mask(text):
    """完整台词保护掩码：引号对内部 + <d>...</d> 标签内部（含标签自身）。"""
    mask = _build_quote_protect_mask(text)
    for m in DIALOGUE_TAG_RE.finditer(text):
        for k in range(m.start(), m.end()):
            mask[k] = True
    return mask


# ==== JSON提取：按端口名提取对应字段，全部输出字符串 ====

class YUAN_TXTJsonExtractor:
    # 输出端口名
    OUTPUT_NAMES = ("整体风格", "档案", "档案编码", "分镜序列", "角色道具场景", "角色索引", "道具索引", "场景索引", "索引时长", "场景上下文")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json": (AnyType("*"), {
                    "forceInput": True,
                    "tooltip": "JSON 数据输入，支持 JSON 字符串或对象。字符串可为多个 JSON 对象拼接（如分镜策划五档案与视频提示词对象合并），自动逐段解析合并提取。"
                }),
                "索引": ("INT", {
                    "default": 1,
                    "min": 1,
                    "step": 1,
                    "tooltip": "对应分镜序列/分镜情节中的「编号」值，选择该编号的分镜。分镜序列端口默认输出 detailed_description(整体风格) + [Shot N]编号的时间段 + overall_soundscape(环境音) + non_diegetic_music(BGM)，标签与内容之间仅换行不空行；BGM开关关闭时 non_diegetic_music 内容输出 N/A；情节开关开启时改为输出「分镜情节」中该编号分镜的「情节」内容；分镜「类型」含武戏时时间段不加 [Shot N] 编号、原行输出；角色道具场景端口优先取「分镜情节」条目的 出现角色/出现道具/出现场景 列表定位角色、道具、场景档案（未提供时在时间段内容中智能匹配角色和道具、按标题智能匹配场景；台词保护：引号对与 <d>...</d> 标签内出现的名称不匹配、不提取），按 角色→道具→场景 顺序输出对应档案的完整描述，整体以 retention_analysis: 开头换行输出，每条描述前加 <Picture N> 序号（从1连续编号）；角色索引/道具索引端口输出匹配到的角色/道具在对应档案中的0基序号（逗号分隔），场景索引端口输出匹配到的场景档案0基序号（未匹配为空），三者均受对应输出开关控制，关时输出空文本；索引时长端口锁定该分镜「类型」字段中的时长（如 武戏：12秒、文戏：8s、MV：15秒，文戏/武戏/MV 通用，单位 s 或无单位均可）。"
                }),
                "档案选择": (["角色档案", "音色档案", "道具档案", "场景档案"], {
                    "default": "角色档案",
                    "tooltip": "选择「档案」输出端口输出的档案类型：角色档案、音色档案、道具档案或场景档案。"
                }),
                "角色开关": ("BOOLEAN", {
                    "default": True,
                    "label_on": "输出",
                    "label_off": "不输出",
                    "display_name": "角色输出",
                    "tooltip": "同时控制「角色道具场景」端口中的角色描述与「角色索引」端口：开=输出，关=不输出（索引输出空文本）。"
                }),
                "道具开关": ("BOOLEAN", {
                    "default": True,
                    "label_on": "输出",
                    "label_off": "不输出",
                    "display_name": "道具输出",
                    "tooltip": "同时控制「角色道具场景」端口中的道具描述与「道具索引」端口：开=输出，关=不输出（索引输出空文本）。"
                }),
                "场景开关": ("BOOLEAN", {
                    "default": True,
                    "label_on": "输出",
                    "label_off": "不输出",
                    "display_name": "场景输出",
                    "tooltip": "同时控制「角色道具场景」端口中的场景描述与「场景索引」端口：开=输出，关=不输出（索引输出空文本）。"
                }),
                "BGM开关": ("BOOLEAN", {
                    "default": True,
                    "label_on": "输出",
                    "label_off": "N/A",
                    "display_name": "BGM输出",
                    "tooltip": "控制「分镜序列」端口中 non_diegetic_music(BGM) 内容：开=正常输出 BGM，关=输出 N/A。"
                }),
                "情节开关": ("BOOLEAN", {
                    "default": False,
                    "label_on": "输出",
                    "label_off": "整合格式",
                    "display_name": "情节输出",
                    "tooltip": "控制「分镜序列」端口输出内容：开=输出「分镜情节」中该编号分镜的「情节」内容（纯文本，替代整合格式）；关=输出 detailed_description(整体风格) + [Shot N]时间段 + 环境音 + BGM 整合格式。未找到对应分镜情节时回退为整合格式。"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING", "STRING", "STRING", "STRING", "FLOAT", "BOOLEAN")
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
        try:
            return str(val)
        except Exception:
            return ""

    @staticmethod
    def _parse_json_text(text):
        """解析 JSON 字符串：先整体解析；失败时按多个 JSON 对象拼接处理（raw_decode 逐段解析并合并，同名键后者覆盖）。

        支持多个 JSON 对象拼接输入（如分镜策划五档案与视频提示词对象合并），
        避免整体 json.loads 抛 JSONDecodeError 导致提取全部为空。
        """
        s = text.strip() if isinstance(text, str) else text
        if not s:
            return {}
        try:
            obj = _json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
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

    @staticmethod
    def _extract_name(entry):
        """从档案条目中提取名称（第一个逗号前的部分）。"""
        s = str(entry) if entry is not None else ""
        for sep in ("，", ","):
            if sep in s:
                return s.split(sep)[0].strip()
        return s.strip()

    @staticmethod
    def _find_appearing_indices(text, char_names, prop_names):
        """在非台词区域查找出现的角色和道具（最长匹配优先，避免子串误匹配），返回按首次出现位置排序的 (角色索引列表, 道具索引列表)。

        台词保护：引号对与 <d>...</d> 标签内部出现的名称不匹配、不提取。
        """
        if not text:
            return [], []

        mask = _build_dialogue_protect_mask(text)
        n = len(text)

        # 收集所有名称：(name, type, index, length)
        all_names = []
        for i, name in enumerate(char_names):
            if name:
                all_names.append((name, 'char', i, len(name)))
        for i, name in enumerate(prop_names):
            if name:
                all_names.append((name, 'prop', i, len(name)))

        # 按长度降序排序（最长优先匹配，避免子串冲突）
        all_names.sort(key=lambda x: -x[3])

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
    def _build_detailed_description(整体风格, 时间段, 环境音, BGM, 类型="", bgm_enabled=True):
        """整合分镜序列：detailed_description(整体风格) → 时间段逐行编号[Shot N] → overall_soundscape(环境音) → non_diegetic_music(BGM)；标签与内容间仅换行不空行。

        类型含「武戏」时时间段不加 [Shot N] 编号，原行输出。
        bgm_enabled=False 时 non_diegetic_music 内容输出 N/A。
        """
        if isinstance(时间段, str):
            shot_lines = 时间段.split("\n") if 时间段.strip() else []
        elif isinstance(时间段, list):
            shot_lines = 时间段
        else:
            shot_lines = []
        is_action = "武戏" in str(类型 or "")
        # 时间段逐行编号 [Shot N]（武戏不加编号），跳过空行，编号连续
        numbered = []
        for line in shot_lines:
            s = str(line) if line is not None else ""
            if s.strip():
                if is_action:
                    numbered.append(s)
                else:
                    numbered.append(f"[Shot {len(numbered) + 1}] {s}")
        blocks = ["detailed_description:" + ("\n" + 整体风格 if 整体风格 else "")]
        if numbered:
            blocks.append("\n\n".join(numbered))
        blocks.append("overall_soundscape:" + ("\n" + 环境音 if 环境音 else ""))
        if bgm_enabled:
            blocks.append("non_diegetic_music:" + ("\n" + BGM if BGM else ""))
        else:
            blocks.append("non_diegetic_music:\nN/A")
        return "\n\n".join(blocks)

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
            scene_str = str(scene) if scene is not None else ""
            scene_name = scene_str
            for sep in ("，", ","):
                if sep in scene_name:
                    scene_name = scene_name.split(sep)[0]
                    break
            scene_name = scene_name.strip()
            if not scene_name:
                continue
            if prefix in scene_name or scene_name in prefix:
                return idx
        return -1

    @staticmethod
    def _archive_indices_by_names(appearing, archive_names):
        """按显式出现名称列表解析档案 0 基索引（去重、保持出现顺序）。

        返回 None 表示 appearing 不是列表（无显式列表，由调用方走文本匹配兜底）。
        匹配规则：精确匹配档案提取名，未命中时再取该名称第一个逗号前的部分匹配。
        """
        if not isinstance(appearing, list):
            return None
        index_map = {name: i for i, name in enumerate(archive_names) if name}
        result = []
        for entry in appearing:
            if entry is None:
                continue
            name = str(entry)
            if name not in index_map:
                for sep in ("，", ","):
                    if sep in name:
                        name = name.split(sep)[0].strip()
                        break
            if name in index_map:
                idx = index_map[name]
                if idx not in result:
                    result.append(idx)
        return result

    def extract_json(self, json=None, 索引=1, 档案选择="角色档案", 角色开关=True, 道具开关=True, 场景开关=True, BGM开关=True, 情节开关=False):
        # 形参名必须与输入端口名一致（ComfyUI 按关键字传参）
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
        档案 = self._list_to_lines(档案表.get(档案选择, 角色档案数据))

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
                    matched_title = str(item.get("标题", ""))
                    matched_type = str(item.get("类型", ""))
                    found_shot = True
                    break
        # 仅命中分镜情节而未命中分镜序列时，标题/类型取自分镜情节条目
        if 情节条目 is not None and not matched_title:
            matched_title = str(情节条目.get("标题", ""))
        if 情节条目 is not None and not matched_type:
            matched_type = str(情节条目.get("类型", ""))
        # 情节开关开启且找到对应分镜情节：分镜序列端口输出「情节」纯文本；否则输出整合格式（未命中分镜时时间段/环境音/BGM 留空，不报错；BGM开关关闭时 non_diegetic_music 输出 N/A）
        if 情节开关 and 情节条目 is not None and str(情节条目.get("情节") or "").strip():
            分镜序列整合 = str(情节条目.get("情节"))
        else:
            分镜序列整合 = self._build_detailed_description(整体风格, 时间段, 环境音, BGM, matched_type, BGM开关)

        # 角色道具场景：未找到分镜输出空；否则按 角色→道具→场景 顺序输出档案完整描述
        # 角色/道具/场景索引：匹配到的档案 0 基序号（角色/道具逗号分隔，场景单个），未匹配为空
        if not found_shot:
            角色道具场景 = ""
            角色索引 = ""
            道具索引 = ""
            场景索引 = ""
            索引时长 = 0.0
        else:
            # 角色、道具：优先取「分镜情节」条目的 出现角色/出现道具 显式列表；未提供时在分镜时间段文本中智能匹配
            char_names = [self._extract_name(e) for e in (角色档案数据 if isinstance(角色档案数据, list) else [])]
            prop_names = [self._extract_name(e) for e in (道具档案数据 if isinstance(道具档案数据, list) else [])]
            出现角色 = 情节条目.get("出现角色") if 情节条目 is not None else None
            出现道具 = 情节条目.get("出现道具") if 情节条目 is not None else None
            char_indices = self._archive_indices_by_names(出现角色, char_names)
            prop_indices = self._archive_indices_by_names(出现道具, prop_names)
            if char_indices is None or prop_indices is None:
                text_char, text_prop = self._find_appearing_indices(分镜序列文本, char_names, prop_names)
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

            # 角色/道具/场景索引：对应档案的 0 基序号，未匹配为空；由对应输出开关控制，关时输出空文本
            角色索引 = ",".join(str(i) for i in char_indices) if 角色开关 else ""
            道具索引 = ",".join(str(i) for i in prop_indices) if 道具开关 else ""
            场景索引 = (str(idx) if idx >= 0 else "") if 场景开关 else ""

            # 按开关过滤后输出：先角色再道具最后场景
            # 情节开关开启时输出纯描述文本（无 retention_analysis: 前缀、无 <Picture N>：序号）；
            # 情节开关关闭时整体以 retention_analysis: 开头、每条加 <Picture N> 序号
            输出块 = []
            if 角色开关:
                输出块.extend(角色描述列表)
            if 道具开关:
                输出块.extend(道具描述列表)
            if 场景开关 and 场景描述:
                输出块.append(场景描述)
            if 输出块:
                if 情节开关:
                    角色道具场景 = "\n".join(输出块) + "\n"
                else:
                    编号行 = [f"<Picture {i + 1}>：{d}" for i, d in enumerate(输出块)]
                    角色道具场景 = "retention_analysis:\n" + "\n".join(编号行) + "\n"
            else:
                角色道具场景 = ""

            # 索引时长：优先取「类型」字段中的时长；未含数字时回退为时间段文本中的最大结束时间
            索引时长 = self._type_duration_seconds(matched_type)
            if 索引时长 is None:
                索引时长 = self._max_duration_seconds(分镜序列文本)

        # 场景上下文：当前分镜（编号=索引）与上一个分镜（编号-1）的场景关键词是否相同（取「-」前部分比较）；索引为1（无前参考）或找不到上一编号则输出 False
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

        return {
            "result": [整体风格, 档案, 档案编码, 分镜序列整合, 角色道具场景, 角色索引, 道具索引, 场景索引, 索引时长, 场景判断]
        }


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
                    "tooltip": "● 列表：输出为包含所有编号文本的字符串列表。\n● 合并文本：将所有带编号的文本合并成一个字符串。"
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
                    "tooltip": "要查找的文本列表，每行对应一组。\n第1行对应替换文本第1行，第2行对应替换文本第2行，以此类推。"
                }),
                "替换文本": ("STRING", {
                    "multiline": True,
                    "placeholder": "每行一个要替换的文本...",
                    "tooltip": "要替换的文本列表，每行对应一组。\n第1行对应查找文本第1行，第2行对应查找文本第2行，以此类推。"
                }),
                "台词开关": ("BOOLEAN", {
                    "default": False,
                    "label_on": "保护台词",
                    "label_off": "正常替换",
                    "display_name": "台词保护",
                    "tooltip": "【台词保护】\n开启后，被以下引号包裹的「人物说话内容」不进行替换，原文保留：\n"
                               "● 半角双引号 / 单引号：\"...\"  '...'\n"
                               "● 中文弯引号：“...”  ‘...’\n"
                               "● 中文直角引号：「...」 『...』\n"
                               "关闭时，整段文本正常执行批量替换。",
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
        find_lines = 查找文本.split("\n")
        replace_lines = 替换文本.split("\n")

        # 台词开关：开启时逐轮以当前文本重建保护掩码进行保护替换
        protect = bool(台词开关)

        # 配对查找/替换，过滤空查找串
        pairs = []
        count = min(len(find_lines), len(replace_lines))
        for i in range(count):
            find_str = find_lines[i]
            replace_str = replace_lines[i] if i < len(replace_lines) else ""
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


# ==== 分镜角色替换 ====

class YUAN_TXTShotReplace:

    # 台词归属边界标点：说话者前必须是句首/句末标点/逗号/空白边界，或紧跟 </d> 之后
    # （「甲对乙说」中的乙由其前的"对"字非边界挡住；「甲看向乙，乙说道」中的乙由逗号边界放行、
    #   归属竞争时离冒号最近者胜；</d>台词结束后紧跟的下一个说话者由 </d> 边界放行，
    #   而「盯住<Picture 2>急道」中的 <Picture 2> 前是标签结束符">"而非 </d>，不会被放行误归属）
    BELONG_BOUNDARY = '。！？!?\n；;，,'

    @staticmethod
    def _build_belong_pattern(name):
        """构建台词归属匹配 pattern：`说话者(<Audio M>)?+引导句+冒号+<d>台词`。

        说话者可为名称或 <Picture N> 标记；可选的已有 <Audio M>（输出重入时）会被跳过并重写新编号。
        引导句允许逗号/顿号等多动作描述（如「在蒲团前刹住脚步，胸口起伏，仰头急道」），
        但不得跨句（。！？!?；;换行）、跨冒号（天然不会吞并其他台词）。
        """
        boundary = YUAN_TXTShotReplace.BELONG_BOUNDARY
        return re.compile(
            r'(?:(?<=</d>)|(?<![^\s' + boundary + r']))'  # 说话者前边界：句末标点/逗号/空白 或 </d> 之后
            r'(' + re.escape(name) + r')'        # 组1：说话者（名称或 <Picture N>）
            r'(?:<Audio \d+>)?'                  # 可选：已有的 <Audio M>（重入时跳过，替换时重写）
            r'([^。！？!?；;\n\r：:]*?)'          # 组2：引导句（允许逗号顿号，不得跨句/跨冒号）
            r'[：:]\s*'                          # 冒号+可选空白
            r'(?=<d>)'                           # 紧跟台词标签
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "角色道具场景": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "tooltip": "接入 JSON提取节点的「角色道具场景」输出。每行 <Picture N>：名称，描述… 取第一个逗号前的名称作为查找对象，在分镜序列中替换为对应的 <Picture N> 标记。"
                }),
                "分镜序列": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "tooltip": "接入 JSON提取节点的「分镜序列」输出。文本中出现的角色/道具/场景名称将被替换为对应的 <Picture N> 标记（最长匹配优先），其余内容原样保留。\n"
                               "【台词归属特殊规则】说话者（名称或 <Picture N> 标记）后跟引导句+冒号+<d>台词时（如「沈惊鸿低声说：<d>…</d>」「<Picture 1>刹住脚步，胸口起伏，仰头急道：<d>…</d>」，引导句可含逗号等多动作描述），说话者替换为 <Picture N> 后追加 <Audio M> 音频标记；同一台词有多个候选说话者时（如「甲看向乙，乙说道：」）取离冒号最近者为说话者。\n"
                               "【兜底追溯】标准规则未命中时（说话者被「将/把」等字挡住，如「炽红火光将<Picture 4>…发抖：<d>…</d>」；或承前省略主语「他怒道：」），从冒号往前追溯最近的未保护 <Picture N> 标记或名称作为说话者（引号内/<d>内候选跳过）。\n"
                               "<Audio M> 按说话角色在分镜序列中首次说话的出现顺序编号：第一个说话的角色 <Audio 1>，第二个 <Audio 2>，第三个 <Audio 3>，最多三个。"
                }),
                "台词开关": ("BOOLEAN", {
                    "default": True,
                    "label_on": "保护台词",
                    "label_off": "正常替换",
                    "display_name": "台词保护",
                    "tooltip": "【台词保护】\n开启后，被以下格式包裹的「人物说话内容」中的名称不替换、原文保留：\n"
                               "● 半角双引号 / 单引号：\"...\"  '...'\n"
                               "● 中文弯引号：“...”  ‘...’\n"
                               "● 中文直角引号：「...」 『...』\n"
                               "● 台词标签：<d>...</d>\n"
                               "关闭时，整段文本正常执行名称替换。",
                }),
                "参考音频开关": ("BOOLEAN", {
                    "default": True,
                    "label_on": "输出<Audio M>",
                    "label_off": "输出<Picture N>",
                    "display_name": "参考音频",
                    "tooltip": "【参考音频】\n"
                               "开启后，台词归属特殊规则生效：`说话者+引导句+：<d>台词` 中说话者替换为 `<Picture N><Audio M>`\n"
                               "（如「沈惊鸿低声说：<d>…</d>」→「<Picture 1><Audio 1>低声说：<d>…</d>」），用于关联参考音频。\n"
                               "引导句允许逗号等多动作描述（如「刹住脚步，胸口起伏，仰头急道：」）；同一台词有多个候选说话者时（如「甲看向乙，乙说道：」）取离冒号最近者为说话者。\n"
                               "标准规则未命中时兜底追溯：从冒号往前追溯最近的未保护 <Picture N> 标记或名称作为说话者（说话者被「将/把」等字挡住或承前省略主语时生效，引号内/<d>内候选跳过）。\n"
                               "<Audio M> 按说话角色首次说话的出现顺序编号：第一个说话的角色 <Audio 1>，第二个 <Audio 2>，第三个 <Audio 3>，最多三个。\n"
                               "超过三个时不报错，第4个及以后的说话角色仍正常替换为 <Picture N>，只是不追加 <Audio M> 标记。\n"
                               "关闭后，不追加 <Audio M>，名称走普通替换输出 <Picture N>"
                               "（如「<Picture 1>低声说：<d>…</d>」），音色索引输出为空。",
                }),
                "角色索引": ("STRING", {
                    "forceInput": True,
                    "multiline": True,
                    "tooltip": "接入 JSON提取节点的「角色索引」输出。逗号分隔的数字串，第 N 个数字对应 <Picture N> 的音色索引，\n"
                               "如「0,4,1」：0 对应 <Picture 1>、4 对应 <Picture 2>、1 对应 <Picture 3>。\n"
                               "【音色索引输出】按说话顺序重新排列：只输出实际说话的角色对应的索引，未说话的角色不输出。\n"
                               "如 <Picture 3> 第一个说话、<Picture 2> 第二个说话、<Picture 1> 未说话，则音色索引输出「1,4」。\n"
                               "未连线或为空时，音色索引输出为空（不影响 <Audio M> 标记的追加）。",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("分镜序列", "音色索引",)
    FUNCTION = "replace_shot_names"
    CATEGORY = "Yuan Tool/文本"
    OUTPUT_NODE = True

    @staticmethod
    def _parse_picture_names(text):
        """从「角色道具场景」解析 <Picture N>：名称，描述… 定义，返回 [(名称, "<Picture N>"), ...]。

        名称取第一个逗号（全角/半角）前的部分；无逗号时取整行描述。
        """
        pairs = []
        if not text:
            return pairs
        for line in text.split("\n"):
            m = re.match(r'^\s*<Picture\s+(\d+)>\s*[：:]\s*(.+?)\s*$', line)
            if not m:
                continue
            desc = m.group(2)
            name = ""
            for sep in ("，", ","):
                if sep in desc:
                    name = desc.split(sep)[0].strip()
                    break
            if not name:
                name = desc.strip()
            if name:
                pairs.append((name, f"<Picture {m.group(1)}>"))
        return pairs

    @staticmethod
    def _build_protect_mask(text):
        """台词保护掩码：引号对内部 + <d>...</d> 标签内部（含标签自身），调用模块级共享实现。"""
        return _build_dialogue_protect_mask(text)

    @staticmethod
    def _parse_voice_indices(text):
        """解析角色索引串（如「0,4,1」）为整数列表；容错中文逗号/空格/非法项，返回 []。"""
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
        """扫描分镜序列，按说话角色首次说话的出现顺序分配 <Audio M> 编号（最多3个）。

        说话的两种形式都识别（只有说话的才有 <Audio M>）：
        - 名称形式：`名称+引导句+冒号+<d>台词`（如「沈惊鸿低声道：<d>…</d>」）
        - 标记形式：`<Picture N>+引导句+冒号+<d>台词`（如「<Picture 1>低声道：<d>…</d>」）

        归属竞争：同一台词（冒号+<d>）有多个候选说话者时（如「甲看向乙，乙说道：」中，
        甲的引导句「看向乙，乙说道」与乙的引导句「说道」同时命中），取离冒号最近者；
        起点相同时取名称更长者（防「林」抢先于「林小雨」）。

        兜底追溯：标准规则未命中的台词（说话者被「将/把/被」等非边界字挡住，
        如「炽红火光将<Picture 4>半边脸映得凶厉，…发抖：<d>」；或承前省略主语「他怒道：」），
        从冒号往前追溯最近的未被保护的 <Picture N> 标记或名称作为说话者
        （不限句界；处于引号内/<d>内的候选跳过继续往前；无候选则该台词无归属）。

        返回 (audio_by_tag, speaking_tags, winners)：
        - audio_by_tag：{<Picture N> 标记: "<Audio M>"}；超过3个不同说话角色的部分不分配
        - speaking_tags：[<Picture N> 标记...] 按 <Audio 1/2/3> 分配顺序（最多3个），用于音色索引输出
        - winners：[(台词目标位置, 替换起点, 替换终点, <Picture N> 标记)] 按台词出现顺序，
          替换区间 [起点, 终点) 覆盖说话者及已有 <Audio M>（重入时重写），用于位置化替换

        - 同一角色（同一 <Picture N>）无论以名称还是标记形式说话，只按首次出现分配一个编号
        - 台词保护开启时，处于被保护位置（引号内/<d>内）的不算说话
        """
        mask = YUAN_TXTShotReplace._build_protect_mask(text) if protect_enabled else None
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
            for m in YUAN_TXTShotReplace._build_belong_pattern(form).finditer(text):
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

    def replace_shot_names(self, 角色道具场景, 分镜序列, 台词开关, 参考音频开关, 角色索引):
        pairs = self._parse_picture_names(角色道具场景 or "")

        # 按名称长度降序排序（最长匹配优先，避免短名误替换长名中的子串）
        pairs.sort(key=lambda x: -len(x[0]))

        result = 分镜序列 or ""
        音色索引 = ""

        # 特殊规则：只有说话的角色才有 <Audio M>（最多3个，按首次说话顺序编号）
        # 两种说话形式都处理：名称+引导句+：<d> 与 <Picture N>+引导句+：<d>
        # 参考音频开关关闭时跳过，走普通 <Picture N> 替换
        if 参考音频开关:
            audio_by_tag, speaking_tags, winners = self._scan_speaking_order(result, pairs, 台词开关)
            # 位置化替换（从右往左，避免位置偏移）：说话者及已有 <Audio M> → <Picture N><Audio M>
            # 追溯命中的区间可能与标准命中的区间重叠（同一说话者先标准命中、后句承前省略又追溯到它），
            # 同角色同标记，重叠区间跳过即可（内容一致，二次替换反而错位）
            replaced_ranges = []
            for _end, s1, _e1, s2, tag in sorted(winners, key=lambda x: -x[0]):
                audio = audio_by_tag.get(tag)
                if audio:
                    if any(s1 < re_ and s2 > rs for rs, re_ in replaced_ranges):
                        continue
                    result = result[:s1] + tag + audio + result[s2:]
                    replaced_ranges.append((s1, s2))

            # 音色索引：按说话顺序（<Audio 1/2/3> 分配顺序）取各角色 <Picture N> 对应的角色索引值
            # （第 N 个数字对应 <Picture N>）；未说话的角色不输出，索引越界的跳过
            if speaking_tags:
                indices = self._parse_voice_indices(角色索引)
                if indices:
                    values = []
                    for tag in speaking_tags:
                        m = re.match(r'^<Picture (\d+)>$', tag)
                        if m:
                            n = int(m.group(1))
                            if 1 <= n <= len(indices):
                                values.append(str(indices[n - 1]))
                    音色索引 = ",".join(values)

        # 普通替换：名称 → <Picture N>
        for name, tag in pairs:
            if 台词开关:
                result = YUAN_TXTReplace._replace_with_protect(
                    result, name, tag, YUAN_TXTShotReplace._build_protect_mask(result))
            else:
                result = result.replace(name, tag)

        return (result, 音色索引)


# ==== 文本处理（分段） ====

class YUAN_TXTParagraphSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "placeholder": "输入需要分割的文本...",
                    "tooltip": "基础文本输入框。\n如果您使用[输入端口]功能连接了其他节点，此处的文本将作为第1部分，其他端口(any_xx)的内容会按顺序拼接在其后。"
                }),
                "输出模式": ("BOOLEAN", {
                    "default": False,
                    "label_on": "输出分段列表",
                    "label_off": "输出原始文本",
                    "tooltip": "控制端口输出的内容：\n● 输出原始文本（执行分段方式、段落优化、选取段落等所有处理规则，最终合并为一段文本输出）。\n● 输出分段列表（输出分割处理后的内容，按分段方式进行分割，以列表形式输出）。"
                }),
                "段落优化": ("BOOLEAN", {
                    "default": True,
                    "label_on": "去除首尾空格",
                    "label_off": "保留原始空格",
                    "tooltip": "优化文本空格\n● 去除首尾空格（自动删除首尾空格、换行符。无论输出原文还是分段均有效）。\n● 保留原始空格（完全保留原始文本的格式和缩进）。"
                }),
                "分段方式": (["端口", "空行", "序号", "段落", "标题", "数字", "地址", "手动"], {
                    "default": "空行",
                    "tooltip": "【核心分割逻辑】\n● 端口：严格按输入端口(any_x)分割。\n● 空行：识别双换行符。\n● 序号：识别 1. / (1) / A. 等列表标记。\n● 段落：每一行算一段。\n● 标题：智能识别章节标题。\n● 数字：仅提取纯数字。\n● 地址：智能从乱码、列表、对象字符串中提取 Windows 文件路径 (如 D:\\Data\\img.png)，并自动清洗格式。\n● 手动：识别 ||| 分隔符进行自定义分割。"
                }),
                "输出段落": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "display": "number",
                    "tooltip": "【动态扩展输出】\n设置节点右侧[段落x]输出端口的数量。\n例如设为 3，右侧会出现 段落1, 段落2, 段落3。\n(需点击节点上的「更新端口」按钮生效)"
                }),
                "输入端口": ("INT", {
                    "default": 1,
                    "min": 1,
                    "step": 1,
                    "display": "number",
                    "tooltip": "【动态扩展输入】\n设置节点左侧[any_x]输入端口的数量。\n用于将多个文本源（如多个加载文本节点）按顺序拼合在一起进行统一分段处理。\n(注意：修改数值后需点击节点上的「更新端口」按钮生效)"}),
                "选取段落": ("STRING", {
                    "default": "-1",
                    "placeholder": "输入要选取的段落，用逗号分隔，如0,2,4；填 -1 输出所有；留空总段输出为空",
                    "tooltip": "【分割后段落选取】\n决定选取哪些段落输出。\n● -1（默认）：输出所有段落。\n● 留空：不选取任何段落，总段输出为空。\n● 0 为第一段、1 为第二段，以此类推。\n● 输入 0,2,4：输出第1、3、5段，丢弃其他。\n此设置会改变[总段]和[段落x]端口的内容。"
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
            return (0, "",) + ("",) * self.MAX_OUTPUTS

        if 分段方式 == "端口":
            if collected_texts:
                # 端口模式下每个 any_x 端口对应一个段落位置：即使端口为空也保留占位，
                # 保证选取段落索引与端口序号一一对应（如 any2 未接入时索引1应输出空文本）
                paras = []
                for i in range(1, input_count + 1):
                    val = kwargs.get(f"any_{i}", None)
                    if val is None:
                        paras.append("")
                    else:
                        s = self._convert_to_str(val)
                        paras.append(s.strip() if 段落优化 else s)
            else:
                paras = [text.strip() if 段落优化 else text] if text else []
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
            lines = text.split('\n')
            paras = []
            for line in lines:
                pl = line.strip() if 段落优化 else line
                if pl: paras.append(pl)
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
            pattern = r'([a-zA-Z]:[\\/][^"\'<>,;\[\]\n\r]+)'
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
                except:
                    continue

        if not 输出模式:
            to = ["\n".join(to)] if to else [""]

        cnt = len(to)

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
                        "选择长度计数方式：\n"
                        "● 字符串：直接输出文本的字符总数（空文本为 0）。\n"
                        "● 段落：每一行算一个段落，统计行数（空行也单独算一行）。\n"
                        "● 空行：按空行（连续换行，段间空白行）分割文本，统计分割后的段落块数量。\n"
                        "● 列表：若上游输出为列表（如文本处理节点的「输出分段列表」），统计列表元素个数；不是列表则按整个文本整体视为 1。\n"
                        "任何模式下，空文本（无内容）均输出 0。",
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("长度",)
    OUTPUT_TOOLTIPS = ("根据所选长度模式统计得到的数量。",)
    FUNCTION = "count_length"
    CATEGORY = "Yuan Tool/文本"
    DESCRIPTION = (
        "统计接入文本的长度：字符串长度 / 段落（行数） / 空行分段块数 / 列表元素个数。"
        "空文本（无内容）在任意模式下均输出 0。"
    )

    @staticmethod
    def _to_plain_text(val):
        """把任意 ComfyUI 输入统一转成纯字符串（list 用换行拼接，其他走 str）。"""
        if val is None:
            return ""
        if isinstance(val, list):
            parts = []
            for v in val:
                parts.append("" if v is None else str(v))
            return "\n".join(parts)
        if isinstance(val, (dict, tuple)):
            return str(val)
        if isinstance(val, (int, float, bool)):
            return str(val)
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


NODE_CLASS_MAPPINGS = {
    "YUAN_TXTJsonExtractor": YUAN_TXTJsonExtractor,
    "YUAN_TXTAppearanceOrder": YUAN_TXTAppearanceOrder,
    "YUAN_TXTConvertAny": YUAN_TXTConvertAny,
    "YUAN_TXTListNumber": YUAN_TXTListNumber,
    "YUAN_TXTReplace": YUAN_TXTReplace,
    "YUAN_TXTShotReplace": YUAN_TXTShotReplace,
    "YUAN_TXTParagraphSplitter": YUAN_TXTParagraphSplitter,
    "YUAN_TXTLength": YUAN_TXTLength,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YUAN_TXTJsonExtractor": "JSON提取",
    "YUAN_TXTAppearanceOrder": "出场排序",
    "YUAN_TXTConvertAny": "格式转换",
    "YUAN_TXTListNumber": "列表编号",
    "YUAN_TXTReplace": "文本批量替换",
    "YUAN_TXTShotReplace": "分镜角色替换",
    "YUAN_TXTParagraphSplitter": "文本处理",
    "YUAN_TXTLength": "长度",
}
