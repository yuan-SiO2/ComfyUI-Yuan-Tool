import re


class AnyType(str):
    def __ne__(self, __value):
        return False


# ==============================================================================
# 出场排序
# ==============================================================================

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


# ==============================================================================
# 格式转换
# ==============================================================================

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


# ==============================================================================
# 列表编号
# ==============================================================================

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
            return ([], 起始编号)

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


# ==============================================================================
# 文本批量替换
# ==============================================================================

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
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "replace_text"
    CATEGORY = "Yuan Tool/文本"
    OUTPUT_NODE = True

    def replace_text(self, text, 查找文本, 替换文本):
        find_lines = 查找文本.split("\n")
        replace_lines = 替换文本.split("\n")

        result = text
        count = min(len(find_lines), len(replace_lines))

        for i in range(count):
            find_str = find_lines[i]
            replace_str = replace_lines[i]
            if find_str:
                result = result.replace(find_str, replace_str)

        return (result,)


# ==============================================================================
# 文本处理（分段）
# ==============================================================================

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
                    "default": "",
                    "placeholder": "输入要选取的段落，用逗号分隔，如1,3,5",
                    "tooltip": "【分割后段落选取】\n决定选取哪些段落输出。\n● 留空：输出所有段落。\n● 0 为第一段、1 为第二段，以此类推。\n● 输入 1,3,5：输出第2、4、6段，丢弃其他。\n此设置会改变[总段]和[段落x]端口的内容。"
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
        ',', '，', '.', '。', '!', '！', '?', '？', ';', '；', '"', '"', "'", "'", '（', '）', '()', '[]', '{}', '、', '…', '—')
        if last_char in forbidden_punctuation: return False
        bracket_patterns = [(r'^【.+】$', '【】'), (r'^《.+》$', '《》'), (r'^<.+>$', '<>')]
        for pattern, bracket_type in bracket_patterns:
            if re.match(pattern, line_stripped): return True
        num_title_pattern = r'^(?:[一二三四五六七八九十百千万]+、|\d+\. |[a-zA-Z]+\. )'
        if re.match(num_title_pattern, line_stripped): return True
        if last_char in (':', '：'): return len(line_stripped) > 1
        if not last_char in (':', '：') and not re.search(r'[^\u4e00-\u9fa5a-zA-Z0-9]', last_char): return True
        return False

    def _convert_to_str(self, val):
        if isinstance(val, (str, int, float, bool)):
            return str(val)
        elif isinstance(val, dict):
            return str(val)
        elif isinstance(val, list):
            return "\n".join([str(x) for x in val])
        else:
            try:
                return str(val)
            except:
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
                paras = []
                for t in collected_texts:
                    paras.append(t.strip() if 段落优化 else t)
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

        if 选取段落.strip() == "":
            to = paras.copy()
        else:
            to = []
            si = re.split(r'[。,，./\\]', 选取段落)
            for i in si:
                try:
                    idx = int(i.strip())
                    if 0 <= idx < len(paras): to.append(paras[idx])
                except:
                    continue

        if not 输出模式:
            to = ["\n".join(to)] if to else [""]

        cnt = len(to)

        max_out = self.MAX_OUTPUTS
        po = [to[i] if i < 输出段落 and i < len(to) else "" for i in range(max_out)]

        return (cnt, to,) + tuple(po)


NODE_CLASS_MAPPINGS = {
    "YUAN_TXTAppearanceOrder": YUAN_TXTAppearanceOrder,
    "YUAN_TXTConvertAny": YUAN_TXTConvertAny,
    "YUAN_TXTListNumber": YUAN_TXTListNumber,
    "YUAN_TXTReplace": YUAN_TXTReplace,
    "YUAN_TXTParagraphSplitter": YUAN_TXTParagraphSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YUAN_TXTAppearanceOrder": "出场排序",
    "YUAN_TXTConvertAny": "格式转换",
    "YUAN_TXTListNumber": "列表编号",
    "YUAN_TXTReplace": "文本批量替换",
    "YUAN_TXTParagraphSplitter": "文本处理",
}
