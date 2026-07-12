class YUAN_TXTReplace:

    def __init__(self):
        pass

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


NODE_CLASS_MAPPINGS = {
    "YUAN_TXTReplace": YUAN_TXTReplace
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YUAN_TXTReplace": "文本批量替换"
}
