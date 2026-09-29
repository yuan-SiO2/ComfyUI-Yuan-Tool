"""自定义选项节点：「编辑」模式填写各选项名称；「选项」模式按名称显示互斥开关，只能开启一个。输出当前开启项的文本与位置索引。"""


class YuanCustomCombo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ["选项", "编辑"],
                    {
                        "default": "编辑",
                        "display_name": "模式",
                        "tooltip": "选项：按名称显示互斥开关，只能开启一个；编辑：填写各选项名称。",
                    },
                ),
                "index": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 9999,
                        "step": 1,
                        "display_name": "索引",
                        "tooltip": "由前端自动维护：当前开启的选项在名称列表中的位置；未开启任何开关时为 -1。",
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "display_name": "文本",
                        "tooltip": "由前端自动维护：当前开启的选项名称；未开启任何开关时为空字符串。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("索引", "文本")
    OUTPUT_TOOLTIPS = (
        "当前开启的选项在名称列表中的位置；未开启任何开关时为 -1。",
        "当前开启的选项名称；未开启任何开关时为空字符串。",
    )
    FUNCTION = "execute"
    CATEGORY = "Yuan Tool/选项"
    DESCRIPTION = (
        "「编辑」模式填写各选项名称（最后一个名称框输入内容即自动新增一栏）；"
        "「选项」模式按名称显示互斥开关，只能开启一个。输出所选选项的名称与位置索引。"
    )

    def execute(self, mode, index, text):
        return (index, text)


NODE_CLASS_MAPPINGS = {
    "YuanCustomCombo": YuanCustomCombo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuanCustomCombo": "自定义选项",
}
