# ComfyUI-Yuan-Tool

ComfyUI 自定义节点工具集，提供图像处理、文本处理、格式转换以及基于时间轴的视频提示词编码等功能，适配 LTX/Wan 视频生成模型工作流。

---

## 节点列表

### 1. 多帧参考 (`YuanTool`)
- **分类**: `Yuan Tool/图像`
- **功能**: 将多张主体图像与一张背景图合成为固定帧数的参考视频，专为 LTX 2.3 Tool (Multiple-Subject-Reference) LoRA 工作流设计。

| 参数 | 类型 | 说明 |
|------|------|------|
| `1` ~ `4` | IMAGE (可选) | 主体图像（按 1 → 2 → 3 → 4 顺序排列） |
| `background` | IMAGE (必填) | 背景图像，始终排在最后一帧 |
| `width` | INT | 输出视频宽度 (默认 736，步长 32) |
| `height` | INT | 输出视频高度 (默认 1280，步长 32) |
| `frame_multiplier` | 选项 | 帧数倍率：8 / 10 / 12 / 16。总帧数 = `图像数 × 倍率 + 1` |
| `list_mode` | BOOL | 开启后使用 `image_list` 输入替代 1~4 独立端口 |
| `image_list` | IMAGE (可选) | 批量图像输入（前 4 张有效） |

- **输出**: `IMAGE` — 合成后的视频帧序列张量

---

### 2. 筛选图像 (`GetImage`)
- **分类**: `Yuan Tool/图像`
- **功能**: 从批量图像中按索引选取一张或多张图像。

| 参数 | 类型 | 说明 |
|------|------|------|
| `images` | IMAGE | 输入批量图像 |
| `indexes` | STRING | 索引列表，逗号分隔，如 `"0, 1, 2"` |

- **输出**: `IMAGE` — 筛选后的图像张量

---

### 3. 文本处理 (`YUAN_TXTParagraphSplitter`)
- **分类**: `Yuan Tool/文本`
- **功能**: 多功能文本分割器，支持 8 种分割模式、动态输入/输出端口扩展、段落优化与选取。

#### 分割模式
| 模式 | 说明 |
|------|------|
| `端口` | 严格按 `any_x` 输入端口分割 |
| `空行` | 识别双换行符 `\n\n` 作为段落分隔 |
| `序号` | 识别 `1.` / `(1)` / `A.` 等列表标记 |
| `段落` | 每一行算一段 |
| `标题` | 智能识别章节标题（括号标题、序号标题等） |
| `数字` | 仅提取纯数字 |
| `地址` | 智能提取 Windows 文件路径（如 `D:\Data\img.png`），自动清洗格式 |
| `手动` | 识别 `|||` 分隔符进行自定义分割 |

#### 主要参数
| 参数 | 类型 | 说明 |
|------|------|------|
| `text` | STRING | 基础文本输入 |
| `any_1` ~ `any_64` | AnyType (可选) | 动态扩展输入端口，可连接任意节点按顺序拼接 |
| `分段方式` | 选项 | 选择分割模式 |
| `段落优化` | BOOL | 去除首尾空格 / 保留原始格式 |
| `输出模式` | BOOL | 输出原始文本 / 输出分段列表 |
| `输出段落` | INT | 动态设置右侧 `段落1` ~ `段落N` 输出端口数量 |
| `输入端口` | INT | 动态设置左侧 `any_x` 输入端口数量 |
| `选取段落` | STRING | 选取指定索引的段落，如 `"1,3,5"`（0 为第一段） |

- **输出**: `数:` (INT) — 段落总数, `总段:` (STRING 列表) — 所有段落, `段落1` ~ `段落100` (STRING) — 单个段落

---

### 4. 列表编号 (`YUAN_TXTListNumber`)
- **分类**: `Yuan Tool/文本`
- **功能**: 为多行文本自动添加编号，支持自定义前缀、后缀和起始编号。

| 参数 | 类型 | 说明 |
|------|------|------|
| `文本` | STRING | 每行一组，自动编号 |
| `起始编号` | INT | 编号起始值 (默认 1) |
| `编号前缀` | STRING | 编号前添加的文本，如 `"第"` |
| `编号后缀` | STRING | 编号后添加的文本，如 `"项"` |
| `输出模式` | 选项 | `列表` 输出字符串列表 / `合并文本` 合并为一个字符串 |
| `合并间隔符` | STRING | 合并时的分隔符（支持 `\n` 换行） |

- **输出**: `输出` (STRING 列表) — 编号后的文本列表, `接续编号` (INT) — 下一批的起始编号

---

### 5. 格式转换 (`YUAN_TXTConvertAny`)
- **分类**: `Yuan Tool/文本`
- **功能**: 接受任意类型输入，转换为指定的目标类型。

| 参数 | 类型 | 说明 |
|------|------|------|
| `*` | AnyType | 接受任何类型的输入 |
| `格式类型` | 选项 | 目标类型：`string` / `int` / `float` / `boolean` |

- **输出**: AnyType — 转换后的值

---

### 6. Yuan CLIP 时间轴 (`YuanCLIPTimeline`)
- **分类**: `Yuan Tool/CLIP`
- **功能**: 可视化时间轴提示词编码节点，复刻自 ComfyUI-PromptRelay 的 Prompt Relay Encode (Timeline) 节点。为视频扩散模型 (Wan / LTX) 注入时间感知的提示词调度，自动生成视频和音频空潜空间。配套 Web 时间轴编辑器 UI。

#### 核心特性
- **时间感知注意力惩罚**: 对交叉注意力注入高斯惩罚矩阵，使不同时间段的提示词在对应帧上具有更高的注意力权重
- **自动潜空间生成**: 未连接 latent 输入时，自动生成 LTXV 兼容的视频/音频空潜空间（零张量），由采样器加噪去噪
- **text_input 智能解析**: 支持时间格式行 `"0-3s 提示词A"` 动态分配帧长，也支持纯文本行自动均分
- **提示词锁定模式**: prompt_lock 开启时，text_input 自动同步到时间轴各段落

#### 主要参数
| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | MODEL | 要补丁的扩散模型 (Wan / LTX) |
| `clip` | CLIP | 用于编码提示词的 CLIP 模型 |
| `audio_vae` | VAE | Audio VAE，用于生成音频潜空间 |
| `global_prompt` | STRING | 全局提示词，贯穿整个视频 |
| `max_frames` | INT | 像素空间总帧数 (默认 129) |
| `timeline_data` | STRING | 时间轴编辑器 JSON 状态（自动管理） |
| `local_prompts` | STRING | 各段提示词，用 `\|` 分隔（自动填充） |
| `segment_lengths` | STRING | 各段帧数，逗号分隔（自动填充） |
| `epsilon` | FLOAT | 惩罚衰减参数 (默认 1e-3，越小边界越锐利) |
| `fps` | FLOAT | 每秒帧数 (默认 24) |
| `time_units` | 选项 | 时间单位：`frames` / `seconds` |
| `width` | INT | 自动生成潜空间的目标宽度 (默认 768) |
| `height` | INT | 自动生成潜空间的目标高度 (默认 512) |
| `latent` | LATENT (可选) | 外部潜空间输入（不连接时自动生成） |
| `text_input` | STRING (可选) | 按行输入的提示词文本（支持时间格式） |
| `prompt_lock` | BOOL (可选) | 提示词锁定模式 (默认 true) |

- **输出**: `model` (MODEL) — 补丁后的模型, `positive` (CONDITIONING) — 正向条件, `video_latent` (LATENT) — 视频潜空间, `audio_latent` (LATENT) — 音频潜空间

---

## 安装

将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yuan-SiO2/ComfyUI-Yuan-Tool.git
```

安装依赖：

```bash
pip install -r ComfyUI-Yuan-Tool/requirements.txt
```

安装完成后重启 ComfyUI。

---

## 依赖

- `torch`
- `numpy`
- `opencv-python`
- `Pillow`
- ComfyUI 核心框架
