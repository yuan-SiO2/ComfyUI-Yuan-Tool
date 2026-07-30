# ComfyUI-Yuan-Tool

ComfyUI 自定义节点工具集，提供图像处理、全景预览、画布合成、色彩匹配、文本处理、格式转换以及基于时间轴的视频提示词编码等功能，适配 LTX/Wan 视频生成模型工作流。

---

## 节点列表

### 1. 多帧参考 (`YuanTool`)
- **分类**: `Yuan Tool/图像`
- **功能**: 将多张主体图像与一张背景图合成为固定帧数的参考视频，专为 LTX 2.3 Tool (Multiple-Subject-Reference) LoRA 工作流设计。

| 参数 | 类型 | 说明 |
|------|------|------|
| `1` ~ `4` | IMAGE (可选) | 主体图像（按 1 → 2 → 3 → 4 顺序排列） |
| `background` | IMAGE (必填) | 背景图像，固定 8 帧，排在最后 |
| `width` | INT | 输出视频宽度 (默认 736，步长 32) |
| `height` | INT | 输出视频高度 (默认 1280，步长 32) |
| `frame_multiplier` | 选项 | 帧数倍率：8 / 16 / 24 / 32 |
| `list_mode` | BOOL | 开启后使用 `image_list` 输入替代 1~4 独立端口 |
| `image_list` | IMAGE (可选) | 批量图像输入（前 4 张有效） |

- **输出**: `output` (IMAGE) — 合成后的视频帧序列张量

#### 帧数计算公式
```
总帧数 = 主体图像数量 × frame_multiplier + 1（首图额外） + 8（背景）
```

#### 图像预处理
- **主体图像**：等比例缩放，保持完整，居中放在白色画布上（不变形、不裁剪）
- **背景图像**：等比例缩放覆盖整个画布，居中裁剪（填满目标尺寸）
- **智能插值**：缩小用 INTER_AREA，放大用 INTER_LANCZOS4，无 OpenCV 时降级为 torch bicubic

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

### 3. 全景预览 (`YuanPanoramaPreview`)
- **分类**: `Yuan Tool/图像`
- **功能**: 交互式预览 ERP 全景图（360°/180°），支持视频批次输入。复刻自 ComfyUI-Panorama-Stickers，配套 WebGL 球面投影编辑器。后端将输入图像落盘为预览图，视频批次额外编码 mp4 预览；前端可输出当前 3D 视角的裁剪截图。

| 参数 | 类型 | 说明 |
|------|------|------|
| `ERP_image` | IMAGE | ERP 全景图（支持单图或视频批次 `[B,H,W,C]`） |
| `Coverage` | 选项 | 全景覆盖范围：`360` / `180` |
| `output_current_view` | BOOL | 全景模式（输出完整 ERP 图）/ 裁剪模式（输出当前 3D 裁剪截图） |
| `view_width` | INT | 裁剪模式下的输出宽度 (默认 1024) |
| `view_height` | INT | 裁剪模式下的输出高度 (默认 512) |
| `current_view_data` | STRING (隐藏) | 前端回传的当前视角截图 data URL（自动管理） |

- **输出**: `输出` (IMAGE) — 全景模式输出完整 ERP 图，裁剪模式输出当前视角截图
- **备注**: 视频预览依赖 PyAV (`pip install av`)，缺失时仅放弃 mp4 预览，不影响图像预览

---

### 4. 全景接缝 (`YuanPanoramaSeamPrep`)
- **分类**: `Yuan Tool/图像`
- **功能**: 为接缝修复（seam-focused inpainting）准备 ERP 图像。将接缝平移到图像中心，并生成接缝带掩码与高斯模糊掩码。

| 参数 | 类型 | 说明 |
|------|------|------|
| `image` | IMAGE | 输入 ERP 图像 `[B,H,W,C]` |
| `seam_width_px` | INT | 接缝带宽度（像素，默认 64） |
| `seam_center_offset_px` | INT | 接缝中心偏移量（正值右移，负值左移，默认 0） |
| `mask_blur_px` | INT | 掩码高斯模糊半径（默认 10，0 表示不模糊） |

- **输出**: `image` (IMAGE) — 接缝平移到中心后的图像, `mask` (MASK) — 接缝带掩码, `mask_blurred` (MASK) — 模糊后的掩码

---

### 5. 色彩匹配 (`YuanColorMatch`)
- **分类**: `Yuan Tool/图像`
- **功能**: 将参考图的色彩分布迁移到目标图，基于 `color-matcher` 库实现。支持批量处理与多线程加速，可用于自动色彩校正、画面调色以及光线/色温统一。复刻自 ComfyUI-KJNodes 的 ColorMatch 节点。

| 参数 | 类型 | 说明 |
|------|------|------|
| `image_ref` | IMAGE | 参考图像（提供色彩分布） |
| `image_target` | IMAGE | 目标图像（被改色的图） |
| `method` | 选项 | 匹配方法：`mkl` / `hm` / `reinhard` / `mvgd` / `hm-mvgd-hm` / `hm-mkl-hm` |
| `strength` | FLOAT (可选) | 混合强度，0 输出原图，1 完全匹配 (默认 1.0) |
| `multithread` | BOOL (可选) | 批量大于 1 时启用多线程 (默认 true) |

- **输出**: `image` (IMAGE) — 色彩匹配后的图像
- **备注**: 依赖 `pip install color-matcher`

#### 匹配方法说明
| 方法 | 说明 |
|------|------|
| `mkl` | Monge-Kantorovich 线性化 (MKL) |
| `hm` | 直方图匹配 (HM) |
| `reinhard` | Reinhard 等人的方法 |
| `mvgd` | 多元高斯分布迁移 (MVGD) |
| `hm-mvgd-hm` | HM-MVGD-HM 复合方法 |
| `hm-mkl-hm` | HM-MKL-HM 复合方法 |

---

### 6. Yuan_画布 (`Yuan_Canvas`)
- **分类**: `Yuan Tool/画布`
- **功能**: 自包含的画布合成器（V3），复刻自 ComfyUI-Yuan。接收最多 8 张图像作为独立图层，在内嵌的 fabric.js 编辑器中可视化放置、旋转、缩放，前端合成后的图像回传后端作为单个 IMAGE 输出。配置变化时节点会暂停执行，便于构建合成后继续。

| 参数 | 类型 | 说明 |
|------|------|------|
| `bg_image` | IMAGE | 背景图像 |
| `fabricData` | STRING | fabric.js 画布状态 JSON（前端自动管理，默认 `"{}"`） |
| `imageName` | STRING | 合成结果文件名（默认 `new.png`） |
| `width` | INT | 画布宽度 (默认 512，步长 32) |
| `height` | INT | 画布高度 (默认 512，步长 32) |
| `padding` | INT | 缓冲区大小，用于暂存不想导出的素材 (默认 100) |
| `images` | IMAGE (可选) | 作为独立图层传入的图像批次 |

- **输出**: `image` (IMAGE) — 合成后的图像

---

### 7. 文本处理 (`YUAN_TXTParagraphSplitter`)
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

### 8. 列表编号 (`YUAN_TXTListNumber`)
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

### 9. 格式转换 (`YUAN_TXTConvertAny`)
- **分类**: `Yuan Tool/文本`
- **功能**: 接受任意类型输入，转换为指定的目标类型。

| 参数 | 类型 | 说明 |
|------|------|------|
| `*` | AnyType | 接受任何类型的输入 |
| `格式类型` | 选项 | 目标类型：`string` / `int` / `float` / `boolean` |

- **输出**: `输出` (AnyType) — 转换后的值

---

### 10. Yuan CLIP 时间轴 (`YuanCLIPTimeline`)
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

### 必需依赖 (`requirements.txt`)
- `torch`
- `numpy`
- `opencv-python`
- `Pillow`
- `color-matcher` — 色彩匹配节点
- ComfyUI 核心框架

### 可选依赖
- `av` (PyAV) — 全景预览节点的视频 mp4 预览编码，缺失时仅放弃 mp4 预览，不影响图像预览

```bash
pip install av
```

---

## 鸣谢

本项目部分节点参考/改编自以下开源项目，在此表示感谢：

- [ComfyUI-PromptRelay](https://github.com/kijai/ComfyUI-PromptRelay) — Yuan CLIP 时间轴节点的实现基础
- [ComfyUI-Licon-MSR](https://github.com/liconstudio/ComfyUI-Licon-MSR) — 多帧参考节点的图像预处理与帧分配算法参考
