# MP4 广告候选审核与批量裁剪

这个项目用于批量处理 `input/*.mp4`：

1. 自动扫描每个视频，找出可能的广告时间段。
2. 生成一个带前/中/后三张截图的 Excel 审核表。
3. 人工在 Excel 里确认哪些时间段要删除。
4. 根据审核表批量裁剪视频，输出到 `output/cleaned/`。

## 目录结构

```text
input/                  原始 mp4 视频
ad_templates/           已确认广告模板库，建议提交到 GitHub
output/                 审核表、检测中间产物、最终视频
output/detect/          检测中间产物，运行 build_review.py 时会重建
output/cleaned/         裁剪后的视频
scripts/                脚本
keypoint.txt            可选：本地人工确认广告时间点，不建议提交
```

## 推荐流程

### 1. 准备 Python 环境

建议使用 Python 3.10 或更新版本。

第一次下载项目后，先创建并激活虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell 可以用：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

然后安装依赖：

```bash
pip install -r requirements.txt
```

后续文档里的命令都假设已经激活虚拟环境，所以直接使用 `python`。

### 2. 放入视频

把待处理视频放到：

```text
input/
```

### 3. 一步生成审核表

```bash
python scripts/build_review.py
```

默认会：

- 扫描 `input/*.mp4`
- 使用 `ad_templates/` 中已有模板匹配广告
- 自动发现其它疑似广告候选
- 生成截图
- 输出审核表：

```text
output/ad_review.xlsx
```

### 4. 审核 Excel

打开：

```text
output/ad_review.xlsx
```

在 `Review` 工作表中，每一行是一个候选广告段。

关键列：

| 列名 | 含义 |
| --- | --- |
| `delete` | 是否删除，填 `YES` 或 `NO` |
| `file` | 视频文件名 |
| `start` | 候选开始时间 |
| `end` | 候选结束时间 |
| `score` | 检测分数 |
| `kind` | 候选来源 |
| `review_required` | 是否需要人工确认 |
| `start_frame` / `middle_frame` / `end_frame` | 前/中/后截图 |

默认规则：

- `template_library`、`manual_confirmed`：默认 `delete=YES`
- `auto_discovery`：默认 `delete=NO`，需要人工看截图确认

你只需要把最终要删除的行设置为：

```text
delete = YES
```

不删除的行设置为：

```text
delete = NO
```

### 5. 根据审核表批量裁剪

```bash
python scripts/cut_ads.py
```

默认读取：

```text
input/
output/ad_review.xlsx
```

输出到：

```text
output/cleaned/
```

输出文件名格式：

```text
原文件名.clean.mp4
```

例如：

```text
output/cleaned/test.clean.mp4
```

## 常用命令

### 一步生成审核表

```bash
python scripts/build_review.py
```

### 根据审核表裁剪

```bash
python scripts/cut_ads.py
```

### 使用 keypoint.txt 重新扩充模板库

如果你已经人工确认了某些广告时间点，可以在本地创建 `keypoint.txt`：

```text
7:56-8:12
12:16-
```

然后运行：

```bash
python scripts/build_review.py --use-keypoints
```

这会把 keypoint 中的确认片段保存到：

```text
ad_templates/
```

后续没有 keypoint 的视频也可以用这些模板自动匹配同款广告。

### 精确裁剪模式

默认裁剪模式是 `copy`，速度快，不重编码，但切点可能受关键帧影响有轻微误差。

如果要更精确地按时间点裁剪：

```bash
python scripts/cut_ads.py --mode reencode
```

### 裁剪时多删一点前后边界

例如广告前后各多删 0.3 秒：

```bash
python scripts/cut_ads.py --padding 0.3
```

## Docker 用法

### 一步生成审核表

```bash
docker compose run --rm build_review
```

### 根据审核表裁剪

```bash
docker compose run --rm cut_ads
```

### Docker 精确裁剪

```bash
CUT_ARGS="--mode reencode" docker compose run --rm cut_ads
```

### Docker 使用 keypoint 扩充模板

```bash
BUILD_REVIEW_ARGS="--use-keypoints" docker compose run --rm build_review
```

## 输出文件说明

默认输出结构：

```text
output/ad_review.xlsx
ad_templates/
output/cleaned/
output/detect/
```

含义：

| 路径 | 说明 |
| --- | --- |
| `output/ad_review.xlsx` | 最终人工审核表，裁剪脚本默认读取它 |
| `ad_templates/` | 已确认广告模板库，建议提交到 GitHub 并长期积累 |
| `output/cleaned/` | 裁剪后的视频 |
| `output/detect/` | 生成审核表所需的中间文件，可以重建 |

`output/detect/` 中每个视频默认只保留必要中间产物：

```text
output/detect/<name>.ads.csv
output/detect/<name>.ad_frames/
```

含义：

| 文件 | 说明 |
| --- | --- |
| `.ads.csv` | 所有候选明细，`review_ads.py` 用它生成 Excel |
| `.ad_frames/` | 每个候选段的前/中/后截图，Excel 会嵌入这些图片 |

这些中间产物不是最终结果。每次运行：

```bash
python scripts/build_review.py
```

都会默认清空并重建 `output/detect/`。

如果需要排查检测细节，可以让检测脚本额外输出调试文件：

```bash
python scripts/detect_ads.py --write-debug-files
```

这会额外生成：

```text
output/detect/<name>.ads.txt
output/detect/<name>.candidates.txt
output/detect/<name>.ads.json
output/detect/<name>.ads.keyframes.jpg
```

正常审核和裁剪流程不依赖这些调试文件。

## 核心实现原理

整体流程不是直接“自动剪掉所有可疑片段”，而是采用：

```text
程序生成候选 -> Excel 人工审核 -> 按审核结果批量裁剪
```

这样做的原因是：广告和正片在画面风格上有时很像，完全自动删除容易误伤正片。当前设计里，程序负责尽量提高召回率，最终删除决策放在 `ad_review.xlsx` 中人工确认。

### 1. 视频时间轴与截图

检测和截图都尽量使用 ffmpeg 的播放时间轴。

这样可以避免一个常见问题：某些 mp4 用 OpenCV 按帧号或毫秒 seek 时，拿到的画面会和播放器里同一时间点看到的画面不一致。项目里截图由 `imageio-ffmpeg` 调用 ffmpeg 完成，审核表中的前/中/后三张图更接近播放器实际时间。

每个候选段会截三张图：

```text
start   候选段开始附近
middle  候选段中间
end     候选段结束附近
```

这些截图保存在：

```text
output/detect/<视频名>.ad_frames/
```

然后嵌入到 Excel 中，方便人工判断。

### 2. 视频抽帧与视觉特征

检测阶段会按固定采样率抽帧，默认每秒抽几帧，而不是逐帧处理完整视频。这样速度更快，也足够判断广告段位置。

每个抽样帧会转换成一组轻量视觉特征，主要包括：

- 缩小后的粗略画面布局
- HSV 颜色直方图
- 亮度均值和亮度变化
- 饱和度
- 对比度
- 边缘密度
- 颜色丰富度
- 与上一帧的画面变化

可以把这些特征理解成每一帧的“视觉指纹”。后续模板匹配、场景切分、候选评分都基于这些指纹。

### 3. keypoint 与模板库

`keypoint.txt` 是本地人工确认广告的入口，通常不提交到 GitHub。例如：

```text
7:56-8:12
12:16-
```

含义：

- `7:56-8:12`：确认这一整段是广告。
- `12:16-`：确认从 `12:16` 开始到视频结束是广告。

运行：

```bash
python scripts/build_review.py --use-keypoints
```

脚本会把这些已确认广告片段转换成模板，保存到：

```text
ad_templates/
```

模板库的作用是：后续处理没有 keypoint 的其它视频时，可以直接寻找“和已确认广告长得很像”的片段。

这类命中在 Excel 里通常显示为：

```text
kind = template_library
review_required = no
delete = YES
```

它们是当前系统里最可信的自动结果。

### 4. 模板匹配逻辑

模板匹配的核心步骤：

1. 读取 `ad_templates/` 中的广告模板。
2. 对待处理视频抽帧并生成视觉指纹。
3. 用一个滑动窗口在整段视频上移动。
4. 比较窗口内的视觉指纹和广告模板的相似度。
5. 相似度超过阈值时，认为找到了同款广告。

模板匹配适合处理“同一批视频插入相同广告”的场景。只要模板库积累得足够多，后续检测会越来越稳定。

### 5. 自动发现候选逻辑

模板库只能找见过的广告。为了发现没见过的新广告，脚本还会做启发式自动发现。

自动发现大致分三步：

1. 找场景切换点。
2. 把单个场景或相邻多个短场景组合成候选片段。
3. 给每个候选片段打分。

候选评分会考虑：

- 片段时长是否像广告
- 是否靠近片头或片尾
- 开始/结束处是否有明显场景突变
- 画面是否高饱和、高对比
- 边缘是否密集
- 颜色是否丰富
- 是否像静态广告卡
- 是否由多个快速切换镜头组成

这类结果在 Excel 中通常显示为：

```text
kind = auto_discovery
review_required = yes
delete = NO
```

也就是说，自动发现只负责“提出怀疑”，默认不会直接删除。你需要看截图后，把确实是广告的行改成 `delete=YES`。

### 6. Excel 审核表生成逻辑

`scripts/review_ads.py` 会读取：

```text
output/detect/<视频名>.ads.csv
output/detect/<视频名>.ad_frames/
```

然后生成：

```text
output/ad_review.xlsx
```

Excel 中一行对应一个候选广告段。默认删除决策如下：

| 来源 | 默认 delete | 原因 |
| --- | --- | --- |
| `manual_confirmed` | `YES` | 来自人工 keypoint |
| `manual_open_to_end` | `YES` | 来自人工 keypoint |
| `template_library` | `YES` | 来自已确认模板库 |
| `auto_discovery` | `NO` | 启发式候选，可能误报 |

最终裁剪只看 Excel 里的 `delete` 列。你可以手动把任何行改成 `YES` 或 `NO`。

### 7. 裁剪逻辑

`scripts/cut_ads.py` 默认读取：

```text
output/ad_review.xlsx
```

它只处理 `delete=YES` 的行。

对每个视频，脚本会：

1. 收集该视频所有需要删除的时间段。
2. 合并重叠或相邻的删除段。
3. 反推出需要保留的时间段。
4. 用 ffmpeg 切出保留段。
5. 把保留段 concat 成一个新 mp4。

默认模式是：

```text
copy
```

优点是速度快，不重编码。缺点是切点可能受关键帧影响，有轻微误差。

如果要更精确的秒级切点，可以使用：

```bash
python scripts/cut_ads.py --mode reencode
```

重编码会慢一些，但切点更准。

### 8. 为什么保留 output/detect/

`output/detect/` 不是最终结果，但它是生成 Excel 的中间输入。

默认只保留两类必要文件：

```text
output/detect/<视频名>.ads.csv
output/detect/<视频名>.ad_frames/
```

其中：

- `.ads.csv` 提供候选段的结构化信息。
- `.ad_frames/` 提供 Excel 里嵌入的截图。

每次运行：

```bash
python scripts/build_review.py
```

都会默认清空并重建 `output/detect/`，避免旧视频的候选混入新审核表。

### 9. 关键设计取舍

当前系统的核心取舍是：

- 模板匹配负责高可信自动删除。
- 自动发现负责召回未知广告。
- Excel 审核负责最终决策。
- 裁剪脚本只相信 Excel 的 `delete=YES`。

这能避免把“可疑但不确定”的片段直接剪掉，也方便你不断积累广告模板库，让后续批量处理越来越省事。

## 检测逻辑简述

快速理解：

- `template_library`：命中已确认广告模板，默认删除。
- `manual_confirmed` / `manual_open_to_end`：来自 `keypoint.txt`，默认删除。
- `auto_discovery`：程序自动发现的疑似广告，默认不删除，需要人工看截图确认。

最终是否裁剪，只看 `output/ad_review.xlsx` 里的 `delete` 列。

## 最短工作流

```bash
# 1. 生成审核表
python scripts/build_review.py

# 2. 打开 output/ad_review.xlsx，把要删的行设为 YES

# 3. 批量裁剪
python scripts/cut_ads.py
```
