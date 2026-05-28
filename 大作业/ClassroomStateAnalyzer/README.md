# 📚 基于多人人脸检测与表情识别的课堂状态分析系统

> **物联网应用软件开发** · 期末大作业  
> 学号：231880098 · 姓名：朱业航

---

## 一、项目简介

本系统是一个**基于多人人脸检测与表情识别的课堂状态分析系统**，能够对输入图片、视频或摄像头画面进行多人人脸检测与表情识别，自动统计表情分布并判断课堂状态，通过 Streamlit 可视化界面展示完整分析结果。

### 系统流程

```
图片/视频/摄像头输入
  → 人脸检测与人脸框定位（MediaPipe + Haar Cascade 双检测器）
  → 人脸区域裁剪与预处理
  → 单个人脸表情识别（CNN3 模型，7 类表情）
  → 多人表情结果汇总
  → 人数统计与表情分布统计
  → 课堂状态判断
  → 结果可视化与记录保存
```

### 示例输出

```
检测人数: 5
表情统计:
  开心(Happy): 2 人 (40.0%)
  中性(Neutral): 2 人 (40.0%)
  悲伤(Sad): 1 人 (20.0%)
  生气(Angry): 0 人 (0.0%)
主要表情: Happy
课堂状态: 课堂状态良好 😊
```

---

## 二、功能特性

| 功能 | 说明 | 状态 |
|------|------|------|
| 📷 **图片上传识别** | 上传多人图片，自动检测+识别+统计 | ✅ 已实现 |
| 🎬 **视频上传分析** | 上传视频，抽帧分析并汇总统计 | ✅ 已实现 |
| 📹 **摄像头实时识别** | 开启摄像头拍照分析 | ✅ 已实现 |
| 🎭 **多人人脸检测** | MediaPipe + Haar Cascade 双检测器，支持多人 | ✅ 已实现 |
| 😀 **7 类表情识别** | Angry/Disgust/Fear/Happy/Sad/Surprise/Neutral | ✅ 已实现 |
| 📊 **表情分布统计** | 各类表情人数、比例、柱状图 | ✅ 已实现 |
| 🎯 **课堂状态判断** | 基于规则自动判断课堂状态 | ✅ 已实现 |
| 📋 **CSV 记录保存** | 每次检测自动保存为 CSV | ✅ 已实现 |
| 🖼️ **结果图片导出** | 导出标注后的检测结果图片 | ✅ 已实现 |
| 👤 **单人识别详情** | 每个人脸的独立识别结果和置信度 | ✅ 已实现 |
| 📈 **历史趋势图** | 历史检测记录的表情变化趋势 | ✅ 已实现 |
| 📤 **CSV 导出** | 导出完整检测记录 CSV | ✅ 已实现 |

---

## 三、系统架构

### 项目目录结构

```
ClassroomStateAnalyzer/
├── app.py                       # Streamlit 主界面入口
├── face_detector.py             # 人脸检测模块（MediaPipe + Haar Cascade）
├── expression_recognizer.py     # 表情识别模块（CNN3 模型）
├── analyzer.py                  # 表情统计与课堂状态分析模块
├── utils.py                     # 工具函数（标注、导出、CSV 等）
├── models/
│   └── cnn3_best_weights.h5    # CNN3 预训练模型权重（来自实验二）
├── assets/
│   └── simsun.ttc              # 中文字体（用于图片标注）
├── test_images/                 # 测试图片目录
├── test_videos/                 # 测试视频目录
├── results/                     # 检测结果图片保存目录
├── records.csv                  # 检测记录 CSV
├── requirements.txt             # Python 依赖
├── README.md                    # 本文件
└── 开发计划.md                  # 开发计划文档
```

### 模块职责

| 模块文件 | 职责 | 关键接口 |
|----------|------|----------|
| `face_detector.py` | 多人人脸检测，支持 MediaPipe 和 Haar Cascade 双检测器 | `detect_faces(image)` → 人脸框列表 |
| `expression_recognizer.py` | 加载 CNN3 模型，对裁剪的人脸进行 7 类表情识别 | `recognizer.recognize_all(image, faces)` → 识别结果列表 |
| `analyzer.py` | 表情统计计算、课堂状态判断、格式化输出 | `calculate_statistics(results)`, `judge_classroom_state(stats)` |
| `utils.py` | 中文标注绘制、结果图片保存、CSV 读写导出 | `draw_results()`, `save_record()`, `load_records()` |
| `app.py` | Streamlit 可视化界面，整合所有模块 | `streamlit run app.py` |

### 数据流

```
用户上传图片/视频/摄像头
        ↓
    app.py（Streamlit 界面）
        ↓
    face_detector.detect_faces()  →  人脸框列表
        ↓
    expression_recognizer.recognize_all()  →  每个人脸的识别结果
        ↓
    analyzer.calculate_statistics()  →  统计信息
        ↓
    analyzer.judge_classroom_state()  →  课堂状态
        ↓
    utils.draw_results()  →  标注后的图片
    utils.save_record()   →  CSV 记录
        ↓
    Streamlit 界面展示（图片 + 统计 + 图表 + 状态）
```

---

## 四、环境搭建与运行

### 4.1 前置要求

- **操作系统**：Windows 10/11（已测试）
- **Conda**：Miniconda 或 Anaconda
- **GPU**：可选（CPU 即可运行，TensorFlow 会自动检测）

### 4.2 创建 Conda 环境

```bash
# 创建 Python 3.11 环境
conda create -n hand python=3.11 -y

# 激活环境
conda activate hand
```

### 4.3 安装依赖

```bash
# 进入项目目录
cd 大作业/ClassroomStateAnalyzer

# 安装所有依赖
pip install -r requirements.txt
```

**依赖清单（`requirements.txt`）**：

```
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=9.0.0
pandas>=1.3.0
tensorflow>=2.8.0,<3.0.0
mediapipe>=0.9.0
streamlit>=1.25.0
matplotlib>=3.5.0
```

> **注意**：TensorFlow 安装可能需要一些时间（约 200MB+）。如果 pip 安装 TensorFlow 较慢，可以尝试使用 conda 安装：
> ```bash
> conda install -c conda-forge tensorflow
> ```

### 4.4 一键启动

```bash
# 确保在 hand 环境中
conda activate hand

# 进入项目目录
cd 大作业/ClassroomStateAnalyzer

# 启动 Streamlit 应用
streamlit run app.py
```

启动后，浏览器会自动打开 `http://localhost:8501`，即可看到系统界面。

### 4.5 常见问题排查

#### ❓ TensorFlow 日志警告

启动时可能看到如下警告，**可以安全忽略**，不影响运行：

```
W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cudart64_101.dll'
```

这表示没有 GPU/CUDA 支持，系统会自动回退到 CPU 模式运行。

#### ❓ MediaPipe 安装失败

如果 `pip install mediapipe` 失败，尝试：

```bash
pip install mediapipe --no-deps
pip install protobuf>=3.20.0
```

#### ❓ 摄像头无法打开

- 确认摄像头设备已连接
- 确认没有其他程序占用摄像头
- 部分虚拟机/远程桌面环境不支持摄像头，可使用图片/视频模式代替

#### ❓ 模型加载失败

确认 `models/cnn3_best_weights.h5` 文件存在且完整。该文件约 45MB，如果从 Git 克隆后缺失，请检查 Git LFS 或重新下载。

---

## 五、使用说明

### 5.1 图片上传模式（基础功能）

1. 在左侧侧边栏选择 **"📷 图片上传"**
2. 点击上传区域，选择一张包含多个人脸的图片（支持 JPG/PNG/BMP）
3. 点击 **"🔍 开始检测"** 按钮
4. 系统将自动完成：人脸检测 → 表情识别 → 统计 → 状态判断
5. 检测完成后，主区域会显示：
   - 原始图片 vs 检测结果图片（带人脸框和表情标签）
   - 检测人数、主要表情、积极/低落占比
   - 表情分布柱状图
   - 课堂状态判断结果（带颜色指示）
   - 每个人脸的详细识别结果

### 5.2 视频上传模式（进阶功能）

1. 在左侧侧边栏选择 **"🎬 视频上传"**
2. 上传一段课堂视频（支持 MP4/AVI/MOV/MKV）
3. 设置采样参数：
   - **采样帧间隔**：每 N 帧采样一次（默认 30，即约每秒 1 帧）
   - **最大采样帧数**：最多分析多少帧（默认 20）
4. 点击 **"🔍 开始视频分析"**
5. 系统将逐帧分析并汇总统计，显示进度条

### 5.3 摄像头实时模式（加分功能）

1. 在左侧侧边栏选择 **"📹 摄像头实时"**
2. 勾选 **"开启摄像头"**
3. 点击 **"📸 拍照并分析"** 捕获当前画面并分析

> 如果摄像头不可用，系统会给出提示。这不影响其他功能的使用。

### 5.4 参数设置

- **置信度阈值**（0.0~1.0，默认 0.3）：低于此阈值的检测结果将被过滤

### 5.5 结果导出

| 按钮 | 功能 | 输出 |
|------|------|------|
| 💾 保存记录 | 将当前检测结果保存到 `records.csv` | CSV 文件 |
| 📊 导出 CSV | 导出所有历史检测记录 | CSV 文件 |
| 🖼️ 导出结果图片 | 保存标注后的检测结果图片 | JPG 图片（保存到 `results/` 目录） |

---

## 六、模块详细说明

### 6.1 `face_detector.py` — 人脸检测模块

**双检测器策略**：

1. **优先使用 MediaPipe Face Detection**：精度高，支持远距离人脸，返回置信度
2. **回退到 OpenCV Haar Cascade**：当 MediaPipe 未检测到人脸时自动启用

**关键函数**：

```python
from face_detector import detect_faces, crop_face, draw_face_boxes

# 检测人脸
faces = detect_faces(image_bgr)  # → [(x1, y1, x2, y2, confidence), ...]

# 裁剪单个人脸
face_img = crop_face(image_bgr, faces[0], margin=10)

# 绘制人脸框
result_img = draw_face_boxes(image_bgr, faces, labels=["Happy"], confidences=[0.95])
```

### 6.2 `expression_recognizer.py` — 表情识别模块

**模型**：CNN3（A Compact Deep Learning Model for Robust Facial Expression Recognition），复用实验二训练的权重。

**7 类表情**：

| 索引 | 英文 | 中文 |
|------|------|------|
| 0 | Angry | 生气 |
| 1 | Disgust | 厌恶 |
| 2 | Fear | 恐惧 |
| 3 | Happy | 开心 |
| 4 | Sad | 悲伤 |
| 5 | Surprise | 惊讶 |
| 6 | Neutral | 中性 |

**增广策略**：对每个人脸使用 6 种变换（裁剪、翻转等），取预测平均值提升准确性。

**关键函数**：

```python
from expression_recognizer import get_recognizer

recognizer = get_recognizer()  # 单例模式，自动加载模型

# 单人识别
label_cn, label_en, confidence, probs = recognizer.predict_single(face_bgr)

# 多人批量识别
results = recognizer.recognize_all(image_bgr, faces)
# → [{'bbox': ..., 'label_cn': '开心', 'label_en': 'Happy', 'confidence': 0.95, 'probs': {...}}, ...]
```

### 6.3 `analyzer.py` — 统计与分析模块

**关键函数**：

```python
from analyzer import calculate_statistics, judge_classroom_state, get_state_color

# 计算统计
stats = calculate_statistics(recognition_results)
# → {'total_count': 5, 'expression_count': {...}, 'expression_ratio': {...}, ...}

# 课堂状态判断
state_text, state_level = judge_classroom_state(stats)
# → ("课堂状态良好 😊", "good")
```

### 6.4 `utils.py` — 工具函数

```python
from utils import draw_results, save_result_image, save_record, load_records, export_csv

# 绘制检测结果（支持中文标注）
result_img = draw_results(image_bgr, recognition_results)

# 保存结果图片
path = save_result_image(result_img, "class1.jpg")

# 保存/读取 CSV 记录
save_record(record_dict)
df = load_records()
```

### 6.5 `app.py` — Streamlit 界面

Streamlit 应用入口，整合所有模块，提供交互式 Web 界面。

**启动命令**：`streamlit run app.py`

---

## 七、课堂状态判断规则

系统根据表情分布统计结果，按以下优先级判断课堂状态：

| 优先级 | 判断条件 | 课堂状态 | 等级 | 颜色 |
|--------|---------|---------|------|------|
| 1 | 检测人数 = 0 | 未检测到学生 | empty | 灰色 |
| 2 | Happy + Neutral 占比 ≥ 70% | 课堂状态良好 😊 | good | 绿色 |
| 3 | Sad + Angry 占比 ≥ 40% | 课堂状态较低落，需要关注 ⚠️ | low | 红色 |
| 4 | Neutral 占比最高 | 课堂状态平稳 😐 | stable | 蓝色 |
| 5 | Surprise 占比 ≥ 30% | 课堂注意力波动较大 😮 | attention | 橙色 |
| 6 | 其他情况 | 课堂状态一般 | normal | 蓝灰色 |

---

## 八、技术选型说明

| 模块 | 技术方案 | 选型理由 |
|------|---------|---------|
| 人脸检测 | **MediaPipe Face Detection**（优先） + **OpenCV Haar Cascade**（回退） | MediaPipe 精度高、速度快；Haar Cascade 作为兜底保证鲁棒性 |
| 表情识别 | **CNN3** + FER2013 预训练权重 | 复用实验二已验证的模型，7 类表情全覆盖，无需重新训练 |
| 可视化界面 | **Streamlit** | 大作业推荐方案，开发效率高，内置文件上传、图表、布局组件 |
| 数据记录 | **CSV**（Pandas 读写） | 实现简单，符合要求，Excel 可直接打开 |
| 中文标注 | **PIL + SimSun 字体** | OpenCV 不支持中文，通过 PIL 绘制中文文本后转回 |

---

## 九、已验证的运行环境

| 项目 | 版本 |
|------|------|
| 操作系统 | Windows 11 |
| Conda 环境 | `hand` |
| Python | 3.11.15 |
| TensorFlow | 2.15.1 |
| OpenCV | 4.11.0 |
| MediaPipe | 0.10.14 |
| Streamlit | 1.57.0 |
| Pandas | 3.0.3 |
| NumPy | 1.26.4 |

---

## 十、已知限制与改进方向

1. **表情识别准确率**：FER2013 数据集训练的模型对真实课堂场景的适配性有限，可考虑使用更大规模的数据集微调
2. **人脸遮挡**：严重遮挡的人脸可能导致检测失败或识别错误
3. **侧脸检测**：MediaPipe 对侧脸检测效果较弱，Haar Cascade 回退可能提升覆盖率
4. **实时性能**：CPU 模式下视频分析速度较慢，GPU 可显著加速
5. **多人重叠**：人群密集时人脸重叠可能导致漏检
6. **延展任务**：时序分析（滑动窗口、三级预警）尚未实现，可作为后续扩展

---

## 十一、许可证

本项目仅用于课程作业，参考了以下开源项目：

- **实验二**：FacialExpressionRecognition（CNN3 模型结构和训练权重）
- **MediaPipe**：Google 开源的人脸检测框架
- **FER2013**：面部表情识别公开数据集

---

**© 2026 物联网应用软件开发 · 期末大作业**