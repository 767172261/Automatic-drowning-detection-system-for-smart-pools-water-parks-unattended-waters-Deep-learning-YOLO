智能泳池、水上乐园、无人值守水域的自动溺水检测系统开发 深度学习YOLOV8模型训练游泳溺水检测数据集 识别溺水人员及游泳人员
@[toc]
### **溺水+游泳目标检测数据集信息表**

| 项目 | 内容 |
|------|------|
| **数据集名称** | 溺水+游泳目标检测数据集 |
| **总图像数量** | 9,000 张（含数据增强处理） |
| **目标类别数** | 2 类 |
| **检测任务** | 溺水行为识别、游泳行为检测 |
| **标注格式** | YOLO 格式（`.txt`），每图对应一个标注文件 |
| **适用模型** | YOLOv5、YOLOv8、YOLOv11 等 YOLO 系列目标检测模型 |
| **使用场景** | 智能泳池监控、溺水预警系统、安防监控、行为识别研究 |

---

### **类别标签对照表**

| 类别编号 | 英文标签 | 中文名称 | 说明 |
|----------|-----------|------------|--------------------------------------------------|
| 0        | swim      | 游泳       | 正常游泳姿态，动作规律，头部露出水面 |
| 1        | drown     | 溺水       | 异常行为，如挣扎、下沉、肢体不协调等特征 |

---

### **数据集特点总结**

| 特点 | 说明 |
|------|------|
| ✅ **行为级分类** | 区分“正常游泳”与“溺水”两种关键行为，支持安全预警 |
| ✅ **含数据增强** | 9,000 张图像已包含旋转、翻转、亮度变化等增强处理，提升模型泛化能力 |
| ✅ **YOLO格式直用** | 标注文件为标准 YOLO 归一化坐标格式，可直接用于训练 |
| ✅ **真实场景覆盖** | 包含不同光照、角度、水面反光、多人场景等复杂环境 |
| ✅ **高实用性** | 可用于智能泳池、水上乐园、无人值守水域的自动溺水检测系统开发 |

---

### **典型应用方向**

- ✅ 实时溺水预警系统（结合摄像头+AI）
- ✅ 智能安防监控平台
- ✅ 儿童泳池安全监护
- ✅ 计算机视觉行为识别科研项目


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/36a75e248604478a97460764acd3671e.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/61ae09bd681944b48a95ee0cebb370be.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/52194373fca74102a07748b240a82678.png)

于构建智能泳池监控、溺水预警系统、无人值守泳场等 AI 应用。

以下是 **从零开始搭建环境 → 数据准备 → 使用 YOLOv8 训练 → 推理 → 评估 → 部署** 的完整全流程，确保你可以高效训练一个高精度的 **溺水行为识别模型**。代码示例，仅供参考学习使用。

---

## ✅ 一、系统环境搭建（CUDA + Anaconda + Python）

### 1. 确认 CUDA 驱动（GPU 加速）

```bash
nvidia-smi
```

- 要求：NVIDIA GPU（建议 ≥ RTX 3060 / 4070）
- CUDA 版本 ≥ 11.8（推荐 12.1）

> 如未安装驱动，请前往 [NVIDIA 官网](https://www.nvidia.com/Download/index.aspx) 下载。

---

### 2. 安装 Anaconda（Python 包管理器）

前往 [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) 下载并安装。

---

### 3. 创建 Python 虚拟环境

```bash
# 创建名为 drowning_detection 的环境
conda create -n drowning_detection python=3.9

# 激活环境
conda activate drowning_detection
```

---

### 4. 安装必要依赖

```bash
# 安装 PyTorch（以 CUDA 11.8 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 YOLOv8 官方库
pip install ultralytics opencv-python numpy matplotlib tqdm scikit-learn pandas pillow

# 可选：TensorBoard 可视化
pip install tensorboard

# 验证 GPU 是否可用
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

---

## ✅ 二、数据集结构与 `data.yaml` 配置

### 目录结构

```
drowning_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

> ✅ 数据已增强处理，共 9000 张图像，YOLO 格式（`.txt`），每张图对应一个标注文件。

---

### `data.yaml` 文件内容

```yaml
# data.yaml
train: ./drowning_dataset/images/train
val: ./drowning_dataset/images/val
test: ./drowning_dataset/images/test

# 类别数量
nc: 2

# 类别名称（英文）
names:
  - swimming    # 正常游泳
  - drowning    # 溺水

# 中文名称（用于可视化）
names_zh:
  - 游泳
  - 溺水
```

> ⚠️ 注意：
> - `swimming` 类 ID = 0
> - `drowning` 类 ID = 1
> - 所有 `.txt` 文件中的 class ID 必须为 0 或 1
> - 建议按 8:1:1 划分 train/val/test

---

## ✅ 三、调用 YOLOv8 官方预训练模型进行训练

由于是 **行为识别类任务**，且“溺水”目标可能表现为静止、姿态异常、头部下沉等特征，建议使用 **YOLOv8m** 或 **YOLOv8l** 提升小目标和姿态变化的检测能力。

```python
from ultralytics import YOLO

# 加载官方预训练模型（推荐使用中等或大型模型）
model = YOLO('yolov8m.pt')  # 也可使用 yolov8s.pt（轻量级）

# 开始训练
results = model.train(
    data='data.yaml',           # 数据配置文件路径
    epochs=150,                 # 溺水数据建议 100~200 轮
    batch=32,                   # 根据显存调整（RTX 3090 可用 64）
    imgsz=640,                  # 图像尺寸（提升小目标或遮挡检测）
    optimizer='AdamW',          # 更稳定优化器
    lr0=0.001,                  # 初始学习率
    weight_decay=0.0005,
    momentum=0.937,

    # 数据增强（关键！提升泛化）
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,

    # 正则化
    dropout=0.2,
    label_smoothing=0.05,       # 缓解过拟合

    # 学习率调度
    cos_lr=True,                # 余弦退火

    # 保存与日志
    project='runs/train',
    name='drowning_yolov8m',
    save=True,
    save_period=10,
    exist_ok=False,

    # 缓存（内存充足时开启）
    cache=True
)
```

> ✅ **提示**：
> - 若显存不足，改用 `yolov8s.pt` 或降低 `batch=16`
> - `mosaic` 和 `hsv_v` 对光照变化、水面反光特别有效
> - 可开启 `close_mosaic` 在最后 10 轮关闭 Mosaic 增强

---

## ✅ 四、推理代码（单图 / 批量 / 视频）

### 1. 单张图像推理

```python
from ultralytics import YOLO
from PIL import Image

# 加载最佳模型
model = YOLO('runs/train/drowning_yolov8m/weights/best.pt')

# 推理
results = model('pool_scene.jpg', conf=0.3)

# 显示结果
for r in results:
    im_array = r.plot()  # 绘制边界框和标签
    im = Image.fromarray(im_array[..., ::-1])  # BGR to RGB
    im.show()
```

---

### 2. 批量图像推理

```python
results = model.predict(
    source='drowning_dataset/images/test',
    save=True,
    project='runs/detect',
    name='drowning_test_results',
    conf=0.3,
    imgsz=640
)
print("批量推理完成")
```

---

### 3. 视频检测（泳池监控视频）

```python
results = model.predict(
    source='swimming_pool_surveillance.mp4',
    save=True,
    project='runs/detect/video',
    name='drowning_detection_video',
    conf=0.3,
    imgsz=640
)
print("视频检测完成")
```

---

### 4. 实时摄像头检测（OpenCV）

```python
import cv2

cap = cv2.VideoCapture(0)  # 或 RTSP 流

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3)
    annotated_frame = results[0].plot()

    # 添加溺水报警逻辑
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 1:  # drowning
                cv2.putText(annotated_frame, "ALERT: DROWNING!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow('Drowning Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ✅ 五、模型评估（验证集 + 测试集）

### 1. 在验证集上评估

```python
metrics = model.val(
    data='data.yaml',
    split='val',
    batch=32,
    imgsz=640,
    save_json=False,
    project='runs/val',
    name='drowning_eval'
)

print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")
print(f"F1-score: {metrics.box.f1.mean():.4f}")
```

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9e70c06a7d0c4fb39a52bb91e78057f8.png)

### 2. 在测试集上独立评估（最终性能）

```python
test_metrics = model.val(
    data='data.yaml',
    split='test',
    save_json=True,
    name='test_final'
)

print(f"Test mAP@0.5: {test_metrics.box.map50:.4f}")
print(f"Test mAP@0.5:0.95: {test_metrics.box.map:.4f}")
```

---

## ✅ 六、可视化与分析

### 1. 使用中文标签显示

```python
# 修改模型类名显示为中文
model.names = ['游泳', '溺水']

results = model('pool_scene.jpg')
results[0].plot()  # 显示中文标签
```

### 2. 查看训练曲线

```bash
tensorboard --logdir runs/train/drowning_yolov8m
```

---

## ✅ 七、模型导出（ONNX / TensorRT / TFLite）

```python
# 导出为 ONNX（通用部署）
model.export(format='onnx', dynamic=True, opset=13, imgsz=640)

# 导出为 TensorRT（NVIDIA GPU 加速）
model.export(format='engine', half=True, dynamic=True)

# 导出为 TFLite（移动端）
model.export(format='tflite', int8=True)

# 导出为 CoreML（iOS）
model.export(format='coreml')
```

---


