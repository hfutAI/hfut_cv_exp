# 合肥工业大学《机器视觉》课程实验

学号：2023217534  
姓名：林圳  
班级：智科23-1班

## 目录结构
- exp1：图像滤波与特征提取（Sobel、直方图、GLCM）
- exp2：车道线检测（颜色过滤 + Canny + ROI + 霍夫变换）
- exp3：学号手写数字识别（连通域分割 + ViT / 自定义 CNN）
- exp4：共享单车目标检测（Faster R-CNN / 自定义检测器）

## 实验一：图像滤波与特征提取（exp1/exp1_backend.py）
功能要点：
- 手写 2D 卷积 + Sobel 边缘检测
- 自定义卷积核滤波
- 颜色直方图（RGB 三通道 256 bin 手写统计）
- GLCM 纹理特征（对比度、能量、同质性、熵、相关性）

运行示例：
```bash
python exp1/exp1_backend.py --input exp1/dog_white.jpg --output_dir exp1/output
```
输出结果：`exp1/output/` 内包含 Sobel 图、直方图、GLCM 热力图与特征文件等。

## 实验二：车道线检测（exp2/exp2_backend.py）
功能要点：
- HLS 白色车道线过滤
- Canny 边缘提取 + ROI 梯形裁剪
- HoughLinesP 检测 + 斜率过滤

运行示例：
```bash
python exp2/exp2_backend.py
```
默认读取 `exp2/images/image_2.jpg`，并将中间步骤与最终结果保存到 `exp2/output/`。

## 实验三：学号手写数字识别（exp3）
### 方案 A：传统分割 + 预训练 ViT（exp3/exp3_backend.py）
功能要点：
- Otsu 二值化 + 轮廓筛选 + 28x28 归一化
- 使用 HuggingFace 预训练 ViT（MNIST）进行识别

运行示例：
```bash
python exp3/exp3_backend.py --image exp3/images/2023217534.jpg --output_dir exp3/output
```
输出：`predicted_id.txt` 与分割调试图 `exp3/output/digits/`。

### 方案 B：自定义 CNN（exp3/exp3_backend2.py）
功能要点：
- 从头训练 CNN（MNIST）
- 复用同样的分割逻辑进行识别

运行示例：
```bash
python exp3/exp3_backend2.py --image exp3/images/2023217534.jpg --output_dir exp3/output_cnn
```

## 实验四：共享单车目标检测（exp4）
### 方案 A：Faster R-CNN + COCO（exp4/exp4_backend.py）
功能要点：
- 使用 COCO 预训练 Faster R-CNN
- 训练/推理阶段可选，仅保留 bicycle 类别

运行示例（推理）：
```bash
python exp4/exp4_backend.py --infer --image exp4/images/1.jpg --output-dir exp4/output
```

### 方案 B：自定义检测模型（exp4/exp4_backend2.py）
功能要点：
- 自定义 Backbone + 检测头
- 自定义检测损失 + NMS 后处理

运行示例：
```bash
python exp4/exp4_backend2.py --image exp4/images/1.jpg --output_dir exp4/output_custom
```

## 环境依赖
- Python 3.8+
- numpy, opencv-python, pillow, matplotlib
- torch, torchvision, transformers, datasets, tqdm

