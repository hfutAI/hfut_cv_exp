#!/usr/bin/env python3
"""
实验三扩展：手写数字识别 - 自定义CNN模型
功能：
1. 搭建自定义CNN模型（不使用预训练模型）
2. 在MNIST数据集上从头训练
3. 使用训练好的模型识别学号
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# 尝试从 exp3_backend 导入分割函数
# 如果直接运行此脚本，且 exp3_backend.py 在同级目录，则可以直接导入
try:
    from exp3_backend import segment_digits_contours
except ImportError:
    # 如果导入失败，可能需要调整 sys.path 或复制函数
    # 这里假设它们在同一目录
    import sys
    sys.path.append(str(Path(__file__).parent))
    from exp3_backend import segment_digits_contours

# 常量定义
MODEL_SAVE_PATH = "/ssdwork/air/linzhen/rebuttal/exp/exp3/mnist_cnn.pth"
DATA_DIR = "/ssdwork/air/linzhen/rebuttal/exp/exp3/data"
DEFAULT_IMAGE = "/ssdwork/air/linzhen/rebuttal/exp/exp3/images/2023217534.jpg"
OUTPUT_DIR = "/ssdwork/air/linzhen/rebuttal/exp/exp3/output_cnn"

# ==== 1. 自定义 CNN 模型 ====
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层 1: 输入 1通道(灰度), 输出 32通道, 核大小 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 卷积层 2: 输入 32通道, 输出 64通道, 核大小 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout 防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层
        # 经过两次池化 (28->14->7), 64通道: 64 * 7 * 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10个数字分类

    def forward(self, x):
        # Layer 1: Conv1 -> ReLU -> Pool
        x = F.relu(self.conv1(x)) # [B, 32, 28, 28]
        x = self.pool(x)          # [B, 32, 14, 14]
        
        # Layer 2: Conv2 -> ReLU -> Pool
        x = F.relu(self.conv2(x)) # [B, 64, 14, 14]
        x = self.pool(x)          # [B, 64, 7, 7]
        
        x = self.dropout1(x)
        x = torch.flatten(x, 1)   # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# ==== 2. 训练代码 ====
def train_model(epochs=5, batch_size=64, learning_rate=0.001):
    print(f"\n========== 开始训练自定义 CNN 模型 ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 均值和标准差
    ])
    
    # 下载/加载数据集
    print("加载 MNIST 数据集...")
    os.makedirs(DATA_DIR, exist_ok=True)
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 初始化模型
    model = SimpleCNN().to(device)
    
    # 使用 DataParallel 利用多卡 (如果有)
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张 GPU 进行训练")
        model = nn.DataParallel(model)
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
        
        # 测试循环
        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        acc = 100. * test_correct / len(test_loader.dataset)
        print(f"Epoch {epoch} 完成. 测试集 Loss: {test_loss:.4f}, Accuracy: {acc:.2f}%")
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
            else:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"新最佳模型已保存! Accuracy: {best_acc:.2f}%")
            
    print(f"训练结束. 最佳 Accuracy: {best_acc:.2f}%")
    print(f"最终模型已保存至: {MODEL_SAVE_PATH}")
    print("========== 训练完成 ==========\n")

# ==== 3. 推理代码 ====
def predict_digits_custom(model: nn.Module, digit_images: list, device: torch.device) -> str:
    model.eval()
    predictions = []
    confidences = []
    
    # MNIST 归一化参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print(f"========== 开始使用 CNN 识别 {len(digit_images)} 个数字 ==========")
    
    with torch.no_grad():
        for idx, digit_img in enumerate(digit_images):
            # digit_img 来自 exp3_backend，是黑字白底 (0为黑, 255为白)
            # MNIST 训练数据通常是白字黑底
            # 所以我们需要反转颜色：255 - digit_img
            digit_inverted = 255 - digit_img
            
            # 转为 PIL Image 以应用 transforms
            # 注意：digit_inverted 是 numpy array (uint8)
            # transforms.ToTensor() 会将 [0, 255] -> [0.0, 1.0]
            digit_tensor = transform(digit_inverted) # [1, 28, 28]
            
            # 增加 batch 维度 [1, 1, 28, 28]
            digit_tensor = digit_tensor.unsqueeze(0).to(device)
            
            output = model(digit_tensor)
            probs = F.softmax(output, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            pred_conf = probs[0, pred_label].item()
            top3_probs, top3_labels = torch.topk(probs[0], 3)
            
            predictions.append(pred_label)
            confidences.append(pred_conf)
            top3_str = ", ".join([f"{top3_labels[i].item()}({top3_probs[i].item():.3f})" for i in range(3)])
            print(f"  数字{idx}: 预测={pred_label}, 置信度={pred_conf:.3f}, Top3=[{top3_str}]")
            
    student_id = ''.join(map(str, predictions))
    if confidences:
        avg_conf = float(np.mean(confidences))
        print(f"平均置信度: {avg_conf:.3f}")
    print(f"\n========== 识别结果 ==========")
    print(f"学号: {student_id}")
    print(f"=============================\n")
    return student_id

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"未找到预训练模型文件: {MODEL_SAVE_PATH}")
        print("将自动开始训练...")
        train_model()
    
    # 加载模型
    print(f"加载模型: {MODEL_SAVE_PATH}")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    
    # 分割数字 (复用 exp3_backend 的逻辑)
    digit_images = segment_digits_contours(args.image, args.output_dir)
    
    if not digit_images:
        print("未检测到数字，退出")
        return

    # 识别
    student_id = predict_digits_custom(model, digit_images, device)
    
    # 保存结果
    output_file = Path(args.output_dir) / "predicted_id_cnn.txt"
    with open(output_file, 'w') as f:
        f.write(student_id)
    print(f"识别结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='实验三扩展：自定义CNN手写数字识别')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE, help='学号图片路径')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--train', action='store_true', help='强制重新训练模型')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.train:
        train_model(epochs=args.epochs)
    
    # 如果指定了 --train，训练完后接着做推理
    # 如果没指定 --train，直接推理（内部会检查模型是否存在，不存在则训练）
    infer(args)

if __name__ == "__main__":
    main()

