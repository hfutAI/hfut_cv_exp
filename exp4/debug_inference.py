#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试推理脚本：分析自定义模型为什么检测不到自行车
"""

import torch
import numpy as np
from PIL import Image
import exp4_backend2

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = exp4_backend2.CustomDetector(
    num_classes=exp4_backend2.NUM_CLASSES,
    num_anchors=exp4_backend2.NUM_ANCHORS
).to(device)

# 加载权重
state_dict = torch.load(exp4_backend2.MODEL_SAVE_PATH, map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

print("模型加载成功")

# 加载测试图像
image_path = exp4_backend2.DEFAULT_IMAGE
img_orig = Image.open(image_path).convert('RGB')
orig_w, orig_h = img_orig.size
print(f"原始图像尺寸: {orig_w} x {orig_h}")

# 预处理
input_size = 416
img = img_orig.resize((input_size, input_size))
img_array = np.array(img).astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

print(f"输入张量形状: {img_tensor.shape}")

# 推理
with torch.no_grad():
    predictions = model(img_tensor)

print(f"预测输出形状: {predictions.shape}")

# 解析预测
pred_xy = torch.sigmoid(predictions[..., 0:2])
pred_wh = predictions[..., 2:4]
pred_conf = torch.sigmoid(predictions[..., 4:5])
pred_cls = torch.sigmoid(predictions[..., 5:])

print(f"\n=== 预测统计 ===")
print(f"置信度 (conf) 范围: [{pred_conf.min():.4f}, {pred_conf.max():.4f}]")
print(f"置信度均值: {pred_conf.mean():.4f}")
print(f"置信度 > 0.01 的数量: {(pred_conf > 0.01).sum().item()}")
print(f"置信度 > 0.05 的数量: {(pred_conf > 0.05).sum().item()}")
print(f"置信度 > 0.1 的数量: {(pred_conf > 0.1).sum().item()}")
print(f"置信度 > 0.25 的数量: {(pred_conf > 0.25).sum().item()}")

# 查看各个类别的预测分数
print(f"\n=== 类别预测分析 ===")
B, H, W, A, num_cls = pred_cls.shape
pred_cls_flat = pred_cls.view(-1, num_cls)

for cls_id in range(min(10, num_cls)):  # 只看前10个类别
    max_score = pred_cls_flat[:, cls_id].max().item()
    mean_score = pred_cls_flat[:, cls_id].mean().item()
    print(f"类别 {cls_id}: max={max_score:.4f}, mean={mean_score:.4f}")

# 特别关注bicycle类别（ID=2）
bicycle_cls_id = 2
bicycle_scores = pred_cls_flat[:, bicycle_cls_id]
print(f"\n=== Bicycle (类别{bicycle_cls_id}) 分析 ===")
print(f"Bicycle类别分数范围: [{bicycle_scores.min():.4f}, {bicycle_scores.max():.4f}]")
print(f"Bicycle类别分数均值: {bicycle_scores.mean():.4f}")
print(f"Bicycle分数 > 0.1 的数量: {(bicycle_scores > 0.1).sum().item()}")
print(f"Bicycle分数 > 0.5 的数量: {(bicycle_scores > 0.5).sum().item()}")

# 查找最高置信度的预测
print(f"\n=== Top 10 高置信度预测 ===")
conf_flat = pred_conf.view(-1)
top_conf_values, top_conf_indices = torch.topk(conf_flat, min(10, conf_flat.numel()))

for i, (idx, conf_val) in enumerate(zip(top_conf_indices, top_conf_values)):
    idx = idx.item()
    # 计算在原始shape中的位置
    total_per_anchor = H * W * A
    b = 0
    pos = idx
    a = pos // (H * W)
    pos = pos % (H * W)
    h = pos // W
    w = pos % W
    
    cls_scores = pred_cls[b, h, w, a]
    cls_id = torch.argmax(cls_scores).item()
    cls_score = cls_scores[cls_id].item()
    
    print(f"#{i+1}: conf={conf_val:.4f}, 位置=({h},{w},{a}), 类别={cls_id}, 类别分数={cls_score:.4f}")
    if cls_id == bicycle_cls_id:
        print(f"     ↑ 这是自行车！")

# 尝试用更低的阈值检测
print(f"\n=== 使用不同阈值的检测结果 ===")
for thresh in [0.01, 0.05, 0.1, 0.15, 0.25]:
    count = 0
    bicycle_count = 0
    
    for h in range(H):
        for w in range(W):
            for a in range(A):
                conf = pred_conf[0, h, w, a, 0].item()
                
                if conf < thresh:
                    continue
                
                cls_scores = pred_cls[0, h, w, a]
                cls_id = torch.argmax(cls_scores).item()
                cls_score = cls_scores[cls_id].item()
                
                final_score = conf * cls_score
                
                if final_score >= thresh:
                    count += 1
                    if cls_id == bicycle_cls_id:
                        bicycle_count += 1
    
    print(f"阈值={thresh:.2f}: 总检测数={count}, 自行车数={bicycle_count}")

print("\n完成！")

