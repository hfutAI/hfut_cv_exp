import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 常量定义
MODEL_ID = "farleyknight-org-username/vit-base-mnist"
DEFAULT_IMAGE = "/ssdwork/air/linzhen/rebuttal/exp/exp3/images/2023217534.jpg"
OUTPUT_DIR = "/ssdwork/air/linzhen/rebuttal/exp/exp3/output"
NUM_STUDENT_ID_DIGITS = 10


def segment_digits_contours(image_path: str, output_dir: str) -> List[np.ndarray]:
    """
    基于连通域的数字分割算法（传统计算机视觉方法）
    
    步骤：
    1. 二值化：使用Otsu将白底黑字分离，黑色数字变为255（白色），背景变为0
    2. 轮廓检测：使用findContours
    3. 矩形过滤：计算外接矩形
    4. 排序：按x坐标从左到右排序
    5. 过滤：剔除面积太小的噪点
    6. 切割与归一化：切出数字，缩放至28x28
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录（保存中间结果）
        
    Returns:
        List[np.ndarray]: 分割出的数字图像列表（28x28）
    """
    print(f"\n========== 开始分割学号图像 ==========")
    print(f"图像路径: {image_path}")
    
    # 创建调试输出目录
    debug_dir = Path(output_dir) / "digits"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # ==== Step 1: 读取图像 ====
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 保存原图
    cv2.imwrite(str(debug_dir / "01_original.png"), gray)
    print(f"[Step 1] 图像尺寸: {w}x{h}")
    
    # ==== Step 2: 二值化（Otsu自动阈值(类间方差)） ====
    # 使用 THRESH_BINARY_INV: 黑色数字变为255（白色），白色背景变为0（黑色）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 保存二值化结果
    cv2.imwrite(str(debug_dir / "02_threshold.png"), binary)
    print(f"[Step 2] 二值化完成（Otsu方法）")
    
    # ==== Step 3: 轮廓检测 ====
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[Step 3] 检测到 {len(contours)} 个轮廓")
    
    # ==== Step 4: 计算外接矩形并过滤 ====
    bounding_boxes = []
    
    for contour in contours:
        # 计算外接矩形
        x, y, w_box, h_box = cv2.boundingRect(contour)
        area = w_box * h_box
        
        # 基本过滤条件
        # 1. 面积过滤：排除太小的噪点
        if area < 100:
            continue
        
        # 2. 尺寸过滤：排除太小或太大的区域
        if w_box < 10 or h_box < 20:
            continue
        if w_box > w * 0.5 or h_box > h * 0.5:
            continue
        
        # 3. 宽高比过滤：数字通常是竖长的
        aspect_ratio = w_box / h_box
        if aspect_ratio < 0.1 or aspect_ratio > 1.5:
            continue
        
        bounding_boxes.append({
            'x': x,
            'y': y,
            'w': w_box,
            'h': h_box,
            'area': area,
            'aspect_ratio': aspect_ratio
        })
    
    print(f"[Step 4] 过滤后剩余 {len(bounding_boxes)} 个候选框")
    
    if len(bounding_boxes) == 0:
        print("[错误] 没有找到任何有效的数字区域")
        return []
    
    # ==== Step 5: 按x坐标从左到右排序 ====
    bounding_boxes.sort(key=lambda box: box['x'])
    print(f"[Step 5] 按x坐标排序完成")
    
    # ==== Step 6: 进一步筛选（如果数量过多） ====
    if len(bounding_boxes) > NUM_STUDENT_ID_DIGITS:
        print(f"[Step 6] 检测到 {len(bounding_boxes)} 个框，超过预期的 {NUM_STUDENT_ID_DIGITS} 个")
        
        # 按面积排序，选择最大的NUM_STUDENT_ID_DIGITS个
        bounding_boxes_sorted_by_area = sorted(bounding_boxes, key=lambda box: box['area'], reverse=True)
        bounding_boxes = bounding_boxes_sorted_by_area[:NUM_STUDENT_ID_DIGITS]
        
        # 重新按x坐标排序
        bounding_boxes.sort(key=lambda box: box['x'])
        print(f"[Step 6] 保留面积最大的 {NUM_STUDENT_ID_DIGITS} 个框")
    
    elif len(bounding_boxes) < NUM_STUDENT_ID_DIGITS:
        print(f"[警告] 只检测到 {len(bounding_boxes)} 个数字框，少于预期的 {NUM_STUDENT_ID_DIGITS} 个")
    
    # ==== Step 7: 切割数字并归一化到28x28 ====
    digit_images = []
    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    for idx, box in enumerate(bounding_boxes):
        x, y, w_box, h_box = box['x'], box['y'], box['w'], box['h']
        
        # 在原图上绘制边界框（用于调试）
        cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.putText(annotated, str(idx), (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 从二值图像中切出数字区域
        digit_binary = binary[y:y + h_box, x:x + w_box]
        
        # 添加边距（保持数字居中，避免被截断）
        pad_size = max(w_box, h_box) // 10
        digit_padded = cv2.copyMakeBorder(
            digit_binary, pad_size, pad_size, pad_size, pad_size,
            cv2.BORDER_CONSTANT, value=0
        )
        
        # 调整为正方形（MNIST是28x28正方形）
        digit_h, digit_w = digit_padded.shape
        if digit_h > digit_w:
            # 高度大于宽度，需要在左右加padding
            pad_left = (digit_h - digit_w) // 2
            pad_right = digit_h - digit_w - pad_left
            digit_square = cv2.copyMakeBorder(
                digit_padded, 0, 0, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
        elif digit_w > digit_h:
            # 宽度大于高度，需要在上下加padding
            pad_top = (digit_w - digit_h) // 2
            pad_bottom = digit_w - digit_h - pad_top
            digit_square = cv2.copyMakeBorder(
                digit_padded, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=0
            )
        else:
            digit_square = digit_padded
        
        # 缩放到28x28（对齐MNIST格式）
        digit_resized = cv2.resize(digit_square, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 注意：此时digit_resized是白字黑底（数字为255，背景为0）
        # 需要反转为黑字白底（数字为0，背景为255）以匹配MNIST的输入格式
        digit_final = 255 - digit_resized
        
        # 保存调试图像
        cv2.imwrite(str(debug_dir / f"digit_{idx}.png"), digit_final)
        
        digit_images.append(digit_final)
    
    # 保存标注图像
    cv2.imwrite(str(debug_dir / "03_annotated.png"), annotated)
    
    print(f"[Step 7] 成功提取 {len(digit_images)} 个数字图像")
    print(f"========== 分割完成 ==========\n")
    
    return digit_images


def predict_digits(model: nn.Module, processor: AutoImageProcessor, 
                   digit_images: List[np.ndarray], device: torch.device) -> str:
    """
    使用模型预测数字
    
    Args:
        model: 预训练模型
        processor: 图像处理器
        digit_images: 数字图像列表（黑字白底，28x28）
        device: 计算设备
        
    Returns:
        str: 识别出的学号字符串
    """
    if len(digit_images) == 0:
        print("[错误] 没有数字图像可供识别")
        return ""
    
    model.eval()
    predictions = []
    confidences = []
    
    print(f"========== 开始识别 {len(digit_images)} 个数字 ==========")
    
    with torch.no_grad():
        for idx, digit_img in enumerate(digit_images):
            # 转换为 PIL Image（灰度图）
            digit_pil = Image.fromarray(digit_img).convert('L')
            
            # 转换为 RGB（ViT 需要 3 通道）
            digit_pil = digit_pil.convert('RGB')
            
            # 使用 processor 处理
            inputs = processor(images=digit_pil, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            
            # 推理
            outputs = model(pixel_values)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            pred_label = torch.argmax(probs, dim=-1).item()
            pred_conf = probs[0, pred_label].item()
            
            # 获取top-3预测（用于调试）
            top3_probs, top3_labels = torch.topk(probs[0], 3)
            top3_str = ", ".join([f"{top3_labels[i].item()}({top3_probs[i].item():.3f})" 
                                 for i in range(3)])
            
            predictions.append(pred_label)
            confidences.append(pred_conf)
            
            print(f"  数字{idx}: 预测={pred_label}, 置信度={pred_conf:.3f}, Top3=[{top3_str}]")
    
    # 组合成学号字符串
    student_id = ''.join(map(str, predictions))
    avg_confidence = np.mean(confidences)
    
    print(f"\n========== 识别结果 ==========")
    print(f"学号: {student_id}")
    print(f"平均置信度: {avg_confidence:.3f}")
    print(f"=============================\n")
    
    return student_id


def infer(args):
    """推理：识别学号图片（直接使用预训练模型）"""
    print("\n" + "=" * 70)
    print("实验三：手写数字识别 - 学号识别")
    print("方法：基于连通域的传统计算机视觉分割 + ViT-Base预训练模型")
    print("=" * 70)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载模型
    print(f"\n加载预训练模型: {MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model = model.to(device)
    print("模型加载完成")
    
    # 分割数字（使用连通域方法）
    digit_images = segment_digits_contours(args.image, args.output_dir)
    
    if len(digit_images) == 0:
        print("[错误] 没有分割出任何数字，退出")
        return
    
    # 识别数字
    student_id = predict_digits(model, processor, digit_images, device)
    
    # 保存结果
    output_file = Path(args.output_dir) / "predicted_id.txt"
    with open(output_file, 'w') as f:
        f.write(student_id)
    
    print(f"识别结果已保存到: {output_file}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='实验三：手写数字识别 - 学号识别')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE, help='学号图片路径')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 直接运行推理
    infer(args)


if __name__ == "__main__":
    main()
