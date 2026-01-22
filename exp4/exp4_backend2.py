import os
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

import os

# 强制在程序内部设置环境变量，确保所有组件使用 SSD 分区
os.environ["HF_HOME"] = "/ssdwork/air/linzhen/hf_home"
os.environ["HF_DATASETS_CACHE"] = "/ssdwork/air/linzhen/hf_home/datasets"
os.environ["TMPDIR"] = "/ssdwork/air/linzhen/tmp"

# 确保这些文件夹已经存在
os.makedirs("/ssdwork/air/linzhen/hf_home", exist_ok=True)
os.makedirs("/ssdwork/air/linzhen/tmp", exist_ok=True)


# 常量定义
MODEL_SAVE_PATH = "/ssdwork/air/linzhen/rebuttal/exp/exp4/bicycle_detector.pth"
DEFAULT_IMAGE = "/ssdwork/air/linzhen/rebuttal/exp/exp4/images/1.jpg"
OUTPUT_DIR = "/ssdwork/air/linzhen/rebuttal/exp/exp4/output_custom"

# COCO 类别 (我们只关注 bicycle；COCO官方ID=2，也有数据可能用顺序ID=1)
BICYCLE_CATEGORY_IDS = {1, 2}
NUM_CLASSES = 2  # 二分类：0=背景, 1=bicycle
GRID_SIZE = 13  # 将图像分成 13x13 的网格
NUM_ANCHORS = 3  # 每个网格预测3个框
FILTER_BICYCLE_ONLY = True   # 默认只使用包含bicycle的样本
MAX_SCAN_FOR_FILTER = 50000  # 最多扫描这么多样本以收集含bicycle的图像

# ==== 1. 自定义 Backbone 网络 ====
class SimpleBackbone(nn.Module):
    """
    简单的卷积特征提取网络
    输入: (B, 3, H, W)
    输出: (B, 512, H/32, W/32) 的特征图
    """
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        
        # Layer 1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Layer 4: 128 -> 256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Layer 5: 256 -> 512
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        # Input: (B, 3, H, W)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 32, H/2, W/2)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, H/4, W/4)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 128, H/8, W/8)
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 256, H/16, W/16)
        x = F.relu(self.bn5(self.conv5(x)))  # (B, 512, H/32, W/32)
        return x

# ==== 2. 检测头 ====
class DetectionHead(nn.Module):
    """
    检测头：预测每个网格的边界框和类别
    输入: (B, 512, H, W) 特征图
    输出: (B, H, W, NUM_ANCHORS * (5 + NUM_CLASSES))
          其中 5 = (x, y, w, h, objectness)
    """
    def __init__(self, in_channels=512, num_anchors=3, num_classes=91):
        super(DetectionHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # 检测层
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        
        # 输出层：每个 anchor 预测 (x, y, w, h, objectness, class_probs)
        self.pred = nn.Conv2d(256, num_anchors * (5 + num_classes), kernel_size=1)
        
    def forward(self, x):
        # x: (B, 512, H, W)
        x = F.relu(self.bn(self.conv(x)))  # (B, 256, H, W)
        x = self.pred(x)  # (B, num_anchors * (5 + num_classes), H, W)
        
        # 重排维度: (B, C, H, W) -> (B, H, W, C)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        
        # 重塑为 (B, H, W, num_anchors, 5+num_classes)
        x = x.view(B, H, W, self.num_anchors, 5 + self.num_classes)
        
        return x

# ==== 3. 完整的检测模型 ====
class CustomDetector(nn.Module):
    """
    自定义单阶段目标检测模型
    """
    def __init__(self, num_classes=91, num_anchors=3):
        super(CustomDetector, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.backbone = SimpleBackbone()
        self.detection_head = DetectionHead(
            in_channels=512,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        
        # Anchor boxes (预定义的宽高比)
        # 格式: (width_ratio, height_ratio) 相对于网格大小
        self.register_buffer('anchors', torch.tensor([
            [0.5, 0.5],   # 小目标
            [1.0, 1.0],   # 中等目标
            [1.5, 1.5],   # 大目标
        ]))
        
    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.backbone(x)  # (B, 512, H', W')
        predictions = self.detection_head(features)  # (B, H', W', num_anchors, 5+num_classes)
        return predictions

# ==== 4. 损失函数 ====
class DetectionLoss(nn.Module):
    """
    目标检测损失函数
    包括：
    1. 定位损失 (bbox regression)
    2. 置信度损失 (objectness)
    3. 分类损失 (class prediction)
    """
    def __init__(self, num_classes=91, lambda_coord=5.0, lambda_noobj=0.5):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord  # 坐标损失权重
        self.lambda_noobj = lambda_noobj  # 无目标置信度损失权重
        
    def forward(self, predictions, targets):
        """
        predictions: (B, H, W, num_anchors, 5+num_classes)
        targets: list of dicts, 每个 dict 包含 'boxes', 'labels'
        """
        B, H, W, num_anchors, _ = predictions.shape
        device = predictions.device
        
        # 解析预测
        pred_xy = torch.sigmoid(predictions[..., 0:2])  # (B, H, W, A, 2) 中心坐标
        pred_wh = predictions[..., 2:4]  # (B, H, W, A, 2) 宽高
        pred_conf = torch.sigmoid(predictions[..., 4:5])  # (B, H, W, A, 1) 置信度
        pred_cls = predictions[..., 5:]  # (B, H, W, A, num_classes) 类别
        
        # 初始化损失
        loss_xy = torch.tensor(0.0, device=device)
        loss_wh = torch.tensor(0.0, device=device)
        loss_conf = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)
        
        # 为每个样本计算损失
        num_pos = 0
        for b in range(B):
            if len(targets[b]['boxes']) == 0:
                # 无目标：只计算置信度损失
                loss_conf += F.binary_cross_entropy(
                    pred_conf[b].view(-1),
                    torch.zeros_like(pred_conf[b].view(-1)),
                    reduction='sum'
                )
                continue
            
            # 获取目标框 (归一化到0-1)
            boxes = targets[b]['boxes']  # (N, 4) [x1, y1, x2, y2]
            labels = targets[b]['labels']  # (N,)
            
            # 转换为中心坐标格式 (cx, cy, w, h)
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            
            # 分配到网格
            for n in range(len(boxes)):
                # 计算所属网格
                grid_x = int(cx[n] * W)
                grid_y = int(cy[n] * H)
                grid_x = min(grid_x, W - 1)
                grid_y = min(grid_y, H - 1)
                
                # 选择最佳 anchor (这里简化为使用第一个)
                anchor_idx = 0
                
                # 目标相对于网格的偏移
                tx = cx[n] * W - grid_x
                ty = cy[n] * H - grid_y
                tw = w[n] * W
                th = h[n] * H
                
                # 定位损失
                loss_xy += F.mse_loss(
                    pred_xy[b, grid_y, grid_x, anchor_idx],
                    torch.tensor([tx, ty], device=device),
                    reduction='sum'
                )
                loss_wh += F.mse_loss(
                    pred_wh[b, grid_y, grid_x, anchor_idx],
                    torch.tensor([tw, th], device=device),
                    reduction='sum'
                )
                
                # 置信度损失 (有目标)
                loss_conf += F.binary_cross_entropy(
                    pred_conf[b, grid_y, grid_x, anchor_idx],
                    torch.ones_like(pred_conf[b, grid_y, grid_x, anchor_idx]),
                    reduction='sum'
                )
                
                # 分类损失 - 给bicycle类别更高权重以处理类别不平衡
                target_cls = torch.zeros(self.num_classes, device=device)
                label_value = labels[n].item()
                target_cls[label_value] = 1.0
                
                # 如果是bicycle类别（标签1），给更高权重
                if label_value == 1:  # bicycle
                    pos_weight = torch.ones(self.num_classes, device=device) * 10.0  # bicycle权重x10
                else:
                    pos_weight = torch.ones(self.num_classes, device=device)
                
                loss_cls += F.binary_cross_entropy_with_logits(
                    pred_cls[b, grid_y, grid_x, anchor_idx],
                    target_cls,
                    pos_weight=pos_weight,
                    reduction='sum'
                )
                
                num_pos += 1
            
            # 对于没有目标的网格，计算置信度损失
            # 创建一个 mask 标记哪些网格有目标
            obj_mask = torch.zeros((H, W, num_anchors), device=device)
            for n in range(len(boxes)):
                grid_x = int(cx[n] * W)
                grid_y = int(cy[n] * H)
                grid_x = min(grid_x, W - 1)
                grid_y = min(grid_y, H - 1)
                obj_mask[grid_y, grid_x, 0] = 1
            
            # 无目标的置信度损失
            noobj_mask = 1 - obj_mask
            loss_conf += F.binary_cross_entropy(
                pred_conf[b].squeeze(-1) * noobj_mask,
                torch.zeros_like(pred_conf[b].squeeze(-1)) * noobj_mask,
                reduction='sum'
            ) * self.lambda_noobj
        
        # 归一化
        if num_pos > 0:
            loss_xy /= num_pos
            loss_wh /= num_pos
            loss_cls /= num_pos
        loss_conf /= (B * H * W * num_anchors)
        
        # 总损失
        total_loss = (
            self.lambda_coord * loss_xy +
            self.lambda_coord * loss_wh +
            loss_conf +
            loss_cls
        )
        
        return {
            'total_loss': total_loss,
            'loss_xy': loss_xy,
            'loss_wh': loss_wh,
            'loss_conf': loss_conf,
            'loss_cls': loss_cls,
        }

# ==== 5. 数据集类 ====
class COCODetectionDataset(torch.utils.data.Dataset):
    """
    COCO 检测数据集（只使用包含bicycle的样本）
    """
    def __init__(self, split='train', max_images=None, input_size=416):
        print(f"加载 COCO 数据集 (split={split})...")
        from huggingface_hub import get_token
        self.ds = load_dataset("detection-datasets/coco", split=split, token=get_token(), cache_dir="/ssdwork/air/linzhen/hf_home/datasets")
        
        # 过滤：只保留包含bicycle的图像
        if FILTER_BICYCLE_ONLY:
            print("过滤数据集：只保留包含bicycle的图像...")
            filtered_indices = []

            # 最多扫描 MAX_SCAN_FOR_FILTER 个样本以收集足够的bicycle样本
            scan_limit = min(MAX_SCAN_FOR_FILTER, len(self.ds))
            for idx in range(scan_limit):
                item = self.ds[idx]
                objs = item.get("objects", {})
                found = False
                if isinstance(objs, dict):
                    category_ids = [int(c) for c in objs.get("category_id", [])]
                    if any(cid in BICYCLE_CATEGORY_IDS for cid in category_ids):
                        found = True
                else:
                    # 备用格式
                    for obj in objs:
                        if not isinstance(obj, dict):
                            continue
                        cid = int(obj.get("category_id", -1))
                        if cid in BICYCLE_CATEGORY_IDS:
                            found = True
                            break
                if found:
                    filtered_indices.append(idx)

                # 进度打印
                if (idx + 1) % 5000 == 0:
                    print(f"  已扫描 {idx + 1}/{scan_limit} 张，找到 {len(filtered_indices)} 张包含bicycle")

            print(f"过滤前: {len(self.ds)} 张图像")
            print(f"过滤后: {len(filtered_indices)} 张图像包含bicycle（扫描上限 {scan_limit}）")

            if len(filtered_indices) == 0:
                print("\n警告：未找到包含bicycle的图像！")
                print("将使用所有数据进行训练（以二分类方式标注）...")
            else:
                self.ds = self.ds.select(filtered_indices)
        
        if max_images:
            self.ds = self.ds.select(range(min(max_images, len(self.ds))))
            print(f"限制为前 {len(self.ds)} 张图像")
        
        self.input_size = input_size
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"].convert('RGB')
        orig_w, orig_h = img.size
        
        # Resize image
        img = img.resize((self.input_size, self.input_size))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        
        # 解析标注
        # HuggingFace COCO 数据集格式: objects 是一个字典，包含多个列表
        boxes = []
        labels = []
        
        # 获取 objects 字典
        objs = item.get("objects", {})
        
        # 检查数据格式
        if isinstance(objs, dict):
            # HuggingFace 格式：objects = {"bbox": [...], "category_id": [...], ...}
            bboxes = objs.get("bbox", [])
            category_ids = objs.get("category_id", [])
            
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                if bbox is None or len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                
                # 归一化坐标
                if x > 1.0:  # 绝对坐标
                    x /= orig_w
                    y /= orig_h
                    w /= orig_w
                    h /= orig_h
                
                if w <= 0 or h <= 0:
                    continue
                
                # 转换为 [x1, y1, x2, y2] 格式 (归一化)
                x1 = max(0, min(1, x))
                y1 = max(0, min(1, y))
                x2 = max(0, min(1, x + w))
                y2 = max(0, min(1, y + h))
                
                boxes.append([x1, y1, x2, y2])
                
                # 获取类别 ID 并映射为二分类
                cat_id = category_ids[i] if i < len(category_ids) else 0
                # 二分类：bicycle=1, 其他=0
                label = 1 if int(cat_id) in BICYCLE_CATEGORY_IDS else 0
                labels.append(label)
        else:
            # 备用格式：objects 是一个列表
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                bbox = obj.get("bbox", None)
                if bbox is None or len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                
                # 归一化坐标
                if x > 1.0:  # 绝对坐标
                    x /= orig_w
                    y /= orig_h
                    w /= orig_w
                    h /= orig_h
                
                if w <= 0 or h <= 0:
                    continue
                
                # 转换为 [x1, y1, x2, y2] 格式 (归一化)
                x1 = max(0, min(1, x))
                y1 = max(0, min(1, y))
                x2 = max(0, min(1, x + w))
                y2 = max(0, min(1, y + h))
                
                boxes.append([x1, y1, x2, y2])
                
                # 二分类标签：bicycle=1, 其他=0
                if "category_id" in obj:
                    cat_id = int(obj["category_id"])
                    label = 1 if cat_id in BICYCLE_CATEGORY_IDS else 0
                    labels.append(label)
                else:
                    labels.append(0)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        return img, target

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, 0)
    return images, targets

# ==== 6. 训练代码 ====
def train_model(epochs=5, batch_size=8, learning_rate=1e-4, max_images=None):
    print(f"\n========== 开始训练自定义检测模型 ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    train_dataset = COCODetectionDataset(split="train", max_images=max_images, input_size=416)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 初始化模型
    model = CustomDetector(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS).to(device)
    
    # 使用 DataParallel 利用多卡
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张 GPU 进行训练")
        model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = DetectionLoss(num_classes=NUM_CLASSES)
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            predictions = model(images)
            
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'xy': f'{loss_dict["loss_xy"].item():.4f}',
                'wh': f'{loss_dict["loss_wh"].item():.4f}',
                'conf': f'{loss_dict["loss_conf"].item():.4f}',
                'cls': f'{loss_dict["loss_cls"].item():.4f}',
            })
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} 完成. 平均 Loss: {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
            else:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"新最佳模型已保存! Loss: {best_loss:.4f}")
    
    print(f"训练结束. 最佳 Loss: {best_loss:.4f}")
    print(f"最终模型已保存至: {MODEL_SAVE_PATH}")
    print("========== 训练完成 ==========\n")

# ==== 7. NMS (非极大值抑制) ====
def nms(boxes, scores, iou_threshold=0.5):
    """
    非极大值抑制
    boxes: (N, 4) [x1, y1, x2, y2]
    scores: (N,)
    """
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    
    keep = []
    while len(order) > 0:
        i = order[0].item()
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # 计算 IoU
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        
        w = torch.maximum(torch.tensor(0.0), xx2 - xx1)
        h = torch.maximum(torch.tensor(0.0), yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留 IoU 小于阈值的框
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

# ==== 8. 推理代码 ====
@torch.no_grad()
def predict_custom(model, image_path: str, device, output_dir: str, 
                   score_thresh: float = 0.25, nms_thresh: float = 0.5,
                   input_size: int = 416):
    """
    使用自定义模型进行推理
    """
    model.eval()
    
    # 加载图像
    img_orig = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img_orig.size
    
    # 预处理
    img = img_orig.resize((input_size, input_size))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 推理
    predictions = model(img_tensor)  # (1, H, W, num_anchors, 5+num_classes)
    
    # 解析预测
    pred_xy = torch.sigmoid(predictions[..., 0:2])  # (1, H, W, A, 2)
    pred_wh = predictions[..., 2:4]  # (1, H, W, A, 2)
    pred_conf = torch.sigmoid(predictions[..., 4:5])  # (1, H, W, A, 1)
    pred_cls = torch.sigmoid(predictions[..., 5:])  # (1, H, W, A, num_classes)
    
    # 提取检测结果
    B, H, W, A, _ = predictions.shape
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for h in range(H):
        for w in range(W):
            for a in range(A):
                conf = pred_conf[0, h, w, a, 0].item()
                
                if conf < score_thresh:
                    continue
                
                # 获取类别分数
                cls_scores = pred_cls[0, h, w, a]  # (num_classes,)
                cls_id = torch.argmax(cls_scores).item()
                cls_score = cls_scores[cls_id].item()
                
                # 二分类模型：类别1是bicycle，类别0是背景
                if cls_id != 1:  # 不是bicycle
                    continue
                
                final_score = conf * cls_score
                
                if final_score < score_thresh:
                    continue
                
                # 计算边界框
                cx = (w + pred_xy[0, h, w, a, 0].item()) / W
                cy = (h + pred_xy[0, h, w, a, 1].item()) / H
                bw = pred_wh[0, h, w, a, 0].item() / W
                bh = pred_wh[0, h, w, a, 1].item() / H
                
                # 转换为 [x1, y1, x2, y2] 格式（归一化）
                x1 = max(0, cx - bw / 2)
                y1 = max(0, cy - bh / 2)
                x2 = min(1, cx + bw / 2)
                y2 = min(1, cy + bh / 2)
                
                # 转换为原图坐标
                x1 = int(x1 * orig_w)
                y1 = int(y1 * orig_h)
                x2 = int(x2 * orig_w)
                y2 = int(y2 * orig_h)
                
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(final_score)
                all_labels.append(cls_id)
    
    # NMS
    if len(all_boxes) > 0:
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=nms_thresh)
        
        final_boxes = [all_boxes[i] for i in keep_indices]
        final_scores = [all_scores[i] for i in keep_indices]
        final_labels = [all_labels[i] for i in keep_indices]
    else:
        final_boxes = []
        final_scores = []
        final_labels = []
    
    # 可视化
    img_draw = np.array(img_orig)
    for box, score in zip(final_boxes, final_scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label_text = f"bicycle {score:.2f}"
        cv2.putText(img_draw, label_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    vis_path = Path(output_dir) / "detection_vis_custom.jpg"
    Image.fromarray(img_draw).save(vis_path)
    
    # 保存 JSON 结果
    result = [
        {"bbox": box, "score": float(score), "label": "bicycle"}
        for box, score in zip(final_boxes, final_scores)
    ]
    with open(Path(output_dir) / "detection_custom.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n========== 检测结果 ==========")
    print(f"检测到 {len(final_boxes)} 个自行车")
    for i, (box, score) in enumerate(zip(final_boxes, final_scores)):
        print(f"  目标{i}: bbox={box}, score={score:.3f}")
    print(f"结果已保存至: {vis_path}")
    print("============================\n")
    
    return final_boxes, final_scores

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"未找到模型文件: {MODEL_SAVE_PATH}")
        print("将自动开始训练...")
        train_model(epochs=args.epochs if hasattr(args, 'epochs') else 5)
    
    # 加载模型
    print(f"加载模型: {MODEL_SAVE_PATH}")
    model = CustomDetector(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    
    # 推理
    predict_custom(model, args.image, device, args.output_dir, 
                  score_thresh=args.score_thresh)

def main():
    parser = argparse.ArgumentParser(description='实验四扩展：自定义目标检测模型')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE, help='输入图片路径')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--train', action='store_true', help='强制重新训练模型')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--max_images', type=int, default=1000, help='最大训练图像数量')
    parser.add_argument('--score_thresh', type=float, default=0.25, help='检测分数阈值')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.train:
        train_model(epochs=args.epochs, batch_size=args.batch_size, max_images=args.max_images)
    
    # 推理
    infer(args)

if __name__ == "__main__":
    main()

