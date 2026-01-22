import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# COCO 类别 id -> name (部分)
COCO_ID_TO_NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
}


def get_transform(train: bool):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # 简单的数据增强
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class HFCOCODetection(torch.utils.data.Dataset):
    """
    直接使用 HuggingFace 数据集 detection-datasets/coco
    输出符合 torchvision detection 接口的 (image, target)
    """
    def __init__(self, split: str, transforms=None, max_images: int = None):
        self.ds = load_dataset("detection-datasets/coco", split=split)
        if max_images:
            self.ds = self.ds.select(range(min(max_images, len(self.ds))))
        self.transforms = transforms
        # 类别名称到 COCO 原始 id 的映射（最常用前几类，若不存在则回退）
        self.name_to_id = {
            "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
            "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
        }

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        w_img, h_img = img.size

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        objs = item.get("objects", [])
        for obj in objs:
            bbox = obj.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            # 如果是归一化坐标，转换为绝对坐标
            if x <= 1.0 and y <= 1.0 and w <= 1.0 and h <= 1.0:
                x *= w_img
                y *= h_img
                w *= w_img
                h *= h_img
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])

            # 标签处理
            if "category_id" in obj:
                labels.append(int(obj["category_id"]))
            elif "category" in obj:
                cat = obj["category"]
                labels.append(self.name_to_id.get(cat, 0))
            else:
                labels.append(0)

            areas.append(w * h)
            iscrowd.append(0)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes: int = 91, pretrained: bool = True) -> torch.nn.Module:
    """
    创建 Faster R-CNN 模型。
    - num_classes: COCO 为 91（含背景）
    """
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1" if pretrained else None)
    # 预训练头已经是91类，无需替换；若自定义类别，可在此处替换 box_predictor
    return model


def filter_bicycle(outputs, score_thresh: float = 0.25):
    """
    仅保留 COCO 类别 id=2 (bicycle) 的检测
    """
    keep_boxes = []
    keep_scores = []
    keep_labels = []
    for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
        if score >= score_thresh and label.item() == 2:
            keep_boxes.append(box)
            keep_scores.append(score)
            keep_labels.append(label)
    if len(keep_boxes) == 0:
        return {"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)), "labels": torch.empty((0,), dtype=torch.int64)}
    return {
        "boxes": torch.stack(keep_boxes),
        "scores": torch.stack(keep_scores),
        "labels": torch.stack(keep_labels),
    }


def train_one_epoch(model, optimizer, data_loader, device, epoch, max_steps=None):
    model.train()
    running_loss = 0.0
    steps = 0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        steps += 1
        pbar.set_postfix(loss=f"{losses.item():.4f}")

        if max_steps and steps >= max_steps:
            break
    return running_loss / steps


def evaluate(model, data_loader, device, max_batches=50):
    model.eval()
    count = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = [img.to(device) for img in images]
            _ = model(images)  # 仅前向，不计算指标
            count += 1
            if max_batches and count >= max_batches:
                break


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载 HuggingFace COCO 数据集：detection-datasets/coco ...")
    train_dataset = HFCOCODetection(split="train", transforms=get_transform(train=True), max_images=args.max_images)
    val_dataset = HFCOCODetection(split="validation", transforms=get_transform(train=False), max_images=1000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 模型
    model = create_model(pretrained=True).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    best_path = Path(args.output_dir) / "best_model.pth"
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1, max_steps=args.max_steps)
        print(f"[Epoch {epoch+1}] loss={loss:.4f}")
        evaluate(model, val_loader, device, max_batches=10)
        torch.save(model.state_dict(), best_path)
        print(f"已保存权重: {best_path}")

    return best_path


@torch.no_grad()
def predict(model, processor, image_path: str, device, output_dir: str, score_thresh: float = 0.25):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    tensor = processor(images=img, return_tensors="pt")["pixel_values"][0]
    outputs = model([tensor.to(device)])[0]

    # 只保留 bicycle 类别
    outputs = filter_bicycle(outputs, score_thresh=score_thresh)

    # 可视化
    if len(outputs["boxes"]) > 0:
        labels = [f"bicycle {s:.2f}" for s in outputs["scores"].cpu().numpy()]
        drawn = draw_bounding_boxes(
            (tensor * 255).byte(),
            boxes=outputs["boxes"],
            labels=labels,
            colors="red",
            width=3,
        )
        vis = torchvision.transforms.ToPILImage()(drawn.cpu())
    else:
        vis = img.copy()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    vis_path = Path(output_dir) / "detection_vis.jpg"
    vis.save(vis_path)

    # 保存JSON结果
    boxes = outputs["boxes"].cpu().numpy().tolist()
    scores = outputs["scores"].cpu().numpy().tolist() if len(outputs["scores"]) > 0 else []
    result = [{"bbox": b, "score": s, "label": "bicycle"} for b, s in zip(boxes, scores)]
    with open(Path(output_dir) / "detection.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"检测完成，结果保存至: {vis_path}")
    if len(result) == 0:
        print("未检测到共享单车")
    else:
        for r in result:
            print(f"bbox={r['bbox']}, score={r['score']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="实验四：共享单车目标检测 (Faster R-CNN + COCO)")
    parser.add_argument("--train", action="store_true", help="是否进行训练")
    parser.add_argument("--infer", action="store_true", help="是否进行推理")
    parser.add_argument("--output-dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--image", type=str, default="/ssdwork/air/linzhen/rebuttal/exp/exp4/images/1.jpg", help="推理图片路径")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None, help="可选：限制每个epoch的训练步数用于快速调试")
    parser.add_argument("--max-images", type=int, default=None, help="可选：限制训练样本数量用于快速调试")
    parser.add_argument("--score-thresh", type=float, default=0.25, help="推理分数阈值")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.output_dir) / "best_model.pth"
    if args.train:
        ckpt_path = train(args)

    # 推理阶段
    if args.infer:
        print("\n加载模型进行推理...")
        processor = torchvision.transforms.Compose([T.ToTensor()])
        model = create_model(pretrained=True).to(device)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"已加载训练权重: {ckpt_path}")
        else:
            print("未找到训练权重，使用预训练权重")

        # 使用更标准的 processor：ToTensor 已经满足模型输入（0-1 归一化）
        class SimpleProcessor:
            def __call__(self, images, return_tensors="pt"):
                tensor = T.ToTensor()(images)
                return {"pixel_values": tensor.unsqueeze(0)}

        processor = SimpleProcessor()

        predict(model, processor, args.image, device, args.output_dir, score_thresh=args.score_thresh)

    if not args.train and not args.infer:
        parser.print_help()


if __name__ == "__main__":
    main()

