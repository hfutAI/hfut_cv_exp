import argparse
import math
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------- 基础工具：图像加载与格式转换 ---------------------- #
def load_image_rgb(path: str) -> np.ndarray:
    """使用 PIL 读取图片并强制转换为标准 RGB 格式的 Numpy 数组。"""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """
    手动实现经典的心理学灰度公式: Gray = 0.299*R + 0.587*G + 0.114*B。
    这是后续进行 Sobel 滤波和 GLCM 计算的必要预处理步骤。
    """
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


def zero_pad(img: np.ndarray, pad: int) -> np.ndarray:
    """
    零填充 (Zero Padding)：在图像四周补 0，防止卷积后图像尺寸变小。
    pad: 填充的像素层数。
    """
    h, w = img.shape
    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=img.dtype)
    padded[pad : pad + h, pad : pad + w] = img
    return padded


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    核心任务：手动实现 2D 卷积。
    逻辑：外两层循环遍历像素点，内两层循环执行窗口内的乘加运算。
    """
    kh, kw = kernel.shape
    assert kh == kw and kh % 2 == 1, "卷积核需为奇数方阵"
    pad = kh // 2
    # 预填充，确保输出图像与输入图像尺寸一致 (Same Padding)
    padded = zero_pad(img, pad)
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    
    # 开始嵌套循环进行空间卷积运算
    for i in range(h):
        for j in range(w):
            acc = 0.0
            # 在图像的当前局部窗口内滑动卷积核
            for ki in range(kh):
                for kj in range(kw):
                    # 卷积核权值 * 对应位置的图像像素值
                    acc += kernel[ki, kj] * padded[i + ki, j + kj]
            out[i, j] = acc
    return out


# ---------------------- Sobel 与自定义滤波 ---------------------- #
def sobel_filter(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    手动实现 Sobel 边缘检测。
    Gx: 检测垂直边缘；Gy: 检测水平边缘。
    """
    # 定义标准 Sobel 算子模板
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    
    gx = convolve2d(gray, sobel_x)
    gy = convolve2d(gray, sobel_y)
    
    # 综合梯度幅值：magnitude = sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(gx * gx + gy * gy)
    return gx, gy, magnitude


def apply_custom_kernel(gray: np.ndarray) -> np.ndarray:
    """
    应用实验要求给定的特定卷积核：[[1,0,-1],[2,0,-2],[1,0,-1]]。
    """
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    return convolve2d(gray, kernel)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    由于卷积计算后数值可能超出 0-255，此函数将 float 图片归一化到 uint8。
    防止保存图片时出现大面积发黑或发白。
    """
    min_v, max_v = float(img.min()), float(img.max())
    if math.isclose(max_v, min_v):
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - min_v) / (max_v - min_v)
    return (norm * 255).clip(0, 255).astype(np.uint8)


# ---------------------- 手写颜色直方图 ---------------------- #
def color_histogram(rgb: np.ndarray) -> np.ndarray:
    """
    手动统计三通道颜色直方图。
    不使用 np.histogram，而是通过双层循环遍历每个像素进行计数。
    """
    hist = np.zeros((3, 256), dtype=np.int64)
    h, w, _ = rgb.shape
    for i in range(h):
        for j in range(w):
            r, g, b = rgb[i, j]
            hist[0, r] += 1
            hist[1, g] += 1
            hist[2, b] += 1
    return hist


# ---------------------- 手写 GLCM 纹理特征 ---------------------- #
def quantize(gray: np.ndarray, levels: int) -> np.ndarray:
    """
    灰度量化：为了减小 GLCM 矩阵的大小（防止 256x256 导致计算太慢），
    将 0-255 的灰度缩放到较小的 levels 级（ 16 级）。
    """
    scaled = gray.astype(np.float32) / 255.0
    q = (scaled * levels).astype(np.int32)
    q[q >= levels] = levels - 1
    return q


def compute_glcm(gray: np.ndarray, levels: int = 16, distance: int = 1, angle: str = "0") -> np.ndarray:
    """
    手动计算灰度共生矩阵 (GLCM)。
    统计特定方向 (angle) 和距离 (distance) 上像素对共生的频次。
    """
    q = quantize(gray, levels)
    h, w = q.shape
    glcm = np.zeros((levels, levels), dtype=np.int64)

    # 定义方向偏移：(行位移, 列位移)
    offsets = {
        "0": (0, distance),      # 水平向右
        "90": (distance, 0),     # 垂直向下
        "45": (distance, distance),   # 右下对角线
        "135": (-distance, distance), # 右上对角线
    }
    dr, dc = offsets.get(angle, (0, distance))

    for i in range(h):
        for j in range(w):
            ni, nj = i + dr, j + dc
            # 判断相邻点是否在图像范围内
            if 0 <= ni < h and 0 <= nj < w:
                a = q[i, j]      # 当前点灰度
                b = q[ni, nj]    # 邻域点灰度
                glcm[a, b] += 1

    # 概率归一化，使得矩阵元素和为 1
    total = glcm.sum()
    if total == 0:
        return glcm.astype(np.float32)
    return glcm.astype(np.float32) / float(total)


def glcm_features(glcm: np.ndarray) -> Dict[str, float]:
    """
    从 GLCM 矩阵中提取二次统计量（纹理特征）。
    包含对比度、能量、同质性、熵和相关性。
    """
    levels = glcm.shape[0]
    contrast = 0.0      # 对比度：反映纹理深浅
    energy = 0.0        # 能量：反映图像均匀程度
    homogeneity = 0.0   # 同质性：反映局部平滑度
    entropy = 0.0       # 熵：反映图像随机性/信息量

    # 为计算相关性预先统计均值和方差
    i_indices = np.arange(levels, dtype=np.float32)
    j_indices = np.arange(levels, dtype=np.float32)
    mean_i = float((glcm.sum(axis=1) * i_indices).sum())
    mean_j = float((glcm.sum(axis=0) * j_indices).sum())
    var_i = float(((i_indices - mean_i) ** 2 * glcm.sum(axis=1)).sum())
    var_j = float(((j_indices - mean_j) ** 2 * glcm.sum(axis=0)).sum())
    var_i = max(var_i, 1e-12) # 防止除零
    var_j = max(var_j, 1e-12)

    correlation_num = 0.0

    # 循环累加各项特征指标
    for i in range(levels):
        for j in range(levels):
            p = float(glcm[i, j])
            if p <= 0: continue
            diff = float(i - j)
            contrast += diff * diff * p
            energy += p * p
            homogeneity += p / (1.0 + abs(diff))
            entropy -= p * math.log(p + 1e-12, 2)
            correlation_num += (i - mean_i) * (j - mean_j) * p

    correlation = correlation_num / math.sqrt(var_i * var_j)

    return {
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity,
        "entropy": entropy,
        "correlation": correlation,
    }


# ---------------------- 可视化与保存 ---------------------- #
def save_image(array_uint8: np.ndarray, path: str) -> None:
    """保存处理后的图像文件。"""
    Image.fromarray(array_uint8).save(path)


def plot_histogram(hist: np.ndarray, out_path: str) -> None:
    """可视化颜色直方图并保存。"""
    colors = ["r", "g", "b"]
    plt.figure(figsize=(10, 4))
    for c, col in enumerate(colors):
        plt.plot(hist[c], color=col, label=f"{col.upper()} channel")
    plt.title("Color Histogram (Manual 256 bins)")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_glcm(glcm: np.ndarray, out_path: str, title: str) -> None:
    """将纹理共生矩阵绘制为热力图展示。"""
    plt.figure(figsize=(5, 4))
    plt.imshow(glcm, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------- 实验一主流水线 ---------------------- #
def process_single_image(img_path: str, out_dir: str) -> Dict[str, str]:
    """
    处理流程：加载图片 -> Sobel滤波 -> 自定义核滤波 -> 直方图统计 -> 纹理特征保存。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 加载图像并预处理灰度
    rgb = load_image_rgb(img_path)
    gray = rgb_to_gray(rgb)

    # 任务 1：Sobel 滤波
    gx, gy, mag = sobel_filter(gray)
    sobel_img = normalize_to_uint8(mag)
    sobel_path = os.path.join(out_dir, "sobel.png")
    save_image(sobel_img, sobel_path)

    # 任务 2：给定卷积核滤波
    custom_resp = apply_custom_kernel(gray)
    custom_img = normalize_to_uint8(custom_resp)
    custom_path = os.path.join(out_dir, "custom_kernel.png")
    save_image(custom_img, custom_path)

    # 任务 3：可视化颜色直方图
    hist = color_histogram(rgb)
    hist_path = os.path.join(out_dir, "color_histogram.png")
    plot_histogram(hist, hist_path)

    # 任务 4：纹理特征 (GLCM) 提取与保存
    glcm = compute_glcm(gray, levels=16, distance=1, angle="0")
    glcm_path = os.path.join(out_dir, "glcm_heatmap.png")
    plot_glcm(glcm, glcm_path, "GLCM (0°)")
    
    features = glcm_features(glcm)
    # 保存结果至 npy 格式
    feat_path = os.path.join(out_dir, "texture_features.npy")
    np.save(feat_path, features, allow_pickle=True)

    # 保存 X/Y 梯度图，用于后续实验分析边缘方向
    gx_path = os.path.join(out_dir, "sobel_gx.png")
    gy_path = os.path.join(out_dir, "sobel_gy.png")
    save_image(normalize_to_uint8(gx), gx_path)
    save_image(normalize_to_uint8(gy), gy_path)

    return {
        "sobel": sobel_path,
        "sobel_gx": gx_path,
        "sobel_gy": gy_path,
        "custom_kernel": custom_path,
        "hist": hist_path,
        "glcm": glcm_path,
        "features": feat_path,
    }


def main():
    parser = argparse.ArgumentParser(description="实验一：图像滤波与特征提取")
    parser.add_argument("--input", type=str, required=True, help="自己拍摄的输入图片路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="结果保存目录")
    args = parser.parse_args()

    results = process_single_image(args.input, args.output_dir)
    print("实验一处理完成，所有中间结果已保存。")


if __name__ == "__main__":
    main()