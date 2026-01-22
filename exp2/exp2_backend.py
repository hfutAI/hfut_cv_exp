import cv2
import numpy as np
import os

"""
在笛卡尔坐标系中, 变量还是(x,y), 而参数用(r,θ)表示。

在霍夫空间中, 变量变成了(r,θ), 而参数变成了(x, y)。

笛卡尔坐标系中的直线 ---> 对应霍夫空间中的点

笛卡尔坐标系中的点 ---> 对应霍夫空间中的曲线

当霍夫空间中，多条曲线交于同一点，就找到了笛卡尔坐标系中的一条直线
"""

def process_lane_image(img_path, output_path):
    """
    车道线检测核心流水线：颜色过滤 -> 边缘提取 -> ROI裁剪 -> 霍夫变换 -> 斜率过滤
    任务输入：校内道路图像
    任务输出：车道线位置（绿色标出）
    """
    # --- 准备工作：自动解析路径和文件名 ---
    output_dir = os.path.dirname(output_path) # 获取输出文件夹路径
    if not os.path.exists(output_dir): os.makedirs(output_dir) # 确保输出目录存在
    base_name = os.path.splitext(os.path.basename(img_path))[0] # 提取文件名（不含后缀）
    
    # 1. 读取原始图像
    img = cv2.imread(img_path)
    if img is None: 
        print(f"错误：无法读取图片 {img_path}")
        return {}
    
    # 将 BGR 转为 HLS 颜色空间。在 HLS 中，亮度(L)通道能更好地分离白色车道线与深色路面
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    # 2. 颜色过滤 (重点提取白色)
    # 定义白色在 HLS 空间中的范围（亮度 L 设为高阈值 200 以上）
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([180, 255, 255])
    # 得到二值化掩码：白色区域为 255，其他为 0
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    
    path_mask = os.path.join(output_dir, f"{base_name}_01_color_mask.jpg")
    cv2.imwrite(path_mask, white_mask) # 保存颜色过滤结果，检查是否过滤掉了浅色路面
    
    # 3. 边缘检测流水线
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度化，降低计算复杂度
    blur = cv2.GaussianBlur(gray, (7, 7), 0) # 高斯模糊，减少路面小石子等高频噪声
    # Canny 边缘检测：利用双阈值提取图像中的亮度突变点
    canny = cv2.Canny(blur, 50, 150)
    
    path_canny = os.path.join(output_dir, f"{base_name}_02_canny.jpg")
    cv2.imwrite(path_canny, canny) # 保存边缘图，检查车道线轮廓是否清晰
    
    # 结合颜色和边缘：只要是“高亮度”或“有边缘”的点都保留，增加虚线检测的稳健性
    combined_binary = cv2.bitwise_or(canny, white_mask)
    
    # 4. 感兴趣区域 (ROI) 掩码
    # 图像上半部分通常是天空和树木，通过定义梯形区域强制排除干扰
    height, width = img.shape[:2]
    mask = np.zeros_like(combined_binary)
    # 顶点定义顺序：左下 -> 左上 -> 右上 -> 右下（比例针对校内道路视角调优）
    # ROI：45% 高度，并稍微放宽顶点左右范围，确保能覆盖远处车道线
    roi_corners = np.array([[
        (int(width * 0.02), height),             # 左下角（更宽）
        (int(width * 0.40), int(height * 0.45)), # 远端左顶点（更高更偏左）
        (int(width * 0.70), int(height * 0.45)), # 远端右顶点（更高更偏右）
        (int(width * 0.98), height)              # 右下角（更宽）
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 255) # 在黑色蒙版上绘制白色填充的梯形
    # 按位“与”操作：只保留梯形内的边缘
    masked_img = cv2.bitwise_and(combined_binary, mask)
    
    path_roi = os.path.join(output_dir, f"{base_name}_03_roi.jpg")
    cv2.imwrite(path_roi, masked_img)

    # 5. 累计概率霍夫变换 (HoughLinesP) 
    # 参数调优：
    # threshold=40: 累加器中最少投票数
    # minLineLength=30: 丢弃过短的噪点线段
    # maxLineGap=150: 允许的最大间断。设大一点可以把断断续续的虚线连成一条长线
    lines = cv2.HoughLinesP(masked_img, 1, np.pi/180, threshold=40, 
                            minLineLength=30, maxLineGap=150)

    # 6. 绘制原始霍夫线段 (红色，用于调试对比)
    hough_canvas = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_canvas, (x1, y1), (x2, y2), (0, 0, 255), 2) # BGR: 红色
    
    path_hough = os.path.join(output_dir, f"{base_name}_04_hough_lines.jpg")
    # 将红线叠加在原图上显示
    cv2.imwrite(path_hough, cv2.addWeighted(img, 0.6, hough_canvas, 1, 0))

    # 7. 最终车道线结果绘制 (斜率过滤后的稳健线条)
    line_canvas = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算斜率 (dy / dx)，加 1e-6 防止除以 0
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            
            # 过滤逻辑：车道线在视野中通常是倾斜的。
            # 过滤掉 abs(slope) < 0.3 的线条（通常是路面横向裂缝、减速带或阴影）
            if abs(slope) < 0.3: continue
            
            # 使用实验要求的绿色绘制最终结果
            cv2.line(line_canvas, (x1, y1), (x2, y2), (0, 255, 0), 8) # BGR: 绿色，线宽 8

    # 8. 图像融合并输出
    # 0.8 和 1 分别为权重，0.0 为偏置值
    result = cv2.addWeighted(img, 0.8, line_canvas, 1, 0)
    cv2.imwrite(output_path, result)
    
    print(f"处理完成，全过程图集已保存至目录: {output_dir}")
    
    # 以字典形式返回，方便在主程序中调用各中间步骤路径
    return {
        "color_mask": path_mask,
        "canny": path_canny,
        "roi": path_roi,
        "hough_lines": path_hough,
        "final_result": output_path
    }

if __name__ == "__main__":
    # 执行检测并打印路径列表
    results = process_lane_image("images/image_2.jpg", "output/image_2.jpg")
    for step, path in results.items():
        print(f"{step}: {path}")