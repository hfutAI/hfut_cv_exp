import streamlit as st
import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import tempfile
import shutil
import json

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

# 设置页面配置
st.set_page_config(
    page_title="机器视觉实验平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS 样式（天蓝色系）
st.markdown("""
<style>
    /* ================= AutoFigure Sky Blue Theme ================= */
    :root {
      /* Primary Colors */
      --af-accent-primary: #0ea5e9;
      --af-accent-secondary: #38bdf8;
      --af-accent-tertiary: #7dd3fc;

      /* Background Colors - Light Mode */
      --af-bg-primary: #f8fafc;
      --af-bg-secondary: #eef2ff;
      --af-bg-tertiary: #e0e7ff;
      --af-bg-elevated: rgba(255, 255, 255, 0.95);
      --af-bg-glass: rgba(255, 255, 255, 0.85);

      /* Text Colors */
      --af-text-primary: #1f2937;
      --af-text-secondary: #475569;
      --af-text-tertiary: #64748b;
      --af-text-muted: #94a3b8;

      /* Border Colors */
      --af-border-primary: rgba(203, 213, 225, 0.8);
      --af-border-secondary: rgba(226, 232, 240, 0.9);
      --af-border-accent: rgba(14, 165, 233, 0.3);

      /* Shadows */
      --af-shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.06);
      --af-shadow-md: 0 4px 12px rgba(15, 23, 42, 0.08);
      --af-shadow-lg: 0 8px 24px rgba(15, 23, 42, 0.12);
      --af-shadow-glow: 0 0 20px rgba(14, 165, 233, 0.2);
      --af-shadow-button: 0 4px 12px rgba(14, 165, 233, 0.3);
      
      /* Transitions */
      --af-transition-fast: 150ms ease;
    }

    /* 全局字体与背景 */
    .stApp {
        background: linear-gradient(180deg, var(--af-bg-primary) 0%, var(--af-bg-secondary) 100%);
        color: var(--af-text-primary);
    }
    
    /* 顶端 Header 条背景色 */
    header[data-testid="stHeader"] {
        background-color: var(--af-bg-elevated);
        border-bottom: 1px solid var(--af-border-primary);
        backdrop-filter: blur(12px);
    }

    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background-color: var(--af-bg-secondary);
        border-right: 1px solid var(--af-border-primary);
    }
    
    /* 标题样式 */
    h1, h2, h3 {
        color: var(--af-accent-primary) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* 普通文本颜色优化，避免在浅蓝背景下看不清 */
    p, label, span, div {
        color: var(--af-text-primary);
    }
    
    /* 按钮样式 - 主按钮 */
    div.stButton > button {
        background: linear-gradient(135deg, var(--af-accent-primary) 0%, var(--af-accent-secondary) 100%);
        color: white !important; /* 强制白字 */
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all var(--af-transition-fast);
        box-shadow: var(--af-shadow-button);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, var(--af-accent-secondary) 0%, var(--af-accent-primary) 100%);
        color: white !important;
        box-shadow: 0 6px 16px rgba(14, 165, 233, 0.4);
        transform: translateY(-1px);
    }
    
    /* 上传文件组件区域美化 */
    [data-testid="stFileUploader"] {
        background-color: var(--af-bg-glass);
        border: 2px dashed var(--af-accent-primary);
        border-radius: 12px;
        padding: 20px;
        box-shadow: var(--af-shadow-sm);
    }
    /* 上传组件内的 Browse 按钮 */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, var(--af-accent-primary) 0%, var(--af-accent-secondary) 100%);
        color: white !important;
        border: none;
        font-weight: bold;
        box-shadow: var(--af-shadow-button);
    }
    [data-testid="stFileUploader"] button:hover {
        background: linear-gradient(135deg, var(--af-accent-secondary) 0%, var(--af-accent-primary) 100%);
        color: white !important;
        box-shadow: 0 6px 16px rgba(14, 165, 233, 0.4);
    }
    /* 上传组件内的提示文字 "Drag and drop..." - 强制改为天蓝色 */
    [data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] p {
         color: var(--af-accent-primary) !important;
         font-weight: 600;
    }
    [data-testid="stFileUploader"] div div {
         color: var(--af-accent-primary);
    }
    [data-testid="stFileUploader"] small {
         color: var(--af-accent-secondary) !important;
    }

    /* 卡片式容器背景 */
    div[data-testid="stVerticalBlock"] > div {
        background-color: transparent;
    }
    
    /* 结果展示区的样式 */
    .result-card {
        background-color: var(--af-bg-elevated);
        padding: 24px;
        border-radius: 12px;
        box-shadow: var(--af-shadow-md);
        margin-bottom: 24px;
        border: 1px solid var(--af-border-primary);
    }
    
    /* 进度条颜色 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--af-accent-primary), var(--af-accent-secondary));
    }
    
    /* Metric 样式 */
    [data-testid="stMetricValue"] {
        color: var(--af-accent-primary);
    }
    
    /* Slider 样式 hack - 强制覆盖所有可能出现的红色滑块 */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: var(--af-accent-primary) !important;
        box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.2) !important;
    }
    div[data-baseweb="slider"] div[role="slider"]:focus {
        box-shadow: 0 0 0 6px rgba(14, 165, 233, 0.3) !important;
    }
    div[data-baseweb="slider"] div[data-testid="stTickBar"] > div {
        background-color: var(--af-accent-primary) !important;
    }
    /* 滑动条轨道颜色 */
    div[data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, var(--af-accent-primary), var(--af-accent-secondary)) !important;
    }
    /* 滑动条数值显示颜色 - 强力覆盖 */
    div[data-testid="stSlider"] * {
        color: var(--af-accent-primary) !important;
    }
    
    /* Radio 按钮选中样式 hack */
    div[role="radiogroup"] > label > div:first-child {
        background-color: var(--af-bg-secondary) !important; 
        border-color: var(--af-accent-primary) !important;
    }
    div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child > div {
        background-color: var(--af-accent-primary) !important;
    }
    /* Radio 按钮文字颜色 */
    div[role="radiogroup"] label p {
        color: var(--af-text-primary) !important;
    }
    
    /* Success/Info/Error 提示框样式 */
    .stAlert {
        background-color: var(--af-bg-glass);
        border: 1px solid var(--af-border-primary);
        color: var(--af-text-primary);
    }
    
</style>
""", unsafe_allow_html=True)

# 添加各个实验的路径到 sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "exp1"))
sys.path.append(os.path.join(BASE_DIR, "exp2"))
sys.path.append(os.path.join(BASE_DIR, "exp3"))
sys.path.append(os.path.join(BASE_DIR, "exp4"))

# 动态导入后端模块
try:
    import exp1_backend
    import exp2_backend
    import exp3_backend
    import exp3_backend2
    import exp4_backend
    import exp4_backend2
except ImportError as e:
    st.error(f"导入后端模块失败: {e}")
    st.stop()


# ==================== 通用辅助函数 ====================

def save_uploaded_file(uploaded_file):
    """保存上传的文件到临时目录"""
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path, temp_dir
    return None, None

def cleanup_temp_dir(temp_dir):
    """清理临时目录"""
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

# ==================== 实验一 ====================

def render_exp1():
    st.markdown('<h2 style="border-bottom: 2px solid var(--af-accent-primary); padding-bottom: 10px;">实验一：图像滤波与纹理特征提取</h2>', unsafe_allow_html=True)
    st.info("""
    **实验内容**：
    1. 使用 Sobel 算子进行滤波。
    2. 使用给定卷积核 `[[1,0,-1],[2,0,-2],[1,0,-1]]` 进行滤波。
    3. 提取图像的颜色直方图。
    4. 提取 GLCM 纹理特征。
    """)

    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"], key="exp1_upload")

    if uploaded_file:
        file_path, temp_dir = save_uploaded_file(uploaded_file)
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### 原始图像")
            st.image(uploaded_file, use_container_width=True)
        
        with col2:
            st.markdown("#### 操作面板")
            if st.button("开始处理", key="exp1_run", use_container_width=True):
                with st.spinner("正在处理..."):
                    try:
                        output_dir = os.path.join(temp_dir, "output")
                        results = exp1_backend.process_single_image(file_path, output_dir)
                        
                        st.success("处理完成！")
                        
                        # 展示结果
                        st.markdown("---")
                        st.markdown("### 滤波结果")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.image(results["sobel"], caption="Sobel 滤波", use_container_width=True)
                        with c2:
                            st.image(results["custom_kernel"], caption="自定义卷积核", use_container_width=True)
                        with c3:
                            st.image(results["sobel_gx"], caption="Sobel Gx", use_container_width=True)

                        st.markdown("### 特征分析")
                        c4, c5 = st.columns(2)
                        with c4:
                            st.image(results["hist"], caption="颜色直方图", use_container_width=True)
                        with c5:
                            st.image(results["glcm"], caption="GLCM 纹理特征", use_container_width=True)
                        
                        # 显示纹理数值特征
                        features = np.load(results["features"], allow_pickle=True).item()
                        st.markdown("### 纹理特征数值 (GLCM)")
                        st.json(features)

                    except Exception as e:
                        st.error(f"处理出错: {e}")
                    finally:
                        # 注意：实际部署可能需要保留文件供下载，这里简化处理不立即删除
                        pass
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== 实验二 ====================

def render_exp2():
    st.markdown('<h2 style="border-bottom: 2px solid var(--af-accent-primary); padding-bottom: 10px;">实验二：车道线检测</h2>', unsafe_allow_html=True)
    st.info("""
    **实验内容**：
    使用霍夫变换检测道路图像中的车道线，并用绿色标记。
    """)

    uploaded_file = st.file_uploader("上传道路图片", type=["jpg", "jpeg", "png"], key="exp2_upload")

    if uploaded_file:
        file_path, temp_dir = save_uploaded_file(uploaded_file)
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 原始图像")
            st.image(uploaded_file, use_container_width=True)

        with col2:
            st.markdown("#### 检测结果")
            if st.button("开始检测", key="exp2_run", use_container_width=True):
                with st.spinner("正在检测..."):
                    try:
                        output_path = os.path.join(temp_dir, "lane_result.jpg")
                        # 这里的 results 是一个字典，包含所有中间图片的路径
                        results = exp2_backend.process_lane_image(file_path, output_path)
                        
                        if results and os.path.exists(results["final_result"]):
                            st.success("检测完成！")
                            st.image(results["final_result"], caption="最终检测结果", use_container_width=True)
                            
                            st.markdown("### 中间过程")
                            c1, c2 = st.columns(2)
                            with c1:
                                if "color_mask" in results:
                                    st.image(results["color_mask"], caption="1. 颜色掩码 (Color Mask)", use_container_width=True)
                                if "roi" in results:
                                    st.image(results["roi"], caption="3. 感兴趣区域 (ROI)", use_container_width=True)
                            with c2:
                                if "canny" in results:
                                    st.image(results["canny"], caption="2. 边缘检测 (Canny)", use_container_width=True)
                                if "hough_lines" in results:
                                    st.image(results["hough_lines"], caption="4. 霍夫变换 (所有线段)", use_container_width=True)

                        else:
                            st.error("未生成结果图像，可能未检测到车道线。")
                    except Exception as e:
                        st.error(f"处理出错: {e}")
                        import traceback
                        st.text(traceback.format_exc())
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== 实验三 ====================

@st.cache_resource
def load_exp3_model():
    """缓存加载实验三模型（预训练）"""
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "farleyknight-org-username/vit-base-mnist"
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        model.to(device)
        return processor, model, device
    except Exception as e:
        return None, None, None

def _get_model_mtime(path: str) -> float:
    """获取模型文件修改时间；不存在则返回0"""
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

@st.cache_resource
def load_exp3_custom_model(model_mtime: float):
    """
    缓存加载实验三模型（自定义CNN）
    缓存键包含 model_mtime，模型文件更新时间变动会触发重新加载
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = exp3_backend2.SimpleCNN().to(device)
        if os.path.exists(exp3_backend2.MODEL_SAVE_PATH):
            # 处理可能的 DataParallel 保存前缀
            state_dict = torch.load(exp3_backend2.MODEL_SAVE_PATH, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model.eval()
            return model, device
        return None, device
    except Exception:
        return None, None

def render_exp3():
    st.markdown('<h2 style="border-bottom: 2px solid var(--af-accent-primary); padding-bottom: 10px;">实验三：手写数字识别 (学号识别)</h2>', unsafe_allow_html=True)
    st.info("""
    **实验内容**：
    基于连通域分割，识别学号照片中的数字。
    支持两种模式：
    1. 使用 **预训练 ViT 模型** (Transfer Learning)
    2. 使用 **自定义 CNN 模型** (Training from Scratch)
    """)
    
    # 模式选择
    mode = st.radio("选择模型模式", ["预训练 ViT 模型", "自定义 CNN 模型 (需训练)"], horizontal=True)
    
    # 模型加载变量
    processor, model, device = None, None, None
    cnn_model = None

    if mode == "预训练 ViT 模型":
        with st.spinner("正在加载预训练模型..."):
            processor, model, device = load_exp3_model()
            if processor is None:
                st.error("模型加载失败，请检查网络或 HuggingFace 配置。")
                return
    else:
        # 自定义 CNN 模式
        pass # 动态加载，允许重新训练

    uploaded_file = st.file_uploader("上传学号图片", type=["jpg", "jpeg", "png"], key="exp3_upload")

    if uploaded_file:
        file_path, temp_dir = save_uploaded_file(uploaded_file)
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### 原始学号图像")
            st.image(uploaded_file, use_container_width=True)
        
        with col2:
            st.markdown("#### 识别操作")
            
            # 操作按钮区域
            if mode == "自定义 CNN 模型 (需训练)":
                # 训练选项
                do_train = st.checkbox("强制重新训练", value=False)
                # 选择 Epoch 数
                epochs = st.number_input("训练轮数 (Epochs)", min_value=1, max_value=20, value=3, step=1)
                # 显式刷新模型按钮
                refresh_model = st.checkbox("刷新模型缓存（模型文件更新后勾选）", value=False)
                
                if st.button("开始识别", key="exp3_run_cnn", use_container_width=True):
                    with st.spinner("正在执行..."):
                        try:
                            output_dir = os.path.join(temp_dir, "output")
                            os.makedirs(output_dir, exist_ok=True)

                            # 如果需要训练或模型不存在
                            if do_train or not os.path.exists(exp3_backend2.MODEL_SAVE_PATH):
                                with st.spinner(f"正在训练自定义 CNN 模型 ({epochs} Epochs)..."):
                                    # 暂时重定向 stdout 以捕获训练进度 (可选)
                                    exp3_backend2.train_model(epochs=epochs)
                                    st.success("训练完成！")
                                    # 清除缓存以重新加载新模型
                                    load_exp3_custom_model.clear()

                            # 如果用户勾选刷新缓存，也清除
                            if refresh_model:
                                load_exp3_custom_model.clear()

                            # 加载模型
                            model_mtime = _get_model_mtime(exp3_backend2.MODEL_SAVE_PATH)
                            cnn_model, device = load_exp3_custom_model(model_mtime)
                            st.info(f"加载自定义CNN模型: {exp3_backend2.MODEL_SAVE_PATH}\nmtime: {model_mtime}")
                            if cnn_model is None:
                                st.error("无法加载自定义模型，请先训练。")
                            else:
                                # 1. 分割
                                digit_images = exp3_backend.segment_digits_contours(file_path, output_dir)
                                
                                if not digit_images:
                                    st.warning("未检测到有效的数字区域。")
                                else:
                                    # 调试信息：数字张量统计
                                    digit_dbg = [
                                        {"idx": idx, "shape": img.shape, "mean": float(np.mean(img)), "min": int(np.min(img)), "max": int(np.max(img))}
                                        for idx, img in enumerate(digit_images)
                                    ]
                                    st.caption("调试：分割出的数字统计（shape/mean/min/max）：")
                                    st.dataframe(digit_dbg, use_container_width=True)
                                    # 2. 识别
                                    student_id = exp3_backend2.predict_digits_custom(cnn_model, digit_images, device)
                                    
                                    st.success(f"识别成功！")
                                    st.metric("识别结果 (学号)", student_id)
                                    
                                    # 展示分割过程
                                    st.markdown("### 分割过程可视化")
                                    debug_dir = Path(output_dir) / "digits"
                                    
                                    dc1, dc2 = st.columns(2)
                                    if (debug_dir / "02_threshold.png").exists():
                                        with dc1:
                                            st.image(str(debug_dir / "02_threshold.png"), caption="二值化结果", use_container_width=True)
                                    
                                    if (debug_dir / "03_annotated.png").exists():
                                        with dc2:
                                            st.image(str(debug_dir / "03_annotated.png"), caption="轮廓标记", use_container_width=True)
                                    
                                    st.markdown("### 提取的数字")
                                    cols = st.columns(min(len(digit_images), 10))
                                    for idx, img in enumerate(digit_images):
                                        with cols[idx % 10]:
                                            st.image(img, caption=f"{idx}", use_container_width=True, clamp=True)

                        except Exception as e:
                            st.error(f"出错: {e}")
                            import traceback
                            st.text(traceback.format_exc())

            else:
                # 预训练模型模式
                if st.button("开始识别", key="exp3_run_vit", use_container_width=True):
                    with st.spinner("正在分割与识别..."):
                        try:
                            output_dir = os.path.join(temp_dir, "output")
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # 1. 分割
                            digit_images = exp3_backend.segment_digits_contours(file_path, output_dir)
                            
                            if not digit_images:
                                st.warning("未检测到有效的数字区域。")
                            else:
                                # 2. 识别
                                student_id = exp3_backend.predict_digits(model, processor, digit_images, device)
                                
                                st.success(f"识别成功！")
                                st.metric("识别结果 (学号)", student_id)
                                
                                # 展示分割过程
                                st.markdown("### 分割过程可视化")
                                debug_dir = Path(output_dir) / "digits"
                                
                                dc1, dc2 = st.columns(2)
                                if (debug_dir / "02_threshold.png").exists():
                                    with dc1:
                                        st.image(str(debug_dir / "02_threshold.png"), caption="二值化结果", use_container_width=True)
                                
                                if (debug_dir / "03_annotated.png").exists():
                                    with dc2:
                                        st.image(str(debug_dir / "03_annotated.png"), caption="轮廓标记", use_container_width=True)
                                
                                st.markdown("### 提取的数字")
                                cols = st.columns(min(len(digit_images), 10))
                                for idx, img in enumerate(digit_images):
                                    with cols[idx % 10]:
                                        st.image(img, caption=f"{idx}", use_container_width=True, clamp=True)

                        except Exception as e:
                            st.error(f"识别出错: {e}")
                            import traceback
                            st.text(traceback.format_exc())
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== 实验四 ====================

@st.cache_resource
def load_exp4_model():
    """缓存加载实验四模型（预训练）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用 exp4_backend 中的函数
    try:
        model = exp4_backend.create_model(pretrained=True)
        model.to(device)
        model.eval()
        
        # 定义简单的 processor
        class SimpleProcessor:
            def __call__(self, images, return_tensors="pt"):
                tensor = exp4_backend.T.ToTensor()(images)
                return {"pixel_values": tensor.unsqueeze(0)}
        
        processor = SimpleProcessor()
        return processor, model, device
    except Exception as e:
        return None, None, None

@st.cache_resource
def load_exp4_custom_model(model_mtime=None):
    """缓存加载实验四自定义模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = exp4_backend2.CustomDetector(
            num_classes=exp4_backend2.NUM_CLASSES,
            num_anchors=exp4_backend2.NUM_ANCHORS
        ).to(device)
        
        if os.path.exists(exp4_backend2.MODEL_SAVE_PATH):
            state_dict = torch.load(exp4_backend2.MODEL_SAVE_PATH, map_location=device)
            # 处理可能的 DataParallel 保存前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model.eval()
            return model, device
        return None, device
    except Exception as e:
        return None, None

def render_exp4():
    st.markdown('<h2 style="border-bottom: 2px solid var(--af-accent-primary); padding-bottom: 10px;">实验四：共享单车目标检测</h2>', unsafe_allow_html=True)
    st.info("""
    **实验内容**：
    检测校园场景中的共享单车。
    支持两种模式：
    1. 使用 **预训练 Faster R-CNN** (Transfer Learning)
    2. 使用 **自定义检测模型** (Training from Scratch)
    """)
    
    # 模式选择
    mode = st.radio("选择模型模式", ["预训练 Faster R-CNN", "自定义检测模型 (需训练)"], horizontal=True)
    
    # 模型加载变量
    processor, model, device = None, None, None
    custom_model = None

    if mode == "预训练 Faster R-CNN":
        with st.spinner("正在加载预训练模型..."):
            processor, model, device = load_exp4_model()
            if model is None:
                st.error("模型加载失败。")
                return
    else:
        # 自定义模型模式
        pass  # 动态加载，允许重新训练

    uploaded_file = st.file_uploader("上传校园场景图片", type=["jpg", "jpeg", "png"], key="exp4_upload")
    score_thresh = st.slider("置信度阈值", 0.1, 0.9, 0.25, 0.05)

    if uploaded_file:
        file_path, temp_dir = save_uploaded_file(uploaded_file)
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 原始图像")
            st.image(uploaded_file, use_container_width=True)

        with col2:
            st.markdown("#### 检测操作")
            
            # 操作按钮区域
            if mode == "自定义检测模型 (需训练)":
                # 训练选项
                do_train = st.checkbox("强制重新训练", value=False)
                # 训练参数
                col_a, col_b = st.columns(2)
                with col_a:
                    epochs = st.number_input("训练轮数", min_value=1, max_value=20, value=3, step=1)
                with col_b:
                    max_images = st.number_input("最大训练图像数", min_value=100, max_value=10000, value=1000, step=100)
                
                refresh_cache = st.checkbox("刷新模型缓存", value=False)
                
                if st.button("开始检测 (自定义)", key="exp4_run_custom", use_container_width=True):
                    with st.spinner("正在执行..."):
                        try:
                            output_dir = os.path.join(temp_dir, "output")
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # 获取模型文件时间戳
                            model_mtime = None
                            if os.path.exists(exp4_backend2.MODEL_SAVE_PATH):
                                model_mtime = os.path.getmtime(exp4_backend2.MODEL_SAVE_PATH)
                            
                            # 如果需要刷新缓存
                            if refresh_cache:
                                load_exp4_custom_model.clear()
                                st.info("已清除模型缓存")

                            # 如果需要训练或模型不存在
                            if do_train or not os.path.exists(exp4_backend2.MODEL_SAVE_PATH):
                                with st.spinner(f"正在训练自定义检测模型 ({epochs} Epochs, {max_images} 图像)..."):
                                    exp4_backend2.train_model(epochs=epochs, batch_size=8, max_images=max_images)
                                    st.success("训练完成！")
                                    # 清除缓存以重新加载新模型
                                    load_exp4_custom_model.clear()
                                    # 更新时间戳
                                    model_mtime = os.path.getmtime(exp4_backend2.MODEL_SAVE_PATH)

                            # 加载模型
                            custom_model, device = load_exp4_custom_model(model_mtime)
                            if custom_model is None:
                                st.error("无法加载自定义模型，请先训练。")
                            else:
                                # 检测
                                boxes, scores = exp4_backend2.predict_custom(
                                    custom_model, file_path, device, output_dir,
                                    score_thresh=score_thresh
                                )
                                
                                vis_path = os.path.join(output_dir, "detection_vis_custom.jpg")
                                json_path = os.path.join(output_dir, "detection_custom.json")
                                
                                if os.path.exists(vis_path):
                                    st.success(f"检测完成！找到 {len(boxes)} 个目标")
                                    st.image(vis_path, use_container_width=True)
                                    
                                    if os.path.exists(json_path):
                                        with open(json_path, 'r') as f:
                                            res_data = json.load(f)
                                        if res_data:
                                            st.markdown("**检测详情:**")
                                            st.dataframe(res_data, height=200)
                                        else:
                                            st.info("未检测到目标。")
                                else:
                                    st.error("未生成结果图像。")

                        except Exception as e:
                            st.error(f"出错: {e}")
                            import traceback
                            st.text(traceback.format_exc())

            else:
                # 预训练模型模式
                if st.button("开始检测 (Faster R-CNN)", key="exp4_run_pretrained", use_container_width=True):
                    with st.spinner("正在检测..."):
                        try:
                            output_dir = os.path.join(temp_dir, "output")
                            # 直接调用 exp4_backend 的 predict 函数
                            exp4_backend.predict(
                                model, processor, file_path, device, output_dir, score_thresh=score_thresh
                            )
                            
                            vis_path = os.path.join(output_dir, "detection_vis.jpg")
                            json_path = os.path.join(output_dir, "detection.json")
                            
                            if os.path.exists(vis_path):
                                st.success("检测完成！")
                                st.image(vis_path, use_container_width=True)
                                
                                if os.path.exists(json_path):
                                    with open(json_path, 'r') as f:
                                        res_data = json.load(f)
                                    if res_data:
                                        st.markdown("**检测详情:**")
                                        st.dataframe(res_data, height=200)
                                    else:
                                        st.info("未检测到目标。")
                            else:
                                st.error("未生成结果图像。")

                        except Exception as e:
                            st.error(f"检测出错: {e}")
                            import traceback
                            st.text(traceback.format_exc())
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== 主界面逻辑 ====================

def main():
    st.sidebar.markdown('<h2 style="color: var(--af-accent-primary);">机器视觉实验平台</h2>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="color: var(--af-text-secondary); font-weight: 500; margin-bottom: 20px;">林圳 2023217534</div>', unsafe_allow_html=True)
    
    exp_selection = st.sidebar.radio(
        "选择实验项目",
        ["实验一：图像滤波与纹理", 
         "实验二：车道线检测", 
         "实验三：学号识别", 
         "实验四：共享单车检测"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("上传图片后点击运行按钮即可查看结果。")

    if exp_selection == "实验一：图像滤波与纹理":
        render_exp1()
    elif exp_selection == "实验二：车道线检测":
        render_exp2()
    elif exp_selection == "实验三：学号识别":
        render_exp3()
    elif exp_selection == "实验四：共享单车检测":
        render_exp4()

if __name__ == "__main__":
    main()

