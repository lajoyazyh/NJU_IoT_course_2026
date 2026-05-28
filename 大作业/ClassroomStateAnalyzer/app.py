"""
课堂状态分析系统 - Streamlit 主界面
基于多人人脸检测与表情识别的课堂状态分析系统
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import tempfile

import streamlit as st

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 抑制 TensorFlow 日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 导入自定义模块
from face_detector import detect_faces
from expression_recognizer import get_recognizer
from analyzer import (
    calculate_statistics,
    judge_classroom_state,
    get_state_color,
    EXPRESSION_CATEGORIES,
    EXPRESSION_CN_MAP,
    format_stats_for_display,
    format_record_for_csv,
)
from utils import (
    draw_results,
    save_result_image,
    save_record,
    load_records,
    get_records_summary,
    export_csv,
)

# 页面配置
st.set_page_config(
    page_title="课堂状态分析系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 会话状态初始化 ====================
if "recognizer" not in st.session_state:
    with st.spinner("正在加载表情识别模型..."):
        st.session_state.recognizer = get_recognizer()
        st.session_state.recognizer.warm_up()
    st.success("✅ 模型加载并预热完成！")

if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = None
if "recognition_results" not in st.session_state:
    st.session_state.recognition_results = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "state_text" not in st.session_state:
    st.session_state.state_text = None
if "state_level" not in st.session_state:
    st.session_state.state_level = None
if "result_image" not in st.session_state:
    st.session_state.result_image = None
if "fast_mode" not in st.session_state:
    st.session_state.fast_mode = True


# ==================== 核心处理函数 ====================
def process_image(image, image_name=None):
    """
    处理单张图片：人脸检测 + 表情识别 + 统计 + 状态判断
    """
    # 转换为 BGR（OpenCV 格式）
    if isinstance(image, Image.Image):
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    # 1. 人脸检测
    with st.spinner("🔍 正在检测人脸..."):
        merge = st.session_state.get('merge_detectors', True)
        sensitivity = st.session_state.get('detection_sensitivity', 0.3)
        faces = detect_faces(image_bgr, merge_detectors=merge, min_conf_mediapipe=sensitivity)
    
    if len(faces) == 0:
        st.warning("⚠️ 未检测到人脸，请尝试其他图片。")
        st.session_state.recognition_results = []
        st.session_state.stats = None
        st.session_state.state_text = "未检测到学生"
        st.session_state.state_level = "empty"
        st.session_state.result_image = image_bgr
        return
    
    # 2. 表情识别（根据快速模式选择方法）
    fast = st.session_state.get('fast_mode', True)
    with st.spinner(f"🎭 正在识别 {len(faces)} 张人脸的表情..."):
        if fast:
            recognition_results = st.session_state.recognizer.recognize_all_fast(image_bgr, faces)
        else:
            recognition_results = st.session_state.recognizer.recognize_all(image_bgr, faces)
    
    # 3. 统计
    stats = calculate_statistics(recognition_results)
    
    # 4. 课堂状态判断
    state_text, state_level = judge_classroom_state(stats)
    
    # 5. 绘制结果
    result_image = draw_results(image_bgr, recognition_results)
    
    # 保存到 session
    st.session_state.recognition_results = recognition_results
    st.session_state.stats = stats
    st.session_state.state_text = state_text
    st.session_state.state_level = state_level
    st.session_state.result_image = result_image
    st.session_state.current_image = image_bgr
    st.session_state.current_image_name = image_name


def save_current_record():
    """保存当前检测记录到 CSV"""
    if st.session_state.stats is None:
        return False
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_name = st.session_state.current_image_name or "unknown.jpg"
    
    record = format_record_for_csv(
        timestamp, image_name,
        st.session_state.stats,
        st.session_state.state_text
    )
    
    return save_record(record)


# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("📚 课堂状态分析系统")
    st.markdown("---")
    
    # 输入方式选择
    st.subheader("📥 输入方式")
    input_mode = st.radio(
        "选择输入方式",
        ["📷 图片上传", "🎬 视频上传", "📹 摄像头实时"],
        index=0,
        help="选择图片、视频或摄像头作为输入源"
    )
    
    st.markdown("---")
    
    # 参数设置
    st.subheader("⚙️ 参数设置")
    
    fast_mode = st.checkbox(
        "⚡ 快速模式",
        value=st.session_state.fast_mode,
        help="开启后跳过数据增广，速度提升约 6 倍；关闭则使用增广策略，准确率略高"
    )
    st.session_state.fast_mode = fast_mode
    
    confidence_threshold = st.slider(
        "置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="低于此置信度的检测结果将被过滤"
    )
    
    detection_sensitivity = st.slider(
        "🔍 人脸检测灵敏度",
        min_value=0.1,
        max_value=0.9,
        value=st.session_state.get('detection_sensitivity', 0.3),
        step=0.05,
        help="越低越灵敏（召回率高但可能误检），越高越精确。推荐 0.3 以检测更多侧脸/模糊人脸"
    )
    st.session_state.detection_sensitivity = detection_sensitivity
    
    merge_detectors = st.checkbox(
        "🔄 双检测器合并",
        value=st.session_state.get('merge_detectors', True),
        help="同时使用 MediaPipe 和 Haar Cascade 检测，合并结果以检测更多人脸"
    )
    st.session_state.merge_detectors = merge_detectors
    
    st.markdown("---")
    
    # 操作按钮
    st.subheader("📋 操作")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 保存记录", use_container_width=True):
            if st.session_state.stats:
                if save_current_record():
                    st.success("记录已保存！")
                else:
                    st.error("保存失败")
            else:
                st.warning("请先完成检测")
    
    with col2:
        if st.button("📊 导出 CSV", use_container_width=True):
            df = load_records()
            if not df.empty:
                path = export_csv(df)
                st.success(f"已导出到: {path}")
            else:
                st.warning("暂无记录可导出")
    
    # 导出检测结果图片
    if st.button("🖼️ 导出结果图片", use_container_width=True):
        if st.session_state.result_image is not None:
            path = save_result_image(
                st.session_state.result_image,
                st.session_state.current_image_name
            )
            st.success(f"结果图片已保存到: {path}")
        else:
            st.warning("请先完成检测")
    
    st.markdown("---")
    st.caption("© 2025 物联网应用软件开发 · 大作业")


# ==================== 主区域 ====================
st.title("📚 基于多人人脸检测与表情识别的课堂状态分析系统")

# ---- 图片上传模式 ----
if input_mode == "📷 图片上传":
    uploaded_file = st.file_uploader(
        "上传一张包含多个人脸的图片",
        type=["jpg", "jpeg", "png", "bmp"],
        help="支持 JPG、PNG、BMP 格式"
    )
    
    if uploaded_file is not None:
        # 读取图片
        image = Image.open(uploaded_file)
        image_name = uploaded_file.name
        
        # 显示原始图片和处理按钮
        col_orig, col_btn = st.columns([3, 1])
        with col_orig:
            st.image(image, caption=f"原始图片: {image_name}", use_container_width=True)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 开始检测", type="primary", use_container_width=True):
                process_image(image, image_name)
                # 自动保存记录
                save_current_record()

# ---- 视频上传模式 ----
elif input_mode == "🎬 视频上传":
    uploaded_video = st.file_uploader(
        "上传一段包含人脸的课堂视频",
        type=["mp4", "avi", "mov", "mkv"],
        help="支持 MP4、AVI、MOV、MKV 格式"
    )
    
    if uploaded_video is not None:
        # 保存临时视频文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        st.video(uploaded_video)
        
        col1, col2 = st.columns(2)
        with col1:
            frame_interval = st.number_input(
                "采样帧间隔（每 N 帧采样一次）",
                min_value=1,
                max_value=100,
                value=30,
                help="值越小采样越密集"
            )
        with col2:
            max_frames = st.number_input(
                "最大采样帧数",
                min_value=1,
                max_value=200,
                value=20,
                help="最多分析多少帧"
            )
        
        if st.button("🔍 开始视频分析", type="primary"):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            st.info(f"视频信息: {total_frames} 帧, {fps:.1f} FPS")
            
            all_results = []
            frame_count = 0
            analyzed_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while cap.isOpened() and analyzed_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    analyzed_count += 1
                    status_text.text(f"正在分析第 {analyzed_count} 帧...")
                    
                    merge = st.session_state.get('merge_detectors', True)
                    sensitivity = st.session_state.get('detection_sensitivity', 0.3)
                    faces = detect_faces(frame, merge_detectors=merge, min_conf_mediapipe=sensitivity)
                    if len(faces) > 0:
                        fast = st.session_state.get('fast_mode', True)
                        if fast:
                            recognition_results = st.session_state.recognizer.recognize_all_fast(frame, faces)
                        else:
                            recognition_results = st.session_state.recognizer.recognize_all(frame, faces)
                        all_results.extend(recognition_results)
                    
                    progress_bar.progress(min(analyzed_count / max_frames, 1.0))
                
                frame_count += 1
            
            cap.release()
            
            # 汇总统计
            if len(all_results) > 0:
                stats = calculate_statistics(all_results)
                state_text, state_level = judge_classroom_state(stats)
                
                st.session_state.recognition_results = all_results
                st.session_state.stats = stats
                st.session_state.state_text = state_text
                st.session_state.state_level = state_level
                st.session_state.current_image_name = uploaded_video.name
                
                # 对最后一帧进行可视化
                cap2 = cv2.VideoCapture(video_path)
                last_frame = None
                while cap2.isOpened():
                    ret, frame = cap2.read()
                    if not ret:
                        break
                    last_frame = frame
                cap2.release()
                
                if last_frame is not None:
                    faces_last = detect_faces(last_frame)
                    if len(faces_last) > 0:
                        rec_last = st.session_state.recognizer.recognize_all(last_frame, faces_last)
                        st.session_state.result_image = draw_results(last_frame, rec_last)
                
                save_current_record()
                status_text.text(f"✅ 分析完成！共分析 {analyzed_count} 帧，检测到 {len(all_results)} 个人脸实例")
            else:
                st.warning("⚠️ 视频中未检测到人脸")
            
            progress_bar.empty()
            # 清理临时文件
            os.unlink(video_path)

# ---- 摄像头实时模式 ----
elif input_mode == "📹 摄像头实时":
    cam_mode = st.radio(
        "摄像头模式",
        ["📸 拍照模式", "🎥 实时流模式"],
        horizontal=True,
        help="拍照模式：点击拍照后分析；实时流模式：持续分析视频画面"
    )
    
    if cam_mode == "📸 拍照模式":
        st.info("📹 点击下方拍照区域即可捕获画面并自动分析")
        
        camera_photo = st.camera_input("📸 点击此处拍照")
        
        if camera_photo is not None:
            image = Image.open(camera_photo)
            timestamp_str = datetime.now().strftime("camera_%Y%m%d_%H%M%S.jpg")
            
            with st.spinner("🔍 正在分析摄像头拍摄的画面..."):
                process_image(image, timestamp_str)
                save_current_record()
    
    elif cam_mode == "🎥 实时流模式":
        st.info("🎥 实时视频流分析，摄像头画面将实时标注人脸和表情")
        st.warning("⚠️ 实时流模式需要浏览器授权摄像头，且对网络和性能有一定要求")
        
        try:
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
            import av
            import threading
            
            # 线程锁，保护 TF 模型调用
            _model_lock = threading.Lock()
            
            class FaceExpressionProcessor(VideoProcessorBase):
                """WebRTC 视频处理器：逐帧进行人脸检测和表情识别"""
                
                def __init__(self):
                    self.recognizer = None
                    self._load_model()
                
                def _load_model(self):
                    """在处理器初始化时加载模型"""
                    try:
                        self.recognizer = get_recognizer()
                        self.recognizer.warm_up()
                    except Exception as e:
                        print(f"[WebRTC] 模型加载失败: {e}")
                
                def recv(self, frame):
                    """处理每一帧视频"""
                    img = frame.to_ndarray(format="bgr24")
                    
                    if self.recognizer is None or self.recognizer.model is None:
                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                    
                    try:
                        with _model_lock:
                            # 人脸检测
                            faces = detect_faces(img)
                            
                            if len(faces) > 0:
                                # 快速模式识别
                                results = self.recognizer.recognize_all_fast(img, faces)
                                
                                # 绘制结果
                                img = draw_results(img, results)
                    except Exception as e:
                        print(f"[WebRTC] 帧处理错误: {e}")
                    
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            webrtc_streamer(
                key="classroom-webrtc",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=FaceExpressionProcessor,
                media_stream_constraints={
                    "video": {"width": 640, "height": 480},
                    "audio": False,
                },
                async_processing=True,
            )
            
        except ImportError:
            st.error("❌ 缺少 streamlit-webrtc 依赖，请运行: `pip install streamlit-webrtc`")
        except Exception as e:
            st.error(f"❌ 实时流启动失败: {e}")
            st.info("提示：实时流模式需要 HTTPS 或 localhost 环境，远程访问可能不支持")


# ==================== 检测结果展示 ====================
if st.session_state.recognition_results is not None:
    st.markdown("---")
    st.header("📊 检测结果")
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        if st.session_state.current_image is not None:
            img_rgb = cv2.cvtColor(st.session_state.current_image, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="原始图片", use_container_width=True)
    
    with col_img2:
        if st.session_state.result_image is not None:
            result_rgb = cv2.cvtColor(st.session_state.result_image, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="检测结果（框+表情标注）", use_container_width=True)
    
    # 统计信息
    if st.session_state.stats:
        st.markdown("---")
        
        col_stats, col_state = st.columns([2, 1])
        
        with col_stats:
            st.subheader("📈 表情统计")
            stats = st.session_state.stats
            
            # 显示关键指标
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("检测人数", stats['total_count'])
            with metric_cols[1]:
                st.metric("主要表情", stats['main_expression_cn'])
            with metric_cols[2]:
                happy_neutral = stats['expression_ratio']['Happy'] + stats['expression_ratio']['Neutral']
                st.metric("积极占比", f"{happy_neutral*100:.1f}%")
            with metric_cols[3]:
                sad_angry = stats['expression_ratio']['Sad'] + stats['expression_ratio']['Angry']
                st.metric("低落占比", f"{sad_angry*100:.1f}%")
            
            # 表情分布柱状图
            st.subheader("表情分布")
            chart_data = pd.DataFrame({
                '表情': [EXPRESSION_CN_MAP.get(cat, cat) for cat in EXPRESSION_CATEGORIES],
                '人数': [stats['expression_count'][cat] for cat in EXPRESSION_CATEGORIES],
                '比例': [stats['expression_ratio'][cat] for cat in EXPRESSION_CATEGORIES],
            })
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.bar_chart(chart_data.set_index('表情')['人数'], use_container_width=True)
            with col_chart2:
                st.bar_chart(chart_data.set_index('表情')['比例'], use_container_width=True)
            
            # 详细统计表格
            st.subheader("详细统计")
            display_df = chart_data.copy()
            display_df['比例'] = display_df['比例'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        with col_state:
            st.subheader("🎯 课堂状态")
            state_text = st.session_state.state_text
            state_level = st.session_state.state_level
            color = get_state_color(state_level)
            
            st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    margin: 10px 0;
                ">
                    {state_text}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # 显示各表情详细比例
            st.markdown("**各类表情比例:**")
            for cat in EXPRESSION_CATEGORIES:
                ratio = stats['expression_ratio'][cat]
                cn_name = EXPRESSION_CN_MAP.get(cat, cat)
                st.progress(ratio, text=f"{cn_name}: {ratio*100:.1f}%")
    
    # 单个人脸详细结果
    if len(st.session_state.recognition_results) > 0:
        st.markdown("---")
        st.subheader("👤 单个人脸识别详情")
        
        detail_cols = st.columns(min(len(st.session_state.recognition_results), 4))
        for i, rec in enumerate(st.session_state.recognition_results[:8]):  # 最多显示8个
            with detail_cols[i % 4]:
                label_cn = rec.get('label_cn', '未知')
                confidence = rec.get('confidence', 0)
                st.metric(
                    f"人脸 {i+1}",
                    f"{label_cn}",
                    delta=f"置信度 {confidence:.2%}"
                )


# ==================== 历史记录 ====================
st.markdown("---")
st.header("📋 历史检测记录")

records_df = get_records_summary(10)

if not records_df.empty:
    # 格式化显示
    display_cols = ['时间', '图片名称', '检测人数', 'Happy', 'Neutral', 'Sad', 'Angry', '主要表情', '课堂状态']
    available_cols = [col for col in display_cols if col in records_df.columns]
    
    st.dataframe(
        records_df[available_cols],
        use_container_width=True,
        hide_index=True
    )
    
    # 历史趋势图
    if len(records_df) >= 2:
        st.subheader("📈 历史趋势")
        trend_df = records_df.copy()
        if '时间' in trend_df.columns:
            trend_df['时间'] = pd.to_datetime(trend_df['时间'])
            trend_df = trend_df.sort_values('时间')
            
            trend_cols = ['Happy', 'Neutral', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust']
            available_trend_cols = [col for col in trend_cols if col in trend_df.columns]
            if available_trend_cols:
                st.line_chart(
                    trend_df.set_index('时间')[available_trend_cols],
                    use_container_width=True
                )
else:
    st.info("暂无历史检测记录，上传图片开始分析吧！")
