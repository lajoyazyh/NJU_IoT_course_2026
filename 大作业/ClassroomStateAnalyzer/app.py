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
import time

import streamlit as st

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 抑制 TensorFlow 日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 导入自定义模块
from expression_recognizer import get_recognizer
from analyzer import (
    get_state_color,
    EXPRESSION_CATEGORIES,
    EXPRESSION_CN_MAP,
    format_record_for_csv,
)
from pipeline import AnalysisConfig, analyze_frame
from temporal_analyzer import analyze_video
from utils import (
    save_record,
    load_records,
    get_records_summary,
    export_csv,
    export_temporal_csv,
    export_frame_report,
    export_video_report,
    make_zip_bytes,
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
if "temporal_records" not in st.session_state:
    st.session_state.temporal_records = None
if "temporal_warning" not in st.session_state:
    st.session_state.temporal_warning = None
if "video_performance" not in st.session_state:
    st.session_state.video_performance = None
if "temporal_csv_path" not in st.session_state:
    st.session_state.temporal_csv_path = None
if "result_mode" not in st.session_state:
    st.session_state.result_mode = None
if "video_summary" not in st.session_state:
    st.session_state.video_summary = None
if "video_keyframes" not in st.session_state:
    st.session_state.video_keyframes = None
if "last_export_bundle" not in st.session_state:
    st.session_state.last_export_bundle = None


# ==================== 核心处理函数 ====================
def get_analysis_config():
    """从侧边栏状态构造统一分析配置。"""
    return AnalysisConfig(
        fast_mode=st.session_state.get('fast_mode', True),
        confidence_threshold=st.session_state.get('confidence_threshold', 0.3),
        detection_sensitivity=st.session_state.get('detection_sensitivity', 0.3),
        merge_detectors=st.session_state.get('merge_detectors', True),
    )


def apply_analysis_to_session(analysis, image_bgr, image_name=None):
    """把一次单帧分析结果写入 session_state，供统一结果区展示。"""
    st.session_state.recognition_results = analysis["recognition_results"]
    st.session_state.stats = analysis["stats"]
    st.session_state.state_text = analysis["state_text"]
    st.session_state.state_level = analysis["state_level"]
    st.session_state.result_image = analysis["result_image"]
    st.session_state.current_image = image_bgr
    st.session_state.current_image_name = image_name
    st.session_state.result_mode = "frame"
    st.session_state.last_export_bundle = None


def clear_temporal_session():
    """清空上一次视频时序结果，避免不同输入模式互相污染。"""
    st.session_state.temporal_records = None
    st.session_state.temporal_warning = None
    st.session_state.video_performance = None
    st.session_state.temporal_csv_path = None
    st.session_state.video_summary = None
    st.session_state.video_keyframes = None
    st.session_state.last_export_bundle = None


def process_image(image, image_name=None):
    """
    处理单张图片：人脸检测 + 表情识别 + 统计 + 状态判断
    """
    # 转换为 BGR（OpenCV 格式）
    if isinstance(image, Image.Image):
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    with st.spinner("🔍 正在检测人脸并识别表情..."):
        analysis = analyze_frame(image_bgr, st.session_state.recognizer, get_analysis_config())

    if len(analysis["faces"]) == 0:
        st.warning("⚠️ 未检测到人脸，请尝试其他图片。")
    elif len(analysis["recognition_results"]) == 0:
        st.warning("⚠️ 检测到人脸，但识别置信度低于当前阈值，未纳入统计。")

    clear_temporal_session()
    apply_analysis_to_session(analysis, image_bgr, image_name)


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


def save_current_result_package():
    """按当前输入模态导出完整结果包。"""
    result_mode = st.session_state.get("result_mode")

    if result_mode == "video" and st.session_state.temporal_records:
        return export_video_report(
            video_name=st.session_state.current_image_name,
            video_summary=st.session_state.video_summary,
            frame_records=st.session_state.temporal_records,
            warning=st.session_state.temporal_warning,
            performance=st.session_state.video_performance,
            keyframes=st.session_state.video_keyframes,
        )

    if st.session_state.result_image is not None:
        source_name = str(st.session_state.current_image_name or "")
        mode = "camera" if "camera" in source_name else "image"
        return export_frame_report(
            result_image=st.session_state.result_image,
            source_name=st.session_state.current_image_name,
            stats=st.session_state.stats,
            recognition_results=st.session_state.recognition_results,
            state_text=st.session_state.state_text,
            mode=mode,
        )

    return None


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
    st.session_state.confidence_threshold = confidence_threshold
    
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
    
    # 保存/下载当前结果包：图片/摄像头保存单帧报告，视频保存时序报告。
    if st.button("📦 保存当前结果包", use_container_width=True):
        if st.session_state.recognition_results is not None:
            try:
                bundle = save_current_result_package()
                if bundle:
                    st.session_state.last_export_bundle = bundle
                    st.success(f"结果包已保存到: {bundle['artifact_dir']}")
                else:
                    st.warning("当前结果暂无可保存内容")
            except Exception as e:
                st.error(f"保存结果包失败: {e}")
        else:
            st.warning("请先完成检测")

    bundle = st.session_state.get("last_export_bundle")
    if bundle and bundle.get("paths"):
        zip_name = os.path.basename(bundle["artifact_dir"]) + ".zip"
        try:
            st.download_button(
                "⬇️ 下载当前结果包",
                data=make_zip_bytes(bundle["paths"]),
                file_name=zip_name,
                mime="application/zip",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"生成结果包下载失败: {e}")
    
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
            sample_every_seconds = st.number_input(
                "采样间隔（秒）",
                min_value=0.2,
                max_value=10.0,
                value=1.0,
                step=0.2,
                help="默认约 1 秒 1 帧，符合时序分析任务要求"
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
            clear_temporal_session()
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_video_progress(analyzed_count, max_count):
                status_text.text(f"正在分析第 {analyzed_count} 个采样帧...")
                progress_bar.progress(min(analyzed_count / max_count, 1.0))

            result = analyze_video(
                video_path,
                st.session_state.recognizer,
                get_analysis_config(),
                sample_every_seconds=float(sample_every_seconds),
                max_frames=int(max_frames),
                progress_callback=update_video_progress,
            )
            performance = result["performance"]
            st.info(
                f"视频信息: {performance['total_video_frames']} 帧, "
                f"{performance['fps']:.1f} FPS, "
                f"采样间隔 {performance['sample_interval_frames']} 帧"
            )

            if result["frame_records"]:
                st.session_state.recognition_results = result["overall_results"]
                st.session_state.stats = result["video_summary"]["video_stats"]
                st.session_state.state_text = result["state_text"]
                st.session_state.state_level = result["state_level"]
                st.session_state.current_image = None
                st.session_state.current_image_name = uploaded_video.name
                st.session_state.result_image = None
                st.session_state.temporal_records = result["frame_records"]
                st.session_state.temporal_warning = result["warning"]
                st.session_state.video_performance = performance
                st.session_state.video_summary = result["video_summary"]
                st.session_state.video_keyframes = result["keyframes"]
                st.session_state.result_mode = "video"
                st.session_state.temporal_csv_path = export_temporal_csv(
                    result["frame_records"],
                    uploaded_video.name,
                )

                save_current_record()
                status_text.text(
                    f"✅ 分析完成！共分析 {performance['analyzed_frames']} 个采样帧，"
                    f"平均每帧 {performance['avg_people_per_frame']} 人，"
                    f"共 {performance['face_instances']} 个人脸实例"
                )
            else:
                st.warning("⚠️ 视频中未检测到人脸")
            
            progress_bar.empty()
            # 清理临时文件
            os.unlink(video_path)

# ---- 摄像头实时模式 ----
elif input_mode == "📹 摄像头实时":
    cam_mode = st.radio(
        "摄像头模式",
        ["📸 拍照模式", "🖥️ 本地实时模式", "🎥 实时流模式"],
        horizontal=True,
        help="拍照模式：点击拍照后分析；本地实时模式：用 OpenCV 直接读取本机摄像头；实时流模式：WebRTC 浏览器视频流"
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

    elif cam_mode == "🖥️ 本地实时模式":
        st.info("🖥️ 本地实时模式不使用 WebRTC，直接由 Python/OpenCV 读取本机摄像头，适合课堂演示")
        st.caption("如果打开失败，请关闭拍照模式预览、微信、腾讯会议或其他占用摄像头的软件后重试。")

        col_cam1, col_cam2, col_cam3 = st.columns(3)
        with col_cam1:
            camera_index = st.number_input(
                "摄像头编号",
                min_value=0,
                max_value=5,
                value=0,
                step=1,
                help="通常内置摄像头为 0；如果打不开可尝试 1 或 2"
            )
        with col_cam2:
            run_seconds = st.number_input(
                "运行时长（秒）",
                min_value=3,
                max_value=60,
                value=15,
                step=1,
                help="避免长时间循环阻塞 Streamlit 页面"
            )
        with col_cam3:
            analyze_interval = st.number_input(
                "分析间隔（帧）",
                min_value=1,
                max_value=30,
                value=5,
                step=1,
                help="值越小越实时，但 CPU 压力越大"
            )

        if st.button("▶️ 启动本地实时分析", type="primary"):
            clear_temporal_session()
            cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(int(camera_index))

            if not cap.isOpened():
                st.error("❌ 无法打开本机摄像头。请确认摄像头编号正确，且没有被其他程序占用。")
            else:
                frame_area = st.empty()
                stats_area = st.empty()
                progress_bar = st.progress(0)

                start_time = time.time()
                frame_count = 0
                last_results = []
                last_stats = None
                last_state_text = None
                last_state_level = None
                last_frame = None
                last_result_image = None

                try:
                    while time.time() - start_time < float(run_seconds):
                        ret, frame = cap.read()
                        if not ret:
                            stats_area.warning("⚠️ 摄像头读取失败，已停止。")
                            break

                        display_frame = frame

                        if frame_count % int(analyze_interval) == 0:
                            analysis = analyze_frame(
                                frame,
                                st.session_state.recognizer,
                                get_analysis_config(),
                            )
                            last_results = analysis["recognition_results"]
                            last_stats = analysis["stats"]
                            last_state_text = analysis["state_text"]
                            last_state_level = analysis["state_level"]
                            last_result_image = analysis["result_image"]

                        if last_result_image is not None:
                            display_frame = last_result_image

                        last_frame = frame
                        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        frame_area.image(frame_rgb, caption="本地实时检测画面", use_container_width=True)

                        if last_stats is not None and last_state_text is not None:
                            stats_area.info(
                                f"检测人数: {last_stats['total_count']} | "
                                f"主要表情: {last_stats['main_expression_cn']} | "
                                f"课堂状态: {last_state_text}"
                            )

                        elapsed = time.time() - start_time
                        progress_bar.progress(min(elapsed / float(run_seconds), 1.0))
                        frame_count += 1
                        time.sleep(0.03)
                finally:
                    cap.release()
                    progress_bar.empty()

                if last_stats is not None:
                    st.session_state.recognition_results = last_results
                    st.session_state.stats = last_stats
                    st.session_state.state_text = last_state_text
                    st.session_state.state_level = last_state_level
                    st.session_state.current_image = last_frame
                    st.session_state.current_image_name = datetime.now().strftime("local_camera_%Y%m%d_%H%M%S.jpg")
                    st.session_state.result_image = last_result_image
                    st.session_state.result_mode = "frame"
                    save_current_record()
                    st.success("✅ 本地实时分析已完成，最后一次检测结果已保存到历史记录。")
    
    elif cam_mode == "🎥 实时流模式":
        st.info("🎥 实时视频流分析，摄像头画面将实时标注人脸和表情")
        st.warning("⚠️ 实时流模式需要浏览器授权摄像头，且对网络和性能有一定要求")
        st.caption(
            "建议使用 Chrome 或 Edge 打开 http://localhost:8501 或 "
            "http://127.0.0.1:8501。若浏览器地址栏左侧摄像头权限不是“允许”，"
            "请改为允许后刷新页面；如果摄像头被微信、腾讯会议或其他标签页占用，"
            "请关闭占用程序后重试。本机演示默认使用本地直连；若实时流仍连接较慢，"
            "可使用拍照模式完成演示。"
        )
        use_stun = st.checkbox(
            "🌐 启用 STUN 服务器（远程/局域网访问时再尝试）",
            value=False,
            help=(
                "localhost 本机演示通常不需要 STUN。部分校园网或代理环境会阻塞 "
                "Google STUN，导致一直显示 Connection is taking longer than expected。"
            )
        )
        
        try:
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
            import av
            import threading
            
            # 线程锁，保护 TF 模型调用
            _model_lock = threading.Lock()
            webrtc_config = get_analysis_config()
            
            class FaceExpressionProcessor(VideoProcessorBase):
                """WebRTC 视频处理器：逐帧进行人脸检测和表情识别"""
                
                def __init__(self):
                    self.recognizer = None
                    self.config = webrtc_config
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
                            analysis = analyze_frame(img, self.recognizer, self.config)
                            img = analysis["result_image"]
                    except Exception as e:
                        print(f"[WebRTC] 帧处理错误: {e}")
                    
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            rtc_configuration = (
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                if use_stun
                else {"iceServers": []}
            )
            webrtc_key = "classroom-webrtc-stun" if use_stun else "classroom-webrtc-local"
            
            webrtc_streamer(
                key=webrtc_key,
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=FaceExpressionProcessor,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 10, "max": 15},
                    },
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

    if st.session_state.result_mode == "video" and st.session_state.temporal_records:
        st.header("🎬 视频分析报告")

        summary = st.session_state.video_summary or {}
        perf = st.session_state.video_performance or {}
        warning = st.session_state.temporal_warning or {}

        st.caption(
            "视频统计按采样帧进行：人数指标展示平均每帧人数、峰值人数和人脸实例数，"
            "不把同一个人跨帧重复出现直接称为“检测人数”。"
        )

        overview_cols = st.columns(5)
        with overview_cols[0]:
            st.metric("采样帧数", summary.get("sampled_frames", 0))
        with overview_cols[1]:
            st.metric("平均每帧人数", f"{summary.get('avg_people_per_frame', 0):.2f}")
        with overview_cols[2]:
            st.metric("峰值人数", summary.get("max_people_per_frame", 0))
        with overview_cols[3]:
            st.metric("人脸实例数", summary.get("face_instances", 0))
        with overview_cols[4]:
            st.metric("主要表情", summary.get("main_expression_cn", "无"))

        state_cols = st.columns([2, 1])
        with state_cols[0]:
            warning_level = warning.get('level', 'normal')
            warning_text = (
                f"{warning.get('level_cn', '正常')}："
                f"{warning.get('reason', '未触发连续状态预警')}"
            )
            if warning_level == 'red':
                st.error(warning_text)
            elif warning_level == 'yellow':
                st.warning(warning_text)
            elif warning_level == 'green':
                st.success(warning_text)
            else:
                st.info(warning_text)
        with state_cols[1]:
            color = get_state_color(summary.get("state_level", "normal"))
            st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    padding: 18px;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                ">
                    {summary.get("state_text", "课堂状态一般")}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("📈 按帧平均表情比例")
        mean_ratio = summary.get("mean_expression_ratio", {})
        ratio_df = pd.DataFrame({
            "表情": [EXPRESSION_CN_MAP.get(cat, cat) for cat in EXPRESSION_CATEGORIES],
            "平均比例": [mean_ratio.get(cat, 0.0) for cat in EXPRESSION_CATEGORIES],
        })
        st.bar_chart(ratio_df.set_index("表情")["平均比例"], use_container_width=True)

        perf_cols = st.columns(4)
        with perf_cols[0]:
            st.metric("总耗时", f"{perf.get('total_elapsed_sec', 0):.2f}s")
        with perf_cols[1]:
            st.metric("平均每帧耗时", f"{perf.get('avg_elapsed_ms_per_frame', 0):.1f}ms")
        with perf_cols[2]:
            st.metric("视频 FPS", perf.get('fps', 0))
        with perf_cols[3]:
            st.metric("采样间隔帧", perf.get('sample_interval_frames', 0))

        temporal_df = pd.DataFrame(st.session_state.temporal_records)
        trend_cols = [f"{cat}_ratio" for cat in EXPRESSION_CATEGORIES if f"{cat}_ratio" in temporal_df.columns]
        if trend_cols:
            st.subheader("⏱️ 表情比例时间趋势")
            chart_df = temporal_df[['time_sec'] + trend_cols].copy()
            rename_map = {f"{cat}_ratio": EXPRESSION_CN_MAP.get(cat, cat) for cat in EXPRESSION_CATEGORIES}
            chart_df = chart_df.rename(columns=rename_map)
            st.line_chart(chart_df.set_index('time_sec'), use_container_width=True)

        keyframes = st.session_state.video_keyframes or []
        if keyframes:
            st.subheader("🖼️ 关键采样帧预览")
            keyframe_cols = st.columns(min(len(keyframes), 3))
            for i, keyframe in enumerate(keyframes[:6]):
                with keyframe_cols[i % len(keyframe_cols)]:
                    image_rgb = cv2.cvtColor(keyframe["image"], cv2.COLOR_BGR2RGB)
                    record = keyframe["record"]
                    st.image(
                        image_rgb,
                        caption=(
                            f"{keyframe['label']} | {record.get('time_sec', 0):.1f}s | "
                            f"{record.get('state_text', '')}"
                        ),
                        use_container_width=True,
                    )

        st.subheader("📋 逐帧时序日志")
        log_cols = [
            'sample_index', 'time_sec', 'total_count', 'main_expression_cn',
            'state_text', 'elapsed_ms'
        ]
        available_log_cols = [col for col in log_cols if col in temporal_df.columns]
        st.dataframe(
            temporal_df[available_log_cols],
            use_container_width=True,
            hide_index=True
        )
        if st.session_state.temporal_csv_path:
            st.caption(f"时序日志已导出到: {st.session_state.temporal_csv_path}")
            if os.path.exists(st.session_state.temporal_csv_path):
                with open(st.session_state.temporal_csv_path, "rb") as f:
                    st.download_button(
                        "⬇️ 下载时序 CSV",
                        data=f.read(),
                        file_name=os.path.basename(st.session_state.temporal_csv_path),
                        mime="text/csv",
                        use_container_width=True,
                    )

    else:
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

        if st.session_state.stats:
            st.markdown("---")

            col_stats, col_state = st.columns([2, 1])

            with col_stats:
                st.subheader("📈 表情统计")
                stats = st.session_state.stats

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

                st.markdown("**各类表情比例:**")
                for cat in EXPRESSION_CATEGORIES:
                    ratio = stats['expression_ratio'][cat]
                    cn_name = EXPRESSION_CN_MAP.get(cat, cat)
                    st.progress(ratio, text=f"{cn_name}: {ratio*100:.1f}%")

        if len(st.session_state.recognition_results) > 0:
            st.markdown("---")
            st.subheader("👤 单个人脸识别详情")

            detail_cols = st.columns(min(len(st.session_state.recognition_results), 4))
            for i, rec in enumerate(st.session_state.recognition_results[:8]):
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
