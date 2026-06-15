"""
工具函数模块
包含中文标注、结果绘制、图片保存、CSV 导出等功能
"""
import os
import json
import re
import zipfile
from io import BytesIO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 中文字体路径
FONT_PATH = os.path.join(PROJECT_ROOT, "assets", "simsun.ttc")

# CSV 记录文件路径
CSV_PATH = os.path.join(PROJECT_ROOT, "records.csv")

# 结果图片保存目录
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)


def cv2_img_add_text(img, text, left, top, text_color=(255, 255, 255), text_size=20):
    """
    在 OpenCV 图像上添加中文文本（使用 PIL）
    从实验二复用
    Args:
        img: OpenCV BGR 图像 (numpy array)
        text: 要添加的文本
        left, top: 文本起始位置
        text_color: RGB 颜色元组
        text_size: 字体大小
    Returns:
        添加文本后的 BGR 图像
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img)
    
    # 尝试加载中文字体
    try:
        if os.path.exists(FONT_PATH):
            font = ImageFont.truetype(FONT_PATH, text_size, encoding="utf-8")
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    draw.text((left, top), text, text_color, font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_results(image_bgr, recognition_results):
    """
    在检测图像上绘制人脸框和表情标注（支持中文）
    Args:
        image_bgr: 原始 BGR 图像
        recognition_results: expression_recognizer.recognize_all() 的返回结果
    Returns:
        标注后的 BGR 图像
    """
    result_img = image_bgr.copy()
    
    # 定义每种表情对应的框颜色
    emotion_colors = {
        '开心': (0, 255, 0),     # 绿色
        'Happy': (0, 255, 0),
        '中性': (255, 255, 0),   # 青色
        'Neutral': (255, 255, 0),
        '悲伤': (255, 0, 0),     # 蓝色
        'Sad': (255, 0, 0),
        '生气': (0, 0, 255),     # 红色
        'Angry': (0, 0, 255),
        '惊讶': (255, 0, 255),   # 品红
        'Surprise': (255, 0, 255),
        '恐惧': (0, 255, 255),   # 黄色
        'Fear': (0, 255, 255),
        '厌恶': (128, 0, 128),   # 紫色
        'Disgust': (128, 0, 128),
    }
    
    for i, rec in enumerate(recognition_results):
        bbox = rec['bbox']
        x1, y1, x2, y2, _ = bbox
        label_cn = rec.get('label_cn', '未知')
        confidence = rec.get('confidence', 0.0)
        
        # 获取颜色
        color = emotion_colors.get(label_cn, (0, 255, 0))
        
        # 绘制人脸框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        # 在人脸框上方添加背景条
        text = f"{label_cn} ({confidence:.2f})"
        text_size_px = 18
        
        # 绘制标签背景
        cv2.rectangle(result_img, 
                      (x1, max(0, y1 - 25)), 
                      (x1 + len(text) * 12 + 10, y1), 
                      color, -1)
        
        # 使用 PIL 绘制中文标签
        result_img = cv2_img_add_text(
            result_img, text, 
            x1 + 5, max(0, y1 - 22), 
            text_color=(255, 255, 255), 
            text_size=text_size_px
        )
    
    return result_img


def save_result_image(image_bgr, image_name=None):
    """
    保存标注后的检测结果图片
    Args:
        image_bgr: 标注后的 BGR 图像
        image_name: 原始图片名称（用于生成结果文件名），为 None 时使用时间戳
    Returns:
        保存的文件路径
    """
    if image_name:
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{base_name}_result_{timestamp}.jpg"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"result_{timestamp}.jpg"
    
    save_path = os.path.join(RESULTS_DIR, save_name)
    success, encoded = cv2.imencode(".jpg", image_bgr)
    if not success:
        raise ValueError("结果图片编码失败")
    encoded.tofile(save_path)
    return save_path


def encode_image_to_jpeg_bytes(image_bgr):
    """
    将 BGR 图像编码为 JPEG 字节，供 Streamlit 下载按钮使用。
    """
    success, encoded = cv2.imencode(".jpg", image_bgr)
    if not success:
        raise ValueError("结果图片编码失败")
    return encoded.tobytes()


def _safe_stem(name, fallback="result"):
    """生成适合文件名使用的短名称。"""
    stem = os.path.splitext(os.path.basename(name or fallback))[0]
    stem = re.sub(r'[\\/:*?"<>|]+', "_", stem).strip(" .")
    return stem or fallback


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def _create_artifact_dir(prefix, source_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _safe_stem(source_name, prefix)
    artifact_dir = os.path.join(RESULTS_DIR, f"{prefix}_{safe_name}_{timestamp}")
    os.makedirs(artifact_dir, exist_ok=True)
    return artifact_dir


def export_frame_report(
    result_image,
    source_name=None,
    stats=None,
    recognition_results=None,
    state_text=None,
    mode="image",
):
    """
    导出图片/摄像头单帧分析结果包。

    输出内容包括标注图、摘要 JSON 和单人脸识别明细 CSV。
    """
    artifact_dir = _create_artifact_dir(mode, source_name)
    paths = []

    image_path = os.path.join(artifact_dir, "annotated_result.jpg")
    success, encoded = cv2.imencode(".jpg", result_image)
    if not success:
        raise ValueError("结果图片编码失败")
    encoded.tofile(image_path)
    paths.append(image_path)

    summary = {
        "type": "frame",
        "mode": mode,
        "source_name": source_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "state_text": state_text,
        "stats": stats or {},
    }
    summary_path = os.path.join(artifact_dir, "summary.json")
    _write_json(summary_path, summary)
    paths.append(summary_path)

    if recognition_results:
        detail_path = os.path.join(artifact_dir, "face_details.csv")
        pd.DataFrame(recognition_results).to_csv(detail_path, index=False, encoding="utf-8-sig")
        paths.append(detail_path)

    return {
        "type": "frame",
        "artifact_dir": artifact_dir,
        "paths": paths,
        "primary_path": image_path,
    }


def export_video_report(
    video_name=None,
    video_summary=None,
    frame_records=None,
    warning=None,
    performance=None,
    keyframes=None,
):
    """
    导出视频分析结果包。

    输出内容包括视频摘要 JSON、逐帧时序 CSV 和关键采样帧 JPG。
    """
    artifact_dir = _create_artifact_dir("video", video_name)
    paths = []

    summary = {
        "type": "video",
        "source_name": video_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_summary": video_summary or {},
        "warning": warning or {},
        "performance": performance or {},
        "statistical_note": (
            "视频人数统计按采样帧展示平均每帧人数、峰值人数和人脸实例数；"
            "人脸实例数是跨帧累计次数，不等同于唯一学生人数。"
        ),
    }
    summary_path = os.path.join(artifact_dir, "video_summary.json")
    _write_json(summary_path, summary)
    paths.append(summary_path)

    if frame_records:
        temporal_path = os.path.join(artifact_dir, "temporal_records.csv")
        pd.DataFrame(frame_records).to_csv(temporal_path, index=False, encoding="utf-8-sig")
        paths.append(temporal_path)

    keyframe_paths = []
    for index, keyframe in enumerate(keyframes or [], start=1):
        label = _safe_stem(keyframe.get("label"), f"keyframe_{index}")
        frame_path = os.path.join(artifact_dir, f"{index:02d}_{label}.jpg")
        success, encoded = cv2.imencode(".jpg", keyframe["image"])
        if success:
            encoded.tofile(frame_path)
            paths.append(frame_path)
            keyframe_paths.append(frame_path)

    return {
        "type": "video",
        "artifact_dir": artifact_dir,
        "paths": paths,
        "primary_path": summary_path,
        "keyframe_paths": keyframe_paths,
    }


def make_zip_bytes(paths):
    """把一组结果文件打包成 ZIP 字节流，供前端下载。"""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in paths:
            if os.path.exists(path):
                zip_file.write(path, arcname=os.path.basename(path))
    buffer.seek(0)
    return buffer.getvalue()


def load_records():
    """
    加载历史检测记录
    Returns:
        pandas DataFrame，若文件不存在则返回空 DataFrame
    """
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
            return df
        except Exception as e:
            print(f"[utils] 读取 CSV 记录失败: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def save_record(record_dict):
    """
    保存一条检测记录到 CSV 文件
    Args:
        record_dict: 记录字典（由 analyzer.format_record_for_csv 生成）
    Returns:
        是否保存成功
    """
    try:
        df_new = pd.DataFrame([record_dict])
        
        if os.path.exists(CSV_PATH):
            df_existing = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        df_combined.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        print(f"[utils] 保存 CSV 记录失败: {e}")
        return False


def get_records_summary(limit=10):
    """
    获取最近 N 条记录的摘要
    Args:
        limit: 返回记录数量
    Returns:
        pandas DataFrame（最新的 limit 条记录）
    """
    df = load_records()
    if df.empty:
        return df
    return df.tail(limit).iloc[::-1]  # 最新的在前


def export_result_image(image_bgr, image_name=None):
    """
    导出检测结果图片（save_result_image 的别名，保持接口兼容）
    """
    return save_result_image(image_bgr, image_name)


def export_csv(df=None, export_path=None):
    """
    导出 CSV 文件
    Args:
        df: 要导出的 DataFrame，为 None 时导出全部记录
        export_path: 导出路径，为 None 时使用默认路径
    Returns:
        导出文件路径
    """
    if df is None:
        df = load_records()
    
    if export_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(RESULTS_DIR, f"export_{timestamp}.csv")
    
    df.to_csv(export_path, index=False, encoding='utf-8-sig')
    return export_path


def export_temporal_csv(frame_records, video_name=None):
    """
    导出视频逐帧时序分析日志。
    Args:
        frame_records: temporal_analyzer.analyze_video() 生成的逐帧记录列表
        video_name: 原始视频名称
    Returns:
        导出的 CSV 路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if video_name:
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        save_name = f"temporal_{base_name}_{timestamp}.csv"
    else:
        save_name = f"temporal_{timestamp}.csv"

    export_path = os.path.join(RESULTS_DIR, save_name)
    pd.DataFrame(frame_records).to_csv(export_path, index=False, encoding='utf-8-sig')
    return export_path


if __name__ == "__main__":
    print("工具函数模块加载成功！")
    print(f"中文字体路径: {FONT_PATH} {'存在' if os.path.exists(FONT_PATH) else '不存在'}")
    print(f"CSV 记录路径: {CSV_PATH}")
    print(f"结果图片目录: {RESULTS_DIR}")
