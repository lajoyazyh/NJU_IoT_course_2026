"""
工具函数模块
包含中文标注、结果绘制、图片保存、CSV 导出等功能
"""
import os
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
        save_name = f"{base_name}_result.jpg"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"result_{timestamp}.jpg"
    
    save_path = os.path.join(RESULTS_DIR, save_name)
    cv2.imwrite(save_path, image_bgr)
    return save_path


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


if __name__ == "__main__":
    print("工具函数模块加载成功！")
    print(f"中文字体路径: {FONT_PATH} {'存在' if os.path.exists(FONT_PATH) else '不存在'}")
    print(f"CSV 记录路径: {CSV_PATH}")
    print(f"结果图片目录: {RESULTS_DIR}")