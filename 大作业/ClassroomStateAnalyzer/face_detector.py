"""
人脸检测模块
支持 MediaPipe Face Detection（优先）和 OpenCV Haar Cascade（回退）双检测器
"""
import os
import cv2
import numpy as np
import mediapipe as mp

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 初始化 MediaPipe 人脸检测（懒加载）
_mp_face_detection = None
_mp_face_detector = None

# OpenCV Haar Cascade 分类器路径
_HAAR_CASCADE_PATH = os.path.join(PROJECT_ROOT, "..", "..", "实验二", "FacialExpressionRecognition", "dataset", "params", "haarcascade_frontalface_alt.xml")
# 备用：尝试在同级目录查找
if not os.path.exists(_HAAR_CASCADE_PATH):
    _HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"

_haar_cascade = None


def _get_mediapipe_detector():
    """懒加载 MediaPipe 人脸检测器"""
    global _mp_face_detection, _mp_face_detector
    if _mp_face_detection is None:
        _mp_face_detection = mp.solutions.face_detection
        _mp_face_detector = _mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = 全范围模型（适合远距离人脸）
            min_detection_confidence=0.5
        )
    return _mp_face_detector


def _get_haar_cascade():
    """懒加载 Haar Cascade 分类器"""
    global _haar_cascade
    if _haar_cascade is None:
        if not os.path.exists(_HAAR_CASCADE_PATH):
            raise FileNotFoundError(f"Haar Cascade 文件未找到: {_HAAR_CASCADE_PATH}")
        _haar_cascade = cv2.CascadeClassifier(_HAAR_CASCADE_PATH)
    return _haar_cascade


def detect_faces_mediapipe(image_bgr):
    """
    使用 MediaPipe 检测人脸
    Args:
        image_bgr: BGR 格式的 numpy 图像数组
    Returns:
        faces: [(x1, y1, x2, y2, confidence), ...] 或空列表
    """
    detector = _get_mediapipe_detector()
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    results = detector.process(img_rgb)
    
    faces = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))
            confidence = detection.score[0] if hasattr(detection, 'score') else 1.0
            faces.append((x1, y1, x2, y2, confidence))
    
    return faces


def detect_faces_haar(image_bgr):
    """
    使用 OpenCV Haar Cascade 检测人脸
    Args:
        image_bgr: BGR 格式的 numpy 图像数组
    Returns:
        faces: [(x1, y1, x2, y2, confidence), ...] 或空列表
    """
    cascade = _get_haar_cascade()
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    faces_raw = cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    faces = []
    for (x, y, w, h) in faces_raw:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_bgr.shape[1], x + w)
        y2 = min(image_bgr.shape[0], y + h)
        faces.append((x1, y1, x2, y2, 0.8))  # Haar 无置信度，给默认值
    
    return faces


def detect_faces(image_bgr, use_mediapipe=True):
    """
    检测图像中的所有人脸
    Args:
        image_bgr: BGR 格式的 numpy 图像数组
        use_mediapipe: 是否优先使用 MediaPipe
    Returns:
        faces: [(x1, y1, x2, y2, confidence), ...]
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    
    faces = []
    
    if use_mediapipe:
        try:
            faces = detect_faces_mediapipe(image_bgr)
        except Exception as e:
            print(f"[face_detector] MediaPipe 检测失败: {e}，回退到 Haar Cascade")
    
    # MediaPipe 未检测到人脸或出错时，回退到 Haar
    if len(faces) == 0:
        try:
            faces = detect_faces_haar(image_bgr)
        except Exception as e:
            print(f"[face_detector] Haar Cascade 检测失败: {e}")
    
    return faces


def crop_face(image_bgr, face_bbox, margin=10):
    """
    根据人脸框裁剪人脸区域
    Args:
        image_bgr: BGR 图像
        face_bbox: (x1, y1, x2, y2, confidence)
        margin: 扩展边距（像素）
    Returns:
        face_img: 裁剪后的人脸 BGR 图像，或 None
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2, _ = face_bbox
    
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    face_img = image_bgr[y1:y2, x1:x2]
    return face_img


def draw_face_boxes(image_bgr, faces, labels=None, confidences=None):
    """
    在图像上绘制人脸框和标签
    Args:
        image_bgr: BGR 图像
        faces: 人脸框列表 [(x1, y1, x2, y2, confidence), ...]
        labels: 表情标签列表（可选）
        confidences: 置信度列表（可选）
    Returns:
        result_img: 标注后的图像
    """
    result_img = image_bgr.copy()
    
    for i, face in enumerate(faces):
        x1, y1, x2, y2, _ = face
        
        # 绘制人脸框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        if labels and i < len(labels):
            label = labels[i]
            conf = confidences[i] if confidences and i < len(confidences) else None
            text = f"{label}"
            if conf is not None:
                text += f" ({conf:.2f})"
            
            # 使用英文标注（OpenCV 不支持中文，中文标注在 utils.py 中处理）
            cv2.putText(result_img, text, (x1, max(y1 - 10, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return result_img


def count_faces(faces):
    """统计检测到的人脸数量"""
    return len(faces)


if __name__ == "__main__":
    # 简单测试
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detect_faces(test_img)
    print(f"空白图像检测结果: {len(faces)} 张人脸")
    print("人脸检测模块加载成功！")