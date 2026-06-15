"""
统一单帧分析管线。

图片、视频采样帧、摄像头帧和 WebRTC 帧都会先变成一张 BGR 图像，
再交给本模块完成检测、识别、统计、状态判断和结果绘制。
"""
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Any

from face_detector import detect_faces
from analyzer import calculate_statistics, judge_classroom_state
from utils import draw_results


@dataclass
class AnalysisConfig:
    """单帧分析参数。"""

    fast_mode: bool = True
    confidence_threshold: float = 0.3
    detection_sensitivity: float = 0.3
    merge_detectors: bool = True


def filter_recognition_results(recognition_results, confidence_threshold):
    """过滤低置信度表情识别结果。"""
    return [
        result for result in recognition_results
        if result.get("confidence", 0.0) >= confidence_threshold
    ]


def analyze_frame(frame_bgr, recognizer, config: AnalysisConfig) -> Dict[str, Any]:
    """
    分析单帧 BGR 图像。

    Returns:
        {
            "faces": 原始检测到的人脸框,
            "raw_recognition_results": 过滤前识别结果,
            "recognition_results": 过滤后识别结果,
            "stats": 过滤后统计,
            "state_text": 课堂状态文本,
            "state_level": 状态等级,
            "result_image": 标注后的图像,
            "elapsed_ms": 单帧耗时,
        }
    """
    start = perf_counter()

    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        stats = calculate_statistics([])
        state_text, state_level = judge_classroom_state(stats)
        return {
            "faces": [],
            "raw_recognition_results": [],
            "recognition_results": [],
            "stats": stats,
            "state_text": state_text,
            "state_level": state_level,
            "result_image": frame_bgr,
            "elapsed_ms": 0.0,
        }

    faces = detect_faces(
        frame_bgr,
        merge_detectors=config.merge_detectors,
        min_conf_mediapipe=config.detection_sensitivity,
    )

    if faces:
        if config.fast_mode:
            raw_results = recognizer.recognize_all_fast(frame_bgr, faces)
        else:
            raw_results = recognizer.recognize_all(frame_bgr, faces)
    else:
        raw_results = []

    recognition_results = filter_recognition_results(
        raw_results,
        config.confidence_threshold,
    )
    stats = calculate_statistics(recognition_results)
    state_text, state_level = judge_classroom_state(stats)
    result_image = draw_results(frame_bgr, recognition_results)
    elapsed_ms = (perf_counter() - start) * 1000

    return {
        "faces": faces,
        "raw_recognition_results": raw_results,
        "recognition_results": recognition_results,
        "stats": stats,
        "state_text": state_text,
        "state_level": state_level,
        "result_image": result_image,
        "elapsed_ms": elapsed_ms,
    }
