"""
视频时序分析模块。

负责把视频拆成采样帧，逐帧调用 pipeline.analyze_frame，
并生成整体统计、时序记录、三级预警和性能指标。
"""
from time import perf_counter
from typing import Dict, Any, Callable, Optional

import cv2

from analyzer import (
    EXPRESSION_CATEGORIES,
    EXPRESSION_CN_MAP,
    calculate_statistics,
    judge_classroom_state,
    calculate_temporal_warning,
)
from pipeline import AnalysisConfig, analyze_frame


def _build_frame_record(sample_index, frame_index, time_sec, analysis):
    stats = analysis["stats"]
    record = {
        "sample_index": sample_index,
        "frame_index": frame_index,
        "time_sec": round(time_sec, 2),
        "total_count": stats["total_count"],
        "main_expression": stats["main_expression"],
        "main_expression_cn": stats["main_expression_cn"],
        "state_text": analysis["state_text"],
        "state_level": analysis["state_level"],
        "elapsed_ms": round(analysis["elapsed_ms"], 2),
    }

    for category in EXPRESSION_CATEGORIES:
        record[category] = stats["expression_count"][category]
        record[f"{category}_ratio"] = stats["expression_ratio"][category]

    return record


def _build_video_summary(frame_records, face_instances):
    """基于逐帧记录计算更适合视频模态的汇总口径。"""
    sampled_frames = len(frame_records)
    if sampled_frames == 0:
        empty_stats = calculate_statistics([])
        state_text, state_level = judge_classroom_state(empty_stats)
        return {
            "sampled_frames": 0,
            "face_instances": face_instances,
            "avg_people_per_frame": 0.0,
            "max_people_per_frame": 0,
            "mean_expression_ratio": {cat: 0.0 for cat in EXPRESSION_CATEGORIES},
            "mean_expression_count": {cat: 0.0 for cat in EXPRESSION_CATEGORIES},
            "main_expression": empty_stats["main_expression"],
            "main_expression_cn": empty_stats["main_expression_cn"],
            "state_text": state_text,
            "state_level": state_level,
            "video_stats": empty_stats,
        }

    avg_people = sum(r.get("total_count", 0) for r in frame_records) / sampled_frames
    max_people = max(r.get("total_count", 0) for r in frame_records)
    mean_ratio = {}
    mean_count = {}
    for category in EXPRESSION_CATEGORIES:
        mean_ratio[category] = round(
            sum(r.get(f"{category}_ratio", 0.0) for r in frame_records) / sampled_frames,
            4,
        )
        mean_count[category] = round(
            sum(r.get(category, 0) for r in frame_records) / sampled_frames,
            2,
        )

    main_expression = max(EXPRESSION_CATEGORIES, key=lambda cat: mean_ratio[cat])
    video_stats = {
        "total_count": round(avg_people, 2),
        "expression_count": mean_count,
        "expression_ratio": mean_ratio,
        "main_expression": main_expression,
        "main_expression_cn": EXPRESSION_CN_MAP.get(main_expression, main_expression),
        "expression_distribution": [mean_count[cat] for cat in EXPRESSION_CATEGORIES],
    }
    state_text, state_level = judge_classroom_state(video_stats)

    return {
        "sampled_frames": sampled_frames,
        "face_instances": face_instances,
        "avg_people_per_frame": round(avg_people, 2),
        "max_people_per_frame": max_people,
        "mean_expression_ratio": mean_ratio,
        "mean_expression_count": mean_count,
        "main_expression": main_expression,
        "main_expression_cn": EXPRESSION_CN_MAP.get(main_expression, main_expression),
        "state_text": state_text,
        "state_level": state_level,
        "video_stats": video_stats,
    }


def _select_keyframes(frame_records, frame_images):
    """选择适合视频报告展示的关键采样帧。"""
    if not frame_records:
        return []

    candidates = [
        ("开始帧", 0),
        ("中间帧", len(frame_records) // 2),
        ("结束帧", len(frame_records) - 1),
    ]

    peak_index = max(
        range(len(frame_records)),
        key=lambda idx: frame_records[idx].get("total_count", 0),
    )
    candidates.append(("人数峰值帧", peak_index))

    warning_index = next(
        (
            idx for idx, record in enumerate(frame_records)
            if record.get("state_level") in ("low", "attention")
        ),
        None,
    )
    if warning_index is not None:
        candidates.append(("预警相关帧", warning_index))

    keyframes = []
    seen = set()
    for label, index in candidates:
        if index in seen or index < 0 or index >= len(frame_records):
            continue
        seen.add(index)
        keyframes.append({
            "label": label,
            "record": frame_records[index],
            "image": frame_images[index],
        })

    return keyframes


def analyze_video(
    video_path,
    recognizer,
    config: AnalysisConfig,
    sample_every_seconds: float = 1.0,
    max_frames: int = 60,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    分析视频并输出整体统计和时序结果。

    Args:
        video_path: 视频文件路径
        recognizer: 表情识别器实例
        config: 单帧分析配置
        sample_every_seconds: 采样间隔，默认 1 秒 1 帧
        max_frames: 最多分析的采样帧数量
        progress_callback: 可选进度回调 (analyzed_count, max_frames)
    """
    start = perf_counter()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_interval = max(1, int(round(fps * sample_every_seconds)))
    overall_results = []
    frame_records = []
    frame_images = []
    frame_count = 0
    analyzed_count = 0
    last_result_image = None
    last_frame = None

    while cap.isOpened() and analyzed_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            analyzed_count += 1
            time_sec = frame_count / fps
            analysis = analyze_frame(frame, recognizer, config)
            overall_results.extend(analysis["recognition_results"])
            frame_records.append(
                _build_frame_record(analyzed_count, frame_count, time_sec, analysis)
            )
            last_result_image = analysis["result_image"]
            last_frame = frame
            frame_images.append(analysis["result_image"])

            if progress_callback:
                progress_callback(analyzed_count, max_frames)

        frame_count += 1

    cap.release()

    overall_stats = calculate_statistics(overall_results)
    summary = _build_video_summary(frame_records, len(overall_results))
    state_text = summary["state_text"]
    state_level = summary["state_level"]
    warning = calculate_temporal_warning(frame_records)
    elapsed_sec = perf_counter() - start

    performance = {
        "total_video_frames": total_frames,
        "fps": round(fps, 2),
        "sample_interval_frames": frame_interval,
        "sample_every_seconds": sample_every_seconds,
        "analyzed_frames": analyzed_count,
        "face_instances": len(overall_results),
        "avg_people_per_frame": summary["avg_people_per_frame"],
        "max_people_per_frame": summary["max_people_per_frame"],
        "total_elapsed_sec": round(elapsed_sec, 2),
        "avg_elapsed_ms_per_frame": round((elapsed_sec * 1000 / analyzed_count), 2)
        if analyzed_count > 0 else 0.0,
    }

    return {
        "overall_results": overall_results,
        "overall_stats": overall_stats,
        "video_summary": summary,
        "state_text": state_text,
        "state_level": state_level,
        "frame_records": frame_records,
        "warning": warning,
        "performance": performance,
        "keyframes": _select_keyframes(frame_records, frame_images),
        "last_result_image": last_result_image,
        "last_frame": last_frame,
    }
