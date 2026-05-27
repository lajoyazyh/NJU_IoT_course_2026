"""
表情统计与课堂状态分析模块
"""
from typing import List, Dict, Tuple

# 表情类别列表（按大作业要求的顺序）
EXPRESSION_CATEGORIES = ['Happy', 'Neutral', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust']
EXPRESSION_CN_MAP = {
    'Happy': '开心',
    'Neutral': '中性',
    'Sad': '悲伤',
    'Angry': '生气',
    'Surprise': '惊讶',
    'Fear': '害怕',
    'Disgust': '厌恶'
}


def calculate_statistics(recognition_results: List[Dict]) -> Dict:
    """
    根据表情识别结果计算统计信息
    Args:
        recognition_results: expression_recognizer.recognize_all() 的返回结果列表
    Returns:
        stats: 统计信息字典
            {
                'total_count': int,           # 总人数
                'expression_count': dict,      # 各类表情人数 {'Happy': 2, ...}
                'expression_ratio': dict,      # 各类表情比例 {'Happy': 0.33, ...}
                'main_expression': str,        # 主要表情（英文）
                'main_expression_cn': str,     # 主要表情（中文）
                'expression_distribution': list # 按顺序的表情数量列表
            }
    """
    total_count = len(recognition_results)
    
    # 初始化计数器
    expression_count = {cat: 0 for cat in EXPRESSION_CATEGORIES}
    
    # 统计各类表情人数
    for result in recognition_results:
        label_en = result.get('label_en', 'Neutral')
        if label_en in expression_count:
            expression_count[label_en] += 1
    
    # 计算比例
    expression_ratio = {}
    for cat in EXPRESSION_CATEGORIES:
        expression_ratio[cat] = round(expression_count[cat] / total_count, 4) if total_count > 0 else 0.0
    
    # 确定主要表情
    main_expression = 'Neutral'
    max_count = 0
    for cat in EXPRESSION_CATEGORIES:
        if expression_count[cat] > max_count:
            max_count = expression_count[cat]
            main_expression = cat
    
    main_expression_cn = EXPRESSION_CN_MAP.get(main_expression, main_expression)
    
    # 按顺序的表情数量列表（方便图表使用）
    expression_distribution = [expression_count[cat] for cat in EXPRESSION_CATEGORIES]
    
    return {
        'total_count': total_count,
        'expression_count': expression_count,
        'expression_ratio': expression_ratio,
        'main_expression': main_expression,
        'main_expression_cn': main_expression_cn,
        'expression_distribution': expression_distribution
    }


def judge_classroom_state(stats: Dict) -> Tuple[str, str]:
    """
    根据表情统计结果判断课堂状态
    Args:
        stats: calculate_statistics() 返回的统计信息字典
    Returns:
        (state_text, state_level)
        state_text: 中文状态描述
        state_level: 'good' | 'stable' | 'attention' | 'low' | 'empty' | 'normal'
    """
    total_count = stats['total_count']
    expression_ratio = stats['expression_ratio']
    expression_count = stats['expression_count']
    main_expression = stats['main_expression']
    
    # 规则 1：检测人数为 0
    if total_count == 0:
        return "未检测到学生", "empty"
    
    # 计算关键指标
    happy_neutral_ratio = expression_ratio['Happy'] + expression_ratio['Neutral']
    sad_angry_ratio = expression_ratio['Sad'] + expression_ratio['Angry']
    surprise_ratio = expression_ratio['Surprise']
    
    # 规则 2：Happy + Neutral 占比 ≥ 70%
    if happy_neutral_ratio >= 0.70:
        return "课堂状态良好 😊", "good"
    
    # 规则 3：Sad + Angry 占比 ≥ 40%
    if sad_angry_ratio >= 0.40:
        return "课堂状态较低落，需要关注 ⚠️", "low"
    
    # 规则 4：Neutral 占比最高
    if main_expression == 'Neutral':
        return "课堂状态平稳 😐", "stable"
    
    # 规则 5：Surprise 占比较高（>30%）
    if surprise_ratio >= 0.30:
        return "课堂注意力波动较大 😮", "attention"
    
    # 规则 6：默认
    return "课堂状态一般", "normal"


def get_state_color(state_level: str) -> str:
    """
    根据状态等级返回对应的颜色（用于界面展示）
    Args:
        state_level: 'good' | 'stable' | 'attention' | 'low' | 'empty' | 'normal'
    Returns:
        颜色代码
    """
    color_map = {
        'good': '#4CAF50',       # 绿色
        'stable': '#2196F3',     # 蓝色
        'attention': '#FF9800',  # 橙色
        'low': '#F44336',        # 红色
        'empty': '#9E9E9E',      # 灰色
        'normal': '#607D8B'      # 蓝灰色
    }
    return color_map.get(state_level, '#607D8B')


def format_stats_for_display(stats: Dict) -> str:
    """
    将统计信息格式化为可展示的字符串
    Args:
        stats: calculate_statistics() 返回的统计信息
    Returns:
        格式化的多行文本
    """
    lines = []
    lines.append(f"检测人数: {stats['total_count']}")
    lines.append("")
    lines.append("表情统计:")
    for cat in EXPRESSION_CATEGORIES:
        count = stats['expression_count'][cat]
        ratio = stats['expression_ratio'][cat]
        cn_name = EXPRESSION_CN_MAP.get(cat, cat)
        lines.append(f"  {cn_name}({cat}): {count} 人 ({ratio*100:.1f}%)")
    lines.append("")
    lines.append(f"主要表情: {stats['main_expression_cn']}({stats['main_expression']})")
    
    return "\n".join(lines)


def format_record_for_csv(timestamp: str, image_name: str, stats: Dict, state_text: str) -> Dict:
    """
    将统计结果格式化为 CSV 记录行
    Args:
        timestamp: 时间戳字符串
        image_name: 图片/视频名称
        stats: 统计信息
        state_text: 课堂状态文本
    Returns:
        包含所有字段的字典
    """
    record = {
        '时间': timestamp,
        '图片名称': image_name,
        '检测人数': stats['total_count'],
    }
    
    for cat in EXPRESSION_CATEGORIES:
        record[cat] = stats['expression_count'][cat]
    
    record['主要表情'] = stats['main_expression']
    record['课堂状态'] = state_text
    
    return record


if __name__ == "__main__":
    # 简单测试
    mock_results = [
        {'label_en': 'Happy', 'confidence': 0.9},
        {'label_en': 'Happy', 'confidence': 0.85},
        {'label_en': 'Neutral', 'confidence': 0.8},
        {'label_en': 'Neutral', 'confidence': 0.75},
        {'label_en': 'Sad', 'confidence': 0.7},
    ]
    
    stats = calculate_statistics(mock_results)
    state_text, state_level = judge_classroom_state(stats)
    
    print(format_stats_for_display(stats))
    print(f"\n课堂状态: {state_text} (等级: {state_level})")
    print(f"状态颜色: {get_state_color(state_level)}")