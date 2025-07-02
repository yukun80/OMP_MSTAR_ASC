"""
统一ASC算法验证脚本
==================

专门验证基于方案一重构的统一ASC算法是否解决了模型失配问题。
"""

import numpy as np
import time
from typing import List, Dict

from unified_asc_extraction import UnifiedASCExtractor
from demo_high_precision import find_best_mstar_file


def detailed_target_analysis(complex_image: np.ndarray, threshold_db: float = 10):
    """详细分析MSTAR图像中的目标区域特征"""
    magnitude = np.abs(complex_image)
    max_val = np.max(magnitude)
    threshold = max_val / (10 ** (threshold_db / 20))

    high_intensity_mask = magnitude > threshold
    rows, cols = np.where(high_intensity_mask)

    if len(rows) == 0:
        return None

    img_h, img_w = complex_image.shape
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    x_min = (min_col / img_w) * 2 - 1
    x_max = (max_col / img_w) * 2 - 1
    y_min = (min_row / img_h) * 2 - 1
    y_max = (max_row / img_h) * 2 - 1

    return {"x_range": (x_min, x_max), "y_range": (y_min, y_max), "center": ((x_min + x_max) / 2, (y_min + y_max) / 2)}


def assess_target_matching(scatterers: List[Dict], target_info: Dict):
    """评估目标区域匹配度"""
    if not scatterers or not target_info:
        return 0, 0

    x_min, x_max = target_info["x_range"]
    y_min, y_max = target_info["y_range"]

    match_count = 0
    for sc in scatterers:
        x, y = sc["x"], sc["y"]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            match_count += 1

    match_ratio = match_count / len(scatterers)
    return match_count, match_ratio


def main():
    """主验证流程"""
    print("🔍 统一ASC算法验证")
    print("=" * 50)

    # 数据加载
    mstar_file = find_best_mstar_file()
    if not mstar_file:
        print("❌ 无法找到MSTAR数据文件")
        return

    # 创建提取器
    extractor = UnifiedASCExtractor(
        max_scatterers=20, adaptive_threshold=0.05, position_samples=32, target_focused=True
    )

    # 加载数据
    core_extractor = extractor.create_core_extractor()
    magnitude, complex_image = core_extractor.load_mstar_data_robust(mstar_file)

    # 分析目标区域
    target_info = detailed_target_analysis(complex_image)

    # 执行提取
    scatterers = extractor.extract_scatterers(complex_image)

    # 验证结果
    if scatterers and target_info:
        match_count, match_ratio = assess_target_matching(scatterers, target_info)
        print(f"\n📊 验证结果:")
        print(f"   散射中心总数: {len(scatterers)}")
        print(f"   目标区域匹配: {match_count}/{len(scatterers)} ({match_ratio:.1%})")

        if match_ratio >= 0.7:
            print("   ✅ 优秀！模型失配问题已解决")
        elif match_ratio >= 0.5:
            print("   ⚠️ 良好！显著改善了目标匹配度")
        else:
            print("   ❌ 仍需改进")
    else:
        print("❌ 验证失败")


if __name__ == "__main__":
    main()
