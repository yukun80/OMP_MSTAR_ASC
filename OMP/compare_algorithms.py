"""
算法对比测试 - 两阶段 vs 统一算法
===================================

对比原有的两阶段算法和新的统一算法的性能差异。
验证doc/next_work_goal.md方案一重构的有效性。
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict

from unified_asc_extraction import UnifiedASCExtractor
from run_two_stage_extraction import hypothesize_locations, estimate_parameters_locally
from asc_extraction_fixed_v2 import ASCExtractionFixedV2
from demo_high_precision import find_best_mstar_file


def run_two_stage_algorithm(complex_image):
    """
    运行原有的两阶段算法
    """
    print("\n🔵 运行两阶段算法...")
    start_time = time.time()
    
    try:
        # Stage 1: 使用alpha=0字典假设位置
        locations = hypothesize_locations(complex_image, (128, 128), n_hypotheses=20, position_grid_size=32)
        
        if not locations:
            return [], time.time() - start_time
        
        # Stage 2: 局部参数估计
        scatterers = estimate_parameters_locally(complex_image, locations, (128, 128))
        
        total_time = time.time() - start_time
        return scatterers, total_time
    
    except Exception as e:
        print(f"   ❌ 两阶段算法执行失败: {str(e)}")
        return [], time.time() - start_time


def run_unified_algorithm(complex_image):
    """
    运行新的统一算法
    """
    print("\n🔴 运行统一算法...")
    start_time = time.time()
    
    try:
        # 创建统一提取器
        extractor = UnifiedASCExtractor(
            max_scatterers=20,
            adaptive_threshold=0.05,
            position_samples=32,
            target_focused=True
        )
        
        # 执行提取
        scatterers = extractor.extract_scatterers(complex_image)
        
        total_time = time.time() - start_time
        return scatterers, total_time
        
    except Exception as e:
        print(f"   ❌ 统一算法执行失败: {str(e)}")
        return [], time.time() - start_time


def analyze_target_coverage(scatterers: List[Dict], complex_image: np.ndarray):
    """
    分析散射中心与目标区域的覆盖情况
    """
    if not scatterers:
        return 0, 0
    
    # 简单的目标区域检测 (高强度区域)
    magnitude = np.abs(complex_image)
    max_val = np.max(magnitude)
    threshold = max_val / 10  # 10dB阈值
    
    high_intensity_mask = magnitude > threshold
    rows, cols = np.where(high_intensity_mask)
    
    if len(rows) == 0:
        return 0, 0
    
    # 目标区域边界框 (归一化坐标)
    img_h, img_w = complex_image.shape
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    x_min = (min_col / img_w) * 2 - 1
    x_max = (max_col / img_w) * 2 - 1
    y_min = (min_row / img_h) * 2 - 1
    y_max = (max_row / img_h) * 2 - 1
    
    # 检查散射中心是否在目标区域内
    in_target_count = 0
    for sc in scatterers:
        x, y = sc["x"], sc["y"]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            in_target_count += 1
    
    coverage_ratio = in_target_count / len(scatterers) if scatterers else 0
    return in_target_count, coverage_ratio


def compare_scatterer_quality(scatterers1: List[Dict], scatterers2: List[Dict], 
                            label1: str, label2: str):
    """
    对比两个算法的散射中心质量
    """
    print(f"\n📊 算法对比分析: {label1} vs {label2}")
    print("="*60)
    
    # 基本统计
    count1, count2 = len(scatterers1), len(scatterers2)
    print(f"散射中心数量: {label1}={count1}, {label2}={count2}")
    
    if count1 == 0 and count2 == 0:
        print("两个算法都未提取到散射中心")
        return
    
    # 优化成功率
    if count1 > 0:
        opt_rate1 = sum(1 for s in scatterers1 if s.get("optimization_success", False)) / count1
        print(f"{label1} 优化成功率: {opt_rate1:.1%}")
    else:
        opt_rate1 = 0
        print(f"{label1} 优化成功率: 0% (无散射中心)")
    
    if count2 > 0:
        opt_rate2 = sum(1 for s in scatterers2 if s.get("optimization_success", False)) / count2
        print(f"{label2} 优化成功率: {opt_rate2:.1%}")
    else:
        opt_rate2 = 0
        print(f"{label2} 优化成功率: 0% (无散射中心)")
    
    # 位置集中度
    def calc_position_stats(scatterers):
        if not scatterers:
            return 0, 0, 0, 0
        positions = [(s["x"], s["y"]) for s in scatterers]
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        return np.mean(x_coords), np.mean(y_coords), np.std(x_coords), np.std(y_coords)
    
    x1_mean, y1_mean, x1_std, y1_std = calc_position_stats(scatterers1)
    x2_mean, y2_mean, x2_std, y2_std = calc_position_stats(scatterers2)
    
    print(f"{label1} 位置分布: 中心({x1_mean:.3f}, {y1_mean:.3f}), 标准差({x1_std:.3f}, {y1_std:.3f})")
    print(f"{label2} 位置分布: 中心({x2_mean:.3f}, {y2_mean:.3f}), 标准差({x2_std:.3f}, {y2_std:.3f})")
    
    # 散射类型分布
    def get_type_distribution(scatterers):
        type_dist = {}
        for s in scatterers:
            stype = s.get("scattering_type", "未知")
            type_dist[stype] = type_dist.get(stype, 0) + 1
        return type_dist
    
    type_dist1 = get_type_distribution(scatterers1)
    type_dist2 = get_type_distribution(scatterers2)
    
    print(f"{label1} 散射类型: {type_dist1}")
    print(f"{label2} 散射类型: {type_dist2}")


def main():
    """
    主对比测试函数
    """
    print("🔬 ASC算法对比测试")
    print("="*60)
    print("对比内容:")
    print("  🔵 两阶段算法 (原有实现)")
    print("  🔴 统一算法 (方案一重构)")
    print("="*60)
    
    # 1. 数据准备
    mstar_file = find_best_mstar_file()
    if not mstar_file:
        print("❌ 无法找到MSTAR数据")
        return
    
    print(f"📂 测试数据: {mstar_file}")
    
    # 加载数据
    extractor = ASCExtractionFixedV2(image_size=(128, 128))
    magnitude, complex_image = extractor.load_mstar_data_robust(mstar_file)
    
    # 2. 运行两个算法
    scatterers_two_stage, time_two_stage = run_two_stage_algorithm(complex_image)
    scatterers_unified, time_unified = run_unified_algorithm(complex_image)
    
    # 3. 性能对比
    print(f"\n⏱️ 运行时间对比:")
    print(f"   两阶段算法: {time_two_stage:.2f}s")
    print(f"   统一算法: {time_unified:.2f}s")
    
    # 4. 目标区域覆盖率对比
    in_target1, coverage1 = analyze_target_coverage(scatterers_two_stage, complex_image)
    in_target2, coverage2 = analyze_target_coverage(scatterers_unified, complex_image)
    
    print(f"\n🎯 目标区域覆盖率对比:")
    print(f"   两阶段算法: {in_target1}/{len(scatterers_two_stage)} ({coverage1:.1%})")
    print(f"   统一算法: {in_target2}/{len(scatterers_unified)} ({coverage2:.1%})")
    
    # 5. 详细质量对比
    compare_scatterer_quality(scatterers_two_stage, scatterers_unified, 
                            "两阶段算法", "统一算法")
    
    # 6. 结论
    print(f"\n🏆 对比结论:")
    if coverage2 > coverage1:
        print(f"   ✅ 统一算法目标覆盖率更高 ({coverage2:.1%} vs {coverage1:.1%})")
    else:
        print(f"   ⚠️ 两阶段算法目标覆盖率更高 ({coverage1:.1%} vs {coverage2:.1%})")
    
    if len(scatterers_unified) > 0 and len(scatterers_two_stage) > 0:
        unified_opt_rate = sum(1 for s in scatterers_unified if s.get("optimization_success", False)) / len(scatterers_unified)
        two_stage_opt_rate = sum(1 for s in scatterers_two_stage if s.get("optimization_success", False)) / len(scatterers_two_stage)
        
        if unified_opt_rate > two_stage_opt_rate:
            print(f"   ✅ 统一算法优化成功率更高 ({unified_opt_rate:.1%} vs {two_stage_opt_rate:.1%})")
        else:
            print(f"   ⚠️ 两阶段算法优化成功率更高 ({two_stage_opt_rate:.1%} vs {unified_opt_rate:.1%})")
    
    print(f"\n📝 方案一重构评估:")
    if coverage2 > 0.5 and coverage2 > coverage1:
        print("   ✅ 重构成功！统一算法显著改善了目标区域匹配度")
    elif coverage2 > coverage1:
        print("   ⚠️ 重构有效，统一算法改善了目标区域匹配度，但仍需优化")
    else:
        print("   ❌ 重构效果不明显，需要进一步调整算法参数")


if __name__ == "__main__":
    main() 