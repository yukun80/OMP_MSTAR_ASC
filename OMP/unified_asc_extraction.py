"""
Unified ASC Extraction Algorithm - 基于方案一重构
===============================================

这个脚本实现了基于doc/next_work_goal.md方案一的统一ASC提取算法：
- 抛弃有缺陷的两阶段架构
- 使用包含所有alpha值的全参数字典进行单阶段迭代式"匹配-优化-减去"
- 直接调用asc_extraction_fixed_v2.py中已实现的核心算法

核心理念：让算法在每一步都能使用最匹配的物理模型，避免模型失配问题。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
import os

from asc_extraction_fixed_v2 import ASCExtractionFixedV2, verify_coordinate_system, visualize_extraction_results
from demo_high_precision import find_best_mstar_file


class UnifiedASCExtractor:
    """
    统一ASC提取器 - 实现方案一的单阶段全参数提取策略
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        max_scatterers: int = 20,
        adaptive_threshold: float = 0.05,
        max_iterations: int = 30,
        position_samples: int = 32,
        target_focused: bool = True,
    ):
        """
        初始化统一提取器

        Args:
            image_size: 图像尺寸
            max_scatterers: 最大散射中心数
            adaptive_threshold: 自适应停止阈值 (能量比例)
            max_iterations: 最大迭代次数
            position_samples: 位置采样密度
            target_focused: 是否启用目标导向模式
        """
        self.image_size = image_size
        self.max_scatterers = max_scatterers
        self.adaptive_threshold = adaptive_threshold
        self.max_iterations = max_iterations
        self.position_samples = position_samples
        self.target_focused = target_focused

        print(f"🚀 统一ASC提取器初始化")
        print(f"   算法策略: 单阶段全参数迭代提取")
        print(f"   位置采样: {position_samples}×{position_samples}")
        print(f"   目标导向: {'启用' if target_focused else '禁用'}")

    def create_core_extractor(self) -> ASCExtractionFixedV2:
        """
        创建核心提取器实例 - 关键：使用全参数字典
        """
        if self.target_focused:
            # 目标导向模式：更密集的位置采样，适度的参数范围
            return ASCExtractionFixedV2(
                image_size=self.image_size,
                extraction_mode="progressive",  # 确保使用全参数字典
                adaptive_threshold=self.adaptive_threshold,
                max_iterations=self.max_iterations,
                max_scatterers=self.max_scatterers,
                # 优化的参数配置
                alpha_values=[-1.0, -0.5, 0.0, 0.5, 1.0],  # 包含所有主要散射类型
                length_values=[0.0, 0.1, 0.5],  # 适量的长度参数
                phi_bar_values=[0.0, np.pi / 4, np.pi / 2],  # 主要方向角
                position_samples=self.position_samples,
            )
        else:
            # 标准模式：使用默认的渐进式配置
            return ASCExtractionFixedV2(
                image_size=self.image_size,
                extraction_mode="progressive",
                adaptive_threshold=self.adaptive_threshold,
                max_iterations=self.max_iterations,
                max_scatterers=self.max_scatterers,
                position_samples=self.position_samples,
            )

    def extract_scatterers(self, complex_image: np.ndarray) -> List[Dict]:
        """
        执行统一的ASC散射中心提取

        Args:
            complex_image: 复数SAR图像

        Returns:
            提取到的散射中心列表
        """
        print("\n" + "=" * 70)
        print("🎯 统一ASC提取 - 单阶段全参数策略")
        print("=" * 70)

        # 1. 创建核心提取器 (关键：使用全参数字典)
        print("\n🔧 初始化核心提取器...")
        core_extractor = self.create_core_extractor()

        # 2. 坐标系验证
        print("\n🔍 验证坐标系修复...")
        if not verify_coordinate_system(core_extractor):
            print("❌ 坐标系验证失败，算法可能无法正常工作")
            return []
        print("✅ 坐标系验证通过")

        # 3. 执行核心提取算法 (直接使用已经实现的完整算法)
        print("\n🚀 开始单阶段全参数提取...")
        print("   策略: 迭代式'匹配-优化-减去'循环")
        print("   字典: 包含所有alpha值的全参数原子")

        start_time = time.time()
        scatterers = core_extractor.extract_asc_scatterers_v2(complex_image)
        extraction_time = time.time() - start_time

        # 4. 结果分析
        print(f"\n📊 提取完成 (耗时: {extraction_time:.2f}s)")

        if not scatterers:
            print("❌ 未提取到任何散射中心")
            return []

        # 按幅度排序
        scatterers.sort(key=lambda s: s["estimated_amplitude"], reverse=True)

        # 详细分析
        self._analyze_extraction_results(scatterers)

        return scatterers

    def _analyze_extraction_results(self, scatterers: List[Dict]):
        """
        详细分析提取结果的质量和分布
        """
        print(f"\n📈 提取结果详细分析:")
        print(f"   散射中心总数: {len(scatterers)}")

        # 散射类型分布
        type_dist = {}
        alpha_dist = {}
        opt_success_count = 0

        for sc in scatterers:
            stype = sc["scattering_type"]
            alpha = sc["alpha"]
            type_dist[stype] = type_dist.get(stype, 0) + 1
            alpha_dist[alpha] = alpha_dist.get(alpha, 0) + 1
            if sc.get("optimization_success", False):
                opt_success_count += 1

        print(f"   散射类型分布: {type_dist}")
        print(f"   α参数分布: {alpha_dist}")
        print(f"   优化成功率: {opt_success_count}/{len(scatterers)} ({opt_success_count/len(scatterers)*100:.1f}%)")

        # 位置分布分析
        positions = [(sc["x"], sc["y"]) for sc in scatterers]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        x_center, y_center = np.mean(x_coords), np.mean(y_coords)
        x_std, y_std = np.std(x_coords), np.std(y_coords)

        print(f"   位置中心: ({x_center:.3f}, {y_center:.3f})")
        print(f"   位置标准差: X={x_std:.3f}, Y={y_std:.3f}")

        # 幅度分析
        amplitudes = [sc["estimated_amplitude"] for sc in scatterers]
        print(f"   幅度范围: {np.min(amplitudes):.3f} - {np.max(amplitudes):.3f}")
        print(f"   平均幅度: {np.mean(amplitudes):.3f}")

        # 质量评估
        self._assess_extraction_quality(scatterers, x_std, y_std, opt_success_count)

    def _assess_extraction_quality(self, scatterers: List[Dict], x_std: float, y_std: float, opt_success_count: int):
        """
        评估提取质量
        """
        print(f"\n🏆 质量评估:")

        quality_score = 0

        # 1. 数量合理性 (10-25个为优秀)
        num_scatterers = len(scatterers)
        if 10 <= num_scatterers <= 25:
            quality_score += 25
            print(f"   ✅ 散射中心数量优秀 ({num_scatterers}) +25分")
        elif 5 <= num_scatterers <= 35:
            quality_score += 15
            print(f"   ⚠️ 散射中心数量合理 ({num_scatterers}) +15分")
        else:
            print(f"   ❌ 散射中心数量异常 ({num_scatterers})")

        # 2. 空间集中度 (标准差 < 0.3为优秀)
        spatial_score = 0
        if x_std < 0.25 and y_std < 0.25:
            spatial_score = 25
            print(f"   ✅ 空间集中度优秀 (X:{x_std:.3f}, Y:{y_std:.3f}) +25分")
        elif x_std < 0.4 and y_std < 0.4:
            spatial_score = 15
            print(f"   ⚠️ 空间集中度良好 (X:{x_std:.3f}, Y:{y_std:.3f}) +15分")
        else:
            print(f"   ❌ 空间分布过于分散 (X:{x_std:.3f}, Y:{y_std:.3f})")
        quality_score += spatial_score

        # 3. 优化成功率
        opt_rate = opt_success_count / len(scatterers) if scatterers else 0
        if opt_rate > 0.8:
            quality_score += 30
            print(f"   ✅ 优化成功率优秀 ({opt_rate:.1%}) +30分")
        elif opt_rate > 0.6:
            quality_score += 20
            print(f"   ⚠️ 优化成功率良好 ({opt_rate:.1%}) +20分")
        else:
            print(f"   ❌ 优化成功率偏低 ({opt_rate:.1%})")

        # 4. 物理合理性 (强散射类型占比)
        strong_scattering_count = sum(1 for sc in scatterers if sc["alpha"] in [-1.0, -0.5])
        strong_ratio = strong_scattering_count / len(scatterers) if scatterers else 0
        if strong_ratio > 0.5:
            quality_score += 20
            print(f"   ✅ 强散射类型占比合理 ({strong_ratio:.1%}) +20分")
        elif strong_ratio > 0.3:
            quality_score += 10
            print(f"   ⚠️ 强散射类型占比一般 ({strong_ratio:.1%}) +10分")
        else:
            print(f"   ❌ 强散射类型占比偏低 ({strong_ratio:.1%})")

        print(f"\n🎖️ 总体质量评分: {quality_score}/100")

        if quality_score >= 80:
            print("   ✅ 算法重构成功！散射中心提取质量优秀")
            return "优秀"
        elif quality_score >= 60:
            print("   ⚠️ 算法基本成功，散射中心提取质量良好")
            return "良好"
        else:
            print("   ❌ 算法仍需改进，散射中心提取质量不佳")
            return "需改进"


def main():
    """
    主运行函数 - 统一ASC提取算法演示
    """
    print("🎯 统一ASC提取算法 - 基于方案一重构")
    print("=" * 70)
    print("核心改进：")
    print("  ❌ 移除有缺陷的两阶段架构")
    print("  ✅ 使用全参数字典的单阶段迭代提取")
    print("  ✅ 避免模型失配问题")
    print("=" * 70)

    # 1. 数据准备
    mstar_file = find_best_mstar_file()
    if not mstar_file:
        print("❌ 无法找到MSTAR数据文件")
        return

    print(f"📂 使用数据文件: {mstar_file}")

    # 2. 创建统一提取器
    extractor = UnifiedASCExtractor(
        image_size=(128, 128),
        max_scatterers=20,
        adaptive_threshold=0.05,  # 5% 的严格阈值
        position_samples=32,  # 平衡精度和计算效率
        target_focused=True,  # 启用目标导向模式
    )

    # 3. 加载数据
    print("\n📊 加载MSTAR数据...")
    core_extractor = extractor.create_core_extractor()
    try:
        magnitude, complex_image = core_extractor.load_mstar_data_robust(mstar_file)
        print("✅ 数据加载成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return

    # 4. 执行提取
    scatterers = extractor.extract_scatterers(complex_image)

    # 5. 可视化结果
    if scatterers:
        print("\n🎨 生成提取结果可视化...")
        visualize_extraction_results(complex_image, scatterers, save_path="unified_extraction_result.png")
        print("✅ 结果已保存到: unified_extraction_result.png")
    else:
        print("❌ 未提取到散射中心，无法可视化")

    print(f"\n🎉 统一ASC提取算法运行完成")


if __name__ == "__main__":
    main()
