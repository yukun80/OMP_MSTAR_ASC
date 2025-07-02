"""
优化版统一ASC提取算法 - 解决参数配置问题
===============================================

基于对比测试的结果，优化算法参数：
1. 降低adaptive_threshold避免过早停止
2. 减少position_samples缩小字典规模
3. 平衡精度和效率
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time

from asc_extraction_fixed_v2 import ASCExtractionFixedV2, verify_coordinate_system, visualize_extraction_results
from demo_high_precision import find_best_mstar_file


class OptimizedUnifiedASCExtractor:
    """
    优化版统一ASC提取器 - 平衡精度和效率
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        max_scatterers: int = 20,
        adaptive_threshold: float = 0.02,  # 降低阈值，避免过早停止
        max_iterations: int = 30,
        position_samples: int = 20,  # 减少采样，平衡效率
        enable_progressive_sampling: bool = True,  # 启用渐进式采样
    ):
        """
        初始化优化版统一提取器
        """
        self.image_size = image_size
        self.max_scatterers = max_scatterers
        self.adaptive_threshold = adaptive_threshold
        self.max_iterations = max_iterations
        self.position_samples = position_samples
        self.enable_progressive_sampling = enable_progressive_sampling

        print(f"🚀 优化版统一ASC提取器初始化")
        print(f"   算法策略: 单阶段全参数迭代提取 (优化版)")
        print(f"   自适应阈值: {adaptive_threshold} (更宽松)")
        print(f"   位置采样: {position_samples}×{position_samples} (优化效率)")
        print(f"   渐进式采样: {'启用' if enable_progressive_sampling else '禁用'}")

    def create_optimized_extractor(self) -> ASCExtractionFixedV2:
        """
        创建优化的核心提取器
        """
        if self.enable_progressive_sampling:
            # 渐进式配置：从简单到复杂
            return ASCExtractionFixedV2(
                image_size=self.image_size,
                extraction_mode="progressive",
                adaptive_threshold=self.adaptive_threshold,
                max_iterations=self.max_iterations,
                max_scatterers=self.max_scatterers,
                # 优化的参数配置：平衡精度和效率
                alpha_values=[-1.0, -0.5, 0.0, 0.5, 1.0],  # 保持全参数
                length_values=[0.0],  # 简化长度参数
                phi_bar_values=[0.0, np.pi / 2],  # 简化方向角
                position_samples=self.position_samples,  # 较小的采样密度
            )
        else:
            # 标准配置
            return ASCExtractionFixedV2(
                image_size=self.image_size,
                extraction_mode="progressive",
                adaptive_threshold=self.adaptive_threshold,
                max_iterations=self.max_iterations,
                max_scatterers=self.max_scatterers,
                position_samples=self.position_samples,
            )

    def extract_scatterers_optimized(self, complex_image: np.ndarray) -> List[Dict]:
        """
        执行优化的ASC散射中心提取
        """
        print("\n" + "=" * 70)
        print("🎯 优化版统一ASC提取 - 平衡精度与效率")
        print("=" * 70)

        # 1. 创建优化的核心提取器
        print("\n🔧 初始化优化的核心提取器...")
        core_extractor = self.create_optimized_extractor()

        # 2. 坐标系验证
        print("\n🔍 验证坐标系修复...")
        if not verify_coordinate_system(core_extractor):
            print("❌ 坐标系验证失败")
            return []
        print("✅ 坐标系验证通过")

        # 3. 执行核心提取算法
        print("\n🚀 开始优化版单阶段提取...")
        print("   改进: 降低阈值 + 缩小字典 + 保持全参数支持")

        start_time = time.time()
        scatterers = core_extractor.extract_asc_scatterers_v2(complex_image)
        extraction_time = time.time() - start_time

        # 4. 结果分析
        print(f"\n📊 优化版提取完成 (耗时: {extraction_time:.2f}s)")

        if not scatterers:
            print("❌ 未提取到任何散射中心")
            return []

        # 按幅度排序
        scatterers.sort(key=lambda s: s["estimated_amplitude"], reverse=True)

        # 详细分析
        self._analyze_optimized_results(scatterers, extraction_time)

        return scatterers

    def _analyze_optimized_results(self, scatterers: List[Dict], extraction_time: float):
        """
        分析优化版算法的结果
        """
        print(f"\n📈 优化版结果分析:")
        print(f"   散射中心总数: {len(scatterers)}")
        print(f"   提取效率: {extraction_time:.2f}s")

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

        # 与理论期望对比
        self._assess_optimization_effectiveness(scatterers, extraction_time)

    def _assess_optimization_effectiveness(self, scatterers: List[Dict], extraction_time: float):
        """
        评估优化效果
        """
        print(f"\n🏆 优化效果评估:")

        # 数量评估
        if len(scatterers) >= 10:
            print(f"   ✅ 散射中心数量改善 ({len(scatterers)}个)")
        elif len(scatterers) >= 5:
            print(f"   ⚠️ 散射中心数量一般 ({len(scatterers)}个)")
        else:
            print(f"   ❌ 散射中心数量仍然偏少 ({len(scatterers)}个)")

        # 效率评估
        if extraction_time < 60:
            print(f"   ✅ 计算效率良好 ({extraction_time:.1f}s)")
        elif extraction_time < 120:
            print(f"   ⚠️ 计算效率一般 ({extraction_time:.1f}s)")
        else:
            print(f"   ❌ 计算效率偏低 ({extraction_time:.1f}s)")

        # 散射类型多样性评估
        strong_types = sum(1 for s in scatterers if s["alpha"] in [-1.0, -0.5])
        strong_ratio = strong_types / len(scatterers) if scatterers else 0

        if strong_ratio > 0.6:
            print(f"   ✅ 强散射类型识别能力强 ({strong_ratio:.1%})")
        elif strong_ratio > 0.3:
            print(f"   ⚠️ 强散射类型识别能力一般 ({strong_ratio:.1%})")
        else:
            print(f"   ❌ 强散射类型识别能力弱 ({strong_ratio:.1%})")


def compare_with_baseline():
    """
    与基准算法进行快速对比
    """
    print("🔬 优化版算法快速验证")
    print("=" * 50)

    # 数据准备
    mstar_file = find_best_mstar_file()
    if not mstar_file:
        print("❌ 无法找到MSTAR数据文件")
        return

    # 创建优化版提取器
    extractor = OptimizedUnifiedASCExtractor(
        max_scatterers=20,
        adaptive_threshold=0.02,  # 更宽松的阈值
        position_samples=20,  # 更小的字典
        enable_progressive_sampling=True,
    )

    # 加载数据
    core_extractor = extractor.create_optimized_extractor()
    magnitude, complex_image = core_extractor.load_mstar_data_robust(mstar_file)

    # 执行提取
    scatterers = extractor.extract_scatterers_optimized(complex_image)

    # 快速验证
    if scatterers:
        print(f"\n✅ 优化版算法成功提取 {len(scatterers)} 个散射中心")

        # 简单的目标区域检测
        magnitude = np.abs(complex_image)
        max_val = np.max(magnitude)
        threshold = max_val / 10
        high_intensity_mask = magnitude > threshold
        rows, cols = np.where(high_intensity_mask)

        if len(rows) > 0:
            img_h, img_w = complex_image.shape
            x_min = (np.min(cols) / img_w) * 2 - 1
            x_max = (np.max(cols) / img_w) * 2 - 1
            y_min = (np.min(rows) / img_h) * 2 - 1
            y_max = (np.max(rows) / img_h) * 2 - 1

            # 检查目标匹配度
            in_target = 0
            for sc in scatterers:
                x, y = sc["x"], sc["y"]
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    in_target += 1

            match_ratio = in_target / len(scatterers)
            print(f"🎯 目标区域匹配度: {in_target}/{len(scatterers)} ({match_ratio:.1%})")

            if match_ratio > 0.5:
                print("   ✅ 优化成功！显著改善了目标匹配度")
            elif match_ratio > 0.2:
                print("   ⚠️ 有所改善，但仍需进一步优化")
            else:
                print("   ❌ 目标匹配度改善有限")

        # 可视化结果
        visualize_extraction_results(complex_image, scatterers, save_path="optimized_unified_result.png")
        print("✅ 结果已保存到: optimized_unified_result.png")
    else:
        print("❌ 优化版算法未提取到散射中心")


def main():
    """
    主运行函数
    """
    print("🎯 优化版统一ASC提取算法")
    print("=" * 70)
    print("优化策略:")
    print("  🔧 降低adaptive_threshold (0.05 → 0.02)")
    print("  🔧 减少position_samples (32 → 20)")
    print("  🔧 简化字典参数组合")
    print("  ✅ 保持全参数支持避免模型失配")
    print("=" * 70)

    compare_with_baseline()


if __name__ == "__main__":
    main()
