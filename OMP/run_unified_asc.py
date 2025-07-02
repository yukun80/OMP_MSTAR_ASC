"""
简化运行脚本 - 统一ASC提取算法
====================================

直接调用重构后的统一ASC提取算法的简化版本。
遵循doc/next_work_goal.md方案一的设计理念。
"""

from unified_asc_extraction import UnifiedASCExtractor
from asc_extraction_fixed_v2 import verify_coordinate_system
from demo_high_precision import find_best_mstar_file


def quick_extraction():
    """
    快速ASC提取 - 使用优化的默认参数
    """
    print("🚀 快速ASC提取 - 统一算法")
    print("=" * 50)

    # 1. 数据准备
    mstar_file = find_best_mstar_file()
    if not mstar_file:
        print("❌ 未找到MSTAR数据")
        return None

    # 2. 创建提取器 (使用优化的参数)
    extractor = UnifiedASCExtractor(
        max_scatterers=20, adaptive_threshold=0.05, position_samples=32, target_focused=True
    )

    # 3. 加载数据
    core_extractor = extractor.create_core_extractor()
    magnitude, complex_image = core_extractor.load_mstar_data_robust(mstar_file)

    # 4. 提取散射中心
    scatterers = extractor.extract_scatterers(complex_image)

    return scatterers, complex_image


def main():
    """主函数"""
    scatterers, complex_image = quick_extraction()

    if scatterers:
        print(f"\n✅ 成功提取 {len(scatterers)} 个散射中心")

        # 显示前5个最强的散射中心
        print("\n🏆 前5个最强散射中心:")
        for i, sc in enumerate(scatterers[:5]):
            print(
                f"   {i+1}. 位置:({sc['x']:.3f}, {sc['y']:.3f}), "
                f"类型:{sc['scattering_type']}, "
                f"幅度:{sc['estimated_amplitude']:.3f}"
            )
    else:
        print("❌ 未提取到散射中心")


if __name__ == "__main__":
    main()
