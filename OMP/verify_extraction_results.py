"""
散射中心提取结果验证脚本
=========================================

用于验证修复后的算法是否能正确提取MSTAR图像中央车辆目标的散射中心。
包括：
1. 数据加载验证
2. 坐标系一致性验证
3. 散射中心位置合理性检查
4. 与车辆目标区域的匹配度分析
"""

import numpy as np
import matplotlib.pyplot as plt
from asc_extraction_fixed_v2 import ASCExtractionFixedV2, verify_coordinate_system
from demo_high_precision import find_best_mstar_file
import time


def analyze_target_region(complex_image, threshold_db=10):
    """
    分析MSTAR图像中的目标区域
    返回高强度区域的边界框，用于验证散射中心是否落在目标上
    """
    magnitude = np.abs(complex_image)
    max_val = np.max(magnitude)
    threshold = max_val / (10 ** (threshold_db / 20))

    # 找到高强度区域
    high_intensity_mask = magnitude > threshold

    # 找到边界框
    rows, cols = np.where(high_intensity_mask)
    if len(rows) == 0:
        return None

    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # 转换为归一化坐标 [-1, 1]
    img_h, img_w = complex_image.shape

    # 像素坐标转归一化坐标
    x_min = (min_col / img_w) * 2 - 1
    x_max = (max_col / img_w) * 2 - 1
    y_min = (min_row / img_h) * 2 - 1
    y_max = (max_row / img_h) * 2 - 1

    return {
        "x_range": (x_min, x_max),
        "y_range": (y_min, y_max),
        "center": ((x_min + x_max) / 2, (y_min + y_max) / 2),
        "coverage_ratio": np.sum(high_intensity_mask) / (img_h * img_w),
    }


def check_scatterers_on_target(scatterers, target_region):
    """
    检查散射中心是否落在目标区域内
    """
    if target_region is None:
        return 0, []

    x_min, x_max = target_region["x_range"]
    y_min, y_max = target_region["y_range"]

    on_target_count = 0
    on_target_scatterers = []

    for sc in scatterers:
        x, y = sc["x"], sc["y"]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            on_target_count += 1
            on_target_scatterers.append(sc)

    return on_target_count, on_target_scatterers


def analyze_scatterer_distribution(scatterers):
    """
    分析散射中心的分布特征
    """
    if not scatterers:
        return {}

    positions = [(sc["x"], sc["y"]) for sc in scatterers]
    amplitudes = [sc["estimated_amplitude"] for sc in scatterers]

    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    analysis = {
        "count": len(scatterers),
        "x_center": np.mean(x_coords),
        "y_center": np.mean(y_coords),
        "x_std": np.std(x_coords),
        "y_std": np.std(y_coords),
        "avg_amplitude": np.mean(amplitudes),
        "max_amplitude": np.max(amplitudes),
        "amplitude_std": np.std(amplitudes),
        "positions": positions,
        "amplitudes": amplitudes,
    }

    return analysis


def visualize_verification_results(complex_image, scatterers, target_region, save_path="verification_result.png"):
    """
    创建详细的验证可视化图
    """
    magnitude = np.abs(complex_image)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 原始SAR图像
    ax1 = axes[0, 0]
    im1 = ax1.imshow(magnitude, cmap="gray", extent=(-1, 1, -1, 1), origin="lower")
    ax1.set_title("Original SAR Image")
    ax1.set_xlabel("X (Normalized)")
    ax1.set_ylabel("Y (Normalized)")
    plt.colorbar(im1, ax=ax1)

    # 2. 目标区域 + 散射中心
    ax2 = axes[0, 1]
    ax2.imshow(magnitude, cmap="gray", extent=(-1, 1, -1, 1), origin="lower", alpha=0.7)

    # 绘制目标区域边界框
    if target_region:
        x_min, x_max = target_region["x_range"]
        y_min, y_max = target_region["y_range"]
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="red",
            linewidth=2,
            label="Target Region",
        )
        ax2.add_patch(rect)

    # 绘制散射中心
    if scatterers:
        x_coords = [sc["x"] for sc in scatterers]
        y_coords = [sc["y"] for sc in scatterers]
        amplitudes = [sc["estimated_amplitude"] for sc in scatterers]

        scatter = ax2.scatter(
            x_coords, y_coords, c=amplitudes, s=50, cmap="hot", alpha=0.8, edgecolors="white", linewidth=1
        )
        plt.colorbar(scatter, ax=ax2, label="Amplitude")

    ax2.set_title(f"Scatterers on Target (Total: {len(scatterers)})")
    ax2.set_xlabel("X (Normalized)")
    ax2.set_ylabel("Y (Normalized)")
    ax2.legend()

    # 3. 散射中心分布分析
    ax3 = axes[1, 0]
    if scatterers:
        amplitudes = [sc["estimated_amplitude"] for sc in scatterers]
        ax3.hist(amplitudes, bins=min(20, len(scatterers)), alpha=0.7)
        ax3.set_title("Amplitude Distribution")
        ax3.set_xlabel("Amplitude")
        ax3.set_ylabel("Count")
        ax3.axvline(np.mean(amplitudes), color="red", linestyle="--", label=f"Mean: {np.mean(amplitudes):.3f}")
        ax3.legend()

    # 4. 位置集中度分析
    ax4 = axes[1, 1]
    if scatterers and len(scatterers) > 1:
        x_coords = [sc["x"] for sc in scatterers]
        y_coords = [sc["y"] for sc in scatterers]

        # 计算到中心的距离
        center_x, center_y = np.mean(x_coords), np.mean(y_coords)
        distances = [np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in zip(x_coords, y_coords)]

        ax4.hist(distances, bins=min(15, len(scatterers)), alpha=0.7)
        ax4.set_title("Distance from Center Distribution")
        ax4.set_xlabel("Distance from Center")
        ax4.set_ylabel("Count")
        ax4.axvline(np.mean(distances), color="red", linestyle="--", label=f"Mean: {np.mean(distances):.3f}")
        ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"✅ 验证结果已保存到: {save_path}")
    plt.show()


def main():
    """
    主验证流程
    """
    print("🔍 MSTAR车辆目标散射中心提取验证")
    print("=" * 60)

    # 1. 数据加载
    mstar_file = find_best_mstar_file()
    if not mstar_file:
        print("❌ 无法找到MSTAR数据文件")
        return

    print(f"📂 使用数据文件: {mstar_file}")

    # 2. 初始化提取器并验证坐标系
    extractor = ASCExtractionFixedV2(
        image_size=(128, 128), extraction_mode="point_only", adaptive_threshold=0.05, max_scatterers=20
    )

    magnitude, complex_image = extractor.load_mstar_data_robust(mstar_file)

    print("\n🔧 验证坐标系修复...")
    if not verify_coordinate_system(extractor):
        print("❌ 坐标系验证失败")
        return

    # 3. 分析目标区域
    print("\n🎯 分析车辆目标区域...")
    target_region = analyze_target_region(complex_image, threshold_db=10)

    if target_region:
        print(f"   目标区域中心: ({target_region['center'][0]:.3f}, {target_region['center'][1]:.3f})")
        print(f"   目标覆盖率: {target_region['coverage_ratio']*100:.2f}%")
    else:
        print("   ⚠️ 未检测到明显的目标区域")

    # 4. 提取散射中心
    print("\n🚀 提取散射中心...")
    start_time = time.time()
    scatterers = extractor.extract_asc_scatterers_v2(complex_image)
    extraction_time = time.time() - start_time

    print(f"   提取时间: {extraction_time:.2f}秒")

    # 5. 验证结果
    print("\n📊 验证提取结果...")

    if not scatterers:
        print("❌ 未提取到任何散射中心")
        return

    # 散射中心分布分析
    distribution = analyze_scatterer_distribution(scatterers)
    print(f"   散射中心数量: {distribution['count']}")
    print(f"   中心位置: ({distribution['x_center']:.3f}, {distribution['y_center']:.3f})")
    print(f"   位置标准差: X={distribution['x_std']:.3f}, Y={distribution['y_std']:.3f}")
    print(f"   平均幅度: {distribution['avg_amplitude']:.3f}")

    # 检查散射中心是否在目标上
    if target_region:
        on_target_count, on_target_scatterers = check_scatterers_on_target(scatterers, target_region)
        target_ratio = on_target_count / len(scatterers) * 100
        print(f"   目标区域内散射中心: {on_target_count}/{len(scatterers)} ({target_ratio:.1f}%)")

        if target_ratio > 70:
            print("   ✅ 散射中心主要集中在目标区域内")
        elif target_ratio > 40:
            print("   ⚠️ 散射中心部分集中在目标区域内")
        else:
            print("   ❌ 散射中心未有效集中在目标区域内")

    # 6. 质量评估
    print("\n🎖️ 提取质量评估:")
    quality_score = 0

    # 数量合理性 (10-30个散射中心为合理)
    if 10 <= len(scatterers) <= 30:
        quality_score += 25
        print("   ✅ 散射中心数量合理 (+25分)")
    else:
        print("   ⚠️ 散射中心数量可能不合理")

    # 集中度 (标准差小于0.3为集中)
    if distribution["x_std"] < 0.3 and distribution["y_std"] < 0.3:
        quality_score += 25
        print("   ✅ 散射中心位置集中 (+25分)")
    else:
        print("   ⚠️ 散射中心位置分散")

    # 目标匹配度
    if target_region and on_target_count > 0:
        target_ratio = on_target_count / len(scatterers)
        if target_ratio > 0.7:
            quality_score += 30
            print("   ✅ 目标匹配度高 (+30分)")
        elif target_ratio > 0.4:
            quality_score += 15
            print("   ⚠️ 目标匹配度中等 (+15分)")

    # 幅度一致性
    if distribution["amplitude_std"] / distribution["avg_amplitude"] < 0.5:
        quality_score += 20
        print("   ✅ 幅度分布合理 (+20分)")
    else:
        print("   ⚠️ 幅度分布不均匀")

    print(f"\n🏆 总体质量评分: {quality_score}/100")

    if quality_score >= 80:
        print("   ✅ 算法修复成功，散射中心提取质量优秀")
    elif quality_score >= 60:
        print("   ⚠️ 算法基本修复，散射中心提取质量良好")
    else:
        print("   ❌ 算法仍需改进，散射中心提取质量不佳")

    # 7. 可视化结果
    print("\n🎨 生成验证可视化...")
    visualize_verification_results(complex_image, scatterers, target_region)

    print("\n✅ 验证完成")


if __name__ == "__main__":
    main()
