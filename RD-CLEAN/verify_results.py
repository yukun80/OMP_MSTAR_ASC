#!/usr/bin/env python3
"""
RD-CLEAN结果验证脚本

加载算法提取的散射中心参数，分析并可视化结果
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 添加src路径
sys.path.insert(0, "src")

from rd_clean_algorithm import ScattererParameters, RDCleanAlgorithm


def load_and_analyze_results(result_file: str):
    """
    加载并分析散射中心提取结果

    Args:
        result_file: 结果文件路径
    """
    print(f"=== 分析结果文件: {result_file} ===\n")

    # 加载结果
    with open(result_file, "rb") as f:
        results = pickle.load(f)

    scatterer_list = results.get("scatterer_objects", [])
    scatter_all = results.get("scatter_all", [])
    algorithm_info = results.get("algorithm_info", {})

    print("算法信息:")
    for key, value in algorithm_info.items():
        print(f"  {key}: {value}")

    print(f"\n散射中心统计:")
    print(f"  总数量: {len(scatterer_list)}")

    if len(scatterer_list) == 0:
        print("  没有提取到散射中心！")
        return

    # 按类型统计
    type_counts = {}
    for scatterer in scatterer_list:
        type_name = get_scatterer_type_name(scatterer.type)
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    print("  类型分布:")
    for type_name, count in type_counts.items():
        print(f"    {type_name}: {count}")

    # 参数统计
    positions = [(s.x, s.y) for s in scatterer_list]
    amplitudes = [s.A for s in scatterer_list]
    alphas = [s.alpha for s in scatterer_list]

    print(f"\n参数统计:")
    print(f"  位置范围:")
    print(f"    X: {min(p[0] for p in positions):.3f} ~ {max(p[0] for p in positions):.3f}")
    print(f"    Y: {min(p[1] for p in positions):.3f} ~ {max(p[1] for p in positions):.3f}")
    print(f"  幅度范围: {np.min(amplitudes):.3f} ~ {np.max(amplitudes):.3f}")
    print(f"  Alpha范围: {np.min(alphas):.3f} ~ {np.max(alphas):.3f}")

    # 详细显示前10个散射中心
    print(f"\n前10个散射中心详情:")
    print("ID  |   X     |   Y     | Alpha |   r   |  θ₀   |   L   |   A   | 类型")
    print("-" * 70)
    for i, s in enumerate(scatterer_list[:10]):
        type_name = get_scatterer_type_name(s.type)
        print(
            f"{i+1:2d}  | {s.x:6.3f}  | {s.y:6.3f}  | {s.alpha:5.2f} | {s.r:5.2f} | {s.theta0:5.1f} | {s.L:5.2f} | {s.A:5.2f} | {type_name}"
        )

    # 可视化结果
    visualize_results(scatterer_list, result_file)


def get_scatterer_type_name(scatterer_type: int) -> str:
    """获取散射中心类型名称"""
    type_names = {0: "分布式", 1: "局部", 2: "多峰"}
    return type_names.get(scatterer_type, "未知")


def visualize_results(scatterer_list, result_file: str):
    """
    可视化散射中心结果

    Args:
        scatterer_list: 散射中心列表
        result_file: 结果文件名（用于保存图片）
    """
    if len(scatterer_list) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"RD-CLEAN 散射中心提取结果\n文件: {os.path.basename(result_file)}", fontsize=14)

    # 1. 散射中心位置分布
    ax1 = axes[0, 0]
    positions = [(s.x, s.y) for s in scatterer_list]
    amplitudes = [s.A for s in scatterer_list]
    types = [s.type for s in scatterer_list]

    # 按类型用不同颜色和形状显示
    type_colors = {0: "red", 1: "blue", 2: "green"}
    type_markers = {0: "s", 1: "o", 2: "^"}
    type_names = {0: "分布式", 1: "局部", 2: "多峰"}

    for type_id in set(types):
        type_positions = [(s.x, s.y) for s in scatterer_list if s.type == type_id]
        type_amplitudes = [s.A for s in scatterer_list if s.type == type_id]

        if type_positions:
            x_coords, y_coords = zip(*type_positions)
            scatter = ax1.scatter(
                x_coords,
                y_coords,
                c=type_amplitudes,
                marker=type_markers[type_id],
                s=60,
                alpha=0.7,
                label=f"{type_names[type_id]} ({len(type_positions)})",
            )

    ax1.set_xlabel("X 坐标 (米)")
    ax1.set_ylabel("Y 坐标 (米)")
    ax1.set_title("散射中心位置分布")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 幅度分布直方图
    ax2 = axes[0, 1]
    ax2.hist(amplitudes, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    ax2.set_xlabel("散射强度 A")
    ax2.set_ylabel("数量")
    ax2.set_title("散射强度分布")
    ax2.grid(True, alpha=0.3)

    # 3. Alpha参数分布
    ax3 = axes[1, 0]
    alphas = [s.alpha for s in scatterer_list]
    ax3.hist(alphas, bins=15, alpha=0.7, color="lightgreen", edgecolor="black")
    ax3.set_xlabel("频率依赖指数 α")
    ax3.set_ylabel("数量")
    ax3.set_title("Alpha参数分布")
    ax3.grid(True, alpha=0.3)

    # 4. 类型分布饼图
    ax4 = axes[1, 1]
    type_counts = {}
    for s in scatterer_list:
        type_name = get_scatterer_type_name(s.type)
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    if type_counts:
        colors = ["lightcoral", "lightskyblue", "lightgreen"]
        ax4.pie(type_counts.values(), labels=type_counts.keys(), autopct="%1.1f%%", colors=colors[: len(type_counts)])
        ax4.set_title("散射中心类型分布")

    plt.tight_layout()

    # 保存图片
    output_name = os.path.splitext(result_file)[0] + "_analysis.png"
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    print(f"\n可视化结果已保存: {output_name}")

    # 显示图片
    plt.show()


def compare_with_original_data(result_file: str, original_raw_file: str):
    """
    与原始数据对比

    Args:
        result_file: 散射中心结果文件
        original_raw_file: 原始.raw文件
    """
    print(f"\n=== 与原始数据对比 ===")

    # 加载结果
    with open(result_file, "rb") as f:
        results = pickle.load(f)
    scatterer_list = results.get("scatterer_objects", [])

    # 创建算法实例重构图像
    algorithm = RDCleanAlgorithm()

    # 加载原始数据
    original_magnitude, original_complex = algorithm.data_loader.load_raw_file(original_raw_file)

    # 重构图像
    reconstructed = algorithm.simulate_scatterers(scatterer_list)

    # 计算重构质量
    original_energy = np.sum(original_magnitude**2)
    residual_energy = np.sum((original_magnitude - reconstructed) ** 2)
    reconstruction_quality = 1 - residual_energy / original_energy if original_energy > 0 else 0

    print(f"重构质量评估:")
    print(f"  原始图像能量: {original_energy:.2e}")
    print(f"  残差能量: {residual_energy:.2e}")
    print(f"  重构质量: {reconstruction_quality:.3f}")

    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(original_magnitude, cmap="hot")
    axes[0].set_title("原始图像")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(reconstructed, cmap="hot")
    axes[1].set_title("重构图像")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1])

    residual = np.abs(original_magnitude - reconstructed)
    im3 = axes[2].imshow(residual, cmap="hot")
    axes[2].set_title("残差图像")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()

    # 保存对比图
    comparison_name = os.path.splitext(result_file)[0] + "_comparison.png"
    plt.savefig(comparison_name, dpi=300, bbox_inches="tight")
    print(f"对比图已保存: {comparison_name}")

    plt.show()


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python verify_results.py <result_file.pkl> [original_raw_file]")
        print("示例: python verify_results.py test_results/HB03344.017.128x128_scatterers.pkl")
        return

    result_file = sys.argv[1]
    original_raw_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(result_file):
        print(f"错误: 结果文件不存在: {result_file}")
        return

    # 分析结果
    load_and_analyze_results(result_file)

    # 可选的原始数据对比
    if original_raw_file and os.path.exists(original_raw_file):
        compare_with_original_data(result_file, original_raw_file)
    elif original_raw_file:
        print(f"警告: 原始文件不存在: {original_raw_file}")


if __name__ == "__main__":
    main()
