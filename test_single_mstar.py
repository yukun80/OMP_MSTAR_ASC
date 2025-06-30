#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单个MSTAR文件OMP测试脚本
Single MSTAR File OMP Testing Script

在批量处理之前，先用单个文件测试算法性能
"""

import numpy as np
import matplotlib.pyplot as plt
from omp_asc_final import OMPASCFinal
import os
import time


def test_single_mstar_file():
    """测试单个MSTAR文件的OMP处理"""
    print("🧪 单个MSTAR文件OMP测试")
    print("=" * 50)

    # 查找第一个可用的RAW文件
    raw_data_dir = "datasets/SAR_ASC_Project/02_Data_Processed_raw/SN_S7"

    if not os.path.exists(raw_data_dir):
        print(f"❌ 数据目录不存在: {raw_data_dir}")
        return None

    raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".raw")]

    if not raw_files:
        print(f"❌ 未找到RAW文件在: {raw_data_dir}")
        return None

    # 选择第一个文件进行测试
    test_file = os.path.join(raw_data_dir, raw_files[0])
    print(f"📁 测试文件: {raw_files[0]}")
    print(f"📁 完整路径: {test_file}")

    # 初始化OMP算法
    print(f"\n🔧 初始化OMP算法...")
    omp_asc = OMPASCFinal(n_scatterers=40, image_size=(128, 128), use_cv=False)

    try:
        start_time = time.time()

        # 步骤1: 加载数据
        print(f"\n📂 步骤1: 加载SAR数据...")
        magnitude, complex_image = omp_asc.load_raw_data(test_file)

        # 步骤2: 预处理
        print(f"⚙️  步骤2: 数据预处理...")
        signal = omp_asc.preprocess_data(complex_image)

        # 步骤3: 构建字典 (快速配置用于测试)
        print(f"📚 步骤3: 构建SAR字典...")
        dictionary, param_grid = omp_asc.build_dictionary(position_grid_size=8, phase_levels=4)  # 快速配置

        # 步骤4: 提取散射中心
        print(f"🎯 步骤4: OMP散射中心提取...")
        results = omp_asc.extract_scatterers(signal)

        # 步骤5: 重构图像
        print(f"🔄 步骤5: 图像重构...")
        reconstructed = omp_asc.reconstruct_image(results["scatterers"])

        processing_time = time.time() - start_time

        # 计算质量指标
        mse = np.mean((magnitude - np.abs(reconstructed)) ** 2)
        max_val = np.max(magnitude)
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float("inf")

        # 显示结果
        print(f"\n✅ 处理完成！")
        print(f"📊 处理结果:")
        print(f"   ⏱️  处理时间: {processing_time:.2f}秒")
        print(f"   🎯 提取散射中心: {len(results['scatterers'])}")
        print(f"   📏 字典大小: {dictionary.shape[1]}")
        print(f"   📈 重构PSNR: {psnr:.2f} dB")
        print(f"   📉 重构误差: {results['reconstruction_error']:.3f}")

        # 可视化结果
        visualize_test_results(magnitude, reconstructed, results["scatterers"], raw_files[0], psnr, processing_time)

        return {
            "file_name": raw_files[0],
            "processing_time": processing_time,
            "scatterers": results["scatterers"],
            "psnr": psnr,
            "reconstruction_error": results["reconstruction_error"],
            "dictionary_size": dictionary.shape[1],
        }

    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def visualize_test_results(magnitude, reconstructed, scatterers, file_name, psnr, processing_time):
    """可视化测试结果"""
    print(f"\n🎨 生成可视化结果...")

    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"MSTAR文件OMP测试结果 - {file_name}", fontsize=16, fontweight="bold")

    # 原始SAR图像
    im1 = axes[0, 0].imshow(magnitude, cmap="gray")
    axes[0, 0].set_title("原始SAR幅度图", fontsize=12)
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    # OMP重构图像
    im2 = axes[0, 1].imshow(np.abs(reconstructed), cmap="gray")
    axes[0, 1].set_title(f"OMP重构图像\nPSNR: {psnr:.1f} dB", fontsize=12)
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

    # 重构误差
    diff = magnitude - np.abs(reconstructed)
    im3 = axes[0, 2].imshow(diff, cmap="seismic")
    axes[0, 2].set_title("重构误差", fontsize=12)
    axes[0, 2].axis("off")
    plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)

    # 散射中心位置分布
    if scatterers:
        x_pos = [s["x"] for s in scatterers]
        y_pos = [s["y"] for s in scatterers]
        amplitudes = [s["estimated_amplitude"] for s in scatterers]

        scatter = axes[1, 0].scatter(x_pos, y_pos, c=amplitudes, s=80, cmap="viridis", alpha=0.8)
        axes[1, 0].set_title(f"散射中心位置\n({len(scatterers)}个)", fontsize=12)
        axes[1, 0].set_xlabel("X位置 (归一化)")
        axes[1, 0].set_ylabel("Y位置 (归一化)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(-1.1, 1.1)
        axes[1, 0].set_ylim(-1.1, 1.1)
        plt.colorbar(scatter, ax=axes[1, 0], shrink=0.8, label="幅度")

        # 幅度分布
        axes[1, 1].hist(amplitudes, bins=15, alpha=0.7, edgecolor="black", color="skyblue")
        axes[1, 1].set_title("散射中心幅度分布", fontsize=12)
        axes[1, 1].set_xlabel("幅度")
        axes[1, 1].set_ylabel("数量")
        axes[1, 1].grid(True, alpha=0.3)

    # 处理信息总结
    info_text = f"""测试结果总结:

📁 文件: {file_name}
⏱️  处理时间: {processing_time:.2f}秒
🎯 散射中心数: {len(scatterers)}
📈 重构PSNR: {psnr:.2f} dB
📊 处理状态: 成功

📍 前5强散射中心:"""

    if scatterers:
        for i, scatterer in enumerate(scatterers[:5]):
            info_text += f"\n{i+1}. ({scatterer['x']:.2f}, {scatterer['y']:.2f})"
            info_text += f" A={scatterer['estimated_amplitude']:.3f}"

    axes[1, 2].text(
        0.05,
        0.95,
        info_text,
        transform=axes[1, 2].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
    )
    axes[1, 2].axis("off")

    plt.tight_layout()

    # 保存结果
    os.makedirs("datasets/SAR_ASC_Project/test_results", exist_ok=True)
    save_path = f"datasets/SAR_ASC_Project/test_results/single_test_{file_name.replace('.raw', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"   💾 可视化结果保存: {save_path}")

    plt.show()


def quick_performance_check():
    """快速性能检查"""
    print(f"\n🚀 执行快速性能检查...")

    result = test_single_mstar_file()

    if result is None:
        print(f"❌ 测试失败，无法进行性能评估")
        return False

    print(f"\n📊 性能评估:")

    # 评估标准
    criteria = {
        "处理时间": result["processing_time"] < 60,  # 小于60秒
        "PSNR质量": result["psnr"] > 20,  # 大于20dB
        "散射中心数": len(result["scatterers"]) > 10,  # 提取到散射中心
        "重构误差": result["reconstruction_error"] < 10,  # 合理的重构误差
    }

    passed = 0
    for criterion, status in criteria.items():
        symbol = "✅" if status else "❌"
        print(f"   {symbol} {criterion}: {'通过' if status else '失败'}")
        if status:
            passed += 1

    success_rate = passed / len(criteria)
    print(f"\n📈 总体评估: {success_rate:.1%} ({passed}/{len(criteria)})")

    if success_rate >= 0.75:
        print(f"🎉 算法性能优秀！可以进行批量处理。")
        recommendation = "proceed"
    elif success_rate >= 0.5:
        print(f"✅ 算法性能良好，可以继续处理。")
        recommendation = "proceed_with_caution"
    else:
        print(f"⚠️  算法性能需要优化，建议检查配置。")
        recommendation = "review_settings"

    return recommendation


def main():
    """主函数"""
    print("🧪 MSTAR单文件OMP测试")
    print("=" * 60)
    print("在批量处理之前进行快速验证")
    print("=" * 60)

    # 执行性能检查
    recommendation = quick_performance_check()

    print(f"\n💡 下一步建议:")
    if recommendation == "proceed":
        print(f"   ✅ 算法测试通过，可以运行批量处理:")
        print(f"      python process_mstar_data.py")
    elif recommendation == "proceed_with_caution":
        print(f"   ⚠️  算法基本正常，建议调整参数后批量处理")
        print(f"      可以继续运行: python process_mstar_data.py")
    else:
        print(f"   🔧 建议检查算法配置后再进行批量处理")

    return recommendation


if __name__ == "__main__":
    result = main()
