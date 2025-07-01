#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSTAR高精度ASC提取测试
测试真实MSTAR数据的高精度ASC提取效果
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings

warnings.filterwarnings("ignore")

# 导入模块
try:
    from asc_extraction_advanced import ASCExtractionAdvanced
    from omp_asc_final import OMPASCExtractor
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def test_mstar_precision():
    """测试真实MSTAR数据的高精度ASC提取"""
    print("🎯 MSTAR高精度ASC提取测试")
    print("=" * 60)

    # 寻找MSTAR数据文件
    mstar_dir = "datasets/SAR_ASC_Project/02_Data_Processed_raw/SN_S7"
    if not os.path.exists(mstar_dir):
        print(f"❌ MSTAR数据目录不存在: {mstar_dir}")
        return

    # 获取第一个MSTAR文件
    mstar_files = [f for f in os.listdir(mstar_dir) if f.endswith(".raw")]
    if not mstar_files:
        print(f"❌ 在 {mstar_dir} 中未找到.raw文件")
        return

    test_file = os.path.join(mstar_dir, mstar_files[0])
    print(f"📂 测试文件: {test_file}")

    # 创建结果目录
    results_dir = "results/mstar_precision_test"
    os.makedirs(results_dir, exist_ok=True)

    # 测试传统OMP系统 (高精度设置)
    print("\n🔄 测试传统OMP系统 (高精度设置)...")
    omp_results = test_traditional_omp_precision(test_file)

    # 测试高级ASC系统 (高精度设置)
    print("\n🎯 测试高级ASC系统 (高精度设置)...")
    asc_results = test_advanced_asc_precision(test_file)

    # 生成对比分析
    print("\n📊 生成精度对比分析...")
    create_precision_comparison(omp_results, asc_results, results_dir)

    print(f"\n✅ MSTAR高精度测试完成!")
    print(f"   结果保存在: {results_dir}")


def test_traditional_omp_precision(test_file):
    """高精度传统OMP测试"""
    start_time = time.time()

    try:
        # 初始化高精度OMP
        omp_system = OMPASCExtractor(n_scatterers=40)

        # 加载MSTAR数据
        magnitude, complex_image = omp_system.load_raw_data(test_file)
        signal = omp_system.preprocess_data(complex_image)

        # 高精度字典构建 (大幅提升精度)
        print("   构建高精度字典...")
        dictionary, param_grid = omp_system.build_dictionary(
            position_grid_size=64, phase_levels=16  # 从默认32提升到64  # 从默认8提升到16
        )

        # 提取散射中心
        scatterers = omp_system.extract_scatterers(signal, dictionary, param_grid)

        # 重构图像
        reconstructed = omp_system.reconstruct_image(scatterers)

        processing_time = time.time() - start_time

        # 计算性能指标
        psnr = calculate_psnr(magnitude, np.abs(reconstructed))

        results = {
            "success": True,
            "system_type": "传统OMP (高精度)",
            "scatterers": scatterers,
            "magnitude": magnitude,
            "reconstructed": reconstructed,
            "num_scatterers": len(scatterers),
            "processing_time": processing_time,
            "psnr": psnr,
            "position_grid_size": 64,
            "phase_levels": 16,
            "dictionary_size": dictionary.shape[1],
        }

        print(f"   ✅ 传统OMP (高精度): {len(scatterers)} 个散射中心")
        print(f"   📊 PSNR: {psnr:.2f} dB")
        print(f"   ⏱️ 处理时间: {processing_time:.1f}s")
        print(f"   🔍 字典规模: {dictionary.shape[1]} 个原子")

        return results

    except Exception as e:
        print(f"   ❌ 传统OMP (高精度) 失败: {str(e)}")
        return {"success": False, "error": str(e)}


def test_advanced_asc_precision(test_file):
    """高精度ASC测试"""
    start_time = time.time()

    try:
        # 初始化高精度ASC系统
        asc_system = ASCExtractionAdvanced(
            adaptive_threshold=0.005,  # 更严格的阈值 (0.5%)
            max_iterations=50,
            min_scatterers=3,
            max_scatterers=25,
            precision_mode="high",  # 使用高精度模式
        )

        # 加载MSTAR数据
        magnitude, complex_image = asc_system.load_raw_data(test_file)
        signal = asc_system.preprocess_data(complex_image)

        # 高精度字典构建（不需要手动指定参数，自动使用precision_mode配置）
        print("   构建高精度ASC字典...")
        dictionary, param_grid = asc_system.build_asc_dictionary()

        # 自适应提取
        scatterers = asc_system.adaptive_asc_extraction(signal, dictionary, param_grid)

        # 参数精化（如果启用）
        if asc_system.enable_refinement:
            print("   执行参数精化...")
            refined_scatterers = asc_system.refine_parameters(scatterers, signal)
        else:
            refined_scatterers = scatterers

        # 重构图像
        reconstructed = asc_system.reconstruct_asc_image(refined_scatterers)

        processing_time = time.time() - start_time

        # 分析结果
        analysis = asc_system.analyze_asc_results(refined_scatterers)
        psnr = calculate_psnr(magnitude, np.abs(reconstructed))

        results = {
            "success": True,
            "system_type": "高级ASC (高精度)",
            "scatterers": refined_scatterers,
            "magnitude": magnitude,
            "reconstructed": reconstructed,
            "num_scatterers": len(refined_scatterers),
            "processing_time": processing_time,
            "psnr": psnr,
            "position_samples": asc_system.position_samples,
            "azimuth_samples": asc_system.azimuth_samples,
            "precision_mode": asc_system.precision_mode,
            "dictionary_size": dictionary.shape[1],
            "analysis": analysis,
            "optimization_success_rate": analysis.get("optimization_success_rate", 0),
        }

        print(f"   ✅ 高级ASC (高精度): {len(refined_scatterers)} 个散射中心")
        print(f"   📊 PSNR: {psnr:.2f} dB")
        print(f"   ⏱️ 处理时间: {processing_time:.1f}s")
        print(f"   🔍 字典规模: {dictionary.shape[1]} 个原子")
        if asc_system.enable_refinement:
            print(f"   🎯 优化成功率: {analysis.get('optimization_success_rate', 0):.1%}")
        print(f"   🔬 α分布: {analysis.get('alpha_distribution', {})}")

        return results

    except Exception as e:
        print(f"   ❌ 高级ASC (高精度) 失败: {str(e)}")
        return {"success": False, "error": str(e)}


def calculate_psnr(original, reconstructed):
    """计算PSNR"""
    if original.shape != reconstructed.shape:
        return 0.0

    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")

    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def create_precision_comparison(omp_results, asc_results, results_dir):
    """创建高精度对比可视化"""

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("MSTAR数据高精度ASC提取对比分析", fontsize=16, fontweight="bold")

    if omp_results["success"] and asc_results["success"]:

        # 原始图像
        im1 = axes[0, 0].imshow(omp_results["magnitude"], cmap="hot", aspect="auto")
        axes[0, 0].set_title("原始MSTAR图像")
        axes[0, 0].axis("off")
        plt.colorbar(im1, ax=axes[0, 0])

        # 传统OMP重构
        im2 = axes[0, 1].imshow(np.abs(omp_results["reconstructed"]), cmap="hot", aspect="auto")
        axes[0, 1].set_title(f'传统OMP重构\nPSNR: {omp_results["psnr"]:.1f}dB')
        axes[0, 1].axis("off")
        plt.colorbar(im2, ax=axes[0, 1])

        # 高级ASC重构
        im3 = axes[0, 2].imshow(np.abs(asc_results["reconstructed"]), cmap="hot", aspect="auto")
        axes[0, 2].set_title(
            f'高级ASC重构 ({asc_results.get("precision_mode", "high")})\nPSNR: {asc_results["psnr"]:.1f}dB'
        )
        axes[0, 2].axis("off")
        plt.colorbar(im3, ax=axes[0, 2])

        # 重构误差对比
        omp_error = omp_results["magnitude"] - np.abs(omp_results["reconstructed"])
        asc_error = asc_results["magnitude"] - np.abs(asc_results["reconstructed"])

        max_error = max(np.max(np.abs(omp_error)), np.max(np.abs(asc_error)))

        im4 = axes[0, 3].imshow(omp_error, cmap="seismic", vmin=-max_error, vmax=max_error, aspect="auto")
        axes[0, 3].set_title("传统OMP重构误差")
        axes[0, 3].axis("off")
        plt.colorbar(im4, ax=axes[0, 3])

        # 散射中心分布对比
        # 传统OMP散射中心
        omp_x = [s["x"] for s in omp_results["scatterers"][:20]]  # 显示前20个
        omp_y = [s["y"] for s in omp_results["scatterers"][:20]]
        omp_amp = [s["estimated_amplitude"] for s in omp_results["scatterers"][:20]]

        scatter1 = axes[1, 0].scatter(omp_x, omp_y, c=omp_amp, s=100, cmap="viridis", alpha=0.8)
        axes[1, 0].set_title(f'传统OMP散射中心\n(显示前20个,共{len(omp_results["scatterers"])}个)')
        axes[1, 0].set_xlabel("X位置")
        axes[1, 0].set_ylabel("Y位置")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(-1, 1)
        axes[1, 0].set_ylim(-1, 1)
        plt.colorbar(scatter1, ax=axes[1, 0], label="幅度")

        # 高级ASC散射中心（按α值分色）
        asc_scatter_data = {}
        alpha_colors = {-1.0: "blue", -0.5: "cyan", 0.0: "green", 0.5: "orange", 1.0: "red"}

        for scatterer in asc_results["scatterers"]:
            alpha = scatterer["alpha"]
            if alpha not in asc_scatter_data:
                asc_scatter_data[alpha] = {"x": [], "y": [], "amp": []}
            asc_scatter_data[alpha]["x"].append(scatterer["x"])
            asc_scatter_data[alpha]["y"].append(scatterer["y"])
            asc_scatter_data[alpha]["amp"].append(scatterer["estimated_amplitude"])

        for alpha, data in asc_scatter_data.items():
            color = alpha_colors.get(alpha, "purple")
            axes[1, 1].scatter(data["x"], data["y"], c=color, s=150, alpha=0.8, label=f"α={alpha}")

        axes[1, 1].set_title(
            f'高级ASC散射中心\n({asc_results.get("precision_mode", "high")} 模式, 共{len(asc_results["scatterers"])}个)'
        )
        axes[1, 1].set_xlabel("X位置")
        axes[1, 1].set_ylabel("Y位置")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(-1, 1)
        axes[1, 1].set_ylim(-1, 1)
        axes[1, 1].legend()

        # ASC重构误差
        im5 = axes[1, 2].imshow(asc_error, cmap="seismic", vmin=-max_error, vmax=max_error, aspect="auto")
        axes[1, 2].set_title("高级ASC重构误差")
        axes[1, 2].axis("off")
        plt.colorbar(im5, ax=axes[1, 2])

        # 性能对比统计
        axes[1, 3].axis("off")

        refinement_info = ""
        if asc_results.get("optimization_success_rate", 0) > 0:
            refinement_info = f"• 优化成功率: {asc_results['optimization_success_rate']:.1%}\n"

        stats_text = f"""高精度性能对比统计

传统OMP (高精度):
• 散射中心数: {omp_results["num_scatterers"]} (固定)
• PSNR: {omp_results["psnr"]:.2f} dB
• 处理时间: {omp_results["processing_time"]:.1f}s
• 字典规模: {omp_results["dictionary_size"]:,}
• 位置网格: {omp_results["position_grid_size"]}×{omp_results["position_grid_size"]}

高级ASC ({asc_results.get("precision_mode", "high")}):
• 散射中心数: {asc_results["num_scatterers"]} (自适应)
• PSNR: {asc_results["psnr"]:.2f} dB
• 处理时间: {asc_results["processing_time"]:.1f}s
• 字典规模: {asc_results["dictionary_size"]:,}
• 位置采样: {asc_results["position_samples"]}×{asc_results["position_samples"]}
{refinement_info}
改进效果:
• PSNR提升: {asc_results["psnr"] - omp_results["psnr"]:+.1f} dB
• 稀疏性: {(1 - asc_results["num_scatterers"]/omp_results["num_scatterers"])*100:.0f}% 减少
• α值识别: {len(asc_results["analysis"]["alpha_distribution"])} 种类型"""

        axes[1, 3].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[1, 3].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

    else:
        # 处理失败情况
        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i == 0:
                ax.text(0.5, 0.5, "测试失败\n请检查数据文件和算法配置", ha="center", va="center", fontsize=14)

    plt.tight_layout()

    # 保存结果
    save_path = os.path.join(results_dir, "mstar_precision_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"   📊 对比图已保存: {save_path}")

    plt.show()

    # 保存详细结果
    save_detailed_results(omp_results, asc_results, results_dir)


def save_detailed_results(omp_results, asc_results, results_dir):
    """保存详细结果数据"""

    # 保存传统OMP结果
    if omp_results["success"]:
        omp_file = os.path.join(results_dir, "traditional_omp_precision_results.txt")
        with open(omp_file, "w", encoding="utf-8") as f:
            f.write("传统OMP高精度提取结果\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"散射中心数: {omp_results['num_scatterers']}\n")
            f.write(f"PSNR: {omp_results['psnr']:.2f} dB\n")
            f.write(f"处理时间: {omp_results['processing_time']:.1f}s\n")
            f.write(f"字典规模: {omp_results['dictionary_size']:,}\n\n")

            f.write("前20个散射中心详情:\n")
            f.write("-" * 60 + "\n")
            f.write("序号   X位置      Y位置      幅度        相位\n")
            f.write("-" * 60 + "\n")

            for i, s in enumerate(omp_results["scatterers"][:20]):
                f.write(
                    f"{i+1:2d}   {s['x']:8.3f}   {s['y']:8.3f}   {s['estimated_amplitude']:8.3f}   {s['estimated_phase']:8.3f}\n"
                )

    # 保存高级ASC结果
    if asc_results["success"]:
        asc_file = os.path.join(results_dir, "advanced_asc_precision_results.txt")
        with open(asc_file, "w", encoding="utf-8") as f:
            f.write("高级ASC高精度提取结果\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"散射中心数: {asc_results['num_scatterers']}\n")
            f.write(f"PSNR: {asc_results['psnr']:.2f} dB\n")
            f.write(f"处理时间: {asc_results['processing_time']:.1f}s\n")
            f.write(f"字典规模: {asc_results['dictionary_size']:,}\n")
            f.write(f"优化成功率: {asc_results['optimization_success_rate']:.1%}\n\n")

            f.write(f"α值分布: {asc_results['analysis']['alpha_distribution']}\n\n")

            f.write("散射中心详情:\n")
            f.write("-" * 80 + "\n")
            f.write("序号   X位置      Y位置      α值     L值      幅度        相位      优化\n")
            f.write("-" * 80 + "\n")

            for i, s in enumerate(asc_results["scatterers"]):
                opt_status = "✓" if s.get("optimization_success", False) else "✗"
                f.write(
                    f"{i+1:2d}   {s['x']:8.3f}   {s['y']:8.3f}   {s['alpha']:4.1f}   {s['length']:6.3f}   {s['estimated_amplitude']:8.3f}   {s['estimated_phase']:8.3f}   {opt_status}\n"
                )

    print(f"   📄 详细结果已保存: {results_dir}")


if __name__ == "__main__":
    test_mstar_precision()
