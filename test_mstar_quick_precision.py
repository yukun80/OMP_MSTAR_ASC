#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSTAR快速高精度ASC提取测试
平衡精度和效率的真实数据测试版本
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


def test_mstar_quick_precision():
    """快速高精度MSTAR测试"""
    print("🎯 MSTAR快速高精度ASC提取测试")
    print("=" * 60)

    # 寻找MSTAR数据文件
    mstar_dir = "datasets/SAR_ASC_Project/02_Data_Processed_raw"
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
    results_dir = "results/mstar_quick_precision"
    os.makedirs(results_dir, exist_ok=True)

    # 测试传统OMP系统 (平衡精度)
    print("\n🔄 测试传统OMP系统 (平衡精度)...")
    omp_results = test_traditional_omp_balanced(test_file)

    # 测试高级ASC系统 (平衡精度)
    print("\n🎯 测试高级ASC系统 (平衡精度)...")
    asc_results = test_advanced_asc_balanced(test_file)

    # 生成对比分析
    print("\n📊 生成对比分析...")
    create_quick_comparison(omp_results, asc_results, results_dir)

    print(f"\n✅ 快速高精度测试完成!")
    print(f"   结果保存在: {results_dir}")


def test_traditional_omp_balanced(test_file):
    """平衡精度传统OMP测试"""
    start_time = time.time()

    try:
        # 初始化OMP系统
        omp_system = OMPASCExtractor(n_scatterers=40)

        # 加载MSTAR数据
        magnitude, complex_image = omp_system.load_raw_data(test_file)
        signal = omp_system.preprocess_data(complex_image)

        # 平衡精度字典构建
        print("   构建平衡精度字典...")
        dictionary, param_grid = omp_system.build_dictionary(
            position_grid_size=24, phase_levels=8  # 24×24 = 576位置 (平衡精度)  # 8个相位层
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
            "system_type": "传统OMP (平衡精度)",
            "scatterers": scatterers,
            "magnitude": magnitude,
            "reconstructed": reconstructed,
            "num_scatterers": len(scatterers),
            "processing_time": processing_time,
            "psnr": psnr,
            "position_grid_size": 24,
            "dictionary_size": dictionary.shape[1],
        }

        print(f"   ✅ 传统OMP (平衡): {len(scatterers)} 个散射中心")
        print(f"   📊 PSNR: {psnr:.2f} dB")
        print(f"   ⏱️ 处理时间: {processing_time:.1f}s")
        print(f"   🔍 字典规模: {dictionary.shape[1]} 个原子")

        return results

    except Exception as e:
        print(f"   ❌ 传统OMP失败: {str(e)}")
        return {"success": False, "error": str(e)}


def test_advanced_asc_balanced(test_file):
    """平衡精度ASC测试"""
    start_time = time.time()

    try:
        # 初始化平衡精度ASC系统
        asc_system = ASCExtractionAdvanced(
            adaptive_threshold=0.01,  # 1% 阈值 (平衡设置)
            max_iterations=30,
            min_scatterers=3,
            max_scatterers=20,
            precision_mode="balanced",  # 平衡模式：32×32采样+精化
        )

        # 加载MSTAR数据
        magnitude, complex_image = asc_system.load_raw_data(test_file)
        signal = asc_system.preprocess_data(complex_image)

        # 构建平衡精度ASC字典
        print("   构建平衡精度ASC字典...")
        dictionary, param_grid = asc_system.build_asc_dictionary()

        # 自适应提取
        scatterers = asc_system.adaptive_asc_extraction(signal, dictionary, param_grid)

        # 参数精化
        if asc_system.enable_refinement and len(scatterers) > 0:
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
            "system_type": "高级ASC (平衡精度)",
            "scatterers": refined_scatterers,
            "magnitude": magnitude,
            "reconstructed": reconstructed,
            "num_scatterers": len(refined_scatterers),
            "processing_time": processing_time,
            "psnr": psnr,
            "dictionary_size": dictionary.shape[1],
            "analysis": analysis,
            "precision_mode": asc_system.precision_mode,
            "optimization_success_rate": analysis.get("optimization_success_rate", 0),
        }

        print(f"   ✅ 高级ASC (平衡): {len(refined_scatterers)} 个散射中心")
        print(f"   📊 PSNR: {psnr:.2f} dB")
        print(f"   ⏱️ 处理时间: {processing_time:.1f}s")
        print(f"   🔍 字典规模: {dictionary.shape[1]} 个原子")
        print(f"   🎯 优化成功率: {analysis.get('optimization_success_rate', 0):.1%}")
        print(f"   🔬 α分布: {analysis.get('alpha_distribution', {})}")

        return results

    except Exception as e:
        print(f"   ❌ 高级ASC失败: {str(e)}")
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


def create_quick_comparison(omp_results, asc_results, results_dir):
    """创建快速对比可视化"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("MSTAR数据快速高精度ASC提取对比", fontsize=16, fontweight="bold")

    if omp_results["success"] and asc_results["success"]:

        # 第一行：原始图像和重构结果
        # 原始图像
        im1 = axes[0, 0].imshow(omp_results["magnitude"], cmap="hot", aspect="auto")
        axes[0, 0].set_title("原始MSTAR图像")
        axes[0, 0].axis("off")
        plt.colorbar(im1, ax=axes[0, 0])

        # 传统OMP重构
        im2 = axes[0, 1].imshow(np.abs(omp_results["reconstructed"]), cmap="hot", aspect="auto")
        axes[0, 1].set_title(
            f'传统OMP重构\n{omp_results["num_scatterers"]}个散射中心\nPSNR: {omp_results["psnr"]:.1f}dB'
        )
        axes[0, 1].axis("off")
        plt.colorbar(im2, ax=axes[0, 1])

        # 高级ASC重构
        im3 = axes[0, 2].imshow(np.abs(asc_results["reconstructed"]), cmap="hot", aspect="auto")
        axes[0, 2].set_title(
            f'高级ASC重构\n{asc_results["num_scatterers"]}个散射中心\nPSNR: {asc_results["psnr"]:.1f}dB'
        )
        axes[0, 2].axis("off")
        plt.colorbar(im3, ax=axes[0, 2])

        # 第二行：散射中心分布和性能对比
        # 传统OMP散射中心
        omp_x = [s["x"] for s in omp_results["scatterers"][:15]]  # 显示前15个
        omp_y = [s["y"] for s in omp_results["scatterers"][:15]]
        omp_amp = [s["estimated_amplitude"] for s in omp_results["scatterers"][:15]]

        scatter1 = axes[1, 0].scatter(omp_x, omp_y, c=omp_amp, s=80, cmap="viridis", alpha=0.8)
        axes[1, 0].set_title(f'传统OMP散射中心\n(显示前15个,共{len(omp_results["scatterers"])}个)')
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
            axes[1, 1].scatter(data["x"], data["y"], c=color, s=120, alpha=0.8, label=f"α={alpha}")

        axes[1, 1].set_title(f'高级ASC散射中心\n({asc_results.get("precision_mode", "balanced")} 模式)')
        axes[1, 1].set_xlabel("X位置")
        axes[1, 1].set_ylabel("Y位置")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(-1, 1)
        axes[1, 1].set_ylim(-1, 1)
        axes[1, 1].legend(fontsize=9)

        # 性能对比统计
        axes[1, 2].axis("off")

        improvement_text = f"""快速高精度对比统计

传统OMP (平衡精度):
• 散射中心数: {omp_results["num_scatterers"]} (固定)
• PSNR: {omp_results["psnr"]:.2f} dB
• 处理时间: {omp_results["processing_time"]:.1f}s
• 字典规模: {omp_results["dictionary_size"]:,}

高级ASC (平衡精度):
• 散射中心数: {asc_results["num_scatterers"]} (自适应)
• PSNR: {asc_results["psnr"]:.2f} dB
• 处理时间: {asc_results["processing_time"]:.1f}s
• 字典规模: {asc_results["dictionary_size"]:,}
• 优化成功率: {asc_results["optimization_success_rate"]:.1%}

改进效果:
• PSNR提升: {asc_results["psnr"] - omp_results["psnr"]:+.1f} dB
• 稀疏性: {(1 - asc_results["num_scatterers"]/omp_results["num_scatterers"])*100:.0f}% 减少
• α值识别: {len(asc_results["analysis"]["alpha_distribution"])} 种类型

🎯 关键突破:
✅ 自适应散射中心数量
✅ 多散射类型识别
✅ 非网格约束定位
✅ 完整ASC参数提取"""

        axes[1, 2].text(
            0.05,
            0.95,
            improvement_text,
            transform=axes[1, 2].transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
        )

    else:
        # 处理失败情况
        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i == 0:
                if not omp_results["success"]:
                    ax.text(
                        0.5, 0.5, f"传统OMP失败\n{omp_results.get('error', '')}", ha="center", va="center", fontsize=12
                    )
                elif not asc_results["success"]:
                    ax.text(
                        0.5, 0.5, f"高级ASC失败\n{asc_results.get('error', '')}", ha="center", va="center", fontsize=12
                    )

    plt.tight_layout()

    # 保存结果
    save_path = os.path.join(results_dir, "mstar_quick_precision_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"   📊 对比图已保存: {save_path}")

    plt.show()

    # 保存简要结果
    save_brief_results(omp_results, asc_results, results_dir)


def save_brief_results(omp_results, asc_results, results_dir):
    """保存简要结果"""

    summary_file = os.path.join(results_dir, "quick_precision_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("MSTAR快速高精度ASC提取测试总结\n")
        f.write("=" * 50 + "\n\n")

        if omp_results["success"]:
            f.write("传统OMP (平衡精度) 结果:\n")
            f.write(f"  散射中心数: {omp_results['num_scatterers']}\n")
            f.write(f"  PSNR: {omp_results['psnr']:.2f} dB\n")
            f.write(f"  处理时间: {omp_results['processing_time']:.1f}s\n")
            f.write(f"  字典规模: {omp_results['dictionary_size']:,}\n\n")
        else:
            f.write(f"传统OMP失败: {omp_results.get('error', '')}\n\n")

        if asc_results["success"]:
            f.write("高级ASC (平衡精度) 结果:\n")
            f.write(f"  散射中心数: {asc_results['num_scatterers']}\n")
            f.write(f"  PSNR: {asc_results['psnr']:.2f} dB\n")
            f.write(f"  处理时间: {asc_results['processing_time']:.1f}s\n")
            f.write(f"  字典规模: {asc_results['dictionary_size']:,}\n")
            f.write(f"  优化成功率: {asc_results['optimization_success_rate']:.1%}\n")
            f.write(f"  α分布: {asc_results['analysis']['alpha_distribution']}\n\n")
        else:
            f.write(f"高级ASC失败: {asc_results.get('error', '')}\n\n")

        if omp_results["success"] and asc_results["success"]:
            f.write("改进效果分析:\n")
            f.write(f"  PSNR提升: {asc_results['psnr'] - omp_results['psnr']:+.1f} dB\n")
            f.write(
                f"  稀疏性提升: {(1 - asc_results['num_scatterers']/omp_results['num_scatterers'])*100:.0f}% 减少\n"
            )
            f.write(f"  散射类型识别: {len(asc_results['analysis']['alpha_distribution'])} 种\n")

    print(f"   📄 结果总结已保存: {summary_file}")


if __name__ == "__main__":
    test_mstar_quick_precision()
