#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASC高级提取系统与传统OMP系统对比测试
Advanced ASC vs Traditional OMP Comparison Test

对比验证:
1. 传统OMP系统 (固定网格 + 固定稀疏度)
2. 高级ASC系统 (自适应 + 多参数)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

# 导入系统模块
try:
    from asc_extraction_advanced import ASCExtractionAdvanced
    from omp_asc_final import OMPASCExtractor
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保所有必要的模块文件都在当前目录中")
    sys.exit(1)


class ASCComparisonTester:
    """ASC高级系统与传统OMP系统对比测试器"""

    def __init__(self, data_dir: str = "datasets/SAR_ASC_Project/02_Data_Processed_raw"):
        """
        初始化对比测试器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.results_dir = "results/asc_comparison"

        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)

        print("🔬 ASC高级系统 vs 传统OMP系统 对比测试")
        print("=" * 70)

    def initialize_systems(self) -> Tuple[ASCExtractionAdvanced, OMPASCExtractor]:
        """初始化两个系统"""
        print("🚀 初始化测试系统...")

        # 1. 高级ASC系统 (自适应参数)
        print("\n📍 初始化高级ASC系统:")
        asc_advanced = ASCExtractionAdvanced(
            image_size=(128, 128),
            adaptive_threshold=0.05,  # 5% 自适应阈值
            max_iterations=50,
            min_scatterers=5,
            max_scatterers=30,
        )

        # 2. 传统OMP系统 (固定参数)
        print("\n📍 初始化传统OMP系统:")
        omp_traditional = OMPASCExtractor(image_size=(128, 128), n_scatterers=40)  # 固定40个散射中心

        return asc_advanced, omp_traditional

    def load_test_data(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载测试数据"""
        file_path = os.path.join(self.data_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"测试数据文件不存在: {file_path}")

        print(f"📂 加载测试数据: {filename}")

        # 使用ASC系统加载数据（两个系统数据加载方式相同）
        asc_system = ASCExtractionAdvanced()
        magnitude, complex_image = asc_system.load_raw_data(file_path)

        return magnitude, complex_image

    def run_traditional_omp(self, complex_image: np.ndarray, omp_system: OMPASCExtractor) -> Dict:
        """运行传统OMP系统"""
        print("\n🔄 运行传统OMP系统...")
        start_time = time.time()

        try:
            # 数据预处理
            signal = omp_system.preprocess_data(complex_image)

            # 构建字典 (固定网格)
            dictionary, param_grid = omp_system.build_dictionary(position_grid_size=12, phase_levels=6)  # 固定12x12网格

            # OMP提取 (固定40个散射中心)
            scatterers = omp_system.extract_scatterers(signal, dictionary, param_grid)

            # 重构图像
            reconstructed = omp_system.reconstruct_image(scatterers, complex_image.shape)

            processing_time = time.time() - start_time

            # 计算性能指标
            original_magnitude = np.abs(complex_image)
            reconstructed_magnitude = np.abs(reconstructed)
            psnr = omp_system.calculate_psnr(original_magnitude, reconstructed_magnitude)

            results = {
                "system_type": "Traditional OMP",
                "scatterers": scatterers,
                "reconstructed": reconstructed,
                "processing_time": processing_time,
                "psnr": psnr,
                "num_scatterers": len(scatterers),
                "dictionary_size": dictionary.shape[1],
                "grid_type": "Fixed Grid",
                "sparsity_type": "Fixed (40 scatterers)",
                "success": True,
            }

            print(f"✅ 传统OMP完成 - 用时: {processing_time:.1f}s, PSNR: {psnr:.1f}dB")
            return results

        except Exception as e:
            print(f"❌ 传统OMP失败: {str(e)}")
            return {"system_type": "Traditional OMP", "success": False, "error": str(e)}

    def run_advanced_asc(self, complex_image: np.ndarray, asc_system: ASCExtractionAdvanced) -> Dict:
        """运行高级ASC系统"""
        print("\n🎯 运行高级ASC系统...")
        start_time = time.time()

        try:
            # 数据预处理
            signal = asc_system.preprocess_data(complex_image)

            # 构建ASC复合字典 (多参数)
            dictionary, param_grid = asc_system.build_asc_dictionary(
                position_samples=8, azimuth_samples=4  # 减少采样以控制计算量
            )

            # 自适应ASC提取
            scatterers = asc_system.adaptive_asc_extraction(signal, dictionary, param_grid)

            # 参数精化
            if len(scatterers) > 0:
                refined_scatterers = asc_system.refine_parameters(scatterers, signal)
            else:
                refined_scatterers = scatterers

            # 重构图像
            reconstructed = asc_system.reconstruct_asc_image(refined_scatterers)

            processing_time = time.time() - start_time

            # 性能分析
            analysis = asc_system.analyze_asc_results(refined_scatterers)

            # 计算PSNR
            original_magnitude = np.abs(complex_image)
            reconstructed_magnitude = np.abs(reconstructed)

            # 简单PSNR计算
            mse = np.mean((original_magnitude - reconstructed_magnitude) ** 2)
            if mse > 0:
                psnr = 20 * np.log10(np.max(original_magnitude) / np.sqrt(mse))
            else:
                psnr = float("inf")

            results = {
                "system_type": "Advanced ASC",
                "scatterers": refined_scatterers,
                "reconstructed": reconstructed,
                "processing_time": processing_time,
                "psnr": psnr,
                "num_scatterers": len(refined_scatterers),
                "dictionary_size": dictionary.shape[1],
                "grid_type": "Adaptive Multi-parameter",
                "sparsity_type": "Adaptive (5-30 scatterers)",
                "analysis": analysis,
                "success": True,
            }

            print(f"✅ 高级ASC完成 - 用时: {processing_time:.1f}s, PSNR: {psnr:.1f}dB")
            return results

        except Exception as e:
            print(f"❌ 高级ASC失败: {str(e)}")
            return {"system_type": "Advanced ASC", "success": False, "error": str(e)}

    def create_comparison_visualization(
        self, original_image: np.ndarray, omp_results: Dict, asc_results: Dict, filename: str
    ) -> None:
        """创建对比可视化"""
        print("🎨 生成对比可视化...")

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"ASC高级系统 vs 传统OMP系统对比 - {filename}", fontsize=16, fontweight="bold")

        # 原始图像
        axes[0, 0].imshow(np.abs(original_image), cmap="hot", aspect="auto")
        axes[0, 0].set_title("原始SAR图像")
        axes[0, 0].set_xlabel("距离维")
        axes[0, 0].set_ylabel("方位维")

        # 传统OMP结果
        if omp_results["success"]:
            omp_reconstructed = np.abs(omp_results["reconstructed"])
            axes[0, 1].imshow(omp_reconstructed, cmap="hot", aspect="auto")
            axes[0, 1].set_title(
                f'传统OMP重构\n{omp_results["num_scatterers"]}个散射中心\nPSNR: {omp_results["psnr"]:.1f}dB'
            )

            # 传统OMP散射中心位置
            axes[0, 2].imshow(np.abs(original_image), cmap="gray", alpha=0.7, aspect="auto")
            for i, scatterer in enumerate(omp_results["scatterers"][:10]):  # 只显示前10个
                x_pixel = int((scatterer["x"] + 1) * 64)  # 转换到像素坐标
                y_pixel = int((scatterer["y"] + 1) * 64)
                circle = Circle((x_pixel, y_pixel), 2, color="red", fill=False, linewidth=2)
                axes[0, 2].add_patch(circle)
            axes[0, 2].set_title("传统OMP散射中心\n(红色圆圈，网格分布)")
            axes[0, 2].set_xlim(0, 128)
            axes[0, 2].set_ylim(0, 128)
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                f'传统OMP失败\n{omp_results.get("error", "")}',
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )
            axes[0, 2].text(0.5, 0.5, "无结果", ha="center", va="center", transform=axes[0, 2].transAxes)

        # 高级ASC结果
        if asc_results["success"]:
            asc_reconstructed = np.abs(asc_results["reconstructed"])
            axes[0, 3].imshow(asc_reconstructed, cmap="hot", aspect="auto")
            axes[0, 3].set_title(
                f'高级ASC重构\n{asc_results["num_scatterers"]}个散射中心\nPSNR: {asc_results["psnr"]:.1f}dB'
            )

            # 高级ASC散射中心位置 (按α值着色)
            axes[1, 2].imshow(np.abs(original_image), cmap="gray", alpha=0.7, aspect="auto")

            alpha_colors = {-1.0: "blue", -0.5: "cyan", 0.0: "green", 0.5: "orange", 1.0: "red"}

            for scatterer in asc_results["scatterers"]:
                x_pixel = int((scatterer["x"] + 1) * 64)
                y_pixel = int((scatterer["y"] + 1) * 64)
                alpha_val = scatterer["alpha"]
                color = alpha_colors.get(alpha_val, "purple")

                circle = Circle((x_pixel, y_pixel), 3, color=color, fill=True, alpha=0.8)
                axes[1, 2].add_patch(circle)

            axes[1, 2].set_title("高级ASC散射中心\n(颜色表示α值)")
            axes[1, 2].set_xlim(0, 128)
            axes[1, 2].set_ylim(0, 128)

            # 添加图例
            from matplotlib.patches import Patch

            legend_elements = [Patch(facecolor=color, label=f"α={alpha}") for alpha, color in alpha_colors.items()]
            axes[1, 2].legend(handles=legend_elements, loc="upper right", fontsize=8)

        else:
            axes[0, 3].text(
                0.5,
                0.5,
                f'高级ASC失败\n{asc_results.get("error", "")}',
                ha="center",
                va="center",
                transform=axes[0, 3].transAxes,
            )
            axes[1, 2].text(0.5, 0.5, "无结果", ha="center", va="center", transform=axes[1, 2].transAxes)

        # 性能对比柱状图
        if omp_results["success"] and asc_results["success"]:
            categories = ["PSNR (dB)", "散射中心数", "处理时间 (s)"]
            omp_values = [omp_results["psnr"], omp_results["num_scatterers"], omp_results["processing_time"]]
            asc_values = [asc_results["psnr"], asc_results["num_scatterers"], asc_results["processing_time"]]

            x = np.arange(len(categories))
            width = 0.35

            axes[1, 0].bar(x - width / 2, omp_values, width, label="传统OMP", alpha=0.8, color="skyblue")
            axes[1, 0].bar(x + width / 2, asc_values, width, label="高级ASC", alpha=0.8, color="lightcoral")

            axes[1, 0].set_xlabel("性能指标")
            axes[1, 0].set_ylabel("数值")
            axes[1, 0].set_title("性能对比")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(categories)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 参数分布对比
        if asc_results["success"] and "analysis" in asc_results:
            analysis = asc_results["analysis"]

            # α值分布饼图
            if "alpha_distribution" in analysis:
                alpha_dist = analysis["alpha_distribution"]
                if alpha_dist:
                    axes[1, 1].pie(
                        alpha_dist.values(),
                        labels=[f"α={k}" for k in alpha_dist.keys()],
                        autopct="%1.1f%%",
                        startangle=90,
                    )
                    axes[1, 1].set_title("高级ASC: α值分布")
                else:
                    axes[1, 1].text(0.5, 0.5, "无α分布数据", ha="center", va="center", transform=axes[1, 1].transAxes)

            # 幅度分布直方图
            if asc_results["scatterers"]:
                amplitudes = [s["estimated_amplitude"] for s in asc_results["scatterers"]]
                axes[1, 3].hist(amplitudes, bins=10, alpha=0.7, color="green", edgecolor="black")
                axes[1, 3].set_xlabel("散射幅度")
                axes[1, 3].set_ylabel("频次")
                axes[1, 3].set_title("高级ASC: 幅度分布")
                axes[1, 3].grid(True, alpha=0.3)

        # 移除空白子图
        for ax in axes.flat:
            if not ax.has_data():
                ax.axis("off")

        plt.tight_layout()

        # 保存图像
        output_path = os.path.join(self.results_dir, f"comparison_{filename[:-4]}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   📊 对比图已保存: {output_path}")

    def generate_comparison_report(self, omp_results: Dict, asc_results: Dict, filename: str) -> Dict:
        """生成对比报告"""
        print("📝 生成对比报告...")

        report = {
            "filename": filename,
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "traditional_omp": {},
            "advanced_asc": {},
            "comparison": {},
        }

        # 传统OMP结果
        if omp_results["success"]:
            report["traditional_omp"] = {
                "success": True,
                "processing_time": omp_results["processing_time"],
                "psnr": omp_results["psnr"],
                "num_scatterers": omp_results["num_scatterers"],
                "dictionary_size": omp_results["dictionary_size"],
                "grid_type": omp_results["grid_type"],
                "sparsity_type": omp_results["sparsity_type"],
            }
        else:
            report["traditional_omp"] = {"success": False, "error": omp_results.get("error", "Unknown error")}

        # 高级ASC结果
        if asc_results["success"]:
            report["advanced_asc"] = {
                "success": True,
                "processing_time": asc_results["processing_time"],
                "psnr": asc_results["psnr"],
                "num_scatterers": asc_results["num_scatterers"],
                "dictionary_size": asc_results["dictionary_size"],
                "grid_type": asc_results["grid_type"],
                "sparsity_type": asc_results["sparsity_type"],
                "analysis": asc_results.get("analysis", {}),
            }
        else:
            report["advanced_asc"] = {"success": False, "error": asc_results.get("error", "Unknown error")}

        # 对比分析
        if omp_results["success"] and asc_results["success"]:
            report["comparison"] = {
                "psnr_improvement": asc_results["psnr"] - omp_results["psnr"],
                "time_ratio": asc_results["processing_time"] / omp_results["processing_time"],
                "scatterer_reduction": omp_results["num_scatterers"] - asc_results["num_scatterers"],
                "dictionary_ratio": asc_results["dictionary_size"] / omp_results["dictionary_size"],
            }

        return report

    def run_single_comparison(self, filename: str) -> Dict:
        """运行单个文件的对比测试"""
        print(f"\n{'='*70}")
        print(f"🎯 对比测试: {filename}")
        print(f"{'='*70}")

        try:
            # 加载数据
            magnitude, complex_image = self.load_test_data(filename)

            # 初始化系统
            asc_advanced, omp_traditional = self.initialize_systems()

            # 运行两个系统
            omp_results = self.run_traditional_omp(complex_image, omp_traditional)
            asc_results = self.run_advanced_asc(complex_image, asc_advanced)

            # 生成可视化
            self.create_comparison_visualization(complex_image, omp_results, asc_results, filename)

            # 生成报告
            report = self.generate_comparison_report(omp_results, asc_results, filename)

            # 打印对比结果
            self.print_comparison_summary(report)

            return report

        except Exception as e:
            print(f"❌ 对比测试失败: {str(e)}")
            return {"filename": filename, "success": False, "error": str(e)}

    def print_comparison_summary(self, report: Dict) -> None:
        """打印对比总结"""
        print("\n📊 对比测试总结:")
        print("-" * 50)

        if report.get("traditional_omp", {}).get("success") and report.get("advanced_asc", {}).get("success"):
            omp = report["traditional_omp"]
            asc = report["advanced_asc"]
            comp = report["comparison"]

            print(f"📈 性能对比:")
            print(f"   PSNR: {omp['psnr']:.1f}dB → {asc['psnr']:.1f}dB (Δ{comp['psnr_improvement']:+.1f}dB)")
            print(
                f"   散射中心数: {omp['num_scatterers']} → {asc['num_scatterers']} (Δ{comp['scatterer_reduction']:+d})"
            )
            print(
                f"   处理时间: {omp['processing_time']:.1f}s → {asc['processing_time']:.1f}s ({comp['time_ratio']:.1f}x)"
            )
            print(f"   字典规模: {omp['dictionary_size']} → {asc['dictionary_size']} ({comp['dictionary_ratio']:.1f}x)")

            print(f"\n🎯 算法特征:")
            print(f"   传统OMP: {omp['grid_type']} + {omp['sparsity_type']}")
            print(f"   高级ASC: {asc['grid_type']} + {asc['sparsity_type']}")

            if "analysis" in asc and "alpha_distribution" in asc["analysis"]:
                alpha_dist = asc["analysis"]["alpha_distribution"]
                print(f"   α值分布: {alpha_dist}")
        else:
            print("⚠️ 部分系统运行失败，无法生成完整对比")


def main():
    """主测试函数"""
    print("🔬 ASC高级系统 vs 传统OMP系统 对比测试")
    print("=" * 70)

    # 检查数据目录
    data_dir = "datasets/SAR_ASC_Project/02_Data_Processed_raw"
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请确保MSTAR数据已经预处理并放置在正确目录中")
        return

    # 获取测试文件
    raw_files = [f for f in os.listdir(data_dir) if f.endswith(".raw")]

    if not raw_files:
        print(f"❌ 在 {data_dir} 中未找到.raw文件")
        return

    print(f"📂 找到 {len(raw_files)} 个RAW文件:")
    for i, filename in enumerate(raw_files[:3], 1):  # 只列出前3个
        print(f"   {i}. {filename}")
    if len(raw_files) > 3:
        print(f"   ... 还有 {len(raw_files) - 3} 个文件")

    # 初始化测试器
    tester = ASCComparisonTester(data_dir)

    # 选择测试文件 (使用第一个文件进行演示)
    test_file = raw_files[0]
    print(f"\n🎯 使用测试文件: {test_file}")

    # 运行对比测试
    report = tester.run_single_comparison(test_file)

    # 保存报告
    if report.get("success", True):
        import json

        report_path = os.path.join(tester.results_dir, f"report_{test_file[:-4]}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📝 详细报告已保存: {report_path}")

    print(f"\n✅ 对比测试完成! 结果保存在: {tester.results_dir}")

    return report


if __name__ == "__main__":
    report = main()
