#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSTAR数据OMP散射中心提取处理脚本
MSTAR Data OMP Scattering Center Extraction Processing Script

将OMP ASC算法应用于实际的MSTAR SAR数据
"""

import numpy as np
import matplotlib.pyplot as plt
from omp_asc_final import OMPASCFinal
import os
import glob
import time
from typing import List, Dict
import pickle


class MSTARProcessor:
    """MSTAR数据处理器"""

    def __init__(self, data_root: str = "datasets/SAR_ASC_Project"):
        self.data_root = data_root
        self.raw_data_dir = os.path.join(data_root, "02_Data_Processed_raw")
        self.results_dir = os.path.join(data_root, "03_OMP_Results")

        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)

        # 初始化OMP算法 - 使用推荐的平衡配置
        self.omp_asc = OMPASCFinal(n_scatterers=40, image_size=(128, 128), use_cv=False)  # 文档要求的40个散射中心

        print(f"MSTAR处理器初始化完成")
        print(f"数据根目录: {self.data_root}")
        print(f"结果保存目录: {self.results_dir}")

    def find_raw_files(self) -> List[str]:
        """查找所有.raw文件"""
        pattern = os.path.join(self.raw_data_dir, "**", "*.raw")
        raw_files = glob.glob(pattern, recursive=True)
        raw_files.sort()

        print(f"找到 {len(raw_files)} 个RAW文件:")
        for file in raw_files:
            rel_path = os.path.relpath(file, self.data_root)
            print(f"  - {rel_path}")

        return raw_files

    def process_single_file(self, raw_file_path: str) -> Dict:
        """处理单个RAW文件"""
        print(f"\n{'='*60}")
        file_name = os.path.basename(raw_file_path)
        print(f"处理文件: {file_name}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # 1. 加载数据
            print("步骤1: 加载SAR数据...")
            magnitude, complex_image = self.omp_asc.load_raw_data(raw_file_path)

            # 2. 预处理
            print("步骤2: 数据预处理...")
            signal = self.omp_asc.preprocess_data(complex_image)

            # 3. 构建字典 (使用平衡配置)
            print("步骤3: 构建SAR字典...")
            dictionary, param_grid = self.omp_asc.build_dictionary(position_grid_size=12, phase_levels=6)  # 平衡配置

            # 4. 提取散射中心
            print("步骤4: OMP散射中心提取...")
            results = self.omp_asc.extract_scatterers(signal)

            # 5. 重构图像
            print("步骤5: 图像重构...")
            reconstructed = self.omp_asc.reconstruct_image(results["scatterers"])

            # 6. 计算质量指标
            mse = np.mean((magnitude - np.abs(reconstructed)) ** 2)
            max_val = np.max(magnitude)
            psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float("inf")

            processing_time = time.time() - start_time

            # 整理结果
            result = {
                "file_name": file_name,
                "file_path": raw_file_path,
                "processing_time": processing_time,
                "scatterers": results["scatterers"],
                "coefficients": results["coefficients"],
                "reconstruction_error": results["reconstruction_error"],
                "psnr": psnr,
                "original_magnitude": magnitude,
                "reconstructed_image": reconstructed,
                "dictionary_size": dictionary.shape[1],
                "extracted_count": len(results["scatterers"]),
            }

            print(f"\n✅ 处理完成！")
            print(f"   处理时间: {processing_time:.2f}s")
            print(f"   提取散射中心: {len(results['scatterers'])}")
            print(f"   重构PSNR: {psnr:.2f} dB")
            print(f"   字典大小: {dictionary.shape[1]}")

            return result

        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
            return {
                "file_name": file_name,
                "file_path": raw_file_path,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def process_all_files(self) -> List[Dict]:
        """批量处理所有文件"""
        print("🚀 开始批量处理MSTAR数据")
        print("=" * 60)

        raw_files = self.find_raw_files()
        all_results = []

        for i, raw_file in enumerate(raw_files, 1):
            print(f"\n📁 [{i}/{len(raw_files)}] 处理进度")
            result = self.process_single_file(raw_file)
            all_results.append(result)

            # 保存单个结果
            self.save_single_result(result)

        # 保存汇总结果
        self.save_summary_results(all_results)

        print(f"\n🎉 批量处理完成！")
        print(f"   总文件数: {len(raw_files)}")
        print(f"   成功处理: {sum(1 for r in all_results if 'error' not in r)}")
        print(f"   处理失败: {sum(1 for r in all_results if 'error' in r)}")

        return all_results

    def save_single_result(self, result: Dict):
        """保存单个文件的处理结果"""
        if "error" in result:
            return

        file_name = result["file_name"]
        base_name = file_name.replace(".raw", "")

        # 保存散射中心数据
        scatterers_file = os.path.join(self.results_dir, f"{base_name}_scatterers.pkl")
        with open(scatterers_file, "wb") as f:
            pickle.dump(result["scatterers"], f)

        # 保存可视化图像
        self.visualize_result(result, save_path=os.path.join(self.results_dir, f"{base_name}_visualization.png"))

        # 保存散射中心列表
        self.save_scatterer_summary(result, os.path.join(self.results_dir, f"{base_name}_summary.txt"))

    def visualize_result(self, result: Dict, save_path: str):
        """可视化单个文件的处理结果"""
        if "error" in result:
            return

        magnitude = result["original_magnitude"]
        reconstructed = result["reconstructed_image"]
        scatterers = result["scatterers"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"OMP散射中心提取结果 - {result['file_name']}", fontsize=14)

        # 原始幅度图像
        im1 = axes[0, 0].imshow(magnitude, cmap="gray")
        axes[0, 0].set_title(f"原始SAR幅度图")
        axes[0, 0].axis("off")
        plt.colorbar(im1, ax=axes[0, 0])

        # 重构幅度图像
        im2 = axes[0, 1].imshow(np.abs(reconstructed), cmap="gray")
        axes[0, 1].set_title(f'OMP重构图像 (PSNR: {result["psnr"]:.1f}dB)')
        axes[0, 1].axis("off")
        plt.colorbar(im2, ax=axes[0, 1])

        # 差值图像
        diff = magnitude - np.abs(reconstructed)
        im3 = axes[0, 2].imshow(diff, cmap="seismic")
        axes[0, 2].set_title("重构误差")
        axes[0, 2].axis("off")
        plt.colorbar(im3, ax=axes[0, 2])

        # 散射中心位置图
        x_pos = [s["x"] for s in scatterers]
        y_pos = [s["y"] for s in scatterers]
        amplitudes = [s["estimated_amplitude"] for s in scatterers]

        scatter = axes[1, 0].scatter(x_pos, y_pos, c=amplitudes, s=100, cmap="viridis")
        axes[1, 0].set_title(f"散射中心位置 ({len(scatterers)}个)")
        axes[1, 0].set_xlabel("X位置 (归一化)")
        axes[1, 0].set_ylabel("Y位置 (归一化)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(-1.1, 1.1)
        axes[1, 0].set_ylim(-1.1, 1.1)
        plt.colorbar(scatter, ax=axes[1, 0], label="幅度")

        # 幅度分布直方图
        axes[1, 1].hist(amplitudes, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 1].set_title("散射中心幅度分布")
        axes[1, 1].set_xlabel("幅度")
        axes[1, 1].set_ylabel("数量")
        axes[1, 1].grid(True, alpha=0.3)

        # 处理信息文本
        info_text = f"""处理信息:
文件: {result['file_name']}
处理时间: {result['processing_time']:.2f}s
散射中心数: {result['extracted_count']}
字典大小: {result['dictionary_size']}
重构PSNR: {result['psnr']:.2f} dB
重构误差: {result['reconstruction_error']:.3f}

前5强散射中心:"""

        # 添加前5强散射中心信息
        for i, scatterer in enumerate(scatterers[:5]):
            info_text += f"\n{i+1}. 位置:({scatterer['x']:.2f},{scatterer['y']:.2f})"
            info_text += f" 幅度:{scatterer['estimated_amplitude']:.3f}"

        axes[1, 2].text(
            0.05,
            0.95,
            info_text,
            transform=axes[1, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   可视化结果保存: {os.path.basename(save_path)}")

    def save_scatterer_summary(self, result: Dict, save_path: str):
        """保存散射中心汇总信息"""
        if "error" in result:
            return

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"MSTAR文件OMP散射中心提取结果\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"文件名: {result['file_name']}\n")
            f.write(f"处理时间: {result['processing_time']:.2f}秒\n")
            f.write(f"散射中心总数: {result['extracted_count']}\n")
            f.write(f"字典大小: {result['dictionary_size']}\n")
            f.write(f"重构PSNR: {result['psnr']:.2f} dB\n")
            f.write(f"重构误差: {result['reconstruction_error']:.3f}\n\n")

            f.write(f"散射中心详细信息:\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{'序号':<4} {'X位置':<8} {'Y位置':<8} {'幅度':<12} {'相位':<12}\n")
            f.write(f"{'-'*80}\n")

            for i, scatterer in enumerate(result["scatterers"], 1):
                f.write(
                    f"{i:<4} {scatterer['x']:<8.3f} {scatterer['y']:<8.3f} "
                    f"{scatterer['estimated_amplitude']:<12.6f} {scatterer['estimated_phase']:<12.3f}\n"
                )

        print(f"   散射中心数据保存: {os.path.basename(save_path)}")

    def save_summary_results(self, all_results: List[Dict]):
        """保存所有文件的汇总结果"""
        summary_file = os.path.join(self.results_dir, "processing_summary.txt")

        successful_results = [r for r in all_results if "error" not in r]
        failed_results = [r for r in all_results if "error" in r]

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("MSTAR数据批量处理汇总报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"处理统计:\n")
            f.write(f"  总文件数: {len(all_results)}\n")
            f.write(f"  成功处理: {len(successful_results)}\n")
            f.write(f"  处理失败: {len(failed_results)}\n\n")

            if successful_results:
                avg_time = np.mean([r["processing_time"] for r in successful_results])
                avg_psnr = np.mean([r["psnr"] for r in successful_results])
                avg_scatterers = np.mean([r["extracted_count"] for r in successful_results])

                f.write(f"成功处理文件统计:\n")
                f.write(f"  平均处理时间: {avg_time:.2f}秒\n")
                f.write(f"  平均PSNR: {avg_psnr:.2f} dB\n")
                f.write(f"  平均散射中心数: {avg_scatterers:.1f}\n\n")

                f.write(f"详细结果:\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{'文件名':<25} {'时间(s)':<8} {'PSNR(dB)':<8} {'散射中心':<8}\n")
                f.write(f"{'-'*80}\n")

                for result in successful_results:
                    f.write(
                        f"{result['file_name']:<25} {result['processing_time']:<8.2f} "
                        f"{result['psnr']:<8.2f} {result['extracted_count']:<8}\n"
                    )

            if failed_results:
                f.write(f"\n失败文件列表:\n")
                f.write(f"{'-'*50}\n")
                for result in failed_results:
                    f.write(f"文件: {result['file_name']}\n")
                    f.write(f"错误: {result['error']}\n\n")

        print(f"汇总报告保存: processing_summary.txt")


def main():
    """主处理函数"""
    print("🎯 MSTAR数据OMP散射中心提取处理")
    print("=" * 60)

    # 初始化处理器
    processor = MSTARProcessor()

    # 批量处理所有文件
    results = processor.process_all_files()

    print(f"\n📊 处理完成！所有结果保存在: {processor.results_dir}")

    return results


if __name__ == "__main__":
    results = main()
