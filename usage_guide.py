#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OMP ASC提取算法使用指南
Usage Guide for OMP-based Scattering Center Extraction

本文件提供了详细的使用指南和实际应用示例
This file provides detailed usage guide and practical examples
"""

import numpy as np
import os
from pathlib import Path
from omp_asc_extraction import OMPASC


class OMPASCGuide:
    """
    OMP ASC算法使用指南类
    """

    def __init__(self):
        self.data_dir = None
        self.output_dir = None

    def setup_directories(self, data_root: str):
        """
        设置数据目录结构

        Parameters:
        -----------
        data_root : str
            项目根目录路径
        """
        self.data_dir = Path(data_root) / "datasets" / "SAR_ASC_Project" / "02_Data_Processed_raw"
        self.output_dir = Path(data_root) / "results" / "omp_asc"

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.output_dir}")

    def find_raw_files(self) -> list:
        """
        查找所有.raw数据文件

        Returns:
        --------
        raw_files : list
            .raw文件路径列表
        """
        if not self.data_dir.exists():
            print(f"警告: 数据目录不存在 {self.data_dir}")
            return []

        raw_files = list(self.data_dir.rglob("*.raw"))
        print(f"找到 {len(raw_files)} 个.raw文件")

        return raw_files

    def process_single_file(self, raw_file_path: str, n_scatterers: int = 40, save_results: bool = True):
        """
        处理单个SAR图像文件

        Parameters:
        -----------
        raw_file_path : str
            .raw文件路径
        n_scatterers : int
            要提取的散射中心数量
        save_results : bool
            是否保存结果

        Returns:
        --------
        results : dict
            处理结果
        """
        print(f"\n=== 处理文件: {Path(raw_file_path).name} ===")

        # 初始化OMP算法
        omp_asc = OMPASC(n_scatterers=n_scatterers, use_cv=False)

        try:
            # 步骤1: 加载数据
            print("1. 加载SAR数据...")
            magnitude, complex_image = omp_asc.load_raw_data(raw_file_path)

            # 步骤2: 数据预处理
            print("2. 数据预处理...")
            signal = omp_asc.preprocess_data(complex_image)

            # 步骤3: 构建字典（计算密集，使用较小的网格）
            print("3. 构建SAR字典...")
            dictionary, param_grid = omp_asc.build_dictionary(
                position_grid_size=16, amplitude_levels=5  # 16x16 位置网格  # 5个幅度级别
            )

            # 步骤4: OMP散射中心提取
            print("4. OMP散射中心提取...")
            extraction_results = omp_asc.extract_scatterers(signal)

            # 步骤5: 图像重构
            print("5. 图像重构...")
            reconstructed = omp_asc.reconstruct_image(extraction_results["scatterers"])

            # 步骤6: 结果保存和可视化
            if save_results:
                file_stem = Path(raw_file_path).stem
                output_file = self.output_dir / f"{file_stem}_omp_results.png"

                print("6. 保存结果...")
                omp_asc.visualize_results(
                    magnitude, reconstructed, extraction_results["scatterers"], save_path=str(output_file)
                )

                # 保存散射中心参数
                self._save_scatterer_params(
                    extraction_results["scatterers"], self.output_dir / f"{file_stem}_scatterers.txt"
                )

            # 计算性能指标
            results = self._calculate_metrics(magnitude, reconstructed, extraction_results)

            print(f"\n=== 处理完成 ===")
            print(f"提取散射中心数量: {len(extraction_results['scatterers'])}")
            print(f"重构误差: {results['reconstruction_error']:.3f}")
            print(f"PSNR: {results['psnr']:.2f} dB")

            return results

        except Exception as e:
            print(f"处理失败: {str(e)}")
            return None

    def batch_process(self, max_files: int = 5):
        """
        批量处理多个文件

        Parameters:
        -----------
        max_files : int
            最大处理文件数量
        """
        raw_files = self.find_raw_files()

        if not raw_files:
            print("未找到.raw文件，请检查数据目录路径")
            return

        # 限制处理文件数量
        process_files = raw_files[:max_files]

        print(f"\n=== 批量处理 {len(process_files)} 个文件 ===")

        all_results = []
        for i, raw_file in enumerate(process_files, 1):
            print(f"\n进度: {i}/{len(process_files)}")

            results = self.process_single_file(str(raw_file))
            if results:
                results["filename"] = raw_file.name
                all_results.append(results)

        # 生成批量处理报告
        self._generate_batch_report(all_results)

    def _save_scatterer_params(self, scatterers: list, output_path: Path):
        """保存散射中心参数到文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# OMP提取的散射中心参数\n")
            f.write("# Index\tX_Position\tY_Position\tAmplitude\tPhase\n")

            for i, scatterer in enumerate(scatterers):
                f.write(
                    f"{i+1}\t{scatterer['x']:.6f}\t{scatterer['y']:.6f}\t"
                    f"{scatterer['estimated_amplitude']:.6f}\t{scatterer['estimated_phase']:.6f}\n"
                )

        print(f"散射中心参数已保存: {output_path}")

    def _calculate_metrics(self, original: np.ndarray, reconstructed: np.ndarray, extraction_results: dict) -> dict:
        """计算性能指标"""

        # 重构误差
        recon_error = np.linalg.norm(original - np.abs(reconstructed))

        # PSNR计算
        mse = np.mean((original - np.abs(reconstructed)) ** 2)
        max_val = np.max(original)
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float("inf")

        # 稀疏度
        sparsity = len(extraction_results["scatterers"])

        return {
            "reconstruction_error": recon_error,
            "psnr": psnr,
            "sparsity": sparsity,
            "compression_ratio": original.size / sparsity,
        }

    def _generate_batch_report(self, all_results: list):
        """生成批量处理报告"""
        if not all_results:
            print("没有成功处理的文件")
            return

        report_path = self.output_dir / "batch_processing_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("OMP ASC 批量处理报告\n")
            f.write("=" * 50 + "\n\n")

            # 统计信息
            avg_psnr = np.mean([r["psnr"] for r in all_results])
            avg_error = np.mean([r["reconstruction_error"] for r in all_results])
            avg_compression = np.mean([r["compression_ratio"] for r in all_results])

            f.write(f"处理文件数量: {len(all_results)}\n")
            f.write(f"平均PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"平均重构误差: {avg_error:.3f}\n")
            f.write(f"平均压缩比: {avg_compression:.1f}:1\n\n")

            # 详细结果
            f.write("详细结果:\n")
            f.write("-" * 80 + "\n")
            f.write("文件名\t\t\tPSNR(dB)\t重构误差\t压缩比\n")
            f.write("-" * 80 + "\n")

            for result in all_results:
                f.write(
                    f"{result['filename']:<25}\t{result['psnr']:.2f}\t\t"
                    f"{result['reconstruction_error']:.3f}\t\t{result['compression_ratio']:.1f}:1\n"
                )

        print(f"\n批量处理报告已保存: {report_path}")


def quick_start_example():
    """
    快速开始示例
    """
    print("=== OMP ASC算法快速开始示例 ===\n")

    # 步骤1: 设置数据路径
    print("1. 设置数据路径...")
    data_root = "E:/Document/paper_library/3rd_paper_250512/code/OMP_MSTAR_ASC"  # 替换为实际路径

    guide = OMPASCGuide()
    guide.setup_directories(data_root)

    # 步骤2: 查找数据文件
    print("\n2. 查找数据文件...")
    raw_files = guide.find_raw_files()

    if raw_files:
        # 步骤3: 处理单个文件
        print("\n3. 处理单个文件示例...")
        guide.process_single_file(str(raw_files[0]), n_scatterers=40)

        # 步骤4: 批量处理（可选）
        print("\n4. 批量处理示例...")
        # guide.batch_process(max_files=3)  # 取消注释以运行批量处理
    else:
        print("未找到数据文件，请检查路径设置")


def parameter_tuning_guide():
    """
    参数调优指南
    """
    print("=== OMP算法参数调优指南 ===\n")

    print("1. 核心参数说明:")
    print("   - n_scatterers: 稀疏度，建议范围 [20, 60]")
    print("   - position_grid_size: 位置网格大小，建议范围 [8, 32]")
    print("   - amplitude_levels: 幅度级别，建议范围 [3, 10]")
    print("")

    print("2. 性能vs精度权衡:")
    print("   - 更大的字典 -> 更高精度，更长计算时间")
    print("   - 更多散射中心 -> 更详细重构，可能过拟合")
    print("")

    print("3. 推荐配置:")
    print("   - 快速测试: n_scatterers=20, grid_size=8, amplitude_levels=3")
    print("   - 标准配置: n_scatterers=40, grid_size=16, amplitude_levels=5")
    print("   - 高精度: n_scatterers=60, grid_size=32, amplitude_levels=10")


if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 快速开始示例")
    print("2. 参数调优指南")

    choice = input("请输入选择 (1 或 2): ").strip()

    if choice == "1":
        quick_start_example()
    elif choice == "2":
        parameter_tuning_guide()
    else:
        print("无效选择")
