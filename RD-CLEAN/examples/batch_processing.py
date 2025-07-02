"""
RD-CLEAN算法批处理示例

对应MATLAB的step3_main_xulu.m，批量处理多个.raw文件
"""

import os
import sys
import glob
import time
from pathlib import Path
from typing import List

# 添加src路径到系统路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rd_clean_algorithm import RDCleanAlgorithm


class BatchProcessor:
    """批处理器"""

    def __init__(self, input_dir: str, output_dir: str):
        """
        初始化批处理器

        Args:
            input_dir: 输入目录 (包含.raw文件)
            output_dir: 输出目录
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.algorithm = RDCleanAlgorithm()

        # 创建输出目录
        self.asc_dir = self.output_dir / "ASC"
        self.recons_dir = self.output_dir / "Reconstruction"
        self.logs_dir = self.output_dir / "Logs"

        for dir_path in [self.asc_dir, self.recons_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def find_raw_files(self) -> List[Path]:
        """查找所有.raw文件"""
        raw_files = []

        # 递归搜索.raw文件
        for pattern in ["**/*.raw", "**/*.RAW"]:
            raw_files.extend(self.input_dir.glob(pattern))

        return sorted(raw_files)

    def process_single_file(self, raw_file: Path) -> dict:
        """
        处理单个.raw文件

        Args:
            raw_file: .raw文件路径

        Returns:
            处理结果字典
        """
        print(f"处理文件: {raw_file.name}")
        start_time = time.time()

        try:
            # 提取散射中心
            scatterer_list = self.algorithm.extract_scatterers(str(raw_file))

            # 生成输出文件名
            base_name = raw_file.stem

            # 保存散射中心结果
            asc_output = self.asc_dir / f"{base_name}_asc.pkl"
            self.algorithm.save_results(scatterer_list, str(asc_output))

            # 生成重构图像
            reconstructed = self.algorithm.simulate_scatterers(scatterer_list)

            # 保存重构结果 (可以保存为numpy文件或图像)
            recons_output = self.recons_dir / f"{base_name}_recons.npy"
            import numpy as np

            np.save(str(recons_output), reconstructed)

            # 计算处理时间
            processing_time = time.time() - start_time

            # 获取统计信息
            stats = self.algorithm.get_algorithm_statistics(scatterer_list)

            result = {
                "file": raw_file.name,
                "status": "success",
                "processing_time": processing_time,
                "num_scatterers": len(scatterer_list),
                "statistics": stats,
                "output_files": {"asc": str(asc_output), "reconstruction": str(recons_output)},
            }

            print(f"  ✓ 成功 - {len(scatterer_list)} 个散射中心，耗时 {processing_time:.1f}s")

        except Exception as e:
            processing_time = time.time() - start_time
            result = {
                "file": raw_file.name,
                "status": "failed",
                "processing_time": processing_time,
                "error": str(e),
                "num_scatterers": 0,
            }

            print(f"  ✗ 失败 - {str(e)}")

        return result

    def process_all_files(self) -> List[dict]:
        """
        批量处理所有文件

        Returns:
            处理结果列表
        """
        # 查找所有.raw文件
        raw_files = self.find_raw_files()

        if not raw_files:
            print(f"在目录 {self.input_dir} 中未找到.raw文件")
            return []

        print(f"找到 {len(raw_files)} 个.raw文件")
        print(f"输出目录: {self.output_dir}")
        print("=" * 50)

        # 逐个处理文件
        results = []
        total_start_time = time.time()

        for i, raw_file in enumerate(raw_files, 1):
            print(f"[{i}/{len(raw_files)}] ", end="")
            result = self.process_single_file(raw_file)
            results.append(result)

        # 总结处理结果
        total_time = time.time() - total_start_time
        self._print_summary(results, total_time)

        # 保存处理日志
        self._save_processing_log(results, total_time)

        return results

    def _print_summary(self, results: List[dict], total_time: float):
        """打印处理总结"""
        print("=" * 50)
        print("处理总结:")

        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]

        print(f"  总文件数: {len(results)}")
        print(f"  成功: {len(successful)}")
        print(f"  失败: {len(failed)}")
        print(f"  总耗时: {total_time:.1f}s")

        if successful:
            total_scatterers = sum(r["num_scatterers"] for r in successful)
            avg_time = sum(r["processing_time"] for r in successful) / len(successful)
            print(f"  总散射中心数: {total_scatterers}")
            print(f"  平均处理时间: {avg_time:.1f}s/文件")

        if failed:
            print("\n失败文件:")
            for result in failed:
                print(f"    {result['file']}: {result['error']}")

    def _save_processing_log(self, results: List[dict], total_time: float):
        """保存处理日志"""
        import json
        from datetime import datetime

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "total_processing_time": total_time,
            "algorithm_info": {"name": "RD-CLEAN", "version": "1.0.0"},
            "results": results,
        }

        log_file = self.logs_dir / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"\n处理日志已保存: {log_file}")


def main():
    """主函数"""
    print("=== RD-CLEAN批处理示例 ===\n")

    # 配置路径 (根据实际情况修改)
    input_directory = "../datasets/raw"  # 输入目录
    output_directory = "../results"  # 输出目录

    # 检查输入目录
    if not os.path.exists(input_directory):
        print(f"输入目录不存在: {input_directory}")
        print("请创建输入目录并放入.raw文件，或修改input_directory变量")

        # 创建示例目录结构
        print("\n创建示例目录结构...")
        os.makedirs(input_directory, exist_ok=True)
        print(f"已创建目录: {input_directory}")
        print("请将.raw文件放入此目录中")
        return

    # 创建批处理器
    processor = BatchProcessor(input_directory, output_directory)

    # 执行批处理
    results = processor.process_all_files()

    # 可选：生成处理报告
    if results:
        generate_processing_report(results, output_directory)


def generate_processing_report(results: List[dict], output_dir: str):
    """生成处理报告"""
    try:
        import matplotlib.pyplot as plt

        # 统计数据
        successful = [r for r in results if r["status"] == "success"]
        processing_times = [r["processing_time"] for r in successful]
        scatterer_counts = [r["num_scatterers"] for r in successful]

        if not successful:
            print("没有成功处理的文件，跳过报告生成")
            return

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 处理时间分布
        axes[0, 0].hist(processing_times, bins=10, alpha=0.7)
        axes[0, 0].set_title("Processing Time Distribution")
        axes[0, 0].set_xlabel("Time (seconds)")
        axes[0, 0].set_ylabel("Frequency")

        # 散射中心数量分布
        axes[0, 1].hist(scatterer_counts, bins=10, alpha=0.7)
        axes[0, 1].set_title("Scatterer Count Distribution")
        axes[0, 1].set_xlabel("Number of Scatterers")
        axes[0, 1].set_ylabel("Frequency")

        # 处理时间 vs 散射中心数量
        axes[1, 0].scatter(scatterer_counts, processing_times, alpha=0.6)
        axes[1, 0].set_title("Processing Time vs Scatterer Count")
        axes[1, 0].set_xlabel("Number of Scatterers")
        axes[1, 0].set_ylabel("Processing Time (seconds)")

        # 成功率饼图
        success_count = len(successful)
        failure_count = len(results) - success_count
        axes[1, 1].pie(
            [success_count, failure_count],
            labels=["Success", "Failed"],
            autopct="%1.1f%%",
            colors=["lightgreen", "lightcoral"],
        )
        axes[1, 1].set_title("Processing Success Rate")

        plt.tight_layout()

        # 保存报告
        report_file = os.path.join(output_dir, "processing_report.png")
        plt.savefig(report_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"处理报告已保存: {report_file}")

    except ImportError:
        print("matplotlib未安装，跳过图表生成")
    except Exception as e:
        print(f"报告生成失败: {e}")


if __name__ == "__main__":
    main()
