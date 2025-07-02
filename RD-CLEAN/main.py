#!/usr/bin/env python3
"""
RD-CLEAN算法主入口脚本

对应MATLAB的step3_main_xulu.m，提供完整的命令行接口
"""

import argparse
import sys
import os
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rd_clean_algorithm import RDCleanAlgorithm
from examples.batch_processing import BatchProcessor


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RD-CLEAN SAR散射中心提取算法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  单文件处理:
    python main.py single input.128x128.raw -o results/
    
  批量处理:
    python main.py batch datasets/raw/ -o results/
    
  测试模式:
    python main.py test
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # 单文件处理模式
    single_parser = subparsers.add_parser("single", help="单文件处理模式")
    single_parser.add_argument("input_file", help="输入.raw文件路径")
    single_parser.add_argument("-o", "--output", default="./results", help="输出目录 (默认: ./results)")
    single_parser.add_argument("--save-reconstruction", action="store_true", help="保存重构图像")
    single_parser.add_argument("--save-visualization", action="store_true", help="保存可视化结果")

    # 批量处理模式
    batch_parser = subparsers.add_parser("batch", help="批量处理模式")
    batch_parser.add_argument("input_dir", help="输入目录路径")
    batch_parser.add_argument("-o", "--output", default="./results", help="输出目录 (默认: ./results)")
    batch_parser.add_argument("--generate-report", action="store_true", help="生成处理报告")

    # 测试模式
    test_parser = subparsers.add_parser("test", help="测试模式")
    test_parser.add_argument(
        "--module", choices=["all", "model", "algorithm", "batch"], default="all", help="测试指定模块"
    )

    args = parser.parse_args()

    if args.mode == "single":
        run_single_file_mode(args)
    elif args.mode == "batch":
        run_batch_mode(args)
    elif args.mode == "test":
        run_test_mode(args)
    else:
        parser.print_help()


def run_single_file_mode(args):
    """单文件处理模式"""
    print("=== RD-CLEAN 单文件处理模式 ===\n")

    # 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        return

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输入文件: {args.input_file}")
    print(f"输出目录: {output_dir}")
    print()

    # 创建算法实例
    algorithm = RDCleanAlgorithm()

    try:
        # 执行提取
        print("开始散射中心提取...")
        scatterer_list = algorithm.extract_scatterers(args.input_file)

        # 保存主要结果
        base_name = Path(args.input_file).stem
        asc_output = output_dir / f"{base_name}_scatterers.pkl"
        algorithm.save_results(scatterer_list, str(asc_output))

        print(f"\n主要结果已保存: {asc_output}")

        # 可选: 保存重构图像
        if args.save_reconstruction:
            reconstructed = algorithm.simulate_scatterers(scatterer_list)
            recons_output = output_dir / f"{base_name}_reconstruction.npy"

            import numpy as np

            np.save(str(recons_output), reconstructed)
            print(f"重构图像已保存: {recons_output}")

        # 可选: 保存可视化结果
        if args.save_visualization:
            try:
                from utils.visualization import plot_scatterer_positions, plot_algorithm_statistics
                import matplotlib.pyplot as plt

                # 绘制散射中心位置
                plt.figure(figsize=(10, 8))
                plot_scatterer_positions(scatterer_list, (128, 128))
                vis_output = output_dir / f"{base_name}_positions.png"
                plt.savefig(str(vis_output), dpi=300, bbox_inches="tight")
                plt.close()

                # 绘制统计图表
                stats = algorithm.get_algorithm_statistics(scatterer_list)
                plot_algorithm_statistics(stats)
                stats_output = output_dir / f"{base_name}_statistics.png"
                plt.savefig(str(stats_output), dpi=300, bbox_inches="tight")
                plt.close()

                print(f"可视化结果已保存: {vis_output}, {stats_output}")

            except ImportError:
                print("matplotlib未安装，跳过可视化保存")

        # 显示统计信息
        stats = algorithm.get_algorithm_statistics(scatterer_list)
        print(f"\n处理完成！")
        print(f"总散射中心数: {stats['total_scatterers']}")
        print("类型分布:")
        for type_name, count in stats["type_distribution"].items():
            print(f"  {type_name}: {count}")

    except Exception as e:
        print(f"处理失败: {e}")
        return


def run_batch_mode(args):
    """批量处理模式"""
    print("=== RD-CLEAN 批量处理模式 ===\n")

    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return

    # 创建批处理器
    processor = BatchProcessor(args.input_dir, args.output)

    # 执行批处理
    results = processor.process_all_files()

    # 可选: 生成报告
    if args.generate_report and results:
        try:
            from examples.batch_processing import generate_processing_report

            generate_processing_report(results, args.output)
        except Exception as e:
            print(f"报告生成失败: {e}")


def run_test_mode(args):
    """测试模式"""
    print("=== RD-CLEAN 测试模式 ===\n")

    if args.module == "all" or args.module == "model":
        print("测试物理模型...")
        try:
            from physical_model import test_physical_model

            test_physical_model()
        except Exception as e:
            print(f"物理模型测试失败: {e}")

    if args.module == "all" or args.module == "algorithm":
        print("\n测试主算法...")
        try:
            from rd_clean_algorithm import test_rd_clean_algorithm

            test_rd_clean_algorithm()
        except Exception as e:
            print(f"主算法测试失败: {e}")

    if args.module == "all" or args.module == "batch":
        print("\n测试批处理功能...")
        try:
            # 创建临时测试数据
            import numpy as np

            test_dir = Path("./test_data")
            test_dir.mkdir(exist_ok=True)

            # 生成测试.raw文件 (模拟格式)
            test_image = np.random.rand(128, 128) * 100
            test_file = test_dir / "test.128x128.raw"

            # 简单的二进制写入 (实际应该按MSTAR格式)
            test_image.astype(np.float32).tofile(str(test_file))

            # 测试批处理
            processor = BatchProcessor(str(test_dir), "./test_results")
            results = processor.process_all_files()

            # 清理测试文件
            import shutil

            shutil.rmtree(str(test_dir), ignore_errors=True)
            shutil.rmtree("./test_results", ignore_errors=True)

        except Exception as e:
            print(f"批处理测试失败: {e}")


if __name__ == "__main__":
    main()
