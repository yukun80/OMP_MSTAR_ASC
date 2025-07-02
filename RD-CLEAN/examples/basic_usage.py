"""
RD-CLEAN算法基础使用示例

展示如何使用RD-CLEAN算法进行SAR散射中心提取
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加src路径到系统路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rd_clean_algorithm import RDCleanAlgorithm
from data_loader import SARDataLoader
from physical_model import SARPhysicalModel


def basic_extraction_example():
    """基础散射中心提取示例"""
    print("=== RD-CLEAN算法基础使用示例 ===\n")

    # 创建算法实例
    print("1. 初始化RD-CLEAN算法...")
    algorithm = RDCleanAlgorithm()
    print("   算法初始化完成\n")

    # 模拟加载一个.raw文件 (实际使用时替换为真实文件路径)
    print("2. 模拟数据加载...")

    # 创建模拟SAR数据
    magnitude_image = np.random.rand(128, 128) * 20
    complex_image = magnitude_image + 1j * magnitude_image * 0.2

    # 添加几个模拟散射中心
    # 强散射中心
    magnitude_image[60:68, 60:68] = 300
    complex_image[60:68, 60:68] = 300 + 1j * 75

    # 中等散射中心
    magnitude_image[40:45, 80:85] = 150
    complex_image[40:45, 80:85] = 150 + 1j * 40

    # 弱散射中心
    magnitude_image[80:85, 40:45] = 80
    complex_image[80:85, 40:45] = 80 + 1j * 20

    print(f"   图像尺寸: {magnitude_image.shape}")
    print(f"   动态范围: {np.min(magnitude_image):.1f} - {np.max(magnitude_image):.1f}\n")

    # 执行散射中心提取
    print("3. 执行散射中心提取...")
    try:
        scatterer_list = algorithm._iterative_extraction(magnitude_image, complex_image)
        print(f"   提取完成，共找到 {len(scatterer_list)} 个散射中心\n")

        # 显示提取结果
        print("4. 提取结果:")
        for i, scatterer in enumerate(scatterer_list):
            print(f"   散射中心 {i+1}:")
            print(f"     位置: ({scatterer.x:.3f}, {scatterer.y:.3f}) 米")
            print(f"     类型: {algorithm.classifier.get_scatterer_type_name(scatterer.type)}")
            print(f"     强度: {scatterer.A:.2f}")
            print(f"     频率依赖指数 α: {scatterer.alpha:.2f}")
            if scatterer.type == 0:  # 分布式散射中心
                print(f"     方向角 θ₀: {scatterer.theta0:.2f}°")
                print(f"     长度 L: {scatterer.L:.3f} 米")
            elif scatterer.type == 1:  # 局部散射中心
                print(f"     角度依赖参数 r: {scatterer.r:.3f}")
            print()

    except Exception as e:
        print(f"   提取失败: {e}\n")
        return

    # 图像重构验证
    print("5. 图像重构验证...")
    try:
        reconstructed = algorithm.simulate_scatterers(scatterer_list)

        # 计算重构质量指标
        original_energy = np.sum(magnitude_image**2)
        reconstructed_energy = np.sum(reconstructed**2)
        residual_energy = np.sum((magnitude_image - reconstructed) ** 2)

        reconstruction_ratio = 1 - residual_energy / original_energy
        energy_ratio = reconstructed_energy / original_energy

        print(f"   重构质量: {reconstruction_ratio:.3f}")
        print(f"   能量比例: {energy_ratio:.3f}")
        print(f"   均方误差: {np.mean((magnitude_image - reconstructed) ** 2):.2f}\n")

    except Exception as e:
        print(f"   重构失败: {e}\n")

    # 获取算法统计信息
    print("6. 算法统计信息:")
    try:
        stats = algorithm.get_algorithm_statistics(scatterer_list)

        print(f"   总散射中心数: {stats['total_scatterers']}")
        print("   类型分布:")
        for type_name, count in stats["type_distribution"].items():
            print(f"     {type_name}: {count}")

        amp_stats = stats["amplitude_stats"]
        print(f"   强度统计:")
        print(f"     最小值: {amp_stats['min']:.2f}")
        print(f"     最大值: {amp_stats['max']:.2f}")
        print(f"     平均值: {amp_stats['mean']:.2f}")
        print(f"     标准差: {amp_stats['std']:.2f}")

    except Exception as e:
        print(f"   统计失败: {e}")

    print("\n=== 示例完成 ===")


def file_processing_example():
    """文件处理示例"""
    print("\n=== 文件处理示例 ===\n")

    # 注意：这里需要真实的.raw文件路径
    sample_file = "sample_data.128x128.raw"

    if not os.path.exists(sample_file):
        print(f"示例文件 {sample_file} 不存在")
        print("请将实际的MSTAR .raw文件放置在当前目录下")
        print("文件命名格式: filename.widthxheight.raw")
        return

    print(f"处理文件: {sample_file}")

    # 创建算法实例
    algorithm = RDCleanAlgorithm()

    try:
        # 直接从文件提取
        scatterer_list = algorithm.extract_scatterers(sample_file)

        # 保存结果
        output_file = "extraction_results.pkl"
        algorithm.save_results(scatterer_list, output_file)

        # 加载并验证
        loaded_results = algorithm.load_results(output_file)
        print(f"保存和加载验证: 原始{len(scatterer_list)} vs 加载{len(loaded_results)}")

    except Exception as e:
        print(f"文件处理失败: {e}")


if __name__ == "__main__":
    # 运行基础示例
    basic_extraction_example()

    # 运行文件处理示例（如果有数据文件）
    file_processing_example()
