#!/usr/bin/env python3
"""
RD-CLEAN算法调试脚本

逐步检查算法的每个组件，修复重构问题
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加src路径
sys.path.insert(0, "src")

from rd_clean_algorithm import RDCleanAlgorithm, ScattererParameters
from physical_model import SARPhysicalModel
from data_loader import SARDataLoader


def debug_data_loading(file_path: str):
    """调试数据加载"""
    print("=== 1. 数据加载调试 ===")

    loader = SARDataLoader()

    # 加载数据
    try:
        fileimage, image_value = loader.load_raw_file(file_path)
        print(f"✓ 数据加载成功")
        print(f"  图像尺寸: {fileimage.shape}")
        print(f"  幅度范围: {np.min(fileimage):.4f} - {np.max(fileimage):.4f}")
        print(f"  复数图像类型: {image_value.dtype}")

        # 可视化原始数据
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        im1 = axes[0].imshow(fileimage, cmap="hot")
        axes[0].set_title("Original SAR Image")
        axes[0].axis("off")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(np.abs(image_value), cmap="hot")
        axes[1].set_title("Complex Image Magnitude")
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.savefig("debug_data_loading.png", dpi=300)
        print(f"  原始数据可视化已保存: debug_data_loading.png")

        return fileimage, image_value

    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return None, None


def debug_physical_model():
    """调试物理模型"""
    print("\n=== 2. 物理模型调试 ===")

    model = SARPhysicalModel()

    # 测试单个散射中心模拟
    x, y = 0.1, 0.05  # 位置 (米)
    alpha, r = 0.5, 0.1  # 频率和角度依赖
    theta0, L = 15.0, 0.2  # 方向和长度
    A = 100.0  # 较大的幅度值

    print(f"  测试参数: x={x}, y={y}, A={A}")

    try:
        # 模拟散射中心
        simulated_image = model.simulate_scatterer(x, y, alpha, r, theta0, L, A)
        print(f"✓ 物理模型模拟成功")
        print(f"  模拟图像尺寸: {simulated_image.shape}")
        print(f"  模拟图像范围: {np.min(simulated_image):.6f} - {np.max(simulated_image):.6f}")
        print(f"  模拟图像总能量: {np.sum(simulated_image**2):.2e}")

        # 找到最大值位置
        max_pos = np.unravel_index(np.argmax(simulated_image), simulated_image.shape)
        print(f"  最大值位置: {max_pos}")

        # 可视化模拟结果
        plt.figure(figsize=(8, 6))
        plt.imshow(simulated_image, cmap="hot")
        plt.colorbar()
        plt.title(f"Simulated Scatterer (x={x}, y={y}, A={A})")
        plt.axis("off")
        plt.savefig("debug_physical_model.png", dpi=300)
        print(f"  物理模型测试可视化已保存: debug_physical_model.png")

        return simulated_image

    except Exception as e:
        print(f"✗ 物理模型测试失败: {e}")
        return None


def debug_multi_scatterer_reconstruction():
    """调试多散射中心重构"""
    print("\n=== 3. 多散射中心重构调试 ===")

    model = SARPhysicalModel()

    # 定义多个散射中心
    scatterers = [
        ScattererParameters(x=0.1, y=0.05, alpha=0.5, r=0.1, theta0=15.0, L=0.2, A=100.0, type=1),
        ScattererParameters(x=-0.05, y=0.1, alpha=1.0, r=0.05, theta0=30.0, L=0.1, A=80.0, type=1),
        ScattererParameters(x=0.08, y=-0.03, alpha=0.0, r=0.0, theta0=0.0, L=0.0, A=120.0, type=1),
    ]

    print(f"  测试{len(scatterers)}个散射中心")

    try:
        # 重构每个散射中心
        total_reconstructed = np.zeros((128, 128))
        individual_images = []

        for i, scatterer in enumerate(scatterers):
            single_image = model.simulate_scatterer(
                scatterer.x, scatterer.y, scatterer.alpha, scatterer.r, scatterer.theta0, scatterer.L, scatterer.A
            )
            individual_images.append(single_image)
            total_reconstructed += single_image

            max_val = np.max(single_image)
            max_pos = np.unravel_index(np.argmax(single_image), single_image.shape)
            print(f"  散射中心{i+1}: max={max_val:.4f}, pos={max_pos}")

        print(f"✓ 多散射中心重构成功")
        print(f"  总重构图像范围: {np.min(total_reconstructed):.6f} - {np.max(total_reconstructed):.6f}")

        # 可视化结果
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 显示前3个单独的散射中心
        for i in range(min(3, len(individual_images))):
            row, col = i // 2, i % 2
            im = axes[row, col].imshow(individual_images[i], cmap="hot")
            axes[row, col].set_title(f"Scatterer {i+1}")
            axes[row, col].axis("off")
            plt.colorbar(im, ax=axes[row, col])

        # 显示总重构结果
        im = axes[1, 1].imshow(total_reconstructed, cmap="hot")
        axes[1, 1].set_title("Total Reconstruction")
        axes[1, 1].axis("off")
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig("debug_multi_scatterer.png", dpi=300)
        print(f"  多散射中心重构可视化已保存: debug_multi_scatterer.png")

        return total_reconstructed, individual_images

    except Exception as e:
        print(f"✗ 多散射中心重构失败: {e}")
        return None, None


def debug_algorithm_on_real_data(file_path: str):
    """调试真实数据上的算法"""
    print("\n=== 4. 真实数据算法调试 ===")

    algorithm = RDCleanAlgorithm()

    try:
        # 加载数据
        fileimage, image_value = algorithm.data_loader.load_raw_file(file_path)
        print(f"✓ 数据加载成功")

        # 运行前几次迭代（限制迭代次数以便调试）
        original_max_iter = algorithm.max_iterations
        algorithm.max_iterations = 3  # 只运行3次迭代

        scatterer_list = algorithm.extract_scatterers(file_path)

        # 恢复原始设置
        algorithm.max_iterations = original_max_iter

        print(f"✓ 提取到{len(scatterer_list)}个散射中心")

        # 检查每个散射中心的参数
        print("  散射中心详情:")
        for i, s in enumerate(scatterer_list[:5]):  # 只显示前5个
            print(f"    {i+1}: x={s.x:.3f}, y={s.y:.3f}, A={s.A:.2f}, type={s.type}")

        # 重构图像
        if len(scatterer_list) > 0:
            reconstructed = algorithm.simulate_scatterers(scatterer_list)
            print(f"✓ 重构图像成功")
            print(f"  重构图像范围: {np.min(reconstructed):.6f} - {np.max(reconstructed):.6f}")

            # 可视化对比
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            im1 = axes[0].imshow(fileimage, cmap="hot")
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            plt.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(reconstructed, cmap="hot")
            axes[1].set_title("Reconstructed Image")
            axes[1].axis("off")
            plt.colorbar(im2, ax=axes[1])

            residual = np.abs(fileimage - reconstructed)
            im3 = axes[2].imshow(residual, cmap="hot")
            axes[2].set_title("Residual Image")
            axes[2].axis("off")
            plt.colorbar(im3, ax=axes[2])

            plt.tight_layout()
            plt.savefig("debug_algorithm_real_data.png", dpi=300)
            print(f"  算法调试可视化已保存: debug_algorithm_real_data.png")

            # 计算重构质量
            original_energy = np.sum(fileimage**2)
            residual_energy = np.sum(residual**2)
            reconstruction_quality = 1 - residual_energy / original_energy if original_energy > 0 else 0
            print(f"  重构质量: {reconstruction_quality:.4f}")

        return scatterer_list

    except Exception as e:
        print(f"✗ 真实数据算法测试失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def analyze_matlab_structure():
    """分析MATLAB数据结构"""
    print("\n=== 5. MATLAB数据结构分析 ===")

    print("MATLAB中的scatter_all结构:")
    print("  - scatter_all: 36x1 cell array")
    print("  - 每个cell包含: [x, y, alpha, r, theta0, L, A] (7个参数)")
    print("  - Python等价结构: List[List[float]]")

    # 模拟MATLAB的数据结构
    scatter_all_matlab_style = []

    # 创建一些示例数据
    for i in range(5):
        scatterer_params = [
            0.1 * i,  # x
            0.05 * i,  # y
            0.5,  # alpha
            0.1,  # r
            15.0,  # theta0
            0.2,  # L
            100.0 - i * 10,  # A
        ]
        scatter_all_matlab_style.append(scatterer_params)

    print(f"  Python模拟结构: {len(scatter_all_matlab_style)}个散射中心")
    for i, params in enumerate(scatter_all_matlab_style):
        print(f"    散射中心{i+1}: {params}")

    return scatter_all_matlab_style


def main():
    """主调试函数"""
    print("RD-CLEAN算法全面调试")
    print("=" * 50)

    # 设置matplotlib支持中文（如果可能）
    plt.rcParams["axes.unicode_minus"] = False

    # 测试文件
    test_file = "../datasets/SAR_ASC_Project/02_Data_Processed_raw/HB03344.017.128x128.raw"

    if not os.path.exists(test_file):
        print(f"错误: 测试文件不存在: {test_file}")
        return

    # 1. 数据加载调试
    fileimage, image_value = debug_data_loading(test_file)

    # 2. 物理模型调试
    simulated_image = debug_physical_model()

    # 3. 多散射中心重构调试
    total_reconstructed, individual_images = debug_multi_scatterer_reconstruction()

    # 4. 真实数据算法调试
    scatterer_list = debug_algorithm_on_real_data(test_file)

    # 5. MATLAB数据结构分析
    matlab_structure = analyze_matlab_structure()

    print("\n" + "=" * 50)
    print("调试完成!")
    print("请检查生成的调试图像:")
    print("  - debug_data_loading.png: 数据加载测试")
    print("  - debug_physical_model.png: 物理模型测试")
    print("  - debug_multi_scatterer.png: 多散射中心重构测试")
    print("  - debug_algorithm_real_data.png: 真实数据算法测试")


if __name__ == "__main__":
    main()
