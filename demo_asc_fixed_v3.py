"""
ASC散射中心提取演示 - v3修复版本
展示完整的MSTAR数据处理和可视化流程

使用修复后的算法解决：
1. 物理尺度不匹配问题
2. 参数精化逻辑缺失问题
3. 迭代收敛性问题

运行方式：
python demo_asc_fixed_v3.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from asc_extraction_fixed_v2 import ASCExtractionFixedV2, visualize_extraction_results


def find_mstar_files():
    """查找MSTAR数据文件"""
    print("🔍 搜索MSTAR数据文件...")

    search_paths = [
        "datasets/SAR_ASC_Project/02_Data_Processed_raw/SN_S7/",
        "datasets/SAR_ASC_Project/02_Data_Processed_raw/",
        "datasets/",
    ]

    mstar_files = []
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(".raw") and "HB" in file:
                        mstar_files.append(os.path.join(root, file))

    if mstar_files:
        print(f"   找到 {len(mstar_files)} 个MSTAR文件")
        for i, file in enumerate(mstar_files[:3]):  # 显示前3个
            print(f"   {i+1}. {file}")
        if len(mstar_files) > 3:
            print(f"   ... 还有 {len(mstar_files)-3} 个文件")
    else:
        print("   ⚠️ 未找到MSTAR数据文件")

    return mstar_files


def demo_real_data():
    """真实MSTAR数据演示"""
    print("🚀 真实MSTAR数据ASC提取演示")
    print("=" * 50)

    # 查找数据文件
    mstar_files = find_mstar_files()

    if not mstar_files:
        print("跳转到合成数据演示...")
        return demo_synthetic_data()

    # 使用第一个文件进行演示
    test_file = mstar_files[0]
    print(f"\n📂 使用数据文件: {os.path.basename(test_file)}")

    # 创建ASC提取器
    print("\n🔧 初始化ASC提取器...")
    asc_extractor = ASCExtractionFixedV2(
        extraction_mode="point_only",  # 专注点散射α值识别
        adaptive_threshold=0.03,  # 适中的阈值
        max_iterations=20,  # 充分的迭代次数
        max_scatterers=15,  # 合理的散射中心数
    )

    try:
        # 步骤1: 数据加载
        print("\n📥 步骤1: 加载MSTAR数据...")
        start_time = time.time()

        magnitude, complex_image = asc_extractor.load_mstar_data_robust(test_file)

        load_time = time.time() - start_time
        print(f"   ✅ 数据加载成功 ({load_time:.2f}s)")
        print(f"      图像尺寸: {complex_image.shape}")
        print(f"      信号能量: {np.linalg.norm(complex_image):.3f}")

        # 步骤2: ASC散射中心提取
        print("\n🎯 步骤2: ASC散射中心提取...")
        extraction_start = time.time()

        scatterers = asc_extractor.extract_asc_scatterers_v2(complex_image)

        extraction_time = time.time() - extraction_start

        # 步骤3: 结果分析
        print(f"\n📊 步骤3: 结果分析...")
        print(f"   提取时间: {extraction_time:.2f}s")
        print(f"   散射中心数: {len(scatterers)}")

        if scatterers:
            print(f"\n✅ 成功提取散射中心!")

            # 详细信息
            print(f"\n🔍 散射中心详细信息:")
            for i, sc in enumerate(scatterers):
                opt_symbol = "✅" if sc.get("optimization_success", False) else "⚠️"
                print(
                    f"   #{i+1}: {opt_symbol} 位置({sc['x']:.3f}, {sc['y']:.3f}), "
                    f"{sc['scattering_type']}, "
                    f"幅度: {sc['estimated_amplitude']:.3f}"
                )

            # 统计分析
            alpha_dist = {}
            opt_success_count = 0
            for sc in scatterers:
                stype = sc["scattering_type"]
                alpha_dist[stype] = alpha_dist.get(stype, 0) + 1
                if sc.get("optimization_success", False):
                    opt_success_count += 1

            print(f"\n📈 统计分析:")
            print(f"   散射类型分布: {alpha_dist}")
            print(
                f"   优化成功率: {opt_success_count}/{len(scatterers)} ({opt_success_count/len(scatterers)*100:.1f}%)"
            )

            # 步骤4: 可视化
            print(f"\n🖼️ 步骤4: 生成可视化...")

            # 确保results目录存在
            os.makedirs("results", exist_ok=True)
            save_path = f"results/asc_demo_real_data.png"

            visualize_extraction_results(complex_image, scatterers, save_path)

            print(f"   ✅ 可视化完成!")
            print(f"   结果保存至: {save_path}")

            # 算法效果评估
            print(f"\n🎯 算法效果评估:")

            quality_score = 0
            if len(scatterers) >= 5:
                quality_score += 30
                print("   ✅ 散射中心数量充足 (+30分)")
            else:
                print(f"   ⚠️ 散射中心数量较少: {len(scatterers)}")

            if opt_success_count / len(scatterers) > 0.6:
                quality_score += 35
                print("   ✅ 参数优化效果良好 (+35分)")
            else:
                print(f"   ⚠️ 参数优化成功率较低: {opt_success_count/len(scatterers)*100:.1f}%")

            if len(alpha_dist) >= 3:
                quality_score += 35
                print("   ✅ 散射机理识别多样 (+35分)")
            else:
                print(f"   ⚠️ 散射机理识别种类较少: {len(alpha_dist)}")

            print(f"\n🏆 总体质量评分: {quality_score}/100")

            if quality_score >= 80:
                print("   🎉 优秀! 算法修复非常成功!")
            elif quality_score >= 60:
                print("   ✅ 良好! 算法修复基本成功!")
            else:
                print("   ⚠️ 需要改进! 建议调整参数或检查数据质量")

        else:
            print("   ❌ 未能提取到散射中心")
            print("   📋 可能的解决方案:")
            print("      1. 降低adaptive_threshold (当前: 0.03)")
            print("      2. 增加max_iterations (当前: 20)")
            print("      3. 检查数据文件是否有效")

            # 尝试合成数据
            print("\n   🔄 尝试合成数据测试...")
            return demo_synthetic_data()

    except Exception as e:
        print(f"   ❌ 演示过程中出现异常: {str(e)}")
        import traceback

        traceback.print_exc()

        print("\n   🔄 尝试合成数据测试...")
        return demo_synthetic_data()


def demo_synthetic_data():
    """合成数据演示"""
    print("\n🧪 合成数据ASC提取演示")
    print("=" * 50)

    # 创建ASC提取器
    asc_extractor = ASCExtractionFixedV2(
        extraction_mode="point_only", adaptive_threshold=0.05, max_iterations=15, max_scatterers=10
    )

    print("🔧 创建合成测试数据...")

    # 创建包含多个散射中心的合成图像
    image_size = (128, 128)
    complex_image = np.zeros(image_size, dtype=complex)

    # 定义理论散射中心
    theoretical_scatterers = [
        {"pos": (0.4, 0.3), "amp": 1.2, "phase": 0.0, "type": "强反射面"},
        {"pos": (-0.3, 0.5), "amp": 0.8, "phase": np.pi / 4, "type": "边缘结构"},
        {"pos": (0.1, -0.4), "amp": 0.6, "phase": np.pi / 2, "type": "标准散射"},
        {"pos": (-0.5, -0.2), "amp": 0.9, "phase": 0.0, "type": "表面散射"},
    ]

    for sc in theoretical_scatterers:
        x, y = sc["pos"]
        amplitude = sc["amp"]
        phase = sc["phase"]

        # 转换到像素坐标
        px = int((x + 1) * image_size[0] / 2)
        py = int((y + 1) * image_size[1] / 2)

        # 创建高斯形状的散射中心
        sigma = 3.0
        for i in range(max(0, px - 10), min(image_size[0], px + 11)):
            for j in range(max(0, py - 10), min(image_size[1], py + 11)):
                distance = np.sqrt((i - px) ** 2 + (j - py) ** 2)
                weight = np.exp(-(distance**2) / (2 * sigma**2))
                complex_image[i, j] += amplitude * weight * np.exp(1j * phase)

    # 添加适量噪声
    noise_level = 0.02
    noise = noise_level * (np.random.randn(*image_size) + 1j * np.random.randn(*image_size))
    complex_image += noise

    print(f"   合成数据特征:")
    print(f"     图像尺寸: {complex_image.shape}")
    print(f"     理论散射中心: {len(theoretical_scatterers)}个")
    print(f"     信号能量: {np.linalg.norm(complex_image):.3f}")
    print(f"     信噪比: ~{1/noise_level:.1f}")

    # ASC提取
    print(f"\n🎯 开始ASC散射中心提取...")
    start_time = time.time()

    scatterers = asc_extractor.extract_asc_scatterers_v2(complex_image)

    extraction_time = time.time() - start_time

    print(f"\n📊 提取结果:")
    print(f"   提取时间: {extraction_time:.2f}s")
    print(f"   提取散射中心: {len(scatterers)}个")
    print(f"   理论散射中心: {len(theoretical_scatterers)}个")

    if scatterers:
        print(f"\n✅ 成功从合成数据中提取散射中心!")

        # 可视化
        print(f"\n🖼️ 生成可视化结果...")
        os.makedirs("results", exist_ok=True)
        save_path = "results/asc_demo_synthetic_data.png"

        visualize_extraction_results(complex_image, scatterers, save_path)

        print(f"   ✅ 合成数据可视化保存至: {save_path}")

        # 与理论值比较
        print(f"\n🔍 与理论值比较:")
        for i, sc in enumerate(scatterers):
            print(
                f"   #{i+1}: 位置({sc['x']:.3f}, {sc['y']:.3f}), "
                f"{sc['scattering_type']}, "
                f"幅度: {sc['estimated_amplitude']:.3f}"
            )

        detection_rate = len(scatterers) / len(theoretical_scatterers)
        print(f"\n📈 检测率: {detection_rate:.1%} ({len(scatterers)}/{len(theoretical_scatterers)})")

        if detection_rate >= 0.75:
            print("   🎉 检测效果优秀!")
        elif detection_rate >= 0.5:
            print("   ✅ 检测效果良好!")
        else:
            print("   ⚠️ 检测效果需要改进!")
    else:
        print("   ❌ 合成数据测试也失败了")
        print("   这可能表明算法实现仍存在问题")


def main():
    """主演示函数"""
    print("🎯 ASC散射中心提取演示 - v3修复版本")
    print("=" * 60)
    print("本演示展示修复后的算法如何:")
    print("  ✅ 解决物理尺度不匹配问题")
    print("  ✅ 实现真正的参数精化优化")
    print("  ✅ 正确的匹配-优化-减去迭代")
    print("  ✅ 完整的散射中心可视化")
    print("=" * 60)

    try:
        # 首先尝试真实数据
        demo_real_data()

    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断演示")
    except Exception as e:
        print(f"\n\n❌ 演示过程中发生未预期的错误: {str(e)}")
        import traceback

        traceback.print_exc()

    print(f"\n🎊 演示完成!")
    print(f"   请查看results/目录中的可视化结果")
    print(f"   如果效果良好，可以尝试:")
    print(f"     - 调整extraction_mode为'progressive'")
    print(f"     - 处理更多的MSTAR数据文件")
    print(f"     - 优化算法参数以获得更好效果")


if __name__ == "__main__":
    main()
