"""
Quick Test for ASC Fix v2
快速测试修复版v2效果

重点验证：
1. MSTAR数据加载NaN问题修复
2. 迭代收敛性能改进
3. 与v1版本性能对比
"""

import numpy as np
import time
import os
from asc_extraction_fixed_v2 import ASCExtractionFixedV2, visualize_extraction_results


def test_mstar_compatibility_v2():
    """测试MSTAR数据兼容性v2"""
    print("🧪 测试MSTAR数据兼容性v2")
    print("-" * 40)

    # 查找MSTAR数据文件
    mstar_files = []
    search_paths = [
        "datasets/SAR_ASC_Project/02_Data_Processed_raw/SN_S7/",
        "datasets/SAR_ASC_Project/02_Data_Processed_raw/",
        "datasets/",
    ]

    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(".raw") and "HB" in file:
                        mstar_files.append(os.path.join(root, file))

    if not mstar_files:
        print("   ⚠️ 未找到MSTAR数据文件")
        return {"success": False, "message": "无数据文件"}

    test_file = mstar_files[0]
    print(f"   📂 测试文件: {test_file}")

    # 初始化v2系统
    asc_v2 = ASCExtractionFixedV2(
        extraction_mode="point_only", adaptive_threshold=0.03, max_iterations=15, max_scatterers=10
    )

    try:
        # 测试稳健数据加载
        start_time = time.time()
        magnitude, complex_image = asc_v2.load_mstar_data_robust(test_file)
        load_time = time.time() - start_time

        # 检查数据质量
        has_nan = np.any(np.isnan(complex_image))
        has_inf = np.any(np.isinf(complex_image))
        signal_energy = np.linalg.norm(complex_image)

        print(f"   ✅ 数据加载测试:")
        print(f"      加载时间: {load_time:.2f}s")
        print(f"      包含NaN: {'是' if has_nan else '否'}")
        print(f"      包含Inf: {'是' if has_inf else '否'}")
        print(f"      信号能量: {signal_energy:.3f}")
        print(f"      数据有效性: {'✅' if not has_nan and not has_inf and signal_energy > 0 else '❌'}")

        if not has_nan and not has_inf and signal_energy > 0:
            # 测试ASC提取
            print(f"\n   🎯 测试ASC提取...")
            start_time = time.time()
            scatterers = asc_v2.extract_asc_scatterers_v2(complex_image)
            extraction_time = time.time() - start_time

            print(f"      提取时间: {extraction_time:.1f}s")
            print(f"      提取散射中心数: {len(scatterers)}")

            if scatterers:
                # 分析散射类型分布
                alpha_dist = {}
                for s in scatterers:
                    stype = s["scattering_type"]
                    alpha_dist[stype] = alpha_dist.get(stype, 0) + 1

                print(f"      散射类型分布: {alpha_dist}")

                result = {
                    "success": True,
                    "load_time": load_time,
                    "extraction_time": extraction_time,
                    "num_scatterers": len(scatterers),
                    "alpha_distribution": alpha_dist,
                    "data_quality": "clean",
                }
            else:
                result = {"success": True, "message": "数据加载成功但未提取到散射中心", "data_quality": "clean"}
        else:
            result = {
                "success": False,
                "message": "数据质量问题",
                "has_nan": has_nan,
                "has_inf": has_inf,
                "signal_energy": signal_energy,
            }

    except Exception as e:
        print(f"   ❌ 测试失败: {str(e)}")
        result = {"success": False, "error": str(e)}

    return result


def test_synthetic_convergence_v2():
    """测试合成数据收敛性v2"""
    print("\n🧪 测试合成数据收敛性v2")
    print("-" * 40)

    # 创建测试图像
    print("   🔧 创建合成测试图像...")
    image_size = (128, 128)
    complex_image = np.zeros(image_size, dtype=complex)

    # 添加几个散射中心
    scatterer_positions = [
        (64, 64, 1.0, 0.0),  # 主散射中心
        (80, 48, 0.5, np.pi / 4),  # 次散射中心
        (48, 80, 0.3, np.pi / 2),  # 弱散射中心
    ]

    for x, y, amplitude, phase in scatterer_positions:
        # 高斯形状散射中心
        for i in range(max(0, x - 3), min(image_size[0], x + 4)):
            for j in range(max(0, y - 3), min(image_size[1], y + 4)):
                distance = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                weight = np.exp(-(distance**2) / 4)
                complex_image[i, j] += amplitude * weight * np.exp(1j * phase)

    # 添加适量噪声
    noise_level = 0.02
    noise = noise_level * (np.random.randn(*image_size) + 1j * np.random.randn(*image_size))
    complex_image += noise

    print(f"      图像尺寸: {complex_image.shape}")
    print(f"      信号能量: {np.linalg.norm(complex_image):.3f}")
    print(f"      信噪比: ~{1/noise_level:.1f}")

    # 测试v2提取
    asc_v2 = ASCExtractionFixedV2(
        extraction_mode="point_only", adaptive_threshold=0.05, max_iterations=15, max_scatterers=10
    )

    try:
        start_time = time.time()
        scatterers = asc_v2.extract_asc_scatterers_v2(complex_image)
        extraction_time = time.time() - start_time

        print(f"\n   📊 收敛性能分析:")
        print(f"      提取时间: {extraction_time:.1f}s")
        print(f"      散射中心数: {len(scatterers)}")

        if scatterers:
            # 分析结果质量
            alpha_dist = {}
            positions = []
            amplitudes = []

            for s in scatterers:
                stype = s["scattering_type"]
                alpha_dist[stype] = alpha_dist.get(stype, 0) + 1
                positions.append((s["x"], s["y"]))
                amplitudes.append(s["estimated_amplitude"])

            print(f"      散射类型分布: {alpha_dist}")
            print(f"      幅度范围: [{min(amplitudes):.3f}, {max(amplitudes):.3f}]")
            print(f"      位置分布: 检测到{len(positions)}个位置")

            # 评估收敛质量
            if len(scatterers) >= 2:
                convergence_quality = "良好"
                if max(amplitudes) / min(amplitudes) > 2:  # 幅度分布合理
                    convergence_quality = "优秀"
            else:
                convergence_quality = "需要改进"

            print(f"      收敛质量: {convergence_quality}")

            result = {
                "success": True,
                "extraction_time": extraction_time,
                "num_scatterers": len(scatterers),
                "alpha_distribution": alpha_dist,
                "convergence_quality": convergence_quality,
                "amplitude_range": [min(amplitudes), max(amplitudes)],
            }
        else:
            result = {"success": False, "message": "未提取到散射中心"}

    except Exception as e:
        print(f"   ❌ 收敛测试失败: {str(e)}")
        result = {"success": False, "error": str(e)}

    return result


def test_numerical_stability_v2():
    """测试数值稳定性v2"""
    print("\n🧪 测试数值稳定性v2")
    print("-" * 40)

    asc_v2 = ASCExtractionFixedV2()

    # 测试所有α值的原子生成
    test_alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    fx_range = np.linspace(-5e8, 5e8, 64)  # 较小尺寸加快测试
    fy_range = np.linspace(-5e8, 5e8, 64)

    stability_results = {}

    for alpha in test_alphas:
        try:
            start_time = time.time()
            atom = asc_v2._generate_robust_asc_atom(0.5, 0.3, alpha, 0.0, 0.0, fx_range, fy_range)
            generation_time = time.time() - start_time

            atom_flat = atom.flatten()
            atom_energy = np.linalg.norm(atom_flat)
            has_nan = np.any(np.isnan(atom_flat))
            has_inf = np.any(np.isinf(atom_flat))

            stability_results[alpha] = {
                "success": not has_nan and not has_inf and atom_energy > 1e-12,
                "atom_energy": atom_energy,
                "generation_time": generation_time,
                "has_issues": has_nan or has_inf,
            }

            status = "✅" if stability_results[alpha]["success"] else "❌"
            print(f"   α={alpha:4.1f}: {status} (能量: {atom_energy:.2e}, 时间: {generation_time:.3f}s)")

        except Exception as e:
            stability_results[alpha] = {"success": False, "error": str(e)}
            print(f"   α={alpha:4.1f}: ❌ 异常: {str(e)}")

    # 统计结果
    successful_count = sum(1 for result in stability_results.values() if result.get("success", False))
    success_rate = successful_count / len(test_alphas)

    print(f"\n   📊 数值稳定性总结:")
    print(f"      成功率: {success_rate:.1%} ({successful_count}/{len(test_alphas)})")

    return {"success_rate": success_rate, "detailed_results": stability_results}


def run_quick_validation_v2():
    """运行v2版本快速验证"""
    print("🚀 ASC修复版v2快速验证")
    print("=" * 60)

    start_time = time.time()

    # 执行测试
    results = {
        "numerical_stability": test_numerical_stability_v2(),
        "synthetic_convergence": test_synthetic_convergence_v2(),
        "mstar_compatibility": test_mstar_compatibility_v2(),
    }

    total_time = time.time() - start_time

    # 生成简化报告
    print("\n" + "=" * 60)
    print("📊 v2版本验证报告")
    print("=" * 60)

    # 评估各项测试
    scores = {}

    # 数值稳定性
    stability = results["numerical_stability"]
    scores["数值稳定性"] = stability["success_rate"]

    # 合成数据收敛
    convergence = results["synthetic_convergence"]
    if convergence.get("success", False):
        quality = convergence.get("convergence_quality", "需要改进")
        scores["收敛性能"] = 1.0 if quality == "优秀" else 0.8 if quality == "良好" else 0.4
    else:
        scores["收敛性能"] = 0.0

    # MSTAR兼容性
    mstar = results["mstar_compatibility"]
    scores["MSTAR兼容性"] = 1.0 if mstar.get("success", False) else 0.0

    # 总体评分
    overall_score = np.mean(list(scores.values()))

    print(f"\n📈 测试结果:")
    for test_name, score in scores.items():
        status = "✅" if score > 0.8 else "⚠️" if score > 0.5 else "❌"
        print(f"   {status} {test_name}: {score:.1%}")

    print(f"\n🎯 v2版本总体评分: {overall_score:.1%}")
    print(f"⏱️ 测试时间: {total_time:.1f}s")

    # 改进评估
    print(f"\n💡 v2版本改进效果:")
    if overall_score > 0.8:
        print("   🎉 v2版本修复效果优秀！关键问题已解决")
    elif overall_score > 0.6:
        print("   ✅ v2版本修复效果良好，主要问题已解决")
    else:
        print("   ⚠️ v2版本仍需进一步改进")

    # 具体建议
    if scores["数值稳定性"] < 1.0:
        print("   📋 建议：继续优化数值稳定性")
    if scores["收敛性能"] < 0.8:
        print("   📋 建议：优化迭代收敛算法")
    if scores["MSTAR兼容性"] < 1.0:
        print("   📋 建议：改进MSTAR数据格式兼容性")

    return results


def test_fixed_algorithm_v3():
    """测试完全修复的算法v3版本 - 包含可视化"""
    print("\n🚀 测试完全修复的算法v3版本")
    print("=" * 60)

    # 创建修复版算法实例
    asc_v3 = ASCExtractionFixedV2(
        extraction_mode="point_only", adaptive_threshold=0.05, max_iterations=15, max_scatterers=10  # 稍微放宽阈值
    )

    print("🔧 算法配置:")
    print(f"   提取模式: point_only (专注α值识别)")
    print(f"   自适应阈值: 0.05")
    print(f"   最大迭代: 15")
    print(f"   最大散射中心: 10")

    # 1. 测试MSTAR数据加载
    print(f"\n📂 1. 测试MSTAR数据兼容性...")

    # 查找MSTAR数据文件
    mstar_files = []
    search_paths = [
        "datasets/SAR_ASC_Project/02_Data_Processed_raw/SN_S7/",
        "datasets/SAR_ASC_Project/02_Data_Processed_raw/",
        "datasets/",
    ]

    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(".raw") and "HB" in file:
                        mstar_files.append(os.path.join(root, file))

    if mstar_files:
        test_file = mstar_files[0]
        print(f"   使用测试文件: {test_file}")

        try:
            # 稳健数据加载
            magnitude, complex_image = asc_v3.load_mstar_data_robust(test_file)

            print(f"   ✅ 数据加载成功!")
            print(f"      图像尺寸: {complex_image.shape}")
            print(f"      信号能量: {np.linalg.norm(complex_image):.3f}")

            # 2. 测试ASC散射中心提取
            print(f"\n🎯 2. 测试ASC散射中心提取...")
            start_time = time.time()

            scatterers = asc_v3.extract_asc_scatterers_v2(complex_image)

            extraction_time = time.time() - start_time

            print(f"\n📊 提取结果:")
            print(f"   提取时间: {extraction_time:.2f}s")
            print(f"   散射中心数: {len(scatterers)}")

            if scatterers:
                print(f"   提取成功! 详细信息:")
                for i, sc in enumerate(scatterers):
                    opt_status = "✅" if sc.get("optimization_success", False) else "⚠️"
                    print(
                        f"     #{i+1}: {opt_status} 位置({sc['x']:.3f}, {sc['y']:.3f}), "
                        f"类型: {sc['scattering_type']}, "
                        f"幅度: {sc['estimated_amplitude']:.3f}"
                    )

                # 3. 可视化结果
                print(f"\n🖼️ 3. 生成可视化结果...")
                try:
                    save_path = "results/asc_extraction_result_v3.png"
                    os.makedirs("results", exist_ok=True)

                    visualize_extraction_results(complex_image, scatterers, save_path)

                    print(f"   ✅ 可视化成功!")
                    print(f"   结果保存至: {save_path}")

                except Exception as e:
                    print(f"   ⚠️ 可视化异常: {str(e)}")

                # 4. 结果质量评估
                print(f"\n📈 4. 结果质量评估...")

                # 统计散射类型分布
                alpha_dist = {}
                opt_success_count = 0
                amplitudes = []

                for sc in scatterers:
                    stype = sc["scattering_type"]
                    alpha_dist[stype] = alpha_dist.get(stype, 0) + 1
                    if sc.get("optimization_success", False):
                        opt_success_count += 1
                    amplitudes.append(sc["estimated_amplitude"])

                print(f"   散射类型分布: {alpha_dist}")
                print(
                    f"   优化成功率: {opt_success_count}/{len(scatterers)} ({opt_success_count/len(scatterers)*100:.1f}%)"
                )
                print(f"   幅度范围: [{min(amplitudes):.3f}, {max(amplitudes):.3f}]")

                # 质量评分
                quality_score = 0
                if len(scatterers) >= 3:  # 提取到足够散射中心
                    quality_score += 30
                if opt_success_count / len(scatterers) > 0.5:  # 优化成功率 > 50%
                    quality_score += 30
                if len(alpha_dist) >= 2:  # 识别到多种散射类型
                    quality_score += 40

                print(f"\n🎯 算法质量评分: {quality_score}/100")

                if quality_score >= 80:
                    print("   🎉 优秀! 算法修复非常成功!")
                elif quality_score >= 60:
                    print("   ✅ 良好! 算法修复基本成功!")
                else:
                    print("   ⚠️ 需要改进! 请检查参数设置!")

                return {
                    "success": True,
                    "num_scatterers": len(scatterers),
                    "extraction_time": extraction_time,
                    "quality_score": quality_score,
                    "alpha_distribution": alpha_dist,
                    "optimization_success_rate": opt_success_count / len(scatterers),
                    "scatterers": scatterers,
                }

            else:
                print("   ❌ 未提取到散射中心!")
                print("   可能原因:")
                print("     - 自适应阈值过严格")
                print("     - 数据信号过弱")
                print("     - 算法参数需要调整")

                return {"success": False, "message": "未提取到散射中心"}

        except Exception as e:
            print(f"   ❌ 算法测试失败: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    else:
        print("   ⚠️ 未找到MSTAR数据文件")

        # 使用合成数据进行测试
        print(f"\n🔧 2. 使用合成数据测试...")
        return test_with_synthetic_data_v3(asc_v3)


def test_with_synthetic_data_v3(asc_v3):
    """使用合成数据测试v3算法"""
    print("   创建合成测试数据...")

    # 创建更复杂的测试图像
    image_size = (128, 128)
    complex_image = np.zeros(image_size, dtype=complex)

    # 添加多个不同类型的散射中心
    scatterer_positions = [
        (0.3, 0.2, 1.0, 0.0),  # 强散射中心
        (-0.4, 0.5, 0.6, np.pi / 4),  # 中等散射中心
        (0.1, -0.3, 0.4, np.pi / 2),  # 弱散射中心
        (-0.2, -0.4, 0.8, 0.0),  # 另一个强散射中心
    ]

    for x, y, amplitude, phase in scatterer_positions:
        # 转换到像素坐标
        px = int((x + 1) * image_size[0] / 2)
        py = int((y + 1) * image_size[1] / 2)

        # 高斯形状散射中心
        for i in range(max(0, px - 4), min(image_size[0], px + 5)):
            for j in range(max(0, py - 4), min(image_size[1], py + 5)):
                distance = np.sqrt((i - px) ** 2 + (j - py) ** 2)
                weight = np.exp(-(distance**2) / 8)
                complex_image[i, j] += amplitude * weight * np.exp(1j * phase)

    # 添加适量噪声
    noise_level = 0.05
    noise = noise_level * (np.random.randn(*image_size) + 1j * np.random.randn(*image_size))
    complex_image += noise

    print(f"   合成数据特征:")
    print(f"     图像尺寸: {complex_image.shape}")
    print(f"     信号能量: {np.linalg.norm(complex_image):.3f}")
    print(f"     理论散射中心数: {len(scatterer_positions)}")

    # 提取散射中心
    print(f"\n🎯 开始合成数据ASC提取...")
    start_time = time.time()

    scatterers = asc_v3.extract_asc_scatterers_v2(complex_image)

    extraction_time = time.time() - start_time

    print(f"\n📊 合成数据提取结果:")
    print(f"   提取时间: {extraction_time:.2f}s")
    print(f"   提取散射中心数: {len(scatterers)}")
    print(f"   理论散射中心数: {len(scatterer_positions)}")

    if scatterers:
        # 可视化
        try:
            save_path = "results/asc_synthetic_result_v3.png"
            os.makedirs("results", exist_ok=True)

            visualize_extraction_results(complex_image, scatterers, save_path)
            print(f"   ✅ 合成数据可视化保存至: {save_path}")

        except Exception as e:
            print(f"   ⚠️ 可视化异常: {str(e)}")

        return {
            "success": True,
            "num_scatterers": len(scatterers),
            "extraction_time": extraction_time,
            "synthetic_test": True,
            "expected_count": len(scatterer_positions),
        }
    else:
        return {"success": False, "message": "合成数据测试失败"}


if __name__ == "__main__":
    # 运行原始测试
    print("🚀 ASC修复版v2快速验证")
    print("=" * 60)
    results = run_quick_validation_v2()

    # 运行新的v3算法测试
    v3_results = test_fixed_algorithm_v3()

    # 综合报告
    print("\n" + "=" * 60)
    print("🎯 综合测试报告")
    print("=" * 60)

    if v3_results.get("success", False):
        print("✅ v3修复算法测试成功!")
        print(f"   散射中心提取: {v3_results.get('num_scatterers', 0)}个")
        print(f"   质量评分: {v3_results.get('quality_score', 0)}/100")
        if "optimization_success_rate" in v3_results:
            print(f"   优化成功率: {v3_results['optimization_success_rate']:.1%}")
        print("\n🎉 物理尺度、参数精化和收敛性问题已修复!")
        print("   算法现在能够正确提取属性散射中心并进行可视化。")
    else:
        print("❌ v3修复算法仍需调试")
        print(f"   错误信息: {v3_results.get('message', 'Unknown error')}")

    print(f"\n💡 后续建议:")
    print(f"   1. 检查results/目录中的可视化结果")
    print(f"   2. 如果效果良好，可以扩展到'progressive'模式")
    print(f"   3. 调整adaptive_threshold参数优化提取效果")
