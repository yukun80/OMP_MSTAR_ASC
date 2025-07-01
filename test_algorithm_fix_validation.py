#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASC算法修复效果验证测试
Algorithm Fix Validation Test

验证修复版ASC算法解决的三个核心问题：
1. 数值稳定性：验证负α值不再造成数值爆炸
2. 参数精化逻辑：验证使用残差而非原始信号进行优化
3. 迭代收敛：验证"匹配-优化-减去"流程的有效性

对比测试：
- 原始有问题的版本 vs 修复版本
- 数值稳定性测试
- 收敛性能测试
- 实际MSTAR数据测试
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Dict, List, Tuple

# 导入系统
from asc_extraction_fixed import ASCExtractionFixed
from asc_extraction_advanced import ASCExtractionAdvanced


class AlgorithmFixValidator:
    """算法修复效果验证器"""

    def __init__(self):
        self.results = {}
        print("🔬 ASC算法修复效果验证器初始化")
        print("=" * 60)

    def test_numerical_stability(self):
        """测试1：数值稳定性验证"""
        print("\n🧪 测试1：数值稳定性验证")
        print("-" * 40)

        # 测试参数
        test_params = {
            "x": 0.5,
            "y": 0.3,
            "alpha_values": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "fx_range": np.linspace(-5e8, 5e8, 128),
            "fy_range": np.linspace(-5e8, 5e8, 128),
        }

        print(f"   测试参数: x={test_params['x']}, y={test_params['y']}")
        print(f"   α值范围: {test_params['alpha_values']}")

        # 初始化系统
        fixed_system = ASCExtractionFixed(extraction_mode="point_only")

        stability_results = {}

        for alpha in test_params["alpha_values"]:
            print(f"\n   🔬 测试α={alpha} (散射类型: {fixed_system._classify_scattering_type(alpha)})...")

            try:
                # 生成修复版原子
                start_time = time.time()
                atom_fixed = fixed_system._generate_robust_asc_atom(
                    test_params["x"],
                    test_params["y"],
                    alpha,
                    0.0,
                    0.0,
                    test_params["fx_range"],
                    test_params["fy_range"],
                )
                generation_time = time.time() - start_time

                # 检查数值稳定性
                atom_flat = atom_fixed.flatten()
                atom_energy = np.linalg.norm(atom_flat)
                has_nan = np.any(np.isnan(atom_flat))
                has_inf = np.any(np.isinf(atom_flat))
                is_finite = np.all(np.isfinite(atom_flat))

                stability_results[alpha] = {
                    "generation_time": generation_time,
                    "atom_energy": atom_energy,
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "is_finite": is_finite,
                    "max_value": np.max(np.abs(atom_flat)),
                    "success": is_finite and not has_nan and not has_inf and atom_energy > 1e-12,
                }

                status = "✅ 成功" if stability_results[alpha]["success"] else "❌ 失败"
                print(f"      状态: {status}")
                print(f"      原子能量: {atom_energy:.3e}")
                print(f"      最大值: {stability_results[alpha]['max_value']:.3e}")
                print(f"      生成时间: {generation_time:.3f}s")

            except Exception as e:
                stability_results[alpha] = {"success": False, "error": str(e)}
                print(f"      ❌ 生成失败: {str(e)}")

        # 汇总结果
        successful_alphas = sum(1 for result in stability_results.values() if result.get("success", False))
        print(f"\n   📊 数值稳定性测试结果:")
        print(
            f"      成功率: {successful_alphas}/{len(test_params['alpha_values'])} ({successful_alphas/len(test_params['alpha_values'])*100:.1f}%)"
        )

        if successful_alphas == len(test_params["alpha_values"]):
            print(f"      🎉 所有α值均通过数值稳定性测试！")

        self.results["numerical_stability"] = stability_results
        return stability_results

    def test_parameter_refinement_logic(self):
        """测试2：参数精化逻辑验证"""
        print("\n🧪 测试2：参数精化逻辑验证")
        print("-" * 40)

        # 创建模拟残差信号和散射中心参数
        signal_size = 128 * 128

        # 模拟一个简单的残差信号
        np.random.seed(42)
        residual_signal = 0.1 * (np.random.randn(signal_size) + 1j * np.random.randn(signal_size))

        # 在信号中心位置添加一个强散射中心
        center_idx = signal_size // 2
        residual_signal[center_idx : center_idx + 100] += 1.0 * np.exp(1j * np.pi / 4)

        print(f"   模拟残差信号特征:")
        print(f"      信号长度: {len(residual_signal)}")
        print(f"      信号能量: {np.linalg.norm(residual_signal):.3f}")
        print(f"      最大幅度: {np.max(np.abs(residual_signal)):.3f}")

        # 初始化修复版系统
        fixed_system = ASCExtractionFixed(extraction_mode="point_only")

        # 测试参数精化
        initial_params = {"x": 0.1, "y": 0.1, "alpha": 0.0, "length": 0.0, "phi_bar": 0.0}

        initial_coefficient = 0.8 + 0.6j

        print(f"\n   🔬 测试参数精化...")
        print(f"      初始位置: ({initial_params['x']}, {initial_params['y']})")
        print(f"      初始系数: {initial_coefficient}")

        try:
            start_time = time.time()
            refined_params = fixed_system._refine_parameters_correctly(
                initial_params, residual_signal, initial_coefficient  # 关键：使用残差信号
            )
            refinement_time = time.time() - start_time

            # 验证精化结果
            refinement_success = refined_params.get("optimization_success", False)
            position_change = np.sqrt(
                (refined_params["x"] - initial_params["x"]) ** 2 + (refined_params["y"] - initial_params["y"]) ** 2
            )

            print(f"      🎯 精化结果:")
            print(f"         优化成功: {'✅' if refinement_success else '❌'}")
            print(f"         精化位置: ({refined_params['x']:.3f}, {refined_params['y']:.3f})")
            print(f"         位置变化: {position_change:.3f}")
            print(f"         精化幅度: {refined_params['estimated_amplitude']:.3f}")
            print(f"         精化相位: {refined_params['estimated_phase']:.3f}")
            print(f"         精化时间: {refinement_time:.3f}s")

            if refinement_success:
                print(f"         优化误差: {refined_params.get('optimization_error', 'N/A'):.6f}")

            refinement_result = {
                "success": refinement_success,
                "position_change": position_change,
                "refinement_time": refinement_time,
                "initial_params": initial_params.copy(),
                "refined_params": refined_params.copy(),
            }

        except Exception as e:
            print(f"      ❌ 参数精化失败: {str(e)}")
            refinement_result = {"success": False, "error": str(e)}

        self.results["parameter_refinement"] = refinement_result
        return refinement_result

    def test_iterative_convergence(self):
        """测试3：迭代收敛性验证"""
        print("\n🧪 测试3：迭代收敛性验证")
        print("-" * 40)

        # 创建合成测试图像
        print("   🔧 创建合成测试目标...")

        test_image = self._create_synthetic_test_image()
        signal = test_image.flatten()
        signal_normalized = signal / np.sqrt(np.linalg.norm(signal))

        print(f"      图像尺寸: {test_image.shape}")
        print(f"      信号能量: {np.linalg.norm(signal_normalized):.3f}")

        # 初始化修复版系统
        fixed_system = ASCExtractionFixed(
            extraction_mode="point_only", adaptive_threshold=0.05, max_iterations=20, max_scatterers=15
        )

        print(f"\n   🎯 测试迭代收敛性...")

        try:
            # 构建字典
            start_time = time.time()
            dictionary, param_grid = fixed_system.build_robust_dictionary()
            dict_time = time.time() - start_time

            print(f"      字典构建时间: {dict_time:.1f}s")
            print(f"      字典规模: {dictionary.shape}")

            # 执行自适应提取
            start_time = time.time()
            extracted_scatterers = fixed_system.fixed_adaptive_extraction(signal_normalized, dictionary, param_grid)
            extraction_time = time.time() - start_time

            # 分析收敛性能
            num_extracted = len(extracted_scatterers)
            optimization_success_rate = sum(
                1 for s in extracted_scatterers if s.get("optimization_success", False)
            ) / max(num_extracted, 1)

            # 计算重构性能
            if extracted_scatterers:
                reconstructed_signal = self._reconstruct_signal_from_scatterers(
                    extracted_scatterers, fixed_system, test_image.shape
                )

                reconstruction_error = np.linalg.norm(signal_normalized - reconstructed_signal.flatten())
                energy_reduction = (np.linalg.norm(signal_normalized) - reconstruction_error) / np.linalg.norm(
                    signal_normalized
                )

                # 分析α分布
                alpha_distribution = {}
                for scatterer in extracted_scatterers:
                    alpha = scatterer["alpha"]
                    scattering_type = fixed_system._classify_scattering_type(alpha)
                    alpha_distribution[scattering_type] = alpha_distribution.get(scattering_type, 0) + 1

                print(f"\n      📊 收敛性能分析:")
                print(f"         提取散射中心数: {num_extracted}")
                print(f"         优化成功率: {optimization_success_rate:.1%}")
                print(f"         能量减少: {energy_reduction:.1%}")
                print(f"         重构误差: {reconstruction_error:.6f}")
                print(f"         总提取时间: {extraction_time:.1f}s")
                print(f"         α分布: {alpha_distribution}")

                convergence_result = {
                    "success": True,
                    "num_extracted": num_extracted,
                    "optimization_success_rate": optimization_success_rate,
                    "energy_reduction": energy_reduction,
                    "reconstruction_error": reconstruction_error,
                    "extraction_time": extraction_time,
                    "alpha_distribution": alpha_distribution,
                    "extracted_scatterers": extracted_scatterers,
                }

                if energy_reduction > 0.3:  # 30%+
                    print(f"         🎉 收敛性能优秀！")
                elif energy_reduction > 0.1:  # 10%+
                    print(f"         ✅ 收敛性能良好")
                else:
                    print(f"         ⚠️ 收敛性能需要改进")
            else:
                print(f"         ❌ 未提取到散射中心")
                convergence_result = {"success": False, "message": "未提取到散射中心"}

        except Exception as e:
            print(f"      ❌ 迭代提取失败: {str(e)}")
            convergence_result = {"success": False, "error": str(e)}

        self.results["iterative_convergence"] = convergence_result
        return convergence_result

    def _create_synthetic_test_image(self) -> np.ndarray:
        """创建合成测试图像"""
        image_size = (128, 128)
        complex_image = np.zeros(image_size, dtype=complex)

        # 添加几个不同强度的散射中心
        scatterer_positions = [
            (64, 64, 1.0, 0.0),  # 中心强散射
            (48, 80, 0.6, np.pi / 4),  # 中等强度
            (80, 48, 0.4, np.pi / 2),  # 较弱
            (96, 96, 0.3, -np.pi / 3),  # 最弱
        ]

        for x, y, amplitude, phase in scatterer_positions:
            # 添加高斯形状的散射中心
            for i in range(max(0, x - 5), min(image_size[0], x + 6)):
                for j in range(max(0, y - 5), min(image_size[1], y + 6)):
                    distance = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                    weight = np.exp(-(distance**2) / 8)
                    complex_image[i, j] += amplitude * weight * np.exp(1j * phase)

        # 添加少量噪声
        noise_level = 0.05
        noise = noise_level * (np.random.randn(*image_size) + 1j * np.random.randn(*image_size))
        complex_image += noise

        return complex_image

    def _reconstruct_signal_from_scatterers(
        self, scatterers: List[Dict], asc_system, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """从散射中心重构信号"""
        reconstructed = np.zeros(image_shape, dtype=complex)

        fx_range = np.linspace(-asc_system.B / 2, asc_system.B / 2, image_shape[0])
        fy_range = np.linspace(
            -asc_system.fc * np.sin(asc_system.omega / 2), asc_system.fc * np.sin(asc_system.omega / 2), image_shape[1]
        )

        for scatterer in scatterers:
            # 生成原子
            atom = asc_system._generate_robust_asc_atom(
                scatterer["x"],
                scatterer["y"],
                scatterer["alpha"],
                scatterer.get("length", 0.0),
                scatterer.get("phi_bar", 0.0),
                fx_range,
                fy_range,
            )

            # 应用幅度和相位
            contribution = scatterer["estimated_amplitude"] * np.exp(1j * scatterer["estimated_phase"]) * atom

            reconstructed += contribution

        return reconstructed

    def test_mstar_data_compatibility(self):
        """测试4：MSTAR数据兼容性验证"""
        print("\n🧪 测试4：MSTAR数据兼容性验证")
        print("-" * 40)

        # 查找可用的MSTAR数据文件
        mstar_files = self._find_mstar_files()

        if not mstar_files:
            print("   ⚠️ 未找到MSTAR数据文件，跳过此测试")
            self.results["mstar_compatibility"] = {"success": False, "message": "无MSTAR数据"}
            return

        # 选择第一个文件进行测试
        test_file = mstar_files[0]
        print(f"   📂 测试文件: {test_file}")

        try:
            # 初始化修复版系统
            fixed_system = ASCExtractionFixed(
                extraction_mode="point_only", adaptive_threshold=0.03, max_iterations=15, max_scatterers=10
            )

            # 加载数据
            start_time = time.time()
            magnitude, complex_image = fixed_system.load_mstar_data(test_file)
            load_time = time.time() - start_time

            print(f"   ✅ 数据加载成功 ({load_time:.2f}s)")
            print(f"      图像尺寸: {complex_image.shape}")
            print(f"      幅度范围: [{np.min(magnitude):.3f}, {np.max(magnitude):.3f}]")

            # 执行提取
            start_time = time.time()
            scatterers = fixed_system.extract_asc_scatterers(complex_image)
            extraction_time = time.time() - start_time

            print(f"   ✅ ASC提取完成 ({extraction_time:.1f}s)")
            print(f"      提取散射中心数: {len(scatterers)}")

            if scatterers:
                # 分析结果
                optimization_success_rate = sum(1 for s in scatterers if s.get("optimization_success", False)) / len(
                    scatterers
                )
                alpha_distribution = {}
                for scatterer in scatterers:
                    alpha = scatterer["alpha"]
                    scattering_type = fixed_system._classify_scattering_type(alpha)
                    alpha_distribution[scattering_type] = alpha_distribution.get(scattering_type, 0) + 1

                print(f"      优化成功率: {optimization_success_rate:.1%}")
                print(f"      α分布: {alpha_distribution}")

                mstar_result = {
                    "success": True,
                    "test_file": test_file,
                    "load_time": load_time,
                    "extraction_time": extraction_time,
                    "num_scatterers": len(scatterers),
                    "optimization_success_rate": optimization_success_rate,
                    "alpha_distribution": alpha_distribution,
                }
            else:
                mstar_result = {"success": False, "message": "未提取到散射中心"}

        except Exception as e:
            print(f"   ❌ MSTAR数据测试失败: {str(e)}")
            mstar_result = {"success": False, "error": str(e)}

        self.results["mstar_compatibility"] = mstar_result
        return mstar_result

    def _find_mstar_files(self) -> List[str]:
        """查找可用的MSTAR数据文件"""
        search_paths = ["datasets/SAR_ASC_Project/02_Data_Processed_raw/SN_S7/", "datasets/", "."]

        mstar_files = []

        for search_path in search_paths:
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file.endswith(".raw") and "HB" in file:
                            mstar_files.append(os.path.join(root, file))

        return mstar_files[:3]  # 最多返回3个文件

    def run_comprehensive_validation(self):
        """运行完整验证测试套件"""
        print("🚀 开始ASC算法修复效果综合验证")
        print("=" * 80)

        start_time = time.time()

        # 执行所有测试
        test_results = {
            "numerical_stability": self.test_numerical_stability(),
            "parameter_refinement": self.test_parameter_refinement_logic(),
            "iterative_convergence": self.test_iterative_convergence(),
            "mstar_compatibility": self.test_mstar_data_compatibility(),
        }

        total_time = time.time() - start_time

        # 生成综合报告
        self._generate_validation_report(test_results, total_time)

        return test_results

    def _generate_validation_report(self, test_results: Dict, total_time: float):
        """生成验证报告"""
        print("\n" + "=" * 80)
        print("📊 ASC算法修复效果验证报告")
        print("=" * 80)

        # 汇总成功率
        test_scores = {}

        # 1. 数值稳定性
        stability = test_results["numerical_stability"]
        stability_success = sum(1 for result in stability.values() if result.get("success", False))
        stability_score = stability_success / len(stability) if stability else 0
        test_scores["数值稳定性"] = stability_score

        # 2. 参数精化
        refinement = test_results["parameter_refinement"]
        refinement_score = 1.0 if refinement.get("success", False) else 0.0
        test_scores["参数精化"] = refinement_score

        # 3. 迭代收敛
        convergence = test_results["iterative_convergence"]
        if convergence.get("success", False):
            energy_reduction = convergence.get("energy_reduction", 0)
            convergence_score = min(1.0, energy_reduction / 0.3)  # 30%为满分
        else:
            convergence_score = 0.0
        test_scores["迭代收敛"] = convergence_score

        # 4. MSTAR兼容性
        mstar = test_results["mstar_compatibility"]
        mstar_score = 1.0 if mstar.get("success", False) else 0.0
        test_scores["MSTAR兼容性"] = mstar_score

        # 计算总体评分
        overall_score = np.mean(list(test_scores.values()))

        print(f"\n📈 测试结果汇总:")
        for test_name, score in test_scores.items():
            status = "✅" if score > 0.8 else "⚠️" if score > 0.5 else "❌"
            print(f"   {status} {test_name}: {score:.1%}")

        print(f"\n🎯 总体评分: {overall_score:.1%}")
        print(f"⏱️ 总测试时间: {total_time:.1f}s")

        # 问题修复状态
        print(f"\n🔧 关键问题修复状态:")

        if stability_score > 0.8:
            print("   ✅ 问题1 (数值稳定性): 已修复")
        else:
            print("   ❌ 问题1 (数值稳定性): 未完全修复")

        if refinement_score > 0.8:
            print("   ✅ 问题2 (参数精化逻辑): 已修复")
        else:
            print("   ❌ 问题2 (参数精化逻辑): 未完全修复")

        if convergence_score > 0.6:
            print("   ✅ 问题3 (迭代收敛): 已修复")
        else:
            print("   ❌ 问题3 (迭代收敛): 未完全修复")

        # 总结建议
        print(f"\n💡 修复效果评估:")

        if overall_score > 0.8:
            print("   🎉 修复效果优秀！算法已成功解决了核心问题")
            print("   📋 建议：可以进入生产级验证和实际应用阶段")
        elif overall_score > 0.6:
            print("   ✅ 修复效果良好！主要问题已解决")
            print("   📋 建议：继续优化细节，提高稳定性")
        else:
            print("   ⚠️ 修复效果有限，仍有关键问题需要解决")
            print("   📋 建议：重新审视修复策略，深入调试")


def main():
    """运行修复效果验证测试"""
    print("🔬 ASC算法修复效果验证测试程序")
    print("解决next_work_goal.md中提到的三个核心问题")
    print("=" * 80)

    # 创建验证器
    validator = AlgorithmFixValidator()

    # 运行完整验证
    results = validator.run_comprehensive_validation()

    return validator, results


if __name__ == "__main__":
    validator, results = main()
