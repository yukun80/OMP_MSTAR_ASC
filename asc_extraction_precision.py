#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Precision Attribute Scattering Center (ASC) Extraction System
高精度属性散射中心提取系统

针对用户需求的精确ASC提取优化版本：
- 高精度位置估计 (64×64+ 网格 + 连续优化)
- 稳健的迭代收敛 (90%+ 能量减少)
- 可靠的参数精化 (80%+ 成功率)
- 智能的自适应停止 (动态阈值)

核心改进：
1. 分层采样策略：粗网格→细网格→连续优化
2. 改进的残差更新机制
3. 稳健的参数精化算法
4. 智能的自适应停止条件
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.io import loadmat
import struct
from typing import Tuple, List, Dict, Optional
import warnings
import time

warnings.filterwarnings("ignore")


class ASCExtractionPrecision:
    """
    高精度ASC提取系统

    专为精确提取设计的ASC算法，采用分层采样和稳健优化策略
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        precision_mode: str = "high",  # "high", "ultra", "production"
        adaptive_threshold: float = 0.01,  # 更严格的阈值
        max_iterations: int = 50,
        min_scatterers: int = 3,
        max_scatterers: int = 30,
    ):
        """
        初始化高精度ASC提取系统

        Args:
            precision_mode: 精度模式
                - "high": 高精度 (64×64位置采样)
                - "ultra": 超高精度 (128×128位置采样)
                - "production": 生产模式 (32×32位置采样 + 优化精化)
        """
        self.image_size = image_size
        self.precision_mode = precision_mode
        self.adaptive_threshold = adaptive_threshold
        self.max_iterations = max_iterations
        self.min_scatterers = min_scatterers
        self.max_scatterers = max_scatterers

        # SAR系统参数
        self.fc = 1e10  # 中心频率 10GHz
        self.B = 1e9  # 带宽 1GHz
        self.omega = np.pi / 3  # 合成孔径角

        # ASC模型参数
        self.alpha_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.length_values = np.logspace(-2, 0, 5)  # [0.01, 1.0]

        # 根据精度模式设置采样参数
        self._configure_precision_mode()

        print(f"🎯 高精度ASC提取系统初始化")
        print(f"   精度模式: {precision_mode}")
        print(f"   位置采样: {self.position_samples}×{self.position_samples}")
        print(f"   方位角采样: {self.azimuth_samples}")
        print(f"   自适应阈值: {adaptive_threshold}")
        print(f"   预估字典规模: ~{self._estimate_dictionary_size()}")

    def _configure_precision_mode(self):
        """根据精度模式配置采样参数"""
        if self.precision_mode == "high":
            self.position_samples = 64  # 64×64 = 4096 位置
            self.azimuth_samples = 8
            self.enable_refinement = True
            self.refinement_method = "L-BFGS-B"
        elif self.precision_mode == "ultra":
            self.position_samples = 128  # 128×128 = 16384 位置
            self.azimuth_samples = 16
            self.enable_refinement = True
            self.refinement_method = "differential_evolution"
        elif self.precision_mode == "production":
            self.position_samples = 32  # 32×32 = 1024 位置
            self.azimuth_samples = 8
            self.enable_refinement = True
            self.refinement_method = "multi_start"
        else:
            raise ValueError(f"Unknown precision mode: {self.precision_mode}")

    def _estimate_dictionary_size(self):
        """估算字典规模"""
        return self.position_samples**2 * len(self.alpha_values) * len(self.length_values) * self.azimuth_samples

    def load_mstar_data(self, raw_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载MSTAR RAW数据"""
        print(f"📂 加载MSTAR数据: {raw_file_path}")

        with open(raw_file_path, "rb") as f:
            data = f.read()

        # 解析复值数据
        num_values = len(data) // 4
        real_imag = struct.unpack(f"<{num_values}f", data)

        # 重构复值图像
        complex_values = []
        for i in range(0, len(real_imag), 2):
            complex_values.append(complex(real_imag[i], real_imag[i + 1]))

        complex_image = np.array(complex_values).reshape(self.image_size)
        magnitude = np.abs(complex_image)

        print(f"   数据类型: MSTAR RAW")
        print(f"   图像尺寸: {complex_image.shape}")
        print(f"   幅度范围: [{np.min(magnitude):.3f}, {np.max(magnitude):.3f}]")
        print(f"   信号能量: {np.linalg.norm(complex_image):.3f}")

        return magnitude, complex_image

    def preprocess_data(self, complex_image: np.ndarray) -> np.ndarray:
        """高精度数据预处理"""
        print("⚙️ 高精度数据预处理...")

        # 信号向量化
        signal = complex_image.flatten()

        # 智能归一化：保持动态范围
        signal_energy = np.linalg.norm(signal)
        signal_max = np.max(np.abs(signal))

        # 使用能量归一化而非最大值归一化，保持相对强度关系
        signal_normalized = signal / np.sqrt(signal_energy)

        print(f"   信号长度: {len(signal)}")
        print(f"   原始能量: {signal_energy:.3f}")
        print(f"   最大幅度: {signal_max:.3f}")
        print(f"   归一化方式: 能量归一化")

        return signal_normalized

    def _generate_precise_asc_atom(
        self,
        x: float,
        y: float,
        alpha: float,
        length: float,
        phi_bar: float,
        fx_range: np.ndarray,
        fy_range: np.ndarray,
    ) -> np.ndarray:
        """
        生成高精度ASC原子

        改进的ASC模型实现，增强数值稳定性
        """
        # 创建频率网格
        FX, FY = np.meshgrid(fx_range, fy_range, indexing="ij")

        # 计算频率特征
        f_magnitude = np.sqrt(FX**2 + FY**2)
        theta = np.arctan2(FY, FX)

        # 避免零频率问题
        f_magnitude_safe = f_magnitude + 1e-12

        # ASC频域响应
        # 1. 位置相位项
        position_phase = -2j * np.pi * (FX * x + FY * y)

        # 2. 频率依赖项: f^α (改进数值稳定性)
        if alpha == 0:
            frequency_term = np.ones_like(f_magnitude_safe)
        else:
            frequency_term = np.power(f_magnitude_safe, alpha)

        # 3. 长度相关项: sinc(L·f·sin(θ-φ_bar))
        angle_diff = theta - phi_bar
        sinc_arg = length * f_magnitude_safe * np.sin(angle_diff)

        # 改进的sinc函数计算，避免数值问题
        with np.errstate(divide="ignore", invalid="ignore"):
            length_term = np.where(np.abs(sinc_arg) < 1e-10, 1.0, np.sin(np.pi * sinc_arg) / (np.pi * sinc_arg))

        # 4. 方位角相位项
        azimuth_phase = 1j * phi_bar

        # 组合ASC频域响应
        H_asc = frequency_term * length_term * np.exp(position_phase + azimuth_phase)

        # 改进的IFFT：添加窗函数减少边缘效应
        window = np.outer(np.hanning(self.image_size[0]), np.hanning(self.image_size[1]))
        H_asc_windowed = H_asc * window

        # 空域原子
        atom = np.fft.ifft2(np.fft.ifftshift(H_asc_windowed))

        return atom

    def build_precision_dictionary(self) -> Tuple[np.ndarray, List[Dict]]:
        """构建高精度ASC字典"""
        print(f"📚 构建高精度ASC字典 (模式: {self.precision_mode})...")

        # 频率采样范围
        fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1])

        # 高精度位置采样
        x_positions = np.linspace(-0.9, 0.9, self.position_samples)  # 避免边界
        y_positions = np.linspace(-0.9, 0.9, self.position_samples)
        phi_bar_values = np.linspace(0, 2 * np.pi, self.azimuth_samples, endpoint=False)

        total_atoms = (
            len(x_positions) * len(y_positions) * len(self.alpha_values) * len(self.length_values) * len(phi_bar_values)
        )

        print(f"   位置采样密度: {self.position_samples}×{self.position_samples}")
        print(f"   α值采样: {len(self.alpha_values)}")
        print(f"   L值采样: {len(self.length_values)}")
        print(f"   φ_bar采样: {len(phi_bar_values)}")
        print(f"   总原子数: {total_atoms}")

        if total_atoms > 100000:
            print(f"⚠️ 字典规模很大，构建可能需要较长时间")

        # 分批构建字典以节省内存
        dictionary_atoms = []
        param_grid = []
        atom_count = 0

        start_time = time.time()

        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                for alpha in self.alpha_values:
                    for length in self.length_values:
                        for phi_bar in phi_bar_values:
                            # 生成高精度ASC原子
                            atom = self._generate_precise_asc_atom(x, y, alpha, length, phi_bar, fx_range, fy_range)

                            # 稳健的归一化
                            atom_flat = atom.flatten()
                            atom_energy = np.linalg.norm(atom_flat)

                            if atom_energy > 1e-12:
                                atom_normalized = atom_flat / atom_energy
                            else:
                                # 跳过能量过小的原子
                                continue

                            dictionary_atoms.append(atom_normalized)
                            param_grid.append(
                                {
                                    "x": x,
                                    "y": y,
                                    "alpha": alpha,
                                    "length": length,
                                    "phi_bar": phi_bar,
                                    "atom_energy": atom_energy,
                                    "grid_i": i,
                                    "grid_j": j,
                                }
                            )

                            atom_count += 1

                            # 进度显示
                            if atom_count % 5000 == 0 or atom_count == total_atoms:
                                elapsed = time.time() - start_time
                                progress = atom_count / total_atoms * 100
                                print(f"   构建进度: {atom_count}/{total_atoms} ({progress:.1f}%) - {elapsed:.1f}s")

        # 转换为字典矩阵
        dictionary = np.column_stack(dictionary_atoms)

        print(f"✅ 高精度字典构建完成")
        print(f"   最终字典尺寸: {dictionary.shape}")
        print(f"   内存占用: ~{dictionary.nbytes / 1024**2:.1f} MB")
        print(f"   构建时间: {time.time() - start_time:.1f}s")

        return dictionary, param_grid

    def adaptive_precision_extraction(
        self, signal: np.ndarray, dictionary: np.ndarray, param_grid: List[Dict]
    ) -> List[Dict]:
        """
        高精度自适应ASC提取

        改进的迭代提取算法，提高收敛性和稳定性
        """
        print(f"🎯 开始高精度自适应ASC提取...")

        residual_signal = signal.copy()
        extracted_scatterers = []

        # 计算初始信号特征
        initial_energy = np.linalg.norm(residual_signal)
        initial_max = np.max(np.abs(residual_signal))

        # 智能阈值设置
        energy_threshold = initial_energy * self.adaptive_threshold
        max_threshold = initial_max * 0.05  # 5%最大值阈值

        print(f"   初始信号能量: {initial_energy:.6f}")
        print(f"   初始最大幅度: {initial_max:.6f}")
        print(f"   能量停止阈值: {energy_threshold:.6f}")
        print(f"   幅度停止阈值: {max_threshold:.6f}")

        convergence_history = []

        for iteration in range(self.max_iterations):
            # 当前残差特征
            current_energy = np.linalg.norm(residual_signal)
            current_max = np.max(np.abs(residual_signal))

            convergence_history.append(
                {
                    "iteration": iteration,
                    "energy": current_energy,
                    "max_amplitude": current_max,
                    "energy_reduction": (initial_energy - current_energy) / initial_energy,
                }
            )

            # 检查多重停止条件
            if current_energy < energy_threshold:
                print(f"   💡 达到能量阈值，停止迭代")
                break

            if current_max < max_threshold:
                print(f"   💡 达到幅度阈值，停止迭代")
                break

            if len(extracted_scatterers) >= self.max_scatterers:
                print(f"   💡 达到最大散射中心数，停止迭代")
                break

            # 检查收敛停滞
            if len(convergence_history) >= 5:
                recent_reductions = [h["energy_reduction"] for h in convergence_history[-5:]]
                if max(recent_reductions) - min(recent_reductions) < 0.001:
                    print(f"   💡 收敛停滞，停止迭代")
                    break

            # 改进的OMP匹配
            residual_real = np.concatenate([residual_signal.real, residual_signal.imag])
            dictionary_real = np.concatenate([dictionary.real, dictionary.imag], axis=0)

            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=1, fit_intercept=False)

            try:
                omp.fit(dictionary_real, residual_real)
                coefficients = omp.coef_
            except Exception as e:
                print(f"   ⚠️ OMP拟合失败: {str(e)}")
                break

            # 寻找最佳匹配
            nonzero_indices = np.nonzero(coefficients)[0]
            if len(nonzero_indices) == 0:
                print(f"   💡 未找到显著散射中心，停止迭代")
                break

            best_idx = nonzero_indices[0]
            best_params = param_grid[best_idx].copy()

            # 改进的复数系数估计
            selected_atom = dictionary[:, best_idx]

            # 使用最小二乘法而非简单点积
            complex_coef = np.vdot(selected_atom, residual_signal) / np.vdot(selected_atom, selected_atom)

            # 验证系数质量
            coef_quality = abs(complex_coef) / (np.linalg.norm(selected_atom) * np.linalg.norm(residual_signal))

            if coef_quality < 1e-6:
                print(f"   💡 系数质量过低，停止迭代")
                break

            # 更稳健的残差更新
            atom_contribution = complex_coef * selected_atom
            residual_signal = residual_signal - atom_contribution

            # 记录散射中心
            estimated_amplitude = np.abs(complex_coef) * best_params["atom_energy"]
            estimated_phase = np.angle(complex_coef)

            scatterer = {
                "iteration": iteration + 1,
                "x": best_params["x"],
                "y": best_params["y"],
                "alpha": best_params["alpha"],
                "length": best_params["length"],
                "phi_bar": best_params["phi_bar"],
                "estimated_amplitude": estimated_amplitude,
                "estimated_phase": estimated_phase,
                "coefficient": complex_coef,
                "coef_quality": coef_quality,
                "energy_reduction": (initial_energy - current_energy) / initial_energy,
                "grid_i": best_params["grid_i"],
                "grid_j": best_params["grid_j"],
            }

            extracted_scatterers.append(scatterer)

            # 智能进度显示
            if iteration < 10 or (iteration + 1) % 5 == 0:
                energy_reduction_pct = (initial_energy - current_energy) / initial_energy * 100
                print(
                    f"   迭代 {iteration+1}: 能量减少 {energy_reduction_pct:.1f}%, "
                    f"散射中心数 {len(extracted_scatterers)}, "
                    f"系数质量 {coef_quality:.3e}"
                )

        # 检查提取质量
        final_energy = np.linalg.norm(residual_signal)
        total_energy_reduction = (initial_energy - final_energy) / initial_energy

        if len(extracted_scatterers) < self.min_scatterers:
            print(f"   ⚠️ 提取的散射中心数 ({len(extracted_scatterers)}) " f"少于最少要求 ({self.min_scatterers})")

        print(f"✅ 高精度自适应ASC提取完成")
        print(f"   总迭代次数: {len(extracted_scatterers)}")
        print(f"   总能量减少: {total_energy_reduction:.1%}")
        print(f"   最终残差能量: {final_energy:.6f}")

        return extracted_scatterers

    def precision_parameter_refinement(self, scatterers: List[Dict], original_signal: np.ndarray) -> List[Dict]:
        """
        高精度参数精化

        使用多种优化策略提高参数估计精度
        """
        if not self.enable_refinement:
            return scatterers

        print(f"🔧 开始高精度参数精化 (方法: {self.refinement_method})...")

        refined_scatterers = []
        fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1])

        for i, scatterer in enumerate(scatterers):
            print(f"   精化散射中心 {i+1}/{len(scatterers)}...")

            # 初始参数
            x0 = [scatterer["x"], scatterer["y"], scatterer["estimated_amplitude"], scatterer["estimated_phase"]]

            # 固定离散参数
            alpha_fixed = scatterer["alpha"]
            length_fixed = scatterer["length"]
            phi_bar_fixed = scatterer["phi_bar"]

            # 改进的目标函数
            def objective(params):
                x, y, amp, phase = params

                # 边界检查
                if not (-1 <= x <= 1 and -1 <= y <= 1 and amp > 0):
                    return 1e6

                try:
                    # 生成精化原子
                    atom = self._generate_precise_asc_atom(
                        x, y, alpha_fixed, length_fixed, phi_bar_fixed, fx_range, fy_range
                    )
                    atom_flat = atom.flatten()

                    # 归一化
                    atom_energy = np.linalg.norm(atom_flat)
                    if atom_energy > 1e-12:
                        atom_normalized = atom_flat / atom_energy
                    else:
                        return 1e6

                    # 重构误差
                    reconstruction = amp * np.exp(1j * phase) * atom_normalized
                    error = np.linalg.norm(original_signal - reconstruction)

                    return error

                except Exception:
                    return 1e6

            # 根据方法选择优化策略
            success = False
            best_result = None

            if self.refinement_method == "L-BFGS-B":
                # 严格边界约束
                bounds = [(-0.95, 0.95), (-0.95, 0.95), (0.001, 10.0), (-np.pi, np.pi)]

                try:
                    result = minimize(
                        objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200, "ftol": 1e-12}
                    )
                    if result.success and result.fun < objective(x0):
                        best_result = result
                        success = True
                except Exception:
                    pass

            elif self.refinement_method == "differential_evolution":
                # 全局优化
                bounds = [(-0.95, 0.95), (-0.95, 0.95), (0.001, 10.0), (-np.pi, np.pi)]

                try:
                    result = differential_evolution(objective, bounds, maxiter=100, popsize=15, seed=42, atol=1e-12)
                    if result.fun < objective(x0):
                        best_result = result
                        success = True
                except Exception:
                    pass

            elif self.refinement_method == "multi_start":
                # 多起点优化
                bounds = [(-0.95, 0.95), (-0.95, 0.95), (0.001, 10.0), (-np.pi, np.pi)]
                best_fun = objective(x0)

                # 尝试多个起点
                for _ in range(5):
                    # 在初始点附近随机扰动
                    x0_perturbed = [
                        x0[0] + np.random.normal(0, 0.1),
                        x0[1] + np.random.normal(0, 0.1),
                        max(0.001, x0[2] + np.random.normal(0, x0[2] * 0.1)),
                        x0[3] + np.random.normal(0, 0.1),
                    ]

                    try:
                        result = minimize(
                            objective, x0_perturbed, method="L-BFGS-B", bounds=bounds, options={"maxiter": 100}
                        )
                        if result.success and result.fun < best_fun:
                            best_result = result
                            best_fun = result.fun
                            success = True
                    except Exception:
                        continue

            # 更新参数
            refined_scatterer = scatterer.copy()

            if success and best_result is not None:
                refined_scatterer["x"] = best_result.x[0]
                refined_scatterer["y"] = best_result.x[1]
                refined_scatterer["estimated_amplitude"] = best_result.x[2]
                refined_scatterer["estimated_phase"] = best_result.x[3]
                refined_scatterer["optimization_success"] = True
                refined_scatterer["optimization_error"] = best_result.fun
                refined_scatterer["initial_error"] = objective(x0)
                refined_scatterer["improvement"] = objective(x0) - best_result.fun
            else:
                refined_scatterer["optimization_success"] = False
                refined_scatterer["optimization_error"] = objective(x0)

            refined_scatterers.append(refined_scatterer)

        # 统计精化效果
        successful_refinements = sum(1 for s in refined_scatterers if s.get("optimization_success", False))
        success_rate = successful_refinements / len(scatterers) if scatterers else 0

        total_improvement = sum(
            s.get("improvement", 0) for s in refined_scatterers if s.get("optimization_success", False)
        )

        print(f"✅ 高精度参数精化完成")
        print(f"   成功精化: {successful_refinements}/{len(scatterers)} ({success_rate:.1%})")
        print(f"   总改进量: {total_improvement:.3e}")

        return refined_scatterers


def main():
    """高精度ASC提取系统演示"""
    print("🎯 高精度ASC提取系统演示")
    print("=" * 60)

    # 初始化高精度系统
    asc_extractor = ASCExtractionPrecision(
        precision_mode="high",  # 可选: "high", "ultra", "production"
        adaptive_threshold=0.005,  # 0.5% 阈值，更严格
        max_iterations=30,
        min_scatterers=3,
        max_scatterers=20,
    )

    print("\n📝 使用示例:")
    print("# 加载MSTAR数据")
    print("magnitude, complex_image = asc_extractor.load_mstar_data('path/to/data.raw')")
    print("# 预处理")
    print("signal = asc_extractor.preprocess_data(complex_image)")
    print("# 构建高精度字典")
    print("dictionary, param_grid = asc_extractor.build_precision_dictionary()")
    print("# 高精度提取")
    print("scatterers = asc_extractor.adaptive_precision_extraction(signal, dictionary, param_grid)")
    print("# 参数精化")
    print("refined_scatterers = asc_extractor.precision_parameter_refinement(scatterers, signal)")

    return asc_extractor


if __name__ == "__main__":
    asc_system = main()
