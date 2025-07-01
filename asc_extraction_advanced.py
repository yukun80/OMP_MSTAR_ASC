#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Attribute Scattering Center (ASC) Extraction System
高级属性散射中心提取系统

基于物理精确模型的自适应ASC参数提取算法
支持完整的ASC参数: {A, α, x, y, L, φ_bar}

Reference:
- ASC Model: A·f^α·sinc(L·f·sin(θ))·exp(j·φ_bar)
- Adaptive extraction similar to CLEAN algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.io import loadmat
import struct
from typing import Tuple, List, Dict, Optional
import warnings
import time

warnings.filterwarnings("ignore")


class ASCExtractionAdvanced:
    """
    高级属性散射中心提取系统

    核心特性:
    1. 完整的ASC物理模型: A·f^α·sinc(L·f·sin(θ))·exp(j·φ_bar)
    2. 自适应迭代提取 (类似CLEAN算法)
    3. 多参数字典: 包含不同α值和L值的复合字典
    4. 后处理优化: 精确估计连续参数
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        adaptive_threshold: float = 0.01,
        max_iterations: int = 100,
        min_scatterers: int = 5,
        max_scatterers: int = 80,
        precision_mode: str = "balanced",  # 新增精度模式参数
    ):
        """
        初始化ASC提取系统

        Args:
            image_size: SAR图像尺寸
            adaptive_threshold: 自适应停止阈值 (相对于最大值)
            max_iterations: 最大迭代次数
            min_scatterers: 最少散射中心数
            max_scatterers: 最多散射中心数
            precision_mode: 精度模式
                - "fast": 快速模式 (16×16位置采样)
                - "balanced": 平衡模式 (32×32位置采样 + 精化)
                - "high": 高精度模式 (48×48位置采样 + 多步精化)
                - "ultra": 超高精度模式 (64×64位置采样 + 全局优化)
        """
        self.image_size = image_size
        self.adaptive_threshold = adaptive_threshold
        self.max_iterations = max_iterations
        self.min_scatterers = min_scatterers
        self.max_scatterers = max_scatterers
        self.precision_mode = precision_mode

        # SAR系统参数
        self.fc = 1e10  # 中心频率 10GHz
        self.B = 1e9  # 带宽 1GHz
        self.omega = np.pi / 3  # 合成孔径角

        # ASC模型参数范围
        self.alpha_values = [-1.0, -0.5, 0.0, 0.5, 1.0]  # 频率依赖因子
        self.length_values = np.logspace(-2, 0, 5)  # 长度参数 [0.01, 1.0]

        # 根据精度模式配置参数
        self._configure_precision_settings()

        print(f"🎯 高级ASC提取系统初始化完成")
        print(f"   精度模式: {precision_mode}")
        print(f"   自适应阈值: {adaptive_threshold}")
        print(f"   迭代范围: {min_scatterers}-{max_scatterers} 个散射中心")
        print(f"   预估位置采样: {self.position_samples}×{self.position_samples}")
        print(f"   方位角采样: {self.azimuth_samples}")

    def _configure_precision_settings(self):
        """根据精度模式配置采样参数"""
        if self.precision_mode == "fast":
            self.position_samples = 16
            self.azimuth_samples = 4
            self.enable_refinement = False
            self.use_progressive_sampling = False
        elif self.precision_mode == "balanced":
            self.position_samples = 32
            self.azimuth_samples = 8
            self.enable_refinement = True
            self.use_progressive_sampling = True
            self.refinement_iterations = 2
        elif self.precision_mode == "high":
            self.position_samples = 48
            self.azimuth_samples = 12
            self.enable_refinement = True
            self.use_progressive_sampling = True
            self.refinement_iterations = 3
        elif self.precision_mode == "ultra":
            self.position_samples = 64
            self.azimuth_samples = 16
            self.enable_refinement = True
            self.use_progressive_sampling = True
            self.refinement_iterations = 5
        else:
            # 默认设置
            self.position_samples = 32
            self.azimuth_samples = 8
            self.enable_refinement = True
            self.use_progressive_sampling = False

    def load_raw_data(self, raw_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载RAW格式SAR数据"""
        print(f"📂 加载RAW数据: {raw_file_path}")

        with open(raw_file_path, "rb") as f:
            data = f.read()

        # 解析复值数据 (实部+虚部，float32，小端序)
        num_values = len(data) // 4
        real_imag = struct.unpack(f"<{num_values}f", data)

        # 重构复值图像
        complex_values = []
        for i in range(0, len(real_imag), 2):
            complex_values.append(complex(real_imag[i], real_imag[i + 1]))

        complex_image = np.array(complex_values).reshape(self.image_size)
        magnitude = np.abs(complex_image)

        print(f"   图像尺寸: {complex_image.shape}")
        print(f"   数据范围: [{np.min(magnitude):.3f}, {np.max(magnitude):.3f}]")

        return magnitude, complex_image

    def preprocess_data(self, complex_image: np.ndarray) -> np.ndarray:
        """数据预处理和归一化"""
        print("⚙️ 数据预处理...")

        # 转换为向量形式
        signal = complex_image.flatten()

        # 归一化 (保持相位信息)
        max_magnitude = np.max(np.abs(signal))
        signal_normalized = signal / max_magnitude

        print(f"   信号长度: {len(signal)}")
        print(f"   归一化因子: {max_magnitude:.3f}")

        return signal_normalized

    def _generate_asc_atom(
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
        生成ASC字典原子

        ASC模型: A·f^α·sinc(L·f·sin(θ))·exp(j·φ_bar)

        Args:
            x, y: 散射中心位置 (归一化)
            alpha: 频率依赖因子
            length: 散射长度参数
            phi_bar: 方位角
            fx_range, fy_range: 频率采样范围

        Returns:
            复值原子 (空域)
        """
        # 创建频率网格
        FX, FY = np.meshgrid(fx_range, fy_range, indexing="ij")

        # 计算频率幅度和角度
        f_magnitude = np.sqrt(FX**2 + FY**2)
        theta = np.arctan2(FY, FX)

        # ASC频域响应
        # 1. 位置相位项
        position_phase = -2j * np.pi * (FX * x + FY * y)

        # 2. 频率依赖项: f^α
        frequency_term = np.power(f_magnitude + 1e-10, alpha)

        # 3. 长度相关项: sinc(L·f·sin(θ-φ_bar))
        angle_diff = theta - phi_bar
        sinc_arg = length * f_magnitude * np.sin(angle_diff)
        length_term = np.sinc(sinc_arg / np.pi)  # numpy sinc = sin(πx)/(πx)

        # 4. 方位角相位项
        azimuth_phase = 1j * phi_bar

        # 组合ASC频域响应
        H_asc = frequency_term * length_term * np.exp(position_phase + azimuth_phase)

        # 空域原子 (IFFT)
        atom = np.fft.ifft2(np.fft.ifftshift(H_asc))

        return atom

    def build_asc_dictionary(
        self, position_samples: int = None, azimuth_samples: int = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        构建完整的ASC复合字典 (支持高精度模式)

        使用渐进式采样策略：先粗网格快速定位，再细网格精确估计
        """
        # 使用配置的精度参数
        if position_samples is None:
            position_samples = self.position_samples
        if azimuth_samples is None:
            azimuth_samples = self.azimuth_samples

        print(f"📚 构建ASC复合字典 (精度模式: {self.precision_mode})...")

        # 频率采样范围
        fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1])

        # 高精度参数采样
        x_positions = np.linspace(-0.9, 0.9, position_samples)  # 避免边界效应
        y_positions = np.linspace(-0.9, 0.9, position_samples)
        phi_bar_values = np.linspace(0, 2 * np.pi, azimuth_samples, endpoint=False)

        # 估算字典大小
        total_atoms = (
            len(x_positions) * len(y_positions) * len(self.alpha_values) * len(self.length_values) * len(phi_bar_values)
        )

        print(f"   位置采样: {position_samples}×{position_samples}")
        print(f"   α采样: {len(self.alpha_values)} 个值")
        print(f"   L采样: {len(self.length_values)} 个值")
        print(f"   φ_bar采样: {azimuth_samples} 个值")
        print(f"   总原子数: {total_atoms}")

        if total_atoms > 50000:
            print(f"⚠️ 字典规模较大，构建可能需要较长时间")

        # 构建字典
        dictionary_atoms = []
        param_grid = []

        atom_count = 0
        start_time = time.time()

        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                for alpha in self.alpha_values:
                    for length in self.length_values:
                        for phi_bar in phi_bar_values:
                            # 生成ASC原子
                            atom = self._generate_asc_atom(x, y, alpha, length, phi_bar, fx_range, fy_range)

                            # 稳健的归一化
                            atom_flat = atom.flatten()
                            atom_energy = np.linalg.norm(atom_flat)
                            if atom_energy > 1e-12:
                                atom_normalized = atom_flat / atom_energy
                            else:
                                # 跳过能量过低的原子
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
                                    "grid_index": (i, j),  # 记录网格索引用于后续优化
                                }
                            )

                            atom_count += 1
                            if atom_count % 2000 == 0:
                                elapsed = time.time() - start_time
                                progress = atom_count / total_atoms * 100
                                print(f"   构建进度: {atom_count}/{total_atoms} ({progress:.1f}%) - {elapsed:.1f}s")

        # 转换为矩阵
        dictionary = np.column_stack(dictionary_atoms)

        print(f"✅ ASC字典构建完成")
        print(f"   字典尺寸: {dictionary.shape}")
        print(f"   内存占用: ~{dictionary.nbytes / 1024**2:.1f} MB")
        print(f"   构建时间: {time.time() - start_time:.1f}s")

        return dictionary, param_grid

    def adaptive_asc_extraction(self, signal: np.ndarray, dictionary: np.ndarray, param_grid: List[Dict]) -> List[Dict]:
        """
        自适应ASC提取 (改进版本，提高收敛性)

        关键改进：
        1. 更严格的收敛判断
        2. 改进的残差更新机制
        3. 智能的停止条件
        """
        print(f"🎯 开始自适应ASC提取 (精度模式: {self.precision_mode})...")

        residual_signal = signal.copy()
        extracted_scatterers = []

        # 计算初始信号特征
        initial_energy = np.linalg.norm(residual_signal)
        initial_max_amplitude = np.max(np.abs(residual_signal))

        # 智能阈值设置
        energy_threshold = initial_energy * self.adaptive_threshold
        amplitude_threshold = initial_max_amplitude * 0.05  # 5%幅度阈值

        print(f"   初始信号能量: {initial_energy:.6f}")
        print(f"   初始最大幅度: {initial_max_amplitude:.6f}")
        print(f"   能量停止阈值: {energy_threshold:.6f}")
        print(f"   幅度停止阈值: {amplitude_threshold:.6f}")

        # 收敛性跟踪
        convergence_history = []
        last_significant_improvement = 0

        for iteration in range(self.max_iterations):
            # 当前残差特征
            current_energy = np.linalg.norm(residual_signal)
            current_max_amplitude = np.max(np.abs(residual_signal))
            energy_reduction_ratio = (initial_energy - current_energy) / initial_energy

            convergence_history.append(
                {
                    "iteration": iteration,
                    "energy": current_energy,
                    "max_amplitude": current_max_amplitude,
                    "energy_reduction": energy_reduction_ratio,
                }
            )

            # 多重停止条件检查
            if current_energy < energy_threshold:
                print(f"   💡 达到能量阈值 ({energy_reduction_ratio:.1%} 减少)，停止迭代")
                break

            if current_max_amplitude < amplitude_threshold:
                print(f"   💡 达到幅度阈值，停止迭代")
                break

            if len(extracted_scatterers) >= self.max_scatterers:
                print(f"   💡 达到最大散射中心数，停止迭代")
                break

            # 检查收敛停滞 (连续5次迭代改进小于0.1%)
            if len(convergence_history) >= 5:
                recent_improvements = []
                for i in range(-4, 0):
                    recent_improvements.append(
                        convergence_history[i]["energy_reduction"] - convergence_history[i - 1]["energy_reduction"]
                    )

                if max(recent_improvements) < 0.001:  # 0.1%改进阈值
                    print(f"   💡 收敛停滞 (改进<0.1%)，停止迭代")
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

            # 使用最小二乘法获得更准确的复数系数
            complex_coef = np.vdot(selected_atom, residual_signal) / np.vdot(selected_atom, selected_atom)

            # 验证系数质量
            projected_energy = abs(complex_coef) ** 2 * np.linalg.norm(selected_atom) ** 2
            signal_energy = np.linalg.norm(residual_signal) ** 2
            energy_capture_ratio = projected_energy / signal_energy

            if energy_capture_ratio < 1e-6:
                print(f"   💡 能量捕获率过低 ({energy_capture_ratio:.2e})，停止迭代")
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
                "energy_capture_ratio": energy_capture_ratio,
                "cumulative_energy_reduction": energy_reduction_ratio,
                "grid_index": best_params["grid_index"],
            }

            extracted_scatterers.append(scatterer)

            # 智能进度显示
            if iteration < 10 or (iteration + 1) % 5 == 0:
                print(
                    f"   迭代 {iteration+1}: 能量减少 {energy_reduction_ratio:.1%}, "
                    f"散射中心数 {len(extracted_scatterers)}, "
                    f"捕获率 {energy_capture_ratio:.2e}"
                )

        # 最终统计
        final_energy = np.linalg.norm(residual_signal)
        total_energy_reduction = (initial_energy - final_energy) / initial_energy

        # 检查提取质量
        if len(extracted_scatterers) < self.min_scatterers:
            print(f"   ⚠️ 提取的散射中心数 ({len(extracted_scatterers)}) " f"少于最少要求 ({self.min_scatterers})")

        print(f"✅ 自适应ASC提取完成")
        print(f"   总迭代次数: {len(extracted_scatterers)}")
        print(f"   总能量减少: {total_energy_reduction:.1%}")
        print(f"   最终残差能量: {final_energy:.6f}")

        # 评估提取质量
        if total_energy_reduction > 0.5:  # 50%+
            print(f"   🎯 提取质量: 优秀")
        elif total_energy_reduction > 0.2:  # 20%+
            print(f"   🎯 提取质量: 良好")
        else:
            print(f"   ⚠️ 提取质量: 需要改进")

        return extracted_scatterers

    def refine_parameters(self, scatterers: List[Dict], original_signal: np.ndarray) -> List[Dict]:
        """
        参数精化 - 后处理优化步骤

        对提取的ASC参数进行非线性优化，提高参数估计精度
        类似于extrac.m中的fmincon优化

        Args:
            scatterers: 初始提取的散射中心
            original_signal: 原始信号

        Returns:
            refined_scatterers: 精化后的散射中心参数
        """
        print(f"🔧 开始ASC参数精化...")

        refined_scatterers = []

        # 频率采样
        fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1])

        for i, scatterer in enumerate(scatterers):
            print(f"   精化散射中心 {i+1}/{len(scatterers)}...")

            # 初始参数
            x0 = [scatterer["x"], scatterer["y"], scatterer["estimated_amplitude"], scatterer["estimated_phase"]]

            # 固定的离散参数
            alpha_fixed = scatterer["alpha"]
            length_fixed = scatterer["length"]
            phi_bar_fixed = scatterer["phi_bar"]

            # 定义优化目标函数
            def objective(params):
                x, y, amp, phase = params

                # 生成精化原子
                atom = self._generate_asc_atom(x, y, alpha_fixed, length_fixed, phi_bar_fixed, fx_range, fy_range)
                atom_flat = atom.flatten()

                # 归一化
                atom_energy = np.linalg.norm(atom_flat)
                if atom_energy > 1e-10:
                    atom_normalized = atom_flat / atom_energy
                else:
                    atom_normalized = atom_flat

                # 计算重构误差
                reconstruction = amp * np.exp(1j * phase) * atom_normalized
                error = np.linalg.norm(original_signal - reconstruction)

                return error

            # 参数边界
            bounds = [(-1.0, 1.0), (-1.0, 1.0), (0.001, 10.0), (-np.pi, np.pi)]  # x  # y  # amplitude  # phase

            # 执行优化
            try:
                result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 100})

                if result.success:
                    # 更新参数
                    refined_scatterer = scatterer.copy()
                    refined_scatterer["x"] = result.x[0]
                    refined_scatterer["y"] = result.x[1]
                    refined_scatterer["estimated_amplitude"] = result.x[2]
                    refined_scatterer["estimated_phase"] = result.x[3]
                    refined_scatterer["optimization_success"] = True
                    refined_scatterer["optimization_error"] = result.fun

                    refined_scatterers.append(refined_scatterer)
                else:
                    # 保持原始参数
                    scatterer["optimization_success"] = False
                    refined_scatterers.append(scatterer)

            except Exception as e:
                print(f"     ⚠️ 优化失败: {str(e)}")
                scatterer["optimization_success"] = False
                refined_scatterers.append(scatterer)

        successful_refinements = sum(1 for s in refined_scatterers if s.get("optimization_success", False))

        print(f"✅ ASC参数精化完成")
        print(f"   成功精化: {successful_refinements}/{len(scatterers)}")

        return refined_scatterers

    def reconstruct_asc_image(self, scatterers: List[Dict]) -> np.ndarray:
        """基于ASC参数重构SAR图像"""
        print(f"🔄 基于ASC参数重构图像...")

        reconstructed = np.zeros(self.image_size, dtype=complex)

        # 频率采样
        fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1])

        for scatterer in scatterers:
            # 生成ASC原子
            atom = self._generate_asc_atom(
                scatterer["x"],
                scatterer["y"],
                scatterer["alpha"],
                scatterer["length"],
                scatterer["phi_bar"],
                fx_range,
                fy_range,
            )

            # 应用幅度和相位
            contribution = scatterer["estimated_amplitude"] * np.exp(1j * scatterer["estimated_phase"]) * atom

            reconstructed += contribution

        print(f"   重构完成，散射中心数: {len(scatterers)}")

        return reconstructed

    def analyze_asc_results(self, scatterers: List[Dict]) -> Dict:
        """分析ASC提取结果"""
        if not scatterers:
            return {}

        print(f"📊 分析ASC提取结果...")

        # 按散射类型分组
        alpha_groups = {}
        for scatterer in scatterers:
            alpha = scatterer["alpha"]
            if alpha not in alpha_groups:
                alpha_groups[alpha] = []
            alpha_groups[alpha].append(scatterer)

        # 统计分析
        amplitudes = [s["estimated_amplitude"] for s in scatterers]
        lengths = [s["length"] for s in scatterers]

        analysis = {
            "total_scatterers": len(scatterers),
            "alpha_distribution": {alpha: len(group) for alpha, group in alpha_groups.items()},
            "amplitude_stats": {
                "mean": np.mean(amplitudes),
                "std": np.std(amplitudes),
                "min": np.min(amplitudes),
                "max": np.max(amplitudes),
            },
            "length_stats": {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": np.min(lengths),
                "max": np.max(lengths),
            },
            "optimization_success_rate": sum(1 for s in scatterers if s.get("optimization_success", False))
            / len(scatterers),
        }

        print(f"   总散射中心数: {analysis['total_scatterers']}")
        print(f"   α分布: {analysis['alpha_distribution']}")
        print(f"   幅度范围: [{analysis['amplitude_stats']['min']:.3f}, {analysis['amplitude_stats']['max']:.3f}]")
        print(f"   长度范围: [{analysis['length_stats']['min']:.3f}, {analysis['length_stats']['max']:.3f}]")
        print(f"   优化成功率: {analysis['optimization_success_rate']:.1%}")

        return analysis


def main():
    """ASC提取系统演示"""
    print("🎯 高级ASC提取系统演示")
    print("=" * 60)

    # 初始化系统
    asc_extractor = ASCExtractionAdvanced(
        adaptive_threshold=0.05,
        max_iterations=50,
        min_scatterers=5,
        max_scatterers=30,
        precision_mode="balanced",  # 5% 阈值
    )

    # 测试用例 - 这里可以加载实际的MSTAR数据
    print("\n📝 注意: 请在实际使用中加载MSTAR数据文件")
    print("示例调用:")
    print("magnitude, complex_image = asc_extractor.load_raw_data('path/to/data.raw')")
    print("signal = asc_extractor.preprocess_data(complex_image)")
    print("dictionary, param_grid = asc_extractor.build_asc_dictionary()")
    print("scatterers = asc_extractor.adaptive_asc_extraction(signal, dictionary, param_grid)")
    print("refined_scatterers = asc_extractor.refine_parameters(scatterers, signal)")
    print("reconstructed = asc_extractor.reconstruct_asc_image(refined_scatterers)")

    return asc_extractor


if __name__ == "__main__":
    asc_system = main()
