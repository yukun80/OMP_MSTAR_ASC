"""
Fixed ASC Extraction System v2
修复版ASC提取系统 v2

在v1基础上进一步修复：
1. MSTAR数据加载NaN问题
2. 重构误差计算逻辑
3. 迭代收敛性能优化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import OrthogonalMatchingPursuit
import struct
from typing import Tuple, List, Dict, Optional
import warnings
import time

warnings.filterwarnings("ignore")


class ASCExtractionFixedV2:
    """修复版ASC提取系统 v2"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        extraction_mode: str = "progressive",
        adaptive_threshold: float = 0.01,
        max_iterations: int = 30,
        max_scatterers: int = 20,
        # New flexible parameters for dictionary building
        alpha_values: Optional[List[float]] = None,
        length_values: Optional[List[float]] = None,
        phi_bar_values: Optional[List[float]] = None,
        position_samples: Optional[int] = None,
    ):
        self.image_size = image_size
        self.extraction_mode = extraction_mode
        self.adaptive_threshold = adaptive_threshold
        self.max_iterations = max_iterations
        self.max_scatterers = max_scatterers

        # SAR系统参数
        self.fc = 1e10  # 中心频率
        self.B = 1e9  # 带宽
        self.omega = np.pi / 3  # 合成孔径角
        self.scene_size = 30.0  # 场景尺寸 (米)

        # 配置参数
        if alpha_values is not None:
            print("   🔧 使用自定义字典参数进行初始化")
            self.alpha_values = alpha_values
            self.length_values = length_values if length_values is not None else [0.0]
            self.phi_bar_values = phi_bar_values if phi_bar_values is not None else [0.0]
            self.position_samples = position_samples if position_samples is not None else 64
        else:
            self._configure_extraction_mode()

        print(f"🔧 修复版ASC提取系统v2初始化")
        print(f"   提取模式: {extraction_mode}")
        print(f"   自适应阈值: {adaptive_threshold}")
        print(f"   场景尺寸: {self.scene_size}m")

    def _configure_extraction_mode(self):
        """配置提取模式参数"""
        if self.extraction_mode == "point_only":
            self.alpha_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
            self.length_values = [0.0]
            self.phi_bar_values = [0.0]
            self.position_samples = 24  # 降低采样提高速度
            print("   🎯 点散射模式：专注α识别")
        else:
            self.alpha_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
            self.length_values = [0.0, 0.1, 0.5]
            self.phi_bar_values = [0.0, np.pi / 4, np.pi / 2]
            self.position_samples = 20

    def load_mstar_data_robust(self, raw_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        稳健的MSTAR数据加载 - 修复版v3
        正确解析[幅度..., 相位...]格式的big-endian数据
        """
        print(f"📂 稳健加载MSTAR数据 (v3): {raw_file_path}")

        try:
            with open(raw_file_path, "rb") as f:
                data = f.read()

            # 根据MATLAB脚本，数据为float32, big-endian
            num_floats = len(data) // 4
            # 使用 '>' 指定 big-endian
            all_values = struct.unpack(f">{num_floats}f", data)
            all_values = np.array(all_values)

            # --- 关键修复：正确分离幅度和相位 ---
            num_pixels = self.image_size[0] * self.image_size[1]
            if len(all_values) != 2 * num_pixels:
                raise ValueError(f"数据尺寸不匹配，期望 {2*num_pixels} 个值，实际得到 {len(all_values)}")

            magnitude_flat = all_values[:num_pixels]
            phase_flat = all_values[num_pixels:]

            # --- 重构复数图像 ---
            complex_image_flat = magnitude_flat * np.exp(1j * phase_flat)
            complex_image = complex_image_flat.reshape(self.image_size)

            # 数据有效性检查
            if np.any(np.isnan(complex_image)):
                print("   ⚠️ 检测到NaN值，进行清理...")
                complex_image = np.nan_to_num(complex_image)

            magnitude = np.abs(complex_image)

            print(f"   ✅ 数据加载成功")
            print(f"      图像尺寸: {complex_image.shape}")
            print(f"      信号能量: {np.linalg.norm(complex_image):.3f}")

            return magnitude, complex_image

        except Exception as e:
            print(f"   ❌ 数据加载失败: {str(e)}")
            # 返回零数据
            return np.zeros(self.image_size), np.zeros(self.image_size, dtype=complex)

    def preprocess_data_robust(self, complex_image: np.ndarray) -> np.ndarray:
        """稳健的数据预处理"""
        print("⚙️ 稳健数据预处理...")

        # 信号向量化
        signal = complex_image.flatten()

        # 检查和清理数据
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            print("   ⚠️ 检测到异常值，进行清理...")
            signal = np.where(np.isnan(signal) | np.isinf(signal), 0.0 + 0.0j, signal)

        # 计算信号特征
        signal_energy = np.linalg.norm(signal)
        signal_max = np.max(np.abs(signal))

        if signal_energy < 1e-12:
            print("   ⚠️ 信号能量过低，使用模拟数据...")
            # 创建简单的测试信号
            signal = 0.1 * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
            signal_energy = np.linalg.norm(signal)

        # 稳健归一化
        signal_normalized = signal / np.sqrt(signal_energy)

        print(f"   信号长度: {len(signal)}")
        print(f"   处理后能量: {np.linalg.norm(signal_normalized):.3f}")
        print(f"   最大幅度: {np.max(np.abs(signal_normalized)):.3f}")

        return signal_normalized

    def _generate_robust_asc_atom(
        self,
        x: float,
        y: float,
        alpha: float,
        length: float = 0.0,  # 默认为点散射体
        phi_bar: float = 0.0,
        fx_range: np.ndarray = None,
        fy_range: np.ndarray = None,
    ) -> np.ndarray:
        """v3版本: 修复了sinc函数参数的物理模型"""
        if fx_range is None:
            fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        if fy_range is None:
            fy_range = np.linspace(
                -self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1]
            )

        FX, FY = np.meshgrid(fx_range, fy_range, indexing="ij")

        # --- 关键修复：统一物理尺度 ---
        C = 299792458.0  # 光速
        x_meters = x * (self.scene_size / 2.0)  # 将归一化坐标[-1,1]转为米
        y_meters = y * (self.scene_size / 2.0)

        f_magnitude = np.sqrt(FX**2 + FY**2)

        theta = np.arctan2(FY, FX)

        # 1. 频率依赖项 (f/fc)^α - 数值稳定版本
        f_magnitude_safe = np.where(f_magnitude < 1e-9, 1e-9, f_magnitude)
        if alpha == 0:
            frequency_term = np.ones_like(f_magnitude_safe)
        else:
            normalized_freq = f_magnitude_safe / self.fc
            frequency_term = np.power(normalized_freq, alpha)

        # 2. 位置相位项 - 修复物理尺度
        # 正确公式: exp(-j*2*pi/c * (FX*x_m + FY*y_m))
        position_phase = -2j * np.pi / C * (FX * x_meters + FY * y_meters)

        # 3. 长度/方位角项 - 修复物理公式
        length_term = np.ones_like(f_magnitude_safe, dtype=float)
        if length > 1e-6:
            k = 2 * np.pi * f_magnitude_safe / C
            angle_diff = theta - phi_bar

            # --- 关键修复：修正sinc函数的参数 ---
            # 物理项 Y = k * length * np.sin(angle_diff) / 2
            # 我们需要计算 sinc(Y/pi)
            Y = k * length * np.sin(angle_diff) / 2  # 注意这里的除2是针对线状散射体模型
            sinc_arg = Y / np.pi
            length_term = np.sinc(sinc_arg)

        # 组合频域响应
        H_asc = frequency_term * length_term * np.exp(position_phase)

        # IFFT 到空域
        atom = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(H_asc)))

        return atom

    def estimate_params_in_roi(
        self, complex_image: np.ndarray, center_x: float, center_y: float, roi_size: int = 24
    ) -> Optional[Dict]:
        """
        第二阶段V2版本：在ROI内进行"模型匹配+局部微调"
        """
        img_h, img_w = self.image_size
        px = int((center_x + 1) / 2 * img_w)
        py = int((center_y + 1) / 2 * img_h)
        half_size = roi_size // 2

        y_start, y_end = py - half_size, py + half_size
        x_start, x_end = px - half_size, px + half_size
        if not (0 <= y_start < y_end <= img_h and 0 <= x_start < x_end <= img_w):
            return None

        roi_signal = complex_image[y_start:y_end, x_start:x_end]

        # --- 步骤A: 模型匹配 ---
        # 在离散的参数空间（alpha, length, phi_bar）中找到最佳模型
        best_match = {"error": float("inf")}

        for alpha in self.alpha_values:
            for length in self.length_values:
                for phi_bar in self.phi_bar_values:
                    # 生成理论原子
                    atom_full = self._generate_robust_asc_atom(center_x, center_y, alpha, length, phi_bar)
                    atom_roi = atom_full[y_start:y_end, x_start:x_end]

                    # 计算该模型下的最佳复幅度 (通过投影)
                    atom_energy = np.linalg.norm(atom_roi)
                    if atom_energy < 1e-9:
                        continue

                    complex_amp = np.vdot(atom_roi, roi_signal) / atom_energy**2

                    # 计算该模型下的拟合误差
                    error = np.linalg.norm(roi_signal - complex_amp * atom_roi)

                    if error < best_match["error"]:
                        best_match = {
                            "error": error,
                            "alpha": alpha,
                            "length": length,
                            "phi_bar": phi_bar,
                            "amp": np.abs(complex_amp),
                            "phase": np.angle(complex_amp),
                        }

        # --- 步骤B: 局部微调 ---
        # 使用上一步找到的最佳参数作为初始值，对连续参数(x, y, A, φ)进行微调

        # 固定的离散参数
        alpha_fixed = best_match["alpha"]
        length_fixed = best_match["length"]
        phi_bar_fixed = best_match["phi_bar"]

        def objective(params):  # x, y, amp, phase
            x, y, amp, phase = params
            atom_full = self._generate_robust_asc_atom(x, y, alpha_fixed, length_fixed, phi_bar_fixed)
            atom_roi = atom_full[y_start:y_end, x_start:x_end]
            reconstruction = amp * np.exp(1j * phase) * atom_roi
            return np.linalg.norm(roi_signal - reconstruction)

        x0 = [center_x, center_y, best_match["amp"], best_match["phase"]]
        # 限制微调范围，防止优化跑飞
        bounds = [
            (center_x - 0.1, center_x + 0.1),
            (center_y - 0.1, center_y + 0.1),
            (0, 2 * best_match["amp"]),
            (-np.pi, np.pi),
        ]

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 50})

        if result.success:
            return {
                "alpha": alpha_fixed,
                "length": length_fixed,
                "phi_bar": phi_bar_fixed,
                "x": result.x[0],
                "y": result.x[1],
                "estimated_amplitude": result.x[2],
                "estimated_phase": result.x[3],
                "scattering_type": self._classify_scattering_type(alpha_fixed),
                "optimization_success": True,
            }
        else:  # 优化失败，返回粗匹配结果
            return {
                "alpha": best_match["alpha"],
                "length": best_match["length"],
                "phi_bar": best_match["phi_bar"],
                "x": center_x,
                "y": center_y,
                "estimated_amplitude": best_match["amp"],
                "estimated_phase": best_match["phase"],
                "scattering_type": self._classify_scattering_type(best_match["alpha"]),
                "optimization_success": False,
            }

    def build_compact_dictionary(self) -> Tuple[np.ndarray, List[Dict]]:
        """构建紧凑高效的字典"""
        print(f"📚 构建紧凑ASC字典...")

        # 频率采样
        fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1])

        # 位置采样
        x_positions = np.linspace(-0.8, 0.8, self.position_samples)
        y_positions = np.linspace(-0.8, 0.8, self.position_samples)

        dictionary_atoms = []
        param_grid = []

        valid_count = 0
        total_count = 0

        for x in x_positions:
            for y in y_positions:
                for alpha in self.alpha_values:
                    for length in self.length_values:
                        for phi_bar in self.phi_bar_values:
                            total_count += 1

                            atom = self._generate_robust_asc_atom(x, y, alpha, length, phi_bar, fx_range, fy_range)

                            atom_flat = atom.flatten()
                            atom_energy = np.linalg.norm(atom_flat)

                            # 检查原子有效性
                            if (
                                atom_energy > 1e-12
                                and np.isfinite(atom_energy)
                                and not np.any(np.isnan(atom_flat))
                                and not np.any(np.isinf(atom_flat))
                            ):

                                atom_normalized = atom_flat / atom_energy
                                dictionary_atoms.append(atom_normalized)
                                param_grid.append(
                                    {
                                        "x": x,
                                        "y": y,
                                        "alpha": alpha,
                                        "length": length,
                                        "phi_bar": phi_bar,
                                        "atom_energy": atom_energy,
                                        "scattering_type": self._classify_scattering_type(alpha),
                                    }
                                )
                                valid_count += 1

        dictionary = np.column_stack(dictionary_atoms)

        print(f"   有效原子: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
        print(f"   字典尺寸: {dictionary.shape}")
        print(f"   内存占用: ~{dictionary.nbytes / 1024**2:.1f} MB")

        return dictionary, param_grid

    def _classify_scattering_type(self, alpha: float) -> str:
        """散射类型分类"""
        types = {-1.0: "尖顶绕射", -0.5: "边缘绕射", 0.0: "标准散射", 0.5: "表面散射", 1.0: "镜面反射"}
        return types.get(alpha, f"α={alpha}")

    def improved_adaptive_extraction(
        self, signal: np.ndarray, dictionary: np.ndarray, param_grid: List[Dict]
    ) -> List[Dict]:
        """改进的自适应提取算法"""
        print(f"🎯 开始改进版自适应提取...")

        residual_signal = signal.copy()
        extracted_scatterers = []

        initial_energy = np.linalg.norm(residual_signal)
        energy_threshold = initial_energy * self.adaptive_threshold

        print(f"   初始能量: {initial_energy:.6f}")
        print(f"   停止阈值: {energy_threshold:.6f}")

        for iteration in range(self.max_iterations):
            current_energy = np.linalg.norm(residual_signal)

            # 多重停止条件
            if current_energy < energy_threshold:
                print(f"   💡 达到能量阈值，停止迭代")
                break

            if len(extracted_scatterers) >= self.max_scatterers:
                print(f"   💡 达到最大散射中心数，停止迭代")
                break

            # 检查停滞
            if len(extracted_scatterers) >= 3:
                recent_energies = [s.get("residual_energy", current_energy) for s in extracted_scatterers[-3:]]
                if max(recent_energies) - min(recent_energies) < current_energy * 0.001:
                    print(f"   💡 能量减少停滞，停止迭代")
                    break

            # 找到最佳匹配
            best_idx, best_coef = self._find_best_match_robust(residual_signal, dictionary)
            if best_idx is None:
                print(f"   💡 未找到有效匹配，停止迭代")
                break

            initial_params = param_grid[best_idx].copy()

            # 参数精化
            refined_params = self._refine_point_scatterer_v2(initial_params, residual_signal, best_coef)

            # 计算贡献并更新残差
            contribution = self._calculate_scatterer_contribution(refined_params)

            if np.linalg.norm(contribution) < current_energy * 0.001:
                print(f"   💡 散射中心贡献过小，停止迭代")
                break

            # 更新残差
            new_residual = residual_signal - contribution
            new_energy = np.linalg.norm(new_residual)

            # 验证能量减少
            if new_energy >= current_energy * 0.999:  # 几乎没有改善
                print(f"   💡 能量减少不足，停止迭代")
                break

            residual_signal = new_residual
            refined_params["residual_energy"] = new_energy
            extracted_scatterers.append(refined_params)

            if iteration < 5 or (iteration + 1) % 5 == 0:
                reduction = (current_energy - new_energy) / current_energy
                print(f"   迭代 {iteration+1}: {current_energy:.6f} → {new_energy:.6f} (减少{reduction:.2%})")

        final_energy = np.linalg.norm(residual_signal)
        total_reduction = (initial_energy - final_energy) / initial_energy

        print(f"✅ 改进版提取完成")
        print(f"   散射中心数: {len(extracted_scatterers)}")
        print(f"   总能量减少: {total_reduction:.1%}")

        return extracted_scatterers

    def _find_best_match_robust(
        self, signal: np.ndarray, dictionary: np.ndarray
    ) -> Tuple[Optional[int], Optional[complex]]:
        """稳健的最佳匹配查找"""
        # 检查输入有效性
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return None, None

        signal_real = np.concatenate([signal.real, signal.imag])
        dictionary_real = np.concatenate([dictionary.real, dictionary.imag], axis=0)

        # 检查字典有效性
        if np.any(np.isnan(dictionary_real)) or np.any(np.isinf(dictionary_real)):
            return None, None

        try:
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=1, fit_intercept=False)
            omp.fit(dictionary_real, signal_real)

            nonzero_indices = np.nonzero(omp.coef_)[0]
            if len(nonzero_indices) == 0:
                return None, None

            best_idx = nonzero_indices[0]
            selected_atom = dictionary[:, best_idx]

            # 计算复数系数
            numerator = np.vdot(selected_atom, signal)
            denominator = np.vdot(selected_atom, selected_atom)

            if abs(denominator) < 1e-12:
                return None, None

            complex_coef = numerator / denominator

            return best_idx, complex_coef

        except Exception as e:
            print(f"   ⚠️ OMP匹配异常: {str(e)}")
            return None, None

    def _refine_point_scatterer_v2(
        self, initial_params: Dict, target_signal: np.ndarray, initial_coef: complex
    ) -> Dict:
        """
        真正的参数精化函数 - 修复优化逻辑
        关键：对当前残差进行优化，而非原始信号
        """
        alpha_fixed = initial_params["alpha"]

        # 优化目标函数
        def objective(params):
            x, y, amp, phase = params
            # 生成原子
            atom = self._generate_robust_asc_atom(x=x, y=y, alpha=alpha_fixed)
            atom_flat = atom.flatten()
            atom_energy = np.linalg.norm(atom_flat)

            if atom_energy < 1e-12:
                return 1e6  # 惩罚无效原子

            atom_normalized = atom_flat / atom_energy
            # 重构
            reconstruction = amp * np.exp(1j * phase) * atom_normalized
            # 关键：计算与当前残差(target_signal)的误差
            return np.linalg.norm(target_signal - reconstruction)

        # 初始值和边界
        x0 = [initial_params["x"], initial_params["y"], np.abs(initial_coef), np.angle(initial_coef)]
        bounds = [(-1, 1), (-1, 1), (0, 10 * np.abs(initial_coef)), (-np.pi, np.pi)]

        # 执行优化
        try:
            from scipy.optimize import minimize

            result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 50})

            refined_params = initial_params.copy()
            if result.success and result.fun < np.linalg.norm(target_signal):
                refined_params.update(
                    {
                        "x": result.x[0],
                        "y": result.x[1],
                        "estimated_amplitude": result.x[2],
                        "estimated_phase": result.x[3],
                        "optimization_success": True,
                        "optimization_error": result.fun,
                    }
                )
            else:  # 优化失败，使用粗匹配结果
                refined_params.update(
                    {
                        "estimated_amplitude": np.abs(initial_coef),
                        "estimated_phase": np.angle(initial_coef),
                        "optimization_success": False,
                    }
                )

        except Exception as e:
            print(f"   ⚠️ 参数优化异常: {str(e)}")
            refined_params = initial_params.copy()
            refined_params.update(
                {
                    "estimated_amplitude": np.abs(initial_coef),
                    "estimated_phase": np.angle(initial_coef),
                    "optimization_success": False,
                }
            )

        return refined_params

    def _calculate_scatterer_contribution(self, scatterer_params: Dict) -> np.ndarray:
        """计算散射中心对信号的贡献"""
        fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1])

        atom = self._generate_robust_asc_atom(
            scatterer_params["x"],
            scatterer_params["y"],
            scatterer_params["alpha"],
            scatterer_params.get("length", 0.0),
            scatterer_params.get("phi_bar", 0.0),
            fx_range,
            fy_range,
        )

        atom_flat = atom.flatten()
        atom_energy = np.linalg.norm(atom_flat)

        if atom_energy > 1e-12:
            atom_normalized = atom_flat / atom_energy
            contribution = (
                scatterer_params["estimated_amplitude"]
                * np.exp(1j * scatterer_params["estimated_phase"])
                * atom_normalized
            )
            return contribution
        else:
            return np.zeros_like(atom_flat)

    def extract_asc_scatterers_v2(self, complex_image: np.ndarray) -> List[Dict]:
        """
        完整的v3版本ASC提取流程 - 带真正的优化
        实现正确的"匹配-优化-减去"循环
        """
        print(f"🚀 开始v3版本ASC提取流程 (带优化)")
        print("=" * 60)

        # 数据预处理
        signal = self.preprocess_data_robust(complex_image)

        # 构建字典
        dictionary, param_grid = self.build_compact_dictionary()

        # 初始化
        residual_signal = signal.copy()
        extracted_scatterers = []

        initial_energy = np.linalg.norm(residual_signal)
        energy_threshold = initial_energy * self.adaptive_threshold

        print(f"   初始能量: {initial_energy:.6f}")
        print(f"   停止阈值: {energy_threshold:.6f}")

        for iteration in range(self.max_iterations):
            current_energy = np.linalg.norm(residual_signal)

            # 停止条件检查
            if current_energy < energy_threshold:
                print(f"   💡 达到能量阈值，停止迭代")
                break

            if len(extracted_scatterers) >= self.max_scatterers:
                print(f"   💡 达到最大散射中心数，停止迭代")
                break

            # --- 1. 匹配 (Matching) ---
            best_idx, initial_coef = self._find_best_match_robust(residual_signal, dictionary)
            if best_idx is None:
                print(f"   💡 未找到有效匹配，停止迭代")
                break

            initial_params = param_grid[best_idx].copy()

            # --- 2. 优化 (Optimization) ---
            # 关键：对当前残差进行优化
            refined_params = self._refine_point_scatterer_v2(initial_params, residual_signal, initial_coef)

            # --- 3. 减去 (Subtraction) ---
            contribution = self._calculate_scatterer_contribution(refined_params)

            # 检查贡献有效性
            contribution_energy = np.linalg.norm(contribution)
            if contribution_energy < current_energy * 0.001:
                print(f"   💡 散射中心贡献过小({contribution_energy:.2e})，停止迭代")
                break

            new_residual_signal = residual_signal - contribution
            new_energy = np.linalg.norm(new_residual_signal)

            # 关键：检查能量是否有效减少
            if new_energy >= current_energy:
                print(f"   ⚠️ 能量增加({current_energy:.6f} → {new_energy:.6f})，优化失败，停止迭代")
                break

            # 更新残差和结果
            residual_signal = new_residual_signal
            refined_params["residual_energy"] = new_energy
            extracted_scatterers.append(refined_params)

            # 进度报告
            reduction = (current_energy - new_energy) / current_energy
            opt_status = "✅" if refined_params.get("optimization_success", False) else "⚠️"
            print(
                f"   迭代 {iteration+1}: {opt_status} 提取 {refined_params['scattering_type']}, "
                f"幅度 {refined_params['estimated_amplitude']:.3f}, "
                f"能量减少 {reduction:.2%}"
            )

        # 最终结果分析
        final_energy = np.linalg.norm(residual_signal)
        total_reduction = (initial_energy - final_energy) / initial_energy

        print(f"\n✅ v3版本提取完成")
        print(f"   散射中心数: {len(extracted_scatterers)}")
        print(f"   总能量减少: {total_reduction:.1%}")

        if extracted_scatterers:
            print(f"\n📊 提取结果分析:")
            alpha_dist = {}
            opt_success_count = 0
            for s in extracted_scatterers:
                stype = s["scattering_type"]
                alpha_dist[stype] = alpha_dist.get(stype, 0) + 1
                if s.get("optimization_success", False):
                    opt_success_count += 1

            print(f"   散射类型分布: {alpha_dist}")
            print(
                f"   优化成功率: {opt_success_count}/{len(extracted_scatterers)} ({opt_success_count/len(extracted_scatterers)*100:.1f}%)"
            )

        return extracted_scatterers


def visualize_extraction_results(complex_image, scatterers, save_path=None):
    """
    可视化散射中心提取结果
    将散射中心叠加在原始SAR图像上显示
    """
    if not scatterers:
        print("⚠️ No scatterers extracted, cannot visualize.")
        return

    magnitude = np.abs(complex_image)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # 1. 显示原始SAR图像作为背景
    ax.imshow(magnitude, cmap="gray", origin="lower", extent=(-1, 1, -1, 1), alpha=0.8)

    # 2. 绘制提取的散射中心
    alpha_colors = {
        -1.0: "blue",  # 尖顶绕射
        -0.5: "cyan",  # 边缘绕射
        0.0: "green",  # 标准散射
        0.5: "orange",  # 表面散射
        1.0: "red",  # 镜面反射
    }

    alpha_names = {
        -1.0: "Dihedral (α=-1.0)",
        -0.5: "Edge Diffraction (α=-0.5)",
        0.0: "Isotropic (α=0.0)",
        0.5: "Surface (α=0.5)",
        1.0: "Specular (α=1.0)",
    }

    # 统计散射中心
    plotted_types = set()

    for i, sc in enumerate(scatterers):
        x, y = sc["x"], sc["y"]
        alpha = sc["alpha"]
        amplitude = sc["estimated_amplitude"]
        opt_success = sc.get("optimization_success", False)

        # 颜色代表散射类型(alpha)
        color = alpha_colors.get(alpha, "purple")
        # 大小代表幅度
        size = 100 + amplitude * 1000  # 调整系数以获得好的视觉效果

        # 边框表示优化成功与否
        edge_color = "white" if opt_success else "black"
        edge_width = 2 if opt_success else 1

        scatter = ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors=edge_color, linewidth=edge_width)

        # 标注散射中心编号
        ax.annotate(
            f"{i+1}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8, color="white", weight="bold"
        )

        plotted_types.add(alpha)

    # 设置图像属性
    ax.set_title(f"ASC Scattering Centers - {len(scatterers)} Extracted", fontsize=14, weight="bold")
    ax.set_xlabel("X Position (Normalized)", fontsize=12)
    ax.set_ylabel("Y Position (Normalized)", fontsize=12)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, linestyle="--", alpha=0.3)

    # 创建图例
    legend_elements = []
    for alpha in sorted(plotted_types):
        color = alpha_colors.get(alpha, "purple")
        name = alpha_names.get(alpha, f"α={alpha}")
        legend_elements.append(
            plt.scatter([], [], c=color, s=100, alpha=0.7, edgecolors="white", linewidth=2, label=name)
        )

    # 添加优化成功说明
    legend_elements.append(
        plt.scatter([], [], c="gray", s=100, alpha=0.7, edgecolors="white", linewidth=2, label="优化成功")
    )
    legend_elements.append(
        plt.scatter([], [], c="gray", s=100, alpha=0.7, edgecolors="black", linewidth=1, label="粗匹配")
    )

    ax.legend(handles=legend_elements, title="散射类型 & 优化状态", loc="upper left", bbox_to_anchor=(1.02, 1))

    # 添加统计信息
    stats_text = f"Statistics:\n"
    stats_text += f"Total Scatterers: {len(scatterers)}\n"
    opt_count = sum(1 for s in scatterers if s.get("optimization_success", False))
    stats_text += f"Optimized: {opt_count}/{len(scatterers)}\n"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"🖼️ 可视化结果已保存到: {save_path}")

    plt.show()


def main():
    """v3版本演示 - 带真正的优化和可视化"""
    print("🔧 修复版ASC提取系统v3")
    print("解决物理尺度、参数精化和收敛性问题")

    asc_v3 = ASCExtractionFixedV2(
        extraction_mode="point_only", adaptive_threshold=0.05, max_iterations=15, max_scatterers=10
    )

    print("\n🚀 完整使用流程:")
    print("1. magnitude, complex_image = asc_v3.load_mstar_data_robust('data.raw')")
    print("2. scatterers = asc_v3.extract_asc_scatterers_v2(complex_image)")
    print("3. visualize_extraction_results(complex_image, scatterers, 'result.png')")

    return asc_v3


if __name__ == "__main__":
    asc_system = main()
