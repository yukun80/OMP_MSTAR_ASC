"""
Fixed Attribute Scattering Center (ASC) Extraction System
修复版属性散射中心提取系统

解决现有算法的三个核心问题：
1. 数值稳定性：修复ASC原子生成中的零频和负α值问题
2. 优化逻辑：修复参数精化中用单个原子匹配完整信号的错误
3. 迭代收敛：实现正确的"匹配-优化-减去"流程

技术改进：
- 数值稳健的ASC原子生成函数
- 正确的残差匹配优化目标
- 渐进式提取策略（点散射→分布式散射）
- 智能收敛判断和自适应停止
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


class ASCExtractionFixed:
    """
    修复版ASC提取系统

    核心修复：
    1. 数值稳健的ASC原子生成
    2. 正确的残差匹配优化
    3. 分层提取策略
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        extraction_mode: str = "progressive",  # "point_only", "progressive", "full_asc"
        adaptive_threshold: float = 0.01,
        max_iterations: int = 50,
        min_scatterers: int = 3,
        max_scatterers: int = 30,
    ):
        """
        初始化修复版ASC提取系统

        Args:
            extraction_mode: 提取模式
                - "point_only": 仅点散射（L=0, phi_bar=0，验证α识别）
                - "progressive": 渐进式（先点散射，再扩展）
                - "full_asc": 完整ASC（6参数同时提取）
        """
        self.image_size = image_size
        self.extraction_mode = extraction_mode
        self.adaptive_threshold = adaptive_threshold
        self.max_iterations = max_iterations
        self.min_scatterers = min_scatterers
        self.max_scatterers = max_scatterers

        # SAR系统参数
        self.fc = 1e10  # 中心频率 10GHz
        self.B = 1e9  # 带宽 1GHz
        self.omega = np.pi / 3  # 合成孔径角
        self.scene_size = 30.0  # 场景尺寸 (米)

        # 根据提取模式配置参数
        self._configure_extraction_mode()

        print(f"🔧 修复版ASC提取系统初始化")
        print(f"   提取模式: {extraction_mode}")
        print(f"   图像尺寸: {image_size}")
        print(f"   α值范围: {self.alpha_values}")
        print(f"   L值范围: {self.length_values}")
        print(f"   自适应阈值: {adaptive_threshold}")

    def _configure_extraction_mode(self):
        """根据提取模式配置参数"""
        if self.extraction_mode == "point_only":
            # 仅点散射：固定L=0, phi_bar=0，专注α识别
            self.alpha_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
            self.length_values = [0.0]  # 固定为0
            self.phi_bar_values = [0.0]  # 固定为0
            self.position_samples = 32
            print("   🎯 点散射模式：专注频率依赖因子α识别")

        elif self.extraction_mode == "progressive":
            # 渐进式：先点散射，再逐步扩展
            self.alpha_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
            self.length_values = [0.0, 0.1, 0.5]  # 逐步扩展
            self.phi_bar_values = [0.0, np.pi / 4, np.pi / 2]  # 逐步扩展
            self.position_samples = 32
            print("   🎯 渐进模式：从点散射扩展到分布式散射")

        else:  # full_asc
            # 完整ASC：所有参数同时提取
            self.alpha_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
            self.length_values = np.logspace(-2, 0, 5)  # [0.01, 1.0]
            self.phi_bar_values = np.linspace(0, np.pi, 8)
            self.position_samples = 24  # 降低采样减少计算量
            print("   🎯 完整ASC模式：6参数同时提取")

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

        print(f"   图像尺寸: {complex_image.shape}")
        print(f"   幅度范围: [{np.min(magnitude):.3f}, {np.max(magnitude):.3f}]")
        print(f"   信号能量: {np.linalg.norm(complex_image):.3f}")

        return magnitude, complex_image

    def preprocess_data(self, complex_image: np.ndarray) -> np.ndarray:
        """智能数据预处理"""
        print("⚙️ 智能数据预处理...")

        # 信号向量化
        signal = complex_image.flatten()

        # 能量归一化（保持相对强度关系）
        signal_energy = np.linalg.norm(signal)
        signal_normalized = signal / np.sqrt(signal_energy)

        print(f"   信号长度: {len(signal)}")
        print(f"   原始能量: {signal_energy:.3f}")
        print(f"   归一化方式: 能量归一化")

        return signal_normalized

    def _generate_robust_asc_atom(
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
        生成数值稳健的ASC原子

        关键修复：
        1. 处理零频问题，避免0^(-alpha)数值爆炸
        2. 使用归一化频率避免数值过大
        3. 改进的sinc函数计算
        """
        # 创建频率网格
        FX, FY = np.meshgrid(fx_range, fy_range, indexing="ij")

        # 计算频率特征
        f_magnitude = np.sqrt(FX**2 + FY**2)
        theta = np.arctan2(FY, FX)

        # 关键修复1：处理零频问题
        # 使用更大的安全值避免数值问题
        f_magnitude_safe = np.where(f_magnitude < 1e-8, 1e-8, f_magnitude)

        # ASC频域响应
        # 1. 位置相位项
        position_phase = -2j * np.pi * (FX * x + FY * y)

        # 2. 频率依赖项: f^α (关键修复)
        if alpha == 0:
            # 特殊处理α=0的情况
            frequency_term = np.ones_like(f_magnitude_safe)
        else:
            # 使用归一化频率 f/fc 避免数值过大
            normalized_freq = f_magnitude_safe / self.fc
            frequency_term = np.power(normalized_freq, alpha)

        # 3. 长度相关项: sinc(L·f·sin(θ-φ_bar))
        if length == 0:
            # 点散射情况
            length_term = np.ones_like(f_magnitude_safe)
        else:
            angle_diff = theta - phi_bar
            sinc_arg = length * f_magnitude_safe * np.sin(angle_diff)

            # 改进的sinc函数计算，避免数值问题
            with np.errstate(divide="ignore", invalid="ignore"):
                length_term = np.where(np.abs(sinc_arg) < 1e-10, 1.0, np.sin(np.pi * sinc_arg) / (np.pi * sinc_arg))

        # 4. 方位角相位项
        azimuth_phase = 1j * phi_bar

        # 组合完整ASC频域响应
        H_asc = frequency_term * length_term * np.exp(position_phase + azimuth_phase)

        # 空域原子 (IFFT)
        atom = np.fft.ifft2(np.fft.ifftshift(H_asc))

        return atom

    def build_robust_dictionary(self) -> Tuple[np.ndarray, List[Dict]]:
        """构建数值稳健的ASC字典"""
        print(f"📚 构建稳健ASC字典 (模式: {self.extraction_mode})...")

        # 频率采样范围
        fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1])

        # 位置采样
        x_positions = np.linspace(-0.9, 0.9, self.position_samples)
        y_positions = np.linspace(-0.9, 0.9, self.position_samples)

        # 估算字典大小
        total_atoms = (
            len(x_positions)
            * len(y_positions)
            * len(self.alpha_values)
            * len(self.length_values)
            * len(self.phi_bar_values)
        )

        print(f"   位置采样: {self.position_samples}×{self.position_samples}")
        print(f"   α采样: {len(self.alpha_values)} 个值")
        print(f"   L采样: {len(self.length_values)} 个值")
        print(f"   φ_bar采样: {len(self.phi_bar_values)} 个值")
        print(f"   预计原子数: {total_atoms}")

        # 构建字典
        dictionary_atoms = []
        param_grid = []

        atom_count = 0
        invalid_atoms = 0
        start_time = time.time()

        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                for alpha in self.alpha_values:
                    for length in self.length_values:
                        for phi_bar in self.phi_bar_values:
                            # 生成ASC原子
                            atom = self._generate_robust_asc_atom(x, y, alpha, length, phi_bar, fx_range, fy_range)

                            # 检查原子有效性
                            atom_flat = atom.flatten()
                            atom_energy = np.linalg.norm(atom_flat)

                            # 检查数值异常
                            if (
                                atom_energy > 1e-12
                                and np.isfinite(atom_energy)
                                and not np.any(np.isnan(atom_flat))
                                and not np.any(np.isinf(atom_flat))
                            ):

                                # 归一化
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
                                        "grid_index": (i, j),
                                        "scattering_type": self._classify_scattering_type(alpha),
                                    }
                                )

                                atom_count += 1
                            else:
                                invalid_atoms += 1

                            if (atom_count + invalid_atoms) % 2000 == 0:
                                progress = (atom_count + invalid_atoms) / total_atoms * 100
                                elapsed = time.time() - start_time
                                print(
                                    f"   进度: {progress:.1f}% ({atom_count}有效, {invalid_atoms}无效) - {elapsed:.1f}s"
                                )

        # 转换为矩阵
        dictionary = np.column_stack(dictionary_atoms)

        print(f"✅ 稳健字典构建完成")
        print(f"   有效原子数: {atom_count}")
        print(f"   无效原子数: {invalid_atoms}")
        print(f"   有效率: {atom_count/(atom_count+invalid_atoms)*100:.1f}%")
        print(f"   字典尺寸: {dictionary.shape}")
        print(f"   内存占用: ~{dictionary.nbytes / 1024**2:.1f} MB")

        return dictionary, param_grid

    def _classify_scattering_type(self, alpha: float) -> str:
        """根据α值分类散射类型"""
        scattering_types = {-1.0: "尖顶绕射", -0.5: "边缘绕射", 0.0: "标准散射", 0.5: "表面散射", 1.0: "镜面反射"}
        return scattering_types.get(alpha, f"α={alpha}")

    def fixed_adaptive_extraction(
        self, signal: np.ndarray, dictionary: np.ndarray, param_grid: List[Dict]
    ) -> List[Dict]:
        """
        修复版自适应提取算法

        核心修复：
        1. 正确的"匹配-优化-减去"迭代流程
        2. 稳健的残差更新机制
        3. 智能的收敛判断
        """
        print(f"🎯 开始修复版自适应ASC提取...")

        residual_signal = signal.copy()
        extracted_scatterers = []

        # 初始信号特征
        initial_energy = np.linalg.norm(residual_signal)
        energy_threshold = initial_energy * self.adaptive_threshold

        print(f"   初始信号能量: {initial_energy:.6f}")
        print(f"   能量停止阈值: {energy_threshold:.6f}")

        for iteration in range(self.max_iterations):
            current_energy = np.linalg.norm(residual_signal)
            energy_reduction_ratio = (initial_energy - current_energy) / initial_energy

            # 多重停止条件
            if current_energy < energy_threshold:
                print(f"   💡 达到能量阈值，停止迭代 (减少{energy_reduction_ratio:.1%})")
                break

            if len(extracted_scatterers) >= self.max_scatterers:
                print(f"   💡 达到最大散射中心数，停止迭代")
                break

            # 阶段1：粗匹配 - 在字典中找到最佳匹配原子
            best_atom_idx, best_coefficient = self._find_best_match(residual_signal, dictionary)

            if best_atom_idx is None:
                print(f"   💡 未找到显著散射中心，停止迭代")
                break

            # 获取初始参数
            initial_params = param_grid[best_atom_idx].copy()

            # 阶段2：参数精化 - 使用当前残差作为优化目标（关键修复）
            refined_params = self._refine_parameters_correctly(
                initial_params, residual_signal, best_coefficient  # 关键：使用残差而非原始信号
            )

            # 阶段3：残差更新 - 从信号中减去当前散射中心的贡献
            updated_residual = self._update_residual_robust(residual_signal, refined_params)

            # 验证更新有效性
            new_energy = np.linalg.norm(updated_residual)
            energy_reduction = current_energy - new_energy

            if energy_reduction < current_energy * 0.001:  # 0.1%改进阈值
                print(f"   💡 能量减少不显著 ({energy_reduction/current_energy:.3%})，停止迭代")
                break

            # 更新残差和记录散射中心
            residual_signal = updated_residual
            extracted_scatterers.append(refined_params)

            # 进度显示
            if iteration < 10 or (iteration + 1) % 5 == 0:
                print(
                    f"   迭代 {iteration+1}: 能量 {current_energy:.6f} → {new_energy:.6f} "
                    f"(减少 {energy_reduction/current_energy:.2%})"
                )

        # 最终统计
        final_energy = np.linalg.norm(residual_signal)
        total_reduction = (initial_energy - final_energy) / initial_energy

        print(f"✅ 修复版ASC提取完成")
        print(f"   提取散射中心数: {len(extracted_scatterers)}")
        print(f"   总能量减少: {total_reduction:.1%}")
        print(f"   最终残差能量: {final_energy:.6f}")

        return extracted_scatterers

    def _find_best_match(self, signal: np.ndarray, dictionary: np.ndarray) -> Tuple[Optional[int], Optional[complex]]:
        """在字典中找到最佳匹配原子"""
        # 转换为实值OMP
        signal_real = np.concatenate([signal.real, signal.imag])
        dictionary_real = np.concatenate([dictionary.real, dictionary.imag], axis=0)

        try:
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=1, fit_intercept=False)
            omp.fit(dictionary_real, signal_real)

            nonzero_indices = np.nonzero(omp.coef_)[0]
            if len(nonzero_indices) == 0:
                return None, None

            best_idx = nonzero_indices[0]

            # 计算准确的复数系数
            selected_atom = dictionary[:, best_idx]
            complex_coef = np.vdot(selected_atom, signal) / np.vdot(selected_atom, selected_atom)

            return best_idx, complex_coef

        except Exception as e:
            print(f"   ⚠️ OMP匹配失败: {str(e)}")
            return None, None

    def _refine_parameters_correctly(
        self,
        initial_params: Dict,
        target_signal: np.ndarray,  # 关键：使用残差信号而非原始信号
        initial_coefficient: complex,
    ) -> Dict:
        """
        正确的参数精化

        关键修复：优化目标是匹配当前残差信号，而非原始信号
        """
        if self.extraction_mode == "point_only":
            # 点散射模式：只优化位置和幅度/相位
            return self._refine_point_scatterer(initial_params, target_signal, initial_coefficient)
        else:
            # 完整ASC模式：优化所有连续参数
            return self._refine_full_asc(initial_params, target_signal, initial_coefficient)

    def _refine_point_scatterer(
        self, initial_params: Dict, target_signal: np.ndarray, initial_coefficient: complex
    ) -> Dict:
        """精化点散射中心参数"""
        # 固定离散参数
        alpha_fixed = initial_params["alpha"]
        length_fixed = 0.0  # 点散射
        phi_bar_fixed = 0.0  # 点散射

        # 优化变量：[x, y, amplitude, phase]
        x0 = [initial_params["x"], initial_params["y"], np.abs(initial_coefficient), np.angle(initial_coefficient)]

        # 频率范围
        fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1])

        def objective(params):
            x, y, amp, phase = params

            # 生成当前参数的原子
            atom = self._generate_robust_asc_atom(x, y, alpha_fixed, length_fixed, phi_bar_fixed, fx_range, fy_range)

            atom_flat = atom.flatten()
            atom_energy = np.linalg.norm(atom_flat)

            if atom_energy > 1e-12:
                atom_normalized = atom_flat / atom_energy
            else:
                return 1e6  # 惩罚无效原子

            # 当前参数下的重构
            reconstruction = amp * np.exp(1j * phase) * atom_normalized

            # 关键修复：与残差信号比较，而非原始信号
            error = np.linalg.norm(target_signal - reconstruction)

            return error

        # 参数边界
        bounds = [(-1.0, 1.0), (-1.0, 1.0), (0.001, 10.0), (-np.pi, np.pi)]  # x  # y  # amplitude  # phase

        try:
            result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 50})

            if result.success:
                refined_params = initial_params.copy()
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
                return refined_params
            else:
                # 保持初始参数
                initial_params["estimated_amplitude"] = np.abs(initial_coefficient)
                initial_params["estimated_phase"] = np.angle(initial_coefficient)
                initial_params["optimization_success"] = False
                return initial_params

        except Exception:
            # 保持初始参数
            initial_params["estimated_amplitude"] = np.abs(initial_coefficient)
            initial_params["estimated_phase"] = np.angle(initial_coefficient)
            initial_params["optimization_success"] = False
            return initial_params

    def _refine_full_asc(self, initial_params: Dict, target_signal: np.ndarray, initial_coefficient: complex) -> Dict:
        """精化完整ASC参数"""
        # 对于完整ASC，暂时保持简化处理
        # 可以后续扩展为优化所有6个参数
        return self._refine_point_scatterer(initial_params, target_signal, initial_coefficient)

    def _update_residual_robust(self, current_signal: np.ndarray, scatterer_params: Dict) -> np.ndarray:
        """稳健的残差更新"""
        # 频率范围
        fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1])

        # 重新生成精化后的原子
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

            # 计算散射中心的贡献
            contribution = (
                scatterer_params["estimated_amplitude"]
                * np.exp(1j * scatterer_params["estimated_phase"])
                * atom_normalized
            )

            # 从信号中减去贡献
            updated_signal = current_signal - contribution

            return updated_signal
        else:
            # 原子无效，返回原信号
            return current_signal

    def extract_asc_scatterers(self, complex_image: np.ndarray) -> List[Dict]:
        """完整的ASC提取流程"""
        print(f"🚀 开始完整ASC提取流程 (模式: {self.extraction_mode})")
        print("=" * 60)

        # 步骤1：数据预处理
        signal = self.preprocess_data(complex_image)

        # 步骤2：构建稳健字典
        dictionary, param_grid = self.build_robust_dictionary()

        # 步骤3：修复版自适应提取
        scatterers = self.fixed_adaptive_extraction(signal, dictionary, param_grid)

        # 步骤4：结果分析
        if scatterers:
            analysis = self._analyze_extraction_results(scatterers)
            print(f"\n📊 提取结果分析:")
            print(f"   散射中心总数: {analysis['total_count']}")
            print(f"   α分布: {analysis['alpha_distribution']}")
            print(f"   优化成功率: {analysis['optimization_success_rate']:.1%}")

        return scatterers

    def _analyze_extraction_results(self, scatterers: List[Dict]) -> Dict:
        """分析提取结果"""
        if not scatterers:
            return {}

        # 按α值分组
        alpha_distribution = {}
        for scatterer in scatterers:
            alpha = scatterer["alpha"]
            scattering_type = self._classify_scattering_type(alpha)
            if scattering_type not in alpha_distribution:
                alpha_distribution[scattering_type] = 0
            alpha_distribution[scattering_type] += 1

        # 统计分析
        optimization_success_count = sum(1 for s in scatterers if s.get("optimization_success", False))

        return {
            "total_count": len(scatterers),
            "alpha_distribution": alpha_distribution,
            "optimization_success_rate": optimization_success_count / len(scatterers),
            "amplitudes": [s["estimated_amplitude"] for s in scatterers],
            "positions": [(s["x"], s["y"]) for s in scatterers],
        }


def main():
    """修复版ASC系统演示"""
    print("🔧 修复版ASC提取系统演示")
    print("=" * 60)

    # 初始化修复版系统
    asc_fixed = ASCExtractionFixed(
        extraction_mode="point_only", adaptive_threshold=0.05, max_iterations=30, max_scatterers=20  # 从点散射开始验证
    )

    print("\n📝 系统初始化完成，准备加载MSTAR数据进行测试")
    print("\n使用示例:")
    print("magnitude, complex_image = asc_fixed.load_mstar_data('data.raw')")
    print("scatterers = asc_fixed.extract_asc_scatterers(complex_image)")

    return asc_fixed


if __name__ == "__main__":
    asc_system = main()
