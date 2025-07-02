"""
参数优化模块

实现散射中心参数的非线性优化，对应MATLAB的extraction_local_*, extraction_dis_*等优化函数
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from scipy.optimize import minimize, differential_evolution
import warnings
from physical_model import SARPhysicalModel


class ParameterOptimizer:
    """散射中心参数优化器"""

    def __init__(self, physical_model: SARPhysicalModel):
        """初始化参数优化器"""
        self.model = physical_model
        self.global_variables = {}  # 存储全局变量 (对应MATLAB的global变量)

        # 坐标转换参数 (对应MATLAB中的坐标转换)
        # 基于SAR成像几何，像素间距约为0.046875米 (6米场景/128像素)
        self.coord_scale = 6.0 / 128  # 像素到米的转换因子，约0.0469米/像素

    def set_global_context(self, roi_image: np.ndarray, roi_complex: np.ndarray):
        """
        设置全局优化上下文 - 对应MATLAB的global变量

        Args:
            roi_image: ROI区域图像
            roi_complex: ROI区域复数图像
        """
        self.global_variables = {"image_interest": roi_image, "complex_temp": roi_complex, "A": 1.0}  # 初始幅度值

    def optimize_local_scatterer(
        self, initial_coords: np.ndarray, roi_image: np.ndarray, roi_complex: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        局部散射中心参数优化 - 对应MATLAB的三种alpha模型优化

        Args:
            initial_coords: 初始坐标 [row, col]
            roi_image: ROI区域图像
            roi_complex: ROI区域复数图像

        Returns:
            optimal_params: 最优参数 [x, y, alpha, r, theta0, L, A]
            optimal_fval: 最优目标函数值
        """
        # 设置全局上下文
        self.set_global_context(roi_image, roi_complex)

        # 坐标转换 (对应MATLAB的坐标转换逻辑)
        x_initial = (initial_coords[1] - 64) * self.coord_scale  # col -> x (距离)
        y_initial = (64 - initial_coords[0]) * self.coord_scale  # row -> y (方位)

        # 计算散射强度 A
        A = self.calculate_scatterer_amplitude(x_initial, y_initial, 0, 0, 0, 0, roi_image)
        self.global_variables["A"] = A

        # 测试三种alpha值模型 (对应MATLAB的并行优化)
        alpha_values = [0.0, 0.5, 1.0]
        results = []

        for alpha in alpha_values:
            try:
                # 初始参数: [x, y, r] (alpha固定)
                initial_params = np.array([x_initial, y_initial, 0.0])

                # 参数边界
                bounds = [
                    (x_initial - 0.1, x_initial + 0.1),  # x边界
                    (y_initial - 0.1, y_initial + 0.1),  # y边界
                    (0.0, 1.0),  # r边界
                ]

                # 优化
                result = minimize(
                    fun=lambda params: self._objective_function_local(params, alpha),
                    x0=initial_params,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"ftol": 1e-9, "maxiter": 1000},
                )

                results.append((result.x, result.fun, alpha))

            except Exception as e:
                warnings.warn(f"Alpha={alpha}优化失败: {e}")
                continue

        # 选择最优结果
        if not results:
            # 如果所有优化都失败，返回初始估计
            return np.array([x_initial, y_initial, 0.0, 0.0, 0.0, 0.0, A]), float("inf")

        best_params, best_fval, best_alpha = min(results, key=lambda x: x[1])

        # 构造完整参数向量 [x, y, alpha, r, theta0, L, A]
        optimal_params = np.array(
            [
                best_params[0],  # x
                best_params[1],  # y
                best_alpha,  # alpha
                best_params[2],  # r
                0.0,  # theta0 (局部散射中心为0)
                0.0,  # L (局部散射中心为0)
                A,  # A
            ]
        )

        return optimal_params, best_fval

    def optimize_distributed_scatterer(
        self, initial_coords: np.ndarray, roi_image: np.ndarray, roi_complex: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        分布式散射中心参数优化 - 对应MATLAB的extraction_dis_a0优化

        Args:
            initial_coords: 初始坐标 [row, col]
            roi_image: ROI区域图像
            roi_complex: ROI区域复数图像

        Returns:
            optimal_params: 最优参数 [x, y, alpha, r, theta0, L, A]
            optimal_fval: 最优目标函数值
        """
        # 设置全局上下文
        self.set_global_context(roi_image, roi_complex)

        # 坐标转换
        x_initial = (initial_coords[1] - 64) * self.coord_scale
        y_initial = (64 - initial_coords[0]) * self.coord_scale

        # 频域分析估计初始参数 (对应MATLAB的频域分析)
        theta0_initial, L_initial = self._estimate_distributed_parameters(roi_complex)

        # 计算散射强度
        A = self.calculate_scatterer_amplitude(x_initial, y_initial, 0, 0, theta0_initial, L_initial, roi_image)
        self.global_variables["A"] = A

        # 初始参数: [x, y, theta0, L]
        initial_params = np.array([x_initial, y_initial, theta0_initial, L_initial])

        # 参数边界
        omega_rad = self.model.omega_rad
        theta0_min = max(theta0_initial - abs(0.5 * theta0_initial), -omega_rad / 2 * 180 / np.pi)
        theta0_max = min(theta0_initial + abs(0.5 * theta0_initial), omega_rad / 2 * 180 / np.pi)

        bounds = [
            (x_initial - 0.1, x_initial + 0.1),  # x边界
            (y_initial - 0.1, y_initial + 0.1),  # y边界
            (theta0_min, theta0_max),  # theta0边界
            (L_initial - 0.3 * L_initial, L_initial + 0.3 * L_initial),  # L边界
        ]

        try:
            # 主优化
            result = minimize(
                fun=self._objective_function_distributed,
                x0=initial_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-9, "maxiter": 1000},
            )

            optimal_params_partial = result.x
            optimal_fval = result.fun

            # 如果优化结果不理想，尝试混沌优化
            if optimal_fval > 1.0:
                chaos_params = self.chaos_optimization(initial_params, bounds)
                chaos_fval = self._objective_function_distributed(chaos_params)

                if chaos_fval < optimal_fval:
                    optimal_params_partial = chaos_params
                    optimal_fval = chaos_fval

        except Exception as e:
            warnings.warn(f"分布式散射中心优化失败: {e}")
            optimal_params_partial = initial_params
            optimal_fval = float("inf")

        # 重新计算最终的散射强度
        final_A = self.calculate_scatterer_amplitude(
            optimal_params_partial[0],
            optimal_params_partial[1],
            0,
            0,
            optimal_params_partial[2],
            optimal_params_partial[3],
            roi_image,
        )

        # 估计alpha参数 (对应MATLAB的finda函数)
        alpha = self._estimate_alpha_parameter(
            roi_image,
            final_A,
            optimal_params_partial[0],
            optimal_params_partial[1],
            optimal_params_partial[2],
            optimal_params_partial[3],
        )

        # 构造完整参数向量 [x, y, alpha, r, theta0, L, A]
        optimal_params = np.array(
            [
                optimal_params_partial[0],  # x
                optimal_params_partial[1],  # y
                alpha,  # alpha
                0.0,  # r (分布式散射中心为0)
                optimal_params_partial[2],  # theta0
                optimal_params_partial[3],  # L
                final_A,  # A
            ]
        )

        return optimal_params, optimal_fval

    def _objective_function_local(self, params: np.ndarray, alpha: float) -> float:
        """
        局部散射中心目标函数 - 对应MATLAB的extraction_local_a0等函数

        Args:
            params: 参数向量 [x, y, r]
            alpha: 频率依赖指数

        Returns:
            目标函数值
        """
        x, y, r = params
        theta0, L = 0.0, 0.0  # 局部散射中心
        A = self.global_variables.get("A", 1.0)

        # 生成频域响应
        freq_response, _ = self.model.spotlight_imaging(x, y, alpha, r, theta0, L, A)

        # 逆变换到图像域
        image_complex = self.model.inverse_transform_to_image(freq_response)

        # 应用ROI掩模
        roi_mask = self.global_variables["image_interest"]
        image_complex = image_complex * roi_mask

        # 计算与实际复数图像的差异 (对应MATLAB的目标函数)
        target_complex = self.global_variables["complex_temp"]
        diff = image_complex - target_complex

        # 计算目标函数值
        objective = np.sum(np.abs(diff) ** 2)

        return objective

    def _objective_function_distributed(self, params: np.ndarray) -> float:
        """
        分布式散射中心目标函数 - 对应MATLAB的extraction_dis_a0函数

        Args:
            params: 参数向量 [x, y, theta0, L]

        Returns:
            目标函数值
        """
        x, y, theta0, L = params
        alpha, r = 0.0, 0.0  # 分布式散射中心
        A = self.global_variables.get("A", 1.0)

        # 生成频域响应
        freq_response, _ = self.model.spotlight_imaging(x, y, alpha, r, theta0, L, A)

        # 逆变换到图像域
        image_complex = self.model.inverse_transform_to_image(freq_response)

        # 应用ROI掩模
        roi_mask = self.global_variables["image_interest"]
        image_complex = image_complex * roi_mask

        # 计算与实际复数图像的差异
        target_complex = self.global_variables["complex_temp"]
        diff = image_complex - target_complex

        # 计算目标函数值
        objective = np.sum(np.abs(diff) ** 2)

        return objective

    def chaos_optimization(
        self, initial_params: np.ndarray, bounds: List[Tuple[float, float]], iterations: int = 100
    ) -> np.ndarray:
        """
        混沌优化算法 - 对应MATLAB的混沌优化策略

        Args:
            initial_params: 初始参数
            bounds: 参数边界
            iterations: 迭代次数

        Returns:
            优化后的参数
        """
        # 初始化混沌变量 (对应MATLAB的hd_temp值)
        hd1 = 3.899 * 0.58 * (1 - 0.58)
        hd2 = 3.899 * 0.57 * (1 - 0.57)

        best_params = initial_params.copy()
        best_fval = self._objective_function_distributed(initial_params)

        for i in range(iterations):
            # 更新混沌变量
            hd1 = 3.899 * hd1 * (1 - hd1)
            hd2 = 3.899 * hd2 * (1 - hd2)

            # 生成新的参数 (对应MATLAB的混沌搜索)
            new_params = initial_params.copy()

            if len(initial_params) >= 4:  # 分布式散射中心
                # theta0扰动
                theta0_range = bounds[2][1] - bounds[2][0]
                new_params[2] = bounds[2][0] + theta0_range * hd1

                # L参数扰动
                L_range = bounds[3][1] - bounds[3][0]
                new_params[3] = bounds[3][0] + L_range * hd2

            # 检查边界
            for j, (low, high) in enumerate(bounds):
                if j < len(new_params):
                    new_params[j] = np.clip(new_params[j], low, high)

            # 计算目标函数值
            try:
                fval = self._objective_function_distributed(new_params)

                if fval < best_fval:
                    best_params = new_params.copy()
                    best_fval = fval

                    # 如果找到足够好的解，提前退出
                    if best_fval < 1.0:
                        break

            except:
                continue

        return best_params

    def calculate_scatterer_amplitude(
        self, x: float, y: float, alpha: float, r: float, theta0: float, L: float, roi_image: np.ndarray
    ) -> float:
        """
        计算散射中心幅度 - 对应MATLAB的findlocalA函数

        Args:
            x, y: 位置参数
            alpha, r, theta0, L: 散射中心参数
            roi_image: ROI区域图像

        Returns:
            散射强度A
        """
        try:
            # 生成理论响应 (A=1)
            freq_response, _ = self.model.spotlight_imaging(x, y, alpha, r, theta0, L, 1.0)

            # 逆变换到图像域
            image_complex = self.model.inverse_transform_to_image(freq_response)

            # 应用ROI掩模
            roi_mask = self.global_variables.get("image_interest", np.ones_like(roi_image))
            image_complex = image_complex * roi_mask

            # 取幅度
            theory_image = np.abs(image_complex)

            # 最小二乘估计幅度参数
            theory_flat = theory_image.flatten()
            roi_flat = roi_image.flatten()

            # 使用伪逆求解 A (对应MATLAB的 A = Z\image)
            if np.sum(theory_flat**2) > 1e-10:
                A = np.dot(theory_flat, roi_flat) / np.dot(theory_flat, theory_flat)
            else:
                A = 1.0

            # 限制A的范围，防止过大的值
            A = max(min(A, 1000.0), 0.1)  # 限制在[0.1, 1000]范围内

            # 如果A仍然异常大，使用ROI最大值作为估计
            if A > 100.0:
                A = min(np.max(roi_image), 100.0)

        except:
            A = 1.0

        return A

    def _estimate_distributed_parameters(self, roi_complex: np.ndarray) -> Tuple[float, float]:
        """
        通过频域分析估计分布式散射中心参数 - 对应MATLAB的频域分析逻辑

        Args:
            roi_complex: ROI复数图像

        Returns:
            theta0: 方向角 (度)
            L: 长度参数 (米)
        """
        try:
            # FFT到频域
            freq_data = np.fft.fftshift(np.fft.fft2(roi_complex))
            freq_magnitude = np.abs(freq_data)

            # 找到频域峰值位置
            max_pos = np.unravel_index(np.argmax(freq_magnitude), freq_magnitude.shape)

            # 估计方向角 (简化版本)
            center_row, center_col = freq_magnitude.shape[0] // 2, freq_magnitude.shape[1] // 2
            delta_row = max_pos[0] - center_row
            delta_col = max_pos[1] - center_col

            if delta_col != 0:
                theta0 = np.arctan2(delta_row, delta_col) * 180 / np.pi
            else:
                theta0 = 0.0

            # 估计长度参数 (基于频域宽度)
            L = 0.2  # 默认值

        except:
            theta0 = 0.0
            L = 0.2

        return theta0, L

    def _estimate_alpha_parameter(
        self, roi_image: np.ndarray, A: float, x: float, y: float, theta0: float, L: float
    ) -> float:
        """
        估计alpha参数 - 对应MATLAB的finda函数

        Args:
            roi_image: ROI图像
            A: 散射强度
            x, y, theta0, L: 其他参数

        Returns:
            alpha: 频率依赖指数
        """
        # 测试不同alpha值，选择最佳匹配
        alpha_candidates = [0.0, 0.5, 1.0]
        best_alpha = 0.0
        best_error = float("inf")

        for alpha in alpha_candidates:
            try:
                # 生成理论图像
                freq_response, _ = self.model.spotlight_imaging(x, y, alpha, 0, theta0, L, A)
                theory_image = np.abs(self.model.inverse_transform_to_image(freq_response))

                # 计算匹配误差
                roi_mask = self.global_variables.get("image_interest", np.ones_like(roi_image))
                theory_masked = theory_image * roi_mask
                error = np.sum((theory_masked - roi_image) ** 2)

                if error < best_error:
                    best_error = error
                    best_alpha = alpha

            except:
                continue

        return best_alpha


def test_parameter_optimizer():
    """测试参数优化器功能"""
    print("测试参数优化器...")

    # 创建物理模型和优化器
    model = SARPhysicalModel()
    optimizer = ParameterOptimizer(model)

    # 创建测试数据
    test_roi = np.random.rand(20, 20) * 50
    test_roi[8:12, 8:12] = 200  # 模拟散射中心

    test_complex = test_roi + 1j * test_roi * 0.1

    # 测试局部散射中心优化
    print("\n测试局部散射中心优化...")
    initial_coords = np.array([10, 10])  # 中心位置

    try:
        params, fval = optimizer.optimize_local_scatterer(initial_coords, test_roi, test_complex)
        print(f"局部散射中心参数: {params}")
        print(f"目标函数值: {fval:.6f}")
    except Exception as e:
        print(f"局部优化失败: {e}")

    # 测试分布式散射中心优化
    print("\n测试分布式散射中心优化...")

    try:
        params, fval = optimizer.optimize_distributed_scatterer(initial_coords, test_roi, test_complex)
        print(f"分布式散射中心参数: {params}")
        print(f"目标函数值: {fval:.6f}")
    except Exception as e:
        print(f"分布式优化失败: {e}")

    print("参数优化器测试完成")


if __name__ == "__main__":
    test_parameter_optimizer()
