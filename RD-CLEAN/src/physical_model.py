"""
SAR物理建模模块

实现严格的SAR散射中心物理模型，对应MATLAB的model_rightangle.m、spotlight.m等函数
"""

import numpy as np
from typing import Tuple, Optional
from scipy.signal import windows
import warnings


class SARPhysicalModel:
    """SAR散射中心物理建模"""

    def __init__(self):
        """初始化SAR物理模型参数"""
        # SAR系统参数 (对应MATLAB设置)
        self.fc = 1e10  # 载频 (Hz)
        self.B = 5e8  # 带宽 (Hz)
        self.omega = 2.86  # 观察角 (度)
        self.p = 84  # 频域采样点数
        self.q = 128  # 图像尺寸
        self.c = 3e8  # 光速 (m/s)

        # 预计算的参数
        self.omega_rad = self.omega * 2 * np.pi / 360  # 转换为弧度
        self.b = self.B / self.fc  # 相对带宽

        # 频率范围计算 (对应MATLAB中的fx1, fx2, fy1, fy2)
        self.fx1 = (1 - self.b / 2) * self.fc
        self.fx2 = (1 + self.b / 2) * self.fc
        self.fy1 = -self.fc * np.sin(self.omega_rad / 2)
        self.fy2 = self.fc * np.sin(self.omega_rad / 2)

    def model_rightangle(
        self, fx: float, fy: float, x: float, y: float, alpha: float, r: float, theta0: float, L: float, A: float
    ) -> complex:
        """
        SAR散射中心物理模型 - 对应MATLAB的model_rightangle.m

        实现公式: E = A * E1 * E2 * E3 * E4

        Args:
            fx, fy: 频率坐标 (Hz)
            x, y: 散射中心位置 (米)
            alpha: 频率依赖指数
            r: 角度依赖参数
            theta0: 方向角 (度)
            L: 长度参数 (米)
            A: 散射强度

        Returns:
            复数散射响应
        """
        # 转换角度为弧度
        theta0_rad = theta0 * np.pi / 180

        # 计算频率和角度
        f = np.sqrt(fx * fx + fy * fy)
        theta = np.arctan2(fy, fx)  # 使用atan2以处理四个象限

        # E1: 频率依赖项
        E1 = (1j * f / self.fc) ** alpha

        # E2: 位置相位项
        phase_term = -1j * 4 * np.pi * f / self.c * (x * np.cos(theta) + y * np.sin(theta))
        E2 = np.exp(phase_term)

        # E3: 长度调制项 (sinc函数)
        # MATLAB的sinc与数学定义不同，需要调整
        sinc_arg = 2 * f * L * np.sin(theta - theta0_rad) / self.c
        if abs(sinc_arg) < 1e-10:
            E3 = 1.0  # sinc(0) = 1
        else:
            E3 = np.sin(np.pi * sinc_arg) / (np.pi * sinc_arg)

        # E4: 方向性项
        E4 = np.exp(((-fy / self.fc) / (2 * np.sin(self.omega_rad / 2))) * r)

        # 组合所有项
        E = A * E1 * E2 * E3 * E4

        return E

    def generate_frequency_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成频率网格

        对应MATLAB中的双重for循环生成fx, fy采样点

        Returns:
            fx_grid, fy_grid: 频率网格 (p x p)
        """
        # 生成频率采样点
        fx_range = np.linspace(self.fx1, self.fx2, self.p)
        fy_range = np.linspace(self.fy1, self.fy2, self.p)

        # 创建网格
        fx_grid, fy_grid = np.meshgrid(fx_range, fy_range, indexing="ij")

        return fx_grid, fy_grid

    def spotlight_imaging(
        self, x: float, y: float, alpha: float, r: float, theta0: float, L: float, A: float
    ) -> Tuple[np.ndarray, float]:
        """
        聚束成像建模 - 对应MATLAB的spotlight.m

        Args:
            x, y: 散射中心位置 (米)
            alpha: 频率依赖指数
            r: 角度依赖参数
            theta0: 方向角 (度)
            L: 长度参数 (米)
            A: 散射强度

        Returns:
            Z: 频域响应 (q x q)
            s: 平均功率
        """
        # 生成频率网格
        fx_grid, fy_grid = self.generate_frequency_grid()

        # 计算频域响应
        K = np.zeros((self.p, self.p), dtype=complex)
        total_power = 0.0

        for i in range(self.p):
            for j in range(self.p):
                fx = fx_grid[i, j]
                fy = fy_grid[i, j]

                # 计算散射响应
                response = self.model_rightangle(fx, fy, x, y, alpha, r, theta0, L, A)
                K[i, j] = response

                # 累计功率
                total_power += abs(response) ** 2

        # 翻转矩阵 (对应MATLAB的flipud)
        K = np.flipud(K)

        # 应用泰勒窗
        K = self.apply_taylor_window(K)

        # 扩展到图像尺寸
        Z = np.zeros((self.q, self.q), dtype=complex)
        Z[: self.p, : self.p] = K

        # 计算平均功率
        s = total_power / (self.p * self.p)

        return Z, s

    def apply_taylor_window(self, data: np.ndarray) -> np.ndarray:
        """
        应用泰勒窗函数 - 对应MATLAB的taylorwin

        Args:
            data: 输入数据 (p x p)

        Returns:
            加窗后的数据
        """
        # 生成泰勒窗 (对应MATLAB: taylorwin(p, 3, -35))
        try:
            # 使用scipy的taylor窗，参数对应MATLAB
            taylor_win = windows.taylor(self.p, nbar=3, sll=-35)

            # 检查是否包含NaN或无穷值
            if np.any(np.isnan(taylor_win)) or np.any(np.isinf(taylor_win)):
                raise ValueError("Taylor窗包含无效值")

        except Exception as e:
            # 如果scipy版本不支持或出现错误，使用汉宁窗作为替代
            warnings.warn(f"泰勒窗失败 ({e})，使用汉宁窗替代")
            taylor_win = windows.hann(self.p)

        # 创建2D窗函数
        window_2d = np.outer(taylor_win, taylor_win)

        # 应用窗函数
        windowed_data = data * window_2d

        # 检查结果是否包含NaN值
        if np.any(np.isnan(windowed_data)):
            warnings.warn("窗函数处理后包含NaN值，使用原始数据")
            return data

        return windowed_data

    def inverse_transform_to_image(self, freq_data: np.ndarray) -> np.ndarray:
        """
        频域到图像域的逆变换

        对应MATLAB的 ifft2 + ifftshift

        Args:
            freq_data: 频域数据

        Returns:
            图像域数据
        """
        # 执行2D逆FFT
        image_data = np.fft.ifft2(freq_data)

        # 执行ifftshift
        image_data = np.fft.ifftshift(image_data)

        return image_data

    def simulate_scatterer_frequency_domain(
        self, x: float, y: float, alpha: float, r: float, theta0: float, L: float, A: float
    ) -> np.ndarray:
        """
        生成单个散射中心的频域响应 - 对应MATLAB的spotlight函数

        注意：这里只返回频域响应，不进行ifft2变换！

        Args:
            x, y: 散射中心位置 (米)
            alpha: 频率依赖指数
            r: 角度依赖参数
            theta0: 方向角 (度)
            L: 长度参数 (米)
            A: 散射强度

        Returns:
            频域响应 (q x q)
        """
        # 频域建模
        freq_response, _ = self.spotlight_imaging(x, y, alpha, r, theta0, L, A)

        return freq_response

    def simulate_scatterers_from_frequency_sum(self, freq_sum: np.ndarray) -> np.ndarray:
        """
        从频域响应总和重构SAR图像 - 对应MATLAB的simulation函数最后步骤

        Args:
            freq_sum: 所有散射中心频域响应的总和

        Returns:
            重构的SAR图像 (q x q)
        """
        # 统一进行ifft2和ifftshift (对应MATLAB: K=ifft2(K); K=ifftshift(K);)
        image_complex = self.inverse_transform_to_image(freq_sum)

        # 取幅度 (对应MATLAB: K=abs(K);)
        image_magnitude = np.abs(image_complex)

        return image_magnitude

    def simulate_scatterer(
        self, x: float, y: float, alpha: float, r: float, theta0: float, L: float, A: float
    ) -> np.ndarray:
        """
        模拟单个散射中心的SAR图像 (保持向后兼容)

        警告：此函数不符合MATLAB算法，仅用于测试！
        正确的方法是使用 simulate_scatterer_frequency_domain + simulate_scatterers_from_frequency_sum

        Args:
            x, y: 散射中心位置 (米)
            alpha: 频率依赖指数
            r: 角度依赖参数
            theta0: 方向角 (度)
            L: 长度参数 (米)
            A: 散射强度

        Returns:
            SAR图像 (q x q)
        """
        # 生成频域响应
        freq_response = self.simulate_scatterer_frequency_domain(x, y, alpha, r, theta0, L, A)

        # 单独变换到图像域 (注意：这不是正确的MATLAB方法！)
        image_magnitude = self.simulate_scatterers_from_frequency_sum(freq_response)

        return image_magnitude

    def validate_parameters(
        self, x: float, y: float, alpha: float, r: float, theta0: float, L: float, A: float
    ) -> bool:
        """
        验证散射中心参数的合理性

        Args:
            散射中心参数

        Returns:
            参数是否有效
        """
        # 位置范围检查 (假设合理范围为±10米)
        if abs(x) > 10 or abs(y) > 10:
            print(f"警告: 位置参数超出合理范围: x={x}, y={y}")
            return False

        # alpha参数检查 (通常在0-2之间)
        if alpha < -1 or alpha > 3:
            print(f"警告: alpha参数超出合理范围: {alpha}")
            return False

        # 角度参数检查
        if abs(theta0) > 180:
            print(f"警告: 角度参数超出范围: {theta0}")
            return False

        # 长度参数检查 (不应为负)
        if L < 0:
            print(f"警告: 长度参数为负值: {L}")
            return False

        # 幅度参数检查
        if A <= 0:
            print(f"警告: 幅度参数非正值: {A}")
            return False

        return True

    def get_system_parameters(self) -> dict:
        """
        获取SAR系统参数

        Returns:
            系统参数字典
        """
        return {
            "fc": self.fc,
            "B": self.B,
            "omega_deg": self.omega,
            "omega_rad": self.omega_rad,
            "p": self.p,
            "q": self.q,
            "c": self.c,
            "fx_range": (self.fx1, self.fx2),
            "fy_range": (self.fy1, self.fy2),
        }


# 工具函数
def create_standard_sar_model() -> SARPhysicalModel:
    """创建标准SAR物理模型"""
    return SARPhysicalModel()


def test_physical_model():
    """测试物理模型功能"""
    print("测试SAR物理模型...")

    # 创建模型
    model = SARPhysicalModel()

    # 显示系统参数
    params = model.get_system_parameters()
    print("SAR系统参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 测试单点散射中心
    print("\n测试单点散射中心模拟...")
    x, y = 0.1, 0.05  # 位置
    alpha, r = 0.5, 0.1  # 频率和角度依赖
    theta0, L = 15.0, 0.2  # 方向和长度
    A = 1.0  # 幅度

    # 验证参数
    if model.validate_parameters(x, y, alpha, r, theta0, L, A):
        print("参数验证通过")

        # 模拟散射中心
        image = model.simulate_scatterer(x, y, alpha, r, theta0, L, A)
        print(f"生成图像尺寸: {image.shape}")
        print(f"图像统计: min={np.min(image):.6f}, max={np.max(image):.6f}, mean={np.mean(image):.6f}")
    else:
        print("参数验证失败")

    print("物理模型测试完成")


if __name__ == "__main__":
    test_physical_model()
