"""
SAR数据加载模块

实现从.raw格式文件加载SAR图像数据，完全对应MATLAB的image_read函数
"""

import os
import re
import numpy as np
from typing import Tuple, Optional
import struct


class SARDataLoader:
    """SAR数据加载器"""

    def __init__(self):
        """初始化数据加载器"""
        self.default_image_size = (128, 128)

    def load_raw_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载.raw文件，返回幅度图像和复数图像

        对应MATLAB函数: image_read.m

        Args:
            file_path: .raw文件路径，格式如 'hb03333.015.128x128.raw'

        Returns:
            fileimage: 幅度图像 (height, width)
            image_value: 复数图像 (height, width)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 从文件名提取图像尺寸
        width, height = self.extract_image_dimensions(file_path)

        # 读取原始二进制数据
        try:
            with open(file_path, "rb") as f:
                # 读取所有数据为float32，使用big-endian字节序 (对应MATLAB的's'格式)
                raw_data = np.frombuffer(f.read(), dtype=">f4")  # big-endian float32
        except Exception as e:
            raise IOError(f"读取文件失败: {file_path}, 错误: {e}")

        # 验证数据长度
        expected_length = width * height * 2  # 幅度 + 相位
        if len(raw_data) < expected_length:
            raise ValueError(f"数据长度不足: 期望{expected_length}, 实际{len(raw_data)}")

        # 解析幅度和相位数据
        fileimage, image_value = self.parse_raw_data(raw_data, width, height)

        return fileimage, image_value

    def extract_image_dimensions(self, filename: str) -> Tuple[int, int]:
        """
        从文件名提取图像维度

        例如: 'hb03333.015.128x128.raw' -> (128, 128)

        Args:
            filename: 文件名或文件路径

        Returns:
            (width, height): 图像宽度和高度
        """
        # 获取文件名部分
        basename = os.path.basename(filename)

        # 反向查找，对应MATLAB的fliplr和findstr逻辑
        reversed_name = basename[::-1]

        # 查找点号位置
        dot_positions = [i for i, char in enumerate(reversed_name) if char == "."]

        if len(dot_positions) < 2:
            # 如果找不到尺寸信息，返回默认值
            print(f"警告: 无法从文件名提取尺寸信息，使用默认值 {self.default_image_size}")
            return self.default_image_size

        # 提取尺寸字符串 (在倒数第一个和第二个点之间)
        size_start = dot_positions[0] + 1
        size_end = dot_positions[1]
        size_str = reversed_name[size_start:size_end][::-1]  # 再次反转回正常顺序

        # 解析 "widthxheight" 格式
        if "x" in size_str:
            parts = size_str.split("x")
            if len(parts) == 2:
                try:
                    width = int(parts[0])
                    height = int(parts[1])
                    return width, height
                except ValueError:
                    pass

        # 如果解析失败，返回默认值
        print(f"警告: 无法解析尺寸信息 '{size_str}'，使用默认值 {self.default_image_size}")
        return self.default_image_size

    def parse_raw_data(self, raw_data: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        解析原始数据为幅度和相位图像

        对应MATLAB image_read.m中的数据处理逻辑

        Args:
            raw_data: 原始float32数组
            width: 图像宽度
            height: 图像高度

        Returns:
            fileimage: 幅度图像 (height, width)
            image_value: 复数图像 (height, width)
        """
        total_pixels = width * height

        # 分离幅度和相位数据
        magnitude = raw_data[:total_pixels]
        phase = raw_data[total_pixels : 2 * total_pixels]

        # 计算复数值和幅度值 (对应MATLAB的for循环逻辑)
        image_magnitude = np.zeros(total_pixels, dtype=np.float32)
        image_value_linear = np.zeros(total_pixels, dtype=np.complex64)

        for i in range(total_pixels):
            temp_mag = magnitude[i]
            temp_phs = phase[i]

            # 计算实部和虚部
            real_part = temp_mag * np.cos(temp_phs)
            imag_part = temp_mag * np.sin(temp_phs)

            # 计算幅度 (对应MATLAB: sqrt(real*real + imag*imag))
            image_magnitude[i] = np.sqrt(real_part * real_part + imag_part * imag_part)

            # 构造复数 (对应MATLAB: real + sqrt(-1)*imag)
            image_value_linear[i] = real_part + 1j * imag_part

        # 重新整形为图像格式 (对应MATLAB的reshape)
        # MATLAB使用列优先顺序，需要转置以匹配行优先的NumPy
        fileimage = image_magnitude.reshape(height, width, order="F")
        image_value = image_value_linear.reshape(height, width, order="F")

        return fileimage, image_value

    def get_image_statistics(self, fileimage: np.ndarray) -> dict:
        """
        获取图像统计信息

        Args:
            fileimage: 幅度图像

        Returns:
            统计信息字典
        """
        return {
            "shape": fileimage.shape,
            "min_value": np.min(fileimage),
            "max_value": np.max(fileimage),
            "mean_value": np.mean(fileimage),
            "std_value": np.std(fileimage),
            "dtype": fileimage.dtype,
        }

    def validate_image_data(self, fileimage: np.ndarray, image_value: np.ndarray) -> bool:
        """
        验证加载的图像数据

        Args:
            fileimage: 幅度图像
            image_value: 复数图像

        Returns:
            验证是否通过
        """
        # 检查形状一致性
        if fileimage.shape != image_value.shape:
            print(f"警告: 幅度图像和复数图像形状不一致: {fileimage.shape} vs {image_value.shape}")
            return False

        # 检查数据类型
        if not np.issubdtype(fileimage.dtype, np.floating):
            print(f"警告: 幅度图像数据类型不正确: {fileimage.dtype}")
            return False

        if not np.issubdtype(image_value.dtype, np.complexfloating):
            print(f"警告: 复数图像数据类型不正确: {image_value.dtype}")
            return False

        # 检查数值范围
        if np.any(fileimage < 0):
            print("警告: 幅度图像包含负值")
            return False

        # 检查是否包含NaN或Inf
        if np.any(~np.isfinite(fileimage)) or np.any(~np.isfinite(image_value)):
            print("警告: 图像数据包含NaN或Inf值")
            return False

        return True


# 测试函数
def test_data_loader():
    """测试数据加载器功能"""
    loader = SARDataLoader()

    # 测试文件名解析
    test_filenames = ["hb03333.015.128x128.raw", "test.256x256.raw", "sample.data.64x64.raw", "invalid_format.raw"]

    print("测试文件名解析:")
    for filename in test_filenames:
        width, height = loader.extract_image_dimensions(filename)
        print(f"  {filename} -> ({width}, {height})")

    print("\n数据加载器初始化完成")


if __name__ == "__main__":
    test_data_loader()
