"""
图像处理模块

实现SAR图像预处理功能，包含目标检测和ROI处理，对应MATLAB的TargetDetect等函数
"""

import numpy as np
from typing import Tuple, Optional
from scipy import ndimage
from skimage import morphology, filters


class ImageProcessor:
    """SAR图像处理器"""

    def __init__(self):
        """初始化图像处理器"""
        self.threshold_3db_factor = 10 ** (3 / 20)  # -3dB阈值因子
        self.threshold_20db_factor = 3  # -20dB阈值因子

    def target_detect(self, image: np.ndarray, neighbor_size: int = 30) -> np.ndarray:
        """
        目标检测函数 - 对应MATLAB的TargetDetect

        Args:
            image: 输入SAR图像
            neighbor_size: 邻域大小

        Returns:
            处理后的图像
        """
        # 创建输出图像的副本
        processed_image = image.copy()

        # 获取图像尺寸
        height, width = image.shape

        # 计算邻域半径
        half_neighbor = neighbor_size // 2

        # 对每个像素进行目标检测处理
        for i in range(half_neighbor, height - half_neighbor):
            for j in range(half_neighbor, width - half_neighbor):
                # 提取邻域
                neighbor_region = image[
                    i - half_neighbor : i + half_neighbor + 1, j - half_neighbor : j + half_neighbor + 1
                ]

                # 计算邻域统计
                neighbor_mean = np.mean(neighbor_region)
                neighbor_std = np.std(neighbor_region)
                current_value = image[i, j]

                # 目标检测判断 (如果当前像素显著高于邻域平均值)
                if current_value > neighbor_mean + 2 * neighbor_std:
                    # 保持原值 (可能是目标)
                    processed_image[i, j] = current_value
                else:
                    # 降低非目标像素的值
                    processed_image[i, j] = current_value * 0.8

        return processed_image

    def apply_roi_mask(self, image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        应用ROI掩模

        Args:
            image: 输入图像
            roi_mask: ROI掩模 (1表示感兴趣区域，0表示其他)

        Returns:
            应用掩模后的图像
        """
        return image * roi_mask

    def enhance_contrast(self, image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
        """
        对比度增强

        Args:
            image: 输入图像
            gamma: 伽马参数

        Returns:
            增强后的图像
        """
        # 归一化到[0,1]
        normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

        # 伽马校正
        enhanced = np.power(normalized, gamma)

        # 恢复原始范围
        result = enhanced * (np.max(image) - np.min(image)) + np.min(image)

        return result

    def adaptive_threshold(self, image: np.ndarray, factor: float = 3.0) -> np.ndarray:
        """
        自适应阈值处理

        Args:
            image: 输入图像
            factor: 阈值因子

        Returns:
            二值化图像
        """
        # 计算自适应阈值
        max_val = np.max(image)
        threshold = max_val / factor

        # 二值化
        binary_image = (image > threshold).astype(np.float32)

        return binary_image

    def remove_noise(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        噪声去除

        Args:
            image: 输入图像
            kernel_size: 滤波核大小

        Returns:
            去噪后的图像
        """
        # 使用中值滤波去除椒盐噪声
        denoised = ndimage.median_filter(image, size=kernel_size)

        return denoised

    def morphological_operations(
        self, binary_image: np.ndarray, operation: str = "opening", kernel_size: int = 3
    ) -> np.ndarray:
        """
        形态学操作

        Args:
            binary_image: 二值图像
            operation: 操作类型 ('opening', 'closing', 'erosion', 'dilation')
            kernel_size: 结构元素大小

        Returns:
            处理后的图像
        """
        # 创建结构元素
        kernel = morphology.disk(kernel_size)

        # 执行指定操作
        if operation == "opening":
            result = morphology.opening(binary_image, kernel)
        elif operation == "closing":
            result = morphology.closing(binary_image, kernel)
        elif operation == "erosion":
            result = morphology.erosion(binary_image, kernel)
        elif operation == "dilation":
            result = morphology.dilation(binary_image, kernel)
        else:
            raise ValueError(f"不支持的形态学操作: {operation}")

        return result.astype(np.float32)

    def extract_roi_statistics(self, image: np.ndarray, roi_mask: np.ndarray) -> dict:
        """
        提取ROI区域统计信息

        Args:
            image: 输入图像
            roi_mask: ROI掩模

        Returns:
            统计信息字典
        """
        # 提取ROI区域像素值
        roi_pixels = image[roi_mask > 0]

        if len(roi_pixels) == 0:
            return {"error": "ROI区域为空"}

        stats = {
            "mean": np.mean(roi_pixels),
            "std": np.std(roi_pixels),
            "min": np.min(roi_pixels),
            "max": np.max(roi_pixels),
            "pixel_count": len(roi_pixels),
            "total_energy": np.sum(roi_pixels**2),
        }

        return stats

    def preprocess_image(
        self,
        image: np.ndarray,
        enable_target_detect: bool = True,
        enable_noise_removal: bool = True,
        enable_contrast_enhancement: bool = False,
    ) -> np.ndarray:
        """
        图像预处理管道

        Args:
            image: 输入SAR图像
            enable_target_detect: 是否启用目标检测
            enable_noise_removal: 是否启用噪声去除
            enable_contrast_enhancement: 是否启用对比度增强

        Returns:
            预处理后的图像
        """
        processed = image.copy()

        # 噪声去除
        if enable_noise_removal:
            processed = self.remove_noise(processed)

        # 目标检测
        if enable_target_detect:
            processed = self.target_detect(processed)

        # 对比度增强
        if enable_contrast_enhancement:
            processed = self.enhance_contrast(processed)

        return processed


def test_image_processor():
    """测试图像处理器功能"""
    print("测试图像处理器...")

    # 创建测试图像
    test_image = np.random.rand(128, 128) * 100
    # 添加一些"目标"(高亮区域)
    test_image[60:68, 60:68] = 500
    test_image[40:45, 80:85] = 300

    # 创建处理器
    processor = ImageProcessor()

    # 测试目标检测
    print("测试目标检测...")
    detected = processor.target_detect(test_image)
    print(f"原始图像统计: min={np.min(test_image):.2f}, max={np.max(test_image):.2f}")
    print(f"检测后统计: min={np.min(detected):.2f}, max={np.max(detected):.2f}")

    # 测试ROI掩模
    print("\n测试ROI掩模...")
    roi_mask = np.zeros_like(test_image)
    roi_mask[55:75, 55:75] = 1  # 创建ROI区域

    masked_image = processor.apply_roi_mask(test_image, roi_mask)
    roi_stats = processor.extract_roi_statistics(test_image, roi_mask)
    print(f"ROI统计: {roi_stats}")

    # 测试预处理管道
    print("\n测试预处理管道...")
    preprocessed = processor.preprocess_image(test_image)
    print(f"预处理后统计: min={np.min(preprocessed):.2f}, max={np.max(preprocessed):.2f}")

    print("图像处理器测试完成")


if __name__ == "__main__":
    test_image_processor()
