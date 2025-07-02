"""
分水岭分割模块

实现双阈值分水岭图像分割，对应MATLAB的watershed_image.m函数
"""

import numpy as np
from typing import Tuple, List
from scipy import ndimage
from skimage import segmentation, measure


class WatershedSegmentation:
    """分水岭分割器"""

    def __init__(self):
        """初始化分水岭分割器"""
        self.max_image_size = 128

    def watershed_image(self, magnitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        分水岭图像分割 - 对应MATLAB的watershed_image函数

        实现双阈值分割和连通组件标记

        Args:
            magnitude: 输入幅度图像

        Returns:
            y1: 3dB阈值分割结果
            y2: 20dB阈值分割结果
            R1: 3dB区域数量
            R2: 20dB区域数量
        """
        # 找到最大值和位置
        max_cell = np.max(magnitude)
        max_i, max_j = np.where(magnitude == max_cell)
        max_i, max_j = max_i[0], max_j[0]  # 取第一个最大值位置

        # 计算阈值 (对应MATLAB逻辑)
        threshold_3db = max_cell / (10 ** (3 / 20))  # -3dB阈值
        threshold_20db = max_cell / 3  # -20dB阈值

        # 找到超过阈值的像素位置
        i_3db, j_3db = np.where(magnitude >= threshold_3db)
        i_20db, j_20db = np.where(magnitude >= threshold_20db)

        # 获取图像尺寸
        magnitude_i, magnitude_j = magnitude.shape

        # 初始化标记图像
        y1 = np.zeros((magnitude_i, magnitude_j), dtype=int)
        y2 = np.zeros((magnitude_i, magnitude_j), dtype=int)

        # 标记阈值区域
        for i in range(len(i_3db)):
            y1[i_3db[i], j_3db[i]] = 1

        for i in range(len(i_20db)):
            y2[i_20db[i], j_20db[i]] = 1

        # 对3dB阈值区域进行排序 (按幅度值从大到小)
        coords_3db = list(zip(i_3db, j_3db))
        coords_3db.sort(key=lambda coord: magnitude[coord[0], coord[1]], reverse=True)

        # 对20dB阈值区域进行排序
        coords_20db = list(zip(i_20db, j_20db))
        coords_20db.sort(key=lambda coord: magnitude[coord[0], coord[1]], reverse=True)

        # 连通组件标记 - 3dB区域
        y1, R1 = self._connected_component_labeling_matlab_style(y1, coords_3db, magnitude)

        # 连通组件标记 - 20dB区域
        y2, R2 = self._connected_component_labeling_matlab_style(y2, coords_20db, magnitude)

        # 将标记从1开始改为从0开始 (对应MATLAB最后的减1操作)
        y1[y1 > 0] -= 1
        y2[y2 > 0] -= 1
        R1 -= 1
        R2 -= 1

        return y1, y2, R1, R2

    def _connected_component_labeling_matlab_style(
        self, binary_image: np.ndarray, sorted_coords: List[Tuple[int, int]], magnitude: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        连通组件标记 - 模拟MATLAB的精确逻辑

        Args:
            binary_image: 二值图像
            sorted_coords: 按幅度值排序的坐标列表
            magnitude: 原始幅度图像

        Returns:
            labeled_image: 标记后的图像
            num_regions: 区域数量
        """
        labeled_image = binary_image.copy()
        R = 1  # 区域计数器，从1开始

        height, width = labeled_image.shape

        for k, (i, j) in enumerate(sorted_coords):
            # 边界检查
            if i == 0 or j == 0 or i == height - 1 or j == width - 1:
                continue

            # 提取3x3邻域 (对应MATLAB的plate)
            plate = np.zeros((3, 3), dtype=int)
            plate[0, 0] = labeled_image[i - 1, j - 1]
            plate[0, 1] = labeled_image[i - 1, j]
            plate[0, 2] = labeled_image[i - 1, j + 1]
            plate[1, 0] = labeled_image[i, j - 1]
            plate[1, 1] = labeled_image[i, j]
            plate[1, 2] = labeled_image[i, j + 1]
            plate[2, 0] = labeled_image[i + 1, j - 1]
            plate[2, 1] = labeled_image[i + 1, j]
            plate[2, 2] = labeled_image[i + 1, j + 1]

            max_plate = np.max(plate)

            if max_plate <= 1:
                # 新区域
                R += 1
                labeled_image[i, j] = R
            else:
                # 找到邻域中的最小非零标记
                temp = max_plate
                merge_positions = []

                for pi in range(3):
                    for pj in range(3):
                        if plate[pi, pj] <= temp and plate[pi, pj] > 1:
                            temp = plate[pi, pj]
                            merge_positions.append((pi, pj))

                # 设置当前像素标记
                labeled_image[i, j] = temp

                # 合并相邻区域 (将所有标记为-1的位置设为temp)
                for pi, pj in merge_positions:
                    if plate[pi, pj] != temp:
                        # 将该位置标记为需要合并
                        ni, nj = i + pi - 1, j + pj - 1
                        labeled_image[ni, nj] = temp

        return labeled_image, R

    def extract_roi(self, segmented_image: np.ndarray, region_id: int) -> np.ndarray:
        """
        提取指定区域的ROI掩模 - 对应MATLAB的ROI函数

        Args:
            segmented_image: 分割后的图像
            region_id: 区域ID

        Returns:
            ROI掩模 (1表示该区域，0表示其他)
        """
        # 找到指定区域的像素位置
        roi_mask = (segmented_image == region_id).astype(int)

        return roi_mask

    def _dual_threshold_segmentation(self, magnitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        双阈值分割

        Args:
            magnitude: 输入幅度图像

        Returns:
            binary_3db: -3dB阈值二值图像
            binary_20db: -20dB阈值二值图像
        """
        max_val = np.max(magnitude)
        threshold_3db = max_val / (10 ** (3 / 20))
        threshold_20db = max_val / 3

        binary_3db = (magnitude >= threshold_3db).astype(int)
        binary_20db = (magnitude >= threshold_20db).astype(int)

        return binary_3db, binary_20db

    def get_region_properties(self, labeled_image: np.ndarray, original_image: np.ndarray) -> List[dict]:
        """
        获取每个区域的属性

        Args:
            labeled_image: 标记后的图像
            original_image: 原始图像

        Returns:
            区域属性列表
        """
        properties = []

        unique_labels = np.unique(labeled_image)
        unique_labels = unique_labels[unique_labels > 0]  # 排除背景(0)

        for label in unique_labels:
            mask = labeled_image == label

            if np.sum(mask) > 0:
                # 计算区域属性
                coords = np.where(mask)
                region_pixels = original_image[mask]

                prop = {
                    "label": label,
                    "area": np.sum(mask),
                    "centroid": (np.mean(coords[0]), np.mean(coords[1])),
                    "bbox": (np.min(coords[0]), np.min(coords[1]), np.max(coords[0]), np.max(coords[1])),
                    "mean_intensity": np.mean(region_pixels),
                    "max_intensity": np.max(region_pixels),
                    "total_intensity": np.sum(region_pixels),
                }

                properties.append(prop)

        return properties

    def visualize_segmentation(self, original_image: np.ndarray, labeled_image: np.ndarray) -> np.ndarray:
        """
        可视化分割结果

        Args:
            original_image: 原始图像
            labeled_image: 分割标记图像

        Returns:
            可视化图像
        """
        # 创建伪彩色图像
        unique_labels = np.unique(labeled_image)
        colored_image = np.zeros_like(labeled_image, dtype=float)

        for i, label in enumerate(unique_labels):
            if label > 0:  # 跳过背景
                colored_image[labeled_image == label] = (i + 1) * 50

        return colored_image


def test_watershed_segmentation():
    """测试分水岭分割功能"""
    print("测试分水岭分割...")

    # 创建测试图像
    test_image = np.random.rand(128, 128) * 50

    # 添加一些高亮区域模拟目标
    test_image[60:70, 60:70] = 500  # 主目标
    test_image[40:45, 80:85] = 300  # 次目标
    test_image[80:85, 40:45] = 200  # 弱目标

    # 创建分割器
    segmenter = WatershedSegmentation()

    # 执行分水岭分割
    print("执行分水岭分割...")
    y1, y2, R1, R2 = segmenter.watershed_image(test_image)

    print(f"3dB阈值区域数量: {R1}")
    print(f"20dB阈值区域数量: {R2}")
    print(f"y1唯一值: {np.unique(y1)}")
    print(f"y2唯一值: {np.unique(y2)}")

    # 测试ROI提取
    print("\n测试ROI提取...")
    if R1 > 0:
        roi_mask = segmenter.extract_roi(y1, 1)  # 提取第一个区域
        print(f"第一个ROI区域像素数: {np.sum(roi_mask)}")

    # 获取区域属性
    print("\n获取区域属性...")
    properties = segmenter.get_region_properties(y1, test_image)
    for prop in properties[:3]:  # 只显示前3个区域
        print(
            f"区域{prop['label']}: 面积={prop['area']}, 质心={prop['centroid']}, 最大强度={prop['max_intensity']:.2f}"
        )

    print("分水岭分割测试完成")


if __name__ == "__main__":
    test_watershed_segmentation()
