"""
散射中心分类模块

实现散射中心类型识别和坐标计算，对应MATLAB的selection.m函数
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import ndimage


class ScattererClassifier:
    """散射中心分类器"""

    def __init__(self):
        """初始化散射中心分类器"""
        self.size_threshold = 2  # 最小区域尺寸 (对应MATLAB: size_thresh=2)
        self.inertia_threshold = 3.0  # 惯性比阈值 (对应MATLAB: k>=3判断)

    def classify_scatterer(self, roi_image: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        散射中心分类 - 对应MATLAB的selection函数

        Args:
            roi_image: ROI区域图像

        Returns:
            type: 散射中心类型
                  -1: 无效区域
                   0: 分布式散射中心
                   1: 局部散射中心
                  >1: 多峰散射中心
            coordinates: 散射中心坐标 [row, col]
        """
        # 获取图像尺寸
        image_x, image_y = roi_image.shape

        # 找到非零像素位置 (对应MATLAB: find(segmented_image~=0))
        x, y = np.where(roi_image != 0)

        # 检查区域大小
        size_x = len(x)
        if size_x <= self.size_threshold:
            return -1, np.array([])

        # 计算边界 (对应MATLAB逻辑)
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        # 计算加权质心 (对应MATLAB的质心计算)
        Cx, Cy = self._calculate_center_of_mass(roi_image, x, y, x_min, y_min)

        # 计算惯性比 (对应MATLAB的I_x/I_y计算)
        k = self._calculate_inertia_ratio(roi_image, x, y, x_min, y_min, Cx, Cy)

        # 寻找局部极大值 (对应MATLAB的3x3邻域最大值搜索)
        local_maxima = self._find_local_maxima(roi_image, x, y)
        number = len(local_maxima)

        # 初始化坐标输出
        temp_coordinate = np.zeros((max(1, number), 2))

        # 根据极大值数量分类 (对应MATLAB的分类逻辑)
        if number <= 1:
            # 单峰或无峰情况
            if k < self.inertia_threshold:
                type_result = 1  # 局部散射中心
            else:
                type_result = 0  # 分布式散射中心

            # 设置质心坐标 (对应MATLAB: Cx+x_min-1, Cy+y_min-1)
            temp_coordinate[0, 0] = Cx + x_min
            temp_coordinate[0, 1] = Cy + y_min

        elif number == 2:
            # 双峰情况
            if k <= 1.5:
                type_result = 2
                # 保留两个极大值坐标
                for i in range(2):
                    temp_coordinate[i, 0] = local_maxima[i][0]
                    temp_coordinate[i, 1] = local_maxima[i][1]
            else:
                # 检查两个峰是否在相近的列位置 (对应MATLAB的y坐标差异检查)
                if abs(local_maxima[0][1] - local_maxima[1][1]) < 2:
                    # 视为分布式散射中心
                    type_result = 0
                    temp_coordinate = np.zeros((1, 2))
                    temp_coordinate[0, 0] = Cx + x_min
                    temp_coordinate[0, 1] = Cy + y_min
                else:
                    type_result = 2
                    for i in range(2):
                        temp_coordinate[i, 0] = local_maxima[i][0]
                        temp_coordinate[i, 1] = local_maxima[i][1]

        else:
            # 多峰情况 (number > 2)
            type_result = number

            # 处理多峰情况的特殊逻辑 (对应MATLAB的复杂判断)
            if k > 1.5:
                # 寻找最大值位置
                max_pix = np.max(roi_image)
                max_coor = np.where(roi_image == max_pix)
                max_coor = (max_coor[0][0], max_coor[1][0])  # 取第一个最大值

                # 检查是否有峰值与最大值在相近列位置
                for i in range(number):
                    peak_pos = local_maxima[i]
                    if peak_pos[0] != max_coor[0] or peak_pos[1] != max_coor[1]:
                        if abs(max_coor[1] - peak_pos[1]) < 2:
                            # 合并为分布式散射中心
                            type_result = 0
                            temp_coordinate = np.zeros((1, 2))
                            temp_coordinate[0, 0] = Cx + x_min
                            temp_coordinate[0, 1] = Cy + y_min
                            break

            # 如果仍然是多峰，计算平均位置 (对应MATLAB的平均计算)
            if type_result > 1:
                # 计算所有峰值的平均位置
                sum_coords = np.zeros(2)
                for i in range(min(number, len(local_maxima))):
                    sum_coords += local_maxima[i]
                    temp_coordinate[i, 0] = local_maxima[i][0]
                    temp_coordinate[i, 1] = local_maxima[i][1]

                # 找到最接近平均位置的峰值 (对应MATLAB的偏差计算)
                avg_coords = sum_coords / number
                min_deviation = float("inf")
                best_coord = temp_coordinate[0]

                for i in range(number):
                    deviation = abs(avg_coords[0] - temp_coordinate[i, 0])
                    if deviation < min_deviation:
                        min_deviation = deviation
                        best_coord = temp_coordinate[i]

                # 只返回最佳坐标
                temp_coordinate = np.array([best_coord])
                type_result = 1  # 简化为局部散射中心

        return type_result, temp_coordinate

    def _calculate_center_of_mass(
        self, roi_image: np.ndarray, x: np.ndarray, y: np.ndarray, x_min: int, y_min: int
    ) -> Tuple[float, float]:
        """
        计算加权质心坐标 - 对应MATLAB的质心计算

        Args:
            roi_image: ROI图像
            x, y: 非零像素坐标
            x_min, y_min: 边界最小值

        Returns:
            Cx, Cy: 质心坐标 (相对于区域边界)
        """
        x_size = len(x)

        Cx_temp = 0.0
        Cy_temp = 0.0
        C_temp = 0.0

        for i in range(x_size):
            # 转换为相对坐标 (对应MATLAB: x_v=x(i)-x_min+1)
            x_v = x[i] - x_min + 1
            y_v = y[i] - y_min + 1

            # 加权累加
            weight = roi_image[x[i], y[i]]
            Cx_temp += x_v * weight
            Cy_temp += y_v * weight
            C_temp += weight

        # 计算质心
        Cx = Cx_temp / C_temp if C_temp > 0 else 0
        Cy = Cy_temp / C_temp if C_temp > 0 else 0

        return Cx, Cy

    def _calculate_inertia_ratio(
        self, roi_image: np.ndarray, x: np.ndarray, y: np.ndarray, x_min: int, y_min: int, Cx: float, Cy: float
    ) -> float:
        """
        计算惯性比 k = I_x/I_y - 对应MATLAB的惯性计算

        Args:
            roi_image: ROI图像
            x, y: 非零像素坐标
            x_min, y_min: 边界最小值
            Cx, Cy: 质心坐标

        Returns:
            惯性比 k
        """
        x_size = len(x)

        I_x = 0.0
        I_y = 0.0

        for i in range(x_size):
            x_v = x[i] - x_min + 1
            y_v = y[i] - y_min + 1
            weight = roi_image[x[i], y[i]]

            I_x += ((x_v - Cx) ** 2) * weight
            I_y += ((y_v - Cy) ** 2) * weight

        # 计算惯性比，避免除零
        k = I_x / I_y if I_y > 1e-10 else float("inf")

        return k

    def _find_local_maxima(self, roi_image: np.ndarray, x: np.ndarray, y: np.ndarray) -> List[Tuple[int, int]]:
        """
        寻找局部极大值点 - 对应MATLAB的3x3邻域最大值搜索

        Args:
            roi_image: ROI图像
            x, y: 非零像素坐标

        Returns:
            局部极大值坐标列表
        """
        local_maxima = []
        x_size = len(x)

        for i in range(x_size):
            row, col = x[i], y[i]

            # 边界检查
            if row <= 0 or row >= roi_image.shape[0] - 1 or col <= 0 or col >= roi_image.shape[1] - 1:
                continue

            # 提取3x3邻域 (对应MATLAB的plate)
            plate = roi_image[row - 1 : row + 2, col - 1 : col + 2]
            current_value = roi_image[row, col]

            # 检查是否为局部最大值
            if current_value == np.max(plate):
                local_maxima.append((row, col))

        return local_maxima

    def get_scatterer_type_name(self, type_code: int) -> str:
        """
        获取散射中心类型名称

        Args:
            type_code: 类型代码

        Returns:
            类型名称
        """
        if type_code == -1:
            return "无效区域"
        elif type_code == 0:
            return "分布式散射中心"
        elif type_code == 1:
            return "局部散射中心"
        else:
            return f"多峰散射中心({type_code}个峰)"

    def validate_classification_result(self, type_result: int, coordinates: np.ndarray) -> bool:
        """
        验证分类结果的合理性

        Args:
            type_result: 分类结果
            coordinates: 坐标数组

        Returns:
            结果是否有效
        """
        # 检查类型代码范围
        if type_result < -1:
            return False

        # 检查坐标数组
        if type_result == -1:
            return len(coordinates) == 0
        elif type_result >= 0:
            return len(coordinates) > 0 and coordinates.shape[1] == 2

        return True


def test_scatterer_classifier():
    """测试散射中心分类器功能"""
    print("测试散射中心分类器...")

    # 创建分类器
    classifier = ScattererClassifier()

    # 测试案例1: 局部散射中心 (单个紧凑区域)
    print("\n测试案例1: 局部散射中心")
    roi1 = np.zeros((20, 20))
    roi1[8:12, 8:12] = 100  # 小的紧凑区域
    roi1[10, 10] = 150  # 中心峰值

    type1, coords1 = classifier.classify_scatterer(roi1)
    print(f"类型: {type1} ({classifier.get_scatterer_type_name(type1)})")
    print(f"坐标: {coords1}")

    # 测试案例2: 分布式散射中心 (细长区域)
    print("\n测试案例2: 分布式散射中心")
    roi2 = np.zeros((20, 20))
    roi2[5:15, 8:12] = 80  # 细长区域

    type2, coords2 = classifier.classify_scatterer(roi2)
    print(f"类型: {type2} ({classifier.get_scatterer_type_name(type2)})")
    print(f"坐标: {coords2}")

    # 测试案例3: 多峰散射中心
    print("\n测试案例3: 多峰散射中心")
    roi3 = np.zeros((20, 20))
    roi3[5:8, 5:8] = 120  # 第一个峰
    roi3[12:15, 12:15] = 110  # 第二个峰
    roi3[6, 6] = 150  # 峰值1
    roi3[13, 13] = 140  # 峰值2

    type3, coords3 = classifier.classify_scatterer(roi3)
    print(f"类型: {type3} ({classifier.get_scatterer_type_name(type3)})")
    print(f"坐标: {coords3}")

    # 测试案例4: 无效区域 (太小)
    print("\n测试案例4: 无效区域")
    roi4 = np.zeros((20, 20))
    roi4[10, 10] = 100  # 只有1个像素

    type4, coords4 = classifier.classify_scatterer(roi4)
    print(f"类型: {type4} ({classifier.get_scatterer_type_name(type4)})")
    print(f"坐标: {coords4}")

    print("\n散射中心分类器测试完成")


if __name__ == "__main__":
    test_scatterer_classifier()
