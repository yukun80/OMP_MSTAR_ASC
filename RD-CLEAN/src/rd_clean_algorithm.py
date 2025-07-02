"""
RD-CLEAN主算法模块

整合所有子模块，实现完整的RD-CLEAN散射中心提取算法，对应MATLAB的extrac.m
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
from dataclasses import dataclass

from data_loader import SARDataLoader
from image_processor import ImageProcessor
from watershed_segmentation import WatershedSegmentation
from scatterer_classifier import ScattererClassifier
from parameter_optimizer import ParameterOptimizer
from physical_model import SARPhysicalModel


@dataclass
class ScattererParameters:
    """散射中心参数结构"""

    x: float  # X坐标 (米)
    y: float  # Y坐标 (米)
    alpha: float  # 频率依赖指数
    r: float  # 角度依赖参数
    theta0: float  # 方向角 (弧度)
    L: float  # 长度参数 (米)
    A: float  # 散射强度
    type: int  # 散射中心类型

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "x": self.x,
            "y": self.y,
            "alpha": self.alpha,
            "r": self.r,
            "theta0": self.theta0,
            "L": self.L,
            "A": self.A,
            "type": self.type,
        }

    def to_matlab_format(self) -> List[float]:
        """转换为MATLAB兼容格式"""
        return [self.x, self.y, self.alpha, self.r, self.theta0, self.L, self.A]


class RDCleanAlgorithm:
    """RD-CLEAN主算法"""

    def __init__(self):
        """初始化RD-CLEAN算法"""
        # 初始化所有子模块
        self.data_loader = SARDataLoader()
        self.image_processor = ImageProcessor()
        self.watershed = WatershedSegmentation()
        self.classifier = ScattererClassifier()
        self.model = SARPhysicalModel()
        self.optimizer = ParameterOptimizer(self.model)

        # 算法参数
        self.max_iterations = 50  # 最大迭代次数
        self.convergence_threshold = 10  # 收敛阈值 (对应MATLAB: max(fileimage)/10)
        self.min_scatterer_strength = 0.1  # 最小散射强度

    def extract_scatterers(self, raw_file_path: str) -> List[ScattererParameters]:
        """
        主提取函数 - 对应MATLAB的extrac.m

        Args:
            raw_file_path: .raw文件路径

        Returns:
            散射中心列表
        """
        print(f"开始处理文件: {raw_file_path}")

        # 1. 数据加载
        print("1. 加载SAR数据...")
        fileimage, image_value = self.data_loader.load_raw_file(raw_file_path)

        # 验证数据
        if not self.data_loader.validate_image_data(fileimage, image_value):
            raise ValueError("数据验证失败")

        print(f"   图像尺寸: {fileimage.shape}")
        print(f"   动态范围: {np.min(fileimage):.2f} - {np.max(fileimage):.2f}")

        # 2. 图像预处理
        print("2. 图像预处理...")
        processed_image = self.image_processor.preprocess_image(
            fileimage, enable_target_detect=True, enable_noise_removal=True
        )

        # 3. 迭代提取
        print("3. 开始迭代提取...")
        scatterer_list = self._iterative_extraction(processed_image, image_value)

        print(f"4. 提取完成，共找到 {len(scatterer_list)} 个散射中心")

        return scatterer_list

    def _iterative_extraction(
        self, magnitude_image: np.ndarray, complex_image: np.ndarray
    ) -> List[ScattererParameters]:
        """
        迭代提取算法核心 - 对应MATLAB的主循环

        Args:
            magnitude_image: 幅度图像
            complex_image: 复数图像

        Returns:
            散射中心列表
        """
        scatterer_list = []
        current_magnitude = magnitude_image.copy()
        current_complex = complex_image.copy()

        # 计算收敛阈值
        initial_max = np.max(current_magnitude)
        threshold = initial_max / self.convergence_threshold

        print(f"   收敛阈值: {threshold:.2f}")

        # 迭代提取
        for iteration in range(self.max_iterations):
            print(f"   迭代 {iteration + 1}/{self.max_iterations}")

            # 检查当前图像最大值
            current_max = np.max(current_magnitude)
            print(f"     当前最大值: {current_max:.2f}")

            if current_max < threshold:
                print("     达到收敛条件，停止迭代")
                break

            # 分水岭分割
            y1, y2, R1, R2 = self.watershed.watershed_image(current_magnitude)
            print(f"     分割结果: 3dB区域={R1}, 20dB区域={R2}")

            if R1 <= 0:
                print("     没有找到有效区域，停止迭代")
                break

            # 处理每个区域
            current_iteration_scatterers = []

            for region_id in range(1, R1 + 1):  # MATLAB索引从1开始
                # 检查region_id是否在有效范围内
                unique_regions = np.unique(y1)
                if region_id not in unique_regions:
                    continue

                # 提取ROI
                roi_mask = self.watershed.extract_roi(y1, region_id)
                roi_image = current_magnitude * roi_mask
                roi_complex = current_complex * roi_mask

                # 检查区域有效性
                if np.sum(roi_mask) == 0:
                    continue

                # 散射中心分类
                scatterer_type, coordinates = self.classifier.classify_scatterer(roi_image)

                if scatterer_type == -1:
                    continue  # 无效区域

                # 参数优化
                try:
                    if scatterer_type == 1:  # 局部散射中心
                        params, fval = self.optimizer.optimize_local_scatterer(coordinates[0], roi_image, roi_complex)
                    elif scatterer_type == 0:  # 分布式散射中心
                        params, fval = self.optimizer.optimize_distributed_scatterer(
                            coordinates[0], roi_image, roi_complex
                        )
                    else:  # 多峰散射中心，简化为局部处理
                        params, fval = self.optimizer.optimize_local_scatterer(coordinates[0], roi_image, roi_complex)

                    # 验证参数合理性
                    if self.model.validate_parameters(*params[:7]):
                        scatterer = ScattererParameters(
                            x=params[0],
                            y=params[1],
                            alpha=params[2],
                            r=params[3],
                            theta0=params[4],
                            L=params[5],
                            A=params[6],
                            type=scatterer_type,
                        )
                        current_iteration_scatterers.append(scatterer)

                except Exception as e:
                    warnings.warn(f"区域 {region_id} 优化失败: {e}")
                    continue

            print(f"     本次迭代提取到 {len(current_iteration_scatterers)} 个散射中心")

            # 重构和残差更新
            if len(current_iteration_scatterers) > 0:
                # 重构当前迭代的散射中心
                reconstructed_magnitude = self._reconstruct_scatterers(current_iteration_scatterers)

                # 检查重构结果是否包含NaN值
                if np.any(np.isnan(reconstructed_magnitude)):
                    warnings.warn("重构图像包含NaN值，跳过本次更新")
                    print("     重构失败，跳过本次迭代")
                    continue

                # 更新残差图像
                current_magnitude = np.maximum(
                    current_magnitude - reconstructed_magnitude, np.zeros_like(current_magnitude)
                )

                # 检查更新后的图像是否包含NaN值
                if np.any(np.isnan(current_magnitude)):
                    warnings.warn("残差更新后包含NaN值，停止迭代")
                    break

                # 添加到总列表
                scatterer_list.extend(current_iteration_scatterers)

                # 计算重构质量指标
                residual_energy = np.sum(current_magnitude**2)
                total_energy = np.sum(magnitude_image**2)
                if total_energy > 0:
                    reconstruction_ratio = 1 - residual_energy / total_energy
                    print(f"     重构比例: {reconstruction_ratio:.3f}")
                else:
                    print("     无法计算重构比例（总能量为0）")

            else:
                print("     本次迭代未找到有效散射中心，停止迭代")
                break

        return scatterer_list

    def _reconstruct_scatterers(self, scatterer_list: List[ScattererParameters]) -> np.ndarray:
        """
        从散射中心列表重构SAR图像 - 对应MATLAB的simulation函数

        正确的MATLAB算法流程：
        1. 累加所有散射中心的频域响应 (K=K+K_temp)
        2. 统一进行ifft2和ifftshift
        3. 取幅度得到最终图像

        Args:
            scatterer_list: 散射中心列表

        Returns:
            重构的SAR图像
        """
        if not scatterer_list:
            return np.zeros((self.model.q, self.model.q))

        # 初始化频域响应总和 (对应MATLAB: K=zeros(q,q))
        freq_sum = np.zeros((self.model.q, self.model.q), dtype=complex)

        print(f"    重构 {len(scatterer_list)} 个散射中心...")

        # 累加所有散射中心的频域响应 (对应MATLAB的循环: K=K+K_temp)
        for i, scatterer in enumerate(scatterer_list):
            try:
                # 生成单个散射中心的频域响应 (对应MATLAB的spotlight调用)
                freq_response = self.model.simulate_scatterer_frequency_domain(
                    scatterer.x, scatterer.y, scatterer.alpha, scatterer.r, scatterer.theta0, scatterer.L, scatterer.A
                )

                # 累加频域响应 (对应MATLAB: K=K+K_temp)
                freq_sum += freq_response

                print(f"      散射中心 {i+1}: x={scatterer.x:.3f}, y={scatterer.y:.3f}, A={scatterer.A:.2f}")

            except Exception as e:
                print(f"      警告: 散射中心 {i+1} 重构失败: {e}")
                continue

        # 统一变换到图像域 (对应MATLAB: K=ifft2(K); K=ifftshift(K); K=abs(K))
        reconstructed_image = self.model.simulate_scatterers_from_frequency_sum(freq_sum)

        print(f"    重构完成，图像范围: {np.min(reconstructed_image):.6f} - {np.max(reconstructed_image):.6f}")

        return reconstructed_image

    def simulate_scatterers(self, scatterer_list: List[ScattererParameters]) -> np.ndarray:
        """
        从散射中心列表重构SAR图像 (公共接口)

        Args:
            scatterer_list: 散射中心列表

        Returns:
            重构的SAR图像
        """
        return self._reconstruct_scatterers(scatterer_list)

    def save_results(self, scatterer_list: List[ScattererParameters], output_path: str):
        """
        保存提取结果 - 对应MATLAB的保存格式

        Args:
            scatterer_list: 散射中心列表
            output_path: 输出文件路径
        """
        import pickle

        # 转换为numpy数组格式 (兼容MATLAB)
        scatter_all = []
        for scatterer in scatterer_list:
            scatter_all.append(scatterer.to_matlab_format())

        scatter_all = np.array(scatter_all)

        # 保存为Python格式
        results = {
            "scatter_all": scatter_all,
            "scatterer_objects": scatterer_list,
            "algorithm_info": {"algorithm": "RD-CLEAN", "version": "1.0.0", "total_scatterers": len(scatterer_list)},
        }

        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        print(f"结果已保存到: {output_path}")

    def load_results(self, input_path: str) -> List[ScattererParameters]:
        """
        加载提取结果

        Args:
            input_path: 输入文件路径

        Returns:
            散射中心列表
        """
        import pickle

        with open(input_path, "rb") as f:
            results = pickle.load(f)

        return results.get("scatterer_objects", [])

    def get_algorithm_statistics(self, scatterer_list: List[ScattererParameters]) -> Dict:
        """
        获取算法统计信息

        Args:
            scatterer_list: 散射中心列表

        Returns:
            统计信息字典
        """
        if not scatterer_list:
            return {"error": "散射中心列表为空"}

        # 按类型统计
        type_counts = {}
        for scatterer in scatterer_list:
            type_name = self.classifier.get_scatterer_type_name(scatterer.type)
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # 参数统计
        positions = [(s.x, s.y) for s in scatterer_list]
        amplitudes = [s.A for s in scatterer_list]

        stats = {
            "total_scatterers": len(scatterer_list),
            "type_distribution": type_counts,
            "amplitude_stats": {
                "min": np.min(amplitudes),
                "max": np.max(amplitudes),
                "mean": np.mean(amplitudes),
                "std": np.std(amplitudes),
            },
            "spatial_extent": {
                "x_range": (min(p[0] for p in positions), max(p[0] for p in positions)),
                "y_range": (min(p[1] for p in positions), max(p[1] for p in positions)),
            },
        }

        return stats


def create_rd_clean_algorithm() -> RDCleanAlgorithm:
    """创建标准RD-CLEAN算法实例"""
    return RDCleanAlgorithm()


def test_rd_clean_algorithm():
    """测试RD-CLEAN算法功能"""
    print("测试RD-CLEAN算法...")

    # 创建算法实例
    algorithm = RDCleanAlgorithm()

    # 创建测试数据 (模拟.raw文件数据)
    print("\n创建测试数据...")
    test_magnitude = np.random.rand(128, 128) * 10
    test_complex = test_magnitude + 1j * test_magnitude * 0.1

    # 添加一些模拟散射中心
    test_magnitude[60:68, 60:68] = 200  # 强散射中心
    test_magnitude[40:45, 80:85] = 100  # 中等散射中心
    test_magnitude[80:85, 40:45] = 50  # 弱散射中心

    # 更新复数图像
    test_complex[60:68, 60:68] = 200 + 1j * 50
    test_complex[40:45, 80:85] = 100 + 1j * 25
    test_complex[80:85, 40:45] = 50 + 1j * 12

    # 测试迭代提取
    print("\n测试迭代提取...")
    try:
        scatterer_list = algorithm._iterative_extraction(test_magnitude, test_complex)
        print(f"提取到 {len(scatterer_list)} 个散射中心")

        # 显示前几个散射中心信息
        for i, scatterer in enumerate(scatterer_list[:3]):
            print(f"散射中心 {i+1}:")
            print(f"  位置: ({scatterer.x:.3f}, {scatterer.y:.3f})")
            print(f"  类型: {algorithm.classifier.get_scatterer_type_name(scatterer.type)}")
            print(f"  强度: {scatterer.A:.2f}")

    except Exception as e:
        print(f"迭代提取测试失败: {e}")

    # 测试重构功能
    if "scatterer_list" in locals() and len(scatterer_list) > 0:
        print("\n测试图像重构...")
        try:
            reconstructed = algorithm.simulate_scatterers(scatterer_list)
            print(f"重构图像尺寸: {reconstructed.shape}")
            print(f"重构图像统计: min={np.min(reconstructed):.2f}, max={np.max(reconstructed):.2f}")

            # 计算重构误差
            original_energy = np.sum(test_magnitude**2)
            residual_energy = np.sum((test_magnitude - reconstructed) ** 2)
            reconstruction_quality = 1 - residual_energy / original_energy
            print(f"重构质量: {reconstruction_quality:.3f}")

        except Exception as e:
            print(f"重构测试失败: {e}")

    # 测试统计功能
    if "scatterer_list" in locals() and len(scatterer_list) > 0:
        print("\n测试统计功能...")
        try:
            stats = algorithm.get_algorithm_statistics(scatterer_list)
            print("算法统计:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"统计测试失败: {e}")

    print("\nRD-CLEAN算法测试完成")


if __name__ == "__main__":
    test_rd_clean_algorithm()
