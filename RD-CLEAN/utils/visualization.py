"""
可视化工具函数
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import warnings


def plot_sar_image(
    image: np.ndarray, title: str = "SAR Image", figsize: Tuple[int, int] = (8, 6), colormap: str = "gray"
) -> None:
    """绘制SAR图像"""
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=colormap, origin="upper")
    plt.colorbar(label="Amplitude")
    plt.title(title)
    plt.xlabel("Range (pixels)")
    plt.ylabel("Azimuth (pixels)")
    plt.show()


def plot_scatterer_positions(
    scatterer_list: List, image_shape: Tuple[int, int], title: str = "Scatterer Positions"
) -> None:
    """绘制散射中心位置"""
    plt.figure(figsize=(10, 8))

    # 创建背景网格
    plt.xlim(0, image_shape[1])
    plt.ylim(0, image_shape[0])
    plt.gca().invert_yaxis()  # 匹配图像坐标系

    # 按类型绘制散射中心
    type_colors = {0: "red", 1: "blue", 2: "green"}
    type_labels = {0: "Distributed", 1: "Local", 2: "Multi-peak"}

    for scatterer in scatterer_list:
        # 转换坐标 (从米回到像素)
        col = scatterer.x / (0.3 * 84 / 128) + 64
        row = 64 - scatterer.y / (0.3 * 84 / 128)

        color = type_colors.get(scatterer.type, "black")
        label = type_labels.get(scatterer.type, f"Type {scatterer.type}")

        plt.scatter(
            col, row, c=color, s=scatterer.A * 10, alpha=0.7, label=label if scatterer == scatterer_list[0] else ""
        )

    plt.xlabel("Range (pixels)")
    plt.ylabel("Azimuth (pixels)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_comparison(original: np.ndarray, reconstructed: np.ndarray, figsize: Tuple[int, int] = (15, 5)) -> None:
    """对比原始图像和重构图像"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 原始图像
    im1 = axes[0].imshow(original, cmap="gray", origin="upper")
    axes[0].set_title("Original Image")
    axes[0].set_xlabel("Range (pixels)")
    axes[0].set_ylabel("Azimuth (pixels)")
    plt.colorbar(im1, ax=axes[0])

    # 重构图像
    im2 = axes[1].imshow(reconstructed, cmap="gray", origin="upper")
    axes[1].set_title("Reconstructed Image")
    axes[1].set_xlabel("Range (pixels)")
    axes[1].set_ylabel("Azimuth (pixels)")
    plt.colorbar(im2, ax=axes[1])

    # 残差图像
    residual = original - reconstructed
    im3 = axes[2].imshow(residual, cmap="seismic", origin="upper")
    axes[2].set_title("Residual Image")
    axes[2].set_xlabel("Range (pixels)")
    axes[2].set_ylabel("Azimuth (pixels)")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()


def plot_algorithm_statistics(stats: dict) -> None:
    """绘制算法统计结果"""
    if "type_distribution" not in stats:
        print("统计数据不完整")
        return

    # 类型分布饼图
    type_dist = stats["type_distribution"]
    if type_dist:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.pie(type_dist.values(), labels=type_dist.keys(), autopct="%1.1f%%")
        plt.title("Scatterer Type Distribution")

        # 幅度分布直方图
        if "amplitude_stats" in stats:
            plt.subplot(1, 3, 2)
            amp_stats = stats["amplitude_stats"]
            plt.bar(["Min", "Max", "Mean"], [amp_stats["min"], amp_stats["max"], amp_stats["mean"]])
            plt.title("Amplitude Statistics")
            plt.ylabel("Amplitude")

        # 总数显示
        plt.subplot(1, 3, 3)
        plt.text(
            0.5,
            0.5,
            f"Total Scatterers:\n{stats['total_scatterers']}",
            ha="center",
            va="center",
            fontsize=16,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis("off")
        plt.title("Summary")

        plt.tight_layout()
        plt.show()
