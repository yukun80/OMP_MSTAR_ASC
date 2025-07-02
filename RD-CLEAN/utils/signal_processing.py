"""
信号处理工具函数
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


def apply_window(data: np.ndarray, window_type: str = "hann") -> np.ndarray:
    """应用窗函数"""
    if window_type == "hann":
        window = np.hanning(data.shape[0])
        if len(data.shape) == 2:
            window = np.outer(window, np.hanning(data.shape[1]))
    elif window_type == "hamming":
        window = np.hamming(data.shape[0])
        if len(data.shape) == 2:
            window = np.outer(window, np.hamming(data.shape[1]))
    else:
        window = np.ones_like(data)

    return data * window


def normalize_image(image: np.ndarray, method: str = "minmax") -> np.ndarray:
    """图像归一化"""
    if method == "minmax":
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    elif method == "zscore":
        return (image - np.mean(image)) / np.std(image)
    else:
        return image


def calculate_snr(signal_image: np.ndarray, noise_image: np.ndarray) -> float:
    """计算信噪比"""
    signal_power = np.mean(signal_image**2)
    noise_power = np.mean(noise_image**2)

    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float("inf")

    return snr


def extract_magnitude_and_phase(complex_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """提取复数图像的幅度和相位"""
    magnitude = np.abs(complex_image)
    phase = np.angle(complex_image)
    return magnitude, phase
