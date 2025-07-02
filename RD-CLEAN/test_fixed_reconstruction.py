#!/usr/bin/env python3
"""
Test Fixed Reconstruction Algorithm

Validates the corrected MATLAB-compliant reconstruction algorithm
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src path
sys.path.insert(0, "src")

from rd_clean_algorithm import RDCleanAlgorithm, ScattererParameters
from data_loader import SARDataLoader

# Set matplotlib to use English only
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False


def test_reconstruction():
    """Test the fixed reconstruction algorithm"""
    print("Testing Fixed Reconstruction Algorithm")
    print("=" * 50)

    # Test file
    test_file = "../datasets/SAR_ASC_Project/02_Data_Processed_raw/HB03344.017.128x128.raw"

    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        return

    # Load data
    loader = SARDataLoader()
    original_image, _ = loader.load_raw_file(test_file)
    print(f"Loaded image: {original_image.shape}")

    # Extract scatterers
    algorithm = RDCleanAlgorithm()
    algorithm.max_iterations = 5

    scatterer_list = algorithm.extract_scatterers(test_file)
    print(f"Extracted {len(scatterer_list)} scatterers")

    # Test reconstruction
    reconstruction = algorithm._reconstruct_scatterers(scatterer_list)

    # Calculate quality
    original_energy = np.sum(original_image**2)
    residual = np.abs(original_image - reconstruction)
    residual_energy = np.sum(residual**2)
    quality = 1 - residual_energy / original_energy if original_energy > 0 else 0

    print(f"Reconstruction quality: {quality:.6f}")

    # Save results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image, cmap="hot")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(reconstruction, cmap="hot")
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    axes[2].imshow(residual, cmap="hot")
    axes[2].set_title("Residual")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("result/test_result.png", dpi=300)
    print("Saved: result/test_result.png")

    return quality


if __name__ == "__main__":
    test_reconstruction()
