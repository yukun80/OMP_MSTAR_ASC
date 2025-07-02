#!/usr/bin/env python3
"""
Simple test for fixed reconstruction algorithm
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src path
sys.path.insert(0, "src")

from rd_clean_algorithm import RDCleanAlgorithm, ScattererParameters
from data_loader import SARDataLoader


def main():
    print("Simple Test - Fixed Reconstruction Algorithm")
    print("=" * 50)

    # Test file
    test_file = "../datasets/SAR_ASC_Project/02_Data_Processed_raw/HB03344.017.128x128.raw"

    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        return

    # Load data
    print("1. Loading data...")
    loader = SARDataLoader()
    original_image, _ = loader.load_raw_file(test_file)
    print(f"   Loaded image: {original_image.shape}")
    print(f"   Image range: {np.min(original_image):.4f} - {np.max(original_image):.4f}")

    # Extract scatterers (limited iterations for testing)
    print("2. Extracting scatterers...")
    algorithm = RDCleanAlgorithm()
    algorithm.max_iterations = 3  # Limit for quick test

    scatterer_list = algorithm.extract_scatterers(test_file)
    print(f"   Extracted {len(scatterer_list)} scatterers")

    if len(scatterer_list) == 0:
        print("   No scatterers extracted!")
        return

    # Show first few scatterers
    print("   First few scatterers:")
    for i, s in enumerate(scatterer_list[:3]):
        print(f"     {i+1}: x={s.x:.3f}, y={s.y:.3f}, A={s.A:.2f}")

    # Test reconstruction
    print("3. Testing reconstruction...")
    reconstruction = algorithm._reconstruct_scatterers(scatterer_list)
    print(f"   Reconstruction range: {np.min(reconstruction):.6f} - {np.max(reconstruction):.6f}")

    # Calculate quality
    original_energy = np.sum(original_image**2)
    residual = np.abs(original_image - reconstruction)
    residual_energy = np.sum(residual**2)
    quality = 1 - residual_energy / original_energy if original_energy > 0 else 0

    print(f"   Reconstruction quality: {quality:.6f}")

    # Create simple visualization
    print("4. Creating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    im1 = axes[0].imshow(original_image, cmap="hot")
    axes[0].set_title("Original SAR Image")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0])

    # Reconstructed
    im2 = axes[1].imshow(reconstruction, cmap="hot")
    axes[1].set_title(f"Reconstructed (Q={quality:.4f})")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1])

    # Residual
    im3 = axes[2].imshow(residual, cmap="hot")
    axes[2].set_title("Residual")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig("result/simple_test_result.png", dpi=300, bbox_inches="tight")
    print("   Saved: result/simple_test_result.png")

    # Summary
    print("\n5. Test Summary:")
    print(f"   Scatterers extracted: {len(scatterer_list)}")
    print(f"   Reconstruction quality: {quality:.6f}")

    if quality > 0.1:
        print("   ✓ SUCCESS: Good reconstruction quality!")
    elif quality > 0.01:
        print("   ⚠ PARTIAL: Moderate reconstruction quality")
    else:
        print("   ✗ POOR: Low reconstruction quality")

    return quality


if __name__ == "__main__":
    main()
