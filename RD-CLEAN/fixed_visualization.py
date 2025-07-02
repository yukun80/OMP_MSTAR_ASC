#!/usr/bin/env python3
"""
Fixed RD-CLEAN Algorithm Visualization Script

Correctly handles MATLAB-style data structures and provides English visualization
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add src path
sys.path.insert(0, "src")

from rd_clean_algorithm import RDCleanAlgorithm, ScattererParameters
from data_loader import SARDataLoader

# Set matplotlib to use English only
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False


def create_matlab_compatible_structure(scatterer_list):
    """
    Create MATLAB-compatible scatter_all structure

    Args:
        scatterer_list: List of ScattererParameters

    Returns:
        scatter_all: List of lists, each containing [x, y, alpha, r, theta0, L, A]
    """
    scatter_all = []

    for scatterer in scatterer_list:
        # Convert to MATLAB format: [x, y, alpha, r, theta0, L, A]
        params = [
            scatterer.x,  # x coordinate (meters)
            scatterer.y,  # y coordinate (meters)
            scatterer.alpha,  # frequency dependence
            scatterer.r,  # angle dependence
            scatterer.theta0,  # orientation angle (degrees)
            scatterer.L,  # length parameter (meters)
            scatterer.A,  # amplitude
        ]
        scatter_all.append(params)

    print(f"Created MATLAB-style structure with {len(scatter_all)} scatterers")
    print("Structure format: scatter_all[i] = [x, y, alpha, r, theta0, L, A]")

    return scatter_all


def visualize_scatterer_reconstruction(scatter_all, original_image, title_prefix=""):
    """
    Properly reconstruct and visualize scatterers from MATLAB-style structure

    Args:
        scatter_all: MATLAB-style scatterer list
        original_image: Original SAR image
        title_prefix: Prefix for plot titles
    """
    print(f"\n=== Reconstructing {len(scatter_all)} scatterers ===")

    # Create algorithm instance for reconstruction
    algorithm = RDCleanAlgorithm()

    # Convert MATLAB structure back to ScattererParameters
    scatterer_objects = []
    for i, params in enumerate(scatter_all):
        if len(params) >= 7:
            scatterer = ScattererParameters(
                x=params[0],
                y=params[1],
                alpha=params[2],
                r=params[3],
                theta0=params[4],
                L=params[5],
                A=params[6],
                type=1,  # Default type
            )
            scatterer_objects.append(scatterer)

            print(f"  Scatterer {i+1}: x={params[0]:.3f}, y={params[1]:.3f}, A={params[6]:.2f}")

    # Reconstruct image
    if len(scatterer_objects) > 0:
        reconstructed_image = algorithm.simulate_scatterers(scatterer_objects)

        # Calculate residual
        residual_image = np.abs(original_image - reconstructed_image)

        # Calculate reconstruction quality
        original_energy = np.sum(original_image**2)
        residual_energy = np.sum(residual_image**2)
        if original_energy > 0:
            reconstruction_quality = 1 - residual_energy / original_energy
        else:
            reconstruction_quality = 0

        print(f"Reconstruction quality: {reconstruction_quality:.4f}")

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        im1 = axes[0].imshow(original_image, cmap="hot")
        axes[0].set_title(f"{title_prefix}Original SAR Image")
        axes[0].axis("off")
        plt.colorbar(im1, ax=axes[0])

        # Reconstructed image
        im2 = axes[1].imshow(reconstructed_image, cmap="hot")
        axes[1].set_title(f"{title_prefix}Reconstructed Image")
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1])

        # Residual image
        im3 = axes[2].imshow(residual_image, cmap="hot")
        axes[2].set_title(f"{title_prefix}Residual Image")
        axes[2].axis("off")
        plt.colorbar(im3, ax=axes[2])

        # Add reconstruction quality text
        fig.suptitle(f"Reconstruction Quality: {reconstruction_quality:.4f}", fontsize=14)

        plt.tight_layout()

        # Save with appropriate filename
        filename = f"fixed_reconstruction_{title_prefix.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Visualization saved: {filename}")

        return reconstructed_image, residual_image, reconstruction_quality

    else:
        print("No valid scatterers found for reconstruction")
        return None, None, 0


def analyze_scatterer_distribution(scatter_all):
    """
    Analyze scatterer parameter distribution

    Args:
        scatter_all: MATLAB-style scatterer list
    """
    print(f"\n=== Scatterer Analysis ===")

    if not scatter_all:
        print("No scatterers to analyze")
        return

    # Extract parameters
    x_coords = [params[0] for params in scatter_all]
    y_coords = [params[1] for params in scatter_all]
    alphas = [params[2] for params in scatter_all]
    amplitudes = [params[6] for params in scatter_all]

    print(f"Total scatterers: {len(scatter_all)}")
    print(f"X coordinate range: {min(x_coords):.3f} to {max(x_coords):.3f} meters")
    print(f"Y coordinate range: {min(y_coords):.3f} to {max(y_coords):.3f} meters")
    print(f"Alpha parameter range: {min(alphas):.3f} to {max(alphas):.3f}")
    print(f"Amplitude range: {min(amplitudes):.2f} to {max(amplitudes):.2f}")

    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatterer positions
    axes[0, 0].scatter(x_coords, y_coords, c=amplitudes, cmap="hot", s=50)
    axes[0, 0].set_xlabel("X Coordinate (m)")
    axes[0, 0].set_ylabel("Y Coordinate (m)")
    axes[0, 0].set_title("Scatterer Positions")
    axes[0, 0].grid(True, alpha=0.3)

    # Alpha distribution
    axes[0, 1].hist(alphas, bins=10, alpha=0.7, color="blue")
    axes[0, 1].set_xlabel("Alpha Parameter")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Alpha Parameter Distribution")
    axes[0, 1].grid(True, alpha=0.3)

    # Amplitude distribution
    axes[1, 0].hist(amplitudes, bins=15, alpha=0.7, color="red")
    axes[1, 0].set_xlabel("Amplitude")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Amplitude Distribution")
    axes[1, 0].grid(True, alpha=0.3)

    # Amplitude vs Distance from center
    center_x, center_y = 0, 0  # Assume image center is at origin
    distances = [np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in zip(x_coords, y_coords)]
    axes[1, 1].scatter(distances, amplitudes, alpha=0.7, color="green")
    axes[1, 1].set_xlabel("Distance from Center (m)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].set_title("Amplitude vs Distance")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scatterer_analysis.png", dpi=300, bbox_inches="tight")
    print("Scatterer analysis saved: scatterer_analysis.png")


def test_fixed_algorithm():
    """
    Test the fixed algorithm with proper data handling
    """
    print("=== Testing Fixed RD-CLEAN Algorithm ===")

    # Test file
    test_file = "../datasets/SAR_ASC_Project/02_Data_Processed_raw/HB03344.017.128x128.raw"

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    # Load data
    loader = SARDataLoader()
    fileimage, image_value = loader.load_raw_file(test_file)
    print(f"Loaded SAR image: {fileimage.shape}")

    # Run algorithm with limited iterations for testing
    algorithm = RDCleanAlgorithm()
    algorithm.max_iterations = 5  # Limit for testing

    scatterer_list = algorithm.extract_scatterers(test_file)
    print(f"Extracted {len(scatterer_list)} scatterers")

    # Create MATLAB-compatible structure
    scatter_all = create_matlab_compatible_structure(scatterer_list)

    # Test reconstruction
    reconstructed, residual, quality = visualize_scatterer_reconstruction(scatter_all, fileimage, "Fixed Algorithm ")

    # Analyze scatterer distribution
    analyze_scatterer_distribution(scatter_all)

    # Save results in multiple formats
    save_results_multiple_formats(scatter_all, scatterer_list, "fixed_results")

    print(f"\n=== Test Complete ===")
    print(f"Reconstruction quality: {quality:.4f}")
    return scatter_all, reconstructed, quality


def save_results_multiple_formats(scatter_all, scatterer_objects, filename_base):
    """
    Save results in multiple formats for compatibility

    Args:
        scatter_all: MATLAB-style structure
        scatterer_objects: Python ScattererParameters objects
        filename_base: Base filename for outputs
    """
    # Save as Python pickle
    results = {
        "scatter_all": scatter_all,
        "scatterer_objects": scatterer_objects,
        "format": "MATLAB_compatible",
        "description": "scatter_all contains List[List[float]] where each inner list is [x,y,alpha,r,theta0,L,A]",
    }

    with open(f"{filename_base}.pkl", "wb") as f:
        pickle.dump(results, f)

    # Save as text format (MATLAB readable)
    with open(f"{filename_base}.txt", "w") as f:
        f.write("% RD-CLEAN Extracted Scatterer Parameters\n")
        f.write("% Format: x(m), y(m), alpha, r, theta0(deg), L(m), A\n")
        f.write(f"% Total scatterers: {len(scatter_all)}\n")
        f.write("%\n")

        for i, params in enumerate(scatter_all):
            f.write(f"{params[0]:.6f}, {params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f}, ")
            f.write(f"{params[4]:.6f}, {params[5]:.6f}, {params[6]:.6f}\n")

    # Save as NumPy array
    scatter_array = np.array(scatter_all)
    np.save(f"{filename_base}.npy", scatter_array)

    print(f"Results saved in multiple formats:")
    print(f"  - {filename_base}.pkl (Python pickle)")
    print(f"  - {filename_base}.txt (MATLAB readable)")
    print(f"  - {filename_base}.npy (NumPy array)")


def main():
    """
    Main function to test the fixed algorithm
    """
    print("RD-CLEAN Algorithm - Fixed Version with Proper MATLAB Structure Handling")
    print("=" * 70)

    # Test the fixed algorithm
    try:
        scatter_all, reconstructed, quality = test_fixed_algorithm()

        if quality > 0:
            print(f"\n✓ Algorithm working correctly!")
            print(f"✓ Reconstruction quality: {quality:.4f}")
            print(f"✓ MATLAB-compatible structure created with {len(scatter_all)} scatterers")
        else:
            print(f"\n⚠ Algorithm needs further debugging")
            print(f"⚠ Reconstruction quality: {quality:.4f}")

    except Exception as e:
        print(f"\n✗ Error in fixed algorithm: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Check the generated files:")
    print("  - fixed_reconstruction_fixed_algorithm_.png")
    print("  - scatterer_analysis.png")
    print("  - fixed_results.pkl/txt/npy")


if __name__ == "__main__":
    main()
