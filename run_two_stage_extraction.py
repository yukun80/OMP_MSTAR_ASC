"""
Two-Stage High-Precision ASC Extraction
=========================================

This script implements the robust two-stage extraction architecture:

Stage 1: Hypothesize Locations
- Use a neutral (alpha=0) dictionary to avoid systemic bias.
- Run a greedy OMP-like algorithm to find potential scatterer locations (x, y).
- This stage answers "WHERE" the scatterers might be.

Stage 2: Estimate Parameters Locally
- For each hypothesized location, define a small Region of Interest (ROI).
- Run a local, full-parameter optimization (including alpha, L, phi_bar)
  within that ROI.
- This stage answers "WHAT" each scatterer is.

This approach decouples location finding from parameter estimation, leading
to more accurate and physically meaningful results.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import time
import os
from tqdm import tqdm

from asc_extraction_fixed_v2 import ASCExtractionFixedV2
from demo_high_precision import find_best_mstar_file, visualize_high_precision_results


def hypothesize_locations(
    complex_image: np.ndarray, image_size: Tuple[int, int], n_hypotheses: int = 15, position_grid_size: int = 64
) -> List[Tuple[float, float]]:
    """
    Stage 1: V2版本 - 获取更少、更可靠的位置假设
    """
    print("\n--- Stage 1: Hypothesizing Scatterer Locations ---")

    # 1. Use a simplified, neutral extractor for location finding.
    # We create a special instance with a dictionary containing only alpha=0 atoms
    # to avoid the systemic bias towards negative alpha values.
    print("   🔧 Creating a neutral extractor with alpha=0 dictionary...")
    hypothesizer = ASCExtractionFixedV2(
        image_size=image_size,
        max_scatterers=n_hypotheses,
        adaptive_threshold=0.1,  # 使用更严格的阈值 (10%) 来提取最强的散射中心
        # Custom dictionary parameters for a neutral, high-density position grid:
        alpha_values=[0.0],
        length_values=[0.0],
        phi_bar_values=[0.0],
        position_samples=position_grid_size,
    )

    # 2. Run the extraction process. We only care about the parameters of the
    #    scatterers found, not the residuals or convergence details.
    print("   🚀 Running OMP on neutral dictionary to find potential locations...")
    scatterers = hypothesizer.extract_asc_scatterers_v2(complex_image)

    if not scatterers:
        print(f"   ✅ Stage 1 complete. Found 0 potential locations.")
        return []

    scatterers.sort(key=lambda s: s["estimated_amplitude"], reverse=True)

    locations = []
    for sc in scatterers[:n_hypotheses]:
        locations.append((sc["x"], sc["y"]))

    print(f"   ✅ Stage 1 complete. Found {len(locations)} high-quality locations.")
    return locations


def estimate_parameters_locally(
    complex_image: np.ndarray, locations: List[Tuple[float, float]], image_size: Tuple[int, int]
) -> List[Dict]:
    """
    Stage 2: For each location hypothesis, perform local full-parameter estimation.
    """
    print("\n--- Stage 2: Estimating Full Parameters Locally ---")

    # 1. Create a full-featured extractor instance. This one knows about all
    #    possible alpha, length, and phi_bar values for the final estimation.
    print("   🔧 Creating a full-featured extractor for parameter estimation...")
    asc_estimator = ASCExtractionFixedV2(
        image_size=image_size, extraction_mode="progressive"  # Ensure all parameters are available
    )

    final_scatterers = []

    # 2. Iterate through each hypothesized location and run local optimization.
    print(f"   🔬 Optimizing parameters for {len(locations)} locations...")
    for x, y in tqdm(locations, desc="Local Parameter Estimation"):
        estimated_params = asc_estimator.estimate_params_in_roi(complex_image, x, y, roi_size=24)
        if estimated_params:
            final_scatterers.append(estimated_params)

    print(f"   ✅ Stage 2 complete. Successfully estimated {len(final_scatterers)} scatterers.")
    return final_scatterers


def main():
    """Main two-stage extraction process."""
    print("🔬 TWO-STAGE HIGH-PRECISION ASC EXTRACTION")
    print("=" * 70)

    IMAGE_SIZE = (128, 128)

    # Use the same robust file finder as the demo
    mstar_file = find_best_mstar_file()
    if not mstar_file:
        print("❌ Cannot proceed without MSTAR data.")
        return

    # Create a temporary loader instance just to get the data
    print("\n📊 Loading and preprocessing MSTAR data...")
    loader = ASCExtractionFixedV2(image_size=IMAGE_SIZE)
    try:
        magnitude, complex_image = loader.load_mstar_data_robust(mstar_file)
        if np.linalg.norm(complex_image) < 1e-6:
            raise ValueError("Loaded data is empty.")
        print(f"   ✅ Data loading successful.")
    except Exception as e:
        print(f"   ❌ Data loading failed: {str(e)}")
        return

    start_time = time.time()

    # --- STAGE 1 ---
    locations = hypothesize_locations(complex_image, image_size=IMAGE_SIZE, n_hypotheses=50, position_grid_size=64)

    if not locations:
        print("❌ Stage 1 did not find any potential locations. Aborting.")
        return

    # --- STAGE 2 ---
    final_scatterers = estimate_parameters_locally(complex_image, locations, image_size=IMAGE_SIZE)

    total_time = time.time() - start_time
    print(f"\n✅ Two-Stage Extraction Complete in {total_time:.2f}s")

    # --- VISUALIZATION ---
    if final_scatterers:
        print(f"\n📊 Final Results Summary:")
        print(f"   Total scatterers estimated: {len(final_scatterers)}")
        # Sort by amplitude for better visualization legend
        final_scatterers.sort(key=lambda s: s["estimated_amplitude"], reverse=True)
        visualize_high_precision_results(complex_image, final_scatterers)
    else:
        print("❌ Stage 2 failed to estimate any scatterers from the hypotheses.")


if __name__ == "__main__":
    main()
