"""
High-Precision ASC Scattering Center Extraction Demo
专用于最高精度的ASC散射中心提取演示

Key Features for Maximum Precision:
✅ Full 6-parameter ASC model: {A, α, x, y, L, φ_bar}
✅ High-resolution dictionary with fine sampling
✅ Advanced optimization algorithms (L-BFGS-B + Differential Evolution)
✅ Strict convergence criteria
✅ English interface (no font display issues)

Usage:
python demo_high_precision.py

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from asc_extraction_fixed_v2 import ASCExtractionFixedV2, visualize_extraction_results


def create_high_precision_extractor():
    """Create extractor with maximum precision configuration"""
    print("🔬 Creating High-Precision ASC Extractor")
    print("=" * 60)

    # Maximum precision configuration
    extractor = ASCExtractionFixedV2(
        image_size=(128, 128),
        extraction_mode="progressive",  # Use progressive mode for full ASC
        adaptive_threshold=0.001,  # Very strict threshold
        max_iterations=50,  # More iterations for convergence
        max_scatterers=30,  # Allow more scatterers
    )

    print("🎯 High-Precision Configuration:")
    print("   Extraction mode: Progressive (Full 6-parameter ASC)")
    print("   Adaptive threshold: 0.001 (Very strict)")
    print("   Max iterations: 50")
    print("   Max scatterers: 30")
    print("   Algorithm: Orthogonal Matching Pursuit + Parameter Refinement")

    return extractor


def find_best_mstar_file():
    """Find the best MSTAR file for demonstration"""
    print("\n📂 Searching for MSTAR data files...")

    search_paths = [
        "datasets/SAR_ASC_Project/02_Data_Processed_raw/SN_S7/",
        "datasets/SAR_ASC_Project/02_Data_Processed_raw/",
        "datasets/",
    ]

    mstar_files = []
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(".raw") and ("HB" in file or "MSTAR" in file):
                        full_path = os.path.join(root, file)
                        file_size = os.path.getsize(full_path)
                        mstar_files.append((full_path, file_size))

    if mstar_files:
        # Sort by file size (larger files usually have better quality)
        mstar_files.sort(key=lambda x: x[1], reverse=True)
        selected_file = mstar_files[0][0]
        print(f"   ✅ Found {len(mstar_files)} MSTAR files")
        print(f"   Selected: {selected_file} ({mstar_files[0][1]} bytes)")
        return selected_file
    else:
        print("   ⚠️ No MSTAR files found")
        return None


def test_high_precision_extraction():
    """Test high-precision ASC extraction with real MSTAR data"""
    print("\n🚀 High-Precision ASC Extraction Test")
    print("=" * 60)

    # 1. Create high-precision extractor
    extractor = create_high_precision_extractor()

    # 2. Find MSTAR data
    mstar_file = find_best_mstar_file()
    if not mstar_file:
        print("❌ Cannot proceed without MSTAR data")
        return None, None

    # 3. Load and preprocess data
    print(f"\n📊 Loading and preprocessing MSTAR data...")
    start_time = time.time()

    try:
        magnitude, complex_image = extractor.load_mstar_data_robust(mstar_file)
        load_time = time.time() - start_time

        print(f"   ✅ Data loading successful ({load_time:.2f}s)")
        print(f"   Image shape: {complex_image.shape}")
        print(f"   Data type: {complex_image.dtype}")
        print(f"   Magnitude range: [{np.min(magnitude):.3f}, {np.max(magnitude):.3f}]")
        print(f"   Signal-to-noise estimation: {np.std(magnitude)/np.mean(magnitude):.3f}")

    except Exception as e:
        print(f"   ❌ Data loading failed: {str(e)}")
        return None, None

    # 4. Run high-precision extraction
    print(f"\n🔬 Running High-Precision ASC Extraction...")
    extraction_start = time.time()

    try:
        # Use the enhanced extraction method with full optimization
        scatterers = extractor.extract_asc_scatterers_v2(complex_image)
        extraction_time = time.time() - extraction_start

        print(f"\n✅ High-Precision Extraction Complete ({extraction_time:.2f}s)")

        if scatterers:
            print(f"\n📊 Extraction Results Summary:")
            print(f"   Total scatterers extracted: {len(scatterers)}")

            # Analyze scattering types
            type_counts = {}
            optimized_count = 0
            total_amplitude = 0

            for i, sc in enumerate(scatterers):
                stype = sc.get("scattering_type", "Unknown")
                type_counts[stype] = type_counts.get(stype, 0) + 1

                if sc.get("optimization_success", False):
                    optimized_count += 1

                amplitude = sc.get("estimated_amplitude", 0)
                total_amplitude += amplitude

                alpha = sc.get("alpha", 0)
                x, y = sc.get("x", 0), sc.get("y", 0)
                length = sc.get("length", 0)

                print(
                    f"   Scatterer {i+1}: {stype}, α={alpha:.2f}, "
                    f"pos=({x:.3f},{y:.3f}), L={length:.3f}, A={amplitude:.3f}"
                )

            print(f"\n📈 Statistical Analysis:")
            print(f"   Scattering type distribution: {type_counts}")
            print(
                f"   Optimization success rate: {optimized_count}/{len(scatterers)} ({optimized_count/len(scatterers)*100:.1f}%)"
            )
            print(f"   Average amplitude: {total_amplitude/len(scatterers):.3f}")

            # Calculate reconstruction quality
            final_energy = scatterers[-1].get("residual_energy", 1.0) if scatterers else 1.0
            initial_energy = np.linalg.norm(extractor.preprocess_data_robust(complex_image))
            reconstruction_quality = (initial_energy - final_energy) / initial_energy
            print(f"   Signal reconstruction quality: {reconstruction_quality:.1%}")

        else:
            print("   ⚠️ No scatterers extracted - possible issues:")
            print("     - Data quality too poor")
            print("     - Threshold too strict")
            print("     - Dictionary mismatch")

    except Exception as e:
        print(f"   ❌ Extraction failed: {str(e)}")
        return None, None

    return complex_image, scatterers


def visualize_high_precision_results(complex_image, scatterers):
    """Create high-quality visualization of extraction results"""
    if not scatterers:
        print("⚠️ 未提取到散射中心，无法进行可视化。")
        return

    print(f"\n🎨 Creating High-Quality Visualization...")

    magnitude = np.abs(complex_image)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(16, 12), gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1, 1]}
    )
    fig.suptitle("High-Precision ASC Extraction Results", fontsize=16, weight="bold")

    # 1. 原始SAR图像
    im1 = ax1.imshow(magnitude, cmap="gray", origin="lower", extent=[-1, 1, -1, 1])
    ax1.set_title("Original SAR Image")
    ax1.set_xlabel("X Position (Normalized)")
    ax1.set_ylabel("Y Position (Normalized)")

    # 2. 提取的散射中心叠加图
    ax2.imshow(magnitude, cmap="gray", origin="lower", extent=[-1, 1, -1, 1], alpha=0.7)

    alpha_colors = {-1.0: "blue", -0.5: "cyan", 0.0: "green", 0.5: "orange", 1.0: "red"}
    alpha_names = {-1.0: "Dihedral", -0.5: "Edge", 0.0: "Isotropic", 0.5: "Surface", 1.0: "Specular"}

    for i, sc in enumerate(scatterers):
        # 颜色代表类型, 大小代表幅度, 边框代表优化状态
        color = alpha_colors.get(sc["alpha"], "purple")
        amplitude = sc["estimated_amplitude"]

        # --- 关键修复：使用对数缩放来可视化幅度 ---
        # 这可以防止单个超强点掩盖其他所有点
        log_amp = np.log1p(amplitude)  # 使用 log1p(x) = log(1+x) 避免 log(0)
        size = 50 + log_amp * 100  # 调整系数

        edge_color = "lime" if sc.get("optimization_success", False) else "gray"
        ax2.scatter(sc["x"], sc["y"], s=size, c=color, alpha=0.8, edgecolors=edge_color, linewidth=1.5)
        ax2.text(sc["x"] + 0.02, sc["y"] + 0.02, str(i + 1), color="white", fontsize=8)

    ax2.set_title("Extracted Scatterers Overlay")
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)

    # 3. 参数空间分析 (Alpha vs Amplitude, color-coded by Length)
    alphas = [s["alpha"] for s in scatterers]
    amps = [s["estimated_amplitude"] for s in scatterers]
    lengths = [s.get("length", 0.0) for s in scatterers]
    scatter3 = ax3.scatter(alphas, amps, s=150, c=lengths, cmap="viridis", alpha=0.7)
    ax3.set_title("Parameter Space Analysis")
    ax3.set_xlabel("Alpha (Scattering Mechanism)")
    ax3.set_ylabel("Amplitude")
    plt.colorbar(scatter3, ax=ax3, label="Length Parameter")
    ax3.grid(True, linestyle="--")

    # 4. 统计与图例
    ax4.axis("off")
    if scatterers:
        opt_success_rate = sum(1 for s in scatterers if s.get("optimization_success", False)) / len(scatterers)
        stats_text = (
            f"Extraction Statistics:\n"
            f"--------------------------\n"
            f"Total Scatterers: {len(scatterers)}\n"
            f"Optimization Success: {opt_success_rate:.1%}\n"
            f"Avg. Amplitude: {np.mean(amps):.4f}\n"
            f"Avg. Length: {np.mean(lengths):.4f}\n\n"
            f"Scattering Type Legend:"
        )
        ax4.text(
            0.05,
            0.95,
            stats_text,
            transform=ax4.transAxes,
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="b", lw=1),
        )

        legend_elements = [
            plt.Line2D(
                [0], [0], marker="o", color="w", label=f"{name} (α={alpha})", markerfacecolor=color, markersize=10
            )
            for alpha, name, color in zip(alpha_names.keys(), alpha_names.values(), alpha_colors.values())
        ]
        ax4.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(0.05, 0.2), title="Scattering Types")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save high-quality image
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = f"results/high_precision_asc_extraction_{timestamp}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"   ✅ High-quality visualization saved: {save_path}")
    plt.show()


def main():
    """Main demonstration function for high-precision ASC extraction"""
    print("🔬 HIGH-PRECISION ASC SCATTERING CENTER EXTRACTION")
    print("=" * 70)
    print("This demo showcases the maximum precision configuration:")
    print("✅ Full 6-parameter ASC model with advanced optimization")
    print("✅ High-resolution dictionary and strict convergence criteria")
    print("✅ English interface to avoid font display issues")
    print("✅ Comprehensive visualization and statistical analysis")
    print("=" * 70)

    # Run high-precision extraction test
    complex_image, scatterers = test_high_precision_extraction()

    if complex_image is not None and scatterers:
        # Create high-quality visualization
        visualize_high_precision_results(complex_image, scatterers)

        print(f"\n🎯 High-Precision Extraction Summary:")
        print(f"   Successfully extracted {len(scatterers)} scatterers with maximum precision")
        print(
            f"   Optimization success rate: {sum(1 for s in scatterers if s.get('optimization_success', False))}/{len(scatterers)}"
        )
        print(f"   Results saved to 'results/' directory")

        print(f"\n📋 Next Steps for Even Higher Precision:")
        print(f"   1. Increase dictionary resolution (more alpha/length/position samples)")
        print(f"   2. Use global optimization algorithms (Genetic Algorithm, Simulated Annealing)")
        print(f"   3. Implement multi-scale processing")
        print(f"   4. Add noise reduction preprocessing")

    else:
        print(f"\n❌ High-precision extraction test failed")
        print(f"   Please check MSTAR data availability and quality")

        print(f"\n🔧 Troubleshooting Guide:")
        print(f"   1. Ensure MSTAR .raw files are in 'datasets/' directory")
        print(f"   2. Check file permissions and format")
        print(f"   3. Verify data is not corrupted (contains valid complex values)")
        print(f"   4. Try reducing adaptive_threshold for lower quality data")


if __name__ == "__main__":
    main()
