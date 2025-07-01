"""
High-Precision ASC Scattering Center Extraction System
高精度ASC散射中心提取系统

配置用于最高精度的6参数ASC提取：
- Full ASC mode: 完整的 {A, α, x, y, L, φ_bar} 6参数提取
- High-resolution dictionary: 高分辨率字典采样
- Advanced optimization: 高级参数优化算法
- English interface: 纯英文界面避免显示问题

Usage:
python asc_extraction_high_precision.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import OrthogonalMatchingPursuit
import struct
from typing import Tuple, List, Dict, Optional
import warnings
import time
import os

warnings.filterwarnings("ignore")


class ASCExtractionHighPrecision:
    """High-Precision ASC Extraction System for Maximum Accuracy"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        extraction_mode: str = "full_asc",  # Maximum precision mode
        adaptive_threshold: float = 0.001,  # Stricter threshold
        max_iterations: int = 50,  # More iterations
        max_scatterers: int = 30,  # More scatterers
        high_resolution: bool = True,  # High-res dictionary
    ):
        self.image_size = image_size
        self.extraction_mode = extraction_mode
        self.adaptive_threshold = adaptive_threshold
        self.max_iterations = max_iterations
        self.max_scatterers = max_scatterers
        self.high_resolution = high_resolution

        # SAR system parameters
        self.fc = 1e10  # Center frequency
        self.B = 1e9  # Bandwidth
        self.omega = np.pi / 3  # Synthetic aperture angle
        self.scene_size = 30.0  # Scene size (meters)

        # Configure extraction parameters
        self._configure_high_precision_mode()

        print(f"🔬 High-Precision ASC Extraction System Initialized")
        print(f"   Mode: {extraction_mode} (Full 6-Parameter ASC)")
        print(f"   Adaptive threshold: {adaptive_threshold}")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Max scatterers: {max_scatterers}")
        print(f"   High resolution: {high_resolution}")
        print(f"   Scene size: {self.scene_size}m")

    def _configure_high_precision_mode(self):
        """Configure parameters for maximum precision"""
        if self.high_resolution:
            # High-resolution sampling for maximum accuracy
            self.alpha_values = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
            self.length_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
            self.phi_bar_values = np.linspace(0, np.pi, 12)  # 12 orientations
            self.position_samples = 48  # High-res position grid
            print("   🎯 High-Resolution Mode: Maximum parameter coverage")
        else:
            # Standard sampling
            self.alpha_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
            self.length_values = [0.0, 0.1, 0.5, 1.0]
            self.phi_bar_values = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
            self.position_samples = 32
            print("   🎯 Standard Mode: Balanced speed and accuracy")

    def load_mstar_data_robust(self, raw_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Robust MSTAR data loading with NaN/Inf handling"""
        print(f"📂 Loading MSTAR data: {raw_file_path}")

        try:
            with open(raw_file_path, "rb") as f:
                data = f.read()

            # Try different data format parsing
            num_values = len(data) // 4

            # Method 1: little-endian float32
            try:
                real_imag = struct.unpack(f"<{num_values}f", data)
                print("   Using little-endian float32 format")
            except:
                # Method 2: big-endian float32
                try:
                    real_imag = struct.unpack(f">{num_values}f", data)
                    print("   Using big-endian float32 format")
                except:
                    # Method 3: int16 format with conversion
                    num_values_int16 = len(data) // 2
                    int_data = struct.unpack(f"<{num_values_int16}h", data)
                    real_imag = [float(x) / 32767.0 for x in int_data]
                    print("   Using int16 format with normalization")

            # Check data validity
            if np.any(np.isnan(real_imag)) or np.any(np.isinf(real_imag)):
                print("   ⚠️ Detected NaN/Inf values, performing data cleaning...")
                real_imag = np.array(real_imag)
                # Replace NaN and Inf with 0
                real_imag = np.where(np.isnan(real_imag) | np.isinf(real_imag), 0.0, real_imag)

            # Reconstruct complex image
            if len(real_imag) % 2 != 0:
                real_imag = real_imag[:-1]  # Ensure even length

            complex_values = []
            for i in range(0, len(real_imag), 2):
                if i + 1 < len(real_imag):
                    complex_values.append(complex(real_imag[i], real_imag[i + 1]))

            # Ensure data length matches image size
            expected_size = self.image_size[0] * self.image_size[1]
            if len(complex_values) > expected_size:
                complex_values = complex_values[:expected_size]
            elif len(complex_values) < expected_size:
                # Pad with zeros
                complex_values.extend([0.0 + 0.0j] * (expected_size - len(complex_values)))

            complex_image = np.array(complex_values).reshape(self.image_size)
            magnitude = np.abs(complex_image)

            # Final data validation
            if np.any(np.isnan(complex_image)) or np.any(np.isinf(complex_image)):
                print("   ⚠️ Complex image still contains NaN/Inf, performing final cleanup...")
                complex_image = np.where(np.isnan(complex_image) | np.isinf(complex_image), 0.0 + 0.0j, complex_image)
                magnitude = np.abs(complex_image)

            print(f"   ✅ Data loading successful")
            print(f"      Image size: {complex_image.shape}")
            print(f"      Magnitude range: [{np.min(magnitude):.3f}, {np.max(magnitude):.3f}]")
            print(f"      Signal energy: {np.linalg.norm(complex_image):.3f}")
            print(f"      Valid data ratio: {np.sum(magnitude > 0) / magnitude.size:.1%}")

            return magnitude, complex_image

        except Exception as e:
            print(f"   ❌ Data loading failed: {str(e)}")
            # Return zero data as fallback
            complex_image = np.zeros(self.image_size, dtype=complex)
            magnitude = np.zeros(self.image_size)
            return magnitude, complex_image

    def preprocess_data_robust(self, complex_image: np.ndarray) -> np.ndarray:
        """Robust data preprocessing"""
        print("⚙️ Robust data preprocessing...")

        # Signal vectorization
        signal = complex_image.flatten()

        # Check and clean data
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            print("   ⚠️ Detected anomalies, performing cleanup...")
            signal = np.where(np.isnan(signal) | np.isinf(signal), 0.0 + 0.0j, signal)

        # Calculate signal characteristics
        signal_energy = np.linalg.norm(signal)
        signal_max = np.max(np.abs(signal))

        if signal_energy < 1e-12:
            print("   ⚠️ Signal energy too low, using simulated data...")
            # Create simple test signal
            signal = 0.1 * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
            signal_energy = np.linalg.norm(signal)

        # Robust normalization
        signal_normalized = signal / np.sqrt(signal_energy)

        print(f"   Signal length: {len(signal)}")
        print(f"   Processed energy: {np.linalg.norm(signal_normalized):.3f}")
        print(f"   Max magnitude: {np.max(np.abs(signal_normalized)):.3f}")

        return signal_normalized

    def _generate_robust_asc_atom(
        self,
        x: float,
        y: float,
        alpha: float,
        length: float = 0.0,
        phi_bar: float = 0.0,
        fx_range: np.ndarray = None,
        fy_range: np.ndarray = None,
    ) -> np.ndarray:
        """
        Generate numerically robust ASC atom with correct physical scaling
        """
        if fx_range is None:
            fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        if fy_range is None:
            fy_range = np.linspace(
                -self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1]
            )

        FX, FY = np.meshgrid(fx_range, fy_range, indexing="ij")

        # Key fix: Unified physical scaling
        C = 299792458.0  # Speed of light
        x_meters = x * (self.scene_size / 2.0)  # Convert normalized coords [-1,1] to meters
        y_meters = y * (self.scene_size / 2.0)

        f_magnitude = np.sqrt(FX**2 + FY**2)
        f_magnitude_safe = np.where(f_magnitude < 1e-9, 1e-9, f_magnitude)
        theta = np.arctan2(FY, FX)

        # 1. Frequency-dependent term (f/fc)^α - numerically stable version
        if alpha == 0:
            frequency_term = np.ones_like(f_magnitude_safe)
        else:
            normalized_freq = f_magnitude_safe / self.fc
            frequency_term = np.power(normalized_freq, alpha)

        # 2. Position phase term - fixed physical scaling
        # Correct formula: exp(-j*2*pi/c * (FX*x_m + FY*y_m))
        position_phase = -2j * np.pi / C * (FX * x_meters + FY * y_meters)

        # 3. Length/orientation term - fixed physical formula
        length_term = np.ones_like(f_magnitude_safe, dtype=float)
        if length > 1e-6:  # Only compute when L is not zero
            k = 2 * np.pi * f_magnitude_safe / C  # Wave number
            angle_diff = theta - phi_bar
            sinc_arg = k * length * np.sin(angle_diff) / (2 * np.pi)  # Correct sinc parameter
            length_term = np.sinc(sinc_arg)  # np.sinc(x) = sin(pi*x)/(pi*x)

        # Combine frequency domain response
        H_asc = frequency_term * length_term * np.exp(position_phase)

        # IFFT to spatial domain
        atom = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(H_asc)))

        return atom

    def build_high_precision_dictionary(self) -> Tuple[np.ndarray, List[Dict]]:
        """Build high-precision dictionary for maximum accuracy"""
        print(f"📚 Building High-Precision ASC Dictionary...")

        # Frequency sampling
        fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1])

        # High-resolution position sampling
        x_positions = np.linspace(-0.9, 0.9, self.position_samples)
        y_positions = np.linspace(-0.9, 0.9, self.position_samples)

        dictionary_atoms = []
        param_grid = []

        valid_count = 0
        total_count = 0

        print(f"   Alpha values: {len(self.alpha_values)} samples")
        print(f"   Length values: {len(self.length_values)} samples")
        print(f"   Orientation values: {len(self.phi_bar_values)} samples")
        print(f"   Position grid: {self.position_samples}x{self.position_samples}")

        for x in x_positions:
            for y in y_positions:
                for alpha in self.alpha_values:
                    for length in self.length_values:
                        for phi_bar in self.phi_bar_values:
                            total_count += 1

                            atom = self._generate_robust_asc_atom(x, y, alpha, length, phi_bar, fx_range, fy_range)

                            atom_flat = atom.flatten()
                            atom_energy = np.linalg.norm(atom_flat)

                            # Check atom validity
                            if (
                                atom_energy > 1e-12
                                and np.isfinite(atom_energy)
                                and not np.any(np.isnan(atom_flat))
                                and not np.any(np.isinf(atom_flat))
                            ):

                                atom_normalized = atom_flat / atom_energy
                                dictionary_atoms.append(atom_normalized)
                                param_grid.append(
                                    {
                                        "x": x,
                                        "y": y,
                                        "alpha": alpha,
                                        "length": length,
                                        "phi_bar": phi_bar,
                                        "atom_energy": atom_energy,
                                        "scattering_type": self._classify_scattering_type(alpha),
                                    }
                                )
                                valid_count += 1

        dictionary = np.column_stack(dictionary_atoms)

        print(f"   Valid atoms: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
        print(f"   Dictionary size: {dictionary.shape}")
        print(f"   Memory usage: ~{dictionary.nbytes / 1024**2:.1f} MB")

        return dictionary, param_grid

    def _classify_scattering_type(self, alpha: float) -> str:
        """Classify scattering mechanism types"""
        types = {
            -1.0: "Dihedral",
            -0.75: "Edge-Dihedral",
            -0.5: "Edge Diffraction",
            -0.25: "Edge-Surface",
            0.0: "Isotropic",
            0.25: "Surface-Edge",
            0.5: "Surface",
            0.75: "Surface-Specular",
            1.0: "Specular",
        }
        return types.get(alpha, f"Alpha={alpha}")

    def _refine_full_asc_parameters(
        self, initial_params: Dict, target_signal: np.ndarray, initial_coef: complex
    ) -> Dict:
        """
        Advanced parameter refinement for full 6-parameter ASC model
        Optimizes: x, y, alpha, length, phi_bar, amplitude, phase
        """
        print(f"   🔬 Advanced 6-parameter optimization...")

        def objective(params):
            x, y, alpha, length, phi_bar, amp, phase = params

            # Generate atom with current parameters
            try:
                atom = self._generate_robust_asc_atom(x=x, y=y, alpha=alpha, length=length, phi_bar=phi_bar)
                atom_flat = atom.flatten()
                atom_energy = np.linalg.norm(atom_flat)

                if atom_energy < 1e-12:
                    return 1e6  # Penalty for invalid atoms

                atom_normalized = atom_flat / atom_energy
                # Reconstruction
                reconstruction = amp * np.exp(1j * phase) * atom_normalized
                # Error with respect to residual signal
                error = np.linalg.norm(target_signal - reconstruction)
                return error

            except:
                return 1e6  # Penalty for computation errors

        # Initial values and bounds
        x0 = [
            initial_params["x"],
            initial_params["y"],
            initial_params["alpha"],
            initial_params.get("length", 0.0),
            initial_params.get("phi_bar", 0.0),
            np.abs(initial_coef),
            np.angle(initial_coef),
        ]

        bounds = [
            (-1, 1),  # x position
            (-1, 1),  # y position
            (-1, 1),  # alpha
            (0, 3.0),  # length
            (0, np.pi),  # phi_bar
            (0, 10 * np.abs(initial_coef)),  # amplitude
            (-np.pi, np.pi),  # phase
        ]

        # Execute optimization with multiple methods
        refined_params = initial_params.copy()

        try:
            # Method 1: L-BFGS-B (fast, local)
            result1 = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 100})

            # Method 2: Differential Evolution (global, robust)
            result2 = differential_evolution(objective, bounds, maxiter=50, seed=42)

            # Choose best result
            if result2.success and result2.fun < result1.fun:
                best_result = result2
                method_used = "Differential Evolution"
            elif result1.success:
                best_result = result1
                method_used = "L-BFGS-B"
            else:
                best_result = None
                method_used = "None"

            if best_result and best_result.fun < np.linalg.norm(target_signal):
                refined_params.update(
                    {
                        "x": best_result.x[0],
                        "y": best_result.x[1],
                        "alpha": best_result.x[2],
                        "length": best_result.x[3],
                        "phi_bar": best_result.x[4],
                        "estimated_amplitude": best_result.x[5],
                        "estimated_phase": best_result.x[6],
                        "optimization_success": True,
                        "optimization_error": best_result.fun,
                        "optimization_method": method_used,
                        "scattering_type": self._classify_scattering_type(best_result.x[2]),
                    }
                )
                print(f"      ✅ {method_used} optimization successful")
            else:
                # Optimization failed, use initial matching results
                refined_params.update(
                    {
                        "estimated_amplitude": np.abs(initial_coef),
                        "estimated_phase": np.angle(initial_coef),
                        "optimization_success": False,
                        "optimization_method": "Initial Match Only",
                    }
                )
                print(f"      ⚠️ Optimization failed, using initial match")

        except Exception as e:
            print(f"      ⚠️ Optimization exception: {str(e)}")
            refined_params.update(
                {
                    "estimated_amplitude": np.abs(initial_coef),
                    "estimated_phase": np.angle(initial_coef),
                    "optimization_success": False,
                    "optimization_method": "Exception Fallback",
                }
            )

        return refined_params

    def _find_best_match_robust(
        self, signal: np.ndarray, dictionary: np.ndarray
    ) -> Tuple[Optional[int], Optional[complex]]:
        """Robust best match finding"""
        # Check input validity
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return None, None

        signal_real = np.concatenate([signal.real, signal.imag])
        dictionary_real = np.concatenate([dictionary.real, dictionary.imag], axis=0)

        # Check dictionary validity
        if np.any(np.isnan(dictionary_real)) or np.any(np.isinf(dictionary_real)):
            return None, None

        try:
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=1, fit_intercept=False)
            omp.fit(dictionary_real, signal_real)

            nonzero_indices = np.nonzero(omp.coef_)[0]
            if len(nonzero_indices) == 0:
                return None, None

            best_idx = nonzero_indices[0]
            selected_atom = dictionary[:, best_idx]

            # Calculate complex coefficient
            numerator = np.vdot(selected_atom, signal)
            denominator = np.vdot(selected_atom, selected_atom)

            if abs(denominator) < 1e-12:
                return None, None

            complex_coef = numerator / denominator

            return best_idx, complex_coef

        except Exception as e:
            print(f"      ⚠️ OMP matching exception: {str(e)}")
            return None, None

    def _calculate_scatterer_contribution(self, scatterer_params: Dict) -> np.ndarray:
        """Calculate scatterer contribution to signal"""
        fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1])

        atom = self._generate_robust_asc_atom(
            scatterer_params["x"],
            scatterer_params["y"],
            scatterer_params["alpha"],
            scatterer_params.get("length", 0.0),
            scatterer_params.get("phi_bar", 0.0),
            fx_range,
            fy_range,
        )

        atom_flat = atom.flatten()
        atom_energy = np.linalg.norm(atom_flat)

        if atom_energy > 1e-12:
            atom_normalized = atom_flat / atom_energy
            contribution = (
                scatterer_params["estimated_amplitude"]
                * np.exp(1j * scatterer_params["estimated_phase"])
                * atom_normalized
            )
            return contribution
        else:
            return np.zeros_like(atom_flat)

    def extract_high_precision_asc(self, complex_image: np.ndarray) -> List[Dict]:
        """
        High-precision ASC extraction with full 6-parameter optimization
        """
        print(f"🚀 High-Precision ASC Extraction (Full 6-Parameter)")
        print("=" * 60)

        # Data preprocessing
        signal = self.preprocess_data_robust(complex_image)

        # Build high-precision dictionary
        dictionary, param_grid = self.build_high_precision_dictionary()

        # Initialize
        residual_signal = signal.copy()
        extracted_scatterers = []

        initial_energy = np.linalg.norm(residual_signal)
        energy_threshold = initial_energy * self.adaptive_threshold

        print(f"   Initial energy: {initial_energy:.6f}")
        print(f"   Stop threshold: {energy_threshold:.6f}")

        for iteration in range(self.max_iterations):
            current_energy = np.linalg.norm(residual_signal)

            # Stop condition checks
            if current_energy < energy_threshold:
                print(f"   💡 Energy threshold reached, stopping iteration")
                break

            if len(extracted_scatterers) >= self.max_scatterers:
                print(f"   💡 Maximum scatterers reached, stopping iteration")
                break

            # --- 1. Matching ---
            best_idx, initial_coef = self._find_best_match_robust(residual_signal, dictionary)
            if best_idx is None:
                print(f"   💡 No valid match found, stopping iteration")
                break

            initial_params = param_grid[best_idx].copy()

            # --- 2. Full 6-Parameter Optimization ---
            refined_params = self._refine_full_asc_parameters(initial_params, residual_signal, initial_coef)

            # --- 3. Subtraction ---
            contribution = self._calculate_scatterer_contribution(refined_params)

            # Check contribution validity
            contribution_energy = np.linalg.norm(contribution)
            if contribution_energy < current_energy * 0.0001:  # Stricter threshold
                print(f"   💡 Scatterer contribution too small ({contribution_energy:.2e}), stopping")
                break

            new_residual_signal = residual_signal - contribution
            new_energy = np.linalg.norm(new_residual_signal)

            # Energy reduction validation
            if new_energy >= current_energy:
                print(f"   ⚠️ Energy increased ({current_energy:.6f} → {new_energy:.6f}), stopping")
                break

            # Update residual and results
            residual_signal = new_residual_signal
            refined_params["residual_energy"] = new_energy
            extracted_scatterers.append(refined_params)

            # Progress report
            reduction = (current_energy - new_energy) / current_energy
            opt_status = "✅" if refined_params.get("optimization_success", False) else "⚠️"
            method = refined_params.get("optimization_method", "Unknown")
            print(
                f"   Iter {iteration+1}: {opt_status} {refined_params['scattering_type']}, "
                f"A={refined_params['estimated_amplitude']:.3f}, "
                f"L={refined_params.get('length', 0):.3f}, "
                f"Energy↓{reduction:.2%} ({method})"
            )

        # Final analysis
        final_energy = np.linalg.norm(residual_signal)
        total_reduction = (initial_energy - final_energy) / initial_energy

        print(f"\n✅ High-Precision Extraction Complete")
        print(f"   Scatterers extracted: {len(extracted_scatterers)}")
        print(f"   Total energy reduction: {total_reduction:.1%}")

        if extracted_scatterers:
            print(f"\n📊 Result Analysis:")
            type_dist = {}
            opt_success_count = 0
            method_count = {}

            for s in extracted_scatterers:
                stype = s["scattering_type"]
                type_dist[stype] = type_dist.get(stype, 0) + 1
                if s.get("optimization_success", False):
                    opt_success_count += 1
                method = s.get("optimization_method", "Unknown")
                method_count[method] = method_count.get(method, 0) + 1

            print(f"   Scattering type distribution: {type_dist}")
            print(
                f"   Optimization success rate: {opt_success_count}/{len(extracted_scatterers)} ({opt_success_count/len(extracted_scatterers)*100:.1f}%)"
            )
            print(f"   Optimization methods used: {method_count}")

        return extracted_scatterers


def visualize_high_precision_results(complex_image, scatterers, save_path=None):
    """
    Visualize high-precision ASC extraction results (English interface)
    """
    if not scatterers:
        print("⚠️ No scatterers extracted, cannot visualize.")
        return

    magnitude = np.abs(complex_image)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Scatterer positions
    ax1.imshow(magnitude, cmap="gray", origin="lower", extent=(-1, 1, -1, 1), alpha=0.8)

    # Color mapping for scattering types
    alpha_colors = {
        -1.0: "blue",  # Dihedral
        -0.75: "cyan",  # Edge-Dihedral
        -0.5: "green",  # Edge Diffraction
        -0.25: "lime",  # Edge-Surface
        0.0: "yellow",  # Isotropic
        0.25: "orange",  # Surface-Edge
        0.5: "red",  # Surface
        0.75: "magenta",  # Surface-Specular
        1.0: "purple",  # Specular
    }

    alpha_names = {
        -1.0: "Dihedral (α=-1.0)",
        -0.75: "Edge-Dihedral (α=-0.75)",
        -0.5: "Edge Diffraction (α=-0.5)",
        -0.25: "Edge-Surface (α=-0.25)",
        0.0: "Isotropic (α=0.0)",
        0.25: "Surface-Edge (α=0.25)",
        0.5: "Surface (α=0.5)",
        0.75: "Surface-Specular (α=0.75)",
        1.0: "Specular (α=1.0)",
    }

    # Plot scatterers
    plotted_types = set()

    for i, sc in enumerate(scatterers):
        x, y = sc["x"], sc["y"]
        alpha = sc["alpha"]
        amplitude = sc["estimated_amplitude"]
        length = sc.get("length", 0.0)
        opt_success = sc.get("optimization_success", False)

        # Color represents scattering type
        color = alpha_colors.get(alpha, "gray")
        # Size represents amplitude
        base_size = 100
        size = base_size + amplitude * 1000

        # Border represents optimization success
        edge_color = "white" if opt_success else "black"
        edge_width = 2 if opt_success else 1

        scatter = ax1.scatter(x, y, s=size, c=color, alpha=0.8, edgecolors=edge_color, linewidth=edge_width)

        # Add scatterer number
        ax1.annotate(
            f"{i+1}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9, color="white", weight="bold"
        )

        # Draw length orientation if significant
        if length > 0.1:
            phi_bar = sc.get("phi_bar", 0.0)
            dx = 0.1 * np.cos(phi_bar)
            dy = 0.1 * np.sin(phi_bar)
            ax1.arrow(x, y, dx, dy, head_width=0.02, head_length=0.02, fc=color, ec=color, alpha=0.6)

        plotted_types.add(alpha)

    ax1.set_title(f"ASC Scatterer Positions - {len(scatterers)} Extracted", fontsize=14, weight="bold")
    ax1.set_xlabel("X Position (Normalized)", fontsize=12)
    ax1.set_ylabel("Y Position (Normalized)", fontsize=12)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Right plot: Parameter analysis
    if scatterers:
        # Alpha distribution
        alphas = [s["alpha"] for s in scatterers]
        lengths = [s.get("length", 0.0) for s in scatterers]
        amplitudes = [s["estimated_amplitude"] for s in scatterers]

        # Scatter plot: Alpha vs Length, size = Amplitude
        scatter2 = ax2.scatter(
            alphas,
            lengths,
            s=[100 + a * 500 for a in amplitudes],
            c=amplitudes,
            cmap="viridis",
            alpha=0.7,
            edgecolors="black",
        )

        ax2.set_xlabel("Alpha (Scattering Mechanism)", fontsize=12)
        ax2.set_ylabel("Length Parameter", fontsize=12)
        ax2.set_title("ASC Parameter Space Analysis", fontsize=14, weight="bold")
        ax2.grid(True, linestyle="--", alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter2, ax=ax2)
        cbar.set_label("Amplitude", fontsize=10)

        # Add alpha tick labels
        alpha_ticks = sorted(list(set(alphas)))
        ax2.set_xticks(alpha_ticks)
        ax2.set_xticklabels([f"{a:.2f}" for a in alpha_ticks], rotation=45)

    # Create legend for left plot
    legend_elements = []
    for alpha in sorted(plotted_types):
        color = alpha_colors.get(alpha, "gray")
        name = alpha_names.get(alpha, f"α={alpha}")
        legend_elements.append(
            plt.scatter([], [], c=color, s=100, alpha=0.8, edgecolors="white", linewidth=2, label=name)
        )

    # Add optimization status to legend
    legend_elements.append(
        plt.scatter([], [], c="gray", s=100, alpha=0.8, edgecolors="white", linewidth=2, label="Optimized")
    )
    legend_elements.append(
        plt.scatter([], [], c="gray", s=100, alpha=0.8, edgecolors="black", linewidth=1, label="Initial Match")
    )

    ax1.legend(
        handles=legend_elements,
        title="Scattering Types & Status",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
    )

    # Add statistics text
    stats_text = f"Statistics:\n"
    stats_text += f"Total scatterers: {len(scatterers)}\n"
    opt_count = sum(1 for s in scatterers if s.get("optimization_success", False))
    stats_text += f"Optimized: {opt_count}/{len(scatterers)}\n"

    if scatterers:
        avg_length = np.mean([s.get("length", 0.0) for s in scatterers])
        stats_text += f"Avg Length: {avg_length:.3f}\n"
        energy_reduction = scatterers[-1].get("residual_energy", 0)
        stats_text += f"Final Energy: {energy_reduction:.2e}"

    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"🖼️ Visualization saved to: {save_path}")

    plt.show()


def main():
    """High-precision ASC extraction demonstration"""
    print("🔬 High-Precision ASC Scattering Center Extraction")
    print("=" * 60)
    print("Maximum accuracy configuration:")
    print("  ✅ Full 6-parameter ASC model")
    print("  ✅ High-resolution dictionary")
    print("  ✅ Advanced optimization methods")
    print("  ✅ English interface (no display issues)")
    print("=" * 60)

    # Create high-precision extractor
    asc_hp = ASCExtractionHighPrecision(
        extraction_mode="full_asc",
        adaptive_threshold=0.001,  # Very strict for maximum precision
        max_iterations=50,
        max_scatterers=30,
        high_resolution=True,
    )

    print("\n🚀 Usage Instructions:")
    print("1. magnitude, complex_image = asc_hp.load_mstar_data_robust('data.raw')")
    print("2. scatterers = asc_hp.extract_high_precision_asc(complex_image)")
    print("3. visualize_high_precision_results(complex_image, scatterers, 'result.png')")

    return asc_hp


if __name__ == "__main__":
    asc_system = main()
