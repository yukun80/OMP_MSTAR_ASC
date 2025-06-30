#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OMP Based Scattering Center Extraction - Final Fixed Version
基于正交匹配追踪的SAR散射中心提取算法 - 最终修复版

Key Fix:
- Consistent amplitude handling
- Unit amplitude dictionary atoms
- Direct coefficient-based amplitude estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from scipy.io import loadmat
import struct
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class OMPASCFinal:
    """
    Final Fixed OMP-based Automatic Scattering Center (ASC) Extraction

    Key improvement: Consistent amplitude handling throughout the pipeline
    """

    def __init__(self, n_scatterers: int = 40, image_size: Tuple[int, int] = (128, 128), use_cv: bool = False):
        self.n_scatterers = n_scatterers
        self.image_size = image_size
        self.use_cv = use_cv

        # SAR system parameters
        self.fc = 1e10  # Center frequency (Hz)
        self.B = 5e8  # Bandwidth (Hz)
        self.omega = 2.86 * np.pi / 180  # Aspect angle (rad)

        # Dictionary parameters
        self.dictionary = None
        self.param_grid = None
        self.omp_model = None

    def load_raw_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load SAR data from .raw file"""
        try:
            with open(filepath, "rb") as f:
                data = np.frombuffer(f.read(), dtype=">f4")

            n_pixels = self.image_size[0] * self.image_size[1]
            if len(data) != 2 * n_pixels:
                raise ValueError(f"Data size mismatch. Expected {2*n_pixels}, got {len(data)}")

            magnitude = data[:n_pixels].reshape(self.image_size)
            phase = data[n_pixels:].reshape(self.image_size)
            complex_image = magnitude * np.exp(1j * phase)

            print(f"Successfully loaded SAR data: {filepath}")
            print(f"Image size: {complex_image.shape}")
            print(f"Magnitude range: [{magnitude.min():.3f}, {magnitude.max():.3f}]")

            return magnitude, complex_image

        except Exception as e:
            raise RuntimeError(f"Failed to load {filepath}: {str(e)}")

    def preprocess_data(self, complex_image: np.ndarray) -> np.ndarray:
        """Preprocess SAR complex image"""
        vectorized_signal = complex_image.flatten()

        print(f"Preprocessed signal shape: {vectorized_signal.shape}")
        print(f"Signal energy: {np.linalg.norm(vectorized_signal):.3f}")

        return vectorized_signal

    def build_dictionary(self, position_grid_size: int = 32, phase_levels: int = 8) -> Tuple[np.ndarray, List[Dict]]:
        """
        Build SAR dictionary with UNIT AMPLITUDE atoms for consistent scaling

        Key fix: All atoms generated with amplitude=1.0, scaling handled by OMP coefficients
        """
        print("Building final SAR dictionary with unit amplitude atoms...")

        M, N = self.image_size

        # Position grid
        x_positions = np.linspace(-1, 1, position_grid_size)
        y_positions = np.linspace(-1, 1, position_grid_size)

        # Phase levels
        phases = np.linspace(0, 2 * np.pi, phase_levels, endpoint=False)

        # Frequency domain grid
        fx_range = np.linspace(-self.B / 2, self.B / 2, M)
        fy_range = np.linspace(-self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), N)

        dictionary_atoms = []
        param_grid = []

        atom_count = 0
        total_atoms = len(x_positions) * len(y_positions) * len(phases)

        for x in x_positions:
            for y in y_positions:
                for phase in phases:
                    # KEY FIX: Use unit amplitude (1.0) for all atoms
                    atom = self._generate_scatterer_atom(x, y, 1.0, fx_range, fy_range, phase)

                    # Normalize atom to unit energy
                    atom_energy = np.linalg.norm(atom.flatten())
                    if atom_energy > 1e-10:
                        atom_normalized = atom / atom_energy
                    else:
                        atom_normalized = atom

                    dictionary_atoms.append(atom_normalized.flatten())
                    param_grid.append(
                        {
                            "x": x,
                            "y": y,
                            "phase": phase,
                            "atom_index": atom_count,
                            "unit_amplitude": 1.0,  # All atoms have unit amplitude
                        }
                    )

                    atom_count += 1
                    if atom_count % 1000 == 0:
                        print(f"Generated {atom_count}/{total_atoms} atoms...")

        dictionary = np.column_stack(dictionary_atoms)
        self.dictionary = dictionary
        self.param_grid = param_grid

        print(f"Dictionary built successfully:")
        print(f"Dictionary shape: {dictionary.shape}")
        print(f"Number of atoms: {len(param_grid)}")

        return dictionary, param_grid

    def _generate_scatterer_atom(
        self,
        x: float,
        y: float,
        amplitude: float,
        fx_range: np.ndarray,
        fy_range: np.ndarray,
        scatterer_phase: float = 0.0,
    ) -> np.ndarray:
        """Generate SAR scatterer atom"""
        M, N = self.image_size

        # Convert to actual positions
        scene_size = 30.0  # meters
        x_actual = x * scene_size / 2
        y_actual = y * scene_size / 2

        # Frequency domain response
        freq_response = np.zeros((M, N), dtype=complex)

        for i, fx in enumerate(fx_range):
            for j, fy in enumerate(fy_range):
                position_phase = -2j * np.pi * (fx * x_actual + fy * y_actual) / 3e8
                total_phase = position_phase + 1j * scatterer_phase
                freq_response[i, j] = amplitude * np.exp(total_phase)

        # Transform to spatial domain
        atom = np.fft.ifft2(np.fft.ifftshift(freq_response))
        return atom

    def extract_scatterers(self, signal: np.ndarray, dictionary: Optional[np.ndarray] = None) -> Dict:
        """Extract scatterers with corrected amplitude estimation"""
        if dictionary is None:
            if self.dictionary is None:
                raise ValueError("Dictionary not built. Call build_dictionary() first.")
            dictionary = self.dictionary

        print("Extracting scattering centers with corrected amplitude handling...")

        # Prepare real-valued data for sklearn
        signal_real = np.concatenate([signal.real, signal.imag])
        dict_real = np.concatenate([dictionary.real, dictionary.imag], axis=0)

        # Initialize OMP model
        if self.use_cv:
            self.omp_model = OrthogonalMatchingPursuitCV(cv=5, max_iter=self.n_scatterers)
            print("Using cross-validation for automatic sparsity selection...")
        else:
            self.omp_model = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_scatterers, fit_intercept=False)
            print(f"Using fixed sparsity: {self.n_scatterers} scatterers")

        # Fit OMP model
        self.omp_model.fit(dict_real, signal_real)
        coefficients = self.omp_model.coef_

        # Find non-zero coefficients
        nonzero_indices = np.nonzero(coefficients)[0]
        nonzero_coefs = coefficients[nonzero_indices]

        print(f"Extracted {len(nonzero_indices)} scattering centers")
        print(f"Reconstruction error: {np.linalg.norm(signal_real - dict_real @ coefficients):.3f}")

        # Extract scatterer parameters with CORRECTED amplitude estimation
        scatterers = []
        for idx, coef in zip(nonzero_indices, nonzero_coefs):
            param = self.param_grid[idx].copy()

            # KEY FIX: Direct amplitude from OMP coefficient magnitude
            # Since dictionary atoms are unit amplitude and normalized,
            # the coefficient magnitude directly represents the amplitude
            estimated_amplitude = abs(coef)
            estimated_phase = np.angle(coef)

            param["coefficient"] = coef
            param["estimated_amplitude"] = estimated_amplitude
            param["estimated_phase"] = estimated_phase
            scatterers.append(param)

        # Sort by amplitude
        scatterers.sort(key=lambda x: x["estimated_amplitude"], reverse=True)

        results = {
            "scatterers": scatterers,
            "coefficients": coefficients,
            "nonzero_indices": nonzero_indices,
            "reconstruction_error": np.linalg.norm(signal_real - dict_real @ coefficients),
            "model": self.omp_model,
        }

        return results

    def reconstruct_image(self, scatterers: List[Dict]) -> np.ndarray:
        """Reconstruct SAR image with corrected amplitude scaling"""
        print("Reconstructing image with corrected amplitude scaling...")

        reconstructed = np.zeros(self.image_size, dtype=complex)

        for scatterer in scatterers:
            x = scatterer["x"]
            y = scatterer["y"]
            amp = scatterer["estimated_amplitude"]
            phase = scatterer["estimated_phase"]

            # Generate scatterer contribution
            fx_range = np.linspace(-self.B / 2, self.B / 2, self.image_size[0])
            fy_range = np.linspace(
                -self.fc * np.sin(self.omega / 2), self.fc * np.sin(self.omega / 2), self.image_size[1]
            )

            # CRITICAL FIX: Generate unit atom and normalize it (consistent with dictionary)
            atom = self._generate_scatterer_atom(x, y, 1.0, fx_range, fy_range, 0.0)
            atom_energy = np.linalg.norm(atom.flatten())
            if atom_energy > 1e-10:
                atom_normalized = atom / atom_energy
            else:
                atom_normalized = atom

            # Apply amplitude and phase to normalized atom
            scaled_atom = amp * np.exp(1j * phase) * atom_normalized
            reconstructed += scaled_atom

        print(f"Reconstruction completed. Energy: {np.linalg.norm(reconstructed):.3f}")
        return reconstructed

    def visualize_results(
        self,
        original_magnitude: np.ndarray,
        reconstructed: np.ndarray,
        scatterers: List[Dict],
        save_path: Optional[str] = None,
    ):
        """Visualize results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original magnitude
        im1 = axes[0, 0].imshow(original_magnitude, cmap="gray")
        axes[0, 0].set_title("Original SAR Magnitude")
        axes[0, 0].axis("off")
        plt.colorbar(im1, ax=axes[0, 0])

        # Reconstructed magnitude
        im2 = axes[0, 1].imshow(np.abs(reconstructed), cmap="gray")
        axes[0, 1].set_title("Reconstructed Magnitude")
        axes[0, 1].axis("off")
        plt.colorbar(im2, ax=axes[0, 1])

        # Difference
        diff = original_magnitude - np.abs(reconstructed)
        im3 = axes[0, 2].imshow(diff, cmap="seismic")
        axes[0, 2].set_title("Difference")
        axes[0, 2].axis("off")
        plt.colorbar(im3, ax=axes[0, 2])

        # Scatterer positions
        x_pos = [s["x"] for s in scatterers]
        y_pos = [s["y"] for s in scatterers]
        amplitudes = [s["estimated_amplitude"] for s in scatterers]

        scatter = axes[1, 0].scatter(x_pos, y_pos, c=amplitudes, s=100, cmap="viridis")
        axes[1, 0].set_title(f"Scatterer Positions ({len(scatterers)} centers)")
        axes[1, 0].set_xlabel("X Position (normalized)")
        axes[1, 0].set_ylabel("Y Position (normalized)")
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label="Amplitude")

        # Amplitude histogram
        axes[1, 1].hist(amplitudes, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 1].set_title("Amplitude Distribution")
        axes[1, 1].set_xlabel("Amplitude")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].grid(True, alpha=0.3)

        # Phase histogram
        phases = [s["estimated_phase"] for s in scatterers]
        axes[1, 2].hist(phases, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 2].set_title("Phase Distribution")
        axes[1, 2].set_xlabel("Phase (radians)")
        axes[1, 2].set_ylabel("Count")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Results saved to: {save_path}")

        plt.show()


def main():
    """Main execution function"""
    print("=== Final OMP-based SAR Scattering Center Extraction ===")

    omp_asc = OMPASCFinal(n_scatterers=40, use_cv=False)

    # Example workflow
    raw_file_path = "path/to/your/data.128x128.raw"

    try:
        magnitude, complex_image = omp_asc.load_raw_data(raw_file_path)
        signal = omp_asc.preprocess_data(complex_image)

        dictionary, param_grid = omp_asc.build_dictionary(position_grid_size=16, phase_levels=4)

        results = omp_asc.extract_scatterers(signal)
        reconstructed = omp_asc.reconstruct_image(results["scatterers"])

        omp_asc.visualize_results(
            magnitude, reconstructed, results["scatterers"], save_path="final_omp_asc_results.png"
        )

        print(f"\n=== Final Extraction Summary ===")
        print(f"Number of scatterers extracted: {len(results['scatterers'])}")
        print(f"Reconstruction error: {results['reconstruction_error']:.3f}")
        print(f"Top 5 strongest scatterers:")
        for i, scatterer in enumerate(results["scatterers"][:5]):
            print(
                f"  {i+1}. Position: ({scatterer['x']:.3f}, {scatterer['y']:.3f}), "
                f"Amplitude: {scatterer['estimated_amplitude']:.3f}"
            )

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your data file path and format.")


if __name__ == "__main__":
    main()
