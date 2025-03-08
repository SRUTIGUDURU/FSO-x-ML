import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import torch
from torch.utils.data import Dataset
import pickle
from matplotlib.colors import Normalize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'oam_generation_{datetime.now().strftime("%Y%m%d")}.log')]
)
logger = logging.getLogger(__name__)


class OAMGenerator:
    """Generates Orbital Angular Momentum (OAM) modes with various distortions."""

    def __init__(self, grid_size=256, r_max=1.0):
        self.grid_size = grid_size
        self.r_max = r_max
        self.setup_coordinate_grid()
        logger.info(f"Initialized OAM Generator with grid size {grid_size}")
        print(f"Initialized OAM Generator")

    def setup_coordinate_grid(self):
        """Initialize coordinate grids for calculations."""
        # Use higher resolution to avoid grid artifacts
        x = np.linspace(-1, 1, self.grid_size) * self.r_max
        y = np.linspace(-1, 1, self.grid_size) * self.r_max
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Phi = np.arctan2(self.Y, self.X)
        logger.info("Coordinate grid setup complete")

    def generate_oam_mode(self, l, w0=0.45):
        """
        Generate OAM mode with topological charge l.
        Args:
            l (int): Topological charge
            w0 (float): Beam waist parameter
        Returns:
            np.ndarray: Complex-valued OAM mode field
        """
        # Smooth profile to avoid grid artifacts
        R_norm = self.R / np.max(self.R)

        # Smooth amplitude profile with better transition
        amplitude = (R_norm/w0) * np.exp(-R_norm**2/(2*w0**2))

        # Apply anti-aliasing by smoothing the edges
        mask = np.ones_like(amplitude)
        mask[R_norm > 0.95] = 0
        edge_region = (R_norm > 0.85) & (R_norm <= 0.95)
        mask[edge_region] = (0.95 - R_norm[edge_region]) / 0.1

        amplitude = amplitude * mask
        phase = l * self.Phi
        amplitude = amplitude / np.max(np.abs(amplitude))
        logger.info(f"Generated OAM mode with l={l}, w0={w0}")
        return amplitude * np.exp(1j * phase)


class TurbulenceSimulator:
    """Simulates atmospheric turbulence effects using Kolmogorov spectrum with von Karman modifications."""

    def __init__(self, grid_size, r0, L0=200.0, l0=0.001, wavelength=1.0e-6, propagation_distance=1000, turbulence_strength=1.5, r_max=1.0):
        self.grid_size = grid_size
        self.r0 = r0
        self.L0 = L0
        self.l0 = l0
        self.wavelength = wavelength
        self.propagation_distance = propagation_distance
        self.turbulence_strength = turbulence_strength
        self.r_max = r_max  # Added parameter for angular spectrum method
        self.setup_frequency_grid()
        logger.info(
            f"Initialized Turbulence Simulator with r0={r0}, L0={L0}, l0={l0}, wavelength={wavelength}m, distance={propagation_distance}m, strength={turbulence_strength}")
        print(f"Initialized Turbulence Simulator")
        self.C_n2 = 0.423 * (self.wavelength**2) / (self.r0**(5/3))
        self.effective_r0 = (0.423 * (2*np.pi/self.wavelength)
                             ** 2 * self.C_n2 * self.propagation_distance)**(-3/5)
        logger.info(f"Effective r0: {self.effective_r0:.6f} m")

    def setup_frequency_grid(self):
        """Setup spatial frequency grid."""
        delta_k = 1.0 / self.grid_size
        k = np.fft.fftfreq(self.grid_size, delta_k)
        self.Kx, self.Ky = np.meshgrid(k, k)
        self.K = np.sqrt(self.Kx**2 + self.Ky**2)
        # Avoid division by zero but use a smaller value
        self.K[self.K == 0] = 1e-12
        logger.info("Frequency grid setup complete")

    def generate_phase_screen(self):
        """Generate random phase screen using von Karman spectrum with enhanced turbulence"""
        k0 = 2*np.pi/self.L0
        km = 5.92/self.l0

        # Enhanced spectrum for stronger turbulence effects
        spectrum = 0.023 * \
            self.r0**(-5/3) * (self.K**2 + k0**2)**(-11/6) * \
            np.exp(-(self.K**2)/(km**2))
        spectrum[self.K == 0] = 0

        # Generate random phase with proper scaling
        random_phase = np.random.normal(size=(self.grid_size, self.grid_size)) + \
            1j * np.random.normal(size=(self.grid_size, self.grid_size))

        # Apply FFT-based phase screen generation
        phase_screen = np.fft.ifft2(np.sqrt(spectrum) * random_phase)
        phase_screen = np.real(phase_screen)

        # Apply additional high-frequency components to create more detail
        high_freq = np.random.normal(0, 0.5, (self.grid_size, self.grid_size))
        high_freq = np.fft.fft2(high_freq)

        # Filter to keep only high frequencies
        high_mask = np.ones_like(self.K)
        high_mask[self.K < 0.3 * np.max(self.K)] = 0
        high_freq = high_freq * high_mask
        high_freq_phase = np.real(np.fft.ifft2(high_freq)) * 0.3

        # Combine with main phase screen for enhanced detail
        phase_screen = phase_screen + high_freq_phase
        normalized_screen = phase_screen * self.turbulence_strength

        logger.info(
            f"Generated phase screen with min={normalized_screen.min():.2f}, max={normalized_screen.max():.2f}, std={np.std(normalized_screen):.2f}")
        return normalized_screen

    def angular_spectrum_propagation(self, field, z, wavelength=None):
        """
        Propagate a complex field using the Angular Spectrum method.

        Args:
            field (np.ndarray): Complex field to propagate
            z (float): Propagation distance in meters
            wavelength (float, optional): Wavelength in meters. Defaults to self.wavelength.

        Returns:
            np.ndarray: Propagated complex field
        """
        if wavelength is None:
            wavelength = self.wavelength

        # Get dimensions
        Ny, Nx = field.shape

        # Calculate pixel size (assuming square grid)
        dx = 2.0 * self.r_max / self.grid_size
        dy = dx

        # Spatial frequencies
        fx = np.fft.fftfreq(Nx, dx)
        fy = np.fft.fftfreq(Ny, dy)
        FX, FY = np.meshgrid(fx, fy)

        # Calculate transfer function
        # Handle evanescent waves by setting sqrt argument to 0 when negative
        sqrt_arg = (2*np.pi/wavelength)**2 - (FX**2 + FY**2)
        sqrt_arg[sqrt_arg < 0] = 0
        H = np.exp(1j * 2 * np.pi * z * np.sqrt(sqrt_arg))

        # Fourier transform, multiply by transfer function, and inverse transform
        F_field = np.fft.fft2(field)
        F_propagated = F_field * H
        propagated_field = np.fft.ifft2(F_propagated)

        return propagated_field


class OAMDataset(Dataset):
    """Dataset class for OAM modes with distortions."""

    def __init__(self, num_samples, grid_size=256, max_l=10, save_samples=False, dataset_type="train", turbulence_strength=1.5):
        self.generator = OAMGenerator(grid_size)
        self.turbulence = TurbulenceSimulator(
            grid_size, r0=0.05, turbulence_strength=turbulence_strength, r_max=self.generator.r_max)
        self.num_samples = num_samples
        self.max_l = max_l
        self.data = []
        self.labels = []
        self.clean_modes = []
        self.distorted_modes = []
        self.save_samples = save_samples
        self.dataset_type = dataset_type
        logger.info(
            f"Creating {dataset_type} dataset with {num_samples} samples")
        print(f"Creating {dataset_type} dataset")
        self.generate_dataset()

    def generate_dataset(self):
        """Generate dataset of OAM modes with propagation and turbulence effects."""
        os.makedirs('samples', exist_ok=True)

        # Define propagation distances (in meters)
        z1 = 1.0  # Distance from OAM generation to turbulence phase screen
        z2 = 200.0  # Distance from turbulence phase screen to receiver

        for i in range(self.num_samples):
            l = np.random.randint(-self.max_l, self.max_l + 1)
            clean_mode = self.generator.generate_oam_mode(l)

            # First propagation (from OAM generation to turbulence screen)
            field_at_screen = self.turbulence.angular_spectrum_propagation(
                clean_mode, z1)

            # Apply turbulence phase screen
            phase_screen = self.turbulence.generate_phase_screen()
            field_after_screen = field_at_screen * np.exp(1j * phase_screen)

            # Second propagation (from turbulence screen to receiver)
            final_field = self.turbulence.angular_spectrum_propagation(
                field_after_screen, z2)

            # Calculate intensity (squared magnitude)
            clean_intensity = np.abs(clean_mode)**2
            distorted_intensity = np.abs(final_field)**2

            # Normalize intensities for visualization and model input
            clean_amplitude = np.sqrt(clean_intensity)
            clean_amplitude = (clean_amplitude - np.min(clean_amplitude)) / \
                (np.max(clean_amplitude) - np.min(clean_amplitude))

            distorted_amplitude = np.sqrt(distorted_intensity)
            distorted_amplitude = (distorted_amplitude - np.min(distorted_amplitude)) / \
                (np.max(distorted_amplitude) - np.min(distorted_amplitude))

            # Store phases for visualization
            clean_phase = np.angle(clean_mode)
            distorted_phase = np.angle(final_field)

            phase_stats = f"Phase min={np.min(distorted_phase):.2f}, max={np.max(distorted_phase):.2f}, std={np.std(distorted_phase):.2f}"
            logger.info(f"Sample {i}, l={l}: {phase_stats}")

            self.data.append(distorted_amplitude)
            self.labels.append(l + self.max_l)

            # Rest of the visualization code remains the same
            if self.save_samples and i < 10:
                self.clean_modes.append(clean_amplitude)
                self.distorted_modes.append(distorted_amplitude)

                # Create side-by-side visualization with shared colorbars
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))

                # Improved visualization with better titles and color normalization
                norm_amp = Normalize(vmin=0, vmax=1)

                # Clean mode
                im0 = axes[0, 0].imshow(
                    clean_amplitude, cmap='viridis', norm=norm_amp)
                axes[0, 0].set_title(
                    f'Clean OAM Mode l={l} (Amplitude)', fontsize=12)
                axes[0, 0].axis('off')

                im1 = axes[0, 1].imshow(
                    clean_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
                axes[0, 1].set_title(
                    f'Clean OAM Mode l={l} (Phase)', fontsize=12)
                axes[0, 1].axis('off')

                # Add phase screen visualization
                im_ps = axes[0, 2].imshow(
                    phase_screen, cmap='RdBu', vmin=-np.pi*2, vmax=np.pi*2)
                axes[0, 2].set_title('Turbulence Phase Screen', fontsize=12)
                axes[0, 2].axis('off')

                # Distorted mode
                im2 = axes[1, 0].imshow(
                    distorted_amplitude, cmap='viridis', norm=norm_amp)
                axes[1, 0].set_title('Turbulent Mode (Amplitude)', fontsize=12)
                axes[1, 0].axis('off')

                im3 = axes[1, 1].imshow(
                    distorted_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
                axes[1, 1].set_title('Turbulent Mode (Phase)', fontsize=12)
                axes[1, 1].axis('off')

                # Add difference visualization
                diff = distorted_amplitude - clean_amplitude
                im_diff = axes[1, 2].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
                axes[1, 2].set_title('Amplitude Difference', fontsize=12)
                axes[1, 2].axis('off')

                # Add colorbars with better positioning
                cbar_ax0 = fig.add_axes([0.02, 0.25, 0.01, 0.5])
                cbar0 = fig.colorbar(im0, cax=cbar_ax0)
                cbar0.set_label('Amplitude', fontsize=10)

                cbar_ax1 = fig.add_axes([0.33, 0.25, 0.01, 0.5])
                cbar1 = fig.colorbar(im1, cax=cbar_ax1)
                cbar1.set_label('Phase (rad)', fontsize=10)

                cbar_ax2 = fig.add_axes([0.64, 0.25, 0.01, 0.5])
                cbar2 = fig.colorbar(im_ps, cax=cbar_ax2)
                cbar2.set_label('Phase Distortion (rad)', fontsize=10)

                fig = plt.gcf()  # Get current figure

                fig.suptitle(
                    f'OAM Mode l={l} - Clean vs Turbulent Comparison', fontsize=14)

                plt.subplots_adjust(left=0.05, right=0.95, top=0.90,
                                    bottom=0.05, wspace=0.2, hspace=0.2)
                plt.subplots_adjust(wspace=0.05, hspace=0.1)

                plt.savefig(
                    f'samples/{self.dataset_type}_sample_{i}_l_{l}.png', dpi=200, bbox_inches='tight')
                plt.close()

        logger.info(
            f"Generated {len(self.data)} samples with l values from {-self.max_l} to {self.max_l}")
        print(f"Dataset generation complete")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx][None, :, :]), self.labels[idx]

    def save_to_file(self, filename):
        """Save the dataset to a file using pickle."""
        data_dict = {
            'data': self.data,
            'labels': self.labels,
            'max_l': self.max_l,
            'dataset_type': self.dataset_type
        }
        with open(filename, 'wb') as f:
            pickle.dump(data_dict, f)
        logger.info(f"Dataset saved to {filename}")
        print(f"Dataset saved to {filename}")


def main():
    """Generate and save datasets for training, validation and testing."""
    np.random.seed(42)
    os.makedirs('datasets', exist_ok=True)

    # Generate training dataset with increased turbulence strength
    train_dataset = OAMDataset(
        num_samples=500, save_samples=True, dataset_type="train",
        grid_size=256, turbulence_strength=2.0)
    train_dataset.save_to_file('datasets/train_dataset.pkl')

    # Generate validation dataset
    val_dataset = OAMDataset(
        num_samples=100, dataset_type="val",
        grid_size=256, turbulence_strength=2.0)
    val_dataset.save_to_file('datasets/val_dataset.pkl')

    # Generate test dataset
    test_dataset = OAMDataset(
        num_samples=50, dataset_type="test",
        grid_size=256, turbulence_strength=2.0)
    test_dataset.save_to_file('datasets/test_dataset.pkl')

    logger.info("All datasets generated and saved successfully")
    print("All datasets generated and saved successfully")


if __name__ == "__main__":
    main()
