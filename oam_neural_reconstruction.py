import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'oam_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OAMGenerator:
    """Generates Orbital Angular Momentum (OAM) modes with various distortions."""

    def __init__(self, grid_size=128, r_max=1.0):
        self.grid_size = grid_size
        self.r_max = r_max
        self.setup_coordinate_grid()
        logger.info(f"Initialized OAM Generator with grid size {grid_size}")
        print(f"Initialized OAM Generator")

    def setup_coordinate_grid(self):
        """Initialize coordinate grids for calculations."""
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
        R_norm = self.R / np.max(self.R)
        amplitude = (R_norm/w0) * np.exp(-R_norm**2/(2*w0**2))
        phase = l * self.Phi
        amplitude = amplitude / np.max(np.abs(amplitude))
        logger.info(f"Generated OAM mode with l={l}, w0={w0}")
        return amplitude * np.exp(1j * phase)


class TurbulenceSimulator:
    """Simulates atmospheric turbulence effects using Kolmogorov spectrum with von Karman modifications."""

    def __init__(self, grid_size, r0, L0=1.0, l0=0.001, wavelength=1.0e-6, propagation_distance=1000):
        self.grid_size = grid_size
        self.r0 = r0
        self.L0 = L0
        self.l0 = l0
        self.wavelength = wavelength
        self.propagation_distance = propagation_distance
        self.setup_frequency_grid()
        logger.info(
            f"Initialized Turbulence Simulator with r0={r0}, L0={L0}, l0={l0}, wavelength={wavelength}m, distance={propagation_distance}m")
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
        self.K[self.K == 0] = 1e-10
        logger.info("Frequency grid setup complete")

    def generate_phase_screen(self):
        """Generate random phase screen using von Karman spectrum"""
        k0 = 2*np.pi/self.L0
        km = 5.92/self.l0
        spectrum = 0.023 * \
            self.r0**(-5/3) * (self.K**2 + k0**2)**(-11/6) * \
            np.exp(-(self.K**2)/(km**2))
        spectrum[self.K == 0] = 0
        random_phase = np.random.normal(size=(self.grid_size, self.grid_size)) + \
            1j * np.random.normal(size=(self.grid_size, self.grid_size))
        phase_screen = np.fft.ifft2(np.sqrt(spectrum) * random_phase)
        phase_screen = np.real(phase_screen)
        scale_factor = 4.0
        normalized_screen = phase_screen * scale_factor
        logger.info(
            f"Generated phase screen with min={normalized_screen.min():.2f}, max={normalized_screen.max():.2f}, std={np.std(normalized_screen):.2f}")
        return normalized_screen


class GenerativeNetwork(nn.Module):
    """Generative Neural Network for reconstructing distorted OAM modes."""

    def __init__(self, input_size=128, latent_dim=64):
        super().__init__()
        logger.info(
            f"Initializing GNN with input_size={input_size}, latent_dim={latent_dim}")
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (input_size//8) * (input_size//8), latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * (input_size//8) * (input_size//8)),
            nn.ReLU(),
            nn.Unflatten(1, (128, input_size//8, input_size//8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


class ClassifierNetwork(nn.Module):
    """CNN for classifying OAM modes."""

    def __init__(self, input_size=128, num_classes=21):
        super().__init__()
        logger.info(
            f"Initializing CNN with input_size={input_size}, num_classes={num_classes}")
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        feature_size = 64 * (input_size//8) * (input_size//8)
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class OAMDataset(Dataset):
    """Dataset class for OAM modes with distortions."""

    def __init__(self, num_samples, grid_size=128, max_l=10, save_samples=False, dataset_type="train"):
        self.generator = OAMGenerator(grid_size)
        self.turbulence = TurbulenceSimulator(grid_size, r0=0.1)
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
        """Generate dataset of OAM modes with stronger distortions."""
        os.makedirs('sample_images', exist_ok=True)
        for i in range(self.num_samples):
            l = np.random.randint(-self.max_l, self.max_l + 1)
            clean_mode = self.generator.generate_oam_mode(l)
            phase_screen = self.turbulence.generate_phase_screen()
            distorted_mode = clean_mode * np.exp(1j * phase_screen)
            amplitude = np.abs(distorted_mode)
            amplitude = (amplitude - np.min(amplitude)) / \
                (np.max(amplitude) - np.min(amplitude))
            phase = np.angle(distorted_mode)
            phase_stats = f"Phase min={np.min(phase):.2f}, max={np.max(phase):.2f}, std={np.std(phase):.2f}"
            logger.info(f"Sample {i}, l={l}: {phase_stats}")
            self.data.append(amplitude)
            self.labels.append(l + self.max_l)
            if self.save_samples and i < 5:
                self.clean_modes.append(np.abs(clean_mode))
                self.distorted_modes.append(amplitude)
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                im0 = axes[0, 0].imshow(np.abs(clean_mode), cmap='viridis')
                axes[0, 0].set_title(f'Clean OAM Mode l={l} (Amplitude)')
                axes[0, 0].axis('off')
                fig.colorbar(im0, ax=axes[0, 0])
                im1 = axes[0, 1].imshow(np.angle(clean_mode), cmap='hsv')
                axes[0, 1].set_title(f'Clean OAM Mode l={l} (Phase)')
                axes[0, 1].axis('off')
                fig.colorbar(im1, ax=axes[0, 1])
                im2 = axes[1, 0].imshow(amplitude, cmap='viridis')
                axes[1, 0].set_title('Distorted Mode (Amplitude)')
                axes[1, 0].axis('off')
                fig.colorbar(im2, ax=axes[1, 0])
                im3 = axes[1, 1].imshow(phase, cmap='hsv')
                axes[1, 1].set_title('Distorted Mode (Phase)')
                axes[1, 1].axis('off')
                fig.colorbar(im3, ax=axes[1, 1])
                plt.tight_layout()
                plt.savefig(
                    f'sample_images/{self.dataset_type}_sample_{i}_l_{l}.png', dpi=150)
                plt.close()
        logger.info(
            f"Generated {len(self.data)} samples with l values from {-self.max_l} to {self.max_l}")
        print(f"Dataset generation complete")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx][None, :, :]), self.labels[idx]


def train_networks(train_loader, val_loader, device='cuda', num_epochs=4):
    """Train both GNN and CNN networks with validation."""
    logger.info(
        f"Starting network training for {num_epochs} epochs on {device}")
    print(f"Starting network training for {num_epochs} epochs")
    gnn = GenerativeNetwork().to(device)
    cnn = ClassifierNetwork().to(device)
    gnn_optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    gnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gnn_optimizer, 'min', patience=5)
    cnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        cnn_optimizer, 'min', patience=5)
    train_gnn_losses = []
    train_cnn_losses = []
    val_gnn_losses = []
    val_cnn_losses = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        gnn.train()
        cnn.train()
        train_gnn_loss = 0
        train_cnn_loss = 0
        correct_predictions = 0
        total_samples = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            if batch_idx % 20 == 0:
                logger.info(f"Training batch {batch_idx}/{len(train_loader)}")
            data, labels = data.to(device), labels.to(device)
            gnn_optimizer.zero_grad()
            reconstructed = gnn(data)
            gnn_loss = mse_loss(reconstructed, data)
            gnn_loss.backward()
            gnn_optimizer.step()
            cnn_optimizer.zero_grad()
            predictions = cnn(reconstructed.detach())
            cnn_loss = ce_loss(predictions, labels)
            cnn_loss.backward()
            cnn_optimizer.step()
            train_gnn_loss += gnn_loss.item()
            train_cnn_loss += cnn_loss.item()
            _, predicted = torch.max(predictions.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        train_accuracy = 100 * correct_predictions / total_samples
        avg_train_gnn_loss = train_gnn_loss / len(train_loader)
        avg_train_cnn_loss = train_cnn_loss / len(train_loader)
        train_gnn_losses.append(avg_train_gnn_loss)
        train_cnn_losses.append(avg_train_cnn_loss)
        logger.info(
            f'Train - GNN Loss: {avg_train_gnn_loss:.4f}, CNN Loss: {avg_train_cnn_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Train Accuracy: {train_accuracy:.2f}%')
        logger.info("Starting validation")
        gnn.eval()
        cnn.eval()
        val_gnn_loss = 0
        val_cnn_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                reconstructed = gnn(data)
                predictions = cnn(reconstructed)
                val_gnn_loss += mse_loss(reconstructed, data).item()
                val_cnn_loss += ce_loss(predictions, labels).item()
                _, predicted = torch.max(predictions.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * val_correct / val_total
        avg_val_gnn_loss = val_gnn_loss / len(val_loader)
        avg_val_cnn_loss = val_cnn_loss / len(val_loader)
        val_gnn_losses.append(avg_val_gnn_loss)
        val_cnn_losses.append(avg_val_cnn_loss)
        gnn_scheduler.step(avg_val_gnn_loss)
        cnn_scheduler.step(avg_val_cnn_loss)
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        logger.info(
            f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds')
        logger.info(
            f'Val - GNN Loss: {avg_val_gnn_loss:.4f}, CNN Loss: {avg_val_cnn_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        if avg_val_gnn_loss + avg_val_cnn_loss < best_val_loss:
            best_val_loss = avg_val_gnn_loss + avg_val_cnn_loss
            torch.save(gnn.state_dict(), 'best_gnn_model.pth')
            torch.save(cnn.state_dict(), 'best_cnn_model.pth')
            logger.info("Saved new best models")
            print("Saved new best models")
        if epoch % 1 == 0:
            visualize_reconstructions(gnn, val_loader, device, epoch)
    plot_training_history(train_gnn_losses, val_gnn_losses,
                          'GNN Loss', 'gnn_loss_history.png')
    plot_training_history(train_cnn_losses, val_cnn_losses,
                          'CNN Loss', 'cnn_loss_history.png')
    logger.info("Training completed")
    print("Training completed")
    return gnn, cnn


def visualize_reconstructions(gnn, data_loader, device, epoch):
    """Visualize and save sample reconstructions."""
    os.makedirs('reconstructions', exist_ok=True)
    gnn.eval()
    with torch.no_grad():
        data, labels = next(iter(data_loader))
        data = data.to(device)
        reconstructed = gnn(data)
        num_samples = min(5, data.size(0))
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2.5*num_samples))
        for i in range(num_samples):
            axes[i, 0].imshow(data[i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 0].set_title(f'Distorted (l={labels[i]-10})')
            axes[i, 0].axis('off')
            axes[i, 1].imshow(
                reconstructed[i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 1].set_title('Reconstructed')
            axes[i, 1].axis('off')
        plt.tight_layout()
        plt.savefig(f'reconstructions/reconstruction_epoch_{epoch+1}.png')
        plt.close()
        logger.info(f"Saved reconstruction samples for epoch {epoch+1}")


def plot_training_history(train_loss, val_loss, title, filename):
    """Plot and save training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved {title} history plot to {filename}")


def load_and_visualize_models(test_dataset, num_samples=5):
    """Load trained models and visualize results."""
    logger.info("Loading trained models for visualization")
    print("Loading trained models for visualization")
    if not os.path.exists('best_gnn_model.pth') or not os.path.exists('best_cnn_model.pth'):
        logger.warning("Model files not found. Run training first.")
        print("Model files not found. Run training first.")
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn = GenerativeNetwork().to(device)
    cnn = ClassifierNetwork().to(device)
    gnn.load_state_dict(torch.load('best_gnn_model.pth', map_location=device))
    cnn.load_state_dict(torch.load('best_cnn_model.pth', map_location=device))
    gnn.eval()
    cnn.eval()
    test_loader = DataLoader(
        test_dataset, batch_size=num_samples, shuffle=True)
    data, labels = next(iter(test_loader))
    data = data.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        reconstructed = gnn(data)
        predictions = cnn(reconstructed)
        _, predicted_classes = torch.max(predictions, 1)
        true_l_values = [l.item() - 10 for l in labels]
        predicted_l_values = [l.item() - 10 for l in predicted_classes]
        for i in range(len(true_l_values)):
            logger.info(f"Sample {i}: True l={true_l_values[i]}, Predicted l={predicted_l_values[i]}, "
                        f"{'Correct' if predicted_l_values[i] == true_l_values[i] else 'Wrong'}")
        correct = sum(1 for tl, pl in zip(
            true_l_values, predicted_l_values) if tl == pl)
        accuracy = 100 * correct / len(true_l_values)
        logger.info(
            f"Test accuracy on {len(true_l_values)} samples: {accuracy:.2f}%")
        print(f"Test accuracy: {accuracy:.2f}%")
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3*num_samples))
        for i in range(num_samples):
            axes[i, 0].imshow(data[i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 0].set_title(f'Distorted (True l={true_l_values[i]})')
            axes[i, 0].axis('off')
            axes[i, 1].imshow(
                reconstructed[i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 1].set_title('Reconstructed')
            axes[i, 1].axis('off')
            text = f"Predicted l={predicted_l_values[i]}\n"
            text += f"{'✓' if predicted_l_values[i] == true_l_values[i] else '✗'}"
            axes[i, 2].text(0.5, 0.5, text, ha='center',
                            va='center', fontsize=12)
            axes[i, 2].axis('off')
        plt.tight_layout()
        plt.savefig('test_results.png')
        plt.close()
        logger.info("Saved test results visualization to test_results.png")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    print(f"Using device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)
    logger.info("Creating datasets")
    print("Creating datasets")
    train_dataset = OAMDataset(
        num_samples=5000, save_samples=True, dataset_type="train")
    val_dataset = OAMDataset(num_samples=500, dataset_type="val")
    test_dataset = OAMDataset(num_samples=100, dataset_type="test")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    logger.info(
        f"Created datasets: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples")
    print(f"Datasets created")
    gnn, cnn = train_networks(train_loader, val_loader, device, num_epochs=4)
    torch.save(gnn.state_dict(), 'final_gnn_model.pth')
    torch.save(cnn.state_dict(), 'final_cnn_model.pth')
    logger.info("Saved final models")
    print("Saved final models")
    load_and_visualize_models(test_dataset)
    logger.info("Process completed successfully")
    print("Process completed successfully")


if __name__ == "__main__":
    main()
