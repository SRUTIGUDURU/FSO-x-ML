import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging
import pickle


class EfficientAttention(nn.Module):
    """Memory-efficient attention module with improved resolution preservation"""

    def __init__(self, in_channels):
        super().__init__()

        self.query = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        spatial_size = max(width, height)
        scale_factor = min(1.0, 32.0 / spatial_size)

        if scale_factor < 1.0:

            x_sampled = F.interpolate(
                x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        else:
            x_sampled = x

        _, _, s_width, s_height = x_sampled.size()

        q = self.query(x_sampled)
        k = self.key(x_sampled)
        v = self.value(x_sampled)

        q_flat = q.view(batch_size, -1, s_width * s_height).permute(0, 2, 1)
        k_flat = k.view(batch_size, -1, s_width * s_height)
        v_flat = v.view(batch_size, -1, s_width * s_height)

        attention = F.softmax(torch.bmm(q_flat, k_flat) /
                              (k.size(1) ** 0.5), dim=2)

        out_flat = torch.bmm(v_flat, attention.permute(0, 2, 1))
        out = out_flat.view(batch_size, C, s_width, s_height)

        if scale_factor < 1.0:
            out = F.interpolate(out, size=(width, height),
                                mode='bilinear', align_corners=False)

        return self.gamma * out + x


class ConvBlock(nn.Module):
    """Enhanced convolutional block with improved regularization"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_residual=True, dropout_rate=0.1):
        super().__init__()
        self.use_residual = use_residual and (
            in_channels == out_channels) and (stride == 1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class GenerativeNetwork(nn.Module):
    """
    Enhanced autoencoder for OAM plot processing with improved capacity
    """

    def __init__(self, input_size=256, latent_dim=256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.initial_downsample = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2)
        )

        self.encoder = nn.Sequential(
            ConvBlock(16, 32, use_residual=False, dropout_rate=0.1),
            nn.AvgPool2d(2),

            ConvBlock(32, 64, use_residual=False,
                      dropout_rate=0.1),
            EfficientAttention(64),
            nn.AvgPool2d(2)
        )

        reduced_size = input_size // 16
        self.bottleneck_size = min(reduced_size, 16)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (self.bottleneck_size, self.bottleneck_size))

        bottleneck_features = 64 * self.bottleneck_size * self.bottleneck_size

        bottleneck_hidden = bottleneck_features // 4

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_features, bottleneck_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(bottleneck_hidden),
            nn.Dropout(0.2),
            nn.Linear(bottleneck_hidden, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, bottleneck_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(bottleneck_hidden),
            nn.Dropout(0.2),
            nn.Linear(bottleneck_hidden, bottleneck_features),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder_reshape = nn.Unflatten(
            1, (64, self.bottleneck_size, self.bottleneck_size))

        self.num_upsample_steps = 4

        target_size = self.input_size
        size_after_initial_upsampling = self.bottleneck_size * \
            (2 ** (self.num_upsample_steps - 1))
        self.final_size = (target_size, target_size)

        self.decoder = nn.Sequential(
            ConvBlock(64, 64, use_residual=True, dropout_rate=0.1),
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=False),

            ConvBlock(64, 32, use_residual=False, dropout_rate=0.1),
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=False),

            ConvBlock(32, 16, use_residual=False, dropout_rate=0.1),
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=False),

            ConvBlock(16, 16, use_residual=True, dropout_rate=0.1),
            nn.Upsample(size=self.final_size,
                        mode='bilinear', align_corners=False),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        original_size = (x.size(2), x.size(3))

        x = self.initial_downsample(x)

        encoded = self.encoder(x)

        pooled = self.adaptive_pool(encoded)

        flattened = pooled.view(pooled.size(0), -1)
        bottleneck = self.bottleneck(flattened)

        reshaped = self.decoder_reshape(bottleneck)

        decoded = self.decoder(reshaped)

        if decoded.size(2) != original_size[0] or decoded.size(3) != original_size[1]:
            decoded = F.interpolate(
                decoded, size=original_size, mode='bilinear', align_corners=False)

        return decoded

    def save(self, path):
        """Save model with metadata"""
        state = {
            'state_dict': self.state_dict(),
            'input_size': self.input_size,
            'latent_dim': self.latent_dim
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path, device=None):
        """Load model with correct initialization parameters"""
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        state = torch.load(path, map_location=device)
        model = cls(input_size=state['input_size'],
                    latent_dim=state['latent_dim'])
        model.load_state_dict(state['state_dict'])
        model.to(device)
        return model


class ClassifierNetwork(nn.Module):
    """
    Enhanced classifier for identifying 'l' values in OAM plots.
    Increased capacity and feature preservation.
    """

    def __init__(self, input_size=64, num_classes=21):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),

            ConvBlock(64, 128, use_residual=False,
                      dropout_rate=0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),


            ConvBlock(128, 256, use_residual=False, dropout_rate=0.2),
            EfficientAttention(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        pooled = self.global_pool(features)
        output = self.classifier(pooled)
        return output

    def save(self, path):
        """Save model with metadata"""
        state = {
            'state_dict': self.state_dict(),
            'input_size': self.input_size,
            'num_classes': self.num_classes
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path, device=None):
        """Load model with correct initialization parameters"""
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        state = torch.load(path, map_location=device)
        model = cls(input_size=state['input_size'],
                    num_classes=state['num_classes'])
        model.load_state_dict(state['state_dict'])
        model.to(device)
        return model


class OAMDatasetLoader(Dataset):
    """
    Dataset loader for pre-generated OAM datasets.  Loads distorted OAM images
    and their corresponding 'l' value labels.  Includes class weight computation
    and data augmentation.
    """

    def __init__(self, file_path, apply_augmentation=False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        self.apply_augmentation = apply_augmentation

        try:
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)

            required_keys = ['data', 'labels', 'max_l', 'dataset_type']
            for key in required_keys:
                if key not in data_dict:
                    raise KeyError(f"Missing required key in dataset: {key}")

            self.data = data_dict['data']
            self.labels = data_dict['labels']
            self.max_l = data_dict['max_l']
            self.dataset_type = data_dict['dataset_type']

            unique_labels = np.unique(self.labels)
            self.label_mapping = {label: i for i,
                                  label in enumerate(unique_labels)}
            self.labels = [self.label_mapping[label] for label in self.labels]

            if len(self.data) != len(self.labels):
                raise ValueError(
                    f"Data length ({len(self.data)}) doesn't match labels length ({len(self.labels)})")

            print(
                f"Loaded {self.dataset_type} dataset with {len(self.data)} samples")
        except Exception as e:
            raise RuntimeError(
                f"Error loading dataset from {file_path}: {str(e)}")

    def compute_class_weights(self):
        """
        Compute weights for each class to address class imbalance.  Returns a
        tensor of weights where less frequent classes have higher weights.
        """
        labels_array = np.array(self.labels)
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))

        total_samples = len(self.labels)
        max_count = max(class_counts.values())

        weights = {label: max_count / count for label,
                   count in class_counts.items()}

        weight_sum = sum(weights.values())
        for label in weights:
            weights[label] *= len(class_counts) / weight_sum

        num_classes = len(unique_labels)
        weight_tensor = torch.ones(num_classes)

        for label, weight in weights.items():

            weight_tensor[label] = weight

        return weight_tensor

    def apply_noise_augmentation(self, image):
        """Apply random noise augmentation to the image"""

        noise_level = np.random.uniform(0.0, 0.05)
        noise = np.random.normal(0, noise_level, image.shape)
        augmented = image + noise

        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            augmented = np.rot90(augmented, k=k)

        augmented = np.clip(augmented, 0, 1)

        return augmented

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        if self.apply_augmentation:
            image = self.apply_noise_augmentation(image)

        label = self.labels[idx]

        return torch.FloatTensor(image[None, :, :]), label


def setup_logging(log_filename=None):
    """Sets up logging configuration"""
    if log_filename is None:
        from datetime import datetime
        log_filename = f'oam_model_{datetime.now().strftime("%Y%m%d")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename)]
    )
    return logging.getLogger(__name__)


log_filename = f'oam_training_{datetime.now().strftime("%Y%m%d")}.log'
logger = setup_logging(log_filename)


def train_networks(train_loader, val_loader, device='cuda', num_epochs=100, model_dir='models',
                   patience=8, continue_training=False):
    """
    Train both the Generative and Classifier networks with validation.
    If continue_training is True, loads existing models and continues training.
    Includes improved training techniques for better accuracy.
    """
    os.makedirs(model_dir, exist_ok=True)
    logger.info(
        f"{'Continuing' if continue_training else 'Starting'} network training for {num_epochs} epochs on {device}")

    sample_data, _ = next(iter(train_loader))
    input_size = sample_data.shape[2]

    gnn = GenerativeNetwork(input_size=input_size, latent_dim=128).to(
        device)

    num_classes = len(torch.unique(torch.tensor(train_loader.dataset.labels)))
    cnn = ClassifierNetwork(input_size=input_size,
                            num_classes=num_classes).to(device)

    if continue_training:
        gnn_path = os.path.join(model_dir, 'final_gnn_model.pth')
        cnn_path = os.path.join(model_dir, 'final_cnn_model.pth')

        if os.path.exists(gnn_path) and os.path.exists(cnn_path):
            gnn = GenerativeNetwork.load(gnn_path, device)
            cnn = ClassifierNetwork.load(cnn_path, device)
            logger.info("Loaded existing models to continue training")
            print("Loaded existing models to continue training")
        else:
            logger.warning(
                "Cannot find existing models, starting from scratch instead")
            print("Cannot find existing models, starting from scratch instead")

    gnn_optimizer = torch.optim.AdamW(
        gnn.parameters(), lr=0.0001, weight_decay=1e-5)
    cnn_optimizer = torch.optim.AdamW(
        cnn.parameters(), lr=0.0001, weight_decay=1e-5)

    warmup_epochs = 5
    max_lr = 0.001

    def get_lr_for_epoch(epoch):
        if epoch < warmup_epochs:

            return max_lr * (epoch + 1) / warmup_epochs
        else:

            return max_lr

    mse_loss = nn.MSELoss()

    def perceptual_loss(output, target):

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(device)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(device)

        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

        output_edges_x = torch.nn.functional.conv2d(output, sobel_x, padding=1)
        output_edges_y = torch.nn.functional.conv2d(output, sobel_y, padding=1)
        target_edges_x = torch.nn.functional.conv2d(target, sobel_x, padding=1)
        target_edges_y = torch.nn.functional.conv2d(target, sobel_y, padding=1)

        edge_diff_x = torch.mean((output_edges_x - target_edges_x) ** 2)
        edge_diff_y = torch.mean((output_edges_y - target_edges_y) ** 2)

        epsilon = 1e-8
        output_magnitude = torch.sqrt(
            output_edges_x**2 + output_edges_y**2 + epsilon)
        target_magnitude = torch.sqrt(
            target_edges_x**2 + target_edges_y**2 + epsilon)

        output_dir_x = output_edges_x / output_magnitude
        output_dir_y = output_edges_y / output_magnitude
        target_dir_x = target_edges_x / target_magnitude
        target_dir_y = target_edges_y / target_magnitude

        direction_diff = 1.0 - \
            (output_dir_x * target_dir_x + output_dir_y * target_dir_y)

        weighted_direction_diff = direction_diff * target_magnitude
        direction_loss = torch.mean(weighted_direction_diff)

        return edge_diff_x + edge_diff_y + direction_loss

    class_weights = train_loader.dataset.compute_class_weights().to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    logger.info("Using weighted CrossEntropyLoss to address class imbalance")

    def focal_loss(inputs, targets, alpha=0.25, gamma=2):
        BCE_loss = nn.CrossEntropyLoss(
            reduction='none', weight=class_weights)(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1-pt)**gamma * BCE_loss
        return torch.mean(F_loss)

    gnn_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        gnn_optimizer, max_lr=max_lr, total_steps=num_epochs *
        len(train_loader),
        pct_start=0.3, div_factor=25, final_div_factor=1000)

    cnn_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        cnn_optimizer, max_lr=max_lr, total_steps=num_epochs *
        len(train_loader),
        pct_start=0.3, div_factor=25, final_div_factor=1000)

    train_gnn_losses = []
    train_cnn_losses = []
    val_gnn_losses = []
    val_cnn_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0
    best_epoch = 0

    no_improve_count = 0

    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        print(f"Epoch {epoch+1}/{num_epochs}")

        if epoch < warmup_epochs:
            new_lr = get_lr_for_epoch(epoch)
            for param_group in gnn_optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = new_lr
            logger.info(f"Warmup LR set to {new_lr:.6f}")

        gnn.train()
        cnn.train()
        train_gnn_loss = 0
        train_cnn_loss = 0
        correct_predictions = 0
        total_samples = 0

        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (data, labels) in enumerate(train_iterator):
            data, labels = data.to(device), labels.to(device)

            gnn_optimizer.zero_grad()
            reconstructed = gnn(data)

            recon_loss = mse_loss(reconstructed, data)

            perceptual_weight = 0.2
            struct_loss = perceptual_loss(
                reconstructed, data) * perceptual_weight

            l1_loss = torch.mean(torch.abs(reconstructed))
            sparsity_weight = 0.05

            gnn_loss = recon_loss + struct_loss + sparsity_weight * l1_loss

            gnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=1.0)
            gnn_optimizer.step()

            cnn_optimizer.zero_grad()

            predictions = cnn(reconstructed.detach())

            standard_loss = ce_loss(predictions, labels)

            focus_loss = focal_loss(predictions, labels, gamma=2.5)
            cnn_loss = 0.5 * standard_loss + 0.5 * focus_loss

            cnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=1.0)
            cnn_optimizer.step()

            if epoch >= warmup_epochs:
                gnn_scheduler.step()
                cnn_scheduler.step()

            train_gnn_loss += gnn_loss.item()
            train_cnn_loss += cnn_loss.item()

            _, predicted = torch.max(predictions.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            train_iterator.set_postfix({
                'GNN Loss': f"{gnn_loss.item():.4f}",
                'CNN Loss': f"{cnn_loss.item():.4f}",
                'Acc': f"{100.0 * correct_predictions / total_samples:.2f}%"
            })

        train_accuracy = 100 * correct_predictions / total_samples
        avg_train_gnn_loss = train_gnn_loss / len(train_loader)
        avg_train_cnn_loss = train_cnn_loss / len(train_loader)
        train_gnn_losses.append(avg_train_gnn_loss)
        train_cnn_losses.append(avg_train_cnn_loss)
        train_accuracies.append(train_accuracy)

        logger.info(
            f'Train - GNN Loss: {avg_train_gnn_loss:.4f}, CNN Loss: {avg_train_cnn_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Train Accuracy: {train_accuracy:.2f}%')

        if 'visualize_gnn_cnn_pipeline' in globals():
            visualize_gnn_cnn_pipeline(gnn, cnn, val_loader, device)

        logger.info("Starting validation")
        gnn.eval()
        cnn.eval()
        val_gnn_loss = 0
        val_cnn_loss = 0
        val_correct = 0
        val_total = 0

        num_classes = cnn.num_classes
        confusion_matrix = torch.zeros(
            num_classes, num_classes, dtype=torch.long)

        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc="Validation")
            for data, labels in val_iterator:
                data, labels = data.to(device), labels.to(device)
                reconstructed = gnn(data)
                predictions = cnn(reconstructed)

                recon_loss = mse_loss(reconstructed, data)
                struct_loss = perceptual_loss(
                    reconstructed, data) * perceptual_weight
                l1_loss = torch.mean(torch.abs(reconstructed))
                val_gnn_loss += (recon_loss + struct_loss +
                                 sparsity_weight * l1_loss).item()

                standard_loss = ce_loss(predictions, labels)
                focus_loss = focal_loss(predictions, labels, gamma=2.5)
                val_cnn_loss += (0.5 * standard_loss + 0.5 * focus_loss).item()

                _, predicted = torch.max(predictions.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                val_iterator.set_postfix({
                    'Acc': f"{100.0 * val_correct / val_total:.2f}%"
                })

        val_accuracy = 100 * val_correct / val_total
        avg_val_gnn_loss = val_gnn_loss / len(val_loader)
        avg_val_cnn_loss = val_cnn_loss / len(val_loader)
        val_gnn_losses.append(avg_val_gnn_loss)
        val_cnn_losses.append(avg_val_cnn_loss)
        val_accuracies.append(val_accuracy)

        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        logger.info(
            f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds')
        logger.info(
            f'Val - GNN Loss: {avg_val_gnn_loss:.4f}, CNN Loss: {avg_val_cnn_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')

        per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        for i, acc in enumerate(per_class_acc):
            if not torch.isnan(acc):
                logger.info(f'Class {i} accuracy: {acc.item()*100:.2f}%')

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch
            no_improve_count = 0
            gnn.save(os.path.join(model_dir, 'best_gnn_model.pth'))
            cnn.save(os.path.join(model_dir, 'best_cnn_model.pth'))
            logger.info(
                f"Saved new best models with accuracy {best_val_acc:.2f}%")
            print(f"Saved new best models with accuracy {best_val_acc:.2f}%")
        else:
            no_improve_count += 1
            logger.info(f"No improvement for {no_improve_count} epochs")

        if no_improve_count >= patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            print(f"Early stopping after {epoch+1} epochs")
            break

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            if 'visualize_reconstructions' in globals():
                visualize_reconstructions(gnn, val_loader, device, epoch)

    gnn.save(os.path.join(model_dir, 'final_gnn_model.pth'))
    cnn.save(os.path.join(model_dir, 'final_cnn_model.pth'))
    logger.info("Saved final models")
    print("Saved final models")

    if 'plot_training_history' in globals():
        plot_training_history(train_gnn_losses, val_gnn_losses, 'GNN Loss', os.path.join(
            model_dir, 'gnn_loss_history.png'))
        plot_training_history(train_cnn_losses, val_cnn_losses, 'CNN Loss', os.path.join(
            model_dir, 'cnn_loss_history.png'))
        plot_training_history(train_accuracies, val_accuracies, 'Accuracy (%)', os.path.join(
            model_dir, 'accuracy_history.png'))

    logger.info(
        f"Training completed. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
    print(
        f"Training completed. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")

    return gnn, cnn


def add_black_quadrants(image):
    """Add black quadrants to the image for visualization"""
    h, w = image.shape
    image_copy = image.copy()
    # Make top-left quadrant black
    image_copy[:h//2, :w//2] = 0
    # Make bottom-right quadrant black
    image_copy[h//2:, w//2:] = 0
    return image_copy

def visualize_reconstructions(gnn, data_loader, device, epoch, output_dir='reconstructions'):
    """
    Visualize and save sample reconstructions from the GNN with black quadrants.
    """
    os.makedirs(output_dir, exist_ok=True)
    gnn.eval()
    with torch.no_grad():
        data, labels = next(iter(data_loader))
        data = data.to(device)
        reconstructed = gnn(data)
        num_samples = min(5, data.size(0))
        fig, axes = plt.subplots(
            num_samples, 3, figsize=(15, 2.5 * num_samples))
        if num_samples == 1:
            axes = np.array([axes])
        for i in range(num_samples):
            orig_img = data[i, 0].cpu().numpy()
            recon_img = reconstructed[i, 0].cpu().numpy()

            # Original image
            axes[i, 0].imshow(orig_img, cmap='viridis')
            axes[i, 0].set_title(f'Original (l={int(labels[i])})')
            axes[i, 0].axis('off')

            # Original with black quadrants
            axes[i, 1].imshow(add_black_quadrants(orig_img), cmap='viridis')
            axes[i, 1].set_title('With Black Quadrants')
            axes[i, 1].axis('off')

            # Reconstructed from black quadrants
            axes[i, 2].imshow(recon_img, cmap='viridis')
            axes[i, 2].set_title('Reconstructed')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f'epoch_{epoch+1}.png'), dpi=300)
        plt.savefig(os.path.join(
            output_dir, f'epoch_{epoch+1}_black_quadrants.png'), dpi=300)
        plt.close()
    logger.info(f"Saved reconstruction samples with black quadrants for epoch {epoch+1}")


def plot_training_history(train_values, val_values, title, filename):
    """
    Plot and save training history.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_values, label='Training')
    plt.plot(val_values, label='Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()
    logger.info(f"Saved {title} history plot to {filename}")


def load_and_evaluate_models(test_loader, gnn=None, cnn=None, model_dir='models', num_samples=10, output_dir='output', device='cuda'):
    """
    Evaluate models on test data.
    If gnn and cnn are provided, uses those models directly.
    Otherwise, loads the models from disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    if gnn is None or cnn is None:
        logger.info("Loading trained models for evaluation")
        print("Loading trained models for evaluation")

        gnn_path = os.path.join(model_dir, 'best_gnn_model.pth')
        cnn_path = os.path.join(model_dir, 'best_cnn_model.pth')
        if not os.path.exists(gnn_path) or not os.path.exists(cnn_path):
            logger.warning("Model files not found. Run training first.")
            print("Model files not found. Run training first.")
            return None

        try:

            gnn = GenerativeNetwork.load(gnn_path, device)
            cnn = ClassifierNetwork.load(cnn_path, device)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return None
    else:
        logger.info("Using provided models for evaluation")
        print("Using provided models for evaluation")

    gnn.eval()
    cnn.eval()
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    test_gnn_loss = 0
    test_cnn_loss = 0
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        test_iterator = tqdm(test_loader, desc="Testing")
        for data, labels in test_iterator:
            data, labels = data.to(device), labels.to(device)
            reconstructed = gnn(data)
            predictions = cnn(reconstructed)
            _, predicted_classes = torch.max(predictions, 1)
            test_gnn_loss += mse_loss(reconstructed, data).item()
            test_cnn_loss += ce_loss(predictions, labels).item()
            total_samples += labels.size(0)
            total_correct += (predicted_classes == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_classes.cpu().numpy())
            test_iterator.set_postfix({
                'Acc': f"{100.0 * total_correct / total_samples:.2f}%"
            })

    test_accuracy = 100.0 * total_correct / total_samples
    avg_test_gnn_loss = test_gnn_loss / len(test_loader)
    avg_test_cnn_loss = test_cnn_loss / len(test_loader)

    logger.info(
        f"Test Results - GNN Loss: {avg_test_gnn_loss:.4f}, CNN Loss: {avg_test_cnn_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    print(f"Test Results - Accuracy: {test_accuracy:.2f}%")

    return test_accuracy


def visualize_gnn_cnn_pipeline(gnn, cnn, data_loader, device, num_samples=5):
    """Visualize the GNN outputs and CNN predictions with black quadrants"""
    gnn.eval()
    cnn.eval()
    with torch.no_grad():
        data, labels = next(iter(data_loader))
        data = data.to(device)
        labels = labels.to(device)

        # Create data with black quadrants
        data_with_black = data.clone()
        for i in range(data.size(0)):
            data_with_black[i, 0] = torch.from_numpy(add_black_quadrants(data[i, 0].cpu().numpy())).to(device)

        reconstructed = gnn(data_with_black)
        predictions = cnn(reconstructed)
        _, predicted_classes = torch.max(predictions, 1)

        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 3 * num_samples))
        if num_samples == 1:
            axes = np.array([[axes[0], axes[1], axes[2], axes[3]]])

        for i in range(min(num_samples, data.size(0))):
            # Original
            axes[i, 0].imshow(data[i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 0].set_title(f'Original (Label: {labels[i].item()})')
            axes[i, 0].axis('off')

            # With black quadrants
            axes[i, 1].imshow(data_with_black[i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 1].set_title('With Black Quadrants')
            axes[i, 1].axis('off')

            # GNN reconstruction
            axes[i, 2].imshow(reconstructed[i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 2].set_title('GNN Output')
            axes[i, 2].axis('off')

            # Prediction info
            axes[i, 3].axis('off')
            axes[i, 3].text(0.1, 0.5, f'True Label: {labels[i].item()}\nPredicted: {predicted_classes[i].item()}\nConfidence: {torch.softmax(predictions[i], 0)[predicted_classes[i]].item():.2f}',
                            fontsize=12)

        plt.tight_layout()
        plt.savefig(f'pipeline_visualization_black_quadrants.png', dpi=300)
        plt.close()


def main():
    """
    Main function to run the training and evaluation pipeline.
    The workflow:
    1. Load datasets
    2. Train networks (GNN for cleaning, CNN for classification)
    3. Evaluate on test data
    """

    os.makedirs('models', exist_ok=True)
    os.makedirs('reconstructions', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        logger.info("Using CUDA for GPU acceleration")
        print("Using CUDA for GPU acceleration")
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available, using CPU instead")
        print("CUDA not available, using CPU instead")

    logger.info(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    dataset_dir = 'datasets'
    if not os.path.exists(dataset_dir):
        logger.error(
            f"Dataset directory '{dataset_dir}' not found. Run data generation script first.")
        print(
            f"Dataset directory '{dataset_dir}' not found. Run data generation script first.")
        return

    try:
        logger.info("Loading datasets")
        print("Loading datasets")

        train_dataset = OAMDatasetLoader(

            os.path.join(dataset_dir, 'train_dataset.pkl'), apply_augmentation=True)
        val_dataset = OAMDatasetLoader(
            os.path.join(dataset_dir, 'val_dataset.pkl'))
        test_dataset = OAMDatasetLoader(
            os.path.join(dataset_dir, 'test_dataset.pkl'))
    except Exception as e:

        logger.exception("Error loading datasets:")
        return

    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(
        val_dataset, batch_size=64,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(
        test_dataset, batch_size=64,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2)

    logger.info(
        f"Created dataloaders: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples")
    print(f"Dataloaders created successfully")

    gnn, cnn = train_networks(
        train_loader, val_loader, device, num_epochs=100, patience=4, continue_training=False)

    test_accuracy = load_and_evaluate_models(
        test_loader, gnn, cnn, device=device)

    if test_accuracy is not None:
        logger.info(f"Final test accuracy: {test_accuracy:.2f}%")
        print(f"Final test accuracy: {test_accuracy:.2f}%")

    logger.info("Process completed successfully")
    print("Process completed successfully")


if __name__ == "__main__":
    main()
