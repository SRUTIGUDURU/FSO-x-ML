import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging
import pickle
import numpy as np


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
