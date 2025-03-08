import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import logging

# Import from our models module
from models import GenerativeNetwork, ClassifierNetwork, OAMDatasetLoader, setup_logging

# Set up logging
log_filename = f'oam_training_{datetime.now().strftime("%Y%m%d")}.log'
logger = setup_logging(log_filename)


def train_networks(train_loader, val_loader, device='mps', num_epochs=100, model_dir='models',
                   patience=8, continue_training=False):
    """
    Train both the Generative and Classifier networks with validation.
    If continue_training is True, loads existing models and continues training.
    Includes improved training techniques for better accuracy.
    """
    import os
    import torch
    import torch.nn as nn
    from tqdm import tqdm
    from datetime import datetime
    import logging
    import numpy as np

    # Setup logging if not already configured
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

    os.makedirs(model_dir, exist_ok=True)
    logger.info(
        f"{'Continuing' if continue_training else 'Starting'} network training for {num_epochs} epochs on {device}")

    # Get input dimensions from the first data batch
    sample_data, _ = next(iter(train_loader))
    input_size = sample_data.shape[2]  # Assuming square input

    # Initialize networks with correct input size and improved architectures
    gnn = GenerativeNetwork(input_size=input_size, latent_dim=128).to(
        device)  # Increased latent dim
    cnn = ClassifierNetwork(input_size=input_size).to(device)

    # Load existing models if continuing training
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

    # Set up improved optimizers with warmup
    gnn_optimizer = torch.optim.AdamW(
        gnn.parameters(), lr=0.0001, weight_decay=1e-5)  # Lower initial LR for warmup
    cnn_optimizer = torch.optim.AdamW(
        cnn.parameters(), lr=0.0001, weight_decay=1e-5)  # Lower initial LR for warmup

    # Warmup schedule for the first few epochs
    warmup_epochs = 5
    max_lr = 0.001

    def get_lr_for_epoch(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return max_lr * (epoch + 1) / warmup_epochs
        else:
            # After warmup, use cosine schedule
            return max_lr

    # Loss functions with improvements
    mse_loss = nn.MSELoss()

    # Enhanced perceptual loss component for better structural preservation
    def perceptual_loss(output, target):
        # Basic edge detection for structural comparison
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(device)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(device)

        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

        # Apply edge detection
        output_edges_x = torch.nn.functional.conv2d(output, sobel_x, padding=1)
        output_edges_y = torch.nn.functional.conv2d(output, sobel_y, padding=1)
        target_edges_x = torch.nn.functional.conv2d(target, sobel_x, padding=1)
        target_edges_y = torch.nn.functional.conv2d(target, sobel_y, padding=1)

        # Standard edge differences
        edge_diff_x = torch.mean((output_edges_x - target_edges_x) ** 2)
        edge_diff_y = torch.mean((output_edges_y - target_edges_y) ** 2)

        # Add gradient direction consistency for better phase preservation
        # This helps maintain the spiral patterns characteristic of OAM modes
        epsilon = 1e-8  # Prevent division by zero
        output_magnitude = torch.sqrt(
            output_edges_x**2 + output_edges_y**2 + epsilon)
        target_magnitude = torch.sqrt(
            target_edges_x**2 + target_edges_y**2 + epsilon)

        # Normalize gradients by magnitude to get direction vectors
        output_dir_x = output_edges_x / output_magnitude
        output_dir_y = output_edges_y / output_magnitude
        target_dir_x = target_edges_x / target_magnitude
        target_dir_y = target_edges_y / target_magnitude

        # Compute cosine similarity between direction vectors (1 - dot product)
        direction_diff = 1.0 - \
            (output_dir_x * target_dir_x + output_dir_y * target_dir_y)

        # Weight more heavily in regions with strong gradients
        weighted_direction_diff = direction_diff * target_magnitude
        direction_loss = torch.mean(weighted_direction_diff)

        return edge_diff_x + edge_diff_y + direction_loss

    # Use class weights from dataset to address imbalance
    train_dataset = train_loader.dataset
    class_weights = train_dataset.compute_class_weights().to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    logger.info("Using weighted CrossEntropyLoss to address class imbalance")

    # Improved focal loss for better handling of hard examples
    def focal_loss(inputs, targets, alpha=0.25, gamma=2):
        BCE_loss = nn.CrossEntropyLoss(
            reduction='none', weight=class_weights)(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1-pt)**gamma * BCE_loss
        return torch.mean(F_loss)

    # Use OneCycleLR instead of cosine annealing for better convergence
    # Start from a low learning rate, quickly increase, then slowly decrease
    gnn_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        gnn_optimizer, max_lr=max_lr, total_steps=num_epochs *
        len(train_loader),
        pct_start=0.3, div_factor=25, final_div_factor=1000)

    cnn_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        cnn_optimizer, max_lr=max_lr, total_steps=num_epochs *
        len(train_loader),
        pct_start=0.3, div_factor=25, final_div_factor=1000)

    # Tracking metrics
    train_gnn_losses = []
    train_cnn_losses = []
    val_gnn_losses = []
    val_cnn_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0
    best_epoch = 0

    no_improve_count = 0  # For early stopping

    # Create validation image indices for consistent visualization
    vis_indices = np.random.choice(len(val_loader.dataset), min(
        10, len(val_loader.dataset)), replace=False)

    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Update learning rate for warmup
        if epoch < warmup_epochs:
            new_lr = get_lr_for_epoch(epoch)
            for param_group in gnn_optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = new_lr
            logger.info(f"Warmup LR set to {new_lr:.6f}")

        # Training phase
        gnn.train()
        cnn.train()
        train_gnn_loss = 0
        train_cnn_loss = 0
        correct_predictions = 0
        total_samples = 0

        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (data, labels) in enumerate(train_iterator):
            data, labels = data.to(device), labels.to(device)

            # Train GNN (cleaning network)
            gnn_optimizer.zero_grad()
            reconstructed = gnn(data)

            # Combined reconstruction loss (MSE + perceptual)
            recon_loss = mse_loss(reconstructed, data)
            # Adjust perceptual loss weight dynamically
            perceptual_weight = 0.2  # Increased from 0.1
            struct_loss = perceptual_loss(
                reconstructed, data) * perceptual_weight

            # Add L1 loss for sparsity where appropriate (OAM patterns often have sparse features)
            l1_loss = torch.mean(torch.abs(reconstructed))
            sparsity_weight = 0.05

            gnn_loss = recon_loss + struct_loss + sparsity_weight * l1_loss

            gnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=1.0)
            gnn_optimizer.step()

            # Train CNN (classifier network)
            cnn_optimizer.zero_grad()
            # Now feed reconstructed images through classifier
            predictions = cnn(reconstructed.detach())

            # Mix standard cross-entropy and focal loss for robust training
            # Adjust weighting to focus more on focal loss which handles harder examples better
            standard_loss = ce_loss(predictions, labels)
            # Increased gamma for harder examples
            focus_loss = focal_loss(predictions, labels, gamma=2.5)
            cnn_loss = 0.5 * standard_loss + 0.5 * focus_loss  # Equal weighting

            cnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=1.0)
            cnn_optimizer.step()

            # Step the schedulers (if past warmup period)
            if epoch >= warmup_epochs:
                gnn_scheduler.step()
                cnn_scheduler.step()

            train_gnn_loss += gnn_loss.item()
            train_cnn_loss += cnn_loss.item()

            # Calculate accuracy for training
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

        # Call visualization function if it exists
        if 'visualize_gnn_cnn_pipeline' in globals():
            visualize_gnn_cnn_pipeline(gnn, cnn, val_loader, device)

        # Validation phase
        logger.info("Starting validation")
        gnn.eval()
        cnn.eval()
        val_gnn_loss = 0
        val_cnn_loss = 0
        val_correct = 0
        val_total = 0

        # Create confusion matrix for better analysis
        num_classes = cnn.num_classes
        confusion_matrix = torch.zeros(
            num_classes, num_classes, dtype=torch.long)

        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc="Validation")
            for data, labels in val_iterator:
                data, labels = data.to(device), labels.to(device)
                reconstructed = gnn(data)
                predictions = cnn(reconstructed)

                # Calculate validation losses using same metrics as training
                recon_loss = mse_loss(reconstructed, data)
                struct_loss = perceptual_loss(
                    reconstructed, data) * perceptual_weight
                l1_loss = torch.mean(torch.abs(reconstructed))
                val_gnn_loss += (recon_loss + struct_loss +
                                 sparsity_weight * l1_loss).item()

                standard_loss = ce_loss(predictions, labels)
                focus_loss = focal_loss(predictions, labels, gamma=2.5)
                val_cnn_loss += (0.5 * standard_loss + 0.5 * focus_loss).item()

                # Calculate accuracy
                _, predicted = torch.max(predictions.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update confusion matrix
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

        # Calculate per-class accuracy from confusion matrix
        per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        for i, acc in enumerate(per_class_acc):
            if not torch.isnan(acc):
                logger.info(f'Class {i} accuracy: {acc.item()*100:.2f}%')

        # Save best models based on validation accuracy
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

        # Visualize reconstructions periodically
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            if 'visualize_reconstructions' in globals():
                visualize_reconstructions(gnn, val_loader, device, epoch)

    # Save final models
    gnn.save(os.path.join(model_dir, 'final_gnn_model.pth'))
    cnn.save(os.path.join(model_dir, 'final_cnn_model.pth'))
    logger.info("Saved final models")
    print("Saved final models")

    # Plot training history if plotting function exists
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


def visualize_reconstructions(gnn, data_loader, device, epoch, output_dir='reconstructions'):
    """
    Visualize and save sample reconstructions from the GNN.
    The key change here is that we removed the incorrect label offset.
    """
    os.makedirs(output_dir, exist_ok=True)
    gnn.eval()
    with torch.no_grad():
        data, labels = next(iter(data_loader))
        data = data.to(device)
        reconstructed = gnn(data)
        num_samples = min(5, data.size(0))
        fig, axes = plt.subplots(
            num_samples, 2, figsize=(10, 2.5 * num_samples))
        if num_samples == 1:
            axes = np.array([axes])
        for i in range(num_samples):
            # Display the original distorted image with the true label (no offset)
            axes[i, 0].imshow(data[i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 0].set_title(f'Distorted (l={int(labels[i])})')
            axes[i, 0].axis('off')
            # Display the reconstructed (cleaned) image
            axes[i, 1].imshow(
                reconstructed[i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 1].set_title('Reconstructed')
            axes[i, 1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f'epoch_{epoch+1}.png'), dpi=300)
        plt.close()
    logger.info(f"Saved reconstruction samples for epoch {epoch+1}")


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


def load_and_evaluate_models(test_loader, gnn=None, cnn=None, model_dir='models', num_samples=10, output_dir='output', device='mps'):
    """
    Evaluate models on test data.
    If gnn and cnn are provided, uses those models directly.
    Otherwise, loads the models from disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    # If models aren't provided, try to load them from disk
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
            # Modified to use the ClassifierNetwork.load and GenerativeNetwork.load methods
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
    """Visualize the GNN outputs and CNN predictions"""
    gnn.eval()
    cnn.eval()
    with torch.no_grad():
        data, labels = next(iter(data_loader))
        data = data.to(device)
        labels = labels.to(device)

        # Get reconstructions and predictions
        reconstructed = gnn(data)
        predictions = cnn(reconstructed)
        _, predicted_classes = torch.max(predictions, 1)

        # Create visualization
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))
        for i in range(min(num_samples, data.size(0))):
            # Original image
            axes[i, 0].imshow(data[i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 0].set_title(f'Original (Label: {labels[i].item()})')
            axes[i, 0].axis('off')

            # Reconstructed image
            axes[i, 1].imshow(
                reconstructed[i, 0].cpu().numpy(), cmap='viridis')
            axes[i, 1].set_title('GNN Output')
            axes[i, 1].axis('off')

            # Prediction info
            axes[i, 2].axis('off')
            axes[i, 2].text(0.1, 0.5, f'True Label: {labels[i].item()}\nPredicted: {predicted_classes[i].item()}\nConfidence: {torch.softmax(predictions[i], 0)[predicted_classes[i]].item():.2f}',
                            fontsize=12)

        plt.tight_layout()
        plt.savefig(f'pipeline_visualization.png', dpi=300)
        plt.close()

# Add this before training


def main():
    """
    Main function to run the training and evaluation pipeline.
    The workflow:
    1. Load datasets
    2. Train networks (GNN for cleaning, CNN for classification)
    3. Evaluate on test data
    """
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reconstructions', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # Check if MPS is available for M-series Macs
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info(
            "Using MPS (Metal Performance Shaders) for GPU acceleration")
        print("Using Metal Performance Shaders for GPU acceleration on M3 Pro")
    else:
        device = torch.device('cpu')
        logger.info("MPS not available, using CPU instead")
        print("MPS not available, using CPU instead")

    logger.info(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Check for datasets
    dataset_dir = 'datasets'
    if not os.path.exists(dataset_dir):
        logger.error(
            f"Dataset directory '{dataset_dir}' not found. Run data generation script first.")
        print(
            f"Dataset directory '{dataset_dir}' not found. Run data generation script first.")
        return

    # Load datasets
    try:
        logger.info("Loading datasets")
        print("Loading datasets")

        train_dataset = OAMDatasetLoader(
            os.path.join(dataset_dir, 'train_dataset.pkl'))
        val_dataset = OAMDatasetLoader(
            os.path.join(dataset_dir, 'val_dataset.pkl'))
        test_dataset = OAMDatasetLoader(
            os.path.join(dataset_dir, 'test_dataset.pkl'))

        # Optimize worker count for M3 Pro (10 cores)
        num_workers = min(4, os.cpu_count() or 1)  # Use at most 4 workers

        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_size=8,
            num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(
            test_dataset, batch_size=8,
            num_workers=num_workers, pin_memory=True)

        logger.info(
            f"Created dataloaders: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples")
        print(f"Dataloaders created successfully")

        # Train models
        gnn, cnn = train_networks(
            train_loader, val_loader, device, num_epochs=100, patience=8, continue_training=False)

        # Evaluate on test set
        test_accuracy = load_and_evaluate_models(
            test_loader, gnn, cnn, device=device)

        if test_accuracy is not None:
            logger.info(f"Final test accuracy: {test_accuracy:.2f}%")
            print(f"Final test accuracy: {test_accuracy:.2f}%")

        logger.info("Process completed successfully")
        print("Process completed successfully")

    except Exception as e:
        logger.error(
            f"An error occurred during execution: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
