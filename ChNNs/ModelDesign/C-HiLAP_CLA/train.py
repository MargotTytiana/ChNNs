import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import json
import logging

# Import custom modules
# 导入自定义模块
from data_loader import create_dataloaders
from speaker_model import (
    ChaoticSpeakerRecognitionSystem,
    SpeakerRecognitionLoss,
    evaluate_speaker_recognition,
    evaluate_speaker_verification,
    plot_embeddings_2d
)
from config import get_config, save_config
from utils import save_dict_to_json


def setup_logging(log_dir: str) -> logging.Logger:
    """
    Set up logging configuration and return logger instance

    设置日志配置并返回日志实例
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"training_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # Configure logging
    # 配置日志
    logger = logging.getLogger("speaker_recognition_trainer")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def save_checkpoint(
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_dir: str,
        is_best: bool = False
) -> None:
    """
    Save model checkpoint with comprehensive metrics

    保存包含全面指标的模型检查点
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    # Save latest checkpoint (overwrite)
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)

    # If this is the best model, save it separately
    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)

    logging.info(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
        checkpoint_path: str
) -> Tuple[nn.Module, Optional[optim.Optimizer], int, Dict[str, float]]:
    """
    Load model checkpoint with metrics

    加载包含指标的模型检查点
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})

    logging.info(f"Loaded checkpoint from epoch {epoch} with metrics: {metrics}")

    return model, optimizer, epoch, metrics


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        epoch: int,
        log_interval: int = 10
) -> Tuple[float, float]:
    """
    Train for one epoch with enhanced progress tracking

    训练一个轮次，增强进度跟踪
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        features = batch['audio'].to(device)
        labels = batch['label'].to(device)

        # Debug info for first batch of first epoch
        if epoch == 0 and batch_idx == 0:
            logging.debug(f"Input audio shape: {features.shape}")
            logging.debug(f"Input audio range: {features.min().item():.4f} to {features.max().item():.4f}")
            logging.debug(f"Input audio mean: {features.mean().item():.4f}, std: {features.std().item():.4f}")

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, labels=labels, mode='identification')

        # Calculate loss
        losses = criterion(outputs, labels)
        loss = losses['total_loss']

        # Debug info for first batch
        if epoch == 0 and batch_idx == 0:
            logging.debug("\n=== DEBUG INFO (First Batch) ===")
            logging.debug(f"Output logits shape: {outputs['logits'].shape}")
            logging.debug(f"Labels min/max: {labels.min().item()} / {labels.max().item()}")
            logging.debug(f"Loss (total): {loss.item()}")
            logging.debug(f"Predictions (first 10): {torch.argmax(outputs['logits'], dim=1)[:10].cpu().numpy()}")
            logging.debug(f"Labels      (first 10): {labels[:10].cpu().numpy()}")
            logging.debug("===============================\n")

        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs['logits'], 1)
        batch_total = labels.size(0)
        batch_correct = (predicted == labels).sum().item()
        total += batch_total
        correct += batch_correct

        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': batch_correct / batch_total
        })

        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            logging.info(
                f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] - Loss: {loss.item():.4f}, Acc: {batch_correct / batch_total:.4f}")

    # Calculate epoch statistics
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total if total > 0 else 0

    logging.info(f"Epoch {epoch} completed - Avg Loss: {epoch_loss:.4f}, Avg Acc: {epoch_accuracy:.4f}")

    return epoch_loss, epoch_accuracy


def validate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        epoch: int
) -> Tuple[float, float]:
    """
    Validate the model with consistent metrics

    使用一致的指标验证模型
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation {epoch}")

        for batch in progress_bar:
            # Get data
            features = batch['audio'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(features, labels=labels, mode='identification')

            # Calculate loss
            losses = criterion(outputs, labels)
            loss = losses['total_loss']

            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs['logits'], 1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            total += batch_total
            correct += batch_correct

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': batch_correct / batch_total
            })

    # Calculate validation statistics
    val_loss = running_loss / len(dataloader)
    val_accuracy = correct / total if total > 0 else 0

    logging.info(f"Validation Epoch {epoch} - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")

    return val_loss, val_accuracy


def extract_embeddings(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract speaker embeddings with additional metadata

    提取说话人嵌入及附加元数据
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_speaker_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Get data
            features = batch['audio'].to(device)
            labels = batch['label']
            speaker_ids = batch['speaker_id']

            # Forward pass to get embeddings
            outputs = model(features, mode='embedding')
            embeddings = outputs['embeddings']

            # Store embeddings and labels
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            all_speaker_ids.extend(speaker_ids)

    # Concatenate all batches
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)

    return all_embeddings, all_labels, all_speaker_ids


def plot_training_curves(
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: List[float],
        val_accuracies: List[float],
        save_dir: str,
        epoch: int
) -> None:
    """
    Plot training curves for loss and accuracy

    绘制损失和准确率的训练曲线
    """
    os.makedirs(save_dir, exist_ok=True)

    # Plot loss curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss (Epoch {epoch + 1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy (Epoch {epoch + 1})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_epoch_{epoch + 1}.png'))
    plt.close()


def train_model(config: Dict) -> None:
    """
    Train the speaker recognition model with enhanced configuration support

    训练说话人识别模型，增强配置支持
    """
    # Set up logging
    logger = setup_logging(config['log_dir'])
    logger.info("Starting training with config:")
    logger.info(json.dumps(config, indent=2))

    # Save actual used config
    save_config(config, os.path.join(config['output_dir'], 'used_config.json'))

    # Set device
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # Create cache directory
    cache_dir = os.path.join(config['output_dir'], 'preprocessed_cache') if config.get('use_cache', True) else None
    if cache_dir:
        logger.info(f"Using feature cache directory: {cache_dir}")

    # Create data loaders with cache support
    logger.info("Creating data loaders...")
    train_loader, dev_loader, test_loader, speaker_to_idx = create_dataloaders(
        train_dir=config['train_dir'],
        dev_dir=config['dev_dir'],
        test_dir=config['test_dir'],
        batch_size=config['batch_size'],
        segment_length=config['segment_length'],
        sampling_rate=config['sampling_rate'],
        num_workers=config['num_workers'],
        cache_dir=cache_dir  # Add cache directory support
    )

    num_speakers = len(speaker_to_idx)
    logger.info(f"Number of speakers: {num_speakers}")

    # Create model
    logger.info("Creating model...")
    model = ChaoticSpeakerRecognitionSystem(
        chaotic_feature_dim=config['chaotic_feature_dim'],
        chaotic_dim=config['chaotic_dim'],
        trajectory_points=config['trajectory_points'],
        embedding_dim=config['embedding_dim'],
        num_speakers=num_speakers,
        use_chaotic_embedding=config['use_chaotic_embedding'],
        use_attractor_pooling=config['use_attractor_pooling'],
        system_type=config['system_type']
    )
    model = model.to(device)

    # Create loss function and optimizer
    criterion = SpeakerRecognitionLoss(
        ce_weight=config['ce_weight'],
        triplet_weight=config['triplet_weight'],
        margin=config['triplet_margin']
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['lr_patience'],
        # verbose=True
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_accuracy = 0.0
    best_metrics = {}

    if config.get('resume_checkpoint'):
        if os.path.isfile(config['resume_checkpoint']):
            logger.info(f"Loading checkpoint: {config['resume_checkpoint']}")
            model, optimizer, start_epoch, metrics = load_checkpoint(
                model, optimizer, config['resume_checkpoint']
            )
            best_val_accuracy = metrics.get('val_accuracy', 0.0)
            start_epoch += 1  # Start from the next epoch
        else:
            logger.warning(f"No checkpoint found at {config['resume_checkpoint']}")

    # Training loop
    logger.info("Starting training...")

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(start_epoch, config['num_epochs']):
        # Train for one epoch
        train_loss, train_accuracy = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=config.get('log_interval', 10)
        )

        # Validate
        val_loss, val_accuracy = validate(
            model=model,
            dataloader=dev_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Store metrics for plotting
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Create metrics dictionary
        metrics = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        }

        # Save metrics
        metrics_dir = os.path.join(config['output_dir'], 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        save_dict_to_json(metrics, os.path.join(metrics_dir, f'metrics_epoch_{epoch}.json'))

        # Check if this is the best model
        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
            best_metrics = metrics.copy()
            best_metrics['epoch'] = epoch

        # Save checkpoint
        checkpoint_interval = config.get('checkpoint_interval', 10)
        if (epoch + 1) % checkpoint_interval == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                checkpoint_dir=config['checkpoint_dir'],
                is_best=is_best
            )

        # Plot and save training curves
        plot_interval = config.get('plot_interval', 10)
        if (epoch + 1) % plot_interval == 0:
            plot_training_curves(
                train_losses=train_losses,
                val_losses=val_losses,
                train_accuracies=train_accuracies,
                val_accuracies=val_accuracies,
                save_dir=config['plot_dir'],
                epoch=epoch
            )

    # Save best metrics
    save_dict_to_json(best_metrics, os.path.join(config['output_dir'], 'best_metrics.json'))

    # Final evaluation on test set
    logger.info("Training completed. Evaluating on test set...")

    # Load best model for evaluation
    best_model_path = os.path.join(config['checkpoint_dir'], "checkpoint_best.pt")
    if os.path.isfile(best_model_path):
        model, _, _, _ = load_checkpoint(model, None, best_model_path)
    else:
        logger.warning("No best model found, using current model for evaluation")

    # Evaluate on test set
    test_metrics = evaluate_speaker_recognition(model, test_loader, device)
    logger.info("Test set evaluation results:")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    save_dict_to_json(test_metrics, os.path.join(config['output_dir'], 'test_metrics.json'))

    # Extract and visualize embeddings
    if config.get('visualize_embeddings', True):
        logger.info("Extracting embeddings for visualization...")
        embeddings, labels, speaker_ids = extract_embeddings(model, test_loader, device)

        # Save embeddings
        embeddings_dir = os.path.join(config['output_dir'], 'embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)

        np.save(os.path.join(embeddings_dir, 'embeddings.npy'), embeddings)
        np.save(os.path.join(embeddings_dir, 'labels.npy'), labels)
        with open(os.path.join(embeddings_dir, 'speaker_ids.txt'), 'w') as f:
            for id in speaker_ids:
                f.write(f"{id}\n")

        # Visualize embeddings
        plot_path = os.path.join(config['plot_dir'], 'embeddings_2d.png')
        plot_embeddings_2d(embeddings, labels, speaker_to_idx, plot_path)
        logger.info(f"Embeddings visualization saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train speaker recognition model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    # Update config with experiment name if provided
    if args.experiment:
        from config import create_experiment_config

        config = create_experiment_config(args.experiment, config)

    # Run training
    train_model(config)
