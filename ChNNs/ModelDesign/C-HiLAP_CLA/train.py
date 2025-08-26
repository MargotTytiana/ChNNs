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
from config import get_config


def setup_logging(log_dir: str) -> None:
    """
    Set up logging configuration

    设置日志配置

    Args:
        log_dir (str): Directory to save log files
                      保存日志文件的目录
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"training_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # Configure logging
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def save_checkpoint(
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        loss: float,
        accuracy: float,
        checkpoint_dir: str,
        is_best: bool = False
) -> None:
    """
    Save model checkpoint

    保存模型检查点

    Args:
        model (nn.Module): Model to save
                          要保存的模型
        optimizer (optim.Optimizer): Optimizer state
                                    优化器状态
        epoch (int): Current epoch
                    当前轮次
        loss (float): Current loss
                     当前损失
        accuracy (float): Current accuracy
                         当前准确率
        checkpoint_dir (str): Directory to save checkpoint
                             保存检查点的目录
        is_best (bool): Whether this is the best model so far
                       这是否是迄今为止最好的模型
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }

    # Save regular checkpoint
    # 保存常规检查点
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    # Save latest checkpoint (overwrite)
    # 保存最新检查点（覆盖）
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)

    # If this is the best model, save it separately
    # 如果这是最好的模型，单独保存
    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)

    logging.info(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
        checkpoint_path: str
) -> Tuple[nn.Module, Optional[optim.Optimizer], int, float, float]:
    """
    Load model checkpoint

    加载模型检查点

    Args:
        model (nn.Module): Model to load weights into
                          要加载权重的模型
        optimizer (optim.Optimizer, optional): Optimizer to load state into
                                             要加载状态的优化器
        checkpoint_path (str): Path to checkpoint file
                              检查点文件的路径

    Returns:
        Tuple: Updated model, optimizer, epoch, loss, and accuracy
               更新的模型、优化器、轮次、损失和准确率
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint.get('accuracy', 0.0)  # Handle older checkpoints without accuracy

    logging.info(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f} and accuracy {accuracy:.4f}")

    return model, optimizer, epoch, loss, accuracy


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
    Train for one epoch

    训练一个轮次

    Args:
        model (nn.Module): Model to train
                          要训练的模型
        dataloader (DataLoader): Training data loader
                                训练数据加载器
        criterion (nn.Module): Loss function
                              损失函数
        optimizer (optim.Optimizer): Optimizer
                                    优化器
        device (torch.device): Device to train on
                              训练设备
        epoch (int): Current epoch number
                    当前轮次编号
        log_interval (int): How often to log progress
                           多久记录一次进度

    Returns:
        Tuple[float, float]: Average loss and accuracy for the epoch
                            该轮次的平均损失和准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        # 获取数据
        features = batch['audio'].to(device)
        labels = batch['label'].to(device)

        # Zero the parameter gradients
        # 清零参数梯度
        optimizer.zero_grad()

        # Forward pass
        # 前向传播
        outputs = model(features, labels=labels, mode='identification')

        # Calculate loss
        # 计算损失
        losses = criterion(outputs, labels)
        loss = losses['total_loss']

        # Backward pass and optimize
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # Update statistics
        # 更新统计信息
        running_loss += loss.item()
        _, predicted = torch.max(outputs['logits'], 1)
        batch_total = labels.size(0)
        batch_correct = (predicted == labels).sum().item()
        total += batch_total
        correct += batch_correct

        # Update progress bar
        # 更新进度条
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': batch_correct / batch_total
        })

        # Log progress
        # 记录进度
        if (batch_idx + 1) % log_interval == 0:
            logging.info(
                f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] - Loss: {loss.item():.4f}, Acc: {batch_correct / batch_total:.4f}")

    # Calculate epoch statistics
    # 计算轮次统计信息
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
    Validate the model

    验证模型

    Args:
        model (nn.Module): Model to validate
                          要验证的模型
        dataloader (DataLoader): Validation data loader
                                验证数据加载器
        criterion (nn.Module): Loss function
                              损失函数
        device (torch.device): Device to validate on
                              验证设备
        epoch (int): Current epoch number
                    当前轮次编号

    Returns:
        Tuple[float, float]: Validation loss and accuracy
                            验证损失和准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation {epoch}")

        for batch in progress_bar:
            # Get data
            # 获取数据
            features = batch['audio'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            # 前向传播
            outputs = model(features, labels=labels, mode='identification')

            # Calculate loss
            # 计算损失
            losses = criterion(outputs, labels)
            loss = losses['total_loss']

            # Update statistics
            # 更新统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs['logits'], 1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            total += batch_total
            correct += batch_correct

            # Update progress bar
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': batch_correct / batch_total
            })

    # Calculate validation statistics
    # 计算验证统计信息
    val_loss = running_loss / len(dataloader)
    val_accuracy = correct / total if total > 0 else 0

    logging.info(f"Validation Epoch {epoch} - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")

    return val_loss, val_accuracy


def extract_embeddings(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract speaker embeddings from the model

    从模型中提取说话人嵌入

    Args:
        model (nn.Module): Trained model
                          训练好的模型
        dataloader (DataLoader): Data loader
                                数据加载器
        device (torch.device): Device to run on
                              运行设备

    Returns:
        Tuple[np.ndarray, np.ndarray]: Speaker embeddings and corresponding labels
                                      说话人嵌入和对应的标签
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Get data
            # 获取数据
            features = batch['audio'].to(device)
            labels = batch['label']

            # Forward pass to get embeddings
            # 前向传播获取嵌入
            outputs = model(features, mode='embedding')
            embeddings = outputs['embeddings']

            # Store embeddings and labels
            # 存储嵌入和标签
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate all batches
    # 连接所有批次
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)

    return all_embeddings, all_labels


def train_model(config: Dict) -> None:
    """
    Train the speaker recognition model

    训练说话人识别模型

    Args:
        config (Dict): Configuration dictionary
                      配置字典
    """
    # Set up logging
    # 设置日志
    setup_logging(config['log_dir'])
    logging.info("Starting training with config:")
    logging.info(json.dumps(config, indent=2))

    # Set device
    # 设置设备
    device = torch.device(config['device'])
    logging.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    # 设置随机种子以确保可重复性
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # Create data loaders
    # 创建数据加载器
    logging.info("Creating data loaders...")
    train_loader, dev_loader, test_loader, speaker_to_idx = create_dataloaders(
        train_dir=config['train_dir'],
        dev_dir=config['dev_dir'],
        test_dir=config['test_dir'],
        batch_size=config['batch_size'],
        segment_length=config['segment_length'],
        sampling_rate=config['sampling_rate'],
        num_workers=config['num_workers']
    )

    num_speakers = len(speaker_to_idx)
    logging.info(f"Number of speakers: {num_speakers}")

    # Create model
    # 创建模型
    logging.info("Creating model...")
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
    # 创建损失函数和优化器
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
    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['lr_patience'],
        # verbose=True
    )

    # Resume from checkpoint if specified
    # 如果指定，从检查点恢复
    start_epoch = 0
    best_val_accuracy = 0.0

    if config['resume_checkpoint']:
        if os.path.isfile(config['resume_checkpoint']):
            logging.info(f"Loading checkpoint: {config['resume_checkpoint']}")
            model, optimizer, start_epoch, _, best_val_accuracy = load_checkpoint(
                model, optimizer, config['resume_checkpoint']
            )
            start_epoch += 1  # Start from the next epoch
        else:
            logging.warning(f"No checkpoint found at {config['resume_checkpoint']}")

    # Training loop
    # 训练循环
    logging.info("Starting training...")

    # Lists to store metrics for plotting
    # 用于绘图的指标列表
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(start_epoch, config['num_epochs']):
        # Train for one epoch
        # 训练一个轮次
        train_loss, train_accuracy = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=config['log_interval']
        )

        # Validate
        # 验证
        val_loss, val_accuracy = validate(
            model=model,
            dataloader=dev_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        # Update learning rate
        # 更新学习率
        scheduler.step(val_loss)

        # Store metrics for plotting
        # 存储指标用于绘图
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Check if this is the best model
        # 检查这是否是最好的模型
        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy

        # Save checkpoint
        # 保存检查点
        if (epoch + 1) % config['checkpoint_interval'] == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                accuracy=val_accuracy,
                checkpoint_dir=config['checkpoint_dir'],
                is_best=is_best
            )

        # Plot and save training curves
        # 绘制并保存训练曲线
        if (epoch + 1) % config['plot_interval'] == 0:
            plot_training_curves(
                train_losses=train_losses,
                val_losses=val_losses,
                train_accuracies=train_accuracies,
                val_accuracies=val_accuracies,
                save_dir=config['plot_dir'],
                epoch=epoch
            )

    # Final evaluation on test set
    # 在测试集上进行最终评估
    logging.info("Training completed. Evaluating on test set...")

    # Load best model for evaluation
    # 加载最佳模型进行评估
    best_model_path = os.path.join(config['checkpoint_dir'], "checkpoint_best.pt")
    if os.path.isfile(best_model_path):
        model, _, _, _, _ = load_checkpoint(model, None, best_model_path)

    # Evaluate on test set
    # 在测试集上评估
    test_metrics = evaluate_speaker_recognition(model, test_loader, device)

    logging.info("Test set evaluation results:")
    logging.info(f"Accuracy: {test_metrics['accuracy']:.4f}")

    # Extract and visualize embeddings
    # 提取并可视化嵌入
    if config['visualize_embeddings']:
        logging.info("Extracting embeddings for visualization...")
        embeddings, labels = extract_embeddings(model, test_loader, device)

        # Save embeddings
        # 保存嵌入
        embeddings_dir = os.path.join(config['output_dir'], 'embeddings')
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir)

        np.save(os.path.join(embeddings_dir, 'embeddings.npy'), embeddings)
        np.save(os.path.join(embeddings_dir, 'labels.npy'), labels)

        # Visualize embeddings
        # 可视化嵌入
        logging.info("Visualizing embeddings...")
        plot_embeddings_2d(
            embeddings=embeddings,
            labels=labels,
            title="Speaker Embeddings (t-SNE Visualization)"
        )
        plt.savefig(os.path.join(config['plot_dir'], 'embeddings_visualization.png'))

    logging.info("Training and evaluation completed.")


def plot_training_curves(
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: List[float],
        val_accuracies: List[float],
        save_dir: str,
        epoch: int
) -> None:
    """
    Plot and save training curves

    绘制并保存训练曲线

    Args:
        train_losses (List[float]): Training losses
                                   训练损失
        val_losses (List[float]): Validation losses
                                 验证损失
        train_accuracies (List[float]): Training accuracies
                                       训练准确率
        val_accuracies (List[float]): Validation accuracies
                                     验证准确率
        save_dir (str): Directory to save plots
                       保存图表的目录
        epoch (int): Current epoch
                    当前轮次
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create figure with two subplots
    # 创建带有两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    # 绘制损失
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    # 绘制准确率
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and save
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_epoch_{epoch}.png'))
    plt.close()


def main():
    """
    Main function

    主函数
    """
    parser = argparse.ArgumentParser(description="Train Speaker Recognition Model")

    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'extract_embeddings'],
                        help='Operation mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation or embedding extraction')

    args = parser.parse_args()

    # Load configuration
    # 加载配置
    config = get_config(args.config)

    # Update config with command line arguments
    # 使用命令行参数更新配置
    if args.checkpoint:
        config['resume_checkpoint'] = args.checkpoint

    # Execute based on mode
    # 根据模式执行
    if args.mode == 'train':
        train_model(config)

    elif args.mode == 'evaluate':
        # Set up logging
        # 设置日志
        setup_logging(config['log_dir'])

        # Set device
        # 设置设备
        device = torch.device(config['device'])

        # Create data loaders
        # 创建数据加载器
        _, _, test_loader, speaker_to_idx = create_dataloaders(
            train_dir=config['train_dir'],
            dev_dir=config['dev_dir'],
            test_dir=config['test_dir'],
            batch_size=config['batch_size'],
            segment_length=config['segment_length'],
            sampling_rate=config['sampling_rate'],
            num_workers=config['num_workers']
        )

        # Create model
        # 创建模型
        model = ChaoticSpeakerRecognitionSystem(
            chaotic_feature_dim=config['chaotic_feature_dim'],
            chaotic_dim=config['chaotic_dim'],
            trajectory_points=config['trajectory_points'],
            embedding_dim=config['embedding_dim'],
            num_speakers=len(speaker_to_idx),
            use_chaotic_embedding=config['use_chaotic_embedding'],
            use_attractor_pooling=config['use_attractor_pooling'],
            system_type=config['system_type']
        )
        model = model.to(device)

        # Load checkpoint
        # 加载检查点
        if not args.checkpoint:
            logging.error("Checkpoint path must be provided for evaluation mode")
            return

        model, _, _, _, _ = load_checkpoint(model, None, args.checkpoint)

        # Evaluate
        # 评估
        logging.info("Evaluating model on test set...")
        test_metrics = evaluate_speaker_recognition(model, test_loader, device)

        logging.info("Test set evaluation results:")
        logging.info(json.dumps(test_metrics, indent=2))

    elif args.mode == 'extract_embeddings':
        # Set up logging
        # 设置日志
        setup_logging(config['log_dir'])

        # Set device
        # 设置设备
        device = torch.device(config['device'])

        # Create data loaders
        # 创建数据加载器
        _, _, test_loader, speaker_to_idx = create_dataloaders(
            train_dir=config['train_dir'],
            dev_dir=config['dev_dir'],
            test_dir=config['test_dir'],
            batch_size=config['batch_size'],
            segment_length=config['segment_length'],
            sampling_rate=config['sampling_rate'],
            num_workers=config['num_workers']
        )

        # Create model
        # 创建模型
        model = ChaoticSpeakerRecognitionSystem(
            chaotic_feature_dim=config['chaotic_feature_dim'],
            chaotic_dim=config['chaotic_dim'],
            trajectory_points=config['trajectory_points'],
            embedding_dim=config['embedding_dim'],
            num_speakers=len(speaker_to_idx),
            use_chaotic_embedding=config['use_chaotic_embedding'],
            use_attractor_pooling=config['use_attractor_pooling'],
            system_type=config['system_type']
        )
        model = model.to(device)

        # Load checkpoint
        # 加载检查点
        if not args.checkpoint:
            logging.error("Checkpoint path must be provided for embedding extraction mode")
            return

        model, _, _, _, _ = load_checkpoint(model, None, args.checkpoint)

        # Extract embeddings
        # 提取嵌入
        logging.info("Extracting embeddings...")
        embeddings, labels = extract_embeddings(model, test_loader, device)

        # Save embeddings
        # 保存嵌入
        embeddings_dir = os.path.join(config['output_dir'], 'embeddings')
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir)

        np.save(os.path.join(embeddings_dir, 'embeddings.npy'), embeddings)
        np.save(os.path.join(embeddings_dir, 'labels.npy'), labels)

        logging.info(f"Embeddings saved to {embeddings_dir}")

        # Visualize embeddings
        # 可视化嵌入
        if config['visualize_embeddings']:
            logging.info("Visualizing embeddings...")
            plot_embeddings_2d(
                embeddings=embeddings,
                labels=labels,
                title="Speaker Embeddings (t-SNE Visualization)"
            )

            plot_dir = config['plot_dir']
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            plt.savefig(os.path.join(plot_dir, 'embeddings_visualization.png'))
            logging.info(f"Visualization saved to {os.path.join(plot_dir, 'embeddings_visualization.png')}")


if __name__ == "__main__":
    main()
