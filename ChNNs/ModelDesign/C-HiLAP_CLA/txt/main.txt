import os
import sys
import argparse
import torch
import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from datetime import datetime

# Import custom modules
# 导入自定义模块
from data_loader import create_dataloaders
from phase_space import PhaseSpaceReconstructor
from chaotic_features import ChaoticFeatureExtractor
from chaotic_embedding import ChaoticEmbeddingLayer, DifferentiableChaoticEmbedding
from attractor_pooling import AttractorPooling, DifferentiableAttractorPooling
from speaker_model import (
    ChaoticSpeakerRecognitionSystem,
    SpeakerRecognitionLoss,
    evaluate_speaker_recognition,
    evaluate_speaker_verification,
    plot_embeddings_2d
)
from train import train_model, validate, extract_embeddings
from config import get_config, save_config, create_experiment_config
from utils import (
    set_seed,
    setup_logger,
    plot_waveform,
    plot_spectrogram,
    plot_mel_spectrogram,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_det_curve,
    plot_training_curves,
    plot_phase_space,
    plot_recurrence_plot,
    calculate_eer,
    calculate_metrics,
    save_dict_to_json,
    load_dict_from_json
)


def parse_arguments():
    """
    Parse command line arguments

    解析命令行参数

    Returns:
        argparse.Namespace: Parsed arguments
                           解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="Chaotic Neural Network Speaker Recognition System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Main operation mode
    # 主操作模式
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'extract_features', 'extract_embeddings',
                                 'visualize', 'demo', 'benchmark'],
                        help='Operation mode')

    # Configuration
    # 配置
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name (creates a new config based on default)')

    # Data paths
    # 数据路径
    parser.add_argument('--train_dir', type=str, default=None,
                        help='Training data directory')
    parser.add_argument('--dev_dir', type=str, default=None,
                        help='Development data directory')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Test data directory')

    # Model parameters
    # 模型参数
    parser.add_argument('--system_type', type=str, default=None,
                        choices=['lorenz', 'rossler', 'chua'],
                        help='Chaotic system type')
    parser.add_argument('--use_chaotic_embedding', action='store_true',
                        help='Use chaotic embedding layer')
    parser.add_argument('--use_attractor_pooling', action='store_true',
                        help='Use attractor pooling layer')

    # Training parameters
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')

    # Checkpoint handling
    # 检查点处理
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation or embedding extraction')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')

    # Output control
    # 输出控制
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    # Visualization options
    # 可视化选项
    parser.add_argument('--visualize_embeddings', action='store_true',
                        help='Visualize speaker embeddings')
    parser.add_argument('--visualize_phase_space', action='store_true',
                        help='Visualize phase space reconstruction')
    parser.add_argument('--visualize_features', action='store_true',
                        help='Visualize chaotic features')

    # Demo options
    # 演示选项
    parser.add_argument('--audio_file', type=str, default=None,
                        help='Audio file for demo mode')
    parser.add_argument('--compare_file', type=str, default=None,
                        help='Audio file to compare with in demo mode')

    # Benchmark options
    # 基准测试选项
    parser.add_argument('--benchmark_systems', action='store_true',
                        help='Benchmark different chaotic systems')
    parser.add_argument('--benchmark_features', action='store_true',
                        help='Benchmark different feature extraction methods')

    # Misc
    # 杂项
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID (default: use CPU)')

    return parser.parse_args()


def update_config_from_args(config: Dict, args: argparse.Namespace) -> Dict:
    """
    Update configuration with command line arguments

    使用命令行参数更新配置

    Args:
        config (Dict): Configuration dictionary
                      配置字典
        args (argparse.Namespace): Command line arguments
                                  命令行参数

    Returns:
        Dict: Updated configuration
              更新后的配置
    """
    # Update configuration with command line arguments if provided
    # 如果提供了命令行参数，则使用其更新配置
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None and arg in config:
            config[arg] = value

    # Handle boolean flags
    # 处理布尔标志
    if args.use_chaotic_embedding:
        config['use_chaotic_embedding'] = True

    if args.use_attractor_pooling:
        config['use_attractor_pooling'] = True

    if args.visualize_embeddings:
        config['visualize_embeddings'] = True

    if args.resume and args.checkpoint:
        config['resume_checkpoint'] = args.checkpoint

    # Set device based on GPU argument
    # 根据GPU参数设置设备
    if args.gpu is not None:
        if torch.cuda.is_available():
            config['device'] = f"cuda:{args.gpu}"
        else:
            print("Warning: GPU requested but CUDA is not available. Using CPU instead.")
            config['device'] = "cpu"

    return config


def setup_environment(config: Dict) -> None:
    """
    Set up environment based on configuration

    根据配置设置环境

    Args:
        config (Dict): Configuration dictionary
                      配置字典
    """
    # Set random seed
    # 设置随机种子
    set_seed(config['seed'])

    # Create output directories
    # 创建输出目录
    for dir_key in ['log_dir', 'checkpoint_dir', 'output_dir', 'plot_dir']:
        os.makedirs(config[dir_key], exist_ok=True)

    # Save configuration
    # 保存配置
    config_path = os.path.join(config['output_dir'], 'config.json')
    save_config(config, config_path)


def run_training(config: Dict) -> None:
    """
    Run training mode

    运行训练模式

    Args:
        config (Dict): Configuration dictionary
                      配置字典
    """
    # Set up logger
    # 设置日志记录器
    logger = setup_logger(config['log_dir'], "chaotic_speaker_recognition_train")
    logger.info("Starting training with configuration:")
    logger.info(json.dumps(config, indent=2))

    # Train model
    # 训练模型
    train_model(config)


def run_evaluation(config: Dict) -> None:
    """
    Run evaluation mode

    运行评估模式

    Args:
        config (Dict): Configuration dictionary
                      配置字典
    """
    # Set up logger
    # 设置日志记录器
    logger = setup_logger(config['log_dir'], "chaotic_speaker_recognition_eval")
    logger.info("Starting evaluation with configuration:")
    logger.info(json.dumps(config, indent=2))

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
    if config['checkpoint'] is None and os.path.exists(os.path.join(config['checkpoint_dir'], 'checkpoint_best.pt')):
        config['checkpoint'] = os.path.join(config['checkpoint_dir'], 'checkpoint_best.pt')

    if config['checkpoint'] is None:
        logger.error("No checkpoint provided for evaluation. Please specify a checkpoint.")
        return

    checkpoint = torch.load(config['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {checkpoint.get('accuracy', 0):.4f}")

    # Evaluate model
    # 评估模型
    logger.info("Evaluating model on test set...")
    test_metrics = evaluate_speaker_recognition(model, test_loader, device)

    # Save metrics
    # 保存指标
    metrics_path = os.path.join(config['output_dir'], 'test_metrics.json')
    save_dict_to_json(test_metrics, metrics_path)

    logger.info("Test set evaluation results:")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")

    # Plot confusion matrix
    # 绘制混淆矩阵
    if 'confusion_matrix' in test_metrics:
        cm = np.array(test_metrics['confusion_matrix'])
        fig = plot_confusion_matrix(
            np.repeat(np.arange(cm.shape[0]), np.sum(cm, axis=1).astype(int)),
            np.concatenate([[i] * np.sum(cm[i, :]) for i in range(cm.shape[0])]),
            title="Speaker Recognition Confusion Matrix"
        )
        plt.savefig(os.path.join(config['plot_dir'], 'confusion_matrix.png'))
        plt.close(fig)

    # Extract and visualize embeddings
    # 提取并可视化嵌入
    if config['visualize_embeddings']:
        logger.info("Extracting embeddings for visualization...")
        embeddings, labels = extract_embeddings(model, test_loader, device)

        # Save embeddings
        # 保存嵌入
        embeddings_dir = os.path.join(config['output_dir'], 'embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)
        np.save(os.path.join(embeddings_dir, 'embeddings.npy'), embeddings)
        np.save(os.path.join(embeddings_dir, 'labels.npy'), labels)

        # Visualize embeddings
        # 可视化嵌入
        logger.info("Visualizing embeddings...")
        fig = plot_embeddings_2d(
            embeddings=embeddings,
            labels=labels,
            title="Speaker Embeddings (t-SNE Visualization)"
        )
        plt.savefig(os.path.join(config['plot_dir'], 'embeddings_visualization.png'))
        plt.close(fig)

    logger.info("Evaluation completed.")


def run_feature_extraction(config: Dict) -> None:
    """
    Run feature extraction mode

    运行特征提取模式

    Args:
        config (Dict): Configuration dictionary
                      配置字典
    """
    # Set up logger
    # 设置日志记录器
    logger = setup_logger(config['log_dir'], "chaotic_speaker_recognition_features")
    logger.info("Starting feature extraction with configuration:")
    logger.info(json.dumps(config, indent=2))

    # Create data loaders
    # 创建数据加载器
    train_loader, dev_loader, test_loader, _ = create_dataloaders(
        train_dir=config['train_dir'],
        dev_dir=config['dev_dir'],
        test_dir=config['test_dir'],
        batch_size=1,  # Process one file at a time
        segment_length=config['segment_length'],
        sampling_rate=config['sampling_rate'],
        num_workers=1
    )

    # Create feature extractor
    # 创建特征提取器
    feature_extractor = ChaoticFeatureExtractor(
        embedding_dim=config['embedding_dim'],
        delay=config.get('delay', 1),
        scales=config.get('mlsa_scales', [1, 2, 4, 8, 16]),
        n_lyapunov_exponents=config.get('n_lyapunov_exponents', 2),
        rqa_threshold=config.get('rqa_threshold', None)
    )

    # Create output directory
    # 创建输出目录
    features_dir = os.path.join(config['output_dir'], 'features')
    os.makedirs(features_dir, exist_ok=True)

    # Extract features from each dataset
    # 从每个数据集提取特征
    for name, loader in [('train', train_loader), ('dev', dev_loader), ('test', test_loader)]:
        logger.info(f"Extracting features from {name} set...")

        all_features = []
        all_labels = []
        all_speaker_ids = []
        all_file_paths = []

        for batch in loader:
            audio = batch['audio'].numpy()[0]  # Get the first (and only) item in batch
            label = batch['label'].item()
            speaker_id = batch['speaker_id'][0]
            file_path = batch['file_path'][0]

            # Extract features
            # 提取特征
            features = feature_extractor.extract(audio)

            # Store features and metadata
            # 存储特征和元数据
            all_features.append(features)
            all_labels.append(label)
            all_speaker_ids.append(speaker_id)
            all_file_paths.append(file_path)

        # Convert to numpy arrays
        # 转换为numpy数组
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)

        # Save features and metadata
        # 保存特征和元数据
        np.save(os.path.join(features_dir, f'{name}_features.npy'), all_features)
        np.save(os.path.join(features_dir, f'{name}_labels.npy'), all_labels)

        # Save metadata as JSON
        # 将元数据保存为JSON
        metadata = {
            'speaker_ids': all_speaker_ids,
            'file_paths': all_file_paths,
            'feature_dim': all_features.shape[1],
            'num_samples': len(all_features)
        }
        save_dict_to_json(metadata, os.path.join(features_dir, f'{name}_metadata.json'))

        logger.info(f"Extracted {len(all_features)} feature vectors from {name} set.")

    logger.info("Feature extraction completed.")


def run_embedding_extraction(config: Dict) -> None:
    """
    Run embedding extraction mode

    运行嵌入提取模式

    Args:
        config (Dict): Configuration dictionary
                      配置字典
    """
    # Set up logger
    # 设置日志记录器
    logger = setup_logger(config['log_dir'], "chaotic_speaker_recognition_embeddings")
    logger.info("Starting embedding extraction with configuration:")
    logger.info(json.dumps(config, indent=2))

    # Set device
    # 设置设备
    device = torch.device(config['device'])

    # Create data loaders
    # 创建数据加载器
    train_loader, dev_loader, test_loader, speaker_to_idx = create_dataloaders(
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
    if config['checkpoint'] is None and os.path.exists(os.path.join(config['checkpoint_dir'], 'checkpoint_best.pt')):
        config['checkpoint'] = os.path.join(config['checkpoint_dir'], 'checkpoint_best.pt')

    if config['checkpoint'] is None:
        logger.error("No checkpoint provided for embedding extraction. Please specify a checkpoint.")
        return

    checkpoint = torch.load(config['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {checkpoint.get('accuracy', 0):.4f}")

    # Create output directory
    # 创建输出目录
    embeddings_dir = os.path.join(config['output_dir'], 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)

    # Extract embeddings from each dataset
    # 从每个数据集提取嵌入
    for name, loader in [('train', train_loader), ('dev', dev_loader), ('test', test_loader)]:
        logger.info(f"Extracting embeddings from {name} set...")

        embeddings, labels = extract_embeddings(model, loader, device)

        # Save embeddings and labels
        # 保存嵌入和标签
        np.save(os.path.join(embeddings_dir, f'{name}_embeddings.npy'), embeddings)
        np.save(os.path.join(embeddings_dir, f'{name}_labels.npy'), labels)

        logger.info(f"Extracted {len(embeddings)} embedding vectors from {name} set.")

        # Visualize embeddings
        # 可视化嵌入
        if config['visualize_embeddings']:
            logger.info(f"Visualizing {name} embeddings...")
            fig = plot_embeddings_2d(
                embeddings=embeddings,
                labels=labels,
                title=f"{name.capitalize()} Speaker Embeddings (t-SNE Visualization)"
            )
            plt.savefig(os.path.join(config['plot_dir'], f'{name}_embeddings_visualization.png'))
            plt.close(fig)

    logger.info("Embedding extraction completed.")


def run_visualization(config: Dict) -> None:
    """
    Run visualization mode

    运行可视化模式

    Args:
        config (Dict): Configuration dictionary
                      配置字典
    """
    # Set up logger
    # 设置日志记录器
    logger = setup_logger(config['log_dir'], "chaotic_speaker_recognition_viz")
    logger.info("Starting visualization with configuration:")
    logger.info(json.dumps(config, indent=2))

    # Create data loaders
    # 创建数据加载器
    train_loader, _, _, _ = create_dataloaders(
        train_dir=config['train_dir'],
        dev_dir=config['dev_dir'],
        test_dir=config['test_dir'],
        batch_size=1,  # Process one file at a time
        segment_length=config['segment_length'],
        sampling_rate=config['sampling_rate'],
        num_workers=1
    )

    # Get a sample audio
    # 获取一个样本音频
    for batch in train_loader:
        audio = batch['audio'].numpy()[0]  # Get the first (and only) item in batch
        speaker_id = batch['speaker_id'][0]
        file_path = batch['file_path'][0]
        break

    # Create output directory
    # 创建输出目录
    viz_dir = os.path.join(config['plot_dir'], 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Plot waveform
    # 绘制波形
    logger.info("Generating waveform visualization...")
    fig = plot_waveform(audio, config['sampling_rate'], title=f"Audio Waveform - Speaker {speaker_id}")
    plt.savefig(os.path.join(viz_dir, 'waveform.png'))
    plt.close(fig)

    # Plot spectrogram
    # 绘制频谱图
    logger.info("Generating spectrogram visualization...")
    fig = plot_spectrogram(audio, config['sampling_rate'], title=f"Spectrogram - Speaker {speaker_id}")
    plt.savefig(os.path.join(viz_dir, 'spectrogram.png'))
    plt.close(fig)

    # Plot mel spectrogram
    # 绘制梅尔频谱图
    logger.info("Generating mel spectrogram visualization...")
    fig = plot_mel_spectrogram(audio, config['sampling_rate'], title=f"Mel Spectrogram - Speaker {speaker_id}")
    plt.savefig(os.path.join(viz_dir, 'mel_spectrogram.png'))
    plt.close(fig)

    # Visualize phase space reconstruction
    # 可视化相空间重构
    if config['visualize_phase_space']:
        logger.info("Generating phase space visualization...")

        # Create phase space reconstructor
        # 创建相空间重构器
        reconstructor = PhaseSpaceReconstructor(
            delay=config.get('delay', None),
            embedding_dim=config.get('embedding_dim', None),
            max_delay=config.get('max_delay', 100),
            max_dim=config.get('max_dim', 10),
            method=config.get('delay_method', 'autocorr')
        )

        # Reconstruct phase space
        # 重构相空间
        phase_space = reconstructor.reconstruct(audio)

        # Plot phase space
        # 绘制相空间
        reconstructor.plot_phase_space(phase_space, f"Phase Space Reconstruction - Speaker {speaker_id}")
        plt.savefig(os.path.join(viz_dir, 'phase_space.png'))
        plt.close()

        # Plot recurrence plot
        # 绘制递归图
        logger.info("Generating recurrence plot visualization...")
        fig = plot_recurrence_plot(audio, title=f"Recurrence Plot - Speaker {speaker_id}")
        plt.savefig(os.path.join(viz_dir, 'recurrence_plot.png'))
        plt.close(fig)

    # Visualize chaotic features
    # 可视化混沌特征
    if config['visualize_features']:
        logger.info("Generating chaotic features visualization...")

        # Create feature extractor
        # 创建特征提取器
        feature_extractor = ChaoticFeatureExtractor(
            embedding_dim=config['embedding_dim'],
            delay=config.get('delay', 1),
            scales=config.get('mlsa_scales', [1, 2, 4, 8, 16]),
            n_lyapunov_exponents=config.get('n_lyapunov_exponents', 2),
            rqa_threshold=config.get('rqa_threshold', None)
        )

        # Extract features with names
        # 提取带有名称的特征
        named_features = feature_extractor.extract_with_names(audio)

        # Plot features as bar chart
        # 将特征绘制为条形图
        fig, ax = plt.subplots(figsize=(12, 6))
        feature_names = list(named_features.keys())
        feature_values = list(named_features.values())

        ax.bar(range(len(feature_names)), feature_values)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=90)
        ax.set_title(f"Chaotic Features - Speaker {speaker_id}")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'chaotic_features.png'))
        plt.close(fig)

    # Visualize embeddings if available
    # 如果可用，可视化嵌入
    embeddings_path = os.path.join(config['output_dir'], 'embeddings', 'train_embeddings.npy')
    labels_path = os.path.join(config['output_dir'], 'embeddings', 'train_labels.npy')

    if os.path.exists(embeddings_path) and os.path.exists(labels_path) and config['visualize_embeddings']:
        logger.info("Generating embeddings visualization...")

        # Load embeddings and labels
        # 加载嵌入和标签
        embeddings = np.load(embeddings_path)
        labels = np.load(labels_path)

        # Visualize embeddings
        # 可视化嵌入
        fig = plot_embeddings_2d(
            embeddings=embeddings,
            labels=labels,
            title="Speaker Embeddings (t-SNE Visualization)"
        )
        plt.savefig(os.path.join(viz_dir, 'embeddings_visualization.png'))
        plt.close(fig)

    logger.info("Visualization completed. Results saved to: " + viz_dir)


def run_demo(config: Dict) -> None:
    """
    Run demo mode

    运行演示模式

    Args:
        config (Dict): Configuration dictionary
                      配置字典
    """
    # Set up logger
    # 设置日志记录器
    logger = setup_logger(config['log_dir'], "chaotic_speaker_recognition_demo")
    logger.info("Starting demo with configuration:")
    logger.info(json.dumps(config, indent=2))

    # Check if audio file is provided
    # 检查是否提供了音频文件
    if config['audio_file'] is None:
        logger.error("No audio file provided for demo. Please specify an audio file.")
        return

    # Set device
    # 设置设备
    device = torch.device(config['device'])

    # Load audio file
    # 加载音频文件
    import librosa
    logger.info(f"Loading audio file: {config['audio_file']}")
    audio, sr = librosa.load(config['audio_file'], sr=config['sampling_rate'])

    # Create output directory
    # 创建输出目录
    demo_dir = os.path.join(config['output_dir'], 'demo')
    os.makedirs(demo_dir, exist_ok=True)

    # Plot waveform
    # 绘制波形
    logger.info("Generating waveform visualization...")
    fig = plot_waveform(audio, sr, title="Input Audio Waveform")
    plt.savefig(os.path.join(demo_dir, 'input_waveform.png'))
    plt.close(fig)

    # Plot spectrogram
    # 绘制频谱图
    logger.info("Generating spectrogram visualization...")
    fig = plot_spectrogram(audio, sr, title="Input Audio Spectrogram")
    plt.savefig(os.path.join(demo_dir, 'input_spectrogram.png'))
    plt.close(fig)

    # Create phase space reconstructor
    # 创建相空间重构器
    logger.info("Reconstructing phase space...")
    reconstructor = PhaseSpaceReconstructor(
        delay=config.get('delay', None),
        embedding_dim=config.get('embedding_dim', None),
        max_delay=config.get('max_delay', 100),
        max_dim=config.get('max_dim', 10),
        method=config.get('delay_method', 'autocorr')
    )

    # Reconstruct phase space
    # 重构相空间
    phase_space = reconstructor.reconstruct(audio)

    # Plot phase space
    # 绘制相空间
    reconstructor.plot_phase_space(phase_space, "Phase Space Reconstruction")
    plt.savefig(os.path.join(demo_dir, 'phase_space.png'))
    plt.close()

    # Create feature extractor
    # 创建特征提取器
    logger.info("Extracting chaotic features...")
    feature_extractor = ChaoticFeatureExtractor(
        embedding_dim=config['embedding_dim'],
        delay=reconstructor.delay,
        scales=config.get('mlsa_scales', [1, 2, 4, 8, 16]),
        n_lyapunov_exponents=config.get('n_lyapunov_exponents', 2),
        rqa_threshold=config.get('rqa_threshold', None)
    )

    # Extract features
    # 提取特征
    features = feature_extractor.extract(audio)

    # If a checkpoint is provided, load the model and process the audio
    # 如果提供了检查点，加载模型并处理音频
    if config['checkpoint'] is not None:
        logger.info("Loading model from checkpoint...")

        # Create data loaders to get speaker_to_idx
        # 创建数据加载器以获取speaker_to_idx
        _, _, _, speaker_to_idx = create_dataloaders(
            train_dir=config['train_dir'],
            dev_dir=config['dev_dir'],
            test_dir=config['test_dir'],
            batch_size=1,
            segment_length=config['segment_length'],
            sampling_rate=config['sampling_rate'],
            num_workers=1
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
        checkpoint = torch.load(config['checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {checkpoint.get('accuracy', 0):.4f}")

        # Convert features to tensor
        # 将特征转换为张量
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        # Get embedding
        # 获取嵌入
        logger.info("Generating speaker embedding...")
        with torch.no_grad():
            outputs = model(features_tensor, mode='embedding')
            embedding = outputs['embeddings'].cpu().numpy()[0]

        # Save embedding
        # 保存嵌入
        np.save(os.path.join(demo_dir, 'embedding.npy'), embedding)

        # If compare_file is provided, compare the two embeddings
        # 如果提供了compare_file，比较两个嵌入
        if config['compare_file'] is not None:
            logger.info(f"Comparing with audio file: {config['compare_file']}")

            # Load comparison audio
            # 加载比较音频
            compare_audio, compare_sr = librosa.load(config['compare_file'], sr=config['sampling_rate'])

            # Reconstruct phase space
            # 重构相空间
            compare_phase_space = reconstructor.reconstruct(compare_audio)

            # Extract features
            # 提取特征
            compare_features = feature_extractor.extract(compare_audio)

            # Convert features to tensor
            # 将特征转换为张量
            compare_features_tensor = torch.tensor(compare_features, dtype=torch.float32).unsqueeze(0).to(device)

            # Get embedding
            # 获取嵌入
            with torch.no_grad():
                compare_outputs = model(compare_features_tensor, mode='embedding')
                compare_embedding = compare_outputs['embeddings'].cpu().numpy()[0]

            # Save comparison embedding
            # 保存比较嵌入
            np.save(os.path.join(demo_dir, 'compare_embedding.npy'), compare_embedding)

            # Calculate cosine similarity
            # 计算余弦相似度
            similarity = np.dot(embedding, compare_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(compare_embedding))

            logger.info(f"Cosine similarity between the two audio files: {similarity:.4f}")

            # Save similarity result
            # 保存相似度结果
            result = {
                'cosine_similarity': float(similarity),
                'is_same_speaker': bool(similarity > 0.7),  # Threshold can be adjusted
                'threshold': 0.7,
                'audio_file': config['audio_file'],
                'compare_file': config['compare_file']
            }

            save_dict_to_json(result, os.path.join(demo_dir, 'comparison_result.json'))

            logger.info(f"Same speaker: {result['is_same_speaker']}")

    logger.info("Demo completed. Results saved to: " + demo_dir)


def run_benchmark(config: Dict) -> None:
    """
    Run benchmark mode

    运行基准测试模式

    Args:
        config (Dict): Configuration dictionary
                      配置字典
    """
    # Set up logger
    # 设置日志记录器
    logger = setup_logger(config['log_dir'], "chaotic_speaker_recognition_benchmark")
    logger.info("Starting benchmark with configuration:")
    logger.info(json.dumps(config, indent=2))

    # Create output directory
    # 创建输出目录
    benchmark_dir = os.path.join(config['output_dir'], 'benchmark')
    os.makedirs(benchmark_dir, exist_ok=True)

    # Benchmark different chaotic systems
    # 对不同的混沌系统进行基准测试
    if config['benchmark_systems']:
        logger.info("Benchmarking different chaotic systems...")

        # Systems to benchmark
        # 要进行基准测试的系统
        systems = ['lorenz', 'rossler', 'chua']
        results = {}

        for system in systems:
            logger.info(f"Testing {system} system...")

            # Create experiment config
            # 创建实验配置
            exp_config = create_experiment_config(
                experiment_name=f"benchmark_{system}",
                base_config=config,
                system_type=system
            )

            # Save experiment config
            # 保存实验配置
            exp_config_path = os.path.join(benchmark_dir, f'{system}_config.json')
            save_config(exp_config, exp_config_path)

            # Run a mini training and evaluation
            # 运行小型训练和评估
            try:
                # Limit epochs for benchmarking
                # 限制用于基准测试的轮次
                exp_config['num_epochs'] = 5

                # Train model
                # 训练模型
                train_model(exp_config)

                # Load best model
                # 加载最佳模型
                best_model_path = os.path.join(exp_config['checkpoint_dir'], "checkpoint_best.pt")

                if os.path.exists(best_model_path):
                    # Set device
                    # 设置设备
                    device = torch.device(exp_config['device'])

                    # Create data loaders
                    # 创建数据加载器
                    _, _, test_loader, speaker_to_idx = create_dataloaders(
                        train_dir=exp_config['train_dir'],
                        dev_dir=exp_config['dev_dir'],
                        test_dir=exp_config['test_dir'],
                        batch_size=exp_config['batch_size'],
                        segment_length=exp_config['segment_length'],
                        sampling_rate=exp_config['sampling_rate'],
                        num_workers=exp_config['num_workers']
                    )

                    # Create model
                    # 创建模型
                    model = ChaoticSpeakerRecognitionSystem(
                        chaotic_feature_dim=exp_config['chaotic_feature_dim'],
                        chaotic_dim=exp_config['chaotic_dim'],
                        trajectory_points=exp_config['trajectory_points'],
                        embedding_dim=exp_config['embedding_dim'],
                        num_speakers=len(speaker_to_idx),
                        use_chaotic_embedding=exp_config['use_chaotic_embedding'],
                        use_attractor_pooling=exp_config['use_attractor_pooling'],
                        system_type=system
                    )
                    model = model.to(device)

                    # Load checkpoint
                    # 加载检查点
                    checkpoint = torch.load(best_model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])

                    # Evaluate model
                    # 评估模型
                    test_metrics = evaluate_speaker_recognition(model, test_loader, device)

                    # Store results
                    # 存储结果
                    results[system] = {
                        'accuracy': test_metrics['accuracy'],
                        'epochs': checkpoint['epoch'] + 1,
                        'loss': checkpoint['loss']
                    }
                else:
                    logger.warning(f"No checkpoint found for {system} system.")
                    results[system] = {'error': 'No checkpoint found'}

            except Exception as e:
                logger.error(f"Error benchmarking {system} system: {str(e)}")
                results[system] = {'error': str(e)}

        # Save benchmark results
        # 保存基准测试结果
        save_dict_to_json(results, os.path.join(benchmark_dir, 'system_benchmark_results.json'))

        # Plot benchmark results
        # 绘制基准测试结果
        if any('accuracy' in results[system] for system in systems):
            fig, ax = plt.subplots(figsize=(10, 6))

            systems_with_results = [s for s in systems if 'accuracy' in results[s]]
            accuracies = [results[s]['accuracy'] for s in systems_with_results]

            ax.bar(systems_with_results, accuracies)
            ax.set_title("Chaotic System Benchmark Results")
            ax.set_xlabel("System")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)

            for i, v in enumerate(accuracies):
                ax.text(i, v + 0.01, f"{v:.4f}", ha='center')

            plt.tight_layout()
            plt.savefig(os.path.join(benchmark_dir, 'system_benchmark_results.png'))
            plt.close(fig)

    # Benchmark different feature extraction methods
    # 对不同的特征提取方法进行基准测试
    if config['benchmark_features']:
        logger.info("Benchmarking different feature extraction methods...")

        # Feature extraction methods to benchmark
        # 要进行基准测试的特征提取方法
        feature_methods = [
            {'name': 'MLSA_only', 'use_mlsa': True, 'use_rqa': False},
            {'name': 'RQA_only', 'use_mlsa': False, 'use_rqa': True},
            {'name': 'MLSA_RQA', 'use_mlsa': True, 'use_rqa': True}
        ]

        results = {}

        for method in feature_methods:
            logger.info(f"Testing {method['name']} feature extraction method...")

            # Create experiment config
            # 创建实验配置
            exp_config = create_experiment_config(
                experiment_name=f"benchmark_{method['name']}",
                base_config=config,
                use_mlsa=method['use_mlsa'],
                use_rqa=method['use_rqa']
            )

            # Save experiment config
            # 保存实验配置
            exp_config_path = os.path.join(benchmark_dir, f'{method["name"]}_config.json')
            save_config(exp_config, exp_config_path)

            # Run a mini training and evaluation
            # 运行小型训练和评估
            try:
                # Limit epochs for benchmarking
                # 限制用于基准测试的轮次
                exp_config['num_epochs'] = 5

                # Train model
                # 训练模型
                train_model(exp_config)

                # Load best model
                # 加载最佳模型
                best_model_path = os.path.join(exp_config['checkpoint_dir'], "checkpoint_best.pt")

                if os.path.exists(best_model_path):
                    # Set device
                    # 设置设备
                    device = torch.device(exp_config['device'])

                    # Create data loaders
                    # 创建数据加载器
                    _, _, test_loader, speaker_to_idx = create_dataloaders(
                        train_dir=exp_config['train_dir'],
                        dev_dir=exp_config['dev_dir'],
                        test_dir=exp_config['test_dir'],
                        batch_size=exp_config['batch_size'],
                        segment_length=exp_config['segment_length'],
                        sampling_rate=exp_config['sampling_rate'],
                        num_workers=exp_config['num_workers']
                    )

                    # Create model
                    # 创建模型
                    model = ChaoticSpeakerRecognitionSystem(
                        chaotic_feature_dim=exp_config['chaotic_feature_dim'],
                        chaotic_dim=exp_config['chaotic_dim'],
                        trajectory_points=exp_config['trajectory_points'],
                        embedding_dim=exp_config['embedding_dim'],
                        num_speakers=len(speaker_to_idx),
                        use_chaotic_embedding=exp_config['use_chaotic_embedding'],
                        use_attractor_pooling=exp_config['use_attractor_pooling'],
                        system_type=exp_config['system_type']
                    )
                    model = model.to(device)

                    # Load checkpoint
                    # 加载检查点
                    checkpoint = torch.load(best_model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])

                    # Evaluate model
                    # 评估模型
                    test_metrics = evaluate_speaker_recognition(model, test_loader, device)

                    # Store results
                    # 存储结果
                    results[method['name']] = {
                        'accuracy': test_metrics['accuracy'],
                        'epochs': checkpoint['epoch'] + 1,
                        'loss': checkpoint['loss']
                    }
                else:
                    logger.warning(f"No checkpoint found for {method['name']} method.")
                    results[method['name']] = {'error': 'No checkpoint found'}

            except Exception as e:
                logger.error(f"Error benchmarking {method['name']} method: {str(e)}")
                results[method['name']] = {'error': str(e)}

        # Save benchmark results
        # 保存基准测试结果
        save_dict_to_json(results, os.path.join(benchmark_dir, 'feature_benchmark_results.json'))

        # Plot benchmark results
        # 绘制基准测试结果
        if any('accuracy' in results[method['name']] for method in feature_methods):
            fig, ax = plt.subplots(figsize=(10, 6))

            methods_with_results = [m['name'] for m in feature_methods if 'accuracy' in results[m['name']]]
            accuracies = [results[m]['accuracy'] for m in methods_with_results]

            ax.bar(methods_with_results, accuracies)
            ax.set_title("Feature Extraction Method Benchmark Results")
            ax.set_xlabel("Method")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)

            for i, v in enumerate(accuracies):
                ax.text(i, v + 0.01, f"{v:.4f}", ha='center')

            plt.tight_layout()
            plt.savefig(os.path.join(benchmark_dir, 'feature_benchmark_results.png'))
            plt.close(fig)

    logger.info("Benchmark completed. Results saved to: " + benchmark_dir)


def main():
    """
    Main function

    主函数
    """
    # Parse command line arguments
    # 解析命令行参数
    args = parse_arguments()

    # Load configuration
    # 加载配置
    config = get_config(args.config)

    # If experiment name is provided, create experiment config
    # 如果提供了实验名称，创建实验配置
    if args.experiment:
        config = create_experiment_config(args.experiment, config)

    # Update configuration with command line arguments
    # 使用命令行参数更新配置
    config = update_config_from_args(config, args)

    # Set up environment
    # 设置环境
    setup_environment(config)

    # Run the specified mode
    # 运行指定的模式
    if args.mode == 'train':
        run_training(config)
    elif args.mode == 'evaluate':
        run_evaluation(config)
    elif args.mode == 'extract_features':
        run_feature_extraction(config)
    elif args.mode == 'extract_embeddings':
        run_embedding_extraction(config)
    elif args.mode == 'visualize':
        run_visualization(config)
    elif args.mode == 'demo':
        run_demo(config)
    elif args.mode == 'benchmark':
        run_benchmark(config)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()