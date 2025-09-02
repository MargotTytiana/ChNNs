import os
import json
import argparse
from typing import Dict, Any, Optional


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration parameters

    获取默认配置参数

    Returns:
        Dict[str, Any]: Default configuration dictionary
                       默认配置字典
    """
    config = {
        # General settings
        # 一般设置
        "seed": 42,
        "device": "cpu",  # 'cuda' or 'cpu'

        # Data settings
        # 数据设置
        "train_dir": "P:/PycharmProjects/pythonProject1/dataset",
        "dev_dir": "P:/PycharmProjects/pythonProject1/devDataset",
        "test_dir": "P:/PycharmProjects/pythonProject1/testDataset",
        "segment_length": 3.0,  # Audio segment length in seconds
        "sampling_rate": 16000,  # Audio sampling rate
        "batch_size": 32,
        "num_workers": 4,

        # Model settings
        # 模型设置
        "chaotic_feature_dim": 64,  # 新增参数：混沌特征维度
        "chaotic_dim": 3,  # 新增参数：混沌系统维度
        "trajectory_points": 100,  # 新增参数：轨迹点数量
        "embedding_dim": 256,
        "use_chaotic_embedding": True,
        "use_attractor_pooling": True,
        "system_type": "lorenz",  # 'lorenz', 'rossler', or 'chua'

        # Training settings
        # 训练设置
        "num_epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "lr_patience": 5,  # Patience for learning rate scheduler
        "ce_weight": 1.0,  # Cross-entropy loss weight
        "triplet_weight": 0.1,  # Triplet loss weight
        "triplet_margin": 0.2,  # Margin for triplet loss

        # Logging and checkpoint settings
        # 日志和检查点设置
        "log_dir": "logs",
        "checkpoint_dir": "checkpoints",
        "output_dir": "output",
        "plot_dir": "plots",
        "log_interval": 10,  # How often to log training progress (in batches)
        "checkpoint_interval": 5,  # How often to save checkpoints (in epochs)
        "plot_interval": 5,  # How often to plot training curves (in epochs)
        "resume_checkpoint": None,  # Path to checkpoint to resume from

        # Evaluation settings
        # 评估设置
        "visualize_embeddings": True,  # Whether to visualize embeddings using t-SNE

        # Phase space reconstruction settings
        # 相空间重构设置
        "max_delay": 100,
        "max_dim": 10,
        "delay_method": "autocorr",  # 'autocorr' or 'mutual_info'

        # Chaotic feature extraction settings
        # 混沌特征提取设置
        "mlsa_scales": [1, 2, 4, 8, 16],
        "rqa_threshold": None,  # If None, it will be estimated automatically

        # Attractor pooling settings
        # 吸引子池化设置
        "pooling_type": "combined",  # 'topological', 'statistical', or 'combined'
        "epsilon_range": [0.01, 0.1, 0.5, 1.0],
        "use_correlation_dim": True,
        "use_lyapunov_dim": True,
        "use_entropy": True
    }

    return config


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or return default configuration

    从文件加载配置或返回默认配置

    Args:
        config_path (str, optional): Path to configuration file
                                    配置文件的路径

    Returns:
        Dict[str, Any]: Configuration dictionary
                       配置字典
    """
    config = get_default_config()

    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)

    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file

    将配置保存到文件

    Args:
        config (Dict[str, Any]): Configuration dictionary
                                配置字典
        config_path (str): Path to save configuration
                          保存配置的路径
    """
    # Create directory if it doesn't exist
    # 如果目录不存在，则创建目录
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments and update configuration

    解析命令行参数并更新配置

    Returns:
        Dict[str, Any]: Updated configuration dictionary
                       更新后的配置字典
    """
    parser = argparse.ArgumentParser(description="Chaotic Neural Network Speaker Recognition")

    # General settings
    # 一般设置
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--device', type=str, help='Device to use (cuda or cpu)')

    # Data settings
    # 数据设置
    parser.add_argument('--train_dir', type=str, help='Training data directory')
    parser.add_argument('--dev_dir', type=str, help='Development data directory')
    parser.add_argument('--test_dir', type=str, help='Test data directory')
    parser.add_argument('--segment_length', type=float, help='Audio segment length in seconds')
    parser.add_argument('--sampling_rate', type=int, help='Audio sampling rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')

    # Model settings
    # 模型设置
    parser.add_argument('--chaotic_feature_dim', type=int, help='Chaotic feature dimension')
    parser.add_argument('--chaotic_dim', type=int, help='Chaotic system dimension')
    parser.add_argument('--trajectory_points', type=int, help='Number of trajectory points')
    parser.add_argument('--embedding_dim', type=int, help='Speaker embedding dimension')
    parser.add_argument('--use_chaotic_embedding', type=bool, help='Whether to use chaotic embedding')
    parser.add_argument('--use_attractor_pooling', type=bool, help='Whether to use attractor pooling')
    parser.add_argument('--system_type', type=str, help='Chaotic system type (lorenz, rossler, chua)')

    # Training settings
    # 训练设置
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--lr_patience', type=int, help='Patience for learning rate scheduler')
    parser.add_argument('--ce_weight', type=float, help='Cross-entropy loss weight')
    parser.add_argument('--triplet_weight', type=float, help='Triplet loss weight')
    parser.add_argument('--triplet_margin', type=float, help='Margin for triplet loss')

    # Logging and checkpoint settings
    # 日志和检查点设置
    parser.add_argument('--log_dir', type=str, help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--plot_dir', type=str, help='Plot directory')
    parser.add_argument('--log_interval', type=int, help='How often to log training progress (in batches)')
    parser.add_argument('--checkpoint_interval', type=int, help='How often to save checkpoints (in epochs)')
    parser.add_argument('--plot_interval', type=int, help='How often to plot training curves (in epochs)')
    parser.add_argument('--resume_checkpoint', type=str, help='Path to checkpoint to resume from')

    # Evaluation settings
    # 评估设置
    parser.add_argument('--visualize_embeddings', type=bool, help='Whether to visualize embeddings using t-SNE')

    # Phase space reconstruction settings
    # 相空间重构设置
    parser.add_argument('--max_delay', type=int, help='Maximum delay for phase space reconstruction')
    parser.add_argument('--max_dim', type=int, help='Maximum dimension for phase space reconstruction')
    parser.add_argument('--delay_method', type=str, help='Method to estimate delay (autocorr or mutual_info)')

    # Chaotic feature extraction settings
    # 混沌特征提取设置
    parser.add_argument('--mlsa_scales', type=int, nargs='+', help='Scales for MLSA')
    parser.add_argument('--rqa_threshold', type=float, help='Threshold for RQA')

    # Attractor pooling settings
    # 吸引子池化设置
    parser.add_argument('--pooling_type', type=str, help='Pooling type (topological, statistical, combined)')
    parser.add_argument('--epsilon_range', type=float, nargs='+', help='Epsilon range for correlation dimension')
    parser.add_argument('--use_correlation_dim', type=bool, help='Whether to use correlation dimension')
    parser.add_argument('--use_lyapunov_dim', type=bool, help='Whether to use Lyapunov dimension')
    parser.add_argument('--use_entropy', type=bool, help='Whether to use Kolmogorov entropy')

    # Parse arguments
    # 解析参数
    args = parser.parse_args()

    # Get default configuration
    # 获取默认配置
    config = get_default_config()

    # Update configuration with command line arguments
    # 使用命令行参数更新配置
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            config[arg] = value

    return config


def create_experiment_config(
        experiment_name: str,
        base_config: Optional[Dict[str, Any]] = None,** kwargs
) -> Dict[str, Any]:
    """
    Create configuration for a specific experiment

    为特定实验创建配置

    Args:
        experiment_name (str): Name of the experiment
                              实验名称
        base_config (Dict[str, Any], optional): Base configuration to extend
                                               要扩展的基础配置
        **kwargs: Additional configuration parameters
                 附加配置参数

    Returns:
        Dict[str, Any]: Experiment configuration
                       实验配置
    """
    # Get base configuration
    # 获取基础配置
    if base_config is None:
        config = get_default_config()
    else:
        config = base_config.copy()

    # Update with experiment-specific settings
    # 使用实验特定设置更新
    config.update(kwargs)

    # Create experiment-specific directories
    # 创建实验特定目录
    for dir_key in ['log_dir', 'checkpoint_dir', 'output_dir', 'plot_dir']:
        config[dir_key] = os.path.join(config[dir_key], experiment_name)

    return config


def get_system_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get configurations for different chaotic systems

    获取不同混沌系统的配置

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of system configurations
                                  系统配置字典
    """
    base_config = get_default_config()

    # Lorenz system configuration
    # 洛伦兹系统配置
    lorenz_config = create_experiment_config(
        experiment_name="lorenz",
        base_config=base_config,
        system_type="lorenz",
        chaotic_dim=3
    )

    # Rössler system configuration
    # Rössler系统配置
    rossler_config = create_experiment_config(
        experiment_name="rossler",
        base_config=base_config,
        system_type="rossler",
        chaotic_dim=3
    )

    # Chua's circuit configuration
    # 蔡氏电路配置
    chua_config = create_experiment_config(
        experiment_name="chua",
        base_config=base_config,
        system_type="chua",
        chaotic_dim=3
    )

    return {
        "lorenz": lorenz_config,
        "rossler": rossler_config,
        "chua": chua_config
    }


if __name__ == "__main__":
    # Get default configuration
    # 获取默认配置
    default_config = get_default_config()
    print("Default configuration:")
    print(json.dumps(default_config, indent=2))

    # Save default configuration to file
    # 将默认配置保存到文件
    os.makedirs("configs", exist_ok=True)
    save_config(default_config, "configs/default.json")

    # Create experiment configurations
    # 创建实验配置
    system_configs = get_system_configs()

    for system_name, system_config in system_configs.items():
        config_path = f"configs/{system_name}.json"
        save_config(system_config, config_path)
        print(f"Saved {system_name} configuration to {config_path}")

    # Create a custom experiment configuration
    # 创建自定义实验配置
    custom_config = create_experiment_config(
        experiment_name="version_0.0",
        base_config=default_config,
        batch_size=64,
        learning_rate=0.0005,
        embedding_dim=512,
        use_chaotic_embedding=True,
        use_attractor_pooling=True,
        system_type="lorenz"
    )

    save_config(custom_config, "configs/ver_0.1.json")
    print("Saved custom experiment configuration to configs/ver_0.1.json")
