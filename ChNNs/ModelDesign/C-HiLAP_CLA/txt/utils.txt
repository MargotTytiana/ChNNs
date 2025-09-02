import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import random
from scipy import signal
import soundfile as sf
import time
import json


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility

    设置随机种子以确保可重复性

    Args:
        seed (int): Random seed
                   随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logger(log_dir: str, name: str = "chaotic_speaker_recognition") -> logging.Logger:
    """
    Set up logger for the project

    为项目设置日志记录器

    Args:
        log_dir (str): Directory to save log files
                      保存日志文件的目录
        name (str): Logger name
                   日志记录器名称

    Returns:
        logging.Logger: Configured logger
                       配置好的日志记录器
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # Create logger
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers
    # 创建处理器
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Create formatter
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_audio(file_path: str, sr: int = 16000) -> np.ndarray:
    """
    Load audio file with librosa

    使用librosa加载音频文件

    Args:
        file_path (str): Path to audio file
                        音频文件路径
        sr (int): Target sampling rate
                 目标采样率

    Returns:
        np.ndarray: Audio signal
                   音频信号
    """
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


def save_audio(audio: np.ndarray, file_path: str, sr: int = 16000) -> None:
    """
    Save audio file with soundfile

    使用soundfile保存音频文件

    Args:
        audio (np.ndarray): Audio signal
                          音频信号
        file_path (str): Path to save audio file
                        保存音频文件的路径
        sr (int): Sampling rate
                 采样率
    """
    # Create directory if it doesn't exist
    # 如果目录不存在，则创建目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save audio file
    # 保存音频文件
    sf.write(file_path, audio, sr)


def plot_waveform(audio: np.ndarray, sr: int = 16000, title: str = "Audio Waveform") -> plt.Figure:
    """
    Plot audio waveform

    绘制音频波形

    Args:
        audio (np.ndarray): Audio signal
                          音频信号
        sr (int): Sampling rate
                 采样率
        title (str): Plot title
                    图表标题

    Returns:
        plt.Figure: Figure object
                   图表对象
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig


def plot_spectrogram(audio: np.ndarray, sr: int = 16000, title: str = "Spectrogram") -> plt.Figure:
    """
    Plot audio spectrogram

    绘制音频频谱图

    Args:
        audio (np.ndarray): Audio signal
                          音频信号
        sr (int): Sampling rate
                 采样率
        title (str): Plot title
                    图表标题

    Returns:
        plt.Figure: Figure object
                   图表对象
    """
    # Compute spectrogram
    # 计算频谱图
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)

    # Plot spectrogram
    # 绘制频谱图
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig


def plot_mel_spectrogram(audio: np.ndarray, sr: int = 16000, n_mels: int = 128,
                         title: str = "Mel Spectrogram") -> plt.Figure:
    """
    Plot mel spectrogram

    绘制梅尔频谱图

    Args:
        audio (np.ndarray): Audio signal
                          音频信号
        sr (int): Sampling rate
                 采样率
        n_mels (int): Number of mel bands
                     梅尔频带数量
        title (str): Plot title
                    图表标题

    Returns:
        plt.Figure: Figure object
                   图表对象
    """
    # Compute mel spectrogram
    # 计算梅尔频谱图
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot mel spectrogram
    # 绘制梅尔频谱图
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None,
                          title: str = "Confusion Matrix") -> plt.Figure:
    """
    Plot confusion matrix

    绘制混淆矩阵

    Args:
        y_true (np.ndarray): True labels
                            真实标签
        y_pred (np.ndarray): Predicted labels
                            预测标签
        class_names (List[str], optional): List of class names
                                          类别名称列表
        title (str): Plot title
                    图表标题

    Returns:
        plt.Figure: Figure object
                   图表对象
    """
    # Compute confusion matrix
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    # 归一化混淆矩阵
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot confusion matrix
    # 绘制混淆矩阵
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    # 设置标签
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate x tick labels and set alignment
    # 旋转x轴刻度标签并设置对齐方式
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    # 遍历数据维度并创建文本注释
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")

    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, title: str = "ROC Curve") -> plt.Figure:
    """
    Plot ROC curve

    绘制ROC曲线

    Args:
        y_true (np.ndarray): True binary labels
                            真实二元标签
        y_score (np.ndarray): Target scores (probability estimates of the positive class)
                             目标分数（正类的概率估计）
        title (str): Plot title
                    图表标题

    Returns:
        plt.Figure: Figure object
                   图表对象
    """
    # Compute ROC curve and ROC area
    # 计算ROC曲线和ROC面积
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Create figure
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot ROC curve
    # 绘制ROC曲线
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_det_curve(y_true: np.ndarray, y_score: np.ndarray, title: str = "DET Curve") -> plt.Figure:
    """
    Plot Detection Error Tradeoff (DET) curve

    绘制检测错误权衡(DET)曲线

    Args:
        y_true (np.ndarray): True binary labels
                            真实二元标签
        y_score (np.ndarray): Target scores (probability estimates of the positive class)
                             目标分数（正类的概率估计）
        title (str): Plot title
                    图表标题

    Returns:
        plt.Figure: Figure object
                   图表对象
    """
    # Compute ROC curve
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Convert to false reject rate (FRR) and false accept rate (FAR)
    # 转换为虚警率(FRR)和误识率(FAR)
    frr = 1 - tpr  # False Reject Rate = 1 - True Positive Rate
    far = fpr  # False Accept Rate = False Positive Rate

    # Create figure
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot DET curve
    # 绘制DET曲线
    ax.plot(far, frr, color='darkorange', lw=2)

    # Find EER point (where FAR = FRR)
    # 找到EER点（FAR = FRR的点）
    abs_diff = np.abs(far - frr)
    min_idx = np.argmin(abs_diff)
    eer = (far[min_idx] + frr[min_idx]) / 2

    # Plot EER point
    # 绘制EER点
    ax.plot(far[min_idx], frr[min_idx], 'ro', label=f'EER = {eer:.4f}')

    # Set log scale for both axes
    # 为两个轴设置对数刻度
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set limits and labels
    # 设置限制和标签
    ax.set_xlim([0.001, 1.0])
    ax.set_ylim([0.001, 1.0])
    ax.set_xlabel('False Accept Rate (FAR)')
    ax.set_ylabel('False Reject Rate (FRR)')
    ax.set_title(title)
    ax.legend(loc="upper right")

    # Add grid
    # 添加网格
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_tsne_embeddings(embeddings: np.ndarray, labels: np.ndarray,
                         title: str = "t-SNE Visualization of Speaker Embeddings") -> plt.Figure:
    """
    Plot t-SNE visualization of embeddings

    绘制嵌入向量的t-SNE可视化

    Args:
        embeddings (np.ndarray): Embedding vectors
                                嵌入向量
        labels (np.ndarray): Labels
                            标签
        title (str): Plot title
                    图表标题

    Returns:
        plt.Figure: Figure object
                   图表对象
    """
    # Apply t-SNE dimensionality reduction
    # 应用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create figure
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get unique labels
    # 获取唯一标签
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    # Plot each label with a different color
    # 用不同的颜色绘制每个标签
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            color=colors[i],
            label=f"Speaker {label}",
            alpha=0.7
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig


def plot_recurrence_plot(time_series: np.ndarray, threshold: float = 0.1, title: str = "Recurrence Plot") -> plt.Figure:
    """
    Plot recurrence plot from time series

    从时间序列绘制递归图

    Args:
        time_series (np.ndarray): Time series data
                                 时间序列数据
        threshold (float): Threshold for recurrence
                          递归阈值
        title (str): Plot title
                    图表标题

    Returns:
        plt.Figure: Figure object
                   图表对象
    """
    # Ensure time series is 1D
    # 确保时间序列是一维的
    time_series = time_series.flatten()

    # Calculate distance matrix
    # 计算距离矩阵
    n = len(time_series)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist = abs(time_series[i] - time_series[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    # Create recurrence plot
    # 创建递归图
    recurrence_plot = dist_matrix < threshold

    # Create figure
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot recurrence plot
    # 绘制递归图
    ax.imshow(recurrence_plot, cmap='binary', origin='lower')
    ax.set_title(title)
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Time Index')
    plt.tight_layout()
    return fig


def plot_phase_space(trajectory: np.ndarray, title: str = "Phase Space Trajectory") -> plt.Figure:
    """
    Plot phase space trajectory

    绘制相空间轨迹

    Args:
        trajectory (np.ndarray): Phase space trajectory
                                相空间轨迹
        title (str): Plot title
                    图表标题

    Returns:
        plt.Figure: Figure object
                   图表对象
    """
    dim = trajectory.shape[1]

    if dim >= 3:
        # 3D phase space
        # 三维相空间
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    elif dim == 2:
        # 2D phase space
        # 二维相空间
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
    else:
        # 1D time series
        # 一维时间序列
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(trajectory[:, 0], 'b-')
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("x")
        ax.grid(True)

    plt.tight_layout()
    return fig


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                         train_metrics: Optional[List[float]] = None, val_metrics: Optional[List[float]] = None,
                         metric_name: str = "Accuracy", title: str = "Training Curves") -> plt.Figure:
    """
    Plot training and validation curves

    绘制训练和验证曲线

    Args:
        train_losses (List[float]): Training losses
                                   训练损失
        val_losses (List[float]): Validation losses
                                 验证损失
        train_metrics (List[float], optional): Training metrics
                                             训练指标
        val_metrics (List[float], optional): Validation metrics
                                           验证指标
        metric_name (str): Name of the metric
                          指标名称
        title (str): Plot title
                    图表标题

    Returns:
        plt.Figure: Figure object
                   图表对象
    """
    # Create figure with subplots
    # 创建带有子图的图表
    if train_metrics is not None and val_metrics is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot losses
    # 绘制损失
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot metrics if provided
    # 如果提供了指标，则绘制指标
    if train_metrics is not None and val_metrics is not None:
        ax2.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
        ax2.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
        ax2.set_title(f'{title} - {metric_name}')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel(metric_name)
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    return fig


def calculate_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER)

    计算等错误率(EER)

    Args:
        y_true (np.ndarray): True binary labels
                            真实二元标签
        y_score (np.ndarray): Target scores (probability estimates of the positive class)
                             目标分数（正类的概率估计）

    Returns:
        Tuple[float, float]: EER value and threshold
                            EER值和阈值
    """
    # Compute ROC curve
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Convert to false reject rate (FRR)
    # 转换为虚警率(FRR)
    frr = 1 - tpr

    # Find the point where FAR = FRR (EER)
    # 找到FAR = FRR的点（EER）
    abs_diff = np.abs(fpr - frr)
    min_idx = np.argmin(abs_diff)
    eer = (fpr[min_idx] + frr[min_idx]) / 2
    eer_threshold = thresholds[min_idx]

    return eer, eer_threshold


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics

    计算分类指标

    Args:
        y_true (np.ndarray): True labels
                            真实标签
        y_pred (np.ndarray): Predicted labels
                            预测标签

    Returns:
        Dict[str, float]: Dictionary of metrics
                         指标字典
    """
    # Calculate accuracy
    # 计算准确率
    accuracy = np.mean(y_true == y_pred)

    # Initialize metrics dictionary
    # 初始化指标字典
    metrics = {'accuracy': accuracy}

    # If binary classification, calculate additional metrics
    # 如果是二元分类，计算额外的指标
    if len(np.unique(y_true)) == 2:
        # Calculate true positives, false positives, true negatives, false negatives
        # 计算真阳性、假阳性、真阴性、假阴性
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Calculate precision, recall, F1 score
        # 计算精确率、召回率、F1分数
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    return metrics


def save_dict_to_json(d: Dict, json_path: str) -> None:
    """
    Save dictionary to JSON file

    将字典保存到JSON文件

    Args:
        d (Dict): Dictionary to save
                 要保存的字典
        json_path (str): Path to save JSON file
                        保存JSON文件的路径
    """
    # Create directory if it doesn't exist
    # 如果目录不存在，则创建目录
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Save dictionary to JSON file
    # 将字典保存到JSON文件
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4)


def load_dict_from_json(json_path: str) -> Dict:
    """
    Load dictionary from JSON file

    从JSON文件加载字典

    Args:
        json_path (str): Path to JSON file
                        JSON文件的路径

    Returns:
        Dict: Loaded dictionary
              加载的字典
    """
    # Load dictionary from JSON file
    # 从JSON文件加载字典
    with open(json_path, 'r') as f:
        d = json.load(f)

    return d


def vad_energy(audio: np.ndarray, frame_length: int = 1024, hop_length: int = 512,
               threshold: float = 0.03) -> np.ndarray:
    """
    Voice Activity Detection using energy threshold

    使用能量阈值进行语音活动检测

    Args:
        audio (np.ndarray): Audio signal
                          音频信号
        frame_length (int): Frame length
                           帧长
        hop_length (int): Hop length
                         跳跃长度
        threshold (float): Energy threshold
                          能量阈值

    Returns:
        np.ndarray: Audio with silence removed
                   去除静音后的音频
    """
    # Calculate energy
    # 计算能量
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Interpolate energy to match audio length
    # 将能量插值到与音频长度匹配
    frames = np.arange(len(energy))
    sample_points = np.linspace(0, len(energy) - 1, len(audio))
    energy_interpolated = np.interp(sample_points, frames, energy)

    # Create mask for non-silent parts
    # 为非静音部分创建掩码
    mask = energy_interpolated > threshold

    # Apply mask to audio
    # 将掩码应用于音频
    return audio[mask]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range

    将音频归一化到[-1, 1]范围

    Args:
        audio (np.ndarray): Audio signal
                          音频信号

    Returns:
        np.ndarray: Normalized audio
                   归一化的音频
    """
    # Normalize audio
    # 归一化音频
    return audio / (np.max(np.abs(audio)) + 1e-6)


def segment_audio(audio: np.ndarray, sr: int = 16000, segment_length: float = 3.0) -> List[np.ndarray]:
    """
    Segment audio into fixed-length segments

    将音频分割成固定长度的片段

    Args:
        audio (np.ndarray): Audio signal
                          音频信号
        sr (int): Sampling rate
                 采样率
        segment_length (float): Segment length in seconds
                               片段长度（秒）

    Returns:
        List[np.ndarray]: List of audio segments
                         音频片段列表
    """
    # Calculate segment length in samples
    # 计算片段长度（样本数）
    segment_samples = int(segment_length * sr)

    # Calculate number of segments
    # 计算片段数量
    num_segments = len(audio) // segment_samples

    # Segment audio
    # 分割音频
    segments = []
    for i in range(num_segments):
        start = i * segment_samples
        end = (i + 1) * segment_samples
        segment = audio[start:end]
        segments.append(segment)

    return segments


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Time stretch audio

    时间拉伸音频

    Args:
        audio (np.ndarray): Audio signal
                          音频信号
        rate (float): Stretch factor (1.0 = no stretch)
                     拉伸因子（1.0 = 不拉伸）

    Returns:
        np.ndarray: Time stretched audio
                   时间拉伸后的音频
    """
    # Time stretch audio
    # 时间拉伸音频
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """
    Pitch shift audio

    音高移动音频

    Args:
        audio (np.ndarray): Audio signal
                          音频信号
        sr (int): Sampling rate
                 采样率
        n_steps (float): Number of semitones to shift
                        要移动的半音数

    Returns:
        np.ndarray: Pitch shifted audio
                   音高移动后的音频
    """
    # Pitch shift audio
    # 音高移动音频
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def add_noise(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """
    Add random noise to audio

    向音频添加随机噪声

    Args:
        audio (np.ndarray): Audio signal
                          音频信号
        noise_level (float): Noise level
                            噪声水平

    Returns:
        np.ndarray: Noisy audio
                   带噪声的音频
    """
    # Generate random noise
    # 生成随机噪声
    noise = np.random.normal(0, noise_level, len(audio))

    # Add noise to audio
    # 向音频添加噪声
    noisy_audio = audio + noise

    # Normalize
    # 归一化
    return normalize_audio(noisy_audio)


def augment_audio(audio: np.ndarray, sr: int = 16000) -> Dict[str, np.ndarray]:
    """
    Apply various augmentations to audio

    对音频应用各种增强

    Args:
        audio (np.ndarray): Audio signal
                          音频信号
        sr (int): Sampling rate
                 采样率

    Returns:
        Dict[str, np.ndarray]: Dictionary of augmented audio
                              增强音频的字典
    """
    augmented = {}

    # Original audio
    # 原始音频
    augmented['original'] = audio

    # Time stretch (slower)
    # 时间拉伸（变慢）
    augmented['stretch_slow'] = time_stretch(audio, 0.9)

    # Time stretch (faster)
    # 时间拉伸（变快）
    augmented['stretch_fast'] = time_stretch(audio, 1.1)

    # Pitch shift (lower)
    # 音高移动（降低）
    augmented['pitch_down'] = pitch_shift(audio, sr, -2)

    # Pitch shift (higher)
    # 音高移动（提高）
    augmented['pitch_up'] = pitch_shift(audio, sr, 2)

    # Add noise
    # 添加噪声
    augmented['noisy'] = add_noise(audio, 0.005)

    return augmented


def extract_mfcc(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 20) -> np.ndarray:
    """
    Extract MFCC features from audio

    从音频中提取MFCC特征

    Args:
        audio (np.ndarray): Audio signal
                          音频信号
        sr (int): Sampling rate
                 采样率
        n_mfcc (int): Number of MFCCs
                     MFCC的数量

    Returns:
        np.ndarray: MFCC features
                   MFCC特征
    """
    # Extract MFCCs
    # 提取MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Add delta and delta-delta features
    # 添加一阶和二阶差分特征
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # Stack features
    # 堆叠特征
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])

    # Transpose to have time as first dimension
    # 转置，使时间成为第一维度
    features = features.T

    return features


def extract_melspectrogram(audio: np.ndarray, sr: int = 16000, n_mels: int = 128) -> np.ndarray:
    """
    Extract mel spectrogram features from audio

    从音频中提取梅尔频谱图特征

    Args:
        audio (np.ndarray): Audio signal
                          音频信号
        sr (int): Sampling rate
                 采样率
        n_mels (int): Number of mel bands
                     梅尔频带数量

    Returns:
        np.ndarray: Mel spectrogram features
                   梅尔频谱图特征
    """
    # Extract mel spectrogram
    # 提取梅尔频谱图
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)

    # Convert to dB scale
    # 转换为分贝刻度
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


if __name__ == "__main__":
    # Set random seed
    # 设置随机种子
    set_seed(42)

    # Setup logger
    # 设置日志记录器
    logger = setup_logger("logs", "chaotic_speaker_recognition_utils")
    logger.info("Testing utility functions...")

    # Generate a test signal (sine wave)
    # 生成测试信号（正弦波）
    sr = 16000
    t = np.linspace(0, 3, 3 * sr)
    audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)

    # Add some silence
    # 添加一些静音
    silence = np.zeros(sr)
    audio = np.concatenate([audio[:sr], silence, audio[sr:]])

    # Add some noise
    # 添加一些噪声
    noisy_audio = add_noise(audio, 0.1)

    # Apply VAD
    # 应用VAD
    vad_audio = vad_energy(noisy_audio)

    # Plot waveform
    # 绘制波形
    fig = plot_waveform(audio, sr)
    plt.savefig("waveform.png")
    plt.close(fig)

    # Plot spectrogram
    # 绘制频谱图
    fig = plot_spectrogram(audio, sr)
    plt.savefig("spectrogram.png")
    plt.close(fig)

    # Plot mel spectrogram
    # 绘制梅尔频谱图
    fig = plot_mel_spectrogram(audio, sr)
    plt.savefig("mel_spectrogram.png")
    plt.close(fig)

    # Extract MFCC features
    # 提取MFCC特征
    mfcc_features = extract_mfcc(audio, sr)
    logger.info(f"MFCC features shape: {mfcc_features.shape}")

    # Generate some dummy classification data
    # 生成一些虚拟分类数据
    y_true = np.random.randint(0, 5, 100)
    y_pred = y_true.copy()
    # Introduce some errors
    # 引入一些错误
    error_idx = np.random.choice(len(y_true), 20, replace=False)
    y_pred[error_idx] = np.random.randint(0, 5, 20)

    # Plot confusion matrix
    # 绘制混淆矩阵
    fig = plot_confusion_matrix(y_true, y_pred)
    plt.savefig("confusion_matrix.png")
    plt.close(fig)

    # Calculate metrics
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred)
    logger.info(f"Classification metrics: {metrics}")

    # Generate some dummy verification data
    # 生成一些虚拟验证数据
    y_true_bin = np.random.randint(0, 2, 100)
    y_score = np.random.rand(100)

    # Plot ROC curve
    # 绘制ROC曲线
    fig = plot_roc_curve(y_true_bin, y_score)
    plt.savefig("roc_curve.png")
    plt.close(fig)

    # Plot DET curve
    # 绘制DET曲线
    fig = plot_det_curve(y_true_bin, y_score)
    plt.savefig("det_curve.png")
    plt.close(fig)

    # Calculate EER
    # 计算EER
    eer, threshold = calculate_eer(y_true_bin, y_score)
    logger.info(f"EER: {eer:.4f}, Threshold: {threshold:.4f}")

    # Generate some dummy embedding data
    # 生成一些虚拟嵌入数据
    embeddings = np.random.rand(50, 10)
    labels = np.random.randint(0, 5, 50)

    # Plot t-SNE visualization
    # 绘制t-SNE可视化
    fig = plot_tsne_embeddings(embeddings, labels)
    plt.savefig("tsne_embeddings.png")
    plt.close(fig)

    # Generate some dummy training history
    # 生成一些虚拟训练历史
    train_losses = [0.9, 0.7, 0.5, 0.3, 0.2]
    val_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
    train_acc = [0.5, 0.7, 0.8, 0.9, 0.95]
    val_acc = [0.4, 0.6, 0.7, 0.8, 0.85]

    # Plot training curves
    # 绘制训练曲线
    fig = plot_training_curves(train_losses, val_losses, train_acc, val_acc)
    plt.savefig("training_curves.png")
    plt.close(fig)

    logger.info("All utility functions tested successfully!")