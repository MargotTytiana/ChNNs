import os
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import librosa
import soundfile as sf
from scipy.linalg import svd
from scipy.signal import find_peaks
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置全局绘图样式
plt.style.use('seaborn')
sns.color_palette('deep')


def plot_training_history(
        train_loss: list,
        val_loss: list,
        train_acc: list = None,
        val_acc: list = None,
        learning_rate: list = None,
        save_path: str = 'training_history.png'
) -> None:
    """
    绘制训练历史记录

    Args:
        train_loss: 训练损失列表
        val_loss: 验证损失列表
        train_acc: 训练准确率列表 (可选)
        val_acc: 验证准确率列表 (可选)
        learning_rate: 学习率列表 (可选)
        save_path: 保存路径
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制损失曲线
    ax1.plot(train_loss, label='Train Loss', color='dodgerblue', linewidth=2)
    ax1.plot(val_loss, label='Validation Loss', color='coral', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left', frameon=True, fontsize=10)

    # 创建第二个y轴用于准确率
    if train_acc is not None and val_acc is not None:
        ax2 = ax1.twinx()
        ax2.plot(train_acc, label='Train Accuracy', color='mediumseagreen', linestyle='--', linewidth=2)
        ax2.plot(val_acc, label='Validation Accuracy', color='purple', linestyle='--', linewidth=2)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.legend(loc='upper right', frameon=True, fontsize=10)

    # 创建第三个y轴用于学习率
    if learning_rate is not None:
        ax3 = ax1.twinx()
        # 调整右侧轴位置
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(learning_rate, label='Learning Rate', color='orange', linestyle=':', linewidth=2)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='tab:orange')
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True, fontsize=10)

    plt.title('Training History', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"训练历史图已保存至 {save_path}")


def plot_det_curve(
        genuine_scores: list,
        impostor_scores: list,
        save_path: str = 'det_curve.png'
) -> None:
    """
    绘制检测错误权衡曲线

    Args:
        genuine_scores: 真分数列表
        impostor_scores: 假分数列表
        save_path: 保存路径
    """
    # 计算EER
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # 计算FAR, FRR
    far, frr, threshold = compute_eer(genuine_scores, impostor_scores)

    # 生成DET曲线数据
    thresholds = np.linspace(min(np.min(genuine_scores), np.min(impostor_scores)),
                             max(np.max(genuine_scores), np.max(impostor_scores)), 1000)

    far_curve = []
    frr_curve = []
    for thresh in thresholds:
        frr_curve.append(np.mean(genuine_scores < thresh))
        far_curve.append(np.mean(impostor_scores >= thresh))

    # 转换为百分比
    far_curve = np.array(far_curve) * 100
    frr_curve = np.array(frr_curve) * 100

    # 绘制DET曲线
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(far_curve, frr_curve, color='royalblue', linewidth=2)
    ax.plot(far, frr, 'ro', markersize=8, label=f'EER: {far * 100:.2f}%')
    ax.set_xlabel('False Acceptance Rate (%)', fontsize=12)
    ax.set_ylabel('False Rejection Rate (%)', fontsize=12)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.title('Detection Error Trade-off (DET) Curve', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"DET曲线已保存至 {save_path}")


def plot_embedding_space(
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = 'tsne',
        n_components: int = 2,
        save_path: str = 'embedding_space.png',
        title: str = 'Speaker Embedding Space'
) -> None:
    """
    可视化嵌入空间

    Args:
        embeddings: 嵌入向量，形状为 (n_samples, n_features)
        labels: 标签数组，形状为 (n_samples,)
        method: 降维方法，'tsne' 或 'pca'
        n_components: 降维后的维度，2 或 3
        save_path: 保存路径
        title: 图标题
    """
    # 降维
    if method == 'tsne':
        model = TSNE(n_components=n_components, random_state=42)
    elif method == 'pca':
        model = PCA(n_components=n_components)
    else:
        raise ValueError(f"不支持的降维方法: {method}")

    reduced_embeddings = model.fit_transform(embeddings)

    # 创建图形
    fig = plt.figure(figsize=(10, 8))

    if n_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                             c=labels, cmap='tab20', s=20, alpha=0.8)

        # 添加图例
        handles, unique_labels = scatter.legend_elements(prop="sizes", num=len(np.unique(labels)))
        ax.legend(handles, unique_labels, title="Speakers", loc="upper right")

    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2],
                             c=labels, cmap='tab20', s=20, alpha=0.8)

        # 添加图例
        legend_elements = [plt.Line2D([0, 0], [0, 1], color=cmap(i), marker='o', linestyle='')
                           for i in range(len(np.unique(labels)))]
        ax.legend(legend_elements, unique_labels, title="Speakers", loc='upper right')

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    if n_components == 3:
        ax.set_zlabel('Dimension 3')

    plt.title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"嵌入空间可视化已保存至 {save_path}")


def plot_phase_space(
        signal: np.ndarray,
        delay: int = 10,
        embed_dim: int = 3,
        save_path: str = 'phase_space.png',
        title: str = 'Phase Space Trajectory'
) -> None:
    """
    绘制相空间轨迹图

    Args:
        signal: 输入信号
        delay: 相空间重构的时间延迟
        embed_dim: 嵌入维度
        save_path: 保存路径
        title: 图标题
    """
    # 相空间重构
    n = len(signal)
    m = embed_dim
    tau = delay

    # 检查信号长度是否足够
    if n < (m - 1) * tau + 1:
        raise ValueError(f"信号太短，无法进行相空间重构。需要至少 {(m - 1) * tau + 1} 个点，但信号只有 {n} 个点")

    # 构建相空间
    phase_space = np.zeros((n - (m - 1) * tau, m))
    for i in range(n - (m - 1) * tau):
        for j in range(m):
            phase_space[i, j] = signal[i + j * tau]

    # 创建图形
    fig = plt.figure(figsize=(10, 8))

    if m == 2:
        ax = fig.add_subplot(111)
        ax.plot(phase_space[:, 0], phase_space[:, 1], color='darkorange', linewidth=1.5)
        ax.set_xlabel('X(t)')
        ax.set_ylabel('X(t + {})'.format(tau))
    elif m == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(phase_space[:, 0], phase_space[:, 1], phase_space[:, 2], color='darkorange', linewidth=1.5)
        ax.set_xlabel('X(t)')
        ax.set_ylabel('X(t + {})'.format(tau))
        ax.set_zlabel('X(t + {})'.format(2 * tau))
    else:
        raise NotImplementedError("目前只支持2D和3D相空间图")

    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"相空间轨迹图已保存至 {save_path}")


def plot_recurrence_plot(
        signal: np.ndarray,
        delay: int = 10,
        embed_dim: int = 2,
        threshold: float = 0.1,
        save_path: str = 'recurrence_plot.png',
        title: str = 'Recurrence Plot'
) -> None:
    """
    绘制递归图

    Args:
        signal: 输入信号
        delay: 时间延迟
        embed_dim: 嵌入维度
        threshold: 递归阈值
        save_path: 保存路径
        title: 图标题
    """
    # 相空间重构
    n = len(signal)
    m = embed_dim
    tau = delay

    # 检查信号长度
    if n < (m - 1) * tau + 1:
        raise ValueError(f"信号太短，无法进行相空间重构。需要至少 {(m - 1) * tau + 1} 个点，但信号只有 {n} 个点")

    # 构建相空间
    phase_space = np.zeros((n - (m - 1) * tau, m))
    for i in range(n - (m - 1) * tau):
        for j in range(m):
            phase_space[i, j] = signal[i + j * tau]

    # 计算距离矩阵
    dist_matrix = np.zeros((len(phase_space), len(phase_space)))
    for i in range(len(phase_space)):
        for j in range(len(phase_space)):
            if i == j:
                dist = 0
            else:
                dist = np.linalg.norm(phase_space[i] - phase_space[j])
            dist_matrix[i, j] = dist

    # 创建递归矩阵
    recurrence_matrix = dist_matrix < threshold

    # 绘制递归图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(recurrence_matrix, cmap='binary', origin='lower', interpolation='none')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Time Index')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 添加颜色条
    plt.colorbar(ax.imshow, ax=ax, orientation='vertical', pad=0.1)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"递归图已保存至 {save_path}")


def plot_lyapunov_spectrum(
        signal: np.ndarray,
        window_size: int = 1024,
        step_size: int = 512,
        save_path: str = 'lyapunov_spectrum.png',
        title: str = 'Lyapunov Exponent Spectrum'
) -> None:
    """
    绘制李雅普诺夫指数谱

    Args:
        signal: 输入信号
        window_size: 窗口大小
        step_size: 步长
        save_path: 保存路径
        title: 图标题
    """
    # 计算每个窗口的李雅普诺夫指数
    lyap_exponents = []
    timestamps = []

    for start in range(0, len(signal) - window_size, step_size):
        window = signal[start:start + window_size]
        try:
            lyap = compute_lyapunov_exponent(window)
            lyap_exponents.append(lyap)
            timestamps.append(start + window_size // 2)
        except Exception as e:
            logger.warning(f"在窗口 {start} 到 {start + window_size} 计算李雅普诺夫指数时出错: {e}")
            lyap_exponents.append(0)
            timestamps.append(start + window_size // 2)

    # 绘制李雅普诺夫指数谱
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timestamps, lyap_exponents, color='forestgreen', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Lyapunov Exponent')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"李雅普诺夫指数谱图已保存至 {save_path}")


def compute_eer(
        genuine_scores: list,
        impostor_scores: list
) -> Tuple[float, float, float]:
    """
    计算等错误率 (EER) 及其对应的阈值

    Args:
        genuine_scores: 真分数列表
        impostor_scores: 假分数列表

    Returns:
        eer: 等错误率
        threshold: 等错误率对应的阈值
        mindcf: 最小检测代价函数
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # 排序
    genuine_scores.sort()
    impostor_scores.sort()

    # 找到EER点
    eer_threshold = find_eer_threshold(genuine_scores, impostor_scores)

    # 计算FAR和FRR
    far = np.mean(impostor_scores >= eer_threshold)
    frr = np.mean(genuine_scores < eer_threshold)

    # 计算EER
    eer = max(far, frr)

    # 计算minDCF
    p_target = 0.01  # 目标先验概率
    mindcf = compute_mindcf(genuine_scores, impostor_scores, p_target)

    return eer, frr, eer_threshold, mindcf


def find_eer_threshold(
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray
) -> float:
    """
    找到EER对应的阈值

    Args:
        genuine_scores: 真分数数组
        impostor_scores: 假分数数组

    Returns:
        阈值
    """
    # 生成所有可能的阈值
    thresholds = np.sort(np.unique(np.concatenate([genuine_scores, impostor_scores])))

    # 初始化最小代价和最佳阈值
    min_cross_ratio = float('inf')
    best_threshold = 0.0

    for threshold in thresholds:
        # 计算FAR和FRR
        far = np.mean(impostor_scores >= threshold)
        frr = np.mean(genuine_scores < threshold)

        # 计算交叉比率 (一个合理的EER替代指标)
        cross_ratio = max(far, frr)

        if cross_ratio < min_cross_ratio:
            min_cross_ratio = cross_ratio
            best_threshold = threshold

    return best_threshold


def compute_mindcf(
        genuine_scores: list,
        impostor_scores: list,
        p_target: float = 0.01,
        c_miss: float = 1.0,
        c_fa: float = 1.0
) -> float:
    """
    计算最小检测代价函数 (minDCF)

    Args:
        genuine_scores: 真分数列表
        impostor_scores: 假分数列表
        p_target: 目标先验概率
        c_miss: 漏识别的代价
        c_fa: 误识别的代价

    Returns:
        最小检测代价
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # 生成所有可能的阈值
    thresholds = np.sort(np.unique(np.concatenate([genuine_scores, impostor_scores])))

    min_dcf = float('inf')

    for threshold in thresholds:
        # 计算漏识别概率 (P(Miss) = P(Target) * FRR)
        p_miss = p_target * np.mean(genuine_scores < threshold)

        # 计算误识别概率 (P(FA) = (1 - P(Target)) * FAR)
        p_fa = (1 - p_target) * np.mean(impostor_scores >= threshold)

        # 计算检测代价
        dcf = c_miss * p_miss + c_fa * p_fa

        if dcf < min_dcf:
            min_dcf = dcf

    return min_dcf


def visualize_chaos_in_time_domain(
        signal: np.ndarray,
        sample_rate: int = 16000,
        save_path: str = 'chaos_time_domain.png',
        title: str = 'Time Domain Signal with Chaos Characteristics'
) -> None:
    """
    可视化时域信号中的混沌特性

    Args:
        signal: 输入信号
        sample_rate: 采样率
        save_path: 保存路径
        title: 图标题
    """
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 原始信号
    time = np.arange(len(signal)) / sample_rate
    ax1.plot(time, signal, color='blue', linewidth=1.0, alpha=0.8)
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original Signal')

    # 放大视图显示细节
    zoom_start = len(signal) // 2
    zoom_end = zoom_start + 500
    ax2.plot(time[zoom_start:zoom_end], signal[zoom_start:zoom_end], color='red', linewidth=1.5)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Zoomed View Showing Nonlinear Behavior')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"时域混沌特性图已保存至 {save_path}")


def visualize_speaker_attractors(
        all_embeddings: np.ndarray,
        labels: np.ndarray,
        n_speakers: int = 10,
        save_path: str = 'speaker_attractors.png',
        title: str = 'Speaker Attractors in Phase Space'
) -> None:
    """
    可视化说话人吸引子

    Args:
        all_embeddings: 所有嵌入向量，形状为 (n_samples, n_features)
        labels: 标签数组，形状为 (n_samples,)
        n_speakers: 要可视化的说话人数量
        save_path: 保存路径
        title: 图标题
    """
    # 获取前n_speakers的说话人
    unique_labels = np.unique(labels)
    if n_speakers > len(unique_labels):
        n_speakers = len(unique_labels)

    selected_labels = unique_labels[:n_speakers]

    # 创建图形
    fig = plt.figure(figsize=(15, 10))

    # 使用SVD降维到2D
    for i, label in enumerate(selected_labels):
        # 获取当前说话人的所有嵌入
        speaker_embeddings = all_embeddings[labels == label]

        # 使用SVD降维
        U, S, Vt = svd(speaker_embeddings, full_matrices=False)
        reduced_embeddings = U[:, :2] * S[:2]

        # 在2D空间中绘制轨迹
        ax = fig.add_subplot(2, (n_speakers + 1) // 2, i + 1)
        ax.plot(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 'o-', markersize=4)
        ax.set_title(f'Speaker {label}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"说话人吸引子图已保存至 {save_path}")


# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    n_samples = 1000
    n_features = 64
    n_speakers = 10

    # 生成随机嵌入和标签
    np.random.seed(42)
    embeddings = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, n_speakers, size=n_samples)

    # 测试可视化函数
    plot_training_history(
        train_loss=[0.5, 0.3, 0.2, 0.15, 0.1],
        val_loss=[0.6, 0.4, 0.25, 0.18, 0.12],
        train_acc=[0.8, 0.85, 0.88, 0.9, 0.92],
        val_acc=[0.78, 0.82, 0.86, 0.88, 0.9],
        learning_rate=[0.1, 0.01, 0.001, 0.0001, 0.00001],
        save_path="training_history example.png"
    )

    plot_det_curve(
        genuine_scores=np.random.randn(500),
        impostor_scores=np.random.randn(500),
        save_path="det_curve example.png"
    )

    plot_embedding_space(
        embeddings=embeddings,
        labels=labels,
        method='tsne',
        n_components=2,
        save_path="embedding_space example.png",
        title="2D t-SNE Visualization of Speaker Embeddings"
    )

    plot_phase_space(
        signal=np.sin(np.linspace(0, 10, 1000)) + 0.1 * np.random.randn(1000),
        delay=5,
        embed_dim=3,
        save_path="phase_space example.png"
    )

    plot_recurrence_plot(
        signal=np.sin(np.linspace(0, 10, 1000)) + 0.1 * np.random.randn(1000),
        delay=5,
        embed_dim=2,
        threshold=0.2,
        save_path="recurrence_plot example.png"
    )

    plot_lyapunov_spectrum(
        signal=np.sin(np.linspace(0, 10, 1000)) + 0.1 * np.random.randn(1000),
        window_size=512,
        step_size=256,
        save_path="lyapunov_spectrum example.png"
    )

    visualize_chaos_in_time_domain(
        signal=np.sin(np.linspace(0, 10, 1000)) + 0.1 * np.random.randn(1000),
        sample_rate=16000,
        save_path="chaos_time_domain example.png"
    )

    visualize_speaker_attractors(
        all_embeddings=embeddings,
        labels=labels,
        n_speakers=5,
        save_path="speaker_attractors example.png"
    )