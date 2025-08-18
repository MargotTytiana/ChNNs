import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.special import entr
from typing import List, Tuple, Dict, Union, Optional


def compute_eer(scores_genuine: np.ndarray, scores_impostor: np.ndarray) -> Tuple[float, float]:
    """
    计算等错误率(Equal Error Rate, EER)和对应的阈值。

    EER是指当False Acceptance Rate (FAR)和False Rejection Rate (FRR)
    相等时的错误率。

    Args:
        scores_genuine: 合法配对的相似度分数
        scores_impostor: 冒充配对的相似度分数

    Returns:
        Tuple包含:
            - EER: 等错误率
            - threshold: 对应的决策阈值
    """
    # 确保输入是numpy数组
    if isinstance(scores_genuine, torch.Tensor):
        scores_genuine = scores_genuine.cpu().numpy()
    if isinstance(scores_impostor, torch.Tensor):
        scores_impostor = scores_impostor.cpu().numpy()

    # 将分数展平为一维数组
    scores_genuine = scores_genuine.flatten()
    scores_impostor = scores_impostor.flatten()

    # 创建标签数组
    labels_genuine = np.ones(len(scores_genuine))
    labels_impostor = np.zeros(len(scores_impostor))

    # 合并分数和标签
    scores = np.concatenate([scores_genuine, scores_impostor])
    labels = np.concatenate([labels_genuine, labels_impostor])

    # 计算不同阈值下的FAR和FRR
    def calculate_far_frr(threshold):
        pred = scores >= threshold
        far = np.sum((pred == 1) & (labels == 0)) / np.sum(labels == 0)
        frr = np.sum((pred == 0) & (labels == 1)) / np.sum(labels == 1)
        return far, frr

    # 寻找FAR=FRR的阈值
    thresholds = np.sort(scores)
    fars, frrs = [], []

    for threshold in thresholds:
        far, frr = calculate_far_frr(threshold)
        fars.append(far)
        frrs.append(frr)

    # 转换为numpy数组
    fars = np.array(fars)
    frrs = np.array(frrs)

    # 找到FAR和FRR最接近的点
    abs_diff = np.abs(fars - frrs)
    idx = np.argmin(abs_diff)
    eer = (fars[idx] + frrs[idx]) / 2
    threshold = thresholds[idx]

    # 使用线性插值获得更精确的EER
    if len(fars) > 1 and len(frrs) > 1:
        try:
            eer = brentq(lambda x: interp1d(thresholds, fars, bounds_error=False, fill_value=(0, 1))(x) -
                                   interp1d(thresholds, frrs, bounds_error=False, fill_value=(0, 1))(x),
                         thresholds[0], thresholds[-1])

            threshold = interp1d(fars, thresholds, bounds_error=False, fill_value=(thresholds[0], thresholds[-1]))(eer)
        except ValueError:
            # 如果brentq失败，使用之前计算的近似值
            pass

    return float(eer), float(threshold)


def compute_mindcf(
        scores_genuine: np.ndarray,
        scores_impostor: np.ndarray,
        p_target: float = 0.01,
        c_miss: float = 1.0,
        c_fa: float = 1.0
) -> Tuple[float, float]:
    """
    计算最小检测代价函数(Minimum Detection Cost Function, minDCF)。

    minDCF是NIST说话人识别评估中使用的标准指标，它考虑了错误拒绝和错误接受的相对代价。

    Args:
        scores_genuine: 合法配对的相似度分数
        scores_impostor: 冒充配对的相似度分数
        p_target: 目标说话人先验概率
        c_miss: 错误拒绝的代价
        c_fa: 错误接受的代价

    Returns:
        Tuple包含:
            - minDCF: 最小检测代价
            - threshold: 对应的最优决策阈值
    """
    # 确保输入是numpy数组
    if isinstance(scores_genuine, torch.Tensor):
        scores_genuine = scores_genuine.cpu().numpy()
    if isinstance(scores_impostor, torch.Tensor):
        scores_impostor = scores_impostor.cpu().numpy()

    # 将分数展平为一维数组
    scores_genuine = scores_genuine.flatten()
    scores_impostor = scores_impostor.flatten()

    # 创建标签数组
    labels_genuine = np.ones(len(scores_genuine))
    labels_impostor = np.zeros(len(scores_impostor))

    # 合并分数和标签
    scores = np.concatenate([scores_genuine, scores_impostor])
    labels = np.concatenate([labels_genuine, labels_impostor])

    # 计算归一化代价
    p_non_target = 1.0 - p_target
    c_miss_norm = c_miss * p_target
    c_fa_norm = c_fa * p_non_target

    # 计算不同阈值下的DCF
    thresholds = np.sort(scores)
    min_dcf = float('inf')
    best_threshold = 0.0

    for threshold in thresholds:
        pred = scores >= threshold
        miss = np.sum((pred == 0) & (labels == 1))
        fa = np.sum((pred == 1) & (labels == 0))

        # 计算错误率
        p_miss = miss / np.sum(labels == 1)
        p_fa = fa / np.sum(labels == 0)

        # 计算DCF
        dcf = c_miss_norm * p_miss + c_fa_norm * p_fa

        # 更新最小DCF
        if dcf < min_dcf:
            min_dcf = dcf
            best_threshold = threshold

    # 归一化minDCF
    min_dcf = min_dcf / min(c_miss_norm, c_fa_norm)

    return float(min_dcf), float(best_threshold)


def compute_entropy(signal: np.ndarray) -> float:
    """
    计算信号的香农熵。

    Args:
        signal: 输入信号

    Returns:
        信号的香农熵
    """
    # 确保输入是numpy数组
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()

    # 归一化信号
    signal = signal.flatten()
    signal = signal - np.min(signal)
    if np.max(signal) > 0:
        signal = signal / np.max(signal)

    # 使用直方图估计概率分布
    hist, bin_edges = np.histogram(signal, bins=100, density=True)
    hist = hist / np.sum(hist)

    # 计算香农熵
    entropy_value = -np.sum(hist * np.log2(hist + 1e-10))

    return float(entropy_value)


def compute_csi(
        eer_clean: float,
        eer_noisy: float,
        signal_clean: np.ndarray,
        signal_noisy: np.ndarray
) -> float:
    """
    计算混沌敏感性指数(Chaotic Sensitivity Index, CSI)。

    CSI = (EER_clean - EER_noisy) / ΔH
    其中ΔH是信号熵的变化率。

    Args:
        eer_clean: 干净条件下的EER
        eer_noisy: 噪声条件下的EER
        signal_clean: 干净信号
        signal_noisy: 噪声信号

    Returns:
        混沌敏感性指数
    """
    # 计算信号熵
    entropy_clean = compute_entropy(signal_clean)
    entropy_noisy = compute_entropy(signal_noisy)

    # 计算熵变化
    delta_h = abs(entropy_noisy - entropy_clean)

    # 避免除零错误
    if delta_h < 1e-10:
        return float('inf')

    # 计算CSI
    csi = (eer_clean - eer_noisy) / delta_h

    return float(csi)


def compute_lyapunov_exponent(signal: np.ndarray, embedding_dim: int = 10, delay: int = 2) -> float:
    """
    使用Rosenstein算法估计信号的最大李雅普诺夫指数。

    Args:
        signal: 输入信号
        embedding_dim: 相空间重构的嵌入维度
        delay: 时间延迟

    Returns:
        最大李雅普诺夫指数
    """
    # 确保输入是numpy数组
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()

    # 归一化信号
    signal = signal.flatten()
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

    # 相空间重构
    m = embedding_dim
    tau = delay
    n = len(signal)

    if n <= (m - 1) * tau:
        return 0.0

    Y = np.zeros((n - (m - 1) * tau, m))
    for i in range(n - (m - 1) * tau):
        for j in range(m):
            Y[i, j] = signal[i + j * tau]

    # 寻找最近邻
    mean_divergence = np.zeros(min(20, n // 2))
    neighbor_count = 0

    for i in range(Y.shape[0]):
        # 计算到所有其他点的距离
        distances = np.zeros(Y.shape[0])
        for j in range(Y.shape[0]):
            if abs(i - j) <= 10:  # 排除时间相近的点
                distances[j] = float('inf')
            else:
                distances[j] = np.sqrt(np.sum((Y[i] - Y[j]) ** 2))

        # 找到最近邻
        nearest_idx = np.argmin(distances)
        if distances[nearest_idx] == float('inf'):
            continue

        # 跟踪发散
        for k in range(min(20, n // 2)):
            if i + k < Y.shape[0] and nearest_idx + k < Y.shape[0]:
                d_k = np.sqrt(np.sum((Y[i + k] - Y[nearest_idx + k]) ** 2))
                if d_k > 0:
                    mean_divergence[k] += np.log(d_k)

        neighbor_count += 1

    if neighbor_count == 0:
        return 0.0

    # 平均发散
    mean_divergence = mean_divergence / neighbor_count

    # 线性回归找斜率(MLE)
    x = np.arange(len(mean_divergence))
    valid_idx = ~np.isnan(mean_divergence) & ~np.isinf(mean_divergence)

    if np.sum(valid_idx) < 2:
        return 0.0

    x = x[valid_idx]
    y = mean_divergence[valid_idx]

    # 线性拟合
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]

    return float(slope)


def compute_recurrence_plot(signal: np.ndarray, embedding_dim: int = 3, delay: int = 1,
                            threshold: float = 0.1) -> np.ndarray:
    """
    计算信号的递归图。

    Args:
        signal: 输入信号
        embedding_dim: 相空间重构的嵌入维度
        delay: 时间延迟
        threshold: 递归图阈值

    Returns:
        递归图矩阵
    """
    # 确保输入是numpy数组
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()

    # 归一化信号
    signal = signal.flatten()
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

    # 相空间重构
    m = embedding_dim
    tau = delay
    n = len(signal)

    if n <= (m - 1) * tau:
        return np.zeros((1, 1))

    Y = np.zeros((n - (m - 1) * tau, m))
    for i in range(n - (m - 1) * tau):
        for j in range(m):
            Y[i, j] = signal[i + j * tau]

    # 计算递归图
    n_points = Y.shape[0]
    rp = np.zeros((n_points, n_points), dtype=np.uint8)

    # 计算距离并阈值化
    for i in range(n_points):
        for j in range(i, n_points):
            dist = np.sqrt(np.sum((Y[i] - Y[j]) ** 2))
            if dist <= threshold:
                rp[i, j] = rp[j, i] = 1

    return rp


def compute_rqa_metrics(rp: np.ndarray) -> Dict[str, float]:
    """
    从递归图计算递归量化分析(RQA)指标。

    Args:
        rp: 递归图矩阵

    Returns:
        包含RQA指标的字典
    """
    n_points = rp.shape[0]

    # 递归率(RR)
    rr = np.sum(rp) / (n_points ** 2)

    # 寻找对角线(简化方法)
    min_diag_length = 2
    diag_lengths = []

    for k in range(-(n_points - min_diag_length), n_points - min_diag_length + 1):
        diag = np.diag(rp, k=k)
        if len(diag) >= min_diag_length:
            # 寻找连续的1
            consec_ones = np.diff(np.hstack(([0], diag, [0])))
            # 连续1的起始索引
            starts = np.where(consec_ones == 1)[0]
            # 连续1的结束索引
            ends = np.where(consec_ones == -1)[0]
            # 对角线长度
            lengths = ends - starts

            for length in lengths:
                if length >= min_diag_length:
                    diag_lengths.append(length)

    # 确定性(DET)
    if len(diag_lengths) > 0 and np.sum(rp) > 0:
        det = np.sum(diag_lengths) / np.sum(rp)
    else:
        det = 0.0

    # 平均对角线长度(L)
    if len(diag_lengths) > 0:
        avg_diag_length = np.mean(diag_lengths)
    else:
        avg_diag_length = 0.0

    # 最大对角线长度(Lmax)
    if len(diag_lengths) > 0:
        max_diag_length = np.max(diag_lengths)
    else:
        max_diag_length = 0.0

    # 发散度(DIV) - Lmax的倒数
    div = 1.0 / (max_diag_length + 1e-10)

    # 寻找垂直线(简化方法)
    min_vert_length = 2
    vert_lengths = []

    for j in range(n_points):
        col = rp[:, j]
        # 寻找连续的1
        consec_ones = np.diff(np.hstack(([0], col, [0])))
        # 连续1的起始索引
        starts = np.where(consec_ones == 1)[0]
        # 连续1的结束索引
        ends = np.where(consec_ones == -1)[0]
        # 垂直线长度
        lengths = ends - starts

        for length in lengths:
            if length >= min_vert_length:
                vert_lengths.append(length)

    # 层流性(LAM)
    if len(vert_lengths) > 0 and np.sum(rp) > 0:
        lam = np.sum(vert_lengths) / np.sum(rp)
    else:
        lam = 0.0

    # 驻留时间(TT)
    if len(vert_lengths) > 0:
        tt = np.mean(vert_lengths)
    else:
        tt = 0.0

    # 熵度量
    if len(diag_lengths) > 0:
        # 计算对角线长度的直方图
        hist, _ = np.histogram(diag_lengths, bins=range(2, max(diag_lengths) + 2))
        prob = hist / np.sum(hist)
        # 香农熵
        entropy = -np.sum(prob * np.log(prob + 1e-10))
    else:
        entropy = 0.0

    return {
        'recurrence_rate': float(rr),
        'determinism': float(det),
        'avg_diag_length': float(avg_diag_length),
        'max_diag_length': float(max_diag_length),
        'divergence': float(div),
        'laminarity': float(lam),
        'trapping_time': float(tt),
        'entropy': float(entropy)
    }


def prepare_tsne_visualization(
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    准备t-SNE可视化的数据。

    Args:
        embeddings: 嵌入向量，形状为 (n_samples, n_features)
        labels: 标签，形状为 (n_samples,)
        n_components: t-SNE降维的目标维度
        perplexity: t-SNE的perplexity参数
        random_state: 随机种子

    Returns:
        Tuple包含:
            - tsne_embeddings: t-SNE降维后的嵌入向量
            - labels: 原始标签
    """
    # 确保输入是numpy数组
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # 应用t-SNE降维
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    tsne_embeddings = tsne.fit_transform(embeddings)

    return tsne_embeddings, labels


def plot_tsne(
        tsne_embeddings: np.ndarray,
        labels: np.ndarray,
        title: str = "t-SNE Visualization of Speaker Embeddings",
        figsize: Tuple[int, int] = (10, 8),
        alpha: float = 0.7,
        marker_size: int = 50
) -> plt.Figure:
    """
    绘制t-SNE可视化图。

    Args:
        tsne_embeddings: t-SNE降维后的嵌入向量
        labels: 标签
        title: 图表标题
        figsize: 图表大小
        alpha: 点的透明度
        marker_size: 点的大小

    Returns:
        matplotlib图表对象
    """
    # 创建图表
    fig = plt.figure(figsize=figsize)

    # 确定维度
    n_components = tsne_embeddings.shape[1]

    if n_components == 2:
        # 2D可视化
        ax = fig.add_subplot(111)

        # 获取唯一标签
        unique_labels = np.unique(labels)

        # 为每个标签分配不同的颜色
        for label in unique_labels:
            idx = labels == label
            ax.scatter(
                tsne_embeddings[idx, 0],
                tsne_embeddings[idx, 1],
                alpha=alpha,
                s=marker_size,
                label=f"Speaker {label}"
            )

        ax.set_title(title)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")

        # 如果标签数量合理，添加图例
        if len(unique_labels) <= 20:
            ax.legend()

    elif n_components == 3:
        # 3D可视化
        ax = fig.add_subplot(111, projection='3d')

        # 获取唯一标签
        unique_labels = np.unique(labels)

        # 为每个标签分配不同的颜色
        for label in unique_labels:
            idx = labels == label
            ax.scatter(
                tsne_embeddings[idx, 0],
                tsne_embeddings[idx, 1],
                tsne_embeddings[idx, 2],
                alpha=alpha,
                s=marker_size,
                label=f"Speaker {label}"
            )

        ax.set_title(title)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")

        # 如果标签数量合理，添加图例
        if len(unique_labels) <= 20:
            ax.legend()

    else:
        raise ValueError(f"Unsupported number of components: {n_components}. Use 2 or 3.")

    plt.tight_layout()
    return fig


def plot_det_curve(
        scores_genuine: np.ndarray,
        scores_impostor: np.ndarray,
        title: str = "DET Curve",
        figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    绘制检测错误权衡(Detection Error Tradeoff, DET)曲线。

    Args:
        scores_genuine: 合法配对的相似度分数
        scores_impostor: 冒充配对的相似度分数
        title: 图表标题
        figsize: 图表大小

    Returns:
        matplotlib图表对象
    """
    # 确保输入是numpy数组
    if isinstance(scores_genuine, torch.Tensor):
        scores_genuine = scores_genuine.cpu().numpy()
    if isinstance(scores_impostor, torch.Tensor):
        scores_impostor = scores_impostor.cpu().numpy()

    # 将分数展平为一维数组
    scores_genuine = scores_genuine.flatten()
    scores_impostor = scores_impostor.flatten()

    # 创建标签数组
    labels_genuine = np.ones(len(scores_genuine))
    labels_impostor = np.zeros(len(scores_impostor))

    # 合并分数和标签
    scores = np.concatenate([scores_genuine, scores_impostor])
    labels = np.concatenate([labels_genuine, labels_impostor])

    # 计算不同阈值下的FAR和FRR
    thresholds = np.sort(scores)
    fars, frrs = [], []

    for threshold in thresholds:
        pred = scores >= threshold
        far = np.sum((pred == 1) & (labels == 0)) / np.sum(labels == 0)
        frr = np.sum((pred == 0) & (labels == 1)) / np.sum(labels == 1)
        fars.append(far)
        frrs.append(frr)

    # 转换为numpy数组
    fars = np.array(fars)
    frrs = np.array(frrs)

    # 创建图表
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # 绘制DET曲线
    ax.plot(fars, frrs, 'b-', lw=2)

    # 绘制EER线
    eer, _ = compute_eer(scores_genuine, scores_impostor)
    ax.plot([0, 1], [0, 1], 'r--', lw=2)
    ax.plot(eer, eer, 'ro', markersize=8)
    ax.annotate(f'EER = {eer:.4f}', xy=(eer, eer), xytext=(eer + 0.1, eer + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05))

    # 设置坐标轴
    ax.set_xlabel('False Acceptance Rate (FAR)')
    ax.set_ylabel('False Rejection Rate (FRR)')
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    return fig


def plot_recurrence_plot(
        rp: np.ndarray,
        title: str = "Recurrence Plot",
        figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    可视化递归图。

    Args:
        rp: 递归图矩阵
        title: 图表标题
        figsize: 图表大小

    Returns:
        matplotlib图表对象
    """
    # 创建图表
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # 绘制递归图
    im = ax.imshow(rp, cmap='binary', origin='lower', interpolation='none')

    # 添加颜色条
    plt.colorbar(im, ax=ax)

    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Time Index")

    plt.tight_layout()
    return fig


def plot_phase_space(
        signal: np.ndarray,
        embedding_dim: int = 3,
        delay: int = 1,
        title: str = "Phase Space Reconstruction",
        figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    可视化相空间重构。

    Args:
        signal: 输入信号
        embedding_dim: 嵌入维度
        delay: 时间延迟
        title: 图表标题
        figsize: 图表大小

    Returns:
        matplotlib图表对象
    """
    # 确保输入是numpy数组
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()

    # 归一化信号
    signal = signal.flatten()
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

    # 相空间重构
    m = min(embedding_dim, 3)  # 最多3维可视化
    tau = delay
    n = len(signal)

    if n <= (m - 1) * tau:
        raise ValueError("Signal too short for the specified embedding dimension and delay")

    Y = np.zeros((n - (m - 1) * tau, m))
    for i in range(n - (m - 1) * tau):
        for j in range(m):
            Y[i, j] = signal[i + j * tau]

    # 创建图表
    fig = plt.figure(figsize=figsize)

    if m == 2:
        # 2D相空间
        ax = fig.add_subplot(111)
        ax.plot(Y[:, 0], Y[:, 1], 'b-', lw=0.5)
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t+τ)")

    elif m == 3:
        # 3D相空间
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], 'b-', lw=0.5)
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t+τ)")
        ax.set_zlabel("x(t+2τ)")

    ax.set_title(title)
    plt.tight_layout()
    return fig


# 示例用法
if __name__ == "__main__":
    # 生成一些示例数据
    np.random.seed(42)

    # 生成合法和冒充分数
    scores_genuine = np.random.normal(0.7, 0.2, 1000)
    scores_impostor = np.random.normal(0.3, 0.2, 1000)

    # 计算EER
    eer, threshold = compute_eer(scores_genuine, scores_impostor)
    print(f"EER: {eer:.4f}, Threshold: {threshold:.4f}")

    # 计算minDCF
    mindcf, threshold = compute_mindcf(scores_genuine, scores_impostor)
    print(f"minDCF: {mindcf:.4f}, Threshold: {threshold:.4f}")

    # 生成示例信号
    t = np.linspace(0, 10, 1000)
    clean_signal = np.sin(t) + 0.1 * np.sin(5 * t)
    noisy_signal = clean_signal + 0.2 * np.random.randn(len(clean_signal))

    # 计算CSI
    csi = compute_csi(eer, eer * 0.9, clean_signal, noisy_signal)
    print(f"CSI: {csi:.4f}")

    # 计算李雅普诺夫指数
    mle = compute_lyapunov_exponent(clean_signal)
    print(f"Maximum Lyapunov Exponent: {mle:.4f}")

    # 计算递归图
    rp = compute_recurrence_plot(clean_signal)
    print(f"Recurrence Plot shape: {rp.shape}")

    # 计算RQA指标
    rqa_metrics = compute_rqa_metrics(rp)
    print("RQA Metrics:")
    for key, value in rqa_metrics.items():
        print(f"  {key}: {value:.4f}")

    # 生成嵌入向量和标签用于t-SNE可视化
    n_speakers = 10
    n_samples_per_speaker = 20
    embeddings = np.random.randn(n_speakers * n_samples_per_speaker, 64)
    labels = np.repeat(np.arange(n_speakers), n_samples_per_speaker)

    # 准备t-SNE可视化
    tsne_embeddings, _ = prepare_tsne_visualization(embeddings, labels)
    print(f"t-SNE Embeddings shape: {tsne_embeddings.shape}")

    # 绘制t-SNE可视化
    fig = plot_tsne(tsne_embeddings, labels)
    plt.savefig("tsne_visualization.png")

    # 绘制DET曲线
    fig = plot_det_curve(scores_genuine, scores_impostor)
    plt.savefig("det_curve.png")

    # 绘制递归图
    fig = plot_recurrence_plot(rp)
    plt.savefig("recurrence_plot.png")

    # 绘制相空间重构
    fig = plot_phase_space(clean_signal)
    plt.savefig("phase_space.png")

    print("所有图表已保存")

