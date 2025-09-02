import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from sklearn.neighbors import NearestNeighbors
import warnings

class MLSAFeatureExtractor:
    """
    Multi-scale Lyapunov Spectrum Analysis (MLSA) feature extractor

    多尺度李雅普诺夫谱分析(MLSA)特征提取器
    """
    def __init__(
        self,
        scales: List[int] = [1, 2, 4, 8, 16],
        embedding_dim: int = 3,
        delay: int = 1,
        n_exponents: int = 2
    ):
        """
        Initialize the MLSA feature extractor

        初始化MLSA特征提取器

        Args:
            scales (List[int]): List of scales for multi-scale analysis
                               多尺度分析的尺度列表
            embedding_dim (int): Embedding dimension for phase space reconstruction
                               相空间重构的嵌入维度
            delay (int): Time delay for phase space reconstruction
                        相空间重构的时间延迟
            n_exponents (int): Number of Lyapunov exponents to compute
                              要计算的李雅普诺夫指数数量
        """
        self.scales = scales
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.n_exponents = n_exponents

    def extract(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract MLSA features from time series

        从时间序列中提取MLSA特征

        Args:
            time_series (np.ndarray): Input time series
                                     输入时间序列

        Returns:
            np.ndarray: MLSA features
                       MLSA特征
        """
        # Ensure time series is 1D
        # 确保时间序列是一维的
        time_series = time_series.flatten()

        # Initialize feature vector
        # 初始化特征向量
        features = []

        # Extract features at different scales
        # 在不同尺度上提取特征
        for scale in self.scales:
            # Downsample the time series
            # 对时间序列进行降采样
            downsampled = time_series[::scale]

            # Skip if downsampled series is too short
            # 如果降采样后的序列太短，则跳过
            if len(downsampled) < (self.embedding_dim - 1) * self.delay + 100:
                warnings.warn(f"Skipping scale {scale}: downsampled series too short")
                # Pad with zeros to maintain feature vector length
                # 用零填充以保持特征向量长度
                features.extend([0.0] * self.n_exponents)
                continue

            # Reconstruct phase space
            # 重构相空间
            phase_space = self._delay_embedding(downsampled, self.embedding_dim, self.delay)

            # Calculate Lyapunov exponents
            # 计算李雅普诺夫指数
            exponents = self._calculate_lyapunov_spectrum(phase_space, n_exponents=self.n_exponents)

            # Add exponents to feature vector
            # 将指数添加到特征向量
            features.extend(exponents)

        return np.array(features)

    def _delay_embedding(self, time_series: np.ndarray, embedding_dim: int, delay: int) -> np.ndarray:
        """
        Perform time delay embedding

        执行时间延迟嵌入

        Args:
            time_series (np.ndarray): Input time series
                                     输入时间序列
            embedding_dim (int): Embedding dimension
                               嵌入维度
            delay (int): Time delay
                        时间延迟

        Returns:
            np.ndarray: Reconstructed phase space
                       重构的相空间
        """
        n_points = len(time_series) - (embedding_dim - 1) * delay

        if n_points <= 0:
            raise ValueError("Time series too short for the given delay and embedding dimension")

        # Initialize the reconstructed phase space
        # 初始化重构的相空间
        phase_space = np.zeros((n_points, embedding_dim))

        # Fill the phase space with delayed values
        # 用延迟值填充相空间
        for i in range(embedding_dim):
            phase_space[:, i] = time_series[i*delay:i*delay + n_points]

        return phase_space

    def _calculate_lyapunov_spectrum(self, phase_space: np.ndarray, n_exponents: int = 2,
                                    min_neighbors: int = 20, trajectory_len: int = 20) -> List[float]:
        """
        Calculate the Lyapunov spectrum using the method of Sano and Sawada

        使用Sano和Sawada的方法计算李雅普诺夫谱

        Args:
            phase_space (np.ndarray): Reconstructed phase space
                                     重构的相空间
            n_exponents (int): Number of Lyapunov exponents to compute
                              要计算的李雅普诺夫指数数量
            min_neighbors (int): Minimum number of neighbors for local linear fit
                                局部线性拟合的最小邻居数
            trajectory_len (int): Length of trajectory for exponent estimation
                                用于指数估计的轨迹长度

        Returns:
            List[float]: Lyapunov exponents
                        李雅普诺夫指数
        """
        n_points, dim = phase_space.shape

        # Ensure we don't try to compute more exponents than dimensions
        # 确保我们不尝试计算比维度更多的指数
        n_exponents = min(n_exponents, dim)

        # Initialize exponents
        # 初始化指数
        exponents = np.zeros(n_exponents)

        # Number of reference points to use
        # 要使用的参考点数量
        n_refs = min(100, n_points // 10)

        # Randomly select reference points
        # 随机选择参考点
        ref_indices = np.random.choice(n_points - trajectory_len, n_refs, replace=False)

        # For each reference point
        # 对于每个参考点
        for ref_idx in ref_indices:
            ref_point = phase_space[ref_idx]

            # Find nearest neighbors
            # 寻找最近邻
            nbrs = NearestNeighbors(n_neighbors=min_neighbors + 1, algorithm='ball_tree').fit(phase_space)
            distances, indices = nbrs.kneighbors([ref_point])

            # Remove the reference point itself
            # 移除参考点本身
            neighbors = indices[0][1:]

            # Calculate local Jacobian matrix
            # 计算局部雅可比矩阵
            for t in range(trajectory_len):
                if ref_idx + t >= n_points:
                    break

                # Current state and next state of reference trajectory
                # 参考轨迹的当前状态和下一状态
                x_t = phase_space[ref_idx + t]
                x_t1 = phase_space[ref_idx + t + 1] if ref_idx + t + 1 < n_points else None

                if x_t1 is None:
                    break

                # Collect neighbor states at time t and t+1
                # 收集时间t和t+1的邻居状态
                neighbor_states_t = []
                neighbor_states_t1 = []

                for neighbor_idx in neighbors:
                    if neighbor_idx + t < n_points and neighbor_idx + t + 1 < n_points:
                        neighbor_states_t.append(phase_space[neighbor_idx + t])
                        neighbor_states_t1.append(phase_space[neighbor_idx + t + 1])

                if len(neighbor_states_t) < dim:
                    continue

                # Convert to numpy arrays
                # 转换为numpy数组
                neighbor_states_t = np.array(neighbor_states_t)
                neighbor_states_t1 = np.array(neighbor_states_t1)

                # Calculate displacement vectors
                # 计算位移向量
                dx_t = neighbor_states_t - x_t
                dx_t1 = neighbor_states_t1 - x_t1

                try:
                    # Estimate local Jacobian matrix using least squares
                    # 使用最小二乘法估计局部雅可比矩阵
                    J, _, _, _ = np.linalg.lstsq(dx_t, dx_t1, rcond=None)

                    # Perform QR decomposition
                    # 执行QR分解
                    Q, R = np.linalg.qr(J)

                    # Update exponents with log of diagonal elements of R
                    # 用R的对角元素的对数更新指数
                    for i in range(n_exponents):
                        if i < len(np.diag(R)):
                            exponents[i] += np.log(abs(np.diag(R)[i]))
                except np.linalg.LinAlgError:
                    # Skip if linear algebra error occurs
                    # 如果发生线性代数错误，则跳过
                    continue

        # Normalize exponents
        # 归一化指数
        if n_refs > 0 and trajectory_len > 0:
            exponents = exponents / (n_refs * trajectory_len)

        # Sort exponents in descending order
        # 按降序排列指数
        exponents = np.sort(exponents)[::-1]

        return exponents.tolist()

class RQAFeatureExtractor:
    """
    Recurrence Quantification Analysis (RQA) feature extractor

    递归定量分析(RQA)特征提取器
    """
    def __init__(
        self,
        embedding_dim: int = 3,
        delay: int = 1,
        threshold: Optional[float] = None,
        lmin: int = 2
    ):
        """
        Initialize the RQA feature extractor

        初始化RQA特征提取器

        Args:
            embedding_dim (int): Embedding dimension for phase space reconstruction
                               相空间重构的嵌入维度
            delay (int): Time delay for phase space reconstruction
                        相空间重构的时间延迟
            threshold (float, optional): Threshold for recurrence plot. If None, it will be estimated
                                        递归图的阈值。如果为None，将进行估计
            lmin (int): Minimum length of diagonal and vertical lines
                       对角线和垂直线的最小长度
        """
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.threshold = threshold
        self.lmin = lmin

    def extract(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        Extract RQA features from time series

        从时间序列中提取RQA特征

        Args:
            time_series (np.ndarray): Input time series
                                     输入时间序列

        Returns:
            Dict[str, float]: Dictionary of RQA features
                             RQA特征字典
        """
        # Ensure time series is 1D
        # 确保时间序列是一维的
        time_series = time_series.flatten()

        # Reconstruct phase space
        # 重构相空间
        phase_space = self._delay_embedding(time_series, self.embedding_dim, self.delay)

        # Calculate distance matrix
        # 计算距离矩阵
        dist_matrix = self._calculate_distance_matrix(phase_space)

        # Determine threshold if not provided
        # 如果未提供阈值，则确定阈值
        threshold = self.threshold
        if threshold is None:
            # Use a percentage of the maximum distance
            # 使用最大距离的百分比
            threshold = 0.1 * np.max(dist_matrix)

        # Create recurrence plot
        # 创建递归图
        recurrence_plot = dist_matrix < threshold

        # Calculate RQA features
        # 计算RQA特征
        features = self._calculate_rqa_features(recurrence_plot)

        return features

    def _delay_embedding(self, time_series: np.ndarray, embedding_dim: int, delay: int) -> np.ndarray:
        """
        Perform time delay embedding

        执行时间延迟嵌入

        Args:
            time_series (np.ndarray): Input time series
                                     输入时间序列
            embedding_dim (int): Embedding dimension
                               嵌入维度
            delay (int): Time delay
                        时间延迟

        Returns:
            np.ndarray: Reconstructed phase space
                       重构的相空间
        """
        n_points = len(time_series) - (embedding_dim - 1) * delay

        if n_points <= 0:
            raise ValueError("Time series too short for the given delay and embedding dimension")

        # Initialize the reconstructed phase space
        # 初始化重构的相空间
        phase_space = np.zeros((n_points, embedding_dim))

        # Fill the phase space with delayed values
        # 用延迟值填充相空间
        for i in range(embedding_dim):
            phase_space[:, i] = time_series[i*delay:i*delay + n_points]

        return phase_space

    def _calculate_distance_matrix(self, phase_space: np.ndarray) -> np.ndarray:
        """
        Calculate the distance matrix between all points in phase space

        计算相空间中所有点之间的距离矩阵

        Args:
            phase_space (np.ndarray): Reconstructed phase space
                                     重构的相空间

        Returns:
            np.ndarray: Distance matrix
                       距离矩阵
        """
        n_points = phase_space.shape[0]
        dist_matrix = np.zeros((n_points, n_points))

        # Calculate Euclidean distance between all pairs of points
        # 计算所有点对之间的欧几里得距离
        for i in range(n_points):
            for j in range(i, n_points):
                dist = np.linalg.norm(phase_space[i] - phase_space[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist

        return dist_matrix

    def _calculate_rqa_features(self, recurrence_plot: np.ndarray) -> Dict[str, float]:
        """
        Calculate RQA features from recurrence plot

        从递归图计算RQA特征

        Args:
            recurrence_plot (np.ndarray): Recurrence plot (binary matrix)
                                         递归图（二值矩阵）

        Returns:
            Dict[str, float]: Dictionary of RQA features
                             RQA特征字典
        """
        n = recurrence_plot.shape[0]

        # Recurrence Rate (RR): Percentage of recurrence points
        # 递归率(RR)：递归点的百分比
        rr = np.sum(recurrence_plot) / (n * n)

        # Find diagonal lines
        # 寻找对角线
        diag_lengths = []
        for i in range(-(n-self.lmin), n-self.lmin+1):
            if i == 0:  # Skip the main diagonal
                continue
            diag = np.diag(recurrence_plot, k=i)
            # Count consecutive True values
            # 计算连续的True值
            if len(diag) > 0:
                count = 0
                for val in diag:
                    if val:
                        count += 1
                    else:
                        if count >= self.lmin:
                            diag_lengths.append(count)
                        count = 0
                if count >= self.lmin:
                    diag_lengths.append(count)

        # Find vertical lines
        # 寻找垂直线
        vert_lengths = []
        for i in range(n):
            col = recurrence_plot[:, i]
            # Count consecutive True values
            # 计算连续的True值
            count = 0
            for val in col:
                if val:
                    count += 1
                else:
                    if count >= self.lmin:
                        vert_lengths.append(count)
                    count = 0
            if count >= self.lmin:
                vert_lengths.append(count)

        # Convert to numpy arrays for easier calculation
        # 转换为numpy数组以便于计算
        diag_lengths = np.array(diag_lengths) if diag_lengths else np.array([0])
        vert_lengths = np.array(vert_lengths) if vert_lengths else np.array([0])

        # Determinism (DET): Percentage of recurrence points forming diagonal lines
        # 确定性(DET)：形成对角线的递归点的百分比
        det = np.sum(diag_lengths) / np.sum(recurrence_plot) if np.sum(recurrence_plot) > 0 else 0

        # Average Diagonal Line Length (L)
        # 平均对角线长度(L)
        avg_diag_length = np.mean(diag_lengths) if len(diag_lengths) > 0 and diag_lengths[0] > 0 else 0

        # Maximum Diagonal Line Length (Lmax)
        # 最大对角线长度(Lmax)
        max_diag_length = np.max(diag_lengths) if len(diag_lengths) > 0 and diag_lengths[0] > 0 else 0

        # Divergence (DIV): Inverse of Lmax
        # 发散(DIV)：Lmax的倒数
        div = 1.0 / max_diag_length if max_diag_length > 0 else float('inf')

        # Entropy of diagonal line lengths (ENTR)
        # 对角线长度的熵(ENTR)
        if len(diag_lengths) > 0 and diag_lengths[0] > 0:
            # Calculate histogram of diagonal lengths
            # 计算对角线长度的直方图
            hist, _ = np.histogram(diag_lengths, bins=range(1, int(max_diag_length) + 2))
            # Normalize histogram to get probability
            # 归一化直方图以获得概率
            prob = hist / np.sum(hist)
            # Calculate entropy
            # 计算熵
            entr = -np.sum(prob * np.log(prob + 1e-10))
        else:
            entr = 0

        # Laminarity (LAM): Percentage of recurrence points forming vertical lines
        # 层流性(LAM)：形成垂直线的递归点的百分比
        lam = np.sum(vert_lengths) / np.sum(recurrence_plot) if np.sum(recurrence_plot) > 0 else 0

        # Trapping Time (TT): Average vertical line length
        # 捕获时间(TT)：平均垂直线长度
        tt = np.mean(vert_lengths) if len(vert_lengths) > 0 and vert_lengths[0] > 0 else 0

        # Return all features
        # 返回所有特征
        return {
            'RR': rr,
            'DET': det,
            'L': avg_diag_length,
            'Lmax': max_diag_length,
            'DIV': div,
            'ENTR': entr,
            'LAM': lam,
            'TT': tt
        }

    def plot_recurrence_plot(self, time_series: np.ndarray, title: str = "Recurrence Plot") -> None:
        """
        Plot the recurrence plot for a time series

        绘制时间序列的递归图

        Args:
            time_series (np.ndarray): Input time series
                                     输入时间序列
            title (str): Plot title
                        图表标题
        """
        # Reconstruct phase space
        # 重构相空间
        phase_space = self._delay_embedding(time_series, self.embedding_dim, self.delay)

        # Calculate distance matrix
        # 计算距离矩阵
        dist_matrix = self._calculate_distance_matrix(phase_space)

        # Determine threshold if not provided
        # 如果未提供阈值，则确定阈值
        threshold = self.threshold
        if threshold is None:
            threshold = 0.1 * np.max(dist_matrix)

        # Create recurrence plot
        # 创建递归图
        recurrence_plot = dist_matrix < threshold

        # Plot
        # 绘图
        plt.figure(figsize=(10, 8))
        plt.imshow(recurrence_plot, cmap='binary', origin='lower')
        plt.colorbar(label='Recurrence')
        plt.title(title)
        plt.xlabel('Time Index')
        plt.ylabel('Time Index')
        plt.tight_layout()
        plt.show()

class ChaoticFeatureExtractor:
    """
    Combined chaotic feature extractor using MLSA and RQA

    使用MLSA和RQA的组合混沌特征提取器
    """
    def __init__(
        self,
        embedding_dim: int = 3,
        delay: int = 1,
        scales: List[int] = [1, 2, 4, 8, 16],
        n_lyapunov_exponents: int = 2,
        rqa_threshold: Optional[float] = None
    ):
        """
        Initialize the chaotic feature extractor

        初始化混沌特征提取器

        Args:
            embedding_dim (int): Embedding dimension for phase space reconstruction
                               相空间重构的嵌入维度
            delay (int): Time delay for phase space reconstruction
                        相空间重构的时间延迟
            scales (List[int]): List of scales for multi-scale analysis
                               多尺度分析的尺度列表
            n_lyapunov_exponents (int): Number of Lyapunov exponents to compute
                                       要计算的李雅普诺夫指数数量
            rqa_threshold (float, optional): Threshold for recurrence plot. If None, it will be estimated
                                           递归图的阈值。如果为None，将进行估计
        """
        self.embedding_dim = embedding_dim
        self.delay = delay

        # Initialize MLSA feature extractor
        # 初始化MLSA特征提取器
        self.mlsa_extractor = MLSAFeatureExtractor(
            scales=scales,
            embedding_dim=embedding_dim,
            delay=delay,
            n_exponents=n_lyapunov_exponents
        )

        # Initialize RQA feature extractor
        # 初始化RQA特征提取器
        self.rqa_extractor = RQAFeatureExtractor(
            embedding_dim=embedding_dim,
            delay=delay,
            threshold=rqa_threshold
        )

    def extract(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract combined chaotic features from time series

        从时间序列中提取组合混沌特征

        Args:
            time_series (np.ndarray): Input time series
                                     输入时间序列

        Returns:
            np.ndarray: Combined chaotic features
                       组合混沌特征
        """
        # Extract MLSA features
        # 提取MLSA特征
        mlsa_features = self.mlsa_extractor.extract(time_series)

        # Extract RQA features
        # 提取RQA特征
        rqa_features_dict = self.rqa_extractor.extract(time_series)

        # Convert RQA features dictionary to array
        # 将RQA特征字典转换为数组
        rqa_features = np.array(list(rqa_features_dict.values()))

        # Combine features
        # 组合特征
        combined_features = np.concatenate([mlsa_features, rqa_features])

        return combined_features

    def extract_with_names(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        Extract combined chaotic features with feature names

        提取带有特征名称的组合混沌特征

        Args:
            time_series (np.ndarray): Input time series
                                     输入时间序列

        Returns:
            Dict[str, float]: Dictionary of named features
                             命名特征的字典
        """
        # Extract MLSA features
        # 提取MLSA特征
        mlsa_features = self.mlsa_extractor.extract(time_series)

        # Extract RQA features
        # 提取RQA特征
        rqa_features = self.rqa_extractor.extract(time_series)

        # Create named feature dictionary
        # 创建命名特征字典
        named_features = {}

        # Add MLSA features with names
        # 添加带有名称的MLSA特征
        for i, scale in enumerate(self.mlsa_extractor.scales):
            for j in range(self.mlsa_extractor.n_exponents):
                feature_idx = i * self.mlsa_extractor.n_exponents + j
                if feature_idx < len(mlsa_features):
                    named_features[f'MLSA_scale{scale}_exp{j+1}'] = mlsa_features[feature_idx]

        # Add RQA features
        # 添加RQA特征
        for name, value in rqa_features.items():
            named_features[f'RQA_{name}'] = value

        return named_features

def extract_chaotic_features_batch(audio_batch: np.ndarray, **kwargs) -> np.ndarray:
    """
    Extract chaotic features from a batch of audio signals

    从一批音频信号中提取混沌特征

    Args:
        audio_batch (np.ndarray): Batch of audio signals [batch_size, signal_length]
                                 一批音频信号 [batch_size, signal_length]
        **kwargs: Additional arguments for ChaoticFeatureExtractor
                 ChaoticFeatureExtractor的其他参数

    Returns:
        np.ndarray: Batch of chaotic features [batch_size, n_features]
                   一批混沌特征 [batch_size, n_features]
    """
    batch_size = audio_batch.shape[0]

    # Initialize feature extractor
    # 初始化特征提取器
    extractor = ChaoticFeatureExtractor(**kwargs)

    # Initialize output array
    # 初始化输出数组
    # Extract features for the first sample to determine feature dimension
    # 提取第一个样本的特征以确定特征维度
    first_features = extractor.extract(audio_batch[0])
    n_features = len(first_features)

    # Initialize output array with the correct shape
    # 用正确的形状初始化输出数组
    features_batch = np.zeros((batch_size, n_features))
    features_batch[0] = first_features

    # Extract features for the rest of the batch
    # 提取批次其余部分的特征
    for i in range(1, batch_size):
        features_batch[i] = extractor.extract(audio_batch[i])

    return features_batch

if __name__ == "__main__":
    # Generate a test signal (Lorenz system)
    # 生成测试信号（洛伦兹系统）
    def lorenz_system(t, xyz, sigma=10, beta=8/3, rho=28):
        x, y, z = xyz
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    from scipy.integrate import solve_ivp

    # Initial conditions
    # 初始条件
    xyz0 = [1.0, 1.0, 1.0]

    # Time points
    # 时间点
    t = np.linspace(0, 50, 5000)

    # Solve the differential equations
    # 求解微分方程
    solution = solve_ivp(lorenz_system, [0, 50], xyz0, t_eval=t)

    # Extract x-coordinate time series
    # 提取x坐标时间序列
    x_series = solution.y[0]

    # Create feature extractors
    # 创建特征提取器
    mlsa_extractor = MLSAFeatureExtractor()
    rqa_extractor = RQAFeatureExtractor()
    combined_extractor = ChaoticFeatureExtractor()

    # Extract features
    # 提取特征
    mlsa_features = mlsa_extractor.extract(x_series)
    rqa_features = rqa_extractor.extract(x_series)
    combined_features = combined_extractor.extract(x_series)
    named_features = combined_extractor.extract_with_names(x_series)

    print("MLSA Features:")
    print(mlsa_features)
    print("\nRQA Features:")
    for name, value in rqa_features.items():
        print(f"{name}: {value:.4f}")
    print("\nCombined Features Shape:", combined_features.shape)
    print("\nNamed Features:")
    for name, value in named_features.items():
        print(f"{name}: {value:.4f}")

    # Plot recurrence plot
    # 绘制递归图
    rqa_extractor.plot_recurrence_plot(x_series, "Lorenz System Recurrence Plot")