import numpy as np
from scipy import signal
from typing import Tuple, Optional, Union, List
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


class PhaseSpaceReconstructor:
    """
    Class for phase space reconstruction using time delay embedding method

    使用时间延迟嵌入方法进行相空间重构的类
    """

    def __init__(
            self,
            delay: Optional[int] = None,
            embedding_dim: Optional[int] = None,
            max_delay: int = 100,
            max_dim: int = 10,
            method: str = 'autocorr'
    ):
        """
        Initialize the phase space reconstructor

        初始化相空间重构器

        Args:
            delay (int, optional): Time delay (tau). If None, will be estimated from data
                                  时间延迟(tau)。如果为None，将从数据中估计
            embedding_dim (int, optional): Embedding dimension. If None, will be estimated from data
                                         嵌入维度。如果为None，将从数据中估计
            max_delay (int): Maximum delay to consider when estimating optimal delay
                           估计最佳延迟时考虑的最大延迟
            max_dim (int): Maximum dimension to consider when estimating optimal embedding dimension
                         估计最佳嵌入维度时考虑的最大维度
            method (str): Method to estimate delay ('autocorr', 'mutual_info')
                        估计延迟的方法('autocorr', 'mutual_info')
        """
        self.delay = delay
        self.embedding_dim = embedding_dim
        self.max_delay = max_delay
        self.max_dim = max_dim
        self.method = method

    def reconstruct(self, time_series: np.ndarray) -> np.ndarray:
        """
        Reconstruct phase space from time series

        从时间序列重构相空间

        Args:
            time_series (np.ndarray): Input time series (1D array)
                                     输入时间序列（一维数组）

        Returns:
            np.ndarray: Reconstructed phase space (2D array)
                       重构的相空间（二维数组）
        """
        # Ensure time series is 1D
        # 确保时间序列是一维的
        time_series = time_series.flatten()

        # Estimate delay if not provided
        # 如果未提供延迟，则进行估计
        if self.delay is None:
            self.delay = self.estimate_delay(time_series)
            print(f"Estimated delay: {self.delay}")

        # Estimate embedding dimension if not provided
        # 如果未提供嵌入维度，则进行估计
        if self.embedding_dim is None:
            self.embedding_dim = self.estimate_embedding_dimension(time_series, self.delay)
            print(f"Estimated embedding dimension: {self.embedding_dim}")

        # Perform time delay embedding
        # 执行时间延迟嵌入
        return self.delay_embedding(time_series, self.delay, self.embedding_dim)

    def estimate_delay(self, time_series: np.ndarray) -> int:
        """
        Estimate optimal time delay using autocorrelation or mutual information

        使用自相关或互信息估计最佳时间延迟

        Args:
            time_series (np.ndarray): Input time series
                                     输入时间序列

        Returns:
            int: Estimated optimal delay
                估计的最佳延迟
        """
        if self.method == 'autocorr':
            return self._estimate_delay_autocorr(time_series)
        elif self.method == 'mutual_info':
            return self._estimate_delay_mutual_info(time_series)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _estimate_delay_autocorr(self, time_series: np.ndarray) -> int:
        """
        Estimate delay using autocorrelation function

        使用自相关函数估计延迟

        Args:
            time_series (np.ndarray): Input time series
                                     输入时间序列

        Returns:
            int: Optimal delay (first zero crossing or 1/e decay)
                最佳延迟（第一次过零点或衰减到1/e）
        """
        # Calculate autocorrelation
        # 计算自相关
        autocorr = signal.correlate(time_series - np.mean(time_series),
                                    time_series - np.mean(time_series),
                                    mode='full')

        # Keep only the second half (positive lags)
        # 只保留后半部分（正时滞）
        autocorr = autocorr[len(autocorr) // 2:]

        # Normalize
        # 归一化
        autocorr = autocorr / autocorr[0]

        # Find first zero crossing
        # 寻找第一个过零点
        zero_crossings = np.where(np.diff(np.signbit(autocorr)))[0]
        if len(zero_crossings) > 0:
            first_zero = zero_crossings[0]
            if first_zero < self.max_delay:
                return first_zero

        # If no zero crossing found or it's too large, find where autocorr decays to 1/e
        # 如果没有找到过零点或过零点太大，找到自相关衰减到1/e的点
        decay_threshold = 1 / np.e
        decay_idx = np.where(autocorr < decay_threshold)[0]
        if len(decay_idx) > 0:
            first_decay = decay_idx[0]
            if first_decay < self.max_delay:
                return first_decay

        # If all else fails, return a reasonable default
        # 如果以上方法都失败，返回一个合理的默认值
        return min(self.max_delay // 10, 10)

    def _estimate_delay_mutual_info(self, time_series: np.ndarray) -> int:
        """
        Estimate delay using mutual information (first minimum)

        使用互信息（第一个最小值）估计延迟

        Args:
            time_series (np.ndarray): Input time series
                                     输入时间序列

        Returns:
            int: Optimal delay
                最佳延迟
        """
        # This is a simplified implementation using histogram-based MI estimation
        # 这是一个使用基于直方图的互信息估计的简化实现
        mi_values = []

        # Calculate MI for different delays
        # 计算不同延迟的互信息
        for delay in range(1, min(self.max_delay, len(time_series) // 3)):
            x1 = time_series[:-delay] if delay > 0 else time_series
            x2 = time_series[delay:] if delay > 0 else time_series

            # Calculate mutual information using histograms
            # 使用直方图计算互信息
            nbins = min(30, len(x1) // 30)  # Adaptive bin size
            hist_2d, _, _ = np.histogram2d(x1, x2, bins=nbins)
            hist_2d = hist_2d / np.sum(hist_2d)  # Normalize to get probability

            hist_x1, _ = np.histogram(x1, bins=nbins)
            hist_x1 = hist_x1 / np.sum(hist_x1)

            hist_x2, _ = np.histogram(x2, bins=nbins)
            hist_x2 = hist_x2 / np.sum(hist_x2)

            # Calculate marginal probabilities
            # 计算边缘概率
            p_x1 = hist_x1.reshape(-1, 1)
            p_x2 = hist_x2.reshape(1, -1)

            # Calculate mutual information
            # 计算互信息
            mutual_info = 0
            for i in range(nbins):
                for j in range(nbins):
                    if hist_2d[i, j] > 0 and p_x1[i, 0] > 0 and p_x2[0, j] > 0:
                        mutual_info += hist_2d[i, j] * np.log(hist_2d[i, j] / (p_x1[i, 0] * p_x2[0, j]))

            mi_values.append(mutual_info)

        # Find first minimum of mutual information
        # 找到互信息的第一个最小值
        mi_values = np.array(mi_values)
        for i in range(1, len(mi_values) - 1):
            if mi_values[i - 1] > mi_values[i] and mi_values[i] < mi_values[i + 1]:
                return i + 1  # +1 because we started from delay=1

        # If no clear minimum is found, return a reasonable default
        # 如果没有找到明显的最小值，返回一个合理的默认值
        return min(self.max_delay // 10, 10)

    def estimate_embedding_dimension(self, time_series: np.ndarray, delay: int) -> int:
        """
        Estimate optimal embedding dimension using the False Nearest Neighbors (FNN) method

        使用伪近邻(FNN)方法估计最佳嵌入维度

        Args:
            time_series (np.ndarray): Input time series
                                     输入时间序列
            delay (int): Time delay
                        时间延迟

        Returns:
            int: Estimated optimal embedding dimension
                估计的最佳嵌入维度
        """
        # Implement False Nearest Neighbors algorithm
        # 实现伪近邻算法
        fnn_fractions = []

        # Calculate FNN for dimensions from 1 to max_dim
        # 计算从1到max_dim维度的伪近邻比例
        for dim in range(1, self.max_dim + 1):
            # Create delay embedding for current dimension
            # 为当前维度创建延迟嵌入
            embedded = self.delay_embedding(time_series, delay, dim)

            # If we can't embed to dim+1, break
            # 如果无法嵌入到dim+1维，则中断
            if (len(time_series) - (dim + 1) * delay) <= 0:
                break

            # Create delay embedding for next dimension (dim+1)
            # 为下一维度(dim+1)创建延迟嵌入
            embedded_next = self.delay_embedding(time_series, delay, dim + 1)

            # Find nearest neighbors in current dimension
            # 在当前维度中找到最近邻
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(embedded)
            distances, indices = nbrs.kneighbors(embedded)

            # Count false nearest neighbors
            # 计算伪近邻的数量
            false_neighbors = 0
            total_points = len(embedded) - 1  # Exclude the last point which has no next-dim embedding

            for i in range(total_points):
                # Get the nearest neighbor (excluding self)
                # 获取最近邻（不包括自身）
                nn_idx = indices[i, 1]

                # Check if it's a false neighbor
                # 检查是否为伪近邻
                if nn_idx < len(embedded_next) and i < len(embedded_next):
                    # Calculate distance in current dimension
                    # 计算当前维度中的距离
                    dist_d = distances[i, 1]

                    # Calculate distance in next dimension
                    # 计算下一维度中的距离
                    dist_d1 = np.abs(embedded_next[i, -1] - embedded_next[nn_idx, -1])

                    # If relative increase in distance is large, it's a false neighbor
                    # 如果距离的相对增加很大，则它是一个伪近邻
                    if dist_d > 0 and dist_d1 / dist_d > 10:
                        false_neighbors += 1

            # Calculate fraction of false neighbors
            # 计算伪近邻的比例
            fnn_fraction = false_neighbors / total_points if total_points > 0 else 0
            fnn_fractions.append(fnn_fraction)

            # If fraction of false neighbors is small enough, we found our dimension
            # 如果伪近邻的比例足够小，我们就找到了合适的维度
            if fnn_fraction < 0.01:
                return dim

        # If no clear dimension is found, find where the FNN fraction decreases significantly
        # 如果没有找到明确的维度，找到伪近邻比例显著下降的地方
        fnn_fractions = np.array(fnn_fractions)
        if len(fnn_fractions) > 1:
            diffs = np.diff(fnn_fractions)
            significant_drops = np.where(diffs < -0.1)[0]
            if len(significant_drops) > 0:
                return significant_drops[0] + 1

        # If all else fails, return a reasonable default
        # 如果以上方法都失败，返回一个合理的默认值
        return min(len(fnn_fractions), 3)

    def delay_embedding(self, time_series: np.ndarray, delay: int, embedding_dim: int) -> np.ndarray:
        """
        Perform time delay embedding

        执行时间延迟嵌入

        Args:
            time_series (np.ndarray): Input time series
                                     输入时间序列
            delay (int): Time delay
                        时间延迟
            embedding_dim (int): Embedding dimension
                               嵌入维度

        Returns:
            np.ndarray: Reconstructed phase space
                       重构的相空间
        """
        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")

        if delay <= 0:
            raise ValueError("Delay must be positive")

        # Calculate the number of points in the reconstructed space
        # 计算重构空间中的点数
        n_points = len(time_series) - (embedding_dim - 1) * delay

        if n_points <= 0:
            raise ValueError("Time series too short for the given delay and embedding dimension")

        # Initialize the reconstructed phase space
        # 初始化重构的相空间
        phase_space = np.zeros((n_points, embedding_dim))

        # Fill the phase space with delayed values
        # 用延迟值填充相空间
        for i in range(embedding_dim):
            phase_space[:, i] = time_series[i * delay:i * delay + n_points]

        return phase_space

    def plot_phase_space(self, phase_space: np.ndarray, title: str = "Reconstructed Phase Space") -> None:
        """
        Plot the reconstructed phase space

        绘制重构的相空间

        Args:
            phase_space (np.ndarray): Reconstructed phase space
                                     重构的相空间
            title (str): Plot title
                        图表标题
        """
        dim = phase_space.shape[1]

        if dim == 1:
            plt.figure(figsize=(10, 6))
            plt.plot(phase_space)
            plt.title(f"{title} (1D)")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()

        elif dim == 2:
            plt.figure(figsize=(10, 8))
            plt.plot(phase_space[:, 0], phase_space[:, 1], 'b.', markersize=0.5)
            plt.title(f"{title} (2D)")
            plt.xlabel("x(t)")
            plt.ylabel("x(t+τ)")
            plt.grid(True)
            plt.show()

        elif dim == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(phase_space[:, 0], phase_space[:, 1], phase_space[:, 2], 'b.', markersize=0.5)
            ax.set_title(f"{title} (3D)")
            ax.set_xlabel("x(t)")
            ax.set_ylabel("x(t+τ)")
            ax.set_zlabel("x(t+2τ)")
            plt.show()

        else:
            print(f"Cannot visualize {dim}-dimensional phase space directly.")
            # Show pairwise plots for the first 3 dimensions
            # 显示前3个维度的成对图
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            axs[0].plot(phase_space[:, 0], phase_space[:, 1], 'b.', markersize=0.5)
            axs[0].set_title("Dimensions 1 vs 2")
            axs[0].set_xlabel("x(t)")
            axs[0].set_ylabel("x(t+τ)")
            axs[0].grid(True)

            axs[1].plot(phase_space[:, 1], phase_space[:, 2], 'b.', markersize=0.5)
            axs[1].set_title("Dimensions 2 vs 3")
            axs[1].set_xlabel("x(t+τ)")
            axs[1].set_ylabel("x(t+2τ)")
            axs[1].grid(True)

            axs[2].plot(phase_space[:, 0], phase_space[:, 2], 'b.', markersize=0.5)
            axs[2].set_title("Dimensions 1 vs 3")
            axs[2].set_xlabel("x(t)")
            axs[2].set_ylabel("x(t+2τ)")
            axs[2].grid(True)

            plt.tight_layout()
            plt.show()


def calculate_lyapunov_exponent(phase_space: np.ndarray, dt: float = 1.0, max_steps: int = 20) -> float:
    """
    Calculate the largest Lyapunov exponent from reconstructed phase space

    从重构的相空间计算最大李雅普诺夫指数

    Args:
        phase_space (np.ndarray): Reconstructed phase space
                                 重构的相空间
        dt (float): Time step
                   时间步长
        max_steps (int): Maximum number of steps for divergence calculation
                        发散计算的最大步数

    Returns:
        float: Largest Lyapunov exponent
               最大李雅普诺夫指数
    """
    # Number of points and dimension
    # 点数和维度
    n_points, dim = phase_space.shape

    # Find nearest neighbors for each point (excluding temporal neighbors)
    # 为每个点找到最近邻（排除时间上的邻居）
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(phase_space)

    # Initialize divergence tracking
    # 初始化发散跟踪
    divergences = np.zeros(max_steps)
    n_pairs = 0

    # Minimum temporal separation to avoid temporal correlations
    # 最小时间分离，以避免时间相关性
    min_temporal_separation = 10

    # For each point, find its nearest neighbor and track divergence
    # 对于每个点，找到其最近邻并跟踪发散
    for i in range(n_points - max_steps):
        # Find nearest neighbors
        # 寻找最近邻
        distances, indices = nbrs.kneighbors([phase_space[i]])

        # Find the nearest neighbor that is temporally separated
        # 找到时间上分离的最近邻
        for j in indices[0][1:]:
            if abs(i - j) > min_temporal_separation:
                # Initial distance
                # 初始距离
                d0 = np.linalg.norm(phase_space[i] - phase_space[j])

                # Track divergence over time
                # 随时间跟踪发散
                valid_pair = True
                for k in range(1, max_steps + 1):
                    if i + k < n_points and j + k < n_points:
                        # Current distance
                        # 当前距离
                        d = np.linalg.norm(phase_space[i + k] - phase_space[j + k])
                        if d > 0:
                            divergences[k - 1] += np.log(d / d0)
                        else:
                            valid_pair = False
                            break
                    else:
                        valid_pair = False
                        break

                if valid_pair:
                    n_pairs += 1
                break

    if n_pairs == 0:
        return 0.0

    # Average divergence at each step
    # 每一步的平均发散
    divergences /= n_pairs

    # Fit a line to the divergence curve
    # 对发散曲线拟合一条直线
    steps = np.arange(1, max_steps + 1)
    slope, _ = np.polyfit(steps * dt, divergences, 1)

    return slope


def calculate_correlation_dimension(phase_space: np.ndarray, max_radius: float = 10.0, n_points: int = 1000) -> float:
    """
    Calculate the correlation dimension from reconstructed phase space

    从重构的相空间计算相关维数

    Args:
        phase_space (np.ndarray): Reconstructed phase space
                                 重构的相空间
        max_radius (float): Maximum radius for correlation sum
                           相关和的最大半径
        n_points (int): Number of points to use for calculation (for efficiency)
                       用于计算的点数（为了效率）

    Returns:
        float: Correlation dimension
               相关维数
    """
    # Subsample if necessary
    # 如果需要，进行子采样
    if len(phase_space) > n_points:
        indices = np.random.choice(len(phase_space), n_points, replace=False)
        phase_space = phase_space[indices]

    # Calculate pairwise distances
    # 计算成对距离
    n = len(phase_space)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(phase_space[i] - phase_space[j])
            distances[i, j] = distances[j, i] = dist

    # Create radius array
    # 创建半径数组
    radii = np.logspace(-3, np.log10(max_radius), 20)

    # Calculate correlation sum for each radius
    # 计算每个半径的相关和
    correlation_sum = np.zeros_like(radii)

    for i, r in enumerate(radii):
        correlation_sum[i] = np.sum(distances < r) / (n * (n - 1))

    # Calculate correlation dimension as the slope of log(C(r)) vs log(r)
    # 计算相关维数，即log(C(r))对log(r)的斜率
    log_radii = np.log(radii)
    log_correlation = np.log(correlation_sum + 1e-10)  # Add small constant to avoid log(0)

    # Fit a line to the middle part of the curve (avoiding very small and very large r)
    # 对曲线的中间部分拟合一条直线（避开非常小和非常大的r）
    mid_start = len(log_radii) // 3
    mid_end = 2 * len(log_radii) // 3

    slope, _ = np.polyfit(log_radii[mid_start:mid_end], log_correlation[mid_start:mid_end], 1)

    return slope


if __name__ == "__main__":
    # Generate a test signal (Lorenz system)
    # 生成测试信号（洛伦兹系统）
    def lorenz_system(t, xyz, sigma=10, beta=8 / 3, rho=28):
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

    # Create phase space reconstructor
    # 创建相空间重构器
    reconstructor = PhaseSpaceReconstructor()

    # Reconstruct phase space
    # 重构相空间
    phase_space = reconstructor.reconstruct(x_series)

    print(f"Time series length: {len(x_series)}")
    print(f"Phase space shape: {phase_space.shape}")
    print(f"Estimated delay: {reconstructor.delay}")
    print(f"Estimated embedding dimension: {reconstructor.embedding_dim}")

    # Calculate Lyapunov exponent
    # 计算李雅普诺夫指数
    lyapunov = calculate_lyapunov_exponent(phase_space)
    print(f"Largest Lyapunov exponent: {lyapunov:.4f}")

    # Calculate correlation dimension
    # 计算相关维数
    corr_dim = calculate_correlation_dimension(phase_space)
    print(f"Correlation dimension: {corr_dim:.4f}")

    # Plot phase space
    # 绘制相空间
    reconstructor.plot_phase_space(phase_space, "Lorenz System Phase Space")