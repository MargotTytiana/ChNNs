import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


class AttractorPooling(nn.Module):
    """
    Attractor pooling layer for chaotic trajectories

    混沌轨迹的吸引子池化层
    """

    def __init__(
            self,
            pooling_type: str = 'topological',
            output_dim: int = 64,
            epsilon_range: List[float] = [0.01, 0.1, 0.5, 1.0],
            use_correlation_dim: bool = True,
            use_lyapunov_dim: bool = True,
            use_entropy: bool = True
    ):
        """
        Initialize the attractor pooling layer

        初始化吸引子池化层

        Args:
            pooling_type (str): Type of pooling ('topological', 'statistical', 'combined')
                               池化类型（'topological'拓扑型, 'statistical'统计型, 'combined'组合型）
            output_dim (int): Output dimension after pooling
                             池化后的输出维度
            epsilon_range (List[float]): Range of epsilon values for correlation dimension calculation
                                        用于计算相关维数的epsilon值范围
            use_correlation_dim (bool): Whether to use correlation dimension
                                       是否使用相关维数
            use_lyapunov_dim (bool): Whether to use Lyapunov dimension
                                    是否使用李雅普诺夫维数
            use_entropy (bool): Whether to use Kolmogorov entropy
                               是否使用柯尔莫哥洛夫熵
        """
        super().__init__()
        self.pooling_type = pooling_type
        self.output_dim = output_dim
        self.epsilon_range = epsilon_range
        self.use_correlation_dim = use_correlation_dim
        self.use_lyapunov_dim = use_lyapunov_dim
        self.use_entropy = use_entropy

        # Initialize projection layer
        # 初始化投影层
        self.projection = nn.Linear(self._get_feature_dim(), output_dim)

        # Initialize normalization layer
        # 初始化归一化层
        self.norm = nn.LayerNorm(output_dim)

    def _get_feature_dim(self) -> int:
        """
        Get the dimension of the raw pooled features

        获取原始池化特征的维度

        Returns:
            int: Feature dimension
                 特征维度
        """
        # Base feature count
        # 基础特征数量
        feature_count = 0

        if self.pooling_type == 'topological' or self.pooling_type == 'combined':
            # Topological invariants
            # 拓扑不变量
            if self.use_correlation_dim:
                feature_count += len(self.epsilon_range)  # Correlation dimension at different scales
            if self.use_lyapunov_dim:
                feature_count += 1  # Lyapunov dimension
            if self.use_entropy:
                feature_count += len(self.epsilon_range)  # Entropy at different scales

        if self.pooling_type == 'statistical' or self.pooling_type == 'combined':
            # Statistical features (mean, std, max, min, etc.)
            # 统计特征（均值、标准差、最大值、最小值等）
            feature_count += 5  # Mean, std, max, min, median per dimension

        return feature_count

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attractor pooling layer

        吸引子池化层的前向传播

        Args:
            trajectories (torch.Tensor): Chaotic trajectories [batch_size, trajectory_points, chaotic_dim]
                                        混沌轨迹 [batch_size, trajectory_points, chaotic_dim]

        Returns:
            torch.Tensor: Pooled features [batch_size, output_dim]
                         池化特征 [batch_size, output_dim]
        """
        batch_size = trajectories.shape[0]

        # Extract features from trajectories
        # 从轨迹中提取特征
        pooled_features = []

        for i in range(batch_size):
            # Get trajectory for this sample
            # 获取此样本的轨迹
            trajectory = trajectories[i].detach().cpu().numpy()

            # Extract features based on pooling type
            # 根据池化类型提取特征
            features = self._extract_features(trajectory)

            pooled_features.append(features)

        # Convert to tensor and move to the same device as input
        # 转换为张量并移动到与输入相同的设备
        pooled_features = torch.tensor(
            np.array(pooled_features),
            dtype=trajectories.dtype,
            device=trajectories.device
        )

        # Project to the desired output dimension
        # 投影到所需的输出维度
        output = self.projection(pooled_features)

        # Apply normalization
        # 应用归一化
        output = self.norm(output)

        return output

    def _extract_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Extract features from a single trajectory

        从单个轨迹中提取特征

        Args:
            trajectory (np.ndarray): Single trajectory [trajectory_points, chaotic_dim]
                                    单个轨迹 [trajectory_points, chaotic_dim]

        Returns:
            np.ndarray: Extracted features
                       提取的特征
        """
        features = []

        if self.pooling_type == 'topological' or self.pooling_type == 'combined':
            # Extract topological invariants
            # 提取拓扑不变量
            if self.use_correlation_dim:
                corr_dims = self._calculate_correlation_dimension(trajectory, self.epsilon_range)
                features.extend(corr_dims)

            if self.use_lyapunov_dim:
                lyap_dim = self._calculate_lyapunov_dimension(trajectory)
                features.append(lyap_dim)

            if self.use_entropy:
                entropies = self._calculate_kolmogorov_entropy(trajectory, self.epsilon_range)
                features.extend(entropies)

        if self.pooling_type == 'statistical' or self.pooling_type == 'combined':
            # Extract statistical features
            # 提取统计特征
            stat_features = self._calculate_statistical_features(trajectory)
            features.extend(stat_features)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return np.array(features)

    def _calculate_correlation_dimension(self, trajectory: np.ndarray, epsilon_range: List[float]) -> List[float]:
        """
        Calculate correlation dimension at different scales

        在不同尺度上计算相关维数

        Args:
            trajectory (np.ndarray): Single trajectory
                                    单个轨迹
            epsilon_range (List[float]): Range of epsilon values
                                        epsilon值范围

        Returns:
            List[float]: Correlation dimensions at different scales
                        不同尺度的相关维数
        """
        n_points = len(trajectory)

        # Subsample if trajectory is too long
        # 如果轨迹太长，则进行子采样
        max_points = 1000
        if n_points > max_points:
            indices = np.random.choice(n_points, max_points, replace=False)
            trajectory = trajectory[indices]
            n_points = max_points

        # Calculate pairwise distances
        # 计算成对距离
        nbrs = NearestNeighbors(n_neighbors=n_points, algorithm='ball_tree').fit(trajectory)
        distances, _ = nbrs.kneighbors(trajectory)

        # Flatten the distances array (excluding self-distances)
        # 展平距离数组（排除自身距离）
        distances = distances[:, 1:].flatten()

        # Calculate correlation sum for each epsilon
        # 计算每个epsilon的相关和
        corr_dims = []
        for epsilon in epsilon_range:
            # Skip very small epsilon to avoid log(0)
            if epsilon < 1e-10:
                corr_dims.append(0.0)
                continue

            # Count pairs with distance less than epsilon
            # 计算距离小于epsilon的对数
            count = np.sum(distances < epsilon)

            # Calculate correlation sum
            # 计算相关和
            corr_sum = count / (n_points * (n_points - 1))

            # Estimate correlation dimension as log(C(r)) / log(r)
            # 估计相关维数为log(C(r)) / log(r)
            if corr_sum > 1e-10 and epsilon > 1e-10 and abs(np.log(epsilon)) > 1e-10:
                corr_dim = np.log(corr_sum) / np.log(epsilon)
                corr_dims.append(corr_dim)
            else:
                corr_dims.append(0.0)

        return corr_dims

    def _calculate_lyapunov_dimension(self, trajectory: np.ndarray) -> float:
        """
        Calculate Lyapunov dimension (Kaplan-Yorke dimension)

        计算李雅普诺夫维数（Kaplan-Yorke维数）

        Args:
            trajectory (np.ndarray): Single trajectory
                                    单个轨迹

        Returns:
            float: Lyapunov dimension
                  李雅普诺夫维数
        """
        # This is a simplified implementation
        # 这是一个简化的实现

        # Calculate local divergence rates
        # 计算局部发散率
        n_points = len(trajectory)
        chaotic_dim = trajectory.shape[1]

        # Subsample if trajectory is too long
        # 如果轨迹太长，则进行子采样
        max_points = 500
        if n_points > max_points:
            indices = np.random.choice(n_points, max_points, replace=False)
            trajectory = trajectory[indices]
            n_points = max_points

        # Find nearest neighbors
        # 寻找最近邻
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(trajectory)
        distances, indices = nbrs.kneighbors(trajectory)

        # Calculate divergence rates
        # 计算发散率
        lyapunov_exponents = np.zeros(chaotic_dim)

        # Use only a subset of points for efficiency
        # 为了效率，仅使用点的子集
        n_samples = min(100, n_points // 2)
        sample_indices = np.random.choice(n_points - 20, n_samples, replace=False)

        for i in sample_indices:
            # Get nearest neighbor
            # 获取最近邻
            j = indices[i, 1]

            # Skip if too close in time
            # 如果在时间上太接近，则跳过
            if abs(i - j) < 10:
                continue

            # Initial state difference
            # 初始状态差异
            dx0 = trajectory[i] - trajectory[j]
            d0 = np.linalg.norm(dx0)

            if d0 < 1e-10:
                continue

            # Normalize initial difference
            # 归一化初始差异
            dx0 = dx0 / d0

            # Track evolution for a few steps
            # 跟踪几步的演化
            for k in range(1, min(20, n_points - max(i, j))):
                # State difference after k steps
                # k步后的状态差异
                dx = trajectory[i + k] - trajectory[j + k]

                # Project along initial direction
                # 沿初始方向投影
                proj = np.dot(dx, dx0)

                # Update Lyapunov exponent estimate
                # 更新李雅普诺夫指数估计
                if proj > 0:
                    lyapunov_exponents[0] += np.log(proj / d0) / k

        # Average over samples
        # 对样本取平均
        if n_samples > 0:
            lyapunov_exponents[0] /= n_samples

        # Estimate other exponents (simplified)
        # 估计其他指数（简化）
        for i in range(1, chaotic_dim):
            lyapunov_exponents[i] = lyapunov_exponents[0] / (i + 1) * (-1 if i % 2 == 1 else 1)

        # Sort exponents in descending order
        # 按降序排列指数
        lyapunov_exponents = np.sort(lyapunov_exponents)[::-1]

        # Calculate Kaplan-Yorke dimension
        # 计算Kaplan-Yorke维数
        j = 0
        sum_exponents = 0

        for j in range(chaotic_dim):
            sum_exponents += lyapunov_exponents[j]
            if sum_exponents < 0:
                break

        if j > 0 and j < chaotic_dim and lyapunov_exponents[j] < 0:
            # Kaplan-Yorke formula
            # Kaplan-Yorke公式
            kaplan_yorke_dim = j + sum_exponents / abs(lyapunov_exponents[j])
            return kaplan_yorke_dim
        else:
            return float(chaotic_dim)

    def _calculate_kolmogorov_entropy(self, trajectory: np.ndarray, epsilon_range: List[float]) -> List[float]:
        """
        Calculate Kolmogorov entropy at different scales

        在不同尺度上计算柯尔莫哥洛夫熵

        Args:
            trajectory (np.ndarray): Single trajectory
                                    单个轨迹
            epsilon_range (List[float]): Range of epsilon values
                                        epsilon值范围

        Returns:
            List[float]: Kolmogorov entropy at different scales
                        不同尺度的柯尔莫哥洛夫熵
        """
        n_points = len(trajectory)

        # Subsample if trajectory is too long
        # 如果轨迹太长，则进行子采样
        max_points = 1000
        if n_points > max_points:
            indices = np.random.choice(n_points, max_points, replace=False)
            trajectory = trajectory[indices]
            n_points = max_points

        # Calculate pairwise distances
        # 计算成对距离
        nbrs = NearestNeighbors(n_neighbors=n_points, algorithm='ball_tree').fit(trajectory)
        distances, _ = nbrs.kneighbors(trajectory)

        # Calculate entropy for each epsilon
        # 计算每个epsilon的熵
        entropies = []
        for epsilon in epsilon_range:
            # Create recurrence plot
            # 创建递归图
            recurrence_plot = distances < epsilon

            # Calculate entropy based on recurrence plot
            # 基于递归图计算熵
            entropy = self._calculate_entropy_from_recurrence(recurrence_plot)
            entropies.append(entropy)

        return entropies

    def _calculate_entropy_from_recurrence(self, recurrence_plot: np.ndarray) -> float:
        """
        Calculate entropy from recurrence plot

        从递归图计算熵

        Args:
            recurrence_plot (np.ndarray): Recurrence plot
                                         递归图

        Returns:
            float: Entropy value
                  熵值
        """
        # Find diagonal lines
        # 寻找对角线
        n = recurrence_plot.shape[0]
        min_length = 2  # Minimum diagonal line length

        # Count diagonal lines of different lengths
        # 计算不同长度的对角线数量
        diag_lengths = []

        for i in range(-(n - min_length), n - min_length + 1):
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
                        if count >= min_length:
                            diag_lengths.append(count)
                        count = 0
                if count >= min_length:
                    diag_lengths.append(count)

        # Calculate entropy from diagonal line length distribution
        # 从对角线长度分布计算熵
        if len(diag_lengths) > 0:
            # Create histogram of diagonal lengths
            # 创建对角线长度的直方图
            max_length = max(diag_lengths) if diag_lengths else 0
            hist, _ = np.histogram(diag_lengths, bins=range(min_length, max_length + 2))

            # Calculate probability distribution
            # 计算概率分布
            p = hist / np.sum(hist) if np.sum(hist) > 0 else np.zeros_like(hist)

            # Calculate entropy
            # 计算熵
            entropy = -np.sum(p * np.log(p + 1e-10))
            return entropy
        else:
            return 0.0

    def _calculate_statistical_features(self, trajectory: np.ndarray) -> List[float]:
        """
        Calculate statistical features from trajectory

        从轨迹计算统计特征

        Args:
            trajectory (np.ndarray): Single trajectory
                                    单个轨迹

        Returns:
            List[float]: Statistical features
                        统计特征
        """
        # Calculate basic statistics
        # 计算基本统计量
        mean = np.mean(trajectory, axis=0)
        std = np.std(trajectory, axis=0)
        max_val = np.max(trajectory, axis=0)
        min_val = np.min(trajectory, axis=0)
        median = np.median(trajectory, axis=0)

        # Flatten and concatenate
        # 展平并连接
        features = np.concatenate([mean, std, max_val, min_val, median])

        return features.tolist()


class DifferentiableAttractorPooling(nn.Module):
    """
    Differentiable attractor pooling layer

    可微分吸引子池化层
    """

    def __init__(
            self,
            chaotic_dim: int = 3,
            hidden_dim: int = 128,
            output_dim: int = 64,
            num_layers: int = 2
    ):
        """
        Initialize the differentiable attractor pooling layer

        初始化可微分吸引子池化层

        Args:
            chaotic_dim (int): Dimension of the chaotic system
                              混沌系统的维度
            hidden_dim (int): Hidden dimension of the network
                             网络的隐藏维度
            output_dim (int): Output dimension after pooling
                             池化后的输出维度
            num_layers (int): Number of layers in the network
                             网络的层数
        """
        super().__init__()
        self.chaotic_dim = chaotic_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Create attention mechanism
        # 创建注意力机制
        self.attention = nn.Sequential(
            nn.Linear(chaotic_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Create feature extraction network
        # 创建特征提取网络
        layers = []
        input_dim = chaotic_dim

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.feature_extractor = nn.Sequential(*layers)

        # Initialize normalization layer
        # 初始化归一化层
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the differentiable attractor pooling layer

        可微分吸引子池化层的前向传播

        Args:
            trajectories (torch.Tensor): Chaotic trajectories [batch_size, trajectory_points, chaotic_dim]
                                        混沌轨迹 [batch_size, trajectory_points, chaotic_dim]

        Returns:
            torch.Tensor: Pooled features [batch_size, output_dim]
                         池化特征 [batch_size, output_dim]
        """
        batch_size, trajectory_points, _ = trajectories.shape

        # Calculate attention weights
        # 计算注意力权重
        attention_scores = self.attention(trajectories)  # [batch_size, trajectory_points, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, trajectory_points, 1]

        # Apply attention to trajectories
        # 将注意力应用于轨迹
        weighted_trajectories = trajectories * attention_weights  # [batch_size, trajectory_points, chaotic_dim]

        # Sum along time dimension
        # 沿时间维度求和
        pooled_trajectories = torch.sum(weighted_trajectories, dim=1)  # [batch_size, chaotic_dim]

        # Extract features
        # 提取特征
        features = self.feature_extractor(pooled_trajectories)  # [batch_size, output_dim]

        # Apply normalization
        # 应用归一化
        output = self.norm(features)

        return output


class HierarchicalAttractorPooling(nn.Module):
    """
    Hierarchical attractor pooling layer

    层次化吸引子池化层
    """

    def __init__(
            self,
            chaotic_dim: int = 3,
            hidden_dim: int = 128,
            output_dim: int = 64,
            num_levels: int = 3,
            window_sizes: List[int] = [10, 20, 50]
    ):
        """
        Initialize the hierarchical attractor pooling layer

        初始化层次化吸引子池化层

        Args:
            chaotic_dim (int): Dimension of the chaotic system
                              混沌系统的维度
            hidden_dim (int): Hidden dimension of the network
                             网络的隐藏维度
            output_dim (int): Output dimension after pooling
                             池化后的输出维度
            num_levels (int): Number of hierarchical levels
                             层次级别的数量
            window_sizes (List[int]): Window sizes for each level
                                     每个级别的窗口大小
        """
        super().__init__()
        self.chaotic_dim = chaotic_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_levels = num_levels
        self.window_sizes = window_sizes[:num_levels]

        # Ensure we have enough window sizes
        # 确保我们有足够的窗口大小
        if len(self.window_sizes) < num_levels:
            self.window_sizes.extend([self.window_sizes[-1]] * (num_levels - len(self.window_sizes)))

        # Create pooling layers for each level
        # 为每个级别创建池化层
        self.level_poolers = nn.ModuleList([
            DifferentiableAttractorPooling(
                chaotic_dim=chaotic_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=2
            )
            for _ in range(num_levels)
        ])

        # Create final projection layer
        # 创建最终投影层
        self.projection = nn.Linear(hidden_dim * num_levels, output_dim)

        # Initialize normalization layer
        # 初始化归一化层
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hierarchical attractor pooling layer

        层次化吸引子池化层的前向传播

        Args:
            trajectories (torch.Tensor): Chaotic trajectories [batch_size, trajectory_points, chaotic_dim]
                                        混沌轨迹 [batch_size, trajectory_points, chaotic_dim]

        Returns:
            torch.Tensor: Pooled features [batch_size, output_dim]
                         池化特征 [batch_size, output_dim]
        """
        batch_size, trajectory_points, _ = trajectories.shape

        # Apply pooling at each level
        # 在每个级别应用池化
        level_features = []

        for i, pooler in enumerate(self.level_poolers):
            # Apply windowing
            # 应用窗口
            window_size = min(self.window_sizes[i], trajectory_points)
            stride = max(1, (trajectory_points - window_size) // (10 - 1))

            windows = []
            for j in range(0, trajectory_points - window_size + 1, stride):
                window = trajectories[:, j:j + window_size, :]
                windows.append(window)

            # If no windows were created, use the entire trajectory
            # 如果没有创建窗口，则使用整个轨迹
            if not windows:
                windows = [trajectories]

            # Process each window
            # 处理每个窗口
            window_features = []
            for window in windows:
                features = pooler(window)
                window_features.append(features)

            # Aggregate window features
            # 聚合窗口特征
            if len(window_features) > 1:
                window_features = torch.stack(window_features, dim=1)  # [batch_size, num_windows, hidden_dim]
                # Max pooling over windows
                # 对窗口进行最大池化
                level_feature, _ = torch.max(window_features, dim=1)  # [batch_size, hidden_dim]
            else:
                level_feature = window_features[0]

            level_features.append(level_feature)

        # Concatenate features from all levels
        # 连接所有级别的特征
        combined_features = torch.cat(level_features, dim=1)  # [batch_size, hidden_dim * num_levels]

        # Project to output dimension
        # 投影到输出维度
        output = self.projection(combined_features)  # [batch_size, output_dim]

        # Apply normalization
        # 应用归一化
        output = self.norm(output)

        return output


def visualize_attractor_pooling(trajectory: np.ndarray, pooled_features: np.ndarray,
                                title: str = "Attractor Pooling Visualization"):
    """
    Visualize the attractor pooling process

    可视化吸引子池化过程

    Args:
        trajectory (np.ndarray): Original trajectory
                                原始轨迹
        pooled_features (np.ndarray): Pooled features
                                     池化特征
        title (str): Plot title
                    图表标题
    """
    fig = plt.figure(figsize=(15, 5))

    # Plot original trajectory
    # 绘制原始轨迹
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=0.5)
    ax1.set_title("Original Chaotic Trajectory")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    # Plot pooled features
    # 绘制池化特征
    ax2 = fig.add_subplot(122)
    ax2.bar(range(len(pooled_features)), pooled_features)
    ax2.set_title("Pooled Topological Features")
    ax2.set_xlabel("Feature Index")
    ax2.set_ylabel("Feature Value")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate a sample trajectory (Lorenz system)
    def lorenz_system(t, xyz, sigma=10, beta=8 / 3, rho=28):
        x, y, z = xyz
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    from scipy.integrate import solve_ivp

    # Initial conditions
    xyz0 = [1.0, 1.0, 1.0]

    # Time points
    t = np.linspace(0, 50, 5000)

    # Solve the differential equations
    solution = solve_ivp(lorenz_system, [0, 50], xyz0, t_eval=t)

    # Extract trajectory
    trajectory = np.column_stack([solution.y[0], solution.y[1], solution.y[2]])

    # Create a batch of trajectories
    batch_size = 4
    trajectories = np.array([trajectory] * batch_size)
    trajectories_tensor = torch.tensor(trajectories, dtype=torch.float32)

    # Create attractor pooling layers
    pooling_layer = AttractorPooling(output_dim=32)

    print("\n===== DEBUG INFO =====")
    print("Feature dim (projection input):", pooling_layer._get_feature_dim())

    # 原始拓扑特征（只看第一个轨迹）
    raw_features = pooling_layer._extract_features(trajectory)
    print("Raw topological features:", raw_features)
    print("Raw features shape:", raw_features.shape)

    # 投影后的特征
    pooled_features = pooling_layer(trajectories_tensor)
    print("Pooled features shape:", pooled_features.shape)
    print("First pooled feature vector:", pooled_features[0].detach().numpy())

    # 可视化对比
    visualize_attractor_pooling(
        trajectory,
        raw_features,  # 原始特征
        "Raw Topological Features"
    )
    visualize_attractor_pooling(
        trajectory,
        pooled_features[0].detach().numpy(),  # 投影后的特征
        "Projected Topological Features"
    )

