import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.neighbors import NearestNeighbors
import scipy.stats  # 新增：用于更丰富的统计特征计算


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
            use_entropy: bool = True,
            subsample_max_points: int = 1000,  # 新增：可配置子采样最大点数
            stat_features: List[str] = None  # 新增：可配置统计特征类型
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
            subsample_max_points (int): Maximum number of points for subsampling
                                       子采样的最大点数（控制计算复杂度）
            stat_features (List[str]): List of statistical features to compute. Default: ['mean', 'std', 'max', 'min', 'median', 'kurtosis', 'skewness']
                                      要计算的统计特征列表
        """
        super().__init__()
        self.pooling_type = pooling_type
        self.output_dim = output_dim
        self.epsilon_range = epsilon_range
        self.use_correlation_dim = use_correlation_dim
        self.use_lyapunov_dim = use_lyapunov_dim
        self.use_entropy = use_entropy
        self.subsample_max_points = subsample_max_points

        # 初始化统计特征列表（默认包含更多特征）
        self.stat_features = stat_features or [
            'mean', 'std', 'max', 'min', 'median', 'kurtosis', 'skewness'
        ]

        # 验证统计特征合法性
        valid_stats = {'mean', 'std', 'max', 'min', 'median', 'kurtosis', 'skewness'}
        if not set(self.stat_features).issubset(valid_stats):
            invalid = set(self.stat_features) - valid_stats
            raise ValueError(f"Invalid statistical features: {invalid}. Valid options: {valid_stats}")

        # 初始化投影层和归一化层
        self.projection = nn.Linear(self._get_feature_dim(), output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def _get_feature_dim(self) -> int:
        """获取原始池化特征的维度"""
        feature_count = 0

        if self.pooling_type in ('topological', 'combined'):
            # 拓扑特征计数
            if self.use_correlation_dim:
                feature_count += len(self.epsilon_range)
            if self.use_lyapunov_dim:
                feature_count += 1
            if self.use_entropy:
                feature_count += len(self.epsilon_range)

        if self.pooling_type in ('statistical', 'combined'):
            # 统计特征计数（每个维度计算指定的统计量）
            chaotic_dim = 3  # 混沌系统通常为3维，与混沌嵌入层保持一致
            feature_count += chaotic_dim * len(self.stat_features)

        return feature_count

    def _subsample_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        轨迹子采样（提取为独立方法，避免代码重复）

        Args:
            trajectory: 原始轨迹 [trajectory_points, chaotic_dim]

        Returns:
            子采样后的轨迹
        """
        n_points = len(trajectory)
        if n_points <= self.subsample_max_points:
            return trajectory

        # 随机子采样（保持轨迹顺序）
        indices = np.sort(np.random.choice(n_points, self.subsample_max_points, replace=False))
        return trajectory[indices]

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        吸引子池化层的前向传播

        Args:
            trajectories: 混沌轨迹 [batch_size, trajectory_points, chaotic_dim]

        Returns:
            池化特征 [batch_size, output_dim]
        """
        batch_size = trajectories.shape[0]
        pooled_features = []

        for i in range(batch_size):
            # 子采样轨迹以提高效率
            trajectory = self._subsample_trajectory(
                trajectories[i].detach().cpu().numpy()
            )
            features = self._extract_features(trajectory)
            pooled_features.append(features)

        # 转换为张量并保持设备一致性
        pooled_features = torch.tensor(
            np.array(pooled_features),
            dtype=trajectories.dtype,
            device=trajectories.device
        )

        # 投影和归一化
        output = self.projection(pooled_features)
        output = self.norm(output)

        return output

    def _extract_features(self, trajectory: np.ndarray) -> np.ndarray:
        """从单个轨迹提取特征"""
        features = []

        if self.pooling_type in ('topological', 'combined'):
            # 提取拓扑不变量
            if self.use_correlation_dim:
                corr_dims = self._calculate_correlation_dimension(trajectory)
                features.extend(corr_dims)

            if self.use_lyapunov_dim:
                lyap_dim = self._calculate_lyapunov_dimension(trajectory)
                features.append(lyap_dim)

            if self.use_entropy:
                entropies = self._calculate_kolmogorov_entropy(trajectory)
                features.extend(entropies)

        if self.pooling_type in ('statistical', 'combined'):
            # 提取统计特征
            stat_features = self._calculate_statistical_features(trajectory)
            features.extend(stat_features)

        # 处理异常值（增强稳定性）
        return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

    def _calculate_correlation_dimension(self, trajectory: np.ndarray) -> List[float]:
        """计算不同尺度下的相关维数"""
        trajectory = self._subsample_trajectory(trajectory)
        n_points = len(trajectory)

        # 计算成对距离（排除自身距离）
        nbrs = NearestNeighbors(n_neighbors=n_points, algorithm='ball_tree').fit(trajectory)
        distances, _ = nbrs.kneighbors(trajectory)
        distances = distances[:, 1:].flatten()  # 移除对角线（自身距离）

        corr_dims = []
        for epsilon in self.epsilon_range:
            if epsilon <= 1e-10:
                corr_dims.append(0.0)
                continue

            # 计算相关和 C(ε)
            count = np.sum(distances < epsilon)
            corr_sum = count / (n_points * (n_points - 1))  # 归一化

            # 计算相关维数（log(C(ε))/log(ε)）
            if corr_sum > 1e-10:
                try:
                    corr_dim = np.log(corr_sum) / np.log(epsilon)
                    # 限制异常值范围
                    corr_dims.append(np.clip(corr_dim, -10, 10))
                except (ValueError, ZeroDivisionError):
                    corr_dims.append(0.0)
            else:
                corr_dims.append(0.0)

        return corr_dims

    def _calculate_lyapunov_dimension(self, trajectory: np.ndarray) -> float:
        """计算李雅普诺夫维数（Kaplan-Yorke维度）"""
        trajectory = self._subsample_trajectory(trajectory)
        n_points, chaotic_dim = trajectory.shape

        # 寻找最近邻点
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(trajectory)  # 增加邻点数量提高稳定性
        distances, indices = nbrs.kneighbors(trajectory)

        # 估计最大李雅普诺夫指数（局部发散率）
        lyapunov_exponents = np.zeros(chaotic_dim)
        n_samples = min(100, n_points // 2)
        sample_indices = np.random.choice(n_points - 20, n_samples, replace=False)
        valid_samples = 0

        for i in sample_indices:
            # 选择多个近邻点提高鲁棒性
            for j in indices[i, 1:3]:  # 取前2个近邻
                if abs(i - j) < 10:
                    continue

                dx0 = trajectory[i] - trajectory[j]
                d0 = np.linalg.norm(dx0)
                if d0 < 1e-10:
                    continue

                dx0_norm = dx0 / d0  # 归一化初始差异
                valid_samples += 1

                # 跟踪多步演化
                for k in range(1, min(20, n_points - max(i, j))):
                    dx = trajectory[i + k] - trajectory[j + k]
                    proj = np.dot(dx, dx0_norm)
                    if proj > 1e-10:
                        lyapunov_exponents[0] += np.log(proj / d0) / k

        if valid_samples > 0:
            lyapunov_exponents[0] /= valid_samples
        else:
            lyapunov_exponents[0] = 0.0  # 无有效样本时设为0

        # 估计其他指数（更合理的衰减分布）
        for i in range(1, chaotic_dim):
            lyapunov_exponents[i] = lyapunov_exponents[0] * (0.5 ** i) * (-1 if i % 2 == 1 else 1)

        # 计算Kaplan-Yorke维度
        lyapunov_exponents = np.sort(lyapunov_exponents)[::-1]  # 降序排列
        sum_exponents = 0.0
        j = 0

        for j in range(chaotic_dim):
            sum_exponents += lyapunov_exponents[j]
            if sum_exponents <= 0:
                break

        if j == 0 and sum_exponents <= 0:
            return 0.0
        elif j < chaotic_dim - 1 and sum_exponents > 0 and lyapunov_exponents[j + 1] < 0:
            return j + sum_exponents / abs(lyapunov_exponents[j + 1])
        else:
            return float(chaotic_dim)

    def _calculate_kolmogorov_entropy(self, trajectory: np.ndarray) -> List[float]:
        """计算不同尺度下的柯尔莫哥洛夫熵"""
        trajectory = self._subsample_trajectory(trajectory)
        n_points = len(trajectory)

        # 计算成对距离
        nbrs = NearestNeighbors(n_neighbors=n_points, algorithm='ball_tree').fit(trajectory)
        distances, _ = nbrs.kneighbors(trajectory)

        entropies = []
        for epsilon in self.epsilon_range:
            if epsilon <= 1e-10:
                entropies.append(0.0)
                continue

            recurrence_plot = distances < epsilon
            entropy = self._calculate_entropy_from_recurrence(recurrence_plot)
            entropies.append(np.clip(entropy, 0, 10))  # 限制熵值范围

        return entropies

    def _calculate_entropy_from_recurrence(self, recurrence_plot: np.ndarray) -> float:
        """从递归图计算熵（基于对角线长度分布）"""
        n = recurrence_plot.shape[0]
        min_length = 2
        diag_lengths = []

        # 遍历所有非主对角线
        for i in range(-(n - min_length), n - min_length + 1):
            if i == 0:
                continue

            diag = np.diag(recurrence_plot, k=i)
            current_length = 0

            for val in diag:
                if val:
                    current_length += 1
                else:
                    if current_length >= min_length:
                        diag_lengths.append(current_length)
                    current_length = 0

            if current_length >= min_length:
                diag_lengths.append(current_length)

        if not diag_lengths:
            return 0.0

        # 计算长度分布的熵
        max_len = max(diag_lengths)
        hist, _ = np.histogram(diag_lengths, bins=range(min_length, max_len + 2))
        hist = hist[hist > 0]  # 过滤零计数
        if len(hist) == 0:
            return 0.0

        p = hist / np.sum(hist)
        return -np.sum(p * np.log(p + 1e-10))

    def _calculate_statistical_features(self, trajectory: np.ndarray) -> List[float]:
        """计算统计特征（支持多种可配置统计量）"""
        features = []
        n_points, chaotic_dim = trajectory.shape

        for dim in range(chaotic_dim):
            series = trajectory[:, dim]  # 每个维度单独计算

            for stat in self.stat_features:
                if stat == 'mean':
                    features.append(np.mean(series))
                elif stat == 'std':
                    features.append(np.std(series))
                elif stat == 'max':
                    features.append(np.max(series))
                elif stat == 'min':
                    features.append(np.min(series))
                elif stat == 'median':
                    features.append(np.median(series))
                elif stat == 'kurtosis':
                    # 峰度（衡量分布的陡峭程度）
                    features.append(scipy.stats.kurtosis(series))
                elif stat == 'skewness':
                    # 偏度（衡量分布的不对称性）
                    features.append(scipy.stats.skew(series))

        return features


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
        super().__init__()
        self.chaotic_dim = chaotic_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(chaotic_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 特征提取网络
        layers = []
        input_dim = chaotic_dim

        for _ in range(num_layers):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.feature_extractor = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the differentiable attractor pooling layer

        可微分吸引子池化层的前向传播

        Args:
            trajectories: Chaotic trajectories [batch_size, trajectory_points, chaotic_dim]
        """
        # 计算注意力权重
        attn_weights = self.attention(trajectories)  # [batch_size, trajectory_points, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # 沿时间维度归一化

        # 加权求和
        weighted_trajectory = torch.sum(attn_weights * trajectories, dim=1)  # [batch_size, chaotic_dim]

        # 特征提取和归一化
        output = self.feature_extractor(weighted_trajectory)
        output = self.norm(output)

        return output
