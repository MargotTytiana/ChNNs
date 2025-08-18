"""
混沌特征提取算法 - 混沌神经网络说话人识别项目
实现最大李雅普诺夫指数(MLE)、递归量化分析(RQA)、分形维数等混沌特征
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
from scipy.spatial.distance import pdist, squareform
from scipy import signal
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class ChaosFeatureExtractor(nn.Module):
    """混沌特征提取器主类"""

    def __init__(self,
                 sample_rate: int = 16000,
                 frame_length: int = 512,
                 hop_length: int = 256,
                 embedding_dim: int = 3,
                 delay: int = 10):
        """
        初始化混沌特征提取器

        Args:
            sample_rate: 音频采样率
            frame_length: 帧长度
            hop_length: 跳跃长度
            embedding_dim: 相空间嵌入维度
            delay: 时间延迟参数τ
        """
        super(ChaosFeatureExtractor, self).__init__()
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.embedding_dim = embedding_dim
        self.delay = delay

        # 初始化传统特征提取器
        self.mel_transform = librosa.filters.mel(
            sr=sample_rate,
            n_fft=frame_length,
            n_mels=80
        )

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播，提取所有混沌特征

        Args:
            audio: 音频张量 [batch_size, 1, samples]

        Returns:
            包含所有特征的字典
        """
        batch_size = audio.shape[0]
        features_dict = {}

        # 处理每个样本
        for i in range(batch_size):
            audio_sample = audio[i, 0].cpu().numpy()  # [samples]

            # 提取传统特征作为基准
            mfcc_feat = self._extract_mfcc(audio_sample)
            mel_feat = self._extract_mel_spectrogram(audio_sample)

            # 提取混沌特征
            mle_feat = self._extract_maximum_lyapunov_exponent(audio_sample)
            rqa_feat = self._extract_recurrence_quantification(audio_sample)
            fractal_feat = self._extract_fractal_dimension(audio_sample)
            phase_space_feat = self._extract_phase_space_features(audio_sample)

            # 组合特征
            if i == 0:
                features_dict['mfcc'] = torch.zeros(batch_size, mfcc_feat.shape[0], mfcc_feat.shape[1])
                features_dict['mel'] = torch.zeros(batch_size, mel_feat.shape[0], mel_feat.shape[1])
                features_dict['mle'] = torch.zeros(batch_size, len(mle_feat))
                features_dict['rqa'] = torch.zeros(batch_size, len(rqa_feat))
                features_dict['fractal'] = torch.zeros(batch_size, len(fractal_feat))
                features_dict['phase_space'] = torch.zeros(batch_size, len(phase_space_feat))

            features_dict['mfcc'][i] = torch.from_numpy(mfcc_feat).float()
            features_dict['mel'][i] = torch.from_numpy(mel_feat).float()
            features_dict['mle'][i] = torch.from_numpy(np.array(mle_feat)).float()
            features_dict['rqa'][i] = torch.from_numpy(np.array(rqa_feat)).float()
            features_dict['fractal'][i] = torch.from_numpy(np.array(fractal_feat)).float()
            features_dict['phase_space'][i] = torch.from_numpy(np.array(phase_space_feat)).float()

        return features_dict

    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """提取MFCC特征"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return mfcc

    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """提取Mel频谱图"""
        stft = librosa.stft(
            audio,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        magnitude = np.abs(stft)
        mel_spec = np.dot(self.mel_transform, magnitude)
        log_mel_spec = librosa.power_to_db(mel_spec)
        return log_mel_spec

    def _extract_maximum_lyapunov_exponent(self, audio: np.ndarray) -> List[float]:
        """
        提取最大李雅普诺夫指数 (MLE)
        使用Wolf算法计算混沌系统的李雅普诺夫指数
        """
        try:
            # 相空间重构
            embedded = self._phase_space_reconstruction(audio)

            # 计算李雅普诺夫指数的多个尺度版本
            mle_features = []

            # 不同时间尺度的MLE
            scales = [1, 5, 10, 20]
            for scale in scales:
                mle = self._compute_lyapunov_exponent(embedded, scale)
                mle_features.append(mle)

            # 添加统计特征
            mle_features.extend([
                np.mean(mle_features),
                np.std(mle_features),
                np.max(mle_features),
                np.min(mle_features)
            ])

            return mle_features

        except Exception as e:
            print(f"MLE计算错误: {e}")
            return [0.0] * 8  # 返回零值作为fallback

    def _phase_space_reconstruction(self, signal: np.ndarray) -> np.ndarray:
        """
        相空间重构 - Takens嵌入定理

        Args:
            signal: 一维时间序列

        Returns:
            重构的相空间矩阵 [n_points, embedding_dim]
        """
        n = len(signal)
        m = self.embedding_dim
        tau = self.delay

        # 计算嵌入向量的数量
        n_vectors = n - (m - 1) * tau

        if n_vectors <= 0:
            raise ValueError("信号长度不足以进行相空间重构")

        # 构建嵌入矩阵
        embedded = np.zeros((n_vectors, m))

        for i in range(m):
            embedded[:, i] = signal[i * tau:i * tau + n_vectors]

        return embedded

    def _compute_lyapunov_exponent(self, embedded: np.ndarray, evolution_time: int = 10) -> float:
        """
        计算李雅普诺夫指数

        Args:
            embedded: 相空间重构后的轨迹
            evolution_time: 轨迹演化时间

        Returns:
            最大李雅普诺夫指数
        """
        try:
            n_points, dim = embedded.shape

            if n_points < evolution_time + 10:
                return 0.0

            # 寻找最近邻点
            nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(embedded)
            distances, indices = nbrs.kneighbors(embedded[:-evolution_time])

            # 计算初始距离
            initial_distances = distances[:, 1]  # 排除自己，取最近邻

            # 计算演化后的距离
            evolved_distances = np.zeros_like(initial_distances)

            for i in range(len(initial_distances)):
                if i + evolution_time < n_points and indices[i, 1] + evolution_time < n_points:
                    evolved_distances[i] = np.linalg.norm(
                        embedded[i + evolution_time] - embedded[indices[i, 1] + evolution_time]
                    )
                else:
                    evolved_distances[i] = initial_distances[i]  # fallback

            # 避免除零和对数为负
            valid_mask = (initial_distances > 1e-8) & (evolved_distances > 1e-8)

            if np.sum(valid_mask) == 0:
                return 0.0

            # 计算李雅普诺夫指数
            lyap = np.mean(np.log(evolved_distances[valid_mask] / initial_distances[valid_mask])) / evolution_time

            return float(lyap)

        except Exception as e:
            print(f"李雅普诺夫指数计算错误: {e}")
            return 0.0

    def _extract_recurrence_quantification(self, audio: np.ndarray) -> List[float]:
        """
        递归量化分析 (RQA)
        分析信号的递归结构和周期性
        """
        try:
            # 相空间重构
            embedded = self._phase_space_reconstruction(audio)

            # 计算递归矩阵
            recurrence_matrix = self._compute_recurrence_matrix(embedded)

            # 提取RQA特征
            rqa_features = []

            # 递归率 (Recurrence Rate)
            rr = np.sum(recurrence_matrix) / (recurrence_matrix.shape[0] ** 2)
            rqa_features.append(rr)

            # 确定性 (Determinism)
            det = self._compute_determinism(recurrence_matrix)
            rqa_features.append(det)

            # 平均对角线长度
            avg_diag_length = self._compute_average_diagonal_length(recurrence_matrix)
            rqa_features.append(avg_diag_length)

            # 最大对角线长度
            max_diag_length = self._compute_max_diagonal_length(recurrence_matrix)
            rqa_features.append(max_diag_length)

            # 熵 (Entropy)
            entropy = self._compute_recurrence_entropy(recurrence_matrix)
            rqa_features.append(entropy)

            # 层流性 (Laminarity)
            lam = self._compute_laminarity(recurrence_matrix)
            rqa_features.append(lam)

            return rqa_features

        except Exception as e:
            print(f"RQA计算错误: {e}")
            return [0.0] * 6  # 返回零值作为fallback

    def _compute_recurrence_matrix(self, embedded: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """计算递归矩阵"""
        distances = squareform(pdist(embedded))
        threshold_value = threshold * np.std(distances)
        recurrence_matrix = (distances < threshold_value).astype(int)
        return recurrence_matrix

    def _compute_determinism(self, recurrence_matrix: np.ndarray, min_length: int = 2) -> float:
        """计算确定性指标"""
        n = recurrence_matrix.shape[0]
        diagonal_lengths = []

        # 计算所有对角线长度
        for offset in range(1 - n, n):
            diagonal = np.diag(recurrence_matrix, k=offset)
            length = 0
            current_length = 0

            for val in diagonal:
                if val == 1:
                    current_length += 1
                else:
                    if current_length >= min_length:
                        diagonal_lengths.append(current_length)
                    current_length = 0

            if current_length >= min_length:
                diagonal_lengths.append(current_length)

        if len(diagonal_lengths) == 0:
            return 0.0

        total_recurrence_points = np.sum(recurrence_matrix)
        diagonal_points = sum(diagonal_lengths)

        return diagonal_points / max(total_recurrence_points, 1)

    def _compute_average_diagonal_length(self, recurrence_matrix: np.ndarray) -> float:
        """计算平均对角线长度"""
        diagonal_lengths = []
        n = recurrence_matrix.shape[0]

        for offset in range(1 - n, n):
            diagonal = np.diag(recurrence_matrix, k=offset)
            current_length = 0

            for val in diagonal:
                if val == 1:
                    current_length += 1
                else:
                    if current_length >= 2:
                        diagonal_lengths.append(current_length)
                    current_length = 0

            if current_length >= 2:
                diagonal_lengths.append(current_length)

        return np.mean(diagonal_lengths) if diagonal_lengths else 0.0

    def _compute_max_diagonal_length(self, recurrence_matrix: np.ndarray) -> float:
        """计算最大对角线长度"""
        max_length = 0
        n = recurrence_matrix.shape[0]

        for offset in range(1 - n, n):
            diagonal = np.diag(recurrence_matrix, k=offset)
            current_length = 0

            for val in diagonal:
                if val == 1:
                    current_length += 1
                    max_length = max(max_length, current_length)
                else:
                    current_length = 0

        return float(max_length)

    def _compute_recurrence_entropy(self, recurrence_matrix: np.ndarray) -> float:
        """计算递归熵"""
        diagonal_lengths = []
        n = recurrence_matrix.shape[0]

        for offset in range(1 - n, n):
            diagonal = np.diag(recurrence_matrix, k=offset)
            current_length = 0

            for val in diagonal:
                if val == 1:
                    current_length += 1
                else:
                    if current_length >= 2:
                        diagonal_lengths.append(current_length)
                    current_length = 0

            if current_length >= 2:
                diagonal_lengths.append(current_length)

        if not diagonal_lengths:
            return 0.0

        # 计算长度分布的熵
        unique_lengths, counts = np.unique(diagonal_lengths, return_counts=True)
        probabilities = counts / len(diagonal_lengths)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        return entropy

    def _compute_laminarity(self, recurrence_matrix: np.ndarray, min_length: int = 2) -> float:
        """计算层流性"""
        n = recurrence_matrix.shape[0]
        vertical_lengths = []

        # 计算垂直线长度
        for col in range(n):
            current_length = 0
            for row in range(n):
                if recurrence_matrix[row, col] == 1:
                    current_length += 1
                else:
                    if current_length >= min_length:
                        vertical_lengths.append(current_length)
                    current_length = 0

            if current_length >= min_length:
                vertical_lengths.append(current_length)

        if not vertical_lengths:
            return 0.0

        total_vertical_points = sum(vertical_lengths)
        total_recurrence_points = np.sum(recurrence_matrix)

        return total_vertical_points / max(total_recurrence_points, 1)

    def _extract_fractal_dimension(self, audio: np.ndarray) -> List[float]:
        """
        提取分形维数特征
        使用盒计数法和关联维数
        """
        try:
            # 相空间重构
            embedded = self._phase_space_reconstruction(audio)

            fractal_features = []

            # 关联维数 (Correlation Dimension)
            corr_dim = self._compute_correlation_dimension(embedded)
            fractal_features.append(corr_dim)

            # 盒计数维数 (Box-counting Dimension)
            box_dim = self._compute_box_counting_dimension(audio)
            fractal_features.append(box_dim)

            # Higuchi分形维数
            higuchi_dim = self._compute_higuchi_dimension(audio)
            fractal_features.append(higuchi_dim)

            # Katz分形维数
            katz_dim = self._compute_katz_dimension(audio)
            fractal_features.append(katz_dim)

            return fractal_features

        except Exception as e:
            print(f"分形维数计算错误: {e}")
            return [0.0] * 4

    def _compute_correlation_dimension(self, embedded: np.ndarray) -> float:
        """计算关联维数"""
        try:
            n_points = embedded.shape[0]

            if n_points < 100:
                return 1.0

            # 计算距离矩阵
            distances = pdist(embedded)

            # 设置半径范围
            r_min = np.percentile(distances, 1)
            r_max = np.percentile(distances, 50)

            if r_min >= r_max or r_min <= 0:
                return 1.0

            # 对数尺度的半径
            radii = np.logspace(np.log10(r_min), np.log10(r_max), 20)
            correlations = []

            for r in radii:
                # 计算关联积分
                correlation = np.mean(distances < r)
                correlations.append(correlation + 1e-10)  # 避免对数为负

            # 线性回归计算维数
            log_r = np.log10(radii)
            log_c = np.log10(correlations)

            # 去除无效值
            valid_mask = np.isfinite(log_r) & np.isfinite(log_c)
            if np.sum(valid_mask) < 5:
                return 1.0

            slope = np.polyfit(log_r[valid_mask], log_c[valid_mask], 1)[0]

            return max(0.1, min(slope, 10.0))  # 限制在合理范围内

        except Exception as e:
            print(f"关联维数计算错误: {e}")
            return 1.0

    def _compute_box_counting_dimension(self, signal: np.ndarray) -> float:
        """计算盒计数维数"""
        try:
            # 将信号映射到2D网格
            n = len(signal)
            if n < 100:
                return 1.0

            # 归一化信号
            signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10)

            # 不同的盒子大小
            box_sizes = [2 ** i for i in range(3, 8)]
            counts = []

            for box_size in box_sizes:
                # 创建网格
                grid_x = np.floor(np.arange(n) / (n / box_size)).astype(int)
                grid_y = np.floor(signal_norm * box_size).astype(int)

                # 计算非空盒子数
                boxes = set(zip(grid_x, grid_y))
                counts.append(len(boxes))

            if len(counts) < 3:
                return 1.0

            # 线性回归计算维数
            log_sizes = np.log10(box_sizes)
            log_counts = np.log10(counts)

            slope = -np.polyfit(log_sizes, log_counts, 1)[0]

            return max(0.1, min(slope, 3.0))

        except Exception as e:
            print(f"盒计数维数计算错误: {e}")
            return 1.0

    def _compute_higuchi_dimension(self, signal: np.ndarray, k_max: int = 10) -> float:
        """计算Higuchi分形维数"""
        try:
            n = len(signal)
            if n < 100:
                return 1.0

            lengths = []

            for k in range(1, k_max + 1):
                length_k = []

                for m in range(k):
                    indices = np.arange(m, n, k)
                    if len(indices) < 2:
                        continue

                    length_m = np.sum(np.abs(np.diff(signal[indices]))) * (n - 1) / (len(indices) - 1) / k
                    length_k.append(length_m)

                if length_k:
                    lengths.append(np.mean(length_k))
                else:
                    lengths.append(1.0)

            if len(lengths) < 3:
                return 1.0

            # 线性回归
            k_values = np.arange(1, len(lengths) + 1)
            log_k = np.log10(k_values)
            log_lengths = np.log10(np.array(lengths) + 1e-10)

            slope = -np.polyfit(log_k, log_lengths, 1)[0]

            return max(0.1, min(slope, 3.0))

        except Exception as e:
            print(f"Higuchi维数计算错误: {e}")
            return 1.0

    def _compute_katz_dimension(self, signal: np.ndarray) -> float:
        """计算Katz分形维数"""
        try:
            n = len(signal)
            if n < 2:
                return 1.0

            # 计算累积长度
            diffs = np.diff(signal)
            lengths = np.sqrt(1 + diffs ** 2)
            total_length = np.sum(lengths)

            # 计算最大距离
            max_distance = np.sqrt((n - 1) ** 2 + (signal[-1] - signal[0]) ** 2)

            if max_distance == 0 or total_length == 0:
                return 1.0

            # Katz维数公式
            katz_dim = np.log10(total_length / max_distance) / (
                        np.log10(total_length / max_distance) + np.log10(max_distance / total_length))

            return max(0.1, min(katz_dim, 3.0))

        except Exception as e:
            print(f"Katz维数计算错误: {e}")
            return 1.0

    def _extract_phase_space_features(self, audio: np.ndarray) -> List[float]:
        """
        提取相空间特征
        包括吸引子几何特征和轨迹统计量
        """
        try:
            # 相空间重构
            embedded = self._phase_space_reconstruction(audio)

            phase_features = []

            # 轨迹长度
            trajectory_length = self._compute_trajectory_length(embedded)
            phase_features.append(trajectory_length)

            # 轨迹曲率统计
            curvatures = self._compute_trajectory_curvature(embedded)
            phase_features.extend([
                np.mean(curvatures),
                np.std(curvatures),
                np.max(curvatures),
                np.min(curvatures)
            ])

            # 相空间体积
            volume = self._compute_phase_space_volume(embedded)
            phase_features.append(volume)

            # 重心和散布特征
            centroid = np.mean(embedded, axis=0)
            phase_features.extend(centroid.tolist())

            # 协方差矩阵的特征值
            cov_matrix = np.cov(embedded.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列
            phase_features.extend(eigenvalues.tolist())

            return phase_features

        except Exception as e:
            print(f"相空间特征计算错误: {e}")
            return [0.0] * (3 + 4 + 1 + self.embedding_dim + self.embedding_dim)

    def _compute_trajectory_length(self, embedded: np.ndarray) -> float:
        """计算轨迹总长度"""
        distances = np.linalg.norm(np.diff(embedded, axis=0), axis=1)
        return np.sum(distances)

    def _compute_trajectory_curvature(self, embedded: np.ndarray) -> np.ndarray:
        """计算轨迹曲率"""
        if embedded.shape[0] < 3:
            return np.array([0.0])

        # 计算一阶和二阶导数
        first_derivative = np.diff(embedded, axis=0)
        second_derivative = np.diff(first_derivative, axis=0)

        # 计算曲率
        curvatures = []
        for i in range(len(second_derivative)):
            v1 = first_derivative[i]
            v2 = second_derivative[i]

            cross_product = np.linalg.norm(np.cross(v1, v2)) if embedded.shape[1] == 2 else np.linalg.norm(v2)
            velocity_magnitude = np.linalg.norm(v1)

            if velocity_magnitude > 1e-8:
                curvature = cross_product / (velocity_magnitude ** 3)
            else:
                curvature = 0.0

            curvatures.append(curvature)

        return np.array(curvatures)

    def _compute_phase_space_volume(self, embedded: np.ndarray) -> float:
        """计算相空间占据体积"""
        try:
            # 使用凸包体积作为近似
            from scipy.spatial import ConvexHull

            if embedded.shape[0] < embedded.shape[1] + 1:
                return 0.0

            hull = ConvexHull(embedded)
            return hull.volume

        except Exception as e:
            # 使用简单的边界框体积作为fallback
            ranges = np.ptp(embedded, axis=0)
            return np.prod(ranges)


# 使用示例和测试
if __name__ == "__main__":
    # 创建测试音频信号
    duration = 5.0  # 5秒
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate))

    # 生成混沌信号（Lorenz系统的简化版本）
    test_signal = np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t) + 0.1 * np.random.randn(len(t))
    test_audio = torch.tensor(test_signal).float().unsqueeze(0).unsqueeze(0)  # [1, 1, samples]

    # 初始化特征提取器
    chaos_extractor = ChaosFeatureExtractor(
        sample_rate=sample_rate,
        embedding_dim=3,
        delay=10
    )

    print("开始混沌特征提取测试...")

    # 提取特征
    features = chaos_extractor(test_audio)

    # 打印特征统计
    print("\n提取的混沌特征:")
    for feature_name, feature_tensor in features.items():
        print(f"{feature_name}: 形状={feature_tensor.shape}, 均值={feature_tensor.mean().item():.4f}")

    print("\n混沌特征提取测试完成！")