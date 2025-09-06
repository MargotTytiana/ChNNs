import numpy as np
import librosa
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist
from pyts.image import RecurrencePlot
import hashlib
import os
import json
from tqdm import tqdm
import functools
import shutil

# 缓存目录
CACHE_DIR = "../chaos_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# 增强的NumPy类型转换函数
def convert_numpy_types(obj):
    """递归转换NumPy类型为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    return obj


def get_signal_hash(signal):
    if isinstance(signal, np.ndarray):
        signal_bytes = signal.tobytes()
    else:
        signal_bytes = str(signal).encode('utf-8')
    return hashlib.md5(signal_bytes).hexdigest()


def cache_to_file(feature_name=None):
    """健壮的缓存装饰器"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 定位signal参数
            if len(args) == 0:
                raise ValueError("被装饰函数必须至少有一个参数(signal)")
            signal = None
            for arg in args:
                if isinstance(arg, np.ndarray):
                    signal = arg
                    break
            if signal is None:
                raise ValueError("无法定位np.ndarray类型的signal参数")

            hash_key = get_signal_hash(signal)
            params_str = ""
            if len(args) > 1:
                params_str += "_" + "_".join(map(str, args[1:]))
            if kwargs:
                params_str += "_" + "_".join(f"{k}={v}" for k, v in kwargs.items())

            # 创建安全的文件名
            fname = f"{hash_key}_{feature_name or func.__name__}{params_str}.json"
            fname = "".join(c for c in fname if c.isalnum() or c in "._-")
            fpath = os.path.join(CACHE_DIR, fname)

            # 尝试加载缓存（健壮版）
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'r') as f:
                        cached_result = json.load(f)
                    print(f"[Cache] 从缓存加载 {func.__name__} 结果")
                    return cached_result
                except (json.JSONDecodeError, IOError) as e:
                    print(f"[Cache] 缓存文件损坏 ({e})，删除并重新计算")
                    os.remove(fpath)

            # 计算并缓存结果
            result = func(*args, **kwargs)

            # 转换并保存结果
            json_result = convert_numpy_types(result)

            try:
                with open(fpath, 'w') as f:
                    json.dump(json_result, f)
                print(f"[Cache] 计算并缓存 {func.__name__} 结果")
            except IOError as e:
                print(f"[Cache] 无法写入缓存: {e}")

            return result

        return wrapper

    return decorator


# 清除所有缓存文件的函数
def clear_all_cache():
    """清除整个缓存目录"""
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"已清除缓存目录: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)


class Config:
    EMBEDDING_DIM = 3
    TIME_DELAY = 10
    TOLERANCE = 0.1
    MIN_WINDOW_LENGTH = 100
    LYAPUNOV_STEPS = 100
    LYAPUNOV_TAU = 1


class ChaosFeatureExtractor:
    @staticmethod
    def phase_space_reconstruction(signal, dim=Config.EMBEDDING_DIM, tau=Config.TIME_DELAY):
        if len(signal) < dim * tau:
            raise ValueError(f"信号长度({len(signal)})过短，无法进行相空间重构(dim={dim}, tau={tau})")

        n = len(signal) - (dim - 1) * tau
        reconstructed = np.zeros((n, dim))

        for i in range(dim):
            reconstructed[:, i] = signal[i * tau: i * tau + n]

        return reconstructed

    @staticmethod
    @cache_to_file("mle")
    def calculate_max_lyapunov_exponent(signal, tau=Config.LYAPUNOV_TAU, steps=Config.LYAPUNOV_STEPS):
        print("[MLE] 开始相空间重构...")
        reconstructed = ChaosFeatureExtractor.phase_space_reconstruction(signal)
        n, dim = reconstructed.shape
        print(f"[MLE] 重构后形状: {reconstructed.shape}")

        distances = []
        print("[MLE] 开始搜索最近邻并计算轨迹距离增长...")
        for i in tqdm(range(n - steps), desc="MLE计算"):
            min_dist = float('inf')
            nearest_idx = -1

            # 搜索最近邻点（排除相邻点）
            for j in range(n):
                if abs(i - j) > 10:  # 确保不是相邻点
                    dist = np.linalg.norm(reconstructed[i] - reconstructed[j])
                    if dist < min_dist and dist > 0:
                        min_dist = dist
                        nearest_idx = j

            if nearest_idx != -1:
                for k in range(steps):
                    if i + k < n and nearest_idx + k < n:
                        new_dist = np.linalg.norm(reconstructed[i + k] - reconstructed[nearest_idx + k])
                        if new_dist > 0:
                            distances.append(np.log(new_dist / min_dist))

        if len(distances) > 0:
            mle = np.mean(distances) / tau
            print(f"[MLE] 计算完成，指数: {mle:.6f}")
            return float(mle)  # 确保返回Python原生float
        else:
            print("[MLE] 无有效点对，返回0")
            return 0.0

    @staticmethod
    @cache_to_file("rqa")
    def recurrence_quantification_analysis(signal, threshold=Config.TOLERANCE):
        print("[RQA] 开始相空间重构...")
        reconstructed = ChaosFeatureExtractor.phase_space_reconstruction(signal)
        print(f"[RQA] 重构形状: {reconstructed.shape}")

        print("[RQA] 计算递归图...")
        reduced = reconstructed[::10, 0:1].T
        rp = RecurrencePlot(threshold=threshold)
        recurrence_matrix = rp.fit_transform(reduced)[0]
        print("[RQA] 递归图计算完成")

        n = recurrence_matrix.shape[0]
        diagonal_lines = []
        vertical_lines = []

        # 提取对角线特征
        for i in range(-n + 1, n):
            diag = np.diag(recurrence_matrix, k=i)
            if len(diag) > 2:
                line_lengths = np.diff(np.where(np.concatenate(([False], diag > 0, [False])))[0])[::2]
                diagonal_lines.extend(line_lengths)

        # 提取垂直线特征
        for j in range(n):
            col = recurrence_matrix[:, j]
            if len(col) > 2:
                line_lengths = np.diff(np.where(np.concatenate(([False], col > 0, [False])))[0])[::2]
                vertical_lines.extend(line_lengths)

        # 计算RQA指标
        recurrence_rate = np.sum(recurrence_matrix) / (n * n)

        if len(diagonal_lines) > 0:
            determinism = np.sum([l for l in diagonal_lines if l >= 2]) / np.sum(diagonal_lines)
            avg_diag_length = np.mean([l for l in diagonal_lines if l >= 2])
        else:
            determinism = 0.0
            avg_diag_length = 0.0

        if len(vertical_lines) > 0:
            laminarity = np.sum([l for l in vertical_lines if l >= 2]) / np.sum(vertical_lines)
            avg_vert_length = np.mean([l for l in vertical_lines if l >= 2])
        else:
            laminarity = 0.0
            avg_vert_length = 0.0

        print("[RQA] RQA 特征提取完成")
        return {
            'recurrence_rate': float(recurrence_rate),
            'determinism': float(determinism),
            'avg_diag_length': float(avg_diag_length),
            'laminarity': float(laminarity),
            'avg_vert_length': float(avg_vert_length)
        }

    @staticmethod
    @cache_to_file("corr_dim")
    def calculate_correlation_dimension(signal, min_r=0.01, max_r=1.0, num_r=10):
        reconstructed = ChaosFeatureExtractor.phase_space_reconstruction(signal)
        distances = pdist(reconstructed, 'euclidean')
        distances = distances[distances > 0]

        r_values = np.logspace(np.log10(min_r), np.log10(max_r), num_r)
        C = np.zeros_like(r_values)

        for i, r in enumerate(r_values):
            C[i] = np.sum(distances < r) / len(distances)

        valid_indices = np.where((C > 0) & (C < 1))[0]
        if len(valid_indices) < 2:
            return 0.0

        log_r = np.log(r_values[valid_indices])
        log_C = np.log(C[valid_indices])

        slope, _ = np.polyfit(log_r, log_C, 1)
        return float(slope)


class MLSAAnalyzer:
    def __init__(self, scales=[1, 2, 4, 8]):
        self.scales = scales

    @cache_to_file("mlsa")
    def analyze(self, signal):
        mle_features = []
        print("[MLSA] 开始多尺度分析")

        for scale in self.scales:
            print(f"[MLSA] 尺度: {scale}")
            if scale > 1:
                scaled_signal = signal[::scale]
            else:
                scaled_signal = signal

            mle = ChaosFeatureExtractor.calculate_max_lyapunov_exponent(scaled_signal)
            mle_features.append(mle)

        print("[MLSA] 完成")
        return mle_features  # 返回Python列表


class ChaoticFeatureExtractor(nn.Module):
    def __init__(self, feature_type='mle', sampling_rate=16000):
        super().__init__()
        self.feature_type = feature_type
        self.sampling_rate = sampling_rate

        if feature_type == 'mlsa':
            self.mlsa_analyzer = MLSAAnalyzer()

    def forward(self, x):
        batch_size = x.shape[0]
        features = []

        for i in range(batch_size):
            signal = x[i].detach().cpu().numpy()

            if self.feature_type == 'mle':
                mle = ChaosFeatureExtractor.calculate_max_lyapunov_exponent(signal)
                feature = torch.tensor([mle], dtype=torch.float32)

            elif self.feature_type == 'rqa':
                rqa_features = ChaosFeatureExtractor.recurrence_quantification_analysis(signal)
                feature = torch.tensor([
                    rqa_features['recurrence_rate'],
                    rqa_features['determinism'],
                    rqa_features['avg_diag_length'],
                    rqa_features['laminarity'],
                    rqa_features['avg_vert_length']
                ], dtype=torch.float32)

            elif self.feature_type == 'mlsa':
                mlsa_features = self.mlsa_analyzer.analyze(signal)
                feature = torch.tensor(mlsa_features, dtype=torch.float32)

            else:
                raise ValueError(f"不支持的特征类型: {self.feature_type}")

            features.append(feature)

        return torch.stack(features).to(x.device)


class TraditionalFeatureExtractor:
    @staticmethod
    def extract_mfcc(signal, sr=16000, n_mfcc=20):
        return librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

    @staticmethod
    def extract_fbank(signal, sr=16000, n_mels=40):
        return librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)

    @staticmethod
    def extract_plp(signal, sr=16000, order=12):
        # 简化用MFCC替代
        return librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=order)


if __name__ == "__main__":
    # 清除所有缓存确保干净测试
    clear_all_cache()

    sample_rate = 16000
    duration = 1  # 减少测试时间
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    test_signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)

    print(f"测试信号长度: {len(test_signal)}")

    # 相空间重构测试
    try:
        reconstructed = ChaosFeatureExtractor.phase_space_reconstruction(test_signal)
        print(f"相空间重构结果形状: {reconstructed.shape}")
    except ValueError as e:
        print(f"相空间重构错误: {e}")

    # 最大李雅普诺夫指数测试
    try:
        mle = ChaosFeatureExtractor.calculate_max_lyapunov_exponent(test_signal)
        print(f"最大李雅普诺夫指数: {mle:.6f} (类型: {type(mle).__name__})")
    except Exception as e:
        print(f"计算MLE错误: {e}")

    # 递归量化分析测试
    try:
        rqa_features = ChaosFeatureExtractor.recurrence_quantification_analysis(test_signal)
        print("递归量化分析结果:")
        for key, value in rqa_features.items():
            print(f"  {key}: {value:.6f} (类型: {type(value).__name__})")
    except Exception as e:
        print(f"计算RQA错误: {e}")

    # 多尺度李雅普诺夫谱分析测试
    try:
        mlsa_analyzer = MLSAAnalyzer()
        mlsa_features = mlsa_analyzer.analyze(test_signal)
        print(f"多尺度李雅普诺夫谱分析: {mlsa_features} (类型: {type(mlsa_features).__name__})")
    except Exception as e:
        print(f"计算MLSA错误: {e}")

    # PyTorch特征提取模块测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 测试不同特征类型
    for feature_type in ['mle', 'rqa', 'mlsa']:
        try:
            print(f"\n测试特征类型: {feature_type}")
            extractor = ChaoticFeatureExtractor(feature_type=feature_type).to(device)
            test_tensor = torch.tensor(test_signal, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                features = extractor(test_tensor)
            print(f"提取的特征: {features}")
            print(f"特征形状: {features.shape}, 数据类型: {features.dtype}")
        except Exception as e:
            print(f"提取 {feature_type} 特征错误: {e}")