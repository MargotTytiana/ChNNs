"""增强版特征提取模块 - 专为说话人识别优化"""
import os
import numpy as np
import librosa
import pandas as pd
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import soundfile as sf


# 配置日志
def setup_logging(log_file='feature_extraction.log'):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=512, hop_length=160,
                 delta_order=2, feature_dir="features", max_duration=10.0,
                 feature_type='mfcc', apply_cmvn=True):
        """
        初始化特征提取器 - 专为说话人识别优化

        参数:
        sample_rate: 音频采样率
        n_mfcc: 提取的MFCC特征数量
        n_fft: FFT窗口大小
        hop_length: 帧移（样本数）
        delta_order: 差分阶数（0-2）
        feature_dir: 特征保存目录
        max_duration: 最大音频时长（秒），超过此长度的音频将被裁剪
        feature_type: 特征类型 ('mfcc', 'fbank', 'spectrogram')
        apply_cmvn: 是否应用倒谱均值方差归一化
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.delta_order = delta_order
        self.feature_dir = feature_dir
        self.max_duration = max_duration
        self.feature_type = feature_type
        self.apply_cmvn = apply_cmvn
        self.frame_rate = sample_rate / hop_length

        # 确保特征目录存在
        os.makedirs(feature_dir, exist_ok=True)

        logging.info(f"特征提取器初始化: 类型={feature_type}, MFCC={n_mfcc}, FFT={n_fft}, 帧移={hop_length}")
        logging.info(f"最大音频时长: {max_duration}秒, 差分阶数: {delta_order}, CMVN: {apply_cmvn}")

    def extract_features(self, audio_path=None, audio_data=None):
        """
        从单个音频文件或音频数据提取特征 - 支持x-vector模型

        返回:
        features: 形状为 (n_features, n_frames) 的NumPy数组
        """
        try:
            # 加载音频（确保采样率匹配）
            if audio_data is not None:
                y = audio_data
                if len(y) > self.max_duration * self.sample_rate:
                    y = y[:int(self.max_duration * self.sample_rate)]
            else:
                y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                # 裁剪过长的音频
                if len(y) > self.max_duration * self.sample_rate:
                    y = y[:int(self.max_duration * self.sample_rate)]

            # 预加重
            y = librosa.effects.preemphasis(y)

            # 提取特征
            if self.feature_type == 'mfcc':
                # 提取MFCC特征
                mfcc = librosa.feature.mfcc(
                    y=y,
                    sr=self.sample_rate,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=40
                )

                # 应用倒谱均值方差归一化 (CMVN)
                if self.apply_cmvn:
                    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-5)

                features = mfcc

            elif self.feature_type == 'fbank':
                # 提取滤波器组能量特征
                mel_spec = librosa.feature.melspectrogram(
                    y=y,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=40
                )
                fbank = librosa.power_to_db(mel_spec, ref=np.max)
                features = fbank

            elif self.feature_type == 'spectrogram':
                # 提取频谱图特征
                stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
                spectrogram = librosa.amplitude_to_db(np.abs(stft))
                features = spectrogram

            # 计算动态特征（差分）
            feature_list = [features]

            if self.delta_order >= 1:
                delta = librosa.feature.delta(features, order=1)
                feature_list.append(delta)

            if self.delta_order >= 2:
                delta2 = librosa.feature.delta(features, order=2)
                feature_list.append(delta2)

            # 合并所有特征
            features = np.vstack(feature_list)

            return features.T  # 转置为 (时间帧, 特征) 适合x-vector输入

        except Exception as e:
            if audio_path:
                logging.error(f"提取特征失败: {audio_path} - {str(e)}")
            else:
                logging.error(f"提取特征失败: {str(e)}")
            return None

    def process_dataset(self, metadata_path, output_csv="feature_metadata.csv",
                        audio_dir=None, use_existing=True):
        """
        处理整个数据集的特征提取 - 支持直接从元数据加载音频

        参数:
        metadata_path: 元数据CSV文件路径
        output_csv: 输出特征元数据文件名
        audio_dir: 音频文件根目录（如果元数据中的路径是相对路径）
        use_existing: 是否使用已存在的特征文件
        """
        # 读取元数据
        metadata = pd.read_csv(metadata_path)
        logging.info(f"读取元数据: 共 {len(metadata)} 条记录")

        # 准备特征元数据
        feature_metadata = []
        skipped_files = []
        processed_count = 0
        skipped_count = 0

        # 进度条
        with tqdm(total=len(metadata), desc="提取特征") as pbar:
            for idx, row in metadata.iterrows():
                audio_path = row['file_path']

                # 处理音频路径
                if audio_dir and not os.path.isabs(audio_path):
                    audio_path = os.path.join(audio_dir, audio_path)

                # 生成特征保存路径
                rel_path = os.path.relpath(audio_path, start=os.path.dirname(metadata_path))
                feature_path = os.path.join(
                    self.feature_dir,
                    os.path.splitext(rel_path)[0] + '.npy'
                )
                os.makedirs(os.path.dirname(feature_path), exist_ok=True)

                # 检查特征文件是否已存在
                if use_existing and os.path.exists(feature_path):
                    features = np.load(feature_path)
                    processed_count += 1
                else:
                    # 提取特征
                    features = self.extract_features(audio_path)

                    if features is not None:
                        # 保存特征
                        np.save(feature_path, features)
                        processed_count += 1
                    else:
                        skipped_files.append(audio_path)
                        skipped_count += 1
                        pbar.update(1)
                        continue

                # 更新元数据
                new_row = row.copy()
                new_row['feature_path'] = feature_path
                new_row['n_frames'] = features.shape[0]
                new_row['duration'] = features.shape[0] / self.frame_rate
                feature_metadata.append(new_row)

                pbar.update(1)

        # 创建特征元数据DataFrame
        feature_df = pd.DataFrame(feature_metadata)

        # 保存特征元数据
        feature_df.to_csv(os.path.join(self.feature_dir, output_csv), index=False)
        logging.info(f"特征元数据已保存: 共 {len(feature_df)} 条记录, 处理: {processed_count}, 跳过: {skipped_count}")

        if skipped_files:
            logging.warning(f"跳过 {len(skipped_files)} 个文件，详情见日志")
            with open(os.path.join(self.feature_dir, "skipped_files.txt"), 'w') as f:
                f.write("\n".join(skipped_files))

        return feature_df

    def compute_global_stats(self, feature_metadata, stats_file="global_stats.joblib"):
        """
        计算全局统计信息（均值、方差）用于在线归一化
        """
        if isinstance(feature_metadata, str):
            feature_metadata = pd.read_csv(feature_metadata)

        # 初始化统计变量
        total_frames = 0
        sum_features = None
        sum_sq_features = None

        # 遍历所有特征文件
        for feature_path in tqdm(feature_metadata['feature_path'], desc="计算全局统计信息"):
            features = np.load(feature_path)

            if sum_features is None:
                sum_features = np.zeros(features.shape[1])
                sum_sq_features = np.zeros(features.shape[1])

            total_frames += features.shape[0]
            sum_features += np.sum(features, axis=0)
            sum_sq_features += np.sum(features ** 2, axis=0)

        # 计算均值和方差
        mean = sum_features / total_frames
        variance = (sum_sq_features / total_frames) - (mean ** 2)
        std = np.sqrt(variance)

        # 保存统计信息
        stats = {
            'mean': mean,
            'std': std,
            'total_frames': total_frames
        }

        stats_path = os.path.join(self.feature_dir, stats_file)
        joblib.dump(stats, stats_path)
        logging.info(f"全局统计信息已保存: {stats_path}")
        logging.info(f"特征均值: {mean[:5]}... 标准差: {std[:5]}...")

        return stats

    def normalize_features(self, features, stats=None):
        """
        使用全局统计信息归一化特征
        """
        if stats is None:
            stats_path = os.path.join(self.feature_dir, "global_stats.joblib")
            if os.path.exists(stats_path):
                stats = joblib.load(stats_path)
            else:
                logging.warning("未找到全局统计信息，跳过归一化")
                return features

        return (features - stats['mean']) / (stats['std'] + 1e-5)

    def extract_features_from_audio(self, audio_data, sample_rate=None):
        """
        直接从音频数据提取特征 - 用于实时处理
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        # 重采样如果需要
        if sample_rate != self.sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)

        return self.extract_features(audio_data=audio_data)

    def save_audio_with_features(self, audio_path, output_dir, speaker_id=None):
        """
        保存音频及其特征可视化 - 用于调试
        """
        os.makedirs(output_dir, exist_ok=True)

        # 加载音频
        y, sr = librosa.load(audio_path, sr=self.sample_rate)

        # 提取特征
        features = self.extract_features(audio_path)
        if features is None:
            return

        # 保存特征
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        feature_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(feature_path, features)

        # 可视化特征
        plt.figure(figsize=(15, 8))

        # 绘制波形
        plt.subplot(211)
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"Audio Waveform - {base_name}")

        # 绘制特征
        plt.subplot(212)
        librosa.display.specshow(features.T, sr=sr, hop_length=self.hop_length, x_axis='time')
        plt.colorbar()
        plt.title(f"{self.feature_type.upper()} Features")

        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}.png"))
        plt.close()

        logging.info(f"音频和特征已保存到: {output_dir}/{base_name}.*")

    def create_feature_dataset(self, metadata_path, output_file="feature_dataset.pt",
                               max_length=1000, audio_dir=None):
        """
        创建可直接加载的特征数据集 - 用于PyTorch训练
        """
        metadata = pd.read_csv(metadata_path)
        features_list = []
        labels_list = []
        speaker_ids = []

        # 创建说话人ID映射
        unique_speakers = metadata['speaker_id'].unique()
        speaker_to_idx = {sp: idx for idx, sp in enumerate(unique_speakers)}

        # 加载全局统计信息
        stats_path = os.path.join(self.feature_dir, "global_stats.joblib")
        stats = joblib.load(stats_path) if os.path.exists(stats_path) else None

        # 处理所有样本
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="创建特征数据集"):
            feature_path = row['feature_path']

            # 处理路径
            if audio_dir and not os.path.isabs(feature_path):
                feature_path = os.path.join(audio_dir, feature_path)

            if not os.path.exists(feature_path):
                continue

            # 加载特征
            features = np.load(feature_path)

            # 归一化
            if stats:
                features = self.normalize_features(features, stats)

            # 处理序列长度
            if features.shape[0] > max_length:
                start = np.random.randint(0, features.shape[0] - max_length)
                features = features[start:start + max_length]
            elif features.shape[0] < max_length:
                padding = np.zeros((max_length - features.shape[0], features.shape[1]))
                features = np.vstack((features, padding))

            # 收集数据
            features_list.append(features)
            labels_list.append(speaker_to_idx[row['speaker_id']])
            speaker_ids.append(row['speaker_id'])

        # 转换为张量
        features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        # 创建数据集
        dataset = {
            'features': features_tensor,
            'labels': labels_tensor,
            'speaker_ids': speaker_ids,
            'speaker_mapping': speaker_to_idx
        }

        # 保存数据集
        torch.save(dataset, os.path.join(self.feature_dir, output_file))
        logging.info(f"特征数据集已保存: {output_file}, 形状: {features_tensor.shape}")

        return dataset


# 使用示例
if __name__ == "__main__":
    # 初始化日志
    setup_logging()

    # 配置路径
    METADATA_PATH = "P:/PycharmProjects/pythonProject1/processed_data/metadata_test.csv"  # 替换为实际路径
    FEATURE_DIR = "extracted_features"
    AUDIO_DIR = "dataset/LibriSpeech/train-clean-100"  # 音频文件根目录

    # 创建特征提取器 (使用滤波器组特征)
    extractor = FeatureExtractor(
        n_mfcc=40,  # 使用更多MFCC系数
        hop_length=160,  # 10ms帧移
        n_fft=512,  # 32ms窗口
        delta_order=2,
        feature_dir=FEATURE_DIR,
        max_duration=15.0,  # 延长最大时长
        feature_type='fbank',  # 使用滤波器组特征
        apply_cmvn=True  # 应用CMVN
    )

    # 处理数据集并提取特征
    feature_metadata = extractor.process_dataset(
        METADATA_PATH,
        audio_dir=AUDIO_DIR,
        use_existing=False  # 强制重新提取
    )

    # 计算全局统计信息
    global_stats = extractor.compute_global_stats(feature_metadata)

    # 创建可直接加载的特征数据集
    feature_dataset = extractor.create_feature_dataset(
        os.path.join(FEATURE_DIR, "feature_metadata.csv"),
        audio_dir=AUDIO_DIR
    )

    # 可视化示例音频
    sample_audio = feature_metadata.iloc[0]['file_path']
    extractor.save_audio_with_features(
        sample_audio,
        os.path.join(FEATURE_DIR, "samples"),
        speaker_id=feature_metadata.iloc[0]['speaker_id']
    )

    logging.info("特征提取流程完成！")
