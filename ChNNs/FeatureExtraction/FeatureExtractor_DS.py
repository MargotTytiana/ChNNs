"""
特征提取模块 - 从音频中提取MFCC及相关特征
"""
import os
import numpy as np
import librosa
import pandas as pd
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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
                 delta_order=2, feature_dir="features", max_duration=10.0):
        """
        初始化特征提取器

        参数:
        sample_rate: 音频采样率
        n_mfcc: 提取的MFCC特征数量
        n_fft: FFT窗口大小
        hop_length: 帧移（样本数）
        delta_order: 差分阶数（0-2）
        feature_dir: 特征保存目录
        max_duration: 最大音频时长（秒），超过此长度的音频将被裁剪
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.delta_order = delta_order
        self.feature_dir = feature_dir
        self.max_duration = max_duration

        # 确保特征目录存在
        os.makedirs(feature_dir, exist_ok=True)

        # 计算帧率（帧/秒）
        self.frame_rate = sample_rate / hop_length

        logging.info(f"特征提取器初始化: MFCC={n_mfcc}, FFT={n_fft}, 帧移={hop_length}")
        logging.info(f"最大音频时长: {max_duration}秒, 差分阶数: {delta_order}")

    def extract_features(self, audio_path):
        """
        从单个音频文件提取特征

        返回:
        features: 形状为 (n_features, n_frames) 的NumPy数组
        """
        try:
            # 加载音频（确保采样率匹配）
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # 裁剪过长的音频
            if len(y) > self.max_duration * self.sample_rate:
                y = y[:int(self.max_duration * self.sample_rate)]
                logging.debug(f"裁剪音频: {audio_path}")

            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=26
            )

            # 计算动态特征（差分）
            features = [mfcc]

            if self.delta_order >= 1:
                delta = librosa.feature.delta(mfcc, order=1)
                features.append(delta)

            if self.delta_order >= 2:
                delta2 = librosa.feature.delta(mfcc, order=2)
                features.append(delta2)

            # 合并所有特征
            features = np.vstack(features)

            return features

        except Exception as e:
            logging.error(f"提取特征失败: {audio_path} - {str(e)}")
            return None

    def process_dataset(self, metadata_path, output_csv="feature_metadata.csv"):
        """
        处理整个数据集的特征提取

        参数:
        metadata_path: 元数据CSV文件路径
        output_csv: 输出特征元数据文件名
        """
        # 读取元数据
        metadata = pd.read_csv(metadata_path)
        logging.info(f"读取元数据: 共 {len(metadata)} 条记录")

        # 准备特征元数据
        feature_metadata = []
        skipped_files = []

        # 进度条
        with tqdm(total=len(metadata), desc="提取特征") as pbar:
            for idx, row in metadata.iterrows():
                audio_path = row['file_path']

                # 提取特征
                features = self.extract_features(audio_path)

                if features is not None:
                    # 生成特征保存路径
                    rel_path = os.path.relpath(audio_path, start=os.path.dirname(metadata_path))
                    feature_path = os.path.join(
                        self.feature_dir,
                        os.path.splitext(rel_path)[0] + '.npy'
                    )
                    os.makedirs(os.path.dirname(feature_path), exist_ok=True)

                    # 保存特征
                    np.save(feature_path, features)

                    # 更新元数据
                    new_row = row.copy()
                    new_row['feature_path'] = feature_path
                    new_row['n_frames'] = features.shape[1]
                    new_row['duration'] = features.shape[1] / self.frame_rate
                    feature_metadata.append(new_row)
                else:
                    skipped_files.append(audio_path)

                pbar.update(1)

        # 创建特征元数据DataFrame
        feature_df = pd.DataFrame(feature_metadata)

        # 保存特征元数据
        feature_df.to_csv(os.path.join(self.feature_dir, output_csv), index=False)
        logging.info(f"特征元数据已保存: 共 {len(feature_df)} 条记录")

        if skipped_files:
            logging.warning(f"跳过 {len(skipped_files)} 个文件，详情见日志")
            with open(os.path.join(self.feature_dir, "skipped_files.txt"), 'w') as f:
                f.write("\n".join(skipped_files))

        return feature_df

    def normalize_features(self, feature_metadata, output_csv="normalized_feature_metadata.csv"):
        """
        特征归一化（使用整个数据集的统计信息）
        使用部分拟合(partial_fit)避免内存溢出

        参数:
        feature_metadata: 特征元数据DataFrame或文件路径
        output_csv: 输出归一化特征元数据文件名
        """
        if isinstance(feature_metadata, str):
            feature_metadata = pd.read_csv(feature_metadata)

        # 使用部分拟合计算统计信息，避免内存溢出
        scaler = StandardScaler()
        logging.info("使用部分拟合计算归一化统计信息...")

        # 分批次处理特征数据
        for feature_path in tqdm(feature_metadata['feature_path'], desc="计算归一化参数"):
            features = np.load(feature_path)
            scaler.partial_fit(features.T)  # 转置为 (时间帧, 特征)

        # 保存归一化器
        import joblib
        scaler_path = os.path.join(self.feature_dir, "feature_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logging.info(f"特征归一化器已保存: {scaler_path}")

        # 应用归一化并保存新特征
        normalized_metadata = []

        for idx, row in tqdm(feature_metadata.iterrows(), total=len(feature_metadata), desc="归一化特征"):
            features = np.load(row['feature_path'])

            # 归一化 (转置->归一化->转置回原始形状)
            normalized = scaler.transform(features.T).T

            # 保存归一化特征
            base_path = os.path.splitext(row['feature_path'])[0]
            normalized_path = base_path + '_normalized.npy'
            np.save(normalized_path, normalized)

            # 更新元数据
            new_row = row.copy()
            new_row['feature_path'] = normalized_path
            normalized_metadata.append(new_row)

        # 创建归一化特征元数据
        normalized_df = pd.DataFrame(normalized_metadata)
        normalized_df.to_csv(os.path.join(self.feature_dir, output_csv), index=False)
        logging.info(f"归一化特征元数据已保存: {output_csv}")

        return normalized_df

    def visualize_features(self, feature_path, speaker_id, save_dir="feature_plots"):
        """
        可视化特征 - 简化字体处理

        参数:
        feature_path: 特征文件路径
        speaker_id: 说话人ID（用于标题）
        save_dir: 保存图像的目录
        """
        # 简化字体处理
        plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用通用字体
        plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

        os.makedirs(save_dir, exist_ok=True)

        # 加载特征
        features = np.load(feature_path)

        # 创建图像
        plt.figure(figsize=(15, 10))

        # MFCC特征
        plt.subplot(3, 1, 1)
        librosa.display.specshow(
            features[:self.n_mfcc],
            x_axis='time',
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'MFCC特征 - 说话人 {speaker_id}')

        # Delta特征
        if self.delta_order >= 1:
            plt.subplot(3, 1, 2)
            librosa.display.specshow(
                features[self.n_mfcc:2 * self.n_mfcc],
                x_axis='time',
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Delta特征 - 说话人 {speaker_id}')

        # Delta-Delta特征
        if self.delta_order >= 2:
            plt.subplot(3, 1, 3)
            librosa.display.specshow(
                features[2 * self.n_mfcc:],
                x_axis='time',
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Delta-Delta特征 - 说话人 {speaker_id}')

        # 保存图像
        filename = os.path.basename(feature_path).replace('.npy', '.png')
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"特征可视化已保存: {save_path}")


# 使用示例
if __name__ == "__main__":
    # 初始化日志
    setup_logging()

    # 配置路径
    METADATA_PATH = "P:/PycharmProjects/pythonProject1/processed_data/metadata_test.csv"  # 替换为实际路径
    FEATURE_DIR = "extracted_features"

    # 创建特征提取器
    extractor = FeatureExtractor(
        n_mfcc=13,
        hop_length=160,  # 10ms帧移 (16000Hz * 0.01s = 160)
        n_fft=512,  # 32ms窗口 (16000Hz * 0.032s ≈ 512)
        delta_order=2,
        feature_dir=FEATURE_DIR,
        max_duration=10.0  # 裁剪超过10秒的音频
    )

    # 处理数据集
    feature_metadata = extractor.process_dataset(METADATA_PATH)

    # 特征归一化
    normalized_metadata = extractor.normalize_features(feature_metadata)

    # 可视化示例特征
    sample_row = normalized_metadata.iloc[0]
    extractor.visualize_features(
        sample_row['feature_path'],
        sample_row['speaker_id'],
        save_dir=os.path.join(FEATURE_DIR, "visualizations")
    )

    logging.info("特征提取完成！")
