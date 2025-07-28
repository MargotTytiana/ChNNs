"""增强版LibriSpeech数据加载器 - 专为x-vector说话人识别优化"""
import os
import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import joblib
from FeatureExtractor import FeatureExtractor  # 使用重写的特征提取器


# 配置日志
def setup_logging(log_file='data_loader.log'):
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


class FeatureAugmenter:
    """特征层面的数据增强 - 更适合x-vector模型"""

    def __init__(self, augment_prob=0.5):
        self.augment_prob = augment_prob
        logging.info(f"初始化特征增强器，增强概率: {augment_prob}")

    def __call__(self, features):
        if random.random() > self.augment_prob:
            return features

        # 应用多种增强技术
        augmented = features.copy()

        # 1. 时间掩蔽 (时间轴上随机遮挡)
        if random.random() < 0.3:
            t_len = augmented.shape[0]
            mask_width = random.randint(1, min(20, t_len // 5))
            mask_start = random.randint(0, t_len - mask_width)
            augmented[mask_start:mask_start + mask_width] = 0

        # 2. 特征掩蔽 (特征维度上随机遮挡)
        if random.random() < 0.3:
            f_len = augmented.shape[1]
            mask_width = random.randint(1, min(5, f_len // 3))
            mask_start = random.randint(0, f_len - mask_width)
            augmented[:, mask_start:mask_start + mask_width] = 0

        # 3. 添加高斯噪声
        if random.random() < 0.4:
            noise_level = random.uniform(0.01, 0.1)
            noise = np.random.normal(0, noise_level, augmented.shape)
            augmented += noise

        # 4. 速度扰动 (通过时间轴插值模拟)
        if random.random() < 0.2:
            speed_factor = random.choice([0.9, 1.1])
            orig_len = augmented.shape[0]
            new_len = int(orig_len * speed_factor)
            x_orig = np.arange(orig_len)
            x_new = np.linspace(0, orig_len - 1, new_len)
            augmented = np.array([np.interp(x_new, x_orig, augmented[:, i]) for i in range(augmented.shape[1])]).T

        return augmented


class LibriSpeechDataset(Dataset):
    def __init__(self, metadata, feature_dir=None, feature_stats=None,
                 max_frames=800, augment=False, use_raw_audio=False,
                 feature_extractor_params=None):
        """
        专为x-vector优化的数据集加载器

        参数:
        metadata: 元数据DataFrame或文件路径
        feature_dir: 特征文件根目录
        feature_stats: 全局特征统计信息
        max_frames: 最大帧数（截断或填充）
        augment: 是否启用特征增强
        use_raw_audio: 是否使用原始音频（实时提取特征）
        feature_extractor_params: 特征提取器参数（当use_raw_audio=True时使用）
        """
        if isinstance(metadata, str):
            self.metadata = pd.read_csv(metadata)
        else:
            self.metadata = metadata

        self.feature_dir = feature_dir
        self.max_frames = max_frames
        self.augment = augment
        self.use_raw_audio = use_raw_audio
        self.feature_stats = feature_stats

        # 初始化特征提取器（如果需要）
        if use_raw_audio:
            if feature_extractor_params is None:
                feature_extractor_params = {
                    'sample_rate': 16000,
                    'n_mfcc': 40,
                    'hop_length': 160,
                    'delta_order': 2
                }
            self.feature_extractor = FeatureExtractor(**feature_extractor_params)
            logging.info("启用实时特征提取")
        else:
            self.feature_extractor = None
            logging.info("使用预提取特征")

        # 初始化特征增强器
        self.augmenter = FeatureAugmenter() if augment else None

        # 构建说话人ID映射
        self.speaker_ids = self.metadata['speaker_id'].unique()
        self.speaker_to_idx = {sp: idx for idx, sp in enumerate(self.speaker_ids)}
        self.num_speakers = len(self.speaker_to_idx)

        # 添加标签列
        self.metadata['label'] = self.metadata['speaker_id'].map(self.speaker_to_idx)

        logging.info(f"数据集初始化: {len(self)} 个样本, {self.num_speakers} 个说话人")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 加载特征
        if self.use_raw_audio:
            # 实时提取特征
            try:
                features = self.feature_extractor.extract_features(row['file_path'])
                if features is None:
                    raise RuntimeError(f"特征提取失败: {row['file_path']}")
            except Exception as e:
                logging.error(f"加载音频失败: {row['file_path']} - {str(e)}")
                # 创建空特征作为回退
                features = np.zeros((self.max_frames, 40))
        else:
            # 加载预提取特征
            feature_path = row['feature_path']
            if self.feature_dir and not os.path.isabs(feature_path):
                feature_path = os.path.join(self.feature_dir, feature_path)

            try:
                features = np.load(feature_path)
            except Exception as e:
                logging.error(f"加载特征失败: {feature_path} - {str(e)}")
                # 创建空特征作为回退
                features = np.zeros((self.max_frames, 40))

        # 归一化特征
        if self.feature_stats:
            features = (features - self.feature_stats['mean']) / (self.feature_stats['std'] + 1e-5)

        # 数据增强
        if self.augment and self.augmenter:
            features = self.augmenter(features)

        # 转换为张量
        features = torch.tensor(features, dtype=torch.float32)

        return {
            'features': features,
            'label': row['label'],
            'speaker_id': row['speaker_id'],
            'length': features.shape[0]  # 实际长度（填充前）
        }

    def get_speaker_weights(self):
        """计算每个说话人的样本权重（用于处理类别不平衡）"""
        speaker_counts = self.metadata['speaker_id'].value_counts().to_dict()
        weights = [1.0 / speaker_counts[sp] for sp in self.metadata['speaker_id']]
        return torch.tensor(weights, dtype=torch.float32)

    def get_speaker_mapping(self):
        """返回说话人到索引的映射"""
        return self.speaker_to_idx.copy()


def collate_fn(batch):
    """
    自定义批处理函数 - 处理变长特征序列

    返回:
    features: 填充后的特征张量 (batch, max_len, feat_dim)
    labels: 标签张量
    lengths: 实际长度列表
    """
    # 分离特征和标签
    features = [item['features'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    speaker_ids = [item['speaker_id'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)

    # 填充特征序列
    features_padded = pad_sequence(features, batch_first=True)

    return {
        'features': features_padded,
        'labels': labels,
        'lengths': lengths,
        'speaker_ids': speaker_ids
    }


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    平衡批次采样器 - 确保每个批次包含多个说话人的样本

    参数:
    dataset: LibriSpeechDataset实例
    batch_size: 批次大小
    speakers_per_batch: 每个批次的说话人数
    samples_per_speaker: 每个说话人的样本数
    """

    def __init__(self, dataset, batch_size, speakers_per_batch=4, samples_per_speaker=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.speakers_per_batch = speakers_per_batch
        self.samples_per_speaker = samples_per_speaker

        # 验证参数
        assert speakers_per_batch * samples_per_speaker == batch_size, \
            "批次大小必须等于speakers_per_batch * samples_per_speaker"

        # 按说话人分组索引
        self.speaker_indices = {}
        for idx, row in enumerate(self.dataset.metadata.itertuples()):
            sp = row.speaker_id
            if sp not in self.speaker_indices:
                self.speaker_indices[sp] = []
            self.speaker_indices[sp].append(idx)

        # 确保每个说话人有足够的样本
        self.valid_speakers = [
            sp for sp, indices in self.speaker_indices.items()
            if len(indices) >= samples_per_speaker
        ]

        logging.info(f"平衡批次采样器: {len(self.valid_speakers)}个有效说话人, "
                     f"每个批次{self.speakers_per_batch}个说话人, "
                     f"每个说话人{self.samples_per_speaker}个样本")

    def __iter__(self):
        # 创建批次索引列表
        indices = []

        # 随机打乱有效说话人
        speaker_list = self.valid_speakers.copy()
        random.shuffle(speaker_list)

        # 为每个批次选择说话人
        for i in range(0, len(speaker_list), self.speakers_per_batch):
            batch_speakers = speaker_list[i:i + self.speakers_per_batch]

            # 如果批次中说话人不足，跳过
            if len(batch_speakers) < self.speakers_per_batch:
                continue

            # 为每个说话人选择样本
            batch_indices = []
            for speaker in batch_speakers:
                # 从该说话人中随机选择样本
                selected = random.sample(self.speaker_indices[speaker], self.samples_per_speaker)
                batch_indices.extend(selected)

            indices.append(batch_indices)

        # 随机打乱批次顺序
        random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return len(self.valid_speakers) // self.speakers_per_batch


def create_data_loaders(metadata_path, feature_dir, feature_stats_path,
                        batch_size=64, num_workers=4, augment_train=True,
                        train_subset='train-clean-100', valid_subset='dev-clean',
                        use_raw_audio=False, balanced_batch=False):
    """
    创建训练和验证数据加载器

    返回:
    train_loader, valid_loader, speaker_mapping
    """
    # 加载全局特征统计信息
    feature_stats = joblib.load(feature_stats_path) if feature_stats_path else None

    # 加载元数据
    full_metadata = pd.read_csv(metadata_path)

    # 创建训练数据集
    train_metadata = full_metadata[full_metadata['subset'] == train_subset]
    train_dataset = LibriSpeechDataset(
        metadata=train_metadata,
        feature_dir=feature_dir,
        feature_stats=feature_stats,
        augment=augment_train,
        use_raw_audio=use_raw_audio
    )

    # 创建验证数据集
    valid_metadata = full_metadata[full_metadata['subset'] == valid_subset]
    valid_dataset = LibriSpeechDataset(
        metadata=valid_metadata,
        feature_dir=feature_dir,
        feature_stats=feature_stats,
        augment=False,  # 验证集不增强
        use_raw_audio=use_raw_audio
    )

    # 获取说话人映射
    speaker_mapping = train_dataset.get_speaker_mapping()

    # 创建数据加载器
    if balanced_batch:
        # 使用平衡批次采样器
        batch_sampler = BalancedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            speakers_per_batch=4,
            samples_per_speaker=batch_size // 4
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    logging.info(f"训练集: {len(train_dataset)}样本, {len(train_loader)}批次")
    logging.info(f"验证集: {len(valid_dataset)}样本, {len(valid_loader)}批次")

    return train_loader, valid_loader, speaker_mapping


# 测试函数
def test_data_loader():
    """测试数据加载器功能"""
    setup_logging()

    # 配置路径
    METADATA_PATH = "P:/PycharmProjects/pythonProject1/ChNNs/FeatureExtraction/extracted_features/feature_metadata.csv"
    FEATURE_DIR = "P:/PycharmProjects/pythonProject1/ChNNs/FeatureExtraction/"
    FEATURE_STATS_PATH = ("P:/PycharmProjects/pythonProject1/ChNNs/FeatureExtraction/extracted_features/feature_scaler"
                          ".joblib")

    logging.info("=== 测试数据加载器 ===")

    # 创建数据加载器
    train_loader, valid_loader, speaker_mapping = create_data_loaders(
        metadata_path=METADATA_PATH,
        feature_dir=FEATURE_DIR,
        feature_stats_path=FEATURE_STATS_PATH,
        batch_size=32,
        num_workers=2,
        augment_train=True,
        balanced_batch=True
    )

    # 获取一个训练批次
    batch = next(iter(train_loader))
    features = batch['features']
    labels = batch['labels']
    lengths = batch['lengths']

    logging.info(f"批次特征形状: {features.shape}")
    logging.info(f"批次标签: {labels[:10]}...")
    logging.info(f"序列长度: {lengths[:10]}...")

    # 获取一个验证批次
    valid_batch = next(iter(valid_loader))
    logging.info(f"验证批次特征形状: {valid_batch['features'].shape}")

    logging.info("数据加载器测试完成！")


if __name__ == "__main__":
    test_data_loader()