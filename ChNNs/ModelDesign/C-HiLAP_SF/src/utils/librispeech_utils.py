import os
import json
import random
import numpy as np
import torch
import torchaudio
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import re
import logging
from tqdm import tqdm
import torch.nn.functional as F


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LibriSpeechDataset:
    """
    LibriSpeech数据集处理类，用于加载和预处理LibriSpeech数据。

    LibriSpeech的目录结构通常为：
    - {root_dir}/{split}/{speaker_id}/{chapter_id}/{speaker_id}_{chapter_id}_{utterance_id}.flac

    例如：
    - train-clean-100/1/1-19-0034/1-19-0034-0001.flac

    Args:
        root_dir: LibriSpeech数据集的根目录
        split: 数据集分割，例如'train-clean-100', 'dev-clean', 'test-clean'等
        sample_rate: 目标采样率
        transform: 可选的特征转换函数
        max_duration: 最大音频时长（秒），超过此长度的音频将被截断
        cache_dir: 缓存目录，用于存储处理后的元数据
        use_cache: 是否使用缓存的元数据
    """

    def __init__(
            self,
            root_dir: str,
            split: str = 'train-clean-100',
            sample_rate: int = 16000,
            transform: Optional[Callable] = None,
            max_duration: Optional[float] = None,
            cache_dir: Optional[str] = None,
            use_cache: bool = True
    ):
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate
        self.transform = transform
        self.max_duration = max_duration
        self.cache_dir = cache_dir
        self.use_cache = use_cache

        # 构建数据集路径
        self.dataset_path = os.path.join(root_dir, split)

        # 加载或解析元数据
        self.metadata = self._load_or_parse_metadata()

        # 构建说话人ID映射
        self.speaker_ids = sorted(list(set(item['speaker_id'] for item in self.metadata)))
        self.speaker_id_map = {spk: idx for idx, spk in enumerate(self.speaker_ids)}

        logger.info(f"加载了 {len(self.metadata)} 条语音，来自 {len(self.speaker_ids)} 个说话人")

    def _load_or_parse_metadata(self) -> List[Dict]:
        """
        加载或解析数据集元数据。

        如果启用了缓存并且缓存文件存在，则从缓存加载元数据；
        否则，解析数据集目录并创建元数据。

        Returns:
            包含元数据的字典列表
        """
        # 检查缓存
        cache_file = None
        if self.cache_dir and self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, f"{self.split}_metadata.json")

            if os.path.exists(cache_file):
                logger.info(f"从缓存加载元数据: {cache_file}")
                with open(cache_file, 'r') as f:
                    return json.load(f)

        # 解析数据集
        logger.info(f"解析数据集: {self.dataset_path}")
        metadata = self._parse_dataset()

        # 保存缓存
        if cache_file:
            logger.info(f"保存元数据到缓存: {cache_file}")
            with open(cache_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        return metadata

    def _parse_dataset(self) -> List[Dict]:
        """
        解析数据集目录，提取所有音频文件的元数据。

        Returns:
            包含元数据的字典列表
        """
        metadata = []

        # 检查数据集路径是否存在
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")

        # 遍历数据集目录
        for speaker_dir in tqdm(os.listdir(self.dataset_path), desc="解析说话人目录"):
            speaker_path = os.path.join(self.dataset_path, speaker_dir)
            if not os.path.isdir(speaker_path):
                continue

            speaker_id = speaker_dir

            for chapter_dir in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter_dir)
                if not os.path.isdir(chapter_path):
                    continue

                chapter_id = chapter_dir

                for audio_file in os.listdir(chapter_path):
                    if not audio_file.endswith('.flac'):
                        continue

                    audio_path = os.path.join(chapter_path, audio_file)

                    # 从文件名提取信息
                    # 文件名格式通常为: {speaker_id}-{chapter_id}-{utterance_id}.flac
                    # 或者: {speaker_id}_{chapter_id}_{utterance_id}.flac
                    file_parts = os.path.splitext(audio_file)[0].replace('-', '_').split('_')
                    if len(file_parts) >= 3:
                        utterance_id = file_parts[-1]
                    else:
                        utterance_id = "unknown"

                    # 获取音频时长
                    try:
                        info = torchaudio.info(audio_path)
                        duration = info.num_frames / info.sample_rate
                    except Exception as e:
                        logger.warning(f"无法获取音频信息 {audio_path}: {e}")
                        duration = 0

                    # 添加元数据
                    metadata.append({
                        'audio_path': audio_path,
                        'speaker_id': speaker_id,
                        'chapter_id': chapter_id,
                        'utterance_id': utterance_id,
                        'duration': duration
                    })

        return metadata

    def __len__(self) -> int:
        """返回数据集中的样本数量"""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        """
        获取指定索引的样本。

        Args:
            idx: 样本索引

        Returns:
            包含音频数据和元数据的字典
        """
        item = self.metadata[idx]
        audio_path = item['audio_path']

        # 加载音频
        waveform, sample_rate = self._load_audio(audio_path)

        # 应用转换
        if self.transform:
            features = self.transform(waveform)
        else:
            features = waveform

        # 获取说话人标签
        speaker_id = item['speaker_id']
        speaker_label = self.speaker_id_map[speaker_id]

        return {
            'features': features,
            'waveform': waveform,
            'speaker_id': speaker_id,
            'speaker_label': speaker_label,
            'chapter_id': item['chapter_id'],
            'utterance_id': item['utterance_id'],
            'audio_path': audio_path
        }

    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        加载音频文件并进行必要的预处理。

        Args:
            audio_path: 音频文件路径

        Returns:
            处理后的波形张量和采样率
        """
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)

        # 重采样
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.sample_rate

        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 截断过长的音频
        if self.max_duration is not None:
            max_samples = int(self.max_duration * sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

        return waveform, sample_rate

    def get_speaker_utterances(self, speaker_id: str) -> List[Dict]:
        """
        获取指定说话人的所有语音。

        Args:
            speaker_id: 说话人ID

        Returns:
            该说话人的所有语音元数据
        """
        return [item for item in self.metadata if item['speaker_id'] == speaker_id]

    def get_speakers(self) -> List[str]:
        """
        获取数据集中的所有说话人ID。

        Returns:
            说话人ID列表
        """
        return self.speaker_ids

    def get_speaker_label(self, speaker_id: str) -> int:
        """
        获取说话人ID对应的标签。

        Args:
            speaker_id: 说话人ID

        Returns:
            说话人标签
        """
        return self.speaker_id_map[speaker_id]

    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息。

        Returns:
            包含统计信息的字典
        """
        # 计算每个说话人的语音数量
        speaker_counts = {}
        for item in self.metadata:
            speaker_id = item['speaker_id']
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

        # 计算总时长
        total_duration = sum(item['duration'] for item in self.metadata)

        # 计算平均时长
        avg_duration = total_duration / len(self.metadata) if self.metadata else 0

        return {
            'total_utterances': len(self.metadata),
            'total_speakers': len(self.speaker_ids),
            'total_duration': total_duration,
            'average_duration': avg_duration,
            'speaker_counts': speaker_counts
        }


class LibriSpeechTrialGenerator:
    """
    LibriSpeech试验对生成器，用于生成说话人验证的试验对。

    Args:
        dataset: LibriSpeechDataset实例
        num_trials: 要生成的试验对数量
        genuine_ratio: 合法配对（同一说话人）的比例
        exclude_same_utterance: 是否排除相同语音的配对
        seed: 随机种子
    """

    def __init__(
            self,
            dataset: LibriSpeechDataset,
            num_trials: int = 1000,
            genuine_ratio: float = 0.5,
            exclude_same_utterance: bool = True,
            seed: int = 42
    ):
        self.dataset = dataset
        self.num_trials = num_trials
        self.genuine_ratio = genuine_ratio
        self.exclude_same_utterance = exclude_same_utterance
        self.seed = seed

        # 设置随机种子
        random.seed(seed)

        # 按说话人ID组织元数据
        self.speaker_utterances = {}
        for item in dataset.metadata:
            speaker_id = item['speaker_id']
            if speaker_id not in self.speaker_utterances:
                self.speaker_utterances[speaker_id] = []
            self.speaker_utterances[speaker_id].append(item)

    def generate_trials(self) -> List[Dict]:
        """
        生成试验对。

        Returns:
            试验对列表，每个试验对是一个字典，包含两个语音的信息和标签
        """
        trials = []
        num_genuine = int(self.num_trials * self.genuine_ratio)
        num_impostor = self.num_trials - num_genuine

        # 生成合法配对（同一说话人）
        for _ in range(num_genuine):
            trial = self._generate_genuine_trial()
            if trial:
                trials.append(trial)

        # 生成冒充配对（不同说话人）
        for _ in range(num_impostor):
            trial = self._generate_impostor_trial()
            if trial:
                trials.append(trial)

        # 打乱试验对
        random.shuffle(trials)

        return trials

    def _generate_genuine_trial(self) -> Optional[Dict]:
        """
        生成一个合法配对（同一说话人）。

        Returns:
            试验对字典，如果无法生成则返回None
        """
        # 选择一个有至少两个语音的说话人
        eligible_speakers = [spk for spk, utts in self.speaker_utterances.items() if len(utts) >= 2]
        if not eligible_speakers:
            return None

        speaker_id = random.choice(eligible_speakers)
        utterances = self.speaker_utterances[speaker_id]

        # 随机选择两个不同的语音
        if len(utterances) >= 2:
            utt1, utt2 = random.sample(utterances, 2)

            return {
                'enrollment': utt1['audio_path'],
                'test': utt2['audio_path'],
                'speaker_id': speaker_id,
                'is_genuine': True,
                'label': 1
            }

        return None

    def _generate_impostor_trial(self) -> Optional[Dict]:
        """
        生成一个冒充配对（不同说话人）。

        Returns:
            试验对字典，如果无法生成则返回None
        """
        # 至少需要两个说话人
        if len(self.speaker_utterances) < 2:
            return None

        # 随机选择两个不同的说话人
        speaker1, speaker2 = random.sample(list(self.speaker_utterances.keys()), 2)

        # 从每个说话人中随机选择一个语音
        utt1 = random.choice(self.speaker_utterances[speaker1])
        utt2 = random.choice(self.speaker_utterances[speaker2])

        return {
            'enrollment': utt1['audio_path'],
            'test': utt2['audio_path'],
            'enrollment_speaker': speaker1,
            'test_speaker': speaker2,
            'is_genuine': False,
            'label': 0
        }

    def save_trials(self, output_path: str) -> None:
        """
        生成试验对并保存到文件。

        Args:
            output_path: 输出文件路径
        """
        trials = self.generate_trials()

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存为JSON格式
        with open(output_path, 'w') as f:
            json.dump(trials, f, indent=2)

        logger.info(f"保存了 {len(trials)} 个试验对到 {output_path}")


class LibriSpeechFeatureExtractor:
    """
    LibriSpeech特征提取器，用于从音频中提取特征。

    Args:
        sample_rate: 音频采样率
        feature_type: 特征类型，'mfcc', 'fbank', 'chaotic' 或 'combined'
        feature_dim: 特征维度
        chaotic_extractor: 可选的混沌特征提取器
    """

    def __init__(
            self,
            sample_rate: int = 16000,
            feature_type: str = 'mfcc',
            feature_dim: int = 40,
            chaotic_extractor: Optional[Callable] = None
    ):
        self.sample_rate = sample_rate
        self.feature_type = feature_type
        self.feature_dim = feature_dim
        self.chaotic_extractor = chaotic_extractor

        # 初始化特征提取器
        if feature_type == 'mfcc':
            self.extractor = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=feature_dim,
                melkwargs={'n_fft': 512, 'n_mels': 80, 'hop_length': 160}
            )
        elif feature_type == 'fbank':
            self.extractor = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=512,
                n_mels=feature_dim,
                hop_length=160
            )
        elif feature_type == 'chaotic' and chaotic_extractor is None:
            raise ValueError("使用'chaotic'特征类型时必须提供chaotic_extractor")
        elif feature_type == 'combined' and chaotic_extractor is None:
            raise ValueError("使用'combined'特征类型时必须提供chaotic_extractor")

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        提取特征。

        Args:
            waveform: 波形张量，形状为 [channels, samples]

        Returns:
            特征张量
        """
        # 确保输入是单声道
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.feature_type == 'mfcc':
            features = self.extractor(waveform)
            # 应用对数变换
            features = torch.log(features + 1e-6)
        elif self.feature_type == 'fbank':
            features = self.extractor(waveform)
            # 应用对数变换
            features = torch.log(features + 1e-6)
        elif self.feature_type == 'chaotic':
            # 使用提供的混沌特征提取器
            features = self.chaotic_extractor(waveform)
        elif self.feature_type == 'combined':
            # 提取常规特征
            if hasattr(self, 'extractor'):
                regular_features = self.extractor(waveform)
                regular_features = torch.log(regular_features + 1e-6)
            else:
                # 如果没有常规提取器，使用MFCC作为默认
                mfcc_extractor = torchaudio.transforms.MFCC(
                    sample_rate=self.sample_rate,
                    n_mfcc=self.feature_dim,
                    melkwargs={'n_fft': 512, 'n_mels': 80, 'hop_length': 160}
                )
                regular_features = mfcc_extractor(waveform)
                regular_features = torch.log(regular_features + 1e-6)

            # 提取混沌特征
            chaotic_features = self.chaotic_extractor(waveform)

            # 组合特征（可能需要调整维度）
            if regular_features.shape[1] != chaotic_features.shape[1]:
                # 调整时间维度
                min_length = min(regular_features.shape[1], chaotic_features.shape[1])
                regular_features = regular_features[:, :min_length]
                chaotic_features = chaotic_features[:, :min_length]

            # 拼接特征
            features = torch.cat([regular_features, chaotic_features], dim=0)
        else:
            raise ValueError(f"不支持的特征类型: {self.feature_type}")

        return features


def create_librispeech_dataloader(
        dataset: LibriSpeechDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """
    创建LibriSpeech数据加载器。

    Args:
        dataset: LibriSpeechDataset实例
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数

    Returns:
        PyTorch数据加载器
    """

    def collate_fn(batch):
        """
        自定义的批次整理函数，处理可变长度的特征。
        """
        # 提取各字段
        features = [item['features'] for item in batch]
        waveforms = [item['waveform'] for item in batch]
        speaker_labels = torch.tensor([item['speaker_label'] for item in batch])
        speaker_ids = [item['speaker_id'] for item in batch]
        audio_paths = [item['audio_path'] for item in batch]

        # 获取每个特征的长度
        lengths = torch.tensor([feat.shape[1] for feat in features])

        # 找到最大长度
        max_length = max(lengths)

        # 填充特征到相同长度
        padded_features = []
        for feat in features:
            if feat.shape[1] < max_length:
                padding = torch.zeros(feat.shape[0], max_length - feat.shape[1], device=feat.device)
                padded_feat = torch.cat([feat, padding], dim=1)
            else:
                padded_feat = feat
            padded_features.append(padded_feat)

        # 堆叠特征
        features_tensor = torch.stack(padded_features)

        return {
            'features': features_tensor,
            'lengths': lengths,
            'waveforms': waveforms,
            'speaker_labels': speaker_labels,
            'speaker_ids': speaker_ids,
            'audio_paths': audio_paths
        }

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def evaluate_speaker_verification(
        model: torch.nn.Module,
        trials: List[Dict],
        feature_extractor: Callable,
        device: torch.device,
        distance_metric: str = 'cosine'
) -> Dict:
    """
    评估说话人验证性能。

    Args:
        model: 说话人嵌入模型
        trials: 试验对列表
        feature_extractor: 特征提取函数
        device: 计算设备
        distance_metric: 距离度量方式，'cosine' 或 'euclidean'

    Returns:
        包含评估结果的字典
    """
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for trial in tqdm(trials, desc="评估试验对"):
            # 加载音频
            enroll_audio, _ = torchaudio.load(trial['enrollment'])
            test_audio, _ = torchaudio.load(trial['test'])

            # 提取特征
            enroll_feat = feature_extractor(enroll_audio).to(device)
            test_feat = feature_extractor(test_audio).to(device)

            # 添加批次维度
            enroll_feat = enroll_feat.unsqueeze(0)
            test_feat = test_feat.unsqueeze(0)

            # 提取嵌入向量
            enroll_emb = model(enroll_feat, extract_embedding=True)
            test_emb = model(test_feat, extract_embedding=True)

            # 计算相似度分数
            if distance_metric == 'cosine':
                # 余弦相似度
                score = F.cosine_similarity(enroll_emb, test_emb).item()
            elif distance_metric == 'euclidean':
                # 欧几里得距离（转换为相似度）
                distance = torch.norm(enroll_emb - test_emb).item()
                score = 1.0 / (1.0 + distance)  # 转换为[0,1]范围的相似度
            else:
                raise ValueError(f"不支持的距离度量: {distance_metric}")

            scores.append(score)
            labels.append(trial['label'])

    # 计算EER和minDCF
    from metrics import compute_eer, compute_mindcf

    scores = np.array(scores)
    labels = np.array(labels)

    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    eer, threshold = compute_eer(genuine_scores, impostor_scores)
    mindcf, _ = compute_mindcf(genuine_scores, impostor_scores)

    # 计算准确率
    predictions = (scores >= threshold).astype(int)
    accuracy = np.mean(predictions == labels)

    return {
        'eer': eer,
        'mindcf': mindcf,
        'threshold': threshold,
        'accuracy': accuracy,
        'scores': scores.tolist(),
        'labels': labels.tolist()
    }


# 示例用法
if __name__ == "__main__":
    # 设置LibriSpeech数据集路径
    librispeech_root = "/path/to/librispeech"

    # 创建数据集
    train_dataset = LibriSpeechDataset(
        root_dir=librispeech_root,
        split="train-clean-100",
        sample_rate=16000,
        cache_dir="./cache"
    )

    # 打印数据集统计信息
    stats = train_dataset.get_statistics()
    print(f"数据集统计信息:")
    print(f"  总语音数: {stats['total_utterances']}")
    print(f"  总说话人数: {stats['total_speakers']}")
    print(f"  总时长: {stats['total_duration']:.2f} 秒")
    print(f"  平均时长: {stats['average_duration']:.2f} 秒")

    # 创建特征提取器
    feature_extractor = LibriSpeechFeatureExtractor(
        sample_rate=16000,
        feature_type='mfcc',
        feature_dim=40
    )

    # 应用特征提取器
    train_dataset.transform = feature_extractor

    # 创建数据加载器
    train_loader = create_librispeech_dataloader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    # 生成试验对
    trial_generator = LibriSpeechTrialGenerator(
        dataset=train_dataset,
        num_trials=1000,
        genuine_ratio=0.5
    )

    trials = trial_generator.generate_trials()
    print(f"生成了 {len(trials)} 个试验对")

    # 保存试验对
    trial_generator.save_trials("./trials/librispeech_trials.json")

    # 加载一个批次的数据
    for batch in train_loader:
        features = batch['features']
        lengths = batch['lengths']
        speaker_labels = batch['speaker_labels']

        print(f"特征形状: {features.shape}")
        print(f"长度形状: {lengths.shape}")
        print(f"说话人标签形状: {speaker_labels.shape}")

        # 只打印一个批次
        break