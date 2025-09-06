import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import glob
import pickle
from typing import Dict, List, Tuple, Optional
import hashlib

from chaos_features import ChaoticFeatureExtractor


def create_global_speaker_mapping_and_files(train_dir, dev_dir, test_dir):
    """创建全局说话人映射并同时收集所有数据集的音频文件路径"""
    all_speakers = set()
    # 收集各数据集的音频文件路径
    train_files = glob.glob(os.path.join(train_dir, "**", "*.flac"), recursive=True) if train_dir else []
    dev_files = glob.glob(os.path.join(dev_dir, "**", "*.flac"), recursive=True) if dev_dir else []
    test_files = glob.glob(os.path.join(test_dir, "**", "*.flac"), recursive=True) if test_dir else []

    # 从所有文件中提取说话人ID
    all_files = train_files + dev_files + test_files
    speaker_ids = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in all_files]
    all_speakers.update(speaker_ids)

    # 创建映射字典
    speaker_to_label = {speaker: idx for idx, speaker in enumerate(sorted(all_speakers))}

    # 保存映射
    mapping_dir = os.path.dirname(train_dir) if train_dir else "."
    mapping_path = os.path.join(mapping_dir, 'speaker_mapping.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump(speaker_to_label, f)

    print(f"创建了全局说话人映射，共 {len(speaker_to_label)} 个说话人")
    return speaker_to_label, train_files, dev_files, test_files


class LibriSpeechDataset(Dataset):
    """支持磁盘缓存的LibriSpeech数据集类"""

    def __init__(
            self,
            audio_files: List[str],  # 直接接收预收集的音频文件列表
            root_dir: str,
            segment_length: float = 3.0,
            sampling_rate: int = 16000,
            transform=None,
            speaker_to_label: Optional[Dict[str, int]] = None,
            cache_dir: Optional[str] = None
    ):
        self.audio_files = audio_files
        self.root_dir = root_dir
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.speaker_to_label = speaker_to_label
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.audio_files)

    def _get_cache_path(self, audio_path: str) -> str:
        """生成唯一的缓存文件路径"""
        if not self.cache_dir:
            return ""
        # 使用文件路径的hash作为缓存文件名，避免路径冲突
        file_hash = hashlib.md5(audio_path.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.npy")
        return cache_file

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """优先从磁盘加载缓存特征，不存在则生成并缓存"""
        audio_path = self.audio_files[idx]
        cache_path = self._get_cache_path(audio_path)

        if cache_path and os.path.exists(cache_path):
            features = np.load(cache_path)
        else:
            audio, _ = librosa.load(audio_path, sr=self.sampling_rate)
            audio = self._process_audio(audio)
            # 修复特征提取方式：实例化后调用extract方法
            extractor = ChaoticFeatureExtractor(sampling_rate=self.sampling_rate)
            features = extractor.extract(audio)
            # 添加特征维度校验
            if not isinstance(features, np.ndarray):
                raise ValueError("ChaoticFeatureExtractor应返回numpy数组")
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, features)

        if self.transform:
            features = self.transform(features)

        speaker_id = os.path.basename(os.path.dirname(os.path.dirname(audio_path)))
        label = self.speaker_to_label[speaker_id] if self.speaker_to_label else -1

        return {
            "audio": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long)
        }

    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """处理音频为目标长度"""
        target_length = int(self.segment_length * self.sampling_rate)
        if len(audio) > target_length:
            start = np.random.randint(0, len(audio) - target_length)
            audio = audio[start:start + target_length]
        else:
            pad_width = target_length - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
        return audio

    def _remove_silence(self, audio: np.ndarray, threshold: float = 0.01, min_duration: int = 160) -> np.ndarray:
        """去除静音片段"""
        # 这里可以保留或删除，视需求而定
        pass

    def _apply_min_duration(self, mask: np.ndarray, min_duration: int) -> np.ndarray:
        """应用最小持续时间过滤"""
        pass


def create_dataloaders(
        train_dir: str,
        dev_dir: str,
        test_dir: str,
        batch_size: int = 32,
        segment_length: float = 3.0,
        sampling_rate: int = 16000,
        num_workers: int = 4,
        cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """创建DataLoader，优化数据加载流程"""
    speaker_to_label, train_files, dev_files, test_files = create_global_speaker_mapping_and_files(
        train_dir, dev_dir, test_dir)

    train_dataset = LibriSpeechDataset(
        train_files, train_dir, segment_length, sampling_rate,
        speaker_to_label=speaker_to_label, cache_dir=cache_dir
    ) if train_files else None

    dev_dataset = LibriSpeechDataset(
        dev_files, dev_dir, segment_length, sampling_rate,
        speaker_to_label=speaker_to_label, cache_dir=cache_dir
    ) if dev_files else None

    test_dataset = LibriSpeechDataset(
        test_files, test_dir, segment_length, sampling_rate,
        speaker_to_label=speaker_to_label, cache_dir=cache_dir
    ) if test_files else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) if train_dataset else None
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if dev_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_dataset else None

    return train_loader, dev_loader, test_loader, speaker_to_label


def extract_features(audio: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
    """特征提取函数，已由chaos_features.py中的extract_chaos_features替代"""
    pass


if __name__ == "__main__":
    train_dir = "P:/PycharmProjects/pythonProject1/dataset"
    dev_dir = "P:/PycharmProjects/pythonProject1/devDataset"
    test_dir = "P:/PycharmProjects/pythonProject1/testDataset"
    cache_root = "P:/PycharmProjects/pythonProject1/preprocessed_cache"

    train_loader, dev_loader, test_loader, speaker_mapping = create_dataloaders(
        train_dir, dev_dir, test_dir,
        batch_size=32,
        cache_dir=cache_root
    )

    print(f"Number of speakers: {len(speaker_mapping)}")
    print(f"Number of training batches: {len(train_loader)}" if train_loader else "No training data")
    print(f"Number of development batches: {len(dev_loader)}" if dev_loader else "No development data")
    print(f"Number of test batches: {len(test_loader)}" if test_loader else "No test data")

    if train_loader:
        for batch in train_loader:
            audio = batch['audio']
            labels = batch['label']
            print(f"Audio batch shape: {audio.shape}")
            print(f"Labels shape: {labels.shape}")
            break
