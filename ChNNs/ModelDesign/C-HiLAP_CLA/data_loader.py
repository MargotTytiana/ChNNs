"""
LibriSpeech数据加载器 - 混沌神经网络说话人识别项目
支持train-clean-100, dev-clean, test-clean数据集
"""

import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
import random
from pathlib import Path
import pickle


class LibriSpeechDataset(Dataset):
    """LibriSpeech数据集类，用于说话人识别任务"""

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 sample_rate: int = 16000,
                 max_length: float = 10.0,
                 min_length: float = 2.0,
                 normalize: bool = True):
        """
        初始化LibriSpeech数据集

        Args:
            data_root: 数据集根目录路径
            split: 数据集分割 ('train', 'dev', 'test')
            sample_rate: 音频采样率
            max_length: 最大音频长度（秒）
            min_length: 最小音频长度（秒）
            normalize: 是否对音频进行归一化
        """
        self.data_root = Path(data_root)
        self.split = split
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.min_length = min_length
        self.normalize = normalize

        # 根据split确定数据文件夹
        self.split_mapping = {
            'train': 'dataset',
            'dev': 'devDataset',
            'test': 'testDataset'
        }

        self.data_folder = self.data_root / self.split_mapping[split]

        # 加载音频文件列表和说话人标签
        self.audio_files, self.speaker_labels = self._load_data()

        # 创建说话人到索引的映射
        self.speaker_to_idx = self._create_speaker_mapping()
        self.num_speakers = len(self.speaker_to_idx)

        print(f"Loaded {len(self.audio_files)} audio files from {split} set")
        print(f"Number of speakers: {self.num_speakers}")

    def _load_data(self) -> Tuple[List[str], List[str]]:
        """加载音频文件路径和对应的说话人标签"""
        audio_files = []
        speaker_labels = []

        # LibriSpeech目录结构: speaker_id/chapter_id/audio_files
        for speaker_dir in sorted(self.data_folder.iterdir()):
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name

            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue

                # 查找.flac音频文件
                for audio_file in chapter_dir.glob("*.flac"):
                    # 检查音频文件长度是否符合要求
                    if self._check_audio_length(audio_file):
                        audio_files.append(str(audio_file))
                        speaker_labels.append(speaker_id)

        return audio_files, speaker_labels

    def _check_audio_length(self, audio_path: Path) -> bool:
        """检查音频长度是否在指定范围内"""
        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
            return self.min_length <= duration <= self.max_length
        except Exception as e:
            print(f"Warning: Could not load audio {audio_path}: {e}")
            return False

    def _create_speaker_mapping(self) -> Dict[str, int]:
        """创建说话人ID到数字索引的映射"""
        unique_speakers = sorted(set(self.speaker_labels))
        return {speaker: idx for idx, speaker in enumerate(unique_speakers)}

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        获取单个样本

        Returns:
            audio: 音频张量 [channels, samples]
            speaker_idx: 说话人索引
            speaker_id: 说话人ID字符串
        """
        audio_path = self.audio_files[idx]
        speaker_id = self.speaker_labels[idx]
        speaker_idx = self.speaker_to_idx[speaker_id]

        # 加载音频
        audio, orig_sr = torchaudio.load(audio_path)

        # 重采样到目标采样率
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            audio = resampler(audio)

        # 转换为单声道
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # 音频长度处理
        audio = self._process_audio_length(audio)

        # 归一化
        if self.normalize:
            audio = self._normalize_audio(audio)

        return audio, speaker_idx, speaker_id

    def _process_audio_length(self, audio: torch.Tensor) -> torch.Tensor:
        """处理音频长度，裁剪或填充到固定长度"""
        target_length = int(self.max_length * self.sample_rate)
        current_length = audio.shape[1]

        if current_length > target_length:
            # 随机裁剪
            start = random.randint(0, current_length - target_length)
            audio = audio[:, start:start + target_length]
        elif current_length < target_length:
            # 重复填充
            repeat_times = (target_length // current_length) + 1
            audio = audio.repeat(1, repeat_times)
            audio = audio[:, :target_length]

        return audio

    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """音频归一化"""
        return audio / (torch.max(torch.abs(audio)) + 1e-8)


class LibriSpeechDataLoader:
    """LibriSpeech数据加载器管理类"""

    def __init__(self,
                 data_root: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 sample_rate: int = 16000,
                 max_length: float = 10.0,
                 min_length: float = 2.0):
        """
        初始化数据加载器

        Args:
            data_root: 数据集根目录
            batch_size: 批处理大小
            num_workers: 数据加载工作进程数
            sample_rate: 音频采样率
            max_length: 最大音频长度
            min_length: 最小音频长度
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.min_length = min_length

        # 初始化数据集
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        # 初始化数据加载器
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None

        self._setup_datasets()
        self._setup_dataloaders()

    def _setup_datasets(self):
        """设置训练、验证、测试数据集"""
        print("设置训练数据集...")
        self.train_dataset = LibriSpeechDataset(
            data_root=self.data_root,
            split='train',
            sample_rate=self.sample_rate,
            max_length=self.max_length,
            min_length=self.min_length,
            normalize=True
        )

        print("设置验证数据集...")
        self.dev_dataset = LibriSpeechDataset(
            data_root=self.data_root,
            split='dev',
            sample_rate=self.sample_rate,
            max_length=self.max_length,
            min_length=self.min_length,
            normalize=True
        )

        print("设置测试数据集...")
        self.test_dataset = LibriSpeechDataset(
            data_root=self.data_root,
            split='test',
            sample_rate=self.sample_rate,
            max_length=self.max_length,
            min_length=self.min_length,
            normalize=True
        )

    def _setup_dataloaders(self):
        """设置数据加载器"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.dev_loader = DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def get_speaker_info(self) -> Dict:
        """获取说话人信息统计"""
        return {
            'num_speakers': self.train_dataset.num_speakers,
            'speaker_to_idx': self.train_dataset.speaker_to_idx,
            'train_samples': len(self.train_dataset),
            'dev_samples': len(self.dev_dataset),
            'test_samples': len(self.test_dataset)
        }

    def save_speaker_mapping(self, save_path: str):
        """保存说话人映射字典"""
        speaker_info = {
            'speaker_to_idx': self.train_dataset.speaker_to_idx,
            'num_speakers': self.train_dataset.num_speakers
        }
        with open(save_path, 'wb') as f:
            pickle.dump(speaker_info, f)
        print(f"说话人映射已保存至: {save_path}")

    def load_speaker_mapping(self, load_path: str):
        """加载说话人映射字典"""
        with open(load_path, 'rb') as f:
            speaker_info = pickle.load(f)
        print(f"说话人映射已加载自: {load_path}")
        return speaker_info


def collate_fn(batch):
    """自定义批处理函数"""
    audios, speaker_indices, speaker_ids = zip(*batch)

    # 堆叠音频张量
    audios = torch.stack(audios, dim=0)

    # 转换说话人索引为张量
    speaker_indices = torch.tensor(speaker_indices, dtype=torch.long)

    return audios, speaker_indices, list(speaker_ids)


# 测试和使用示例
if __name__ == "__main__":
    # 数据加载器使用示例
    data_root = "/path/to/your/librispeech"  # 替换为实际路径

    # 初始化数据加载器
    dataloader_manager = LibriSpeechDataLoader(
        data_root=data_root,
        batch_size=16,
        num_workers=4,
        sample_rate=16000,
        max_length=8.0,
        min_length=2.0
    )

    # 获取说话人信息
    speaker_info = dataloader_manager.get_speaker_info()
    print("说话人信息:", speaker_info)

    # 保存说话人映射
    dataloader_manager.save_speaker_mapping("speaker_mapping.pkl")

    # 测试数据加载
    print("\n测试训练数据加载...")
    for i, (audios, speaker_indices, speaker_ids) in enumerate(dataloader_manager.train_loader):
        print(f"Batch {i + 1}:")
        print(f"  音频形状: {audios.shape}")
        print(f"  说话人索引: {speaker_indices.shape}")
        print(f"  说话人ID样例: {speaker_ids[:3]}")

        if i >= 2:  # 只测试前几个batch
            break

    print("数据加载器测试完成！")