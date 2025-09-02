import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import glob
import pickle
from typing import Dict, List, Tuple, Optional, Union
import hashlib


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
        self.root_dir = root_dir
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.cache_dir = cache_dir  # 确保缓存目录属性存在
        self.audio_files = audio_files  # 使用预收集的文件列表，避免重复glob

        # 初始化缓存目录
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"启用特征缓存，目录：{self.cache_dir}")

        # 提取说话人ID
        self.speaker_ids = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in self.audio_files]

        # 说话人映射
        if speaker_to_label is not None:
            self.speaker_to_idx = speaker_to_label
        else:
            mapping_path = os.path.join(os.path.dirname(root_dir), 'speaker_mapping.pkl')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    self.speaker_to_idx = pickle.load(f)
            else:
                unique_speakers = sorted(list(set(self.speaker_ids)))
                self.speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}

        self.idx_to_speaker = {idx: speaker for speaker, idx in self.speaker_to_idx.items()}
        self.labels = [self.speaker_to_idx[speaker] for speaker in self.speaker_ids]

        print(f"Loaded {len(self.audio_files)} audio files from {len(set(self.speaker_ids))} speakers")

    def __len__(self) -> int:
        return len(self.audio_files)

    def _get_cache_path(self, audio_path: str) -> str:
        """生成唯一的缓存文件路径"""
        rel_path = os.path.relpath(audio_path, self.root_dir)
        param_str = f"seg_{self.segment_length}_sr_{self.sampling_rate}"
        hash_str = hashlib.md5(f"{rel_path}_{param_str}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_str}.npy")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """优先从磁盘加载缓存特征，不存在则生成并缓存"""
        audio_file = self.audio_files[idx]
        speaker_id = os.path.basename(os.path.dirname(os.path.dirname(audio_file)))
        label = self.speaker_to_idx[speaker_id]

        # 尝试从缓存加载
        if self.cache_dir:
            cache_path = self._get_cache_path(audio_file)
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path, allow_pickle=True).item()
                    return {
                        'audio': torch.FloatTensor(data['features']),
                        'label': torch.LongTensor([data['label']])[0],
                        'speaker_id': data['speaker_id'],
                        'file_path': audio_file
                    }
                except Exception as e:
                    print(f"缓存加载失败 {cache_path}: {e}，将重新生成")

        # 缓存不存在时，执行原处理逻辑
        audio, sr = librosa.load(audio_file, sr=self.sampling_rate)
        audio = self._process_audio(audio)

        if self.transform:
            audio = self.transform(audio)

        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sampling_rate,
            n_mfcc=40,
            n_fft=512,
            hop_length=256
        )
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs]).T

        # 标准化
        feature_mean = np.mean(features, axis=0)
        feature_std = np.std(features, axis=0)
        feature_std[feature_std == 0] = 1e-6
        features = (features - feature_mean) / feature_std

        # 构建结果
        result = {
            'audio': torch.FloatTensor(features),
            'label': torch.LongTensor([label])[0],
            'speaker_id': speaker_id,
            'file_path': audio_file
        }

        # 保存到缓存
        if self.cache_dir:
            try:
                cache_data = {
                    'features': features,
                    'label': label,
                    'speaker_id': speaker_id
                }
                np.save(cache_path, cache_data)
            except Exception as e:
                print(f"缓存保存失败 {cache_path}: {e}")

        return result

    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """处理音频为目标长度"""
        target_length = int(self.segment_length * self.sampling_rate)
        audio = self._remove_silence(audio)

        if len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        elif len(audio) > target_length:
            start = np.random.randint(0, len(audio) - target_length)
            audio = audio[start:start + target_length]

        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        return audio

    def _remove_silence(self, audio: np.ndarray, threshold: float = 0.03, min_duration: int = 320) -> np.ndarray:
        """去除静音片段"""
        energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
        frames = np.arange(len(energy))
        sample_points = np.linspace(0, len(energy) - 1, len(audio))
        energy_interpolated = np.interp(sample_points, frames, energy)
        mask = energy_interpolated > threshold
        mask = self._apply_min_duration(mask, min_duration)
        return audio[mask]

    def _apply_min_duration(self, mask: np.ndarray, min_duration: int) -> np.ndarray:
        """应用最小持续时间过滤"""
        changes = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        new_mask = np.zeros_like(mask)
        for start, end in zip(starts, ends):
            if end - start >= min_duration:
                new_mask[start:end] = True
        return new_mask


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
    # 一次性获取全局说话人映射和所有音频文件路径（避免重复glob）
    speaker_to_idx, train_files, dev_files, test_files = create_global_speaker_mapping_and_files(
        train_dir, dev_dir, test_dir
    )

    # 为不同数据集创建独立缓存子目录
    train_cache_dir = os.path.join(cache_dir, "train") if cache_dir else None
    dev_cache_dir = os.path.join(cache_dir, "dev") if cache_dir else None
    test_cache_dir = os.path.join(cache_dir, "test") if cache_dir else None

    # 创建数据集（使用预收集的文件列表）
    train_dataset = LibriSpeechDataset(
        audio_files=train_files,
        root_dir=train_dir,
        segment_length=segment_length,
        sampling_rate=sampling_rate,
        speaker_to_label=speaker_to_idx,
        cache_dir=train_cache_dir
    ) if train_files else None

    dev_dataset = LibriSpeechDataset(
        audio_files=dev_files,
        root_dir=dev_dir,
        segment_length=segment_length,
        sampling_rate=sampling_rate,
        speaker_to_label=speaker_to_idx,
        cache_dir=dev_cache_dir
    ) if dev_files else None

    test_dataset = LibriSpeechDataset(
        audio_files=test_files,
        root_dir=test_dir,
        segment_length=segment_length,
        sampling_rate=sampling_rate,
        speaker_to_label=speaker_to_idx,
        cache_dir=test_cache_dir
    ) if test_files else None

    pin_memory = torch.cuda.is_available()
    num_workers = min(num_workers, os.cpu_count() or 4)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    ) if train_dataset else None

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    ) if dev_dataset else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    ) if test_dataset else None

    # 优化数据集验证：一次遍历同时收集speakers和labels，减少数据加载次数
    print("Validating dataset consistency...")
    print(f"Global speakers: {len(speaker_to_idx)}")

    # 验证训练集
    if train_loader:
        train_speakers = set()
        train_labels = set()
        for batch in train_loader:
            labels = batch['label'].numpy()
            train_speakers.update(labels)
            train_labels.update(labels)
        print(f"Training speakers: {len(train_speakers)}")
        print(f"Training labels range: {min(train_labels)} - {max(train_labels)}" if train_labels else "No training data")

    # 验证开发集
    if dev_loader:
        dev_speakers = set()
        dev_labels = set()
        for batch in dev_loader:
            labels = batch['label'].numpy()
            dev_speakers.update(labels)
            dev_labels.update(labels)
        print(f"Development speakers: {len(dev_speakers)}")
        print(f"Development labels range: {min(dev_labels)} - {max(dev_labels)}" if dev_labels else "No development data")

    # 验证测试集
    if test_loader:
        test_speakers = set()
        test_labels = set()
        for batch in test_loader:
            labels = batch['label'].numpy()
            test_speakers.update(labels)
            test_labels.update(labels)
        print(f"Test speakers: {len(test_speakers)}")
        print(f"Test labels range: {min(test_labels)} - {max(test_labels)}" if test_labels else "No test data")

    # 检查标签范围
    all_labels = set()
    if train_loader and train_labels:
        all_labels.update(train_labels)
    if dev_loader and dev_labels:
        all_labels.update(dev_labels)
    if test_loader and test_labels:
        all_labels.update(test_labels)

    if all_labels:
        max_label = max(all_labels)
        if max_label >= len(speaker_to_idx):
            print(f"WARNING: Label {max_label} exceeds model output dimension {len(speaker_to_idx)}")
        else:
            print("Dataset validation passed - all labels are within model output range")

    return train_loader, dev_loader, test_loader, speaker_to_idx


def extract_features(audio: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
    """特征提取函数"""
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sampling_rate,
        n_mfcc=20,
        n_fft=512,
        hop_length=256
    )
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs]).T
    return features


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
