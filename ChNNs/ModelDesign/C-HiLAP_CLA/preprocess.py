"""
数据预处理模块 - 混沌神经网络说话人识别项目
支持VoxCeleb、LibriSpeech等多种数据集的预处理
包含音频标准化、VAD、特征提取和缓存等功能
"""

import os
import json
import pickle
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import warnings
import webrtcvad
from scipy import signal
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """音频预处理器主类"""

    def __init__(self,
                 target_sr: int = 16000,
                 n_mels: int = 80,
                 n_mfcc: int = 13,
                 hop_length: int = 256,
                 win_length: int = 512,
                 n_fft: int = 512,
                 normalize: bool = True,
                 use_vad: bool = True,
                 vad_mode: int = 3,
                 min_duration: float = 2.0,
                 max_duration: float = 10.0):
        """
        初始化音频预处理器

        Args:
            target_sr: 目标采样率
            n_mels: Mel滤波器数量
            n_mfcc: MFCC系数数量
            hop_length: 帧移
            win_length: 窗长
            n_fft: FFT点数
            normalize: 是否归一化
            use_vad: 是否使用VAD
            vad_mode: VAD模式 (0-3, 3最严格)
            min_duration: 最小音频长度(秒)
            max_duration: 最大音频长度(秒)
        """
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.normalize = normalize
        self.use_vad = use_vad
        self.vad_mode = vad_mode
        self.min_duration = min_duration
        self.max_duration = max_duration

        # 初始化VAD
        if self.use_vad:
            self.vad = webrtcvad.Vad(vad_mode)

        # Mel滤波器组
        self.mel_filters = librosa.filters.mel(
            sr=target_sr,
            n_fft=n_fft,
            n_mels=n_mels
        )

        print(f"AudioPreprocessor initialized with:")
        print(f"  - Target sample rate: {target_sr} Hz")
        print(f"  - Feature dimensions: Mel={n_mels}, MFCC={n_mfcc}")
        print(f"  - VAD enabled: {use_vad}")
        print(f"  - Duration range: {min_duration}-{max_duration}s")

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        加载音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            audio: 音频信号
            sr: 采样率
        """
        try:
            # 使用librosa加载，自动转换为目标采样率
            audio, sr = librosa.load(
                str(audio_path),
                sr=self.target_sr,
                mono=True
            )
            return audio, sr

        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None, None

    def apply_vad(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        应用语音活动检测 (VAD)

        Args:
            audio: 输入音频信号
            sr: 采样率

        Returns:
            voiced_audio: 去除静音后的音频
        """
        if not self.use_vad or len(audio) == 0:
            return audio

        try:
            # 确保采样率为8k, 16k或32k (webrtcvad要求)
            if sr not in [8000, 16000, 32000]:
                # 重采样到16kHz
                audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            else:
                audio_resampled = audio

            # 转换为16位整数
            audio_int16 = (audio_resampled * 32767).astype(np.int16)

            # 分帧处理 (10ms frames for VAD)
            frame_duration = 0.01  # 10ms
            frame_length = int(sr * frame_duration)

            voiced_frames = []

            for i in range(0, len(audio_int16) - frame_length + 1, frame_length):
                frame = audio_int16[i:i + frame_length]

                # VAD需要特定帧长度
                if len(frame) == frame_length:
                    is_speech = self.vad.is_speech(frame.tobytes(), sr)
                    if is_speech:
                        voiced_frames.extend(frame)

            if len(voiced_frames) > 0:
                voiced_audio = np.array(voiced_frames, dtype=np.float32) / 32767.0
                return voiced_audio
            else:
                # 如果没有检测到语音，返回原始音频
                return audio

        except Exception as e:
            print(f"VAD processing error: {e}")
            return audio

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """音频归一化"""
        if not self.normalize or len(audio) == 0:
            return audio

        # 移除直流分量
        audio = audio - np.mean(audio)

        # 幅度归一化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95  # 留一点余量

        return audio

    def trim_or_pad_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """裁剪或填充音频到指定长度范围"""
        duration = len(audio) / sr

        # 如果太短，重复填充
        if duration < self.min_duration:
            target_length = int(self.min_duration * sr)
            repeat_times = int(np.ceil(target_length / len(audio)))
            audio = np.tile(audio, repeat_times)[:target_length]

        # 如果太长，随机裁剪
        elif duration > self.max_duration:
            target_length = int(self.max_duration * sr)
            if len(audio) > target_length:
                start = np.random.randint(0, len(audio) - target_length + 1)
                audio = audio[start:start + target_length]

        return audio

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """提取Mel频谱图"""
        try:
            # 计算STFT
            stft = librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window='hann'
            )

            # 计算幅度谱
            magnitude = np.abs(stft)

            # 应用Mel滤波器
            mel_spec = np.dot(self.mel_filters, magnitude)

            # 转换为对数刻度
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            return log_mel_spec.T  # [time, mel_bins]

        except Exception as e:
            print(f"Mel spectrogram extraction error: {e}")
            return np.zeros((100, self.n_mels))  # 返回默认形状

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """提取MFCC特征"""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.target_sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length
            )

            # 添加一阶和二阶差分
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            # 拼接特征
            features = np.vstack([mfcc, delta, delta2])

            return features.T  # [time, mfcc_dim * 3]

        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return np.zeros((100, self.n_mfcc * 3))

    def process_single_audio(self,
                             audio_path: Union[str, Path],
                             output_dir: Optional[Path] = None) -> Dict:
        """
        处理单个音频文件

        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录

        Returns:
            result: 处理结果字典
        """
        audio_path = Path(audio_path)

        result = {
            'audio_path': str(audio_path),
            'success': False,
            'error': None,
            'duration': 0.0,
            'features': {}
        }

        try:
            # 加载音频
            audio, sr = self.load_audio(audio_path)
            if audio is None:
                result['error'] = 'Failed to load audio'
                return result

            # 记录原始时长
            original_duration = len(audio) / sr
            result['original_duration'] = original_duration

            # 应用VAD
            audio = self.apply_vad(audio, sr)

            # 音频归一化
            audio = self.normalize_audio(audio)

            # 长度处理
            audio = self.trim_or_pad_audio(audio, sr)

            # 记录处理后时长
            result['duration'] = len(audio) / sr

            # 提取特征
            mel_spec = self.extract_mel_spectrogram(audio)
            mfcc = self.extract_mfcc(audio)

            result['features'] = {
                'mel_spectrogram': mel_spec,
                'mfcc': mfcc,
                'audio_processed': audio  # 可选：保存处理后的音频
            }

            # 保存特征到文件 (可选)
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # 生成输出文件名
                feature_file = output_dir / f"{audio_path.stem}_features.npz"

                np.savez_compressed(
                    feature_file,
                    mel_spectrogram=mel_spec,
                    mfcc=mfcc,
                    audio=audio,
                    sample_rate=sr,
                    original_duration=original_duration,
                    processed_duration=result['duration']
                )

                result['feature_file'] = str(feature_file)

            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            print(f"Error processing {audio_path}: {e}")

        return result


class DatasetPreprocessor:
    """数据集预处理器"""

    def __init__(self,
                 preprocessor: AudioPreprocessor,
                 num_workers: int = 4):
        """
        初始化数据集预处理器

        Args:
            preprocessor: 音频预处理器
            num_workers: 并行处理进程数
        """
        self.preprocessor = preprocessor
        self.num_workers = num_workers

    def process_voxceleb_dataset(self,
                                 data_root: str,
                                 output_dir: str,
                                 splits: List[str] = ['train', 'test']) -> Dict:
        """
        处理VoxCeleb数据集

        Args:
            data_root: VoxCeleb数据集根目录
            output_dir: 输出目录
            splits: 要处理的数据分割

        Returns:
            processing_stats: 处理统计信息
        """
        data_root = Path(data_root)
        output_dir = Path(output_dir)

        stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'speaker_counts': {},
            'duration_stats': []
        }

        for split in splits:
            print(f"\nProcessing VoxCeleb {split} split...")

            # VoxCeleb目录结构: split/speaker_id/audio_files
            split_dir = data_root / split
            if not split_dir.exists():
                print(f"Warning: Split directory {split_dir} not found")
                continue

            # 收集音频文件
            audio_files = []
            speaker_labels = []

            for speaker_dir in sorted(split_dir.iterdir()):
                if not speaker_dir.is_dir():
                    continue

                speaker_id = speaker_dir.name
                speaker_audio_files = list(speaker_dir.glob("**/*.wav"))

                audio_files.extend(speaker_audio_files)
                speaker_labels.extend([speaker_id] * len(speaker_audio_files))

                stats['speaker_counts'][speaker_id] = len(speaker_audio_files)

            stats['total_files'] += len(audio_files)

            # 创建输出目录
            split_output_dir = output_dir / split
            split_output_dir.mkdir(parents=True, exist_ok=True)

            # 并行处理音频文件
            process_func = partial(
                self.preprocessor.process_single_audio,
                output_dir=split_output_dir
            )

            if self.num_workers > 1:
                with mp.Pool(self.num_workers) as pool:
                    results = list(tqdm(
                        pool.imap(process_func, audio_files),
                        total=len(audio_files),
                        desc=f"Processing {split}"
                    ))
            else:
                results = [
                    process_func(audio_file)
                    for audio_file in tqdm(audio_files, desc=f"Processing {split}")
                ]

            # 统计处理结果
            for result in results:
                if result['success']:
                    stats['processed_files'] += 1
                    stats['duration_stats'].append(result['duration'])
                else:
                    stats['failed_files'] += 1

            # 保存文件列表和标签
            metadata = {
                'audio_files': [str(f) for f in audio_files],
                'speaker_labels': speaker_labels,
                'processing_results': results
            }

            with open(split_output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

        # 计算统计信息
        if stats['duration_stats']:
            durations = np.array(stats['duration_stats'])
            stats['duration_mean'] = float(np.mean(durations))
            stats['duration_std'] = float(np.std(durations))
            stats['duration_min'] = float(np.min(durations))
            stats['duration_max'] = float(np.max(durations))

        # 保存统计信息
        with open(output_dir / 'processing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def process_librispeech_dataset(self,
                                    data_root: str,
                                    output_dir: str,
                                    subsets: List[str] = ['train-clean-100', 'dev-clean', 'test-clean']) -> Dict:
        """
        处理LibriSpeech数据集

        Args:
            data_root: LibriSpeech数据集根目录
            output_dir: 输出目录
            subsets: 要处理的子集

        Returns:
            processing_stats: 处理统计信息
        """
        data_root = Path(data_root)
        output_dir = Path(output_dir)

        stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'speaker_counts': {},
            'duration_stats': []
        }

        for subset in subsets:
            print(f"\nProcessing LibriSpeech {subset}...")

            # LibriSpeech目录结构: subset/speaker_id/chapter_id/audio_files
            subset_dir = data_root / subset
            if not subset_dir.exists():
                print(f"Warning: Subset directory {subset_dir} not found")
                continue

            # 收集音频文件
            audio_files = []
            speaker_labels = []

            for speaker_dir in sorted(subset_dir.iterdir()):
                if not speaker_dir.is_dir():
                    continue

                speaker_id = speaker_dir.name
                speaker_audio_files = list(speaker_dir.glob("**/*.flac"))

                audio_files.extend(speaker_audio_files)
                speaker_labels.extend([speaker_id] * len(speaker_audio_files))

                if speaker_id not in stats['speaker_counts']:
                    stats['speaker_counts'][speaker_id] = 0
                stats['speaker_counts'][speaker_id] += len(speaker_audio_files)

            stats['total_files'] += len(audio_files)

            # 创建输出目录
            subset_output_dir = output_dir / subset.replace('-', '_')
            subset_output_dir.mkdir(parents=True, exist_ok=True)

            # 并行处理音频文件
            process_func = partial(
                self.preprocessor.process_single_audio,
                output_dir=subset_output_dir
            )

            if self.num_workers > 1:
                with mp.Pool(self.num_workers) as pool:
                    results = list(tqdm(
                        pool.imap(process_func, audio_files),
                        total=len(audio_files),
                        desc=f"Processing {subset}"
                    ))
            else:
                results = [
                    process_func(audio_file)
                    for audio_file in tqdm(audio_files, desc=f"Processing {subset}")
                ]

            # 统计处理结果
            for result in results:
                if result['success']:
                    stats['processed_files'] += 1
                    stats['duration_stats'].append(result['duration'])
                else:
                    stats['failed_files'] += 1

            # 保存文件列表和标签
            metadata = {
                'audio_files': [str(f) for f in audio_files],
                'speaker_labels': speaker_labels,
                'processing_results': results
            }

            with open(subset_output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

        # 计算统计信息
        if stats['duration_stats']:
            durations = np.array(stats['duration_stats'])
            stats['duration_mean'] = float(np.mean(durations))
            stats['duration_std'] = float(np.std(durations))
            stats['duration_min'] = float(np.min(durations))
            stats['duration_max'] = float(np.max(durations))

        # 保存统计信息
        with open(output_dir / 'processing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def create_speaker_mapping(self, metadata_files: List[str], output_path: str) -> Dict:
        """
        创建统一的说话人标签映射

        Args:
            metadata_files: 元数据文件路径列表
            output_path: 输出映射文件路径

        Returns:
            speaker_mapping: 说话人映射字典
        """
        all_speakers = set()

        # 收集所有说话人ID
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            all_speakers.update(metadata['speaker_labels'])

        # 创建映射
        sorted_speakers = sorted(list(all_speakers))
        speaker_to_idx = {speaker: idx for idx, speaker in enumerate(sorted_speakers)}
        idx_to_speaker = {idx: speaker for speaker, idx in speaker_to_idx.items()}

        mapping = {
            'speaker_to_idx': speaker_to_idx,
            'idx_to_speaker': idx_to_speaker,
            'num_speakers': len(sorted_speakers)
        }

        # 保存映射
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)

        print(f"Created speaker mapping for {len(sorted_speakers)} speakers")
        print(f"Mapping saved to: {output_path}")

        return mapping


def create_data_augmentation_pipeline():
    """创建数据增强流水线"""

    def time_stretch(audio: np.ndarray, stretch_factor: float = 1.1) -> np.ndarray:
        """时间拉伸"""
        return librosa.effects.time_stretch(audio, rate=stretch_factor)

    def pitch_shift(audio: np.ndarray, sr: int, n_steps: int = 2) -> np.ndarray:
        """音调变换"""
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    def add_noise(audio: np.ndarray, noise_factor: float = 0.02) -> np.ndarray:
        """添加随机噪声"""
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise

    def volume_change(audio: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """音量变化"""
        return np.clip(audio * factor, -1.0, 1.0)

    augmentation_functions = {
        'time_stretch': time_stretch,
        'pitch_shift': pitch_shift,
        'add_noise': add_noise,
        'volume_change': volume_change
    }

    return augmentation_functions


# 使用示例和测试
if __name__ == "__main__":
    # 创建预处理器
    audio_preprocessor = AudioPreprocessor(
        target_sr=16000,
        n_mels=80,
        n_mfcc=13,
        use_vad=True,
        min_duration=2.0,
        max_duration=10.0
    )

    # 创建数据集预处理器
    dataset_preprocessor = DatasetPreprocessor(
        preprocessor=audio_preprocessor,
        num_workers=4
    )

    # 示例：处理VoxCeleb数据集
    # voxceleb_stats = dataset_preprocessor.process_voxceleb_dataset(
    #     data_root="/path/to/voxceleb",
    #     output_dir="./processed_data/voxceleb",
    #     splits=['dev', 'test']
    # )

    # 示例：处理LibriSpeech数据集
    # librispeech_stats = dataset_preprocessor.process_librispeech_dataset(
    #     data_root="/path/to/librispeech",
    #     output_dir="./processed_data/librispeech",
    #     subsets=['train-clean-100', 'dev-clean']
    # )

    # 创建统一说话人映射
    # speaker_mapping = dataset_preprocessor.create_speaker_mapping(
    #     metadata_files=[
    #         "./processed_data/voxceleb/dev/metadata.json",
    #         "./processed_data/voxceleb/test/metadata.json"
    #     ],
    #     output_path="./processed_data/speaker_mapping.json"
    # )

    # 测试单个音频文件处理
    print("AudioPreprocessor ready for use!")
    print("Uncomment the example code above to process datasets.")