import os
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union, Callable
import random
from pathlib import Path
import logging
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    音频预处理器，提供基本的音频处理功能。

    Args:
        sample_rate: 目标采样率
        mono: 是否转换为单声道
        normalize: 是否归一化音频
        trim_silence: 是否去除静音
        silence_threshold: 静音检测阈值（dB）
        silence_padding: 静音片段前后保留的填充长度（秒）
        max_duration: 最大音频时长（秒），超过此长度的音频将被截断
    """

    def __init__(
            self,
            sample_rate: int = 16000,
            mono: bool = True,
            normalize: bool = True,
            trim_silence: bool = True,
            silence_threshold: float = -60.0,
            silence_padding: float = 0.1,
            max_duration: Optional[float] = None
    ):
        self.sample_rate = sample_rate
        self.mono = mono
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.silence_threshold = silence_threshold
        self.silence_padding = silence_padding
        self.max_duration = max_duration

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件。

        Args:
            audio_path: 音频文件路径

        Returns:
            音频数据和采样率
        """
        try:
            # 使用librosa加载音频
            audio, sr = librosa.load(audio_path, sr=None, mono=self.mono)

            # 如果加载失败，尝试使用torchaudio
            if audio is None:
                waveform, sr = torchaudio.load(audio_path)
                audio = waveform.numpy()
                if self.mono and audio.shape[0] > 1:
                    audio = np.mean(audio, axis=0)
        except Exception as e:
            # 如果上述方法都失败，尝试使用scipy
            try:
                sr, audio = wavfile.read(audio_path)
                # 将int16转换为float32
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                # 确保单声道
                if self.mono and len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
            except Exception as e2:
                logger.error(f"无法加载音频文件 {audio_path}: {e2}")
                raise e2

        return audio, sr

    def process_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        处理音频数据。

        Args:
            audio: 音频数据
            sr: 采样率

        Returns:
            处理后的音频数据
        """
        # 重采样
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # 转换为单声道
        if self.mono and len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)

        # 去除静音
        if self.trim_silence:
            non_silent_intervals = librosa.effects.split(
                audio,
                top_db=-self.silence_threshold,
                frame_length=2048,
                hop_length=512
            )

            if len(non_silent_intervals) > 0:
                # 添加填充
                pad_samples = int(self.silence_padding * self.sample_rate)

                # 合并间隔过近的非静音片段
                merged_intervals = []
                current_start, current_end = non_silent_intervals[0]

                for start, end in non_silent_intervals[1:]:
                    if start - current_end <= 2 * pad_samples:
                        # 如果间隔很小，合并片段
                        current_end = end
                    else:
                        # 否则，保存当前片段并开始新片段
                        merged_intervals.append((current_start, current_end))
                        current_start, current_end = start, end

                # 添加最后一个片段
                merged_intervals.append((current_start, current_end))

                # 应用填充并提取非静音部分
                audio_segments = []
                for start, end in merged_intervals:
                    start = max(0, start - pad_samples)
                    end = min(len(audio), end + pad_samples)
                    audio_segments.append(audio[start:end])

                # 合并所有非静音片段
                audio = np.concatenate(audio_segments)

        # 归一化
        if self.normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

        # 截断过长的音频
        if self.max_duration is not None:
            max_samples = int(self.max_duration * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

        return audio

    def __call__(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        加载并处理音频文件。

        Args:
            audio_path: 音频文件路径

        Returns:
            处理后的音频数据和采样率
        """
        audio, sr = self.load_audio(audio_path)
        processed_audio = self.process_audio(audio, sr)
        return processed_audio, self.sample_rate

    def process_directory(
            self,
            input_dir: str,
            output_dir: str,
            file_extension: str = ".wav",
            recursive: bool = True
    ) -> List[str]:
        """
        处理目录中的所有音频文件。

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            file_extension: 音频文件扩展名
            recursive: 是否递归处理子目录

        Returns:
            处理后的文件路径列表
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 查找所有音频文件
        if recursive:
            audio_files = []
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(file_extension):
                        audio_files.append(os.path.join(root, file))
        else:
            audio_files = [
                os.path.join(input_dir, file)
                for file in os.listdir(input_dir)
                if file.endswith(file_extension)
            ]

        # 处理所有音频文件
        processed_files = []
        for audio_file in tqdm(audio_files, desc="处理音频文件"):
            # 确定输出路径
            rel_path = os.path.relpath(audio_file, input_dir)
            output_path = os.path.join(output_dir, rel_path)

            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                # 处理音频
                audio, sr = self(audio_file)

                # 保存处理后的音频
                sf.write(output_path, audio, sr)

                # 添加到处理文件列表
                processed_files.append(output_path)
            except Exception as e:
                logger.error(f"处理文件 {audio_file} 时出错: {e}")

        return processed_files


class AudioAugmentor:
    """
    音频增强器，提供多种数据增强方法。

    Args:
        sample_rate: 音频采样率
        noise_dir: 噪声文件目录（可选）
        rir_dir: 房间冲激响应文件目录（可选）
    """

    def __init__(
            self,
            sample_rate: int = 16000,
            noise_dir: Optional[str] = None,
            rir_dir: Optional[str] = None
    ):
        self.sample_rate = sample_rate
        self.noise_dir = noise_dir
        self.rir_dir = rir_dir

        # 加载噪声样本
        self.noise_samples = []
        if noise_dir and os.path.exists(noise_dir):
            self._load_noise_samples()

        # 加载RIR样本
        self.rir_samples = []
        if rir_dir and os.path.exists(rir_dir):
            self._load_rir_samples()

    def _load_noise_samples(self):
        """加载噪声样本"""
        noise_files = [
            os.path.join(self.noise_dir, file)
            for file in os.listdir(self.noise_dir)
            if file.endswith((".wav", ".flac", ".mp3"))
        ]

        for noise_file in noise_files:
            try:
                noise, sr = librosa.load(noise_file, sr=self.sample_rate, mono=True)
                self.noise_samples.append(noise)
            except Exception as e:
                logger.warning(f"加载噪声文件 {noise_file} 时出错: {e}")

        logger.info(f"加载了 {len(self.noise_samples)} 个噪声样本")

    def _load_rir_samples(self):
        """加载房间冲激响应样本"""
        rir_files = [
            os.path.join(self.rir_dir, file)
            for file in os.listdir(self.rir_dir)
            if file.endswith((".wav", ".flac", ".mp3"))
        ]

        for rir_file in rir_files:
            try:
                rir, sr = librosa.load(rir_file, sr=self.sample_rate, mono=True)
                self.rir_samples.append(rir)
            except Exception as e:
                logger.warning(f"加载RIR文件 {rir_file} 时出错: {e}")

        logger.info(f"加载了 {len(self.rir_samples)} 个RIR样本")

    def add_noise(
            self,
            audio: np.ndarray,
            snr_db: float = 10.0,
            noise_type: str = "gaussian"
    ) -> np.ndarray:
        """
        添加噪声。

        Args:
            audio: 音频数据
            snr_db: 信噪比（dB）
            noise_type: 噪声类型，'gaussian'、'babble' 或 'file'

        Returns:
            添加噪声后的音频
        """
        # 计算信号功率
        signal_power = np.mean(audio ** 2)

        # 将SNR从dB转换为线性比例
        snr_linear = 10 ** (snr_db / 10)

        # 计算目标噪声功率
        noise_power_target = signal_power / snr_linear

        if noise_type == "gaussian":
            # 生成高斯白噪声
            noise = np.random.normal(0, 1, size=len(audio))
        elif noise_type == "babble":
            # 生成有色噪声（近似人声噪声）
            noise = np.random.normal(0, 1, size=len(audio))
            # 应用低通滤波器模拟人声噪声频谱
            b, a = signal.butter(4, 0.2)
            noise = signal.filtfilt(b, a, noise)
        elif noise_type == "file" and self.noise_samples:
            # 使用预加载的噪声样本
            noise_sample = random.choice(self.noise_samples)

            # 如果噪声样本太短，重复它
            if len(noise_sample) < len(audio):
                repeats = int(np.ceil(len(audio) / len(noise_sample)))
                noise_sample = np.tile(noise_sample, repeats)

            # 随机选择一段合适长度的噪声
            start = random.randint(0, len(noise_sample) - len(audio))
            noise = noise_sample[start:start + len(audio)]
        else:
            # 默认使用高斯噪声
            noise = np.random.normal(0, 1, size=len(audio))

        # 缩放噪声到目标功率
        current_noise_power = np.mean(noise ** 2)
        noise = noise * np.sqrt(noise_power_target / (current_noise_power + 1e-10))

        # 添加噪声
        noisy_audio = audio + noise

        # 归一化以防止截断
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val

        return noisy_audio

    def apply_reverb(
            self,
            audio: np.ndarray,
            reverb_level: float = 0.3
    ) -> np.ndarray:
        """
        应用混响。

        Args:
            audio: 音频数据
            reverb_level: 混响强度（0.0 到 1.0）

        Returns:
            添加混响后的音频
        """
        if not self.rir_samples:
            # 如果没有RIR样本，创建合成混响
            reverb_time = reverb_level * 0.5  # 最大500ms混响
            reverb_samples = int(reverb_time * self.sample_rate)

            # 创建简单的指数衰减
            decay = np.exp(-np.arange(reverb_samples) / (reverb_samples * 0.1))
            reverb_audio = signal.convolve(audio, decay, mode='full')[:len(audio)]

            # 混合原始信号和混响信号
            result = (1 - reverb_level) * audio + reverb_level * reverb_audio
        else:
            # 使用随机RIR样本
            rir = random.choice(self.rir_samples)

            # 归一化RIR
            rir = rir / np.max(np.abs(rir))

            # 应用RIR
            reverb_audio = signal.convolve(audio, rir, mode='full')[:len(audio)]

            # 混合原始信号和混响信号
            result = (1 - reverb_level) * audio + reverb_level * reverb_audio

        # 归一化以防止截断
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val

        return result

    def time_stretch(
            self,
            audio: np.ndarray,
            rate: float = 1.0
    ) -> np.ndarray:
        """
        时间拉伸。

        Args:
            audio: 音频数据
            rate: 拉伸因子（<1.0 加速，>1.0 减速）

        Returns:
            时间拉伸后的音频
        """
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(
            self,
            audio: np.ndarray,
            n_steps: float = 0.0
    ) -> np.ndarray:
        """
        音高偏移。

        Args:
            audio: 音频数据
            n_steps: 偏移半音数（正值升高，负值降低）

        Returns:
            音高偏移后的音频
        """
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)

    def apply_augmentation(
            self,
            audio: np.ndarray,
            augmentation_type: str,
            **kwargs
    ) -> np.ndarray:
        """
        应用指定类型的增强。

        Args:
            audio: 音频数据
            augmentation_type: 增强类型
            **kwargs: 增强参数

        Returns:
            增强后的音频
        """
        if augmentation_type == "noise":
            return self.add_noise(
                audio,
                snr_db=kwargs.get("snr_db", 10.0),
                noise_type=kwargs.get("noise_type", "gaussian")
            )
        elif augmentation_type == "reverb":
            return self.apply_reverb(
                audio,
                reverb_level=kwargs.get("reverb_level", 0.3)
            )
        elif augmentation_type == "time_stretch":
            return self.time_stretch(
                audio,
                rate=kwargs.get("rate", 1.0)
            )
        elif augmentation_type == "pitch_shift":
            return self.pitch_shift(
                audio,
                n_steps=kwargs.get("n_steps", 0.0)
            )
        else:
            logger.warning(f"未知的增强类型: {augmentation_type}")
            return audio

    def apply_augmentation_chain(
            self,
            audio: np.ndarray,
            augmentation_chain: List[Dict]
    ) -> np.ndarray:
        """
        应用增强链。

        Args:
            audio: 音频数据
            augmentation_chain: 增强链，格式为 [{'type': '增强类型', 'params': {参数字典}}, ...]

        Returns:
            增强后的音频
        """
        result = audio.copy()

        for aug in augmentation_chain:
            aug_type = aug['type']
            params = aug.get('params', {})

            result = self.apply_augmentation(result, aug_type, **params)

        return result

    def create_augmented_copy(
            self,
            audio: np.ndarray,
            augmentation_config: Dict
    ) -> np.ndarray:
        """
        创建增强副本。

        Args:
            audio: 音频数据
            augmentation_config: 增强配置

        Returns:
            增强后的音频
        """
        # 提取增强类型和参数
        aug_type = augmentation_config['type']
        params = augmentation_config.get('params', {})

        # 应用增强
        return self.apply_augmentation(audio, aug_type, **params)

    def create_augmented_dataset(
            self,
            audio_files: List[str],
            output_dir: str,
            augmentation_configs: List[Dict],
            preprocessor: Optional[AudioPreprocessor] = None
    ) -> Dict[str, List[str]]:
        """
        创建增强数据集。

        Args:
            audio_files: 音频文件路径列表
            output_dir: 输出目录
            augmentation_configs: 增强配置列表
            preprocessor: 音频预处理器（可选）

        Returns:
            增强文件映射，格式为 {原始文件: [增强文件列表]}
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化结果映射
        augmented_files = {}

        # 处理每个音频文件
        for audio_file in tqdm(audio_files, desc="创建增强数据集"):
            # 加载音频
            if preprocessor:
                audio, sr = preprocessor(audio_file)
            else:
                audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)

            # 获取文件名（不带扩展名）
            file_name = os.path.splitext(os.path.basename(audio_file))[0]

            # 初始化增强文件列表
            augmented_files[audio_file] = []

            # 应用每个增强配置
            for i, config in enumerate(augmentation_configs):
                # 创建增强副本
                augmented_audio = self.create_augmented_copy(audio, config)

                # 构建输出文件路径
                aug_type = config['type']
                output_file = os.path.join(output_dir, f"{file_name}_{aug_type}_{i}.wav")

                # 保存增强后的音频
                sf.write(output_file, augmented_audio, self.sample_rate)

                # 添加到结果映射
                augmented_files[audio_file].append(output_file)

        return augmented_files


class ChaoticFeaturePreprocessor:
    """
    混沌特征预处理器，为混沌特征提取准备数据。

    Args:
        sample_rate: 音频采样率
        frame_length: 帧长度（毫秒）
        frame_shift: 帧移（毫秒）
        embedding_dim: 相空间重构的嵌入维度
        delay: 相空间重构的时间延迟
    """

    def __init__(
            self,
            sample_rate: int = 16000,
            frame_length: int = 25,
            frame_shift: int = 10,
            embedding_dim: int = 10,
            delay: int = 5
    ):
        self.sample_rate = sample_rate
        self.frame_length = int(frame_length * sample_rate / 1000)
        self.frame_shift = int(frame_shift * sample_rate / 1000)
        self.embedding_dim = embedding_dim
        self.delay = delay

    def preemphasize(
            self,
            signal: np.ndarray,
            coef: float = 0.97
    ) -> np.ndarray:
        """
        预加重。

        Args:
            signal: 音频信号
            coef: 预加重系数

        Returns:
            预加重后的信号
        """
        return np.append(signal[0], signal[1:] - coef * signal[:-1])

    def framing(
            self,
            signal: np.ndarray
    ) -> np.ndarray:
        """
        分帧。

        Args:
            signal: 音频信号

        Returns:
            分帧后的信号，形状为 (num_frames, frame_length)
        """
        signal_length = len(signal)
        num_frames = 1 + int(np.floor((signal_length - self.frame_length) / self.frame_shift))

        frames = np.zeros((num_frames, self.frame_length))
        for i in range(num_frames):
            frames[i] = signal[i * self.frame_shift: i * self.frame_shift + self.frame_length]

        return frames

    def windowing(
            self,
            frames: np.ndarray
    ) -> np.ndarray:
        """
        加窗。

        Args:
            frames: 分帧后的信号

        Returns:
            加窗后的信号
        """
        return frames * np.hamming(frames.shape[1])

    def reconstruct_phase_space(
            self,
            signal: np.ndarray
    ) -> np.ndarray:
        """
        重构相空间。

        Args:
            signal: 输入信号

        Returns:
            重构的相空间轨迹，形状为 (n_points, embedding_dim)
        """
        n = len(signal)
        m = self.embedding_dim
        tau = self.delay

        # 计算重构空间中的点数
        n_points = n - (m - 1) * tau

        if n_points <= 0:
            # 如果信号太短，减小嵌入维度或延迟
            reduced_m = min(m, n // 2)
            reduced_tau = 1
            n_points = n - (reduced_m - 1) * reduced_tau

            logger.warning(
                f"信号太短，无法进行嵌入。将嵌入维度减小到 {reduced_m}，"
                f"延迟减小到 {reduced_tau}。"
            )

            m = reduced_m
            tau = reduced_tau

        # 初始化重构的相空间
        phase_space = np.zeros((n_points, m))

        # 填充相空间
        for i in range(n_points):
            for j in range(m):
                phase_space[i, j] = signal[i + j * tau]

        return phase_space

    def process_for_chaotic_features(
            self,
            audio: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        为混沌特征提取准备数据。

        Args:
            audio: 音频数据

        Returns:
            元组，包含 (分帧信号列表, 相空间轨迹列表)
        """
        # 预加重
        preemphasized = self.preemphasize(audio)

        # 分帧
        frames = self.framing(preemphasized)

        # 加窗
        windowed_frames = self.windowing(frames)

        # 重构相空间
        phase_spaces = []
        for frame in windowed_frames:
            phase_space = self.reconstruct_phase_space(frame)
            phase_spaces.append(phase_space)

        return windowed_frames, phase_spaces

    def __call__(
            self,
            audio_path: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        加载并处理音频文件，准备混沌特征提取。

        Args:
            audio_path: 音频文件路径

        Returns:
            元组，包含 (分帧信号列表, 相空间轨迹列表)
        """
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # 处理音频
        return self.process_for_chaotic_features(audio)


class SpeakerAugmentationPipeline:
    """
    说话人增强流水线，结合预处理和增强。

    Args:
        preprocessor: 音频预处理器
        augmentor: 音频增强器
        output_dir: 输出目录
    """

    def __init__(
            self,
            preprocessor: AudioPreprocessor,
            augmentor: AudioAugmentor,
            output_dir: str
    ):
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

    def process_speaker(
            self,
            speaker_dir: str,
            augmentation_configs: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        处理单个说话人的所有音频。

        Args:
            speaker_dir: 说话人目录
            augmentation_configs: 增强配置列表

        Returns:
            增强文件映射
        """
        # 查找所有音频文件
        audio_files = []
        for root, _, files in os.walk(speaker_dir):
            for file in files:
                if file.endswith((".wav", ".flac", ".mp3")):
                    audio_files.append(os.path.join(root, file))

        # 创建说话人输出目录
        speaker_id = os.path.basename(speaker_dir)
        speaker_output_dir = os.path.join(self.output_dir, speaker_id)
        os.makedirs(speaker_output_dir, exist_ok=True)

        # 预处理音频文件
        preprocessed_dir = os.path.join(speaker_output_dir, "preprocessed")
        preprocessed_files = self.preprocessor.process_directory(
            speaker_dir, preprocessed_dir
        )

        # 创建增强数据
        augmented_dir = os.path.join(speaker_output_dir, "augmented")
        augmented_files = self.augmentor.create_augmented_dataset(
            preprocessed_files, augmented_dir, augmentation_configs
        )

        return augmented_files

    def process_dataset(
            self,
            dataset_dir: str,
            augmentation_configs: List[Dict]
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        处理整个数据集。

        Args:
            dataset_dir: 数据集目录
            augmentation_configs: 增强配置列表

        Returns:
            嵌套字典，格式为 {说话人ID: {原始文件: [增强文件列表]}}
        """
        # 查找所有说话人目录
        speaker_dirs = [
            os.path.join(dataset_dir, d)
            for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]

        # 初始化结果字典
        results = {}

        # 处理每个说话人
        for speaker_dir in tqdm(speaker_dirs, desc="处理说话人"):
            speaker_id = os.path.basename(speaker_dir)
            results[speaker_id] = self.process_speaker(speaker_dir, augmentation_configs)

        return results

    def visualize_augmentations(
            self,
            original_audio: np.ndarray,
            augmented_audios: Dict[str, np.ndarray],
            output_path: str
    ):
        """
        可视化原始音频和增强后的音频。

        Args:
            original_audio: 原始音频数据
            augmented_audios: 增强后的音频字典，格式为 {增强类型: 音频数据}
            output_path: 输出文件路径
        """
        n_augmentations = len(augmented_audios)
        fig, axs = plt.subplots(n_augmentations + 1, 2, figsize=(15, 3 * (n_augmentations + 1)))

        # 绘制原始音频
        axs[0, 0].plot(original_audio)
        axs[0, 0].set_title("Original Audio - Waveform")
        axs[0, 0].set_xlabel("Sample")
        axs[0, 0].set_ylabel("Amplitude")

        # 计算并绘制原始音频的频谱图
        D = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
        librosa.display.specshow(D, sr=self.preprocessor.sample_rate, x_axis='time', y_axis='log', ax=axs[0, 1])
        axs[0, 1].set_title("Original Audio - Spectrogram")

        # 绘制增强后的音频
        for i, (aug_type, aug_audio) in enumerate(augmented_audios.items(), 1):
            # 波形图
            axs[i, 0].plot(aug_audio)
            axs[i, 0].set_title(f"{aug_type} - Waveform")
            axs[i, 0].set_xlabel("Sample")
            axs[i, 0].set_ylabel("Amplitude")

            # 频谱图
            D = librosa.amplitude_to_db(np.abs(librosa.stft(aug_audio)), ref=np.max)
            librosa.display.specshow(D, sr=self.preprocessor.sample_rate, x_axis='time', y_axis='log', ax=axs[i, 1])
            axs[i, 1].set_title(f"{aug_type} - Spectrogram")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)


# 示例用法
if __name__ == "__main__":
    # 创建音频预处理器
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        mono=True,
        normalize=True,
        trim_silence=True,
        silence_threshold=-60.0,
        silence_padding=0.1,
        max_duration=10.0
    )

    # 创建音频增强器
    augmentor = AudioAugmentor(
        sample_rate=16000,
        noise_dir="./noise_samples",
        rir_dir="./rir_samples"
    )

    # 定义增强配置
    augmentation_configs = [
        {"type": "noise", "params": {"snr_db": 15.0, "noise_type": "gaussian"}},
        {"type": "noise", "params": {"snr_db": 10.0, "noise_type": "babble"}},
        {"type": "reverb", "params": {"reverb_level": 0.3}},
        {"type": "time_stretch", "params": {"rate": 0.9}},
        {"type": "time_stretch", "params": {"rate": 1.1}},
        {"type": "pitch_shift", "params": {"n_steps": -2.0}},
        {"type": "pitch_shift", "params": {"n_steps": 2.0}}
    ]

    # 创建说话人增强流水线
    pipeline = SpeakerAugmentationPipeline(
        preprocessor=preprocessor,
        augmentor=augmentor,
        output_dir="./augmented_data"
    )

    # 处理数据集
    results = pipeline.process_dataset(
        dataset_dir="./librispeech/train-clean-100",
        augmentation_configs=augmentation_configs
    )

    # 打印结果
    print(f"处理了 {len(results)} 个说话人")

    # 创建混沌特征预处理器
    chaotic_preprocessor = ChaoticFeaturePreprocessor(
        sample_rate=16000,
        frame_length=25,
        frame_shift=10,
        embedding_dim=10,
        delay=5
    )

    # 处理单个音频文件，准备混沌特征提取
    audio_path = "./librispeech/train-clean-100/1272/128104/1272-128104-0000.flac"
    frames, phase_spaces = chaotic_preprocessor(audio_path)

    print(f"提取了 {len(frames)} 帧")
    print(f"相空间轨迹形状: {phase_spaces[0].shape}")