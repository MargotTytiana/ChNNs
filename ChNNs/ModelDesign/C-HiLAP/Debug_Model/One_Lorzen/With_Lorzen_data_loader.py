import os
import random
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import soundfile as sf
from tqdm import tqdm
import gc


# 配置参数
class Config:
    DEBUG_MODE = True  # 调试模式，使用小规模数据集
    DEBUG_SAMPLE_SIZE = 5000  # 调试模式下使用的样本数量

    # 更新数据集路径
    TRAIN_DIR = "P:/PycharmProjects/pythonProject1/dataset"
    DEV_DIR = "P:/PycharmProjects/pythonProject1/devDataset"
    TEST_DIR = "P:/PycharmProjects/pythonProject1/testDataset"

    # 具体数据集路径
    LIBRISPEECH_PATH = os.path.join(TRAIN_DIR, "LibriSpeech")
    CHIME6_PATH = os.path.join(DEV_DIR, "CHiME6_dev", "CHiME6", "audio", "dev")
    RAVDESS_PATH = os.path.join(DEV_DIR, "ravdess")

    # 音频处理参数
    SAMPLE_RATE = 16000
    DURATION = 1.0  # 1秒音频（16000个采样点）
    MAX_SAMPLES = int(SAMPLE_RATE * DURATION)
    MAX_SEQ_LEN = 16000  # 1秒音频长度

    # 噪声参数
    NOISE_TYPES = ["white", "babble"]
    SNR_LEVELS = [0, 5, 10]  # SNR级别

    # 数据加载器参数
    BATCH_SIZE = 8  # 减小批次大小，复杂混沌模块需要更多内存
    NUM_WORKERS = 2  # 减少工作线程数
    VALID_RATIO = 0.1

    # 混沌系统特定参数
    CHAOS_DIM = 3  # Lorenz系统维度


# 噪声生成与注入
class NoiseInjector:
    @staticmethod
    def generate_white_noise(length):
        return np.random.randn(length).astype(np.float32)

    @staticmethod
    def generate_babble_noise(length, num_speakers=3):
        noise = np.zeros(length, dtype=np.float32)
        for _ in range(num_speakers):
            start = random.randint(0, max(0, length - Config.SAMPLE_RATE))
            end = min(start + Config.SAMPLE_RATE, length)
            noise[start:end] += np.random.randn(end - start).astype(np.float32)
        return noise / num_speakers

    @staticmethod
    def add_noise(signal, noise_type="white", snr_db=10):
        if len(signal) == 0:
            return signal

        signal_power = np.mean(signal ** 2)
        if signal_power < 1e-10:
            return signal

        signal_db = 10 * np.log10(signal_power)

        if noise_type == "white":
            noise = NoiseInjector.generate_white_noise(len(signal))
        elif noise_type == "babble":
            noise = NoiseInjector.generate_babble_noise(len(signal))
        else:
            raise ValueError(f"不支持的噪声类型：{noise_type}")

        noise_power = np.mean(noise ** 2)
        if noise_power < 1e-10:
            noise_db = -100
        else:
            noise_db = 10 * np.log10(noise_power)

        target_noise_db = signal_db - snr_db
        noise_scale = 10 ** ((target_noise_db - noise_db) / 20)
        noisy_signal = signal + noise * noise_scale

        # 归一化防止溢出
        max_val = np.max(np.abs(noisy_signal))
        if max_val > 1e-5:
            noisy_signal = noisy_signal / max_val

        return noisy_signal


# 数据集类
class SpeakerRecognitionDataset(Dataset):
    def __init__(self, dataset_name, split="train", add_noise=False, noise_type="white", snr_db=10):
        self.dataset_name = dataset_name
        self.split = split
        self.add_noise = add_noise
        self.noise_type = noise_type
        self.snr_db = snr_db

        # 验证数据集路径
        self._validate_path()

        # 加载数据集
        self.audio_paths, self.labels = self._load_dataset()
        self.speaker_to_idx = self._build_speaker_map()

        # 调试模式：使用小样本
        if Config.DEBUG_MODE:
            if split == "train":
                self.audio_paths = self.audio_paths[:Config.DEBUG_SAMPLE_SIZE]
                self.labels = self.labels[:Config.DEBUG_SAMPLE_SIZE]
            elif split == "val":
                self.audio_paths = self.audio_paths[:min(Config.DEBUG_SAMPLE_SIZE // 2, len(self.audio_paths))]
                self.labels = self.labels[:min(Config.DEBUG_SAMPLE_SIZE // 2, len(self.labels))]

        # 验证数据集
        self._validate_dataset()

        print(f"最终 {split} 数据集大小: {len(self.audio_paths)} 个样本")

    def _validate_path(self):
        """验证数据集路径是否存在"""
        if self.dataset_name == "librispeech":
            path = Config.LIBRISPEECH_PATH
        elif self.dataset_name == "chime6":
            path = Config.CHIME6_PATH
        elif self.dataset_name == "ravdess":
            path = Config.RAVDESS_PATH
        else:
            return

        if not os.path.exists(path):
            print(f"警告: {self.dataset_name} 路径不存在 - {path}")
            print("请确保数据集已下载并放置在正确位置")

    def _load_dataset(self):
        audio_paths = []
        labels = []

        if self.dataset_name == "librispeech":
            root = Config.LIBRISPEECH_PATH
            if not os.path.exists(root):
                print(f"警告: LibriSpeech路径不存在 - {root}")
                return [], []

            print(f"加载LibriSpeech数据集: {root}")

            # 遍历所有子集目录
            for subset_dir in os.listdir(root):
                subset_path = os.path.join(root, subset_dir)
                if not os.path.isdir(subset_path):
                    continue

                print(f"处理子集: {subset_dir}")

                # 遍历子集中的所有说话人目录
                for speaker_dir in os.listdir(subset_path):
                    speaker_path = os.path.join(subset_path, speaker_dir)
                    if not os.path.isdir(speaker_path):
                        continue

                    # 遍历说话人目录中的所有章节
                    for chapter_dir in os.listdir(speaker_path):
                        chapter_path = os.path.join(speaker_path, chapter_dir)
                        if not os.path.isdir(chapter_path):
                            continue

                        # 遍历章节目录中的所有文件
                        for file in os.listdir(chapter_path):
                            if file.endswith(".flac"):
                                full_path = os.path.join(chapter_path, file)
                                audio_paths.append(full_path)
                                labels.append(speaker_dir)

                print(f"在 {subset_dir} 中找到 {len(audio_paths)} 个.flac文件")

            # 打印前5个文件路径作为示例
            if audio_paths:
                print("找到的音频文件示例:")
                for i in range(min(5, len(audio_paths))):
                    print(f"  {audio_paths[i]} -> 说话人: {labels[i]}")

        elif self.dataset_name == "chime6":
            root = Config.CHIME6_PATH
            if not os.path.exists(root):
                print(f"警告: CHiME-6路径不存在 - {root}")
                return [], []

            print("加载CHiME-6数据集...")
            for speaker_id in os.listdir(root):
                speaker_dir = os.path.join(root, speaker_id)
                if not os.path.isdir(speaker_dir):
                    continue

                for file in os.listdir(speaker_dir):
                    if file.endswith(".wav"):
                        full_path = os.path.join(speaker_dir, file)
                        audio_paths.append(full_path)
                        labels.append(speaker_id)

        elif self.dataset_name == "ravdess":
            root = Config.RAVDESS_PATH
            if not os.path.exists(root):
                print(f"警告: RAVDESS路径不存在 - {root}")
                return [], []

            print("加载RAVDESS数据集...")
            for file in os.listdir(root):
                if file.endswith(".wav"):
                    full_path = os.path.join(root, file)
                    speaker_id = file.split("-")[0]
                    audio_paths.append(full_path)
                    labels.append(speaker_id)

        else:
            raise ValueError(f"不支持的数据集：{self.dataset_name}")

        # 数据集分割
        if self.split != "test" and len(audio_paths) > 0:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                audio_paths, labels, test_size=Config.VALID_RATIO, random_state=42
            )
            if self.split == "train":
                return train_paths, train_labels
            else:
                return val_paths, val_labels

        return audio_paths, labels

    def _build_speaker_map(self):
        unique_speakers = sorted(set(self.labels))
        print(f"找到 {len(unique_speakers)} 个不同的说话人")
        return {speaker: idx for idx, speaker in enumerate(unique_speakers)}

    def _validate_dataset(self):
        """验证数据集完整性并移除无效文件"""
        if len(self.audio_paths) == 0:
            print(f"警告: {self.split}数据集为空，跳过验证")
            return

        print(f"验证{self.split}数据集...")
        valid_count = 0
        invalid_indices = []

        # 反向遍历，避免索引错位
        for i in range(len(self.audio_paths) - 1, -1, -1):
            path = self.audio_paths[i]
            try:
                # 检查文件是否存在
                if not os.path.exists(path):
                    raise FileNotFoundError(f"文件不存在")

                # 检查文件大小
                if os.path.getsize(path) < 1024:
                    raise ValueError("文件太小可能已损坏")

                # 尝试读取文件
                if path.endswith('.flac'):
                    signal, sr = sf.read(path)
                else:
                    signal, sr = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)

                # 检查音频长度
                if len(signal) < Config.SAMPLE_RATE // 2:
                    raise ValueError(f"音频过短 ({len(signal)}采样点)")

                # 检查信号幅度
                if np.max(np.abs(signal)) < 1e-5:
                    raise ValueError("信号幅度太小（接近静音）")

                valid_count += 1

            except Exception as e:
                invalid_indices.append(i)
                print(f"无效文件: {path} - 原因: {str(e)}")

        # 移除无效文件
        for i in invalid_indices:
            self.audio_paths.pop(i)
            self.labels.pop(i)

        print(f"有效文件: {valid_count}/{len(self.audio_paths) + len(invalid_indices)}")
        print(f"移除 {len(invalid_indices)} 个无效文件")

        # 重新构建说话人映射
        self.speaker_to_idx = self._build_speaker_map()

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        speaker_id = self.labels[idx]
        label = self.speaker_to_idx[speaker_id]

        try:
            # 优先使用soundfile加载.flac文件
            if path.endswith('.flac'):
                try:
                    signal, sr = sf.read(path)
                    if sr != Config.SAMPLE_RATE:
                        signal = librosa.resample(signal, orig_sr=sr, target_sr=Config.SAMPLE_RATE)
                except Exception as e:
                    print(f"SoundFile加载失败 {path}: {e}, 尝试Librosa")
                    signal, sr = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
            else:
                signal, sr = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"加载音频错误 {path}: {str(e)}, 使用静音替代")
            signal = np.zeros(Config.MAX_SAMPLES, dtype=np.float32)

        # 确保音频长度正确
        if len(signal) > Config.MAX_SAMPLES:
            signal = signal[:Config.MAX_SAMPLES]
        elif len(signal) < Config.MAX_SAMPLES:
            pad_len = Config.MAX_SAMPLES - len(signal)
            signal = np.pad(signal, (0, pad_len), mode='constant')

        # 归一化
        max_val = np.max(np.abs(signal))
        if max_val > 1e-5:
            signal = signal / max_val

        # 添加噪声
        if self.add_noise and self.split == "train":
            noise_type = random.choice(Config.NOISE_TYPES)
            snr_db = random.choice(Config.SNR_LEVELS)
            signal = NoiseInjector.add_noise(signal, noise_type, snr_db)

        # 为复杂混沌系统准备数据格式
        # 返回原始音频信号和标签
        return torch.FloatTensor(signal), label

    def get_chaos_features(self, idx):
        """
        为复杂混沌系统提取特征
        :param idx: 样本索引
        :return: 适合混沌系统的特征格式
        """
        signal, label = self.__getitem__(idx)

        # 将信号转换为适合混沌系统的格式
        # 这里可以根据需要添加特定的特征提取逻辑
        signal = signal.numpy()

        # 简化版本：直接返回原始信号
        # 实际应用中可能需要更复杂的特征提取
        return torch.FloatTensor(signal), label


# 数据加载器构造函数
def get_dataloaders(dataset_name="librispeech", batch_size=None):
    """创建数据加载器"""
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    print(f"创建数据加载器: 数据集={dataset_name}, 批大小={batch_size}")

    # 创建数据集
    try:
        train_dataset = SpeakerRecognitionDataset(dataset_name, split="train")
        val_dataset = SpeakerRecognitionDataset(dataset_name, split="val")
        test_dataset = SpeakerRecognitionDataset(dataset_name, split="test")

        # 如果训练集为空，尝试直接加载测试集
        if len(train_dataset) == 0 and len(test_dataset) > 0:
            print("警告: 训练集为空，使用测试集作为训练集")
            train_dataset = test_dataset

        # 如果所有数据集都为空，抛出错误
        if len(train_dataset) == 0 and len(val_dataset) == 0 and len(test_dataset) == 0:
            raise RuntimeError("所有数据集均为空，请检查数据集路径和加载逻辑")

        # 创建带噪声的测试集
        noisy_test_dataset = SpeakerRecognitionDataset(
            dataset_name, split="test", add_noise=True, noise_type="white", snr_db=5
        )

        # 打印数据集统计信息
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
        print(f"测试集: {len(test_dataset)} 样本")
        print(f"带噪声测试集: {len(noisy_test_dataset)} 样本")
        print(f"总说话人数: {len(train_dataset.speaker_to_idx)}")

        # 创建数据加载器
        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=Config.NUM_WORKERS, pin_memory=True),
            "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=Config.NUM_WORKERS, pin_memory=True),
            "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=Config.NUM_WORKERS, pin_memory=True),
            "noisy_test": DataLoader(noisy_test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=Config.NUM_WORKERS, pin_memory=True),
        }

        return dataloaders

    except Exception as e:
        print(f"创建数据集时出错: {e}")
        return None


# 专门用于混沌系统的数据加载器
def get_chaos_dataloaders(dataset_name="librispeech", batch_size=None):
    """
    创建专门用于复杂混沌系统的数据加载器
    返回适合混沌系统输入格式的数据
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    print(f"创建混沌系统数据加载器: 数据集={dataset_name}, 批大小={batch_size}")

    # 创建数据集
    try:
        chaos_dataset = SpeakerRecognitionDataset(dataset_name, split="train")

        # 创建自定义数据集类，专门用于混沌系统
        class ChaosDataset(Dataset):
            def __init__(self, base_dataset):
                self.base_dataset = base_dataset

            def __len__(self):
                return len(self.base_dataset)

            def __getitem__(self, idx):
                # 使用专门为混沌系统准备的数据格式
                return self.base_dataset.get_chaos_features(idx)

        chaos_dataset = ChaosDataset(chaos_dataset)

        # 创建数据加载器
        chaos_dataloader = DataLoader(
            chaos_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )

        return chaos_dataloader

    except Exception as e:
        print(f"创建混沌系统数据加载器时出错: {e}")
        return None


# 测试加载
if __name__ == "__main__":
    # 测试数据加载器
    print("测试数据加载器...")
    try:
        dataloaders = get_dataloaders("librispeech")

        if dataloaders is None:
            print("无法创建数据加载器")
            exit(1)

        # 获取一个批次的数据
        if "train" in dataloaders and len(dataloaders["train"].dataset) > 0:
            train_loader = dataloaders["train"]
            x, y = next(iter(train_loader))

            print(f"音频数据形状: {x.shape} (应为 [batch, {Config.MAX_SAMPLES}])")
            print(f"标签数据形状: {y.shape}")
            print(f"音频范围: {torch.min(x)} 到 {torch.max(x)}")
            print(f"标签示例: {y[:5]}")

            # 检查数据集统计信息
            print("\n数据集统计信息:")
            print(f"训练集说话人数: {len(train_loader.dataset.speaker_to_idx)}")
            if "val" in dataloaders:
                print(f"验证集说话人数: {len(dataloaders['val'].dataset.speaker_to_idx)}")
            if "test" in dataloaders:
                print(f"测试集说话人数: {len(dataloaders['test'].dataset.speaker_to_idx)}")

            # 测试单个样本
            print("\n测试单个样本加载...")
            sample_idx = 0
            signal, label = train_loader.dataset[sample_idx]
            print(f"样本{sample_idx} - 标签: {label}")
            print(f"信号长度: {len(signal)} 采样点")
            print(f"信号范围: {signal.min()} 到 {signal.max()}")

        else:
            print("训练集为空，无法获取样本")
    except Exception as e:
        print(f"创建数据加载器时出错: {e}")
        import traceback

        traceback.print_exc()
