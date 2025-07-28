import os
import random
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import soundfile as sf
from tqdm import tqdm


# 配置参数
class Config:
    LIBRISPEECH_PATH = r"P:/PycharmProjects/pythonProject1/dataset/LibriSpeech"
    CHIME6_PATH = r"P:/PycharmProjects/pythonProject1/devDataset/CHiME6_dev/CHiME6/audio/dev"
    RAVDESS_PATH = r"P:/PycharmProjects/pythonProject1/devDataset/ravdess"

    SAMPLE_RATE = 16000
    DURATION = 3
    MAX_SAMPLES = SAMPLE_RATE * DURATION

    NOISE_TYPES = ["white", "babble"]
    SNR_LEVELS = [-5, 0, 5, 10, 15, 20]

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    VALID_RATIO = 0.1


# 噪声生成与注入
class NoiseInjector:
    @staticmethod
    def generate_white_noise(length, sample_rate):
        return np.random.randn(length).astype(np.float32)

    @staticmethod
    def generate_babble_noise(length, sample_rate, num_speakers=5):
        noise = np.zeros(length, dtype=np.float32)
        for _ in range(num_speakers):
            start = random.randint(0, max(0, length - sample_rate))
            end = min(start + sample_rate, length)
            noise[start:end] += np.random.randn(end - start).astype(np.float32)
        return noise / num_speakers

    @staticmethod
    def add_noise(signal, noise_type, snr_db):
        signal_power = np.mean(signal ** 2)
        signal_db = 10 * np.log10(signal_power + 1e-10)

        if noise_type == "white":
            noise = NoiseInjector.generate_white_noise(len(signal), Config.SAMPLE_RATE)
        elif noise_type == "babble":
            noise = NoiseInjector.generate_babble_noise(len(signal), Config.SAMPLE_RATE)
        else:
            raise ValueError(f"不支持的噪声类型：{noise_type}")

        noise_power = np.mean(noise ** 2)
        noise_db = 10 * np.log10(noise_power + 1e-10)
        target_noise_db = signal_db - snr_db
        noise_scale = 10 ** ((target_noise_db - noise_db) / 20)
        noisy_signal = signal + noise * noise_scale

        max_val = np.max(np.abs(noisy_signal))
        if max_val > 0:
            noisy_signal /= max_val
        return noisy_signal


# 数据集类
class SpeakerRecognitionDataset(Dataset):
    def __init__(self, dataset_name, split="train", add_noise=False, noise_type="white", snr_db=10):
        self.dataset_name = dataset_name
        self.split = split
        self.add_noise = add_noise
        self.noise_type = noise_type
        self.snr_db = snr_db
        self.audio_paths, self.labels = self._load_dataset()
        self.speaker_to_idx = self._build_speaker_map()

    def _load_dataset(self):
        audio_paths = []
        labels = []

        if self.dataset_name == "librispeech":
            root = Config.LIBRISPEECH_PATH
            if not os.path.exists(root):
                print(f"路径 {root} 不存在")
                return [], []
            for dirpath, _, filenames in os.walk(root):
                for file in filenames:
                    if file.endswith(".flac"):
                        full_path = os.path.join(dirpath, file)
                        speaker_id = os.path.normpath(full_path).split(os.sep)[-3]
                        audio_paths.append(full_path)
                        labels.append(speaker_id)

        elif self.dataset_name == "chime6":
            root = Config.CHIME6_PATH
            if not os.path.exists(root):
                print(f"路径 {root} 不存在")
                return [], []
            for speaker_id in os.listdir(root):
                speaker_dir = os.path.join(root, speaker_id)
                if not os.path.isdir(speaker_dir):
                    continue
                for file in os.listdir(speaker_dir):
                    if file.endswith(".wav"):
                        audio_paths.append(os.path.join(speaker_dir, file))
                        labels.append(speaker_id)

        elif self.dataset_name == "ravdess":
            root = Config.RAVDESS_PATH
            if not os.path.exists(root):
                print(f"路径 {root} 不存在")
                return [], []
            for file in os.listdir(root):
                if file.endswith(".wav"):
                    full_path = os.path.join(root, file)
                    speaker_id = file.split("_")[6]  # RAVDESS 特定格式
                    audio_paths.append(full_path)
                    labels.append(speaker_id)

        else:
            raise ValueError(f"不支持的数据集：{self.dataset_name}")

        print(f"加载到 {len(audio_paths)} 个音频文件。")

        if self.split != "test":
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                audio_paths, labels, test_size=Config.VALID_RATIO, random_state=42
            )
            if self.split == "train":
                return train_paths, train_labels
            else:
                return val_paths, val_labels

        return audio_paths, labels

    def _build_speaker_map(self):
        unique_speakers = sorted(list(set(self.labels)))
        return {speaker: idx for idx, speaker in enumerate(unique_speakers)}

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        speaker_id = self.labels[idx]
        label = self.speaker_to_idx[speaker_id]

        signal, sr = librosa.load(path, sr=Config.SAMPLE_RATE)

        if len(signal) > Config.MAX_SAMPLES:
            signal = signal[:Config.MAX_SAMPLES]
        else:
            signal = np.pad(signal, (0, Config.MAX_SAMPLES - len(signal)), mode="constant")

        signal = signal / (np.max(np.abs(signal)) + 1e-10)

        if self.add_noise:
            signal = NoiseInjector.add_noise(signal, self.noise_type, self.snr_db)

        return torch.FloatTensor(signal), label


# 数据加载器构造函数
def get_dataloaders(dataset_name, batch_size=Config.BATCH_SIZE):
    train_dataset = SpeakerRecognitionDataset(dataset_name, split="train")
    val_dataset = SpeakerRecognitionDataset(dataset_name, split="val")
    test_dataset = SpeakerRecognitionDataset(dataset_name, split="test")

    noisy_test_dataset = SpeakerRecognitionDataset(
        dataset_name, split="test", add_noise=True, noise_type="white", snr_db=5
    )

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=Config.NUM_WORKERS),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS),
        "noisy_test": DataLoader(noisy_test_dataset, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS),
    }


# 测试加载
if __name__ == "__main__":
    dataloaders = get_dataloaders("librispeech")
    print(f"训练集批次：{len(dataloaders['train'])}，说话人数量：{len(dataloaders['train'].dataset.speaker_to_idx)}")
    x, y = next(iter(dataloaders["train"]))
    print(f"音频形状：{x.shape}，标签形状：{y.shape}")
