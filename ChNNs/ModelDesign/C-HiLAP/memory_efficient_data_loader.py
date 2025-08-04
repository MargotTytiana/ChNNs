import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import warnings

# 尝试多种音频加载方式
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("警告: librosa 不可用")

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("警告: torchaudio 不可用")

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
    # 检查是否有SoundFileRuntimeError属性
    if not hasattr(sf, 'SoundFileRuntimeError'):
        print("警告: soundfile版本不兼容，缺少SoundFileRuntimeError属性")
        # 创建一个临时的异常类
        sf.SoundFileRuntimeError = RuntimeError
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("警告: soundfile 不可用")


class RobustSpeakerRecognitionDataset(Dataset):
    """
    鲁棒的说话人识别数据集，能处理各种音频加载错误
    """

    def __init__(self, dataset_name, split="train", sr=16000, max_length=48000):
        """
        初始化数据集
        :param dataset_name: 数据集名称
        :param split: 数据分割 ('train', 'val', 'test')
        :param sr: 采样率
        :param max_length: 最大音频长度
        """
        self.dataset_name = dataset_name
        self.split = split
        self.sr = sr
        self.max_length = max_length

        # 错误统计
        self.corrupted_files = []
        self.loading_errors = {}

        # 加载数据列表
        self.audio_files, self.labels, self.speaker_to_id = self._load_dataset()
        print(f"加载到 {len(self.audio_files)} 个音频文件。")

        # 验证音频加载能力
        self._check_audio_loading_capability()

    def _check_audio_loading_capability(self):
        """检查音频加载能力并选择最佳方法"""
        methods = []
        if LIBROSA_AVAILABLE:
            methods.append("librosa")
        if TORCHAUDIO_AVAILABLE:
            methods.append("torchaudio")
        if SOUNDFILE_AVAILABLE:
            methods.append("soundfile")

        print(f"可用的音频加载方法: {methods}")

        # 测试加载第一个文件
        if len(self.audio_files) > 0:
            test_file = self.audio_files[0]
            for method in methods:
                try:
                    audio = self._load_audio_with_method(test_file, method)
                    if audio is not None:
                        print(f"选择音频加载方法: {method}")
                        self.preferred_method = method
                        return
                except Exception as e:
                    print(f"方法 {method} 测试失败: {e}")

        print("警告: 没有可用的音频加载方法！")
        self.preferred_method = None

    def _load_audio_with_method(self, file_path, method):
        """使用指定方法加载音频"""
        try:
            if method == "librosa" and LIBROSA_AVAILABLE:
                audio, _ = librosa.load(file_path, sr=self.sr)
                return audio

            elif method == "torchaudio" and TORCHAUDIO_AVAILABLE:
                waveform, original_sr = torchaudio.load(file_path)
                # 转换为单声道
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                # 重采样
                if original_sr != self.sr:
                    resampler = torchaudio.transforms.Resample(original_sr, self.sr)
                    waveform = resampler(waveform)
                return waveform.squeeze().numpy()

            elif method == "soundfile" and SOUNDFILE_AVAILABLE:
                audio, original_sr = sf.read(file_path)
                if len(audio.shape) > 1:  # 多声道转单声道
                    audio = np.mean(audio, axis=1)
                # 简单重采样（实际应用中建议使用专业的重采样库）
                if original_sr != self.sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * self.sr / original_sr))
                return audio

        except Exception as e:
            raise e

        return None

    def _load_audio_robust(self, file_path, max_retries=3):
        """
        鲁棒的音频加载，尝试多种方法
        """
        methods_to_try = []
        if hasattr(self, 'preferred_method') and self.preferred_method:
            methods_to_try.append(self.preferred_method)

        # 添加其他可用方法作为备选
        for method in ["librosa", "torchaudio", "soundfile"]:
            if method != getattr(self, 'preferred_method', None):
                if (method == "librosa" and LIBROSA_AVAILABLE) or \
                        (method == "torchaudio" and TORCHAUDIO_AVAILABLE) or \
                        (method == "soundfile" and SOUNDFILE_AVAILABLE):
                    methods_to_try.append(method)

        last_error = None
        for retry in range(max_retries):
            for method in methods_to_try:
                try:
                    audio = self._load_audio_with_method(file_path, method)
                    if audio is not None:
                        return self._process_audio(audio)
                except Exception as e:
                    last_error = e
                    # 记录错误
                    error_key = f"{method}_{type(e).__name__}"
                    self.loading_errors[error_key] = self.loading_errors.get(error_key, 0) + 1

                    # 如果是soundfile的SoundFileRuntimeError，跳过这个方法
                    if "soundfile" in str(e).lower() or "SoundFileRuntimeError" in str(e):
                        continue

        # 所有方法都失败了
        self.corrupted_files.append(file_path)
        print(f"无法加载音频文件: {file_path}, 最后错误: {last_error}")
        return None

    def _process_audio(self, audio):
        """处理音频：填充或裁剪到固定长度"""
        if len(audio) > self.max_length:
            # 随机裁剪
            start = np.random.randint(0, len(audio) - self.max_length + 1)
            audio = audio[start:start + self.max_length]
        else:
            # 填充到固定长度
            padding = self.max_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)

        return audio.astype(np.float32)

    def _load_dataset(self):
        """
        加载数据集 - 这里需要根据你的实际数据集结构进行修改
        """
        # 示例：假设你有一个包含音频文件路径和标签的列表
        # 这里需要根据你的实际数据集进行修改

        # 临时示例数据 - 请替换为你的实际数据加载逻辑
        audio_files = []
        labels = []
        speaker_to_id = {}

        # 这里应该是你的实际数据加载逻辑
        # 例如：从LibriSpeech或其他数据集加载

        # 示例：假设你有一个数据目录
        data_dir = f"./data/{self.dataset_name}/{self.split}"
        if os.path.exists(data_dir):
            speaker_id = 0
            for speaker_dir in os.listdir(data_dir):
                speaker_path = os.path.join(data_dir, speaker_dir)
                if os.path.isdir(speaker_path):
                    if speaker_dir not in speaker_to_id:
                        speaker_to_id[speaker_dir] = speaker_id
                        speaker_id += 1

                    for audio_file in os.listdir(speaker_path):
                        if audio_file.endswith(('.wav', '.flac', '.mp3')):
                            audio_path = os.path.join(speaker_path, audio_file)
                            audio_files.append(audio_path)
                            labels.append(speaker_to_id[speaker_dir])

        # 如果没有找到数据，创建一些虚拟数据用于测试
        if len(audio_files) == 0:
            print(f"警告: 在 {data_dir} 中没有找到音频文件，创建虚拟数据")
            # 创建虚拟数据
            num_samples = 1000 if self.split == "train" else 200
            for i in range(num_samples):
                # 虚拟音频文件路径
                audio_files.append(f"dummy_audio_{i}.wav")
                labels.append(i % 100)  # 100个说话人

            # 虚拟说话人映射
            for i in range(100):
                speaker_to_id[f"speaker_{i}"] = i

        return audio_files, labels, speaker_to_id

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        获取数据项，包含错误处理
        """
        max_retries = 3
        for retry in range(max_retries):
            try:
                file_path = self.audio_files[idx]
                label = self.labels[idx]

                # 检查是否是虚拟数据
                if file_path.startswith("dummy_audio_"):
                    # 生成虚拟音频数据
                    audio = np.random.randn(self.max_length).astype(np.float32)
                else:
                    # 加载真实音频
                    audio = self._load_audio_robust(file_path)

                if audio is not None:
                    return torch.FloatTensor(audio), torch.LongTensor([label]).squeeze()
                else:
                    # 如果加载失败，尝试下一个文件
                    idx = (idx + 1) % len(self.audio_files)
                    continue

            except Exception as e:
                print(f"数据加载错误 (索引 {idx}, 重试 {retry + 1}): {e}")
                if retry == max_retries - 1:
                    # 最后一次重试失败，返回虚拟数据
                    audio = np.random.randn(self.max_length).astype(np.float32)
                    label = self.labels[idx] if idx < len(self.labels) else 0
                    return torch.FloatTensor(audio), torch.LongTensor([label]).squeeze()
                continue

    def get_error_statistics(self):
        """获取错误统计信息"""
        return {
            'corrupted_files_count': len(self.corrupted_files),
            'corrupted_files': self.corrupted_files[:10],  # 只显示前10个
            'loading_errors': self.loading_errors
        }


def create_robust_dataloaders(dataset_name, batch_size=4, num_workers=0):
    """
    创建鲁棒的数据加载器
    """
    try:
        train_dataset = RobustSpeakerRecognitionDataset(dataset_name, split="train")
        val_dataset = RobustSpeakerRecognitionDataset(dataset_name, split="val")
        test_dataset = RobustSpeakerRecognitionDataset(dataset_name, split="test")

        # 打印错误统计
        print("训练集错误统计:", train_dataset.get_error_statistics())
        print("验证集错误统计:", val_dataset.get_error_statistics())
        print("测试集错误统计:", test_dataset.get_error_statistics())

        return {
            "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=False, drop_last=True),
            "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False, drop_last=False),
            "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=False, drop_last=False),
        }
    except Exception as e:
        print(f"数据加载器创建错误: {e}")
        return None


# 测试代码
if __name__ == "__main__":
    print("测试鲁棒数据加载器...")

    # 创建数据加载器
    dataloaders = create_robust_dataloaders("librispeech", batch_size=4)

    if dataloaders:
        print("数据加载器创建成功！")

        # 测试训练数据加载器
        train_loader = dataloaders["train"]
        print(f"训练集批次数: {len(train_loader)}")

        # 测试加载几个批次
        for i, (audio, labels) in enumerate(train_loader):
            print(f"批次 {i}: 音频形状 {audio.shape}, 标签形状 {labels.shape}")
            if i >= 2:  # 只测试前3个批次
                break

        print("数据加载测试完成！")
    else:
        print("数据加载器创建失败！")