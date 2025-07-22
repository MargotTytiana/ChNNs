import librosa
import torch
import numpy as np

"""
豆包 version 的特征处理：

1. __init__方法：初始化特征提取器的参数，包括采样率、MFCC 特征数量、梅尔滤波器组数量、FFT 窗口大小和帧移。

2. extract_mfcc方法：使用librosa.feature.mfcc函数提取 MFCC 特征，并将其转换为 PyTorch 张量。

3. extract_fbank方法：首先使用librosa.feature.melspectrogram函数提取梅尔频谱，然后将其转换为对数刻度，最后将其转换为 PyTorch 张量。

"""


# 定义特征提取类
class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=13, n_mels=40, n_fft=512, hop_length=160):
        """
        初始化特征提取器
        :param sample_rate: 音频采样率
        :param n_mfcc: MFCC特征的数量
        :param n_mels: 梅尔滤波器组的数量
        :param n_fft: FFT窗口大小
        :param hop_length: 帧移
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_mfcc(self, audio):
        """
        提取MFCC特征
        :param audio: 音频信号
        :return: MFCC特征
        """
        # 使用librosa库提取MFCC特征
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
                                    hop_length=self.hop_length)
        # 将特征转换为PyTorch张量
        mfcc = torch.FloatTensor(mfcc)
        return mfcc

    def extract_fbank(self, audio):
        """
        提取FBANK特征
        :param audio: 音频信号
        :return: FBANK特征
        """
        # 使用librosa库提取梅尔频谱
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_fft=self.n_fft,
                                                  hop_length=self.hop_length, n_mels=self.n_mels)
        # 将梅尔频谱转换为对数刻度
        fbank = librosa.power_to_db(mel_spec, ref=np.max)
        # 将特征转换为PyTorch张量
        fbank = torch.FloatTensor(fbank)
        return fbank


# 修改LibriSpeechDataset类，添加特征提取功能
from ChNNs.PreProcessData.LibriSpeechDataLoader import LibriSpeechDataset


class FeatureExtractedLibriSpeechDataset(LibriSpeechDataset):
    def __init__(self, metadata, subset=None, chunk_size=48000, augment=False, background_paths=None,
                 extract_mfcc=True, extract_fbank=True):
        """
        初始化数据集类，添加特征提取选项
        :param metadata: 合并后的元数据DataFrame
        :param subset: 可选，指定子集（如'train-clean-100'）；None表示使用所有
        :param chunk_size: 裁剪长度(样本数)
        :param augment: 是否启用数据增强
        :param background_paths: 背景噪声路径列表
        :param extract_mfcc: 是否提取MFCC特征
        :param extract_fbank: 是否提取FBANK特征
        """
        super().__init__(metadata, subset, chunk_size, augment, background_paths)
        self.extract_mfcc = extract_mfcc
        self.extract_fbank = extract_fbank
        self.feature_extractor = FeatureExtractor()

    def __getitem__(self, idx):
        # 调用父类的__getitem__方法获取音频和其他信息
        sample = super().__getitem__(idx)
        audio = sample['audio'].numpy()

        if self.extract_mfcc:
            # 提取MFCC特征
            mfcc = self.feature_extractor.extract_mfcc(audio)
            sample['mfcc'] = mfcc

        if self.extract_fbank:
            # 提取FBANK特征
            fbank = self.feature_extractor.extract_fbank(audio)
            sample['fbank'] = fbank

        return sample


# 使用示例
if __name__ == "__main__":
    import pandas as pd

    # 加载元数据
    metadata_path = r"P:\PycharmProjects\pythonProject1\processed_data\metadata_all.csv"
    metadata = pd.read_csv(metadata_path)

    # 创建特征提取后的数据集
    train_set = FeatureExtractedLibriSpeechDataset(
        metadata,
        subset='train-clean-100',
        augment=True,
        background_paths=[
            r"P:\PycharmProjects\pythonProject1\backgroundNoise\musan\noise\free-sound",
            r"P:\PycharmProjects\pythonProject1\backgroundNoise\musan\noise\sound-bible"
        ]
    )

    # 查看样本
    if len(train_set) > 0:
        sample = train_set[0]
        print(f"说话人: {sample['speaker_id']}，数据源: {sample['source']}")
        if 'mfcc' in sample:
            print(f"MFCC特征形状: {sample['mfcc'].shape}")
        if 'fbank' in sample:
            print(f"FBANK特征形状: {sample['fbank'].shape}")
