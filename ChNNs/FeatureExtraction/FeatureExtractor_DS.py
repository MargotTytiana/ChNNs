import librosa
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt


class SpeakerFeatureExtractor:
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=40, n_mfcc=13,
                 win_length=0.025, hop_length=0.01, pre_emphasis=0.97):
        """
        说话人特征提取器

        参数:
        sample_rate: 采样率 (Hz)
        n_fft: FFT窗口大小
        n_mels: 梅尔滤波器数量
        n_mfcc: MFCC系数数量
        win_length: 窗口长度 (秒)
        hop_length: 帧移 (秒)
        pre_emphasis: 预加重系数
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.win_length = int(win_length * sample_rate)  # 转换为样本数
        self.hop_length = int(hop_length * sample_rate)  # 转换为样本数
        self.pre_emphasis = pre_emphasis

        # 创建梅尔滤波器组
        self.mel_filters = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )

    def pre_emphasize(self, audio):
        """预加重处理：增强高频信息，平衡频谱"""
        return np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])

    def framing(self, audio):
        """分帧：将音频信号分成短时帧"""
        frames = []
        n_frames = 1 + (len(audio) - self.win_length) // self.hop_length

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.win_length
            frame = audio[start:end]

            # 应用汉明窗减少频谱泄漏
            frame = frame * np.hamming(self.win_length)
            frames.append(frame)

        return np.array(frames)

    def compute_power_spectrum(self, frames):
        """计算功率谱：获取每帧的频率能量分布"""
        mag_frames = np.absolute(np.fft.rfft(frames, self.n_fft))
        power_frames = (1.0 / self.n_fft) * (mag_frames ** 2)
        return power_frames

    def apply_mel_filterbank(self, power_spectrum):
        """应用梅尔滤波器组：模拟人耳听觉特性"""
        mel_energy = np.dot(power_spectrum, self.mel_filters.T)

        # 避免log(0)错误
        mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
        return np.log(mel_energy)

    def compute_mfcc(self, mel_energy):
        """计算MFCC：提取倒谱系数"""
        # 使用DCT获得倒谱系数
        mfcc = dct(mel_energy, axis=1, norm='ortho')[:, :self.n_mfcc]

        # 提升高频MFCC系数（倒谱提升）
        n_coeff = mfcc.shape[1]
        n = np.arange(n_coeff)
        cep_lifter = 1 + (22 / 2) * np.sin(np.pi * n / 22)
        return mfcc * cep_lifter

    def compute_delta(self, features, N=2):
        """计算动态特征：捕捉特征随时间的变化"""
        deltas = np.zeros_like(features)
        padding = np.zeros((N, features.shape[1]))
        padded = np.vstack((padding, features, padding))

        for t in range(features.shape[0]):
            # 使用回归方法计算一阶导数
            numerator = 0
            for n in range(1, N + 1):
                numerator += n * (padded[t + N + n] - padded[t + N - n])
            deltas[t] = numerator / (2 * sum([n ** 2 for n in range(1, N + 1)]))

        return deltas

    def extract_features(self, audio):
        """
        完整的特征提取流程
        返回: (静态MFCC, Delta, Delta-Delta) 的拼接特征
        """
        # 1. 预加重
        emphasized = self.pre_emphasize(audio)

        # 2. 分帧加窗
        frames = self.framing(emphasized)

        # 3. 计算功率谱
        power_spectrum = self.compute_power_spectrum(frames)

        # 4. 应用梅尔滤波器组
        mel_energy = self.apply_mel_filterbank(power_spectrum)

        # 5. 计算静态MFCC
        mfcc = self.compute_mfcc(mel_energy)

        # 6. 计算动态特征
        delta = self.compute_delta(mfcc)
        delta_delta = self.compute_delta(delta)

        # 7. 特征归一化 (按帧归一化)
        normalized_mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
        normalized_delta = (delta - np.mean(delta, axis=0)) / (np.std(delta, axis=0) + 1e-8)
        normalized_delta_delta = (delta_delta - np.mean(delta_delta, axis=0)) / (np.std(delta_delta, axis=0) + 1e-8)

        # 8. 特征拼接
        features = np.hstack((normalized_mfcc, normalized_delta, normalized_delta_delta))

        return features

    def visualize_features(self, audio, features, title="MFCC特征"):
        """可视化特征提取过程"""
        plt.figure(figsize=(15, 10))

        # 原始音频波形
        plt.subplot(3, 1, 1)
        plt.plot(np.linspace(0, len(audio) / self.sample_rate, len(audio)), audio)
        plt.title("原始音频波形")
        plt.xlabel("时间 (秒)")
        plt.ylabel("振幅")

        # 频谱图
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=self.sample_rate, x_axis='time', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title("频谱图")

        # MFCC特征
        plt.subplot(3, 1, 3)
        librosa.display.specshow(features.T, sr=self.sample_rate, x_axis='time')
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 加载示例音频
    audio_path = "path/to/your/audio.wav"  # 替换为实际路径
    audio, sr = librosa.load(audio_path, sr=16000)

    # 初始化特征提取器
    extractor = SpeakerFeatureExtractor(sample_rate=sr)

    # 提取特征
    features = extractor.extract_features(audio)

    print(f"提取的特征形状: {features.shape} (帧数 x 特征维度)")

    # 可视化特征
    extractor.visualize_features(audio, features[:, :extractor.n_mfcc], "静态MFCC特征")