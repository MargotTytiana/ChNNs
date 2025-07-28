import torch
import numpy as np
import librosa
import sounddevice as sd
from scipy.signal import butter, lfilter
import queue
import threading
from .c_hilap_model import CHiLAPModel
from .chaos_features import ChaosFeatureExtractor


# 配置参数
class Config:
    # 音频处理参数
    SAMPLE_RATE = 16000  # 采样率
    BLOCK_SIZE = 1024  # 音频块大小
    HOP_LENGTH = 512  # 跳跃长度
    N_FFT = 2048  # FFT大小
    N_MELS = 80  # 梅尔滤波器数量
    WIN_LENGTH = 400  # 窗长
    PREEMPHASIS = 0.97  # 预加重系数

    # 实时处理参数
    BUFFER_SECONDS = 3  # 缓冲区大小(秒)
    MIN_SPEECH_SECONDS = 1.0  # 最小语音检测长度(秒)
    SPEECH_THRESHOLD = 0.5  # 语音活动检测阈值

    # 模型参数
    MODEL_PATH = "./checkpoints/best_model.pth"  # 模型路径
    EMBEDDING_DIM = 192  # 嵌入维度

    # 识别参数
    TOP_K = 5  # 返回前K个候选说话人
    COSINE_THRESHOLD = 0.5  # 余弦相似度阈值


# 实时音频处理类
class AudioProcessor:
    def __init__(self, config=Config):
        """初始化音频处理器"""
        self.config = config
        self.buffer = np.zeros(int(config.SAMPLE_RATE * config.BUFFER_SECONDS), dtype=np.float32)
        self.buffer_idx = 0
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.speech_detected = False

        # 创建低通滤波器
        nyquist = 0.5 * config.SAMPLE_RATE
        normal_cutoff = 8000 / nyquist  # 8kHz截止频率
        self.b, self.a = butter(4, normal_cutoff, btype='low', analog=False)

        # 初始化语音活动检测
        self.silence_threshold = self._estimate_silence_threshold()

    def _estimate_silence_threshold(self):
        """估计环境静默阈值"""
        print("正在估计环境噪声阈值，请保持安静...")
        duration = 2  # 2秒
        recording = sd.rec(int(duration * self.config.SAMPLE_RATE),
                           samplerate=self.config.SAMPLE_RATE,
                           channels=1,
                           dtype='float32')
        sd.wait()

        # 计算能量
        energy = np.sum(recording ** 2, axis=0) / len(recording)
        threshold = np.percentile(energy, 95)  # 取95%分位数作为阈值

        print(f"环境噪声阈值: {threshold:.6f}")
        return threshold * 1.5  # 设置稍高一点的阈值

    def start_recording(self):
        """开始录音"""
        self.is_recording = True
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()

    def _record_audio(self):
        """录音线程函数"""
        with sd.InputStream(samplerate=self.config.SAMPLE_RATE,
                            blocksize=self.config.BLOCK_SIZE,
                            channels=1,
                            dtype='float32',
                            callback=self._audio_callback):
            while self.is_recording:
                pass

    def _audio_callback(self, indata, frames, time, status):
        """音频回调函数"""
        if status:
            print(f"状态: {status}")

        # 应用低通滤波器
        filtered_data = lfilter(self.b, self.a, indata[:, 0])

        # 更新缓冲区
        block_size = len(filtered_data)
        if self.buffer_idx + block_size > len(self.buffer):
            # 缓冲区溢出，移动数据
            remaining = len(self.buffer) - self.buffer_idx
            self.buffer[:remaining] = self.buffer[self.buffer_idx:]
            self.buffer_idx = remaining

        self.buffer[self.buffer_idx:self.buffer_idx + block_size] = filtered_data
        self.buffer_idx += block_size

        # 检测语音活动
        energy = np.sum(filtered_data ** 2) / len(filtered_data)
        self.speech_detected = energy > self.silence_threshold

        # 如果检测到语音，将数据放入队列
        if self.speech_detected:
            self.audio_queue.put(filtered_data.copy())

    def get_speech_segment(self):
        """获取语音片段"""
        min_samples = int(self.config.MIN_SPEECH_SECONDS * self.config.SAMPLE_RATE)
        speech_buffer = []

        # 收集足够的语音数据
        while True:
            try:
                data = self.audio_queue.get(timeout=1.0)
                speech_buffer.append(data)

                # 检查是否有足够的语音
                total_samples = sum(len(d) for d in speech_buffer)
                if total_samples >= min_samples:
                    break
            except queue.Empty:
                # 如果队列为空且没有检测到语音，返回None
                if not self.speech_detected and len(speech_buffer) == 0:
                    return None

        # 合并语音数据
        speech_data = np.concatenate(speech_buffer)

        # 确保长度不超过缓冲区大小
        if len(speech_data) > len(self.buffer):
            speech_data = speech_data[-len(self.buffer):]

        return speech_data

    def preprocess(self, audio_signal):
        """
        预处理音频信号
        :param audio_signal: 音频信号
        :return: 预处理后的特征
        """
        # 预加重
        pre_emphasized = np.append(audio_signal[0], audio_signal[1:] - self.config.PREEMPHASIS * audio_signal[:-1])

        # 提取梅尔频谱
        mel_spec = librosa.feature.melspectrogram(
            y=pre_emphasized,
            sr=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH,
            n_mels=self.config.N_MELS
        )

        # 转换为对数刻度
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # 归一化
        mean = np.mean(log_mel_spec, axis=1, keepdims=True)
        std = np.std(log_mel_spec, axis=1, keepdims=True) + 1e-9
        normalized = (log_mel_spec - mean) / std

        return normalized.T  # 返回 [时间帧数, 特征维度]


# 说话人识别系统
class SpeakerRecognizer:
    def __init__(self, config=Config):
        """初始化说话人识别系统"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.model = CHiLAPModel()
        self.model.load_state_dict(torch.load(config.MODEL_PATH, map_location=self.device)["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # 初始化音频处理器
        self.audio_processor = AudioProcessor(config)

        # 注册的说话人数据库
        self.speaker_db = {}  # {speaker_id: embedding}

        # 混沌特征提取器
        self.chaos_extractor = ChaosFeatureExtractor()

    def enroll_speaker(self, speaker_id, audio_signal):
        """
        注册说话人
        :param speaker_id: 说话人ID
        :param audio_signal: 音频信号
        :return: 注册的嵌入向量
        """
        # 预处理音频
        features = self.audio_processor.preprocess(audio_signal)

        # 转换为张量
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # 获取嵌入向量
        with torch.no_grad():
            embedding, _ = self.model(features_tensor)
            embedding = embedding.squeeze(0).cpu().numpy()

        # 存储到数据库
        self.speaker_db[speaker_id] = embedding

        return embedding

    def recognize_speaker(self, audio_signal, top_k=None):
        """
        识别说话人
        :param audio_signal: 音频信号
        :param top_k: 返回前K个候选说话人
        :return: 候选说话人列表 [(speaker_id, similarity), ...]
        """
        top_k = top_k or self.config.TOP_K

        # 预处理音频
        features = self.audio_processor.preprocess(audio_signal)

        # 转换为张量
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # 获取嵌入向量
        with torch.no_grad():
            embedding, _ = self.model(features_tensor)
            embedding = embedding.squeeze(0).cpu().numpy()

        # 计算与数据库中所有说话人的相似度
        similarities = []
        for speaker_id, enrolled_embedding in self.speaker_db.items():
            similarity = np.dot(embedding, enrolled_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(enrolled_embedding))
            similarities.append((speaker_id, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def verify_speaker(self, audio_signal, speaker_id, threshold=None):
        """
        验证说话人身份
        :param audio_signal: 音频信号
        :param speaker_id: 说话人ID
        :param threshold: 相似度阈值
        :return: (是否匹配, 相似度)
        """
        threshold = threshold or self.config.COSINE_THRESHOLD

        # 检查说话人是否已注册
        if speaker_id not in self.speaker_db:
            return False, 0.0

        # 获取测试音频的嵌入向量
        features = self.audio_processor.preprocess(audio_signal)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding, _ = self.model(features_tensor)
            embedding = embedding.squeeze(0).cpu().numpy()

        # 计算相似度
        enrolled_embedding = self.speaker_db[speaker_id]
        similarity = np.dot(embedding, enrolled_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(enrolled_embedding))

        # 判断是否匹配
        is_match = similarity >= threshold

        return is_match, similarity

    def get_chaos_features(self, audio_signal):
        """
        获取音频的混沌特征
        :param audio_signal: 音频信号
        :return: 混沌特征字典
        """
        # 预处理音频
        features = self.audio_processor.preprocess(audio_signal)

        # 提取混沌特征
        mle = self.chaos_extractor.calculate_max_lyapunov_exponent(features.flatten())
        rqa_features = self.chaos_extractor.recurrence_quantification_analysis(features.flatten())

        return {
            'max_lyapunov_exponent': mle,
            'recurrence_rate': rqa_features['recurrence_rate'],
            'determinism': rqa_features['determinism'],
            'avg_diag_length': rqa_features['avg_diag_length'],
            'laminarity': rqa_features['laminarity'],
            'avg_vert_length': rqa_features['avg_vert_length']
        }

    def start_real_time_recognition(self):
        """开始实时说话人识别"""
        self.audio_processor.start_recording()
        print("开始实时说话人识别...")

        try:
            while True:
                # 获取语音片段
                speech_segment = self.audio_processor.get_speech_segment()

                if speech_segment is not None:
                    # 识别说话人
                    candidates = self.recognize_speaker(speech_segment)

                    # 显示结果
                    print("\n识别结果:")
                    for i, (speaker_id, similarity) in enumerate(candidates):
                        print(f"{i + 1}. {speaker_id}: 相似度 {similarity:.4f}")

                    # 判断是否匹配
                    if candidates and candidates[0][1] >= self.config.COSINE_THRESHOLD:
                        print(f"识别为: {candidates[0][0]}")
                    else:
                        print("未识别到已知说话人")
                else:
                    print("未检测到语音")
        except KeyboardInterrupt:
            print("\n停止实时识别")
            self.audio_processor.stop_recording()


# 主函数示例
if __name__ == "__main__":
    # 创建说话人识别系统
    recognizer = SpeakerRecognizer()

    # 示例: 注册几个说话人
    print("请录制第一个说话人的语音...")
    duration = 5  # 录制5秒
    speaker1_audio = sd.rec(int(duration * Config.SAMPLE_RATE),
                            samplerate=Config.SAMPLE_RATE,
                            channels=1,
                            dtype='float32')
    sd.wait()
    recognizer.enroll_speaker("speaker1", speaker1_audio[:, 0])

    print("请录制第二个说话人的语音...")
    speaker2_audio = sd.rec(int(duration * Config.SAMPLE_RATE),
                            samplerate=Config.SAMPLE_RATE,
                            channels=1,
                            dtype='float32')
    sd.wait()
    recognizer.enroll_speaker("speaker2", speaker2_audio[:, 0])

    # 示例: 识别说话人
    print("请录制要识别的说话人的语音...")
    test_audio = sd.rec(int(duration * Config.SAMPLE_RATE),
                        samplerate=Config.SAMPLE_RATE,
                        channels=1,
                        dtype='float32')
    sd.wait()

    candidates = recognizer.recognize_speaker(test_audio[:, 0])
    print("\n识别结果:")
    for i, (speaker_id, similarity) in enumerate(candidates):
        print(f"{i + 1}. {speaker_id}: 相似度 {similarity:.4f}")

    # 示例: 获取混沌特征
    chaos_features = recognizer.get_chaos_features(test_audio[:, 0])
    print("\n混沌特征:")
    for key, value in chaos_features.items():
        print(f"{key}: {value:.6f}")

    # 示例: 实时识别
    print("\n开始实时识别 (按Ctrl+C停止)...")
    recognizer.start_real_time_recognition()