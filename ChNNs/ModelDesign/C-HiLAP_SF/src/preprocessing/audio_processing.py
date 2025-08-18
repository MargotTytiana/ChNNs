import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, List, Union, Optional
import random


class AudioProcessor:
    """
    Handles audio processing, feature extraction, and augmentation
    for speaker recognition tasks.
    """

    def __init__(self,
                 target_sr: int = 16000,
                 n_mfcc: int = 40,
                 n_fbank: int = 80):
        """
        Initialize with processing parameters

        Args:
            target_sr: Target sampling rate for resampling
            n_mfcc: Number of MFCC coefficients to extract
            n_fbank: Number of filterbanks for Fbank features
        """
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.n_fbank = n_fbank

    def load_audio(self,
                   audio_path: str,
                   sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file from disk with resampling support

        Args:
            audio_path: Path to audio file
            sr: Target sample rate (None uses original rate)

        Returns:
            Tuple of (audio signal, sample rate)
        """
        signal, sr_orig = sf.read(audio_path)
        if sr is not None and sr != sr_orig:
            signal = librosa.resample(signal, orig_sr=sr_orig, target_sr=sr)
        return signal, self.target_sr if sr is not None else sr_orig

    def preemphasize(self,
                     signal: np.ndarray,
                     preemphasis: float = 0.97) -> np.ndarray:
        """
        Apply preemphasis filter to enhance high frequencies

        Args:
            signal: Raw audio signal
            preemphasis: Preemphasis coefficient

        Returns:
            Preemphasized signal
        """
        return np.append(signal[0], signal[1:] - preemphasis * signal[:-1])

    def extract_mfcc(self,
                     signal: np.ndarray,
                     sr: int) -> np.ndarray:
        """
        Extract MFCC features from audio signal

        Args:
            signal: Audio signal array
            sr: Sample rate of signal

        Returns:
            MFCC features (n_mfcc x time)
        """
        # Preemphasis to enhance high frequencies
        signal = self.preemphasize(signal)

        # Compute MFCCs
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        )
        return mfcc

    def extract_fbank(self,
                      signal: np.ndarray,
                      sr: int) -> np.ndarray:
        """
        Extract filterbank energies from audio signal

        Args:
            signal: Audio signal array
            sr: Sample rate of signal

        Returns:
            Filterbank energies (n_fbank x time)
        """
        # Preemphasis
        signal = self.preemphasize(signal)

        # Compute Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=self.n_fbank
        )

        # Convert to log scale
        log_fbank = librosa.power_to_db(mel_spec, ref=np.max)
        return log_fbank

    def compute_mle(self,
                    signal: np.ndarray,
                    emb_dim: int = 10,
                    delay: int = 2,
                    min_lyap: float = 0.1) -> float:
        """
        Compute maximum Lyapunov exponent as chaos feature

        Args:
            signal: Audio signal (vocal segment)
            emb_dim: Embedding dimension for phase reconstruction
            delay: Time delay for embedding
            min_lyap: Minimum valid Lyapunov value

        Returns:
            Estimated maximum Lyapunov exponent
        """
        # Normalize signal to unit variance
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        # Phase space reconstruction
        m = emb_dim
        tau = delay
        n = len(signal)
        embeddings = np.zeros((m, n - (m - 1) * tau))

        for i in range(n - (m - 1) * tau):
            for j in range(m):
                embeddings[j, i] = signal[i + j * tau]

        # Compute pairwise distances
        distances = np.zeros((n - (m - 1) * tau,))
        for i in range(n - (m - 1) * tau):
            for j in range(i + 1, n - (m - 1) * tau):
                dist = np.linalg.norm(embeddings[:, i] - embeddings[:, j])
                distances[i] += dist
                distances[j] += dist

        # Normalize and compute MLE
        distances = distances / (n - (m - 1) * tau - 1)
        mle = np.mean(np.log(distances + 1e-8)) / (tau * 0.001)  # Convert to Hz

        return max(mle, min_lyap)

    def compute_recurrence_metrics(self,
                                   signal: np.ndarray,
                                   emb_dim: int = 3,
                                   delay: int = 1,
                                   radius: float = 0.1) -> dict:
        """
        Compute recurrence quantification analysis (RQA) features

        Args:
            signal: Audio signal (vocal segment)
            emb_dim: Embedding dimension
            delay: Time delay
            radius: Nearest neighbor radius

        Returns:
            Dictionary of RQA metrics
        """
        # Phase space reconstruction
        m = emb_dim
        tau = delay
        n = len(signal)
        embeddings = np.zeros((m, n - (m - 1) * tau))

        for i in range(n - (m - 1) * tau):
            for j in range(m):
                embeddings[j, i] = signal[i + j * tau]

        # Create recurrence plot
        recurrence_matrix = np.zeros((n - (m - 1) * tau, n - (m - 1) * tau))
        for i in range(n - (m - 1) * tau):
            for j in range(n - (m - 1) * tau):
                dist = np.linalg.norm(embeddings[:, i] - embeddings[:, j])
                recurrence_matrix[i, j] = 1 if dist <= radius else 0

        # Compute RQA metrics
        diagonal_lines = []
        vertical_lines = []
        white_vertical_lines = []

        # Find line structures (simplified implementation)
        for i in range(1, n - (m - 1) * tau):
            for j in range(n - (m - 1) * tau):
                if recurrence_matrix[i, j] and recurrence_matrix[i - 1, j]:
                    diagonal_lines.append((i, j))
                if recurrence_matrix[i, j] and recurrence_matrix[i, j - 1]:
                    vertical_lines.append((i, j))
                if not recurrence_matrix[i, j] and recurrence_matrix[i, j - 1]:
                    white_vertical_lines.append((i, j))

        # Calculate metrics
        recurrencerate = np.mean(recurrence_matrix)
        determinism = len(diagonal_lines) / (recurrencerate * n + 1e-8)
        divergence = 1.0 / (np.mean([len(np.arange(i[0], i[0] - min(10, i[0])))
                                     for i in diagonal_lines]) + 1e-8)

        return {
            'recurrence_rate': float(recurrencerate),
            'determinism': float(determinism),
            'divergence': float(divergence),
            'diagonal_lines': len(diagonal_lines),
            'vertical_lines': len(vertical_lines)
        }

    def augment_add_noise(self,
                          signal: np.ndarray,
                          noise_level: float = 0.001) -> np.ndarray:
        """
        Add background noise to audio signal

        Args:
            signal: Original audio signal
            noise_level: Relative amplitude of added noise

        Returns:
            Noisy audio signal
        """
        noise = np.random.randn(len(signal))
        noisy_signal = signal + noise_level * np.max(np.abs(signal)) * noise
        return np.clip(noisy_signal, -1.0, 1.0)

    def augment_speed_perturb(self,
                              signal: np.ndarray,
                              factors: List[float] = [0.9, 1.0, 1.1]) -> List[np.ndarray]:
        """
        Apply speed perturbations to create augmented copies

        Args:
            signal: Original audio signal
            factors: Speed adjustment factors

        Returns:
            List of augmented audio signals
        """
        augmented = []
        for factor in factors:
            augmented.append(librosa.effects.time_stretch(signal, rate=factor))
        return augmented

    def segment_speech(self,
                       signal: np.ndarray,
                       sr: int,
                       min_duration: float = 1.0) -> List[np.ndarray]:
        """
        Segment long audio into fixed-duration chunks

        Args:
            signal: Full audio signal
            sr: Sample rate
            min_duration: Minimum segment duration in seconds

        Returns:
            List of segmented audio signals
        """
        seg_length = int(min_duration * sr)
        segments = []

        for start in range(0, len(signal) - seg_length, seg_length // 2):
            segment = signal[start:start + seg_length]
            if len(segment) >= seg_length * 0.8:  # Skip very short segments
                segments.append(segment)

        return segments if segments else [signal[-seg_length:]]


# Example usage
if __name__ == "__main__":
    processor = AudioProcessor(target_sr=16000, n_mfcc=40)

    # Process a sample file
    audio_path = "P:/PycharmProjects/pythonProject1/dataset/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"
    signal, sr = processor.load_audio(audio_path)

    # Extract features
    mfcc = processor.extract_mfcc(signal, sr)
    fbank = processor.extract_fbank(signal, sr)
    mle = processor.compute_mle(signal, emb_dim=10, delay=2)
    rqa = processor.compute_recurrence_metrics(signal, emb_dim=3, delay=1)

    print(f"MFCC shape: {mfcc.shape}")
    print(f"Fbank shape: {fbank.shape}")
    print(f"Maximum Lyapunov exponent: {mle:.4f}")
    print("RQA metrics:", rqa)

    # Create augmented versions
    noisy = processor.augment_add_noise(signal, noise_level=0.001)
    speed_perturbed = processor.augment_speed_perturb(signal, factors=[0.9, 1.1])

    # Segment long recordings
    segments = processor.segment_speech(signal, sr, min_duration=1.0)
    print(f"Created {len(segments)} segments")
    