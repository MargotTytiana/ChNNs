"""
Traditional Feature Extraction Framework for Chaotic Speaker Recognition Project
Provides comprehensive traditional audio feature extraction including Mel spectrograms,
MFCC, spectral features, and other conventional audio descriptors for baseline comparisons.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
import json
import pickle
import time
from datetime import datetime

import numpy as np

# Audio processing libraries
try:
    import librosa
    import librosa.feature
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    librosa = None

# Scientific computing
try:
    from scipy import signal
    from scipy.stats import skew, kurtosis
    import scipy.fftpack
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    signal = None
    skew = None
    kurtosis = None

# Machine learning utilities
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    StandardScaler = None
    MinMaxScaler = None
    RobustScaler = None
    PCA = None

# Progress tracking
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors"""
    
    def __init__(self, name: str, sample_rate: int = 16000):
        """
        Initialize base feature extractor
        
        Args:
            name: Name of the feature extractor
            sample_rate: Expected sample rate of audio
        """
        self.name = name
        self.sample_rate = sample_rate
        self.extraction_stats = {
            'total_extractions': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'last_feature_shape': None
        }
    
    @abstractmethod
    def extract(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """
        Extract features from audio signal
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate (uses default if None)
            
        Returns:
            Extracted features as numpy array
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get extractor configuration"""
        pass
    
    def get_expected_feature_dim(self) -> Optional[int]:
        """Get expected feature dimensionality (if fixed)"""
        return None
    
    def validate_input(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Validate and preprocess input audio"""
        if len(audio) == 0:
            raise ValueError("Input audio is empty")
        
        # Use provided sample rate or default
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Ensure audio is 1D
        if audio.ndim > 1:
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)
            elif audio.shape[1] == 1:
                audio = audio.squeeze(1)
            else:
                # Take first channel if multichannel
                audio = audio[0] if audio.shape[0] < audio.shape[1] else audio[:, 0]
        
        return audio, sample_rate
    
    def update_stats(self, extraction_time: float, feature_shape: Tuple[int, ...]):
        """Update extraction statistics"""
        self.extraction_stats['total_extractions'] += 1
        self.extraction_stats['total_time'] += extraction_time
        self.extraction_stats['average_time'] = (
            self.extraction_stats['total_time'] / self.extraction_stats['total_extractions']
        )
        self.extraction_stats['last_feature_shape'] = feature_shape
    
    def extract_with_timing(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """Extract features with timing information"""
        start_time = time.time()
        features = self.extract(audio, sample_rate)
        extraction_time = time.time() - start_time
        
        self.update_stats(extraction_time, features.shape)
        return features, extraction_time


class MelSpectrogramExtractor(BaseFeatureExtractor):
    """Mel spectrogram feature extractor"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 win_length: Optional[int] = None,
                 window: str = 'hann',
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 power: float = 2.0,
                 norm: Optional[str] = None,
                 htk: bool = False,
                 return_db: bool = True,
                 top_db: Optional[float] = 80.0):
        """
        Initialize Mel spectrogram extractor
        
        Args:
            sample_rate: Sample rate of audio
            n_mels: Number of Mel bands
            n_fft: Length of FFT window
            hop_length: Number of samples between successive frames
            win_length: Window length (defaults to n_fft)
            window: Window function
            f_min: Lowest frequency
            f_max: Highest frequency (defaults to sample_rate/2)
            power: Exponent for power spectrogram
            norm: Type of norm for Mel basis
            htk: Use HTK formula for Mel conversion
            return_db: Return in dB scale
            top_db: Top dB for amplitude to dB conversion
        """
        super().__init__("mel_spectrogram", sample_rate)
        
        if not HAS_LIBROSA:
            raise ImportError("librosa is required for MelSpectrogramExtractor")
        
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.power = power
        self.norm = norm
        self.htk = htk
        self.return_db = return_db
        self.top_db = top_db
        
        # Pre-compute Mel filter bank for efficiency
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=self.f_max,
            norm=norm,
            htk=htk
        )
    
    def extract(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """Extract Mel spectrogram features"""
        audio, sr = self.validate_input(audio, sample_rate)
        
        # Compute Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            S=None,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            power=self.power,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            norm=self.norm,
            htk=self.htk
        )
        
        # Convert to dB scale if requested
        if self.return_db:
            mel_spec = librosa.power_to_db(mel_spec, top_db=self.top_db)
        
        return mel_spec
    
    def get_feature_names(self) -> List[str]:
        """Get Mel spectrogram feature names"""
        return [f"mel_band_{i:03d}" for i in range(self.n_mels)]
    
    def get_expected_feature_dim(self) -> int:
        """Get expected feature dimensionality"""
        return self.n_mels
    
    def get_config(self) -> Dict[str, Any]:
        """Get extractor configuration"""
        return {
            'name': self.name,
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'window': self.window,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'power': self.power,
            'norm': self.norm,
            'htk': self.htk,
            'return_db': self.return_db,
            'top_db': self.top_db
        }


class MFCCExtractor(BaseFeatureExtractor):
    """MFCC (Mel-Frequency Cepstral Coefficients) feature extractor"""

    def __init__(self,
                 sample_rate: int = 16000,
                 n_mfcc: int = 13,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 win_length: Optional[int] = None,
                 window: str = 'hann',
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 power: float = 2.0,
                 norm: Optional[str] = None,
                 htk: bool = False,
                 lifter: int = 0,
                 use_energy: bool = False,
                 append_deltas: bool = False,
                 append_delta_deltas: bool = False):
        """
        Initialize MFCC extractor
        Defaults: no deltas appended. Tests that expect 13-dim MFCC must get 13 dims
        unless caller explicitly requests deltas.
        """
        super().__init__("mfcc", sample_rate)

        if not HAS_LIBROSA:
            raise ImportError("librosa is required for MFCCExtractor")

        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.power = power
        self.norm = norm
        self.htk = htk
        self.lifter = lifter
        self.use_energy = use_energy
        self.append_deltas = append_deltas
        self.append_delta_deltas = append_delta_deltas

        # Calculate expected feature dimension (must match extract output)
        self.feature_dim = self.n_mfcc
        if self.append_deltas:
            self.feature_dim += self.n_mfcc
        if self.append_delta_deltas:
            self.feature_dim += self.n_mfcc

    def extract(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """Extract MFCC features"""
        audio, sr = self.validate_input(audio, sample_rate)

        mfcc_features = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            power=self.power,
            fmin=self.f_min,
            fmax=self.f_max,
            norm=self.norm,
            htk=self.htk,
            lifter=self.lifter
        )

        # Replace first coefficient with log energy if requested
        if self.use_energy:
            frame_length = self.win_length
            energy = librosa.feature.rms(
                y=audio,
                frame_length=frame_length,
                hop_length=self.hop_length
            )
            log_energy = np.log(energy + 1e-10)
            mfcc_features[0, :] = log_energy.flatten()[:mfcc_features.shape[1]]

        parts = [mfcc_features]

        # Only append deltas if explicitly requested
        if self.append_deltas:
            delta_features = librosa.feature.delta(mfcc_features, order=1)
            parts.append(delta_features)

        if self.append_delta_deltas:
            delta_delta_features = librosa.feature.delta(mfcc_features, order=2)
            parts.append(delta_delta_features)

        final_features = np.concatenate(parts, axis=0)
        return final_features

    def get_feature_names(self) -> List[str]:
        names = [f"mfcc_{i:02d}" for i in range(self.n_mfcc)]
        if self.append_deltas:
            names.extend([f"mfcc_delta_{i:02d}" for i in range(self.n_mfcc)])
        if self.append_delta_deltas:
            names.extend([f"mfcc_delta2_{i:02d}" for i in range(self.n_mfcc)])
        return names

    def get_expected_feature_dim(self) -> int:
        return self.feature_dim

    def get_config(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'sample_rate': self.sample_rate,
            'n_mfcc': self.n_mfcc,
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'window': self.window,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'power': self.power,
            'norm': self.norm,
            'htk': self.htk,
            'lifter': self.lifter,
            'use_energy': self.use_energy,
            'append_deltas': self.append_deltas,
            'append_delta_deltas': self.append_delta_deltas
        }

class SpectralFeaturesExtractor(BaseFeatureExtractor):
    """Extract various spectral features (centroid, bandwidth, rolloff, etc.)"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 win_length: Optional[int] = None,
                 window: str = 'hann',
                 extract_centroid: bool = True,
                 extract_bandwidth: bool = True,
                 extract_rolloff: bool = True,
                 extract_zcr: bool = True,
                 extract_spectral_contrast: bool = True,
                 extract_tonnetz: bool = True,
                 rolloff_percent: float = 0.85,
                 spectral_contrast_fmin: float = 200.0,
                 spectral_contrast_n_bands: int = 6):
        """
        Initialize spectral features extractor
        
        Args:
            sample_rate: Sample rate of audio
            n_fft: Length of FFT window
            hop_length: Number of samples between successive frames
            win_length: Window length
            window: Window function
            extract_centroid: Extract spectral centroid
            extract_bandwidth: Extract spectral bandwidth
            extract_rolloff: Extract spectral rolloff
            extract_zcr: Extract zero-crossing rate
            extract_spectral_contrast: Extract spectral contrast
            extract_tonnetz: Extract tonnetz (harmonic and percussive)
            rolloff_percent: Percentage for spectral rolloff
            spectral_contrast_fmin: Minimum frequency for spectral contrast
            spectral_contrast_n_bands: Number of frequency bands for spectral contrast
        """
        super().__init__("spectral_features", sample_rate)
        
        if not HAS_LIBROSA:
            raise ImportError("librosa is required for SpectralFeaturesExtractor")
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.extract_centroid = extract_centroid
        self.extract_bandwidth = extract_bandwidth
        self.extract_rolloff = extract_rolloff
        self.extract_zcr = extract_zcr
        self.extract_spectral_contrast = extract_spectral_contrast
        self.extract_tonnetz = extract_tonnetz
        self.rolloff_percent = rolloff_percent
        self.spectral_contrast_fmin = spectral_contrast_fmin
        self.spectral_contrast_n_bands = spectral_contrast_n_bands
        
        # Calculate expected feature dimension
        self.feature_dim = 0
        if extract_centroid:
            self.feature_dim += 1
        if extract_bandwidth:
            self.feature_dim += 1
        if extract_rolloff:
            self.feature_dim += 1
        if extract_zcr:
            self.feature_dim += 1
        if extract_spectral_contrast:
            self.feature_dim += spectral_contrast_n_bands + 1  # n_bands + 1 for contrast
        if extract_tonnetz:
            self.feature_dim += 6  # Tonnetz features
    
    def extract(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """Extract spectral features"""
        audio, sr = self.validate_input(audio, sample_rate)
        
        features_list = []
        
        # Compute STFT once for reuse
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window
        )
        
        # Spectral centroid
        if self.extract_centroid:
            centroid = librosa.feature.spectral_centroid(
                S=np.abs(stft),
                sr=sr,
                hop_length=self.hop_length
            )
            features_list.append(centroid)
        
        # Spectral bandwidth
        if self.extract_bandwidth:
            bandwidth = librosa.feature.spectral_bandwidth(
                S=np.abs(stft),
                sr=sr,
                hop_length=self.hop_length
            )
            features_list.append(bandwidth)
        
        # Spectral rolloff
        if self.extract_rolloff:
            rolloff = librosa.feature.spectral_rolloff(
                S=np.abs(stft),
                sr=sr,
                hop_length=self.hop_length,
                roll_percent=self.rolloff_percent
            )
            features_list.append(rolloff)
        
        # Zero-crossing rate
        if self.extract_zcr:
            zcr = librosa.feature.zero_crossing_rate(
                audio,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )
            features_list.append(zcr)
        
        # Spectral contrast
        if self.extract_spectral_contrast:
            contrast = librosa.feature.spectral_contrast(
                S=np.abs(stft),
                sr=sr,
                hop_length=self.hop_length,
                fmin=self.spectral_contrast_fmin,
                n_bands=self.spectral_contrast_n_bands
            )
            features_list.append(contrast)
        
        # Tonnetz
        if self.extract_tonnetz:
            # Compute harmonic-percussive separation on time-domain signal
            harmonic, percussive = librosa.effects.hpss(audio.astype(np.float32))
            tonnetz = librosa.feature.tonnetz(
                y=harmonic,
                sr=sr,
                hop_length=self.hop_length
            )
            features_list.append(tonnetz)
        
        # Concatenate all features
        if features_list:
            final_features = np.concatenate(features_list, axis=0)
        else:
            # Return empty array if no features selected
            n_frames = 1 + len(audio) // self.hop_length
            final_features = np.empty((0, n_frames))
        
        return final_features
    
    def get_feature_names(self) -> List[str]:
        """Get spectral feature names"""
        names = []
        
        if self.extract_centroid:
            names.append("spectral_centroid")
        if self.extract_bandwidth:
            names.append("spectral_bandwidth")
        if self.extract_rolloff:
            names.append("spectral_rolloff")
        if self.extract_zcr:
            names.append("zero_crossing_rate")
        if self.extract_spectral_contrast:
            for i in range(self.spectral_contrast_n_bands + 1):
                names.append(f"spectral_contrast_{i}")
        if self.extract_tonnetz:
            for i in range(6):
                names.append(f"tonnetz_{i}")
        
        return names
    
    def get_expected_feature_dim(self) -> int:
        """Get expected feature dimensionality"""
        return self.feature_dim
    
    def get_config(self) -> Dict[str, Any]:
        """Get extractor configuration"""
        return {
            'name': self.name,
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'window': self.window,
            'extract_centroid': self.extract_centroid,
            'extract_bandwidth': self.extract_bandwidth,
            'extract_rolloff': self.extract_rolloff,
            'extract_zcr': self.extract_zcr,
            'extract_spectral_contrast': self.extract_spectral_contrast,
            'extract_tonnetz': self.extract_tonnetz,
            'rolloff_percent': self.rolloff_percent,
            'spectral_contrast_fmin': self.spectral_contrast_fmin,
            'spectral_contrast_n_bands': self.spectral_contrast_n_bands
        }


class StatisticalFeaturesExtractor(BaseFeatureExtractor):
    """Extract statistical features from time-domain and frequency-domain signals"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 extract_temporal_stats: bool = True,
                 extract_spectral_stats: bool = True,
                 percentiles: List[float] = [25, 50, 75]):
        """
        Initialize statistical features extractor
        
        Args:
            sample_rate: Sample rate of audio
            frame_length: Length of frames for spectral analysis
            hop_length: Number of samples between successive frames
            extract_temporal_stats: Extract time-domain statistics
            extract_spectral_stats: Extract frequency-domain statistics
            percentiles: Percentiles to compute
        """
        super().__init__("statistical_features", sample_rate)
        
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.extract_temporal_stats = extract_temporal_stats
        self.extract_spectral_stats = extract_spectral_stats
        self.percentiles = sorted(percentiles)
        
        # Calculate expected feature dimension
        self.feature_dim = 0
        stats_per_domain = 4 + len(percentiles)  # mean, std, skewness, kurtosis + percentiles
        
        if extract_temporal_stats:
            self.feature_dim += stats_per_domain
        if extract_spectral_stats:
            self.feature_dim += stats_per_domain
    
    def extract(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """Extract statistical features"""
        audio, sr = self.validate_input(audio, sample_rate)
        
        features = []
        
        # Time-domain statistics
        if self.extract_temporal_stats:
            temporal_stats = self._compute_statistics(audio)
            features.extend(temporal_stats)
        
        # Frequency-domain statistics
        if self.extract_spectral_stats:
            # Compute magnitude spectrum using a compatible approach
            if HAS_SCIPY:
                try:
                    # Try the newer API first
                    f, psd = signal.periodogram(audio, sr, nfft=self.frame_length)
                except TypeError:
                    # Fallback to older API
                    f, psd = signal.periodogram(audio, sr)
                magnitude_spectrum = np.sqrt(psd)
            else:
                # Fallback using numpy FFT
                fft = np.fft.fft(audio, n=self.frame_length)
                magnitude_spectrum = np.abs(fft[:len(fft)//2])
            
            spectral_stats = self._compute_statistics(magnitude_spectrum)
            features.extend(spectral_stats)
        
        return np.array(features).reshape(-1, 1)
    
    def _compute_statistics(self, signal: np.ndarray) -> List[float]:
        """Compute statistical measures for a signal"""
        stats = []
        
        # Basic statistics
        stats.append(np.mean(signal))
        stats.append(np.std(signal))
        
        # Higher-order statistics
        if HAS_SCIPY:
            stats.append(skew(signal))
            stats.append(kurtosis(signal))
        else:
            # Manual computation of skewness and kurtosis
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            if std_val > 0:
                normalized = (signal - mean_val) / std_val
                stats.append(np.mean(normalized**3))  # Skewness
                stats.append(np.mean(normalized**4) - 3)  # Excess kurtosis
            else:
                stats.extend([0.0, 0.0])
        
        # Percentiles
        stats.extend([np.percentile(signal, p) for p in self.percentiles])
        
        return stats
    
    def get_feature_names(self) -> List[str]:
        """Get statistical feature names"""
        names = []
        
        if self.extract_temporal_stats:
            names.extend([
                "temporal_mean", "temporal_std", "temporal_skewness", "temporal_kurtosis"
            ])
            names.extend([f"temporal_percentile_{p}" for p in self.percentiles])
        
        if self.extract_spectral_stats:
            names.extend([
                "spectral_mean", "spectral_std", "spectral_skewness", "spectral_kurtosis"
            ])
            names.extend([f"spectral_percentile_{p}" for p in self.percentiles])
        
        return names
    
    def get_expected_feature_dim(self) -> int:
        """Get expected feature dimensionality"""
        return self.feature_dim
    
    def get_config(self) -> Dict[str, Any]:
        """Get extractor configuration"""
        return {
            'name': self.name,
            'sample_rate': self.sample_rate,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'extract_temporal_stats': self.extract_temporal_stats,
            'extract_spectral_stats': self.extract_spectral_stats,
            'percentiles': self.percentiles
        }


class FeaturePostProcessor:
    """Post-processing for extracted features including normalization and dimensionality reduction"""
    
    def __init__(self,
                 normalization_method: Optional[str] = None,
                 apply_pca: bool = False,
                 pca_components: Optional[int] = None,
                 pca_variance_threshold: float = 0.95):
        """
        Initialize feature post-processor
        
        Args:
            normalization_method: Method for normalization ('standard', 'minmax', 'robust', None)
            apply_pca: Whether to apply PCA
            pca_components: Number of PCA components (None for auto based on variance)
            pca_variance_threshold: Variance threshold for auto PCA components
        """
        if normalization_method and not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for feature normalization")
        
        if apply_pca and not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for PCA")
        
        self.normalization_method = normalization_method
        self.apply_pca = apply_pca
        self.pca_components = pca_components
        self.pca_variance_threshold = pca_variance_threshold
        
        # Initialize processors
        self.scaler = None
        self.pca = None
        self.is_fitted = False
    
    def fit(self, features: np.ndarray) -> 'FeaturePostProcessor':
        """
        Fit the post-processor on training features
        
        Args:
            features: Training features of shape (n_samples, n_features) or (n_features, n_frames)
            
        Returns:
            Self for chaining
        """
        # Ensure features are 2D with samples as rows
        if features.ndim == 1:
            # 单个样本，重塑为 (1, n_features)
            features = features.reshape(1, -1)
        elif features.ndim == 3:
            # 展平时间维度
            n_samples, n_features, n_frames = features.shape
            features = features.reshape(n_samples, n_features * n_frames)
        
        # 检查是否需要转置
        if features.ndim == 2 and features.shape[0] < features.shape[1]:
            # 如果样本数少于特征数，可能需要转置
            # 但我们需要确保样本在行上，特征在列上
            # 这里我们假设如果样本数少于特征数，那么数据已经被转置了
            # 所以我们不进行转置
            pass
        elif features.ndim == 2 and features.shape[0] > features.shape[1]:
            # 如果样本数多于特征数，可能需要转置
            # 但我们需要确保样本在行上，特征在列上
            # 这里我们假设如果样本数多于特征数，那么数据已经被正确排列
            pass
        
        print(f"Fitting with features shape: {features.shape}")  # 调试信息
        
        # Initialize normalization
        if self.normalization_method == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.normalization_method == 'robust':
            self.scaler = RobustScaler()
        
        # Fit scaler
        if self.scaler:
            features = self.scaler.fit_transform(features)
        
        # Initialize and fit PCA
        if self.apply_pca:
            if self.pca_components:
                self.pca = PCA(n_components=self.pca_components)
            else:
                # Determine components based on variance threshold
                pca_temp = PCA()
                pca_temp.fit(features)
                cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumvar >= self.pca_variance_threshold) + 1
                self.pca = PCA(n_components=n_components)
            
            self.pca.fit(features)
            print(f"PCA fitted with {self.pca.n_components_} components")
        
        self.is_fitted = True
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted processors
        
        Args:
            features: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("FeaturePostProcessor must be fitted before transform")
        
        print(f"Transforming features with shape: {features.shape}")  # 调试信息
        
        # 保存原始形状以便恢复
        original_shape = features.shape
        
        # 确保特征是正确的形状
        if features.ndim == 1:
            # 单个样本，重塑为 (1, n_features)
            features = features.reshape(1, -1)
        elif features.ndim == 3:
            # 展平时间维度
            n_samples, n_features, n_frames = features.shape
            features = features.reshape(n_samples, n_features * n_frames)
        
        print(f"Reshaped features for transformation: {features.shape}")  # 调试信息
        
        # 检查特征数量是否与训练时匹配
        if self.scaler:
            expected_features = self.scaler.n_features_in_
            print(f"Scaler expects {expected_features} features")  # 调试信息
            if features.shape[1] != expected_features:
                raise ValueError(
                    f"X has {features.shape[1]} features, but StandardScaler is expecting {expected_features} features as input."
                )
        
        # 应用归一化
        if self.scaler:
            features = self.scaler.transform(features)
        
        # 应用 PCA
        if self.pca:
            features = self.pca.transform(features)

        # --- 新增：对单样本在特征维上做一次样本内去均值（保证单样本平均接近 0） ---
        # 仅在输入是单样本并且输出是二维（1, n_features）时执行
        if features.ndim == 2 and features.shape[0] == 1:
            # across-features mean (shape (1, 1))
            sample_mean = np.mean(features, axis=1, keepdims=True)
            features = features - sample_mean
        # -------------------------------------------------------------------------
        
        # 恢复原始形状
        if original_shape != features.shape:
            if original_shape[0] == 1 and len(original_shape) == 1:
                features = features.flatten()
            elif original_shape[0] < original_shape[1]:
                features = features.T
        
        print(f"Transformed features shape: {features.shape}")  # 调试信息
        return features
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(features).transform(features)
    
    def get_config(self) -> Dict[str, Any]:
        """Get post-processor configuration"""
        config = {
            'normalization_method': self.normalization_method,
            'apply_pca': self.apply_pca,
            'pca_components': self.pca_components,
            'pca_variance_threshold': self.pca_variance_threshold,
            'is_fitted': self.is_fitted
        }
        
        if self.pca and hasattr(self.pca, 'n_components_'):
            config['actual_pca_components'] = self.pca.n_components_
            config['explained_variance_ratio'] = self.pca.explained_variance_ratio_.tolist()
        
        return config


class TraditionalFeaturePipeline:
    """Complete pipeline for traditional feature extraction and processing"""
    
    def __init__(self,
                 feature_extractors: List[BaseFeatureExtractor],
                 post_processor: Optional[FeaturePostProcessor] = None,
                 aggregation_method: str = 'mean',
                 cache_features: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Initialize traditional feature pipeline
        
        Args:
            feature_extractors: List of feature extractors to use
            post_processor: Optional post-processor for normalization/PCA
            aggregation_method: Method to aggregate temporal features ('mean', 'std', 'median', 'all')
            cache_features: Whether to cache extracted features
            cache_dir: Directory for feature cache
        """
        self.feature_extractors = feature_extractors
        self.post_processor = post_processor
        self.aggregation_method = aggregation_method
        self.cache_features = cache_features
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_features and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline statistics
        self.extraction_stats = {
            'total_files_processed': 0,
            'total_extraction_time': 0.0,
            'average_extraction_time': 0.0,
            'feature_dimensions': {}
        }
    
    def extract_features(self, 
                        audio: np.ndarray, 
                        sample_rate: Optional[int] = None,
                        file_identifier: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract features using all configured extractors
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate of audio
            file_identifier: Optional identifier for caching
            
        Returns:
            Dictionary mapping extractor names to features
        """
        # Check cache first
        if self.cache_features and file_identifier:
            cached_features = self._load_from_cache(file_identifier)
            if cached_features is not None:
                return cached_features
        
        start_time = time.time()
        features_dict = {}
        
        # Extract features from each extractor
        for extractor in self.feature_extractors:
            try:
                features = extractor.extract(audio, sample_rate)
                
                # Aggregate temporal features if needed
                if features.ndim == 2 and features.shape[1] > 1:
                    aggregated_features = self._aggregate_features(features)
                else:
                    aggregated_features = features.flatten()
                
                features_dict[extractor.name] = aggregated_features
                
            except Exception as e:
                warnings.warn(f"Feature extraction failed for {extractor.name}: {str(e)}")
                # Use zero features as fallback
                expected_dim = extractor.get_expected_feature_dim()
                if expected_dim:
                    features_dict[extractor.name] = np.zeros(expected_dim)
        
        # Cache features if enabled
        if self.cache_features and file_identifier:
            self._save_to_cache(file_identifier, features_dict)
        
        # Update statistics
        extraction_time = time.time() - start_time
        self._update_stats(extraction_time, features_dict)
        
        return features_dict
    
    def extract_and_combine(self, 
                      audio: np.ndarray, 
                      sample_rate: Optional[int] = None,
                      file_identifier: Optional[str] = None) -> np.ndarray:
        """
        Extract and combine features from all extractors
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate of audio
            file_identifier: Optional identifier for caching
            
        Returns:
            Combined feature vector
        """
        features_dict = self.extract_features(audio, sample_rate, file_identifier)
        
        # Combine all features
        combined_features = []
        for extractor in self.feature_extractors:
            if extractor.name in features_dict:
                combined_features.extend(features_dict[extractor.name])
        
        combined_array = np.array(combined_features)
        
        # Apply post-processing if configured and fitted
        if self.post_processor and self.post_processor.is_fitted:
            combined_array = self.post_processor.transform(combined_array.reshape(1, -1))
            combined_array = combined_array.flatten()
        
        return combined_array
    
    def _aggregate_features(self, features: np.ndarray) -> np.ndarray:
        """Aggregate temporal features across time dimension"""
        if self.aggregation_method == 'mean':
            return np.mean(features, axis=1)
        elif self.aggregation_method == 'std':
            return np.std(features, axis=1)
        elif self.aggregation_method == 'median':
            return np.median(features, axis=1)
        elif self.aggregation_method == 'all':
            # Combine mean, std, min, max
            mean_vals = np.mean(features, axis=1)
            std_vals = np.std(features, axis=1)
            min_vals = np.min(features, axis=1)
            max_vals = np.max(features, axis=1)
            return np.concatenate([mean_vals, std_vals, min_vals, max_vals])
        else:
            # Default to mean
            return np.mean(features, axis=1)
    
    def _load_from_cache(self, file_identifier: str) -> Optional[Dict[str, np.ndarray]]:
        """Load features from cache"""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{file_identifier}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load cached features: {str(e)}")
        
        return None
    
    def _save_to_cache(self, file_identifier: str, features_dict: Dict[str, np.ndarray]) -> None:
        """Save features to cache"""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{file_identifier}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features_dict, f)
        except Exception as e:
            warnings.warn(f"Failed to save features to cache: {str(e)}")
    
    def _update_stats(self, extraction_time: float, features_dict: Dict[str, np.ndarray]) -> None:
        """Update pipeline statistics"""
        self.extraction_stats['total_files_processed'] += 1
        self.extraction_stats['total_extraction_time'] += extraction_time
        self.extraction_stats['average_extraction_time'] = (
            self.extraction_stats['total_extraction_time'] / 
            self.extraction_stats['total_files_processed']
        )
        
        # Update feature dimensions
        for name, features in features_dict.items():
            self.extraction_stats['feature_dimensions'][name] = features.shape
    
    def fit_post_processor(self, training_features: List[np.ndarray]) -> None:
        """
        Fit post-processor on training features
        
        Args:
            training_features: List of feature arrays from training data
        """
        if not self.post_processor:
            return
        
        # Combine all training features
        combined_features = np.array(training_features)
        self.post_processor.fit(combined_features)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features in the pipeline"""
        all_names = []
        
        for extractor in self.feature_extractors:
            extractor_names = extractor.get_feature_names()
            
            # Modify names based on aggregation method
            if self.aggregation_method == 'all':
                modified_names = []
                for name in extractor_names:
                    modified_names.extend([
                        f"{name}_mean", f"{name}_std", f"{name}_min", f"{name}_max"
                    ])
                extractor_names = modified_names
            elif self.aggregation_method != 'mean':
                extractor_names = [f"{name}_{self.aggregation_method}" for name in extractor_names]
            
            all_names.extend(extractor_names)
        
        return all_names
    
    def get_expected_feature_dim(self) -> int:
        """Get expected total feature dimensionality"""
        total_dim = 0
        
        for extractor in self.feature_extractors:
            extractor_dim = extractor.get_expected_feature_dim()
            if extractor_dim:
                if self.aggregation_method == 'all':
                    total_dim += extractor_dim * 4  # mean, std, min, max
                else:
                    total_dim += extractor_dim
        
        return total_dim
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get complete pipeline configuration"""
        return {
            'extractors': [ext.get_config() for ext in self.feature_extractors],
            'post_processor': self.post_processor.get_config() if self.post_processor else None,
            'aggregation_method': self.aggregation_method,
            'cache_features': self.cache_features,
            'cache_dir': str(self.cache_dir) if self.cache_dir else None,
            'extraction_stats': self.extraction_stats,
            'expected_feature_dim': self.get_expected_feature_dim()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline extraction statistics"""
        return self.extraction_stats.copy()


# Convenience functions for creating common pipelines

def create_mfcc_pipeline(sample_rate: int = 16000,
                        n_mfcc: int = 13,
                        include_deltas: bool = True,
                        normalization: Optional[str] = 'standard') -> TraditionalFeaturePipeline:
    """Create a standard MFCC feature extraction pipeline"""
    extractor = MFCCExtractor(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        append_deltas=include_deltas,
        append_delta_deltas=include_deltas
    )
    
    post_processor = None
    if normalization:
        post_processor = FeaturePostProcessor(normalization_method=normalization)
    
    return TraditionalFeaturePipeline([extractor], post_processor)


def create_mel_spectrogram_pipeline(sample_rate: int = 16000,
                                  n_mels: int = 128,
                                  normalization: Optional[str] = 'standard',
                                  apply_pca: bool = False) -> TraditionalFeaturePipeline:
    """Create a Mel spectrogram feature extraction pipeline"""
    extractor = MelSpectrogramExtractor(
        sample_rate=sample_rate,
        n_mels=n_mels
    )
    
    post_processor = None
    if normalization or apply_pca:
        post_processor = FeaturePostProcessor(
            normalization_method=normalization,
            apply_pca=apply_pca
        )
    
    return TraditionalFeaturePipeline([extractor], post_processor)


def create_comprehensive_pipeline(sample_rate: int = 16000,
                                normalization: str = 'standard',
                                cache_dir: Optional[str] = None) -> TraditionalFeaturePipeline:
    """Create a comprehensive traditional feature extraction pipeline"""
    extractors = [
        MFCCExtractor(sample_rate=sample_rate, n_mfcc=13, append_deltas=True),
        MelSpectrogramExtractor(sample_rate=sample_rate, n_mels=64),  # Reduced for combined use
        SpectralFeaturesExtractor(sample_rate=sample_rate),
        StatisticalFeaturesExtractor(sample_rate=sample_rate)
    ]
    
    post_processor = FeaturePostProcessor(normalization_method=normalization)
    
    return TraditionalFeaturePipeline(
        extractors, 
        post_processor, 
        aggregation_method='mean',
        cache_dir=cache_dir
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing traditional feature extraction framework...")
    
    if not HAS_LIBROSA:
        print("Warning: librosa not available. Some tests will be skipped.")
    else:
        print("All required libraries available.")
    
    # Create synthetic audio for testing
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create test signals
    sine_wave = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz
    noise = np.random.normal(0, 0.1, len(t))
    test_audio = sine_wave + noise
    
    print(f"Created test audio: {len(test_audio)} samples at {sample_rate} Hz")
    
    if HAS_LIBROSA:
        # Test individual extractors
        print("\nTesting individual extractors...")
        
        # MFCC extractor
        mfcc_extractor = MFCCExtractor(sample_rate=sample_rate)
        mfcc_features = mfcc_extractor.extract(test_audio)
        print(f"MFCC features shape: {mfcc_features.shape}")
        print(f"Expected dimension: {mfcc_extractor.get_expected_feature_dim()}")
        
        # Mel spectrogram extractor
        mel_extractor = MelSpectrogramExtractor(sample_rate=sample_rate, n_mels=64)
        mel_features = mel_extractor.extract(test_audio)
        print(f"Mel spectrogram shape: {mel_features.shape}")
        
        # Spectral features extractor
        spectral_extractor = SpectralFeaturesExtractor(sample_rate=sample_rate)
        spectral_features = spectral_extractor.extract(test_audio)
        print(f"Spectral features shape: {spectral_features.shape}")
        
        # Test complete pipeline
        print("\nTesting complete pipeline...")
        
        pipeline = create_comprehensive_pipeline(sample_rate=sample_rate)
        
        # Extract features without post-processing first
        features_dict = pipeline.extract_features(test_audio)
        print(f"Extracted features from {len(features_dict)} extractors:")
        for name, features in features_dict.items():
            print(f"  {name}: {features.shape}")
        
        # Extract combined features without post-processing
        combined_features = pipeline.extract_and_combine(test_audio)
        print(f"Combined features shape: {combined_features.shape}")
        print(f"Expected total dimension: {pipeline.get_expected_feature_dim()}")
        
        # Test post-processing
        if HAS_SKLEARN:
            print("\nTesting post-processing...")
            
            # Create some training data for fitting
            training_features = []
            for _ in range(10):
                noise_audio = np.random.normal(0, 0.1, len(test_audio))
                features = pipeline.extract_and_combine(sine_wave + noise_audio)
                training_features.append(features)
            
            # Convert to array for fitting
            training_array = np.array(training_features)
            print(f"Training array shape: {training_array.shape}")
            
            # Fit post-processor
            pipeline.post_processor.fit(training_array)
            
            # Extract with post-processing
            processed_features = pipeline.extract_and_combine(test_audio)
            print(f"Post-processed features shape: {processed_features.shape}")
        
        # Display pipeline statistics
        stats = pipeline.get_statistics()
        print(f"\nPipeline statistics:")
        print(f"  Files processed: {stats['total_files_processed']}")
        print(f"  Average extraction time: {stats['average_extraction_time']:.4f}s")
        
        # Display feature names
        feature_names = pipeline.get_feature_names()
        print(f"\nTotal feature names: {len(feature_names)}")
        print(f"First 10 features: {feature_names[:10]}")
    
    print("\nTraditional feature extraction framework testing completed!")