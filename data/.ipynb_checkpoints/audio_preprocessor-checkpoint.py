"""
Audio Preprocessing System for Chaotic Speaker Recognition Project
Provides comprehensive audio preprocessing pipeline including loading, normalization,
augmentation, and feature preparation for speaker recognition tasks.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
import hashlib
import pickle
import json
from datetime import datetime
import math
import random

import numpy as np

# Audio processing libraries
try:
    import librosa
    import soundfile as sf
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    librosa = None
    sf = None

# Scientific computing
try:
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    import scipy.io.wavfile as wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    signal = None
    gaussian_filter1d = None

# Audio augmentation (optional)
try:
    import audiomentations as AA
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False
    AA = None

# Progress tracking
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable


class AudioPreprocessorBase(ABC):
    """Abstract base class for audio preprocessing components"""
    
    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Process audio signal
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate of input audio
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters"""
        pass


class AudioLoader(AudioPreprocessorBase):
    """Audio file loader with format support and error handling"""
    
    def __init__(self, 
                 target_sample_rate: Optional[int] = 16000,
                 mono: bool = True,
                 normalize_on_load: bool = False,
                 offset: float = 0.0,
                 duration: Optional[float] = None):
        """
        Initialize audio loader
        
        Args:
            target_sample_rate: Target sample rate (None to keep original)
            mono: Whether to convert to mono
            normalize_on_load: Whether to normalize amplitude on load
            offset: Start offset in seconds
            duration: Duration to load in seconds (None for full file)
        """
        if not HAS_AUDIO_LIBS:
            raise ImportError("Audio libraries (librosa, soundfile) required for AudioLoader")
        
        self.target_sample_rate = target_sample_rate
        self.mono = mono
        self.normalize_on_load = normalize_on_load
        self.offset = offset
        self.duration = duration
    
    def process(self, file_path: Union[str, Path], **kwargs) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            file_path: Path to audio file
            **kwargs: Override parameters
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Override parameters if provided
        sample_rate = kwargs.get('sample_rate', self.target_sample_rate)
        mono = kwargs.get('mono', self.mono)
        normalize = kwargs.get('normalize', self.normalize_on_load)
        offset = kwargs.get('offset', self.offset)
        duration = kwargs.get('duration', self.duration)
        
        try:
            # Load audio using librosa
            audio, sr = librosa.load(
                str(file_path),
                sr=sample_rate,
                mono=mono,
                offset=offset,
                duration=duration
            )
            
            # Basic validation
            if len(audio) == 0:
                raise ValueError("Loaded audio is empty")
            
            # Normalize if requested
            if normalize:
                audio = self._normalize_audio(audio)
            
            return audio, sr
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from {file_path}: {str(e)}")
    
    def _normalize_audio(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """Normalize audio to target RMS level"""
        if len(audio) == 0:
            return audio
        
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            scaling_factor = target_rms / current_rms
            audio = audio * scaling_factor
        
        return audio
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'target_sample_rate': self.target_sample_rate,
            'mono': self.mono,
            'normalize_on_load': self.normalize_on_load,
            'offset': self.offset,
            'duration': self.duration
        }


class AudioNormalizer(AudioPreprocessorBase):
    """Audio normalization and level adjustment"""
    
    def __init__(self, 
                 method: str = 'rms',
                 target_level: float = 0.1,
                 clip_threshold: Optional[float] = None):
        """
        Initialize audio normalizer
        
        Args:
            method: Normalization method ('rms', 'peak', 'lufs')
            target_level: Target level for normalization
            clip_threshold: Clipping threshold (None to disable)
        """
        self.method = method.lower()
        self.target_level = target_level
        self.clip_threshold = clip_threshold
        
        if self.method not in ['rms', 'peak', 'lufs']:
            raise ValueError(f"Unsupported normalization method: {method}")
    
    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Normalize audio signal
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            **kwargs: Override parameters
            
        Returns:
            Tuple of (normalized_audio, sample_rate)
        """
        method = kwargs.get('method', self.method)
        target_level = kwargs.get('target_level', self.target_level)
        clip_threshold = kwargs.get('clip_threshold', self.clip_threshold)
        
        if len(audio) == 0:
            return audio, sample_rate
        
        # Apply normalization based on method
        if method == 'rms':
            normalized_audio = self._rms_normalize(audio, target_level)
        elif method == 'peak':
            normalized_audio = self._peak_normalize(audio, target_level)
        elif method == 'lufs':
            normalized_audio = self._lufs_normalize(audio, sample_rate, target_level)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Apply clipping if specified
        if clip_threshold is not None:
            normalized_audio = np.clip(normalized_audio, -clip_threshold, clip_threshold)
        
        return normalized_audio, sample_rate
    
    def _rms_normalize(self, audio: np.ndarray, target_rms: float) -> np.ndarray:
        """RMS-based normalization"""
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            scaling_factor = target_rms / current_rms
            return audio * scaling_factor
        return audio
    
    def _peak_normalize(self, audio: np.ndarray, target_peak: float) -> np.ndarray:
        """Peak-based normalization"""
        current_peak = np.max(np.abs(audio))
        if current_peak > 0:
            scaling_factor = target_peak / current_peak
            return audio * scaling_factor
        return audio
    
    def _lufs_normalize(self, audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
        """LUFS-based normalization (simplified implementation)"""
        # Simplified LUFS calculation - for production use proper LUFS library
        # This is a basic approximation
        windowed_rms = self._windowed_rms(audio, sample_rate, window_size=0.4)
        current_lufs = -23.0 + 20 * np.log10(np.mean(windowed_rms) + 1e-10)
        
        gain_db = target_lufs - current_lufs
        gain_linear = 10**(gain_db / 20.0)
        
        return audio * gain_linear
    
    def _windowed_rms(self, audio: np.ndarray, sample_rate: int, window_size: float) -> np.ndarray:
        """Calculate windowed RMS for LUFS calculation"""
        window_samples = int(window_size * sample_rate)
        hop_samples = window_samples // 4
        
        windowed_rms = []
        for i in range(0, len(audio) - window_samples + 1, hop_samples):
            window = audio[i:i + window_samples]
            rms = np.sqrt(np.mean(window**2))
            windowed_rms.append(rms)
        
        return np.array(windowed_rms)
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'target_level': self.target_level,
            'clip_threshold': self.clip_threshold
        }


class SilenceTrimmer(AudioPreprocessorBase):
    """Trim silence from beginning and end of audio"""
    
    def __init__(self, 
                 top_db: int = 30,
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 margin_seconds: float = 0.1):
        """
        Initialize silence trimmer
        
        Args:
            top_db: Threshold in dB below reference for silence detection
            frame_length: Frame length for analysis
            hop_length: Hop length between frames
            margin_seconds: Margin to keep around speech in seconds
        """
        if not HAS_AUDIO_LIBS:
            raise ImportError("Librosa required for SilenceTrimmer")
        
        self.top_db = top_db
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.margin_seconds = margin_seconds
    
    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Trim silence from audio
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            **kwargs: Override parameters
            
        Returns:
            Tuple of (trimmed_audio, sample_rate)
        """
        top_db = kwargs.get('top_db', self.top_db)
        margin_samples = int(kwargs.get('margin_seconds', self.margin_seconds) * sample_rate)
        
        if len(audio) == 0:
            return audio, sample_rate
        
        try:
            # Use librosa's trim function
            trimmed_audio, trim_indices = librosa.effects.trim(
                audio,
                top_db=top_db,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            
            # Add margin if requested
            if margin_samples > 0:
                start_idx = max(0, trim_indices[0] - margin_samples)
                end_idx = min(len(audio), trim_indices[1] + margin_samples)
                trimmed_audio = audio[start_idx:end_idx]
            
            return trimmed_audio, sample_rate
            
        except Exception as e:
            warnings.warn(f"Silence trimming failed: {str(e)}")
            return audio, sample_rate
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'top_db': self.top_db,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'margin_seconds': self.margin_seconds
        }


class NoiseReducer(AudioPreprocessorBase):
    """Basic noise reduction using spectral subtraction"""
    
    def __init__(self, 
                 noise_factor: float = 0.1,
                 noise_duration: float = 0.5,
                 smoothing_factor: float = 0.1):
        """
        Initialize noise reducer
        
        Args:
            noise_factor: Factor for noise reduction strength
            noise_duration: Duration of noise profile estimation in seconds
            smoothing_factor: Spectral smoothing factor
        """
        if not HAS_AUDIO_LIBS:
            raise ImportError("Librosa required for NoiseReducer")
        
        self.noise_factor = noise_factor
        self.noise_duration = noise_duration
        self.smoothing_factor = smoothing_factor
    
    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Apply basic noise reduction
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            **kwargs: Override parameters
            
        Returns:
            Tuple of (denoised_audio, sample_rate)
        """
        if len(audio) == 0:
            return audio, sample_rate
        
        noise_factor = kwargs.get('noise_factor', self.noise_factor)
        noise_duration = kwargs.get('noise_duration', self.noise_duration)
        
        try:
            # Estimate noise from the beginning of the audio
            noise_samples = int(noise_duration * sample_rate)
            noise_samples = min(noise_samples, len(audio) // 4)
            
            if noise_samples < 1024:  # Not enough samples for noise estimation
                return audio, sample_rate
            
            # Compute STFT
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise spectrum from the beginning
            noise_spectrum = np.mean(magnitude[:, :noise_samples // 512], axis=1, keepdims=True)
            
            # Apply spectral subtraction
            enhanced_magnitude = magnitude - noise_factor * noise_spectrum
            enhanced_magnitude = np.maximum(enhanced_magnitude, 
                                          magnitude * self.smoothing_factor)
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            denoised_audio = librosa.istft(enhanced_stft, length=len(audio))
            
            return denoised_audio, sample_rate
            
        except Exception as e:
            warnings.warn(f"Noise reduction failed: {str(e)}")
            return audio, sample_rate
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'noise_factor': self.noise_factor,
            'noise_duration': self.noise_duration,
            'smoothing_factor': self.smoothing_factor
        }


class AudioAugmenter(AudioPreprocessorBase):
    """Audio data augmentation for training robustness"""
    
    def __init__(self, 
                 augmentation_config: Optional[Dict[str, Any]] = None,
                 augmentation_prob: float = 0.5,
                 random_seed: Optional[int] = None):
        """
        Initialize audio augmenter
        
        Args:
            augmentation_config: Configuration for augmentation parameters
            augmentation_prob: Probability of applying augmentation
            random_seed: Random seed for reproducible augmentation
        """
        self.augmentation_prob = augmentation_prob
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Default augmentation configuration
        default_config = {
            'time_stretch': {'rate_range': (0.8, 1.2), 'prob': 0.3},
            'pitch_shift': {'semitone_range': (-2, 2), 'prob': 0.3},
            'noise_injection': {'noise_factor_range': (0.001, 0.01), 'prob': 0.2},
            'volume_change': {'gain_range': (-6, 6), 'prob': 0.4},  # in dB
            'frequency_masking': {'freq_mask_param': 15, 'prob': 0.2},
            'time_masking': {'time_mask_param': 35, 'prob': 0.2}
        }
        
        self.config = {**default_config, **(augmentation_config or {})}
        
        # Initialize audiomentations if available
        self.audiomentations_available = HAS_AUDIOMENTATIONS
        if self.audiomentations_available:
            self._setup_audiomentations()
    
    def _setup_audiomentations(self):
        """Setup audiomentations transformations"""
        transforms = []
        
        # Add gain transformation
        if 'volume_change' in self.config:
            cfg = self.config['volume_change']
            transforms.append(
                AA.Gain(
                    min_gain_in_db=cfg['gain_range'][0],
                    max_gain_in_db=cfg['gain_range'][1],
                    p=cfg['prob']
                )
            )
        
        # Add noise injection
        if 'noise_injection' in self.config:
            cfg = self.config['noise_injection']
            transforms.append(
                AA.AddGaussianNoise(
                    min_amplitude=cfg['noise_factor_range'][0],
                    max_amplitude=cfg['noise_factor_range'][1],
                    p=cfg['prob']
                )
            )
        
        # Add time shifting
        transforms.append(
            AA.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.3)
        )
        
        self.audiomentations_transform = AA.Compose(transforms)
    
    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Apply audio augmentation
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            **kwargs: Override parameters
            
        Returns:
            Tuple of (augmented_audio, sample_rate)
        """
        augment_prob = kwargs.get('augmentation_prob', self.augmentation_prob)
        
        # Skip augmentation based on probability
        if random.random() > augment_prob:
            return audio, sample_rate
        
        if len(audio) == 0:
            return audio, sample_rate
        
        augmented_audio = audio.copy()
        current_sr = sample_rate
        
        try:
            # Apply audiomentations if available
            if self.audiomentations_available and hasattr(self, 'audiomentations_transform'):
                augmented_audio = self.audiomentations_transform(
                    samples=augmented_audio, sample_rate=current_sr
                )
            
            # Apply custom augmentations
            augmented_audio, current_sr = self._apply_time_stretch(augmented_audio, current_sr)
            augmented_audio, current_sr = self._apply_pitch_shift(augmented_audio, current_sr)
            augmented_audio = self._apply_frequency_masking(augmented_audio, current_sr)
            augmented_audio = self._apply_time_masking(augmented_audio, current_sr)
            
            return augmented_audio, current_sr
            
        except Exception as e:
            warnings.warn(f"Audio augmentation failed: {str(e)}")
            return audio, sample_rate
    
    def _apply_time_stretch(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """Apply time stretching"""
        if 'time_stretch' not in self.config or not HAS_AUDIO_LIBS:
            return audio, sample_rate
        
        cfg = self.config['time_stretch']
        if random.random() > cfg['prob']:
            return audio, sample_rate
        
        rate = random.uniform(*cfg['rate_range'])
        stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
        return stretched_audio, sample_rate
    
    def _apply_pitch_shift(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """Apply pitch shifting"""
        if 'pitch_shift' not in self.config or not HAS_AUDIO_LIBS:
            return audio, sample_rate
        
        cfg = self.config['pitch_shift']
        if random.random() > cfg['prob']:
            return audio, sample_rate
        
        n_steps = random.uniform(*cfg['semitone_range'])
        shifted_audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
        return shifted_audio, sample_rate
    
    def _apply_frequency_masking(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply frequency masking in spectral domain"""
        if 'frequency_masking' not in self.config or not HAS_AUDIO_LIBS:
            return audio
        
        cfg = self.config['frequency_masking']
        if random.random() > cfg['prob']:
            return audio
        
        # Compute spectrogram
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply frequency masking
        freq_bins = magnitude.shape[0]
        mask_param = cfg['freq_mask_param']
        mask_size = random.randint(0, min(mask_param, freq_bins // 4))
        mask_start = random.randint(0, freq_bins - mask_size)
        
        masked_magnitude = magnitude.copy()
        masked_magnitude[mask_start:mask_start + mask_size, :] *= 0.1
        
        # Reconstruct audio
        masked_stft = masked_magnitude * np.exp(1j * phase)
        masked_audio = librosa.istft(masked_stft, length=len(audio))
        
        return masked_audio
    
    def _apply_time_masking(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply time masking"""
        if 'time_masking' not in self.config:
            return audio
        
        cfg = self.config['time_masking']
        if random.random() > cfg['prob']:
            return audio
        
        # Apply time masking directly in time domain
        time_samples = len(audio)
        mask_param = cfg['time_mask_param']
        mask_size = random.randint(0, min(mask_param * sample_rate // 1000, time_samples // 4))
        mask_start = random.randint(0, time_samples - mask_size)
        
        masked_audio = audio.copy()
        masked_audio[mask_start:mask_start + mask_size] *= 0.1
        
        return masked_audio
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'augmentation_prob': self.augmentation_prob,
            'config': self.config,
            'random_seed': self.random_seed
        }


class AudioSegmenter(AudioPreprocessorBase):
    """Segment audio into fixed-length chunks"""
    
    def __init__(self, 
                 segment_duration: float = 3.0,
                 hop_duration: Optional[float] = None,
                 min_segment_duration: float = 1.0,
                 pad_mode: str = 'constant'):
        """
        Initialize audio segmenter
        
        Args:
            segment_duration: Target segment duration in seconds
            hop_duration: Hop between segments (None for non-overlapping)
            min_segment_duration: Minimum segment duration to keep
            pad_mode: Padding mode for short segments ('constant', 'reflect', 'wrap')
        """
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration or segment_duration
        self.min_segment_duration = min_segment_duration
        self.pad_mode = pad_mode
    
    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> List[np.ndarray]:
        """
        Segment audio into chunks
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            **kwargs: Override parameters
            
        Returns:
            List of audio segments
        """
        segment_duration = kwargs.get('segment_duration', self.segment_duration)
        hop_duration = kwargs.get('hop_duration', self.hop_duration)
        min_segment_duration = kwargs.get('min_segment_duration', self.min_segment_duration)
        
        if len(audio) == 0:
            return []
        
        segment_samples = int(segment_duration * sample_rate)
        hop_samples = int(hop_duration * sample_rate)
        min_samples = int(min_segment_duration * sample_rate)
        
        segments = []
        
        # Extract segments
        for start in range(0, len(audio), hop_samples):
            end = start + segment_samples
            
            if start + min_samples > len(audio):
                # Remaining audio is too short
                break
            
            segment = audio[start:end]
            
            # Pad if segment is shorter than target
            if len(segment) < segment_samples:
                if len(segment) >= min_samples:
                    # Pad the segment
                    pad_length = segment_samples - len(segment)
                    if self.pad_mode == 'constant':
                        segment = np.pad(segment, (0, pad_length), mode='constant')
                    elif self.pad_mode == 'reflect':
                        segment = np.pad(segment, (0, pad_length), mode='reflect')
                    elif self.pad_mode == 'wrap':
                        segment = np.pad(segment, (0, pad_length), mode='wrap')
                    else:
                        # Zero padding as fallback
                        segment = np.pad(segment, (0, pad_length), mode='constant')
                else:
                    # Skip segment if too short
                    continue
            
            segments.append(segment)
        
        return segments
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'segment_duration': self.segment_duration,
            'hop_duration': self.hop_duration,
            'min_segment_duration': self.min_segment_duration,
            'pad_mode': self.pad_mode
        }


class AudioPreprocessingPipeline:
    """Complete audio preprocessing pipeline with caching"""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 cache_dir: Optional[Union[str, Path]] = None,
                 use_cache: bool = True):
        """
        Initialize preprocessing pipeline
        
        Args:
            config: Pipeline configuration
            cache_dir: Directory for caching processed audio
            use_cache: Whether to use caching
        """
        self.config = config or self._get_default_config()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache and self.cache_dir is not None
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processing components
        self.components = self._build_pipeline()
        
        # Pipeline statistics
        self.stats = {
            'files_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': []
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return {
            'loader': {
                'target_sample_rate': 16000,
                'mono': True,
                'normalize_on_load': False
            },
            'normalizer': {
                'method': 'rms',
                'target_level': 0.1
            },
            'silence_trimmer': {
                'top_db': 30,
                'margin_seconds': 0.1
            },
            'noise_reducer': {
                'enabled': False,
                'noise_factor': 0.1
            },
            'augmenter': {
                'enabled': False,
                'augmentation_prob': 0.5
            },
            'segmenter': {
                'enabled': False,
                'segment_duration': 3.0
            }
        }
    
    def _build_pipeline(self) -> Dict[str, AudioPreprocessorBase]:
        """Build preprocessing components"""
        components = {}
        
        # Audio loader
        if 'loader' in self.config:
            components['loader'] = AudioLoader(**self.config['loader'])
        
        # Normalizer
        if 'normalizer' in self.config:
            components['normalizer'] = AudioNormalizer(**self.config['normalizer'])
        
        # Silence trimmer
        if 'silence_trimmer' in self.config:
            components['silence_trimmer'] = SilenceTrimmer(**self.config['silence_trimmer'])
        
        # Noise reducer
        if 'noise_reducer' in self.config and self.config['noise_reducer'].get('enabled', False):
            noise_config = {k: v for k, v in self.config['noise_reducer'].items() if k != 'enabled'}
            components['noise_reducer'] = NoiseReducer(**noise_config)
        
        # Augmenter
        if 'augmenter' in self.config and self.config['augmenter'].get('enabled', False):
            aug_config = {k: v for k, v in self.config['augmenter'].items() if k != 'enabled'}
            components['augmenter'] = AudioAugmenter(**aug_config)
        
        # Segmenter
        if 'segmenter' in self.config and self.config['segmenter'].get('enabled', False):
            seg_config = {k: v for k, v in self.config['segmenter'].items() if k != 'enabled'}
            components['segmenter'] = AudioSegmenter(**seg_config)
        
        return components
    
    def _get_cache_key(self, file_path: Union[str, Path], config_hash: str) -> str:
        """Generate cache key for file and configuration"""
        file_path_str = str(Path(file_path).resolve())
        file_stats = Path(file_path).stat()
        
        # Combine file path, modification time, size, and config
        key_data = f"{file_path_str}_{file_stats.st_mtime}_{file_stats.st_size}_{config_hash}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_config_hash(self) -> str:
        """Get hash of current configuration"""
        config_str = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[Tuple[Union[np.ndarray, List[np.ndarray]], int]]:
        """Load processed audio from cache"""
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.stats['cache_hits'] += 1
            return cached_data
            
        except Exception as e:
            warnings.warn(f"Failed to load from cache: {str(e)}")
            return None
    
    def _save_to_cache(self, cache_key: str, 
                      data: Union[np.ndarray, List[np.ndarray]], 
                      sample_rate: int) -> None:
        """Save processed audio to cache"""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((data, sample_rate), f)
                
        except Exception as e:
            warnings.warn(f"Failed to save to cache: {str(e)}")
    
    def process_file(self, file_path: Union[str, Path], 
                    **override_params) -> Tuple[Union[np.ndarray, List[np.ndarray]], int]:
        """
        Process a single audio file through the pipeline
        
        Args:
            file_path: Path to audio file
            **override_params: Parameters to override in components
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        start_time = datetime.now()
        
        # Check cache
        config_hash = self._get_config_hash()
        cache_key = self._get_cache_key(file_path, config_hash)
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['processing_times'].append(processing_time)
            return cached_result
        
        self.stats['cache_misses'] += 1
        
        try:
            # Load audio
            if 'loader' in self.components:
                audio, sample_rate = self.components['loader'].process(
                    file_path, **override_params.get('loader', {})
                )
            else:
                # Fallback direct loading
                if not HAS_AUDIO_LIBS:
                    raise ImportError("Audio libraries required for processing")
                audio, sample_rate = librosa.load(str(file_path))
            
            # Apply preprocessing steps
            processing_order = ['normalizer', 'silence_trimmer', 'noise_reducer', 'augmenter']
            
            for component_name in processing_order:
                if component_name in self.components:
                    component_params = override_params.get(component_name, {})
                    audio, sample_rate = self.components[component_name].process(
                        audio, sample_rate, **component_params
                    )
            
            # Apply segmentation last (changes return type)
            if 'segmenter' in self.components:
                segments = self.components['segmenter'].process(
                    audio, sample_rate, **override_params.get('segmenter', {})
                )
                result = (segments, sample_rate)
            else:
                result = (audio, sample_rate)
            
            # Cache result
            self._save_to_cache(cache_key, result[0], result[1])
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['processing_times'].append(processing_time)
            self.stats['files_processed'] += 1
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to process {file_path}: {str(e)}")
    
    def process_batch(self, file_paths: List[Union[str, Path]], 
                     show_progress: bool = True,
                     **override_params) -> List[Tuple[Union[np.ndarray, List[np.ndarray]], int]]:
        """
        Process a batch of audio files
        
        Args:
            file_paths: List of file paths to process
            show_progress: Whether to show progress bar
            **override_params: Parameters to override in components
            
        Returns:
            List of (processed_audio, sample_rate) tuples
        """
        results = []
        
        # Setup progress bar
        if show_progress and HAS_TQDM:
            file_iterator = tqdm(file_paths, desc="Processing audio files")
        else:
            file_iterator = file_paths
        
        for file_path in file_iterator:
            try:
                result = self.process_file(file_path, **override_params)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Failed to process {file_path}: {str(e)}")
                results.append((None, None))  # Placeholder for failed processing
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['total_processing_time'] = np.sum(stats['processing_times'])
        
        if stats['cache_hits'] + stats['cache_misses'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear processing cache"""
        if not self.use_cache:
            return
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                warnings.warn(f"Failed to delete cache file {cache_file}: {str(e)}")
        
        print(f"Cleared {len(cache_files)} cache files")
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get complete pipeline configuration"""
        return {
            'config': self.config,
            'components': {name: comp.get_config() for name, comp in self.components.items()},
            'cache_enabled': self.use_cache,
            'cache_dir': str(self.cache_dir) if self.cache_dir else None
        }


# Convenience functions
def create_preprocessing_pipeline(config: Optional[Dict[str, Any]] = None,
                                cache_dir: Optional[str] = None) -> AudioPreprocessingPipeline:
    """Create a preprocessing pipeline with given configuration"""
    return AudioPreprocessingPipeline(config=config, cache_dir=cache_dir)


def preprocess_audio_file(file_path: Union[str, Path],
                         target_sample_rate: int = 16000,
                         normalize: bool = True,
                         trim_silence: bool = True) -> Tuple[np.ndarray, int]:
    """Quick audio preprocessing function"""
    config = {
        'loader': {
            'target_sample_rate': target_sample_rate,
            'mono': True
        },
        'normalizer': {
            'method': 'rms',
            'target_level': 0.1
        } if normalize else {},
        'silence_trimmer': {
            'top_db': 30
        } if trim_silence else {}
    }
    
    # Filter out empty configs
    config = {k: v for k, v in config.items() if v}
    
    pipeline = AudioPreprocessingPipeline(config=config, use_cache=False)
    audio, sample_rate = pipeline.process_file(file_path)
    
    return audio, sample_rate


# Example usage and testing
if __name__ == "__main__":
    print("Testing audio preprocessing system...")
    
    if not HAS_AUDIO_LIBS:
        print("Warning: Audio libraries not available. Some tests will be skipped.")
    
    # Test individual components
    print("Testing individual components...")
    
    # Create sample audio for testing
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine wave
    
    # Test normalizer
    normalizer = AudioNormalizer(method='rms', target_level=0.1)
    normalized_audio, sr = normalizer.process(test_audio, sample_rate)
    print(f"✓ AudioNormalizer: RMS before={np.sqrt(np.mean(test_audio**2)):.3f}, after={np.sqrt(np.mean(normalized_audio**2)):.3f}")
    
    # Test augmenter
    augmenter = AudioAugmenter(augmentation_prob=1.0)  # Always augment for testing
    augmented_audio, sr = augmenter.process(test_audio, sample_rate)
    print(f"✓ AudioAugmenter: Original length={len(test_audio)}, augmented length={len(augmented_audio)}")
    
    # Test segmenter
    segmenter = AudioSegmenter(segment_duration=0.5, hop_duration=0.3)
    segments = segmenter.process(test_audio, sample_rate)
    print(f"✓ AudioSegmenter: Created {len(segments)} segments from {duration}s audio")
    
    # Test complete pipeline
    print("Testing complete pipeline...")
    
    pipeline_config = {
        'normalizer': {
            'method': 'rms',
            'target_level': 0.1
        },
        'augmenter': {
            'enabled': True,
            'augmentation_prob': 0.5
        },
        'segmenter': {
            'enabled': True,
            'segment_duration': 1.0
        }
    }
    
    pipeline = AudioPreprocessingPipeline(config=pipeline_config, use_cache=False)
    
    # Since we don't have a real file, we'll test with synthetic data
    print("Pipeline configuration:")
    print(json.dumps(pipeline.get_pipeline_config(), indent=2, default=str))
    
    print("Audio preprocessing system testing completed!")
    
    # Performance statistics
    stats = pipeline.get_statistics()
    if stats['processing_times']:
        print(f"Processing statistics: {stats}")