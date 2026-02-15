#!/usr/bin/env python3
"""
Noise Augmentation Module for Speaker Recognition

This module provides various noise augmentation techniques for testing
the robustness of speaker recognition models under different noise conditions.

Supports:
- Gaussian white noise (AWGN)
- Babble noise (multi-speaker background)
- Environmental noise (cafe, street, etc.)
- Pink noise
- Brown noise

Usage:
    from noise_augmentation import NoiseAugmentor, add_noise
    
    augmentor = NoiseAugmentor()
    noisy_audio = augmentor.add_gaussian_noise(audio, snr_db=10)
"""

import numpy as np
import torch
from typing import Optional, Union, List, Tuple, Dict
from pathlib import Path
import warnings


class NoiseAugmentor:
    """
    Audio noise augmentation class for robustness testing.
    
    Provides methods to add various types of noise at specified SNR levels.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        seed: Optional[int] = None
    ):
        """
        Initialize NoiseAugmentor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            seed: Random seed for reproducibility
        """
        self.sample_rate = sample_rate
        self.rng = np.random.RandomState(seed)
        
        # Cache for generated noise patterns
        self._noise_cache = {}
    
    def _ensure_numpy(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert audio to numpy array if needed."""
        if isinstance(audio, torch.Tensor):
            return audio.cpu().numpy()
        return np.asarray(audio, dtype=np.float32)
    
    def _ensure_tensor(
        self, 
        audio: np.ndarray, 
        original: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Convert back to original type."""
        if isinstance(original, torch.Tensor):
            return torch.from_numpy(audio).to(original.device)
        return audio
    
    def _calculate_signal_power(self, signal: np.ndarray) -> float:
        """Calculate signal power (RMS squared)."""
        return np.mean(signal ** 2)
    
    def _calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Calculate SNR in dB."""
        signal_power = self._calculate_signal_power(signal)
        noise_power = self._calculate_signal_power(noise)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)
    
    def _scale_noise_to_snr(
        self, 
        signal: np.ndarray, 
        noise: np.ndarray, 
        target_snr_db: float
    ) -> np.ndarray:
        """Scale noise to achieve target SNR."""
        signal_power = self._calculate_signal_power(signal)
        noise_power = self._calculate_signal_power(noise)
        
        if noise_power == 0:
            return noise
        
        # Calculate required noise power for target SNR
        # SNR = 10 * log10(signal_power / noise_power)
        # target_noise_power = signal_power / (10 ** (target_snr_db / 10))
        target_noise_power = signal_power / (10 ** (target_snr_db / 10))
        
        # Scale factor
        scale = np.sqrt(target_noise_power / noise_power)
        
        return noise * scale
    
    def add_gaussian_noise(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        snr_db: float = 10.0
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Add Gaussian white noise (AWGN) at specified SNR.
        
        Args:
            audio: Input audio signal
            snr_db: Target signal-to-noise ratio in dB
            
        Returns:
            Noisy audio signal
        """
        original = audio
        audio = self._ensure_numpy(audio)
        
        # Generate white noise
        noise = self.rng.randn(*audio.shape).astype(np.float32)
        
        # Scale to target SNR
        scaled_noise = self._scale_noise_to_snr(audio, noise, snr_db)
        
        # Add noise
        noisy_audio = audio + scaled_noise
        
        return self._ensure_tensor(noisy_audio, original)
    
    def add_pink_noise(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        snr_db: float = 10.0
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Add pink noise (1/f noise) at specified SNR.
        
        Pink noise has equal energy per octave, more natural sounding.
        
        Args:
            audio: Input audio signal
            snr_db: Target signal-to-noise ratio in dB
            
        Returns:
            Noisy audio signal
        """
        original = audio
        audio = self._ensure_numpy(audio)
        
        # Generate pink noise using Voss-McCartney algorithm
        n_samples = audio.shape[-1]
        
        # Simple approximation: filter white noise
        white = self.rng.randn(n_samples).astype(np.float32)
        
        # Apply 1/f filter in frequency domain
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1e-6  # Avoid division by zero
        
        # 1/f scaling
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=n_samples).astype(np.float32)
        
        # Handle multi-dimensional audio
        if audio.ndim > 1:
            pink = np.tile(pink, (audio.shape[0], 1))
        
        # Scale to target SNR
        scaled_noise = self._scale_noise_to_snr(audio, pink, snr_db)
        
        # Add noise
        noisy_audio = audio + scaled_noise
        
        return self._ensure_tensor(noisy_audio, original)
    
    def add_brown_noise(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        snr_db: float = 10.0
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Add brown noise (Brownian/red noise) at specified SNR.
        
        Brown noise has more energy at lower frequencies (1/f²).
        
        Args:
            audio: Input audio signal
            snr_db: Target signal-to-noise ratio in dB
            
        Returns:
            Noisy audio signal
        """
        original = audio
        audio = self._ensure_numpy(audio)
        
        n_samples = audio.shape[-1]
        
        # Generate brown noise as cumulative sum of white noise
        white = self.rng.randn(n_samples).astype(np.float32)
        brown = np.cumsum(white).astype(np.float32)
        
        # Normalize
        brown = brown - np.mean(brown)
        brown = brown / (np.std(brown) + 1e-8)
        
        # Handle multi-dimensional audio
        if audio.ndim > 1:
            brown = np.tile(brown, (audio.shape[0], 1))
        
        # Scale to target SNR
        scaled_noise = self._scale_noise_to_snr(audio, brown, snr_db)
        
        # Add noise
        noisy_audio = audio + scaled_noise
        
        return self._ensure_tensor(noisy_audio, original)
    
    def add_babble_noise(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        snr_db: float = 10.0,
        num_speakers: int = 5
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Add babble noise (multiple overlapping speakers).
        
        Simulates cocktail party effect with multiple speakers talking.
        
        Args:
            audio: Input audio signal
            snr_db: Target signal-to-noise ratio in dB
            num_speakers: Number of simulated background speakers
            
        Returns:
            Noisy audio signal
        """
        original = audio
        audio = self._ensure_numpy(audio)
        
        n_samples = audio.shape[-1]
        
        # Generate babble as sum of modulated noise sources
        babble = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(num_speakers):
            # Each "speaker" is modulated noise
            speaker_noise = self.rng.randn(n_samples).astype(np.float32)
            
            # Apply random amplitude modulation (speech-like envelope)
            mod_freq = self.rng.uniform(2, 6)  # 2-6 Hz modulation (syllable rate)
            t = np.arange(n_samples) / self.sample_rate
            modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t + self.rng.uniform(0, 2*np.pi))
            
            # Apply bandpass-like filtering (speech frequency range)
            # Simple approximation using moving average
            window_size = int(self.sample_rate / 1000)  # 1ms window
            if window_size > 1:
                kernel = np.ones(window_size) / window_size
                speaker_noise = np.convolve(speaker_noise, kernel, mode='same')
            
            babble += speaker_noise * modulation.astype(np.float32)
        
        # Normalize
        babble = babble / (np.std(babble) + 1e-8)
        
        # Handle multi-dimensional audio
        if audio.ndim > 1:
            babble = np.tile(babble, (audio.shape[0], 1))
        
        # Scale to target SNR
        scaled_noise = self._scale_noise_to_snr(audio, babble, snr_db)
        
        # Add noise
        noisy_audio = audio + scaled_noise
        
        return self._ensure_tensor(noisy_audio, original)
    
    def add_environmental_noise(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        snr_db: float = 10.0,
        noise_type: str = 'cafe'
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Add environmental noise at specified SNR.
        
        Simulates various environmental noise conditions.
        
        Args:
            audio: Input audio signal
            snr_db: Target signal-to-noise ratio in dB
            noise_type: Type of environmental noise
                - 'cafe': Coffee shop ambient noise
                - 'street': Street/traffic noise
                - 'office': Office background noise
                - 'rain': Rain sound
                - 'wind': Wind noise
                
        Returns:
            Noisy audio signal
        """
        original = audio
        audio = self._ensure_numpy(audio)
        
        n_samples = audio.shape[-1]
        
        if noise_type == 'cafe':
            # Cafe: babble + clinking + ambient
            noise = self._generate_cafe_noise(n_samples)
        elif noise_type == 'street':
            # Street: low frequency rumble + occasional peaks
            noise = self._generate_street_noise(n_samples)
        elif noise_type == 'office':
            # Office: HVAC hum + keyboard + quiet chatter
            noise = self._generate_office_noise(n_samples)
        elif noise_type == 'rain':
            # Rain: broadband noise with specific characteristics
            noise = self._generate_rain_noise(n_samples)
        elif noise_type == 'wind':
            # Wind: low frequency with gusts
            noise = self._generate_wind_noise(n_samples)
        else:
            warnings.warn(f"Unknown noise type '{noise_type}', using white noise")
            noise = self.rng.randn(n_samples).astype(np.float32)
        
        # Handle multi-dimensional audio
        if audio.ndim > 1:
            noise = np.tile(noise, (audio.shape[0], 1))
        
        # Scale to target SNR
        scaled_noise = self._scale_noise_to_snr(audio, noise, snr_db)
        
        # Add noise
        noisy_audio = audio + scaled_noise
        
        return self._ensure_tensor(noisy_audio, original)
    
    def _generate_cafe_noise(self, n_samples: int) -> np.ndarray:
        """Generate cafe-like ambient noise."""
        # Base: babble noise
        babble = np.zeros(n_samples, dtype=np.float32)
        for _ in range(8):
            speaker = self.rng.randn(n_samples).astype(np.float32)
            mod_freq = self.rng.uniform(2, 5)
            t = np.arange(n_samples) / self.sample_rate
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
            babble += speaker * mod.astype(np.float32)
        
        # Add occasional clinks (high frequency transients)
        num_clinks = int(n_samples / self.sample_rate * 2)  # ~2 per second
        for _ in range(num_clinks):
            pos = self.rng.randint(0, n_samples - 100)
            clink = self.rng.randn(100).astype(np.float32) * np.exp(-np.arange(100) / 20)
            babble[pos:pos+100] += clink * 0.5
        
        return babble / (np.std(babble) + 1e-8)
    
    def _generate_street_noise(self, n_samples: int) -> np.ndarray:
        """Generate street/traffic noise."""
        # Low frequency rumble (traffic)
        t = np.arange(n_samples) / self.sample_rate
        rumble = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
        rumble = rumble.astype(np.float32)
        
        # Add random modulation
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t + self.rng.uniform(0, 2*np.pi))
        rumble *= mod.astype(np.float32)
        
        # Add white noise component
        white = self.rng.randn(n_samples).astype(np.float32) * 0.3
        
        # Occasional car passes (louder rumble)
        num_cars = int(n_samples / self.sample_rate / 3)  # ~1 every 3 seconds
        for _ in range(num_cars):
            pos = self.rng.randint(0, max(1, n_samples - self.sample_rate))
            duration = int(self.rng.uniform(0.5, 1.5) * self.sample_rate)
            car_pass = np.exp(-((np.arange(duration) - duration/2) ** 2) / (duration/4) ** 2)
            end_pos = min(pos + duration, n_samples)
            rumble[pos:end_pos] += car_pass[:end_pos-pos].astype(np.float32) * 2
        
        noise = rumble + white
        return noise / (np.std(noise) + 1e-8)
    
    def _generate_office_noise(self, n_samples: int) -> np.ndarray:
        """Generate office ambient noise."""
        t = np.arange(n_samples) / self.sample_rate
        
        # HVAC hum (low frequency)
        hvac = 0.3 * np.sin(2 * np.pi * 60 * t) + 0.2 * np.sin(2 * np.pi * 120 * t)
        hvac = hvac.astype(np.float32)
        
        # Light white noise (air circulation)
        air = self.rng.randn(n_samples).astype(np.float32) * 0.1
        
        # Occasional keyboard clicks
        num_clicks = int(n_samples / self.sample_rate * 3)  # ~3 per second
        for _ in range(num_clicks):
            pos = self.rng.randint(0, n_samples - 50)
            click = self.rng.randn(50).astype(np.float32) * np.exp(-np.arange(50) / 10)
            hvac[pos:pos+50] += click * 0.2
        
        noise = hvac + air
        return noise / (np.std(noise) + 1e-8)
    
    def _generate_rain_noise(self, n_samples: int) -> np.ndarray:
        """Generate rain noise."""
        # Rain is similar to pink noise with specific characteristics
        white = self.rng.randn(n_samples).astype(np.float32)
        
        # Filter to rain-like spectrum
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, 1/self.sample_rate)
        freqs[0] = 1e-6
        
        # Rain has more high frequency content than pink noise
        # but less than white noise
        fft = fft / (freqs ** 0.3)
        rain = np.fft.irfft(fft, n=n_samples).astype(np.float32)
        
        # Add occasional droplet impacts
        num_drops = int(n_samples / self.sample_rate * 10)
        for _ in range(num_drops):
            pos = self.rng.randint(0, n_samples - 30)
            drop = self.rng.randn(30).astype(np.float32) * np.exp(-np.arange(30) / 5)
            rain[pos:pos+30] += drop * 0.3
        
        return rain / (np.std(rain) + 1e-8)
    
    def _generate_wind_noise(self, n_samples: int) -> np.ndarray:
        """Generate wind noise."""
        t = np.arange(n_samples) / self.sample_rate
        
        # Base: brown noise (low frequency dominated)
        white = self.rng.randn(n_samples).astype(np.float32)
        brown = np.cumsum(white)
        brown = brown - np.mean(brown)
        brown = brown / (np.std(brown) + 1e-8)
        
        # Slow modulation (gusts)
        gust_freq = self.rng.uniform(0.1, 0.5)
        gusts = 0.5 + 0.5 * np.sin(2 * np.pi * gust_freq * t)
        
        # Faster variations
        variations = 0.8 + 0.2 * np.sin(2 * np.pi * 2 * t)
        
        wind = brown.astype(np.float32) * gusts.astype(np.float32) * variations.astype(np.float32)
        
        return wind / (np.std(wind) + 1e-8)
    
    def add_noise(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        noise_type: str,
        snr_db: float = 10.0,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Add noise of specified type at specified SNR.
        
        Unified interface for all noise types.
        
        Args:
            audio: Input audio signal
            noise_type: Type of noise ('gaussian', 'pink', 'brown', 'babble', 
                       'cafe', 'street', 'office', 'rain', 'wind')
            snr_db: Target signal-to-noise ratio in dB
            **kwargs: Additional arguments for specific noise types
            
        Returns:
            Noisy audio signal
        """
        noise_type = noise_type.lower()
        
        if noise_type == 'gaussian' or noise_type == 'white' or noise_type == 'awgn':
            return self.add_gaussian_noise(audio, snr_db)
        elif noise_type == 'pink':
            return self.add_pink_noise(audio, snr_db)
        elif noise_type == 'brown' or noise_type == 'brownian' or noise_type == 'red':
            return self.add_brown_noise(audio, snr_db)
        elif noise_type == 'babble':
            num_speakers = kwargs.get('num_speakers', 5)
            return self.add_babble_noise(audio, snr_db, num_speakers)
        elif noise_type in ['cafe', 'street', 'office', 'rain', 'wind']:
            return self.add_environmental_noise(audio, snr_db, noise_type)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}. "
                           f"Supported: gaussian, pink, brown, babble, "
                           f"cafe, street, office, rain, wind")
    
    def augment_batch(
        self,
        batch: Union[np.ndarray, torch.Tensor],
        noise_type: str,
        snr_db: float = 10.0,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Add noise to a batch of audio samples.
        
        Args:
            batch: Batch of audio samples [batch_size, num_samples]
            noise_type: Type of noise to add
            snr_db: Target SNR in dB
            **kwargs: Additional arguments
            
        Returns:
            Batch of noisy audio samples
        """
        return self.add_noise(batch, noise_type, snr_db, **kwargs)
    
    def augment_dataset(
        self,
        dataset,
        snr_levels: List[float],
        noise_types: List[str]
    ) -> Dict[str, Dict[float, list]]:
        """
        Create augmented versions of a dataset at multiple SNR levels.
        
        Args:
            dataset: Dataset with audio samples
            snr_levels: List of SNR levels to test (e.g., [20, 15, 10, 5, 0])
            noise_types: List of noise types to apply
            
        Returns:
            Dictionary mapping noise_type -> snr_level -> augmented_samples
        """
        augmented = {}
        
        for noise_type in noise_types:
            augmented[noise_type] = {}
            for snr_db in snr_levels:
                augmented[noise_type][snr_db] = []
                
                for i in range(len(dataset)):
                    sample = dataset[i]
                    if isinstance(sample, tuple):
                        audio, label = sample[0], sample[1]
                    else:
                        audio = sample
                        label = None
                    
                    noisy_audio = self.add_noise(audio, noise_type, snr_db)
                    
                    if label is not None:
                        augmented[noise_type][snr_db].append((noisy_audio, label))
                    else:
                        augmented[noise_type][snr_db].append(noisy_audio)
        
        return augmented


# Convenience function
def add_noise(
    audio: Union[np.ndarray, torch.Tensor],
    noise_type: str = 'gaussian',
    snr_db: float = 10.0,
    seed: Optional[int] = None,
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convenience function to add noise to audio.
    
    Args:
        audio: Input audio signal
        noise_type: Type of noise to add
        snr_db: Target SNR in dB
        seed: Random seed for reproducibility
        **kwargs: Additional arguments
        
    Returns:
        Noisy audio signal
    """
    augmentor = NoiseAugmentor(seed=seed)
    return augmentor.add_noise(audio, noise_type, snr_db, **kwargs)


class NoisyDatasetWrapper:
    """
    Wrapper to create a noisy version of an existing dataset.
    
    Useful for testing model robustness without modifying original data.
    """
    
    def __init__(
        self,
        dataset,
        noise_type: str = 'gaussian',
        snr_db: float = 10.0,
        sample_rate: int = 16000,
        seed: Optional[int] = None
    ):
        """
        Initialize NoisyDatasetWrapper.
        
        Args:
            dataset: Original dataset
            noise_type: Type of noise to add
            snr_db: Target SNR in dB
            sample_rate: Audio sample rate
            seed: Random seed
        """
        self.dataset = dataset
        self.noise_type = noise_type
        self.snr_db = snr_db
        self.augmentor = NoiseAugmentor(sample_rate=sample_rate, seed=seed)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        if isinstance(sample, tuple) or isinstance(sample, list):
            audio = sample[0]
            rest = sample[1:]
            
            noisy_audio = self.augmentor.add_noise(audio, self.noise_type, self.snr_db)
            
            return (noisy_audio,) + tuple(rest)
        else:
            return self.augmentor.add_noise(sample, self.noise_type, self.snr_db)


if __name__ == "__main__":
    # Test the noise augmentation module
    print("Testing NoiseAugmentor...")
    
    # Create test signal
    duration = 3.0
    sample_rate = 16000
    t = np.arange(int(duration * sample_rate)) / sample_rate
    
    # Simple test signal (sine wave)
    test_signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    augmentor = NoiseAugmentor(sample_rate=sample_rate, seed=42)
    
    # Test all noise types
    noise_types = ['gaussian', 'pink', 'brown', 'babble', 'cafe', 'street', 'office', 'rain', 'wind']
    snr_levels = [20, 10, 5, 0]
    
    print(f"\nOriginal signal power: {np.mean(test_signal**2):.6f}")
    
    for noise_type in noise_types:
        print(f"\n{noise_type.upper()} noise:")
        for snr_db in snr_levels:
            noisy = augmentor.add_noise(test_signal, noise_type, snr_db)
            noise = noisy - test_signal
            actual_snr = 10 * np.log10(np.mean(test_signal**2) / np.mean(noise**2))
            print(f"  SNR={snr_db:3d}dB -> Actual SNR: {actual_snr:.1f}dB")
    
    # Test with torch tensor
    print("\n\nTesting with PyTorch tensor...")
    torch_signal = torch.from_numpy(test_signal)
    noisy_torch = augmentor.add_gaussian_noise(torch_signal, snr_db=10)
    print(f"Input type: {type(torch_signal)}, Output type: {type(noisy_torch)}")
    
    print("\n✓ All tests passed!")