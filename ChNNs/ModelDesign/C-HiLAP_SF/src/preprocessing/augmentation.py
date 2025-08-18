import numpy as np
import librosa
import random
import torch
import torch.nn.functional as F
from scipy.signal import convolve
from typing import Tuple, List, Dict, Union, Optional


class AudioAugmentor:
    """
    Advanced audio augmentation toolkit for speaker recognition robustness.
    Implements various signal transformations to simulate real-world conditions.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 noise_dir: Optional[str] = None,
                 rir_dir: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize the augmentor with configuration parameters.

        Args:
            sample_rate: Target sample rate for all audio processing
            noise_dir: Directory containing noise samples (optional)
            rir_dir: Directory containing room impulse responses (optional)
            device: Computation device ('cpu' or 'cuda')
        """
        self.sample_rate = sample_rate
        self.noise_dir = noise_dir
        self.rir_dir = rir_dir
        self.device = device

        # Preload noise samples if directory is provided
        self.noise_samples = []
        if noise_dir:
            self._load_noise_samples()

        # Preload RIR samples if directory is provided
        self.rir_samples = []
        if rir_dir:
            self._load_rir_samples()

    def _load_noise_samples(self):
        """Load noise samples from the specified directory"""
        import os
        import glob

        noise_files = glob.glob(os.path.join(self.noise_dir, "*.wav"))
        for noise_file in noise_files:
            try:
                noise, sr = librosa.load(noise_file, sr=self.sample_rate)
                if len(noise) > 0.5 * self.sample_rate:  # At least 0.5 seconds
                    self.noise_samples.append(noise)
            except Exception as e:
                print(f"Error loading noise file {noise_file}: {e}")

        print(f"Loaded {len(self.noise_samples)} noise samples")

    def _load_rir_samples(self):
        """Load room impulse response samples from the specified directory"""
        import os
        import glob

        rir_files = glob.glob(os.path.join(self.rir_dir, "*.wav"))
        for rir_file in rir_files:
            try:
                rir, sr = librosa.load(rir_file, sr=self.sample_rate)
                self.rir_samples.append(rir)
            except Exception as e:
                print(f"Error loading RIR file {rir_file}: {e}")

        print(f"Loaded {len(self.rir_samples)} RIR samples")

    def add_noise(self,
                  signal: np.ndarray,
                  snr_db: float = 10.0,
                  noise_type: str = "gaussian") -> np.ndarray:
        """
        Add noise to the audio signal at specified SNR level.

        Args:
            signal: Clean audio signal
            snr_db: Signal-to-Noise Ratio in decibels
            noise_type: Type of noise ('gaussian', 'babble', 'sample')

        Returns:
            Noisy audio signal
        """
        # Calculate signal power
        signal_power = np.mean(signal ** 2)

        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)

        # Calculate target noise power
        noise_power_target = signal_power / snr_linear

        if noise_type == "gaussian":
            # Generate Gaussian noise
            noise = np.random.normal(0, 1, size=len(signal))

        elif noise_type == "babble":
            # Generate colored noise (approximating babble)
            noise = np.random.normal(0, 1, size=len(signal))
            # Apply low-pass filter to simulate babble noise spectrum
            b = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            noise = convolve(noise, b, mode='same')

        elif noise_type == "sample" and self.noise_samples:
            # Use a random pre-loaded noise sample
            noise_idx = random.randint(0, len(self.noise_samples) - 1)
            noise_sample = self.noise_samples[noise_idx]

            # If noise sample is too short, repeat it
            if len(noise_sample) < len(signal):
                repeats = int(np.ceil(len(signal) / len(noise_sample)))
                noise_sample = np.tile(noise_sample, repeats)

            # Extract a random segment of appropriate length
            start = random.randint(0, len(noise_sample) - len(signal))
            noise = noise_sample[start:start + len(signal)]
        else:
            # Default to gaussian if no valid option
            noise = np.random.normal(0, 1, size=len(signal))

        # Scale noise to desired power
        current_noise_power = np.mean(noise ** 2)
        noise = noise * np.sqrt(noise_power_target / (current_noise_power + 1e-10))

        # Add noise to signal
        noisy_signal = signal + noise

        # Normalize to prevent clipping
        max_val = np.max(np.abs(noisy_signal))
        if max_val > 1.0:
            noisy_signal = noisy_signal / max_val

        return noisy_signal

    def apply_reverb(self,
                     signal: np.ndarray,
                     reverb_level: float = 0.3) -> np.ndarray:
        """
        Apply reverberation to the audio signal.

        Args:
            signal: Clean audio signal
            reverb_level: Strength of reverberation (0.0 to 1.0)

        Returns:
            Reverberated audio signal
        """
        if not self.rir_samples:
            # Synthetic reverb if no RIR samples available
            reverb_time = reverb_level * 0.5  # Max 500ms reverb
            reverb_samples = int(reverb_time * self.sample_rate)

            # Create a simple exponential decay
            decay = np.exp(-np.arange(reverb_samples) / (reverb_samples * 0.1))
            reverb_signal = convolve(signal, decay, mode='full')[:len(signal)]

            # Mix with original signal
            result = (1 - reverb_level) * signal + reverb_level * reverb_signal

        else:
            # Use a random RIR sample
            rir_idx = random.randint(0, len(self.rir_samples) - 1)
            rir = self.rir_samples[rir_idx]

            # Normalize RIR
            rir = rir / np.max(np.abs(rir))

            # Apply RIR through convolution
            reverb_signal = convolve(signal, rir, mode='full')[:len(signal)]

            # Mix with original signal based on reverb level
            result = (1 - reverb_level) * signal + reverb_level * reverb_signal

        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val

        return result

    def time_stretch(self,
                     signal: np.ndarray,
                     rate: float = 1.0) -> np.ndarray:
        """
        Apply time stretching to the audio signal.

        Args:
            signal: Input audio signal
            rate: Stretching factor (0.8 = faster, 1.2 = slower)

        Returns:
            Time-stretched audio signal
        """
        return librosa.effects.time_stretch(signal, rate=rate)

    def pitch_shift(self,
                    signal: np.ndarray,
                    n_steps: float = 0.0) -> np.ndarray:
        """
        Apply pitch shifting to the audio signal.

        Args:
            signal: Input audio signal
            n_steps: Number of semitones to shift (-2.0 to 2.0)

        Returns:
            Pitch-shifted audio signal
        """
        return librosa.effects.pitch_shift(
            signal,
            sr=self.sample_rate,
            n_steps=n_steps
        )

    def apply_codec_distortion(self,
                               signal: np.ndarray,
                               quality: float = 0.7) -> np.ndarray:
        """
        Simulate codec distortion by applying MP3-like compression artifacts.

        Args:
            signal: Input audio signal
            quality: Quality factor (0.1 = low quality, 1.0 = high quality)

        Returns:
            Distorted audio signal
        """
        # Convert to torch tensor
        signal_tensor = torch.tensor(signal).float().to(self.device)

        # Apply low-pass filter to simulate codec artifacts
        cutoff_freq = int(quality * self.sample_rate / 4)  # Nyquist limit
        filter_length = 101

        # Create low-pass filter
        h = np.sinc(2 * cutoff_freq * (np.arange(filter_length) - (filter_length - 1) / 2) / self.sample_rate)
        h = h * np.hamming(filter_length)
        h = h / np.sum(h)

        # Convert filter to tensor
        h_tensor = torch.tensor(h).float().to(self.device)

        # Apply filtering
        signal_tensor = signal_tensor.view(1, 1, -1)
        h_tensor = h_tensor.view(1, 1, -1)
        filtered = F.conv1d(signal_tensor, h_tensor, padding=filter_length // 2)

        # Add quantization noise
        bits = int(2 + quality * 14)  # 2-16 bits
        max_val = torch.max(torch.abs(filtered))
        if max_val > 0:
            filtered = filtered / max_val
        steps = 2 ** bits
        filtered = torch.round(filtered * steps) / steps

        # Convert back to numpy
        result = filtered.view(-1).cpu().numpy()

        return result

    def apply_clipping(self,
                       signal: np.ndarray,
                       clip_factor: float = 0.8) -> np.ndarray:
        """
        Apply soft clipping distortion to simulate overload.

        Args:
            signal: Input audio signal
            clip_factor: Clipping threshold (0.1 = heavy, 0.9 = light)

        Returns:
            Clipped audio signal
        """
        # Normalize signal
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            normalized = signal / max_val
        else:
            return signal

        # Apply soft clipping using tanh function
        threshold = clip_factor
        clipped = np.zeros_like(normalized)

        # Linear region
        mask = np.abs(normalized) <= threshold
        clipped[mask] = normalized[mask]

        # Soft clipping region
        mask = np.abs(normalized) > threshold
        clipped[mask] = np.sign(normalized[mask]) * (
                threshold + (1 - threshold) * np.tanh(
            (np.abs(normalized[mask]) - threshold) / (1 - threshold)
        )
        )

        # Restore original scale
        return clipped * max_val

    def apply_band_reject(self,
                          signal: np.ndarray,
                          center_freq: float = 1000.0,
                          bandwidth: float = 200.0) -> np.ndarray:
        """
        Apply band-reject filter to simulate frequency masking.

        Args:
            signal: Input audio signal
            center_freq: Center frequency of the rejected band (Hz)
            bandwidth: Width of the rejected band (Hz)

        Returns:
            Filtered audio signal
        """
        # Convert to frequency domain
        stft = librosa.stft(signal)

        # Calculate frequency bins
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2 * (stft.shape[0] - 1))

        # Create band-reject mask
        lower_bin = np.argmin(np.abs(freqs - (center_freq - bandwidth / 2)))
        upper_bin = np.argmin(np.abs(freqs - (center_freq + bandwidth / 2)))

        # Apply smooth rejection (Hann window)
        mask = np.ones(stft.shape[0])
        reject_width = upper_bin - lower_bin
        if reject_width > 0:
            hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(reject_width) / reject_width)
            mask[lower_bin:upper_bin] = 1.0 - hann

        # Apply mask to STFT
        stft = stft * mask[:, np.newaxis]

        # Convert back to time domain
        filtered = librosa.istft(stft, length=len(signal))

        return filtered

    def generate_adversarial_fgsm(self,
                                  signal: np.ndarray,
                                  model: torch.nn.Module,
                                  target_id: int = None,
                                  epsilon: float = 0.01) -> np.ndarray:
        """
        Generate adversarial example using Fast Gradient Sign Method.

        Args:
            signal: Original audio signal
            model: PyTorch model to attack
            target_id: Target speaker ID (None for untargeted attack)
            epsilon: Perturbation magnitude

        Returns:
            Adversarial audio signal
        """
        # Ensure model is in evaluation mode
        model.eval()

        # Convert signal to tensor with gradient tracking
        signal_tensor = torch.tensor(signal, requires_grad=True).float().to(self.device)
        signal_tensor = signal_tensor.view(1, 1, -1)  # Add batch and channel dims

        # Forward pass
        output = model(signal_tensor)

        # Determine target
        if target_id is not None:
            # Targeted attack: minimize target class score
            target = torch.tensor([target_id]).long().to(self.device)
            loss = -F.cross_entropy(output, target)
        else:
            # Untargeted attack: maximize original class score
            _, predicted = torch.max(output, 1)
            loss = F.cross_entropy(output, predicted)

        # Backward pass to get gradients
        model.zero_grad()
        loss.backward()

        # Create perturbation using gradient sign
        perturbation = epsilon * torch.sign(signal_tensor.grad.data)

        # Apply perturbation
        adversarial = signal_tensor + perturbation

        # Ensure values stay in valid range
        adversarial = torch.clamp(adversarial, -1.0, 1.0)

        # Convert back to numpy
        return adversarial.view(-1).detach().cpu().numpy()

    def augment_batch(self,
                      signals: List[np.ndarray],
                      augmentation_chain: List[Dict] = None) -> List[np.ndarray]:
        """
        Apply a chain of augmentations to a batch of signals.

        Args:
            signals: List of audio signals
            augmentation_chain: List of augmentation specifications
                [{'type': 'noise', 'params': {'snr_db': 10.0}}, ...]

        Returns:
            List of augmented audio signals
        """
        if augmentation_chain is None:
            # Default augmentation chain
            augmentation_chain = [
                {'type': 'noise', 'params': {'snr_db': random.uniform(5, 20)}},
                {'type': 'reverb', 'params': {'reverb_level': random.uniform(0.1, 0.5)}},
                {'type': 'time_stretch', 'params': {'rate': random.uniform(0.9, 1.1)}}
            ]

        results = []
        for signal in signals:
            augmented = signal.copy()

            for aug in augmentation_chain:
                aug_type = aug['type']
                params = aug['params']

                if aug_type == 'noise':
                    augmented = self.add_noise(augmented, **params)
                elif aug_type == 'reverb':
                    augmented = self.apply_reverb(augmented, **params)
                elif aug_type == 'time_stretch':
                    augmented = self.time_stretch(augmented, **params)
                elif aug_type == 'pitch_shift':
                    augmented = self.pitch_shift(augmented, **params)
                elif aug_type == 'codec':
                    augmented = self.apply_codec_distortion(augmented, **params)
                elif aug_type == 'clipping':
                    augmented = self.apply_clipping(augmented, **params)
                elif aug_type == 'band_reject':
                    augmented = self.apply_band_reject(augmented, **params)

            results.append(augmented)

        return results

    def create_chaotic_augmentation(self,
                                    signal: np.ndarray,
                                    chaos_level: float = 0.5) -> np.ndarray:
        """
        Apply chaos-inspired augmentation to simulate non-linear distortions.

        Args:
            signal: Input audio signal
            chaos_level: Intensity of chaotic transformation (0.0 to 1.0)

        Returns:
            Chaotically augmented signal
        """
        # Normalize signal
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            normalized = signal / max_val
        else:
            return signal

        # Apply logistic map transformation
        r = 3.7 + 0.3 * chaos_level  # Chaotic regime: 3.7 to 4.0
        x = normalized.copy()

        # Apply iterative chaotic transformation
        for _ in range(int(chaos_level * 3) + 1):
            x = r * x * (1 - x)

        # Mix with original signal
        result = (1 - chaos_level) * normalized + chaos_level * x

        # Restore original scale
        return result * max_val

    def create_multi_condition_training_set(self,
                                            signals: List[np.ndarray],
                                            num_augmentations: int = 3) -> List[np.ndarray]:
        """
        Create multiple augmented versions of each signal for multi-condition training.

        Args:
            signals: List of original audio signals
            num_augmentations: Number of augmented versions to create per signal

        Returns:
            List of all augmented signals
        """
        augmented_signals = []

        # Define possible augmentation types
        aug_types = [
            'noise', 'reverb', 'time_stretch', 'pitch_shift',
            'codec', 'clipping', 'band_reject'
        ]

        for signal in signals:
            # Always include the original signal
            augmented_signals.append(signal)

            # Create specified number of augmented versions
            for _ in range(num_augmentations):
                # Randomly select 1-3 augmentation types
                num_augs = random.randint(1, 3)
                selected_augs = random.sample(aug_types, num_augs)

                # Create augmentation chain
                aug_chain = []
                for aug_type in selected_augs:
                    if aug_type == 'noise':
                        aug_chain.append({
                            'type': 'noise',
                            'params': {
                                'snr_db': random.uniform(5, 20),
                                'noise_type': random.choice(['gaussian', 'babble', 'sample'])
                            }
                        })
                    elif aug_type == 'reverb':
                        aug_chain.append({
                            'type': 'reverb',
                            'params': {'reverb_level': random.uniform(0.1, 0.5)}
                        })
                    elif aug_type == 'time_stretch':
                        aug_chain.append({
                            'type': 'time_stretch',
                            'params': {'rate': random.uniform(0.9, 1.1)}
                        })
                    elif aug_type == 'pitch_shift':
                        aug_chain.append({
                            'type': 'pitch_shift',
                            'params': {'n_steps': random.uniform(-2.0, 2.0)}
                        })
                    elif aug_type == 'codec':
                        aug_chain.append({
                            'type': 'codec',
                            'params': {'quality': random.uniform(0.5, 0.9)}
                        })
                    elif aug_type == 'clipping':
                        aug_chain.append({
                            'type': 'clipping',
                            'params': {'clip_factor': random.uniform(0.7, 0.95)}
                        })
                    elif aug_type == 'band_reject':
                        aug_chain.append({
                            'type': 'band_reject',
                            'params': {
                                'center_freq': random.uniform(500, 3000),
                                'bandwidth': random.uniform(100, 500)
                            }
                        })

                # Apply augmentation chain
                augmented = self.augment_batch([signal], aug_chain)[0]
                augmented_signals.append(augmented)

        return augmented_signals


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create augmentor
    augmentor = AudioAugmentor(sample_rate=16000)

    # Load a sample audio file
    audio_file = "dataset/sample.wav"
    signal, sr = librosa.load(audio_file, sr=16000)

    # Apply various augmentations
    noisy = augmentor.add_noise(signal, snr_db=10.0)
    reverbed = augmentor.apply_reverb(signal, reverb_level=0.3)
    stretched = augmentor.time_stretch(signal, rate=0.9)
    pitched = augmentor.pitch_shift(signal, n_steps=1.0)
    clipped = augmentor.apply_clipping(signal, clip_factor=0.8)
    chaotic = augmentor.create_chaotic_augmentation(signal, chaos_level=0.4)

    # Plot original and augmented signals
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 2, 1)
    plt.title("Original Signal")
    plt.plot(signal)

    plt.subplot(4, 2, 2)
    plt.title("Noisy Signal (SNR=10dB)")
    plt.plot(noisy)

    plt.subplot(4, 2, 3)
    plt.title("Reverberated Signal")
    plt.plot(reverbed)

    plt.subplot(4, 2, 4)
    plt.title("Time-stretched Signal (0.9x)")
    plt.plot(stretched)

    plt.subplot(4, 2, 5)
    plt.title("Pitch-shifted Signal (+1 semitone)")
    plt.plot(pitched)

    plt.subplot(4, 2, 6)
    plt.title("Clipped Signal")
    plt.plot(clipped)

    plt.subplot(4, 2, 7)
    plt.title("Chaotic Augmentation")
    plt.plot(chaotic)

    plt.tight_layout()
    plt.savefig("augmentation_examples.png")
    print("Saved visualization to augmentation_examples.png")