import numpy as np
import librosa
import python_speech_features
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional
import warnings
import scipy.signal


class ConventionalFeatureExtractor:
    """
    Extracts conventional speaker recognition features including:
    - MFCC (Mel-Frequency Cepstral Coefficients)
    - Fbank (Log Mel-filterbank energies)
    - LPCC (Linear Prediction Cepstral Coefficients)
    - PLP (Perceptual Linear Prediction)
    - Spectral features (spectral centroid, bandwidth, etc.)
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 n_mfcc: int = 40,
                 n_fbank: int = 80,
                 n_lpcc: int = 20,
                 frame_length: int = 25,  # ms
                 frame_shift: int = 10,  # ms
                 preemphasis: float = 0.97,
                 delta_order: int = 2,
                 cmvn: bool = True):
        """
        Initialize feature extractor with configuration parameters.

        Args:
            sample_rate: Audio sample rate in Hz
            n_mfcc: Number of MFCC coefficients
            n_fbank: Number of filterbank channels
            n_lpcc: Number of LPCC coefficients
            frame_length: Frame length in milliseconds
            frame_shift: Frame shift in milliseconds
            preemphasis: Preemphasis coefficient
            delta_order: Order of delta features (0=no deltas, 1=+deltas, 2=+delta-deltas)
            cmvn: Whether to apply Cepstral Mean and Variance Normalization
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fbank = n_fbank
        self.n_lpcc = n_lpcc
        self.frame_length = int(frame_length * sample_rate / 1000)
        self.frame_shift = int(frame_shift * sample_rate / 1000)
        self.preemphasis = preemphasis
        self.delta_order = delta_order
        self.cmvn = cmvn

        # Initialize PLP parameters
        self.n_plp = 13  # Default PLP order

    def preemphasize(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply preemphasis filter to enhance high frequencies.

        Args:
            signal: Input audio signal

        Returns:
            Preemphasized signal
        """
        return np.append(signal[0], signal[1:] - self.preemphasis * signal[:-1])

    def framing(self,
                signal: np.ndarray,
                frame_length: Optional[int] = None,
                frame_shift: Optional[int] = None) -> np.ndarray:
        """
        Split signal into overlapping frames.

        Args:
            signal: Input audio signal
            frame_length: Frame length in samples (optional)
            frame_shift: Frame shift in samples (optional)

        Returns:
            Framed signal of shape (num_frames, frame_length)
        """
        if frame_length is None:
            frame_length = self.frame_length
        if frame_shift is None:
            frame_shift = self.frame_shift

        signal_length = len(signal)
        num_frames = 1 + int(np.floor((signal_length - frame_length) / frame_shift))

        frames = np.zeros((num_frames, frame_length))
        for i in range(num_frames):
            frames[i] = signal[i * frame_shift: i * frame_shift + frame_length]

        return frames

    def windowing(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply Hamming window to frames.

        Args:
            frames: Framed signal of shape (num_frames, frame_length)

        Returns:
            Windowed frames
        """
        return frames * np.hamming(frames.shape[1])

    def compute_deltas(self, features: np.ndarray, width: int = 2) -> np.ndarray:
        """
        Compute delta features.

        Args:
            features: Input features of shape (num_frames, feature_dim)
            width: Width of the delta window

        Returns:
            Delta features of same shape as input
        """
        # Create denominator for normalization
        denominator = 2 * sum([i ** 2 for i in range(1, width + 1)])

        # Create delta weights
        delta_weights = np.array([i for i in range(-width, width + 1)])

        # Pad features for edge frames
        padded = np.pad(features, ((width, width), (0, 0)), mode='edge')

        # Compute deltas
        deltas = np.zeros_like(features)
        for t in range(features.shape[0]):
            # Apply delta weights to a window of frames
            deltas[t] = np.dot(delta_weights, padded[t:t + 2 * width + 1]) / denominator

        return deltas

    def apply_cmvn(self, features: np.ndarray, variance_norm: bool = True) -> np.ndarray:
        """
        Apply Cepstral Mean and Variance Normalization.

        Args:
            features: Input features of shape (num_frames, feature_dim)
            variance_norm: Whether to normalize variance

        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) if variance_norm else 1.0

        # Avoid division by zero
        std = np.maximum(std, 1e-10)

        return (features - mean) / std

    def extract_mfcc(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio signal.

        Args:
            signal: Input audio signal

        Returns:
            MFCC features of shape (num_frames, n_mfcc * (1 + delta_order))
        """
        # Apply preemphasis
        preemphasized = self.preemphasize(signal)

        # Extract MFCCs using python_speech_features
        mfcc = python_speech_features.mfcc(
            preemphasized,
            samplerate=self.sample_rate,
            numcep=self.n_mfcc,
            nfilt=2 * self.n_mfcc,
            nfft=2048,
            winlen=self.frame_length / self.sample_rate,
            winstep=self.frame_shift / self.sample_rate,
            preemph=0.0,  # Already applied
            ceplifter=22,
            appendEnergy=True
        )

        # Compute deltas if requested
        features = [mfcc]
        if self.delta_order >= 1:
            delta = self.compute_deltas(mfcc)
            features.append(delta)
        if self.delta_order >= 2:
            delta_delta = self.compute_deltas(delta)
            features.append(delta_delta)

        # Concatenate features
        features = np.hstack(features)

        # Apply CMVN if requested
        if self.cmvn:
            features = self.apply_cmvn(features)

        return features

    def extract_fbank(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract log Mel-filterbank energies from audio signal.

        Args:
            signal: Input audio signal

        Returns:
            Fbank features of shape (num_frames, n_fbank * (1 + delta_order))
        """
        # Apply preemphasis
        preemphasized = self.preemphasize(signal)

        # Extract filterbank features using python_speech_features
        fbank = python_speech_features.logfbank(
            preemphasized,
            samplerate=self.sample_rate,
            nfilt=self.n_fbank,
            nfft=2048,
            winlen=self.frame_length / self.sample_rate,
            winstep=self.frame_shift / self.sample_rate,
            preemph=0.0  # Already applied
        )

        # Compute deltas if requested
        features = [fbank]
        if self.delta_order >= 1:
            delta = self.compute_deltas(fbank)
            features.append(delta)
        if self.delta_order >= 2:
            delta_delta = self.compute_deltas(delta)
            features.append(delta_delta)

        # Concatenate features
        features = np.hstack(features)

        # Apply CMVN if requested
        if self.cmvn:
            features = self.apply_cmvn(features)

        return features

    def extract_lpcc(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract Linear Prediction Cepstral Coefficients.

        Args:
            signal: Input audio signal

        Returns:
            LPCC features of shape (num_frames, n_lpcc * (1 + delta_order))
        """
        # Apply preemphasis
        preemphasized = self.preemphasize(signal)

        # Frame and window the signal
        frames = self.framing(preemphasized)
        windowed_frames = self.windowing(frames)

        # Initialize LPCC features
        num_frames = frames.shape[0]
        lpcc = np.zeros((num_frames, self.n_lpcc))

        # Compute LPCC for each frame
        for i in range(num_frames):
            # Compute autocorrelation
            autocorr = np.correlate(windowed_frames[i], windowed_frames[i], mode='full')
            autocorr = autocorr[len(autocorr) // 2:][:self.n_lpcc + 1]

            # Levinson-Durbin recursion to get LP coefficients
            if autocorr[0] == 0:
                # Avoid division by zero
                lp_coeffs = np.zeros(self.n_lpcc)
            else:
                lp_coeffs = self._levinson_durbin(autocorr, self.n_lpcc)

            # Convert LP coefficients to cepstral coefficients
            lpcc[i] = self._lp_to_cepstrum(lp_coeffs, self.n_lpcc)

        # Compute deltas if requested
        features = [lpcc]
        if self.delta_order >= 1:
            delta = self.compute_deltas(lpcc)
            features.append(delta)
        if self.delta_order >= 2:
            delta_delta = self.compute_deltas(delta)
            features.append(delta_delta)

        # Concatenate features
        features = np.hstack(features)

        # Apply CMVN if requested
        if self.cmvn:
            features = self.apply_cmvn(features)

        return features

    def _levinson_durbin(self, autocorr: np.ndarray, order: int) -> np.ndarray:
        """
        Levinson-Durbin recursion to compute LP coefficients.

        Args:
            autocorr: Autocorrelation sequence
            order: LP order

        Returns:
            LP coefficients
        """
        a = np.zeros(order)
        e = autocorr[0]

        for i in range(order):
            k = -autocorr[i + 1]
            for j in range(i):
                k -= a[j] * autocorr[i - j]
            k /= e

            a_new = np.zeros(order)
            a_new[i] = k
            for j in range(i):
                a_new[j] = a[j] + k * a[i - j - 1]

            a = a_new
            e *= (1 - k * k)

        return a

    def _lp_to_cepstrum(self, lp_coeffs: np.ndarray, n_ceps: int) -> np.ndarray:
        """
        Convert LP coefficients to cepstral coefficients.

        Args:
            lp_coeffs: LP coefficients
            n_ceps: Number of cepstral coefficients

        Returns:
            Cepstral coefficients
        """
        ceps = np.zeros(n_ceps)

        # First cepstral coefficient
        ceps[0] = -np.log(1.0)

        # Recursion to compute remaining coefficients
        for n in range(1, n_ceps):
            sum_term = 0
            for k in range(1, min(n, len(lp_coeffs)) + 1):
                sum_term += k * ceps[n - k] * lp_coeffs[k - 1] / n

            if n < len(lp_coeffs):
                ceps[n] = lp_coeffs[n - 1] + sum_term
            else:
                ceps[n] = sum_term

        return ceps

    def extract_plp(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract Perceptual Linear Prediction features.

        Args:
            signal: Input audio signal

        Returns:
            PLP features of shape (num_frames, n_plp * (1 + delta_order))
        """
        try:
            from bob.ap import Ceps
        except ImportError:
            warnings.warn("bob.ap package not found. Using MFCC as fallback for PLP.")
            return self.extract_mfcc(signal)

        # Apply preemphasis
        preemphasized = self.preemphasize(signal)

        # Initialize PLP extractor
        plp_extractor = Ceps(
            self.sample_rate,
            win_length_ms=self.frame_length * 1000 / self.sample_rate,
            win_shift_ms=self.frame_shift * 1000 / self.sample_rate,
            n_filters=self.n_fbank,
            n_ceps=self.n_plp,
            f_min=0.0,
            f_max=self.sample_rate / 2,
            delta_win=2,
            pre_emphasis_coef=0.0,  # Already applied
            mel_scale=False,  # Use PLP instead of MFCC
            dct_norm=True,
            normalize_spectrum=True
        )

        # Extract PLP features
        plp = plp_extractor(preemphasized.astype(np.float64))

        # Compute deltas if requested
        features = [plp]
        if self.delta_order >= 1:
            delta = self.compute_deltas(plp)
            features.append(delta)
        if self.delta_order >= 2:
            delta_delta = self.compute_deltas(delta)
            features.append(delta_delta)

        # Concatenate features
        features = np.hstack(features)

        # Apply CMVN if requested
        if self.cmvn:
            features = self.apply_cmvn(features)

        return features

    def extract_spectral_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract spectral features (centroid, bandwidth, flatness, roll-off).

        Args:
            signal: Input audio signal

        Returns:
            Spectral features of shape (num_frames, 4)
        """
        # Apply preemphasis
        preemphasized = self.preemphasize(signal)

        # Frame and window the signal
        frames = self.framing(preemphasized)
        windowed_frames = self.windowing(frames)

        # Initialize spectral features
        num_frames = frames.shape[0]
        spectral_features = np.zeros((num_frames, 4))

        # Compute FFT for each frame
        for i in range(num_frames):
            fft = np.abs(np.fft.rfft(windowed_frames[i], n=2048))
            fft = fft[:1024]  # Keep only positive frequencies

            # Frequency axis
            freqs = np.linspace(0, self.sample_rate / 2, len(fft))

            # Spectral centroid
            if np.sum(fft) > 0:
                spectral_features[i, 0] = np.sum(freqs * fft) / np.sum(fft)

            # Spectral bandwidth
            if np.sum(fft) > 0:
                spectral_features[i, 1] = np.sqrt(
                    np.sum(((freqs - spectral_features[i, 0]) ** 2) * fft) / np.sum(fft)
                )

            # Spectral flatness
            if np.all(fft > 0):
                geometric_mean = np.exp(np.mean(np.log(fft + 1e-10)))
                arithmetic_mean = np.mean(fft)
                if arithmetic_mean > 0:
                    spectral_features[i, 2] = geometric_mean / arithmetic_mean

            # Spectral roll-off
            cumsum = np.cumsum(fft)
            if cumsum[-1] > 0:
                threshold = 0.85 * cumsum[-1]
                spectral_features[i, 3] = freqs[np.where(cumsum >= threshold)[0][0]]

        # Apply CMVN if requested
        if self.cmvn:
            spectral_features = self.apply_cmvn(spectral_features)

        return spectral_features

    def extract_all_features(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all conventional features from audio signal.

        Args:
            signal: Input audio signal

        Returns:
            Dictionary containing all extracted features
        """
        return {
            'mfcc': self.extract_mfcc(signal),
            'fbank': self.extract_fbank(signal),
            'lpcc': self.extract_lpcc(signal),
            'spectral': self.extract_spectral_features(signal)
        }

    def extract_combined_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract and combine multiple feature types.

        Args:
            signal: Input audio signal

        Returns:
            Combined features
        """
        # Extract individual features
        mfcc = self.extract_mfcc(signal)
        spectral = self.extract_spectral_features(signal)

        # Ensure same number of frames
        min_frames = min(mfcc.shape[0], spectral.shape[0])
        mfcc = mfcc[:min_frames]
        spectral = spectral[:min_frames]

        # Combine features
        combined = np.hstack([mfcc, spectral])

        return combined


class IVectorExtractor:
    """
    i-vector feature extractor for speaker recognition.
    Implements the traditional i-vector/PLDA pipeline.
    """

    def __init__(self,
                 ubm_components: int = 512,
                 tv_dim: int = 400,
                 feature_dim: int = 40):
        """
        Initialize i-vector extractor.

        Args:
            ubm_components: Number of UBM components
            tv_dim: Total variability subspace dimension
            feature_dim: Input feature dimension
        """
        self.ubm_components = ubm_components
        self.tv_dim = tv_dim
        self.feature_dim = feature_dim

        # Initialize UBM parameters
        self.ubm_means = None
        self.ubm_covs = None
        self.ubm_weights = None

        # Initialize T matrix
        self.T = None

        # Initialize PLDA parameters
        self.plda_mean = None
        self.plda_F = None
        self.plda_G = None
        self.plda_sigma = None

    def train_ubm(self, features_list: List[np.ndarray]) -> None:
        """
        Train Universal Background Model (GMM-UBM).

        Args:
            features_list: List of feature matrices from multiple speakers
        """
        # Concatenate features from all utterances
        all_features = np.vstack(features_list)

        # Initialize GMM parameters
        n_samples = all_features.shape[0]
        n_components = self.ubm_components

        # Random initialization
        np.random.seed(42)
        indices = np.random.choice(n_samples, n_components, replace=False)
        means = all_features[indices]

        # Initialize covariances as identity matrices
        covs = np.ones((n_components, self.feature_dim))

        # Initialize weights uniformly
        weights = np.ones(n_components) / n_components

        # EM algorithm for GMM training (simplified)
        max_iters = 10
        for _ in range(max_iters):
            # E-step: Compute responsibilities
            log_probs = np.zeros((n_samples, n_components))
            for k in range(n_components):
                diff = all_features - means[k]
                log_probs[:, k] = -0.5 * np.sum(diff ** 2 / covs[k], axis=1) - \
                                  0.5 * np.sum(np.log(covs[k])) + np.log(weights[k])

            # Normalize responsibilities
            log_probs_max = np.max(log_probs, axis=1, keepdims=True)
            probs = np.exp(log_probs - log_probs_max)
            responsibilities = probs / np.sum(probs, axis=1, keepdims=True)

            # M-step: Update parameters
            Nk = np.sum(responsibilities, axis=0)

            # Update weights
            weights = Nk / n_samples

            # Update means
            means = np.zeros((n_components, self.feature_dim))
            for k in range(n_components):
                means[k] = np.sum(responsibilities[:, k:k + 1] * all_features, axis=0) / Nk[k]

            # Update covariances (diagonal)
            covs = np.zeros((n_components, self.feature_dim))
            for k in range(n_components):
                diff = all_features - means[k]
                covs[k] = np.sum(responsibilities[:, k:k + 1] * diff ** 2, axis=0) / Nk[k]
                covs[k] = np.maximum(covs[k], 1e-6)  # Ensure positive values

        # Store UBM parameters
        self.ubm_means = means
        self.ubm_covs = covs
        self.ubm_weights = weights

    def train_tv_space(self, features_list: List[np.ndarray], speaker_ids: List[int]) -> None:
        """
        Train total variability space (T matrix).

        Args:
            features_list: List of feature matrices from multiple speakers
            speaker_ids: List of speaker IDs corresponding to features_list
        """
        if self.ubm_means is None:
            raise ValueError("UBM must be trained before TV space")

        # Get unique speaker IDs
        unique_speakers = np.unique(speaker_ids)
        n_speakers = len(unique_speakers)

        # Initialize T matrix randomly
        np.random.seed(42)
        self.T = np.random.randn(self.ubm_components * self.feature_dim, self.tv_dim) * 0.01

        # Compute Baum-Welch statistics for each utterance
        n_stats = []
        f_stats = []

        for features in features_list:
            # Compute posteriors
            posteriors = self._compute_posteriors(features)

            # Zero-order statistics
            n_stat = np.sum(posteriors, axis=0)
            n_stats.append(n_stat)

            # First-order statistics
            f_stat = np.zeros((self.ubm_components, self.feature_dim))
            for c in range(self.ubm_components):
                f_stat[c] = np.sum(posteriors[:, c:c + 1] * features, axis=0)
            f_stats.append(f_stat)

        # EM algorithm for T matrix estimation (simplified)
        max_iters = 5
        for _ in range(max_iters):
            # E-step: Estimate i-vectors
            i_vectors = []
            for i in range(len(features_list)):
                i_vector = self._estimate_ivector(n_stats[i], f_stats[i])
                i_vectors.append(i_vector)

            # M-step: Update T matrix
            # (Simplified implementation - in practice, this would be more complex)
            for c in range(self.ubm_components):
                for d in range(self.feature_dim):
                    idx = c * self.feature_dim + d

                    # Accumulate statistics
                    A = np.zeros((self.tv_dim, self.tv_dim))
                    B = np.zeros(self.tv_dim)

                    for i in range(len(features_list)):
                        i_vector = i_vectors[i]
                        n_stat = n_stats[i][c]
                        f_stat = f_stats[i][c, d]

                        A += n_stat * np.outer(i_vector, i_vector)
                        B += (f_stat - n_stat * self.ubm_means[c, d]) * i_vector

                    # Solve for T
                    A += 1e-4 * np.eye(self.tv_dim)  # Regularization
                    self.T[idx] = np.linalg.solve(A, B)

    def train_plda(self, i_vectors: List[np.ndarray], speaker_ids: List[int]) -> None:
        """
        Train PLDA model for i-vector scoring.

        Args:
            i_vectors: List of i-vectors
            speaker_ids: List of speaker IDs corresponding to i_vectors
        """
        # Convert to numpy array
        i_vectors = np.vstack(i_vectors)
        speaker_ids = np.array(speaker_ids)

        # Get unique speaker IDs
        unique_speakers = np.unique(speaker_ids)
        n_speakers = len(unique_speakers)

        # Compute global mean
        self.plda_mean = np.mean(i_vectors, axis=0)

        # Center i-vectors
        i_vectors_centered = i_vectors - self.plda_mean

        # Initialize PLDA parameters
        self.plda_F = np.random.randn(self.tv_dim, self.tv_dim // 2) * 0.01
        self.plda_G = np.random.randn(self.tv_dim, self.tv_dim // 2) * 0.01
        self.plda_sigma = np.ones(self.tv_dim) * 0.01

        # EM algorithm for PLDA training (simplified)
        max_iters = 10
        for _ in range(max_iters):
            # E-step: Estimate latent variables
            speaker_factors = {}
            channel_factors = {}

            for spk in unique_speakers:
                spk_ivectors = i_vectors_centered[speaker_ids == spk]

                # Compute speaker factor (simplified)
                n_utts = len(spk_ivectors)
                F_proj = self.plda_F.T @ self.plda_F
                precision = np.diag(1.0 / self.plda_sigma)

                speaker_cov = np.linalg.inv(np.eye(self.plda_F.shape[1]) + n_utts * F_proj @ precision)
                speaker_mean = speaker_cov @ F_proj @ precision @ np.sum(spk_ivectors, axis=0)

                speaker_factors[spk] = speaker_mean

                # Compute channel factors for each utterance
                for i, ivec in enumerate(spk_ivectors):
                    residual = ivec - self.plda_F @ speaker_mean

                    G_proj = self.plda_G.T @ self.plda_G
                    channel_cov = np.linalg.inv(np.eye(self.plda_G.shape[1]) + G_proj @ precision)
                    channel_mean = channel_cov @ self.plda_G.T @ precision @ residual

                    channel_factors[(spk, i)] = channel_mean

            # M-step: Update PLDA parameters
            # (Simplified implementation - in practice, this would be more complex)

            # Update F
            F_num = np.zeros((self.tv_dim, self.plda_F.shape[1]))
            F_denom = np.zeros((self.plda_F.shape[1], self.plda_F.shape[1]))

            for spk in unique_speakers:
                spk_ivectors = i_vectors_centered[speaker_ids == spk]
                y = speaker_factors[spk]

                for i, ivec in enumerate(spk_ivectors):
                    z = channel_factors[(spk, i)]
                    residual = ivec - self.plda_G @ z

                    F_num += np.outer(residual, y)
                    F_denom += np.outer(y, y)

            F_denom += 1e-4 * np.eye(self.plda_F.shape[1])  # Regularization
            self.plda_F = F_num @ np.linalg.inv(F_denom)

            # Update G (similar to F update)
            G_num = np.zeros((self.tv_dim, self.plda_G.shape[1]))
            G_denom = np.zeros((self.plda_G.shape[1], self.plda_G.shape[1]))

            for spk in unique_speakers:
                spk_ivectors = i_vectors_centered[speaker_ids == spk]
                y = speaker_factors[spk]

                for i, ivec in enumerate(spk_ivectors):
                    z = channel_factors[(spk, i)]
                    residual = ivec - self.plda_F @ y

                    G_num += np.outer(residual, z)
                    G_denom += np.outer(z, z)

            G_denom += 1e-4 * np.eye(self.plda_G.shape[1])  # Regularization
            self.plda_G = G_num @ np.linalg.inv(G_denom)

            # Update sigma
            sigma_acc = np.zeros(self.tv_dim)
            n_samples = 0

            for spk in unique_speakers:
                spk_ivectors = i_vectors_centered[speaker_ids == spk]
                y = speaker_factors[spk]

                for i, ivec in enumerate(spk_ivectors):
                    z = channel_factors[(spk, i)]
                    residual = ivec - self.plda_F @ y - self.plda_G @ z

                    sigma_acc += residual ** 2
                    n_samples += 1

            self.plda_sigma = sigma_acc / n_samples
            self.plda_sigma = np.maximum(self.plda_sigma, 1e-6)  # Ensure positive values

    def _compute_posteriors(self, features: np.ndarray) -> np.ndarray:
        """
        Compute GMM posteriors for features.

        Args:
            features: Input feature matrix of shape (num_frames, feature_dim)

        Returns:
            Posterior probabilities of shape (num_frames, ubm_components)
        """
        n_samples = features.shape[0]
        log_probs = np.zeros((n_samples, self.ubm_components))

        for k in range(self.ubm_components):
            diff = features - self.ubm_means[k]
            log_probs[:, k] = -0.5 * np.sum(diff ** 2 / self.ubm_covs[k], axis=1) - \
                              0.5 * np.sum(np.log(self.ubm_covs[k])) + np.log(self.ubm_weights[k])

        # Normalize posteriors
        log_probs_max = np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - log_probs_max)
        posteriors = probs / np.sum(probs, axis=1, keepdims=True)

        return posteriors

    def _estimate_ivector(self, n_stat: np.ndarray, f_stat: np.ndarray) -> np.ndarray:
        """
        Estimate i-vector from sufficient statistics.

        Args:
            n_stat: Zero-order statistics
            f_stat: First-order statistics

        Returns:
            i-vector of dimension tv_dim
        """
        # Compute precision matrix
        precision = np.eye(self.tv_dim)

        for c in range(self.ubm_components):
            Tc = self.T[c * self.feature_dim:(c + 1) * self.feature_dim]
            precision += n_stat[c] * Tc.T @ np.diag(1.0 / self.ubm_covs[c]) @ Tc

        # Compute i-vector
        i_vector = np.zeros(self.tv_dim)

        for c in range(self.ubm_components):
            Tc = self.T[c * self.feature_dim:(c + 1) * self.feature_dim]
            f_c = f_stat[c] - n_stat[c] * self.ubm_means[c]
            i_vector += Tc.T @ np.diag(1.0 / self.ubm_covs[c]) @ f_c

        # Solve for i-vector
        i_vector = np.linalg.solve(precision, i_vector)

        return i_vector

    def extract_ivector(self, features: np.ndarray) -> np.ndarray:
        """
        Extract i-vector from features.

        Args:
            features: Input feature matrix

        Returns:
            i-vector of dimension tv_dim
        """
        if self.T is None:
            raise ValueError("TV space must be trained before extracting i-vectors")

        # Compute posteriors
        posteriors = self._compute_posteriors(features)

        # Compute sufficient statistics
        n_stat = np.sum(posteriors, axis=0)
        f_stat = np.zeros((self.ubm_components, self.feature_dim))
        for c in range(self.ubm_components):
            f_stat[c] = np.sum(posteriors[:, c:c + 1] * features, axis=0)

        # Estimate i-vector
        i_vector = self._estimate_ivector(n_stat, f_stat)

        return i_vector

    def score_plda(self, ivector1: np.ndarray, ivector2: np.ndarray) -> float:
        """
        Compute PLDA score between two i-vectors.

        Args:
            ivector1: First i-vector
            ivector2: Second i-vector

        Returns:
            PLDA score (log-likelihood ratio)
        """
        if self.plda_mean is None:
            raise ValueError("PLDA must be trained before scoring")

        # Center i-vectors
        ivector1 = ivector1 - self.plda_mean
        ivector2 = ivector2 - self.plda_mean

        # Compute precision matrix
        precision = np.diag(1.0 / self.plda_sigma)

        # Compute matrices for scoring
        F_proj = self.plda_F @ self.plda_F.T
        G_proj = self.plda_G @ self.plda_G.T

        # Compute same-speaker and different-speaker covariances
        cov_same = F_proj + G_proj + np.diag(self.plda_sigma)
        cov_diff = G_proj + np.diag(self.plda_sigma)

        # Compute precisions
        prec_same = np.linalg.inv(cov_same)
        prec_diff = np.linalg.inv(cov_diff)

        # Compute log-determinants
        logdet_same = np.linalg.slogdet(cov_same)[1]
        logdet_diff = np.linalg.slogdet(cov_diff)[1]

        # Compute score components
        term1 = ivector1.T @ (prec_same - 2 * prec_diff) @ ivector2
        term2 = -0.5 * ivector1.T @ (prec_same - prec_diff) @ ivector1
        term3 = -0.5 * ivector2.T @ (prec_same - prec_diff) @ ivector2
        term4 = 0.5 * (logdet_diff - logdet_same)

        # Compute final score
        score = term1 + term2 + term3 + term4

        return float(score)


class XVectorExtractor:
    """
    PyTorch implementation of x-vector extraction for speaker recognition.
    Based on the TDNN architecture proposed by Snyder et al.
    """

    def __init__(self,
                 input_dim: int = 40,
                 embedding_dim: int = 512,
                 num_classes: int = 0,
                 device: str = "cpu"):
        """
        Initialize x-vector extractor.

        Args:
            input_dim: Input feature dimension
            embedding_dim: Embedding dimension (x-vector size)
            num_classes: Number of speakers for training (0 for feature extraction only)
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.device = device

        # Initialize model
        self.model = self._create_model()
        self.model.to(device)

    def _create_model(self) -> nn.Module:
        """
        Create x-vector model architecture.

        Returns:
            PyTorch model
        """

        class XVectorNet(nn.Module):
            def __init__(self, input_dim, embedding_dim, num_classes):
                super().__init__()

                # Frame-level layers
                self.tdnn1 = nn.Conv1d(input_dim, 512, kernel_size=5, dilation=1)
                self.tdnn2 = nn.Conv1d(512, 512, kernel_size=3, dilation=2)
                self.tdnn3 = nn.Conv1d(512, 512, kernel_size=3, dilation=3)
                self.tdnn4 = nn.Conv1d(512, 512, kernel_size=1, dilation=1)
                self.tdnn5 = nn.Conv1d(512, 1500, kernel_size=1, dilation=1)

                # Statistics pooling
                self.segment6 = nn.Linear(3000, embedding_dim)
                self.segment7 = nn.Linear(embedding_dim, embedding_dim)

                # Output layer
                self.output = nn.Linear(embedding_dim, num_classes) if num_classes > 0 else None

                # Activation functions
                self.relu = nn.ReLU()
                self.bn1 = nn.BatchNorm1d(512)
                self.bn2 = nn.BatchNorm1d(512)
                self.bn3 = nn.BatchNorm1d(512)
                self.bn4 = nn.BatchNorm1d(512)
                self.bn5 = nn.BatchNorm1d(1500)
                self.bn6 = nn.BatchNorm1d(embedding_dim)
                self.bn7 = nn.BatchNorm1d(embedding_dim)

            def forward(self, x, extract_xvector=False):
                # Input: (batch_size, time_steps, input_dim)
                # Convert to (batch_size, input_dim, time_steps) for 1D convolution
                x = x.transpose(1, 2)

                # Frame-level layers
                x = self.relu(self.bn1(self.tdnn1(x)))
                x = self.relu(self.bn2(self.tdnn2(x)))
                x = self.relu(self.bn3(self.tdnn3(x)))
                x = self.relu(self.bn4(self.tdnn4(x)))
                x = self.relu(self.bn5(self.tdnn5(x)))

                # Statistics pooling
                mean = torch.mean(x, dim=2)
                std = torch.std(x, dim=2)
                x = torch.cat([mean, std], dim=1)

                # Segment-level layers
                x = self.relu(self.bn6(self.segment6(x)))
                x = self.relu(self.bn7(self.segment7(x)))

                # Extract x-vector if requested
                if extract_xvector:
                    return x

                # Output layer
                if self.output is not None:
                    x = self.output(x)

                return x

        return XVectorNet(self.input_dim, self.embedding_dim, self.num_classes)

    def extract_xvector(self, features: np.ndarray) -> np.ndarray:
        """
        Extract x-vector from features.

        Args:
            features: Input feature matrix of shape (num_frames, feature_dim)

        Returns:
            x-vector of dimension embedding_dim
        """
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Extract x-vector
        self.model.eval()
        with torch.no_grad():
            xvector = self.model(features_tensor, extract_xvector=True)

        # Convert to numpy
        xvector = xvector.cpu().numpy().squeeze()

        return xvector

    def train_model(self,
                    features_list: List[np.ndarray],
                    speaker_ids: List[int],
                    num_epochs: int = 10,
                    batch_size: int = 32,
                    learning_rate: float = 0.001) -> None:
        """
        Train x-vector model.

        Args:
            features_list: List of feature matrices
            speaker_ids: List of speaker IDs corresponding to features_list
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        if self.num_classes == 0:
            raise ValueError("Model must be initialized with num_classes > 0 for training")

        # Create dataset
        class SpeakerDataset(torch.utils.data.Dataset):
            def __init__(self, features_list, speaker_ids):
                self.features = []
                self.labels = []

                for i, features in enumerate(features_list):
                    self.features.append(features)
                    self.labels.append(speaker_ids[i])

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return self.features[idx], self.labels[idx]

        # Create data loader
        dataset = SpeakerDataset(features_list, speaker_ids)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: (
                torch.tensor([item[0] for item in batch], dtype=torch.float32),
                torch.tensor([item[1] for item in batch], dtype=torch.long)
            )
        )

        # Set up optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Train model
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(features)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    def save_model(self, path: str) -> None:
        """
        Save model to file.

        Args:
            path: Path to save model
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        """
        Load model from file.

        Args:
            path: Path to load model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load a sample audio file
    audio_file = "P:/PycharmProjects/pythonProject1/dataset/LibriSpeech/train-clean-100/19/198/19-198-0001.flac"
    signal, sr = librosa.load(audio_file, sr=16000)

    # Create feature extractor
    feature_extractor = ConventionalFeatureExtractor(sample_rate=sr)

    # Extract features
    mfcc = feature_extractor.extract_mfcc(signal)
    fbank = feature_extractor.extract_fbank(signal)
    lpcc = feature_extractor.extract_lpcc(signal)
    spectral = feature_extractor.extract_spectral_features(signal)

    # Plot features
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title("MFCC")
    plt.imshow(mfcc.T, aspect='auto', origin='lower')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("Fbank")
    plt.imshow(fbank.T, aspect='auto', origin='lower')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("LPCC")
    plt.imshow(lpcc.T, aspect='auto', origin='lower')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("Spectral Features")
    plt.imshow(spectral.T, aspect='auto', origin='lower')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("conventional_features.png")
    print("Saved visualization to conventional_features.png")