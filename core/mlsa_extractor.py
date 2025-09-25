"""
Multi-scale Lyapunov Spectrum Analysis (MLSA) Extractor.

This module implements the core innovation of the C-HiLAP project: multi-scale
Lyapunov spectrum analysis for extracting chaotic features from audio signals
across different temporal scales.

Author: C-HiLAP Project
Date: 2025
"""

from abc import ABC, abstractmethod
import scipy.signal as signal
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# AFTER (修复后的统一导入方式):
import os
import sys
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# 统一导入设置
# =============================================================================
def setup_module_imports(current_file: str = __file__):
    """Setup imports for current module.""" 
    try:
        from setup_imports import setup_project_imports
        return setup_project_imports(current_file), True
    except ImportError:
        current_dir = Path(current_file).resolve().parent  # core目录
        project_root = current_dir.parent  # core -> Model
        
        paths_to_add = [
            str(project_root),
            str(project_root / 'core'),
            str(project_root / 'utils'),
        ]
        
        for path in paths_to_add:
            if Path(path).exists() and path not in sys.path:
                sys.path.insert(0, path)
        
        return project_root, False

# Setup imports
PROJECT_ROOT, USING_IMPORT_MANAGER = setup_module_imports()

# =============================================================================
# 项目模块导入 (带安全检查)
# =============================================================================
try:
    from chaos_utils import (
        largest_lyapunov_from_data, correlation_dimension, 
        hurst_exponent, kolmogorov_entropy
    )
    HAS_CHAOS_UTILS = True
except ImportError as e:
    HAS_CHAOS_UTILS = False
    warnings.warn(f"chaos_utils not available: {e}")
    # 简单fallback，不需要大量代码
    largest_lyapunov_from_data = lambda x: 0.0
    correlation_dimension = lambda x: 2.0

try:
    from phase_space_reconstruction import PhaseSpaceReconstructor, EmbeddingConfig
    HAS_PHASE_SPACE = True
except ImportError as e:
    HAS_PHASE_SPACE = False
    warnings.warn(f"phase_space_reconstruction not available: {e}")

try:
    from utils.numerical_stability import NumericalConfig, safe_divide
    HAS_NUMERICAL_UTILS = True
except ImportError as e:
    HAS_NUMERICAL_UTILS = False
    warnings.warn(f"numerical_stability not available: {e}")
    # Simple fallback
    safe_divide = lambda x, y: x / (y + 1e-12)

    
    
    # Provide default class definitions to avoid NameError
    @dataclass
    class EmbeddingConfig:
        """Default EmbeddingConfig when import fails"""
        embedding_dim: int = 3
        time_delay: int = 1
        max_embedding_dim: int = 20
        fnn_threshold: float = 15.0
        autocorr_threshold: float = 0.1
    
    @dataclass
    class NumericalConfig:
        """Default NumericalConfig when import fails"""
        epsilon: float = 1e-12
        max_iterations: int = 1000
        tolerance: float = 1e-8
        convergence_tolerance: float = 1e-8
        rtol: float = 1e-5
        atol: float = 1e-8
        outlier_threshold: float = 3.0
        stability_check: bool = True
        zero_threshold: float = 1e-12
        inf_threshold: float = 1e12
        enable_overflow_check: bool = True
        default_fill_value: float = 0.0
    
    class PhaseSpaceReconstructor:
        """Default PhaseSpaceReconstructor when import fails"""
        def __init__(self, config):
            self.config = config
        
        def reconstruct(self, signal, delay=None, dimension=None):
            if delay is None:
                delay = 1
            if dimension is None:
                dimension = 3
            
            N = len(signal)
            if N < dimension * delay:
                return {
                    'embedding_success': False,
                    'error': 'Signal too short for embedding'
                }
            
            embedded = np.zeros((N - (dimension-1)*delay, dimension))
            for i in range(dimension):
                embedded[:, i] = signal[i*delay:N-(dimension-1-i)*delay]
            
            return {
                'embedding_success': True,
                'embedded_data': embedded,
                'delay': delay,
                'dimension': dimension,
                'n_embedded_points': embedded.shape[0]
            }
    
    class PrecisionManager:
        def __init__(self, config):
            self.config = config
    
    class OutlierDetector:
        def __init__(self, config):
            self.config = config
        
        def remove_outliers(self, data, method='zscore'):
            if len(data) == 0:
                return np.array([]), np.array([])
                
            if method == 'zscore':
                data_std = np.std(data)
                if data_std < 1e-12:  # Constant signal
                    return data, np.array([])
                
                z_scores = np.abs((data - np.mean(data)) / data_std)
                mask = z_scores < 3.0
                return data[mask], np.where(~mask)[0]
            else:
                return data, np.array([])
    
    class NumericalValidator:
        def __init__(self, config):
            self.config = config
        
        def validate_array(self, arr, name="array"):
            issues = []
            
            if not isinstance(arr, np.ndarray):
                issues.append(f"{name} must be numpy array")
            elif arr.size == 0:
                issues.append(f"{name} cannot be empty")
            elif not np.isfinite(arr).all():
                issues.append(f"{name} contains non-finite values")
            
            return {
                'is_valid': len(issues) == 0,
                'issues': issues
            }
    
    def largest_lyapunov_from_data(data, dt=0.01, tau=1, min_neighbors=10):
        try:
            N = len(data)
            if N < 100:
                return 0.0
            
            m = 3
            embedded = np.zeros((N - (m-1)*tau, m))
            for i in range(m):
                embedded[:, i] = data[i*tau:N-(m-1-i)*tau]
            
            n_points = min(200, embedded.shape[0] - 1)
            divergence_rates = []
            
            for i in range(0, n_points, 10):
                distances = np.linalg.norm(embedded - embedded[i], axis=1)
                distances[i] = np.inf
                
                if np.min(distances) > 0:
                    j = np.argmin(distances)
                    track_length = min(20, embedded.shape[0] - max(i, j) - 1)
                    if track_length > 5:
                        initial_dist = distances[j]
                        final_dist = np.linalg.norm(embedded[i + track_length] - 
                                                  embedded[j + track_length])
                        if initial_dist > 0 and final_dist > initial_dist:
                            rate = np.log(final_dist / initial_dist) / (track_length * dt)
                            divergence_rates.append(rate)
            
            return np.mean(divergence_rates) if divergence_rates else 0.0
            
        except Exception:
            return 0.0
    
    def correlation_dimension(data, r_points=30):
        try:
            N = len(data) if data.ndim == 1 else data.shape[0]
            
            if N < 50:
                return np.array([]), np.array([]), 2.0
            
            if data.ndim == 1:
                distances = np.abs(data[:, None] - data[None, :])
            else:
                from scipy.spatial.distance import pdist, squareform
                distances = squareform(pdist(data))
            
            max_dist = np.max(distances)
            min_dist = np.min(distances[distances > 0])
            
            radii = np.logspace(np.log10(min_dist), np.log10(max_dist), r_points)
            correlations = []
            
            for r in radii:
                count = np.sum(distances <= r)
                correlation = count / (N * N)
                correlations.append(correlation)
            
            correlations = np.array(correlations)
            
            valid_mask = (correlations > 0) & (correlations < 1)
            if np.sum(valid_mask) > 5:
                log_r = np.log(radii[valid_mask])
                log_c = np.log(correlations[valid_mask])
                slope = np.polyfit(log_r, log_c, 1)[0]
                corr_dim = max(0, slope)
            else:
                corr_dim = 2.0
            
            return radii, correlations, corr_dim
            
        except Exception:
            return np.array([]), np.array([]), 2.0
    
    def hurst_exponent(data):
        try:
            N = len(data)
            if N < 20:
                return 0.5
            
            lags = np.arange(2, min(N//4, 100))
            rs_values = []
            
            for lag in lags:
                mean_val = np.mean(data)
                cumulative_dev = np.cumsum(data - mean_val)
                
                n_segments = N // lag
                rs_segment = []
                
                for i in range(n_segments):
                    segment = cumulative_dev[i*lag:(i+1)*lag]
                    if len(segment) > 1:
                        R = np.max(segment) - np.min(segment)
                        S = np.std(data[i*lag:(i+1)*lag])
                        if S > 0:
                            rs_segment.append(R / S)
                
                if rs_segment:
                    rs_values.append(np.mean(rs_segment))
                else:
                    rs_values.append(1.0)
            
            if len(rs_values) > 3:
                log_lags = np.log(lags[:len(rs_values)])
                log_rs = np.log(rs_values)
                hurst = np.polyfit(log_lags, log_rs, 1)[0]
                return np.clip(hurst, 0.0, 1.0)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def kolmogorov_entropy(data):
        return 1.0
    
    def safe_divide(numerator, denominator, default=0.0, epsilon=1e-12):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(np.abs(denominator) > epsilon, 
                             numerator / denominator, default)
        return result
    
    def safe_log(x, epsilon=1e-12):
        return np.log(np.maximum(x, epsilon))


def safe_psd_normalize(psd, eps=1e-12):
    """Safe PSD normalization to avoid divide by zero errors."""
    psd_sum = np.sum(psd)
    if psd_sum > eps:
        return psd / psd_sum
    else:
        # If PSD is all zeros, return uniform distribution
        return np.ones_like(psd) / len(psd) if len(psd) > 0 else np.array([1.0])


def safe_signal_normalize(signal, eps=1e-12):
    """Safe signal standardization to avoid divide by zero errors."""
    signal = np.asarray(signal)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    
    if std_val > eps:
        return (signal - mean_val) / std_val
    else:
        # If constant signal, return zero-mean signal
        return signal - mean_val


@dataclass
class MLSAConfig:
    """Configuration for Multi-scale Lyapunov Spectrum Analysis."""
    
    # Scale decomposition parameters
    n_scales: int = 5
    scale_factors: List[float] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    decomposition_method: str = 'wavelet'  # 'wavelet', 'emd', 'fourier'
    wavelet_name: str = 'db4'
    
    # Lyapunov analysis parameters
    min_segment_length: int = 100
    max_segment_length: int = 2000
    overlap_ratio: float = 0.5
    lyapunov_method: str = 'rosenstein'  # 'rosenstein', 'wolf', 'kantz'
    
    # Phase space parameters
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    auto_embedding: bool = True
    
    # Feature extraction parameters
    spectral_bins: int = 50
    entropy_bins: int = 20
    correlation_r_points: int = 30
    
    # Quality control
    min_positive_lyapunov_ratio: float = 0.1
    max_condition_number: float = 1e10
    outlier_threshold: float = 3.0
    
    # Numerical stability
    numerical_config: NumericalConfig = field(default_factory=NumericalConfig)
    enable_validation: bool = True


class ScaleDecomposer(ABC):
    """Abstract base class for multi-scale decomposition."""
    
    @abstractmethod
    def decompose(self, signal: np.ndarray, scales: List[float]) -> Dict[float, np.ndarray]:
        """Decompose signal into multiple scales."""
        pass
    
    @abstractmethod
    def get_scale_info(self, scale: float) -> Dict[str, Any]:
        """Get information about a specific scale."""
        pass


class WaveletScaleDecomposer(ScaleDecomposer):
    """Wavelet-based multi-scale decomposition."""
    
    def __init__(self, wavelet_name: str = 'db4'):
        self.wavelet_name = wavelet_name
        try:
            import pywt
            self.pywt = pywt
            self.available = True
        except ImportError:
            warnings.warn("PyWavelets not available, using Fourier fallback")
            self.available = False
    
    def decompose(self, signal: np.ndarray, scales: List[float]) -> Dict[float, np.ndarray]:
        """Decompose signal using wavelet transform."""
        if not self.available:
            return self._fourier_fallback(signal, scales)
        
        decomposed = {}
        
        for scale in scales:
            try:
                # Use continuous wavelet transform
                coeffs, freqs = self.pywt.cwt(
                    signal, 
                    np.array([scale]), 
                    self.wavelet_name,
                    sampling_period=1.0
                )
                
                # Extract the real part and flatten
                decomposed[scale] = np.real(coeffs[0])
                
            except Exception as e:
                warnings.warn(f"Wavelet decomposition failed for scale {scale}: {e}")
                # Fallback to simple downsampling
                decomposed[scale] = self._simple_scale_extraction(signal, scale)
        
        return decomposed
    
    def _fourier_fallback(self, signal: np.ndarray, scales: List[float]) -> Dict[float, np.ndarray]:
        """Fourier-based fallback decomposition."""
        decomposed = {}
        
        # Get frequency domain representation
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        
        for scale in scales:
            # Create bandpass filter around scale frequency
            center_freq = 1.0 / scale
            bandwidth = center_freq * 0.5
            
            # Filter in frequency domain
            filtered_fft = fft_signal.copy()
            mask = (np.abs(freqs - center_freq) > bandwidth) & (np.abs(freqs + center_freq) > bandwidth)
            filtered_fft[mask] = 0
            
            # Convert back to time domain
            filtered_signal = np.real(np.fft.ifft(filtered_fft))
            decomposed[scale] = filtered_signal
        
        return decomposed
    
    def _simple_scale_extraction(self, signal: np.ndarray, scale: float) -> np.ndarray:
        """Simple scale extraction by downsampling."""
        downsample_factor = max(1, int(scale))
        if downsample_factor == 1:
            return signal
        
        # Downsample and upsample to maintain length
        downsampled = signal[::downsample_factor]
        
        # Interpolate back to original length
        original_indices = np.arange(len(signal))
        downsampled_indices = np.arange(0, len(signal), downsample_factor)
        
        if len(downsampled_indices) > 1:
            upsampled = np.interp(original_indices, downsampled_indices, downsampled)
        else:
            upsampled = np.full(len(signal), downsampled[0] if len(downsampled) > 0 else 0.0)
        
        return upsampled
    
    def get_scale_info(self, scale: float) -> Dict[str, Any]:
        """Get information about wavelet scale."""
        return {
            'scale_factor': scale,
            'wavelet': self.wavelet_name,
            'equivalent_frequency': 1.0 / scale,
            'time_resolution': scale,
            'method': 'wavelet'
        }


class FourierScaleDecomposer(ScaleDecomposer):
    """Fourier-based multi-scale decomposition."""
    
    def decompose(self, signal: np.ndarray, scales: List[float]) -> Dict[float, np.ndarray]:
        """Decompose signal using Fourier filtering."""
        decomposed = {}
        
        # Compute FFT
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        
        for scale in scales:
            # Define frequency band for this scale
            center_freq = 1.0 / scale
            bandwidth = center_freq * 0.8  # Adjustable bandwidth
            
            # Create bandpass filter
            low_freq = max(center_freq - bandwidth/2, 0)
            high_freq = center_freq + bandwidth/2
            
            # Apply filter
            filtered_fft = fft_signal.copy()
            freq_mask = (np.abs(freqs) < low_freq) | (np.abs(freqs) > high_freq)
            filtered_fft[freq_mask] = 0
            
            # Convert back to time domain
            filtered_signal = np.real(np.fft.ifft(filtered_fft))
            decomposed[scale] = filtered_signal
        
        return decomposed
    
    def get_scale_info(self, scale: float) -> Dict[str, Any]:
        """Get information about Fourier scale."""
        return {
            'scale_factor': scale,
            'center_frequency': 1.0 / scale,
            'bandwidth': 0.8 / scale,
            'method': 'fourier'
        }


class LyapunovSpectrumCalculator:
    """Calculate Lyapunov spectrum for a single scale."""
    
    def __init__(self, config: MLSAConfig):
        self.config = config
        self.reconstructor = PhaseSpaceReconstructor(config.embedding_config)
        self.outlier_detector = OutlierDetector(config.numerical_config)
        self.validator = NumericalValidator(config.numerical_config)
    
    def calculate_spectrum(self, signal: np.ndarray, 
                          scale: float) -> Dict[str, Any]:
        """
        Calculate Lyapunov spectrum for signal at given scale with robust error handling.
        """
        results = {
            'scale': scale,
            'signal_length': len(signal),
            'success': False,
            'error': None
        }
        
        try:
            # Early validation
            if len(signal) == 0:
                results['error'] = "Empty signal provided"
                return results
                
            if len(signal) < 20:  # Minimum viable length
                results['error'] = f"Signal too short: {len(signal)} points"
                return results
            
            # Validate input signal
            if self.config.enable_validation:
                validation = self.validator.validate_array(signal, f"scale_{scale}_signal")
                if not validation['is_valid']:
                    results['error'] = f"Invalid input signal: {validation['issues']}"
                    return results
            
            # Remove outliers with size check
            clean_signal, outlier_indices = self.outlier_detector.remove_outliers(
                signal, method='zscore'
            )
            
            if len(clean_signal) == 0:
                results['error'] = "All points removed as outliers"
                return results
                
            if len(clean_signal) < self.config.min_segment_length:
                results['error'] = f"Signal too short after cleaning: {len(clean_signal)}"
                return results
            
            results['outliers_removed'] = len(outlier_indices)
            
            # Perform phase space reconstruction
            if self.config.auto_embedding:
                reconstruction = self.reconstructor.reconstruct(
                    clean_signal,
                    delay=None,
                    dimension=None
                )
            else:
                reconstruction = self.reconstructor.reconstruct(
                    clean_signal,
                    delay=1,
                    dimension=3
                )
            
            if not reconstruction['embedding_success']:
                results['error'] = f"Phase space reconstruction failed: {reconstruction.get('error', 'Unknown')}"
                return results
            
            results['embedding_info'] = {
                'delay': reconstruction['delay'],
                'dimension': reconstruction['dimension'],
                'embedded_points': reconstruction['n_embedded_points']
            }
            
            # Calculate various Lyapunov-related measures
            lyapunov_features = self._extract_lyapunov_features(
                clean_signal, reconstruction['embedded_data'], scale
            )
            
            results.update(lyapunov_features)
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            warnings.warn(f"Lyapunov calculation failed for scale {scale}: {e}")
        
        return results
    
    def _extract_lyapunov_features(self, signal: np.ndarray, 
                                 embedded_data: np.ndarray,
                                 scale: float) -> Dict[str, Any]:
        """Extract comprehensive Lyapunov-based features."""
        features = {}
        
        # 1. Largest Lyapunov exponent from time series
        try:
            dt = 1.0 / scale  # Effective sampling time
            largest_lyap = largest_lyapunov_from_data(
                signal, dt=dt, tau=1, min_neighbors=10
            )
            features['largest_lyapunov'] = largest_lyap
        except Exception as e:
            features['largest_lyapunov'] = np.nan
        
        # 2. Correlation dimension
        try:
            radii, correlations, corr_dim = correlation_dimension(embedded_data)
            features['correlation_dimension'] = corr_dim
            
            # Safe entropy calculation
            valid_correlations = correlations[correlations > 1e-12]
            if len(valid_correlations) > 0:
                features['correlation_entropy'] = entropy(valid_correlations)
            else:
                features['correlation_entropy'] = 0.0
        except Exception as e:
            features['correlation_dimension'] = np.nan
            features['correlation_entropy'] = np.nan
        
        # 3. Lyapunov spectrum estimation from embedded data
        lyap_spectrum = self._estimate_lyapunov_spectrum(embedded_data)
        features['lyapunov_spectrum'] = lyap_spectrum
        
        # 4. Spectral Lyapunov features
        spectral_features = self._calculate_spectral_lyapunov_features(embedded_data)
        features.update(spectral_features)
        
        # 5. Hurst exponent for long-range correlations
        try:
            hurst = hurst_exponent(signal)
            features['hurst_exponent'] = hurst
        except Exception as e:
            features['hurst_exponent'] = 0.5  # Default value
        
        # 6. Complexity measures
        complexity_features = self._calculate_complexity_measures(signal, embedded_data)
        features.update(complexity_features)
        
        return features
    
    def _estimate_lyapunov_spectrum(self, embedded_data: np.ndarray) -> np.ndarray:
        """Estimate Lyapunov spectrum from embedded data."""
        try:
            # Use local linear approximation method
            n_points, dimension = embedded_data.shape
            
            if n_points < 50 or dimension < 2:
                return np.full(dimension, np.nan)
            
            # Sample subset for efficiency
            n_sample = min(n_points, 500)
            indices = np.random.choice(n_points, n_sample, replace=False)
            sample_data = embedded_data[indices]
            
            # Estimate local Jacobians and calculate spectrum
            spectrum = self._local_jacobian_spectrum(sample_data)
            
            return spectrum
            
        except Exception as e:
            warnings.warn(f"Lyapunov spectrum estimation failed: {e}")
            return np.full(embedded_data.shape[1], np.nan)
    
    def _local_jacobian_spectrum(self, data: np.ndarray) -> np.ndarray:
        """Estimate Lyapunov spectrum using local Jacobian approximation with robust error handling."""
        n_points, dimension = data.shape
        
        # Validate input size
        if n_points < max(10, dimension + 2):  # Need minimum points
            return np.full(dimension, np.nan)
        
        lyap_sums = np.zeros(dimension)
        n_valid = 0
        
        # Find neighbors and estimate local Jacobians
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Limit neighbors to available points
            n_neighbors = min(dimension + 2, n_points - 1)
            if n_neighbors < 2:
                return np.full(dimension, np.nan)
                
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
            
            # Limit iterations to avoid excessive computation
            max_iterations = min(50, n_points - 1)
            
            for i in range(max_iterations):
                try:
                    # Get neighbors
                    distances, indices = nbrs.kneighbors([data[i]])
                    
                    # Validate results
                    if distances.size == 0 or indices.size == 0:
                        continue
                        
                    neighbor_indices = indices[0][1:]  # Exclude self
                    if len(neighbor_indices) == 0:
                        continue
                    
                    # Create local coordinate system
                    neighbors = data[neighbor_indices]
                    center = data[i]
                    
                    # Approximate local Jacobian
                    jacobian = self._approximate_jacobian(center, neighbors, data)
                    
                    if jacobian is not None:
                        # Calculate eigenvalues
                        try:
                            eigenvals = np.linalg.eigvals(jacobian)
                            real_parts = np.real(eigenvals)
                            
                            # Sort and accumulate
                            real_parts = np.sort(real_parts)[::-1]
                            lyap_sums[:len(real_parts)] += real_parts
                            n_valid += 1
                            
                        except np.linalg.LinAlgError:
                            continue
                            
                except Exception as e:
                    warnings.warn(f"Neighbor finding failed for point {i}: {e}")
                    continue
            
            # Return results with validation
            if n_valid > 0:
                return lyap_sums / n_valid
            else:
                return np.full(dimension, np.nan)
                
        except Exception as e:
            warnings.warn(f"NearestNeighbors setup failed: {e}")
            return np.full(dimension, np.nan)
    
    def _approximate_jacobian(self, center: np.ndarray, 
                            neighbors: np.ndarray, 
                            data: np.ndarray) -> Optional[np.ndarray]:
        """Approximate local Jacobian matrix with robust error handling."""
        try:
            dimension = len(center)
            
            if len(neighbors) < dimension:
                return None
            
            # Use finite differences with neighbors
            differences = neighbors - center
            
            # Remove too similar points
            norms = np.linalg.norm(differences, axis=1)
            valid_mask = norms > 1e-10
            
            if np.sum(valid_mask) < dimension:
                return None
            
            differences = differences[valid_mask]
            
            # Simple finite difference approximation
            # This is a simplified approach - full implementation would be more sophisticated
            if differences.shape[0] >= differences.shape[1]:
                jacobian = np.cov(differences.T)
            else:
                # If we don't have enough points, create a simple approximation
                jacobian = np.outer(differences[0], differences[0]) / (np.linalg.norm(differences[0]) + 1e-12)
            
            # Ensure it's square and well-conditioned
            if jacobian.shape[0] != jacobian.shape[1]:
                return None
            
            # Check condition number
            try:
                cond_num = np.linalg.cond(jacobian)
                if cond_num > self.config.max_condition_number:
                    return None
            except Exception:
                return None
            
            return jacobian
            
        except Exception:
            return None
    
    def _calculate_spectral_lyapunov_features(self, embedded_data: np.ndarray) -> Dict[str, float]:
        """Calculate spectral features related to Lyapunov analysis with robust error handling."""
        features = {}
        
        try:
            # Power spectral density of each dimension
            n_points, dimension = embedded_data.shape
            
            spectral_entropies = []
            spectral_peaks = []
            
            for dim in range(dimension):
                try:
                    # Calculate PSD
                    freqs, psd = signal.welch(
                        embedded_data[:, dim], 
                        nperseg=min(256, n_points // 4)
                    )
                    
                    # Safe PSD normalization
                    psd_norm = safe_psd_normalize(psd)
                    
                    # Spectral entropy with numerical safety
                    valid_psd = psd_norm[psd_norm > 1e-12]
                    if len(valid_psd) > 0:
                        spectral_entropy = entropy(valid_psd)
                    else:
                        spectral_entropy = 0.0
                    spectral_entropies.append(spectral_entropy)
                    
                    # Dominant frequency
                    peak_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0.0
                    spectral_peaks.append(peak_freq)
                    
                except Exception as e:
                    # Add default values for this dimension
                    spectral_entropies.append(0.0)
                    spectral_peaks.append(0.0)
            
            # Calculate statistics
            features['spectral_entropy_mean'] = np.mean(spectral_entropies) if spectral_entropies else 0.0
            features['spectral_entropy_std'] = np.std(spectral_entropies) if spectral_entropies else 0.0
            features['dominant_frequency_mean'] = np.mean(spectral_peaks) if spectral_peaks else 0.0
            features['dominant_frequency_std'] = np.std(spectral_peaks) if spectral_peaks else 0.0
            
        except Exception as e:
            warnings.warn(f"Spectral feature calculation failed: {e}")
            # Provide default values instead of NaN
            features.update({
                'spectral_entropy_mean': 0.0,
                'spectral_entropy_std': 0.0,
                'dominant_frequency_mean': 0.0,
                'dominant_frequency_std': 0.0
            })
        
        return features
    
    def _calculate_complexity_measures(self, signal: np.ndarray, 
                                     embedded_data: np.ndarray) -> Dict[str, float]:
        """Calculate various complexity measures."""
        features = {}
        
        # 1. Sample entropy
        try:
            features['sample_entropy'] = self._sample_entropy(signal)
        except Exception:
            features['sample_entropy'] = np.nan
        
        # 2. Approximate entropy
        try:
            features['approximate_entropy'] = self._approximate_entropy(signal)
        except Exception:
            features['approximate_entropy'] = np.nan
        
        # 3. Embedding dimension complexity
        try:
            variances = np.var(embedded_data, axis=0)
            total_variance = np.sum(variances)
            if total_variance > 0:
                norm_variances = variances / total_variance
                valid_variances = norm_variances[norm_variances > 1e-12]
                if len(valid_variances) > 0:
                    entropy_dims = entropy(valid_variances)
                    features['dimension_entropy'] = entropy_dims
                    features['effective_dimension'] = np.exp(entropy_dims)
                else:
                    features['dimension_entropy'] = 0.0
                    features['effective_dimension'] = 1.0
            else:
                features['dimension_entropy'] = 0.0
                features['effective_dimension'] = 1.0
        except Exception:
            features['dimension_entropy'] = np.nan
            features['effective_dimension'] = np.nan
        
        # 4. Trajectory complexity
        try:
            # Calculate path length in phase space
            if embedded_data.shape[0] > 1:
                path_diffs = np.diff(embedded_data, axis=0)
                path_lengths = np.linalg.norm(path_diffs, axis=1)
                if len(path_lengths) > 0:
                    features['path_length_mean'] = np.mean(path_lengths)
                    features['path_length_std'] = np.std(path_lengths)
                    mean_length = np.mean(path_lengths)
                    features['path_complexity'] = np.std(path_lengths) / (mean_length + 1e-15)
                else:
                    features.update({
                        'path_length_mean': 0.0,
                        'path_length_std': 0.0,
                        'path_complexity': 0.0
                    })
            else:
                features.update({
                    'path_length_mean': 0.0,
                    'path_length_std': 0.0,
                    'path_complexity': 0.0
                })
        except Exception:
            features.update({
                'path_length_mean': np.nan,
                'path_length_std': np.nan,
                'path_complexity': np.nan
            })
        
        return features
    
    def _sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy with robust error handling."""
        N = len(signal)
        
        if N < m + 1:
            return np.nan
        
        # Safe signal normalization
        signal_norm = safe_signal_normalize(signal)
        
        def _maxdist(data, i, j, m):
            return max([abs(ua - va) for ua, va in zip(data[i:i + m], data[j:j + m])])
        
        try:
            patterns_m = np.zeros(N - m + 1)
            patterns_m1 = np.zeros(N - m)
            
            for i in range(N - m + 1):
                for j in range(i + 1, N - m + 1):
                    if _maxdist(signal_norm, i, j, m) <= r:
                        patterns_m[i] += 1
                        patterns_m[j] += 1
                        
                        if j < N - m and _maxdist(signal_norm, i, j, m + 1) <= r:
                            patterns_m1[i] += 1
                            patterns_m1[j] += 1
            
            phi_m = np.sum(patterns_m) / (N - m + 1)
            phi_m1 = np.sum(patterns_m1) / (N - m)
            
            if phi_m == 0 or phi_m1 == 0 or phi_m1 >= phi_m:
                return np.nan
            
            return -np.log(phi_m1 / phi_m)
            
        except Exception:
            return np.nan
    
    def _approximate_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate approximate entropy with robust error handling."""
        N = len(signal)
        
        if N < m + 1:
            return np.nan
        
        signal_norm = safe_signal_normalize(signal)
        
        def _maxdist(data, i, j, m):
            return max([abs(ua - va) for ua, va in zip(data[i:i + m], data[j:j + m])])
        
        def _phi(m):
            try:
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    for j in range(N - m + 1):
                        if _maxdist(signal_norm, i, j, m) <= r:
                            C[i] += 1.0
                            
                C = C / (N - m + 1)
                valid_C = C[C > 1e-12]
                if len(valid_C) > 0:
                    phi = np.sum(np.log(valid_C)) / (N - m + 1)
                else:
                    phi = 0.0
                return phi
            except Exception:
                return 0.0
        
        try:
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            return phi_m - phi_m1
        except Exception:
            return np.nan


class MLSAExtractor:
    """Main Multi-scale Lyapunov Spectrum Analysis extractor."""
    
    def __init__(self, config: MLSAConfig = None):
        self.config = config or MLSAConfig()
        
        # Initialize decomposer
        if self.config.decomposition_method == 'wavelet':
            self.decomposer = WaveletScaleDecomposer(self.config.wavelet_name)
        elif self.config.decomposition_method == 'fourier':
            self.decomposer = FourierScaleDecomposer()
        else:
            warnings.warn(f"Unknown decomposition method {self.config.decomposition_method}, using Fourier")
            self.decomposer = FourierScaleDecomposer()
        
        # Initialize Lyapunov calculator
        self.lyapunov_calculator = LyapunovSpectrumCalculator(self.config)
        
        # Initialize feature scaler
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Extract multi-scale Lyapunov features from signal.
        
        Args:
            signal: Input time series signal
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        if signal.ndim != 1:
            raise ValueError("Input signal must be 1-dimensional")
        
        if len(signal) < self.config.min_segment_length:
            raise ValueError(f"Signal too short: {len(signal)} < {self.config.min_segment_length}")
        
        results = {
            'original_length': len(signal),
            'scales_analyzed': [],
            'scale_results': {},
            'aggregated_features': {},
            'success': False
        }
        
        try:
            # Step 1: Multi-scale decomposition
            scale_signals = self.decomposer.decompose(signal, self.config.scale_factors)
            
            # Step 2: Analyze each scale
            valid_scales = []
            scale_features_list = []
            
            for scale in self.config.scale_factors:
                if scale not in scale_signals:
                    warnings.warn(f"Scale {scale} not available in decomposition")
                    continue
                
                scale_signal = scale_signals[scale]
                
                # Analyze this scale
                scale_result = self.lyapunov_calculator.calculate_spectrum(
                    scale_signal, scale
                )
                
                results['scale_results'][scale] = scale_result
                
                if scale_result['success']:
                    valid_scales.append(scale)
                    scale_features_list.append(scale_result)
            
            results['scales_analyzed'] = valid_scales
            
            if not valid_scales:
                results['error'] = "No scales analyzed successfully"
                return results
            
            # Step 3: Aggregate features across scales
            aggregated_features = self._aggregate_scale_features(scale_features_list, valid_scales)
            results['aggregated_features'] = aggregated_features
            
            # Step 4: Generate final feature vector
            feature_vector = self._create_feature_vector(aggregated_features)
            results['feature_vector'] = feature_vector
            results['feature_names'] = self.feature_names
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            warnings.warn(f"MLSA feature extraction failed: {e}")
        
        return results
    
    def _aggregate_scale_features(self, scale_features_list: List[Dict], 
                                scales: List[float]) -> Dict[str, Any]:
        """Aggregate features across all scales."""
        aggregated = {}
        
        # Define feature keys to aggregate
        feature_keys = [
            'largest_lyapunov', 'correlation_dimension', 'correlation_entropy',
            'hurst_exponent', 'spectral_entropy_mean', 'spectral_entropy_std',
            'dominant_frequency_mean', 'dominant_frequency_std',
            'sample_entropy', 'approximate_entropy', 'dimension_entropy',
            'effective_dimension', 'path_length_mean', 'path_length_std',
            'path_complexity'
        ]
        
        # Collect values for each feature across scales
        for feature_key in feature_keys:
            values = []
            for scale_result in scale_features_list:
                if feature_key in scale_result and not np.isnan(scale_result[feature_key]):
                    values.append(scale_result[feature_key])
            
            if values:
                aggregated[f'{feature_key}_mean'] = np.mean(values)
                aggregated[f'{feature_key}_std'] = np.std(values)
                aggregated[f'{feature_key}_min'] = np.min(values)
                aggregated[f'{feature_key}_max'] = np.max(values)
                aggregated[f'{feature_key}_range'] = np.max(values) - np.min(values)
                
                # Scale-weighted average (higher scales get more weight)
                if len(values) == len(scales):
                    weights = np.array(scales) / np.sum(scales)
                    aggregated[f'{feature_key}_weighted'] = np.average(values, weights=weights)
                else:
                    aggregated[f'{feature_key}_weighted'] = np.mean(values)
            else:
                # Fill with default values instead of NaN for better stability
                for suffix in ['_mean', '_std', '_min', '_max', '_range', '_weighted']:
                    aggregated[f'{feature_key}{suffix}'] = 0.0
        
        # Scale-specific features
        aggregated['n_valid_scales'] = len(scales)
        aggregated['scale_coverage'] = len(scales) / len(self.config.scale_factors)
        
        # Lyapunov spectrum aggregation
        lyap_spectra = []
        for scale_result in scale_features_list:
            if 'lyapunov_spectrum' in scale_result:
                spectrum = scale_result['lyapunov_spectrum']
                if spectrum is not None and not np.all(np.isnan(spectrum)):
                    lyap_spectra.append(spectrum)
        
        if lyap_spectra:
            # Pad spectra to same length
            max_len = max(len(spec) for spec in lyap_spectra)
            padded_spectra = []
            for spec in lyap_spectra:
                padded = np.full(max_len, np.nan)
                padded[:len(spec)] = spec
                padded_spectra.append(padded)
            
            spectra_array = np.array(padded_spectra)
            
            # Aggregate spectrum statistics with safe calculations
            aggregated['spectrum_mean'] = np.nanmean(spectra_array, axis=0)
            aggregated['spectrum_std'] = np.nanstd(spectra_array, axis=0)
            
            positive_values = spectra_array[spectra_array > 0]
            aggregated['spectrum_max_positive'] = np.nanmax(positive_values) if len(positive_values) > 0 else 0.0
            aggregated['spectrum_sum_positive'] = np.nansum(positive_values) if len(positive_values) > 0 else 0.0
            
            # Kolmogorov entropy approximation
            aggregated['kolmogorov_entropy_approx'] = aggregated['spectrum_sum_positive']
        else:
            # Default values when no spectra available
            aggregated.update({
                'spectrum_mean': np.array([0.0]),
                'spectrum_std': np.array([0.0]),
                'spectrum_max_positive': 0.0,
                'spectrum_sum_positive': 0.0,
                'kolmogorov_entropy_approx': 0.0
            })
        
        return aggregated
    
    def _create_feature_vector(self, aggregated_features: Dict[str, Any]) -> np.ndarray:
        """Create final feature vector from aggregated features."""
        # Define feature order for consistent vector creation
        base_features = [
            'largest_lyapunov', 'correlation_dimension', 'correlation_entropy',
            'hurst_exponent', 'spectral_entropy_mean', 'spectral_entropy_std',
            'dominant_frequency_mean', 'dominant_frequency_std',
            'sample_entropy', 'approximate_entropy', 'dimension_entropy',
            'effective_dimension', 'path_length_mean', 'path_length_std',
            'path_complexity'
        ]
        
        suffixes = ['_mean', '_std', '_min', '_max', '_range', '_weighted']
        
        feature_vector = []
        feature_names = []
        
        # Add base features with all suffixes
        for feature in base_features:
            for suffix in suffixes:
                key = f'{feature}{suffix}'
                value = aggregated_features.get(key, 0.0)  # Use 0.0 instead of np.nan
                feature_vector.append(value)
                feature_names.append(key)
        
        # Add scale-specific features
        scale_features = [
            'n_valid_scales', 'scale_coverage', 'spectrum_max_positive',
            'spectrum_sum_positive', 'kolmogorov_entropy_approx'
        ]
        
        for feature in scale_features:
            value = aggregated_features.get(feature, 0.0)
            feature_vector.append(value)
            feature_names.append(feature)
        
        # Add spectrum statistics (first few components)
        spectrum_mean = aggregated_features.get('spectrum_mean', np.array([0.0]))
        spectrum_std = aggregated_features.get('spectrum_std', np.array([0.0]))
        
        # Ensure arrays are properly handled
        if not isinstance(spectrum_mean, np.ndarray):
            spectrum_mean = np.array([spectrum_mean]) if spectrum_mean is not None else np.array([0.0])
        if not isinstance(spectrum_std, np.ndarray):
            spectrum_std = np.array([spectrum_std]) if spectrum_std is not None else np.array([0.0])
        
        # Include first 5 components of spectrum statistics
        for i in range(5):
            if i < len(spectrum_mean):
                value = spectrum_mean[i] if not np.isnan(spectrum_mean[i]) else 0.0
            else:
                value = 0.0
            feature_vector.append(value)
            feature_names.append(f'spectrum_mean_{i}')
        
        for i in range(5):
            if i < len(spectrum_std):
                value = spectrum_std[i] if not np.isnan(spectrum_std[i]) else 0.0
            else:
                value = 0.0
            feature_vector.append(value)
            feature_names.append(f'spectrum_std_{i}')
        
        # Store feature names for later use
        self.feature_names = feature_names
        
        return np.array(feature_vector, dtype=np.float64)
    
    def fit_scaler(self, feature_vectors: List[np.ndarray]):
        """Fit scaler on a collection of feature vectors."""
        if not feature_vectors:
            raise ValueError("No feature vectors provided for scaler fitting")
        
        # Stack all feature vectors
        features_matrix = np.vstack(feature_vectors)
        
        # Handle NaN values by replacing with column means
        col_means = np.nanmean(features_matrix, axis=0)
        for i, mean_val in enumerate(col_means):
            if np.isnan(mean_val):
                col_means[i] = 0.0
        
        # Fill NaN values
        for i in range(features_matrix.shape[1]):
            mask = np.isnan(features_matrix[:, i])
            features_matrix[mask, i] = col_means[i]
        
        # Fit scaler
        self.scaler.fit(features_matrix)
    
    def transform_features(self, feature_vector: np.ndarray) -> np.ndarray:
        """Transform feature vector using fitted scaler."""
        # Handle NaN values
        feature_vector_clean = feature_vector.copy()
        nan_mask = np.isnan(feature_vector_clean)
        
        if hasattr(self.scaler, 'mean_'):
            # Scaler has been fitted
            feature_vector_clean[nan_mask] = self.scaler.mean_[nan_mask]
        else:
            # Scaler not fitted, use zeros
            feature_vector_clean[nan_mask] = 0.0
        
        # Transform
        return self.scaler.transform(feature_vector_clean.reshape(1, -1))[0]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on variance."""
        if not hasattr(self.scaler, 'var_'):
            return {}
        
        # Use variance as simple importance measure
        importance = {}
        for i, name in enumerate(self.feature_names):
            importance[name] = self.scaler.var_[i] if i < len(self.scaler.var_) else 0.0
        
        # Normalize
        total_var = sum(importance.values())
        if total_var > 0:
            importance = {k: v/total_var for k, v in importance.items()}
        
        return importance


if __name__ == "__main__":
    print(f"✓ Project Root: {PROJECT_ROOT}")
    print(f"✓ Import Manager: {USING_IMPORT_MANAGER}")
    print(f"✓ Module imports successful")
    
    # Example usage and testing
    print("Testing Multi-scale Lyapunov Spectrum Analysis...")
    
    # Generate test signal (chaotic Lorenz system)
    from scipy.integrate import solve_ivp
    
    def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    
    # Generate Lorenz time series
    t_span = (0, 20)
    t_eval = np.arange(0, 20, 0.01)
    initial_state = [1.0, 1.0, 1.0]
    
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')
    test_signal = sol.y[0]  # Use x-component
    
    print(f"Generated test signal with {len(test_signal)} points")
    
    # Create MLSA extractor
    config = MLSAConfig(
        n_scales=3,
        scale_factors=[1, 2, 4],
        decomposition_method='fourier',
        min_segment_length=100
    )
    
    mlsa = MLSAExtractor(config)
    
    # Extract features
    print("Extracting MLSA features...")
    results = mlsa.extract_features(test_signal)
    
    if results['success']:
        print(f"✓ Feature extraction successful!")
        print(f"Scales analyzed: {results['scales_analyzed']}")
        print(f"Feature vector shape: {results['feature_vector'].shape}")
        print(f"Non-NaN features: {np.sum(~np.isnan(results['feature_vector']))}")
        
        # Print some key aggregated features
        key_features = [
            'largest_lyapunov_mean', 'correlation_dimension_mean',
            'hurst_exponent_mean', 'n_valid_scales'
        ]
        
        print("\nKey aggregated features:")
        for feature in key_features:
            value = results['aggregated_features'].get(feature, np.nan)
            print(f"  {feature}: {value:.6f}")
    
    else:
        print(f"✗ Feature extraction failed: {results.get('error', 'Unknown error')}")
    
    print("MLSA testing completed!")