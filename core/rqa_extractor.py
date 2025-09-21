"""
Recurrence Quantification Analysis (RQA) Extractor.

This module implements comprehensive recurrence quantification analysis for 
extracting temporal structure features from chaotic time series, complementing
the MLSA approach with recursive pattern analysis.

Author: C-HiLAP Project
Date: 2025
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import scipy.spatial.distance as distance
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
import logging
import os
import sys

# 导入路径设置
try:
    from setup_imports import setup_project_imports
    setup_project_imports()
except ImportError:
    # 手动设置路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    
# Import project modules
try:
    from phase_space_reconstruction import (
        PhaseSpaceReconstructor, EmbeddingConfig
    )
    from utils.numerical_stability import (
        NumericalConfig, NumericalValidator, OutlierDetector,
        safe_divide, safe_log
    )
    from chaos_utils import (
        correlation_dimension, largest_lyapunov_from_data
    )
except ImportError as e:
    warnings.warn(f"Failed to import required modules: {e}")
    
    # 提供默认类定义以避免NameError
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
        rtol: float = 1e-5
        atol: float = 1e-8
        outlier_threshold: float = 3.0
        stability_check: bool = True
    
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
    
    class NumericalValidator:
        def __init__(self, config=None):
            self.config = config or NumericalConfig()
        
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
    
    class OutlierDetector:
        def __init__(self, config=None):
            self.config = config or NumericalConfig()
        
        def remove_outliers(self, data, method='zscore'):
            if method == 'zscore':
                z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-12))
                mask = z_scores < 3.0
                return data[mask], np.where(~mask)[0]
            else:
                return data, np.array([])
    
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
    
    def safe_divide(numerator, denominator, default=0.0, epsilon=1e-12):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(np.abs(denominator) > epsilon, 
                             numerator / denominator, default)
        return result
    
    def safe_log(x, epsilon=1e-12):
        return np.log(np.maximum(x, epsilon))

@dataclass
class RQAConfig:
    """Configuration for Recurrence Quantification Analysis."""
    
    # Recurrence matrix parameters
    distance_metric: str = 'euclidean'  # 'euclidean', 'manhattan', 'chebyshev', 'cosine'
    threshold_method: str = 'fixed_distance'  # 'fixed_distance', 'fixed_amount', 'fan'
    threshold_value: float = 0.1  # For fixed_distance method
    recurrence_rate_target: float = 0.05  # For fixed_amount method (5% recurrence)
    
    # RQA analysis parameters
    min_diagonal_length: int = 2
    min_vertical_length: int = 2
    theiler_window: int = 1  # Exclude points too close in time
    
    # Multi-scale parameters
    scale_factors: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    scale_method: str = 'coarse_graining'  # 'coarse_graining', 'moving_average'
    
    # Phase space parameters
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    auto_embedding: bool = True
    
    # Quality control
    max_matrix_size: int = 2000  # Maximum size for memory control
    min_recurrence_rate: float = 0.001
    max_recurrence_rate: float = 0.5
    
    # Numerical stability
    numerical_config: NumericalConfig = field(default_factory=NumericalConfig)
    enable_validation: bool = True

class DistanceCalculator:
    """Calculate distances for recurrence matrix construction."""
    
    @staticmethod
    def calculate_distance_matrix(data: np.ndarray, 
                                metric: str = 'euclidean',
                                theiler_window: int = 1) -> np.ndarray:
        """
        Calculate pairwise distance matrix.
        
        Args:
            data: Phase space vectors (n_points, n_dimensions)
            metric: Distance metric
            theiler_window: Exclude diagonal band
            
        Returns:
            Distance matrix
        """
        n_points = data.shape[0]
        
        # Calculate distance matrix using scipy
        if metric == 'euclidean':
            dist_matrix = distance.squareform(distance.pdist(data, 'euclidean'))
        elif metric == 'manhattan':
            dist_matrix = distance.squareform(distance.pdist(data, 'cityblock'))
        elif metric == 'chebyshev':
            dist_matrix = distance.squareform(distance.pdist(data, 'chebyshev'))
        elif metric == 'cosine':
            dist_matrix = distance.squareform(distance.pdist(data, 'cosine'))
        else:
            # Custom metric implementation
            dist_matrix = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(i, n_points):
                    if metric == 'correlation':
                        # Correlation distance
                        corr = np.corrcoef(data[i], data[j])[0, 1]
                        dist = 1 - abs(corr) if not np.isnan(corr) else 1.0
                    else:
                        # Default to euclidean
                        dist = np.linalg.norm(data[i] - data[j])
                    
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
        
        # Apply Theiler window (exclude temporal neighbors)
        if theiler_window > 0:
            for i in range(n_points):
                start_idx = max(0, i - theiler_window)
                end_idx = min(n_points, i + theiler_window + 1)
                dist_matrix[i, start_idx:end_idx] = np.inf
        
        return dist_matrix


class RecurrenceMatrix:
    """Construct and manage recurrence matrices."""
    
    def __init__(self, config: RQAConfig):
        self.config = config
        self.distance_calc = DistanceCalculator()
    
    def construct_matrix(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Construct recurrence matrix from phase space data.
        
        Args:
            data: Phase space vectors
            
        Returns:
            Recurrence matrix and metadata
        """
        metadata = {
            'original_size': data.shape[0],
            'threshold_used': None,
            'recurrence_rate': None,
            'method': self.config.threshold_method
        }
        
        # Limit matrix size for memory efficiency
        if data.shape[0] > self.config.max_matrix_size:
            # Subsample data
            indices = np.linspace(0, data.shape[0] - 1, self.config.max_matrix_size, dtype=int)
            data = data[indices]
            metadata['subsampled'] = True
            metadata['subsample_indices'] = indices
        else:
            metadata['subsampled'] = False
        
        # Calculate distance matrix
        distance_matrix = self.distance_calc.calculate_distance_matrix(
            data, self.config.distance_metric, self.config.theiler_window
        )
        
        # Determine threshold
        if self.config.threshold_method == 'fixed_distance':
            threshold = self.config.threshold_value
            
        elif self.config.threshold_method == 'fixed_amount':
            # Find threshold for target recurrence rate
            finite_distances = distance_matrix[np.isfinite(distance_matrix)]
            if len(finite_distances) == 0:
                threshold = 0.1
            else:
                threshold = np.percentile(finite_distances, 
                                        self.config.recurrence_rate_target * 100)
                
        elif self.config.threshold_method == 'fan':
            # Variable threshold method (advanced)
            threshold = self._calculate_fan_threshold(distance_matrix)
            
        else:
            threshold = self.config.threshold_value
        
        metadata['threshold_used'] = threshold
        
        # Create recurrence matrix
        recurrence_matrix = (distance_matrix <= threshold).astype(np.int8)
        
        # Calculate actual recurrence rate
        total_points = np.sum(np.isfinite(distance_matrix))
        if total_points > 0:
            recurrence_rate = np.sum(recurrence_matrix) / total_points
        else:
            recurrence_rate = 0.0
        
        metadata['recurrence_rate'] = recurrence_rate
        
        # Quality check
        if (recurrence_rate < self.config.min_recurrence_rate or 
            recurrence_rate > self.config.max_recurrence_rate):
            warnings.warn(f"Recurrence rate {recurrence_rate:.4f} outside optimal range "
                         f"[{self.config.min_recurrence_rate}, {self.config.max_recurrence_rate}]")
        
        return recurrence_matrix, metadata
    
    def _calculate_fan_threshold(self, distance_matrix: np.ndarray) -> float:
        """Calculate adaptive threshold using fan method."""
        n_points = distance_matrix.shape[0]
        thresholds = np.zeros(n_points)
        
        for i in range(n_points):
            # Get distances from point i
            distances = distance_matrix[i, :]
            finite_distances = distances[np.isfinite(distances)]
            
            if len(finite_distances) > 1:
                # Use a percentile of local distances
                thresholds[i] = np.percentile(finite_distances, 10)  # 10th percentile
            else:
                thresholds[i] = self.config.threshold_value
        
        return np.median(thresholds)


class RQAAnalyzer:
    """Analyze recurrence matrices and compute RQA measures."""
    
    def __init__(self, config: RQAConfig):
        self.config = config
    
    def analyze(self, recurrence_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive RQA measures.
        
        Args:
            recurrence_matrix: Binary recurrence matrix
            
        Returns:
            Dictionary of RQA measures
        """
        measures = {}
        
        # Basic measures
        measures['RR'] = self._recurrence_rate(recurrence_matrix)
        measures['DET'] = self._determinism(recurrence_matrix)
        measures['LAM'] = self._laminarity(recurrence_matrix)
        
        # Length-based measures
        diagonal_lengths = self._diagonal_line_lengths(recurrence_matrix)
        vertical_lengths = self._vertical_line_lengths(recurrence_matrix)
        
        measures['L_max'] = np.max(diagonal_lengths) if len(diagonal_lengths) > 0 else 0
        measures['L_mean'] = np.mean(diagonal_lengths) if len(diagonal_lengths) > 0 else 0
        measures['L_entropy'] = self._line_entropy(diagonal_lengths)
        
        measures['V_max'] = np.max(vertical_lengths) if len(vertical_lengths) > 0 else 0
        measures['V_mean'] = np.mean(vertical_lengths) if len(vertical_lengths) > 0 else 0
        measures['V_entropy'] = self._line_entropy(vertical_lengths)
        
        # Divergence and trend
        measures['DIV'] = safe_divide(1.0, measures['L_max'], default=np.inf)
        measures['TREND'] = self._trend(recurrence_matrix)
        
        # Entropy measures
        measures['ENTR'] = self._shannon_entropy(diagonal_lengths)
        measures['RATIO'] = safe_divide(measures['DET'], measures['RR'], default=0.0)
        
        # Advanced measures
        measures['TT'] = self._trapping_time(recurrence_matrix)
        measures['RPDE'] = self._recurrence_period_density_entropy(recurrence_matrix)
        
        # Complexity measures
        measures['COMP'] = self._complexity(recurrence_matrix)
        measures['TC'] = self._transitivity_coefficient(recurrence_matrix)
        
        return measures
    
    def _recurrence_rate(self, matrix: np.ndarray) -> float:
        """Calculate recurrence rate (RR)."""
        total_points = matrix.size
        if total_points == 0:
            return 0.0
        return np.sum(matrix) / total_points
    
    def _determinism(self, matrix: np.ndarray) -> float:
        """Calculate determinism (DET) - ratio of recurrent points in diagonal lines."""
        diagonal_lengths = self._diagonal_line_lengths(matrix)
        
        if len(diagonal_lengths) == 0:
            return 0.0
        
        points_in_diagonals = np.sum(diagonal_lengths)
        total_recurrent_points = np.sum(matrix)
        
        return safe_divide(points_in_diagonals, total_recurrent_points, default=0.0)
    
    def _laminarity(self, matrix: np.ndarray) -> float:
        """Calculate laminarity (LAM) - ratio of recurrent points in vertical lines."""
        vertical_lengths = self._vertical_line_lengths(matrix)
        
        if len(vertical_lengths) == 0:
            return 0.0
        
        points_in_verticals = np.sum(vertical_lengths)
        total_recurrent_points = np.sum(matrix)
        
        return safe_divide(points_in_verticals, total_recurrent_points, default=0.0)
    
    def _diagonal_line_lengths(self, matrix: np.ndarray) -> np.ndarray:
        """Find lengths of diagonal lines."""
        n = matrix.shape[0]
        diagonal_lengths = []
        
        # Check all diagonals (both directions from main diagonal)
        for offset in range(-n + 1, n):
            diagonal = np.diagonal(matrix, offset=offset)
            lengths = self._find_line_lengths(diagonal)
            diagonal_lengths.extend(lengths)
        
        # Filter by minimum length
        diagonal_lengths = [length for length in diagonal_lengths 
                          if length >= self.config.min_diagonal_length]
        
        return np.array(diagonal_lengths)
    
    def _vertical_line_lengths(self, matrix: np.ndarray) -> np.ndarray:
        """Find lengths of vertical lines."""
        vertical_lengths = []
        
        # Check each column
        for col in range(matrix.shape[1]):
            column = matrix[:, col]
            lengths = self._find_line_lengths(column)
            vertical_lengths.extend(lengths)
        
        # Filter by minimum length
        vertical_lengths = [length for length in vertical_lengths 
                          if length >= self.config.min_vertical_length]
        
        return np.array(vertical_lengths)
    
    def _find_line_lengths(self, line: np.ndarray) -> List[int]:
        """Find consecutive sequences of 1s in a binary array."""
        if len(line) == 0:
            return []
        
        lengths = []
        current_length = 0
        
        for value in line:
            if value == 1:
                current_length += 1
            else:
                if current_length > 0:
                    lengths.append(current_length)
                    current_length = 0
        
        # Don't forget the last sequence
        if current_length > 0:
            lengths.append(current_length)
        
        return lengths
    
    def _line_entropy(self, lengths: np.ndarray) -> float:
        """Calculate entropy of line length distribution."""
        if len(lengths) == 0:
            return 0.0
        
        # Create histogram
        unique_lengths, counts = np.unique(lengths, return_counts=True)
        probabilities = counts / np.sum(counts)
        
        return entropy(probabilities)
    
    def _shannon_entropy(self, lengths: np.ndarray) -> float:
        """Calculate Shannon entropy of diagonal line lengths."""
        if len(lengths) == 0:
            return 0.0
        
        # Bin the lengths
        max_length = int(np.max(lengths)) if len(lengths) > 0 else 1
        bins = np.arange(1, max_length + 2)  # +2 to include max_length
        
        hist, _ = np.histogram(lengths, bins=bins)
        hist = hist[hist > 0]  # Remove empty bins
        
        if len(hist) == 0:
            return 0.0
        
        probabilities = hist / np.sum(hist)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _trend(self, matrix: np.ndarray) -> float:
        """Calculate trend of recurrence matrix."""
        n = matrix.shape[0]
        if n < 3:
            return 0.0
        
        # Calculate recurrence rate for different bands around main diagonal
        bands = []
        band_distances = []
        
        for offset in range(0, min(n//4, 50)):  # Check first quarter of diagonals
            # Upper diagonal
            upper_diag = np.diagonal(matrix, offset=offset)
            if len(upper_diag) > 10:  # Only consider long enough diagonals
                bands.append(np.mean(upper_diag))
                band_distances.append(offset)
            
            # Lower diagonal (if offset > 0)
            if offset > 0:
                lower_diag = np.diagonal(matrix, offset=-offset)
                if len(lower_diag) > 10:
                    bands.append(np.mean(lower_diag))
                    band_distances.append(offset)
        
        if len(bands) < 3:
            return 0.0
        
        # Linear regression to find trend
        try:
            coeffs = np.polyfit(band_distances, bands, 1)
            return coeffs[0]  # Slope is the trend
        except:
            return 0.0
    
    def _trapping_time(self, matrix: np.ndarray) -> float:
        """Calculate trapping time (TT) - average length of vertical lines."""
        vertical_lengths = self._vertical_line_lengths(matrix)
        
        if len(vertical_lengths) == 0:
            return 0.0
        
        return np.mean(vertical_lengths)
    
    def _recurrence_period_density_entropy(self, matrix: np.ndarray) -> float:
        """Calculate Recurrence Period Density Entropy (RPDE)."""
        try:
            # Find recurrence periods by analyzing diagonal structure
            n = matrix.shape[0]
            periods = []
            
            # Look for periodic patterns in recurrence matrix
            for i in range(n//2):  # Check up to half the length
                if i == 0:
                    continue
                
                # Check if there's a periodic pattern with period i
                correlation = 0
                count = 0
                
                for j in range(n - i):
                    if j + i < n:
                        correlation += matrix[j, j] * matrix[j + i, j + i]
                        count += 1
                
                if count > 0:
                    normalized_corr = correlation / count
                    if normalized_corr > 0.1:  # Threshold for significant correlation
                        periods.append(i)
            
            if len(periods) == 0:
                return 0.0
            
            # Calculate period density
            period_hist, _ = np.histogram(periods, bins=20)
            period_density = period_hist / np.sum(period_hist)
            period_density = period_density[period_density > 0]
            
            # Calculate entropy
            return entropy(period_density)
            
        except:
            return 0.0
    
    def _complexity(self, matrix: np.ndarray) -> float:
        """Calculate complexity measure."""
        # Complexity based on distribution of diagonal and vertical lines
        diagonal_lengths = self._diagonal_line_lengths(matrix)
        vertical_lengths = self._vertical_line_lengths(matrix)
        
        if len(diagonal_lengths) == 0 and len(vertical_lengths) == 0:
            return 0.0
        
        # Combine length distributions
        all_lengths = np.concatenate([diagonal_lengths, vertical_lengths])
        
        if len(all_lengths) == 0:
            return 0.0
        
        # Complexity as normalized variance
        mean_length = np.mean(all_lengths)
        if mean_length == 0:
            return 0.0
        
        complexity = np.std(all_lengths) / mean_length
        return complexity
    
    def _transitivity_coefficient(self, matrix: np.ndarray) -> float:
        """Calculate transitivity coefficient."""
        try:
            n = matrix.shape[0]
            if n < 3:
                return 0.0
            
            triangles = 0
            connected_triples = 0
            
            # Sample a subset for efficiency
            sample_size = min(n, 200)
            indices = np.random.choice(n, sample_size, replace=False)
            
            for i in indices:
                for j in indices:
                    if i != j and matrix[i, j] == 1:
                        for k in indices:
                            if k != i and k != j:
                                if matrix[i, k] == 1 and matrix[j, k] == 1:
                                    triangles += 1
                                if matrix[i, k] == 1 or matrix[j, k] == 1:
                                    connected_triples += 1
            
            if connected_triples == 0:
                return 0.0
            
            return triangles / connected_triples
            
        except:
            return 0.0


class MultiScaleRQA:
    """Perform multi-scale recurrence quantification analysis."""
    
    def __init__(self, config: RQAConfig):
        self.config = config
        self.analyzer = RQAAnalyzer(config)
        self.matrix_constructor = RecurrenceMatrix(config)
    
    def analyze_multiscale(self, data: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Perform RQA analysis at multiple scales.
        
        Args:
            data: Phase space embedded data
            
        Returns:
            Dictionary with results for each scale
        """
        results = {}
        successful_results = []     
        
        for scale in self.config.scale_factors:
            try:
                # Create coarse-grained series
                scaled_data = self._coarse_grain(data, scale)
                
                if scaled_data.shape[0] < 10:  # Too short after coarse-graining
                    warnings.warn(f"Data too short for scale {scale}")
                    continue
                
                # Construct recurrence matrix
                recurrence_matrix, metadata = self.matrix_constructor.construct_matrix(scaled_data)
                
                # Analyze matrix
                rqa_measures = self.analyzer.analyze(recurrence_matrix)
                
                # Store results
                results[scale] = {
                    'rqa_measures': rqa_measures,
                    'matrix_metadata': metadata,
                    'scaled_data_shape': scaled_data.shape,
                    'success': True
                }
                
            except Exception as e:
                results[scale] = {
                    'success': False,
                    'error': str(e)
                }
                warnings.warn(f"RQA analysis failed for scale {scale}: {e}")
        
        return results
    
    def _coarse_grain(self, data: np.ndarray, scale: int) -> np.ndarray:
        """Apply coarse-graining to data."""
        if scale == 1:
            return data
        
        if self.config.scale_method == 'coarse_graining':
            # Non-overlapping averaging
            n_points, n_dims = data.shape
            n_coarse = n_points // scale
            
            coarse_data = np.zeros((n_coarse, n_dims))
            
            for i in range(n_coarse):
                start_idx = i * scale
                end_idx = start_idx + scale
                coarse_data[i] = np.mean(data[start_idx:end_idx], axis=0)
            
            return coarse_data
            
        elif self.config.scale_method == 'moving_average':
            # Moving average downsampling
            from scipy.ndimage import uniform_filter1d
            
            # Apply moving average along time axis
            smoothed = uniform_filter1d(data, size=scale, axis=0)
            
            # Downsample
            return smoothed[::scale]
        
        else:
            # Simple downsampling
            return data[::scale]


class RQAExtractor:
    """Main RQA feature extractor."""
    
    def __init__(self, config: RQAConfig = None):
        self.config = config or RQAConfig()
        
        # Initialize components
        self.reconstructor = PhaseSpaceReconstructor(self.config.embedding_config)
        self.multiscale_rqa = MultiScaleRQA(self.config)
        
        # Initialize preprocessing
        self.outlier_detector = OutlierDetector(self.config.numerical_config)
        self.validator = NumericalValidator(self.config.numerical_config)
        self.scaler = StandardScaler()
        
        self.feature_names = []
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Extract RQA features from time series signal.
        
        Args:
            signal: Input time series
            
        Returns:
            Dictionary containing RQA features and metadata
        """
        if signal.ndim != 1:
            raise ValueError("Input signal must be 1-dimensional")
        
        results = {
            'original_length': len(signal),
            'success': False,
            'error': None
        }
        
        try:
            # Step 1: Input validation
            if self.config.enable_validation:
                validation = self.validator.validate_array(signal, "input_signal")
                if not validation['is_valid']:
                    results['error'] = f"Invalid input signal: {validation['issues']}"
                    return results
            
            # Step 2: Preprocessing
            clean_signal, outlier_indices = self.outlier_detector.remove_outliers(
                signal, method='zscore'
            )
            results['outliers_removed'] = len(outlier_indices)
            
            # Step 3: Phase space reconstruction
            if self.config.auto_embedding:
                reconstruction = self.reconstructor.reconstruct(
                    clean_signal, delay=None, dimension=None
                )
            else:
                reconstruction = self.reconstructor.reconstruct(
                    clean_signal, delay=1, dimension=3
                )
            
            if not reconstruction['embedding_success']:
                results['error'] = f"Phase space reconstruction failed: {reconstruction.get('error', 'Unknown')}"
                return results
            
            results['embedding_info'] = {
                'delay': reconstruction['delay'],
                'dimension': reconstruction['dimension'],
                'embedded_points': reconstruction['n_embedded_points']
            }
            
            # Step 4: Multi-scale RQA analysis
            embedded_data = reconstruction['embedded_data']
            scale_results = self.multiscale_rqa.analyze_multiscale(embedded_data)
            
            results['scale_results'] = scale_results
            
            # Step 5: Aggregate features across scales
            aggregated_features = self._aggregate_rqa_features(scale_results)
            results['aggregated_features'] = aggregated_features
            
            # Step 6: Create feature vector
            feature_vector = self._create_feature_vector(aggregated_features)
            results['feature_vector'] = feature_vector
            results['feature_names'] = self.feature_names
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            warnings.warn(f"RQA feature extraction failed: {e}")
        
        return results
    
    def _aggregate_rqa_features(self, scale_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate RQA features across scales."""
        aggregated = {}
        
        # Extract successful results
        successful_results = {
            scale: result for scale, result in scale_results.items()
            if result.get('success', False)
        }
        
        if not successful_results:
            return {}
        
        # Define RQA measures to aggregate
        rqa_measures = [
            'RR', 'DET', 'LAM', 'L_max', 'L_mean', 'L_entropy',
            'V_max', 'V_mean', 'V_entropy', 'DIV', 'TREND',
            'ENTR', 'RATIO', 'TT', 'RPDE', 'COMP', 'TC'
        ]
        
        # Aggregate each measure across scales
        for measure in rqa_measures:
            values = []
            scales = []
            
            for scale, result in successful_results.items():
                if measure in result['rqa_measures']:
                    value = result['rqa_measures'][measure]
                    if np.isfinite(value):
                        values.append(value)
                        scales.append(scale)
            
            if values:
                # Basic statistics
                aggregated[f'{measure}_mean'] = np.mean(values)
                aggregated[f'{measure}_std'] = np.std(values)
                aggregated[f'{measure}_min'] = np.min(values)
                aggregated[f'{measure}_max'] = np.max(values)
                aggregated[f'{measure}_range'] = np.max(values) - np.min(values)
                
                # Scale-weighted average
                if len(values) == len(scales):
                    weights = np.array(scales) / np.sum(scales)
                    aggregated[f'{measure}_weighted'] = np.average(values, weights=weights)
                else:
                    aggregated[f'{measure}_weighted'] = np.mean(values)
                
                # Trend across scales
                if len(values) >= 3:
                    try:
                        trend_coeffs = np.polyfit(scales, values, 1)
                        aggregated[f'{measure}_trend'] = trend_coeffs[0]
                    except:
                        aggregated[f'{measure}_trend'] = 0.0
                else:
                    aggregated[f'{measure}_trend'] = 0.0
            else:
                # Fill with NaN if no valid values
                for suffix in ['_mean', '_std', '_min', '_max', '_range', '_weighted', '_trend']:
                    aggregated[f'{measure}{suffix}'] = np.nan
        
        # Scale-specific aggregated features
        aggregated['n_successful_scales'] = len(successful_results)
        aggregated['scale_success_rate'] = len(successful_results) / len(self.config.scale_factors)
        
        # Recurrence matrix quality metrics
        recurrence_rates = []
        matrix_sizes = []
        
        for result in successful_results.values():
            if 'matrix_metadata' in result:
                metadata = result['matrix_metadata']
                if 'recurrence_rate' in metadata:
                    recurrence_rates.append(metadata['recurrence_rate'])
                if 'original_size' in metadata:
                    matrix_sizes.append(metadata['original_size'])
        
        if recurrence_rates:
            aggregated['recurrence_rate_mean'] = np.mean(recurrence_rates)
            aggregated['recurrence_rate_std'] = np.std(recurrence_rates)
        else:
            aggregated['recurrence_rate_mean'] = np.nan
            aggregated['recurrence_rate_std'] = np.nan
        
        # Complexity-based aggregated features
        complexity_measures = ['DET', 'LAM', 'COMP']
        complexity_values = []
        
        for measure in complexity_measures:
            key = f'{measure}_mean'
            if key in aggregated and np.isfinite(aggregated[key]):
                complexity_values.append(aggregated[key])
        
        if complexity_values:
            aggregated['overall_complexity'] = np.mean(complexity_values)
            aggregated['complexity_variance'] = np.var(complexity_values)
        else:
            aggregated['overall_complexity'] = np.nan
            aggregated['complexity_variance'] = np.nan
        
        return aggregated
    
    def _create_feature_vector(self, aggregated_features: Dict[str, Any]) -> np.ndarray:
        """Create feature vector from aggregated RQA features."""
        # Define RQA measures and their suffixes
        base_measures = [
            'RR', 'DET', 'LAM', 'L_max', 'L_mean', 'L_entropy',
            'V_max', 'V_mean', 'V_entropy', 'DIV', 'TREND',
            'ENTR', 'RATIO', 'TT', 'RPDE', 'COMP', 'TC'
        ]
        
        suffixes = ['_mean', '_std', '_min', '_max', '_range', '_weighted', '_trend']
        
        feature_vector = []
        feature_names = []
        
        # Add base RQA measures with all statistics
        for measure in base_measures:
            for suffix in suffixes:
                key = f'{measure}{suffix}'
                value = aggregated_features.get(key, np.nan)
                feature_vector.append(value)
                feature_names.append(key)
        
        # Add meta-features
        meta_features = [
            'n_successful_scales', 'scale_success_rate',
            'recurrence_rate_mean', 'recurrence_rate_std',
            'overall_complexity', 'complexity_variance'
        ]
        
        for feature in meta_features:
            value = aggregated_features.get(feature, np.nan)
            feature_vector.append(value)
            feature_names.append(feature)
        
        # Store feature names
        self.feature_names = feature_names
        
        return np.array(feature_vector)
    
    def fit_scaler(self, feature_vectors: List[np.ndarray]):
        """Fit scaler on collection of feature vectors."""
        if not feature_vectors:
            raise ValueError("No feature vectors provided for scaler fitting")
        
        # Stack feature vectors
        features_matrix = np.vstack(feature_vectors)
        
        # Handle NaN values
        col_means = np.nanmean(features_matrix, axis=0)
        for i, mean_val in enumerate(col_means):
            if np.isnan(mean_val):
                col_means[i] = 0.0
        
        # Fill NaN values with column means
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
            feature_vector_clean[nan_mask] = self.scaler.mean_[nan_mask]
        else:
            feature_vector_clean[nan_mask] = 0.0
        
        return self.scaler.transform(feature_vector_clean.reshape(1, -1))[0]


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Recurrence Quantification Analysis...")
    
    # Generate test signal from Lorenz system
    from scipy.integrate import solve_ivp
    
    def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    
    # Generate Lorenz time series
    t_span = (0, 20)
    t_eval = np.arange(0, 20, 0.02)  # Slightly coarser for efficiency
    initial_state = [1.0, 1.0, 1.0]
    
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')
    test_signal = sol.y[0]  # Use x-component
    
    print(f"Generated test signal with {len(test_signal)} points")
    
    # Create RQA extractor
    config = RQAConfig(
        scale_factors=[1, 2, 4],
        threshold_method='fixed_amount',
        recurrence_rate_target=0.05,
        distance_metric='euclidean',
        min_diagonal_length=2,
        min_vertical_length=2
    )
    
    rqa = RQAExtractor(config)
    
    # Extract features
    print("Extracting RQA features...")
    results = rqa.extract_features(test_signal)
    
    if results['success']:
        print(f"✓ RQA feature extraction successful!")
        print(f"Embedding: delay={results['embedding_info']['delay']}, "
              f"dimension={results['embedding_info']['dimension']}")
        print(f"Successful scales: {results['scale_results'].keys()}")
        print(f"Feature vector shape: {results['feature_vector'].shape}")
        print(f"Non-NaN features: {np.sum(~np.isnan(results['feature_vector']))}")
        
        # Print some key RQA features
        key_features = [
            'RR_mean', 'DET_mean', 'LAM_mean', 'L_max_mean',
            'COMP_mean', 'overall_complexity'
        ]
        
        print("\nKey RQA features:")
        for feature in key_features:
            value = results['aggregated_features'].get(feature, np.nan)
            print(f"  {feature}: {value:.6f}")
    
    else:
        print(f"✗ RQA feature extraction failed: {results.get('error', 'Unknown error')}")
    
    print("RQA testing completed!")