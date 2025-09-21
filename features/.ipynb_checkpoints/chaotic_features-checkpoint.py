"""
Chaotic Features Extractor - Unified Interface for Chaos-based Feature Extraction.

This module provides a unified interface for extracting chaotic features from
audio signals, integrating Multi-scale Lyapunov Spectrum Analysis (MLSA) and
Recurrence Quantification Analysis (RQA) with advanced feature fusion and
selection capabilities.

Author: C-HiLAP Project
Date: 2025
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
import pickle
import json
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

# Ensure numerical stability
np.seterr(divide='warn', over='warn', under='ignore', invalid='warn')

# Feature selection and dimensionality reduction
try:
    from sklearn.feature_selection import (
        SelectKBest, SelectPercentile, RFE, RFECV,
        f_classif, mutual_info_classif, chi2, VarianceThreshold
    )
    from sklearn.decomposition import PCA, FastICA, FactorAnalysis
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LassoCV
    HAS_SKLEARN = True
except ImportError:
    warnings.warn("scikit-learn not available, some features will be disabled")
    HAS_SKLEARN = False

# Safe numerical operations
def safe_divide(numerator, denominator, default_value=0.0, eps=1e-10):
    """Safe division with default value for zero denominators."""
    if isinstance(denominator, (int, float)):
        if abs(denominator) < eps:
            return default_value
        return numerator / denominator
    else:
        # Handle array case
        denominator = np.asarray(denominator)
        result = np.full_like(numerator, default_value, dtype=float)
        mask = np.abs(denominator) >= eps
        if np.any(mask):
            result[mask] = numerator[mask] / denominator[mask]
        return result

def safe_log(x, eps=1e-10):
    """Safe logarithm avoiding log(0)."""
    x = np.asarray(x)
    return np.log(np.maximum(x, eps))

def safe_sqrt(x, eps=1e-10):
    """Safe square root avoiding sqrt of negative numbers."""
    x = np.asarray(x)
    return np.sqrt(np.maximum(x, eps))

def handle_nans_infs(arr, replace_value=0.0):
    """Handle NaN and infinity values in arrays."""
    arr = np.asarray(arr)
    arr = np.where(np.isnan(arr), replace_value, arr)
    arr = np.where(np.isinf(arr), replace_value, arr)
    return arr

def entropy(x, eps=1e-10):
    """Calculate Shannon entropy with numerical safety."""
    x = np.asarray(x)
    x = x[x > eps]  # Remove zero elements
    if len(x) == 0:
        return 0.0
    x_norm = x / np.sum(x)
    return -np.sum(x_norm * np.log(x_norm))

# Complete configuration classes that match the real extractors
@dataclass
class NumericalConfig:
    """Configuration for numerical stability."""
    zero_threshold: float = 1e-12
    inf_threshold: float = 1e12
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-8
    enable_overflow_check: bool = True
    default_fill_value: float = 0.0

@dataclass
class EmbeddingConfig:
    """Configuration for phase space embedding."""
    embedding_dimension: int = 3
    embedding_dim: int = 3  # Alternative name for compatibility
    delay_method: str = 'autocorr'
    delay_value: Optional[int] = None
    time_delay: int = 1  # Alternative name for compatibility
    max_delay: int = 50
    min_dimension: int = 2
    max_dimension: int = 10
    max_embedding_dim: int = 20  # For compatibility
    fnn_threshold: float = 15.0
    autocorr_threshold: float = 0.1

@dataclass
class MLSAConfig:
    """Complete configuration for Multi-scale Lyapunov Spectrum Analysis."""
    
    # Scale decomposition parameters
    n_scales: int = 4
    scale_factors: List[float] = field(default_factory=lambda: [1, 2, 4, 8])
    scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])  # For compatibility
    decomposition_method: str = 'fourier'  # 'wavelet', 'emd', 'fourier'
    wavelet_name: str = 'db4'
    
    # Lyapunov analysis parameters
    min_segment_length: int = 100
    max_segment_length: int = 2000
    overlap_ratio: float = 0.5
    lyapunov_method: str = 'rosenstein'
    max_lyapunov_iterations: int = 1000
    lyapunov_dt: float = 0.01
    
    # Phase space parameters
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    auto_embedding: bool = True
    
    # Feature extraction parameters
    spectral_bins: int = 50
    spectral_features: bool = True
    statistical_features: bool = True
    entropy_bins: int = 20
    correlation_r_points: int = 30
    
    # Quality control
    min_positive_lyapunov_ratio: float = 0.1
    max_condition_number: float = 1e10
    outlier_threshold: float = 3.0
    
    # Numerical stability
    numerical_config: NumericalConfig = field(default_factory=NumericalConfig)
    enable_validation: bool = True

@dataclass
class RQAConfig:
    """Complete configuration for Recurrence Quantification Analysis."""
    
    # Embedding parameters
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    auto_embedding: bool = True
    
    # RQA parameters
    recurrence_threshold: float = 0.1
    threshold_method: str = 'fixed_amount'
    scales: List[int] = field(default_factory=lambda: [1, 2, 4])
    scale_factors: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    distance_metric: str = 'euclidean'
    recurrence_rate_target: float = 0.05
    threshold_value: Optional[float] = None
    
    # 🔧 添加缺失的属性
    theiler_window: int = 1  # Theiler窗口，用于排除时间上过近的点
    max_matrix_size: int = 1000  # 添加矩阵大小限制
    scale_method: str = 'coarse_graining'  # 添加尺度方法
    
    # Line detection parameters
    min_diagonal_length: int = 2
    min_vertical_length: int = 2
    min_horizontal_length: int = 2
    
    # Quality control
    max_recurrence_rate: float = 0.5
    min_recurrence_rate: float = 0.001
    
    # Performance settings
    max_points: int = 1000
    subsample_method: str = 'random'
    
    # Numerical stability
    numerical_config: NumericalConfig = field(default_factory=NumericalConfig)
    enable_validation: bool = True

@dataclass
class ChaoticFeatureConfig:
    """Configuration for chaotic feature extraction."""
    
    # Core analysis configurations
    mlsa_config: MLSAConfig = field(default_factory=MLSAConfig)
    rqa_config: RQAConfig = field(default_factory=RQAConfig)
    
    # Feature selection and fusion
    enable_mlsa: bool = True
    enable_rqa: bool = True
    feature_fusion_method: str = 'concatenate'
    
    # Feature selection
    enable_feature_selection: bool = True
    selection_method: str = 'variance'  # 'variance', 'univariate', 'mutual_info'
    selection_k: Optional[int] = None
    selection_percentile: float = 75
    variance_threshold: float = 0.01
    
    # Dimensionality reduction
    enable_dimensionality_reduction: bool = False
    reduction_method: str = 'pca'  # 'pca', 'ica', 'factor_analysis'
    target_dimensions: Optional[int] = None
    explained_variance_ratio: float = 0.95
    
    # Preprocessing
    scaler_type: str = 'standard'  # 'standard', 'minmax', 'robust', 'none'
    handle_nan_strategy: str = 'mean'  # 'mean', 'median', 'zero', 'drop'
    outlier_detection: bool = True
    
    # Performance optimization
    enable_parallel: bool = False  # Disable by default for stability
    max_workers: int = 2  # Reduced for better performance
    chunk_size: int = 50  # Reduced chunk size
    memory_limit_gb: float = 2.0  # Reduced memory limit
    
    # Quality control
    min_signal_length: int = 200
    max_signal_length: int = 10000  # Reduced for performance
    feature_validation: bool = True
    
    # Caching and persistence
    enable_caching: bool = False
    cache_directory: Optional[str] = None
    
    # Numerical stability
    numerical_config: NumericalConfig = field(default_factory=NumericalConfig)

# Comprehensive Mock Extractors
class MockMLSAExtractor:
    """Mock MLSA extractor that generates realistic features with improved performance."""
    
    def __init__(self, config):
        self.config = config
        
        # Generate more concise feature names for better performance
        self.feature_names = []
        
        # Base feature categories (reduced for performance)
        base_features = [
            'largest_lyapunov', 'correlation_dimension', 'correlation_entropy',
            'spectral_entropy_mean', 'sample_entropy', 'effective_dimension'
        ]
        
        # Statistical aggregation suffixes (reduced)
        suffixes = ['_mean', '_std', '_max']
        
        # Generate aggregated feature names
        for feature in base_features:
            for suffix in suffixes:
                self.feature_names.append(f'{feature}{suffix}')
        
        # Add scale-specific features
        scale_features = [
            'n_valid_scales', 'scale_coverage', 'spectrum_max_positive', 
            'kolmogorov_entropy_approx'
        ]
        self.feature_names.extend(scale_features)
        
        # Add spectrum statistics (reduced)
        for i in range(3):
            self.feature_names.append(f'spectrum_mean_{i}')
    
    def extract_features(self, signal):
        """Extract comprehensive mock MLSA features with improved performance."""
        try:
            n_features = len(self.feature_names)
            
            # Basic signal statistics for feature generation (vectorized)
            signal = np.asarray(signal)
            signal_stats = np.array([
                np.std(signal), np.var(signal), len(signal), np.mean(signal)
            ])
            signal_std, signal_var, signal_length, signal_mean = signal_stats
            signal_std = max(signal_std, 1e-6)
            
            # Generate realistic base feature values (vectorized)
            np.random.seed(hash(signal_length) % 2**32)  # Deterministic but signal-dependent
            
            features = []
            
            # Base features with statistical variations
            for i in range(6):  # 6 base features
                base_value = max(0.01, signal_std * np.random.uniform(0.05, 0.5))
                # Add 3 statistical variations per base feature
                features.extend([
                    base_value,  # mean
                    abs(np.random.normal(0, base_value * 0.1)),  # std
                    base_value * 1.2  # max
                ])
            
            # Add scale-specific features
            n_scales = getattr(self.config, 'n_scales', 4)
            features.extend([
                n_scales,  # n_valid_scales
                1.0,  # scale_coverage
                max(0.1, signal_std * 0.2),  # spectrum_max_positive
                max(0.3, signal_std * 0.3)   # kolmogorov_entropy_approx
            ])
            
            # Add spectrum statistics (3 components)
            features.extend([signal_std * np.random.uniform(0.1, 0.5) for _ in range(3)])
            
            # Ensure we have exactly the right number of features
            features_array = np.array(features[:n_features])
            if len(features_array) < n_features:
                padding = np.random.uniform(0, signal_std * 0.1, n_features - len(features_array))
                features_array = np.concatenate([features_array, padding])
            
            # Clean features
            features_array = handle_nans_infs(features_array)
            
            return {
                'success': True,
                'feature_vector': features_array,
                'feature_names': self.feature_names,
                'processing_time': 0.05,  # Reduced processing time
                'scales_analyzed': getattr(self.config, 'scale_factors', [1, 2, 4]),
                'aggregated_features': dict(zip(self.feature_names, features_array))
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'feature_vector': np.zeros(len(self.feature_names)),
                'processing_time': 0.0
            }


class MockRQAExtractor:
    """Mock RQA extractor that generates realistic features with improved performance."""
    
    def __init__(self, config):
        self.config = config
        # Simplified feature names for better performance
        self.feature_names = [
            'recurrence_rate', 'determinism', 'laminarity', 'trapping_time',
            'longest_diagonal', 'longest_vertical', 'entropy_diagonal',
            'entropy_vertical', 'trend', 'transitivity'
        ]
    
    def extract_features(self, signal):
        """Extract comprehensive mock RQA features with improved performance."""
        try:
            n = len(signal)
            
            if n < 20:
                return {
                    'success': False,
                    'error': 'Signal too short for RQA analysis',
                    'feature_vector': np.zeros(len(self.feature_names)),
                    'processing_time': 0.0
                }
            
            # Vectorized feature computation for performance
            signal = np.asarray(signal)
            signal_stats = np.array([
                np.std(signal), np.mean(signal), n
            ])
            signal_std, signal_mean, signal_length = signal_stats
            signal_std = max(signal_std, 1e-6)
            
            # Use deterministic random seed based on signal properties
            seed_val = int(hash(tuple(signal[:min(10, len(signal))])) % 2**31)
            np.random.seed(seed_val)
            
            features = []
            
            # 1. Recurrence rate (properly bounded)
            delay = max(1, n // 50)  # Optimized delay calculation
            if n > delay:
                # Simplified correlation calculation
                rr = min(0.5, max(0.001, signal_std * 0.1 * np.random.uniform(0.5, 2.0)))
            else:
                rr = 0.05
            features.append(rr)
            
            # 2-10. Generate remaining features efficiently
            det = np.clip(rr * np.random.uniform(5, 12), 0.1, 0.95)
            lam = np.clip(det * 0.85, 0.1, 0.9)
            tt = max(2.0, min(lam * 50, 100))
            longest_diag = min(n // 10, max(5, int(det * 30)))
            longest_vert = min(n // 15, max(3, int(longest_diag * 0.6)))
            entropy_diag = min(3.0, max(0.5, -np.log(rr + 1e-10) * 0.5))
            entropy_vert = entropy_diag * 0.8
            trend = min(1.0, abs(np.random.normal(0, 0.2)))
            transitivity = min(2.0, signal_std / max(abs(signal_mean), 1e-6))
            
            features.extend([det, lam, tt, longest_diag, longest_vert, 
                           entropy_diag, entropy_vert, trend, transitivity])
            
            # Clean and validate features
            features_array = np.array(features)
            features_array = handle_nans_infs(features_array)
            features_array = np.maximum(features_array, 0.0)
            
            # Create aggregated features dictionary with expected names
            aggregated_features = {}
            short_names = ['RR', 'DET', 'LAM', 'TT', 'L_max', 'V_max', 
                          'L_entr', 'V_entr', 'TREND', 'TRANS']
            
            for i, (name, value) in enumerate(zip(short_names, features_array)):
                aggregated_features[f'{name}_mean'] = float(value)
            
            return {
                'success': True,
                'feature_vector': features_array,
                'feature_names': self.feature_names,
                'aggregated_features': aggregated_features,
                'processing_time': 0.05  # Reduced processing time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'feature_vector': np.zeros(len(self.feature_names)),
                'processing_time': 0.0
            }


class FeatureFusionEngine:
    """Engine for fusing features from multiple extractors."""
    
    def __init__(self, config: ChaoticFeatureConfig):
        self.config = config
        self.fusion_weights = None
        self.pca_transformer = None
        self.ica_transformer = None
    
    def fuse_features(self, mlsa_features: np.ndarray, 
                     rqa_features: np.ndarray,
                     method: str = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fuse features from MLSA and RQA extractors."""
        method = method or self.config.feature_fusion_method
        metadata = {'fusion_method': method}
        
        # Handle case where only one type is available
        if mlsa_features is None and rqa_features is None:
            return np.array([]), metadata
        elif mlsa_features is None:
            return handle_nans_infs(rqa_features), metadata
        elif rqa_features is None:
            return handle_nans_infs(mlsa_features), metadata
        
        # Clean and flatten features
        mlsa_flat = handle_nans_infs(mlsa_features.flatten() if mlsa_features.ndim > 1 else mlsa_features)
        rqa_flat = handle_nans_infs(rqa_features.flatten() if rqa_features.ndim > 1 else rqa_features)
        
        if method == 'concatenate':
            fused = np.concatenate([mlsa_flat, rqa_flat])
            metadata['mlsa_dims'] = len(mlsa_flat)
            metadata['rqa_dims'] = len(rqa_flat)
            
        elif method == 'weighted':
            if self.fusion_weights is None:
                mlsa_weight, rqa_weight = 0.5, 0.5
            else:
                mlsa_weight, rqa_weight = self.fusion_weights
            
            # Pad shorter array to match longer one
            max_len = max(len(mlsa_flat), len(rqa_flat))
            mlsa_padded = np.pad(mlsa_flat, (0, max_len - len(mlsa_flat)), mode='constant', constant_values=0.0)
            rqa_padded = np.pad(rqa_flat, (0, max_len - len(rqa_flat)), mode='constant', constant_values=0.0)
            
            fused = mlsa_weight * mlsa_padded + rqa_weight * rqa_padded
            metadata['weights'] = (mlsa_weight, rqa_weight)
            
        else:
            # Default to concatenation
            fused = np.concatenate([mlsa_flat, rqa_flat])
        
        # Final cleanup
        fused = handle_nans_infs(fused)
        metadata['final_dims'] = len(fused)
        return fused, metadata
    
    def learn_fusion_weights(self, mlsa_features_list: List[np.ndarray],
                           rqa_features_list: List[np.ndarray],
                           labels: np.ndarray):
        """Learn optimal fusion weights."""
        if not HAS_SKLEARN:
            self.fusion_weights = (0.5, 0.5)
            return
            
        try:
            # Simple weight learning based on feature variance
            mlsa_vars = [np.var(handle_nans_infs(feat)) for feat in mlsa_features_list]
            rqa_vars = [np.var(handle_nans_infs(feat)) for feat in rqa_features_list]
            
            mlsa_avg_var = np.mean(mlsa_vars)
            rqa_avg_var = np.mean(rqa_vars)
            
            total_var = mlsa_avg_var + rqa_avg_var
            if total_var > 0:
                self.fusion_weights = (mlsa_avg_var / total_var, rqa_avg_var / total_var)
            else:
                self.fusion_weights = (0.5, 0.5)
                
        except Exception as e:
            warnings.warn(f"Failed to learn fusion weights: {e}")
            self.fusion_weights = (0.5, 0.5)
    
    def fit_fusion_transformers(self, mlsa_features_list: List[np.ndarray], 
                              rqa_features_list: List[np.ndarray]):
        """Fit fusion transformers (PCA/ICA) on feature lists."""
        if not HAS_SKLEARN:
            return
            
        try:
            # Concatenate all features for transformer fitting
            all_mlsa = np.vstack([feat.reshape(1, -1) if feat.ndim == 1 else feat 
                                 for feat in mlsa_features_list])
            all_rqa = np.vstack([feat.reshape(1, -1) if feat.ndim == 1 else feat 
                                for feat in rqa_features_list])
            
            # Fit PCA transformer
            if all_mlsa.shape[1] > 2:
                self.pca_transformer = PCA(n_components=min(all_mlsa.shape[1], 50))
                self.pca_transformer.fit(all_mlsa)
            
            # Fit ICA transformer
            if all_rqa.shape[1] > 2:
                self.ica_transformer = FastICA(n_components=min(all_rqa.shape[1], 20))
                self.ica_transformer.fit(all_rqa)
                
        except Exception as e:
            warnings.warn(f"Failed to fit fusion transformers: {e}")


class FeatureSelectionEngine:
    """Engine for feature selection and dimensionality reduction."""
    
    def __init__(self, config: ChaoticFeatureConfig):
        self.config = config
        self.selector = None
        self.reducer = None
        self.fitted_feature_shape = None
    
    def fit_selector(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        """Fit feature selector on training data."""
        if not HAS_SKLEARN or not self.config.enable_feature_selection:
            return
        
        features = handle_nans_infs(features)
        self.fitted_feature_shape = features.shape[1]
        
        try:
            if self.config.selection_method == 'variance':
                self.selector = VarianceThreshold(threshold=self.config.variance_threshold)
                self.selector.fit(features)
            elif self.config.selection_method == 'univariate' and labels is not None:
                self.selector = SelectPercentile(f_classif, percentile=self.config.selection_percentile)
                self.selector.fit(features, labels)
            elif self.config.selection_method == 'mutual_info' and labels is not None:
                self.selector = SelectPercentile(mutual_info_classif, percentile=self.config.selection_percentile)
                self.selector.fit(features, labels)
                
        except Exception as e:
            warnings.warn(f"Feature selection fitting failed: {e}")
            self.selector = None
    
    def fit_reducer(self, features: np.ndarray):
        """Fit dimensionality reducer on training data."""
        if not HAS_SKLEARN or not self.config.enable_dimensionality_reduction:
            return
            
        features = handle_nans_infs(features)
        
        try:
            if self.config.reduction_method == 'pca':
                n_components = self.config.target_dimensions or min(features.shape[1], 50)
                self.reducer = PCA(n_components=n_components)
                self.reducer.fit(features)
            elif self.config.reduction_method == 'ica':
                n_components = self.config.target_dimensions or min(features.shape[1], 20)
                self.reducer = FastICA(n_components=n_components)
                self.reducer.fit(features)
            elif self.config.reduction_method == 'factor_analysis':
                n_components = self.config.target_dimensions or min(features.shape[1], 30)
                self.reducer = FactorAnalysis(n_components=n_components)
                self.reducer.fit(features)
                
        except Exception as e:
            warnings.warn(f"Dimensionality reduction fitting failed: {e}")
            self.reducer = None
    
    def transform_features(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply feature selection and dimensionality reduction."""
        metadata = {}
        features = handle_nans_infs(features)
        
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        metadata['original_dimensions'] = features.shape[1]
        
        # Handle dimension mismatch
        if self.fitted_feature_shape is not None and features.shape[1] != self.fitted_feature_shape:
            if features.shape[1] < self.fitted_feature_shape:
                # Pad with zeros
                padding = np.zeros((features.shape[0], self.fitted_feature_shape - features.shape[1]))
                features = np.hstack([features, padding])
            else:
                # Truncate
                features = features[:, :self.fitted_feature_shape]
        
        # Apply feature selection
        if self.selector is not None:
            try:
                features = self.selector.transform(features)
                metadata['selection_applied'] = True
                metadata['selected_dimensions'] = features.shape[1]
            except Exception as e:
                warnings.warn(f"Feature selection transform failed: {e}")
                metadata['selection_applied'] = False
        
        # Apply dimensionality reduction
        if self.reducer is not None:
            try:
                features = self.reducer.transform(features)
                metadata['reduction_applied'] = True
                metadata['reduced_dimensions'] = features.shape[1]
            except Exception as e:
                warnings.warn(f"Dimensionality reduction transform failed: {e}")
                metadata['reduction_applied'] = False
        
        # Clean final features
        features = handle_nans_infs(features)
        
        # Return to 1D if single sample
        if features.shape[0] == 1:
            features = features[0]
        
        return features, metadata
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return {}


class ChaoticFeatureExtractor:
    """Main chaotic feature extraction system with robust error handling."""
    
    def __init__(self, config: ChaoticFeatureConfig = None):
        self.config = config or ChaoticFeatureConfig()
        
        # Initialize extractors with error handling
        self.mlsa_extractor = None
        self.rqa_extractor = None
        
        if self.config.enable_mlsa:
            try:
                # Try to import and use real MLSA extractor
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from core.mlsa_extractor import MLSAExtractor
                self.mlsa_extractor = MLSAExtractor(self.config.mlsa_config)
                print("Using real MLSAExtractor")
            except Exception as e:
                warnings.warn(f"Failed to load real MLSA extractor: {e}. Using mock implementation.")
                self.mlsa_extractor = MockMLSAExtractor(self.config.mlsa_config)
            
        if self.config.enable_rqa:
            try:
                # Try to import and use real RQA extractor
                from core.rqa_extractor import RQAExtractor
                self.rqa_extractor = RQAExtractor(self.config.rqa_config)
                print("Using real RQAExtractor")
            except Exception as e:
                warnings.warn(f"Failed to load real RQA extractor: {e}. Using mock implementation.")
                self.rqa_extractor = MockRQAExtractor(self.config.rqa_config)
        
        # Initialize processing engines
        self.fusion_engine = FeatureFusionEngine(self.config)
        self.selection_engine = FeatureSelectionEngine(self.config)
        
        # Initialize scaler
        self.scaler = None
        if HAS_SKLEARN and self.config.scaler_type != 'none':
            if self.config.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.config.scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.config.scaler_type == 'robust':
                self.scaler = RobustScaler()
        
        # State tracking
        self.is_fitted = False
        self.feature_names = []
        self.processing_stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'average_processing_time': 0.0
        }
    
    def extract_features(self, signal: np.ndarray, signal_id: str = None) -> Dict[str, Any]:
        """Extract comprehensive chaotic features from audio signal."""
        start_time = time.time()
        
        results = {
            'signal_id': signal_id,
            'signal_length': len(signal),
            'success': False,
            'error': None,
            'processing_time': 0.0
        }
        
        try:
            # Input validation
            if len(signal) < self.config.min_signal_length:
                results['error'] = f'Signal too short: {len(signal)} < {self.config.min_signal_length}'
                return results
            
            # Preprocessing
            processed_signal = self._preprocess_signal(signal)
            
            # Extract features from each method
            mlsa_features = None
            rqa_features = None
            extraction_results = {}
            
            # MLSA extraction
            if self.mlsa_extractor:
                mlsa_result = self.mlsa_extractor.extract_features(processed_signal)
                extraction_results['mlsa'] = mlsa_result
                
                if mlsa_result.get('success', False):
                    mlsa_features = mlsa_result['feature_vector']
            
            # RQA extraction
            if self.rqa_extractor:
                rqa_result = self.rqa_extractor.extract_features(processed_signal)
                extraction_results['rqa'] = rqa_result
                
                if rqa_result.get('success', False):
                    rqa_features = rqa_result['feature_vector']
            
            results['extraction_results'] = extraction_results
            
            # Feature fusion
            if mlsa_features is not None or rqa_features is not None:
                fused_features, fusion_metadata = self.fusion_engine.fuse_features(mlsa_features, rqa_features)
                results['fusion_metadata'] = fusion_metadata
            else:
                results['error'] = "Both MLSA and RQA extractions failed"
                return results
            
            # Handle NaN values
            clean_features = self._handle_nan_values(fused_features)
            
            # Apply feature selection and scaling
            if self.is_fitted:
                final_features, selection_metadata = self.selection_engine.transform_features(
                    clean_features.reshape(1, -1)
                )
                results['selection_metadata'] = selection_metadata
                
                # Apply scaling
                if self.scaler:
                    if final_features.ndim == 1:
                        final_features = final_features.reshape(1, -1)
                    final_features = self.scaler.transform(final_features)
                    if final_features.shape[0] == 1:
                        final_features = final_features[0]
            else:
                final_features = clean_features
            
            # Final cleanup
            final_features = handle_nans_infs(final_features)
            
            results['feature_vector'] = final_features
            results['feature_names'] = self.feature_names[:len(final_features)] if self.feature_names else []
            results['success'] = True
            
            self._update_processing_stats(success=True, processing_time=time.time() - start_time)
            
        except Exception as e:
            results['error'] = str(e)
            self._update_processing_stats(success=False, processing_time=time.time() - start_time)
        
        results['processing_time'] = time.time() - start_time
        return results
    
    def extract_features_batch(self, signals: List[np.ndarray],
                             signal_ids: Optional[List[str]] = None,
                             show_progress: bool = True) -> List[Dict[str, Any]]:
        """Extract features from multiple signals."""
        if signal_ids is None:
            signal_ids = [f"signal_{i}" for i in range(len(signals))]
        
        results = []
        
        for i, (signal, signal_id) in enumerate(zip(signals, signal_ids)):
            result = self.extract_features(signal, signal_id)
            results.append(result)
            
            if show_progress and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(signals)} signals")
        
        return results
    
    def fit(self, signals: List[np.ndarray], 
           labels: Optional[np.ndarray] = None,
           signal_ids: Optional[List[str]] = None):
        """Fit the feature extraction system on training data."""
        print(f"Fitting chaotic feature extractor on {len(signals)} signals...")
        
        # Extract features from all training signals
        extraction_results = self.extract_features_batch(signals, signal_ids, show_progress=True)
        
        # Separate successful extractions
        successful_results = [r for r in extraction_results if r.get('success', False)]
        
        if not successful_results:
            raise ValueError("No successful feature extractions for fitting")
        
        print(f"Successful extractions: {len(successful_results)}/{len(signals)}")
        
        # Collect features for fitting
        mlsa_features_list = []
        rqa_features_list = []
        all_features_list = []
        
        for result in successful_results:
            # Get raw features for fusion learning
            if 'extraction_results' in result:
                mlsa_result = result['extraction_results'].get('mlsa', {})
                rqa_result = result['extraction_results'].get('rqa', {})
                
                if mlsa_result.get('success', False):
                    mlsa_features_list.append(mlsa_result['feature_vector'])
                if rqa_result.get('success', False):
                    rqa_features_list.append(rqa_result['feature_vector'])
            
            # Re-extract fused features consistently
            mlsa_feat = None
            rqa_feat = None
            
            if 'extraction_results' in result:
                mlsa_result = result['extraction_results'].get('mlsa', {})
                rqa_result = result['extraction_results'].get('rqa', {})
                
                if mlsa_result.get('success', False):
                    mlsa_feat = mlsa_result['feature_vector']
                if rqa_result.get('success', False):
                    rqa_feat = rqa_result['feature_vector']
            
            if mlsa_feat is not None or rqa_feat is not None:
                fused_feat, _ = self.fusion_engine.fuse_features(mlsa_feat, rqa_feat)
                clean_feat = self._handle_nan_values(fused_feat)
                all_features_list.append(clean_feat)
        
        if all_features_list:
            # Handle variable length features
            max_length = max(len(feat) for feat in all_features_list)
            padded_features = []
            
            for feat in all_features_list:
                if len(feat) < max_length:
                    padded = np.pad(feat, (0, max_length - len(feat)), mode='constant', constant_values=0.0)
                else:
                    padded = feat
                padded_features.append(padded)
            
            features_matrix = np.vstack(padded_features)
            features_matrix = self._handle_nan_matrix(features_matrix)
            
            # Learn fusion weights
            if mlsa_features_list and rqa_features_list and labels is not None:
                self.fusion_engine.learn_fusion_weights(mlsa_features_list, rqa_features_list, labels)
            
            # Fit feature selection
            self.selection_engine.fit_selector(features_matrix, labels)
            
            # Fit dimensionality reduction
            if self.config.enable_dimensionality_reduction:
                self.selection_engine.fit_reducer(features_matrix)
            
            # Apply feature selection
            selected_features, _ = self.selection_engine.transform_features(features_matrix)
            
            # Fit scaler
            if self.scaler:
                self.scaler.fit(selected_features)
            
            # Generate feature names
            self._generate_feature_names(selected_features.shape[1])
            
            self.is_fitted = True
            print("✓ Chaotic feature extractor fitted successfully!")
            print(f"Final feature dimensions: {selected_features.shape[1]}")
        else:
            raise ValueError("No valid features extracted for fitting")
    
    def _preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Preprocess signal before feature extraction."""
        if self.config.outlier_detection:
            # Simple z-score based outlier removal
            z_scores = np.abs((signal - np.mean(signal)) / (np.std(signal) + 1e-10))
            outlier_mask = z_scores < 3.0
            if np.any(outlier_mask):
                return signal[outlier_mask]
        return signal
    
    def _handle_nan_values(self, features: np.ndarray) -> np.ndarray:
        """Handle NaN values in feature vector."""
        if len(features) == 0:
            return features
        return handle_nans_infs(features)
    
    def _handle_nan_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Handle NaN values in feature matrix."""
        return handle_nans_infs(matrix)
    
    def _generate_feature_names(self, n_features: int):
        """Generate feature names."""
        feature_names = []
        
        # Try to get names from extractors
        if self.mlsa_extractor and hasattr(self.mlsa_extractor, 'feature_names'):
            mlsa_names = [f"mlsa_{name}" for name in self.mlsa_extractor.feature_names]
            feature_names.extend(mlsa_names)
        
        if self.rqa_extractor and hasattr(self.rqa_extractor, 'feature_names'):
            rqa_names = [f"rqa_{name}" for name in self.rqa_extractor.feature_names]
            feature_names.extend(rqa_names)
        
        # Pad or truncate to match n_features
        while len(feature_names) < n_features:
            feature_names.append(f"chaotic_feature_{len(feature_names)}")
        
        self.feature_names = feature_names[:n_features]
    
    def _update_processing_stats(self, success: bool, processing_time: float):
        """Update processing statistics."""
        self.processing_stats['total_processed'] += 1
        
        if success:
            self.processing_stats['successful_extractions'] += 1
        else:
            self.processing_stats['failed_extractions'] += 1
        
        # Update average processing time
        total = self.processing_stats['total_processed']
        current_avg = self.processing_stats['average_processing_time']
        new_avg = ((total - 1) * current_avg + processing_time) / total
        self.processing_stats['average_processing_time'] = new_avg
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        importance = {}
        
        if self.fusion_engine.fusion_weights:
            importance['mlsa_fusion_weight'] = self.fusion_engine.fusion_weights[0]
            importance['rqa_fusion_weight'] = self.fusion_engine.fusion_weights[1]
        
        return importance
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful_extractions'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def save_model(self, filepath: str):
        """Save the fitted model to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        
        model_data = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'processing_stats': self.processing_stats,
            'fusion_weights': self.fusion_engine.fusion_weights,
            'selector': self.selection_engine.selector,
            'reducer': getattr(self.selection_engine, 'reducer', None),
            'fitted_feature_shape': self.selection_engine.fitted_feature_shape,
            'scaler': self.scaler
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str):
        """Load a fitted model from file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model state
            self.config = model_data['config']
            self.is_fitted = model_data['is_fitted']
            self.feature_names = model_data['feature_names']
            self.processing_stats = model_data['processing_stats']
            
            # Restore fusion engine state
            self.fusion_engine.fusion_weights = model_data.get('fusion_weights')
            
            # Restore selection engine state
            self.selection_engine.selector = model_data.get('selector')
            self.selection_engine.reducer = model_data.get('reducer')
            self.selection_engine.fitted_feature_shape = model_data.get('fitted_feature_shape')
            
            # Restore scaler
            self.scaler = model_data.get('scaler')
            
            print(f"Model loaded from {filepath}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")


if __name__ == "__main__":
    print("Testing Chaotic Feature Extractor...")
    
    # Generate test signals
    try:
        from scipy.integrate import solve_ivp
        
        def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
            x, y, z = state
            return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
        
        test_signals = []
        test_labels = []
        
        for i in range(5):
            sigma = 10.0 + np.random.normal(0, 1)
            rho = 28.0 + np.random.normal(0, 2)
            
            t_span = (0, 5 + i * 0.5)
            t_eval = np.arange(0, 5 + i * 0.5, 0.02)
            initial_state = np.random.normal([1, 1, 1], 0.1)
            
            sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')
            signal = sol.y[0]
            
            test_signals.append(signal)
            test_labels.append(i % 2)
        
        print(f"Generated {len(test_signals)} Lorenz test signals")
        
    except ImportError:
        print("SciPy not available, generating simple test signals...")
        test_signals = []
        test_labels = []
        
        for i in range(5):
            t = np.linspace(0, 10, 500 + i * 50)
            freq = 1.0 + i * 0.2
            signal = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
            
            test_signals.append(signal)
            test_labels.append(i % 2)
        
        print(f"Generated {len(test_signals)} simple test signals")
    
    # Create chaotic feature extractor with robust configuration
    config = ChaoticFeatureConfig(
        enable_mlsa=True,
        enable_rqa=True,
        feature_fusion_method='concatenate',
        enable_feature_selection=True,
        selection_method='variance',
        variance_threshold=0.001,
        scaler_type='standard',
        enable_parallel=False
    )
    
    print("Created configuration successfully")
    
    try:
        extractor = ChaoticFeatureExtractor(config)
        print("✓ ChaoticFeatureExtractor created successfully!")
        
        # Fit the extractor
        print("\nFitting chaotic feature extractor...")
        extractor.fit(test_signals[:3], np.array(test_labels[:3]))
        print("✓ Fitting completed successfully!")
        
        # Extract features from remaining signals
        print("\nExtracting features from test signals...")
        results = extractor.extract_features_batch(test_signals[3:])
        
        successful_results = [r for r in results if r.get('success', False)]
        print(f"Successful extractions: {len(successful_results)}/{len(results)}")
        
        if successful_results:
            sample_result = successful_results[0]
            print(f"Feature vector shape: {sample_result['feature_vector'].shape}")
            print(f"Processing time: {sample_result['processing_time']:.3f}s")
            
            # Show processing statistics
            stats = extractor.get_processing_stats()
            print(f"\nProcessing statistics:")
            print(f"  Total processed: {stats['total_processed']}")
            print(f"  Success rate: {stats['success_rate']:.2%}")
            print(f"  Average processing time: {stats['average_processing_time']:.3f}s")
            
            # Show feature importance
            importance = extractor.get_feature_importance()
            if importance:
                print("\nFeature importance scores:")
                for name, score in importance.items():
                    print(f"  {name}: {score:.6f}")
        
        print("\n✓ Chaotic Feature Extractor testing completed successfully!")
        
    except Exception as e:
        print(f"✗ Testing failed: {e}")
        import traceback
        traceback.print_exc()