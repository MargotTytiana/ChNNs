"""
Feature Adapter System for Chaotic Speaker Recognition Project
Provides comprehensive feature adaptation capabilities including dimension matching,
format conversion, and model-specific preprocessing for seamless integration
between traditional and chaotic features with different model architectures.
"""

import warnings
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
import json
from pathlib import Path

import numpy as np

# Machine learning frameworks (optional imports)
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None

# Scientific computing
try:
    from scipy import interpolate
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    interpolate = None
    uniform_filter1d = None

# Machine learning utilities
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.utils import resample
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    StandardScaler = None
    MinMaxScaler = None
    RobustScaler = None
    PCA = None


class BaseFeatureAdapter(ABC):
    """Abstract base class for feature adapters"""
    
    def __init__(self, name: str, input_shape: Optional[Tuple[int, ...]] = None,
                 output_shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize base feature adapter
        
        Args:
            name: Name of the adapter
            input_shape: Expected input shape (None for dynamic)
            output_shape: Target output shape (None for dynamic)
        """
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.is_fitted = False
        self.adaptation_stats = {
            'total_adaptations': 0,
            'input_shapes_seen': set(),
            'output_shapes_produced': set()
        }
    
    @abstractmethod
    def adapt(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adapt features to target format
        
        Args:
            features: Input features to adapt
            **kwargs: Additional adaptation parameters
            
        Returns:
            Adapted features
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get adapter configuration"""
        pass
    
    def fit(self, features: Union[np.ndarray, List[np.ndarray]], **kwargs):
        """
        Fit adapter parameters on training data
        
        Args:
            features: Training features
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        self.is_fitted = True
        return self
    
    def validate_input(self, features: np.ndarray) -> np.ndarray:
        """Validate and preprocess input features"""
        if features is None:
            raise ValueError("Input features cannot be None")
        
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Record input shape
        self.adaptation_stats['input_shapes_seen'].add(features.shape)
        
        return features
    
    def update_stats(self, output_features: np.ndarray):
        """Update adaptation statistics"""
        self.adaptation_stats['total_adaptations'] += 1
        self.adaptation_stats['output_shapes_produced'].add(output_features.shape)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter usage statistics"""
        stats = self.adaptation_stats.copy()
        stats['input_shapes_seen'] = list(stats['input_shapes_seen'])
        stats['output_shapes_produced'] = list(stats['output_shapes_produced'])
        return stats


class DimensionAdapter(BaseFeatureAdapter):
    """Adapt feature dimensions through padding, truncation, or interpolation"""
    
    def __init__(self,
                 target_dim: int,
                 method: str = 'pad',
                 pad_value: float = 0.0,
                 pad_mode: str = 'constant',
                 interpolation_kind: str = 'linear',
                 axis: int = -1):
        """
        Initialize dimension adapter
        
        Args:
            target_dim: Target dimension size
            method: Adaptation method ('pad', 'truncate', 'interpolate', 'repeat')
            pad_value: Value for padding (if method is 'pad')
            pad_mode: Padding mode ('constant', 'edge', 'wrap', 'reflect')
            interpolation_kind: Interpolation method for 'interpolate' mode
            axis: Axis along which to adapt dimensions
        """
        super().__init__("dimension_adapter", output_shape=(target_dim,))
        
        self.target_dim = target_dim
        self.method = method.lower()
        self.pad_value = pad_value
        self.pad_mode = pad_mode
        self.interpolation_kind = interpolation_kind
        self.axis = axis
        
        # Validate method
        valid_methods = ['pad', 'truncate', 'interpolate', 'repeat', 'pool']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")
        
        if self.method == 'interpolate' and not HAS_SCIPY:
            warnings.warn("SciPy not available. Falling back to linear interpolation.")
    
    def adapt(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """Adapt feature dimensions"""
        features = self.validate_input(features)
        
        # Get current dimension along specified axis
        current_dim = features.shape[self.axis]
        
        if current_dim == self.target_dim:
            # No adaptation needed
            adapted_features = features
        elif self.method == 'pad':
            adapted_features = self._pad_features(features, current_dim)
        elif self.method == 'truncate':
            adapted_features = self._truncate_features(features, current_dim)
        elif self.method == 'interpolate':
            adapted_features = self._interpolate_features(features, current_dim)
        elif self.method == 'repeat':
            adapted_features = self._repeat_features(features, current_dim)
        elif self.method == 'pool':
            adapted_features = self._pool_features(features, current_dim)
        else:
            raise ValueError(f"Unknown adaptation method: {self.method}")
        
        self.update_stats(adapted_features)
        return adapted_features
    
    def _pad_features(self, features: np.ndarray, current_dim: int) -> np.ndarray:
        """Pad features to target dimension"""
        if current_dim >= self.target_dim:
            return features
        
        pad_width = [(0, 0)] * features.ndim
        pad_amount = self.target_dim - current_dim
        
        # Distribute padding
        if self.pad_mode == 'constant':
            pad_before = pad_amount // 2
            pad_after = pad_amount - pad_before
            pad_width[self.axis] = (pad_before, pad_after)
            
            return np.pad(features, pad_width, mode='constant', constant_values=self.pad_value)
        else:
            pad_width[self.axis] = (0, pad_amount)
            return np.pad(features, pad_width, mode=self.pad_mode)
    
    def _truncate_features(self, features: np.ndarray, current_dim: int) -> np.ndarray:
        """Truncate features to target dimension"""
        if current_dim <= self.target_dim:
            return features
        
        # Create slice objects
        slices = [slice(None)] * features.ndim
        
        # Center truncation
        start_idx = (current_dim - self.target_dim) // 2
        end_idx = start_idx + self.target_dim
        slices[self.axis] = slice(start_idx, end_idx)
        
        return features[tuple(slices)]
    
    def _interpolate_features(self, features: np.ndarray, current_dim: int) -> np.ndarray:
        """Interpolate features to target dimension"""
        if current_dim == self.target_dim:
            return features
        
        # Handle different array dimensions
        if features.ndim == 1:
            return self._interpolate_1d(features)
        elif features.ndim == 2:
            return self._interpolate_2d(features)
        else:
            # For higher dimensions, work along the specified axis
            return self._interpolate_nd(features)
    
    def _interpolate_1d(self, features: np.ndarray) -> np.ndarray:
        """Interpolate 1D features"""
        current_dim = len(features)
        old_indices = np.linspace(0, current_dim - 1, current_dim)
        new_indices = np.linspace(0, current_dim - 1, self.target_dim)
        
        if HAS_SCIPY:
            f = interpolate.interp1d(old_indices, features, kind=self.interpolation_kind,
                                   bounds_error=False, fill_value='extrapolate')
            return f(new_indices)
        else:
            # Simple linear interpolation fallback
            return np.interp(new_indices, old_indices, features)
    
    def _interpolate_2d(self, features: np.ndarray) -> np.ndarray:
        """Interpolate 2D features"""
        if self.axis == 0:
            # Interpolate along rows
            return np.array([self._interpolate_1d(row) for row in features])
        else:
            # Interpolate along columns
            return np.array([self._interpolate_1d(features[:, col]) 
                           for col in range(features.shape[1])]).T
    
    def _interpolate_nd(self, features: np.ndarray) -> np.ndarray:
        """Interpolate n-dimensional features"""
        # Move target axis to the last position
        features_moved = np.moveaxis(features, self.axis, -1)
        original_shape = features_moved.shape
        
        # Reshape to 2D for processing
        reshaped = features_moved.reshape(-1, original_shape[-1])
        
        # Interpolate each row
        interpolated_rows = []
        for row in reshaped:
            interpolated_row = self._interpolate_1d(row)
            interpolated_rows.append(interpolated_row)
        
        # Reshape back and move axis back
        new_shape = original_shape[:-1] + (self.target_dim,)
        interpolated = np.array(interpolated_rows).reshape(new_shape)
        
        return np.moveaxis(interpolated, -1, self.axis)
    
    def _repeat_features(self, features: np.ndarray, current_dim: int) -> np.ndarray:
        """Repeat features to reach target dimension"""
        if current_dim >= self.target_dim:
            return self._truncate_features(features, current_dim)
        
        # Calculate repetition factor
        repeat_factor = self.target_dim // current_dim
        remainder = self.target_dim % current_dim
        
        # Repeat features
        repeated = np.repeat(features, repeat_factor, axis=self.axis)
        
        # Handle remainder
        if remainder > 0:
            slices = [slice(None)] * features.ndim
            slices[self.axis] = slice(0, remainder)
            extra_part = features[tuple(slices)]
            repeated = np.concatenate([repeated, extra_part], axis=self.axis)
        
        return repeated
    
    def _pool_features(self, features: np.ndarray, current_dim: int) -> np.ndarray:
        """Pool features to target dimension"""
        if current_dim <= self.target_dim:
            return self._pad_features(features, current_dim)
        
        # Calculate pooling parameters
        pool_size = current_dim // self.target_dim
        
        # Simple average pooling
        if features.ndim == 1:
            pooled = []
            for i in range(self.target_dim):
                start_idx = i * pool_size
                end_idx = min(start_idx + pool_size, current_dim)
                pooled.append(np.mean(features[start_idx:end_idx]))
            return np.array(pooled)
        else:
            # For multi-dimensional arrays, pool along the specified axis
            return self._pool_nd(features, pool_size)
    
    def _pool_nd(self, features: np.ndarray, pool_size: int) -> np.ndarray:
        """Pool n-dimensional features"""
        # Move target axis to the last position
        features_moved = np.moveaxis(features, self.axis, -1)
        original_shape = features_moved.shape
        current_dim = original_shape[-1]
        
        # Reshape for pooling
        reshaped = features_moved.reshape(-1, current_dim)
        
        # Pool each row
        pooled_rows = []
        for row in reshaped:
            pooled_row = []
            for i in range(self.target_dim):
                start_idx = i * pool_size
                end_idx = min(start_idx + pool_size, current_dim)
                pooled_row.append(np.mean(row[start_idx:end_idx]))
            pooled_rows.append(pooled_row)
        
        # Reshape back and move axis back
        new_shape = original_shape[:-1] + (self.target_dim,)
        pooled = np.array(pooled_rows).reshape(new_shape)
        
        return np.moveaxis(pooled, -1, self.axis)
    
    def get_config(self) -> Dict[str, Any]:
        """Get adapter configuration"""
        return {
            'name': self.name,
            'target_dim': self.target_dim,
            'method': self.method,
            'pad_value': self.pad_value,
            'pad_mode': self.pad_mode,
            'interpolation_kind': self.interpolation_kind,
            'axis': self.axis
        }


class TemporalAdapter(BaseFeatureAdapter):
    """Adapt temporal features for different model requirements"""
    
    def __init__(self,
                 target_length: Optional[int] = None,
                 aggregation_method: str = 'mean',
                 window_size: Optional[int] = None,
                 hop_size: Optional[int] = None,
                 padding_mode: str = 'constant'):
        """
        Initialize temporal adapter
        
        Args:
            target_length: Target temporal length (None to keep original)
            aggregation_method: Method for temporal aggregation ('mean', 'max', 'std', 'all')
            window_size: Window size for windowed operations
            hop_size: Hop size for windowed operations
            padding_mode: Padding mode for length adjustment
        """
        super().__init__("temporal_adapter")
        
        self.target_length = target_length
        self.aggregation_method = aggregation_method.lower()
        self.window_size = window_size
        self.hop_size = hop_size or (window_size // 2 if window_size else None)
        self.padding_mode = padding_mode
        
        # Validate aggregation method
        valid_methods = ['mean', 'max', 'min', 'std', 'median', 'all', 'flatten', 'none']
        if self.aggregation_method not in valid_methods:
            raise ValueError(f"Invalid aggregation method '{aggregation_method}'. "
                           f"Must be one of {valid_methods}")
    
    def adapt(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """Adapt temporal features"""
        features = self.validate_input(features)
        
        # Assume temporal dimension is the last one for 2D features
        if features.ndim < 2:
            # No temporal dimension to adapt
            return features
        
        # Apply windowing if specified
        if self.window_size and self.window_size > 1:
            features = self._apply_windowing(features)
        
        # Apply length adaptation if specified
        if self.target_length:
            features = self._adapt_length(features)
        
        # Apply aggregation
        if self.aggregation_method != 'none':
            features = self._aggregate_temporal(features)
        
        self.update_stats(features)
        return features
    
    def _apply_windowing(self, features: np.ndarray) -> np.ndarray:
        """Apply windowing to temporal features"""
        if features.ndim != 2:
            return features
        
        n_features, n_time = features.shape
        
        # Calculate number of windows
        if self.hop_size:
            n_windows = (n_time - self.window_size) // self.hop_size + 1
        else:
            n_windows = n_time // self.window_size
        
        if n_windows <= 0:
            return features
        
        # Create windowed features
        windowed_features = []
        
        for i in range(n_windows):
            if self.hop_size:
                start_idx = i * self.hop_size
            else:
                start_idx = i * self.window_size
            
            end_idx = start_idx + self.window_size
            if end_idx <= n_time:
                window = features[:, start_idx:end_idx]
                windowed_features.append(window)
        
        if windowed_features:
            # Stack windows along new axis
            return np.stack(windowed_features, axis=-1)  # Shape: (n_features, window_size, n_windows)
        else:
            return features
    
    def _adapt_length(self, features: np.ndarray) -> np.ndarray:
        """Adapt temporal length"""
        if features.ndim < 2:
            return features
        
        current_length = features.shape[-1]  # Assume last dimension is temporal
        
        if current_length == self.target_length:
            return features
        elif current_length < self.target_length:
            # Pad
            pad_width = [(0, 0)] * features.ndim
            pad_amount = self.target_length - current_length
            pad_width[-1] = (0, pad_amount)
            
            return np.pad(features, pad_width, mode=self.padding_mode)
        else:
            # Truncate or interpolate
            slices = [slice(None)] * features.ndim
            slices[-1] = slice(0, self.target_length)
            
            return features[tuple(slices)]
    
    def _aggregate_temporal(self, features: np.ndarray) -> np.ndarray:
        """Aggregate temporal features"""
        if features.ndim < 2:
            return features
        
        # Apply aggregation along temporal axis (assumed to be last)
        temporal_axis = -1
        
        if self.aggregation_method == 'mean':
            return np.mean(features, axis=temporal_axis)
        elif self.aggregation_method == 'max':
            return np.max(features, axis=temporal_axis)
        elif self.aggregation_method == 'min':
            return np.min(features, axis=temporal_axis)
        elif self.aggregation_method == 'std':
            return np.std(features, axis=temporal_axis)
        elif self.aggregation_method == 'median':
            return np.median(features, axis=temporal_axis)
        elif self.aggregation_method == 'all':
            # Combine multiple statistics
            stats = [
                np.mean(features, axis=temporal_axis),
                np.std(features, axis=temporal_axis),
                np.max(features, axis=temporal_axis),
                np.min(features, axis=temporal_axis)
            ]
            return np.concatenate(stats, axis=-1)
        elif self.aggregation_method == 'flatten':
            # Flatten temporal dimension
            return features.reshape(features.shape[:-2] + (-1,))
        else:
            return features
    
    def get_config(self) -> Dict[str, Any]:
        """Get adapter configuration"""
        return {
            'name': self.name,
            'target_length': self.target_length,
            'aggregation_method': self.aggregation_method,
            'window_size': self.window_size,
            'hop_size': self.hop_size,
            'padding_mode': self.padding_mode
        }


class NormalizationAdapter(BaseFeatureAdapter):
    """Normalize features for model compatibility"""
    
    def __init__(self,
                 method: str = 'standard',
                 feature_range: Tuple[float, float] = (0, 1),
                 clip_values: Optional[Tuple[float, float]] = None,
                 handle_nan: str = 'warn'):
        """
        Initialize normalization adapter
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust', 'unit_norm', 'none')
            feature_range: Target range for minmax scaling
            clip_values: Optional clipping range (min, max)
            handle_nan: How to handle NaN values ('warn', 'replace', 'raise')
        """
        super().__init__("normalization_adapter")
        
        self.method = method.lower()
        self.feature_range = feature_range
        self.clip_values = clip_values
        self.handle_nan = handle_nan
        
        # Initialize scaler
        self.scaler = None
        
        # Validate method
        valid_methods = ['standard', 'minmax', 'robust', 'unit_norm', 'none']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid normalization method '{method}'. "
                           f"Must be one of {valid_methods}")
        
        if self.method in ['standard', 'minmax', 'robust'] and not HAS_SKLEARN:
            warnings.warn("scikit-learn not available. Using manual implementation.")
    
    def fit(self, features: Union[np.ndarray, List[np.ndarray]], **kwargs):
        """Fit normalization parameters"""
        if self.method == 'none':
            self.is_fitted = True
            return self
        
        # Convert to numpy array if needed
        if isinstance(features, list):
            features = np.array(features)
        
        # Handle different shapes
        if features.ndim == 1:
            # Reshape 1D array to 2D for sklearn compatibility
            features = features.reshape(1, -1)
        elif features.ndim > 2:
            # Flatten to 2D for fitting
            original_shape = features.shape
            features = features.reshape(-1, original_shape[-1])
        
        # Handle NaN values
        features = self._handle_nan_values(features)
        
        # Fit scaler
        if self.method == 'standard' and HAS_SKLEARN:
            self.scaler = StandardScaler()
            self.scaler.fit(features)
        elif self.method == 'minmax' and HAS_SKLEARN:
            self.scaler = MinMaxScaler(feature_range=self.feature_range)
            self.scaler.fit(features)
        elif self.method == 'robust' and HAS_SKLEARN:
            self.scaler = RobustScaler()
            self.scaler.fit(features)
        else:
            # Manual computation of parameters
            self._fit_manual(features)
        
        self.is_fitted = True
        return self
    
    def _fit_manual(self, features: np.ndarray):
        """Manually fit normalization parameters"""
        if self.method == 'standard':
            self.mean_ = np.mean(features, axis=0)
            self.std_ = np.std(features, axis=0)
            # Avoid division by zero
            self.std_[self.std_ == 0] = 1.0
        elif self.method == 'minmax':
            self.min_ = np.min(features, axis=0)
            self.max_ = np.max(features, axis=0)
            # Avoid division by zero
            self.scale_ = self.max_ - self.min_
            self.scale_[self.scale_ == 0] = 1.0
    
    def adapt(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """Apply normalization to features"""
        features = self.validate_input(features)
        
        if self.method == 'none':
            normalized = features
        elif not self.is_fitted:
            warnings.warn("Adapter not fitted. Fitting on current data.")
            self.fit(features)
            normalized = self._apply_normalization(features)
        else:
            normalized = self._apply_normalization(features)
        
        # Apply clipping if specified
        if self.clip_values:
            normalized = np.clip(normalized, self.clip_values[0], self.clip_values[1])
        
        self.update_stats(normalized)
        return normalized
    
    def _apply_normalization(self, features: np.ndarray) -> np.ndarray:
        """Apply the normalization transformation"""
        # Handle NaN values
        features = self._handle_nan_values(features)
        
        original_shape = features.shape
        is_1d = features.ndim == 1
        
        # Reshape for processing if needed - sklearn always needs 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        elif features.ndim > 2:
            features = features.reshape(-1, original_shape[-1])
        
        if self.method == 'standard':
            if self.scaler:
                normalized = self.scaler.transform(features)
            else:
                normalized = (features - self.mean_) / self.std_
        elif self.method == 'minmax':
            if self.scaler:
                normalized = self.scaler.transform(features)
            else:
                normalized = (features - self.min_) / self.scale_
                normalized = normalized * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        elif self.method == 'robust':
            if self.scaler:
                normalized = self.scaler.transform(features)
            else:
                # Manual robust scaling
                median = np.median(features, axis=0)
                mad = np.median(np.abs(features - median), axis=0)
                mad[mad == 0] = 1.0
                normalized = (features - median) / mad
        elif self.method == 'unit_norm':
            # L2 normalization along feature axis
            norms = np.linalg.norm(features, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = features / norms
        else:
            normalized = features
        
        # Reshape back to original shape
        if is_1d:
            normalized = normalized.reshape(-1)
        elif len(original_shape) > 2:
            normalized = normalized.reshape(original_shape)
        
        return normalized
    
    def _handle_nan_values(self, features: np.ndarray) -> np.ndarray:
        """Handle NaN values in features"""
        if not np.any(np.isnan(features)):
            return features
        
        if self.handle_nan == 'raise':
            raise ValueError("NaN values found in features")
        elif self.handle_nan == 'warn':
            warnings.warn("NaN values found in features. Replacing with zeros.")
            return np.nan_to_num(features, nan=0.0)
        elif self.handle_nan == 'replace':
            return np.nan_to_num(features, nan=0.0)
        else:
            return features
    
    def get_config(self) -> Dict[str, Any]:
        """Get adapter configuration"""
        config = {
            'name': self.name,
            'method': self.method,
            'feature_range': self.feature_range,
            'clip_values': self.clip_values,
            'handle_nan': self.handle_nan,
            'is_fitted': self.is_fitted
        }
        
        # Add fitted parameters if available
        if self.is_fitted and hasattr(self, 'mean_'):
            config['mean'] = self.mean_.tolist() if hasattr(self.mean_, 'tolist') else None
            config['std'] = self.std_.tolist() if hasattr(self.std_, 'tolist') else None
        
        return config


class ModelSpecificAdapter(BaseFeatureAdapter):
    """Adapt features for specific model architectures"""
    
    def __init__(self,
                 model_type: str,
                 framework: str = 'numpy',
                 batch_dimension: bool = True,
                 channel_dimension: bool = False,
                 sequence_first: bool = True):
        """
        Initialize model-specific adapter
        
        Args:
            model_type: Type of model ('mlp', 'cnn', 'rnn', 'chaotic_network')
            framework: Target framework ('numpy', 'torch', 'tensorflow')
            batch_dimension: Whether to add/ensure batch dimension
            channel_dimension: Whether to add/ensure channel dimension
            sequence_first: For RNN models, whether sequence comes first
        """
        super().__init__("model_specific_adapter")
        
        self.model_type = model_type.lower()
        self.framework = framework.lower()
        self.batch_dimension = batch_dimension
        self.channel_dimension = channel_dimension
        self.sequence_first = sequence_first
        
        # Validate parameters
        valid_models = ['mlp', 'cnn', 'rnn', 'lstm', 'gru', 'chaotic_network']
        if self.model_type not in valid_models:
            raise ValueError(f"Invalid model type '{model_type}'. Must be one of {valid_models}")
        
        valid_frameworks = ['numpy', 'torch', 'tensorflow']
        if self.framework not in valid_frameworks:
            raise ValueError(f"Invalid framework '{framework}'. Must be one of {valid_frameworks}")
    
    def adapt(self, features: np.ndarray, **kwargs) -> Union[np.ndarray, 'torch.Tensor', 'tf.Tensor']:
        """Adapt features for specific model architecture"""
        features = self.validate_input(features)
        
        # Apply model-specific transformations
        if self.model_type == 'mlp':
            adapted = self._adapt_for_mlp(features)
        elif self.model_type == 'cnn':
            adapted = self._adapt_for_cnn(features)
        elif self.model_type in ['rnn', 'lstm', 'gru']:
            adapted = self._adapt_for_rnn(features)
        elif self.model_type == 'chaotic_network':
            adapted = self._adapt_for_chaotic_network(features)
        else:
            adapted = features
        
        # Convert to target framework
        adapted = self._convert_to_framework(adapted)
        
        self.update_stats(adapted if isinstance(adapted, np.ndarray) else 
                         adapted.detach().cpu().numpy() if HAS_TORCH and isinstance(adapted, torch.Tensor) else
                         adapted.numpy() if HAS_TF and hasattr(adapted, 'numpy') else adapted)
        
        return adapted
    
    def _adapt_for_mlp(self, features: np.ndarray) -> np.ndarray:
        """Adapt features for MLP (flatten to 1D per sample)"""
        if features.ndim > 2:
            # Flatten all dimensions except batch dimension
            if self.batch_dimension and features.ndim > 1:
                # Keep first dimension as batch, flatten rest
                batch_size = features.shape[0]
                adapted = features.reshape(batch_size, -1)
            else:
                # Flatten everything
                adapted = features.reshape(-1)
        else:
            adapted = features
        
        # Ensure batch dimension if required
        if self.batch_dimension and adapted.ndim == 1:
            adapted = adapted.reshape(1, -1)
        
        return adapted
    
    def _adapt_for_cnn(self, features: np.ndarray) -> np.ndarray:
        """Adapt features for CNN (add channel dimension if needed)"""
        adapted = features
        
        # CNN typically expects (batch, channels, height, width) or (batch, height, width, channels)
        if adapted.ndim == 2:
            # Add channel dimension
            if self.channel_dimension:
                adapted = np.expand_dims(adapted, axis=1)  # (batch, 1, features)
        elif adapted.ndim == 3:
            # Already has proper dimensions, might need channel dimension
            if self.channel_dimension and adapted.shape[1] != 1:
                adapted = np.expand_dims(adapted, axis=1)
        
        # Ensure batch dimension
        if self.batch_dimension and adapted.ndim == 2:
            adapted = np.expand_dims(adapted, axis=0)
        
        return adapted
    
    def _adapt_for_rnn(self, features: np.ndarray) -> np.ndarray:
        """Adapt features for RNN models"""
        adapted = features
        
        # RNN expects (batch, sequence, features) or (sequence, batch, features)
        if adapted.ndim == 1:
            # Single feature vector - treat as one timestep
            adapted = adapted.reshape(1, 1, -1) if self.batch_dimension else adapted.reshape(1, -1)
        elif adapted.ndim == 2:
            if self.batch_dimension:
                # Assume (batch, features) - add sequence dimension
                adapted = np.expand_dims(adapted, axis=1)  # (batch, 1, features)
            else:
                # Assume (sequence, features) - add batch dimension at the end
                adapted = np.expand_dims(adapted, axis=0)  # (1, sequence, features)
                if not self.sequence_first:
                    adapted = np.transpose(adapted, (1, 0, 2))  # (sequence, 1, features)
        
        # Handle sequence_first parameter
        if not self.sequence_first and adapted.ndim == 3:
            # Convert from (sequence, batch, features) to (batch, sequence, features)
            adapted = np.transpose(adapted, (1, 0, 2))
        
        return adapted
    
    def _adapt_for_chaotic_network(self, features: np.ndarray) -> np.ndarray:
        """Adapt features for chaotic network (preserve structure)"""
        adapted = features
        
        # Chaotic networks might expect specific input formats
        # This depends on the specific implementation
        
        # Ensure batch dimension if required
        if self.batch_dimension and adapted.ndim == 1:
            adapted = adapted.reshape(1, -1)
        
        return adapted
    
    def _convert_to_framework(self, features: np.ndarray) -> Union[np.ndarray, 'torch.Tensor', 'tf.Tensor']:
        """Convert features to target framework format"""
        if self.framework == 'numpy':
            return features
        elif self.framework == 'torch':
            if HAS_TORCH:
                return torch.from_numpy(features).float()
            else:
                warnings.warn("PyTorch not available. Returning NumPy array.")
                return features
        elif self.framework == 'tensorflow':
            if HAS_TF:
                return tf.constant(features, dtype=tf.float32)
            else:
                warnings.warn("TensorFlow not available. Returning NumPy array.")
                return features
        else:
            return features
    
    def get_config(self) -> Dict[str, Any]:
        """Get adapter configuration"""
        return {
            'name': self.name,
            'model_type': self.model_type,
            'framework': self.framework,
            'batch_dimension': self.batch_dimension,
            'channel_dimension': self.channel_dimension,
            'sequence_first': self.sequence_first
        }


class FeatureAdapterPipeline:
    """Pipeline combining multiple feature adapters"""
    
    def __init__(self,
                 adapters: List[BaseFeatureAdapter],
                 fit_adapters: bool = True):
        """
        Initialize adapter pipeline
        
        Args:
            adapters: List of adapters to apply in order
            fit_adapters: Whether to fit adapters on training data
        """
        self.adapters = adapters
        self.fit_adapters = fit_adapters
        self.is_fitted = False
        
        # Pipeline statistics
        self.pipeline_stats = {
            'total_adaptations': 0,
            'adapter_stats': {}
        }
    
    def fit(self, features: Union[np.ndarray, List[np.ndarray]], **kwargs):
        """Fit all adapters in the pipeline"""
        if not self.fit_adapters:
            self.is_fitted = True
            return self
        
        current_features = features
        
        for adapter in self.adapters:
            if hasattr(adapter, 'fit'):
                adapter.fit(current_features, **kwargs)
                
                # Transform for next adapter
                if isinstance(current_features, list):
                    current_features = [adapter.adapt(f) for f in current_features]
                else:
                    current_features = adapter.adapt(current_features)
        
        self.is_fitted = True
        return self
    
    def adapt(self, features: np.ndarray, **kwargs) -> Any:
        """Apply all adapters in sequence"""
        if self.fit_adapters and not self.is_fitted:
            warnings.warn("Pipeline not fitted. Fitting on current data.")
            self.fit(features, **kwargs)
        
        adapted_features = features
        
        # Apply each adapter in sequence
        for adapter in self.adapters:
            adapted_features = adapter.adapt(adapted_features, **kwargs)
        
        # Update pipeline statistics
        self.pipeline_stats['total_adaptations'] += 1
        for adapter in self.adapters:
            adapter_name = adapter.name
            if adapter_name not in self.pipeline_stats['adapter_stats']:
                self.pipeline_stats['adapter_stats'][adapter_name] = 0
            self.pipeline_stats['adapter_stats'][adapter_name] += 1
        
        return adapted_features
    
    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration"""
        return {
            'adapters': [adapter.get_config() for adapter in self.adapters],
            'fit_adapters': self.fit_adapters,
            'is_fitted': self.is_fitted,
            'pipeline_stats': self.pipeline_stats
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        stats = {
            'pipeline_stats': self.pipeline_stats.copy(),
            'adapter_stats': {}
        }
        
        for adapter in self.adapters:
            stats['adapter_stats'][adapter.name] = adapter.get_statistics()
        
        return stats


# Convenience functions for creating common adapter pipelines

def create_mlp_adapter_pipeline(target_dim: Optional[int] = None,
                              normalize: bool = True,
                              framework: str = 'numpy') -> FeatureAdapterPipeline:
    """Create adapter pipeline for MLP models"""
    adapters = []
    
    # Dimension adaptation
    if target_dim:
        adapters.append(DimensionAdapter(target_dim=target_dim, method='pad'))
    
    # Normalization
    if normalize:
        adapters.append(NormalizationAdapter(method='standard'))
    
    # Model-specific adaptation
    adapters.append(ModelSpecificAdapter(
        model_type='mlp',
        framework=framework,
        batch_dimension=True
    ))
    
    return FeatureAdapterPipeline(adapters)


def create_chaotic_network_adapter_pipeline(normalize: bool = True,
                                          framework: str = 'numpy') -> FeatureAdapterPipeline:
    """Create adapter pipeline for chaotic network models"""
    adapters = []
    
    # Normalization (important for chaotic systems)
    if normalize:
        adapters.append(NormalizationAdapter(method='robust'))
    
    # Model-specific adaptation
    adapters.append(ModelSpecificAdapter(
        model_type='chaotic_network',
        framework=framework,
        batch_dimension=True
    ))
    
    return FeatureAdapterPipeline(adapters)


def create_comparison_adapter(traditional_dim: int,
                            chaotic_dim: int,
                            framework: str = 'numpy') -> Dict[str, FeatureAdapterPipeline]:
    """Create adapters for comparison experiments"""
    adapters = {}
    
    # Traditional features to MLP
    adapters['traditional_mlp'] = create_mlp_adapter_pipeline(
        target_dim=None, normalize=True, framework=framework
    )
    
    # Traditional features to chaotic network (dimension adaptation needed)
    adapters['traditional_chaotic'] = FeatureAdapterPipeline([
        DimensionAdapter(target_dim=chaotic_dim, method='pad'),
        NormalizationAdapter(method='robust'),
        ModelSpecificAdapter(model_type='chaotic_network', framework=framework)
    ])
    
    # Chaotic features to MLP (dimension adaptation needed)  
    adapters['chaotic_mlp'] = FeatureAdapterPipeline([
        DimensionAdapter(target_dim=traditional_dim, method='interpolate'),
        NormalizationAdapter(method='standard'),
        ModelSpecificAdapter(model_type='mlp', framework=framework)
    ])
    
    # Chaotic features to chaotic network
    adapters['chaotic_chaotic'] = create_chaotic_network_adapter_pipeline(
        normalize=True, framework=framework
    )
    
    return adapters


# Example usage and testing
if __name__ == "__main__":
    print("Testing feature adapter system...")
    
    # Create test features
    np.random.seed(42)
    
    # Traditional features (e.g., MFCC)
    traditional_features = np.random.randn(13, 100)  # 13 MFCC coeffs, 100 time frames
    print(f"Traditional features shape: {traditional_features.shape}")
    
    # Chaotic features (different dimension)
    chaotic_features = np.random.randn(8, 50)  # 8 chaotic features, 50 time frames
    print(f"Chaotic features shape: {chaotic_features.shape}")
    
    # Test dimension adapter
    print("\nTesting DimensionAdapter...")
    dim_adapter = DimensionAdapter(target_dim=16, method='pad')
    adapted_dim = dim_adapter.adapt(traditional_features[0])  # Test on first feature
    print(f"Dimension adapted: {len(traditional_features[0])} -> {len(adapted_dim)}")
    
    # Test temporal adapter
    print("\nTesting TemporalAdapter...")
    temporal_adapter = TemporalAdapter(aggregation_method='mean')
    adapted_temporal = temporal_adapter.adapt(traditional_features)
    print(f"Temporal adapted shape: {traditional_features.shape} -> {adapted_temporal.shape}")
    
    # Test normalization adapter
    print("\nTesting NormalizationAdapter...")
    norm_adapter = NormalizationAdapter(method='standard')
    norm_adapter.fit(adapted_temporal)
    normalized = norm_adapter.adapt(adapted_temporal)
    print(f"Normalized features - mean: {np.mean(normalized):.4f}, std: {np.std(normalized):.4f}")
    
    # Test model-specific adapter
    print("\nTesting ModelSpecificAdapter...")
    model_adapter = ModelSpecificAdapter(model_type='mlp', framework='numpy')
    model_adapted = model_adapter.adapt(normalized)
    print(f"Model adapted shape: {normalized.shape} -> {model_adapted.shape}")
    
    # Test complete pipeline
    print("\nTesting complete pipeline...")
    pipeline = create_mlp_adapter_pipeline(target_dim=20, normalize=True)
    
    # Fit pipeline
    pipeline.fit(traditional_features.T)  # Transpose for batch dimension
    
    # Apply adaptation
    final_features = pipeline.adapt(traditional_features.T)
    print(f"Pipeline result shape: {traditional_features.T.shape} -> {final_features.shape}")
    
    # Test comparison adapters
    print("\nTesting comparison adapters...")
    comparison_adapters = create_comparison_adapter(
        traditional_dim=39,  # 13 MFCC + deltas + delta-deltas
        chaotic_dim=8,       # 8 chaotic features
        framework='numpy'
    )
    
    for name, adapter in comparison_adapters.items():
        print(f"Created adapter: {name}")
    
    # Display statistics
    print(f"\nPipeline statistics: {pipeline.get_statistics()}")
    
    print("\nFeature adapter system testing completed!")