"""
Numerical Stability Tools for Chaotic System Analysis.

This module provides comprehensive tools for ensuring numerical stability
in chaotic system computations, including error detection, conditioning,
precision control, and robust numerical algorithms.

Author: C-HiLAP Project
Date: 2025
"""

import numpy as np
import warnings
from typing import Tuple, Union, Optional, Dict, Any, Callable
from scipy import linalg
from scipy.stats import zscore
import time
import psutil
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class NumericalConfig:
    """Configuration for numerical stability parameters."""
    float_precision: str = 'float64'  # 'float32', 'float64', 'float128'
    tolerance: float = 1e-12
    max_condition_number: float = 1e12
    max_iterations: int = 10000
    convergence_threshold: float = 1e-8
    outlier_threshold: float = 5.0  # Z-score threshold
    memory_limit_gb: float = 8.0
    time_limit_seconds: float = 3600.0  # 1 hour
    adaptive_precision: bool = True
    numerical_warnings: bool = True


class NumericalStabilityError(Exception):
    """Custom exception for numerical stability issues."""
    pass


class PrecisionManager:
    """Manages numerical precision and data types."""
    
    def __init__(self, config: NumericalConfig):
        self.config = config
        self._original_precision = None
    
    def __enter__(self):
        """Enter precision context."""
        # Store current numpy precision settings
        self._original_precision = np.finfo(np.float64).eps
        
        # Set precision based on config
        if self.config.float_precision == 'float32':
            self.dtype = np.float32
        elif self.config.float_precision == 'float64':
            self.dtype = np.float64
        elif self.config.float_precision == 'float128':
            self.dtype = np.longdouble  # Highest available precision
        else:
            self.dtype = np.float64
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit precision context."""
        pass
    
    def ensure_precision(self, array: np.ndarray) -> np.ndarray:
        """Ensure array has correct precision."""
        return array.astype(self.dtype)
    
    def get_machine_epsilon(self) -> float:
        """Get machine epsilon for current precision."""
        return np.finfo(self.dtype).eps
    
    def is_numerically_zero(self, value: Union[float, np.ndarray], 
                           relative_tolerance: float = None) -> bool:
        """Check if value is numerically zero."""
        if relative_tolerance is None:
            relative_tolerance = self.config.tolerance
        
        return np.abs(value) < relative_tolerance


class ConditionNumberAnalyzer:
    """Analyzes matrix conditioning for numerical stability."""
    
    def __init__(self, config: NumericalConfig):
        self.config = config
    
    def condition_number(self, matrix: np.ndarray, norm: str = '2') -> float:
        """Calculate condition number of a matrix."""
        try:
            if norm == '2':
                return np.linalg.cond(matrix, p=2)
            elif norm == 'fro':
                return np.linalg.cond(matrix, p='fro')
            elif norm == 'inf':
                return np.linalg.cond(matrix, p=np.inf)
            else:
                return np.linalg.cond(matrix)
        except np.linalg.LinAlgError:
            return np.inf
    
    def is_well_conditioned(self, matrix: np.ndarray) -> bool:
        """Check if matrix is well-conditioned."""
        cond_num = self.condition_number(matrix)
        return cond_num < self.config.max_condition_number
    
    def regularize_matrix(self, matrix: np.ndarray, 
                         regularization: float = None) -> np.ndarray:
        """Add regularization to improve conditioning."""
        if regularization is None:
            regularization = self.config.tolerance
        
        # Add small diagonal perturbation
        regularized = matrix.copy()
        np.fill_diagonal(regularized, 
                        np.diag(regularized) + regularization)
        return regularized
    
    def svd_conditioning(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze conditioning using SVD."""
        try:
            U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
            
            # Filter out tiny singular values
            s_filtered = s[s > self.config.tolerance]
            
            if len(s_filtered) == 0:
                return {
                    'condition_number': np.inf,
                    'rank': 0,
                    'singular_values': s,
                    'effective_rank': 0,
                    'is_stable': False
                }
            
            condition_number = s_filtered[0] / s_filtered[-1]
            effective_rank = len(s_filtered)
            
            return {
                'condition_number': condition_number,
                'rank': len(s),
                'singular_values': s,
                'effective_rank': effective_rank,
                'is_stable': condition_number < self.config.max_condition_number
            }
        
        except np.linalg.LinAlgError as e:
            return {
                'condition_number': np.inf,
                'rank': 0,
                'singular_values': np.array([]),
                'effective_rank': 0,
                'is_stable': False,
                'error': str(e)
            }


class OutlierDetector:
    """Detects and handles numerical outliers."""
    
    def __init__(self, config: NumericalConfig):
        self.config = config
    
    def detect_outliers_zscore(self, data: np.ndarray, 
                              threshold: float = None) -> np.ndarray:
        """Detect outliers using z-score method."""
        if threshold is None:
            threshold = self.config.outlier_threshold
        
        if data.ndim == 1:
            z_scores = np.abs(zscore(data, nan_policy='omit'))
            return z_scores > threshold
        else:
            # For multi-dimensional data, check each dimension
            outliers = np.zeros(data.shape[0], dtype=bool)
            for i in range(data.shape[1]):
                z_scores = np.abs(zscore(data[:, i], nan_policy='omit'))
                outliers |= (z_scores > threshold)
            return outliers
    
    def detect_outliers_iqr(self, data: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Detect outliers using Interquartile Range (IQR) method."""
        if data.ndim == 1:
            Q1 = np.nanpercentile(data, 25)
            Q3 = np.nanpercentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return (data < lower_bound) | (data > upper_bound)
        else:
            outliers = np.zeros(data.shape[0], dtype=bool)
            for i in range(data.shape[1]):
                Q1 = np.nanpercentile(data[:, i], 25)
                Q3 = np.nanpercentile(data[:, i], 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                col_outliers = (data[:, i] < lower_bound) | (data[:, i] > upper_bound)
                outliers |= col_outliers
            return outliers
    
    def remove_outliers(self, data: np.ndarray, 
                       method: str = 'zscore') -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers from data."""
        if method == 'zscore':
            outlier_mask = self.detect_outliers_zscore(data)
        elif method == 'iqr':
            outlier_mask = self.detect_outliers_iqr(data)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        clean_data = data[~outlier_mask]
        outlier_indices = np.where(outlier_mask)[0]
        
        return clean_data, outlier_indices
    
    def robust_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate robust statistical measures."""
        return {
            'median': np.nanmedian(data),
            'mad': np.nanmedian(np.abs(data - np.nanmedian(data))),  # Median Absolute Deviation
            'iqr': np.nanpercentile(data, 75) - np.nanpercentile(data, 25),
            'trimmed_mean': np.mean(data[(data >= np.nanpercentile(data, 10)) & 
                                       (data <= np.nanpercentile(data, 90))]),
            'robust_std': 1.4826 * np.nanmedian(np.abs(data - np.nanmedian(data)))
        }


class ConvergenceChecker:
    """Checks numerical convergence of iterative algorithms."""
    
    def __init__(self, config: NumericalConfig):
        self.config = config
        self.history = []
    
    def reset(self):
        """Reset convergence history."""
        self.history = []
    
    def check_convergence(self, current_value: Union[float, np.ndarray],
                         previous_value: Union[float, np.ndarray] = None) -> bool:
        """Check if algorithm has converged."""
        self.history.append(current_value)
        
        if previous_value is None and len(self.history) < 2:
            return False
        
        if previous_value is None:
            previous_value = self.history[-2]
        
        # Calculate relative change
        if isinstance(current_value, np.ndarray):
            relative_change = np.linalg.norm(current_value - previous_value) / (
                np.linalg.norm(previous_value) + self.config.tolerance
            )
        else:
            relative_change = abs(current_value - previous_value) / (
                abs(previous_value) + self.config.tolerance
            )
        
        return relative_change < self.config.convergence_threshold
    
    def oscillation_detection(self, window_size: int = 10) -> bool:
        """Detect if the algorithm is oscillating."""
        if len(self.history) < window_size:
            return False
        
        recent_values = self.history[-window_size:]
        
        # Check for oscillating pattern
        if isinstance(recent_values[0], np.ndarray):
            # For arrays, check the variance of norms
            norms = [np.linalg.norm(val) for val in recent_values]
            return np.std(norms) / (np.mean(norms) + self.config.tolerance) > 0.1
        else:
            # For scalars, check variance directly
            return np.std(recent_values) / (np.mean(np.abs(recent_values)) + self.config.tolerance) > 0.1
    
    def get_convergence_rate(self) -> float:
        """Estimate convergence rate."""
        if len(self.history) < 3:
            return np.nan
        
        # Calculate successive ratios
        ratios = []
        for i in range(2, len(self.history)):
            if isinstance(self.history[i], np.ndarray):
                curr_error = np.linalg.norm(self.history[i] - self.history[i-1])
                prev_error = np.linalg.norm(self.history[i-1] - self.history[i-2])
            else:
                curr_error = abs(self.history[i] - self.history[i-1])
                prev_error = abs(self.history[i-1] - self.history[i-2])
            
            if prev_error > self.config.tolerance:
                ratios.append(curr_error / prev_error)
        
        return np.median(ratios) if ratios else np.nan


class ResourceMonitor:
    """Monitors computational resources during numerical computations."""
    
    def __init__(self, config: NumericalConfig):
        self.config = config
        self.start_time = None
        self.initial_memory = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = time.time()
        self.initial_memory = self._get_memory_usage()
    
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage."""
        if self.start_time is None:
            self.start_monitoring()
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        current_memory = self._get_memory_usage()
        memory_increase = current_memory - self.initial_memory
        
        status = {
            'elapsed_time': elapsed_time,
            'current_memory_gb': current_memory,
            'memory_increase_gb': memory_increase,
            'time_limit_exceeded': elapsed_time > self.config.time_limit_seconds,
            'memory_limit_exceeded': current_memory > self.config.memory_limit_gb
        }
        
        return status
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)  # Convert to GB
    
    def should_terminate(self) -> Tuple[bool, str]:
        """Check if computation should be terminated due to resource limits."""
        status = self.check_resources()
        
        if status['time_limit_exceeded']:
            return True, f"Time limit exceeded: {status['elapsed_time']:.1f}s"
        
        if status['memory_limit_exceeded']:
            return True, f"Memory limit exceeded: {status['current_memory_gb']:.2f}GB"
        
        return False, ""


class StableAlgorithms:
    """Collection of numerically stable algorithms."""
    
    def __init__(self, config: NumericalConfig):
        self.config = config
        self.precision_mgr = PrecisionManager(config)
    
    def stable_qr_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Numerically stable QR decomposition with pivoting."""
        with self.precision_mgr:
            matrix = self.precision_mgr.ensure_precision(matrix)
            
            try:
                # Use scipy's QR with pivoting for better stability
                Q, R, P = linalg.qr(matrix, pivoting=True, mode='economic')
                return Q, R
            except np.linalg.LinAlgError:
                # Fallback to regularized version
                regularized = matrix + self.config.tolerance * np.eye(matrix.shape[0])
                Q, R = np.linalg.qr(regularized)
                return Q, R
    
    def stable_eigenvalue_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Stable eigenvalue decomposition."""
        with self.precision_mgr:
            matrix = self.precision_mgr.ensure_precision(matrix)
            
            try:
                # Check if matrix is symmetric/hermitian for better stability
                if np.allclose(matrix, matrix.T, rtol=self.config.tolerance):
                    eigenvals, eigenvecs = linalg.eigh(matrix)
                else:
                    eigenvals, eigenvecs = linalg.eig(matrix)
                
                # Sort by eigenvalue magnitude
                idx = np.argsort(np.abs(eigenvals))[::-1]
                return eigenvals[idx], eigenvecs[:, idx]
            
            except np.linalg.LinAlgError:
                warnings.warn("Eigenvalue decomposition failed, returning NaN")
                n = matrix.shape[0]
                return np.full(n, np.nan), np.full((n, n), np.nan)
    
    def stable_matrix_inversion(self, matrix: np.ndarray) -> np.ndarray:
        """Stable matrix inversion using SVD."""
        with self.precision_mgr:
            matrix = self.precision_mgr.ensure_precision(matrix)
            
            try:
                U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
                
                # Threshold small singular values
                s_inv = np.zeros_like(s)
                s_thresh = s > self.config.tolerance * s[0]
                s_inv[s_thresh] = 1.0 / s[s_thresh]
                
                # Reconstruct inverse
                return Vt.T @ np.diag(s_inv) @ U.T
            
            except np.linalg.LinAlgError:
                warnings.warn("Matrix inversion failed, using pseudo-inverse")
                return linalg.pinv(matrix, rcond=self.config.tolerance)
    
    def stable_lyapunov_calculation(self, jacobian: np.ndarray, 
                                  dt: float) -> np.ndarray:
        """Stable calculation of Lyapunov exponents using QR method."""
        n = jacobian.shape[0]
        Q = np.eye(n)
        
        # Evolution matrix
        A = np.eye(n) + dt * jacobian
        
        # QR decomposition with error checking
        try:
            Q_new, R = self.stable_qr_decomposition(A @ Q)
            
            # Extract diagonal elements (Lyapunov contributions)
            lyap_contributions = np.log(np.abs(np.diag(R)))
            
            # Check for numerical issues
            if np.any(~np.isfinite(lyap_contributions)):
                warnings.warn("Numerical instability in Lyapunov calculation")
                lyap_contributions = np.nan_to_num(lyap_contributions, 
                                                 nan=0.0, posinf=0.0, neginf=0.0)
            
            return lyap_contributions, Q_new
        
        except Exception as e:
            warnings.warn(f"Stable Lyapunov calculation failed: {e}")
            return np.zeros(n), Q


class NumericalValidator:
    """Validates numerical results and detects issues."""
    
    def __init__(self, config: NumericalConfig):
        self.config = config
    
    def validate_array(self, array: np.ndarray, name: str = "array") -> Dict[str, Any]:
        """Comprehensive validation of numerical array."""
        validation_result = {
            'name': name,
            'shape': array.shape,
            'dtype': array.dtype,
            'is_valid': True,
            'issues': []
        }
        
        # Check for NaN values
        nan_count = np.sum(np.isnan(array))
        if nan_count > 0:
            validation_result['issues'].append(f"Contains {nan_count} NaN values")
            validation_result['is_valid'] = False
        
        # Check for infinite values
        inf_count = np.sum(np.isinf(array))
        if inf_count > 0:
            validation_result['issues'].append(f"Contains {inf_count} infinite values")
            validation_result['is_valid'] = False
        
        # Check for extremely large values
        max_val = np.nanmax(np.abs(array))
        if max_val > 1e10:
            validation_result['issues'].append(f"Contains very large values (max: {max_val:.2e})")
        
        # Check for extremely small values
        min_val = np.nanmin(np.abs(array[array != 0]))
        if min_val < 1e-15:
            validation_result['issues'].append(f"Contains very small values (min: {min_val:.2e})")
        
        # Check dynamic range
        if not np.all(array == 0):
            dynamic_range = max_val / (min_val + self.config.tolerance)
            if dynamic_range > 1e12:
                validation_result['issues'].append(f"Large dynamic range: {dynamic_range:.2e}")
        
        return validation_result
    
    def validate_convergence_result(self, final_value: Union[float, np.ndarray],
                                  history: list, name: str = "result") -> Dict[str, Any]:
        """Validate convergence result."""
        validation = {
            'name': name,
            'converged': False,
            'final_value': final_value,
            'iterations': len(history),
            'issues': []
        }
        
        if len(history) < 2:
            validation['issues'].append("Insufficient iteration history")
            return validation
        
        # Check for monotonic decrease (for error measures)
        if len(history) > 5:
            recent_trend = np.polyfit(range(len(history[-5:])), history[-5:], 1)[0]
            if recent_trend > 0:
                validation['issues'].append("Non-decreasing trend in recent iterations")
        
        # Check final convergence
        if isinstance(final_value, np.ndarray):
            final_change = np.linalg.norm(final_value - history[-2])
            validation['final_change'] = final_change
            validation['converged'] = final_change < self.config.convergence_threshold
        else:
            final_change = abs(final_value - history[-2])
            validation['final_change'] = final_change  
            validation['converged'] = final_change < self.config.convergence_threshold
        
        if not validation['converged']:
            validation['issues'].append(f"Not converged (final change: {final_change:.2e})")
        
        return validation


def create_stable_config(precision: str = 'float64',
                        tolerance: float = 1e-12,
                        max_iterations: int = 10000) -> NumericalConfig:
    """Create a numerical configuration optimized for stability."""
    return NumericalConfig(
        float_precision=precision,
        tolerance=tolerance,
        max_condition_number=1e10,
        max_iterations=max_iterations,
        convergence_threshold=tolerance * 10,
        outlier_threshold=5.0,
        memory_limit_gb=8.0,
        time_limit_seconds=3600.0,
        adaptive_precision=True,
        numerical_warnings=True
    )


def safe_divide(numerator: Union[float, np.ndarray], 
               denominator: Union[float, np.ndarray],
               default_value: float = 0.0,
               tolerance: float = 1e-15) -> Union[float, np.ndarray]:
    """Safe division with handling of near-zero denominators."""
    if isinstance(denominator, np.ndarray):
        result = np.full_like(numerator, default_value, dtype=float)
        safe_mask = np.abs(denominator) > tolerance
        result[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
        return result
    else:
        if abs(denominator) > tolerance:
            return numerator / denominator
        else:
            return default_value


def safe_log(x: Union[float, np.ndarray], 
            minimum_value: float = 1e-15) -> Union[float, np.ndarray]:
    """Safe logarithm with handling of non-positive values."""
    if isinstance(x, np.ndarray):
        safe_x = np.maximum(x, minimum_value)
        return np.log(safe_x)
    else:
        return np.log(max(x, minimum_value))


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Numerical Stability Tools...")
    
    # Create configuration
    config = create_stable_config(precision='float64', tolerance=1e-12)
    
    # Test precision manager
    with PrecisionManager(config) as pm:
        test_array = np.array([1.0, 2.0, 3.0])
        precise_array = pm.ensure_precision(test_array)
        print(f"Array dtype: {precise_array.dtype}")
    
    # Test condition number analysis
    cond_analyzer = ConditionNumberAnalyzer(config)
    test_matrix = np.array([[1, 2], [3, 4]], dtype=np.float64)
    cond_num = cond_analyzer.condition_number(test_matrix)
    print(f"Condition number: {cond_num:.2f}")
    print(f"Well conditioned: {cond_analyzer.is_well_conditioned(test_matrix)}")
    
    # Test outlier detection
    outlier_detector = OutlierDetector(config)
    test_data = np.array([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])  # 100 is outlier
    outliers = outlier_detector.detect_outliers_zscore(test_data)
    print(f"Outliers detected: {np.sum(outliers)} positions")
    
    # Test convergence checker
    conv_checker = ConvergenceChecker(config)
    for i in range(10):
        value = 1.0 / (i + 1)  # Converging series
        converged = conv_checker.check_convergence(value)
        if converged:
            print(f"Converged at iteration {i}")
            break
    
    # Test resource monitoring
    resource_monitor = ResourceMonitor(config)
    resource_monitor.start_monitoring()
    status = resource_monitor.check_resources()
    print(f"Memory usage: {status['current_memory_gb']:.3f} GB")
    
    # Test validator
    validator = NumericalValidator(config)
    test_result = validator.validate_array(precise_array, "test_array")
    print(f"Array validation: {'PASS' if test_result['is_valid'] else 'FAIL'}")
    
    print("Numerical Stability Tools testing completed!")