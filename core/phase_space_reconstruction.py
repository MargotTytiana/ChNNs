"""
Phase Space Reconstruction for Chaotic Time Series Analysis.

This module implements comprehensive phase space reconstruction techniques
including time delay embedding, optimal parameter estimation, and various
embedding methods based on Takens' embedding theorem.

Author: C-HiLAP Project
Date: 2025
"""

import numpy as np
import warnings
from typing import Tuple, Union, Optional, Dict, Any, List
from scipy import spatial
from scipy.stats import entropy
from scipy.signal import correlate, find_peaks
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

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
    
@dataclass
class EmbeddingConfig:
    """Configuration for phase space reconstruction parameters."""
    max_delay: int = 50
    max_dimension: int = 20
    min_dimension: int = 2
    correlation_threshold: float = 0.1
    false_neighbor_threshold: float = 15.0
    cao_threshold: float = 0.95
    mutual_info_bins: int = 50
    nearest_neighbors: int = 10
    rtol: float = 15.0  # Relative tolerance for false neighbors
    atol: float = 2.0   # Absolute tolerance for false neighbors
    min_samples_fraction: float = 0.1
    saturation_tolerance: float = 0.01


class EmbeddingMethod(ABC):
    """Abstract base class for embedding parameter estimation methods."""
    
    @abstractmethod
    def estimate(self, data: np.ndarray, config: EmbeddingConfig) -> Union[int, float]:
        """Estimate embedding parameter from data."""
        pass


class AutocorrelationDelayEstimator(EmbeddingMethod):
    """Estimate optimal delay time using autocorrelation function."""
    
    def estimate(self, data: np.ndarray, config: EmbeddingConfig) -> int:
        """
        Estimate delay time as first minimum of autocorrelation function.
        
        Args:
            data: Input time series
            config: Embedding configuration
            
        Returns:
            Optimal delay time
        """
        # Calculate normalized autocorrelation
        autocorr = self._calculate_autocorrelation(data, config.max_delay)
        
        # Find first minimum
        delay = self._find_first_minimum(autocorr, config.correlation_threshold)
        
        return max(1, min(delay, config.max_delay))
    
    def _calculate_autocorrelation(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate normalized autocorrelation function."""
        n = len(data)
        data_centered = data - np.mean(data)
        
        # Use scipy correlate for efficiency
        full_corr = correlate(data_centered, data_centered, mode='full')
        
        # Take the second half (positive lags)
        autocorr = full_corr[n-1:n-1+max_lag+1]
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        return autocorr
    
    def _find_first_minimum(self, autocorr: np.ndarray, threshold: float) -> int:
        """Find first minimum or crossing of threshold."""
        # Find where autocorrelation drops below threshold
        below_threshold = np.where(autocorr < threshold)[0]
        if len(below_threshold) > 0:
            return below_threshold[0]
        
        # If no threshold crossing, find first local minimum
        if len(autocorr) < 3:
            return 1
        
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] < autocorr[i-1] and autocorr[i] < autocorr[i+1]:
                return i
        
        # Fallback to 1/e time
        e_time = np.where(autocorr < 1.0/np.e)[0]
        return e_time[0] if len(e_time) > 0 else len(autocorr) // 4


class MutualInformationDelayEstimator(EmbeddingMethod):
    """Estimate optimal delay time using mutual information."""
    
    def estimate(self, data: np.ndarray, config: EmbeddingConfig) -> int:
        """
        Estimate delay time as first minimum of mutual information.
        
        Args:
            data: Input time series
            config: Embedding configuration
            
        Returns:
            Optimal delay time
        """
        mutual_info = self._calculate_mutual_information(data, config)
        
        # Find first minimum
        delay = self._find_first_minimum(mutual_info)
        
        return max(1, min(delay, config.max_delay))
    
    def _calculate_mutual_information(self, data: np.ndarray, config: EmbeddingConfig) -> np.ndarray:
        """Calculate mutual information between x(t) and x(t+tau)."""
        n = len(data)
        max_delay = min(config.max_delay, n // 2)
        mutual_info = np.zeros(max_delay)
        
        for tau in range(max_delay):
            if tau == 0:
                mutual_info[tau] = self._calculate_entropy(data, config.mutual_info_bins)
            else:
                x1 = data[:-tau]
                x2 = data[tau:]
                mutual_info[tau] = self._calculate_mutual_info_pair(
                    x1, x2, config.mutual_info_bins
                )
        
        return mutual_info
    
    def _calculate_entropy(self, data: np.ndarray, bins: int) -> float:
        """Calculate Shannon entropy of data."""
        hist, _ = np.histogram(data, bins=bins)
        hist = hist[hist > 0]  # Remove zero bins
        prob = hist / np.sum(hist)
        return -np.sum(prob * np.log2(prob))
    
    def _calculate_mutual_info_pair(self, x1: np.ndarray, x2: np.ndarray, bins: int) -> float:
        """Calculate mutual information between two variables."""
        # Calculate joint histogram
        hist_2d, _, _ = np.histogram2d(x1, x2, bins=bins)
        
        # Calculate marginal histograms
        hist_x1 = np.histogram(x1, bins=bins)[0]
        hist_x2 = np.histogram(x2, bins=bins)[0]
        
        # Convert to probabilities
        p_joint = hist_2d / np.sum(hist_2d)
        p_x1 = hist_x1 / np.sum(hist_x1)
        p_x2 = hist_x2 / np.sum(hist_x2)
        
        # Calculate mutual information
        mutual_info = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_joint[i, j] > 0 and p_x1[i] > 0 and p_x2[j] > 0:
                    mutual_info += p_joint[i, j] * np.log2(
                        p_joint[i, j] / (p_x1[i] * p_x2[j])
                    )
        
        return mutual_info
    
    def _find_first_minimum(self, mutual_info: np.ndarray) -> int:
        """Find first local minimum in mutual information."""
        if len(mutual_info) < 3:
            return 1
        
        # Find local minima
        for i in range(1, len(mutual_info) - 1):
            if (mutual_info[i] < mutual_info[i-1] and 
                mutual_info[i] < mutual_info[i+1]):
                return i
        
        # If no minimum found, return the point of greatest decrease
        diff = np.diff(mutual_info)
        min_idx = np.argmin(diff)
        return min_idx + 1


class FalseNearestNeighborsDimensionEstimator(EmbeddingMethod):
    """Estimate optimal embedding dimension using false nearest neighbors method."""
    
    def estimate(self, data: np.ndarray, config: EmbeddingConfig) -> int:
        """
        Estimate embedding dimension using false nearest neighbors.
        
        Args:
            data: Input time series
            config: Embedding configuration
            
        Returns:
            Optimal embedding dimension
        """
        delay = 1  # Use delay=1 or estimate separately
        
        false_neighbor_fraction = []
        
        for dim in range(config.min_dimension, config.max_dimension + 1):
            # Create embedding
            embedded = self._time_delay_embedding(data, dim, delay)
            
            # Calculate false neighbor fraction
            fraction = self._calculate_false_neighbors(embedded, config)
            false_neighbor_fraction.append(fraction)
            
            # Stop if fraction is below threshold
            if fraction < config.false_neighbor_threshold / 100.0:
                return dim
        
        # If no clear minimum, find elbow point
        return self._find_elbow_point(false_neighbor_fraction, config.min_dimension)
    
    def _time_delay_embedding(self, data: np.ndarray, dim: int, delay: int) -> np.ndarray:
        """Create time delay embedding."""
        n = len(data)
        m = n - (dim - 1) * delay
        
        if m <= 0:
            raise ValueError("Insufficient data length for embedding")
        
        embedded = np.zeros((m, dim))
        for i in range(dim):
            embedded[:, i] = data[i * delay:i * delay + m]
        
        return embedded
    
    def _calculate_false_neighbors(self, embedded: np.ndarray, config: EmbeddingConfig) -> float:
        """Calculate fraction of false nearest neighbors."""
        n_points, dim = embedded.shape
        
        # Use subset for efficiency
        n_test = min(n_points, max(100, int(n_points * config.min_samples_fraction)))
        indices = np.random.choice(n_points, n_test, replace=False)
        test_points = embedded[indices]
        
        # Build nearest neighbor tree
        nbrs = NearestNeighbors(n_neighbors=2).fit(embedded)
        distances, neighbor_indices = nbrs.kneighbors(test_points)
        
        false_neighbors = 0
        
        for i, point_idx in enumerate(indices):
            # Get nearest neighbor
            neighbor_idx = neighbor_indices[i, 1]  # Skip self (index 0)
            neighbor_dist = distances[i, 1]
            
            if neighbor_dist == 0:
                continue
            
            # Check if it's a false neighbor by extending dimension
            if self._is_false_neighbor(
                point_idx, neighbor_idx, embedded, neighbor_dist, config
            ):
                false_neighbors += 1
        
        return false_neighbors / n_test
    
    def _is_false_neighbor(self, point_idx: int, neighbor_idx: int, 
                          embedded: np.ndarray, neighbor_dist: float,
                          config: EmbeddingConfig) -> bool:
        """Check if neighbor is false by dimension extension criteria."""
        dim = embedded.shape[1]
        
        # Calculate distance in current dimension
        current_dist = neighbor_dist
        
        # Calculate distance if we had one more dimension (simulate)
        # Use the actual data points if available
        if point_idx + dim < len(embedded) and neighbor_idx + dim < len(embedded):
            # This is a simplified check - in practice, you'd need the original time series
            extended_diff = abs(
                embedded[point_idx, -1] - embedded[neighbor_idx, -1]
            )
            
            # Relative tolerance criterion
            if extended_diff / current_dist > config.rtol:
                return True
            
            # Absolute tolerance criterion (simplified)
            if extended_diff > config.atol * np.std(embedded[:, -1]):
                return True
        
        return False
    
    def _find_elbow_point(self, fractions: List[float], min_dim: int) -> int:
        """Find elbow point in false neighbor fraction curve."""
        if len(fractions) < 2:
            return min_dim
        
        # Calculate second derivative to find elbow
        fractions = np.array(fractions)
        if len(fractions) >= 3:
            second_deriv = np.diff(fractions, 2)
            elbow_idx = np.argmax(second_deriv) + 2  # Account for diff operations
            return min_dim + elbow_idx
        
        # Fallback: find largest decrease
        diff = np.diff(fractions)
        max_decrease_idx = np.argmax(-diff)
        return min_dim + max_decrease_idx + 1


class CaoDimensionEstimator(EmbeddingMethod):
    """Estimate optimal embedding dimension using Cao's method."""
    
    def estimate(self, data: np.ndarray, config: EmbeddingConfig) -> int:
        """
        Estimate embedding dimension using Cao's method.
        
        Args:
            data: Input time series
            config: Embedding configuration
            
        Returns:
            Optimal embedding dimension
        """
        delay = 1  # Use fixed delay or estimate separately
        
        E1_values = []
        E2_values = []
        
        for dim in range(config.min_dimension, config.max_dimension):
            E1, E2 = self._calculate_cao_statistics(data, dim, delay)
            E1_values.append(E1)
            if E2 is not None:
                E2_values.append(E2)
        
        # Find saturation point in E1
        optimal_dim = self._find_saturation_point(E1_values, config)
        
        return config.min_dimension + optimal_dim
    
    def _calculate_cao_statistics(self, data: np.ndarray, dim: int, delay: int) -> Tuple[float, Optional[float]]:
        """Calculate E1 and E2 statistics for Cao's method."""
        # Create embeddings for dimension m and m+1
        embedded_m = self._time_delay_embedding(data, dim, delay)
        
        try:
            embedded_m1 = self._time_delay_embedding(data, dim + 1, delay)
        except ValueError:
            return 0.0, None
        
        n_points = embedded_m.shape[0]
        
        # Find nearest neighbors in m-dimensional space
        nbrs = NearestNeighbors(n_neighbors=2).fit(embedded_m)
        distances_m, indices_m = nbrs.kneighbors(embedded_m)
        
        # Calculate a(i,m) values
        a_values = []
        
        for i in range(n_points):
            nearest_idx = indices_m[i, 1]  # Skip self
            
            # Distance in m dimensions
            dist_m = distances_m[i, 1]
            
            if dist_m > 0:
                # Distance in m+1 dimensions
                dist_m1 = np.linalg.norm(embedded_m1[i] - embedded_m1[nearest_idx])
                
                a_values.append(dist_m1 / dist_m)
        
        if not a_values:
            return 0.0, None
        
        E1 = np.mean(a_values)
        
        # Calculate E2 for deterministic vs stochastic test
        E2 = None
        if dim > 1:
            # This is a simplified version - full implementation would require
            # comparing with surrogate data
            E2 = np.std(a_values) / np.mean(a_values) if np.mean(a_values) > 0 else 0
        
        return E1, E2
    
    def _time_delay_embedding(self, data: np.ndarray, dim: int, delay: int) -> np.ndarray:
        """Create time delay embedding."""
        n = len(data)
        m = n - (dim - 1) * delay
        
        if m <= 0:
            raise ValueError("Insufficient data length for embedding")
        
        embedded = np.zeros((m, dim))
        for i in range(dim):
            embedded[:, i] = data[i * delay:i * delay + m]
        
        return embedded
    
    def _find_saturation_point(self, E1_values: List[float], config: EmbeddingConfig) -> int:
        """Find saturation point in E1 values."""
        if len(E1_values) < 2:
            return 0
        
        E1_array = np.array(E1_values)
        
        # Look for where E1(m+1)/E1(m) approaches 1
        ratios = E1_array[1:] / (E1_array[:-1] + 1e-15)
        
        # Find first point where ratio is close to 1
        for i, ratio in enumerate(ratios):
            if abs(ratio - 1.0) < config.saturation_tolerance:
                return i + 1
        
        # If no clear saturation, find minimum slope
        if len(E1_array) >= 3:
            slopes = np.diff(E1_array)
            min_slope_idx = np.argmin(np.abs(slopes))
            return min_slope_idx + 1
        
        return len(E1_values) // 2


class TimeDelayEmbedding:
    """Core time delay embedding implementation."""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
    
    def embed(self, data: np.ndarray, dimension: int, delay: int) -> np.ndarray:
        """
        Create time delay embedding of time series.
        
        Args:
            data: Input time series
            dimension: Embedding dimension
            delay: Time delay
            
        Returns:
            Embedded data matrix
        """
        if data.ndim != 1:
            raise ValueError("Input data must be 1-dimensional")
        
        n = len(data)
        m = n - (dimension - 1) * delay
        
        if m <= 0:
            raise ValueError(f"Insufficient data length. Need at least {(dimension - 1) * delay + 1} points")
        
        # Create embedding matrix
        embedded = np.zeros((m, dimension))
        
        for i in range(dimension):
            start_idx = i * delay
            end_idx = start_idx + m
            embedded[:, i] = data[start_idx:end_idx]
        
        return embedded
    
    def multivariate_embed(self, data: np.ndarray, dimensions: List[int], 
                          delays: List[int]) -> np.ndarray:
        """
        Create multivariate time delay embedding.
        
        Args:
            data: Input multivariate time series (n_samples, n_variables)
            dimensions: Embedding dimensions for each variable
            delays: Time delays for each variable
            
        Returns:
            Multivariate embedded data matrix
        """
        if data.ndim != 2:
            raise ValueError("Input data must be 2-dimensional for multivariate embedding")
        
        n_samples, n_vars = data.shape
        
        if len(dimensions) != n_vars or len(delays) != n_vars:
            raise ValueError("Dimensions and delays must match number of variables")
        
        # Calculate total embedding dimension and required samples
        total_dim = sum(dimensions)
        max_delay_span = max((dim - 1) * delay for dim, delay in zip(dimensions, delays))
        m = n_samples - max_delay_span
        
        if m <= 0:
            raise ValueError("Insufficient data length for multivariate embedding")
        
        # Create embedding matrix
        embedded = np.zeros((m, total_dim))
        col_idx = 0
        
        for var_idx in range(n_vars):
            dim = dimensions[var_idx]
            delay = delays[var_idx]
            
            for i in range(dim):
                start_idx = i * delay
                end_idx = start_idx + m
                embedded[:, col_idx] = data[start_idx:end_idx, var_idx]
                col_idx += 1
        
        return embedded


class PhaseSpaceReconstructor:
    """Main phase space reconstruction class."""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        
        # Initialize parameter estimators
        self.delay_estimators = {
            'autocorrelation': AutocorrelationDelayEstimator(),
            'mutual_information': MutualInformationDelayEstimator()
        }
        
        self.dimension_estimators = {
            'false_neighbors': FalseNearestNeighborsDimensionEstimator(),
            'cao': CaoDimensionEstimator()
        }
        
        self.embedding = TimeDelayEmbedding(config)
    
    def reconstruct(self, data: np.ndarray, 
                   delay: Optional[int] = None,
                   dimension: Optional[int] = None,
                   delay_method: str = 'autocorrelation',
                   dimension_method: str = 'false_neighbors') -> Dict[str, Any]:
        """
        Perform complete phase space reconstruction.
        
        Args:
            data: Input time series
            delay: Time delay (estimated if None)
            dimension: Embedding dimension (estimated if None)
            delay_method: Method for delay estimation
            dimension_method: Method for dimension estimation
            
        Returns:
            Dictionary containing reconstruction results
        """
        results = {
            'original_data': data,
            'config': self.config
        }
        
        # Estimate delay if not provided
        if delay is None:
            if delay_method in self.delay_estimators:
                delay = self.delay_estimators[delay_method].estimate(data, self.config)
                results['delay_estimated'] = True
                results['delay_method'] = delay_method
            else:
                delay = 1
                results['delay_estimated'] = False
                warnings.warn(f"Unknown delay method {delay_method}, using delay=1")
        else:
            results['delay_estimated'] = False
        
        results['delay'] = delay
        
        # Estimate dimension if not provided
        if dimension is None:
            if dimension_method in self.dimension_estimators:
                dimension = self.dimension_estimators[dimension_method].estimate(data, self.config)
                results['dimension_estimated'] = True
                results['dimension_method'] = dimension_method
            else:
                dimension = 3
                results['dimension_estimated'] = False
                warnings.warn(f"Unknown dimension method {dimension_method}, using dimension=3")
        else:
            results['dimension_estimated'] = False
        
        results['dimension'] = dimension
        
        # Perform embedding
        try:
            embedded_data = self.embedding.embed(data, dimension, delay)
            results['embedded_data'] = embedded_data
            results['embedding_success'] = True
            results['n_embedded_points'] = embedded_data.shape[0]
        except ValueError as e:
            results['embedded_data'] = None
            results['embedding_success'] = False
            results['error'] = str(e)
            warnings.warn(f"Embedding failed: {e}")
        
        # Calculate reconstruction quality metrics
        if results['embedding_success']:
            results['quality_metrics'] = self._calculate_quality_metrics(
                data, embedded_data, delay, dimension
            )
        
        return results
    
    def _calculate_quality_metrics(self, original_data: np.ndarray, 
                                 embedded_data: np.ndarray,
                                 delay: int, dimension: int) -> Dict[str, float]:
        """Calculate quality metrics for phase space reconstruction."""
        metrics = {}
        
        # Embedding efficiency
        original_length = len(original_data)
        embedded_length = embedded_data.shape[0]
        metrics['embedding_efficiency'] = embedded_length / original_length
        
        # Fill factor (how well the embedding fills the phase space)
        try:
            # Calculate convex hull volume (simplified)
            from scipy.spatial import ConvexHull
            if embedded_data.shape[1] >= 2 and embedded_data.shape[0] > embedded_data.shape[1]:
                hull = ConvexHull(embedded_data)
                metrics['convex_hull_volume'] = hull.volume
            else:
                metrics['convex_hull_volume'] = np.nan
        except:
            metrics['convex_hull_volume'] = np.nan
        
        # Variance explained by each dimension
        variances = np.var(embedded_data, axis=0)
        total_variance = np.sum(variances)
        if total_variance > 0:
            metrics['variance_ratios'] = (variances / total_variance).tolist()
            metrics['effective_dimensions'] = np.sum((variances / total_variance) > 0.01)
        else:
            metrics['variance_ratios'] = [np.nan] * dimension
            metrics['effective_dimensions'] = 0
        
        # Correlation between embedding dimensions
        correlation_matrix = np.corrcoef(embedded_data.T)
        off_diagonal = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        metrics['mean_interdim_correlation'] = np.mean(np.abs(off_diagonal))
        metrics['max_interdim_correlation'] = np.max(np.abs(off_diagonal))
        
        return metrics
    
    def estimate_parameters(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Estimate optimal embedding parameters using multiple methods.
        
        Args:
            data: Input time series
            
        Returns:
            Dictionary with parameter estimates from different methods
        """
        results = {}
        
        # Estimate delay using different methods
        delay_estimates = {}
        for method_name, estimator in self.delay_estimators.items():
            try:
                delay_estimates[method_name] = estimator.estimate(data, self.config)
            except Exception as e:
                delay_estimates[method_name] = None
                warnings.warn(f"Delay estimation failed for {method_name}: {e}")
        
        results['delay_estimates'] = delay_estimates
        
        # Estimate dimension using different methods
        dimension_estimates = {}
        for method_name, estimator in self.dimension_estimators.items():
            try:
                dimension_estimates[method_name] = estimator.estimate(data, self.config)
            except Exception as e:
                dimension_estimates[method_name] = None
                warnings.warn(f"Dimension estimation failed for {method_name}: {e}")
        
        results['dimension_estimates'] = dimension_estimates
        
        # Consensus estimates
        valid_delays = [d for d in delay_estimates.values() if d is not None]
        valid_dimensions = [d for d in dimension_estimates.values() if d is not None]
        
        results['consensus_delay'] = int(np.median(valid_delays)) if valid_delays else 1
        results['consensus_dimension'] = int(np.median(valid_dimensions)) if valid_dimensions else 3
        
        return results


def plot_embedding(embedded_data: np.ndarray, title: str = "Phase Space Reconstruction"):
    """
    Plot phase space reconstruction.
    
    Args:
        embedded_data: Embedded data matrix
        title: Plot title
    """
    if embedded_data.shape[1] < 2:
        warnings.warn("Cannot plot embedding with dimension < 2")
        return
    
    fig = plt.figure(figsize=(12, 4))
    
    if embedded_data.shape[1] == 2:
        plt.plot(embedded_data[:, 0], embedded_data[:, 1], 'b-', alpha=0.7, linewidth=0.5)
        plt.plot(embedded_data[0, 0], embedded_data[0, 1], 'ro', markersize=8, label='Start')
        plt.xlabel('x(t)')
        plt.ylabel('x(t+τ)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    elif embedded_data.shape[1] >= 3:
        ax = fig.add_subplot(121, projection='3d')
        ax.plot(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], 
                'b-', alpha=0.7, linewidth=0.5)
        ax.scatter(embedded_data[0, 0], embedded_data[0, 1], embedded_data[0, 2], 
                  c='red', s=50, label='Start')
        ax.set_xlabel('x(t)')
        ax.set_ylabel('x(t+τ)')
        ax.set_zlabel('x(t+2τ)')
        ax.set_title('3D ' + title)
        ax.legend()
        
        # 2D projection
        ax2 = fig.add_subplot(122)
        ax2.plot(embedded_data[:, 0], embedded_data[:, 1], 'b-', alpha=0.7, linewidth=0.5)
        ax2.plot(embedded_data[0, 0], embedded_data[0, 1], 'ro', markersize=8, label='Start')
        ax2.set_xlabel('x(t)')
        ax2.set_ylabel('x(t+τ)')
        ax2.set_title('2D Projection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Phase Space Reconstruction...")
    
    # Generate test data (Lorenz system)
    def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
        x, y, z = state
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
    
    # Generate Lorenz time series
    from scipy.integrate import solve_ivp
    
    t_span = (0, 50)
    t_eval = np.arange(0, 50, 0.01)
    initial_state = [1.0, 1.0, 1.0]
    
    sol = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t_eval)
    lorenz_x = sol.y[0]  # Use x-component
    
    print(f"Generated {len(lorenz_x)} data points from Lorenz system")
    
    # Create reconstructor
    config = EmbeddingConfig(max_delay=50, max_dimension=10)
    reconstructor = PhaseSpaceReconstructor(config)
    
    # Estimate parameters
    param_estimates = reconstructor.estimate_parameters(lorenz_x)
    print(f"Parameter estimates: {param_estimates}")
    
    # Perform reconstruction
    results = reconstructor.reconstruct(
        lorenz_x, 
        delay=None, 
        dimension=None,
        delay_method='autocorrelation',
        dimension_method='false_neighbors'
    )
    
    print(f"Reconstruction successful: {results['embedding_success']}")
    if results['embedding_success']:
        print(f"Optimal delay: {results['delay']}")
        print(f"Optimal dimension: {results['dimension']}")
        print(f"Embedded points: {results['n_embedded_points']}")
        print(f"Quality metrics: {results['quality_metrics']}")
        
        # Plot reconstruction (if matplotlib available)
        try:
            plot_embedding(results['embedded_data'], "Lorenz System Reconstruction")
        except:
            print("Plotting not available")
    
    print("Phase Space Reconstruction testing completed!")