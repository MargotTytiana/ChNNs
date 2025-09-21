import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging


class AttractorPooling(nn.Module):
    """
    Strange Attractor Pooling Layer - Extracts topological invariants from chaotic trajectories.
    
    Converts dynamic attractor trajectories into static topological features by computing
    geometric and dynamical invariants that are robust to time shifts and minor perturbations.
    """
    
    def __init__(
        self,
        pooling_type: str = 'comprehensive',
        correlation_radii: Optional[List[float]] = None,
        min_radius: float = 0.001,
        max_radius: float = 10.0,
        num_radii: int = 20,
        lyapunov_window: int = 50,
        entropy_bins: int = 32,
        device: str = 'cpu',
        eps: float = 1e-8
    ):
        """
        Initialize the Attractor Pooling Layer.
        
        Args:
            pooling_type: Type of pooling ('basic', 'comprehensive', 'learnable')
            correlation_radii: List of radii for correlation dimension calculation
            min_radius: Minimum radius for correlation integral
            max_radius: Maximum radius for correlation integral  
            num_radii: Number of radii points for dimension estimation
            lyapunov_window: Window size for local Lyapunov exponent estimation
            entropy_bins: Number of bins for entropy estimation
            device: Computation device
            eps: Small value for numerical stability
        """
        super(AttractorPooling, self).__init__()
        
        self.pooling_type = pooling_type
        self.lyapunov_window = lyapunov_window
        self.entropy_bins = entropy_bins
        self.device = device
        self.eps = eps
        
        # Setup correlation radii for dimension calculation
        if correlation_radii is None:
            self.correlation_radii = torch.logspace(
                np.log10(min_radius), np.log10(max_radius), num_radii,
                device=device, dtype=torch.float32
            )
        else:
            self.correlation_radii = torch.tensor(
                correlation_radii, device=device, dtype=torch.float32
            )
        
        # Learnable pooling components
        if pooling_type == 'learnable':
            self._setup_learnable_components()
        
        # Output dimensions based on pooling type
        self.output_dim = self._get_output_dim()
    
    def _setup_learnable_components(self):
        """Setup learnable parameters for adaptive pooling."""
        # Learnable weights for different invariants
        self.invariant_weights = nn.Parameter(torch.ones(5))  # 5 main invariants
        
        # Learnable radius selection network
        self.radius_selector = nn.Sequential(
            nn.Linear(3, 16),  # Input: trajectory statistics
            nn.ReLU(),
            nn.Linear(16, len(self.correlation_radii)),
            nn.Softmax(dim=-1)
        )
        
        # Feature combination network
        self.feature_combiner = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(), 
            nn.Linear(8, 3),
            nn.Tanh()
        )
    
    def _get_output_dim(self) -> int:
        """Get output dimension based on pooling type."""
        if self.pooling_type == 'basic':
            return 3  # D2, DL, K
        elif self.pooling_type == 'comprehensive':
            return 5  # D2, DL, K, + additional invariants
        elif self.pooling_type == 'learnable':
            return 3  # Learned combination
        else:
            return 3
    
    def _compute_pairwise_distances(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between trajectory points.
        
        Args:
            trajectory: Trajectory tensor [batch_size, num_steps, 3]
            
        Returns:
            Distance matrix [batch_size, num_steps, num_steps]
        """
        batch_size, num_steps, dim = trajectory.shape
        
        # Efficient pairwise distance computation
        # |a - b|^2 = |a|^2 + |b|^2 - 2<a,b>
        trajectory_norm = torch.sum(trajectory ** 2, dim=2, keepdim=True)  # [B, T, 1]
        
        distances_sq = (
            trajectory_norm +  # [B, T, 1]
            trajectory_norm.transpose(1, 2) -  # [B, 1, T]
            2 * torch.bmm(trajectory, trajectory.transpose(1, 2))  # [B, T, T]
        )
        
        # Clamp to avoid numerical errors and take sqrt
        distances = torch.sqrt(torch.clamp(distances_sq, min=self.eps))
        
        return distances
    
    def _compute_correlation_integral(
        self, 
        distances: torch.Tensor, 
        radius: float
    ) -> torch.Tensor:
        """
        Compute correlation integral C(r) for given radius.
        
        Args:
            distances: Pairwise distance matrix [batch_size, N, N]
            radius: Correlation radius
            
        Returns:
            Correlation integral values [batch_size]
        """
        batch_size, N, _ = distances.shape
        
        # Count pairs with distance < radius (excluding diagonal)
        mask = (distances < radius) & ~torch.eye(N, device=self.device).bool()
        counts = torch.sum(mask.float(), dim=(1, 2))  # [batch_size]
        
        # Normalize by total number of pairs
        total_pairs = N * (N - 1)
        correlation_integral = counts / total_pairs
        
        return correlation_integral
    
    def _compute_correlation_dimension(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation dimension D2 using correlation integral.
        
        Args:
            trajectory: Trajectory tensor [batch_size, num_steps, 3]
            
        Returns:
            Correlation dimensions [batch_size]
        """
        distances = self._compute_pairwise_distances(trajectory)
        batch_size = trajectory.shape[0]
        
        # Compute correlation integrals for different radii
        log_radii = torch.log(self.correlation_radii + self.eps)
        log_correlations = []
        
        for radius in self.correlation_radii:
            C_r = self._compute_correlation_integral(distances, radius)
            log_C_r = torch.log(C_r + self.eps)
            log_correlations.append(log_C_r)
        
        log_correlations = torch.stack(log_correlations, dim=1)  # [batch_size, num_radii]
        
        # Estimate dimension using linear regression in log-log space
        # D2 = d(log C)/d(log r)
        dimensions = []
        for b in range(batch_size):
            # Simple finite difference approximation
            valid_mask = torch.isfinite(log_correlations[b])
            if torch.sum(valid_mask) > 2:
                valid_log_radii = log_radii[valid_mask]
                valid_log_corr = log_correlations[b][valid_mask]
                
                # Linear regression slope
                if len(valid_log_radii) > 1:
                    slope = torch.mean(
                        (valid_log_corr[1:] - valid_log_corr[:-1]) / 
                        (valid_log_radii[1:] - valid_log_radii[:-1])
                    )
                    dimensions.append(torch.clamp(slope, 0.1, 3.0))
                else:
                    dimensions.append(torch.tensor(1.5, device=self.device))
            else:
                dimensions.append(torch.tensor(1.5, device=self.device))
        
        return torch.stack(dimensions)
    
    def _compute_local_lyapunov_exponents(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute local Lyapunov exponents from trajectory.
        
        Args:
            trajectory: Trajectory tensor [batch_size, num_steps, 3]
            
        Returns:
            Local Lyapunov exponents [batch_size, 3]
        """
        batch_size, num_steps, dim = trajectory.shape
        
        if num_steps < self.lyapunov_window + 1:
            # Not enough data, return default values
            return torch.ones(batch_size, dim, device=self.device) * 0.1
        
        lyapunov_exponents = []
        
        for b in range(batch_size):
            traj = trajectory[b]  # [num_steps, 3]
            local_exponents = []
            
            # Compute local divergence rates
            for i in range(num_steps - self.lyapunov_window):
                start_point = traj[i]  # [3]
                end_segment = traj[i:i+self.lyapunov_window]  # [window, 3]
                
                # Find nearest neighbor at start
                distances_to_start = torch.norm(traj - start_point, dim=1)
                distances_to_start[i] = float('inf')  # Exclude self
                
                nearest_idx = torch.argmin(distances_to_start)
                nearest_point = traj[nearest_idx]
                
                # Track evolution of separation
                if nearest_idx + self.lyapunov_window < num_steps:
                    nearest_segment = traj[nearest_idx:nearest_idx+self.lyapunov_window]
                    
                    initial_sep = torch.norm(start_point - nearest_point)
                    final_sep = torch.norm(
                        end_segment[-1] - nearest_segment[-1]
                    )
                    
                    if initial_sep > self.eps and final_sep > self.eps:
                        local_exp = torch.log(final_sep / initial_sep) / self.lyapunov_window
                        local_exponents.append(local_exp)
            
            if local_exponents:
                # Average local exponents and replicate for each dimension
                avg_exp = torch.mean(torch.stack(local_exponents))
                lyapunov_exponents.append(
                    torch.tensor([avg_exp, avg_exp*0.5, avg_exp*0.1], device=self.device)
                )
            else:
                lyapunov_exponents.append(
                    torch.tensor([0.1, 0.05, 0.01], device=self.device)
                )
        
        return torch.stack(lyapunov_exponents)
    
    def _compute_lyapunov_dimension(self, lyapunov_exponents: torch.Tensor) -> torch.Tensor:
        """
        Compute Lyapunov dimension from exponents.
        
        Args:
            lyapunov_exponents: Lyapunov exponents [batch_size, 3]
            
        Returns:
            Lyapunov dimensions [batch_size]
        """
        batch_size = lyapunov_exponents.shape[0]
        dimensions = []
        
        for b in range(batch_size):
            exps = lyapunov_exponents[b]
            sorted_exps, _ = torch.sort(exps, descending=True)
            
            cumsum = torch.cumsum(sorted_exps, dim=0)
            
            # Find largest k where sum of first k exponents is positive
            k = 0
            for i in range(len(sorted_exps)):
                if cumsum[i] > 0:
                    k = i + 1
                else:
                    break
            
            if k > 0 and k < len(sorted_exps):
                # DL = k + sum(λ_i, i=1..k) / |λ_{k+1}|
                sum_positive = cumsum[k-1]
                next_exp = torch.abs(sorted_exps[k])
                if next_exp > self.eps:
                    dimension = k + sum_positive / next_exp
                else:
                    dimension = k
            else:
                dimension = 1.0
            
            dimensions.append(torch.clamp(torch.tensor(dimension, device=self.device), 0.1, 3.0))
        
        return torch.stack(dimensions)
    
    def _compute_kolmogorov_entropy(self, lyapunov_exponents: torch.Tensor) -> torch.Tensor:
        """
        Compute Kolmogorov entropy (sum of positive Lyapunov exponents).
        
        Args:
            lyapunov_exponents: Lyapunov exponents [batch_size, 3]
            
        Returns:
            Kolmogorov entropies [batch_size]
        """
        positive_exponents = torch.clamp(lyapunov_exponents, min=0)
        entropy = torch.sum(positive_exponents, dim=1)
        return entropy
    
    def _compute_additional_invariants(self, trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute additional topological invariants for comprehensive pooling.
        
        Args:
            trajectory: Trajectory tensor [batch_size, num_steps, 3]
            
        Returns:
            Dictionary of additional invariants
        """
        invariants = {}
        
        # 1. Trajectory length (total path length)
        diff = torch.diff(trajectory, dim=1)  # [batch_size, num_steps-1, 3]
        segment_lengths = torch.norm(diff, dim=2)  # [batch_size, num_steps-1]
        invariants['total_length'] = torch.sum(segment_lengths, dim=1)  # [batch_size]
        
        # 2. Gyration radius (spread of trajectory)
        centroid = torch.mean(trajectory, dim=1, keepdim=True)  # [batch_size, 1, 3]
        deviations = trajectory - centroid  # [batch_size, num_steps, 3]
        gyration_radius_sq = torch.mean(torch.sum(deviations**2, dim=2), dim=1)
        invariants['gyration_radius'] = torch.sqrt(gyration_radius_sq)  # [batch_size]
        
        # 3. Fractal box-counting dimension (simplified)
        # Use range in each dimension as proxy
        ranges = torch.max(trajectory, dim=1)[0] - torch.min(trajectory, dim=1)[0]
        invariants['box_dimension'] = torch.mean(ranges, dim=1)  # [batch_size]
        
        return invariants
    
    def _basic_pooling(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Basic pooling: D2, DL, K only.
        
        Args:
            trajectory: Trajectory tensor [batch_size, num_steps, 3]
            
        Returns:
            Basic features [batch_size, 3]
        """
        # Compute correlation dimension
        D2 = self._compute_correlation_dimension(trajectory)
        
        # Compute Lyapunov exponents
        lyap_exps = self._compute_local_lyapunov_exponents(trajectory)
        
        # Compute Lyapunov dimension and Kolmogorov entropy
        DL = self._compute_lyapunov_dimension(lyap_exps)
        K = self._compute_kolmogorov_entropy(lyap_exps)
        
        return torch.stack([D2, DL, K], dim=1)
    
    def _comprehensive_pooling(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Comprehensive pooling: D2, DL, K + additional invariants.
        
        Args:
            trajectory: Trajectory tensor [batch_size, num_steps, 3]
            
        Returns:
            Comprehensive features [batch_size, 5]
        """
        # Get basic features
        basic_features = self._basic_pooling(trajectory)  # [batch_size, 3]
        
        # Get additional invariants
        additional = self._compute_additional_invariants(trajectory)
        
        # Combine features
        extra_features = torch.stack([
            additional['gyration_radius'],
            additional['box_dimension']
        ], dim=1)  # [batch_size, 2]
        
        return torch.cat([basic_features, extra_features], dim=1)
    
    def _learnable_pooling(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Learnable pooling: Adaptive feature combination.
        
        Args:
            trajectory: Trajectory tensor [batch_size, num_steps, 3]
            
        Returns:
            Learned features [batch_size, 3]
        """
        # Get comprehensive features
        comprehensive_features = self._comprehensive_pooling(trajectory)  # [batch_size, 5]
        
        # Apply learnable weights
        weighted_features = comprehensive_features * self.invariant_weights.unsqueeze(0)
        
        # Combine with learned network
        learned_features = self.feature_combiner(weighted_features)
        
        return learned_features
    
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attractor pooling layer.
        
        Args:
            trajectory: Chaotic trajectory [batch_size, num_steps, 3]
            
        Returns:
            Pooled features [batch_size, output_dim]
        """
        # Choose pooling strategy
        if self.pooling_type == 'basic':
            return self._basic_pooling(trajectory)
        elif self.pooling_type == 'comprehensive':
            return self._comprehensive_pooling(trajectory)
        elif self.pooling_type == 'learnable':
            return self._learnable_pooling(trajectory)
        else:
            return self._basic_pooling(trajectory)
    
    def analyze_trajectory(self, trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Comprehensive analysis of trajectory properties.
        
        Args:
            trajectory: Trajectory tensor [batch_size, num_steps, 3]
            
        Returns:
            Dictionary of analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['mean'] = torch.mean(trajectory, dim=1)  # [batch_size, 3]
        analysis['std'] = torch.std(trajectory, dim=1)    # [batch_size, 3]
        analysis['min'] = torch.min(trajectory, dim=1)[0] # [batch_size, 3]
        analysis['max'] = torch.max(trajectory, dim=1)[0] # [batch_size, 3]
        
        # Topological invariants
        analysis['correlation_dimension'] = self._compute_correlation_dimension(trajectory)
        lyap_exps = self._compute_local_lyapunov_exponents(trajectory)
        analysis['lyapunov_exponents'] = lyap_exps
        analysis['lyapunov_dimension'] = self._compute_lyapunov_dimension(lyap_exps)
        analysis['kolmogorov_entropy'] = self._compute_kolmogorov_entropy(lyap_exps)
        
        # Additional invariants
        additional = self._compute_additional_invariants(trajectory)
        analysis.update(additional)
        
        return analysis
    
    def visualize_correlation_dimension(self, trajectory: torch.Tensor, sample_idx: int = 0):
        """
        Visualize correlation dimension calculation for debugging.
        
        Args:
            trajectory: Trajectory tensor [batch_size, num_steps, 3]
            sample_idx: Which sample to visualize
        """
        try:
            import matplotlib.pyplot as plt
            
            distances = self._compute_pairwise_distances(trajectory)
            log_radii = torch.log(self.correlation_radii + self.eps)
            
            log_correlations = []
            for radius in self.correlation_radii:
                C_r = self._compute_correlation_integral(distances, radius)
                log_C_r = torch.log(C_r[sample_idx] + self.eps)
                log_correlations.append(log_C_r.item())
            
            plt.figure(figsize=(10, 6))
            plt.plot(log_radii.cpu().numpy(), log_correlations, 'o-')
            plt.xlabel('log(r)')
            plt.ylabel('log(C(r))')
            plt.title(f'Correlation Integral - Sample {sample_idx}')
            plt.grid(True)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")


# Factory function for easy instantiation
def create_attractor_pooling(config: dict) -> AttractorPooling:
    """
    Factory function to create attractor pooling layer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AttractorPooling instance
    """
    return AttractorPooling(**config)


if __name__ == "__main__":
    # Test the attractor pooling layer
    print("Testing Attractor Pooling Layer...")
    
    # Create test configuration
    config = {
        'pooling_type': 'comprehensive',
        'num_radii': 15,
        'device': 'cpu'
    }
    
    # Create pooling layer
    pooling = create_attractor_pooling(config)
    
    # Generate test trajectory (simulated chaotic trajectory)
    batch_size = 4
    num_steps = 100
    
    # Create synthetic Lorenz-like trajectory
    t = torch.linspace(0, 10, num_steps)
    x = 10 * torch.sin(t).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
    y = 8 * torch.cos(t).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1) 
    z = 5 * torch.sin(2*t).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
    
    # Add some noise and nonlinearity
    noise = torch.randn(batch_size, num_steps, 3) * 0.5
    test_trajectory = torch.cat([x, y, z], dim=2) + noise
    
    print(f"Input trajectory shape: {test_trajectory.shape}")
    
    # Test forward pass
    with torch.no_grad():
        pooled_features = pooling(test_trajectory)
        analysis = pooling.analyze_trajectory(test_trajectory)
    
    print(f"Output features shape: {pooled_features.shape}")
    print(f"Features sample: {pooled_features[0]}")
    
    print(f"\nTrajectory Analysis:")
    for key, value in analysis.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} - sample: {value[0] if len(value.shape) > 0 else value}")
    
    # Test different pooling types
    for ptype in ['basic', 'comprehensive', 'learnable']:
        config['pooling_type'] = ptype
        test_pooling = create_attractor_pooling(config)
        
        with torch.no_grad():
            features = test_pooling(test_trajectory)
        
        print(f"\nPooling type '{ptype}': output shape {features.shape}")
    
    print("\nAttractor Pooling Layer test completed!")