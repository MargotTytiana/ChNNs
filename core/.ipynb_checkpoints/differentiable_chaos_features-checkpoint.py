import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class DifferentiableLyapunovEstimator(nn.Module):
    """
    Differentiable approximation of Lyapunov exponent estimation.
    Uses neural network to learn the mapping from trajectory to Lyapunov features.
    """
    
    def __init__(self, trajectory_dim: int = 3, output_dim: int = 8):
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.output_dim = output_dim
        
        # Temporal convolution for trajectory analysis
        self.conv1d = nn.Sequential(
            nn.Conv1d(trajectory_dim, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Feature extraction
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: [batch, time_steps, dim]
        Returns:
            lyapunov_features: [batch, output_dim]
        """
        # [batch, time_steps, dim] -> [batch, dim, time_steps]
        x = trajectory.transpose(1, 2)
        x = self.conv1d(x)  # [batch, 64, 1]
        x = x.squeeze(-1)   # [batch, 64]
        return self.fc(x)   # [batch, output_dim]


class DifferentiableRQAEstimator(nn.Module):
    """
    Differentiable approximation of RQA features.
    Uses attention mechanism to capture recurrence patterns.
    """
    
    def __init__(self, trajectory_dim: int = 3, output_dim: int = 6):
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.output_dim = output_dim
        
        # Project trajectory points
        self.point_encoder = nn.Linear(trajectory_dim, 32)
        
        # Self-attention for recurrence detection
        self.attention = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, batch_first=True
        )
        
        # RQA feature extraction
        self.rqa_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: [batch, time_steps, dim]
        Returns:
            rqa_features: [batch, output_dim]
        """
        # Encode trajectory points
        x = self.point_encoder(trajectory)  # [batch, T, 32]
        
        # Self-attention to capture recurrence
        attn_out, attn_weights = self.attention(x, x, x)  # [batch, T, 32]
        
        # Global pooling
        x = attn_out.mean(dim=1)  # [batch, 32]
        
        return self.rqa_head(x)  # [batch, output_dim]


class DifferentiableChaoticFeatures(nn.Module):
    """
    Fully differentiable chaotic feature extraction module.
    Replaces NumPy-based MLSA and RQA extractors.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        lyapunov_dim: int = 8,
        rqa_dim: int = 6,
        additional_stats: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.additional_stats = additional_stats
        
        # Lyapunov feature estimator
        self.lyapunov_estimator = DifferentiableLyapunovEstimator(
            trajectory_dim=input_dim, output_dim=lyapunov_dim
        )
        
        # RQA feature estimator
        self.rqa_estimator = DifferentiableRQAEstimator(
            trajectory_dim=input_dim, output_dim=rqa_dim
        )
        
        # Calculate output dimension
        self.output_dim = lyapunov_dim + rqa_dim
        if additional_stats:
            self.output_dim += input_dim * 4  # mean, std, min, max
        
    def forward(self, phase_space: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phase_space: [batch, time_steps, dim] or [batch, dim, time_steps]
        Returns:
            chaotic_features: [batch, output_dim]
        """
        # Ensure correct shape [batch, time, dim]
        if phase_space.shape[1] < phase_space.shape[2]:
            phase_space = phase_space.transpose(1, 2)
        
        features = []
        
        # Lyapunov features
        lyap_feat = self.lyapunov_estimator(phase_space)
        features.append(lyap_feat)
        
        # RQA features
        rqa_feat = self.rqa_estimator(phase_space)
        features.append(rqa_feat)
        
        # Additional statistics (fully differentiable)
        if self.additional_stats:
            stats_mean = phase_space.mean(dim=1)
            stats_std = phase_space.std(dim=1)
            stats_min = phase_space.min(dim=1)[0]
            stats_max = phase_space.max(dim=1)[0]
            features.extend([stats_mean, stats_std, stats_min, stats_max])
        
        return torch.cat(features, dim=1)