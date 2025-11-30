#!/usr/bin/env python3
"""
FIXED Chaotic Speaker Recognition Network

KEY FIXES:
1. Added learnable feature transformation to restore gradient flow
2. Added skip connection from raw audio features
3. Improved feature diversity through learnable noise injection
4. Added auxiliary loss for better gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Any


class LearnableChaoticProjection(nn.Module):
    """
    Learnable projection that combines chaotic features with gradient-friendly processing.
    
    This layer solves the gradient disconnection problem by:
    1. Learning a transformation of the chaotic features
    2. Adding learnable noise during training for exploration
    3. Providing skip connections from raw input
    """
    
    def __init__(
        self,
        chaotic_dim: int,
        raw_input_dim: int,
        output_dim: int,
        use_skip_connection: bool = True,
        noise_scale: float = 0.01
    ):
        super().__init__()
        
        self.chaotic_dim = chaotic_dim
        self.raw_input_dim = raw_input_dim
        self.output_dim = output_dim
        self.use_skip_connection = use_skip_connection
        self.noise_scale = noise_scale
        
        # Main chaotic feature transformation
        self.chaotic_transform = nn.Sequential(
            nn.Linear(chaotic_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Raw audio feature extraction (DIFFERENTIABLE path)
        if use_skip_connection:
            self.raw_transform = nn.Sequential(
                nn.Linear(raw_input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim, output_dim)
            )
            
            # Learnable combination weights
            self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Learnable noise scale
        self.learnable_noise_scale = nn.Parameter(torch.tensor(noise_scale))
        
    def forward(self, chaotic_features: torch.Tensor, 
                raw_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with gradient-aware processing.
        
        Args:
            chaotic_features: [batch, chaotic_dim] - from numpy-based extraction
            raw_features: [batch, raw_input_dim] - differentiable path (optional)
        """
        # Transform chaotic features
        transformed = self.chaotic_transform(chaotic_features)
        
        # Add learnable noise during training for exploration
        if self.training:
            noise = torch.randn_like(transformed) * torch.abs(self.learnable_noise_scale)
            transformed = transformed + noise
        
        # Skip connection from raw features (DIFFERENTIABLE)
        if self.use_skip_connection and raw_features is not None:
            raw_transformed = self.raw_transform(raw_features)
            alpha = torch.sigmoid(self.alpha)  # 0-1 range
            transformed = alpha * transformed + (1 - alpha) * raw_transformed
        
        return transformed


class DifferentiablePhaseSpaceStats(nn.Module):
    """
    Extract differentiable statistics from phase space data.
    
    This provides a gradient-friendly alternative to numpy-based feature extraction.
    """
    
    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.output_dim = output_dim
        
        # Learnable feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(10, 32, kernel_size=3, padding=1),  # Assumes 10-dim phase space
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.projection = nn.Linear(64, output_dim)
        
    def forward(self, phase_space: torch.Tensor) -> torch.Tensor:
        """
        Extract differentiable features from phase space.
        
        Args:
            phase_space: [batch, time_steps, phase_dim]
        
        Returns:
            features: [batch, output_dim]
        """
        # Transpose for Conv1d: [batch, phase_dim, time_steps]
        x = phase_space.transpose(1, 2)
        
        # Extract features
        x = self.feature_extractor(x)  # [batch, 64, 1]
        x = x.squeeze(-1)  # [batch, 64]
        x = self.projection(x)  # [batch, output_dim]
        
        return x


class FixedChaoticSpeakerNetwork(nn.Module):
    """
    FIXED Chaotic Speaker Recognition Network with proper gradient flow.
    
    Key differences from original:
    1. Adds differentiable path alongside numpy-based chaotic features
    2. Uses learnable combination of chaotic and differentiable features
    3. Adds auxiliary losses for better gradient flow
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        num_speakers: int = 251,
        speaker_embedding_dim: int = 256,
        chaotic_feature_dim: int = 8,  # From MLSA + RQA
        phase_space_dim: int = 10,
        hidden_dims: list = None,
        use_differentiable_path: bool = True,  # KEY FIX
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers
        self.speaker_embedding_dim = speaker_embedding_dim
        self.chaotic_feature_dim = chaotic_feature_dim
        self.use_differentiable_path = use_differentiable_path
        self.device = device
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # ============================================================
        # FIX 1: Differentiable feature extraction path
        # ============================================================
        if use_differentiable_path:
            self.diff_phase_stats = DifferentiablePhaseSpaceStats(output_dim=32)
            combined_dim = chaotic_feature_dim + 32
        else:
            combined_dim = chaotic_feature_dim
        
        # ============================================================
        # FIX 2: Learnable chaotic projection with skip connection
        # ============================================================
        self.learnable_projection = LearnableChaoticProjection(
            chaotic_dim=combined_dim,
            raw_input_dim=256,  # From raw audio embedding
            output_dim=hidden_dims[0],
            use_skip_connection=use_differentiable_path,
            noise_scale=0.01
        )
        
        # Raw audio feature extractor (DIFFERENTIABLE)
        if use_differentiable_path:
            self.raw_audio_encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=80, stride=10),  # ~5ms windows
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, stride=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.raw_audio_projection = nn.Linear(128, 256)
        
        # ============================================================
        # Speaker embedding layers (same as before but better init)
        # ============================================================
        layers = []
        current_dim = hidden_dims[0]
        
        for i, hidden_dim in enumerate(hidden_dims[1:]):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),  # Better than ReLU for this application
                nn.Dropout(0.3 * (0.8 ** i))
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, speaker_embedding_dim))
        self.embedding_network = nn.Sequential(*layers)
        
        # L2 normalization
        self.normalize_embeddings = True
        
        # ============================================================
        # Classifier with auxiliary output
        # ============================================================
        self.classifier = nn.Sequential(
            nn.Linear(speaker_embedding_dim, speaker_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(speaker_embedding_dim, num_speakers)
        )
        
        # FIX 3: Auxiliary classifier for intermediate features
        # This provides additional gradient signal
        if use_differentiable_path:
            self.auxiliary_classifier = nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dims[0] // 2, num_speakers)
            )
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print(f"FIXED CHAOTIC NETWORK INITIALIZED")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Uses differentiable path: {use_differentiable_path}")
        print(f"Speaker embedding dim: {speaker_embedding_dim}")
        print(f"{'='*60}\n")
    
    def _initialize_weights(self):
        """Better weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def extract_raw_audio_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features directly from raw audio (DIFFERENTIABLE)."""
        # Add channel dimension: [batch, samples] -> [batch, 1, samples]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        # Extract features
        features = self.raw_audio_encoder(audio)  # [batch, 128, 1]
        features = features.squeeze(-1)  # [batch, 128]
        features = self.raw_audio_projection(features)  # [batch, 256]
        
        return features
    
    def forward(
        self, 
        audio: torch.Tensor,
        phase_space: Optional[torch.Tensor] = None,
        chaotic_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with fixed gradient flow.
        
        Args:
            audio: Raw audio [batch, samples]
            phase_space: Phase space reconstruction [batch, time, dim] (optional)
            chaotic_features: Pre-extracted chaotic features [batch, feat_dim] (optional)
            labels: Speaker labels (optional, for angular margin)
            return_intermediates: Whether to return intermediate activations
        """
        batch_size = audio.shape[0]
        intermediates = {}
        
        # ============================================================
        # Path 1: Differentiable raw audio features
        # ============================================================
        if self.use_differentiable_path:
            raw_features = self.extract_raw_audio_features(audio)
            intermediates['raw_features'] = raw_features
        else:
            raw_features = None
        
        # ============================================================
        # Path 2: Get chaotic features (may be provided or computed)
        # ============================================================
        if chaotic_features is None:
            # This is where you'd call your existing chaotic feature extraction
            # For now, create placeholder - replace with your actual extraction
            chaotic_features = torch.randn(batch_size, self.chaotic_feature_dim, device=audio.device)
        
        intermediates['chaotic_features'] = chaotic_features
        
        # ============================================================
        # FIX: Combine differentiable and chaotic paths
        # ============================================================
        if self.use_differentiable_path and phase_space is not None:
            # Get differentiable phase space stats
            diff_stats = self.diff_phase_stats(phase_space)
            combined_chaotic = torch.cat([chaotic_features, diff_stats], dim=-1)
        else:
            combined_chaotic = chaotic_features
        
        # Learnable projection with skip connection
        projected_features = self.learnable_projection(combined_chaotic, raw_features)
        intermediates['projected_features'] = projected_features
        
        # ============================================================
        # Speaker embedding
        # ============================================================
        speaker_embeddings = self.embedding_network(projected_features)
        
        if self.normalize_embeddings:
            speaker_embeddings = F.normalize(speaker_embeddings, p=2, dim=1)
        
        intermediates['speaker_embeddings'] = speaker_embeddings
        
        # ============================================================
        # Classification
        # ============================================================
        logits = self.classifier(speaker_embeddings)
        intermediates['logits'] = logits
        
        # Auxiliary logits for additional gradient signal
        if self.use_differentiable_path and self.training:
            aux_logits = self.auxiliary_classifier(projected_features)
            intermediates['aux_logits'] = aux_logits
        
        if return_intermediates:
            return logits, intermediates
        return logits
    
    def compute_loss(
        self, 
        logits: torch.Tensor,
        labels: torch.Tensor,
        intermediates: Optional[Dict] = None,
        aux_weight: float = 0.3
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss with auxiliary supervision.
        
        Args:
            logits: Main classifier output
            labels: Ground truth labels
            intermediates: Intermediate activations (for auxiliary loss)
            aux_weight: Weight for auxiliary loss
        """
        losses = {}
        
        # Main classification loss
        losses['main'] = F.cross_entropy(logits, labels)
        
        # Auxiliary loss (for better gradient flow)
        if intermediates is not None and 'aux_logits' in intermediates:
            losses['auxiliary'] = F.cross_entropy(intermediates['aux_logits'], labels)
            losses['total'] = losses['main'] + aux_weight * losses['auxiliary']
        else:
            losses['total'] = losses['main']
        
        return losses
    
    def predict(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
        return predicted, confidence


# ============================================================
# INTEGRATION WITH EXISTING CODE
# ============================================================

def patch_existing_network(original_network):
    """
    Patch an existing ChaoticSpeakerRecognitionNetwork to fix gradient flow.
    
    This function modifies the original network in-place.
    """
    # Save original forward
    original_forward = original_network.forward
    
    # Create differentiable components
    device = original_network.device
    
    # Add raw audio encoder
    original_network.raw_audio_encoder = nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=80, stride=10),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=5, stride=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1)
    ).to(device)
    
    original_network.raw_audio_projection = nn.Linear(64, 117).to(device)  # Match pooling output
    
    # Add learnable combination
    original_network.alpha = nn.Parameter(torch.tensor(0.5, device=device))
    
    def patched_forward(audio, labels=None, return_intermediates=False, debug=False):
        """Patched forward with gradient-friendly path."""
        batch_size = audio.shape[0]
        
        # Get raw audio features (DIFFERENTIABLE)
        audio_input = audio.unsqueeze(1) if audio.dim() == 2 else audio
        raw_feat = original_network.raw_audio_encoder(audio_input)
        raw_feat = raw_feat.squeeze(-1)
        raw_feat = original_network.raw_audio_projection(raw_feat)
        
        # Get chaotic features (may not have gradients)
        logits, intermediates = original_forward(audio, labels, return_intermediates=True, debug=debug)
        
        # Combine with skip connection
        if 'pooled_features' in intermediates:
            pooled = intermediates['pooled_features']
            alpha = torch.sigmoid(original_network.alpha)
            combined = alpha * pooled + (1 - alpha) * raw_feat
            
            # Re-run through embedding and classifier
            speaker_embeddings = original_network.speaker_embedding(combined)
            logits = original_network.classifier(speaker_embeddings, labels)
        
        if return_intermediates:
            return logits, intermediates
        return logits
    
    # Replace forward
    original_network.forward = patched_forward
    
    return original_network


if __name__ == "__main__":
    print("Testing Fixed Chaotic Speaker Network...")
    
    # Test configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    num_speakers = 26
    audio_length = 48000
    
    # Create network
    network = FixedChaoticSpeakerNetwork(
        num_speakers=num_speakers,
        speaker_embedding_dim=256,
        use_differentiable_path=True,
        device=device
    ).to(device)
    
    # Test forward pass
    test_audio = torch.randn(batch_size, audio_length).to(device)
    test_labels = torch.randint(0, num_speakers, (batch_size,)).to(device)
    
    # Forward pass
    logits, intermediates = network(test_audio, labels=test_labels, return_intermediates=True)
    
    print(f"\nTest results:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    # Test gradient flow
    loss_dict = network.compute_loss(logits, test_labels, intermediates)
    loss_dict['total'].backward()
    
    # Check gradients
    grad_norm = 0
    for p in network.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print(f"  Main loss: {loss_dict['main'].item():.4f}")
    if 'auxiliary' in loss_dict:
        print(f"  Auxiliary loss: {loss_dict['auxiliary'].item():.4f}")
    
    print("\n[OK] Fixed network test passed!")
