# chaotic_network.py - FIXED VERSION WITH ENHANCED CAPACITY
"""
Enhanced Chaotic Speaker Recognition Network
CRITICAL FIX: Increased model capacity for 251 speakers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any, List
import logging
import os
import sys
from pathlib import Path
import numpy as np

def fix_imports():
    current_file = Path(__file__).resolve()
    model_dir = current_file.parent.parent
    paths = [
        str(model_dir),
        str(model_dir/'experiments'), 
        str(model_dir/'models'),
        str(model_dir/'features'),
        str(model_dir/'data'),
        str(model_dir/'utils')
    ]
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    return model_dir

MODEL_DIR = fix_imports()

from core.phase_space_reconstruction import PhaseSpaceReconstructor
from core.mlsa_extractor import MLSAExtractor
from core.rqa_extractor import RQAExtractor
from core.chaotic_embedding import ChaoticEmbedding
from core.attractor_pooling import AttractorPooling

class MockComponent(nn.Module):
    """Mock component for testing when core modules are not available."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = None
    
    def forward(self, x):
        if self.linear is None:
            if len(x.shape) == 2:
                actual_input_dim = x.shape[-1]
            elif len(x.shape) == 3:
                actual_input_dim = x.shape[1] * x.shape[2]
                x = x.view(x.shape[0], -1)
            else:
                actual_input_dim = x.shape[-1]
            
            self.linear = nn.Linear(actual_input_dim, self.output_dim).to(x.device)
        
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
            
        return self.linear(x)


class EnhancedSpeakerEmbedding(nn.Module):
    """
    ENHANCED Speaker Embedding Layer with MUCH larger capacity.
    
    CRITICAL FIX: Increased hidden dimensions for 251 speakers.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        embedding_dim: int = 256,  # INCREASED from 128
        hidden_dims: list = [512, 256, 128],  # INCREASED from [64, 32]
        dropout_rate: float = 0.3,  # INCREASED dropout
        activation: str = 'relu'
    ):
        super(EnhancedSpeakerEmbedding, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # Build ENHANCED embedding network
        layers = []
        current_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate * (0.8 ** i))  # Progressive dropout
            ])
            current_dim = hidden_dim
        
        # Final embedding layer
        layers.append(nn.Linear(current_dim, embedding_dim))
        
        self.embedding_network = nn.Sequential(*layers)
        
        # L2 normalization for embedding vectors
        self.normalize = True
        
        # Weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Better weight initialization for deeper network"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_network(features)
        
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings


class EnhancedChaoticClassifier(nn.Module):
    """
    ENHANCED classifier with better architecture for 251 speakers.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,  # INCREASED from 128
        num_speakers: int = 251,
        classifier_type: str = 'linear',
        temperature: float = 30.0,
        margin: float = 0.35
    ):
        super(EnhancedChaoticClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.classifier_type = classifier_type
        self.temperature = temperature
        self.margin = margin
        
        if classifier_type == 'linear':
            # Add intermediate layer for better capacity
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(embedding_dim, num_speakers)
            )
        elif classifier_type in ['cosine', 'angular']:
            # Learnable speaker prototypes
            self.weight = nn.Parameter(torch.randn(num_speakers, embedding_dim))
            nn.init.xavier_normal_(self.weight)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.classifier_type == 'linear':
            return self.classifier(embeddings)
        
        elif self.classifier_type == 'cosine':
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            normalized_weight = F.normalize(self.weight, p=2, dim=1)
            
            cosine_sim = F.linear(normalized_embeddings, normalized_weight)
            logits = cosine_sim * self.temperature
            
            return logits
        
        elif self.classifier_type == 'angular':
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            normalized_weight = F.normalize(self.weight, p=2, dim=1)
            
            cosine_sim = F.linear(normalized_embeddings, normalized_weight)
            cosine_sim = torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7)
            
            if self.training and labels is not None:
                theta = torch.acos(cosine_sim)
                target_logits = torch.cos(theta + self.margin)
                
                one_hot = torch.zeros_like(cosine_sim)
                one_hot.scatter_(1, labels.view(-1, 1), 1)
                
                logits = (one_hot * target_logits) + ((1.0 - one_hot) * cosine_sim)
            else:
                logits = cosine_sim
                
            logits = logits * self.temperature
            return logits


class ChaoticSpeakerRecognitionNetwork(nn.Module):
    """
    ENHANCED Chaotic Speaker Recognition Network with MUCH larger capacity.
    
    CRITICAL CHANGES:
    1. Increased speaker embedding dimensions: 128 -> 256
    2. Deeper speaker embedding network: [64,32] -> [512,256,128]
    3. Better dropout strategy
    4. Improved weight initialization
    
    Expected parameters: ~300K+ (vs original 46K)
    """
        
    def __init__(
        self,
        # Audio processing parameters
        sample_rate: int = 16000,
        frame_length: int = 400,
        hop_length: int = 160,
        
        # Phase space reconstruction parameters
        embedding_dim: int = 10,
        delay_method: str = 'autocorr',
        
        # Chaotic feature parameters
        mlsa_scales: int = 5,
        rqa_radius_ratio: float = 0.1,
        
        # Chaotic embedding parameters
        chaotic_system: str = 'lorenz',
        evolution_time: float = 0.5,
        time_step: float = 0.01,
        
        # Attractor pooling parameters
        pooling_type: str = 'comprehensive',
        
        # ENHANCED Speaker embedding parameters
        speaker_embedding_dim: int = 256,  # INCREASED from 128
        embedding_hidden_dims: list = None,  # Will default to [512, 256, 128]
        
        # Classification parameters
        num_speakers: int = 251,  # Default to actual dataset size
        classifier_type: str = 'linear',
        
        # Optional modules
        use_bifurcation_control: bool = False,  # NEW: Enable bifurcation control
        
        # Device
        device: str = 'cpu',

        fixed_delay: int = 10
    ):
        super(ChaoticSpeakerRecognitionNetwork, self).__init__()
        
        # Save all parameters
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.embedding_dim = embedding_dim
        self.delay_method = delay_method
        self.fixed_delay = fixed_delay
        self.mlsa_scales = mlsa_scales
        self.rqa_radius_ratio = rqa_radius_ratio
        self.chaotic_system = chaotic_system
        self.evolution_time = evolution_time
        self.time_step = time_step
        self.pooling_type = pooling_type
        
        # ENHANCED embedding parameters
        self.speaker_embedding_dim = speaker_embedding_dim
        if embedding_hidden_dims is None:
            self.embedding_hidden_dims = [512, 256, 128]  # MUCH larger
        else:
            self.embedding_hidden_dims = embedding_hidden_dims
        
        self.num_speakers = num_speakers
        self.classifier_type = classifier_type
        self.device = device
        self.use_bifurcation_control = use_bifurcation_control  # NEW
        
        # Initialize components
        self._initialize_components()

        # ========== GRADIENT FIX: Add differentiable raw audio path ==========
        self.raw_audio_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=80, stride=10),
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
        
        # Project to match pooled feature dimension (117 for comprehensive pooling)
        self.raw_audio_projection = nn.Linear(128, 117)
        
        # Learnable mixing weight for chaotic vs differentiable path
        self.mix_alpha = nn.Parameter(torch.tensor(0.3))
        
        # Move to device
        self.raw_audio_encoder = self.raw_audio_encoder.to(self.device)
        self.raw_audio_projection = self.raw_audio_projection.to(self.device)
        
        print("[GRADIENT FIX] Added differentiable raw audio path")

        # Loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Print model capacity
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print(f"ENHANCED CHAOTIC NETWORK INITIALIZED")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Parameters per speaker: {total_params/num_speakers:.0f}")
        print(f"Speaker embedding dim: {speaker_embedding_dim}")
        print(f"Hidden dims: {self.embedding_hidden_dims}")
        print(f"{'='*60}\n")

    def _initialize_components(self):
        """Initialize all network components with ENHANCED capacity."""
        
        # Phase space reconstruction - FIX CONFIGURATION
        from core.phase_space_reconstruction import EmbeddingConfig
        
        phase_space_config = EmbeddingConfig()
        phase_space_config.max_dimension = self.embedding_dim
        
        # Get delay and dimension methods from config (with defaults)
        self.delay_method = getattr(self, 'delay_method', 'autocorr')
        self.dimension_method = getattr(self, 'dimension_method', 'false_neighbors')
        
        print(f"[CONFIG DEBUG] Phase space: dim={self.embedding_dim}, "
              f"delay_method={self.delay_method}, dimension_method={self.dimension_method}")
        
        base_reconstructor = PhaseSpaceReconstructor(config=phase_space_config)
        
        self.phase_space = BatchPhaseSpaceReconstructor(
            reconstructor=base_reconstructor,
            fixed_output_length=200,
            device=self.device,
            delay_method=self.delay_method,
            dimension_method=self.dimension_method,
            fixed_delay=self.fixed_delay
        )
        
        # MLSA extractor
        if MLSAExtractor is not None:
            try:
                from core.mlsa_extractor import MLSAConfig
                mlsa_config = MLSAConfig(
                    n_scales=self.mlsa_scales,
                    scale_factors=[1, 2, 4, 8, 16][:self.mlsa_scales],
                    decomposition_method='fourier',
                    min_segment_length=100
                )
                self.mlsa_extractor = MLSAExtractor(config=mlsa_config)
            except Exception as e:
                logging.warning(f"Failed to create MLSAExtractor: {e}")
                self.mlsa_extractor = MockComponent(None, self.mlsa_scales)
        else:
            self.mlsa_extractor = MockComponent(None, self.mlsa_scales)
            
        # RQA extractor
        if RQAExtractor is not None:
            try:
                from core.rqa_extractor import RQAConfig
                rqa_config = RQAConfig(
                    threshold_method='fixed_amount',
                    recurrence_rate_target=self.rqa_radius_ratio,
                    scale_factors=[1, 2, 4],
                    min_diagonal_length=2,
                    min_vertical_length=2
                )
                self.rqa_extractor = RQAExtractor(config=rqa_config)
            except Exception as e:
                logging.warning(f"Failed to create RQAExtractor: {e}")
                self.rqa_extractor = MockComponent(None, 3)
        else:
            self.rqa_extractor = MockComponent(None, 3)

        # Determine chaotic feature dimension
        try:
            dummy_signal = np.random.randn(1000)
            
            try:
                mlsa_result = self.mlsa_extractor.extract_features(dummy_signal)
                if mlsa_result['success'] and 'feature_vector' in mlsa_result:
                    mlsa_feature_dim = len(mlsa_result['feature_vector'])
                    self.mlsa_feature_dim = mlsa_feature_dim
                else:
                    mlsa_feature_dim = self.mlsa_scales
                    self.mlsa_feature_dim = mlsa_feature_dim
            except:
                mlsa_feature_dim = self.mlsa_scales
                self.mlsa_feature_dim = mlsa_feature_dim
            
            try:
                rqa_result = self.rqa_extractor.extract_features(dummy_signal)
                if rqa_result['success'] and 'feature_vector' in rqa_result:
                    rqa_feature_dim = len(rqa_result['feature_vector'])
                    self.rqa_feature_dim = rqa_feature_dim
                else:
                    rqa_feature_dim = 3
                    self.rqa_feature_dim = rqa_feature_dim
            except:
                rqa_feature_dim = 3
                self.rqa_feature_dim = rqa_feature_dim
            
            chaotic_feature_dim = mlsa_feature_dim + rqa_feature_dim
            print(f"Chaotic feature dimensions: MLSA={mlsa_feature_dim}, RQA={rqa_feature_dim}, Total={chaotic_feature_dim}")
            
        except Exception as e:
            mlsa_feature_dim = self.mlsa_scales
            rqa_feature_dim = 3
            chaotic_feature_dim = mlsa_feature_dim + rqa_feature_dim
            self.mlsa_feature_dim = mlsa_feature_dim
            self.rqa_feature_dim = rqa_feature_dim
        
        # Chaotic embedding layer
        if ChaoticEmbedding is not None:
            # FIX: Use self.use_bifurcation_control directly (already set from __init__ parameter)
            # Previous bug: was re-reading from self.config which doesn't exist
            self.chaotic_embedding = ChaoticEmbedding(
                input_dim=chaotic_feature_dim,
                system_type=self.chaotic_system,
                evolution_time=self.evolution_time,
                time_step=self.time_step,
                device=self.device,
                use_bifurcation_control=self.use_bifurcation_control  # FIX: use instance variable
            )
            print(f"[NETWORK DEBUG] ChaoticEmbedding created with:")
            print(f"  use_bifurcation_control={self.use_bifurcation_control}")
            _bif_exists = hasattr(self.chaotic_embedding, 'bifurcation_net') and self.chaotic_embedding.bifurcation_net is not None
            print(f"  bifurcation_net exists: {_bif_exists}")
            if _bif_exists:
                print(f"  bifurcation_net: {self.chaotic_embedding.bifurcation_net}")
        else:
            trajectory_dim = int(self.evolution_time / self.time_step) * 3
            self.chaotic_embedding = MockComponent(None, trajectory_dim)
            
        # Attractor pooling
        if AttractorPooling is not None:
            self.attractor_pooling = AttractorPooling(
                pooling_type=self.pooling_type,
                device=self.device
            )
            pooling_output_dim = 117 if self.pooling_type == 'comprehensive' else 3
        else:
            pooling_output_dim = 5
            self.attractor_pooling = MockComponent(None, pooling_output_dim)
            
        # ENHANCED Speaker embedding
        self.speaker_embedding = EnhancedSpeakerEmbedding(
            input_dim=pooling_output_dim,
            embedding_dim=self.speaker_embedding_dim,
            hidden_dims=self.embedding_hidden_dims,
            dropout_rate=0.3
        )
        
        # ENHANCED Final classifier
        self.classifier = EnhancedChaoticClassifier(
            embedding_dim=self.speaker_embedding_dim,
            num_speakers=self.num_speakers,
            classifier_type=self.classifier_type
        )
        # CRITICAL: Add learnable feature projection to restore gradient flow
        chaotic_feature_dim = self.mlsa_feature_dim + self.rqa_feature_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(chaotic_feature_dim, chaotic_feature_dim),
            nn.LayerNorm(chaotic_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(chaotic_feature_dim, chaotic_feature_dim)
        )
        
        # ============ DIFFERENTIABLE CHAOTIC FEATURES ============
        # Fully differentiable replacement for NumPy-based MLSA/RQA
        try:
            # Try core subdirectory first, then root directory
            try:
                from core.differentiable_chaos_features import DifferentiableChaoticFeatures
            except ImportError:
                from differentiable_chaos_features import DifferentiableChaoticFeatures
            
            self.diff_chaos_features = DifferentiableChaoticFeatures(
                input_dim=self.embedding_dim,  # phase space dimension
                lyapunov_dim=self.mlsa_feature_dim,
                rqa_dim=self.rqa_feature_dim,
                additional_stats=True
            )
            self.use_differentiable_features = True
            
            # === FIX: Pre-register projection layer in __init__ ===
            diff_output_dim = self.diff_chaos_features.output_dim
            expected_dim = self.mlsa_feature_dim + self.rqa_feature_dim
            
            if diff_output_dim != expected_dim:
                self.diff_feat_projection = nn.Linear(diff_output_dim, expected_dim)
                print(f"[INIT] Registered diff_feat_projection: {diff_output_dim} -> {expected_dim}")
            else:
                self.diff_feat_projection = nn.Identity()
            
            print(f"[GRADIENT FIX] Differentiable chaos features enabled: output_dim={self.diff_chaos_features.output_dim}")
        except ImportError as e:
            print(f"[WARNING] DifferentiableChaoticFeatures not available: {e}")
            self.use_differentiable_features = False
            self.diff_feat_projection = None
        
    def extract_chaotic_features(self, phase_space_data: torch.Tensor) -> torch.Tensor:
        """Extract chaotic features using MLSA and RQA with gradient preservation."""
        batch_size = phase_space_data.shape[0]
        device = phase_space_data.device
        all_features = []
        success_count = 0
    
        mlsa_dim = self.mlsa_feature_dim if hasattr(self, 'mlsa_feature_dim') else self.mlsa_scales
        rqa_dim = self.rqa_feature_dim if hasattr(self, 'rqa_feature_dim') else 3
    
        for i in range(batch_size):
            sample = phase_space_data[i].cpu().detach().numpy()
    
            if sample.ndim == 2:
                signal_1d = sample[:, 0].copy()
            else:
                signal_1d = sample.copy()
    
            # Ensure signal has sufficient variance
            signal_std = np.std(signal_1d)
            if signal_std < 1e-8:
                signal_1d = signal_1d + np.random.randn(len(signal_1d)) * 1e-6
    
            # Extract MLSA features
            try:
                mlsa_result = self.mlsa_extractor.extract_features(signal_1d)
                if mlsa_result.get('success', False):
                    success_count += 1
                    mlsa_vec = np.array(mlsa_result['feature_vector'], dtype=np.float32).flatten()
                else:
                    mlsa_vec = np.random.randn(mlsa_dim).astype(np.float32) * 0.01
            except:
                mlsa_vec = np.random.randn(mlsa_dim).astype(np.float32) * 0.01
    
            # Extract RQA features
            try:
                rqa_result = self.rqa_extractor.extract_features(signal_1d)
                if rqa_result.get('success', False) and 'feature_vector' in rqa_result:
                    rqa_vec = np.array(rqa_result['feature_vector'], dtype=np.float32).flatten()
                    if len(rqa_vec) < rqa_dim:
                        rqa_vec = np.pad(rqa_vec, (0, rqa_dim - len(rqa_vec)), mode='constant')
                    elif len(rqa_vec) > rqa_dim:
                        rqa_vec = rqa_vec[:rqa_dim]
                else:
                    rqa_vec = np.random.randn(rqa_dim).astype(np.float32) * 0.01
            except:
                rqa_vec = np.random.randn(rqa_dim).astype(np.float32) * 0.01
    
            combined = np.concatenate([mlsa_vec, rqa_vec])
            combined = np.nan_to_num(combined, nan=0.0, posinf=1e3, neginf=-1e3)
            combined = np.clip(combined, -1e3, 1e3)
    
            all_features.append(combined)
    
        features_np = np.stack(all_features, axis=0)
        chaotic_features = torch.from_numpy(features_np).float().to(device)
    
        # ========== GRADIENT FIX: Add learnable noise during training ==========
        if self.training:
            # Small noise helps with gradient exploration
            noise = torch.randn_like(chaotic_features) * 0.01
            chaotic_features = chaotic_features + noise
    
        return chaotic_features

    def forward(self, audio, labels=None, return_intermediates=False, debug=False):
        """Forward pass with FIXED gradient flow."""
    
        batch_size = audio.shape[0]
        device = audio.device

        # --- 强力补丁：强制对齐子模块设备 ---
        if hasattr(self, 'diff_feat_projection') and self.diff_feat_projection is not None:
            self.diff_feat_projection.to(device) # 强制将投影层移动到 cuda:0
            
        if hasattr(self, 'diff_chaos_features'):
            self.diff_chaos_features.to(device) # 确保微分特征提取器也在 cuda:0
            
        if hasattr(self, 'raw_audio_encoder'):
            self.raw_audio_encoder.to(device)
            self.raw_audio_projection.to(device)
        # ----------------------------------
        
        if debug:
            print("\n" + "="*60)
            print("FIXED FORWARD PASS - GRADIENT AWARE")
            print("="*60)
            print(f"Audio shape: {audio.shape}")
    
        # ========== GRADIENT FIX: Extract differentiable raw audio features ==========
        if hasattr(self, 'raw_audio_encoder'):
            audio_input = audio.unsqueeze(1) if audio.dim() == 2 else audio
            raw_feat = self.raw_audio_encoder(audio_input)
            raw_feat = raw_feat.squeeze(-1)
            raw_feat = self.raw_audio_projection(raw_feat)
    
            if debug:
                print(f"Raw audio features: shape={raw_feat.shape}, "
                      f"range=[{raw_feat.min():.4f}, {raw_feat.max():.4f}]")
        else:
            raw_feat = None
    
        # Phase space reconstruction
        phase_space_data = self.phase_space(audio)
    
        if debug:
            print(f"Phase space: shape={phase_space_data.shape}")
    
        # ============ CHAOTIC FEATURES WITH GRADIENT FLOW ============
        if hasattr(self, 'use_differentiable_features') and self.use_differentiable_features:
            # Use differentiable chaos features (GRADIENT FLOWS!)
            chaotic_features = self.diff_chaos_features(phase_space_data)
            
            # Use pre-registered projection layer
            if self.diff_feat_projection is not None:
                chaotic_features = self.diff_feat_projection(chaotic_features)
            
            if debug:
                print(f"[DIFF] Chaotic features: shape={chaotic_features.shape}, "
                      f"requires_grad={chaotic_features.requires_grad}")
        else:
            # Fallback: NumPy-based extraction (no gradients)
            chaotic_features = self.extract_chaotic_features(phase_space_data)
            chaotic_features = self.feature_projection(chaotic_features)
    
        if debug:
            print(f"Chaotic features (projected): shape={chaotic_features.shape}")
    
        # Chaotic trajectories
        if hasattr(self.chaotic_embedding, 'forward'):
            chaotic_trajectories = self.chaotic_embedding(chaotic_features)
        else:
            chaotic_trajectories = self.chaotic_embedding(chaotic_features.view(batch_size, -1))
            trajectory_length = chaotic_trajectories.shape[1] // 3
            chaotic_trajectories = chaotic_trajectories.view(batch_size, trajectory_length, 3)
    
        # Attractor pooling
        pooled_features = self.attractor_pooling(chaotic_trajectories)
    
        if debug:
            print(f"Pooled features: shape={pooled_features.shape}")
    
        # ========== GRADIENT FIX: Combine chaotic and differentiable paths ==========
        if raw_feat is not None:
            # 获取输入音频当前的设备 (应当是 cuda:0)
            target_device = audio.device
            
            # 强制确保融合参数 alpha 在正确的设备上
            self.mix_alpha.data = self.mix_alpha.data.to(target_device)
            
            # 确保 raw_feat 在正确的设备上
            raw_feat = raw_feat.to(target_device)

            # Ensure dimensions match
            if raw_feat.shape[-1] != pooled_features.shape[-1]:
                if not hasattr(self, '_raw_feat_adjust'):
                    self._raw_feat_adjust = nn.Linear(
                        raw_feat.shape[-1], pooled_features.shape[-1]
                    ).to(target_device) # 显式移动
                
                # 再次确认调整层也在正确的设备
                self._raw_feat_adjust.to(target_device)
                raw_feat = self._raw_feat_adjust(raw_feat)
    
            # Learnable combination
            alpha = torch.sigmoid(self.mix_alpha)
            # 此时 pooled_features, alpha, raw_feat 均在 target_device 上
            pooled_features = alpha * pooled_features + (1 - alpha) * raw_feat

            
    
            if debug:
                print(f"Mixed features: alpha={alpha.item():.4f}")
    
        # Speaker embeddings
        speaker_embeddings = self.speaker_embedding(pooled_features)
    
        if debug:
            print(f"Speaker embeddings: shape={speaker_embeddings.shape}, "
                  f"norm={speaker_embeddings.norm(dim=1).mean():.4f}")
    
        # Classification
        logits = self.classifier(speaker_embeddings, labels)
    
        if debug:
            probs = F.softmax(logits, dim=1)
            print(f"Logits: shape={logits.shape}, "
                  f"max_prob={probs.max(dim=1)[0].mean():.4f}")
            print("="*60 + "\n")
    
        if return_intermediates:
            intermediates = {
                'phase_space': phase_space_data,
                'chaotic_features': chaotic_features,
                'chaotic_trajectories': chaotic_trajectories,
                'pooled_features': pooled_features,
                'speaker_embeddings': speaker_embeddings,
                'logits': logits,
                'raw_features': raw_feat
            }
            return logits, intermediates
    
        return logits

    def compute_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss."""
        losses = {}
        
        losses['classification'] = self.cross_entropy_loss(logits, labels)
        
        if loss_weights:
            total_loss = losses['classification'] * loss_weights.get('classification', 1.0)
            
            if 'l2_reg' in loss_weights:
                l2_reg = 0.0
                for param in self.parameters():
                    l2_reg += torch.norm(param, p=2)
                losses['l2_reg'] = l2_reg
                total_loss += l2_reg * loss_weights['l2_reg']
            
            losses['total'] = total_loss
        else:
            losses['total'] = losses['classification']
        
        return losses
    
    def predict(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions on input audio."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio)
            probabilities = F.softmax(logits, dim=1)
            confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
        
        return predicted_classes, confidence_scores
    
    def extract_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embeddings."""
        self.eval()
        with torch.no_grad():
            _, intermediates = self.forward(audio, return_intermediates=True)
        
        return intermediates['speaker_embeddings']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'parameters_per_speaker': total_params / self.num_speakers,
            'speaker_embedding_dim': self.speaker_embedding_dim,
            'hidden_dims': self.embedding_hidden_dims,
            'num_speakers': self.num_speakers
        }
        
        return info
    
    def save_checkpoint(self, filepath: str, additional_info: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath: str, strict: bool = True) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        return checkpoint


class BatchPhaseSpaceReconstructor(nn.Module):
    """Batch-enabled wrapper for PhaseSpaceReconstructor."""
    
    def __init__(self, reconstructor: PhaseSpaceReconstructor, 
                 fixed_output_length: int = 200,
                 device: str = 'cpu',
                 delay_method: str = 'autocorr',
                 dimension_method: str = 'false_neighbors',
                 fixed_delay: int = None):
        super().__init__()
        self.reconstructor = reconstructor
        self.delay_method = delay_method
        self.dimension_method = dimension_method
        self.fixed_delay = fixed_delay if fixed_delay is not None else 10
        self.fixed_output_length = fixed_output_length
        self.device = device
        
    def forward(self, audio_batch: torch.Tensor) -> torch.Tensor:
            """Process a batch of audio signals."""
            batch_size = audio_batch.shape[0]
            results = []
            
            for i in range(batch_size):
                embedded_tensor = self._create_fallback_embedding(audio_batch[i])
                results.append(embedded_tensor)
            
            standardized = self._standardize_length(results)
        
            return standardized.to(audio_batch.device)
    
    def _create_fallback_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Create simple fallback phase space embedding with CORRECT dimension."""
        delay = getattr(self, 'fixed_delay', None) or 10
        dim = getattr(self.reconstructor.config, 'max_dimension', 10)
        
        audio_np = audio.cpu().numpy()
        n_points = len(audio_np) - (dim - 1) * delay
        
        if n_points <= 0:
            # If audio too short, return zeros with correct dimension
            return torch.zeros(self.fixed_output_length, dim)
        
        embedded = np.zeros((n_points, dim))
        for i in range(dim):
            start_idx = i * delay
            end_idx = start_idx + n_points
            if end_idx <= len(audio_np):
                embedded[:, i] = audio_np[start_idx:end_idx]
            else:
                # Pad with zeros if needed
                available = len(audio_np) - start_idx
                if available > 0:
                    embedded[:available, i] = audio_np[start_idx:]
                # Remainder stays zero
        
        return torch.from_numpy(embedded).float()
    
    def _standardize_length(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Standardize all embeddings to same length."""
        max_len = max(emb.shape[0] for emb in embeddings)
        target_len = min(max_len, self.fixed_output_length)
        
        standardized = []
        for emb in embeddings:
            if emb.shape[0] > target_len:
                emb = emb[:target_len]
            elif emb.shape[0] < target_len:
                padding = torch.zeros(target_len - emb.shape[0], emb.shape[1])
                emb = torch.cat([emb, padding], dim=0)
            
            standardized.append(emb)
        
        return torch.stack(standardized)


# Factory function for easy model creation
def create_chaotic_speaker_network(config: Dict) -> ChaoticSpeakerRecognitionNetwork:
    """
    Factory function to create ENHANCED chaotic speaker recognition network.
    
    CRITICAL: This version creates the enhanced model with larger capacity.
    """
    return ChaoticSpeakerRecognitionNetwork(**config)

SpeakerEmbedding = EnhancedSpeakerEmbedding
ChaoticClassifier = EnhancedChaoticClassifier


# ============================================================
# ADVERSARIAL CHAOTIC AUGMENTATION
# ============================================================

class AdversarialChaoticAugmentation(nn.Module):
    """
    Adversarial Chaos Injection for robust training.
    
    Adds controlled perturbations to chaotic system parameters
    or trajectories during training to improve model robustness.
    """
    
    def __init__(
        self,
        perturbation_scale: float = 0.1,
        parameter_noise: bool = True,
        trajectory_noise: bool = True,
        adversarial_steps: int = 1
    ):
        super().__init__()
        self.perturbation_scale = perturbation_scale
        self.parameter_noise = parameter_noise
        self.trajectory_noise = trajectory_noise
        self.adversarial_steps = adversarial_steps
        
    def perturb_parameters(
        self, 
        params: torch.Tensor,
        grad_direction: torch.Tensor = None
    ) -> torch.Tensor:
        """Add adversarial perturbation to chaotic system parameters."""
        if grad_direction is not None:
            perturbation = self.perturbation_scale * torch.sign(grad_direction)
        else:
            perturbation = self.perturbation_scale * torch.randn_like(params)
        return params + perturbation
    
    def perturb_trajectory(
        self, 
        trajectory: torch.Tensor,
        epsilon: float = None
    ) -> torch.Tensor:
        """Add noise to trajectory for robustness."""
        eps = epsilon if epsilon else self.perturbation_scale * 0.1
        noise = eps * torch.randn_like(trajectory)
        return trajectory + noise
    
    def forward(
        self,
        chaotic_embedding: nn.Module = None,
        features: torch.Tensor = None,
        original_trajectory: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate adversarially perturbed trajectories.
        
        Returns:
            Tuple of (clean_trajectory, perturbed_trajectory)
        """
        if not self.training:
            return original_trajectory, original_trajectory
        
        if original_trajectory is None and chaotic_embedding is not None and features is not None:
            with torch.no_grad():
                clean_trajectory = chaotic_embedding(features)
        else:
            clean_trajectory = original_trajectory
        
        if clean_trajectory is None:
            raise ValueError("Either original_trajectory or (chaotic_embedding + features) must be provided")
        
        perturbed = clean_trajectory.clone()
        
        if self.trajectory_noise:
            perturbed = self.perturb_trajectory(perturbed)
        
        return clean_trajectory, perturbed


# ============================================================
# LYAPUNOV STABILITY REGULARIZATION
# ============================================================

class LyapunovStabilityLoss(nn.Module):
    """
    Lyapunov Stability Regularization Loss.
    
    Encourages chaotic trajectories to remain in a bounded, stable
    attractor region while maintaining positive Lyapunov exponents
    (i.e., staying chaotic but not exploding or collapsing).
    
    This addresses common training issues:
    - Trajectory explosion (values -> infinity)
    - Trajectory collapse (values -> constant)
    - Loss of chaotic dynamics
    """
    
    def __init__(
        self,
        target_lyapunov_range: Tuple[float, float] = (0.1, 2.0),
        trajectory_bound: float = 50.0,
        stability_weight: float = 0.1,
        diversity_weight: float = 0.05,
        collapse_threshold: float = 0.1
    ):
        """
        Args:
            target_lyapunov_range: (min, max) acceptable Lyapunov exponent range
            trajectory_bound: Maximum allowed trajectory norm
            stability_weight: Weight for stability losses
            diversity_weight: Weight for embedding diversity loss
            collapse_threshold: Minimum trajectory std to prevent collapse
        """
        super().__init__()
        self.target_lyap_min = target_lyapunov_range[0]
        self.target_lyap_max = target_lyapunov_range[1]
        self.trajectory_bound = trajectory_bound
        self.stability_weight = stability_weight
        self.diversity_weight = diversity_weight
        self.collapse_threshold = collapse_threshold
    
    def estimate_lyapunov(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Estimate local Lyapunov exponent from trajectory divergence.
        
        Args:
            trajectories: [batch, time_steps, dim]
        Returns:
            lyapunov_estimates: [batch]
        """
        # Compute local divergence rate from trajectory differences
        diffs = trajectories[:, 1:, :] - trajectories[:, :-1, :]  # [batch, T-1, dim]
        diff_norms = torch.norm(diffs, dim=2) + 1e-8  # [batch, T-1]
        
        # Log of expansion rate (proxy for local Lyapunov)
        # Avoid division by zero
        ratios = diff_norms[:, 1:] / diff_norms[:, :-1].clamp(min=1e-8)
        log_expansion = torch.log(ratios.clamp(min=1e-8))
        
        # Mean Lyapunov estimate per sample
        mean_lyapunov = log_expansion.mean(dim=1)  # [batch]
        
        return mean_lyapunov
    
    def forward(
        self, 
        trajectories: torch.Tensor,
        embeddings: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute stability regularization losses.
        
        Args:
            trajectories: Chaotic trajectories [batch, time_steps, dim]
            embeddings: Speaker embeddings [batch, embed_dim] (optional)
            
        Returns:
            Dictionary containing individual losses and total
        """
        losses = {}
        batch_size, T, dim = trajectories.shape
        device = trajectories.device
        
        # 1. Boundedness Loss: Penalize trajectories that explode
        trajectory_norms = torch.norm(trajectories, dim=2)  # [batch, T]
        max_norms = trajectory_norms.max(dim=1)[0]  # [batch]
        boundedness_loss = F.relu(max_norms - self.trajectory_bound).mean()
        losses['boundedness'] = boundedness_loss
        
        # 2. Lyapunov Range Loss: Keep Lyapunov in target range
        mean_lyapunov = self.estimate_lyapunov(trajectories)
        
        # Penalize if Lyapunov is too small (not chaotic) or too large (unstable)
        lyap_too_small = F.relu(self.target_lyap_min - mean_lyapunov)
        lyap_too_large = F.relu(mean_lyapunov - self.target_lyap_max)
        lyapunov_loss = (lyap_too_small + lyap_too_large).mean()
        losses['lyapunov_range'] = lyapunov_loss
        
        # 3. Collapse Prevention: Ensure trajectory doesn't collapse to a point
        trajectory_std = trajectories.std(dim=1).mean(dim=1)  # [batch]
        collapse_loss = F.relu(self.collapse_threshold - trajectory_std).mean()
        losses['anti_collapse'] = collapse_loss
        
        # 4. NaN/Inf Detection: Heavy penalty for numerical issues
        has_nan = torch.isnan(trajectories).any()
        has_inf = torch.isinf(trajectories).any()
        numerical_loss = torch.tensor(0.0, device=device)
        if has_nan or has_inf:
            numerical_loss = torch.tensor(100.0, device=device)
        losses['numerical_stability'] = numerical_loss
        
        # 5. Embedding Diversity (if provided): Prevent mode collapse
        if embeddings is not None and embeddings.shape[0] > 1:
            # Pairwise distances between embeddings
            embed_normalized = F.normalize(embeddings, p=2, dim=1)
            similarity_matrix = torch.mm(embed_normalized, embed_normalized.t())
            
            # Exclude diagonal
            mask = ~torch.eye(embeddings.shape[0], dtype=torch.bool, device=device)
            avg_similarity = similarity_matrix[mask].mean()
            
            # Penalize high similarity (mode collapse)
            diversity_loss = F.relu(avg_similarity - 0.5)
            losses['embedding_diversity'] = diversity_loss * self.diversity_weight
        else:
            losses['embedding_diversity'] = torch.tensor(0.0, device=device)
        
        # Total stability loss
        total = (
            self.stability_weight * boundedness_loss +
            self.stability_weight * lyapunov_loss +
            self.stability_weight * collapse_loss +
            numerical_loss +
            losses['embedding_diversity']
        )
        losses['total_stability'] = total
        
        # Add diagnostic info
        losses['mean_lyapunov'] = mean_lyapunov.mean()
        losses['max_trajectory_norm'] = max_norms.max()
        losses['min_trajectory_std'] = trajectory_std.min()
        
        return losses


# ============================================================
# PHASE SYNCHRONIZATION LOSS
# ============================================================

class PhaseSynchronizationLoss(nn.Module):
    """
    Phase Synchronization Loss for speaker embedding learning.
    
    Encourages trajectories from the same speaker to synchronize,
    while trajectories from different speakers should remain desynchronized.
    
    This leverages the fundamental property of chaotic systems where
    coupled systems can achieve synchronization under certain conditions.
    """
    
    def __init__(
        self,
        sync_weight: float = 0.1,
        desync_weight: float = 0.1,
        margin: float = 1.0,
        distance_type: str = 'euclidean'
    ):
        super().__init__()
        self.sync_weight = sync_weight
        self.desync_weight = desync_weight
        self.margin = margin
        self.distance_type = distance_type
        
    def trajectory_distance(
        self, 
        traj1: torch.Tensor,
        traj2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute synchronization distance between two trajectories.
        
        Args:
            traj1: First trajectory [T, dim]
            traj2: Second trajectory [T, dim]
            
        Returns:
            Scalar distance value
        """
        if self.distance_type == 'euclidean':
            diff = traj1 - traj2
            dist = torch.norm(diff, dim=1).mean()
            
        elif self.distance_type == 'cosine':
            traj1_flat = traj1.reshape(traj1.size(0), -1)
            traj2_flat = traj2.reshape(traj2.size(0), -1)
            cos_sim = F.cosine_similarity(traj1_flat, traj2_flat, dim=-1)
            dist = 1 - cos_sim
            
        elif self.distance_type == 'manhattan':
            # manhattan distance
            diff = torch.abs(traj1 - traj2)
            dist = diff.sum(dim=-1).mean(dim=-1)
            
        else:
            diff = traj1 - traj2
            dist = torch.norm(diff, dim=1).mean()
        return dist
    
    def forward(
        self, 
        trajectories: torch.Tensor,
        labels: torch.Tensor,
        sample_ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute phase synchronization loss.
        
        Args:
            trajectories: Chaotic trajectories [batch, T, dim]
            labels: Speaker labels [batch]
            sample_ratio: Ratio of pairs to sample (for efficiency)
            
        Returns:
            Dictionary with sync_loss, desync_loss, and total
        """
        batch_size = trajectories.shape[0]
        device = trajectories.device
        
        sync_loss = torch.tensor(0.0, device=device)
        desync_loss = torch.tensor(0.0, device=device)
        sync_count = 0
        desync_count = 0
        
        # Sample pairs for efficiency
        num_pairs = int(batch_size * (batch_size - 1) / 2 * sample_ratio)
        num_pairs = max(num_pairs, min(10, batch_size * (batch_size - 1) // 2))
        
        indices = torch.randperm(batch_size * (batch_size - 1) // 2)[:num_pairs]
        
        pair_idx = 0
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if pair_idx not in indices:
                    pair_idx += 1
                    continue
                pair_idx += 1
                
                dist = self.trajectory_distance(trajectories[i], trajectories[j])
                
                if labels[i] == labels[j]:
                    # Same speaker: encourage synchronization (minimize distance)
                    sync_loss = sync_loss + dist
                    sync_count += 1
                else:
                    # Different speakers: encourage desynchronization
                    desync_loss = desync_loss + F.relu(self.margin - dist)
                    desync_count += 1
        
        # Normalize
        if sync_count > 0:
            sync_loss = sync_loss / sync_count
        if desync_count > 0:
            desync_loss = desync_loss / desync_count
        
        total_loss = self.sync_weight * sync_loss + self.desync_weight * desync_loss
        
        return {
            'sync_loss': sync_loss,
            'desync_loss': desync_loss,
            'total_sync': total_loss,
            'sync_pairs': sync_count,
            'desync_pairs': desync_count
        }


# ============================================================
# BIFURCATION CONTROL MODULE (Optional - Advanced)
# ============================================================

class BifurcationControlModule(nn.Module):
    """
    Bifurcation Control Module for adaptive chaos regime selection.
    
    This module learns to adjust chaotic system parameters to navigate
    between different dynamical regimes (fixed point, periodic, chaotic)
    based on input features, potentially improving speaker discrimination.
    
    Theory: By controlling the bifurcation parameter (e.g., rho in Lorenz),
    we can push the system into regimes where speaker-specific features
    are more distinguishable.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        num_regimes: int = 4,
        control_strength: float = 0.5,
        learnable_boundaries: bool = True
    ):
        """
        Args:
            input_dim: Dimension of trajectory state (3 for Lorenz)
            hidden_dim: Hidden layer dimension
            num_regimes: Number of distinct dynamical regimes to learn
            control_strength: How strongly to modify trajectories
            learnable_boundaries: Whether regime boundaries are learnable
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        self.control_strength = control_strength
        
        # Regime classifier: determines which regime the trajectory is in
        self.regime_classifier = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # mean + std of trajectory
            nn.ReLU(),
            nn.Linear(hidden_dim, num_regimes),
            nn.Softmax(dim=-1)
        )
        
        # Regime-specific transformation parameters
        # Each regime has its own scaling and bias
        self.regime_scales = nn.Parameter(torch.ones(num_regimes, input_dim))
        self.regime_biases = nn.Parameter(torch.zeros(num_regimes, input_dim))
        
        # Bifurcation parameter predictor
        # Predicts adjustment to system parameters based on trajectory
        self.bifurcation_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),  # sigma, rho, beta adjustments
            nn.Tanh()  # Bounded adjustments
        )
        
        # Learnable regime boundaries (in terms of Lyapunov-like measure)
        if learnable_boundaries:
            self.regime_boundaries = nn.Parameter(
                torch.linspace(0, 1, num_regimes + 1)[1:-1]
            )
        else:
            self.register_buffer(
                'regime_boundaries',
                torch.linspace(0, 1, num_regimes + 1)[1:-1]
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def compute_trajectory_statistics(
        self, 
        trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute statistics for regime classification.
        
        Args:
            trajectory: [batch, T, dim]
        Returns:
            statistics: [batch, dim * 2]
        """
        mean = trajectory.mean(dim=1)  # [batch, dim]
        std = trajectory.std(dim=1)    # [batch, dim]
        return torch.cat([mean, std], dim=-1)
    
    def estimate_local_lyapunov(
        self, 
        trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate local Lyapunov-like measure for regime detection.
        
        Args:
            trajectory: [batch, T, dim]
        Returns:
            lyapunov_estimate: [batch]
        """
        # Compute local divergence rate
        diffs = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        diff_norms = torch.norm(diffs, dim=2) + 1e-8
        
        # Log expansion rate
        log_expansion = torch.log(diff_norms[:, 1:] / diff_norms[:, :-1].clamp(min=1e-8))
        
        # Mean and normalize to [0, 1]
        lyap = log_expansion.mean(dim=1)
        lyap_normalized = torch.sigmoid(lyap)
        
        return lyap_normalized
    
    def forward(
        self, 
        trajectory: torch.Tensor,
        return_regime_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Apply bifurcation control to trajectory.
        
        Args:
            trajectory: Input trajectory [batch, T, dim]
            return_regime_info: Whether to return regime information
            
        Returns:
            controlled_trajectory: Modified trajectory [batch, T, dim]
            regime_info: (optional) Dictionary with regime details
        """
        batch_size, T, dim = trajectory.shape
        
        # Compute trajectory statistics
        stats = self.compute_trajectory_statistics(trajectory)
        
        # Classify regime
        regime_probs = self.regime_classifier(stats)  # [batch, num_regimes]
        
        # Get regime-specific transformations (soft selection)
        # [batch, num_regimes] @ [num_regimes, dim] -> [batch, dim]
        selected_scales = torch.matmul(regime_probs, self.regime_scales)
        selected_biases = torch.matmul(regime_probs, self.regime_biases)
        
        # Apply transformation
        # Expand to match trajectory shape
        scales = selected_scales.unsqueeze(1)  # [batch, 1, dim]
        biases = selected_biases.unsqueeze(1)  # [batch, 1, dim]
        
        controlled = trajectory * (1 + self.control_strength * scales) + \
                     self.control_strength * biases
        
        # Predict bifurcation adjustments (for monitoring/analysis)
        bifurcation_params = self.bifurcation_predictor(stats)
        
        if return_regime_info:
            regime_info = {
                'regime_probs': regime_probs,
                'dominant_regime': torch.argmax(regime_probs, dim=1),
                'bifurcation_adjustments': bifurcation_params,
                'local_lyapunov': self.estimate_local_lyapunov(trajectory)
            }
            return controlled, regime_info
        
        return controlled
    
    def get_regime_statistics(
        self, 
        trajectory: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get detailed regime statistics for analysis."""
        stats = self.compute_trajectory_statistics(trajectory)
        regime_probs = self.regime_classifier(stats)
        lyap = self.estimate_local_lyapunov(trajectory)
        
        return {
            'regime_distribution': regime_probs.mean(dim=0),
            'mean_lyapunov': lyap.mean(),
            'lyapunov_std': lyap.std()
        }


# ============================================================
# PLDA CLASSIFIER (Traditional Method for Comparison)
# ============================================================

class PLDAClassifier(nn.Module):
    """
    Probabilistic Linear Discriminant Analysis (PLDA) Classifier.
    
    A traditional speaker verification approach that models speaker
    embeddings using a generative model with between-speaker and
    within-speaker variability.
    
    This implementation provides a neural approximation of PLDA
    for end-to-end training compatibility.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        num_speakers: int = 251,
        latent_dim: int = 128,
        use_length_norm: bool = True
    ):
        """
        Args:
            embedding_dim: Input speaker embedding dimension
            num_speakers: Number of speakers (for enrollment)
            latent_dim: Dimension of latent speaker space
            use_length_norm: Whether to apply length normalization
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.latent_dim = latent_dim
        self.use_length_norm = use_length_norm
        
        # Between-class projection (speaker identity subspace)
        self.between_class_proj = nn.Linear(embedding_dim, latent_dim, bias=False)
        
        # Within-class projection (session/channel variability)
        self.within_class_proj = nn.Linear(embedding_dim, latent_dim, bias=False)
        
        # Global mean (centering)
        self.register_buffer('global_mean', torch.zeros(embedding_dim))
        
        # Speaker centroids for classification
        self.speaker_centroids = nn.Parameter(torch.randn(num_speakers, latent_dim) * 0.1)
        
        # Precision matrices (simplified as diagonal)
        self.between_precision = nn.Parameter(torch.ones(latent_dim))
        self.within_precision = nn.Parameter(torch.ones(latent_dim))
        
        # Scoring layer
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.score_bias = nn.Parameter(torch.tensor(0.0))
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.orthogonal_(self.between_class_proj.weight)
        nn.init.orthogonal_(self.within_class_proj.weight)
        
    def length_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply length normalization."""
        return F.normalize(x, p=2, dim=-1)
    
    def center(self, x: torch.Tensor) -> torch.Tensor:
        """Center embeddings by subtracting global mean."""
        return x - self.global_mean
    
    def project_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Project embedding to PLDA latent space."""
        if self.use_length_norm:
            x = self.length_normalize(x)
        x = self.center(x)
        
        # Combined projection
        between = self.between_class_proj(x)
        within = self.within_class_proj(x)
        
        # Latent representation emphasizes between-class
        latent = between + 0.1 * within
        
        return latent
    
    def compute_llr_scores(
        self, 
        embeddings: torch.Tensor,
        target_speakers: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute log-likelihood ratio scores.
        
        Args:
            embeddings: Speaker embeddings [batch, embedding_dim]
            target_speakers: Target speaker indices [batch] (optional)
            
        Returns:
            scores: LLR scores against all speakers [batch, num_speakers]
                   or against targets [batch] if target_speakers provided
        """
        # Project to latent space
        latent = self.project_to_latent(embeddings)  # [batch, latent_dim]
        
        # Compute distances to all speaker centroids
        # Using Mahalanobis-like distance with learned precision
        diff = latent.unsqueeze(1) - self.speaker_centroids.unsqueeze(0)  # [batch, num_speakers, latent_dim]
        
        # Weighted squared distance
        precision = F.softplus(self.between_precision)  # Ensure positive
        weighted_diff = diff * precision.unsqueeze(0).unsqueeze(0)
        distances = (weighted_diff * diff).sum(dim=-1)  # [batch, num_speakers]
        
        # Convert to scores (negative distance = higher similarity)
        scores = -distances * self.score_scale + self.score_bias
        
        if target_speakers is not None:
            # Return scores for specific targets
            batch_idx = torch.arange(embeddings.shape[0], device=embeddings.device)
            return scores[batch_idx, target_speakers]
        
        return scores
    
    def forward(
        self, 
        embeddings: torch.Tensor,
        labels: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            embeddings: Speaker embeddings [batch, embedding_dim]
            labels: Speaker labels [batch] (used for training metrics)
            
        Returns:
            logits: Classification logits [batch, num_speakers]
        """
        scores = self.compute_llr_scores(embeddings)
        return scores
    
    def verify(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        threshold: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Speaker verification between two embeddings.
        
        Args:
            embedding1: First embedding [batch, embedding_dim]
            embedding2: Second embedding [batch, embedding_dim]
            threshold: Decision threshold
            
        Returns:
            decisions: Boolean verification decisions [batch]
            scores: Verification scores [batch]
        """
        latent1 = self.project_to_latent(embedding1)
        latent2 = self.project_to_latent(embedding2)
        
        # Compute similarity score
        precision = F.softplus(self.between_precision)
        diff = latent1 - latent2
        weighted_diff = diff * precision
        distance = (weighted_diff * diff).sum(dim=-1)
        
        scores = -distance * self.score_scale + self.score_bias
        decisions = scores > threshold
        
        return decisions, scores
    
    def update_global_mean(self, embeddings: torch.Tensor):
        """Update global mean from training data (call during training)."""
        with torch.no_grad():
            batch_mean = embeddings.mean(dim=0)
            # Exponential moving average
            momentum = 0.99
            self.global_mean = momentum * self.global_mean + (1 - momentum) * batch_mean

__all__ = [
    'ChaoticSpeakerRecognitionNetwork',
    'EnhancedSpeakerEmbedding',
    'EnhancedChaoticClassifier',
    'AdversarialChaoticAugmentation',
    'LyapunovStabilityLoss',
    'PhaseSynchronizationLoss',
    'BifurcationControlModule',
    'PLDAClassifier',
    'SpeakerEmbedding',
    'ChaoticClassifier',
    'BatchPhaseSpaceReconstructor',
    'create_chaotic_speaker_network'
]

if __name__ == "__main__":
    print("Testing ENHANCED Chaotic Speaker Recognition Network...")
    
    # Create test configuration
    config = {
        'sample_rate': 16000,
        'embedding_dim': 8,
        'mlsa_scales': 3,
        'evolution_time': 0.001,
        'time_step': 0.02,
        'pooling_type': 'comprehensive',
        'speaker_embedding_dim': 256,  # ENHANCED
        'embedding_hidden_dims': [512, 256, 128],  # ENHANCED
        'num_speakers': 251,  # Real dataset size
        'classifier_type': 'linear',
        'device': 'cpu'
    }
    
    # Create network
    network = create_chaotic_speaker_network(config)
    
    # Print model information
    model_info = network.get_model_info()
    print(f"\nModel created successfully!")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Parameters per speaker: {model_info['parameters_per_speaker']:.0f}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Test forward pass
    batch_size = 4
    sequence_length = 1000
    test_audio = torch.randn(batch_size, sequence_length)
    test_labels = torch.randint(0, config['num_speakers'], (batch_size,))
    
    print(f"\nTesting forward pass:")
    print(f"Input audio shape: {test_audio.shape}")
    
    network.eval()
    with torch.no_grad():
        logits, intermediates = network(test_audio, return_intermediates=True)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Speaker embeddings shape: {intermediates['speaker_embeddings'].shape}")
    
    # Test prediction
    predicted_classes, confidence_scores = network.predict(test_audio)
    print(f"\nPredictions: {predicted_classes}")
    print(f"Confidence scores: {confidence_scores}")
    
    print("\nENHANCED Chaotic Speaker Recognition Network test completed!")