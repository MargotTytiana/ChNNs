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
        classifier_type: str = 'cosine',
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
        classifier_type: str = 'cosine',
        
        # Device
        device: str = 'cpu'
    ):
        super(ChaoticSpeakerRecognitionNetwork, self).__init__()
        
        # Save all parameters
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.embedding_dim = embedding_dim
        self.delay_method = delay_method
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
        
        # Initialize components
        self._initialize_components()
        
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
        
        # Phase space reconstruction
        from core.phase_space_reconstruction import EmbeddingConfig
        
        phase_space_config = EmbeddingConfig()
        base_reconstructor = PhaseSpaceReconstructor(config=phase_space_config)
        
        self.phase_space = BatchPhaseSpaceReconstructor(
            reconstructor=base_reconstructor,
            fixed_output_length=200,
            device=self.device
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
            self.chaotic_embedding = ChaoticEmbedding(
                input_dim=chaotic_feature_dim,
                system_type=self.chaotic_system,
                evolution_time=self.evolution_time,
                time_step=self.time_step,
                device=self.device
            )
        else:
            trajectory_dim = int(self.evolution_time / self.time_step) * 3
            self.chaotic_embedding = MockComponent(None, trajectory_dim)
            
        # Attractor pooling
        if AttractorPooling is not None:
            self.attractor_pooling = AttractorPooling(
                pooling_type=self.pooling_type,
                device=self.device
            )
            pooling_output_dim = 5 if self.pooling_type == 'comprehensive' else 3
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
    
    def extract_chaotic_features(self, phase_space_data: torch.Tensor) -> torch.Tensor:
        """Extract chaotic features using MLSA and RQA."""
        batch_size = phase_space_data.shape[0]
        all_mlsa_features = []
        all_rqa_features = []

        mlsa_dim = self.mlsa_feature_dim if hasattr(self, 'mlsa_feature_dim') else None
        rqa_dim = self.rqa_feature_dim if hasattr(self, 'rqa_feature_dim') else None
        
        for i in range(batch_size):
            sample = phase_space_data[i].cpu().detach().numpy()
            
            if sample.ndim == 2:
                signal_1d = sample[:, 0]
            else:
                signal_1d = sample
            
            # Extract MLSA features
            try:
                mlsa_result = self.mlsa_extractor.extract_features(signal_1d)
                if mlsa_result['success']:
                    mlsa_vec = mlsa_result['feature_vector']
                    if mlsa_dim is None:
                        mlsa_dim = len(mlsa_vec)
                    if len(mlsa_vec) != mlsa_dim:
                        if len(mlsa_vec) < mlsa_dim:
                            mlsa_vec = np.pad(mlsa_vec, (0, mlsa_dim - len(mlsa_vec)), 
                                             mode='constant', constant_values=0)
                        else:
                            mlsa_vec = mlsa_vec[:mlsa_dim]
                else:
                    if mlsa_dim is None:
                        mlsa_dim = self.mlsa_scales
                    mlsa_vec = np.zeros(mlsa_dim)
            except:
                if mlsa_dim is None:
                    mlsa_dim = self.mlsa_scales
                mlsa_vec = np.zeros(mlsa_dim)
            
            # Extract RQA features
            try:
                rqa_result = self.rqa_extractor.extract_features(signal_1d)
                if rqa_result['success']:
                    rqa_vec = rqa_result['feature_vector']
                    if rqa_dim is None:
                        rqa_dim = len(rqa_vec)
                    if len(rqa_vec) != rqa_dim:
                        if len(rqa_vec) < rqa_dim:
                            rqa_vec = np.pad(rqa_vec, (0, rqa_dim - len(rqa_vec)), 
                                            mode='constant', constant_values=0)
                        else:
                            rqa_vec = rqa_vec[:rqa_dim]
                else:
                    if rqa_dim is None:
                        rqa_dim = 3
                    rqa_vec = np.zeros(rqa_dim)
            except:
                if rqa_dim is None:
                    rqa_dim = 3
                rqa_vec = np.zeros(rqa_dim)
            
            # Handle NaN values
            mlsa_vec = np.nan_to_num(mlsa_vec, nan=0.0, posinf=0.0, neginf=0.0)
            rqa_vec = np.nan_to_num(rqa_vec, nan=0.0, posinf=0.0, neginf=0.0)
            
            mlsa_vec = np.atleast_1d(mlsa_vec).flatten()
            rqa_vec = np.atleast_1d(rqa_vec).flatten()
            
            all_mlsa_features.append(mlsa_vec)
            all_rqa_features.append(rqa_vec)
        
        try:
            mlsa_array = np.stack(all_mlsa_features)
            rqa_array = np.stack(all_rqa_features)
            
            mlsa_features = torch.FloatTensor(mlsa_array).to(phase_space_data.device)
            rqa_features = torch.FloatTensor(rqa_array).to(phase_space_data.device)
            
        except ValueError:
            max_mlsa_len = max(len(vec) for vec in all_mlsa_features)
            max_rqa_len = max(len(vec) for vec in all_rqa_features)
            
            padded_mlsa = []
            for vec in all_mlsa_features:
                if len(vec) < max_mlsa_len:
                    vec = np.pad(vec, (0, max_mlsa_len - len(vec)), 
                               mode='constant', constant_values=0)
                padded_mlsa.append(vec)
            
            padded_rqa = []
            for vec in all_rqa_features:
                if len(vec) < max_rqa_len:
                    vec = np.pad(vec, (0, max_rqa_len - len(vec)), 
                               mode='constant', constant_values=0)
                padded_rqa.append(vec)
            
            mlsa_features = torch.FloatTensor(np.stack(padded_mlsa)).to(phase_space_data.device)
            rqa_features = torch.FloatTensor(np.stack(padded_rqa)).to(phase_space_data.device)
        
        chaotic_features = torch.cat([mlsa_features, rqa_features], dim=-1)
        
        return chaotic_features
    
    def forward(
        self, 
        audio: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass through the complete network."""
        intermediates = {} if return_intermediates else None
        
        # Step 1: Phase space reconstruction
        phase_space_data = self.phase_space(audio)
        if return_intermediates:
            intermediates['phase_space'] = phase_space_data
        
        # Step 2: Chaotic feature extraction
        chaotic_features = self.extract_chaotic_features(phase_space_data)
        if return_intermediates:
            intermediates['chaotic_features'] = chaotic_features
        
        # Step 3: Chaotic embedding
        if hasattr(self.chaotic_embedding, 'forward'):
            chaotic_trajectories = self.chaotic_embedding(chaotic_features)
        else:
            chaotic_trajectories = self.chaotic_embedding(chaotic_features.view(chaotic_features.shape[0], -1))
            batch_size = chaotic_trajectories.shape[0]
            trajectory_length = chaotic_trajectories.shape[1] // 3
            chaotic_trajectories = chaotic_trajectories.view(batch_size, trajectory_length, 3)
            
        if return_intermediates:
            intermediates['chaotic_trajectories'] = chaotic_trajectories
        
        # Step 4: Attractor pooling
        pooled_features = self.attractor_pooling(chaotic_trajectories)
        if return_intermediates:
            intermediates['pooled_features'] = pooled_features
        
        # Step 5: ENHANCED Speaker embedding
        speaker_embeddings = self.speaker_embedding(pooled_features)
        if return_intermediates:
            intermediates['speaker_embeddings'] = speaker_embeddings
        
        # Step 6: ENHANCED Classification
        logits = self.classifier(speaker_embeddings, labels)
        if return_intermediates:
            intermediates['logits'] = logits
        
        if return_intermediates:
            return logits, intermediates
        else:
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
                 fixed_output_length: int = 100,
                 device: str = 'cpu'):
        super().__init__()
        self.reconstructor = reconstructor
        self.fixed_output_length = fixed_output_length
        self.device = device
        
    def forward(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """Process a batch of audio signals."""
        batch_size = audio_batch.shape[0]
        results = []
        
        for i in range(batch_size):
            single_audio = audio_batch[i].cpu().numpy()
            
            try:
                reconstruction = self.reconstructor.reconstruct(
                    single_audio, delay=None, dimension=None
                )
                
                if reconstruction['embedding_success']:
                    embedded = reconstruction['embedded_data']
                    embedded_tensor = torch.from_numpy(embedded).float()
                else:
                    embedded_tensor = self._create_fallback_embedding(audio_batch[i])
                    
            except:
                embedded_tensor = self._create_fallback_embedding(audio_batch[i])
            
            results.append(embedded_tensor)
        
        standardized = self._standardize_length(results)
        return standardized.to(self.device)
    
    def _create_fallback_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Create simple fallback phase space embedding."""
        delay = 1
        dim = 3
        
        audio_np = audio.cpu().numpy()
        n_points = len(audio_np) - (dim - 1) * delay
        
        if n_points <= 0:
            return torch.zeros(self.fixed_output_length, dim)
        
        embedded = np.zeros((n_points, dim))
        for i in range(dim):
            embedded[:, i] = audio_np[i*delay : i*delay + n_points]
        
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

# 在文件末尾，create_chaotic_speaker_network 函数之后添加：

# ============== 向后兼容性别名 ==============
# 为了兼容 hybrid_models.py 的导入
SpeakerEmbedding = EnhancedSpeakerEmbedding
ChaoticClassifier = EnhancedChaoticClassifier

# 导出所有需要的类
__all__ = [
    'ChaoticSpeakerRecognitionNetwork',
    'EnhancedSpeakerEmbedding',
    'EnhancedChaoticClassifier',
    'SpeakerEmbedding',  # 别名
    'ChaoticClassifier',  # 别名
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
        'evolution_time': 0.1,
        'time_step': 0.02,
        'pooling_type': 'comprehensive',
        'speaker_embedding_dim': 256,  # ENHANCED
        'embedding_hidden_dims': [512, 256, 128],  # ENHANCED
        'num_speakers': 251,  # Real dataset size
        'classifier_type': 'cosine',
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