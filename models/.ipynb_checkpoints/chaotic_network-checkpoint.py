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
    model_dir = current_file.parent.parent  # models -> Model
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

from phase_space_reconstruction import PhaseSpaceReconstructor
from mlsa_extractor import MLSAExtractor
from rqa_extractor import RQAExtractor
from chaotic_embedding import ChaoticEmbedding
from attractor_pooling import AttractorPooling

from numerical_stability import OutlierDetector

class MockComponent(nn.Module):
    """Mock component for testing when core modules are not available."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim  # Can be None for auto-adaptation
        self.output_dim = output_dim
        self.linear = None  # Will be initialized on first forward pass
    
    def forward(self, x):
        # Initialize linear layer based on actual input shape
        if self.linear is None:
            # Handle different input shapes
            if len(x.shape) == 2:
                # [batch_size, features]
                actual_input_dim = x.shape[-1]
            elif len(x.shape) == 3:
                # [batch_size, sequence, features] - flatten sequence dimension
                actual_input_dim = x.shape[1] * x.shape[2]
                x = x.view(x.shape[0], -1)  # Flatten to [batch_size, sequence*features]
            else:
                # Fallback: use last dimension
                actual_input_dim = x.shape[-1]
            
            self.linear = nn.Linear(actual_input_dim, self.output_dim).to(x.device)
        
        # Ensure input matches expected shape for linear layer
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
            
        return self.linear(x)


class SpeakerEmbedding(nn.Module):
    """
    Speaker Embedding Layer - Converts chaotic features to speaker identity vectors.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        embedding_dim: int = 128,
        hidden_dims: list = [64, 32],
        dropout_rate: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Initialize Speaker Embedding Layer.
        
        Args:
            input_dim: Dimension of input features from attractor pooling
            embedding_dim: Dimension of speaker embedding vectors
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'tanh', 'gelu')
        """
        super(SpeakerEmbedding, self).__init__()
        
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
        
        # Build embedding network
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        # Final embedding layer (no activation to allow full range)
        layers.append(nn.Linear(current_dim, embedding_dim))
        
        self.embedding_network = nn.Sequential(*layers)
        
        # L2 normalization for embedding vectors
        self.normalize = True
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through speaker embedding layer.
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            Speaker embeddings [batch_size, embedding_dim]
        """
        embeddings = self.embedding_network(features)
        
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings


class ChaoticClassifier(nn.Module):
    """
    Final classifier for speaker recognition using chaotic features.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_speakers: int = 100,
        classifier_type: str = 'cosine',
        temperature: float = 30.0,
        margin: float = 0.35
    ):
        """
        Initialize Chaotic Classifier.
        
        Args:
            embedding_dim: Dimension of input speaker embeddings
            num_speakers: Number of speaker classes
            classifier_type: Type of classifier ('linear', 'cosine', 'angular')
            temperature: Temperature scaling for cosine similarity
            margin: Margin for angular margin loss
        """
        super(ChaoticClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.classifier_type = classifier_type
        self.temperature = temperature
        self.margin = margin
        
        if classifier_type == 'linear':
            self.classifier = nn.Linear(embedding_dim, num_speakers)
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
        """
        Forward pass through classifier.
        
        Args:
            embeddings: Speaker embeddings [batch_size, embedding_dim]
            labels: True labels for training (optional)
            
        Returns:
            Classification logits [batch_size, num_speakers]
        """
        if self.classifier_type == 'linear':
            return self.classifier(embeddings)
        
        elif self.classifier_type == 'cosine':
            # Cosine similarity classification
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            normalized_weight = F.normalize(self.weight, p=2, dim=1)
            
            cosine_sim = F.linear(normalized_embeddings, normalized_weight)
            logits = cosine_sim * self.temperature
            
            return logits
        
        elif self.classifier_type == 'angular':
            # Angular margin loss (ArcFace-style)
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            normalized_weight = F.normalize(self.weight, p=2, dim=1)
            
            cosine_sim = F.linear(normalized_embeddings, normalized_weight)
            cosine_sim = torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7)
            
            if self.training and labels is not None:
                # Add angular margin during training
                theta = torch.acos(cosine_sim)
                target_logits = torch.cos(theta + self.margin)
                
                # Create one-hot mask
                one_hot = torch.zeros_like(cosine_sim)
                one_hot.scatter_(1, labels.view(-1, 1), 1)
                
                # Apply margin only to target class
                logits = (one_hot * target_logits) + ((1.0 - one_hot) * cosine_sim)
            else:
                logits = cosine_sim
                
            logits = logits * self.temperature
            return logits


class ChaoticSpeakerRecognitionNetwork(nn.Module):
    """
    Complete Chaotic Speaker Recognition Network integrating all components.
    
    This network implements the full pipeline from the research paper:
    1. Phase Space Reconstruction
    2. Chaotic Feature Extraction (MLSA + RQA)
    3. Chaotic Embedding Layer  
    4. Strange Attractor Pooling
    5. Speaker Embedding
    6. Classification
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
        
        # Speaker embedding parameters
        speaker_embedding_dim: int = 128,
        
        # Classification parameters
        num_speakers: int = 100,
        classifier_type: str = 'cosine',
        
        # Device
        device: str = 'cpu'
    ):
        """
        Initialize the complete chaotic speaker recognition network.
        """
        super(ChaoticSpeakerRecognitionNetwork, self).__init__()
        
        # Save all parameters as instance attributes
        # Audio processing
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        
        # Phase space reconstruction
        self.embedding_dim = embedding_dim
        self.delay_method = delay_method
        
        # Chaotic features
        self.mlsa_scales = mlsa_scales
        self.rqa_radius_ratio = rqa_radius_ratio
        
        # Chaotic embedding
        self.chaotic_system = chaotic_system
        self.evolution_time = evolution_time
        self.time_step = time_step
        
        # Attractor pooling
        self.pooling_type = pooling_type
        
        # Speaker embedding
        self.speaker_embedding_dim = speaker_embedding_dim
        
        # Classification
        self.num_speakers = num_speakers
        self.classifier_type = classifier_type
        
        # Device
        self.device = device
        
        # Initialize components based on availability
        self._initialize_components()
        
        # Loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.training_stats = {
            'total_loss': 0.0,
            'accuracy': 0.0,
            'num_batches': 0
        }
        

    def _initialize_components(self):
        """Initialize all network components."""
        
        # Phase space reconstruction with EmbeddingConfig
        from phase_space_reconstruction import EmbeddingConfig
        
        phase_space_config = EmbeddingConfig()
        base_reconstructor = PhaseSpaceReconstructor(config=phase_space_config)
        
        # Wrap with batch processor
        self.phase_space = BatchPhaseSpaceReconstructor(
            reconstructor=base_reconstructor,
            fixed_output_length=200,  # Or from config
            device=self.device
        )
        
        # MLSA extractor - needs MLSAConfig
        if MLSAExtractor is not None:
            try:
                from mlsa_extractor import MLSAConfig
                
                mlsa_config = MLSAConfig(
                    n_scales=self.mlsa_scales,
                    scale_factors=[1, 2, 4, 8, 16][:self.mlsa_scales],  # Use first n_scales factors
                    decomposition_method='fourier',  # Use Fourier for stability
                    min_segment_length=100
                )
                
                self.mlsa_extractor = MLSAExtractor(config=mlsa_config)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to create MLSAExtractor: {e}, using MockComponent")
                self.mlsa_extractor = MockComponent(None, self.mlsa_scales)
        else:
            self.mlsa_extractor = MockComponent(None, self.mlsa_scales)
            
        # RQA extractor - needs RQAConfig  
        if RQAExtractor is not None:
            try:
                from rqa_extractor import RQAConfig
                
                rqa_config = RQAConfig(
                    threshold_method='fixed_amount',
                    recurrence_rate_target=self.rqa_radius_ratio,  # Use this as recurrence rate
                    scale_factors=[1, 2, 4],  # Standard multi-scale factors
                    min_diagonal_length=2,
                    min_vertical_length=2
                )
                
                self.rqa_extractor = RQAExtractor(config=rqa_config)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to create RQAExtractor: {e}, using MockComponent")
                self.rqa_extractor = MockComponent(None, 3)
        else:
            self.rqa_extractor = MockComponent(None, 3)

        # Determine chaotic feature dimension more reliably
        try:
            # Create a dummy signal to determine feature dimensions
            dummy_signal = np.random.randn(1000)  # Longer signal for better feature extraction
            
            # Get MLSA feature dimension
            try:
                mlsa_result = self.mlsa_extractor.extract_features(dummy_signal)
                if mlsa_result['success'] and 'feature_vector' in mlsa_result:
                    mlsa_feature_dim = len(mlsa_result['feature_vector'])
                    self.mlsa_feature_dim = mlsa_feature_dim  # Store for later use
                else:
                    mlsa_feature_dim = self.mlsa_scales
                    self.mlsa_feature_dim = mlsa_feature_dim
            except Exception as e:
                print(f"Warning: Could not determine MLSA feature dimension: {e}")
                mlsa_feature_dim = self.mlsa_scales
                self.mlsa_feature_dim = mlsa_feature_dim
            
            # Get RQA feature dimension
            try:
                rqa_result = self.rqa_extractor.extract_features(dummy_signal)
                if rqa_result['success'] and 'feature_vector' in rqa_result:
                    rqa_feature_dim = len(rqa_result['feature_vector'])
                    self.rqa_feature_dim = rqa_feature_dim  # Store for later use
                else:
                    rqa_feature_dim = 3
                    self.rqa_feature_dim = rqa_feature_dim
            except Exception as e:
                print(f"Warning: Could not determine RQA feature dimension: {e}")
                rqa_feature_dim = 3
                self.rqa_feature_dim = rqa_feature_dim
            
            chaotic_feature_dim = mlsa_feature_dim + rqa_feature_dim
            print(f"Chaotic feature dimensions: MLSA={mlsa_feature_dim}, RQA={rqa_feature_dim}, Total={chaotic_feature_dim}")
            
        except Exception as e:
            print(f"Warning: Error determining feature dimensions: {e}")
            mlsa_feature_dim = self.mlsa_scales
            rqa_feature_dim = 3
            chaotic_feature_dim = mlsa_feature_dim + rqa_feature_dim
            self.mlsa_feature_dim = mlsa_feature_dim
            self.rqa_feature_dim = rqa_feature_dim
        
        # Chaotic embedding layer - this one accepts direct parameters
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
            
        # Attractor pooling - this one also accepts direct parameters
        if AttractorPooling is not None:
            self.attractor_pooling = AttractorPooling(
                pooling_type=self.pooling_type,
                device=self.device
            )
            pooling_output_dim = 5 if self.pooling_type == 'comprehensive' else 3
        else:
            pooling_output_dim = 5
            self.attractor_pooling = MockComponent(None, pooling_output_dim)
            
        # Speaker embedding
        self.speaker_embedding = SpeakerEmbedding(
            input_dim=pooling_output_dim,
            embedding_dim=self.speaker_embedding_dim
        )
        
        # Final classifier
        self.classifier = ChaoticClassifier(
            embedding_dim=self.speaker_embedding_dim,
            num_speakers=self.num_speakers,
            classifier_type=self.classifier_type
        )
    
    
    def extract_chaotic_features(self, phase_space_data: torch.Tensor) -> torch.Tensor:
        """
        Extract chaotic features using MLSA and RQA.
        
        Args:
            phase_space_data: Phase space reconstructed data [batch_size, time_steps, embedding_dim]
            
        Returns:
            Combined chaotic features [batch_size, feature_dim]
        """
        batch_size = phase_space_data.shape[0]
        all_mlsa_features = []
        all_rqa_features = []

        # Use pre-determined feature dimensions if available
        mlsa_dim = self.mlsa_feature_dim if hasattr(self, 'mlsa_feature_dim') else None
        rqa_dim = self.rqa_feature_dim if hasattr(self, 'rqa_feature_dim') else None
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Convert to numpy for feature extraction
            sample = phase_space_data[i].cpu().detach().numpy()  # [time_steps, embedding_dim]
            
            # Flatten to 1D signal (use first dimension or mean)
            if sample.ndim == 2:
                signal_1d = sample[:, 0]  # Use first dimension
            else:
                signal_1d = sample
            
            # Extract MLSA features
            try:
                mlsa_result = self.mlsa_extractor.extract_features(signal_1d)
                if mlsa_result['success']:
                    mlsa_vec = mlsa_result['feature_vector']
                    # Set dimension from first successful extraction
                    if mlsa_dim is None:
                        mlsa_dim = len(mlsa_vec)
                    # Ensure consistent dimension
                    if len(mlsa_vec) != mlsa_dim:
                        # Pad or truncate to match expected dimension
                        if len(mlsa_vec) < mlsa_dim:
                            mlsa_vec = np.pad(mlsa_vec, (0, mlsa_dim - len(mlsa_vec)), 
                                             mode='constant', constant_values=0)
                        else:
                            mlsa_vec = mlsa_vec[:mlsa_dim]
                else:
                    # Use default dimension if available
                    if mlsa_dim is None:
                        mlsa_dim = self.mlsa_scales
                    mlsa_vec = np.zeros(mlsa_dim)
            except Exception as e:
                if mlsa_dim is None:
                    mlsa_dim = self.mlsa_scales
                mlsa_vec = np.zeros(mlsa_dim)
            
            # Extract RQA features
            try:
                rqa_result = self.rqa_extractor.extract_features(signal_1d)
                if rqa_result['success']:
                    rqa_vec = rqa_result['feature_vector']
                    # Set dimension from first successful extraction
                    if rqa_dim is None:
                        rqa_dim = len(rqa_vec)
                    # Ensure consistent dimension
                    if len(rqa_vec) != rqa_dim:
                        if len(rqa_vec) < rqa_dim:
                            rqa_vec = np.pad(rqa_vec, (0, rqa_dim - len(rqa_vec)), 
                                            mode='constant', constant_values=0)
                        else:
                            rqa_vec = rqa_vec[:rqa_dim]
                else:
                    if rqa_dim is None:
                        rqa_dim = 3  # Default RQA feature dimension
                    rqa_vec = np.zeros(rqa_dim)
            except Exception as e:
                if rqa_dim is None:
                    rqa_dim = 3
                rqa_vec = np.zeros(rqa_dim)
            
            # Handle NaN values
            mlsa_vec = np.nan_to_num(mlsa_vec, nan=0.0, posinf=0.0, neginf=0.0)
            rqa_vec = np.nan_to_num(rqa_vec, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure vectors are 1D
            mlsa_vec = np.atleast_1d(mlsa_vec).flatten()
            rqa_vec = np.atleast_1d(rqa_vec).flatten()
            
            all_mlsa_features.append(mlsa_vec)
            all_rqa_features.append(rqa_vec)
        
        # Convert to tensors with explicit shape checking
        try:
            # Stack arrays - this will fail if shapes are inconsistent
            mlsa_array = np.stack(all_mlsa_features)  # [batch_size, mlsa_dim]
            rqa_array = np.stack(all_rqa_features)    # [batch_size, rqa_dim]
            
            mlsa_features = torch.FloatTensor(mlsa_array).to(phase_space_data.device)
            rqa_features = torch.FloatTensor(rqa_array).to(phase_space_data.device)
            
        except ValueError as e:
            # Fallback: pad all vectors to maximum length
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
        
        # Combine features
        chaotic_features = torch.cat([mlsa_features, rqa_features], dim=-1)
        
        return chaotic_features
    
    def forward(
        self, 
        audio: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through the complete network.
        
        Args:
            audio: Input audio tensor [batch_size, num_samples] or [batch_size, frames]
            labels: True speaker labels (for training)
            return_intermediates: Whether to return intermediate representations
            
        Returns:
            logits: Classification logits [batch_size, num_speakers]
            intermediates: Dict of intermediate representations (if requested)
        """
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
            # Handle mock component
            chaotic_trajectories = self.chaotic_embedding(chaotic_features.view(chaotic_features.shape[0], -1))
            # Reshape to trajectory format
            batch_size = chaotic_trajectories.shape[0]
            trajectory_length = chaotic_trajectories.shape[1] // 3
            chaotic_trajectories = chaotic_trajectories.view(batch_size, trajectory_length, 3)
            
        if return_intermediates:
            intermediates['chaotic_trajectories'] = chaotic_trajectories
        
        # Step 4: Attractor pooling
        pooled_features = self.attractor_pooling(chaotic_trajectories)
        if return_intermediates:
            intermediates['pooled_features'] = pooled_features
        
        # Step 5: Speaker embedding
        speaker_embeddings = self.speaker_embedding(pooled_features)
        if return_intermediates:
            intermediates['speaker_embeddings'] = speaker_embeddings
        
        # Step 6: Classification
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
        """
        Compute training loss.
        
        Args:
            logits: Classification logits
            labels: True labels
            loss_weights: Optional weights for different loss components
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Main classification loss
        losses['classification'] = self.cross_entropy_loss(logits, labels)
        
        # Optional regularization losses
        if loss_weights:
            total_loss = losses['classification'] * loss_weights.get('classification', 1.0)
            
            # Add L2 regularization if specified
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
        """
        Make predictions on input audio.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            predicted_classes: Predicted speaker IDs
            confidence_scores: Confidence scores for predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio)
            probabilities = F.softmax(logits, dim=1)
            confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
        
        return predicted_classes, confidence_scores
    
    def extract_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embeddings for the input audio.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Speaker embeddings
        """
        self.eval()
        with torch.no_grad():
            _, intermediates = self.forward(audio, return_intermediates=True)
        
        return intermediates['speaker_embeddings']
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'components': {
                'phase_space': type(self.phase_space).__name__,
                'mlsa_extractor': type(self.mlsa_extractor).__name__,
                'rqa_extractor': type(self.rqa_extractor).__name__,
                'chaotic_embedding': type(self.chaotic_embedding).__name__,
                'attractor_pooling': type(self.attractor_pooling).__name__,
                'speaker_embedding': type(self.speaker_embedding).__name__,
                'classifier': type(self.classifier).__name__
            }
        }
        
        return info
    
    def save_checkpoint(self, filepath: str, additional_info: Optional[Dict] = None):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            additional_info: Additional information to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'training_stats': self.training_stats
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath: str, strict: bool = True) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Checkpoint information
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
            
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
        """
        Process a batch of audio signals.
        
        Args:
            audio_batch: [batch_size, num_samples]
            
        Returns:
            phase_space_batch: [batch_size, time_steps, embedding_dim]
        """
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
                    # Fallback
                    embedded_tensor = self._create_fallback_embedding(audio_batch[i])
                    
            except Exception as e:
                # Robust fallback
                embedded_tensor = self._create_fallback_embedding(audio_batch[i])
            
            results.append(embedded_tensor)
        
        # Standardize length
        standardized = self._standardize_length(results)
        return standardized.to(self.device)
    
    def _create_fallback_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Create simple fallback phase space embedding."""
        # Use time-delay embedding directly
        delay = 1
        dim = 3
        
        audio_np = audio.cpu().numpy()
        n_points = len(audio_np) - (dim - 1) * delay
        
        if n_points <= 0:
            # Audio too short, return minimal embedding
            return torch.zeros(self.fixed_output_length, dim)
        
        embedded = np.zeros((n_points, dim))
        for i in range(dim):
            embedded[:, i] = audio_np[i*delay : i*delay + n_points]
        
        return torch.from_numpy(embedded).float()
    
    def _standardize_length(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Standardize all embeddings to same length."""
        # Option 1: Pad to max length
        max_len = max(emb.shape[0] for emb in embeddings)
        target_len = min(max_len, self.fixed_output_length)
        
        standardized = []
        for emb in embeddings:
            if emb.shape[0] > target_len:
                # Truncate
                emb = emb[:target_len]
            elif emb.shape[0] < target_len:
                # Pad
                padding = torch.zeros(target_len - emb.shape[0], emb.shape[1])
                emb = torch.cat([emb, padding], dim=0)
            
            standardized.append(emb)
        
        return torch.stack(standardized)

# Factory function for easy model creation
def create_chaotic_speaker_network(config: Dict) -> ChaoticSpeakerRecognitionNetwork:
    """
    Factory function to create chaotic speaker recognition network.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ChaoticSpeakerRecognitionNetwork instance
    """
    return ChaoticSpeakerRecognitionNetwork(**config)


if __name__ == "__main__":
    # Test the complete chaotic network
    print("Testing Chaotic Speaker Recognition Network...")
    
    # Create test configuration
    config = {
        'sample_rate': 16000,
        'embedding_dim': 8,
        'mlsa_scales': 3,
        'evolution_time': 0.1,
        'time_step': 0.02,
        'pooling_type': 'comprehensive',
        'speaker_embedding_dim': 64,
        'num_speakers': 10,
        'classifier_type': 'cosine',
        'device': 'cpu'
    }
    
    # Create network
    network = create_chaotic_speaker_network(config)
    
    # Print model information
    model_info = network.get_model_info()
    print(f"Model created successfully!")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Test forward pass
    batch_size = 4
    sequence_length = 1000  # Audio samples or frames
    test_audio = torch.randn(batch_size, sequence_length)
    test_labels = torch.randint(0, config['num_speakers'], (batch_size,))
    
    print(f"\nTesting forward pass:")
    print(f"Input audio shape: {test_audio.shape}")
    print(f"Labels: {test_labels}")
    
    # Forward pass with intermediates
    network.eval()
    with torch.no_grad():
        logits, intermediates = network(test_audio, return_intermediates=True)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Intermediate shapes:")
    for name, tensor in intermediates.items():
        print(f"  {name}: {tensor.shape}")
    
    # Test prediction
    predicted_classes, confidence_scores = network.predict(test_audio)
    print(f"\nPredictions: {predicted_classes}")
    print(f"Confidence scores: {confidence_scores}")
    
    # Test loss computation
    network.train()
    losses = network.compute_loss(logits, test_labels)
    print(f"\nLoss: {losses['total'].item():.4f}")
    
    # Test embedding extraction
    embeddings = network.extract_embeddings(test_audio)
    print(f"Speaker embeddings shape: {embeddings.shape}")
    
    print("\nChaotic Speaker Recognition Network test completed!")