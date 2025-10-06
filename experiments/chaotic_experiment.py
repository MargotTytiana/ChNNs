import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

import sys
from pathlib import Path

def setup_module_imports(current_file: str = __file__):
    """Setup imports for current module."""
    try:
        from setup_imports import setup_project_imports
        return setup_project_imports(current_file), True
    except ImportError:
        # Fallback: manual path setup
        current_dir = Path(current_file).resolve().parent  # experiments
        project_root = current_dir.parent  # experiments -> Model
        
        paths_to_add = [
            str(project_root),
            str(project_root / 'core'),
            str(project_root / 'models'), 
            str(project_root / 'features'),
            str(project_root / 'data'),
            str(project_root / 'utils'),
            str(project_root / 'evaluation'),
        ]
        
        for path in paths_to_add:
            if Path(path).exists() and path not in sys.path:
                sys.path.insert(0, path)
        
        return project_root, False

# Setup imports
PROJECT_ROOT, USING_IMPORT_MANAGER = setup_module_imports()

# =============================================================================
from base_experiment import BaseExperiment
from hybrid_models import TraditionalMLPBaseline, HybridModelManager  
from mlp_classifier import MLPClassifier
from traditional_features import MelSpectrogramExtractor, MFCCExtractor
from dataset_loader import create_speaker_dataloaders, LibriSpeechChaoticDataset
from chaotic_network import ChaoticSpeakerRecognitionNetwork

class ChaoticExperiment(BaseExperiment):
    """
    Chaotic Network Experiment for robust speaker recognition using chaos theory.
    
    This experiment implements the complete chaotic neural network pipeline:
    1. Phase space reconstruction
    2. Chaotic feature extraction (MLSA + RQA)
    3. Chaotic embedding layer
    4. Strange attractor pooling
    5. Speaker embedding and classification
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: str = 'chaotic_experiment',
        output_dir: str = './experiments/outputs',
        device: str = 'auto',
        seed: int = 42
    ):
        """
        Initialize Chaotic Network Experiment.
        
        Args:
            config: Experiment configuration
            experiment_name: Name of the experiment  
            output_dir: Output directory for results
            device: Device to use ('auto', 'cpu', 'cuda')
            seed: Random seed
        """
        super().__init__(config, experiment_name, output_dir, device, seed)
        
        # Validate chaotic-specific config
        self._validate_config()
        
        # Initialize chaotic components
        self.feature_visualizer = None
        self.chaotic_analyzer = None
        
        # Experiment tracking
        self.chaotic_metrics = {
            'attractor_dimensions': [],
            'lyapunov_exponents': [],
            'embedding_quality': []
        }
        
        self.logger.info(f"Initialized chaotic network experiment with {config['chaotic_system']} system")
    
    def _validate_config(self):
        """Validate chaotic experiment configuration."""
        required_keys = ['num_speakers', 'batch_size', 'chaotic_system']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate chaotic system type
        valid_systems = ['lorenz', 'rossler', 'mackey_glass', 'chua']
        if self.config['chaotic_system'] not in valid_systems:
            raise ValueError(f"Invalid chaotic_system. Must be one of: {valid_systems}")
        
        # Set default values for chaotic network
        self.config.setdefault('learning_rate', 0.0005)  # Lower LR for chaotic systems
        self.config.setdefault('weight_decay', 1e-5)
        
        # Phase space reconstruction defaults
        self.config.setdefault('embedding_dim', 10)
        self.config.setdefault('delay_method', 'autocorr')
        
        # Chaotic feature extraction defaults  
        self.config.setdefault('mlsa_scales', 5)
        self.config.setdefault('rqa_radius_ratio', 0.1)
        
        # Chaotic embedding defaults
        self.config.setdefault('evolution_time', 0.5)
        self.config.setdefault('time_step', 0.01)
        self.config.setdefault('coupling_strength', 1.0)
        self.config.setdefault('noise_level', 0.001)
        
        # Attractor pooling defaults
        self.config.setdefault('pooling_type', 'comprehensive')
        self.config.setdefault('correlation_radii', None)
        
        # Speaker embedding defaults
        self.config.setdefault('speaker_embedding_dim', 128)
        self.config.setdefault('embedding_hidden_dims', [64, 32])
        
        # Classification defaults
        self.config.setdefault('classifier_type', 'cosine')
        self.config.setdefault('temperature', 30.0)
        self.config.setdefault('margin', 0.35)
        
        # Audio processing defaults
        self.config.setdefault('sample_rate', 16000)
        self.config.setdefault('frame_length', 400)
        self.config.setdefault('hop_length', 160)
        self.config.setdefault('max_audio_length', 3.0)
        
        # Training specific defaults
        self.config.setdefault('gradient_clipping', 1.0)  # Important for chaotic systems
        self.config.setdefault('adaptive_embedding', False)
    
    def create_model(self) -> nn.Module:
        """Create chaotic network model based on configuration."""
        model_type = self.config.get('model_type', 'full_chaotic')

        # CRITICAL: Ensure num_speakers matches actual dataset
        # This should be set after dataloaders are created
        if not hasattr(self, 'actual_num_speakers'):
            # If we haven't created dataloaders yet, use config value
            num_speakers = self.config.get('num_speakers', 251)
            self.logger.warning(f"Using num_speakers from config: {num_speakers}")
        else:
            num_speakers = self.actual_num_speakers
            self.logger.info(f"Using actual num_speakers from dataset: {num_speakers}")
            # Update config to reflect reality
            self.config['num_speakers'] = num_speakers
        
        if model_type == 'full_chaotic':
            # Complete chaotic network
            if ChaoticSpeakerRecognitionNetwork is not None:
                model = ChaoticSpeakerRecognitionNetwork(
                    # Audio processing parameters
                    sample_rate=self.config['sample_rate'],
                    frame_length=self.config['frame_length'],
                    hop_length=self.config['hop_length'],
                    
                    # Phase space reconstruction  
                    embedding_dim=self.config['embedding_dim'],
                    delay_method=self.config['delay_method'],
                    
                    # Chaotic features
                    mlsa_scales=self.config['mlsa_scales'],
                    rqa_radius_ratio=self.config['rqa_radius_ratio'],
                    
                    # Chaotic embedding
                    chaotic_system=self.config['chaotic_system'],
                    evolution_time=self.config['evolution_time'],
                    time_step=self.config['time_step'],
                    
                    # Attractor pooling
                    pooling_type=self.config['pooling_type'],
                    
                    # Speaker embedding
                    speaker_embedding_dim=self.config['speaker_embedding_dim'],
                    
                    # Classification
                    num_speakers=num_speakers,
                    classifier_type=self.config['classifier_type'],
                    
                    device=self.device
                )
            else:
                # Mock implementation
                model = MockChaoticNetwork(self.config)
        
        elif model_type == 'traditional_chaotic':
            # Traditional features + chaotic processing
            if TraditionalChaoticHybrid is not None:
                model = TraditionalChaoticHybrid(
                    feature_type=self.config.get('feature_type', 'mel'),
                    n_mels=self.config.get('n_mels', 80),
                    n_mfcc=self.config.get('n_mfcc', 13),
                    sample_rate=self.config['sample_rate'],
                    evolution_time=self.config['evolution_time'],
                    pooling_type=self.config['pooling_type'],
                    speaker_embedding_dim=self.config['speaker_embedding_dim'],
                    num_speakers=self.config['num_speakers'],
                    classifier_type=self.config['classifier_type'],
                    device=self.device
                )
            else:
                model = MockChaoticNetwork(self.config)
        
        elif model_type == 'chaotic_mlp':
            # Chaotic features + MLP classifier
            if ChaoticMLPHybrid is not None:
                model = ChaoticMLPHybrid(
                    sample_rate=self.config['sample_rate'],
                    embedding_dim=self.config['embedding_dim'],
                    mlsa_scales=self.config['mlsa_scales'],
                    rqa_radius_ratio=self.config['rqa_radius_ratio'],
                    hidden_dims=self.config.get('mlp_hidden_dims', [128, 64, 32]),
                    dropout_rate=self.config.get('dropout_rate', 0.2),
                    num_speakers=self.config['num_speakers'],
                    device=self.device
                )
            else:
                model = MockChaoticNetwork(self.config)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for chaotic network training."""
        if create_speaker_dataloaders is not None:
            # Use real data loading with chaotic preprocessing
            train_loader, val_loader, test_loader = create_speaker_dataloaders(
                data_dir=self.config.get('data_dir', './data'),
                batch_size=self.config['batch_size'],
                sample_rate=self.config['sample_rate'],
                max_length=self.config['max_audio_length'],
                num_workers=self.config.get('num_workers', 4),
                train_split=self.config.get('train_split', 0.7),
                val_split=self.config.get('val_split', 0.15),
                seed=self.seed,
                # Chaotic-specific preprocessing
                # apply_chaotic_preprocessing=True,
                # embedding_dim=self.config['embedding_dim']
            )
            # CRITICAL: Extract actual number of speakers from dataset
            # Assuming the dataset has a way to report this
            try:
                if hasattr(train_loader.dataset, 'num_speakers'):
                    self.actual_num_speakers = train_loader.dataset.num_speakers
                elif hasattr(train_loader.dataset, 'num_classes'):
                    self.actual_num_speakers = train_loader.dataset.num_classes
                else:
                    # Try to infer from labels
                    all_labels = []
                    for _, labels in train_loader:
                        all_labels.extend(labels.tolist())
                    self.actual_num_speakers = len(set(all_labels))
                
                self.logger.info(f"Detected {self.actual_num_speakers} speakers in dataset")
                
            except Exception as e:
                self.logger.warning(f"Could not detect num_speakers: {e}")
                # Use a safe default
                self.actual_num_speakers = 251
        else:
            # Create mock data loaders
            self.logger.warning("Using mock data loaders for testing")
            
            train_dataset = MockDataset(1000, self.config['num_speakers'])
            val_dataset = MockDataset(200, self.config['num_speakers'])
            test_dataset = MockDataset(200, self.config['num_speakers'])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=0
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=0
            )
        
        return train_loader, val_loader, test_loader
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer optimized for chaotic networks."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw').lower()
        
        if optimizer_type == 'adamw':
            # Get additional parameters from config
            extra_params = optimizer_config.get('params', {})
            
            # Create base parameters
            optimizer_params = {
                'lr': self.config['learning_rate'],
                'weight_decay': self.config['weight_decay']
            }
            
            # Add betas only if not in extra_params
            if 'betas' not in extra_params:
                optimizer_params['betas'] = (0.9, 0.999)
            
            # Merge with extra parameters (extra_params will override defaults)
            optimizer_params.update(extra_params)
            
            optimizer = optim.AdamW(
                model.parameters(),
                **optimizer_params
            )
            
        elif optimizer_type == 'adam':
            extra_params = optimizer_config.get('params', {})
            optimizer_params = {
                'lr': self.config['learning_rate'],
                'weight_decay': self.config['weight_decay']
            }
            optimizer_params.update(extra_params)
            
            optimizer = optim.Adam(
                model.parameters(),
                **optimizer_params
            )
            
        elif optimizer_type == 'sgd':
            extra_params = optimizer_config.get('params', {})
            optimizer_params = {
                'lr': self.config['learning_rate'],
                'weight_decay': self.config['weight_decay'],
                'momentum': extra_params.get('momentum', 0.9),
                'nesterov': extra_params.get('nesterov', True)
            }
            
            optimizer = optim.SGD(
                model.parameters(),
                **optimizer_params
            )
            
        else:
            self.logger.warning(f"Unknown optimizer type: {optimizer_type}, using AdamW")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        
        return optimizer
    
    
    def forward_pass(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward pass through chaotic network.
        
        Args:
            batch: Tuple of (audio, speaker_labels)
            training: Whether in training mode
            
        Returns:
            Tuple of (loss, predictions, targets)
        """
        audio, targets = batch
        
        # Forward pass through chaotic network
        if hasattr(self.model, 'forward') and 'labels' in self.model.forward.__code__.co_varnames:
            # Full chaotic network with labels support
            logits = self.model(audio, labels=targets if training else None)
        else:
            # Standard forward pass
            logits = self.model(audio)
        
        # Compute loss
        loss = self.criterion(logits, targets)
        
        # Get predictions
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
        
        return loss, predictions, targets
    
    def calculate_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        loss: float
    ) -> Dict[str, float]:
        """Calculate chaotic network specific metrics."""
        metrics = super().calculate_metrics(predictions, targets, loss)
        
        # Add chaotic-specific metrics
        with torch.no_grad():
            # Prediction confidence analysis
            if hasattr(self.model, 'predict'):
                # Get confidence scores
                sample_audio = torch.randn(1, 8000).to(self.device)  # Sample for confidence analysis
                try:
                    _, confidence = self.model.predict(sample_audio)
                    metrics['avg_confidence'] = confidence.mean().item()
                except:
                    pass  # Skip if predict method fails
            
            # Embedding quality metrics
            if hasattr(self.model, 'extract_embeddings'):
                try:
                    sample_audio = torch.randn(min(16, predictions.shape[0]), 8000).to(self.device)
                    embeddings = self.model.extract_embeddings(sample_audio)
                    
                    # Embedding diversity (average pairwise distance)
                    if embeddings.shape[0] > 1:
                        pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
                        # Exclude diagonal (distance to self)
                        mask = ~torch.eye(embeddings.shape[0], dtype=bool, device=self.device)
                        avg_distance = pairwise_distances[mask].mean().item()
                        metrics['embedding_diversity'] = avg_distance
                    
                    # Embedding norm consistency
                    embedding_norms = torch.norm(embeddings, dim=1)
                    metrics['embedding_norm_mean'] = embedding_norms.mean().item()
                    metrics['embedding_norm_std'] = embedding_norms.std().item()
                    
                except:
                    pass  # Skip if extraction fails
        
        return metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Train epoch with chaotic network specific monitoring."""
        # Call parent train_epoch
        epoch_metrics = super().train_epoch()
        
        # Add chaotic system monitoring
        if hasattr(self.model, 'forward') and hasattr(self.model, 'chaotic_embedding'):
            self._monitor_chaotic_dynamics()
        
        return epoch_metrics
    
    def _monitor_chaotic_dynamics(self):
        """Monitor chaotic dynamics during training."""
        try:
            # Sample a batch for analysis
            sample_batch = next(iter(self.val_loader))
            sample_audio, _ = sample_batch
            sample_audio = sample_audio[:4].to(self.device)  # Small batch for analysis
            
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    _, intermediates = self.model(sample_audio, return_intermediates=True)
                    
                    # Analyze chaotic trajectories
                    if 'chaotic_trajectories' in intermediates:
                        trajectories = intermediates['chaotic_trajectories']
                        
                        # Calculate basic trajectory statistics
                        trajectory_std = torch.std(trajectories).item()
                        trajectory_range = (torch.max(trajectories) - torch.min(trajectories)).item()
                        
                        self.chaotic_metrics['trajectory_std'] = trajectory_std
                        self.chaotic_metrics['trajectory_range'] = trajectory_range
                        
                        # Log to tensorboard
                        self.writer.add_scalar('Chaotic/trajectory_std', trajectory_std, self.state.epoch)
                        self.writer.add_scalar('Chaotic/trajectory_range', trajectory_range, self.state.epoch)
                    
                    # Analyze pooled features
                    if 'pooled_features' in intermediates:
                        pooled = intermediates['pooled_features']
                        feature_diversity = torch.std(pooled, dim=0).mean().item()
                        
                        self.writer.add_scalar('Chaotic/feature_diversity', feature_diversity, self.state.epoch)
                        
        except Exception as e:
            self.logger.debug(f"Chaotic monitoring failed: {e}")
    
    def analyze_chaotic_features(self, num_samples: int = 100) -> Dict[str, Any]:
        """Comprehensive analysis of chaotic features and dynamics."""
        self.logger.info("Analyzing chaotic features and dynamics...")
        
        analysis_results = {}
        
        # Collect samples for analysis
        self.model.eval()
        sample_data = []
        sample_labels = []
        
        with torch.no_grad():
            samples_collected = 0
            for batch in self.test_loader:
                if samples_collected >= num_samples:
                    break
                    
                audio, labels = batch
                audio = audio.to(self.device)
                
                batch_size = min(audio.shape[0], num_samples - samples_collected)
                audio = audio[:batch_size]
                labels = labels[:batch_size]
                
                sample_data.append(audio)
                sample_labels.extend(labels.tolist())
                samples_collected += batch_size
            
            if sample_data:
                sample_audio = torch.cat(sample_data, dim=0)
                
                # Extract intermediate representations
                if hasattr(self.model, 'forward'):
                    try:
                        _, intermediates = self.model(sample_audio, return_intermediates=True)
                        
                        # Analyze each stage
                        analysis_results['phase_space'] = self._analyze_phase_space(
                            intermediates.get('phase_space')
                        )
                        analysis_results['chaotic_features'] = self._analyze_chaotic_features_dist(
                            intermediates.get('chaotic_features')
                        )
                        analysis_results['trajectories'] = self._analyze_trajectories(
                            intermediates.get('chaotic_trajectories')
                        )
                        analysis_results['pooled_features'] = self._analyze_pooled_features(
                            intermediates.get('pooled_features')
                        )
                        analysis_results['embeddings'] = self._analyze_speaker_embeddings(
                            intermediates.get('speaker_embeddings'),
                            sample_labels
                        )
                        
                    except Exception as e:
                        self.logger.warning(f"Feature analysis failed: {e}")
                        analysis_results['error'] = str(e)
        
        # Save analysis results
        analysis_file = os.path.join(self.results_dir, 'chaotic_analysis.json')
        with open(analysis_file, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_results = self._convert_tensors_for_json(analysis_results)
            json.dump(json_results, f, indent=2)
        
        self.logger.info("Chaotic analysis completed")
        return analysis_results
    
    def _analyze_phase_space(self, phase_space_data: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Analyze phase space reconstruction quality."""
        if phase_space_data is None:
            return {'error': 'No phase space data available'}
        
        analysis = {}
        
        # Basic statistics
        analysis['mean'] = phase_space_data.mean().item()
        analysis['std'] = phase_space_data.std().item()
        analysis['min'] = phase_space_data.min().item()
        analysis['max'] = phase_space_data.max().item()
        
        # Dimensionality analysis
        analysis['shape'] = list(phase_space_data.shape)
        analysis['effective_dim'] = self._estimate_effective_dimension(phase_space_data)
        
        return analysis
    
    def _analyze_chaotic_features_dist(self, features: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Analyze distribution of chaotic features."""
        if features is None:
            return {'error': 'No chaotic features available'}
        
        analysis = {}
        
        # Feature statistics per dimension
        analysis['per_dim_stats'] = []
        for dim in range(features.shape[-1]):
            dim_data = features[..., dim]
            dim_stats = {
                'mean': dim_data.mean().item(),
                'std': dim_data.std().item(),
                'min': dim_data.min().item(),
                'max': dim_data.max().item()
            }
            analysis['per_dim_stats'].append(dim_stats)
        
        # Overall feature diversity
        analysis['feature_diversity'] = torch.std(features, dim=0).mean().item()
        analysis['feature_range'] = (features.max() - features.min()).item()
        
        return analysis
    
    def _analyze_trajectories(self, trajectories: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Analyze chaotic trajectory properties."""
        if trajectories is None:
            return {'error': 'No trajectory data available'}
        
        analysis = {}
        
        # Trajectory statistics
        analysis['num_trajectories'] = trajectories.shape[0]
        analysis['trajectory_length'] = trajectories.shape[1] if len(trajectories.shape) > 1 else 0
        analysis['state_dimension'] = trajectories.shape[2] if len(trajectories.shape) > 2 else 0
        
        # Trajectory properties
        trajectory_norms = torch.norm(trajectories, dim=-1)
        analysis['avg_trajectory_norm'] = trajectory_norms.mean().item()
        analysis['trajectory_norm_std'] = trajectory_norms.std().item()
        
        # Path length analysis
        if len(trajectories.shape) == 3 and trajectories.shape[1] > 1:
            diffs = torch.diff(trajectories, dim=1)
            path_lengths = torch.norm(diffs, dim=-1).sum(dim=1)
            analysis['avg_path_length'] = path_lengths.mean().item()
            analysis['path_length_std'] = path_lengths.std().item()
        
        return analysis
    
    def _analyze_pooled_features(self, pooled_features: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Analyze pooled attractor features."""
        if pooled_features is None:
            return {'error': 'No pooled features available'}
        
        analysis = {}
        
        # Feature dimensionality
        analysis['feature_dim'] = pooled_features.shape[-1] if len(pooled_features.shape) > 0 else 0
        
        # Feature statistics
        analysis['mean'] = pooled_features.mean().item()
        analysis['std'] = pooled_features.std().item()
        
        # Per-feature analysis
        if len(pooled_features.shape) > 1:
            per_feature_std = torch.std(pooled_features, dim=0)
            analysis['per_feature_std'] = per_feature_std.tolist()
            analysis['feature_diversity'] = per_feature_std.mean().item()
        
        return analysis
    
    def _analyze_speaker_embeddings(
        self, 
        embeddings: Optional[torch.Tensor], 
        labels: List[int]
    ) -> Dict[str, Any]:
        """Analyze speaker embedding quality."""
        if embeddings is None:
            return {'error': 'No speaker embeddings available'}
        
        analysis = {}
        
        # Embedding properties
        analysis['embedding_dim'] = embeddings.shape[-1] if len(embeddings.shape) > 0 else 0
        analysis['num_embeddings'] = embeddings.shape[0] if len(embeddings.shape) > 0 else 0
        
        # Embedding norms (should be normalized)
        embedding_norms = torch.norm(embeddings, dim=-1)
        analysis['norm_mean'] = embedding_norms.mean().item()
        analysis['norm_std'] = embedding_norms.std().item()
        
        # Inter-class and intra-class distances
        if len(set(labels)) > 1 and len(embeddings.shape) > 1:
            unique_labels = list(set(labels))
            intra_class_distances = []
            inter_class_distances = []
            
            for label in unique_labels:
                label_indices = [i for i, l in enumerate(labels) if l == label]
                if len(label_indices) > 1:
                    label_embeddings = embeddings[label_indices]
                    # Intra-class distances
                    pairwise_dist = torch.cdist(label_embeddings, label_embeddings, p=2)
                    mask = ~torch.eye(len(label_indices), dtype=bool)
                    intra_class_distances.extend(pairwise_dist[mask].tolist())
                
                # Inter-class distances
                other_indices = [i for i, l in enumerate(labels) if l != label]
                if other_indices and label_indices:
                    label_embeddings = embeddings[label_indices]
                    other_embeddings = embeddings[other_indices]
                    inter_dist = torch.cdist(label_embeddings, other_embeddings, p=2)
                    inter_class_distances.extend(inter_dist.flatten().tolist())
            
            if intra_class_distances:
                analysis['intra_class_distance'] = {
                    'mean': np.mean(intra_class_distances),
                    'std': np.std(intra_class_distances)
                }
            
            if inter_class_distances:
                analysis['inter_class_distance'] = {
                    'mean': np.mean(inter_class_distances),
                    'std': np.std(inter_class_distances)
                }
            
            # Separation ratio
            if intra_class_distances and inter_class_distances:
                separation_ratio = np.mean(inter_class_distances) / np.mean(intra_class_distances)
                analysis['separation_ratio'] = separation_ratio
        
        return analysis
    
    def _estimate_effective_dimension(self, data: torch.Tensor) -> float:
        """Estimate effective dimension of data using PCA."""
        try:
            # Flatten data for PCA analysis
            if len(data.shape) > 2:
                data_flat = data.view(data.shape[0], -1)
            else:
                data_flat = data
            
            # Center the data
            data_centered = data_flat - data_flat.mean(dim=0, keepdim=True)
            
            # Compute SVD
            U, S, V = torch.svd(data_centered)
            
            # Compute explained variance ratio
            explained_variance = S ** 2
            total_variance = explained_variance.sum()
            explained_ratio = explained_variance / total_variance
            
            # Find effective dimension (95% variance)
            cumsum_ratio = torch.cumsum(explained_ratio, dim=0)
            effective_dim = (cumsum_ratio < 0.95).sum().item() + 1
            
            return min(effective_dim, data_flat.shape[1])
            
        except Exception:
            return data.shape[-1] if len(data.shape) > 0 else 0
    
    def _convert_tensors_for_json(self, obj: Any) -> Any:
        """Convert tensors to lists for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_tensors_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_for_json(item) for item in obj]
        else:
            return obj
    
    def visualize_chaotic_dynamics(self, save_plots: bool = True) -> Dict[str, str]:
        """Create visualizations of chaotic dynamics."""
        if not save_plots:
            return {}
        
        self.logger.info("Creating chaotic dynamics visualizations...")
        
        plot_paths = {}
        plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            # Sample data for visualization
            sample_batch = next(iter(self.test_loader))
            sample_audio, sample_labels = sample_batch
            sample_audio = sample_audio[:8].to(self.device)
            
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    _, intermediates = self.model(sample_audio, return_intermediates=True)
                    
                    # Plot chaotic trajectories
                    if 'chaotic_trajectories' in intermediates:
                        trajectory_plot = self._plot_trajectories(
                            intermediates['chaotic_trajectories'],
                            os.path.join(plots_dir, 'chaotic_trajectories.png')
                        )
                        if trajectory_plot:
                            plot_paths['trajectories'] = trajectory_plot
                    
                    # Plot feature distributions
                    if 'pooled_features' in intermediates:
                        feature_plot = self._plot_feature_distributions(
                            intermediates['pooled_features'],
                            os.path.join(plots_dir, 'feature_distributions.png')
                        )
                        if feature_plot:
                            plot_paths['features'] = feature_plot
                    
                    # Plot embeddings
                    if 'speaker_embeddings' in intermediates:
                        embedding_plot = self._plot_embeddings(
                            intermediates['speaker_embeddings'],
                            sample_labels,
                            os.path.join(plots_dir, 'speaker_embeddings.png')
                        )
                        if embedding_plot:
                            plot_paths['embeddings'] = embedding_plot
        
        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")
        
        return plot_paths
    
    def _plot_trajectories(self, trajectories: torch.Tensor, save_path: str) -> Optional[str]:
        """Plot chaotic trajectories in 3D."""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            trajectories_cpu = trajectories.cpu().numpy()
            
            fig = plt.figure(figsize=(15, 5))
            
            # Plot first 3 trajectories
            for i in range(min(3, trajectories_cpu.shape[0])):
                ax = fig.add_subplot(1, 3, i+1, projection='3d')
                traj = trajectories_cpu[i]
                
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=0.8)
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=50, label='Start')
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=50, label='End')
                
                ax.set_title(f'Trajectory {i+1}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            self.logger.warning(f"Trajectory plotting failed: {e}")
            return None
    
    def _plot_feature_distributions(self, features: torch.Tensor, save_path: str) -> Optional[str]:
        """Plot distribution of pooled features."""
        try:
            
            features_cpu = features.cpu().numpy()
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i in range(min(features_cpu.shape[1], len(axes))):
                ax = axes[i]
                ax.hist(features_cpu[:, i], bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'Feature {i+1}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(features_cpu.shape[1], len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except ImportError:
            return None
        except Exception as e:
            self.logger.warning(f"Feature distribution plotting failed: {e}")
            return None
    
    def _plot_embeddings(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor, 
        save_path: str
    ) -> Optional[str]:
        """Plot speaker embeddings using t-SNE."""
        try:
            from sklearn.manifold import TSNE
            
            embeddings_cpu = embeddings.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            
            # Use t-SNE for dimensionality reduction
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_cpu)-1))
            embeddings_2d = tsne.fit_transform(embeddings_cpu)
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            unique_labels = np.unique(labels_cpu)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels_cpu == label
                ax.scatter(
                    embeddings_2d[mask, 0], 
                    embeddings_2d[mask, 1], 
                    c=[colors[i]], 
                    label=f'Speaker {label}',
                    alpha=0.7,
                    s=50
                )
            
            ax.set_title('Speaker Embeddings (t-SNE)')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except ImportError:
            self.logger.warning("Scikit-learn not available for t-SNE plotting")
            return None
        except Exception as e:
            self.logger.warning(f"Embedding plotting failed: {e}")
            return None
    
    def run_chaotic_analysis(self) -> Dict[str, Any]:
        """Run complete chaotic network analysis."""
        self.logger.info("Running comprehensive chaotic network analysis...")
        
        analysis_results = {}
        
        # Model complexity analysis
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        analysis_results['model_complexity'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }
        
        # Chaotic system configuration
        analysis_results['chaotic_config'] = {
            'system_type': self.config['chaotic_system'],
            'evolution_time': self.config['evolution_time'],
            'time_step': self.config['time_step'],
            'embedding_dim': self.config['embedding_dim'],
            'pooling_type': self.config['pooling_type']
        }
        
        # Feature analysis
        feature_analysis = self.analyze_chaotic_features(num_samples=200)
        analysis_results['feature_analysis'] = feature_analysis
        
        # Create visualizations
        visualization_paths = self.visualize_chaotic_dynamics(save_plots=True)
        analysis_results['visualizations'] = visualization_paths
        
        # Training dynamics
        analysis_results['training_dynamics'] = {
            'best_epoch': self.state.best_epoch,
            'best_metric': self.state.best_metric,
            'final_training_loss': self.state.train_losses[-1] if self.state.train_losses else None,
            'chaotic_metrics': self.chaotic_metrics
        }
        
        # Save comprehensive analysis
        analysis_file = os.path.join(self.results_dir, 'comprehensive_chaotic_analysis.json')
        with open(analysis_file, 'w') as f:
            json_results = self._convert_tensors_for_json(analysis_results)
            json.dump(json_results, f, indent=2)
        
        self.logger.info("Chaotic network analysis completed")
        return analysis_results


def create_chaotic_experiments(base_config: Dict[str, Any]) -> Dict[str, ChaoticExperiment]:
    """
    Create multiple chaotic experiments with different configurations.
    
    Args:
        base_config: Base configuration for experiments
        
    Returns:
        Dictionary of chaotic experiments
    """
    chaotic_systems = ['lorenz', 'rossler']
    model_types = ['full_chaotic', 'traditional_chaotic']
    
    experiments = {}
    
    for system in chaotic_systems:
        for model_type in model_types:
            config = base_config.copy()
            config['chaotic_system'] = system
            config['model_type'] = model_type
            
            # Adjust parameters based on system
            if system == 'lorenz':
                config['evolution_time'] = 0.5
                config['coupling_strength'] = 1.0
            elif system == 'rossler':
                config['evolution_time'] = 0.8
                config['coupling_strength'] = 0.8
            
            experiment_name = f'chaotic_{system}_{model_type}'
            experiment = ChaoticExperiment(
                config=config,
                experiment_name=experiment_name,
                seed=config.get('seed', 42)
            )
            
            experiments[f'{system}_{model_type}'] = experiment
    
    return experiments


if __name__ == "__main__":
    print(f"✓ Project Root: {PROJECT_ROOT}")
    print(f"✓ Import Manager: {USING_IMPORT_MANAGER}")
    print(f"✓ Module imports successful")
    # Example usage and testing
    
    # Test configuration
    test_config = {
        'chaotic_system': 'lorenz',
        'model_type': 'full_chaotic',
        'num_speakers': 10,
        'batch_size': 8,
        'learning_rate': 0.0005,
        'embedding_dim': 8,
        'mlsa_scales': 3,
        'evolution_time': 0.2,
        'pooling_type': 'comprehensive',
        'speaker_embedding_dim': 64,
        'classifier_type': 'cosine',
        'sample_rate': 16000,
        'primary_metric': 'accuracy',
        'log_interval': 5
    }
    
    print("Testing ChaoticExperiment...")
    
    # Test single experiment
    experiment = ChaoticExperiment(
        config=test_config,
        experiment_name='test_chaotic_lorenz'
    )
    
    print("Setting up experiment...")
    experiment.setup()
    
    print("Running chaotic analysis...")
    analysis = experiment.run_chaotic_analysis()
    print(f"Model parameters: {analysis['model_complexity']['total_parameters']:,}")
    
    print("Training for 2 epochs...")
    experiment.train(num_epochs=2)
    
    print("Chaotic experiment test completed!")
    
    # Test multiple experiments creation
    print("\nTesting multiple chaotic experiments creation...")
    base_config = {
        'num_speakers': 5,
        'batch_size': 4,
        'learning_rate': 0.0005,
        'sample_rate': 16000
    }
    
    chaotic_experiments = create_chaotic_experiments(base_config)
    print(f"Created {len(chaotic_experiments)} chaotic experiments:")
    for name in chaotic_experiments.keys():
        print(f"  - {name}")
    
    print("All tests completed successfully!")