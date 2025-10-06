import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
import numpy as np
import sys
from pathlib import Path

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

def fix_imports():
    current_file = Path(__file__).resolve()
    model_dir = current_file.parent.parent  # experiments -> Model
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

from base_experiment import BaseExperiment
from hybrid_models import TraditionalMLPBaseline, HybridModelManager
from traditional_features import MelSpectrogramExtractor, MFCCExtractor
from dataset_loader import create_speaker_dataloaders, LibriSpeechChaoticDataset

class BaselineExperiment(BaseExperiment):
    """
    Baseline Experiment for traditional speaker recognition methods.
    
    This experiment implements standard approaches using traditional features
    (Mel-spectrogram, MFCC) combined with MLP classifiers as baseline 
    comparisons for the chaotic neural network approach.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: str = 'baseline_experiment',
        output_dir: str = './experiments/outputs',
        device: str = 'auto',
        seed: int = 42
    ):
        """
        Initialize Baseline Experiment.
        
        Args:
            config: Experiment configuration
            experiment_name: Name of the experiment
            output_dir: Output directory for results
            device: Device to use ('auto', 'cpu', 'cuda')
            seed: Random seed
        """
        super().__init__(config, experiment_name, output_dir, device, seed)
        
        # Validate baseline-specific config
        self._validate_config()
        
        # Initialize feature extractor
        self.feature_extractor = None

        # Add MetricsCalculator Initialization
        from evaluation.metrics import MetricsCalculator
        self.metrics_calculator = MetricsCalculator(
            num_classes=self.config['num_speakers'],
            class_names=[f"speaker_{i}" for i in range(self.config['num_speakers'])]
        )
        
        self.logger.info(f"Initialized {config['baseline_type']} baseline experiment")
    
    def _validate_config(self):
        """Validate baseline experiment configuration."""
        required_keys = ['baseline_type', 'num_speakers', 'batch_size']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate baseline type
        valid_baseline_types = ['mel_mlp', 'mfcc_mlp', 'mel_cnn', 'mfcc_cnn']
        if self.config['baseline_type'] not in valid_baseline_types:
            raise ValueError(f"Invalid baseline_type. Must be one of: {valid_baseline_types}")
        
        # Set default values
        self.config.setdefault('learning_rate', 0.001)
        self.config.setdefault('weight_decay', 1e-4)
        self.config.setdefault('hidden_dims', [256, 128, 64])
        self.config.setdefault('dropout_rate', 0.3)
        self.config.setdefault('use_batch_norm', True)
        
        # Feature-specific defaults
        if 'mel' in self.config['baseline_type']:
            self.config.setdefault('n_mels', 80)
            self.config.setdefault('hop_length', 512)
            self.config.setdefault('n_fft', 2048)
        elif 'mfcc' in self.config['baseline_type']:
            self.config.setdefault('n_mfcc', 13)
            self.config.setdefault('n_fft', 2048)
            self.config.setdefault('hop_length', 512)
        
        self.config.setdefault('sample_rate', 16000)
        self.config.setdefault('max_audio_length', 3.0)  # seconds
    
    def create_model(self) -> nn.Module:
        """Create baseline model based on configuration."""
        baseline_type = self.config['baseline_type']
        num_speakers = self.config['num_speakers']
        
        if baseline_type == 'mel_mlp':
            if TraditionalMLPBaseline is not None:
                model = TraditionalMLPBaseline(
                    feature_type='mel',
                    n_mels=self.config['n_mels'],
                    sample_rate=self.config['sample_rate'],
                    hidden_dims=self.config['hidden_dims'],
                    dropout_rate=self.config['dropout_rate'],
                    use_batch_norm=self.config['use_batch_norm'],
                    num_speakers=num_speakers,
                    device=self.device
                )
            else:
                # Mock implementation
                feature_dim = self.config['n_mels']
                model = nn.Sequential(
                    MockFeatureExtractor(feature_dim),
                    MockMLPClassifier(feature_dim, num_speakers, self.config['hidden_dims'])
                )
                
        elif baseline_type == 'mfcc_mlp':
            if TraditionalMLPBaseline is not None:
                model = TraditionalMLPBaseline(
                    feature_type='mfcc',
                    n_mfcc=self.config['n_mfcc'],
                    sample_rate=self.config['sample_rate'],
                    hidden_dims=self.config['hidden_dims'],
                    dropout_rate=self.config['dropout_rate'],
                    use_batch_norm=self.config['use_batch_norm'],
                    num_speakers=num_speakers,
                    device=self.device
                )
            else:
                # Mock implementation
                feature_dim = self.config['n_mfcc']
                model = nn.Sequential(
                    MockFeatureExtractor(feature_dim),
                    MockMLPClassifier(feature_dim, num_speakers, self.config['hidden_dims'])
                )
                
        elif baseline_type == 'mel_cnn':
            model = self._create_cnn_model('mel')
            
        elif baseline_type == 'mfcc_cnn':
            model = self._create_cnn_model('mfcc')
            
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

        model = model.to(self.device)
        
        for name, param in model.named_parameters():
            if param.device.type != self.device:
                self.logger.warning(f"Parameter {name} not on correct device: {param.device} vs {self.device}")
        
        return model
        
    def _create_cnn_model(self, feature_type: str) -> nn.Module:
        """Create CNN-based baseline model."""
        if feature_type == 'mel':
            feature_dim = self.config['n_mels']
        else:  # mfcc
            feature_dim = self.config['n_mfcc']
        
        # Store config values in local variables for use in the inner class
        sample_rate = self.config['sample_rate']
        outer_config = self.config  # Store reference to outer config
        
        class CNNBaseline(nn.Module):
            def __init__(self, feature_dim, num_classes, dropout_rate=0.3):
                super(CNNBaseline, self).__init__()
                
                # Feature extraction layer
                if feature_type == 'mel':
                    if MelSpectrogramExtractor is not None:
                        self.feature_extractor = MelSpectrogramExtractor(
                            n_mels=feature_dim,
                            sample_rate=sample_rate
                        )
                    else:
                        self.feature_extractor = MockFeatureExtractor(feature_dim)
                else:
                    if MFCCExtractor is not None:
                        self.feature_extractor = MFCCExtractor(
                            n_mfcc=feature_dim,
                            sample_rate=sample_rate
                        )
                    else:
                        self.feature_extractor = MockFeatureExtractor(feature_dim)
                
                # CNN layers for temporal modeling
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout_rate),
                    
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout_rate),
                    
                    nn.Conv1d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Dropout(dropout_rate)
                )
                
                # Classification layers
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                import torch
                
                # Ensure input is on correct device
                device = next(self.parameters()).device
                if x.device != device:
                    x = x.to(device)
                
                # Extract features
                # Check if feature_extractor has extract method (real extractors)
                if hasattr(self.feature_extractor, 'extract'):
                    # Real feature extractors need numpy arrays
                    # Process each sample in the batch
                    batch_size = x.shape[0]
                    feature_list = []
                    
                    for i in range(batch_size):
                        # Convert single audio sample to numpy
                        audio_numpy = x[i].cpu().numpy()
                        
                        # Extract features using the extractor
                        # Output shape: (n_mels, time_frames) or (n_mfcc, time_frames)
                        sample_features = self.feature_extractor.extract(audio_numpy)
                        
                        # Convert back to tensor
                        feature_list.append(torch.from_numpy(sample_features))
                    
                    # Stack features back into a batch
                    # Shape: (batch, feature_dim, time_frames)
                    features = torch.stack(feature_list).to(device)
                else:
                    # For mock or callable extractors (already handle tensors)
                    features = self.feature_extractor(x)
                
                # Ensure features are on correct device
                if features.device != device:
                    features = features.to(device)
            
                # Handle different feature shapes for Conv1d
                # Conv1d expects input shape: (batch, channels, length)
                # where channels = feature_dim and length = time_frames
                
                if len(features.shape) == 2:
                    # Shape: (batch, features) - need to add time dimension
                    # This shouldn't happen with real audio features, but handle it anyway
                    features = features.unsqueeze(-1)  # (batch, features, 1)
                    
                elif len(features.shape) == 3:
                    # Shape should be (batch, feature_dim, time_frames)
                    # Check if we need to transpose
                    # If second dimension is not feature_dim, we need to fix it
                    if features.shape[1] != feature_dim and features.shape[2] == feature_dim:
                        # Wrong order: (batch, time_frames, feature_dim)
                        # Need to transpose to: (batch, feature_dim, time_frames)
                        features = features.transpose(1, 2)
                    # else: already in correct format (batch, feature_dim, time_frames)
                
                # CNN processing
                conv_out = self.conv_layers(features)
                
                # Classification
                logits = self.classifier(conv_out)
                
                return logits
        
        return CNNBaseline(
            feature_dim=feature_dim,
            num_classes=self.config['num_speakers'],
            dropout_rate=self.config['dropout_rate']
        )
    

        
    def _apply_feature_extraction(self, train_loader, val_loader, test_loader):
        """Apply Feature Extraction when Training"""
        
        self.logger.info("Apply Mel Feature Extraction...")
        
        mel_extractor = MelSpectrogramExtractor(
            sample_rate=self.config['sample_rate'],
            n_mels=self.config.get('n_mels', 80),
            n_fft=1024,
            hop_length=512
        )
    
        def extract_features_from_loader(dataloader):
            features_list = []
            labels_list = []
            
            for batch_audio, batch_labels in dataloader:
                batch_features = []
                for audio in batch_audio:
                    mel_features = mel_extractor.extract(audio.numpy())
                    batch_features.append(torch.tensor(mel_features))
                
                features_list.append(torch.stack(batch_features))
                labels_list.append(batch_labels)
            
            all_features = torch.cat(features_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            
            from torch.utils.data import TensorDataset, DataLoader
            dataset = TensorDataset(all_features, all_labels)
            return DataLoader(
                dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=(dataloader.dataset == train_loader.dataset),
                collate_fn=smart_collate_fn
            )
        
        new_train_loader = extract_features_from_loader(train_loader)
        new_val_loader = extract_features_from_loader(val_loader)
        new_test_loader = extract_features_from_loader(test_loader)
        
        return new_train_loader, new_val_loader, new_test_loader

    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for training, validation, and testing."""
        print(f"create_speaker_dataloaders VALUE: {create_speaker_dataloaders}")
        print(f"create_speaker_dataloaders TYPE: {type(create_speaker_dataloaders)}")
        
        try:
            from dataset_loader import create_speaker_dataloaders as manual_import
            print(f"Manually Import: {manual_import}")
        except ImportError as e:
            print(f"✖️ Manually Import Fail: {e}")
        
        import sys
        print(f"Current Python Path: {sys.path[:3]}...")
        
        if 'data_dir' not in self.config:
            model_dir = Path(__file__).parent.parent  # experiments --> Model
            project_root = model_dir.parent  # Model --> project
            librispeech_path = project_root / "dataset" / "train-clean-100" / "LibriSpeech" / "train-clean-100"
            self.config['data_dir'] = str(librispeech_path)
            
        print(f"✅ Data Path: {self.config['data_dir']}")
        
        if create_speaker_dataloaders is not None:
            # Use real data loading
            train_loader, val_loader, test_loader = create_speaker_dataloaders(
                data_dir=self.config['data_dir'],
                batch_size=self.config['batch_size'],
                sample_rate=self.config['sample_rate'],
                max_length=self.config['max_audio_length'],
                num_workers=0,
                train_split=self.config.get('train_split', 0.7),
                val_split=self.config.get('val_split', 0.15),
                seed=self.seed,
                target_num_speakers=self.config['num_speakers']
            )
            
            self.dataloaders = (train_loader, val_loader, test_loader)
            
            try:
                self.logger.info("Scanning Every Data to verify the real Label Range...")
                
                sample_batch = next(iter(train_loader))
                sample_input, sample_labels = sample_batch
                
                all_labels = []
                
                self.logger.info("Scanning the Training Dataset...")
                for batch_idx, (_, labels) in enumerate(train_loader):
                    all_labels.extend(labels.tolist())
                    if batch_idx % 50 == 0:
                        current_min = min(all_labels)
                        current_max = max(all_labels)
                        self.logger.info(f"Finish Scan {batch_idx + 1} batch, current label range: {current_min}-{current_max}")

                self.logger.info("Scanning the Val Dataset...")
                for _, labels in val_loader:
                    all_labels.extend(labels.tolist())
                
                self.logger.info("Scanning the Test Dataset...")
                for _, labels in test_loader:
                    all_labels.extend(labels.tolist())
                
                min_label = min(all_labels)
                max_label = max(all_labels)
                actual_num_speakers = max_label + 1
                unique_labels = set(all_labels)
                
                self.logger.info(f"===== Dataset Labels Analysis =====")
                self.logger.info(f"INPUT SHAPE: {sample_input.shape}")
                self.logger.info(f"LABEL RANGE: {min_label} to {max_label}")
                self.logger.info(f"ACTUAL SPEAKERS: {actual_num_speakers}")
                self.logger.info(f"UNIQUE LABEL NUMBER: {len(unique_labels)}")
                self.logger.info(f"SPEAKER NUMBER IN CONFIG: {self.config['num_speakers']}")
                self.logger.info(f"BASELINE TYPE: {self.config['baseline_type']}")
                
                expected_labels = set(range(min_label, max_label + 1))
                missing_labels = expected_labels - unique_labels
                if missing_labels:
                    self.logger.warning(f"‼️DETECT MISSING LABEL: {sorted(list(missing_labels))[:10]}{'...' if len(missing_labels) > 10 else ''}")
                
                if 'mel' in self.config['baseline_type'] and sample_input.shape[1] == 48000:
                    self.logger.warning(f"Detect original dataset: {sample_input.shape}，but baseline type is mel_mlp")
                    self.logger.warning("Probably will lead to dim unmatch，model will auto-fit in")
                
                model_needs_recreation = False
                if actual_num_speakers > self.config['num_speakers']:
                    old_num_speakers = self.config['num_speakers']
                    self.config['num_speakers'] = actual_num_speakers
                    self.logger.warning(f"Dataset need speaker number: {actual_num_speakers}, surpass the nuber in config: {old_num_speakers}")
                    self.logger.info(f"Adjust the  num_speakers from {old_num_speakers} to {actual_num_speakers}")
                    model_needs_recreation = True
                
                if hasattr(self, 'model') and self.model is not None and model_needs_recreation:
                    self.logger.info("Recreat Model to Match the Speaker Number...")
                    self.model = None
                    model_needs_recreation = True
                
                if model_needs_recreation or not hasattr(self, 'model') or self.model is None:
                    self.logger.info("Create the model that suits the full dataset...")
                    self._create_model_with_input_detection(sample_input)
                    
                    if self.model is None:
                        self.logger.error("Model creation failed，trying the other plan...")
                        self._create_model_fallback_with_detection(sample_input)
                
                if min_label < 0:
                    raise ValueError(f"DETECT NEGATIVE LABEL: {min_label}")
                if max_label >= self.config['num_speakers']:
                    raise ValueError(f"MAX LABEL: {max_label} --> EXCEED RANGE [0, {self.config['num_speakers']-1}]")
                    
                self.logger.info(f"✅ FALL LABEL RANGE VERIFICATION PASS: [{min_label}, {max_label}] in {self.config['num_speakers']} category")
                
                train_loader, val_loader, test_loader = create_speaker_dataloaders(
                    data_dir=self.config['data_dir'],
                    batch_size=self.config['batch_size'],
                    sample_rate=self.config['sample_rate'],
                    max_length=self.config['max_audio_length'],
                    num_workers=0,
                    train_split=self.config.get('train_split', 0.7),
                    val_split=self.config.get('val_split', 0.15),
                    seed=self.seed,
                    target_num_speakers=self.config['num_speakers']
                )
                
            except Exception as e:
                self.logger.error(f"❌ FALL LABEL RANGE VERIFICATION FAILED: {e}")
                import traceback
                traceback.print_exc()
                self.logger.info("USING NUMBER "260" instead")
                self.config['num_speakers'] = 260
            
        else:
            # Create mock data loaders for testing
            self.logger.warning("Using mock data loaders for testing")
            
            feature_dim = self.config.get('n_mels', 80) if 'mel' in self.config['baseline_type'] else self.config.get('n_mfcc', 13)
            
            train_dataset = MockDataset(1000, self.config['num_speakers'], feature_dim)
            val_dataset = MockDataset(200, self.config['num_speakers'], feature_dim)
            test_dataset = MockDataset(200, self.config['num_speakers'], feature_dim)
            
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
            
            self.dataloaders = (train_loader, val_loader, test_loader)
        
        return train_loader, val_loader, test_loader
    
    def _create_model_with_input_detection(self, sample_input):
        """Creater Model Based on the Real Input"""
        try:
            actual_input_shape = sample_input.shape
            self.logger.info(f"DETECT REAL INPUT SHAPE: {actual_input_shape}")
            
            if len(actual_input_shape) == 2:  # [batch_size, features]
                input_dim = actual_input_shape[1]
            elif len(actual_input_shape) == 3:  # [batch_size, feature_dim, time_frames]
                input_dim = actual_input_shape[1] * actual_input_shape[2]
            else:
                raise ValueError(f"NOT SUPPORT INPUT SHAPE: {actual_input_shape}")
            
            model_config = {
                'input_dim': input_dim,
                'num_classes': self.config['num_speakers'],
                'hidden_dims': self.config.get('hidden_dims', [512, 256, 128]),
                'dropout_rate': self.config.get('dropout_rate', 0.3),
                'use_batch_norm': self.config.get('use_batch_norm', True),
                'baseline_type': self.config['baseline_type']
            }
            
            self.logger.info(f"CREATE MODEL CONFIG: INPUT DIM={input_dim}, OUTPUT TYPE={self.config['num_speakers']}")
            
            if hasattr(self, '_create_baseline_model'):
                self.model = self._create_baseline_model(model_config)
            else:
                from hybrid_models import TraditionalMLPBaseline
                self.model = TraditionalMLPBaseline(model_config)
            
            if self.model is not None:
                self.model = self.model.to(self.device)
                self.logger.info("MODEL CREATE SUCCESS")
                self.logger.info(f"INPUT DIM: {input_dim}, OUTPUT DIM: {self.config['num_speakers']}")
            else:
                self.logger.error("MODEL RETURN: None")
                
        except Exception as e:
            self.logger.error(f"MODEL CREATE FAILED: {e}")
            self.model = None
    
    def _create_model_fallback_with_detection(self, sample_input):
        """PLAN B model, based on the real input dim"""
        try:
            import torch.nn as nn
            
            actual_input_shape = sample_input.shape
            self.logger.info(f"PLAN B MODEL：DETECT REAL INPUT DIM: {actual_input_shape}")
            
            if len(actual_input_shape) == 2:  # [batch_size, features]
                input_dim = actual_input_shape[1]
            elif len(actual_input_shape) == 3:  # [batch_size, feature_dim, time_frames]
                input_dim = actual_input_shape[1] * actual_input_shape[2]
            else:
                input_dim = sample_input.view(sample_input.size(0), -1).shape[1]
            
            self.logger.info(f"PLAN B MODEL USING DIM: {input_dim}")
            
            if input_dim > 10000:
                hidden_dims = [1024, 512, 256]
            elif input_dim > 1000:
                hidden_dims = [512, 256, 128]
            else:
                hidden_dims = [256, 128, 64]
            
            layers = []
            layers.append(nn.Flatten())
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))
            
            layers.append(nn.Linear(hidden_dims[-1], self.config['num_speakers']))
            
            self.model = nn.Sequential(*layers).to(self.device)
            
            total_params = sum(p.numel() for p in self.model.parameters())
            
            self.logger.info(f"✓ PLAN B MODEL CREATE SUCCESS")
            self.logger.info(f"  INPUT DIM: {input_dim}")
            self.logger.info(f"  HIDDEN LAYER: {hidden_dims}")
            self.logger.info(f"  OUTPUT DIM: {self.config['num_speakers']}")
            self.logger.info(f"  TOTAL PARAMS: {total_params:,}")
            
        except Exception as e:
            self.logger.error(f"❌PLAN B MODEL CREATE FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("‼️CAN NOT CREATE MODEL")
                
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer for the baseline model."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam').lower()
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                **optimizer_config.get('params', {})
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                momentum=optimizer_config.get('params', {}).get('momentum', 0.9),
                **{k: v for k, v in optimizer_config.get('params', {}).items() if k != 'momentum'}
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                **optimizer_config.get('params', {})
            )
        else:
            self.logger.warning(f"Unknown optimizer type: {optimizer_type}, using Adam")
            optimizer = optim.Adam(
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
        Perform forward pass on a batch.
        
        Args:
            batch: Tuple of (audio, speaker_labels)
            training: Whether in training mode
            
        Returns:
            Tuple of (loss, predictions, targets)
        """
        audio, labels = batch
        
        # Get current epoch safely
        if hasattr(self.state, 'epoch'):
            current_epoch = self.state.epoch
        elif hasattr(self, 'current_epoch'):
            current_epoch = self.current_epoch
        else:
            current_epoch = 0
        
        # Debug: Check label distribution in first validation epoch
        if not training and current_epoch == 0:
            print(f"VAL Dataset Label Range: {labels.min().item()} - {labels.max().item()}")
            print(f"VAL Dataset Unique Label Number: {len(torch.unique(labels))}")
        
        # Forward pass through model
        logits = self.model(audio)
        
        # Debug: Check prediction distribution in first validation epoch
        if not training and current_epoch == 0:
            preds = torch.argmax(logits, dim=1)
            print(f"Prediction Dataset Label Range: {preds.min().item()} - {preds.max().item()}")
            print(f"Prediction Unique Label Number: {len(torch.unique(preds))}")
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1)
        
        return loss, predictions, labels
        
    def calculate_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        loss: float
    ) -> Dict[str, float]:
        """Calculate baseline-specific metrics."""
        # Use MetricsCalculator for basic metrics
        self.metrics_calculator.reset()
        self.metrics_calculator.update(predictions, targets)
        
        basic_metrics = self.metrics_calculator.compute_basic_metrics()
        
        # Add loss and custom metrics
        metrics = {
            'loss': loss,
            **basic_metrics
        }
        
        # Add additional metrics specific to speaker recognition
        with torch.no_grad():
            # Per-class accuracy (if needed for analysis)
            num_classes = self.config['num_speakers']
            if num_classes <= 20:  # Only compute for smaller number of speakers
                per_class_correct = torch.zeros(num_classes)
                per_class_total = torch.zeros(num_classes)
                
                for i in range(num_classes):
                    mask = (targets == i)
                    if mask.sum() > 0:
                        per_class_correct[i] = (predictions[mask] == i).float().sum()
                        per_class_total[i] = mask.sum()
                
                # Average per-class accuracy
                per_class_acc = per_class_correct / (per_class_total + 1e-8)
                metrics['mean_per_class_accuracy'] = per_class_acc.mean().item()
            
            # Balanced accuracy
            metrics['balanced_accuracy'] = self._compute_balanced_accuracy(predictions, targets)
        
        return metrics
    
    def _compute_balanced_accuracy(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> float:
        """Compute balanced accuracy for imbalanced datasets."""
        num_classes = self.config['num_speakers']
        class_accuracies = []
        
        for class_id in range(num_classes):
            # Find samples of this class
            class_mask = (targets == class_id)
            if class_mask.sum() == 0:
                continue  # Skip classes with no samples
            
            # Compute accuracy for this class
            class_predictions = predictions[class_mask]
            class_accuracy = (class_predictions == class_id).float().mean().item()
            class_accuracies.append(class_accuracy)
        
        return np.mean(class_accuracies) if class_accuracies else 0.0
    
    def run_baseline_analysis(self) -> Dict[str, Any]:
        """Run comprehensive baseline analysis."""
        self.logger.info("Running baseline analysis...")
        
        analysis_results = {}
        
        # Model complexity analysis
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        analysis_results['model_complexity'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        # Feature analysis (if possible)
        if hasattr(self.model, 'feature_extractor'):
            analysis_results['feature_type'] = self.config['baseline_type']
            if 'mel' in self.config['baseline_type']:
                analysis_results['feature_dim'] = self.config['n_mels']
            elif 'mfcc' in self.config['baseline_type']:
                analysis_results['feature_dim'] = self.config['n_mfcc']
        
        # Training efficiency
        analysis_results['training_config'] = {
            'learning_rate': self.config['learning_rate'],
            'batch_size': self.config['batch_size'],
            'optimizer': self.config.get('optimizer', {}).get('type', 'adam'),
            'weight_decay': self.config['weight_decay']
        }
        
        # Save analysis results
        analysis_file = os.path.join(self.results_dir, 'baseline_analysis.json')
        import json
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        self.logger.info("Baseline analysis completed")
        return analysis_results

    def _validate_data_path(self):
        """varify data path"""
        data_dir = Path(self.config['data_dir'])
        
        if not data_dir.exists():
            self.logger.error(f"‼️Data path doesn't exsit: {data_dir}")
            alternatives = [
                data_dir.parent.parent / "dataset" / "train-clean-100" / "LibriSpeech" / "train-clean-100",
                Path("/dataset/train-clean-100/LibriSpeech/train-clean-100/"),
            ]
            
            for alt_path in alternatives:
                if alt_path.exists():
                    self.logger.info(f"Find Dataset in: {alt_path}")
                    self.config['data_dir'] = str(alt_path)
                    return
            
            raise FileNotFoundError(f"Didn't find LibriSpeech Dataset，Checked Path: {[data_dir] + alternatives}")
        
        flac_files = list(data_dir.rglob("*.flac"))
        if len(flac_files) == 0:
            raise ValueError(f"Data Path {data_dir} didn't find FLAC files")
        
        self.logger.info(f"✅ DATA PATH VARIFICATION SUCCESS: {data_dir}")
        self.logger.info(f"FOUND FLAC FILES: {len(flac_files)}")

    def setup(self):
        """Set up all experiment components."""
        self.logger.info("Setting up experiment components...")
        
        self._validate_data_path()
        
        # Create model
        self.model = self.create_model()
        self.model.to(self.device)
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            self.logger.error("NO PARAMETERS IN MODEL! MODEL COULD BE EMPTY!")
            self.logger.error(f"MODEL TYPE: {type(self.model)}")
            self.logger.error(f"MODEL STURCTURE: {self.model}")
            raise ValueError("CANNOT ACQUIRE DEVICE INFO WITHOUT PARAMETERS")
        if str(model_device) != str(self.device):
            self.logger.error(f"Model device mismatch: {model_device} vs {self.device}")
        
        self.logger.info(f"Model created and moved to device: {model_device}")
    
        # Log model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = self.create_dataloaders()
        self.logger.info(f"Dataloaders created:")
        self.logger.info(f"  Train: {len(self.train_loader)} batches")
        self.logger.info(f"  Val: {len(self.val_loader)} batches")
        self.logger.info(f"  Test: {len(self.test_loader)} batches")
        
        # Create optimizer
        self.optimizer = self.create_optimizer(self.model)
        self.logger.info(f"Optimizer: {type(self.optimizer).__name__}")
        
        # Create scheduler
        self.scheduler = self.create_scheduler(self.optimizer)
        if self.scheduler:
            self.logger.info(f"Scheduler: {type(self.scheduler).__name__}")
        
        # Create criterion
        self.criterion = self.create_criterion()
        self.criterion.to(self.device)
        self.logger.info(f"Criterion: {type(self.criterion).__name__}")
        
        # Save experiment configuration
        self.save_config()
        
        self.logger.info("Experiment setup completed")

    def _create_model(self):
        """Recreate model"""
        try:
            if 'mel' in self.config['baseline_type']:
                input_dim = self.config.get('n_mels', 80)
            elif 'mfcc' in self.config['baseline_type']:
                input_dim = self.config.get('n_mfcc', 13)
            else:
                input_dim = 80  # defalue
            
            model_config = {
                'input_dim': input_dim,
                'num_classes': self.config['num_speakers'],
                'hidden_dims': self.config.get('hidden_dims', [256, 128, 64]),
                'dropout_rate': self.config.get('dropout_rate', 0.3),
                'use_batch_norm': self.config.get('use_batch_norm', True)
            }
            
            if hasattr(self, '_create_baseline_model'):
                self.model = self._create_baseline_model(model_config)
            else:
                from hybrid_models import TraditionalMLPBaseline
                self.model = TraditionalMLPBaseline(model_config)
            
            if self.model is not None:
                self.model = self.model.to(self.device)
                self.logger.info(f"✓ MODEL CREATE SUCCESS，OUTPUT DIM: {self.config['num_speakers']}")
            else:
                self.logger.error("MODEL RETURN None")
                
        except Exception as e:
            self.logger.error(f"MODEL CREATE FAILED: {e}")
            self.model = None
    
    def _create_model_fallback(self):
        """PLAN B MODEL"""
        try:
            import torch.nn as nn
            
            self.logger.info("DETECT REAL INPUT DIM...")
            
            train_loader, _, _ = self.dataloaders if hasattr(self, 'dataloaders') else (None, None, None)
            if train_loader is not None:
                sample_batch = next(iter(train_loader))
                sample_input, _ = sample_batch
                actual_input_shape = sample_input.shape
                self.logger.info(f"DETECT ACTUAL INPUT SHAPE: {actual_input_shape}")
                
                if len(actual_input_shape) == 2:  # [batch_size, features]
                    input_dim = actual_input_shape[1]
                elif len(actual_input_shape) == 3:  # [batch_size, feature_dim, time_frames]
                    input_dim = actual_input_shape[1] * actual_input_shape[2]
                else:
                    raise ValueError(f"NOT SUPPORT INPUT SHAPE: {actual_input_shape}")
            else:
                if 'mel' in self.config['baseline_type']:
                    input_dim = 80 * 300  # mel
                elif 'mfcc' in self.config['baseline_type']:
                    input_dim = 13 * 300  # mfcc
                else:
                    input_dim = 48000  # origin
            
            self.logger.info(f"USING DIM: {input_dim}")
            
            hidden_dims = [512, 256, 128]
            layers = []
            
            layers.append(nn.Flatten())
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))
            
            layers.append(nn.Linear(hidden_dims[-1], self.config['num_speakers']))
            
            self.model = nn.Sequential(*layers).to(self.device)
            
            self.logger.info(f"✓ PLAN B MODEL CREATE SUCCESS，INPUT DIM: {input_dim}, OUTPUT DIM: {self.config['num_speakers']}")
            
        except Exception as e:
            self.logger.error(f"PLAN B MODEL CREATE FAILED: {e}")
            raise RuntimeError("CANNOT CREATE MODEL")


def create_baseline_experiments(base_config: Dict[str, Any]) -> Dict[str, BaselineExperiment]:
    """
    Create multiple baseline experiments for comparison.
    
    Args:
        base_config: Base configuration to use for all baselines
        
    Returns:
        Dictionary of baseline experiments
    """
    baseline_types = ['mel_mlp', 'mfcc_mlp', 'mel_cnn', 'mfcc_cnn']
    experiments = {}
    
    for baseline_type in baseline_types:
        config = base_config.copy()
        config['baseline_type'] = baseline_type
        
        # Adjust learning rate for CNN models (typically need lower LR)
        if 'cnn' in baseline_type:
            config['learning_rate'] = config.get('learning_rate', 0.001) * 0.5
        
        experiment_name = f'baseline_{baseline_type}'
        experiment = BaselineExperiment(
            config=config,
            experiment_name=experiment_name,
            seed=config.get('seed', 42)
        )
        
        experiments[baseline_type] = experiment
    
    return experiments

def test(self) -> Dict[str, float]:
    """Test the trained baseline model."""
    if self.model is None:
        raise RuntimeError("Model has not been created. Call setup() first.")
    
    self.model.eval()
    test_loader = self.test_loader
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            loss, predictions, targets = self.forward_pass(batch, training=False)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total_loss += loss.item()
            num_batches += 1
    
    # Calculate metrics using MetricsCalculator
    avg_loss = total_loss / max(num_batches, 1)
    
    # Update metrics calculator
    self.metrics_calculator.reset()
    self.metrics_calculator.update(
        predictions=np.array(all_predictions),
        targets=np.array(all_targets)
    )
    
    # Compute comprehensive metrics
    basic_metrics = self.metrics_calculator.compute_basic_metrics()
    advanced_metrics = self.metrics_calculator.compute_advanced_metrics()
    
    # Combine all metrics
    test_metrics = {
        'loss': avg_loss,
        **basic_metrics,
        **advanced_metrics
    }
    
    # Add baseline-specific metrics
    test_metrics['balanced_accuracy'] = self._compute_balanced_accuracy(
        torch.tensor(all_predictions), 
        torch.tensor(all_targets)
    )
    
    return test_metrics

    
if __name__ == "__main__":
    # print(f"✓ Project Root: {PROJECT_ROOT}")
    # print(f"✓ Import Manager: {USING_IMPORT_MANAGER}")
    # print(f"✓ Module imports successful")
    
    # Example usage and testing
    
    # Test configuration
    test_config = {
        'baseline_type': 'mel_mlp',
        'num_speakers': 10,
        'batch_size': 16,
        'learning_rate': 0.001,
        'hidden_dims': [128, 64],
        'n_mels': 80,
        'sample_rate': 16000,
        'primary_metric': 'accuracy',
        'log_interval': 5
    }
    
    print("Testing BaselineExperiment...")
    
    # Test single experiment
    experiment = BaselineExperiment(
        config=test_config,
        experiment_name='test_mel_mlp_baseline'
    )
    
    print("Setting up experiment...")
    experiment.setup()
    
    print("Running baseline analysis...")
    analysis = experiment.run_baseline_analysis()
    print(f"Model parameters: {analysis['model_complexity']['total_parameters']:,}")
    
    print("Training for 2 epochs...")
    experiment.train(num_epochs=2)
    
    print("Baseline experiment test completed!")
    
    # Test multiple baselines creation
    print("\nTesting multiple baseline creation...")
    base_config = {
        'num_speakers': 5,
        'batch_size': 8,
        'learning_rate': 0.001,
        'sample_rate': 16000
    }
    
    baseline_experiments = create_baseline_experiments(base_config)
    print(f"Created {len(baseline_experiments)} baseline experiments:")
    for name in baseline_experiments.keys():
        print(f"  - {name}")
    
    print("All tests completed successfully!")