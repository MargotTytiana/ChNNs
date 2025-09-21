import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
import numpy as np

import os
import sys
from pathlib import Path

# 简单添加路径，不调用setup_imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import base experiment class
try:
    from experiments.base_experiment import BaseExperiment
except ImportError:
    # Fallback for testing
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from experiments.base_experiment import BaseExperiment

# Import models and components
try:
    from models.hybrid_models import TraditionalMLPBaseline, HybridModelManager
    from models.mlp_classifier import MLPClassifier
    from features.traditional_features import MelExtractor, MFCCExtractor
    from data.dataset_loader import SpeakerDataset, create_speaker_dataloaders
    
    print(f"Successfully imported: {create_speaker_dataloaders}")
    from data.audio_preprocessor import AudioPreprocessor
except ImportError:
    # Mock implementations for testing
    print("Warning: Some modules not found. Using mock implementations for testing.")
    
    TraditionalMLPBaseline = None
    HybridModelManager = None
    MLPClassifier = None
    MelExtractor = None
    MFCCExtractor = None
    SpeakerDataset = None
    create_speaker_dataloaders = None
    AudioPreprocessor = None
    
    class MockMLPClassifier(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
            super().__init__()
            layers = []
            current_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, output_dim))
            self.model = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.model(x)
    
    class MockFeatureExtractor(nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            self.output_dim = output_dim
            
        def forward(self, x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, self.output_dim, 
                              device=x.device, dtype=x.dtype)
    
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, num_classes, feature_dim):
            self.num_samples = num_samples
            self.num_classes = num_classes
            self.feature_dim = feature_dim
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # Return (audio, speaker_id)
            audio = torch.randn(8000)  # 0.5 seconds at 16kHz
            speaker_id = torch.randint(0, self.num_classes, (1,)).item()
            return audio, speaker_id


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

        # 添加 MetricsCalculator 初始化
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

        # 确保模型完全移动到设备
        model = model.to(self.device)
        
        # 检查所有参数是否在正确设备上
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
        
        class CNNBaseline(nn.Module):
            def __init__(self, feature_dim, num_classes, dropout_rate=0.3):
                super(CNNBaseline, self).__init__()
                
                # Feature extraction layer
                if feature_type == 'mel':
                    if MelExtractor is not None:
                        self.feature_extractor = MelExtractor(
                            n_mels=feature_dim,
                            sample_rate=self.config['sample_rate']
                        )
                    else:
                        self.feature_extractor = MockFeatureExtractor(feature_dim)
                else:
                    if MFCCExtractor is not None:
                        self.feature_extractor = MFCCExtractor(
                            n_mfcc=feature_dim,
                            sample_rate=self.config['sample_rate']
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
                    nn.AdaptiveAvgPool1d(1),  # Global average pooling
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
                # 确保输入在正确设备上
                device = next(self.parameters()).device
                if x.device != device:
                    x = x.to(device)
                    
                # Extract features
                features = self.feature_extractor(x)
                
                # 确保特征在正确设备上
                if features.device != device:
                    features = features.to(device)
            
                # Handle different feature shapes
                if len(features.shape) == 2:
                    # Add time dimension if needed
                    features = features.unsqueeze(-1)
                elif len(features.shape) == 3:
                    # Transpose to (batch, feature, time) for Conv1d
                    features = features.transpose(1, 2)
                
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
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for training, validation, and testing."""
        # 添加这些调试语句
        print(f"create_speaker_dataloaders 变量值: {create_speaker_dataloaders}")
        print(f"create_speaker_dataloaders 类型: {type(create_speaker_dataloaders)}")
        
        # 尝试手动导入测试
        try:
            from dataset_loader import create_speaker_dataloaders as manual_import
            print(f"手动导入成功: {manual_import}")
        except ImportError as e:
            print(f"手动导入失败: {e}")
        
        # 检查当前Python路径
        import sys
        print(f"当前Python路径: {sys.path[:3]}...")  # 只显示前3个路径

        
        # 根据项目结构设置正确的数据路径
        if 'data_dir' not in self.config:
            # 从Model目录向上找到project根目录，然后到dataset
            model_dir = Path(__file__).parent.parent  # 从experiments回到Model
            project_root = model_dir.parent  # 从Model回到project
            librispeech_path = project_root / "dataset" / "train-clean-100" / "LibriSpeech" / "train-clean-100"
            self.config['data_dir'] = str(librispeech_path)
            
        print(f"使用数据路径: {self.config['data_dir']}")
        
        if create_speaker_dataloaders is not None:
            # Use real data loading
            train_loader, val_loader, test_loader = create_speaker_dataloaders(
                data_dir=self.config.get('data_dir'),
                batch_size=self.config['batch_size'],
                sample_rate=self.config['sample_rate'],
                max_length=self.config['max_audio_length'],
                num_workers=self.config.get('num_workers', 4),
                train_split=self.config.get('train_split', 0.7),
                val_split=self.config.get('val_split', 0.15),
                seed=self.seed
            )
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
        
        return train_loader, val_loader, test_loader
    
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
        audio, targets = batch
        
        # Forward pass through model
        logits = self.model(audio)
        
        # Compute loss
        loss = self.criterion(logits, targets)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1)
        
        return loss, predictions, targets
    
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
        """验证数据路径是否正确"""
        data_dir = Path(self.config['data_dir'])
        
        if not data_dir.exists():
            self.logger.error(f"数据路径不存在: {data_dir}")
            # 尝试其他可能的路径
            alternatives = [
                data_dir.parent.parent / "dataset" / "train-clean-100" / "LibriSpeech" / "train-clean-100",
                Path("/dataset/train-clean-100/LibriSpeech/train-clean-100/"),
            ]
            
            for alt_path in alternatives:
                if alt_path.exists():
                    self.logger.info(f"找到替代路径: {alt_path}")
                    self.config['data_dir'] = str(alt_path)
                    return
            
            raise FileNotFoundError(f"未找到LibriSpeech数据集，检查过的路径: {[data_dir] + alternatives}")
        
        # 检查是否包含音频文件
        flac_files = list(data_dir.rglob("*.flac"))
        if len(flac_files) == 0:
            raise ValueError(f"数据路径 {data_dir} 中未找到FLAC音频文件")
        
        self.logger.info(f"数据路径验证成功: {data_dir}")
        self.logger.info(f"找到 {len(flac_files)} 个FLAC文件")

    def setup(self):
        """Set up all experiment components."""
        self.logger.info("Setting up experiment components...")
        
        # 验证数据路径
        self._validate_data_path()
        
        # Create model
        self.model = self.create_model()
        self.model.to(self.device)
        model_device = next(self.model.parameters()).device
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