import os
import time
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 标准的文件头部导入设置
import os
import sys
from pathlib import Path

# Setup project imports - this is our safety net
try:
    from setup_imports import setup_project_imports
    setup_project_imports()
    print("Import setup completed via setup_imports module")
except ImportError:
    # Fallback method if setup_imports is not found
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    print(f"Fallback import setup: added {project_root} to path")

# Now we can use clean, absolute imports based on the Model package
try:
    from Model.models.hybrid_models import TraditionalMLPBaseline, HybridModelManager
    from Model.models.mlp_classifier import MLPClassifier
    from Model.data.dataset_loader import create_speaker_dataloaders, LibriSpeechChaoticDataset
    from Model.features.traditional_features import MelExtractor, MFCCExtractor
    print("All required modules imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please check that all required files exist and the project structure is correct")
    

# Import utilities (with fallbacks for testing)
try:
    from utils.logger import setup_logger
    from utils.checkpoint import CheckpointManager
    from utils.reproducibility import set_seed, get_system_info
    from evaluation.metrics import MetricsCalculator
except ImportError:
    # Mock implementations for testing
    def setup_logger(name, log_file=None, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    class CheckpointManager:
        def __init__(self, checkpoint_dir):
            self.checkpoint_dir = checkpoint_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        def save_checkpoint(self, state, filename):
            filepath = os.path.join(self.checkpoint_dir, filename)
            torch.save(state, filepath)
            return filepath
        
        def load_checkpoint(self, filename):
            filepath = os.path.join(self.checkpoint_dir, filename)
            return torch.load(filepath, map_location='cpu')
        
        def list_checkpoints(self):
            return [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
    
    def set_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def get_system_info():
        return {"python_version": "3.8+", "pytorch_version": torch.__version__}
    
    class MetricsCalculator:
        @staticmethod
        def compute_accuracy(predictions, targets):
            correct = (predictions == targets).float()
            return correct.mean().item()
        
        @staticmethod
        def compute_top_k_accuracy(logits, targets, k=5):
            _, pred_k = torch.topk(logits, k, dim=1)
            correct = pred_k.eq(targets.view(-1, 1).expand_as(pred_k))
            return correct.any(dim=1).float().mean().item()


class ExperimentState:
    """Class to track experiment state and progress."""
    
    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_metrics = []
        self.start_time = None
        self.end_time = None
        self.status = 'initialized'  # initialized, running, completed, failed
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'epoch': self.epoch,
            'step': self.step,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status
        }
    
    def from_dict(self, state_dict: Dict[str, Any]):
        """Load state from dictionary."""
        self.epoch = state_dict.get('epoch', 0)
        self.step = state_dict.get('step', 0)
        self.best_metric = state_dict.get('best_metric', 0.0)
        self.best_epoch = state_dict.get('best_epoch', 0)
        self.train_losses = state_dict.get('train_losses', [])
        self.val_metrics = state_dict.get('val_metrics', [])
        self.status = state_dict.get('status', 'initialized')
        
        start_time_str = state_dict.get('start_time')
        if start_time_str:
            self.start_time = datetime.fromisoformat(start_time_str)
        
        end_time_str = state_dict.get('end_time')
        if end_time_str:
            self.end_time = datetime.fromisoformat(end_time_str)


class BaseExperiment(ABC):
    """
    Abstract base class for all experiments.
    
    Provides common functionality for training, validation, testing, and experiment management.
    Subclasses should implement the abstract methods to define specific experiment logic.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: str,
        output_dir: str = './experiments/outputs',
        device: str = 'auto',
        seed: int = 42
    ):
        """
        Initialize base experiment.
        
        Args:
            config: Experiment configuration dictionary
            experiment_name: Name of the experiment
            output_dir: Directory to save experiment outputs
            device: Device to use ('auto', 'cpu', 'cuda')
            seed: Random seed for reproducibility
        """
        self.config = config
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.seed = seed
        
        # Set up device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Set up directories
        self.experiment_dir = os.path.join(output_dir, experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.results_dir = os.path.join(self.experiment_dir, 'results')
        
        for directory in [self.experiment_dir, self.checkpoint_dir, self.log_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set up logging
        log_file = os.path.join(self.log_dir, f'{experiment_name}.log')
        self.logger = setup_logger(f'experiment_{experiment_name}', log_file)
        
        # Set up tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Initialize state
        self.state = ExperimentState()
        
        # Set up checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        
        # Initialize components (to be set by subclasses)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.metrics_calculator = MetricsCalculator(
            num_classes=self.config['num_speakers'],
            class_names=[f"speaker_{i}" for i in range(self.config['num_speakers'])]
        )
        
        # Set reproducibility
        set_seed(self.seed)
        
        self.logger.info(f"Initialized experiment: {experiment_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output directory: {self.experiment_dir}")
        
    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create and return the model for this experiment."""
        pass
    
    @abstractmethod
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create and return train, validation, and test dataloaders."""
        pass
    
    @abstractmethod
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create and return optimizer for the model."""
        pass
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create and return learning rate scheduler (optional)."""
        scheduler_config = self.config.get('scheduler', {})
        
        if not scheduler_config or scheduler_config.get('type') is None:
            return None
        
        scheduler_type = scheduler_config['type']
        scheduler_params = scheduler_config.get('params', {})
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler_type == 'multistep':
            return optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        elif scheduler_type == 'reduce_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
        else:
            self.logger.warning(f"Unknown scheduler type: {scheduler_type}")
            return None
    
    def create_criterion(self) -> nn.Module:
        """Create and return loss criterion."""
        criterion_config = self.config.get('criterion', {'type': 'crossentropy'})
        
        criterion_type = criterion_config.get('type', 'crossentropy').lower()
        criterion_params = criterion_config.get('params', {})
        
        if criterion_type == 'crossentropy':
            return nn.CrossEntropyLoss(**criterion_params)
        elif criterion_type == 'mse':
            return nn.MSELoss(**criterion_params)
        elif criterion_type == 'bce':
            return nn.BCEWithLogitsLoss(**criterion_params)
        else:
            self.logger.warning(f"Unknown criterion type: {criterion_type}, using CrossEntropyLoss")
            return nn.CrossEntropyLoss()
    
    def setup(self):
        """Set up all experiment components."""
        self.logger.info("Setting up experiment components...")
        
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
    
    def save_config(self):
        """Save experiment configuration to file."""
        config_file = os.path.join(self.experiment_dir, 'config.json')
        
        # Add system information
        full_config = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'device': self.device,
            'seed': self.seed,
            'system_info': get_system_info(),
            'created_at': datetime.now().isoformat()
        }
        
        with open(config_file, 'w') as f:
            json.dump(full_config, f, indent=2)
        
        self.logger.info(f"Configuration saved to: {config_file}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            loss, predictions, targets = self.forward_pass(batch, training=True)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if specified
            if self.config.get('gradient_clipping'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clipping']
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            batch_size = targets.size(0) if torch.is_tensor(targets) else len(targets)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            all_predictions.append(predictions)
            all_targets.append(targets)
            
            # Log batch metrics
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self.logger.info(
                    f'Train Epoch {self.state.epoch}: [{batch_idx * batch_size}/{len(self.train_loader.dataset)} '
                    f'({100. * batch_idx / len(self.train_loader):.1f}%)]\tLoss: {loss.item():.6f}'
                )
            
            self.state.step += 1
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        
        # Combine predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate additional metrics
        epoch_metrics = self.calculate_metrics(all_predictions, all_targets, avg_loss)
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary containing validation metrics for the epoch
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                loss, predictions, targets = self.forward_pass(batch, training=False)
                
                # Update metrics
                batch_size = targets.size(0) if torch.is_tensor(targets) else len(targets)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        
        # Combine predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate additional metrics
        epoch_metrics = self.calculate_metrics(all_predictions, all_targets, avg_loss)
        
        return epoch_metrics
    
    def test(self) -> Dict[str, float]:
        """
        Test the model on test set.
        
        Returns:
            Dictionary containing test metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                loss, predictions, targets = self.forward_pass(batch, training=False)
                
                # Update metrics
                batch_size = targets.size(0) if torch.is_tensor(targets) else len(targets)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Calculate test metrics
        avg_loss = total_loss / total_samples
        
        # Combine predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate additional metrics
        test_metrics = self.calculate_metrics(all_predictions, all_targets, avg_loss)
        
        return test_metrics
    
    @abstractmethod
    def forward_pass(self, batch: Any, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward pass on a batch.
        
        Args:
            batch: Batch of data
            training: Whether in training mode
            
        Returns:
            Tuple of (loss, predictions, targets)
        """
        pass
    
    def calculate_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        loss: float
    ) -> Dict[str, float]:
        """
        Calculate metrics from predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: True targets
            loss: Loss value
            
        Returns:
            Dictionary of metrics
        """
        metrics = {'loss': loss}
        
        # Convert logits to class predictions if necessary
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions
        
        # Calculate accuracy
        accuracy = self.metrics_calculator.compute_accuracy(pred_classes, targets)
        metrics['accuracy'] = accuracy
        
        # Calculate top-k accuracy if multi-class
        if len(predictions.shape) > 1 and predictions.shape[1] > 5:
            top5_acc = self.metrics_calculator.compute_top_k_accuracy(predictions, targets, k=5)
            metrics['top5_accuracy'] = top5_acc
        
        return metrics
    
    def train(self, num_epochs: int, resume_from_checkpoint: bool = False):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from_checkpoint: Whether to resume from latest checkpoint
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        if resume_from_checkpoint:
            self.load_latest_checkpoint()
        
        self.state.start_time = datetime.now()
        self.state.status = 'running'
        
        try:
            for epoch in range(self.state.epoch, num_epochs):
                self.state.epoch = epoch
                
                # Train epoch
                train_metrics = self.train_epoch()
                self.state.train_losses.append(train_metrics['loss'])
                
                # Validate epoch
                val_metrics = self.validate_epoch()
                self.state.val_metrics.append(val_metrics)
                
                # Log metrics
                self.log_epoch_metrics(epoch, train_metrics, val_metrics)
                
                # Update learning rate scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get('loss', val_metrics.get('accuracy', 0)))
                    else:
                        self.scheduler.step()
                
                # Check for best model
                primary_metric = self.config.get('primary_metric', 'accuracy')
                current_metric = val_metrics.get(primary_metric, 0.0)
                
                is_best = False
                if primary_metric in ['accuracy', 'top5_accuracy']:  # Higher is better
                    is_best = current_metric > self.state.best_metric
                else:  # Lower is better (e.g., loss)
                    is_best = current_metric < self.state.best_metric or self.state.best_metric == 0.0
                
                if is_best:
                    self.state.best_metric = current_metric
                    self.state.best_epoch = epoch
                    self.save_checkpoint(epoch, is_best=True)
                
                # Save regular checkpoint
                if epoch % self.config.get('checkpoint_interval', 10) == 0:
                    self.save_checkpoint(epoch, is_best=False)
                
                # Early stopping check
                if self.should_early_stop(epoch):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            self.state.status = 'completed'
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            self.state.status = 'failed'
            raise
        
        finally:
            self.state.end_time = datetime.now()
            self.save_experiment_state()
            
            # Final test evaluation
            if self.state.status == 'completed':
                self.logger.info("Running final test evaluation...")
                test_metrics = self.test()
                self.log_test_metrics(test_metrics)
        
        self.logger.info("Training completed")
    
    def log_epoch_metrics(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ):
        """Log metrics for an epoch."""
        # Console logging
        self.logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics.get('accuracy', 0):.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
        )
        
        # Tensorboard logging
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{metric_name}', value, epoch)
        
        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f'Validation/{metric_name}', value, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)
    
    def log_test_metrics(self, test_metrics: Dict[str, float]):
        """Log test metrics."""
        self.logger.info("Test Results:")
        for metric_name, value in test_metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")
        
        # Save test results
        results_file = os.path.join(self.results_dir, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
    
    def should_early_stop(self, epoch: int) -> bool:
        """Check if early stopping should be triggered."""
        early_stopping_config = self.config.get('early_stopping')
        if not early_stopping_config:
            return False
        
        patience = early_stopping_config.get('patience', 10)
        epochs_without_improvement = epoch - self.state.best_epoch
        
        return epochs_without_improvement >= patience
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'experiment_state': self.state.to_dict(),
            'config': self.config,
            'best_metric': self.state.best_metric,
            'best_epoch': self.state.best_epoch
        }
        
        # Save regular checkpoint
        filename = f'checkpoint_epoch_{epoch}.pth'
        self.checkpoint_manager.save_checkpoint(checkpoint_state, filename)
        
        # Save best checkpoint
        if is_best:
            self.checkpoint_manager.save_checkpoint(checkpoint_state, 'best_model.pth')
            self.logger.info(f"Saved new best model at epoch {epoch}")
        
        # Save latest checkpoint
        self.checkpoint_manager.save_checkpoint(checkpoint_state, 'latest.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'experiment_state' in checkpoint:
            self.state.from_dict(checkpoint['experiment_state'])
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def load_latest_checkpoint(self):
        """Load the latest checkpoint."""
        try:
            latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
            if os.path.exists(latest_path):
                self.load_checkpoint(latest_path)
                self.logger.info("Resumed from latest checkpoint")
            else:
                self.logger.info("No checkpoint found, starting from scratch")
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
    
    def save_experiment_state(self):
        """Save experiment state to file."""
        state_file = os.path.join(self.experiment_dir, 'experiment_state.json')
        with open(state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    def _move_batch_to_device(self, batch: Any) -> Any:
        """Move batch to the specified device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return [self._move_batch_to_device(item) for item in batch]
        elif isinstance(batch, dict):
            return {key: self._move_batch_to_device(value) for key, value in batch.items()}
        else:
            return batch
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'writer'):
            self.writer.close()
        
        self.logger.info("Experiment cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


if __name__ == "__main__":
    # Example usage and testing
    class DummyExperiment(BaseExperiment):
        """Dummy experiment for testing."""
        
        def create_model(self):
            return nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 5)
            )
        
        def create_dataloaders(self):
            # Create dummy datasets
            train_data = [(torch.randn(10), torch.randint(0, 5, (1,)).item()) for _ in range(100)]
            val_data = [(torch.randn(10), torch.randint(0, 5, (1,)).item()) for _ in range(20)]
            test_data = [(torch.randn(10), torch.randint(0, 5, (1,)).item()) for _ in range(20)]
            
            train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
            
            return train_loader, val_loader, test_loader
        
        def create_optimizer(self, model):
            return optim.Adam(model.parameters(), lr=0.001)
        
        def forward_pass(self, batch, training=True):
            audio, targets = batch
            self.logger.debug(f"Audio device: {audio.device}, Model device: {next(self.model.parameters()).device}")
            audio = audio.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(audio)
            self.logger.debug(f"Logits device: {logits.device}, Targets device: {targets.device}")
            loss = self.criterion(logits, targets)
            predictions = torch.argmax(logits, dim=1)
            return loss, predictions, targets
    
    # Test the base experiment
    config = {
        'primary_metric': 'accuracy',
        'log_interval': 10,
        'checkpoint_interval': 5,
        'early_stopping': {'patience': 5}
    }
    
    experiment = DummyExperiment(
        config=config,
        experiment_name='test_experiment',
        output_dir='./test_outputs'
    )
    
    print("Setting up experiment...")
    experiment.setup()
    
    print("Training for 3 epochs...")
    experiment.train(num_epochs=3)
    
    print("Experiment completed!")
    experiment.cleanup()