"""
Base Model Interface and Abstract Classes for C-HiLAP Project.

This module provides unified interfaces and abstract base classes for all machine
learning models in the C-HiLAP system, ensuring consistent API design and
standardized model lifecycle management.

Author: C-HiLAP Project
Date: 2025
"""

import numpy as np
import warnings
import json
import pickle
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

# ML Framework imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning models will be disabled.")

try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score, roc_curve
    )
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some functionality will be limited.")


class ModelType(Enum):
    """Enumeration of supported model types."""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    AUTOENCODER = "autoencoder"
    CLUSTERING = "clustering"


class TaskType(Enum):
    """Enumeration of supported task types."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    REPRESENTATION_LEARNING = "representation_learning"


class OptimizationType(Enum):
    """Enumeration of optimization algorithms."""
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAMW = "adamw"
    LBFGS = "lbfgs"


@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    
    # Model architecture
    model_type: ModelType = ModelType.CLASSIFIER
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])

    # 添加 activation 字段
    activation: str = 'relu'
    output_activation: Optional[str] = None  # 输出层激活函数
    activation_params: Dict[str, Any] = field(default_factory=dict)  # 激活函数参数
    
    # 批量归一化相关
    batch_norm: bool = True
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    validation_split: float = 0.2
    
    # Optimization
    optimizer_type: OptimizationType = OptimizationType.ADAM
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Regularization
    dropout_rate: float = 0.2
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Loss function
    loss_function: str = "cross_entropy"  # "cross_entropy", "mse", "mae", "bce"
    class_weights: Optional[Dict[int, float]] = None
    
    # Data preprocessing
    normalize_features: bool = True
    feature_scaling: str = "standard"  # "standard", "minmax", "robust"
    
    # Model persistence
    save_best_model: bool = True
    checkpoint_frequency: int = 10  # Save every N epochs
    
    # Hardware settings
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    random_seed: int = 42
    
    # Logging and monitoring
    verbose: bool = True
    log_metrics_frequency: int = 1
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        
        # 验证 activation
        valid_activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu', 'elu', 'swish']
        
        if self.activation not in valid_activations:
            raise ValueError(f"Invalid activation '{self.activation}'. Must be one of {valid_activations}")
        
        # 验证 output_activation（如果指定）
        if self.output_activation and self.output_activation not in valid_activations + ['softmax']:
            raise ValueError(f"Invalid output activation '{self.output_activation}'")
            
        if self.input_dim is not None and self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        
        if self.output_dim is not None and self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")


@dataclass
class TrainingMetrics:
    """Container for training metrics and history."""
    
    # Loss history
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    
    # Accuracy history
    train_accuracy_history: List[float] = field(default_factory=list)
    val_accuracy_history: List[float] = field(default_factory=list)
    
    # Other metrics
    train_metrics_history: Dict[str, List[float]] = field(default_factory=dict)
    val_metrics_history: Dict[str, List[float]] = field(default_factory=dict)
    
    # Training info
    epoch_times: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    # Best model info
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    best_val_accuracy: float = 0.0
    
    # Training status
    training_completed: bool = False
    early_stopped: bool = False
    total_training_time: float = 0.0
    
    def update_epoch(self, epoch: int, train_loss: float, val_loss: float,
                    train_accuracy: float = None, val_accuracy: float = None,
                    additional_metrics: Dict[str, Tuple[float, float]] = None,
                    epoch_time: float = None, learning_rate: float = None):
        """Update metrics for a single epoch."""
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        
        if train_accuracy is not None:
            self.train_accuracy_history.append(train_accuracy)
        if val_accuracy is not None:
            self.val_accuracy_history.append(val_accuracy)
            
            # Update best model tracking
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
        
        if additional_metrics:
            for metric_name, (train_val, val_val) in additional_metrics.items():
                if metric_name not in self.train_metrics_history:
                    self.train_metrics_history[metric_name] = []
                    self.val_metrics_history[metric_name] = []
                self.train_metrics_history[metric_name].append(train_val)
                self.val_metrics_history[metric_name].append(val_val)
        
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        return {
            'total_epochs': len(self.train_loss_history),
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'final_train_loss': self.train_loss_history[-1] if self.train_loss_history else None,
            'final_val_loss': self.val_loss_history[-1] if self.val_loss_history else None,
            'training_completed': self.training_completed,
            'early_stopped': self.early_stopped,
            'total_training_time': self.total_training_time,
            'average_epoch_time': np.mean(self.epoch_times) if self.epoch_times else None
        }


class ModelCallback(ABC):
    """Abstract base class for training callbacks."""
    
    @abstractmethod
    def on_training_start(self, model, config: ModelConfig):
        """Called at the start of training."""
        pass
    
    @abstractmethod
    def on_epoch_start(self, model, epoch: int, config: ModelConfig):
        """Called at the start of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, model, epoch: int, metrics: Dict[str, float], config: ModelConfig):
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_training_end(self, model, metrics: TrainingMetrics, config: ModelConfig):
        """Called at the end of training."""
        pass


class EarlyStoppingCallback(ModelCallback):
    """Early stopping callback implementation."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def on_training_start(self, model, config: ModelConfig):
        """Reset early stopping state."""
        self.best_value = float('inf') if self.mode == 'min' else -float('inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def on_epoch_start(self, model, epoch: int, config: ModelConfig):
        """No action needed at epoch start."""
        pass
    
    def on_epoch_end(self, model, epoch: int, metrics: Dict[str, float], config: ModelConfig):
        """Check for early stopping condition."""
        if self.monitor not in metrics:
            warnings.warn(f"Monitor metric '{self.monitor}' not found in metrics")
            return
        
        current_value = metrics[self.monitor]
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.should_stop = True
            logging.info(f"Early stopping triggered at epoch {epoch}")
    
    def on_training_end(self, model, metrics: TrainingMetrics, config: ModelConfig):
        """Mark if training was early stopped."""
        if self.should_stop:
            metrics.early_stopped = True


class LearningRateSchedulerCallback(ModelCallback):
    """Learning rate scheduling callback."""
    
    def __init__(self, schedule_type: str = 'step', **schedule_params):
        self.schedule_type = schedule_type
        self.schedule_params = schedule_params
        self.scheduler = None
    
    def on_training_start(self, model, config: ModelConfig):
        """Initialize scheduler."""
        if hasattr(model, 'optimizer'):
            if self.schedule_type == 'step':
                from torch.optim.lr_scheduler import StepLR
                step_size = self.schedule_params.get('step_size', 30)
                gamma = self.schedule_params.get('gamma', 0.1)
                self.scheduler = StepLR(model.optimizer, step_size=step_size, gamma=gamma)
            
            elif self.schedule_type == 'exponential':
                from torch.optim.lr_scheduler import ExponentialLR
                gamma = self.schedule_params.get('gamma', 0.95)
                self.scheduler = ExponentialLR(model.optimizer, gamma=gamma)
    
    def on_epoch_start(self, model, epoch: int, config: ModelConfig):
        """No action needed at epoch start."""
        pass
    
    def on_epoch_end(self, model, epoch: int, metrics: Dict[str, float], config: ModelConfig):
        """Update learning rate."""
        if self.scheduler:
            self.scheduler.step()
    
    def on_training_end(self, model, metrics: TrainingMetrics, config: ModelConfig):
        """No action needed at training end."""
        pass


class BaseModel(ABC):
    """
    Abstract base class for all models in the C-HiLAP system.
    
    This class defines the standard interface that all models must implement,
    ensuring consistency across different model types and frameworks.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.metrics = TrainingMetrics()
        self.is_fitted = False
        self.model = None
        self.callbacks: List[ModelCallback] = []
        
        # Set random seed for reproducibility
        self._set_random_seed(config.random_seed)
        
        # Setup device
        self.device = self._setup_device(config.device)
        
        # Initialize logging
        self.logger = self._setup_logging()
    
    def _set_random_seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        if PYTORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def _setup_device(self, device: str):
        """Setup computation device."""
        if not PYTORCH_AVAILABLE:
            return "cpu"
        
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logger for the model."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        return logger
    
    def add_callback(self, callback: ModelCallback):
        """Add a training callback."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback_type: type):
        """Remove callbacks of specific type."""
        self.callbacks = [cb for cb in self.callbacks if not isinstance(cb, callback_type)]
    
    @abstractmethod
    def _build_model(self) -> Any:
        """Build the actual model architecture."""
        pass
    
    @abstractmethod
    def _compile_model(self):
        """Compile the model with optimizer and loss function."""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Test features
            y: True labels
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions = self.predict(X)
        
        # Default metrics based on task type
        if metrics is None:
            if self.config.task_type == TaskType.BINARY_CLASSIFICATION:
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            else:
                metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        results = {}
        
        if SKLEARN_AVAILABLE:
            for metric in metrics:
                try:
                    if metric == 'accuracy':
                        results[metric] = accuracy_score(y, predictions)
                    elif metric == 'precision':
                        results[metric] = precision_score(y, predictions, average='binary')
                    elif metric == 'precision_macro':
                        results[metric] = precision_score(y, predictions, average='macro')
                    elif metric == 'recall':
                        results[metric] = recall_score(y, predictions, average='binary')
                    elif metric == 'recall_macro':
                        results[metric] = recall_score(y, predictions, average='macro')
                    elif metric == 'f1':
                        results[metric] = f1_score(y, predictions, average='binary')
                    elif metric == 'f1_macro':
                        results[metric] = f1_score(y, predictions, average='macro')
                    elif metric == 'auc':
                        if self.config.task_type == TaskType.BINARY_CLASSIFICATION:
                            proba = self.predict_proba(X)
                            if proba.shape[1] == 2:  # Binary classification
                                results[metric] = roc_auc_score(y, proba[:, 1])
                            else:
                                results[metric] = roc_auc_score(y, proba, multi_class='ovr')
                except Exception as e:
                    self.logger.warning(f"Failed to compute {metric}: {e}")
                    results[metric] = np.nan
        
        return results
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if supported by the model.
        
        Returns:
            Feature importance scores or None
        """
        # Default implementation returns None
        # Should be overridden by models that support feature importance
        return None
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'config': self.config,
            'metrics': self.metrics,
            'model_state': self._get_model_state(),
            'is_fitted': self.is_fitted,
            'model_class': self.__class__.__name__
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Verify model class matches
        if model_data['model_class'] != self.__class__.__name__:
            warnings.warn(f"Loading model of different class: {model_data['model_class']}")
        
        self.config = model_data['config']
        self.metrics = model_data['metrics']
        self.is_fitted = model_data['is_fitted']
        
        self._set_model_state(model_data['model_state'])
        
        self.logger.info(f"Model loaded from {filepath}")
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for saving."""
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]):
        """Set model state from loaded data."""
        pass
    
    def get_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.config
    
    def get_metrics(self) -> TrainingMetrics:
        """Get training metrics."""
        return self.metrics
    
    def summary(self) -> str:
        """Get model summary."""
        summary_lines = [
            f"Model: {self.__class__.__name__}",
            f"Type: {self.config.model_type.value}",
            f"Task: {self.config.task_type.value}",
            f"Input Dimension: {self.config.input_dim}",
            f"Output Dimension: {self.config.output_dim}",
            f"Fitted: {self.is_fitted}",
            f"Device: {self.device}"
        ]
        
        if self.is_fitted:
            metrics_summary = self.metrics.get_summary()
            summary_lines.extend([
                f"Training Epochs: {metrics_summary['total_epochs']}",
                f"Best Validation Loss: {metrics_summary['best_val_loss']:.6f}",
                f"Best Validation Accuracy: {metrics_summary['best_val_accuracy']:.4f}",
                f"Training Time: {metrics_summary['total_training_time']:.2f}s"
            ])
        
        return "\n".join(summary_lines)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"


class SklearnCompatibleModel(BaseModel):
    """
    Base class for sklearn-compatible models.
    
    This class provides sklearn-style interface while maintaining
    the BaseModel abstraction.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.classes_ = None
        self.n_features_in_ = None
        self.n_classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> 'SklearnCompatibleModel':
        """Sklearn-style fit method."""
        # Store data properties
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Update config with inferred dimensions
        if self.config.input_dim is None:
            self.config.input_dim = self.n_features_in_
        if self.config.output_dim is None:
            self.config.output_dim = self.n_classes_
        
        return self
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Sklearn-style score method (returns accuracy)."""
        results = self.evaluate(X, y, metrics=['accuracy'])
        return results.get('accuracy', 0.0)


def create_default_callbacks(config: ModelConfig) -> List[ModelCallback]:
    """Create default callbacks based on configuration."""
    callbacks = []
    
    # Add early stopping if configured
    if config.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        ))
    
    return callbacks


def validate_model_inputs(X: np.ndarray, y: np.ndarray, 
                         task_type: TaskType) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and preprocess model inputs.
    
    Args:
        X: Input features
        y: Target labels
        task_type: Type of task
        
    Returns:
        Validated and preprocessed X, y
    """
    # Validate shapes
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}")
    
    # Validate task-specific requirements
    if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
        unique_labels = np.unique(y)
        if task_type == TaskType.BINARY_CLASSIFICATION and len(unique_labels) != 2:
            raise ValueError(f"Binary classification requires exactly 2 classes, got {len(unique_labels)}")
    
    return X, y


# Utility functions for model creation and management

def create_model_config(model_type: str = "classifier",
                       task_type: str = "binary_classification",
                        activation: str = 'relu', 
                       **kwargs) -> ModelConfig:
    """
    Create model configuration with validation.
    
    Args:
        model_type: Type of model
        task_type: Type of task
        **kwargs: Additional configuration parameters
        
    Returns:
        Validated ModelConfig instance
    """
    # Convert string enums
    if isinstance(model_type, str):
        model_type = ModelType(model_type)
    if isinstance(task_type, str):
        task_type = TaskType(task_type)
    
    config = ModelConfig(
        model_type=model_type,
        task_type=task_type,
        activation=activation,
        **kwargs
    )
    
    return config


def get_model_info(model: BaseModel) -> Dict[str, Any]:
    """
    Get comprehensive information about a model.
    
    Args:
        model: Model instance
        
    Returns:
        Dictionary containing model information
    """
    info = {
        'model_class': model.__class__.__name__,
        'model_type': model.config.model_type.value,
        'task_type': model.config.task_type.value,
        'is_fitted': model.is_fitted,
        'device': model.device,
        'config': model.config,
        'parameter_count': getattr(model, 'parameter_count', None)
    }
    
    if model.is_fitted:
        info['metrics'] = model.metrics.get_summary()
        
        # Add feature importance if available
        feature_importance = model.get_feature_importance()
        if feature_importance is not None:
            info['feature_importance'] = feature_importance
    
    return info


def compare_models(models: List[BaseModel], 
                  X_test: np.ndarray, 
                  y_test: np.ndarray,
                  metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same test set.
    
    Args:
        models: List of fitted models
        X_test: Test features
        y_test: Test labels
        metrics: Metrics to compute
        
    Returns:
        Dictionary of model names to metric values
    """
    results = {}
    
    for model in models:
        if not model.is_fitted:
            warnings.warn(f"Model {model.__class__.__name__} is not fitted, skipping")
            continue
        
        try:
            model_results = model.evaluate(X_test, y_test, metrics)
            results[model.__class__.__name__] = model_results
        except Exception as e:
            warnings.warn(f"Failed to evaluate {model.__class__.__name__}: {e}")
            results[model.__class__.__name__] = {}
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Base Model Interface...")
    
    # Test configuration creation
    config = create_model_config(
        model_type="classifier",
        task_type="binary_classification",
        input_dim=100,
        output_dim=2,
        learning_rate=0.001,
        batch_size=32
    )
    
    print(f"✓ Model configuration created: {config.model_type.value}")
    print(f"  Task type: {config.task_type.value}")
    print(f"  Input dimension: {config.input_dim}")
    print(f"  Output dimension: {config.output_dim}")
    
    # Test training metrics
    metrics = TrainingMetrics()
    metrics.update_epoch(0, 0.5, 0.4, 0.8, 0.85, epoch_time=1.2, learning_rate=0.001)
    metrics.update_epoch(1, 0.3, 0.35, 0.85, 0.87, epoch_time=1.1, learning_rate=0.001)
    
    summary = metrics.get_summary()
    print(f"✓ Training metrics summary:")
    print(f"  Best validation loss: {summary['best_val_loss']:.3f}")
    print(f"  Best validation accuracy: {summary['best_val_accuracy']:.3f}")
    print(f"  Total epochs: {summary['total_epochs']}")
    
    # Test callbacks
    early_stopping = EarlyStoppingCallback(patience=5, min_delta=0.01)
    print(f"✓ Early stopping callback created with patience: {early_stopping.patience}")
    
    # Test input validation
    X = np.random.randn(100, 50)
    y = np.random.choice([0, 1], 100)
    
    X_val, y_val = validate_model_inputs(X, y, TaskType.BINARY_CLASSIFICATION)
    print(f"✓ Input validation passed: X shape {X_val.shape}, y shape {y_val.shape}")
    
    print("Base Model Interface testing completed!")