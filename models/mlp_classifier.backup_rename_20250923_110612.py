"""
Multi-Layer Perceptron Classifier for C-HiLAP Project.

This module implements MLP classifiers using both PyTorch and sklearn backends,
serving as baseline models for comparing against chaotic neural networks.
Inherits from BaseModel to ensure consistent API across the project.

Author: C-HiLAP Project
Date: 2025
"""

import numpy as np
import warnings
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
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
    from Model.data.dataset_loader import create_speaker_dataloaders, LibriSpeechChaoticDataset
    from Model.features.traditional_features import MelExtractor, MFCCExtractor
    from Model.experiments.base_experiment import BaseExperiment
    print("All required modules imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please check that all required files exist and the project structure is correct")
    
try:
    from base_model import (
        BaseModel, SklearnCompatibleModel, ModelConfig, TrainingMetrics,
        ModelType, TaskType, OptimizationType, create_default_callbacks,
        validate_model_inputs
    )
except ImportError:
    from models.base_model import (
        BaseModel, SklearnCompatibleModel, ModelConfig, TrainingMetrics,
        ModelType, TaskType, OptimizationType, create_default_callbacks,
        validate_model_inputs
    )

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. PyTorch-based MLP will be disabled.")

# Sklearn imports
try:
    from sklearn.neural_network import MLPClassifier as SklearnMLP
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Sklearn-based MLP will be disabled.")

# Additional utilities
from sklearn.utils.class_weight import compute_class_weight


class MLPNetwork(nn.Module):
    """PyTorch MLP network implementation."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int],
                 activation: str = 'relu', dropout_rate: float = 0.2,
                 batch_norm: bool = True, output_activation: str = None):
        super(MLPNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Build network layers
        layers = []
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(layer_dims) - 1):
            # Linear layer
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            
            # Don't add activation/dropout/batch_norm to output layer
            if i < len(layer_dims) - 2:
                # Batch normalization
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                
                # Activation function
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU())
                elif activation.lower() == 'tanh':
                    layers.append(nn.Tanh())
                elif activation.lower() == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation.lower() == 'leaky_relu':
                    layers.append(nn.LeakyReLU())
                elif activation.lower() == 'gelu':
                    layers.append(nn.GELU())
                else:
                    layers.append(nn.ReLU())  # Default to ReLU
                
                # Dropout
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
        
        # Add output activation if specified
        if output_activation:
            if output_activation.lower() == 'softmax':
                layers.append(nn.Softmax(dim=1))
            elif output_activation.lower() == 'sigmoid':
                layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation.lower() in ['relu', 'leaky_relu']:
                    # He initialization for ReLU-like activations
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', 
                                          nonlinearity=self.activation.lower())
                else:
                    # Xavier initialization for tanh/sigmoid
                    nn.init.xavier_normal_(module.weight)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPClassifier(BaseModel):
    """MLP classifier implementation."""
    
    def __init__(self, config: ModelConfig):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MLPClassifier")
        
        super().__init__(config)
        
        # Additional MLP-specific configuration
        self.activation = getattr(config, 'activation', 'relu')
        self.batch_norm = getattr(config, 'batch_norm', True)
        self.output_activation = getattr(config, 'output_activation', None)
        
        # Training components
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        self.label_encoder = None
        
        # Build model if dimensions are specified
        if config.input_dim and config.output_dim:
            self.model = self._build_model()
            self._compile_model()
    
    def _build_model(self) -> MLPNetwork:
        """Build the MLP network."""
        network = MLPNetwork(
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.activation,
            dropout_rate=self.config.dropout_rate,
            batch_norm=self.batch_norm,
            output_activation=self.output_activation
        )
        
        # Move to device
        network = network.to(self.device)
        self.parameter_count = network.get_parameter_count()
        
        return network
    
    def _compile_model(self):
        """Compile model with optimizer and loss function."""
        if self.model is None:
            return
        
        # Setup optimizer
        if self.config.optimizer_type == OptimizationType.ADAM:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **self.config.optimizer_params
            )
        elif self.config.optimizer_type == OptimizationType.SGD:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.optimizer_params.get('momentum', 0.9),
                **{k: v for k, v in self.config.optimizer_params.items() if k != 'momentum'}
            )
        elif self.config.optimizer_type == OptimizationType.ADAMW:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **self.config.optimizer_params
            )
        elif self.config.optimizer_type == OptimizationType.RMSPROP:
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **self.config.optimizer_params
            )
        
        # Setup loss function
        if self.config.loss_function == "cross_entropy":
            # Handle class weights if provided
            weight = None
            if self.config.class_weights:
                weight = torch.FloatTensor([
                    self.config.class_weights.get(i, 1.0) 
                    for i in range(self.config.output_dim)
                ]).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight)
            
        elif self.config.loss_function == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.config.loss_function == "mse":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'MLPClassifier':
        """Train the MLP classifier."""
        # Validate inputs
        X, y = validate_model_inputs(X, y, self.config.task_type)
        
        # Setup data preprocessing
        self._setup_preprocessing(X, y)
        
        # Preprocess data
        X_processed = self._preprocess_features(X)
        y_processed = self._preprocess_labels(y)
        
        # Handle validation data
        if X_val is not None and y_val is not None:
            X_val, y_val = validate_model_inputs(X_val, y_val, self.config.task_type)
            X_val_processed = self._preprocess_features(X_val)
            y_val_processed = self._preprocess_labels(y_val)
        else:
            # Split training data for validation
            if self.config.validation_split > 0:
                X_processed, X_val_processed, y_processed, y_val_processed = train_test_split(
                    X_processed, y_processed, 
                    test_size=self.config.validation_split,
                    random_state=self.config.random_seed,
                    stratify=y_processed
                )
            else:
                X_val_processed = y_val_processed = None
        
        # Build model if not already built
        if self.model is None:
            self.config.input_dim = X_processed.shape[1]
            self.config.output_dim = len(np.unique(y_processed))
            self.model = self._build_model()
            self._compile_model()
        
        # Create data loaders
        train_loader = self._create_data_loader(X_processed, y_processed, shuffle=True)
        val_loader = None
        if X_val_processed is not None:
            val_loader = self._create_data_loader(X_val_processed, y_val_processed, shuffle=False)
        
        # Setup callbacks
        if not self.callbacks:
            self.callbacks = create_default_callbacks(self.config)
        
        # Training loop
        self._train_loop(train_loader, val_loader)
        
        self.is_fitted = True
        return self
    
    def _setup_preprocessing(self, X: np.ndarray, y: np.ndarray):
        """Setup preprocessing components."""
        # Feature scaling
        if self.config.normalize_features:
            if self.config.feature_scaling == 'standard':
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            elif self.config.feature_scaling == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
            elif self.config.feature_scaling == 'robust':
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
            
            if self.scaler:
                self.scaler.fit(X)
        
        # Label encoding
        if self.config.task_type in [TaskType.MULTICLASS_CLASSIFICATION, TaskType.BINARY_CLASSIFICATION]:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
    
    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """Preprocess features."""
        if self.scaler:
            return self.scaler.transform(X)
        return X
    
    def _preprocess_labels(self, y: np.ndarray) -> np.ndarray:
        """Preprocess labels."""
        if self.label_encoder:
            return self.label_encoder.transform(y)
        return y
    
    def _create_data_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create PyTorch data loader."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)
    
    def _train_loop(self, train_loader: DataLoader, val_loader: Optional[DataLoader]):
        """Main training loop."""
        start_time = time.time()
        
        # Call training start callbacks
        for callback in self.callbacks:
            callback.on_training_start(self, self.config)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Call epoch start callbacks
            for callback in self.callbacks:
                callback.on_epoch_start(self, epoch, self.config)
            
            # Training phase
            train_loss, train_accuracy = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss = val_accuracy = None
            if val_loader:
                val_loss, val_accuracy = self._validate_epoch(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update metrics
            self.metrics.update_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss or train_loss,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                epoch_time=epoch_time,
                learning_rate=current_lr
            )
            
            # Logging
            if self.config.verbose and epoch % self.config.log_metrics_frequency == 0:
                log_msg = f"Epoch {epoch+1}/{self.config.num_epochs} - "
                log_msg += f"Loss: {train_loss:.4f}"
                if train_accuracy is not None:
                    log_msg += f" - Acc: {train_accuracy:.4f}"
                if val_loss is not None:
                    log_msg += f" - Val Loss: {val_loss:.4f}"
                if val_accuracy is not None:
                    log_msg += f" - Val Acc: {val_accuracy:.4f}"
                log_msg += f" - Time: {epoch_time:.2f}s"
                self.logger.info(log_msg)
            
            # Call epoch end callbacks
            epoch_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss or train_loss,
                'train_accuracy': train_accuracy or 0,
                'val_accuracy': val_accuracy or 0
            }
            
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, epoch_metrics, self.config)
            
            # Check for early stopping (from callbacks)
            early_stop = any(
                getattr(callback, 'should_stop', False) 
                for callback in self.callbacks
            )
            
            if early_stop:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final metrics update
        self.metrics.training_completed = True
        self.metrics.total_training_time = time.time() - start_time
        
        # Call training end callbacks
        for callback in self.callbacks:
            callback.on_training_end(self, self.metrics, self.config)
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, Optional[float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            if self.config.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else None
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Optional[float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                if self.config.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else None
        
        return avg_loss, accuracy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if self.config.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                predictions = outputs.argmax(dim=1).cpu().numpy()
                
                # Decode labels if encoder was used
                if self.label_encoder:
                    predictions = self.label_encoder.inverse_transform(predictions)
            else:
                predictions = outputs.cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.config.task_type not in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            raise ValueError("predict_proba is only available for classification tasks")
        
        self.model.eval()
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
        
        return probabilities
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance based on weight magnitudes."""
        if not self.is_fitted or self.model is None:
            return None
        
        # Get first layer weights
        first_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        
        if first_layer is None:
            return None
        
        # Calculate importance as absolute sum of weights per input feature
        weights = first_layer.weight.data.cpu().numpy()  # Shape: (hidden_dim, input_dim)
        importance = np.sum(np.abs(weights), axis=0)  # Sum over hidden dimension
        
        # Normalize
        importance = importance / np.sum(importance)
        
        return importance
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for saving."""
        state = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'parameter_count': getattr(self, 'parameter_count', None)
        }
        return state
    
    def _set_model_state(self, state: Dict[str, Any]):
        """Set model state from loaded data."""
        # Rebuild model if needed
        if self.model is None and self.config.input_dim and self.config.output_dim:
            self.model = self._build_model()
            self._compile_model()
        
        # Load model state
        if self.model and state['model_state_dict']:
            self.model.load_state_dict(state['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer and state['optimizer_state_dict']:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        # Load preprocessors
        self.scaler = state.get('scaler')
        self.label_encoder = state.get('label_encoder')
        self.parameter_count = state.get('parameter_count')


class SklearnMLPClassifier(SklearnCompatibleModel):
    """Sklearn-based MLP classifier wrapper."""
    
    def __init__(self, config: ModelConfig):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for SklearnMLPClassifier")
        
        super().__init__(config)
        
        # Convert config to sklearn parameters
        sklearn_params = self._convert_config_to_sklearn_params(config)
        self.model = SklearnMLP(**sklearn_params)
        self.scaler = None
    
    def _convert_config_to_sklearn_params(self, config: ModelConfig) -> Dict[str, Any]:
        """Convert ModelConfig to sklearn MLP parameters."""
        params = {
            'hidden_layer_sizes': tuple(config.hidden_dims),
            'learning_rate_init': config.learning_rate,
            'max_iter': config.num_epochs,
            'random_state': config.random_seed,
            'verbose': config.verbose
        }
        
        # Map activation function
        activation_map = {
            'relu': 'relu',
            'tanh': 'tanh',
            'sigmoid': 'logistic'
        }
        activation = getattr(config, 'activation', 'relu')
        params['activation'] = activation_map.get(activation, 'relu')
        
        # Map optimizer
        optimizer_map = {
            OptimizationType.ADAM: 'adam',
            OptimizationType.SGD: 'sgd',
            OptimizationType.LBFGS: 'lbfgs'
        }
        params['solver'] = optimizer_map.get(config.optimizer_type, 'adam')
        
        # Early stopping
        if config.early_stopping_patience > 0:
            params['early_stopping'] = True
            params['n_iter_no_change'] = config.early_stopping_patience
            params['validation_fraction'] = config.validation_split
        
        # Regularization
        params['alpha'] = config.weight_decay
        
        return params
    
    def _build_model(self) -> Any:
        """Model is already built in __init__."""
        return self.model
    
    def _compile_model(self):
        """No compilation needed for sklearn."""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'SklearnMLPClassifier':
        """Train the sklearn MLP."""
        # Call parent fit for setup
        super().fit(X, y, X_val, y_val)
        
        # Validate inputs
        X, y = validate_model_inputs(X, y, self.config.task_type)
        
        # Feature scaling
        if self.config.normalize_features:
            if self.config.feature_scaling == 'standard':
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            elif self.config.feature_scaling == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
            elif self.config.feature_scaling == 'robust':
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
            
            if self.scaler:
                X = self.scaler.fit_transform(X)
        
        # Train model
        start_time = time.time()
        self.model.fit(X, y)
        training_time = time.time() - start_time
        
        # Update metrics
        self.metrics.total_training_time = training_time
        self.metrics.training_completed = True
        
        # Get final loss if available
        if hasattr(self.model, 'loss_'):
            self.metrics.train_loss_history = [self.model.loss_]
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply scaling if used during training
        if self.scaler:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply scaling if used during training
        if self.scaler:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance based on weight magnitudes."""
        if not self.is_fitted:
            return None
        
        # Get first layer weights
        if hasattr(self.model, 'coefs_') and len(self.model.coefs_) > 0:
            weights = self.model.coefs_[0]  # First layer weights
            importance = np.sum(np.abs(weights), axis=1)  # Sum over hidden units
            importance = importance / np.sum(importance)  # Normalize
            return importance
        
        return None
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for saving."""
        return {
            'sklearn_model': self.model,
            'scaler': self.scaler
        }
    
    def _set_model_state(self, state: Dict[str, Any]):
        """Set model state from loaded data."""
        self.model = state['sklearn_model']
        self.scaler = state.get('scaler')


def create_mlp_classifier(backend: str = 'pytorch', config: ModelConfig = None, **kwargs) -> BaseModel:
    """
    Factory function to create MLP classifier with specified backend.
    
    Args:
        backend: 'pytorch' or 'sklearn'
        config: Model configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        MLP classifier instance
    """
    # Create default config if not provided
    if config is None:
        from .base_model import create_model_config
        config = create_model_config(
            model_type="classifier",
            task_type="binary_classification",
            **kwargs
        )
    
    # Update config with additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create classifier based on backend
    if backend.lower() == 'pytorch':
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Install torch to use PyTorch backend.")
        return MLPClassifier(config)
    
    elif backend.lower() == 'sklearn':
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is not available. Install scikit-learn to use sklearn backend.")
        return SklearnMLPClassifier(config)
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'pytorch' or 'sklearn'.")


MLPClassifier = MLPClassifier  # Backward compatibility alias

PyTorchMLPClassifier = MLPClassifier  # Backward compatibility alias

__all__ = ['MLPClassifier']

if __name__ == "__main__":
    # Example usage and testing
    print("Testing MLP Classifier...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice(n_classes, n_samples)
    
    print(f"Generated synthetic data: {X.shape} features, {len(np.unique(y))} classes")
    
    # Test both backends if available
    backends = []
    if PYTORCH_AVAILABLE:
        backends.append('pytorch')
    if SKLEARN_AVAILABLE:
        backends.append('sklearn')
    
    for backend in backends:
        print(f"\n=== Testing {backend.upper()} Backend ===")
        
        try:
            # Create configuration
            from .base_model import create_model_config
            config = create_model_config(
                model_type="classifier",
                task_type="multiclass_classification",
                input_dim=n_features,
                output_dim=n_classes,
                hidden_dims=[64, 32],
                learning_rate=0.001,
                num_epochs=20 if backend == 'pytorch' else 100,
                batch_size=32,
                verbose=True
            )
            
            # Create classifier
            classifier = create_mlp_classifier(backend=backend, config=config)
            print(f"✓ {backend.capitalize()} MLP classifier created")
            print(f"  Model type: {classifier.config.model_type.value}")
            print(f"  Hidden layers: {classifier.config.hidden_dims}")
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            print(f"Training {backend} MLP...")
            start_time = time.time()
            classifier.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            print(f"✓ Training completed in {training_time:.2f}s")
            
            # Make predictions
            predictions = classifier.predict(X_test)
            probabilities = classifier.predict_proba(X_test)
            
            print(f"✓ Predictions generated: {predictions.shape}")
            print(f"✓ Probabilities generated: {probabilities.shape}")
            
            # Evaluate model
            metrics = classifier.evaluate(X_test, y_test)
            print(f"✓ Evaluation metrics:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
            
            # Get feature importance if available
            feature_importance = classifier.get_feature_importance()
            if feature_importance is not None:
                print(f"✓ Feature importance computed: shape {feature_importance.shape}")
                print(f"    Top 3 features: {np.argsort(feature_importance)[-3:]}")
            
            # Test model summary
            print(f"✓ Model summary:")
            print(classifier.summary())
            
        except Exception as e:
            print(f"✗ Error testing {backend} backend: {e}")
    
    print("\nMLP Classifier testing completed!")