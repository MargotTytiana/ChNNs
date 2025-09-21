"""
Checkpoint Management System for Chaotic Speaker Recognition Project
Provides comprehensive model saving, loading, and checkpoint management
with support for different frameworks and automatic backup strategies.
"""

import os
import json
import pickle
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Callable
from collections import defaultdict
import warnings

# Import ML frameworks with error handling
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class CheckpointManager:
    """
    Comprehensive checkpoint management system for machine learning experiments.
    Supports automatic saving, loading, and management of model checkpoints
    with configurable backup strategies and metadata tracking.
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 experiment_name: Optional[str] = None,
                 max_checkpoints: int = 5,
                 save_frequency: int = 1,
                 auto_save_best: bool = True,
                 metric_for_best: str = "val_loss",
                 metric_mode: str = "min"):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
            max_checkpoints: Maximum number of checkpoints to keep
            save_frequency: Save checkpoint every N epochs
            auto_save_best: Whether to automatically save best model
            metric_for_best: Metric to track for best model
            metric_mode: "min" or "max" for the metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.max_checkpoints = max_checkpoints
        self.save_frequency = save_frequency
        self.auto_save_best = auto_save_best
        self.metric_for_best = metric_for_best
        self.metric_mode = metric_mode.lower()
        
        # Create experiment checkpoint directory
        self.experiment_dir = self.checkpoint_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoints and metrics
        self.checkpoint_history = []
        self.best_metric_value = float('inf') if self.metric_mode == 'min' else float('-inf')
        self.best_checkpoint_path = None
        
        # Metadata file
        self.metadata_file = self.experiment_dir / "checkpoint_metadata.json"
        
        # Load existing metadata if available
        self._load_metadata()
        
        print(f"CheckpointManager initialized for experiment: {self.experiment_name}")
        print(f"Checkpoint directory: {self.experiment_dir}")
    
    def _load_metadata(self):
        """Load existing checkpoint metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.checkpoint_history = metadata.get('checkpoint_history', [])
                self.best_metric_value = metadata.get('best_metric_value', 
                                                    float('inf') if self.metric_mode == 'min' else float('-inf'))
                self.best_checkpoint_path = metadata.get('best_checkpoint_path', None)
                
                print(f"Loaded existing metadata with {len(self.checkpoint_history)} checkpoints")
            
            except Exception as e:
                print(f"Warning: Could not load existing metadata: {e}")
                self.checkpoint_history = []
    
    def _save_metadata(self):
        """Save checkpoint metadata to file"""
        metadata = {
            'experiment_name': self.experiment_name,
            'checkpoint_history': self.checkpoint_history,
            'best_metric_value': self.best_metric_value,
            'best_checkpoint_path': self.best_checkpoint_path,
            'metric_for_best': self.metric_for_best,
            'metric_mode': self.metric_mode,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _is_better_metric(self, current_value: float, best_value: float) -> bool:
        """Check if current metric is better than best"""
        if self.metric_mode == 'min':
            return current_value < best_value
        else:
            return current_value > best_value
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit"""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by timestamp and remove oldest (except best checkpoint)
        sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['timestamp'])
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint_info in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint_info['path'])
            
            # Don't remove the best checkpoint
            if str(checkpoint_path) == self.best_checkpoint_path:
                continue
            
            try:
                if checkpoint_path.exists():
                    if checkpoint_path.is_dir():
                        shutil.rmtree(checkpoint_path)
                    else:
                        checkpoint_path.unlink()
                    print(f"Removed old checkpoint: {checkpoint_path}")
                
                # Remove from history
                self.checkpoint_history.remove(checkpoint_info)
                
            except Exception as e:
                print(f"Warning: Could not remove old checkpoint {checkpoint_path}: {e}")
    
    def save_checkpoint(self, 
                       model: Any,
                       optimizer: Optional[Any] = None,
                       scheduler: Optional[Any] = None,
                       epoch: int = 0,
                       metrics: Optional[Dict[str, float]] = None,
                       additional_info: Optional[Dict[str, Any]] = None,
                       checkpoint_name: Optional[str] = None) -> str:
        """
        Save a complete checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler to save
            epoch: Current epoch number
            metrics: Dictionary of metrics
            additional_info: Additional information to save
            checkpoint_name: Custom checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint name if not provided
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint_path = self.experiment_dir / f"{checkpoint_name}.pt"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'additional_info': additional_info or {}
        }
        
        # Save model based on framework
        if HAS_TORCH and isinstance(model, nn.Module):
            checkpoint_data.update({
                'model_state_dict': model.state_dict(),
                'model_class_name': model.__class__.__name__,
                'framework': 'pytorch'
            })
            
            if optimizer is not None:
                checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
                checkpoint_data['optimizer_class_name'] = optimizer.__class__.__name__
            
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                checkpoint_data['scheduler_class_name'] = scheduler.__class__.__name__
            
            # Save using PyTorch
            torch.save(checkpoint_data, checkpoint_path)
            
        elif HAS_TF and hasattr(model, 'save_weights'):
            # TensorFlow/Keras model
            model_dir = self.experiment_dir / f"{checkpoint_name}_tf_model"
            model_dir.mkdir(exist_ok=True)
            
            # Save TF model weights
            model.save_weights(str(model_dir / "model_weights"))
            
            # Save additional checkpoint data
            checkpoint_data.update({
                'framework': 'tensorflow',
                'model_config': model.get_config() if hasattr(model, 'get_config') else None
            })
            
            with open(model_dir / "checkpoint_data.json", 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            checkpoint_path = model_dir
            
        else:
            # Generic Python object - use pickle
            checkpoint_data.update({
                'model': model,
                'framework': 'generic'
            })
            
            with open(checkpoint_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            checkpoint_path = checkpoint_path.with_suffix('.pkl')
        
        # Update checkpoint history
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'checkpoint_name': checkpoint_name
        }
        
        self.checkpoint_history.append(checkpoint_info)
        
        # Check if this is the best checkpoint
        if self.auto_save_best and metrics and self.metric_for_best in metrics:
            current_metric = metrics[self.metric_for_best]
            
            if self._is_better_metric(current_metric, self.best_metric_value):
                self.best_metric_value = current_metric
                self.best_checkpoint_path = str(checkpoint_path)
                
                # Create a symbolic link to best checkpoint
                best_link_path = self.experiment_dir / "best_checkpoint"
                if best_link_path.exists():
                    best_link_path.unlink()
                
                # Copy instead of symlink for better compatibility
                if checkpoint_path.is_dir():
                    shutil.copytree(checkpoint_path, best_link_path)
                else:
                    shutil.copy2(checkpoint_path, best_link_path.with_suffix(checkpoint_path.suffix))
                
                print(f"New best checkpoint saved! {self.metric_for_best}: {current_metric:.4f}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save metadata
        self._save_metadata()
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, 
                       checkpoint_path: Optional[str] = None,
                       load_best: bool = False,
                       map_location: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a checkpoint
        
        Args:
            checkpoint_path: Path to specific checkpoint
            load_best: Whether to load the best checkpoint
            map_location: Device to map tensors to (PyTorch)
            
        Returns:
            Checkpoint data dictionary
        """
        # Determine which checkpoint to load
        if load_best:
            if self.best_checkpoint_path is None:
                raise ValueError("No best checkpoint found")
            checkpoint_path = self.best_checkpoint_path
        elif checkpoint_path is None:
            # Load latest checkpoint
            if not self.checkpoint_history:
                raise ValueError("No checkpoints found")
            checkpoint_path = self.checkpoint_history[-1]['path']
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load based on file type/framework
        if checkpoint_path.suffix == '.pt':
            # PyTorch checkpoint
            if not HAS_TORCH:
                raise ImportError("PyTorch not available for loading checkpoint")
            
            checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
            
        elif checkpoint_path.suffix == '.pkl':
            # Pickle checkpoint
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
                
        elif checkpoint_path.is_dir():
            # TensorFlow checkpoint directory
            checkpoint_data_file = checkpoint_path / "checkpoint_data.json"
            if checkpoint_data_file.exists():
                with open(checkpoint_data_file, 'r') as f:
                    checkpoint_data = json.load(f)
                checkpoint_data['weights_path'] = str(checkpoint_path / "model_weights")
            else:
                raise ValueError(f"Invalid checkpoint directory: {checkpoint_path}")
        
        else:
            raise ValueError(f"Unknown checkpoint format: {checkpoint_path}")
        
        print(f"Checkpoint loaded successfully from epoch {checkpoint_data.get('epoch', 'unknown')}")
        return checkpoint_data
    
    def restore_model(self, 
                     model: Any,
                     checkpoint_path: Optional[str] = None,
                     load_best: bool = False,
                     optimizer: Optional[Any] = None,
                     scheduler: Optional[Any] = None,
                     strict: bool = True) -> Dict[str, Any]:
        """
        Restore model from checkpoint
        
        Args:
            model: Model to restore
            checkpoint_path: Path to checkpoint
            load_best: Whether to load best checkpoint
            optimizer: Optimizer to restore
            scheduler: Scheduler to restore
            strict: Whether to strictly enforce state dict loading
            
        Returns:
            Checkpoint information
        """
        checkpoint_data = self.load_checkpoint(checkpoint_path, load_best)
        
        # Restore model based on framework
        if checkpoint_data.get('framework') == 'pytorch' and HAS_TORCH:
            if 'model_state_dict' in checkpoint_data:
                model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
                print("Model state restored from PyTorch checkpoint")
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                print("Optimizer state restored")
            
            if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                print("Scheduler state restored")
                
        elif checkpoint_data.get('framework') == 'tensorflow' and HAS_TF:
            if 'weights_path' in checkpoint_data:
                model.load_weights(checkpoint_data['weights_path'])
                print("Model weights restored from TensorFlow checkpoint")
                
        elif checkpoint_data.get('framework') == 'generic':
            # For generic objects, replace the model
            if 'model' in checkpoint_data:
                print("Warning: Loading generic model - this replaces the entire model object")
                return checkpoint_data
        
        return checkpoint_data
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        return sorted(self.checkpoint_history, key=lambda x: x['epoch'])
    
    def get_best_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the best checkpoint"""
        if self.best_checkpoint_path is None:
            return None
        
        return {
            'path': self.best_checkpoint_path,
            'metric_value': self.best_metric_value,
            'metric_name': self.metric_for_best,
            'metric_mode': self.metric_mode
        }
    
    def delete_checkpoint(self, checkpoint_path: str):
        """Delete a specific checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            if checkpoint_path.is_dir():
                shutil.rmtree(checkpoint_path)
            else:
                checkpoint_path.unlink()
            
            # Remove from history
            self.checkpoint_history = [
                cp for cp in self.checkpoint_history 
                if cp['path'] != str(checkpoint_path)
            ]
            
            self._save_metadata()
            print(f"Checkpoint deleted: {checkpoint_path}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    def export_checkpoint(self, checkpoint_path: str, export_path: str):
        """Export a checkpoint to a different location"""
        checkpoint_path = Path(checkpoint_path)
        export_path = Path(export_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if checkpoint_path.is_dir():
            shutil.copytree(checkpoint_path, export_path)
        else:
            shutil.copy2(checkpoint_path, export_path)
        
        print(f"Checkpoint exported to: {export_path}")
    
    def should_save_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be saved based on frequency"""
        return (epoch + 1) % self.save_frequency == 0
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get a summary of all checkpoints"""
        if not self.checkpoint_history:
            return {"total_checkpoints": 0}
        
        latest_checkpoint = max(self.checkpoint_history, key=lambda x: x['epoch'])
        
        summary = {
            "total_checkpoints": len(self.checkpoint_history),
            "latest_epoch": latest_checkpoint['epoch'],
            "latest_checkpoint": latest_checkpoint['path'],
            "best_checkpoint": self.best_checkpoint_path,
            "best_metric_value": self.best_metric_value,
            "metric_for_best": self.metric_for_best,
            "checkpoint_directory": str(self.experiment_dir)
        }
        
        return summary


class AutoCheckpoint:
    """Context manager for automatic checkpoint saving during training"""
    
    def __init__(self, 
                 checkpoint_manager: CheckpointManager,
                 model: Any,
                 optimizer: Optional[Any] = None,
                 scheduler: Optional[Any] = None):
        self.manager = checkpoint_manager
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Save emergency checkpoint on exception
            print("Exception occurred, saving emergency checkpoint...")
            self.manager.save_checkpoint(
                self.model, 
                self.optimizer, 
                self.scheduler,
                self.epoch,
                checkpoint_name=f"emergency_epoch_{self.epoch}"
            )
    
    def step(self, epoch: int, metrics: Optional[Dict[str, float]] = None):
        """Step function to call after each epoch"""
        self.epoch = epoch
        
        if self.manager.should_save_checkpoint(epoch):
            self.manager.save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                metrics
            )


# Example usage and testing
if __name__ == "__main__":
    # Test the checkpoint system
    print("Testing CheckpointManager...")
    
    # Create a simple test model
    if HAS_TORCH:
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        # Test PyTorch model
        model = TestModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir="test_checkpoints",
            experiment_name="test_experiment",
            max_checkpoints=3
        )
        
        # Test saving checkpoints
        for epoch in range(5):
            metrics = {
                'train_loss': 0.5 - epoch * 0.05,
                'val_loss': 0.6 - epoch * 0.04,
                'accuracy': 0.7 + epoch * 0.02
            }
            
            if manager.should_save_checkpoint(epoch):
                checkpoint_path = manager.save_checkpoint(
                    model, optimizer, None, epoch, metrics
                )
                print(f"Saved checkpoint for epoch {epoch}")
        
        # Test loading
        print("\nTesting checkpoint loading...")
        checkpoint_data = manager.load_checkpoint(load_best=True)
        print(f"Loaded best checkpoint from epoch {checkpoint_data['epoch']}")
        
        # Test model restoration
        new_model = TestModel()
        manager.restore_model(new_model, load_best=True)
        print("Model restored successfully")
        
        # Print summary
        summary = manager.get_checkpoint_summary()
        print(f"\nCheckpoint Summary: {summary}")
        
        # Test auto checkpoint context manager
        print("\nTesting AutoCheckpoint context manager...")
        with AutoCheckpoint(manager, model, optimizer) as auto_cp:
            for epoch in range(2):
                metrics = {'loss': 0.3 - epoch * 0.1}
                auto_cp.step(epoch, metrics)
        
    else:
        print("PyTorch not available, testing with generic objects...")
        
        # Test with generic Python object
        test_model = {"weights": [1, 2, 3, 4, 5], "bias": 0.1}
        
        manager = CheckpointManager(
            checkpoint_dir="test_checkpoints_generic",
            experiment_name="test_generic"
        )
        
        manager.save_checkpoint(
            test_model,
            epoch=1,
            metrics={'loss': 0.5},
            checkpoint_name="generic_test"
        )
        
        loaded = manager.load_checkpoint()
        print(f"Generic checkpoint loaded: {loaded['model']}")
    
    print("\nCheckpointManager testing completed!")