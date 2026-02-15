#!/usr/bin/env python3
"""
Experiment 1: Unified Training Script for Baseline Comparison

This script trains all models (baseline and chaotic) under identical conditions
for fair comparison. All checkpoints are saved in a unified format.

Features:
- Unified data loading for all models
- Consistent checkpoint format (.pth with best_model.pth)
- Same training hyperparameters
- Comprehensive logging and result saving

Usage:
    python run_experiment1.py --all
    python run_experiment1.py --model mel_mlp
    python run_experiment1.py --model mfcc_mlp
    python run_experiment1.py --list
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import importlib
import sys

# 清除可能的缓存
for mod_name in list(sys.modules.keys()):
    if 'hybrid_models' in mod_name or 'traditional_features' in mod_name:
        del sys.modules[mod_name]
# ============================================================
# Setup Imports
# ============================================================
def setup_imports():
    """Setup Python path for project imports."""
    current_file = Path(__file__).resolve()
    
    # scripts/run_experiment1.py -> Model/
    project_root = current_file.parent.parent
    
    paths = [
        str(project_root),
        str(project_root / 'experiments'),
        str(project_root / 'models'),
        str(project_root / 'features'),
        str(project_root / 'data'),
        str(project_root / 'utils'),
        str(project_root / 'evaluation'),
        str(project_root / 'core')
    ]
    
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    return project_root

PROJECT_ROOT = setup_imports()

# Import project modules - use correct module names
from data.dataset_loader import create_speaker_dataloaders
from models.hybrid_models import TraditionalMLPBaseline
from utils.reproducibility import set_seed
from features.traditional_features import MelSpectrogramExtractor, MFCCExtractor

# ============================================================
# Configuration
# ============================================================
DEFAULT_CONFIG = {
    # Data settings
    'data_dir': '/scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2',
    'sample_rate': 16000,
    'max_length': 3.0,
    'train_split': 0.7,
    'val_split': 0.15,
    
    # Training settings
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'early_stopping_patience': 20,
    'scheduler_patience': 10,
    'scheduler_factor': 0.5,
    
    # Model settings
    'hidden_dims': [512, 256, 128],
    'dropout_rate': 0.3,
    'use_batch_norm': True,
    
    # Output settings
    'output_dir': './outputs/experiment1_unified',
    'seed': 42,
}


# ============================================================
# Logger Setup
# ============================================================
def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# Trainer Class
# ============================================================
class UnifiedTrainer:
    """Unified trainer for all model types."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        model_name: str,
        output_dir: Path,
        device: str,
        logger: logging.Logger
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.logger = logger
        
        # Create output directories
        self.checkpoint_dir = output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config['scheduler_patience'],
            factor=config['scheduler_factor'],
            min_lr=1e-6
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch in pbar:
            # Unpack batch
            if len(batch) == 3:
                audio, labels, _ = batch
            else:
                audio, labels = batch[0], batch[1]
            
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(audio)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * audio.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, loader: DataLoader, desc: str = 'Validation') -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                if len(batch) == 3:
                    audio, labels, _ = batch
                else:
                    audio, labels = batch[0], batch[1]
                
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(audio)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item() * audio.size(0)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint in unified format."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'config': self.config,
            'model_name': self.model_name,
            'history': self.history
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"  ★ New best model saved! Val Acc: {self.best_val_acc*100:.2f}%")
    
    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        num_epochs = self.config['num_epochs']
        patience = self.config['early_stopping_patience']
        
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        self.logger.info(f"Early stopping patience: {patience}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Log progress
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% | "
                f"LR: {lr:.2e}"
            )
            
            # Early stopping
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        
        # Final test evaluation
        self.logger.info("\nRunning final test evaluation...")
        
        # Load best model for testing
        best_ckpt = torch.load(self.checkpoint_dir / 'best_model.pth', weights_only=False)
        self.model.load_state_dict(best_ckpt['model_state_dict'])
        
        test_loss, test_acc = self.validate(self.test_loader, desc='Testing')
        
        self.logger.info(f"Test Results: Loss={test_loss:.4f}, Accuracy={test_acc*100:.2f}%")
        
        # Compile results
        results = {
            'model_name': self.model_name,
            'status': 'completed',
            'best_epoch': self.best_epoch + 1,
            'best_val_acc': self.best_val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'total_epochs': epoch + 1,
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'history': self.history
        }
        
        # Save results
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


# ============================================================
# Model Factory
# ============================================================
def create_model(
    model_type: str,
    num_speakers: int,
    config: Dict[str, Any],
    device: str
) -> nn.Module:
    """Create model based on type."""
    
    if model_type == 'mel_mlp':
        model = TraditionalMLPBaseline(
            feature_type='mel',
            n_mels=80,
            n_mfcc=40,
            sample_rate=config['sample_rate'],
            hidden_dims=config['hidden_dims'],
            dropout_rate=config['dropout_rate'],
            use_batch_norm=config['use_batch_norm'],
            num_speakers=num_speakers,
            device=device
        )
    elif model_type == 'mfcc_mlp':
        model = TraditionalMLPBaseline(
            feature_type='mfcc',
            n_mels=80,
            n_mfcc=40,
            sample_rate=config['sample_rate'],
            hidden_dims=config['hidden_dims'],
            dropout_rate=config['dropout_rate'],
            use_batch_norm=config['use_batch_norm'],
            num_speakers=num_speakers,
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


# ============================================================
# Main Experiment Runner
# ============================================================
class Experiment1Runner:
    """Main runner for Experiment 1."""
    
    AVAILABLE_MODELS = ['mel_mlp', 'mfcc_mlp']
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger(
            'experiment1',
            str(self.output_dir / 'experiment1.log')
        )
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set seed
        set_seed(config['seed'])
        
        # Results storage
        self.results = {}
        
        # Data loaders (created once, shared by all models)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.num_speakers = None
    
    def setup_data(self):
        """Setup data loaders (once for all models)."""
        self.logger.info("=" * 70)
        self.logger.info("Loading dataset...")
        self.logger.info(f"Data dir: {self.config['data_dir']}")
        self.logger.info("=" * 70)
        
        self.train_loader, self.val_loader, self.test_loader = create_speaker_dataloaders(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            sample_rate=self.config['sample_rate'],
            max_length=self.config['max_length'],
            train_split=self.config['train_split'],
            val_split=self.config['val_split'],
            seed=self.config['seed']
        )
        
        # Get number of speakers
        self.num_speakers = 26  # Default
        try:
            if hasattr(self.train_loader.dataset, 'num_classes'):
                self.num_speakers = self.train_loader.dataset.num_classes
        except:
            pass
        
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
        self.logger.info(f"Number of speakers: {self.num_speakers}")
    
    def train_model(self, model_type: str) -> Dict[str, Any]:
        """Train a single model."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"Training model: {model_type.upper()}")
        self.logger.info("=" * 70)
        
        # Create model output directory
        model_output_dir = self.output_dir / model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        model = create_model(
            model_type=model_type,
            num_speakers=self.num_speakers,
            config=self.config,
            device=self.device
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {num_params:,} (trainable: {trainable_params:,})")
        
        # Create trainer
        trainer = UnifiedTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            config=self.config,
            model_name=model_type,
            output_dir=model_output_dir,
            device=self.device,
            logger=self.logger
        )
        
        # Train
        results = trainer.train()
        results['num_parameters'] = num_params
        
        return results
    
    def run_single(self, model_type: str):
        """Run training for a single model."""
        if self.train_loader is None:
            self.setup_data()
        
        results = self.train_model(model_type)
        self.results[model_type] = results
        
        return results
    
    def run_all(self):
        """Run training for all models."""
        self.logger.info("=" * 70)
        self.logger.info("EXPERIMENT 1: UNIFIED BASELINE COMPARISON")
        self.logger.info("=" * 70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Seed: {self.config['seed']}")
        self.logger.info(f"Models: {self.AVAILABLE_MODELS}")
        
        # Setup data once
        self.setup_data()
        
        # Train each model
        for model_type in self.AVAILABLE_MODELS:
            try:
                results = self.train_model(model_type)
                self.results[model_type] = results
                self.logger.info(f"✓ {model_type} completed: Test Acc = {results['test_acc']*100:.2f}%")
            except Exception as e:
                self.logger.error(f"✗ {model_type} failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.results[model_type] = {'status': 'failed', 'error': str(e)}
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate experiment summary."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("EXPERIMENT 1 SUMMARY")
        self.logger.info("=" * 70)
        
        summary_lines = []
        summary_lines.append(f"{'Model':<15} {'Val Acc':>10} {'Test Acc':>10} {'Time (min)':>12} {'Params':>12}")
        summary_lines.append("-" * 65)
        
        for model_name, results in self.results.items():
            if results.get('status') == 'failed':
                summary_lines.append(f"{model_name:<15} {'ERROR':>10}")
            else:
                val_acc = results.get('best_val_acc', 0) * 100
                test_acc = results.get('test_acc', 0) * 100
                time_min = results.get('training_time_minutes', 0)
                params = results.get('num_parameters', 0)
                summary_lines.append(
                    f"{model_name:<15} {val_acc:>9.2f}% {test_acc:>9.2f}% {time_min:>11.1f} {params:>12,}"
                )
        
        summary_lines.append("=" * 65)
        
        for line in summary_lines:
            self.logger.info(line)
        
        # Save summary
        summary = {
            'experiment': 'Experiment 1: Unified Baseline Comparison',
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': self.results,
            'summary_table': summary_lines
        }
        
        summary_path = self.output_dir / 'experiment1_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"\nSummary saved to: {summary_path}")
        
        # Print checkpoint locations
        self.logger.info("\nCheckpoint locations:")
        for model_name in self.results.keys():
            if self.results[model_name].get('status') != 'failed':
                ckpt_path = self.output_dir / model_name / 'checkpoints' / 'best_model.pth'
                self.logger.info(f"  {model_name}: {ckpt_path}")


# ============================================================
# Main Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Experiment 1: Unified Baseline Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--all', action='store_true',
        help='Train all models'
    )
    parser.add_argument(
        '--model', type=str, choices=['mel_mlp', 'mfcc_mlp'],
        help='Train a specific model'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List available models'
    )
    parser.add_argument(
        '--data_dir', type=str,
        default=DEFAULT_CONFIG['data_dir'],
        help='Path to dataset'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=DEFAULT_CONFIG['output_dir'],
        help='Output directory'
    )
    parser.add_argument(
        '--epochs', type=int,
        default=DEFAULT_CONFIG['num_epochs'],
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int,
        default=DEFAULT_CONFIG['batch_size'],
        help='Batch size'
    )
    parser.add_argument(
        '--seed', type=int,
        default=DEFAULT_CONFIG['seed'],
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available models:")
        for model in Experiment1Runner.AVAILABLE_MODELS:
            print(f"  - {model}")
        return
    
    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['seed'] = args.seed
    
    # Create runner
    runner = Experiment1Runner(config)
    
    if args.all:
        runner.run_all()
    elif args.model:
        runner.run_single(args.model)
    else:
        print("Please specify --all to train all models or --model <name> to train a specific model")
        print("Use --list to see available models")
        parser.print_help()


if __name__ == "__main__":
    main()