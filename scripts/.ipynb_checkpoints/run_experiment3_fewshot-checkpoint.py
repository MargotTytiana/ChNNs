#!/usr/bin/env python3
"""
Experiment 3: Few-Shot Learning Evaluation

This script evaluates how well speaker recognition models perform
when trained with limited samples per speaker.

Hypothesis: Chaotic features may capture more essential speaker characteristics,
leading to better performance with fewer training samples.

Test conditions:
- Samples per speaker: 1, 2, 3, 5, 10, 15, all
- Same test set for all conditions
- Multiple runs for statistical significance

Usage:
    python scripts/run_experiment3_fewshot.py --all
    python scripts/run_experiment3_fewshot.py --samples 5,10
    python scripts/run_experiment3_fewshot.py --model mel_mlp --samples 5
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
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import logging
import copy

# ============================================================
# Setup Imports
# ============================================================
def setup_imports():
    """Setup Python path for project imports."""
    current_file = Path(__file__).resolve()
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

# Import project modules
from dataset_loader import create_speaker_dataloaders
from hybrid_models import TraditionalMLPBaseline
from reproducibility import set_seed


# ============================================================
# Configuration
# ============================================================
DEFAULT_CONFIG = {
    # Data settings
    'data_dir': '/scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2',
    'sample_rate': 16000,
    'max_length': 3.0,
    
    # Few-shot settings
    'samples_per_speaker': [1, 2, 3, 5, 10, 15, -1],  # -1 means all samples
    'num_runs': 3,  # Number of runs per condition for statistical significance
    
    # Training settings
    'batch_size': 16,  # Smaller batch for few-shot
    'num_epochs': 50,  # Fewer epochs for few-shot (less data)
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'early_stopping_patience': 15,
    
    # Model settings
    'hidden_dims': [512, 256, 128],
    'dropout_rate': 0.3,
    'use_batch_norm': True,
    
    # Output settings
    'output_dir': './outputs/experiment3_fewshot',
    'seed': 42,
}


# ============================================================
# Logger Setup
# ============================================================
def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# Few-Shot Dataset Sampler
# ============================================================
class FewShotSampler:
    """
    Samples a limited number of examples per class/speaker for few-shot learning.
    """
    
    def __init__(self, dataset, num_samples_per_speaker: int, seed: int = 42):
        """
        Args:
            dataset: Original dataset
            num_samples_per_speaker: Number of samples to keep per speaker (-1 for all)
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_samples = num_samples_per_speaker
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Build index mapping: speaker -> sample indices
        self.speaker_indices = self._build_speaker_indices()
    
    def _build_speaker_indices(self) -> Dict[int, List[int]]:
        """Build mapping from speaker ID to sample indices."""
        speaker_indices = {}
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            
            # Get label (speaker ID)
            if len(sample) == 3:
                _, label, _ = sample
            elif len(sample) == 2:
                _, label = sample
            else:
                continue
            
            # Convert to int if tensor
            if hasattr(label, 'item'):
                label = label.item()
            
            if label not in speaker_indices:
                speaker_indices[label] = []
            speaker_indices[label].append(idx)
        
        return speaker_indices
    
    def get_few_shot_indices(self) -> List[int]:
        """Get indices for few-shot subset."""
        selected_indices = []
        
        for speaker_id, indices in self.speaker_indices.items():
            if self.num_samples == -1 or self.num_samples >= len(indices):
                # Use all samples
                selected_indices.extend(indices)
            else:
                # Randomly sample
                sampled = self.rng.choice(indices, size=self.num_samples, replace=False)
                selected_indices.extend(sampled.tolist())
        
        return selected_indices
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the sampling."""
        total_samples = sum(len(indices) for indices in self.speaker_indices.values())
        num_speakers = len(self.speaker_indices)
        
        if self.num_samples == -1:
            selected_per_speaker = [len(indices) for indices in self.speaker_indices.values()]
        else:
            selected_per_speaker = [min(self.num_samples, len(indices)) 
                                   for indices in self.speaker_indices.values()]
        
        return {
            'total_samples': total_samples,
            'num_speakers': num_speakers,
            'samples_per_speaker_setting': self.num_samples,
            'actual_selected': sum(selected_per_speaker),
            'min_per_speaker': min(selected_per_speaker),
            'max_per_speaker': max(selected_per_speaker),
            'avg_per_speaker': np.mean(selected_per_speaker)
        }


class FewShotDataset(Dataset):
    """Wrapper dataset for few-shot learning."""
    
    def __init__(self, original_dataset, indices: List[int]):
        self.original_dataset = original_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]


# ============================================================
# Few-Shot Trainer
# ============================================================
class FewShotTrainer:
    """Trainer for few-shot learning experiments."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: str,
        logger: logging.Logger
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            if len(batch) == 3:
                audio, labels, _ = batch
            else:
                audio, labels = batch[0], batch[1]
            
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(audio)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * audio.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / total, correct / total
    
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
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
        
        return total_loss / total, correct / total
    
    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        num_epochs = self.config['num_epochs']
        patience = self.config['early_stopping_patience']
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            self.scheduler.step(val_acc)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Load best model and test
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        test_loss, test_acc = self.evaluate(self.test_loader)
        
        return {
            'best_val_acc': self.best_val_acc,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'epochs_trained': epoch + 1
        }


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
class Experiment3Runner:
    """Runner for Experiment 3: Few-Shot Learning."""
    
    AVAILABLE_MODELS = ['mel_mlp', 'mfcc_mlp']
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            'experiment3',
            str(self.output_dir / 'experiment3.log')
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_seed = config['seed']
        
        self.results = {}
        
        # Store dataloaders
        self.full_train_dataset = None
        self.val_loader = None
        self.test_loader = None
        self.num_speakers = None
        self.collate_fn = None
    
    def setup_data(self):
        """Load full dataset."""
        self.logger.info("=" * 70)
        self.logger.info("Loading dataset...")
        self.logger.info(f"Data dir: {self.config['data_dir']}")
        self.logger.info("=" * 70)
        
        train_loader, val_loader, test_loader = create_speaker_dataloaders(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            sample_rate=self.config['sample_rate'],
            max_length=self.config['max_length'],
            train_split=0.7,
            val_split=0.15,
            seed=self.base_seed
        )
        
        self.full_train_dataset = train_loader.dataset
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.collate_fn = train_loader.collate_fn
        
        # Get number of speakers
        self.num_speakers = 26
        try:
            if hasattr(train_loader.dataset, 'num_classes'):
                self.num_speakers = train_loader.dataset.num_classes
        except:
            pass
        
        self.logger.info(f"Full train samples: {len(self.full_train_dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
        self.logger.info(f"Number of speakers: {self.num_speakers}")
    
    def create_fewshot_loader(
        self, 
        samples_per_speaker: int, 
        run_seed: int
    ) -> Tuple[DataLoader, Dict]:
        """Create a few-shot training dataloader."""
        sampler = FewShotSampler(
            self.full_train_dataset,
            num_samples_per_speaker=samples_per_speaker,
            seed=run_seed
        )
        
        indices = sampler.get_few_shot_indices()
        stats = sampler.get_statistics()
        
        fewshot_dataset = FewShotDataset(self.full_train_dataset, indices)
        
        fewshot_loader = DataLoader(
            fewshot_dataset,
            batch_size=min(self.config['batch_size'], len(fewshot_dataset)),
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn
        )
        
        return fewshot_loader, stats
    
    def run_single_condition(
        self,
        model_type: str,
        samples_per_speaker: int,
        run_id: int
    ) -> Dict[str, Any]:
        """Run a single few-shot condition."""
        run_seed = self.base_seed + run_id * 1000 + samples_per_speaker
        set_seed(run_seed)
        
        # Create few-shot loader
        fewshot_loader, stats = self.create_fewshot_loader(samples_per_speaker, run_seed)
        
        samples_str = "all" if samples_per_speaker == -1 else str(samples_per_speaker)
        self.logger.info(
            f"  Run {run_id+1}: {samples_str} samples/speaker, "
            f"total={stats['actual_selected']} samples"
        )
        
        # Create fresh model
        model = create_model(
            model_type=model_type,
            num_speakers=self.num_speakers,
            config=self.config,
            device=self.device
        )
        
        # Train
        trainer = FewShotTrainer(
            model=model,
            train_loader=fewshot_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            config=self.config,
            device=self.device,
            logger=self.logger
        )
        
        results = trainer.train()
        results['samples_per_speaker'] = samples_per_speaker
        results['actual_train_samples'] = stats['actual_selected']
        results['run_id'] = run_id
        results['seed'] = run_seed
        
        self.logger.info(
            f"    Val: {results['best_val_acc']*100:.2f}%, "
            f"Test: {results['test_acc']*100:.2f}%"
        )
        
        return results
    
    def run_model(self, model_type: str) -> Dict[str, Any]:
        """Run all few-shot conditions for a model."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Model: {model_type.upper()}")
        self.logger.info(f"{'='*70}")
        
        model_results = {}
        
        for samples_per_speaker in self.config['samples_per_speaker']:
            samples_str = "all" if samples_per_speaker == -1 else str(samples_per_speaker)
            self.logger.info(f"\nSamples per speaker: {samples_str}")
            
            run_results = []
            for run_id in range(self.config['num_runs']):
                try:
                    result = self.run_single_condition(
                        model_type, samples_per_speaker, run_id
                    )
                    run_results.append(result)
                except Exception as e:
                    self.logger.error(f"    Run {run_id+1} failed: {e}")
            
            if run_results:
                # Aggregate results
                val_accs = [r['best_val_acc'] for r in run_results]
                test_accs = [r['test_acc'] for r in run_results]
                
                model_results[samples_str] = {
                    'samples_per_speaker': samples_per_speaker,
                    'num_runs': len(run_results),
                    'val_acc_mean': np.mean(val_accs),
                    'val_acc_std': np.std(val_accs),
                    'test_acc_mean': np.mean(test_accs),
                    'test_acc_std': np.std(test_accs),
                    'runs': run_results
                }
                
                self.logger.info(
                    f"  → Mean Test Acc: {np.mean(test_accs)*100:.2f}% "
                    f"± {np.std(test_accs)*100:.2f}%"
                )
        
        return model_results
    
    def run_all(self) -> Dict[str, Any]:
        """Run all models and conditions."""
        self.logger.info("=" * 70)
        self.logger.info("EXPERIMENT 3: FEW-SHOT LEARNING EVALUATION")
        self.logger.info("=" * 70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Samples per speaker: {self.config['samples_per_speaker']}")
        self.logger.info(f"Runs per condition: {self.config['num_runs']}")
        
        # Setup data
        self.setup_data()
        
        # Run each model
        for model_type in self.AVAILABLE_MODELS:
            try:
                results = self.run_model(model_type)
                self.results[model_type] = results
            except Exception as e:
                self.logger.error(f"Model {model_type} failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # Generate summary
        self._generate_summary()
        self._save_results()
        
        return self.results
    
    def run_single_model(self, model_type: str) -> Dict[str, Any]:
        """Run few-shot evaluation for a single model."""
        if self.full_train_dataset is None:
            self.setup_data()
        
        results = self.run_model(model_type)
        self.results[model_type] = results
        
        self._generate_summary()
        self._save_results()
        
        return results
    
    def _generate_summary(self):
        """Generate summary table."""
        self.logger.info("\n" + "=" * 90)
        self.logger.info("FEW-SHOT LEARNING SUMMARY")
        self.logger.info("=" * 90)
        
        # Header
        samples_list = self.config['samples_per_speaker']
        header = f"{'Model':<12}"
        for s in samples_list:
            label = "all" if s == -1 else str(s)
            header += f" {label:>10}"
        self.logger.info(header)
        self.logger.info("-" * 90)
        
        # Data rows
        for model_name, model_results in self.results.items():
            row = f"{model_name:<12}"
            for s in samples_list:
                label = "all" if s == -1 else str(s)
                if label in model_results:
                    acc = model_results[label]['test_acc_mean'] * 100
                    std = model_results[label]['test_acc_std'] * 100
                    row += f" {acc:>5.1f}±{std:<4.1f}"
                else:
                    row += f" {'N/A':>10}"
            self.logger.info(row)
        
        self.logger.info("=" * 90)
        self.logger.info("(Values show Test Accuracy % ± std)")
    
    def _save_results(self):
        """Save results to JSON."""
        output = {
            'experiment': 'Experiment 3: Few-Shot Learning',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'samples_per_speaker': self.config['samples_per_speaker'],
                'num_runs': self.config['num_runs'],
                'num_epochs': self.config['num_epochs']
            },
            'results': self.results
        }
        
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        output = convert(output)
        
        results_path = self.output_dir / 'fewshot_results.json'
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"\nResults saved to: {results_path}")


# ============================================================
# Main Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Experiment 3: Few-Shot Learning Evaluation'
    )
    
    parser.add_argument(
        '--all', action='store_true',
        help='Run all models'
    )
    parser.add_argument(
        '--model', type=str, choices=['mel_mlp', 'mfcc_mlp'],
        help='Run a specific model'
    )
    parser.add_argument(
        '--samples', type=str, default=None,
        help='Comma-separated samples per speaker (e.g., "1,2,5,10,-1")'
    )
    parser.add_argument(
        '--runs', type=int, default=DEFAULT_CONFIG['num_runs'],
        help='Number of runs per condition'
    )
    parser.add_argument(
        '--epochs', type=int, default=DEFAULT_CONFIG['num_epochs'],
        help='Number of training epochs'
    )
    parser.add_argument(
        '--data_dir', type=str, default=DEFAULT_CONFIG['data_dir'],
        help='Path to dataset'
    )
    parser.add_argument(
        '--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'],
        help='Output directory'
    )
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_CONFIG['seed'],
        help='Base random seed'
    )
    
    args = parser.parse_args()
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    config['num_runs'] = args.runs
    config['num_epochs'] = args.epochs
    config['seed'] = args.seed
    
    if args.samples:
        config['samples_per_speaker'] = [int(x) for x in args.samples.split(',')]
    
    # Create runner
    runner = Experiment3Runner(config)
    
    if args.all:
        runner.run_all()
    elif args.model:
        runner.run_single_model(args.model)
    else:
        print("Please specify --all or --model <name>")
        print("Example: python run_experiment3_fewshot.py --all")
        print("Example: python run_experiment3_fewshot.py --model mel_mlp --samples 1,5,10")
        parser.print_help()


if __name__ == "__main__":
    main()