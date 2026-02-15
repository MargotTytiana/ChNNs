#!/usr/bin/env python3
"""
Short Utterance Robustness Experiment

This script evaluates speaker recognition models under short utterance conditions
to validate the robustness advantages of chaotic features (C-HiLAP) over traditional
approaches (MFCC, Mel-spectrogram).

Key experiments:
1. Performance degradation curve across durations (0.5s - 3.0s)
2. Comparison of C-HiLAP vs baselines
3. Statistical significance testing
4. Visualization of results
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
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

        project_root = current_dir.parent.parent.parent  # experiments -> Model

        

        paths_to_add = [

            str(project_root),

            str(project_root / 'core'),

            str(project_root / 'models'), 

            str(project_root / 'features'),

            str(project_root / 'data'),

            str(project_root / 'utils'),

            str(project_root / 'evaluation'),

            str(project_root / 'experiments'),

            

        ]

        

        for path in paths_to_add:

            if Path(path).exists() and path not in sys.path:

                sys.path.insert(0, path)

        

        return project_root, False

# Setup imports

PROJECT_ROOT, USING_IMPORT_MANAGER = setup_module_imports()

from short_utterance_transform import (
    ShortUtteranceTransform,
    create_short_utterance_dataset_wrapper,
    compute_duration_performance_curve
)
from dataset_loader import create_chaotic_speaker_dataset
from baseline_experiment import BaselineExperiment
from chaotic_experiment import ChaoticExperiment
from logger import setup_logger
from reproducibility import set_seed


class ShortUtteranceExperiment:
    """
    Experiment manager for short utterance robustness testing.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = './outputs/short_utterance',
        verbose: bool = True
    ):
        """
        Initialize short utterance experiment.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory
            verbose: Verbose logging
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger(
            name='short_utterance',
            log_file=str(self.output_dir / 'experiment.log'),
            level='INFO' if verbose else 'WARNING'
        )
        
        # Set seed
        if 'seed' in config:
            set_seed(config['seed'])
        
        self.logger.info("="*60)
        self.logger.info("SHORT UTTERANCE ROBUSTNESS EXPERIMENT")
        self.logger.info("="*60)
        self.logger.info(f"Output: {self.output_dir}")
        
        # Storage for results
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': config
            },
            'models': {},
            'duration_curves': {},
            'comparisons': {}
        }
    
    def prepare_datasets(self):
        """Prepare datasets for testing."""
        self.logger.info("Preparing datasets...")
        
        data_config = self.config.get('dataset', {})
        data_dir = data_config.get('path', './data/LibriSpeech')
        
        # Create base dataset
        self.logger.info(f"Loading dataset from: {data_dir}")
        
        base_dataset = create_chaotic_speaker_dataset(
            dataset_path=data_dir,
            min_samples_per_speaker=data_config.get('min_samples_per_speaker', 2),
            max_samples_per_speaker=data_config.get('max_samples_per_speaker', 30),
            target_num_speakers=data_config.get('num_speakers', 50)
        )
        
        # Create splits
        splits = base_dataset.create_data_splits(
            train_ratio=data_config.get('train_split', 0.7),
            val_ratio=data_config.get('val_split', 0.15),
            test_ratio=data_config.get('test_split', 0.15)
        )
        
        # Create PyTorch datasets
        self.train_dataset = splits['train'].create_pytorch_dataset()
        self.val_dataset = splits['val'].create_pytorch_dataset()
        self.test_dataset = splits['test'].create_pytorch_dataset()
        
        self.logger.info(f"Dataset prepared:")
        self.logger.info(f"  Train: {len(self.train_dataset)} samples")
        self.logger.info(f"  Val: {len(self.val_dataset)} samples")
        self.logger.info(f"  Test: {len(self.test_dataset)} samples")
        self.logger.info(f"  Speakers: {base_dataset.num_speakers}")
        
        return base_dataset
    
    def train_or_load_models(self, force_retrain: bool = False):
        """
        Train or load models for comparison.
        
        Args:
            force_retrain: Whether to force retraining even if checkpoints exist
        """
        self.logger.info("Setting up models...")
        
        models_config = self.config.get('models', {})
        self.models = {}
        
        for model_name, model_config in models_config.items():
            self.logger.info(f"\nProcessing model: {model_name}")
            
            # Check if checkpoint path is provided in config
            checkpoint_from_config = model_config.get('checkpoint')
            
            # Default checkpoint path
            default_checkpoint_path = self.output_dir / 'models' / f'{model_name}_best.pth'
            
            # Determine checkpoint to use
            if checkpoint_from_config and Path(checkpoint_from_config).exists():
                # Use checkpoint from config
                checkpoint_path = Path(checkpoint_from_config)
                self.logger.info(f"  Using checkpoint from config: {checkpoint_path}")
                self.models[model_name] = self._load_model(
                    model_name, model_config, checkpoint_path
                )
            elif default_checkpoint_path.exists() and not force_retrain:
                # Use default checkpoint
                self.logger.info(f"  Loading from default checkpoint: {default_checkpoint_path}")
                self.models[model_name] = self._load_model(
                    model_name, model_config, default_checkpoint_path
                )
            else:
                # Train new model
                if checkpoint_from_config and not Path(checkpoint_from_config).exists():
                    self.logger.warning(f"  Checkpoint not found: {checkpoint_from_config}")
                self.logger.info(f"  Training new model...")
                self.models[model_name] = self._train_model(
                    model_name, model_config
                )
        
        self.logger.info(f"\nTotal models prepared: {len(self.models)}")
    
    def _train_model(self, model_name: str, model_config: Dict):
        """Train a single model."""
        model_type = model_config.get('type', 'baseline')
        
        # Prepare config for experiment
        if model_type == 'chaotic':
            # ChaoticExperiment needs complete config structure
            exp_config = {
                **self.config.get('training', {}),
                'num_speakers': self.test_dataset.num_classes,
                'output_dir': str(self.output_dir / 'models' / model_name),
                
                # Audio config
                'sample_rate': self.config.get('audio', {}).get('sample_rate', 16000),
                'frame_length': self.config.get('audio', {}).get('frame_length', 400),
                'hop_length': self.config.get('audio', {}).get('hop_length', 160),
                'max_audio_length': self.config.get('audio', {}).get('max_audio_length', 3.0),
                
                # Phase space config
                'embedding_dim': model_config.get('params', {}).get('input_dim', 10),
                'delay_method': 'autocorr',
                
                # Chaotic features
                'mlsa_scales': model_config.get('params', {}).get('mlsa_scales', 5),
                'rqa_radius_ratio': model_config.get('params', {}).get('rqa_radius_ratio', 0.1),
                
                # Chaotic embedding
                'chaotic_system': 'lorenz',
                'evolution_time': 0.5,
                'time_step': 0.01,
                
                # Attractor pooling
                'pooling_type': 'comprehensive',
                
                # Speaker embedding
                'speaker_embedding_dim': model_config.get('params', {}).get('embedding_dim', 256),
                
                # Classifier
                'classifier_type': model_config.get('params', {}).get('classifier_type', 'cosine'),
                'temperature': model_config.get('params', {}).get('temperature', 30.0),
                
                # Model type
                'model_type': 'full_chaotic',
                
                # Data config
                'data_dir': self.config.get('dataset', {}).get('path', './data'),
                'train_split': self.config.get('dataset', {}).get('train_split', 0.7),
                'val_split': self.config.get('dataset', {}).get('val_split', 0.15),
                'test_split': self.config.get('dataset', {}).get('test_split', 0.15),
            }
            experiment = ChaoticExperiment(config=exp_config)
            experiment.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            experiment.model = experiment.model.to(experiment.device)

        else:
            # Baseline experiment - needs baseline_type
            feature_type = model_config.get('feature_type', 'mel')
            baseline_type = f"{feature_type}_mlp"  # mel_mlp or mfcc_mlp
            
            exp_config = {
                **self.config.get('training', {}),
                'num_speakers': self.test_dataset.num_classes,
                'baseline_type': baseline_type,  # Required by BaselineExperiment
                'output_dir': str(self.output_dir / 'models' / model_name),
                'data_dir': self.config.get('dataset', {}).get('path', './data/LibriSpeech'),
                # 【关键修复点】：将 optimizer 包装成字典
                'optimizer': {
                    'type': self.config.get('training', {}).get('optimizer', 'adam'),
                    'lr': self.config.get('training', {}).get('learning_rate', 0.001),
                    'weight_decay': 1e-4
                },
                
                # 【额外修复点】：将 scheduler 也包装成字典，防止下一个报错
                'scheduler': {
                    'type': 'plateau',
                    'patience': 5,
                    'factor': 0.5
                },
                # Audio config
                'sample_rate': self.config.get('audio', {}).get('sample_rate', 16000),
                'max_audio_length': self.config.get('audio', {}).get('max_audio_length', 3.0),
                
                # Feature config
                'n_mels': model_config.get('params', {}).get('n_mels', 80) if feature_type == 'mel' else None,
                'n_mfcc': model_config.get('params', {}).get('n_mfcc', 40) if feature_type == 'mfcc' else None,
                'hidden_dims': model_config.get('params', {}).get('hidden_dims', [512, 256]),
                'dropout_rate': model_config.get('params', {}).get('dropout', 0.3),
                'use_batch_norm': True,
            }
            experiment = BaselineExperiment(config=exp_config)
            
            if hasattr(experiment, 'setup'):
                experiment.setup()
            
            # 检查模型是否存在
            if experiment.model is None:
                raise ValueError(f"Failed to initialize model for {model_name}. Check BaselineExperiment config.")
            experiment.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            experiment.model = experiment.model.to(experiment.device)
        
        # Setup data loaders (use normal length for training)
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=exp_config.get('batch_size', 32),
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=exp_config.get('batch_size', 32),
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=exp_config.get('batch_size', 32),
            shuffle=False,
            num_workers=0
        )
        
        num_epochs = self.config.get('training', {}).get('num_epochs', 100)
        
        print(f"  Training {model_name} for {num_epochs} epochs...")
        
        results = experiment.train(num_epochs=num_epochs)
        
        self.results['models'][model_name] = {
            'config': model_config,
            'training_results': results
        }
        
        return experiment.model
    
    def _load_model(self, model_name: str, model_config: Dict, checkpoint_path: Path):
        """Load a trained model from checkpoint."""
        model_type = model_config.get('type', 'baseline')
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if model_type == 'chaotic':
            from chaotic_network import ChaoticSpeakerRecognitionNetwork
            
            if 'config' in checkpoint:
                config = checkpoint['config']
                self.logger.info(f"Using config from checkpoint for {model_name}")
            else:
                config = {
                    'phase_space': {'embedding_dim': 10},
                    'speaker_embedding': {'dim': 256, 'hidden_dims': [512, 256, 128]},
                    'classifier': {'type': 'cosine', 'temperature': 30.0},
                }
                self.logger.info(f"Using default config for {model_name}")
            
            # Create model on CPU first
            model = ChaoticSpeakerRecognitionNetwork(
                num_speakers=self.test_dataset.num_classes,
                embedding_dim=config.get('phase_space', {}).get('embedding_dim', 10),
                speaker_embedding_dim=config.get('speaker_embedding', {}).get('dim', 256),
                embedding_hidden_dims=config.get('speaker_embedding', {}).get('hidden_dims', [512, 256, 128]),
                classifier_type=config.get('classifier', {}).get('type', 'cosine'),
                device='cpu'  # Force CPU initially
            )
        else:
            from hybrid_models import MelMLP, MFCCMLP
            
            feature_type = model_config.get('feature_type', 'mel')
            if feature_type == 'mel':
                model = MelMLP(num_speakers=self.test_dataset.num_classes)
            else:
                model = MFCCMLP(num_speakers=self.test_dataset.num_classes)
        
        # Load weights on CPU
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        # Keep on CPU, will move to GPU later
        model = model.cpu()
        model.eval()
        
        return model
    
    def evaluate_short_utterances(self):
        """Evaluate all models on short utterances."""
        self.logger.info("\n" + "="*60)
        self.logger.info("EVALUATING SHORT UTTERANCE PERFORMANCE")
        self.logger.info("="*60)
        
        test_config = self.config.get('short_utterance_test', {})
        durations = test_config.get('durations', [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        sample_rate = test_config.get('sample_rate', 16000)
        batch_size = test_config.get('batch_size', 32)
        device = test_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        for model_name, model in self.models.items():
            self.logger.info(f"\nEvaluating: {model_name}")
            
            # Move entire model to device and force all submodules
            model = model.to(device)
            for module in model.modules():
                module.to(device)
            model.eval()
            
            results = compute_duration_performance_curve(
                model=model,
                test_dataset=self.test_dataset,
                durations=durations,
                sample_rate=sample_rate,
                batch_size=batch_size,
                device=device
            )
            
            self.results['duration_curves'][model_name] = results
            
            # Log results
            self.logger.info(f"  Results for {model_name}:")
            for dur, acc in sorted(results.items()):
                self.logger.info(f"    {dur:4.1f}s: {acc:6.2f}%")
                
    def analyze_results(self):
        """Analyze and compare results."""
        self.logger.info("\n" + "="*60)
        self.logger.info("ANALYZING RESULTS")
        self.logger.info("="*60)
        
        curves = self.results['duration_curves']
        
        if not curves:
            self.logger.warning("No results to analyze")
            return
        
        # Compute relative performance
        self.logger.info("\nRelative Performance (vs longest duration):")
        for model_name, results in curves.items():
            durations = sorted(results.keys())
            longest_acc = results[max(durations)]
            
            self.logger.info(f"\n{model_name}:")
            relative_perf = {}
            for dur in durations:
                rel = (results[dur] / longest_acc) * 100 if longest_acc > 0 else 0
                relative_perf[dur] = rel
                self.logger.info(f"  {dur:4.1f}s: {rel:6.2f}% of max")
            
            self.results['comparisons'][model_name] = {
                'absolute': results,
                'relative': relative_perf
            }
        
        # Compute degradation slope
        self.logger.info("\nPerformance Degradation (1.0s -> 0.5s):")
        for model_name, results in curves.items():
            if 1.0 in results and 0.5 in results:
                degradation = results[1.0] - results[0.5]
                self.logger.info(f"  {model_name}: {degradation:+6.2f}%")
    
    def visualize_results(self):
        """Create visualization plots."""
        self.logger.info("\n" + "="*60)
        self.logger.info("CREATING VISUALIZATIONS")
        self.logger.info("="*60)
        
        curves = self.results['duration_curves']
        
        if not curves:
            self.logger.warning("No results to visualize")
            return
        
        # Plot 1: Performance vs Duration
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, results in curves.items():
            durations = sorted(results.keys())
            accuracies = [results[d] for d in durations]
            
            ax.plot(durations, accuracies, marker='o', linewidth=2, label=model_name)
        
        ax.set_xlabel('Utterance Duration (seconds)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Speaker Recognition Performance vs Utterance Duration', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / 'figures' / 'duration_performance.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved: {plot_path}")
        
        # Plot 2: Relative Performance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name in curves.keys():
            if model_name in self.results['comparisons']:
                relative = self.results['comparisons'][model_name]['relative']
                durations = sorted(relative.keys())
                rel_perf = [relative[d] for d in durations]
                
                ax.plot(durations, rel_perf, marker='s', linewidth=2, label=model_name)
        
        ax.set_xlabel('Utterance Duration (seconds)', fontsize=12)
        ax.set_ylabel('Relative Performance (% of max)', fontsize=12)
        ax.set_title('Relative Performance Degradation', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='k', linestyle='--', alpha=0.3)
        
        plot_path = self.output_dir / 'figures' / 'relative_performance.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved: {plot_path}")
    
    def save_results(self):
        """Save all results to JSON."""
        self.logger.info("\nSaving results...")
        
        results_path = self.output_dir / 'results' / 'short_utterance_results.json'
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"  Results saved to: {results_path}")
        
        # Also save summary
        summary_path = self.output_dir / 'results' / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write("SHORT UTTERANCE EXPERIMENT SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            for model_name, results in self.results['duration_curves'].items():
                f.write(f"{model_name}:\n")
                for dur, acc in sorted(results.items()):
                    f.write(f"  {dur:4.1f}s: {acc:6.2f}%\n")
                f.write("\n")
        
        self.logger.info(f"  Summary saved to: {summary_path}")
    
    def run(self, force_retrain: bool = False):
        """
        Run complete short utterance experiment.
        
        Args:
            force_retrain: Whether to force model retraining
        """
        try:
            # Step 1: Prepare data
            self.prepare_datasets()
            
            # Step 2: Train/load models
            self.train_or_load_models(force_retrain=force_retrain)
            
            # Step 3: Evaluate on short utterances
            self.evaluate_short_utterances()
            
            # Step 4: Analyze results
            self.analyze_results()
            
            # Step 5: Create visualizations
            self.visualize_results()
            
            # Step 6: Save results
            self.save_results()
            
            self.logger.info("\n" + "="*60)
            self.logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
            self.logger.info("="*60)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Short Utterance Robustness Experiment for Speaker Recognition'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='short_utterance_config.yaml',
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/short_utterance',
        help='Output directory'
    )
    parser.add_argument(
        '--force_retrain',
        action='store_true',
        help='Force model retraining even if checkpoints exist'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration")
        config = get_default_config()
    
    # Override seed if specified
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Create and run experiment
    experiment = ShortUtteranceExperiment(
        config=config,
        output_dir=args.output_dir,
        verbose=True
    )
    
    results = experiment.run(force_retrain=args.force_retrain)
    
    print("\nExperiment completed!")
    print(f"Results saved to: {args.output_dir}")


def get_default_config():
    """Get default experiment configuration."""
    return {
        'seed': 42,
        'dataset': {
            'path': './data/LibriSpeech',
            'num_speakers': 50,
            'min_samples_per_speaker': 2,
            'max_samples_per_speaker': 30,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15
        },
        'training': {
            'num_epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'early_stopping_patience': 20
        },
        'models': {
            'C-HiLAP': {
                'type': 'chaotic',
                'params': {
                    'chaotic_system': 'lorenz',
                    'embedding_dim': 256
                }
            },
            'MFCC-MLP': {
                'type': 'baseline',
                'feature_type': 'mfcc'
            },
            'Mel-MLP': {
                'type': 'baseline',
                'feature_type': 'mel'
            }
        },
        'short_utterance_test': {
            'durations': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            'sample_rate': 16000,
            'batch_size': 32,
            'device': 'cuda'
        }
    }


if __name__ == '__main__':
    main()