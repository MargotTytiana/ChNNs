#!/usr/bin/env python3
"""
Experiment 1: Standard Performance Comparison Runner

This script runs a fair comparison between baseline and chaotic models
under identical training conditions.

Usage:
    python scripts/run_experiment1.py --all
    python scripts/run_experiment1.py --model mel_mlp
    python scripts/run_experiment1.py --model chaotic_hybrid
    python scripts/run_experiment1.py --list  # Show available models
"""

import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import torch
import numpy as np

# Fix imports
def fix_imports():
    current_file = Path(__file__).resolve()
    model_dir = current_file.parent.parent
    paths = [
        str(model_dir),
        str(model_dir / 'experiments'),
        str(model_dir / 'models'),
        str(model_dir / 'features'),
        str(model_dir / 'data'),
        str(model_dir / 'utils'),
        str(model_dir / 'evaluation'),
        str(model_dir / 'core')
    ]
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    return model_dir

MODEL_DIR = fix_imports()

# Import project modules
try:
    from baseline_experiment import BaselineExperiment
    from chaotic_experiment import ChaoticExperiment
    from reproducibility import set_seed, get_system_info
    from logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are in the correct directories")
    sys.exit(1)


class Experiment1Runner:
    """
    Runner for Experiment 1: Standard Performance Comparison
    
    Manages the execution of multiple models under identical conditions
    and generates comparison reports.
    """
    
    def __init__(
        self,
        config_path: str,
        output_dir: Optional[str] = None,
        device: str = 'auto'
    ):
        """
        Initialize the experiment runner.
        
        Args:
            config_path: Path to experiment configuration YAML
            output_dir: Override output directory
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set output directory
        self.output_dir = Path(output_dir or self.config['output']['base_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(
            name='experiment1',
            log_file=str(self.output_dir / 'experiment1.log'),
            level='INFO'
        )
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize results storage
        self.results = {}
        self.seed = self.config['experiment'].get('seed', 42)
        
        # Log initialization
        self.logger.info("=" * 70)
        self.logger.info("EXPERIMENT 1: STANDARD PERFORMANCE COMPARISON")
        self.logger.info("=" * 70)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Seed: {self.seed}")
        
        # Log system info
        try:
            sys_info = get_system_info()
            self.logger.info(f"System: {sys_info}")
        except:
            pass
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled models from config."""
        enabled = []
        for name, cfg in self.config['models'].items():
            if cfg.get('enabled', True):
                enabled.append(name)
        return enabled
    
    def _build_experiment_config(
        self,
        model_name: str,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build complete experiment configuration for a model."""
        # Start with data config
        exp_config = dict(self.config['data'])
        
        # Add training config
        exp_config.update(self.config['training'])
        
        # Add model-specific config
        if model_config['type'] == 'baseline':
            exp_config['baseline_type'] = model_config['baseline_type']
            exp_config.update(model_config.get('feature_config', {}))
            exp_config.update(model_config.get('model_config', {}))
        else:
            # Chaotic model
            exp_config['chaotic_system'] = model_config.get('chaotic_system', 'lorenz')
            exp_config['model_type'] = model_config.get('model_type', 'full_chaotic')
            exp_config['use_gradient_fix'] = model_config.get('use_gradient_fix', True)
            exp_config['mix_alpha'] = model_config.get('mix_alpha', 0.3)
            
            # Phase space config
            if 'phase_space' in model_config:
                exp_config['embedding_dim'] = model_config['phase_space'].get('embedding_dim', 10)
            
            # Chaotic embedding config
            if 'chaotic_embedding' in model_config:
                exp_config['evolution_time'] = model_config['chaotic_embedding'].get('evolution_time', 0.5)
                exp_config['time_step'] = model_config['chaotic_embedding'].get('time_step', 0.01)
            
            # Pooling config
            if 'attractor_pooling' in model_config:
                exp_config['pooling_type'] = model_config['attractor_pooling'].get('pooling_type', 'comprehensive')
            
            # Classifier config
            if 'classifier' in model_config:
                exp_config['speaker_embedding_dim'] = model_config['classifier'].get('speaker_embedding_dim', 256)
                exp_config['hidden_dims'] = model_config['classifier'].get('hidden_dims', [512, 256, 128])
        
        # Placeholder for num_speakers (will be detected from dataset)
        exp_config['num_speakers'] = 26
        
        return exp_config
    
    def run_baseline_model(
        self,
        model_name: str,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a baseline model experiment."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running BASELINE model: {model_name}")
        self.logger.info(f"Type: {model_config.get('baseline_type')}")
        self.logger.info(f"{'='*60}")
        
        # Set seed for reproducibility
        set_seed(self.seed)
        
        # Build config
        exp_config = self._build_experiment_config(model_name, model_config)
        
        # Get num_epochs from config
        num_epochs = self.config['training'].get('num_epochs', 100)
        
        # Create experiment
        experiment = BaselineExperiment(
            config=exp_config,
            experiment_name=f"exp1_{model_name}",
            output_dir=str(self.output_dir / model_name),
            device=self.device,
            seed=self.seed
        )
        
        # Setup and train (not run!)
        experiment.setup()
        experiment.train(num_epochs=num_epochs)
        
        # Collect results
        results = self._collect_experiment_results(experiment)
        
        return results
    
    def run_chaotic_model(
        self,
        model_name: str,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a chaotic model experiment."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running CHAOTIC model: {model_name}")
        self.logger.info(f"System: {model_config.get('chaotic_system', 'lorenz')}")
        self.logger.info(f"Gradient Fix: {model_config.get('use_gradient_fix', True)}")
        self.logger.info(f"{'='*60}")
        
        # Set seed for reproducibility
        set_seed(self.seed)
        
        # Build config
        exp_config = self._build_experiment_config(model_name, model_config)
        
        # Get num_epochs from config
        num_epochs = self.config['training'].get('num_epochs', 100)
        
        # Create experiment
        experiment = ChaoticExperiment(
            config=exp_config,
            experiment_name=f"exp1_{model_name}",
            output_dir=str(self.output_dir / model_name),
            device=self.device,
            seed=self.seed
        )
        
        # Setup and train (not run!)
        experiment.setup()
        experiment.train(num_epochs=num_epochs)
        
        # Collect results
        results = self._collect_experiment_results(experiment)
        
        return results
    
    def _collect_experiment_results(self, experiment) -> Dict[str, Any]:
        """Collect results from a completed experiment."""
        results = {
            'status': experiment.state.status,
            'best_epoch': experiment.state.best_epoch,
            'best_metric': experiment.state.best_metric,
            'total_epochs': experiment.state.epoch + 1,
        }
        
        # Get validation metrics
        if experiment.state.val_metrics:
            last_val = experiment.state.val_metrics[-1]
            results['val_accuracy'] = last_val.get('accuracy', 0)
            results['val_loss'] = last_val.get('loss', 0)
            
            # Find best validation accuracy
            best_val_acc = max(
                [m.get('accuracy', 0) for m in experiment.state.val_metrics],
                default=0
            )
            results['best_val_accuracy'] = best_val_acc
        
        # Run final test if not already done
        try:
            test_metrics = experiment.test()
            results['test_accuracy'] = test_metrics.get('accuracy', 0)
            results['test_loss'] = test_metrics.get('loss', 0)
            results['eer'] = test_metrics.get('eer', 0)
        except Exception as e:
            self.logger.warning(f"Could not run test: {e}")
            results['test_accuracy'] = 0
            results['test_error'] = str(e)
        
        return results
    
    def run_single(self, model_name: str) -> Dict[str, Any]:
        """Run a single model by name."""
        if model_name not in self.config['models']:
            available = list(self.config['models'].keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
        
        model_config = self.config['models'][model_name]
        
        # Check if model is enabled
        if not model_config.get('enabled', True):
            self.logger.info(f"Skipping disabled model: {model_name}")
            return {'status': 'skipped', 'reason': 'disabled in config'}
        
        start_time = time.time()
        
        try:
            if model_config['type'] == 'baseline':
                results = self.run_baseline_model(model_name, model_config)
            else:
                results = self.run_chaotic_model(model_name, model_config)
            
            elapsed = time.time() - start_time
            results['training_time_seconds'] = elapsed
            results['training_time_minutes'] = elapsed / 60
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error running {model_name}: {e}")
            self.logger.error(traceback.format_exc())
            results = {
                'status': 'failed',
                'error': str(e),
                'training_time_seconds': time.time() - start_time
            }
        
        self.results[model_name] = results
        
        # Save individual results
        self._save_model_results(model_name, results)
        
        return results
    
    def run_all(self) -> Dict[str, Dict]:
        """Run all enabled models."""
        enabled_models = self.get_enabled_models()
        
        self.logger.info(f"\nRunning {len(enabled_models)} models: {enabled_models}")
        
        for i, model_name in enumerate(enabled_models):
            self.logger.info(f"\n{'#'*70}")
            self.logger.info(f"# [{i+1}/{len(enabled_models)}] Model: {model_name}")
            self.logger.info(f"{'#'*70}")
            
            results = self.run_single(model_name)
            
            if results.get('status') == 'failed':
                self.logger.error(f"✗ Failed {model_name}: {results.get('error', 'Unknown error')}")
            else:
                train_time = results.get('training_time_minutes', 0)
                self.logger.info(f"✓ Completed {model_name} in {train_time:.1f} minutes")
        
        # Generate comparison report
        self._save_comparison_results()
        self._print_comparison_table()
        self._generate_summary_report()
        
        return self.results
    
    def _save_model_results(self, model_name: str, results: Dict):
        """Save individual model results."""
        results_dir = self.output_dir / model_name / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = results_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _save_comparison_results(self):
        """Save comparison results to JSON."""
        results_path = self.output_dir / 'comparison_results.json'
        
        # Add metadata
        output = {
            'experiment': 'Experiment 1: Standard Performance Comparison',
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'device': self.device,
            'config': self.config['experiment'],
            'results': self.results
        }
        
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        self.logger.info(f"\nComparison results saved to: {results_path}")
    
    def _print_comparison_table(self):
        """Print comparison table to console and log."""
        self.logger.info("\n" + "=" * 90)
        self.logger.info("EXPERIMENT 1 RESULTS: STANDARD PERFORMANCE COMPARISON")
        self.logger.info("=" * 90)
        
        # Header
        header = f"{'Model':<20} {'Type':<10} {'Val Acc':>10} {'Test Acc':>10} {'EER':>8} {'Time(min)':>10}"
        self.logger.info(header)
        self.logger.info("-" * 90)
        
        # Results rows
        for model_name, results in self.results.items():
            model_type = self.config['models'][model_name]['type']
            
            if 'error' in results:
                self.logger.info(f"{model_name:<20} {model_type:<10} {'FAILED':>10}")
                continue
            
            # Extract metrics (handle different result formats)
            val_acc = results.get('best_val_accuracy', 
                     results.get('val_accuracy', 0))
            if isinstance(val_acc, dict):
                val_acc = val_acc.get('accuracy', 0)
            val_acc = float(val_acc) * 100
            
            test_acc = results.get('test_accuracy', 
                      results.get('final_test_accuracy', 0))
            if isinstance(test_acc, dict):
                test_acc = test_acc.get('accuracy', 0)
            test_acc = float(test_acc) * 100
            
            eer = results.get('eer', results.get('test_eer', 0))
            eer = float(eer) * 100
            
            train_time = results.get('training_time_minutes', 
                        results.get('training_time', 0) / 60)
            
            self.logger.info(
                f"{model_name:<20} {model_type:<10} "
                f"{val_acc:>9.2f}% {test_acc:>9.2f}% "
                f"{eer:>7.2f}% {train_time:>10.1f}"
            )
        
        self.logger.info("=" * 90)
    
    def _generate_summary_report(self):
        """Generate a text summary report."""
        report_path = self.output_dir / 'experiment1_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EXPERIMENT 1: STANDARD PERFORMANCE COMPARISON - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("RESULTS\n")
            f.write("-" * 70 + "\n\n")
            
            # Find best model
            best_model = None
            best_acc = 0
            
            for model_name, results in self.results.items():
                if 'error' in results:
                    continue
                
                val_acc = results.get('best_val_accuracy', 
                         results.get('val_accuracy', 0))
                if isinstance(val_acc, dict):
                    val_acc = val_acc.get('accuracy', 0)
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = model_name
                
                f.write(f"Model: {model_name}\n")
                f.write(f"  Validation Accuracy: {float(val_acc)*100:.2f}%\n")
                
                test_acc = results.get('test_accuracy', 0)
                if isinstance(test_acc, dict):
                    test_acc = test_acc.get('accuracy', 0)
                f.write(f"  Test Accuracy: {float(test_acc)*100:.2f}%\n")
                
                train_time = results.get('training_time_minutes', 0)
                f.write(f"  Training Time: {train_time:.1f} minutes\n")
                f.write("\n")
            
            f.write("-" * 70 + "\n")
            f.write("CONCLUSION\n")
            f.write("-" * 70 + "\n\n")
            
            if best_model:
                f.write(f"Best performing model: {best_model}\n")
                f.write(f"Best validation accuracy: {best_acc*100:.2f}%\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        self.logger.info(f"Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 1: Standard Performance Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_experiment1.py --all
    python run_experiment1.py --model mel_mlp
    python run_experiment1.py --model chaotic_hybrid
    python run_experiment1.py --list
        """
    )
    
    parser.add_argument(
        '--config', type=str,
        default='experiments/configs/experiment1_config.yaml',
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Run specific model (e.g., mel_mlp, mfcc_mlp, chaotic_hybrid)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all enabled models'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List available models'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    
    args = parser.parse_args()
    
    # Check config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print("Please create the config file or specify correct path with --config")
        sys.exit(1)
    
    # Initialize runner
    runner = Experiment1Runner(
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device
    )
    
    if args.list:
        print("\nAvailable models:")
        for name, cfg in runner.config['models'].items():
            status = "✓ enabled" if cfg.get('enabled', True) else "✗ disabled"
            print(f"  {name:<20} [{cfg['type']}] {status}")
        return
    
    if args.model:
        print(f"\nRunning single model: {args.model}")
        results = runner.run_single(args.model)
        print(f"\nResults for {args.model}:")
        print(json.dumps(results, indent=2, default=str))
    elif args.all:
        print("\nRunning all enabled models...")
        runner.run_all()
    else:
        parser.print_help()
        print("\n⚠️  Please specify --model <name> or --all")


if __name__ == "__main__":
    main()