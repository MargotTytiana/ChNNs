#!/usr/bin/env python3
"""
Baseline Models Training Script for Speaker Recognition

This script trains traditional baseline methods for comparison with chaotic networks.
Supports multiple baseline architectures and provides comprehensive training management.

Usage:
    python scripts/train_baselines.py --config configs/baseline_config.yaml
    python scripts/train_baselines.py --method mel_mlp --epochs 100
    python scripts/train_baselines.py --all --data_dir ./data/voxceleb
"""
#!/usr/bin/env python3
"""
Baseline Training Script - Fixed Version
This script trains baseline models for the ChNNs project with proper import handling.
"""
import logging
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# ============================================================================
# IMPORT SETUP SECTION
# This section ensures all project modules can be found and imported correctly
# Think of this as preparing your workspace before starting an experiment
# ============================================================================

print("Setting up project imports...")

# Method 1: Try to use our setup_imports module
try:
    from setup_imports import setup_project_imports
    project_root = setup_project_imports()
    print("✓ Import setup completed using setup_imports module")
except ImportError:
    print("setup_imports module not found, using fallback method...")
    
    # Method 2: Fallback - manually configure paths
    # Find the project root by looking for the Model directory
    current_file = Path(__file__).absolute()
    
    # Start from current file location and search upward
    search_path = current_file.parent
    project_root = None
    
    for _ in range(5):  # Search up to 5 levels up
        model_dir = search_path / "Model"
        if model_dir.exists() and model_dir.is_dir():
            project_root = search_path
            break
        if search_path == search_path.parent:  # Reached filesystem root
            break
        search_path = search_path.parent
    
    # If still not found, try the specific path from your system
    if project_root is None:
        specific_path = Path("/scratch/project_2003370/yueyao")
        if (specific_path / "Model").exists():
            project_root = specific_path
    
    if project_root is None:
        raise RuntimeError(
            "Could not locate the Model package directory. "
            "Please ensure your project structure is correct and "
            "the Model directory exists with proper __init__.py files."
        )
    
    # Add the project root to Python path
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
        print(f"✓ Added to Python path: {project_root_str}")

print(f"Project root: {project_root}")

# ============================================================================
# LOGGER SETUP
# Import and configure logging before importing other project modules
# ============================================================================

# First, try to import our logger utilities
try:
    from logger_utils import setup_experiment_logger, log_system_info, TrainingProgressLogger
    print("✓ Logger utilities imported successfully")
except ImportError:
    print("logger_utils not found, creating basic logger setup...")
    
    # Fallback logger setup if logger_utils is not available
    import logging
    
    def setup_experiment_logger(name, log_dir='logs'):
        """Basic logger setup as fallback"""
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger, log_file
    
    def log_system_info(logger):
        """Basic system info logging"""
        import platform
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"Platform: {platform.platform()}")
    
    class TrainingProgressLogger:
        """Simple training progress logger"""
        def __init__(self, logger, total_epochs):
            self.logger = logger
            self.total_epochs = total_epochs
        
        def log_epoch_start(self, epoch):
            if epoch % 10 == 0 or epoch == 1:
                self.logger.info(f"Starting epoch {epoch}/{self.total_epochs}")
        
        def log_epoch_end(self, epoch, **metrics):
            if epoch % 10 == 0 or epoch == self.total_epochs:
                msg = f"Epoch {epoch} completed"
                for key, value in metrics.items():
                    if value is not None:
                        msg += f" | {key}: {value:.4f}"
                self.logger.info(msg)

# For compatibility with your existing code
setup_logger = setup_experiment_logger

# ============================================================================
# PROJECT MODULE IMPORTS
# Import all the project-specific modules with proper error handling
# ============================================================================

print("Importing project modules...")

# Import with detailed error reporting for easier debugging
def safe_import(module_path, item_names, fallback_values=None):
    """
    Safely import modules with detailed error reporting.
    This helps identify exactly which imports are failing and why.
    """
    imported_items = {}
    
    if isinstance(item_names, str):
        item_names = [item_names]
    
    if fallback_values is None:
        fallback_values = [None] * len(item_names)
    elif not isinstance(fallback_values, list):
        fallback_values = [fallback_values] * len(item_names)
    
    try:
        module = __import__(module_path, fromlist=item_names)
        for i, item_name in enumerate(item_names):
            try:
                imported_items[item_name] = getattr(module, item_name)
                print(f"  ✓ {module_path}.{item_name}")
            except AttributeError:
                imported_items[item_name] = fallback_values[i]
                print(f"  ⚠ {module_path}.{item_name} not found, using fallback")
    except ImportError as e:
        print(f"  ❌ Failed to import {module_path}: {e}")
        for i, item_name in enumerate(item_names):
            imported_items[item_name] = fallback_values[i]
    
    return imported_items

# Import base experiment class
base_imports = safe_import('Model.experiments.base_experiment', 'BaseExperiment')
BaseExperiment = base_imports['BaseExperiment']

# Import model classes
model_imports = safe_import('Model.models.hybrid_models', 
                           ['TraditionalMLPBaseline', 'HybridModelManager'])
TraditionalMLPBaseline = model_imports['TraditionalMLPBaseline']
HybridModelManager = model_imports['HybridModelManager']

mlp_imports = safe_import('Model.models.mlp_classifier', 'MLPClassifier')
MLPClassifier = mlp_imports['MLPClassifier']

# Import data handling modules
data_imports = safe_import('Model.data.dataset_loader', 
                          ['create_speaker_dataloaders', 'LibriSpeechChaoticDataset'])
create_speaker_dataloaders = data_imports['create_speaker_dataloaders']
LibriSpeechChaoticDataset = data_imports['LibriSpeechChaoticDataset']

# Import feature extractors
feature_imports = safe_import('Model.features.traditional_features', 
                             ['MelExtractor', 'MFCCExtractor'])
MelExtractor = feature_imports['MelExtractor']
MFCCExtractor = feature_imports['MFCCExtractor']

print("Import setup completed!")

# ============================================================================
# TRAINING CONFIGURATION AND CLASSES
# ============================================================================






class BaselineTrainingManager:
    """
    Manager class for training baseline models.
    
    Handles configuration, experiment setup, training orchestration,
    and result collection for all baseline methods.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = './outputs/baselines',
        verbose: bool = True
    ):
        """
        Initialize baseline training manager.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory for saving outputs
            verbose: Whether to enable verbose logging
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logger(
            name='baseline_trainer',
            log_file=str(self.output_dir / 'training.log'),
            level=logging.INFO if verbose else logging.WARNING
        )
        
        # Initialize tracking
        self.training_results = {}
        self.failed_experiments = []
        self.training_times = {}
        
        # Set reproducibility
        if 'seed' in config:
            set_seed(config['seed'])
        
        self.logger.info("="*60)
        self.logger.info("BASELINE MODELS TRAINING")
        self.logger.info("="*60)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Log system information
        system_info = get_system_info()
        self.logger.info(f"System info: {system_info}")
    
    def get_available_methods(self) -> List[str]:
        """Get list of available baseline methods."""
        return ['mel_mlp', 'mfcc_mlp', 'mel_cnn', 'mfcc_cnn']
    
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        required_keys = ['num_speakers', 'batch_size', 'num_epochs']
        
        for key in required_keys:
            if key not in self.config:
                self.logger.error(f"Missing required config key: {key}")
                return False
        
        # Validate method if specified
        if 'method' in self.config:
            if self.config['method'] not in self.get_available_methods():
                self.logger.error(f"Invalid method: {self.config['method']}")
                self.logger.error(f"Available methods: {self.get_available_methods()}")
                return False
        
        # Set defaults
        self.config.setdefault('learning_rate', 0.001)
        self.config.setdefault('weight_decay', 1e-4)
        self.config.setdefault('hidden_dims', [256, 128, 64])
        self.config.setdefault('dropout_rate', 0.3)
        self.config.setdefault('use_batch_norm', True)
        self.config.setdefault('optimizer', {'type': 'adam'})
        self.config.setdefault('scheduler', {'type': 'step', 'params': {'step_size': 30, 'gamma': 0.5}})
        self.config.setdefault('early_stopping', {'patience': 15})
        
        # Feature-specific defaults
        self.config.setdefault('n_mels', 80)
        self.config.setdefault('n_mfcc', 13)
        self.config.setdefault('sample_rate', 16000)
        self.config.setdefault('max_audio_length', 3.0)
        
        return True
    
    def create_experiment_config(self, method: str) -> Dict[str, Any]:
        """Create experiment-specific configuration."""
        experiment_config = self.config.copy()
        experiment_config['baseline_type'] = method
        
        # Method-specific optimizations
        if 'cnn' in method:
            # CNNs often need different hyperparameters
            experiment_config['learning_rate'] = experiment_config.get('learning_rate', 0.001) * 0.5
            experiment_config['hidden_dims'] = [128, 64]  # Smaller for CNN
        
        if 'mfcc' in method:
            # MFCC has fewer dimensions than Mel
            if 'mlp' in method:
                experiment_config['hidden_dims'] = [128, 64, 32]
        
        return experiment_config
    
    def train_single_method(
        self,
        method: str,
        run_id: int = 0,
        save_model: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Train a single baseline method.
        
        Args:
            method: Baseline method name
            run_id: Run identifier for multiple runs
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary or None if failed
        """
        self.logger.info(f"Training {method} (Run {run_id + 1})...")
        
        try:
            # Create experiment configuration
            experiment_config = self.create_experiment_config(method)
            
            # 确保 num_speakers 在配置中
            if 'num_speakers' not in experiment_config:
                experiment_config['num_speakers'] = self.config.get('num_speakers', 100)
    
            # Create experiment
            experiment_name = f"baseline_{method}_run_{run_id}"
            experiment = BaselineExperiment(
                config=experiment_config,
                experiment_name=experiment_name,
                output_dir=str(self.output_dir / 'experiments'),
                device=self.config.get('device', 'auto'),
                seed=self.config.get('seed', 42) + run_id  # Different seed for each run
            )
            
            # Setup experiment
            start_time = time.time()
            self.logger.info(f"Setting up {method}...")
            experiment.setup()
            
            # Log model information
            if hasattr(experiment, 'model') and experiment.model is not None:
                total_params = sum(p.numel() for p in experiment.model.parameters())
                trainable_params = sum(p.numel() for p in experiment.model.parameters() if p.requires_grad)
                self.logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
            
            # Train model
            self.logger.info(f"Starting training for {self.config['num_epochs']} epochs...")
            experiment.train(self.config['num_epochs'])
            
            training_time = time.time() - start_time
            
            # Get test results
            self.logger.info(f"Evaluating {method}...")
            test_metrics = experiment.test()
            
            # Run baseline analysis
            analysis_results = experiment.run_baseline_analysis()
            
            # Collect results
            results = {
                'method': method,
                'run_id': run_id,
                'experiment_name': experiment_name,
                'training_time': training_time,
                'test_metrics': test_metrics,
                'analysis': analysis_results,
                'best_epoch': experiment.state.best_epoch,
                'best_metric': experiment.state.best_metric,
                'config': experiment_config
            }
            
            self.logger.info(
                f"Completed {method} (Run {run_id + 1}): "
                f"Accuracy: {test_metrics.get('accuracy', 0):.4f}, "
                f"Loss: {test_metrics.get('loss', float('inf')):.4f}, "
                f"Time: {training_time:.1f}s"
            )
            
            # Save model if requested
            if save_model:
                model_save_path = self.output_dir / 'models' / f"{method}_run_{run_id}.pth"
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                
                if hasattr(experiment, 'model'):
                    experiment.save_checkpoint(str(model_save_path))
                    results['model_path'] = str(model_save_path)
            
            # Save detailed results
            results_file = self.output_dir / 'results' / f"{method}_run_{run_id}_results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to train {method} (Run {run_id + 1}): {e}")
            self.failed_experiments.append({
                'method': method,
                'run_id': run_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def train_all_methods(
        self,
        methods: Optional[List[str]] = None,
        num_runs: int = 1,
        save_models: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Train all baseline methods.
        
        Args:
            methods: List of methods to train (if None, train all)
            num_runs: Number of runs per method
            save_models: Whether to save trained models
            
        Returns:
            Dictionary mapping methods to their results
        """
        if methods is None:
            methods = self.get_available_methods()
        
        self.logger.info(f"Training {len(methods)} methods with {num_runs} runs each...")
        self.logger.info(f"Methods: {methods}")
        
        all_results = {}
        total_experiments = len(methods) * num_runs
        completed_experiments = 0
        
        for method in methods:
            method_results = []
            
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"TRAINING METHOD: {method.upper()}")
            self.logger.info(f"{'='*50}")
            
            for run_id in range(num_runs):
                result = self.train_single_method(
                    method=method,
                    run_id=run_id,
                    save_model=save_models
                )
                
                if result:
                    method_results.append(result)
                    self.training_results[f"{method}_run_{run_id}"] = result
                
                completed_experiments += 1
                progress = (completed_experiments / total_experiments) * 100
                
                self.logger.info(f"Progress: {completed_experiments}/{total_experiments} ({progress:.1f}%)")
            
            all_results[method] = method_results
            
            # Method summary
            if method_results:
                accuracies = [r['test_metrics'].get('accuracy', 0) for r in method_results]
                mean_acc = sum(accuracies) / len(accuracies)
                std_acc = (sum((acc - mean_acc)**2 for acc in accuracies) / len(accuracies))**0.5
                
                self.logger.info(f"{method} Summary: {mean_acc:.4f} ± {std_acc:.4f}")
            else:
                self.logger.warning(f"No successful runs for {method}")
        
        return all_results
    
    def analyze_results(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze training results across methods."""
        self.logger.info("Analyzing baseline training results...")
        
        analysis = {
            'summary': {},
            'comparisons': {},
            'best_methods': {},
            'statistics': {}
        }
        
        # Method summaries
        for method, method_results in results.items():
            if not method_results:
                continue
            
            accuracies = [r['test_metrics'].get('accuracy', 0) for r in method_results]
            losses = [r['test_metrics'].get('loss', float('inf')) for r in method_results]
            times = [r['training_time'] for r in method_results]
            
            # Calculate statistics
            mean_acc, lower_acc, upper_acc = StatisticalAnalyzer.compute_confidence_interval(accuracies)
            
            analysis['summary'][method] = {
                'num_runs': len(method_results),
                'accuracy': {
                    'mean': float(sum(accuracies) / len(accuracies)),
                    'std': float((sum((acc - mean_acc)**2 for acc in accuracies) / len(accuracies))**0.5),
                    'min': float(min(accuracies)),
                    'max': float(max(accuracies)),
                    'ci_lower': float(lower_acc),
                    'ci_upper': float(upper_acc)
                },
                'loss': {
                    'mean': float(sum(losses) / len(losses)),
                    'std': float((sum((loss - sum(losses)/len(losses))**2 for loss in losses) / len(losses))**0.5),
                    'min': float(min(losses)),
                    'max': float(max(losses))
                },
                'training_time': {
                    'mean': float(sum(times) / len(times)),
                    'std': float((sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5),
                    'total': float(sum(times))
                }
            }
        
        # Statistical comparisons
        methods_list = list(results.keys())
        if len(methods_list) > 1:
            for i, method1 in enumerate(methods_list):
                for method2 in methods_list[i+1:]:
                    if results[method1] and results[method2]:
                        acc1 = [r['test_metrics'].get('accuracy', 0) for r in results[method1]]
                        acc2 = [r['test_metrics'].get('accuracy', 0) for r in results[method2]]
                        
                        if len(acc1) > 1 and len(acc2) > 1:
                            comparison = StatisticalAnalyzer.perform_t_test(acc1, acc2)
                            analysis['comparisons'][f"{method1}_vs_{method2}"] = comparison
        
        # Best methods
        if analysis['summary']:
            best_by_accuracy = max(
                analysis['summary'].items(),
                key=lambda x: x[1]['accuracy']['mean']
            )
            analysis['best_methods']['accuracy'] = {
                'method': best_by_accuracy[0],
                'score': best_by_accuracy[1]['accuracy']['mean']
            }
            
            best_by_efficiency = min(
                analysis['summary'].items(),
                key=lambda x: x[1]['training_time']['mean']
            )
            analysis['best_methods']['efficiency'] = {
                'method': best_by_efficiency[0],
                'time': best_by_efficiency[1]['training_time']['mean']
            }
        
        # Overall statistics
        analysis['statistics'] = {
            'total_experiments': sum(len(results[m]) for m in results),
            'successful_experiments': sum(len(results[m]) for m in results),
            'failed_experiments': len(self.failed_experiments),
            'total_training_time': sum(
                sum(r['training_time'] for r in method_results)
                for method_results in results.values()
            )
        }
        
        return analysis
    
    def generate_report(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        analysis: Dict[str, Any]
    ):
        """Generate comprehensive training report."""
        self.logger.info("Generating baseline training report...")
        
        # Text report
        report_file = self.output_dir / 'baseline_training_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("BASELINE MODELS TRAINING REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {json.dumps(self.config, indent=2)}\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            stats = analysis['statistics']
            f.write(f"Total experiments: {stats['total_experiments']}\n")
            f.write(f"Successful experiments: {stats['successful_experiments']}\n")
            f.write(f"Failed experiments: {stats['failed_experiments']}\n")
            f.write(f"Total training time: {stats['total_training_time']:.1f}s ({stats['total_training_time']/3600:.2f}h)\n\n")
            
            # Method results
            f.write("METHOD RESULTS\n")
            f.write("-" * 15 + "\n")
            for method, summary in analysis['summary'].items():
                f.write(f"\n{method.upper()}:\n")
                acc = summary['accuracy']
                loss = summary['loss']
                time_info = summary['training_time']
                
                f.write(f"  Accuracy: {acc['mean']:.4f} ± {acc['std']:.4f} [{acc['ci_lower']:.4f}, {acc['ci_upper']:.4f}]\n")
                f.write(f"  Loss: {loss['mean']:.4f} ± {loss['std']:.4f}\n")
                f.write(f"  Training Time: {time_info['mean']:.1f}s ± {time_info['std']:.1f}s\n")
                f.write(f"  Number of runs: {summary['num_runs']}\n")
            
            # Best methods
            f.write("\nBEST METHODS\n")
            f.write("-" * 12 + "\n")
            if 'best_methods' in analysis and analysis['best_methods']:
                best = analysis['best_methods']
                if 'accuracy' in best:
                    f.write(f"Best Accuracy: {best['accuracy']['method']} ({best['accuracy']['score']:.4f})\n")
                if 'efficiency' in best:
                    f.write(f"Most Efficient: {best['efficiency']['method']} ({best['efficiency']['time']:.1f}s)\n")
            
            # Statistical comparisons
            if 'comparisons' in analysis and analysis['comparisons']:
                f.write("\nSTATISTICAL COMPARISONS\n")
                f.write("-" * 23 + "\n")
                for comparison, result in analysis['comparisons'].items():
                    f.write(f"{comparison}:\n")
                    f.write(f"  p-value: {result['p_value']:.6f}\n")
                    f.write(f"  Significant: {result['significant']}\n")
                    f.write(f"  Effect size: {result['effect_size']:.3f}\n\n")
            
            # Failed experiments
            if self.failed_experiments:
                f.write("FAILED EXPERIMENTS\n")
                f.write("-" * 18 + "\n")
                for failure in self.failed_experiments:
                    f.write(f"Method: {failure['method']}, Run: {failure['run_id']}\n")
                    f.write(f"Error: {failure['error']}\n")
                    f.write(f"Time: {failure['timestamp']}\n\n")
        
        # JSON report
        json_report_file = self.output_dir / 'baseline_training_results.json'
        
        full_report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'config': self.config,
                'system_info': get_system_info()
            },
            'results': results,
            'analysis': analysis,
            'failed_experiments': self.failed_experiments
        }
        
        with open(json_report_file, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
        
        self.logger.info(f"Reports saved:")
        self.logger.info(f"  Text report: {report_file}")
        self.logger.info(f"  JSON report: {json_report_file}")
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Baseline training completed.")
        self.logger.info("="*60)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    # 计算相对于train_baselines.py的LibriSpeech路径
    script_dir = Path(__file__).parent  # scripts目录
    model_root = script_dir.parent  # Model根目录
    project_root = model_root.parent  # Project 根目录 
    librispeech_path = project_root / "dataset" / "train-clean-100" / "LibriSpeech" / "train-clean-100"
    
    return {
        'num_speakers': 100,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.3,
        'use_batch_norm': True,
        'n_mels': 80,
        'n_mfcc': 13,
        'sample_rate': 16000,
        'max_audio_length': 3.0,
        'data_dir': str(librispeech_path),
        'device': 'auto',
        'seed': 42,
        'optimizer': {
            'type': 'adam',
            'params': {}
        },
        'scheduler': {
            'type': 'step',
            'params': {
                'step_size': 30,
                'gamma': 0.5
            }
        },
        'early_stopping': {
            'patience': 15
        }
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train baseline models for speaker recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train all baseline methods
    python scripts/train_baselines.py --all
    
    # Train specific method
    python scripts/train_baselines.py --method mel_mlp
    
    # Train with custom config
    python scripts/train_baselines.py --config configs/baseline.yaml
    
    # Train multiple runs for statistical significance
    python scripts/train_baselines.py --method mfcc_cnn --runs 5
    
    # Train on specific dataset
    python scripts/train_baselines.py --all --data_dir ./data/voxceleb --epochs 100
        """
    )
    
    # Configuration arguments
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--output_dir', type=str, default='./outputs/baselines',
                       help='Output directory for results')
    
    # Method selection
    parser.add_argument('--method', type=str, 
                       choices=['mel_mlp', 'mfcc_mlp', 'mel_cnn', 'mfcc_cnn'],
                       help='Specific baseline method to train')
    parser.add_argument('--all', action='store_true',
                       help='Train all baseline methods')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--runs', type=int, default=1, 
                       help='Number of runs per method (for statistical significance)')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, help='Path to dataset')
    parser.add_argument('--num_speakers', type=int, help='Number of speakers in dataset')
    
    # System parameters
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    # Options
    parser.add_argument('--save_models', action='store_true', default=True,
                       help='Save trained models')
    parser.add_argument('--no_save_models', action='store_false', dest='save_models',
                       help='Do not save trained models')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose logging')
    parser.add_argument('--quiet', action='store_false', dest='verbose',
                       help='Disable verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        if not config:
            print(f"Failed to load config from {args.config}, using defaults")
            config = create_default_config()
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.num_speakers:
        config['num_speakers'] = args.num_speakers
    if args.device:
        config['device'] = args.device
    if args.seed:
        config['seed'] = args.seed
    
    # Validate arguments
    if not args.all and not args.method:
        parser.error("Must specify either --all or --method")
    
    if args.all and args.method:
        parser.error("Cannot specify both --all and --method")
    
    # Create training manager
    try:
        manager = BaselineTrainingManager(
            config=config,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        # Validate configuration
        if not manager.validate_config():
            print("Configuration validation failed. Please check your settings.")
            sys.exit(1)
        
        # Determine methods to train
        if args.all:
            methods = manager.get_available_methods()
            print(f"Training all baseline methods: {methods}")
        else:
            methods = [args.method]
            print(f"Training method: {args.method}")
        
        # Train models
        print(f"Starting training with {args.runs} runs per method...")
        results = manager.train_all_methods(
            methods=methods,
            num_runs=args.runs,
            save_models=args.save_models
        )
        
        # Analyze results
        analysis = manager.analyze_results(results)
        
        # Generate report
        manager.generate_report(results, analysis)
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        if analysis['summary']:
            print("RESULTS SUMMARY:")
            for method, summary in analysis['summary'].items():
                acc = summary['accuracy']
                print(f"  {method}: {acc['mean']:.4f} ± {acc['std']:.4f}")
        
        if analysis['best_methods']:
            best = analysis['best_methods']
            if 'accuracy' in best:
                print(f"Best method: {best['accuracy']['method']} ({best['accuracy']['score']:.4f})")
        
        print(f"Results saved to: {args.output_dir}")
        
        # Cleanup
        manager.cleanup()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()