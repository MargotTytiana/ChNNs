#!/usr/bin/env python3
"""
Chaotic Neural Networks Training Script for Speaker Recognition

This script trains chaotic neural network models for robust speaker recognition.
Supports multiple chaotic systems, hybrid architectures, and comprehensive analysis.

Usage:
    python scripts/train_chaotic.py --config configs/chaotic_config.yaml
    python scripts/train_chaotic.py --system lorenz --model_type full_chaotic
    python scripts/train_chaotic.py --all --data_dir ./data/voxceleb --epochs 100
"""

import os
import sys
import argparse
import yaml
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import torch
import torch.nn as nn

def fix_imports():
    current_file = Path(__file__).resolve()
    model_dir = current_file.parent.parent  # experiments -> Model
    paths = [
        str(model_dir),
        str(model_dir/'experiments'), 
        str(model_dir/'models'),
        str(model_dir/'features'),
        str(model_dir/'data'),
        str(model_dir/'utils')
    ]
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    return model_dir

MODEL_DIR = fix_imports()

from chaotic_experiment import ChaoticExperiment, create_chaotic_experiments
from logger import setup_logger
from reproducibility import set_seed, get_system_info
from metrics import evaluate_model_comprehensive, StatisticalAnalyzer
from chaos_utils import validate_chaotic_parameters, optimize_chaotic_parameters

# Mock chaos utilities if not available
try:
    from core.chaos_utils import validate_chaotic_parameters, optimize_chaotic_parameters
except ImportError:
    def validate_chaotic_parameters(system_type, **params):
        return True, "Mock validation"
    
    def optimize_chaotic_parameters(system_type, **params):
        return params


class ChaoticTrainingManager:
    """
    Manager class for training chaotic neural networks.
    
    Handles configuration, chaos parameter optimization, training orchestration,
    dynamics monitoring, and comprehensive result analysis for chaotic models.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = './outputs/chaotic',
        verbose: bool = True,
        args = None
    ):
        """
        Initialize chaotic training manager.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory for saving outputs
            verbose: Whether to enable verbose logging
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.args = args
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'experiments').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'analysis').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = setup_logger(
            name='chaotic_trainer',
            log_file=str(self.output_dir / 'training.log'),
            level=logging.INFO if verbose else logging.WARNING
        )
        
        # Initialize tracking
        self.training_results = {}
        self.failed_experiments = []
        self.chaotic_dynamics = {}
        self.convergence_history = {}
        
        # Set reproducibility
        if 'seed' in config:
            set_seed(config['seed'])
        
        self.logger.info("="*60)
        self.logger.info("CHAOTIC NEURAL NETWORKS TRAINING")
        self.logger.info("="*60)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Log system information
        system_info = get_system_info()
        self.logger.info(f"System info: {system_info}")
    
    def get_available_systems(self) -> List[str]:
        """Get list of available chaotic systems."""
        return ['lorenz', 'rossler', 'mackey_glass', 'chua']
    
    def get_available_model_types(self) -> List[str]:
        """Get list of available model types."""
        return ['full_chaotic', 'traditional_chaotic', 'chaotic_mlp']
    
    def validate_config(self) -> bool:
        """Validate configuration parameters with chaotic system checks."""
        required_keys = ['num_speakers', 'batch_size', 'num_epochs', 'chaotic_system']
        
        for key in required_keys:
            if key not in self.config:
                self.logger.error(f"Missing required config key: {key}")
                return False
        
        # Validate chaotic system
        if self.config['chaotic_system'] not in self.get_available_systems():
            self.logger.error(f"Invalid chaotic system: {self.config['chaotic_system']}")
            self.logger.error(f"Available systems: {self.get_available_systems()}")
            return False
        
        # Validate model type if specified
        if 'model_type' in self.config:
            if self.config['model_type'] not in self.get_available_model_types():
                self.logger.error(f"Invalid model type: {self.config['model_type']}")
                self.logger.error(f"Available types: {self.get_available_model_types()}")
                return False
        
        # Set chaotic network defaults (lower learning rates for stability)
        self.config.setdefault('learning_rate', 0.0005)
        self.config.setdefault('weight_decay', 1e-5)
        self.config.setdefault('gradient_clipping', 1.0)
        
        # Phase space reconstruction defaults
        self.config.setdefault('embedding_dim', 10)
        self.config.setdefault('delay_method', 'autocorr')
        
        # Chaotic feature extraction defaults
        self.config.setdefault('mlsa_scales', 5)
        self.config.setdefault('rqa_radius_ratio', 0.1)
        
        # Chaotic embedding defaults
        self.config.setdefault('evolution_time', 0.5)
        self.config.setdefault('time_step', 0.01)
        self.config.setdefault('coupling_strength', 1.0)
        self.config.setdefault('noise_level', 0.001)
        
        # Attractor pooling defaults
        self.config.setdefault('pooling_type', 'comprehensive')
        
        # Speaker embedding defaults
        self.config.setdefault('speaker_embedding_dim', 128)
        self.config.setdefault('embedding_hidden_dims', [64, 32])
        
        # Classification defaults
        self.config.setdefault('classifier_type', 'cosine')
        self.config.setdefault('temperature', 30.0)
        self.config.setdefault('margin', 0.35)
        
        # Audio processing defaults
        self.config.setdefault('sample_rate', 16000)
        self.config.setdefault('frame_length', 400)
        self.config.setdefault('hop_length', 160)
        self.config.setdefault('max_audio_length', 3.0)
        
        # Optimizer defaults (AdamW better for chaotic systems)
        self.config.setdefault('optimizer', {
            'type': 'adamw',
            'params': {
                'betas': [0.9, 0.999],
                'eps': 1e-8
            }
        })
        
        # Scheduler defaults
        self.config.setdefault('scheduler', {
            'type': 'cosine',
            'params': {
                'T_max': self.config['num_epochs'],
                'eta_min': 1e-6
            }
        })
        
        # Early stopping (more patience for chaotic systems)
        self.config.setdefault('early_stopping', {'patience': 20})
        
        # Validate chaotic parameters
        chaotic_params = {
            'evolution_time': self.config['evolution_time'],
            'time_step': self.config['time_step'],
            'coupling_strength': self.config['coupling_strength'],
            'noise_level': self.config['noise_level']
        }
        
        is_valid, message = validate_chaotic_parameters(
            self.config['chaotic_system'], 
            **chaotic_params
        )
        
        if not is_valid:
            self.logger.warning(f"Chaotic parameter validation warning: {message}")
            # Attempt parameter optimization
            optimized_params = optimize_chaotic_parameters(
                self.config['chaotic_system'],
                **chaotic_params
            )
            self.config.update(optimized_params)
            self.logger.info("Applied chaotic parameter optimization")
        
        return True
    
    def create_experiment_config(
        self, 
        system: str, 
        model_type: str = 'full_chaotic'
    ) -> Dict[str, Any]:
        """Create experiment-specific configuration."""
        experiment_config = self.config.copy()
        experiment_config['chaotic_system'] = system
        experiment_config['model_type'] = model_type
        
        # System-specific optimizations
        if system == 'lorenz':
            # Lorenz system is well-behaved, can use standard parameters
            experiment_config.setdefault('evolution_time', 0.5)
            experiment_config.setdefault('coupling_strength', 1.0)
            
        elif system == 'rossler':
            # Rössler system needs longer evolution time
            experiment_config.setdefault('evolution_time', 0.8)
            experiment_config.setdefault('coupling_strength', 0.8)
            experiment_config.setdefault('time_step', 0.005)  # Smaller step for stability
            
        elif system == 'mackey_glass':
            # Mackey-Glass is a delay differential equation
            experiment_config.setdefault('evolution_time', 1.0)
            experiment_config.setdefault('coupling_strength', 0.5)
            experiment_config.setdefault('delay_tau', 17)  # Characteristic delay
            
        elif system == 'chua':
            # Chua's circuit can be sensitive
            experiment_config.setdefault('evolution_time', 0.3)
            experiment_config.setdefault('coupling_strength', 1.2)
            experiment_config.setdefault('noise_level', 0.0005)  # Less noise
        
        # Model type specific adjustments
        if model_type == 'traditional_chaotic':
            # Traditional features + chaotic processing
            experiment_config.setdefault('feature_type', 'mel')
            experiment_config.setdefault('n_mels', 80)
            
        elif model_type == 'chaotic_mlp':
            # Chaotic features + MLP classifier
            experiment_config.setdefault('mlp_hidden_dims', [128, 64, 32])
            experiment_config.setdefault('dropout_rate', 0.2)
        
        return experiment_config
    
    def optimize_chaotic_parameters(
        self, 
        system: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize chaotic system parameters for better performance."""
        self.logger.info(f"Optimizing parameters for {system} system...")
        
        original_params = {
            'evolution_time': config['evolution_time'],
            'time_step': config['time_step'],
            'coupling_strength': config['coupling_strength'],
            'noise_level': config['noise_level']
        }
        
        # Apply optimization
        optimized_params = optimize_chaotic_parameters(system, **original_params)
        
        # Log changes
        for param, original_val in original_params.items():
            new_val = optimized_params.get(param, original_val)
            if abs(new_val - original_val) > 1e-6:
                self.logger.info(f"  {param}: {original_val:.4f} -> {new_val:.4f}")
        
        # Update config
        config.update(optimized_params)
        return config
    
    def monitor_chaotic_dynamics(
        self,
        experiment: ChaoticExperiment,
        epoch: int
    ) -> Dict[str, float]:
        """Monitor chaotic system dynamics during training."""
        dynamics = {}
        
        try:
            # Extract sample trajectories for analysis
            if hasattr(experiment.model, 'forward'):
                # Get a small batch for analysis
                sample_batch = next(iter(experiment.val_loader))
                sample_audio, _ = sample_batch
                sample_audio = sample_audio[:4].to(experiment.device)
                
                with torch.no_grad():
                    _, intermediates = experiment.model(sample_audio, return_intermediates=True)
                    
                    if 'chaotic_trajectories' in intermediates:
                        trajectories = intermediates['chaotic_trajectories']
                        
                        # Trajectory stability
                        traj_std = torch.std(trajectories).item()
                        traj_range = (torch.max(trajectories) - torch.min(trajectories)).item()
                        
                        dynamics['trajectory_std'] = traj_std
                        dynamics['trajectory_range'] = traj_range
                        
                        # Lyapunov-like measure (local divergence)
                        if trajectories.shape[1] > 1:
                            diffs = torch.diff(trajectories, dim=1)
                            div_rate = torch.mean(torch.norm(diffs, dim=2)).item()
                            dynamics['divergence_rate'] = div_rate
                    
                    if 'pooled_features' in intermediates:
                        pooled = intermediates['pooled_features']
                        
                        # Feature diversity
                        feature_diversity = torch.std(pooled, dim=0).mean().item()
                        dynamics['feature_diversity'] = feature_diversity
                        
                        # Feature magnitude
                        feature_magnitude = torch.norm(pooled, dim=1).mean().item()
                        dynamics['feature_magnitude'] = feature_magnitude
                    
                    if 'speaker_embeddings' in intermediates:
                        embeddings = intermediates['speaker_embeddings']
                        
                        # Embedding separation
                        if embeddings.shape[0] > 1:
                            pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
                            mask = ~torch.eye(embeddings.shape[0], dtype=bool, device=embeddings.device)
                            avg_separation = pairwise_dist[mask].mean().item()
                            dynamics['embedding_separation'] = avg_separation
        
        except Exception as e:
            self.logger.debug(f"Chaotic dynamics monitoring failed: {e}")
        
        return dynamics
    
    def train_single_experiment(
        self,
        system: str,
        model_type: str = 'full_chaotic',
        run_id: int = 0,
        save_model: bool = True,
        analyze_dynamics: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Train a single chaotic network experiment.
        
        Args:
            system: Chaotic system type
            model_type: Model architecture type
            run_id: Run identifier
            save_model: Whether to save trained model
            analyze_dynamics: Whether to perform dynamics analysis
            
        Returns:
            Training results or None if failed
        """
        experiment_name = f"chaotic_{system}_{model_type}_run_{run_id}"
        self.logger.info(f"Training {experiment_name}...")
        
        try:
            # Create experiment configuration
            experiment_config = self.create_experiment_config(system, model_type)
            
            # Optimize chaotic parameters
            if self.config.get('optimize_parameters', True):
                experiment_config = self.optimize_chaotic_parameters(system, experiment_config)
            
            # Create experiment
            experiment = ChaoticExperiment(
                config=experiment_config,
                experiment_name=experiment_name,
                output_dir=str(self.output_dir / 'experiments'),
                device=self.config.get('device', 'auto'),
                seed=self.config.get('seed', 42) + run_id
            )
            
            # Setup experiment
            start_time = time.time()
            self.logger.info(f"Setting up {experiment_name}...")
            experiment.setup()
            
            # ==================== 健壮的检查点恢复逻辑 ====================
            start_epoch = 0
            best_metric = 0.0
            
            if hasattr(self, 'args') and self.args and getattr(self.args, 'resume', None):
                checkpoint_path = self.args.resume
                self.logger.info(f"🔄 Attempting to resume from: {checkpoint_path}")
                
                if os.path.exists(checkpoint_path):
                    try:
                        # 智能文件加载
                        if checkpoint_path.endswith(('.pth', '.pt')):
                            checkpoint = torch.load(checkpoint_path, map_location=experiment.device, weights_only=False)
                            self.logger.info("✅ Loaded with torch.load")
                        elif checkpoint_path.endswith('.pkl'):
                            import pickle
                            with open(checkpoint_path, 'rb') as f:
                                checkpoint = pickle.load(f)
                            self.logger.info("✅ Loaded with pickle")
                        else:
                            # 尝试两种方法
                            try:
                                checkpoint = torch.load(checkpoint_path, map_location=experiment.device, weights_only=False)
                                self.logger.info("✅ Loaded unknown extension with torch.load")
                            except:
                                import pickle
                                with open(checkpoint_path, 'rb') as f:
                                    checkpoint = pickle.load(f)
                                self.logger.info("✅ Loaded unknown extension with pickle")
                        
                        self.logger.info(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
                        
                        # 提取模型状态字典
                        model_state_dict = None
                        if 'model_state_dict' in checkpoint:
                            model_state_dict = checkpoint['model_state_dict']
                            self.logger.info("📦 Using model_state_dict")
                        elif 'model' in checkpoint:
                            model_state_dict = checkpoint['model'] 
                            self.logger.info("📦 Using model (converted)")
                        elif all(not k.startswith(('epoch', 'best_', 'training_', 'config')) 
                                for k in checkpoint.keys()):
                            model_state_dict = checkpoint
                            self.logger.info("📦 Checkpoint is pure model state")
                        
                        # 健壮的模型加载（非严格模式）
                        if model_state_dict and hasattr(experiment, 'model') and experiment.model is not None:
                            current_state = experiment.model.state_dict()
                            
                            # 只加载形状匹配的参数
                            pretrained_dict = {}
                            for key, value in model_state_dict.items():
                                if key in current_state and value.shape == current_state[key].shape:
                                    pretrained_dict[key] = value
                            
                            # 更新模型参数
                            current_state.update(pretrained_dict)
                            experiment.model.load_state_dict(current_state)
                            
                            loaded_count = len(pretrained_dict)
                            total_count = len(current_state)
                            match_rate = loaded_count / total_count
                            
                            self.logger.info(f"🔧 Loaded {loaded_count}/{total_count} parameters ({match_rate:.1%})")
                            
                            if loaded_count > 0:
                                self.logger.info("✅ Model parameters partially loaded successfully")
                            else:
                                self.logger.warning("⚠️  No parameters loaded - architecture mismatch")
                        
                        # 恢复训练状态
                        if 'epoch' in checkpoint:
                            start_epoch = checkpoint['epoch'] + 1
                            self.logger.info(f"⏩ Resuming from epoch {start_epoch}")
                        else:
                            self.logger.warning("📅 No epoch information in checkpoint")
                        
                        if 'best_metric' in checkpoint:
                            best_metric = checkpoint['best_metric']
                            self.logger.info(f"🏆 Previous best metric: {best_metric:.4f}")
                        
                    except Exception as e:
                        self.logger.error(f"❌ Checkpoint recovery failed: {e}")
                        import traceback
                        self.logger.error(f"🔍 Traceback: {traceback.format_exc()}")
                        self.logger.info("🔄 Starting training from scratch")
                        start_epoch = 0
                else:
                    self.logger.error(f"❌ Checkpoint file not found: {checkpoint_path}")
                    self.logger.info("🔄 Starting training from scratch")
                    start_epoch = 0
            else:
                self.logger.info("🚀 No checkpoint specified, starting from scratch")
            # ==================== 检查点恢复逻辑结束 ====================

            
            # Log model information
            if hasattr(experiment, 'model') and experiment.model is not None:
                total_params = sum(p.numel() for p in experiment.model.parameters())
                trainable_params = sum(p.numel() for p in experiment.model.parameters() if p.requires_grad)
                self.logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

            # Initialize dynamics tracking
            epoch_dynamics = []
            
            # Custom training loop with dynamics monitoring
            if analyze_dynamics:
                self.logger.info("Training with chaotic dynamics monitoring...")
                
                # Override train method to include monitoring
                original_train_epoch = experiment.train_epoch
                
                def monitored_train_epoch():
                    metrics = original_train_epoch()
                    dynamics = self.monitor_chaotic_dynamics(experiment, experiment.state.epoch)
                    epoch_dynamics.append({
                        'epoch': experiment.state.epoch,
                        'metrics': metrics,
                        'dynamics': dynamics
                    })
                    return metrics
                
                experiment.train_epoch = monitored_train_epoch

            # Train model - support start_epoch
            remaining_epochs = self.config['num_epochs'] - start_epoch
            if remaining_epochs > 0:
                self.logger.info(f"Starting training from epoch {start_epoch} for {remaining_epochs} more epochs...")
                # 注意：这里需要修改 experiment.train 方法以支持 start_epoch
                # 如果 train 方法不支持，需要手动实现循环
                if hasattr(experiment.train, '__code__') and 'start_epoch' in experiment.train.__code__.co_varnames:
                    experiment.train(remaining_epochs, start_epoch=start_epoch)
                else:
                    # Fallback: 手动设置 epoch 并训练
                    if hasattr(experiment, 'state'):
                        experiment.state.epoch = start_epoch
                    experiment.train(remaining_epochs)
            else:
                self.logger.info(f"Training already completed (reached epoch {start_epoch})")
                # Skip to evaluation
            
            training_time = time.time() - start_time
            
            # Test evaluation
            self.logger.info(f"Evaluating {experiment_name}...")
            test_metrics = experiment.test()
            
            # Comprehensive chaotic analysis
            if analyze_dynamics:
                self.logger.info("Running chaotic analysis...")
                chaotic_analysis = experiment.run_chaotic_analysis()
            else:
                chaotic_analysis = {}
            
            # Collect results
            results = {
                'system': system,
                'model_type': model_type,
                'run_id': run_id,
                'experiment_name': experiment_name,
                'training_time': training_time,
                'test_metrics': test_metrics,
                'chaotic_analysis': chaotic_analysis,
                'dynamics_history': epoch_dynamics if analyze_dynamics else [],
                'best_epoch': experiment.state.best_epoch,
                'best_metric': experiment.state.best_metric,
                'config': experiment_config,
                'convergence_info': {
                    'final_loss': experiment.state.train_losses[-1] if experiment.state.train_losses else None,
                    'best_val_metric': experiment.state.best_metric,
                    'total_epochs': len(experiment.state.train_losses)
                }
            }
            
            self.logger.info(
                f"Completed {experiment_name}: "
                f"Accuracy: {test_metrics.get('accuracy', 0):.4f}, "
                f"Loss: {test_metrics.get('loss', float('inf')):.4f}, "
                f"Time: {training_time:.1f}s"
            )
            
            # Save model if requested
            if save_model:
                model_save_path = self.output_dir / 'models' / f"{system}_{model_type}_run_{run_id}.pth"
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                
                if hasattr(experiment, 'model'):
                    experiment.save_checkpoint(str(model_save_path))
                    results['model_path'] = str(model_save_path)
            
            # Save detailed results
            results_file = self.output_dir / 'analysis' / f"{system}_{model_type}_run_{run_id}_results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'w') as f:
                # Convert tensors and numpy arrays for JSON serialization
                json_results = self._convert_for_json(results)
                json.dump(json_results, f, indent=2, default=str)
            
            # Store in manager
            self.chaotic_dynamics[experiment_name] = epoch_dynamics
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to train {experiment_name}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            self.failed_experiments.append({
                'system': system,
                'model_type': model_type,
                'run_id': run_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def train_all_systems(
        self,
        systems: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        num_runs: int = 1,
        save_models: bool = True,
        analyze_dynamics: bool = True
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Train all chaotic systems and model types.
        
        Args:
            systems: List of chaotic systems to train
            model_types: List of model types to train
            num_runs: Number of runs per configuration
            save_models: Whether to save trained models
            analyze_dynamics: Whether to perform dynamics analysis
            
        Returns:
            Nested dictionary of results
        """
        if systems is None:
            systems = self.get_available_systems()
        if model_types is None:
            model_types = ['full_chaotic']
        
        self.logger.info(f"Training {len(systems)} systems × {len(model_types)} types × {num_runs} runs...")
        self.logger.info(f"Systems: {systems}")
        self.logger.info(f"Model types: {model_types}")
        
        all_results = {}
        total_experiments = len(systems) * len(model_types) * num_runs
        completed_experiments = 0
        
        for system in systems:
            all_results[system] = {}
            
            for model_type in model_types:
                type_results = []
                
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"TRAINING: {system.upper()} + {model_type.upper()}")
                self.logger.info(f"{'='*60}")
                
                for run_id in range(num_runs):
                    result = self.train_single_experiment(
                        system=system,
                        model_type=model_type,
                        run_id=run_id,
                        save_model=save_models,
                        analyze_dynamics=analyze_dynamics
                    )
                    
                    if result:
                        type_results.append(result)
                        experiment_key = f"{system}_{model_type}_run_{run_id}"
                        self.training_results[experiment_key] = result
                    
                    completed_experiments += 1
                    progress = (completed_experiments / total_experiments) * 100
                    self.logger.info(f"Progress: {completed_experiments}/{total_experiments} ({progress:.1f}%)")
                
                all_results[system][model_type] = type_results
                
                # System-model combination summary
                if type_results:
                    accuracies = [r['test_metrics'].get('accuracy', 0) for r in type_results]
                    mean_acc = sum(accuracies) / len(accuracies)
                    std_acc = (sum((acc - mean_acc)**2 for acc in accuracies) / len(accuracies))**0.5
                    
                    self.logger.info(f"{system} + {model_type} Summary: {mean_acc:.4f} ± {std_acc:.4f}")
                else:
                    self.logger.warning(f"No successful runs for {system} + {model_type}")
        
        return all_results
    
    def analyze_chaotic_results(
        self, 
        results: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """Comprehensive analysis of chaotic training results."""
        self.logger.info("Analyzing chaotic training results...")
        
        analysis = {
            'system_comparison': {},
            'model_type_comparison': {},
            'dynamics_analysis': {},
            'convergence_analysis': {},
            'best_configurations': {},
            'statistical_tests': {},
            'summary': {}
        }
        
        # System comparison
        for system in results:
            system_results = []
            for model_type in results[system]:
                system_results.extend(results[system][model_type])
            
            if system_results:
                accuracies = [r['test_metrics'].get('accuracy', 0) for r in system_results]
                times = [r['training_time'] for r in system_results]
                
                mean_acc, lower_acc, upper_acc = StatisticalAnalyzer.compute_confidence_interval(accuracies)
                
                analysis['system_comparison'][system] = {
                    'num_experiments': len(system_results),
                    'accuracy': {
                        'mean': float(mean_acc),
                        'std': float(np.std(accuracies)),
                        'ci_lower': float(lower_acc),
                        'ci_upper': float(upper_acc)
                    },
                    'training_time': {
                        'mean': float(np.mean(times)),
                        'std': float(np.std(times))
                    },
                    'success_rate': len(system_results) / max(1, len(system_results) + 
                                   len([f for f in self.failed_experiments if f['system'] == system]))
                }
        
        # Model type comparison
        model_type_results = {}
        for system in results:
            for model_type in results[system]:
                if model_type not in model_type_results:
                    model_type_results[model_type] = []
                model_type_results[model_type].extend(results[system][model_type])
        
        for model_type, type_results in model_type_results.items():
            if type_results:
                accuracies = [r['test_metrics'].get('accuracy', 0) for r in type_results]
                analysis['model_type_comparison'][model_type] = {
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies))
                }
        
        # Dynamics analysis
        if self.chaotic_dynamics:
            analysis['dynamics_analysis'] = self._analyze_dynamics_evolution()
        
        # Find best configurations
        all_results = []
        for system in results:
            for model_type in results[system]:
                all_results.extend(results[system][model_type])
        
        if all_results:
            best_accuracy = max(all_results, key=lambda x: x['test_metrics'].get('accuracy', 0))
            analysis['best_configurations']['accuracy'] = {
                'system': best_accuracy['system'],
                'model_type': best_accuracy['model_type'],
                'accuracy': best_accuracy['test_metrics'].get('accuracy', 0),
                'config': best_accuracy['config']
            }
            
            fastest_training = min(all_results, key=lambda x: x['training_time'])
            analysis['best_configurations']['efficiency'] = {
                'system': fastest_training['system'],
                'model_type': fastest_training['model_type'],
                'training_time': fastest_training['training_time']
            }
        
        # Statistical tests between systems
        if len(analysis['system_comparison']) > 1:
            systems_list = list(analysis['system_comparison'].keys())
            for i, system1 in enumerate(systems_list):
                for system2 in systems_list[i+1:]:
                    sys1_results = []
                    sys2_results = []
                    
                    for model_type in results.get(system1, {}):
                        sys1_results.extend([r['test_metrics'].get('accuracy', 0) 
                                           for r in results[system1][model_type]])
                    
                    for model_type in results.get(system2, {}):
                        sys2_results.extend([r['test_metrics'].get('accuracy', 0) 
                                           for r in results[system2][model_type]])
                    
                    if len(sys1_results) > 1 and len(sys2_results) > 1:
                        t_test = StatisticalAnalyzer.perform_t_test(sys1_results, sys2_results)
                        analysis['statistical_tests'][f"{system1}_vs_{system2}"] = t_test
        
        return analysis
    
    def _analyze_dynamics_evolution(self) -> Dict[str, Any]:
        """Analyze evolution of chaotic dynamics during training."""
        dynamics_analysis = {}
        
        for experiment_name, epoch_dynamics in self.chaotic_dynamics.items():
            if not epoch_dynamics:
                continue
                
            # Extract dynamics time series
            dynamics_keys = set()
            for epoch_data in epoch_dynamics:
                dynamics_keys.update(epoch_data.get('dynamics', {}).keys())
            
            experiment_analysis = {}
            for key in dynamics_keys:
                values = [epoch_data['dynamics'].get(key, 0) for epoch_data in epoch_dynamics]
                if values:
                    experiment_analysis[key] = {
                        'initial': values[0] if values else 0,
                        'final': values[-1] if values else 0,
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'trend': 'increasing' if values[-1] > values[0] else 'decreasing' if values[-1] < values[0] else 'stable'
                    }
            
            dynamics_analysis[experiment_name] = experiment_analysis
        
        return dynamics_analysis
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert objects for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
    
    def generate_comprehensive_report(
        self,
        results: Dict[str, Dict[str, List[Dict[str, Any]]]],
        analysis: Dict[str, Any]
    ):
        """Generate comprehensive chaotic training report."""
        self.logger.info("Generating comprehensive chaotic training report...")
        
        # Text report
        report_file = self.output_dir / 'chaotic_training_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("CHAOTIC NEURAL NETWORKS TRAINING REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {json.dumps(self.config, indent=2)}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 17 + "\n")
            total_experiments = sum(len(results[s][m]) for s in results for m in results[s])
            total_successful = total_experiments - len(self.failed_experiments)
            
            f.write(f"Total experiments: {total_experiments}\n")
            f.write(f"Successful experiments: {total_successful}\n")
            f.write(f"Success rate: {(total_successful/max(1,total_experiments))*100:.1f}%\n")
            
            if analysis.get('best_configurations', {}).get('accuracy'):
                best = analysis['best_configurations']['accuracy']
                f.write(f"Best configuration: {best['system']} + {best['model_type']} ({best['accuracy']:.4f})\n")
            f.write("\n")
            
            # System Comparison
            f.write("CHAOTIC SYSTEMS COMPARISON\n")
            f.write("-" * 27 + "\n")
            for system, stats in analysis.get('system_comparison', {}).items():
                acc = stats['accuracy']
                time_info = stats['training_time']
                
                f.write(f"{system.upper()}:\n")
                f.write(f"  Accuracy: {acc['mean']:.4f} ± {acc['std']:.4f} [{acc['ci_lower']:.4f}, {acc['ci_upper']:.4f}]\n")
                f.write(f"  Training Time: {time_info['mean']:.1f}s ± {time_info['std']:.1f}s\n")
                f.write(f"  Success Rate: {stats['success_rate']*100:.1f}%\n")
                f.write(f"  Experiments: {stats['num_experiments']}\n\n")
            
            # Model Types Comparison
            f.write("MODEL TYPES COMPARISON\n")
            f.write("-" * 22 + "\n")
            for model_type, stats in analysis.get('model_type_comparison', {}).items():
                f.write(f"{model_type}: {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}\n")
            f.write("\n")
            
            # Dynamics Analysis Summary
            if analysis.get('dynamics_analysis'):
                f.write("CHAOTIC DYNAMICS ANALYSIS\n")
                f.write("-" * 25 + "\n")
                f.write("Key dynamics trends observed during training:\n")
                
                # Aggregate trends
                trend_summary = {}
                for exp_name, exp_analysis in analysis['dynamics_analysis'].items():
                    for metric, metric_data in exp_analysis.items():
                        trend = metric_data.get('trend', 'unknown')
                        if metric not in trend_summary:
                            trend_summary[metric] = {}
                        trend_summary[metric][trend] = trend_summary[metric].get(trend, 0) + 1
                
                for metric, trends in trend_summary.items():
                    f.write(f"  {metric}: {trends}\n")
                f.write("\n")
            
            # Statistical Significance Tests
            if analysis.get('statistical_tests'):
                f.write("STATISTICAL SIGNIFICANCE TESTS\n")
                f.write("-" * 31 + "\n")
                for comparison, result in analysis['statistical_tests'].items():
                    f.write(f"{comparison}:\n")
                    f.write(f"  p-value: {result['p_value']:.6f}\n")
                    f.write(f"  Significant: {result['significant']}\n")
                    f.write(f"  Effect size: {result['effect_size']:.3f}\n\n")
            
            # Failed Experiments
            if self.failed_experiments:
                f.write("FAILED EXPERIMENTS ANALYSIS\n")
                f.write("-" * 27 + "\n")
                
                # Group by error type
                error_summary = {}
                for failure in self.failed_experiments:
                    error_type = type(failure['error']).__name__ if hasattr(failure['error'], '__name__') else 'Unknown'
                    if error_type not in error_summary:
                        error_summary[error_type] = []
                    error_summary[error_type].append(failure)
                
                for error_type, failures in error_summary.items():
                    f.write(f"{error_type}: {len(failures)} failures\n")
                    for failure in failures[:3]:  # Show first 3
                        f.write(f"  {failure['system']} + {failure['model_type']}: {failure['error']}\n")
                    if len(failures) > 3:
                        f.write(f"  ... and {len(failures)-3} more\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            if analysis.get('best_configurations', {}).get('accuracy'):
                best = analysis['best_configurations']['accuracy']
                f.write(f"1. Best performing system: {best['system']} with {best['model_type']}\n")
            
            if analysis.get('system_comparison'):
                # Find most stable system
                most_stable = min(analysis['system_comparison'].items(), 
                                key=lambda x: x[1]['accuracy']['std'])
                f.write(f"2. Most stable system: {most_stable[0]} (lowest variance)\n")
            
            f.write("3. Consider parameter optimization for failed experiments\n")
            f.write("4. Monitor chaotic dynamics for early stopping criteria\n")
        
        # JSON report
        json_report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'config': self.config,
                'system_info': get_system_info()
            },
            'results': self._convert_for_json(results),
            'analysis': self._convert_for_json(analysis),
            'failed_experiments': self.failed_experiments,
            'chaotic_dynamics': self._convert_for_json(self.chaotic_dynamics)
        }
        
        json_report_file = self.output_dir / 'chaotic_training_results.json'
        with open(json_report_file, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # Generate visualizations if possible
        self._generate_visualizations(results, analysis)
        
        self.logger.info(f"Reports generated:")
        self.logger.info(f"  Text report: {report_file}")
        self.logger.info(f"  JSON report: {json_report_file}")
    
    def _generate_visualizations(
        self, 
        results: Dict[str, Dict[str, List[Dict[str, Any]]]], 
        analysis: Dict[str, Any]
    ):
        """Generate visualization plots for chaotic training results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            viz_dir = self.output_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            
            # System performance comparison
            if analysis.get('system_comparison'):
                systems = list(analysis['system_comparison'].keys())
                accuracies = [analysis['system_comparison'][s]['accuracy']['mean'] for s in systems]
                errors = [analysis['system_comparison'][s]['accuracy']['std'] for s in systems]
                
                plt.figure(figsize=(10, 6))
                plt.bar(systems, accuracies, yerr=errors, capsize=5, alpha=0.7)
                plt.xlabel('Chaotic Systems')
                plt.ylabel('Test Accuracy')
                plt.title('Chaotic Systems Performance Comparison')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / 'systems_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Dynamics evolution plots
            if self.chaotic_dynamics:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                metrics_to_plot = ['trajectory_std', 'feature_diversity', 'embedding_separation', 'divergence_rate']
                
                for i, metric in enumerate(metrics_to_plot):
                    if i >= len(axes):
                        break
                        
                    ax = axes[i]
                    
                    for exp_name, epoch_dynamics in list(self.chaotic_dynamics.items())[:3]:  # Plot first 3 experiments
                        epochs = [ed['epoch'] for ed in epoch_dynamics]
                        values = [ed['dynamics'].get(metric, 0) for ed in epoch_dynamics]
                        
                        if any(v != 0 for v in values):
                            ax.plot(epochs, values, label=exp_name.replace('chaotic_', ''), alpha=0.7)
                    
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    ax.set_title(f'{metric.replace("_", " ").title()} Evolution')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(viz_dir / 'dynamics_evolution.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Visualizations saved to: {viz_dir}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available for visualization generation")
        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")
    
    def cleanup(self):
        """Clean up resources and log summary."""
        self.logger.info("Chaotic training completed.")
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
    """Create default chaotic network configuration."""
    return {
        'chaotic_system': 'lorenz',
        'model_type': 'full_chaotic',
        'num_speakers': 100,
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.0005,
        'weight_decay': 1e-5,
        'gradient_clipping': 1.0,
        
        # Chaotic system parameters
        'embedding_dim': 10,
        'mlsa_scales': 5,
        'rqa_radius_ratio': 0.1,
        'evolution_time': 0.5,
        'time_step': 0.01,
        'coupling_strength': 1.0,
        'noise_level': 0.001,
        
        # Network architecture
        'pooling_type': 'comprehensive',
        'speaker_embedding_dim': 128,
        'classifier_type': 'cosine',
        'temperature': 30.0,
        'margin': 0.35,
        
        # Audio processing
        'sample_rate': 16000,
        'max_audio_length': 3.0,
        'data_dir': './data',
        
        # Training settings
        'device': 'auto',
        'seed': 42,
        'optimize_parameters': True,
        
        'optimizer': {
            'type': 'adamw',
            'params': {'betas': [0.9, 0.999]}
        },
        'scheduler': {
            'type': 'cosine',
            'params': {'eta_min': 1e-6}
        },
        'early_stopping': {
            'patience': 20
        }
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train chaotic neural networks for speaker recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train all chaotic systems with full architecture
    python scripts/train_chaotic.py --all
    
    # Train specific system
    python scripts/train_chaotic.py --system lorenz --model_type full_chaotic
    
    # Train with custom config
    python scripts/train_chaotic.py --config configs/chaotic.yaml
    
    # Train multiple systems and architectures
    python scripts/train_chaotic.py --systems lorenz rossler --model_types full_chaotic chaotic_mlp
    
    # Train with optimization and dynamics monitoring
    python scripts/train_chaotic.py --system lorenz --runs 5 --optimize --analyze_dynamics
        """
    )
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--output_dir', type=str, default='./outputs/chaotic',
                       help='Output directory for results')
    
    # System and model selection
    parser.add_argument('--system', type=str, 
                       choices=['lorenz', 'rossler', 'mackey_glass', 'chua'],
                       help='Chaotic system to use')
    parser.add_argument('--systems', type=str, nargs='+',
                       choices=['lorenz', 'rossler', 'mackey_glass', 'chua'],
                       help='Multiple chaotic systems to train')
    parser.add_argument('--model_type', type=str,
                       choices=['full_chaotic', 'traditional_chaotic', 'chaotic_mlp'],
                       help='Model architecture type')
    parser.add_argument('--model_types', type=str, nargs='+',
                       choices=['full_chaotic', 'traditional_chaotic', 'chaotic_mlp'],
                       help='Multiple model types to train')
    parser.add_argument('--all', action='store_true',
                       help='Train all available systems and model types')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs per configuration')
    
    # Chaotic system parameters
    parser.add_argument('--evolution_time', type=float, help='Chaotic evolution time')
    parser.add_argument('--coupling_strength', type=float, help='Coupling strength')
    parser.add_argument('--optimize', action='store_true', default=True,
                       help='Optimize chaotic parameters')
    parser.add_argument('--no_optimize', action='store_false', dest='optimize',
                       help='Disable parameter optimization')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, help='Path to dataset')
    parser.add_argument('--num_speakers', type=int, help='Number of speakers')
    
    # Analysis options
    parser.add_argument('--analyze_dynamics', action='store_true', default=True,
                       help='Enable chaotic dynamics analysis')
    parser.add_argument('--no_analyze_dynamics', action='store_false', dest='analyze_dynamics',
                       help='Disable dynamics analysis')
    parser.add_argument('--save_models', action='store_true', default=True,
                       help='Save trained models')
    parser.add_argument('--no_save_models', action='store_false', dest='save_models',
                       help='Do not save models')
    
    # System parameters
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose logging')
    parser.add_argument('--quiet', action='store_false', dest='verbose',
                       help='Disable verbose logging')


    # Add import from checkpoint
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    
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
    overrides = ['epochs', 'batch_size', 'learning_rate', 'evolution_time', 
                'coupling_strength', 'data_dir', 'num_speakers', 'device', 'seed']
    
    for override in overrides:
        if hasattr(args, override) and getattr(args, override) is not None:
            if override == 'num_epochs':
                config['num_epochs'] = args.epochs
            else:
                config[override] = getattr(args, override)
    
    if hasattr(args, 'optimize'):
        config['optimize_parameters'] = args.optimize
    
    # Determine what to train
    systems = None
    model_types = None
    
    if args.all:
        print("Training all chaotic systems and model types")
    else:
        if args.systems:
            systems = args.systems
        elif args.system:
            systems = [args.system]
        
        if args.model_types:
            model_types = args.model_types
        elif args.model_type:
            model_types = [args.model_type]
        
        if not systems and not model_types:
            parser.error("Must specify --all, --system(s), or --model_type(s)")
    
    # Create training manager
    try:
        manager = ChaoticTrainingManager(
            config=config,
            output_dir=args.output_dir,
            verbose=args.verbose,
            args=args
        )
        
        # Validate configuration
        if not manager.validate_config():
            print("Configuration validation failed. Please check your settings.")
            sys.exit(1)
        
        print(f"Starting chaotic neural network training...")
        if systems:
            print(f"Systems: {systems}")
        if model_types:
            print(f"Model types: {model_types}")
        print(f"Runs per configuration: {args.runs}")
        
        # Train models
        results = manager.train_all_systems(
            systems=systems,
            model_types=model_types,
            num_runs=args.runs,
            save_models=args.save_models,
            analyze_dynamics=args.analyze_dynamics
        )
        
        # Analyze results
        analysis = manager.analyze_chaotic_results(results)
        
        # Generate comprehensive report
        manager.generate_comprehensive_report(results, analysis)
        
        # Print summary
        print("\n" + "="*60)
        print("CHAOTIC TRAINING COMPLETED")
        print("="*60)
        
        if analysis.get('best_configurations', {}).get('accuracy'):
            best = analysis['best_configurations']['accuracy']
            print(f"Best configuration: {best['system']} + {best['model_type']}")
            print(f"Best accuracy: {best['accuracy']:.4f}")
        
        if analysis.get('system_comparison'):
            print("\nSystems performance:")
            for system, stats in analysis['system_comparison'].items():
                acc = stats['accuracy']
                print(f"  {system}: {acc['mean']:.4f} ± {acc['std']:.4f}")
        
        print(f"Results saved to: {args.output_dir}")
        
        # Cleanup
        manager.cleanup()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed with error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()