#!/usr/bin/env python3
"""
Experiment 4: Computational Efficiency Evaluation

This script evaluates the computational efficiency of different models:
- Number of parameters
- FLOPs (Floating Point Operations)
- Training time per epoch
- Inference time per sample
- Memory usage
- GPU utilization

Usage:
    python scripts/run_experiment4_efficiency.py --all
    python scripts/run_experiment4_efficiency.py --model mel_mlp
    python scripts/run_experiment4_efficiency.py --benchmark
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import gc

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
    'batch_size': 32,
    
    # Benchmark settings
    'num_warmup_batches': 5,
    'num_benchmark_batches': 20,
    'num_inference_runs': 100,
    
    # Model settings
    'hidden_dims': [512, 256, 128],
    'dropout_rate': 0.3,
    'use_batch_norm': True,
    
    # Output settings
    'output_dir': './outputs/experiment4_efficiency',
    'seed': 42,
}


# ============================================================
# Logger Setup
# ============================================================
def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger."""
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
# Efficiency Metrics
# ============================================================
class EfficiencyProfiler:
    """Profiler for measuring model efficiency metrics."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.cuda_available = torch.cuda.is_available() and device == 'cuda'
    
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Count by layer type
        layer_params = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                module_params = sum(p.numel() for p in module.parameters())
                if module_params > 0:
                    layer_type = type(module).__name__
                    if layer_type not in layer_params:
                        layer_params[layer_type] = 0
                    layer_params[layer_type] += module_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params,
            'by_layer_type': layer_params
        }
    
    def estimate_flops(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """
        Estimate FLOPs for a forward pass.
        
        Note: This is an approximation. For accurate FLOPs, use tools like
        fvcore or thop.
        """
        try:
            # Try using thop if available
            from thop import profile, clever_format
            dummy_input = torch.randn(1, *input_shape).to(self.device)
            model = model.to(self.device)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
            return {
                'flops': flops,
                'flops_formatted': flops_formatted,
                'method': 'thop'
            }
        except ImportError:
            pass
        
        # Simple estimation based on linear layers
        total_flops = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # FLOPs for linear: 2 * in_features * out_features (multiply-add)
                flops = 2 * module.in_features * module.out_features
                total_flops += flops
            elif isinstance(module, nn.Conv1d):
                # FLOPs for conv1d: 2 * kernel_size * in_channels * out_channels * output_length
                # Approximate output_length as input_length / stride
                flops = 2 * module.kernel_size[0] * module.in_channels * module.out_channels
                total_flops += flops * input_shape[-1]  # Approximate
            elif isinstance(module, nn.Conv2d):
                flops = 2 * module.kernel_size[0] * module.kernel_size[1] * \
                        module.in_channels * module.out_channels
                total_flops += flops
        
        # Format
        if total_flops >= 1e9:
            formatted = f"{total_flops/1e9:.2f} GFLOPs"
        elif total_flops >= 1e6:
            formatted = f"{total_flops/1e6:.2f} MFLOPs"
        else:
            formatted = f"{total_flops/1e3:.2f} KFLOPs"
        
        return {
            'flops': total_flops,
            'flops_formatted': formatted,
            'method': 'estimation'
        }
    
    def measure_inference_time(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """Measure inference time."""
        model = model.to(self.device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        if self.cuda_available:
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if self.cuda_available:
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = model(dummy_input)
                
                if self.cuda_available:
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append(end - start)
        
        times = np.array(times)
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'median_ms': np.median(times) * 1000,
            'samples_per_second': batch_size / np.mean(times),
            'batch_size': batch_size,
            'num_runs': num_runs
        }
    
    def measure_training_step_time(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_batches: int = 20,
        num_warmup: int = 5
    ) -> Dict[str, float]:
        """Measure training step time."""
        model = model.to(self.device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        times = []
        batch_iter = iter(dataloader)
        
        # Warmup
        for _ in range(num_warmup):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                batch = next(batch_iter)
            
            if len(batch) == 3:
                audio, labels, _ = batch
            else:
                audio, labels = batch[0], batch[1]
            
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(audio)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if self.cuda_available:
            torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(num_batches):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                batch = next(batch_iter)
            
            if len(batch) == 3:
                audio, labels, _ = batch
            else:
                audio, labels = batch[0], batch[1]
            
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            if self.cuda_available:
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            optimizer.zero_grad()
            outputs = model(audio)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if self.cuda_available:
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
        
        times = np.array(times)
        batch_size = dataloader.batch_size
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'samples_per_second': batch_size / np.mean(times),
            'batch_size': batch_size,
            'num_batches': num_batches
        }
    
    def measure_memory_usage(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """Measure GPU memory usage."""
        if not self.cuda_available:
            return {'error': 'CUDA not available'}
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        model = model.to(self.device)
        
        # Measure model memory
        torch.cuda.reset_peak_memory_stats()
        model_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Create input and measure forward pass memory
        dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
        
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        
        inference_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # Measure training memory
        torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        labels = torch.randint(0, 26, (batch_size,)).to(self.device)
        
        optimizer.zero_grad()
        outputs = model(dummy_input)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        training_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            'model_memory_mb': model_memory,
            'inference_peak_mb': inference_memory,
            'training_peak_mb': training_memory,
            'batch_size': batch_size
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
class Experiment4Runner:
    """Runner for Experiment 4: Computational Efficiency."""
    
    AVAILABLE_MODELS = ['mel_mlp', 'mfcc_mlp']
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            'experiment4',
            str(self.output_dir / 'experiment4.log')
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(config['seed'])
        
        self.profiler = EfficiencyProfiler(device=self.device)
        self.results = {}
        
        # Data
        self.train_loader = None
        self.num_speakers = 26
        
        # Input shape for the models (audio length in samples)
        self.input_length = int(config['sample_rate'] * config['max_length'])
    
    def setup_data(self):
        """Load dataset for training time measurement."""
        self.logger.info("Loading dataset for benchmarking...")
        
        train_loader, _, _ = create_speaker_dataloaders(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            sample_rate=self.config['sample_rate'],
            max_length=self.config['max_length'],
            train_split=0.7,
            val_split=0.15,
            seed=self.config['seed']
        )
        
        self.train_loader = train_loader
        
        try:
            if hasattr(train_loader.dataset, 'num_classes'):
                self.num_speakers = train_loader.dataset.num_classes
        except:
            pass
        
        self.logger.info(f"Dataset loaded: {len(train_loader.dataset)} samples")
    
    def profile_model(self, model_type: str) -> Dict[str, Any]:
        """Profile a single model."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Profiling: {model_type.upper()}")
        self.logger.info(f"{'='*60}")
        
        # Create model
        model = create_model(
            model_type=model_type,
            num_speakers=self.num_speakers,
            config=self.config,
            device=self.device
        )
        
        results = {
            'model_type': model_type,
            'device': self.device
        }
        
        # 1. Parameter count
        self.logger.info("\n1. Counting parameters...")
        param_stats = self.profiler.count_parameters(model)
        results['parameters'] = param_stats
        self.logger.info(f"   Total: {param_stats['total']:,}")
        self.logger.info(f"   Trainable: {param_stats['trainable']:,}")
        
        # 2. FLOPs estimation
        self.logger.info("\n2. Estimating FLOPs...")
        flops_stats = self.profiler.estimate_flops(model, (self.input_length,))
        results['flops'] = flops_stats
        self.logger.info(f"   FLOPs: {flops_stats['flops_formatted']}")
        
        # 3. Inference time
        self.logger.info("\n3. Measuring inference time...")
        
        # Single sample
        inference_single = self.profiler.measure_inference_time(
            model, (self.input_length,),
            num_runs=self.config['num_inference_runs'],
            batch_size=1
        )
        results['inference_single'] = inference_single
        self.logger.info(f"   Single sample: {inference_single['mean_ms']:.2f} ± {inference_single['std_ms']:.2f} ms")
        
        # Batch inference
        inference_batch = self.profiler.measure_inference_time(
            model, (self.input_length,),
            num_runs=self.config['num_inference_runs'],
            batch_size=self.config['batch_size']
        )
        results['inference_batch'] = inference_batch
        self.logger.info(f"   Batch ({self.config['batch_size']}): {inference_batch['mean_ms']:.2f} ± {inference_batch['std_ms']:.2f} ms")
        self.logger.info(f"   Throughput: {inference_batch['samples_per_second']:.1f} samples/sec")
        
        # 4. Training step time
        if self.train_loader is not None:
            self.logger.info("\n4. Measuring training step time...")
            
            # Recreate model (training modifies state)
            model = create_model(
                model_type=model_type,
                num_speakers=self.num_speakers,
                config=self.config,
                device=self.device
            )
            
            training_stats = self.profiler.measure_training_step_time(
                model, self.train_loader,
                num_batches=self.config['num_benchmark_batches'],
                num_warmup=self.config['num_warmup_batches']
            )
            results['training_step'] = training_stats
            self.logger.info(f"   Step time: {training_stats['mean_ms']:.2f} ± {training_stats['std_ms']:.2f} ms")
            self.logger.info(f"   Throughput: {training_stats['samples_per_second']:.1f} samples/sec")
        
        # 5. Memory usage
        if self.device == 'cuda':
            self.logger.info("\n5. Measuring memory usage...")
            
            # Recreate model
            model = create_model(
                model_type=model_type,
                num_speakers=self.num_speakers,
                config=self.config,
                device=self.device
            )
            
            memory_stats = self.profiler.measure_memory_usage(
                model, (self.input_length,),
                batch_size=self.config['batch_size']
            )
            results['memory'] = memory_stats
            self.logger.info(f"   Model: {memory_stats['model_memory_mb']:.1f} MB")
            self.logger.info(f"   Inference peak: {memory_stats['inference_peak_mb']:.1f} MB")
            self.logger.info(f"   Training peak: {memory_stats['training_peak_mb']:.1f} MB")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if self.device == 'cuda' else None
        gc.collect()
        
        return results
    
    def run_all(self) -> Dict[str, Any]:
        """Run efficiency profiling for all models."""
        self.logger.info("=" * 70)
        self.logger.info("EXPERIMENT 4: COMPUTATIONAL EFFICIENCY EVALUATION")
        self.logger.info("=" * 70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Input length: {self.input_length} samples ({self.config['max_length']}s)")
        self.logger.info(f"Batch size: {self.config['batch_size']}")
        
        # Setup data
        self.setup_data()
        
        # Profile each model
        for model_type in self.AVAILABLE_MODELS:
            try:
                results = self.profile_model(model_type)
                self.results[model_type] = results
            except Exception as e:
                self.logger.error(f"Error profiling {model_type}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.results[model_type] = {'error': str(e)}
        
        # Generate summary
        self._generate_summary()
        self._save_results()
        
        return self.results
    
    def run_single_model(self, model_type: str) -> Dict[str, Any]:
        """Profile a single model."""
        if self.train_loader is None:
            self.setup_data()
        
        results = self.profile_model(model_type)
        self.results[model_type] = results
        
        self._generate_summary()
        self._save_results()
        
        return results
    
    def _generate_summary(self):
        """Generate summary table."""
        self.logger.info("\n" + "=" * 90)
        self.logger.info("COMPUTATIONAL EFFICIENCY SUMMARY")
        self.logger.info("=" * 90)
        
        # Header
        self.logger.info(
            f"{'Model':<12} {'Params':>12} {'FLOPs':>12} "
            f"{'Infer(ms)':>12} {'Train(ms)':>12} {'Memory(MB)':>12}"
        )
        self.logger.info("-" * 90)
        
        for model_name, results in self.results.items():
            if 'error' in results:
                self.logger.info(f"{model_name:<12} ERROR")
                continue
            
            params = results['parameters']['total']
            flops = results['flops']['flops_formatted']
            
            infer_ms = results.get('inference_batch', {}).get('mean_ms', 0)
            train_ms = results.get('training_step', {}).get('mean_ms', 0)
            memory_mb = results.get('memory', {}).get('training_peak_mb', 0)
            
            # Format params
            if params >= 1e6:
                params_str = f"{params/1e6:.2f}M"
            elif params >= 1e3:
                params_str = f"{params/1e3:.1f}K"
            else:
                params_str = str(params)
            
            self.logger.info(
                f"{model_name:<12} {params_str:>12} {flops:>12} "
                f"{infer_ms:>11.2f} {train_ms:>11.2f} {memory_mb:>11.1f}"
            )
        
        self.logger.info("=" * 90)
        self.logger.info("Note: Infer/Train times are per batch, Memory is peak during training")
    
    def _save_results(self):
        """Save results to JSON."""
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
        
        output = {
            'experiment': 'Experiment 4: Computational Efficiency',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'device': self.device,
                'batch_size': self.config['batch_size'],
                'input_length': self.input_length,
                'audio_duration': self.config['max_length']
            },
            'results': convert(self.results)
        }
        
        results_path = self.output_dir / 'efficiency_results.json'
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"\nResults saved to: {results_path}")


# ============================================================
# Quick Benchmark (without dataset)
# ============================================================
def quick_benchmark(config: Dict[str, Any], logger: logging.Logger):
    """Run a quick benchmark without loading the full dataset."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    profiler = EfficiencyProfiler(device=device)
    input_length = int(config['sample_rate'] * config['max_length'])
    
    logger.info("=" * 70)
    logger.info("QUICK BENCHMARK (No Dataset Required)")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    
    results = {}
    
    for model_type in ['mel_mlp', 'mfcc_mlp']:
        logger.info(f"\n--- {model_type.upper()} ---")
        
        model = create_model(
            model_type=model_type,
            num_speakers=26,
            config=config,
            device=device
        )
        
        # Parameters
        params = profiler.count_parameters(model)
        logger.info(f"Parameters: {params['total']:,}")
        
        # FLOPs
        flops = profiler.estimate_flops(model, (input_length,))
        logger.info(f"FLOPs: {flops['flops_formatted']}")
        
        # Inference time
        inference = profiler.measure_inference_time(
            model, (input_length,), num_runs=50, batch_size=32
        )
        logger.info(f"Inference (batch=32): {inference['mean_ms']:.2f} ms")
        logger.info(f"Throughput: {inference['samples_per_second']:.1f} samples/sec")
        
        # Memory (if CUDA)
        if device == 'cuda':
            memory = profiler.measure_memory_usage(model, (input_length,), batch_size=32)
            logger.info(f"Memory (training): {memory['training_peak_mb']:.1f} MB")
        
        results[model_type] = {
            'parameters': params['total'],
            'flops': flops['flops'],
            'inference_ms': inference['mean_ms'],
            'throughput': inference['samples_per_second']
        }
        
        del model
        torch.cuda.empty_cache() if device == 'cuda' else None
    
    return results


# ============================================================
# Main Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Experiment 4: Computational Efficiency Evaluation'
    )
    
    parser.add_argument(
        '--all', action='store_true',
        help='Profile all models'
    )
    parser.add_argument(
        '--model', type=str, choices=['mel_mlp', 'mfcc_mlp'],
        help='Profile a specific model'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Quick benchmark without loading dataset'
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
        '--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
        help='Batch size for benchmarking'
    )
    
    args = parser.parse_args()
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    config['batch_size'] = args.batch_size
    
    # Setup logger
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('experiment4', str(output_dir / 'experiment4.log'))
    
    if args.benchmark:
        quick_benchmark(config, logger)
    elif args.all:
        runner = Experiment4Runner(config)
        runner.run_all()
    elif args.model:
        runner = Experiment4Runner(config)
        runner.run_single_model(args.model)
    else:
        print("Please specify --all, --model <name>, or --benchmark")
        print("Example: python run_experiment4_efficiency.py --all")
        print("Example: python run_experiment4_efficiency.py --benchmark")
        parser.print_help()


if __name__ == "__main__":
    main()