#!/usr/bin/env python3
"""
Experiment 2: Noise Robustness Testing for All Models

Tests both baseline models (Mel-MLP, MFCC-MLP) and C-HiLAP under various noise conditions.

Usage:
    python run_experiment2_noise_robustness.py --all
    python run_experiment2_noise_robustness.py --chaotic_only
    python run_experiment2_noise_robustness.py --baseline_only
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# =============================================================================
# 1. Project Path Setup
# =============================================================================
def setup_project_path():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    paths = [
        str(project_root),
        str(project_root / 'core'),
        str(project_root / 'models'), 
        str(project_root / 'features'),
        str(project_root / 'data'),
        str(project_root / 'utils'),
        str(project_root / 'experiments')
    ]
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    return project_root

PROJECT_ROOT = setup_project_path()

# Import modules
try:
    from data.dataset_loader import create_speaker_dataloaders
    from utils.reproducibility import set_seed
    print("Core modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Import noise augmentor
try:
    from data.noise_augmentation import NoiseAugmentor
except ImportError:
    try:
        from utils.noise_augmentation import NoiseAugmentor
    except ImportError:
        # Inline implementation if not found
        class NoiseAugmentor:
            def __init__(self, sample_rate=16000, seed=42):
                self.sample_rate = sample_rate
                np.random.seed(seed)
            
            def add_noise(self, audio, noise_type, snr_db):
                if isinstance(audio, torch.Tensor):
                    audio = audio.numpy()
                
                signal_power = np.mean(audio ** 2, axis=-1, keepdims=True)
                snr_linear = 10 ** (snr_db / 10)
                noise_power = signal_power / snr_linear
                
                if noise_type == 'gaussian':
                    noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
                elif noise_type == 'babble':
                    noise = np.random.randn(*audio.shape) * np.sqrt(noise_power) * 0.8
                    noise = np.convolve(noise.flatten(), np.ones(10)/10, mode='same').reshape(audio.shape)
                elif noise_type in ['cafe', 'street']:
                    noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
                else:
                    noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
                
                return audio + noise

# =============================================================================
# 2. Model Loading Functions
# =============================================================================

def load_baseline_model(ckpt_path, model_name, device):
    """Load baseline model (Mel-MLP or MFCC-MLP)."""
    from models.hybrid_models import TraditionalMLPBaseline
    
    print(f"Loading baseline model: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get('config', {})
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer feature type from model name
    if 'mfcc' in model_name.lower():
        feature_type = 'mfcc'
    else:
        feature_type = 'mel'
    
    # Auto-detect speaker count from state dict
    num_speakers = 26  # default
    for k, v in state_dict.items():
        if 'classifier' in k and 'weight' in k and len(v.shape) == 2:
            num_speakers = v.shape[0]
    
    print(f"  Feature type: {feature_type}, Speakers: {num_speakers}")
    
    model = TraditionalMLPBaseline(
        feature_type=feature_type,
        n_mels=config.get('n_mels', 80),
        n_mfcc=config.get('n_mfcc', 40),
        sample_rate=config.get('sample_rate', 16000),
        hidden_dims=config.get('hidden_dims', [256, 128, 64]),
        num_speakers=num_speakers,
        device=device
    )
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model


def load_chaotic_model(ckpt_path, device, config=None):
    """Load C-HiLAP (Chaotic Speaker Recognition) model."""
    from models.chaotic_network import ChaoticSpeakerRecognitionNetwork
    
    print(f"Loading C-HiLAP model: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Get config from checkpoint or use provided config
    if config is None:
        config = checkpoint.get('config', {})
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # AUTO-DETECT num_speakers from classifier weight
    num_speakers = config.get('num_speakers', 251)
    for key, value in state_dict.items():
        if 'classifier.weight' in key and len(value.shape) == 2:
            num_speakers = value.shape[0]
            print(f"  Auto-detected num_speakers from checkpoint: {num_speakers}")
            break
            
    # Extract model parameters from config
    model_params = {
        'sample_rate': config.get('sample_rate', 16000),
        'frame_length': config.get('frame_length', 400),
        'hop_length': config.get('hop_length', 160),
        'embedding_dim': config.get('embedding_dim', 10),
        'delay_method': config.get('delay_method', 'autocorr'),
        'mlsa_scales': config.get('mlsa_scales', 5),
        'rqa_radius_ratio': config.get('rqa_radius_ratio', 0.1),
        'chaotic_system': config.get('chaotic_system', 'lorenz'),
        'evolution_time': config.get('evolution_time', 0.5),
        'time_step': config.get('time_step', 0.01),
        'pooling_type': config.get('pooling_type', 'comprehensive'),
        'speaker_embedding_dim': config.get('speaker_embedding_dim', 256),
        'embedding_hidden_dims': config.get('embedding_hidden_dims', [512, 256, 128]),
        'num_speakers': config.get('num_speakers', num_speakers),
        'classifier_type': config.get('classifier_type', 'cosine'),
        'device': device
    }
    
    print(f"  Chaotic system: {model_params['chaotic_system']}")
    print(f"  Speakers: {model_params['num_speakers']}")
    print(f"  Embedding dim: {model_params['embedding_dim']}")
    
    model = ChaoticSpeakerRecognitionNetwork(**model_params)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print("  Loaded with strict=True")
    except RuntimeError as e:
        print(f"  Strict loading failed: {e}")
        model.load_state_dict(state_dict, strict=False)
        print("  Loaded with strict=False")
    
    model.to(device)
    model.eval()
    return model


# =============================================================================
# 3. Evaluation Functions
# =============================================================================

def evaluate_model(model, data_loader, device, model_type='baseline'):
    """Evaluate model accuracy on given data loader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) >= 2:
                audio, labels = batch[0], batch[1]
            else:
                continue
            
            audio = audio.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(audio)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output', None))
                if logits is None:
                    logits = list(outputs.values())[0]
            else:
                logits = outputs
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def evaluate_with_noise(model, data_loader, noise_augmentor, noise_type, snr_db, device, model_type='baseline'):
    """Evaluate model with added noise."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) >= 2:
                audio, labels = batch[0], batch[1]
            else:
                continue
            
            # Add noise
            audio_np = audio.numpy()
            noisy_audio = noise_augmentor.add_noise(audio_np, noise_type, snr_db)
            audio = torch.from_numpy(noisy_audio).float()
            
            audio = audio.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(audio)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output', None))
                if logits is None:
                    logits = list(outputs.values())[0]
            else:
                logits = outputs
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


# =============================================================================
# 4. Main Experiment
# =============================================================================

def run_experiment(args):
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device
    set_seed(args.seed)
    
    print("=" * 70)
    print("EXPERIMENT 2: NOISE ROBUSTNESS TESTING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {output_dir}")
    
    # Load data
    print("\nLoading dataset...")
    _, _, test_loader = create_speaker_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_split=0.7,
        val_split=0.15,
        max_length=3.0,
        sample_rate=16000,
        seed=args.seed
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize noise augmentor
    noise_augmentor = NoiseAugmentor(sample_rate=16000, seed=args.seed)
    
    # Parse noise types and SNR levels
    noise_types = args.noise_types.split(',')
    snr_levels = [int(x) for x in args.snr.split(',')]
    
    print(f"\nNoise types: {noise_types}")
    print(f"SNR levels: {snr_levels}")
    
    # Results storage
    all_results = {}
    
    # =================================
    # Test Baseline Models
    # =================================
    if not args.chaotic_only:
        print("\n" + "=" * 70)
        print("TESTING BASELINE MODELS")
        print("=" * 70)
        
        baseline_dir = Path(args.baseline_checkpoint_dir)
        baseline_models = {}
        
        # Find baseline checkpoints
        for name in ['mel_mlp', 'mfcc_mlp']:
            matches = list(baseline_dir.rglob(f"*{name}*/**/best_model.pth"))
            if matches:
                best = sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                baseline_models[name] = str(best)
                print(f"Found {name}: {best}")
        
        for model_name, ckpt_path in baseline_models.items():
            print(f"\n--- Testing {model_name} ---")
            try:
                model = load_baseline_model(ckpt_path, model_name, device)
                model_results = {'model_type': 'baseline', 'feature_type': model_name}
                
                # Clean accuracy
                clean_acc = evaluate_model(model, test_loader, device, 'baseline')
                model_results['clean'] = {'accuracy': clean_acc}
                print(f"  Clean accuracy: {clean_acc*100:.2f}%")
                
                # Noise tests
                for noise_type in noise_types:
                    model_results[noise_type] = {}
                    for snr in snr_levels:
                        acc = evaluate_with_noise(model, test_loader, noise_augmentor, 
                                                  noise_type, snr, device, 'baseline')
                        model_results[noise_type][snr] = {'accuracy': acc}
                        drop = (clean_acc - acc) * 100
                        print(f"  {noise_type} SNR={snr}dB: {acc*100:.2f}% (drop: {drop:+.2f}%)")
                
                all_results[model_name] = model_results
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # =================================
    # Test C-HiLAP Model
    # =================================
    if not args.baseline_only and args.chaotic_checkpoint:
        print("\n" + "=" * 70)
        print("TESTING C-HiLAP MODEL")
        print("=" * 70)
        
        try:
            # Load C-HiLAP config if provided
            chaotic_config = None
            if args.chaotic_config:
                with open(args.chaotic_config, 'r') as f:
                    chaotic_config = json.load(f)
            
            model = load_chaotic_model(args.chaotic_checkpoint, device, chaotic_config)
            model_results = {'model_type': 'chaotic', 'system': 'lorenz'}
            
            # Clean accuracy
            clean_acc = evaluate_model(model, test_loader, device, 'chaotic')
            model_results['clean'] = {'accuracy': clean_acc}
            print(f"  Clean accuracy: {clean_acc*100:.2f}%")
            
            # Noise tests
            for noise_type in noise_types:
                model_results[noise_type] = {}
                for snr in snr_levels:
                    acc = evaluate_with_noise(model, test_loader, noise_augmentor,
                                              noise_type, snr, device, 'chaotic')
                    model_results[noise_type][snr] = {'accuracy': acc}
                    drop = (clean_acc - acc) * 100
                    retention = (acc / clean_acc * 100) if clean_acc > 0 else 0
                    print(f"  {noise_type} SNR={snr}dB: {acc*100:.2f}% (drop: {drop:+.2f}%, retention: {retention:.1f}%)")
            
            all_results['c_hilap'] = model_results
            
        except Exception as e:
            print(f"  ERROR loading C-HiLAP: {e}")
            import traceback
            traceback.print_exc()
    
    # =================================
    # Generate Summary
    # =================================
    print("\n" + "=" * 70)
    print("NOISE ROBUSTNESS SUMMARY")
    print("=" * 70)
    
    # Header
    header = f"{'Model':<15} {'Clean':>8}"
    for noise_type in noise_types:
        header += f" {noise_type[:6]+'@20dB':>12}"
    print(header)
    print("-" * len(header))
    
    # Results
    for model_name, results in all_results.items():
        clean_acc = results.get('clean', {}).get('accuracy', 0) * 100
        line = f"{model_name:<15} {clean_acc:>7.2f}%"
        
        for noise_type in noise_types:
            if noise_type in results and 20 in results[noise_type]:
                acc = results[noise_type][20].get('accuracy', 0) * 100
                line += f" {acc:>11.2f}%"
            else:
                line += f" {'N/A':>12}"
        print(line)
    
    # Save results
    results_file = output_dir / 'noise_robustness_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate comparison table
    if 'c_hilap' in all_results and len(all_results) > 1:
        print("\n" + "=" * 70)
        print("RETENTION RATE COMPARISON (% of clean accuracy retained)")
        print("=" * 70)
        
        header = f"{'Noise@SNR':<15}"
        for model_name in all_results.keys():
            header += f" {model_name:>12}"
        print(header)
        print("-" * len(header))
        
        for noise_type in noise_types:
            for snr in snr_levels:
                line = f"{noise_type}@{snr}dB"
                line = f"{line:<15}"
                for model_name, results in all_results.items():
                    clean_acc = results.get('clean', {}).get('accuracy', 0)
                    if noise_type in results and snr in results[noise_type]:
                        noisy_acc = results[noise_type][snr].get('accuracy', 0)
                        retention = (noisy_acc / clean_acc * 100) if clean_acc > 0 else 0
                        line += f" {retention:>11.1f}%"
                    else:
                        line += f" {'N/A':>12}"
                print(line)
    
    return all_results


# =============================================================================
# 5. Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Noise Robustness Testing')
    
    # Data
    parser.add_argument('--data_dir', 
                        default='/scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2')
    
    # Checkpoints
    parser.add_argument('--baseline_checkpoint_dir', 
                        default='./outputs/experiment1_unified')
    parser.add_argument('--chaotic_checkpoint',
                        default=None,
                        help='Path to C-HiLAP checkpoint (.pkl or .pth)')
    parser.add_argument('--chaotic_config',
                        default=None,
                        help='Path to C-HiLAP config JSON (optional)')
    
    # Output
    parser.add_argument('--output_dir', 
                        default='./outputs/experiment2_noise_robustness')
    
    # Noise settings
    parser.add_argument('--noise_types', default='gaussian,babble,cafe,street')
    parser.add_argument('--snr', default='20,15,10,5,0')
    
    # Model selection
    parser.add_argument('--all', action='store_true', help='Test all models')
    parser.add_argument('--chaotic_only', action='store_true', help='Test only C-HiLAP')
    parser.add_argument('--baseline_only', action='store_true', help='Test only baselines')
    
    # Other
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Run experiment
    run_experiment(args)