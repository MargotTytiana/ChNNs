#!/usr/bin/env python3
"""
Experiment 2: Noise Robustness via Classification Accuracy

This script tests noise robustness using CLASSIFICATION ACCURACY.
Updated to support external configuration files via --chaotic_config.

Usage:
    python run_experiment2_classification.py \
        --baseline_checkpoint_dir ./outputs/experiment1_unified \
        --chaotic_checkpoint /path/to/chaotic/checkpoint.pkl \
        --chaotic_config /tmp/chaotic_26spk_config.json \
        --output_dir ./outputs/experiment2_classification
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import torch
from pathlib import Path

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
    print("✓ Core modules imported")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Import noise augmentor
try:
    from data.noise_augmentation import NoiseAugmentor
    print("✓ NoiseAugmentor imported")
except ImportError:
    try:
        from utils.noise_augmentation import NoiseAugmentor
        print("✓ NoiseAugmentor imported from utils")
    except ImportError:
        print("⚠ Using built-in NoiseAugmentor")
        class NoiseAugmentor:
            def __init__(self, sample_rate=16000, seed=42):
                self.sample_rate = sample_rate
                np.random.seed(seed)
            
            def add_noise(self, audio, noise_type, snr_db):
                if isinstance(audio, torch.Tensor):
                    audio = audio.numpy()
                
                signal_power = np.mean(audio ** 2, axis=-1, keepdims=True) + 1e-10
                snr_linear = 10 ** (snr_db / 10)
                noise_power = signal_power / snr_linear
                
                if noise_type == 'gaussian':
                    noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
                elif noise_type == 'babble':
                    noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
                    for i in range(audio.shape[0]):
                        noise[i] = np.convolve(noise[i], np.ones(5)/5, mode='same')
                else:
                    noise = np.random.randn(*audio.shape) * np.sqrt(noise_power)
                
                return (audio + noise).astype(np.float32)


# =============================================================================
# 2. Model Loading
# =============================================================================

def load_baseline_model(ckpt_path, model_name, device):
    """Load baseline model (Mel-MLP or MFCC-MLP) - KEEP CLASSIFIER."""
    from models.hybrid_models import TraditionalMLPBaseline
    
    print(f"  Loading: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get('config', {})
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer feature type
    feature_type = 'mfcc' if 'mfcc' in model_name.lower() else 'mel'
    
    # Detect speaker count
    num_speakers = 26
    for k, v in state_dict.items():
        if 'classifier' in k and 'weight' in k and len(v.shape) == 2:
            num_speakers = v.shape[0]
    
    print(f"    Feature: {feature_type}, Speakers: {num_speakers}")
    
    model = TraditionalMLPBaseline(
        feature_type=feature_type,
        n_mels=config.get('n_mels', 80),
        n_mfcc=config.get('n_mfcc', 40),
        sample_rate=config.get('sample_rate', 16000),
        hidden_dims=config.get('hidden_dims', [256, 128, 64]),
        num_speakers=num_speakers,
        device=device
    )
    
    model.load_state_dict(state_dict, strict=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print("\n" + "!"*50)
    print("[DEBUG] Weight Loading Diagnosis:")
    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} layers failed to load (using random init)!")
        print(f"  First 5 missing: {missing[:5]}")
        # 如果输入层（通常叫 feature_projection 或 input_layer）在 missing 列表里，说明维度不对
    else:
        print("  SUCCESS: All layers loaded matching weights.")
    print("!"*50 + "\n")
    
    model.to(device)
    model.eval()
    
    return model, num_speakers


def load_chaotic_model(ckpt_path, device, config_path=None):
    """
    Load C-HiLAP model.
    Arguments:
        ckpt_path: Path to the .pt or .pkl checkpoint.
        device: 'cuda' or 'cpu'.
        config_path: Optional path to a JSON config file to override checkpoint settings.
    """
    from models.chaotic_network import ChaoticSpeakerRecognitionNetwork
    
    print(f"  Loading: {ckpt_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except:
        with open(ckpt_path, 'rb') as f:
            checkpoint = pickle.load(f)
    
    # Base config from checkpoint
    config = checkpoint.get('config', {})
    
    # ---------------------------------------------------------
    # Override with external JSON config if provided (KEY FIX)
    # ---------------------------------------------------------
    if config_path and os.path.exists(config_path):
        print(f"  Overriding config with: {config_path}")
        with open(config_path, 'r') as f:
            json_config = json.load(f)
            # Update the config dictionary with values from JSON
            config.update(json_config)
    
    # Extract parameters with defaults
    num_speakers = config.get('num_speakers', 26)
    
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
        'num_speakers': num_speakers,
        'classifier_type': config.get('classifier_type', 'cosine'),
        'device': device,
        # === 请添加以下 3 行，强制覆盖默认值 ===
        'include_mfcc': False,
        'use_raw': False,
        'use_differentiable_raw': False  # 这一行至关重要！
    }
    
    print(f"    System: {model_params['chaotic_system']}, Speakers: {num_speakers}")
    print(f"    Classifier: {model_params['classifier_type']}")
    
    # Initialize model
    model = ChaoticSpeakerRecognitionNetwork(**model_params)
    
    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    return model, num_speakers


# =============================================================================
# 3. Evaluation Functions
# =============================================================================

def evaluate_accuracy(model, data_loader, device):
    """Evaluate classification accuracy."""
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
            
            outputs = model(audio)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))
            else:
                logits = outputs
            
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total if total > 0 else 0.0


def evaluate_with_noise(model, data_loader, noise_aug, noise_type, snr_db, device):
    """Evaluate classification accuracy under noise."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) >= 2:
                audio, labels = batch[0], batch[1]
            else:
                continue
            
            audio_np = audio.numpy()
            noisy = noise_aug.add_noise(audio_np, noise_type, snr_db)
            audio = torch.from_numpy(noisy).float()
            
            audio = audio.to(device)
            labels = labels.to(device)
            
            outputs = model(audio)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))
            else:
                logits = outputs
            
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total if total > 0 else 0.0


# =============================================================================
# 4. Main Experiment
# =============================================================================

def run_experiment(args):
    print("=" * 70)
    print("EXPERIMENT 2: NOISE ROBUSTNESS (CLASSIFICATION ACCURACY)")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")
    
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
    
    # Noise setup
    noise_aug = NoiseAugmentor(sample_rate=16000, seed=args.seed)
    noise_types = args.noise_types.split(',')
    snr_levels = [int(x) for x in args.snr.split(',')]
    
    results = {}
    
    # =================================
    # Test Baseline Models
    # =================================
    if not args.chaotic_only:
        print("\n" + "-" * 70)
        print("BASELINE MODELS")
        print("-" * 70)
        
        baseline_dir = Path(args.baseline_checkpoint_dir)
        
        for model_name in ['mel_mlp', 'mfcc_mlp']:
            matches = list(baseline_dir.rglob(f"*{model_name}*/**/best_model.pth"))
            if not matches:
                print(f"  {model_name}: checkpoint not found")
                continue
            
            ckpt_path = sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            print(f"\n[{model_name.upper()}]")
            
            try:
                model, num_spk = load_baseline_model(str(ckpt_path), model_name, device)
                
                model_results = {'clean': {}, 'noise': {}}
                
                clean_acc = evaluate_accuracy(model, test_loader, device)
                model_results['clean']['accuracy'] = clean_acc
                print(f"  Clean: {clean_acc*100:.2f}%")
                
                for noise_type in noise_types:
                    model_results['noise'][noise_type] = {}
                    for snr in snr_levels:
                        acc = evaluate_with_noise(model, test_loader, noise_aug, noise_type, snr, device)
                        model_results['noise'][noise_type][snr] = acc
                        retention = (acc / clean_acc * 100) if clean_acc > 0 else 0
                        print(f"  {noise_type}@{snr}dB: {acc*100:.2f}% (retention: {retention:.1f}%)")
                
                results[model_name] = model_results
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # =================================
    # Test C-HiLAP Model
    # =================================
    if args.chaotic_checkpoint and not args.baseline_only:
        print("\n" + "-" * 70)
        print("C-HiLAP MODEL")
        print("-" * 70)
        
        try:
            # Pass the chaotic config path to the loader
            model, num_spk = load_chaotic_model(
                args.chaotic_checkpoint, 
                device, 
                config_path=args.chaotic_config
            )
            
            model_results = {'clean': {}, 'noise': {}}
            
            clean_acc = evaluate_accuracy(model, test_loader, device)
            model_results['clean']['accuracy'] = clean_acc
            print(f"  Clean: {clean_acc*100:.2f}%")
            
            for noise_type in noise_types:
                model_results['noise'][noise_type] = {}
                for snr in snr_levels:
                    acc = evaluate_with_noise(model, test_loader, noise_aug, noise_type, snr, device)
                    model_results['noise'][noise_type][snr] = acc
                    retention = (acc / clean_acc * 100) if clean_acc > 0 else 0
                    print(f"  {noise_type}@{snr}dB: {acc*100:.2f}% (retention: {retention:.1f}%)")
            
            results['c_hilap'] = model_results
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # =================================
    # Summary
    # =================================
    print("\n" + "=" * 70)
    print("SUMMARY: CLASSIFICATION ACCURACY (%)")
    print("=" * 70)
    
    header = f"{'Model':<12} {'Clean':>8}"
    for nt in noise_types[:4]:
        header += f" {nt[:4]+'@20':>8}"
    print(header)
    print("-" * len(header))
    
    for model_name, res in results.items():
        clean = res['clean']['accuracy'] * 100
        line = f"{model_name:<12} {clean:>7.2f}%"
        for nt in noise_types[:4]:
            if nt in res['noise'] and 20 in res['noise'][nt]:
                acc = res['noise'][nt][20] * 100
                line += f" {acc:>7.2f}%"
        print(line)
    
    # Save
    results_file = output_dir / 'classification_accuracy_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
                        default='/scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2')
    parser.add_argument('--baseline_checkpoint_dir', default='./outputs/experiment1_unified')
    parser.add_argument('--chaotic_checkpoint', default=None)
    parser.add_argument('--chaotic_config', default=None, help="Path to JSON config file for chaotic model") # Added argument
    parser.add_argument('--output_dir', default='./outputs/experiment2_classification')
    parser.add_argument('--noise_types', default='gaussian,babble,cafe,street')
    parser.add_argument('--snr', default='20,15,10,5,0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chaotic_only', action='store_true')
    parser.add_argument('--baseline_only', action='store_true')
    
    run_experiment(parser.parse_args())