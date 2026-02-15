#!/usr/bin/env python3
"""
Quick Short Utterance Evaluation Script

Evaluates a single trained model on short utterances without full experiment setup.
Useful for testing existing models quickly.

Usage:
    python quick_short_utterance_test.py --model path/to/model.pth --data ./data
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

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

        ]

        

        for path in paths_to_add:

            if Path(path).exists() and path not in sys.path:

                sys.path.insert(0, path)

        

        return project_root, False


# Setup imports

PROJECT_ROOT, USING_IMPORT_MANAGER = setup_module_imports()
from short_utterance_transform import ShortUtteranceTransform, create_short_utterance_dataset_wrapper
from dataset_loader import create_chaotic_speaker_dataset


def evaluate_model_at_duration(
    model,
    test_dataset,
    duration: float,
    sample_rate: int = 16000,
    batch_size: int = 32,
    device: str = 'cuda',
    num_crops: int = 3
):
    """
    Evaluate model at a specific duration with multiple random crops.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        duration: Target duration in seconds
        sample_rate: Sample rate
        batch_size: Batch size
        device: Device to use
        num_crops: Number of random crops per sample
        
    Returns:
        Average accuracy across crops
    """
    # Force CPU for stability
    device = 'cpu'
    
    # Move model to device and ensure eval mode
    model = model.cpu()
    
    # Force all submodules to CPU
    for module in model.modules():
        module.to('cpu')
    
    model.eval()
    
    # Verify model is on CPU
    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  Model in eval mode: {not model.training}")
    
    accuracies = []
    
    for crop_idx in range(num_crops):
        # # Create wrapped dataset with random crop
        # wrapped_dataset = create_short_utterance_dataset_wrapper(
        #     test_dataset,
        #     target_duration=duration,
        #     sample_rate=sample_rate,
        #     crop_mode='random'
        # )

        def collate_fn(batch):
            audios, labels = zip(*batch)
            target_length = 48000  # 3秒 @ 16kHz
            fixed_audios = []
            for audio in audios:
                if audio.shape[0] > target_length:
                    audio = audio[:target_length]
                elif audio.shape[0] < target_length:
                    padding = torch.zeros(target_length - audio.shape[0])
                    audio = torch.cat([audio, padding])
                fixed_audios.append(audio)
            return torch.stack(fixed_audios), torch.tensor(labels)

        def make_collate_fn(duration, sample_rate=16000):
            target_length = int(duration * sample_rate)
            
            def collate_fn(batch):
                audios, labels = zip(*batch)
                fixed_audios = []
                for audio in audios:
                    if audio.shape[0] > target_length:
                        audio = audio[:target_length]
                    elif audio.shape[0] < target_length:
                        padding = torch.zeros(target_length - audio.shape[0])
                        audio = torch.cat([audio, padding])
                    fixed_audios.append(audio)
                return torch.stack(fixed_audios), torch.tensor(labels)
            
            return collate_fn
        
        loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            # collate_fn=collate_fn  # 添加
            collate_fn=make_collate_fn(duration)  # 传入当前的duration
        )
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for audio, labels in loader:
                # Ensure data is on CPU
                audio = audio.cpu()
                labels = labels.cpu()
                
                try:
                    # Handle different model output formats
                    outputs = model(audio, labels=None, return_intermediates=False)
                    
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Take logits if tuple
                    
                    _, predicted = torch.max(outputs, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                except Exception as e:
                    print(f"  ERROR during inference: {e}")
                    print(f"  Audio shape: {audio.shape}")
                    print(f"  Audio device: {audio.device}")
                    raise
        
        accuracy = 100.0 * correct / total
        accuracies.append(accuracy)
        
        print(f"  Crop {crop_idx + 1}/{num_crops}: {accuracy:.2f}%")
    
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    return avg_accuracy, std_accuracy


def quick_test(
    model_path: str,
    data_dir: str,
    model_type: str = 'chaotic',
    durations: list = None,
    num_speakers: int = 50,
    batch_size: int = 32,
    device: str = 'cuda',
    num_crops: int = 3,
    output_dir: str = './quick_test_results'
):
    """
    Quick test of a model on short utterances.
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to dataset
        model_type: Type of model ('chaotic' or 'baseline')
        durations: List of durations to test
        num_speakers: Number of speakers
        batch_size: Batch size
        device: Device to use
        num_crops: Number of random crops per sample
        output_dir: Output directory
    """
    if durations is None:
        durations = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("QUICK SHORT UTTERANCE TEST")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Durations: {durations}")
    print(f"Device: {device}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = create_chaotic_speaker_dataset(
        dataset_path=data_dir,
        min_samples_per_speaker=2,
        max_samples_per_speaker=30,
        target_num_speakers=num_speakers
    )
    
    # Create test split
    splits = dataset.create_data_splits(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    test_dataset = splits['test'].create_pytorch_dataset()
    print(f"Test dataset: {len(test_dataset)} samples")
    print()
    
    # Load model
    print("Loading model...")
    
    # Load checkpoint first to get config
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Print checkpoint info for debugging
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'train_accuracy' in checkpoint:
        print(f"Training accuracy: {checkpoint.get('train_accuracy', 'N/A')}")
    if 'val_accuracy' in checkpoint:
        print(f"Validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
    
    if model_type == 'chaotic':
        from chaotic_network import ChaoticSpeakerRecognitionNetwork
        
        # Extract config from checkpoint if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"Using config from checkpoint")
        else:
            # Use default config matching base_chaotic.yaml
            config = {
                'phase_space': {'embedding_dim': 10},
                'speaker_embedding': {'dim': 256, 'hidden_dims': [512, 256, 128]},
                'classifier': {'type': 'cosine', 'temperature': 30.0},
            }
            print("Using default config matching base_chaotic.yaml")
        
        # Create model on CPU with correct parameters
        print(f"Creating model with num_speakers={num_speakers}")
        model = ChaoticSpeakerRecognitionNetwork(
            num_speakers=num_speakers,
            embedding_dim=config.get('phase_space', {}).get('embedding_dim', 10),
            speaker_embedding_dim=config.get('speaker_embedding', {}).get('dim', 256),
            embedding_hidden_dims=config.get('speaker_embedding', {}).get('hidden_dims', [512, 256, 128]),
            classifier_type=config.get('classifier', {}).get('type', 'cosine'),
            device='cpu'  # Force CPU
        )
    else:
        from hybrid_models import MelMLP
        model = MelMLP(num_speakers=num_speakers)
    
    # Load state dict - remove strict=False to see what's missing
    print("Loading weights...")
    try:
        if 'model_state_dict' in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                print(f"WARNING: Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"WARNING: Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
            print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            if missing_keys:
                print(f"WARNING: Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"WARNING: Unexpected keys: {unexpected_keys[:5]}...")
    except Exception as e:
        print(f"ERROR loading weights: {e}")
        print("Attempting to continue anyway...")
    
    # Force CPU and eval mode
    model = model.cpu()
    
    # Force all submodules to CPU
    for module in model.modules():
        module.to('cpu')
    
    model.eval()
    
    # Verify
    print(f"Model ready - device: {next(model.parameters()).device}, training: {model.training}")
    
    # Test a forward pass
    print("Testing forward pass...")
    print("\n=== Model Configuration Check ===")
    print(f"use_differentiable_features: {getattr(model, 'use_differentiable_features', 'NOT SET')}")
    print(f"Has diff_chaos_features: {hasattr(model, 'diff_chaos_features')}")
    print(f"Has diff_feat_projection: {hasattr(model, 'diff_feat_projection')}")

    try:
        with torch.no_grad():
            test_input = torch.randn(1, 48000).cpu()
            output = model(test_input, labels=None, return_intermediates=True)
            
            # 处理返回值
            if isinstance(output, tuple):
                test_output, intermediates = output
            else:
                test_output = output
                intermediates = {}
            
            if isinstance(test_output, tuple):
                test_output = test_output[0]
                
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {test_output.shape}")
            print(f"  Expected output shape: (1, {num_speakers})")
            
            if test_output.shape[1] != num_speakers:
                print(f"  WARNING: Output dimension mismatch!")
            
            print("\nIntermediate features:")
            for key, value in intermediates.items():
                if torch.is_tensor(value):
                    print(f"  {key}: shape={value.shape}, mean={value.mean():.4f}, std={value.std():.4f}")
            
            if isinstance(test_output, tuple):
                test_output = test_output[0]
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {test_output.shape}")
            print(f"  Expected output shape: (1, {num_speakers})")
            if test_output.shape[1] != num_speakers:
                print(f"  WARNING: Output dimension mismatch!")
    except Exception as e:
        print(f"  ERROR in forward pass: {e}")
        raise
    
    print()
    
    # Test at each duration
    print("Testing at different durations...")
    print("-"*60)
    
    results = {}
    
    for duration in durations:
        print(f"\nDuration: {duration}s")
        
        avg_acc, std_acc = evaluate_model_at_duration(
            model=model,
            test_dataset=test_dataset,
            duration=duration,
            batch_size=batch_size,
            device=device,
            num_crops=num_crops
        )
        
        results[duration] = {
            'mean': avg_acc,
            'std': std_acc
        }
        
        print(f"  Average: {avg_acc:.2f}% ± {std_acc:.2f}%")
    
    # Save results
    results_file = output_path / 'quick_test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'model_path': str(model_path),
            'model_type': model_type,
            'durations': durations,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Plot results
    print("\nGenerating plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    durs = sorted(results.keys())
    means = [results[d]['mean'] for d in durs]
    stds = [results[d]['std'] for d in durs]
    
    ax.errorbar(durs, means, yerr=stds, marker='o', linewidth=2, 
                capsize=5, capthick=2, label='Model Performance')
    
    ax.set_xlabel('Utterance Duration (seconds)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Speaker Recognition Performance vs Duration', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plot_file = output_path / 'performance_curve.png'
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {plot_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    best_duration = max(results.keys(), key=lambda d: results[d]['mean'])
    worst_duration = min(results.keys(), key=lambda d: results[d]['mean'])
    
    print(f"Best performance:  {best_duration}s -> {results[best_duration]['mean']:.2f}%")
    print(f"Worst performance: {worst_duration}s -> {results[worst_duration]['mean']:.2f}%")
    
    if 1.0 in results and 0.5 in results:
        degradation = results[1.0]['mean'] - results[0.5]['mean']
        print(f"Degradation (1.0s -> 0.5s): {degradation:+.2f}%")
    
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Quick short utterance test for trained models'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='chaotic',
        choices=['chaotic', 'baseline'],
        help='Type of model'
    )
    parser.add_argument(
        '--durations',
        type=float,
        nargs='+',
        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        help='Durations to test (in seconds)'
    )
    parser.add_argument(
        '--num_speakers',
        type=int,
        default=50,
        help='Number of speakers'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--num_crops',
        type=int,
        default=3,
        help='Number of random crops per sample'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./quick_test_results',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run test
    results = quick_test(
        model_path=args.model,
        data_dir=args.data,
        model_type=args.model_type,
        durations=args.durations,
        num_speakers=args.num_speakers,
        batch_size=args.batch_size,
        device=args.device,
        num_crops=args.num_crops,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()