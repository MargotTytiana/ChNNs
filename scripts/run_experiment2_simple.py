#!/usr/bin/env python3
"""
Experiment 2: Noise Robustness Testing (Simplified Version)

This script tests trained baseline models under various noise conditions.
It uses the checkpoints from experiment 1.

Usage:
    python scripts/run_experiment2_simple.py
    python scripts/run_experiment2_simple.py --snr 20,10,5,0
    python scripts/run_experiment2_simple.py --noise_types gaussian,babble
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
from dataset_loader import create_speaker_dataloaders
from logger import setup_logger
from reproducibility import set_seed
from noise_augmentation import NoiseAugmentor


class NoisyDataset(Dataset):
    """Dataset wrapper that adds noise to audio samples on-the-fly."""
    
    def __init__(
        self,
        original_dataset: Dataset,
        noise_augmentor: NoiseAugmentor,
        noise_type: str,
        snr_db: float
    ):
        self.original_dataset = original_dataset
        self.noise_augmentor = noise_augmentor
        self.noise_type = noise_type
        self.snr_db = snr_db
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        sample = self.original_dataset[idx]
        
        # Unpack sample (audio, label, speaker_id)
        if len(sample) == 3:
            audio, label, speaker_id = sample
        elif len(sample) == 2:
            audio, label = sample
            speaker_id = label
        else:
            raise ValueError(f"Unexpected sample format: {len(sample)} elements")
        
        # Add noise
        noisy_audio = self.noise_augmentor.add_noise(
            audio, self.noise_type, self.snr_db
        )
        
        if len(sample) == 3:
            return noisy_audio, label, speaker_id
        else:
            return noisy_audio, label


def evaluate_model_on_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    description: str = ""
) -> Dict[str, float]:
    """Evaluate model accuracy on a dataloader."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=description, leave=False)
        for batch in pbar:
            # Handle different batch formats
            if len(batch) == 3:
                audio, labels, _ = batch
            else:
                audio, labels = batch[0], batch[1]
            
            audio = audio.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(audio)
            
            # Handle tuple outputs
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_acc = np.mean(np.array(all_predictions) == np.array(all_labels))
            pbar.set_postfix({'acc': f'{current_acc*100:.1f}%'})
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_predictions == all_labels)
    avg_loss = total_loss / max(num_batches, 1)
    
    return {
        'accuracy': float(accuracy),
        'loss': float(avg_loss),
        'num_samples': len(all_labels),
        'num_correct': int(np.sum(all_predictions == all_labels))
    }


def load_baseline_model(
    checkpoint_path: str,
    model_type: str,
    num_speakers: int,
    device: str
) -> nn.Module:
    """Load a baseline model from checkpoint (.pth or .pkl)."""
    from hybrid_models import TraditionalMLPBaseline
    
    # Determine feature type from model_type
    if 'mel' in model_type:
        feature_type = 'mel'
        n_mels = 80
        n_mfcc = 40
    else:
        feature_type = 'mfcc'
        n_mels = 80
        n_mfcc = 40
    
    # Create model
    model = TraditionalMLPBaseline(
        feature_type=feature_type,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        sample_rate=16000,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.3,
        use_batch_norm=True,
        num_speakers=num_speakers,
        device=device
    )
    
    # Load checkpoint (support both .pth and .pkl)
    import pickle
    
    if checkpoint_path.endswith('.pkl'):
        # Load pickle format
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
    else:
        # Load torch format
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state dict from various possible formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Assume the dict itself is the state dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load weights
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Could not load state dict directly: {e}")
        # Try loading with strict=False
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model


def run_experiment2(
    data_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    noise_types: List[str],
    snr_levels: List[int],
    device: str = 'auto',
    seed: int = 42
):
    """Run experiment 2: noise robustness testing."""
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        name='experiment2',
        log_file=str(output_path / 'experiment2.log')
    )
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    set_seed(seed)
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: NOISE ROBUSTNESS TESTING")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Noise types: {noise_types}")
    logger.info(f"SNR levels: {snr_levels}")
    
    # Create noise augmentor
    noise_augmentor = NoiseAugmentor(sample_rate=16000, seed=seed)
    
    # Load test dataset
    logger.info("\nLoading test dataset...")
    _, _, test_loader = create_speaker_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        train_split=0.7,
        val_split=0.15,
        max_length=3.0,
        sample_rate=16000,
        seed=seed
    )
    
    num_speakers = 26  # Default for LibriSpeech dev-clean-2
    # Try to get actual number from dataset
    try:
        if hasattr(test_loader.dataset, 'num_classes'):
            num_speakers = test_loader.dataset.num_classes
        elif hasattr(test_loader.dataset, 'num_speakers'):
            num_speakers = test_loader.dataset.num_speakers
    except:
        pass
    
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    logger.info(f"Number of speakers: {num_speakers}")
    
    # Find checkpoints
    checkpoint_path = Path(checkpoint_dir)
    models_to_test = {}
    
    # Helper function to find checkpoint
    def find_checkpoint(model_name):
        """Find checkpoint file for a model, supporting unified format."""
        # Unified format: outputs/experiment1_unified/mel_mlp/checkpoints/best_model.pth
        search_patterns = [
            # Unified format (primary)
            f"{model_name}/checkpoints/best_model.pth",
            # Alternative patterns
            f"{model_name}/**/best_model.pth",
            f"{model_name}/**/best_model.pkl",
            f"*{model_name}*/**/best_model.pth",
            f"*{model_name}*/**/best_model.pkl",
            f"{model_name}/**/checkpoint*.pth",
            f"{model_name}/**/checkpoint*.pkl",
        ]
        
        for pattern in search_patterns:
            matches = list(checkpoint_path.glob(pattern))
            if matches:
                # Return the most recent one
                return str(sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0])
        return None
    
    # Look for mel_mlp checkpoint
    mel_ckpt = find_checkpoint('mel_mlp')
    if mel_ckpt:
        models_to_test['mel_mlp'] = mel_ckpt
        logger.info(f"Found mel_mlp checkpoint: {mel_ckpt}")
    
    # Look for mfcc_mlp checkpoint
    mfcc_ckpt = find_checkpoint('mfcc_mlp')
    if mfcc_ckpt:
        models_to_test['mfcc_mlp'] = mfcc_ckpt
        logger.info(f"Found mfcc_mlp checkpoint: {mfcc_ckpt}")
    
    logger.info(f"\nFound {len(models_to_test)} models: {list(models_to_test.keys())}")
    
    if not models_to_test:
        logger.error("No model checkpoints found!")
        logger.info(f"Searched in: {checkpoint_path}")
        return
    
    # Results storage
    all_results = {}
    
    # Test each model
    for model_name, ckpt_path in models_to_test.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {model_name}")
        logger.info(f"Checkpoint: {ckpt_path}")
        logger.info(f"{'='*60}")
        
        try:
            # Load model
            model = load_baseline_model(ckpt_path, model_name, num_speakers, device)
            
            model_results = {}
            
            # Evaluate on clean data
            logger.info("\nEvaluating on CLEAN data...")
            clean_metrics = evaluate_model_on_loader(
                model, test_loader, device, "Clean"
            )
            model_results['clean'] = clean_metrics
            logger.info(f"Clean accuracy: {clean_metrics['accuracy']*100:.2f}%")
            
            clean_acc = clean_metrics['accuracy']
            
            # Evaluate on each noise condition
            for noise_type in noise_types:
                logger.info(f"\n--- Noise: {noise_type.upper()} ---")
                model_results[noise_type] = {}
                
                for snr_db in snr_levels:
                    # Create noisy dataset
                    noisy_dataset = NoisyDataset(
                        test_loader.dataset,
                        noise_augmentor,
                        noise_type,
                        snr_db
                    )
                    
                    # IMPORTANT: Use the same collate_fn as original loader
                    noisy_loader = DataLoader(
                        noisy_dataset,
                        batch_size=test_loader.batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=test_loader.collate_fn  # Preserve collate function
                    )
                    
                    # Evaluate
                    metrics = evaluate_model_on_loader(
                        model, noisy_loader, device, f"SNR={snr_db}dB"
                    )
                    model_results[noise_type][snr_db] = metrics
                    
                    acc = metrics['accuracy']
                    drop = (clean_acc - acc) * 100
                    
                    logger.info(
                        f"  SNR={snr_db:3d}dB: {acc*100:5.2f}% "
                        f"(drop: {drop:+5.2f}%)"
                    )
            
            all_results[model_name] = model_results
            
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            all_results[model_name] = {'error': str(e)}
    
    # Print comparison table
    logger.info("\n" + "=" * 90)
    logger.info("NOISE ROBUSTNESS COMPARISON SUMMARY")
    logger.info("=" * 90)
    
    # Table header
    header = f"{'Noise Type':<12} {'SNR(dB)':<8}"
    for model_name in all_results.keys():
        if 'error' not in all_results[model_name]:
            header += f" {model_name:>12}"
    logger.info(header)
    logger.info("-" * 90)
    
    # Clean row
    row = f"{'Clean':<12} {'-':<8}"
    for model_name, results in all_results.items():
        if 'error' not in results:
            acc = results['clean']['accuracy'] * 100
            row += f" {acc:>11.2f}%"
    logger.info(row)
    logger.info("-" * 90)
    
    # Noise rows
    for noise_type in noise_types:
        for snr_db in snr_levels:
            row = f"{noise_type:<12} {snr_db:<8}"
            for model_name, results in all_results.items():
                if 'error' not in results and noise_type in results:
                    acc = results[noise_type][snr_db]['accuracy'] * 100
                    row += f" {acc:>11.2f}%"
            logger.info(row)
        logger.info("")
    
    logger.info("=" * 90)
    
    # Calculate average degradation
    logger.info("\nAVERAGE ACCURACY DEGRADATION (Clean - Noisy):")
    for model_name, results in all_results.items():
        if 'error' in results:
            continue
        
        clean_acc = results['clean']['accuracy']
        total_drop = 0
        count = 0
        
        for noise_type in noise_types:
            if noise_type in results:
                for snr_db, metrics in results[noise_type].items():
                    total_drop += clean_acc - metrics['accuracy']
                    count += 1
        
        avg_drop = total_drop / max(count, 1) * 100
        logger.info(f"  {model_name}: {avg_drop:+.2f}%")
    
    # Save results
    results_file = output_path / 'noise_robustness_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Experiment 2: Noise Robustness',
            'timestamp': datetime.now().isoformat(),
            'noise_types': noise_types,
            'snr_levels': snr_levels,
            'results': all_results
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Experiment 2: Noise Robustness')
    
    parser.add_argument(
        '--data_dir', type=str,
        default='/scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2',
        help='Path to dataset'
    )
    parser.add_argument(
        '--checkpoint_dir', type=str,
        default='./outputs/experiment1_unified',
        help='Directory containing trained model checkpoints'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='./outputs/experiment2_noise_robustness',
        help='Output directory'
    )
    parser.add_argument(
        '--noise_types', type=str,
        default='gaussian,babble,cafe,street',
        help='Comma-separated noise types'
    )
    parser.add_argument(
        '--snr', type=str,
        default='20,15,10,5,0',
        help='Comma-separated SNR levels in dB'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda']
    )
    parser.add_argument(
        '--seed', type=int, default=42
    )
    
    args = parser.parse_args()
    
    noise_types = args.noise_types.split(',')
    snr_levels = [int(x) for x in args.snr.split(',')]
    
    run_experiment2(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        noise_types=noise_types,
        snr_levels=snr_levels,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()