#!/usr/bin/env python3
"""
Experiment 2: Noise Robustness Testing (Import Fixed Version)

This script tests trained baseline models under various noise conditions.
It dynamically loads model configurations from Experiment 1 checkpoints.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional, List

# =============================================================================
# 1. Setup Imports (Robust Path Handling)
# =============================================================================
def setup_project_path():
    """Add project root and subdirs to sys.path."""
    current_file = Path(__file__).resolve()
    # Assuming structure: Project/scripts/run_experiment2.py
    project_root = current_file.parent.parent
    
    # Define paths to add
    paths_to_add = [
        str(project_root),
        str(project_root / 'core'),
        str(project_root / 'models'), 
        str(project_root / 'features'),  # This allows 'import noise_augmentation'
        str(project_root / 'data'),
        str(project_root / 'utils'),
    ]
    
    # Add to sys.path if not present
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return project_root

PROJECT_ROOT = setup_project_path()

# =============================================================================
# 2. Dynamic Imports
# =============================================================================
try:
    from models.hybrid_models import TraditionalMLPBaseline
    from data.dataset_loader import create_speaker_dataloaders
    from utils.logger import setup_logger
    from utils.reproducibility import set_seed
    
    # --- ROBUST IMPORT FOR NOISE AUGMENTOR ---
    # Try multiple ways to import NoiseAugmentor because file locations vary
    try:
        # 1. Try importing from features package (if Model is in path)
        from features.noise_augmentation import NoiseAugmentor
    except ImportError:
        try:
            # 2. Try importing directly (if features/ is in path)
            import noise_augmentation
            NoiseAugmentor = noise_augmentation.NoiseAugmentor
        except ImportError:
            try:
                # 3. Try importing from utils (common alternative location)
                from utils.noise_augmentation import NoiseAugmentor
            except ImportError:
                # 4. Try importing directly from utils/ (if utils/ is in path)
                import noise_augmentation as na_utils
                NoiseAugmentor = na_utils.NoiseAugmentor

except ImportError as e:
    print("\n" + "!"*80)
    print(f"CRITICAL IMPORT ERROR: {e}")
    print("!"*80)
    print("\nDebug Info:")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Script Location: {__file__}")
    print("sys.path:")
    for p in sys.path:
        print(f"  - {p}")
    print("\nPlease verify that 'noise_augmentation.py' exists in 'features/' or 'utils/'.")
    sys.exit(1)


# =============================================================================
# 3. Dataset Wrapper
# =============================================================================
class NoisyDataset(Dataset):
    """Dataset wrapper that adds noise to audio samples on-the-fly."""
    
    def __init__(
        self,
        original_dataset: Dataset,
        noise_augmentor: Any, 
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
        
        # Robust unpacking
        if isinstance(sample, (tuple, list)):
            if len(sample) >= 2:
                audio = sample[0]
                label = sample[1]
                others = sample[2:]
            else:
                raise ValueError(f"Sample has insufficient elements: {len(sample)}")
        else:
            raise ValueError(f"Unexpected sample type: {type(sample)}")
        
        # Ensure audio is numpy array
        if isinstance(audio, torch.Tensor):
            audio_np = audio.numpy()
        else:
            audio_np = audio
            
        # Add noise
        noisy_audio = self.noise_augmentor.add_noise(
            audio_np, self.noise_type, self.snr_db
        )
        
        # Convert back to float tensor
        noisy_audio = torch.from_numpy(noisy_audio).float()
            
        if others:
            return (noisy_audio, label, *others)
        else:
            return (noisy_audio, label)


# =============================================================================
# 4. Evaluation Logic
# =============================================================================
def evaluate_model_on_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    description: str = ""
) -> Dict[str, float]:
    """Evaluate model accuracy on a dataloader."""
    model.eval()
    
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=description, leave=False)
        for batch in pbar:
            if len(batch) >= 2:
                audio, labels = batch[0], batch[1]
            else:
                continue
            
            audio = audio.to(device)
            labels = labels.to(device)
            
            outputs = model(audio)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / max(num_batches, 1)
    
    return {
        'accuracy': float(accuracy),
        'loss': float(avg_loss),
        'num_samples': total
    }


# =============================================================================
# 5. Model Loading
# =============================================================================
def load_baseline_model(
    checkpoint_path: str,
    model_name: str, 
    num_speakers: int,
    device: str
) -> nn.Module:
    """Load a baseline model using config from checkpoint."""
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        raise

    # 1. Retrieve Configuration
    config = checkpoint.get('config', {})
    
    # 2. Infer Feature Type and Parameters
    if 'feature_type' in config:
        feature_type = config['feature_type']
    elif 'mel' in model_name:
        feature_type = 'mel'
    else:
        feature_type = 'mfcc'
        
    n_mels = config.get('n_mels', 80)
    # Default MFCC n_mfcc is usually 13 or 40. 
    # If not in config, guess based on feature type logic in exp1
    if feature_type == 'mfcc':
        n_mfcc = config.get('n_mfcc', 40) # Baseline usually uses 40 or 13
    else:
        n_mfcc = config.get('n_mfcc', 40)
        
    sample_rate = config.get('sample_rate', 16000)
    
    # 3. Retrieve Architecture Parameters
    hidden_dims = config.get('hidden_dims', [256, 128, 64])
    dropout_rate = config.get('dropout_rate', 0.3)
    use_batch_norm = config.get('use_batch_norm', True)
    
    print(f"Initializing {feature_type.upper()} model with dims: {hidden_dims}")
    
    # 4. Instantiate Model
    model = TraditionalMLPBaseline(
        feature_type=feature_type,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        sample_rate=sample_rate,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        num_speakers=num_speakers,
        device=device
    )
    
    # 5. Load State Dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Attempting non-strict loading...")
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model


# =============================================================================
# 6. Main Experiment Loop
# =============================================================================
def run_experiment2(
    data_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    noise_types: List[str],
    snr_levels: List[int],
    device: str = 'auto',
    seed: int = 42
):
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
    
    # Initialize Augmentor
    try:
        noise_augmentor = NoiseAugmentor(sample_rate=16000, seed=seed)
        logger.info("Noise Augmentor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize NoiseAugmentor: {e}")
        return

    # Load Data
    logger.info("\nLoading test dataset...")
    try:
        _, _, test_loader = create_speaker_dataloaders(
            data_dir=data_dir,
            batch_size=32,
            train_split=0.7,
            val_split=0.15,
            max_length=3.0,
            sample_rate=16000,
            seed=seed
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    num_speakers = 26
    try:
        if hasattr(test_loader.dataset, 'num_classes'):
            num_speakers = test_loader.dataset.num_classes
    except:
        pass
    
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Find Checkpoints
    checkpoint_path = Path(checkpoint_dir)
    models_to_test = {}
    
    search_names = ['mel_mlp', 'mfcc_mlp']
    for name in search_names:
        potential_path = checkpoint_path / name / 'checkpoints' / 'best_model.pth'
        if potential_path.exists():
            models_to_test[name] = str(potential_path)
            logger.info(f"Found {name} checkpoint: {potential_path}")
        else:
            matches = list(checkpoint_path.rglob(f"*{name}*/**/best_model.pth"))
            if matches:
                # Pick the most recent one
                best = sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                models_to_test[name] = str(best)
                logger.info(f"Found {name} checkpoint: {best}")

    if not models_to_test:
        logger.error(f"No model checkpoints found in {checkpoint_dir}")
        return

    # Run Tests
    all_results = {}
    
    for model_name, ckpt_path in models_to_test.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            model = load_baseline_model(ckpt_path, model_name, num_speakers, device)
            
            model_results = {}
            
            # 1. Clean Baseline
            logger.info("Evaluating on CLEAN data...")
            clean_metrics = evaluate_model_on_loader(
                model, test_loader, device, "Clean"
            )
            model_results['clean'] = clean_metrics
            clean_acc = clean_metrics['accuracy']
            logger.info(f"Clean accuracy: {clean_acc*100:.2f}%")
            
            # 2. Noise Tests
            for noise_type in noise_types:
                logger.info(f"\n--- Noise: {noise_type.upper()} ---")
                model_results[noise_type] = {}
                
                for snr_db in snr_levels:
                    noisy_dataset = NoisyDataset(
                        test_loader.dataset,
                        noise_augmentor,
                        noise_type,
                        snr_db
                    )
                    
                    noisy_loader = DataLoader(
                        noisy_dataset,
                        batch_size=test_loader.batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=test_loader.collate_fn
                    )
                    
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

    # Summary and Save
    results_path = output_path / 'noise_robustness_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nExperiment completed. Results saved to: {results_path}")


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