#!/usr/bin/env python3
"""
Unified Experiment Runner for Chaotic Model

This script runs Experiments 2-5 on the trained chaotic model:
- Experiment 2: Noise Robustness Testing
- Experiment 3: Few-Shot Learning
- Experiment 4: Computational Efficiency
- Experiment 5: Feature Visualization

Usage:
    python scripts/run_chaotic_experiments.py --all
    python scripts/run_chaotic_experiments.py --exp 2
    python scripts/run_chaotic_experiments.py --exp 2,3,4
"""

import os
import sys
import argparse
import json
import time
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import warnings
import gc

warnings.filterwarnings('ignore')

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
from reproducibility import set_seed

# Try to import chaotic model
try:
    from chaotic_network import ChaoticSpeakerRecognitionNetwork
    HAS_CHAOTIC = True
except ImportError:
    try:
        # Try alternative import path
        from models.chaotic_network import ChaoticSpeakerRecognitionNetwork
        HAS_CHAOTIC = True
    except ImportError:
        HAS_CHAOTIC = False
        print("Warning: ChaoticSpeakerRecognitionNetwork not found")

# Try to import noise augmentation
try:
    from noise_augmentation import NoiseAugmentor
    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False
    print("Warning: NoiseAugmentor not found, run: cp noise_augmentation.py data/")

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================
# Configuration
# ============================================================
DEFAULT_CONFIG = {
    # Data settings
    'data_dir': '/scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2',
    'sample_rate': 16000,
    'max_length': 3.0,
    'batch_size': 32,
    
    # Chaotic model checkpoint
    'chaotic_checkpoint': './outputs/experiment1_comparison/chaotic_hybrid/exp1_chaotic_hybrid/checkpoints',
    
    # Noise settings (Exp 2)
    'noise_types': ['gaussian', 'babble', 'cafe', 'street'],
    'snr_levels': [20, 15, 10, 5, 0],
    
    # Few-shot settings (Exp 3)
    'samples_per_speaker': [10, 15, -1], # change 12.03
    'num_runs': 3,
    'fewshot_epochs': 50,
    
    # Output settings
    'output_dir': './outputs/chaotic_experiments',
    'seed': 42,
}


# ============================================================
# Logger Setup
# ============================================================
def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# Model Loading
# ============================================================
def find_chaotic_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the best checkpoint in the directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    print(f"Searching for checkpoints in: {checkpoint_path}")
    
    # Look for best_model.pth
    best_model = checkpoint_path / 'best_model.pth'
    if best_model.exists():
        print(f"Found: {best_model}")
        return str(best_model)
    
    # Look for any .pth file
    pth_files = list(checkpoint_path.glob('*.pth'))
    if pth_files:
        print(f"Found .pth files: {pth_files}")
        return str(pth_files[0])
    
    # Look for .pkl files
    pkl_files = list(checkpoint_path.glob('*.pkl'))
    if pkl_files:
        # Sort by modification time, get newest
        pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"Found .pkl files: {pkl_files[:3]}...")
        return str(pkl_files[0])
    
    # Search in subdirectories
    print(f"Searching subdirectories...")
    for subdir in checkpoint_path.iterdir():
        if subdir.is_dir():
            # Check for best_model.pth in subdir
            best_in_subdir = subdir / 'best_model.pth'
            if best_in_subdir.exists():
                print(f"Found in subdir: {best_in_subdir}")
                return str(best_in_subdir)
            
            # Check for any .pth or .pkl in subdir
            for pattern in ['*.pth', '*.pkl']:
                files = list(subdir.glob(pattern))
                if files:
                    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    print(f"Found in subdir: {files[0]}")
                    return str(files[0])
            
            # Recursive search one more level
            for subsubdir in subdir.iterdir():
                if subsubdir.is_dir():
                    for pattern in ['*.pth', '*.pkl']:
                        files = list(subsubdir.glob(pattern))
                        if files:
                            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            print(f"Found in nested subdir: {files[0]}")
                            return str(files[0])
    
    print("No checkpoint found!")
    return None


def load_chaotic_model(
    checkpoint_path: str,
    num_speakers: int,
    device: str
) -> nn.Module:
    """Load chaotic model from checkpoint."""
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint - handle both .pth and .pkl formats
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("Loaded with torch.load")
    except RuntimeError as e:
        if "Invalid magic number" in str(e) or "corrupt file" in str(e):
            import pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print("Loaded with pickle")
        else:
            raise e
    
    # Debug: show checkpoint structure
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check if 'model' key contains the actual model object
        if 'model' in checkpoint:
            model_obj = checkpoint['model']
            print(f"Model object type: {type(model_obj)}")
            
            # If it's already a nn.Module, use it directly
            if isinstance(model_obj, nn.Module):
                print("Found nn.Module object, using directly")
                model = model_obj
                model = model.to(device)
                model.eval()
                
                # Debug: Show chaotic embedding parameters
                if hasattr(model, 'chaotic_embedding'):
                    ce = model.chaotic_embedding
                    print(f"\nChaotic Embedding Parameters:")
                    print(f"  evolution_time: {getattr(ce, 'evolution_time', 'N/A')}")
                    print(f"  time_step: {getattr(ce, 'time_step', 'N/A')}")
                    print(f"  num_steps: {getattr(ce, 'num_steps', 'N/A')}")
                    print(f"  state_dim: {getattr(ce, 'state_dim', 'N/A')}")
                    print(f"  system_type: {getattr(ce, 'system_type', 'N/A')}")
                    
                    # Fix num_steps if it's 0
                    if hasattr(ce, 'num_steps') and ce.num_steps == 0:
                        print("\n*** WARNING: num_steps is 0! Fixing... ***")
                        ce.num_steps = 50  # Set reasonable default
                        ce.evolution_time = ce.num_steps * ce.time_step
                        print(f"  Fixed num_steps to: {ce.num_steps}")
                        print(f"  Fixed evolution_time to: {ce.evolution_time}")
                
                # Verify model info
                if hasattr(model, 'get_model_info'):
                    info = model.get_model_info()
                    print(f"Model info: {info}")
                
                return model
            
            # If 'model' is a nested dict containing 'model_state_dict'
            elif isinstance(model_obj, dict):
                if 'model_state_dict' in model_obj:
                    print(f"Found nested structure: model -> model_state_dict")
                    state_dict = model_obj['model_state_dict']
                    print(f"state_dict has {len(state_dict)} keys:")
                    for i, (k, v) in enumerate(list(state_dict.items())[:10]):
                        if hasattr(v, 'shape'):
                            print(f"  {i+1}. {k}: {v.shape}")
                        else:
                            print(f"  {i+1}. {k}: {type(v)}")
                    if len(state_dict) > 10:
                        print(f"  ... and {len(state_dict)-10} more keys")
                    
                    # Get config from nested structure if available
                    if 'config' in model_obj:
                        config = model_obj['config']
                        print(f"\nConfig from checkpoint:")
                        for k, v in config.items():
                            print(f"  {k}: {v}")
                    
                    # Get best metric info
                    if 'best_metric' in model_obj:
                        print(f"\nBest metric: {model_obj['best_metric']}")
                    if 'best_epoch' in model_obj:
                        print(f"Best epoch: {model_obj['best_epoch']}")
                else:
                    print(f"'model' is dict with {len(model_obj)} keys: {list(model_obj.keys())}")
                    state_dict = model_obj
            else:
                raise ValueError(f"Unknown model type: {type(model_obj)}")
        
        # Try other common keys
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Using model_state_dict with {len(state_dict)} keys")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"Using state_dict with {len(state_dict)} keys")
        else:
            # Check if checkpoint itself is a state_dict
            first_key = list(checkpoint.keys())[0]
            if isinstance(checkpoint.get(first_key), torch.Tensor):
                state_dict = checkpoint
                print(f"Checkpoint is state_dict directly with {len(state_dict)} keys")
            else:
                raise ValueError(f"Cannot find model in checkpoint. Keys: {list(checkpoint.keys())}")
    else:
        # Checkpoint might be the model itself
        if isinstance(checkpoint, nn.Module):
            print("Checkpoint is nn.Module directly")
            model = checkpoint
            model = model.to(device)
            model.eval()
            return model
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")
    
    # If we reach here, we have a state_dict and need to create model
    if not HAS_CHAOTIC:
        raise ImportError("ChaoticSpeakerRecognitionNetwork not available")
    
    # CRITICAL: Infer actual architecture from state_dict shapes
    # The config may be wrong, but the weights tell the truth
    inferred_hidden_dims = []
    
    # Look at speaker_embedding.embedding_network weights to infer hidden dims
    for key, tensor in state_dict.items():
        if 'speaker_embedding.embedding_network' in key and 'weight' in key:
            # Extract layer index
            parts = key.split('.')
            try:
                layer_idx = int(parts[2])
                if layer_idx % 4 == 0:  # Linear layers are at indices 0, 4, 8, 12...
                    out_features = tensor.shape[0]
                    inferred_hidden_dims.append((layer_idx, out_features))
            except (ValueError, IndexError):
                pass
    
    # Sort by layer index and extract dimensions
    inferred_hidden_dims.sort(key=lambda x: x[0])
    actual_hidden_dims = [dim for _, dim in inferred_hidden_dims]
    
    # The last dimension in embedding network should be speaker_embedding_dim
    # Hidden dims are all except the last
    if len(actual_hidden_dims) >= 2:
        embedding_hidden_dims = actual_hidden_dims[:-1]  # All except last
        speaker_embedding_dim = actual_hidden_dims[-1]   # Last one
    else:
        # Fallback to config
        embedding_hidden_dims = config.get('embedding_hidden_dims', [64, 32])
        speaker_embedding_dim = config.get('speaker_embedding_dim', 128)
    
    print(f"\nInferred architecture from state_dict:")
    print(f"  embedding_hidden_dims: {embedding_hidden_dims}")
    print(f"  speaker_embedding_dim: {speaker_embedding_dim}")
    print(f"  (Config said: {config.get('embedding_hidden_dims', 'N/A')}, {config.get('speaker_embedding_dim', 'N/A')})")
    
    # Use inferred values, but fall back to config for other parameters
    if not config:
        config = {
            'chaotic_system': 'lorenz',
            'embedding_dim': 10,
            'evolution_time': 0.5,
            'time_step': 0.01,
            'mlsa_scales': 5,
            'pooling_type': 'comprehensive',
            'classifier_type': 'cosine',
        }
    
    # CRITICAL FIX: Ensure evolution_time > time_step
    evolution_time = config.get('evolution_time', 0.5)
    time_step = config.get('time_step', 0.01)
    if evolution_time <= time_step:
        print(f"WARNING: evolution_time ({evolution_time}) <= time_step ({time_step})")
        evolution_time = 0.5
        print(f"Fixed evolution_time to: {evolution_time}")
    
    print(f"\nCreating model with:")
    print(f"  speaker_embedding_dim: {speaker_embedding_dim}")
    print(f"  embedding_hidden_dims: {embedding_hidden_dims}")
    print(f"  classifier_type: {config.get('classifier_type', 'cosine')}")
    
    # Create model with INFERRED architecture
    model = ChaoticSpeakerRecognitionNetwork(
        # Audio processing parameters
        sample_rate=config.get('sample_rate', 16000),
        frame_length=config.get('frame_length', 400),
        hop_length=config.get('hop_length', 160),
        
        # Phase space reconstruction
        embedding_dim=config.get('embedding_dim', 10),
        delay_method=config.get('delay_method', 'autocorr'),
        
        # Chaotic features
        mlsa_scales=config.get('mlsa_scales', 5),
        rqa_radius_ratio=config.get('rqa_radius_ratio', 0.1),
        
        # Chaotic embedding
        chaotic_system=config.get('chaotic_system', 'lorenz'),
        evolution_time=evolution_time,
        time_step=time_step,
        
        # Attractor pooling
        pooling_type=config.get('pooling_type', 'comprehensive'),
        
        # Speaker embedding - USE INFERRED VALUES
        speaker_embedding_dim=speaker_embedding_dim,
        embedding_hidden_dims=embedding_hidden_dims,
        
        # Classification
        num_speakers=num_speakers,
        classifier_type=config.get('classifier_type', 'cosine'),
        
        # Device
        device=device
    )
    
    # Fix chaotic embedding num_steps if needed
    if hasattr(model, 'chaotic_embedding'):
        ce = model.chaotic_embedding
        print(f"\nChaotic Embedding created with:")
        print(f"  evolution_time: {getattr(ce, 'evolution_time', 'N/A')}")
        print(f"  time_step: {getattr(ce, 'time_step', 'N/A')}")
        print(f"  num_steps: {getattr(ce, 'num_steps', 'N/A')}")
        
        if hasattr(ce, 'num_steps') and ce.num_steps == 0:
            print(f"\n*** Fixing chaotic_embedding.num_steps (was 0) ***")
            ce.num_steps = 50
            ce.evolution_time = ce.num_steps * ce.time_step
            print(f"  num_steps: {ce.num_steps}, evolution_time: {ce.evolution_time}")
    
    # Load weights
    try:
        model.load_state_dict(state_dict, strict=True)
        print("\nLoaded state_dict successfully (strict=True)")
    except RuntimeError as e:
        print(f"\nStrict loading failed: {e}")
        print("Trying with strict=False...")
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded with strict=False:")
            print(f"  Missing keys: {len(missing)}")
            print(f"  Unexpected keys: {len(unexpected)}")
            if missing:
                print(f"  Missing: {missing[:5]}...")
            if unexpected:
                print(f"  Unexpected: {unexpected[:5]}...")
        except RuntimeError as e2:
            print(f"strict=False also failed: {e2}")
            raise e2
    
    model = model.to(device)
    model.eval()
    
    return model


# ============================================================
# Noisy Dataset Wrapper
# ============================================================
class NoisyDataset(Dataset):
    """Dataset wrapper that adds noise to audio samples."""
    
    def __init__(self, original_dataset, noise_augmentor, noise_type, snr_db):
        self.original_dataset = original_dataset
        self.noise_augmentor = noise_augmentor
        self.noise_type = noise_type
        self.snr_db = snr_db
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        sample = self.original_dataset[idx]
        
        if len(sample) == 3:
            audio, label, speaker_id = sample
        else:
            audio, label = sample
            speaker_id = label
        
        noisy_audio = self.noise_augmentor.add_noise(
            audio, self.noise_type, self.snr_db
        )
        
        if len(sample) == 3:
            return noisy_audio, label, speaker_id
        return noisy_audio, label


# ============================================================
# Experiment 2: Noise Robustness
# ============================================================
def run_experiment2_chaotic(
    model: nn.Module,
    test_loader: DataLoader,
    config: Dict[str, Any],
    logger: logging.Logger,
    output_dir: Path,
    device: str
) -> Dict[str, Any]:
    """Run noise robustness test on chaotic model."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: NOISE ROBUSTNESS (Chaotic Model)")
    logger.info("=" * 70)
    
    if not HAS_NOISE:
        logger.error("NoiseAugmentor not available!")
        return {'error': 'NoiseAugmentor not found'}
    
    noise_augmentor = NoiseAugmentor(sample_rate=config['sample_rate'])
    results = {}
    
    # Evaluate on clean data
    logger.info("\nEvaluating on CLEAN data...")
    clean_acc = evaluate_model(model, test_loader, device)
    results['clean'] = {'accuracy': clean_acc}
    logger.info(f"Clean accuracy: {clean_acc*100:.2f}%")
    
    # Test each noise type and SNR
    for noise_type in config['noise_types']:
        logger.info(f"\n--- Noise: {noise_type.upper()} ---")
        results[noise_type] = {}
        
        for snr_db in config['snr_levels']:
            noisy_dataset = NoisyDataset(
                test_loader.dataset, noise_augmentor, noise_type, snr_db
            )
            noisy_loader = DataLoader(
                noisy_dataset,
                batch_size=test_loader.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=test_loader.collate_fn
            )
            
            acc = evaluate_model(model, noisy_loader, device)
            results[noise_type][snr_db] = {'accuracy': acc}
            drop = (clean_acc - acc) * 100
            logger.info(f"  SNR={snr_db:3d}dB: {acc*100:5.2f}% (drop: {drop:+5.2f}%)")
    
    # Save results
    results_path = output_dir / 'exp2_noise_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    return results


def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                audio, labels, _ = batch
            else:
                audio, labels = batch[0], batch[1]
            
            audio = audio.to(device)
            labels = labels.to(device)
            
            outputs = model(audio)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total if total > 0 else 0.0


# ============================================================
# Experiment 3: Few-Shot Learning
# ============================================================
class FewShotSampler:
    """Sample limited examples per speaker."""
    
    def __init__(self, dataset, num_samples_per_speaker, seed=42):
        self.dataset = dataset
        self.num_samples = num_samples_per_speaker
        self.rng = np.random.RandomState(seed)
        self.speaker_indices = self._build_speaker_indices()
    
    def _build_speaker_indices(self):
        speaker_indices = {}
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            label = sample[1] if len(sample) >= 2 else sample[0]
            if hasattr(label, 'item'):
                label = label.item()
            if label not in speaker_indices:
                speaker_indices[label] = []
            speaker_indices[label].append(idx)
        return speaker_indices
    
    def get_few_shot_indices(self):
        selected = []
        for speaker_id, indices in self.speaker_indices.items():
            if self.num_samples == -1 or self.num_samples >= len(indices):
                selected.extend(indices)
            else:
                sampled = self.rng.choice(indices, size=self.num_samples, replace=False)
                selected.extend(sampled.tolist())
        return selected


class FewShotDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]


def run_experiment3_chaotic(
    config: Dict[str, Any],
    logger: logging.Logger,
    output_dir: Path,
    device: str
) -> Dict[str, Any]:
    """Run few-shot learning experiment with chaotic model."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: FEW-SHOT LEARNING (Chaotic Model)")
    logger.info("=" * 70)
    
    if not HAS_CHAOTIC:
        logger.error("ChaoticSpeakerNetwork not available!")
        return {'error': 'ChaoticSpeakerNetwork not found'}
    
    # Load dataset
    train_loader, val_loader, test_loader = create_speaker_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        sample_rate=config['sample_rate'],
        max_length=config['max_length'],
        train_split=0.7,
        val_split=0.15,
        seed=config['seed']
    )
    
    full_train_dataset = train_loader.dataset
    collate_fn = train_loader.collate_fn
    num_speakers = 26
    
    results = {}
    
    for samples_per_speaker in config['samples_per_speaker']:
        samples_str = "all" if samples_per_speaker == -1 else str(samples_per_speaker)
        logger.info(f"\nSamples per speaker: {samples_str}")
        
        run_results = []
        for run_id in range(config['num_runs']):
            run_seed = config['seed'] + run_id * 1000 + samples_per_speaker
            set_seed(run_seed)
            
            # Create few-shot dataset
            sampler = FewShotSampler(full_train_dataset, samples_per_speaker, run_seed)
            indices = sampler.get_few_shot_indices()
            fewshot_dataset = FewShotDataset(full_train_dataset, indices)
            
            fewshot_loader = DataLoader(
                fewshot_dataset,
                batch_size=min(config['batch_size'], len(fewshot_dataset)),
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn
            )
            
            logger.info(f"  Run {run_id+1}: {len(fewshot_dataset)} samples")
            
            # Create fresh model
            model = ChaoticSpeakerRecognitionNetwork(
                num_speakers=num_speakers,
                sample_rate=config['sample_rate'],
                embedding_dim=8,
                speaker_embedding_dim=256,
                chaotic_system='lorenz',
                device=device
            ).to(device)
            
            # Train
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            best_val_acc = 0.0
            best_model_state = None
            patience_counter = 0
            
            for epoch in range(config['fewshot_epochs']):
                # Train
                model.train()
                for batch in fewshot_loader:
                    if len(batch) == 3:
                        audio, labels, _ = batch
                    else:
                        audio, labels = batch[0], batch[1]
                    
                    audio, labels = audio.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(audio)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                # Validate
                val_acc = evaluate_model(model, val_loader, device)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 15:
                    break
            
            # Test with best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            test_acc = evaluate_model(model, test_loader, device)
            
            run_results.append({
                'val_acc': best_val_acc,
                'test_acc': test_acc,
                'run_id': run_id
            })
            
            logger.info(f"    Val: {best_val_acc*100:.2f}%, Test: {test_acc*100:.2f}%")
            
            del model
            torch.cuda.empty_cache()
        
        # Aggregate results
        test_accs = [r['test_acc'] for r in run_results]
        results[samples_str] = {
            'test_acc_mean': float(np.mean(test_accs)),
            'test_acc_std': float(np.std(test_accs)),
            'runs': run_results
        }
        logger.info(f"  → Mean Test Acc: {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}%")
    
    # Save results
    results_path = output_dir / 'exp3_fewshot_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    return results


# ============================================================
# Experiment 4: Computational Efficiency
# ============================================================
def run_experiment4_chaotic(
    model: nn.Module,
    train_loader: DataLoader,
    config: Dict[str, Any],
    logger: logging.Logger,
    output_dir: Path,
    device: str
) -> Dict[str, Any]:
    """Measure computational efficiency of chaotic model."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 4: COMPUTATIONAL EFFICIENCY (Chaotic Model)")
    logger.info("=" * 70)
    
    results = {}
    input_length = int(config['sample_rate'] * config['max_length'])
    
    # 1. Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results['parameters'] = {
        'total': total_params,
        'trainable': trainable_params
    }
    logger.info(f"\n1. Parameters: {total_params:,} ({total_params/1000:.1f}K)")
    
    # 2. Inference time
    logger.info("\n2. Measuring inference time...")
    model.eval()
    
    # Single sample
    dummy_input = torch.randn(1, input_length).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(50):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    results['inference_single'] = {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000)
    }
    logger.info(f"   Single sample: {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")
    
    # Batch inference
    dummy_batch = torch.randn(config['batch_size'], input_length).to(device)
    
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_batch)
    
    times = []
    with torch.no_grad():
        for _ in range(20):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_batch)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    results['inference_batch'] = {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'throughput': float(config['batch_size'] / np.mean(times))
    }
    logger.info(f"   Batch ({config['batch_size']}): {np.mean(times)*1000:.2f} ms")
    logger.info(f"   Throughput: {config['batch_size']/np.mean(times):.1f} samples/sec")
    
    # 3. Training step time
    logger.info("\n3. Measuring training step time...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    batch_iter = iter(train_loader)
    
    # Warmup
    for _ in range(3):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            batch = next(batch_iter)
        
        if len(batch) == 3:
            audio, labels, _ = batch
        else:
            audio, labels = batch[0], batch[1]
        
        audio, labels = audio.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(audio)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Benchmark
    times = []
    for _ in range(10):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            batch = next(batch_iter)
        
        if len(batch) == 3:
            audio, labels, _ = batch
        else:
            audio, labels = batch[0], batch[1]
        
        audio, labels = audio.to(device), labels.to(device)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        optimizer.zero_grad()
        outputs = model(audio)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    results['training_step'] = {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000)
    }
    logger.info(f"   Step time: {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")
    
    # 4. Memory usage
    if device == 'cuda':
        logger.info("\n4. Memory usage:")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        dummy = torch.randn(config['batch_size'], input_length).to(device)
        with torch.no_grad():
            _ = model(dummy)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        results['memory'] = {'peak_mb': float(peak_memory)}
        logger.info(f"   Peak memory: {peak_memory:.1f} MB")
    
    # Save results
    results_path = output_dir / 'exp4_efficiency_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    return results


# ============================================================
# Experiment 5: Feature Visualization
# ============================================================
def run_experiment5_chaotic(
    model: nn.Module,
    test_loader: DataLoader,
    config: Dict[str, Any],
    logger: logging.Logger,
    output_dir: Path,
    device: str
) -> Dict[str, Any]:
    """Visualize features from chaotic model."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 5: FEATURE VISUALIZATION (Chaotic Model)")
    logger.info("=" * 70)
    
    results = {}
    
    # Extract features
    logger.info("\nExtracting features...")
    model.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting", leave=False):
            if len(batch) == 3:
                audio, labels, _ = batch
            else:
                audio, labels = batch[0], batch[1]
            
            audio = audio.to(device)
            outputs = model(audio)
            
            if isinstance(outputs, tuple):
                features = outputs[0]  # Use logits as features
            else:
                features = outputs
            
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    logger.info(f"Features shape: {features.shape}")
    results['feature_shape'] = features.shape
    
    # Feature statistics
    results['statistics'] = {
        'mean': float(np.mean(features)),
        'std': float(np.std(features)),
        'min': float(np.min(features)),
        'max': float(np.max(features))
    }
    logger.info(f"Mean: {np.mean(features):.4f}, Std: {np.std(features):.4f}")
    
    # Cluster metrics
    if HAS_SKLEARN and len(np.unique(labels)) > 1:
        logger.info("\nComputing cluster metrics...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        try:
            silhouette = silhouette_score(features_scaled, labels)
            results['silhouette_score'] = float(silhouette)
            logger.info(f"Silhouette score: {silhouette:.4f}")
        except Exception as e:
            logger.warning(f"Silhouette score failed: {e}")
    
    # t-SNE visualization
    if HAS_SKLEARN and HAS_MATPLOTLIB:
        logger.info("\nComputing t-SNE...")
        try:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            tsne = TSNE(
                n_components=2,
                perplexity=min(30, len(features) - 1),
                n_iter=1000,
                random_state=config['seed']
            )
            embedding = tsne.fit_transform(features_scaled)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            unique_labels = np.unique(labels)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    c=[colors[i]], label=f'Speaker {label}',
                    alpha=0.6, s=30
                )
            
            ax.set_title('t-SNE: Chaotic Model Features')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            
            if len(unique_labels) <= 15:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            plt.tight_layout()
            plot_path = output_dir / 'chaotic_tsne.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            results['tsne_plot'] = str(plot_path)
            logger.info(f"t-SNE plot saved to: {plot_path}")
        except Exception as e:
            logger.warning(f"t-SNE failed: {e}")
    
    # Save results
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results_path = output_dir / 'exp5_visualization_results.json'
    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    return results


# ============================================================
# Main Runner
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Run experiments on chaotic model')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--exp', type=str, help='Comma-separated experiment numbers (2,3,4,5)')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CONFIG['chaotic_checkpoint'],
                       help='Path to chaotic model checkpoint')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_CONFIG['data_dir'])
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'])
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    if args.all:
        experiments = [2, 3, 4, 5]
    elif args.exp:
        experiments = [int(x) for x in args.exp.split(',')]
    else:
        print("Please specify --all or --exp <numbers>")
        print("Example: python run_chaotic_experiments.py --all")
        print("Example: python run_chaotic_experiments.py --exp 2,4")
        return
    
    # Setup
    config = DEFAULT_CONFIG.copy()
    config['chaotic_checkpoint'] = args.checkpoint
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger('chaotic_exp', str(output_dir / 'chaotic_experiments.log'))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(config['seed'])
    
    logger.info("=" * 70)
    logger.info("CHAOTIC MODEL EXPERIMENTS")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Experiments: {experiments}")
    
    # Find checkpoint
    checkpoint_path = find_chaotic_checkpoint(config['chaotic_checkpoint'])
    if not checkpoint_path:
        logger.error(f"No checkpoint found in: {config['chaotic_checkpoint']}")
        return
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Load data
    logger.info("\nLoading dataset...")
    train_loader, val_loader, test_loader = create_speaker_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        sample_rate=config['sample_rate'],
        max_length=config['max_length'],
        train_split=0.7,
        val_split=0.15,
        seed=config['seed']
    )
    num_speakers = 26
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model (for exp 2, 4, 5)
    model = None
    if any(exp in experiments for exp in [2, 4, 5]):
        logger.info("\nLoading chaotic model...")
        try:
            model = load_chaotic_model(checkpoint_path, num_speakers, device)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    all_results = {}
    
    # Run experiments
    if 2 in experiments and model is not None:
        try:
            results = run_experiment2_chaotic(
                model, test_loader, config, logger, output_dir, device
            )
            all_results['experiment2'] = results
        except Exception as e:
            logger.error(f"Experiment 2 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    if 3 in experiments:
        try:
            results = run_experiment3_chaotic(config, logger, output_dir, device)
            all_results['experiment3'] = results
        except Exception as e:
            logger.error(f"Experiment 3 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    if 4 in experiments and model is not None:
        try:
            results = run_experiment4_chaotic(
                model, train_loader, config, logger, output_dir, device
            )
            all_results['experiment4'] = results
        except Exception as e:
            logger.error(f"Experiment 4 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    if 5 in experiments and model is not None:
        try:
            results = run_experiment5_chaotic(
                model, test_loader, config, logger, output_dir, device
            )
            all_results['experiment5'] = results
        except Exception as e:
            logger.error(f"Experiment 5 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Save all results
    all_results_path = output_dir / 'all_chaotic_results.json'
    
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(all_results_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()