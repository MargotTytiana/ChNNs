#!/usr/bin/env python3
"""
Experiment 2b: Noise Robustness via Speaker Verification (Final Fix)

Fixes:
1. Auto-detects feature type (Mel vs MFCC) from model name to prevent input dimension mismatch.
2. Auto-detects output speaker count to prevent output dimension mismatch.
3. Performs robust verification using embeddings.
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

# =============================================================================
# 1. Robust Import Setup
# =============================================================================
def setup_project_path():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    paths = [
        str(project_root),
        str(project_root/'core'),
        str(project_root/'models'), 
        str(project_root/'features'),
        str(project_root/'data'),
        str(project_root/'utils')
    ]
    for path in paths:
        if path not in sys.path: sys.path.insert(0, path)
    return project_root

PROJECT_ROOT = setup_project_path()

# Import Project Modules
try:
    from models.hybrid_models import TraditionalMLPBaseline
    from data.dataset_loader import create_speaker_dataloaders
    from utils.logger import setup_logger
    from utils.reproducibility import set_seed
    
    try:
        from features.noise_augmentation import NoiseAugmentor
    except ImportError:
        try:
            import noise_augmentation
            NoiseAugmentor = noise_augmentation.NoiseAugmentor
        except ImportError:
            from utils.noise_augmentation import NoiseAugmentor

except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

# =============================================================================
# 2. Model Helpers
# =============================================================================
def load_feature_extractor_model(ckpt_path, model_name, device):
    """
    Loads model with smart parameter inference.
    Args:
        ckpt_path: Path to checkpoint
        model_name: Name of the model (e.g., 'mel_mlp', 'mfcc_mlp')
    """
    print(f"Loading: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get('config', {})
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # --- FIX 1: INFER FEATURE TYPE FROM NAME ---
    # The config often lacks 'feature_type', causing default to 'mel' (80)
    # forcing mismatch for mfcc (40).
    if 'mel' in model_name:
        feature_type = 'mel'
    elif 'mfcc' in model_name:
        feature_type = 'mfcc'
    else:
        feature_type = config.get('feature_type', 'mel')
        
    print(f"  Inferred feature type: {feature_type}")

    # --- FIX 2: AUTO-DETECT SPEAKER COUNT ---
    # Fixes output size mismatch (e.g. 251 vs 26)
    detected_speakers = 26 # Fallback
    for k, v in state_dict.items():
        if 'classifier' in k and 'weight' in k and len(v.shape) == 2:
            # Linear layer weight shape is [out_features, in_features]
            # We want the output features of the LAST layer.
            # Usually the loop order preserves structure, so the last one found is likely output.
            detected_speakers = v.shape[0]
            
    print(f"  Detected original speaker count: {detected_speakers}")
    
    # Initialize model with CORRECT dimensions
    # Note: We use the hardcoded defaults from run_experiment1.py for n_mels/n_mfcc
    # because they might not be in config either.
    model = TraditionalMLPBaseline(
        feature_type=feature_type,
        n_mels=config.get('n_mels', 80),
        n_mfcc=config.get('n_mfcc', 40),
        sample_rate=config.get('sample_rate', 16000),
        hidden_dims=config.get('hidden_dims', [256, 128, 64]), # Use fallback if missing
        num_speakers=detected_speakers,
        device=device
    )
    
    # Load weights
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"  Warning: Strict loading failed ({e}). Retrying with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.eval()
    
    # --- STRIP CLASSIFIER HEAD ---
    # Replace the final linear layer with Identity to get embeddings
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        print("  Stripping final classification layer for verification...")
        model.classifier[-1] = nn.Identity()
        
    return model

def compute_similarity_stats(embeddings, labels):
    """Computes similarity gap between same-speaker and diff-speaker pairs."""
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Similarity Matrix (N x N)
    sim_matrix = torch.mm(embeddings, embeddings.t())
    
    # Masks
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    diag_mask = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    
    pos_mask = label_matrix & (~diag_mask) # Same speaker, not self
    neg_mask = ~label_matrix               # Different speaker
    
    if pos_mask.sum() == 0: return 0.0, 0.0, 0.0
    
    pos_sim = sim_matrix[pos_mask].mean().item()
    neg_sim = sim_matrix[neg_mask].mean().item()
    gap = pos_sim - neg_sim
    
    return pos_sim, neg_sim, gap

# =============================================================================
# 3. Main Experiment
# =============================================================================
def run_experiment(args):
    # Setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('exp2_verify', str(out_dir / 'verification.log'))
    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else 'cpu'
    set_seed(args.seed)
    
    logger.info("="*60)
    logger.info("EXPERIMENT 2b: ROBUST VERIFICATION")
    logger.info("="*60)
    
    # Data & Noise
    _, _, test_loader = create_speaker_dataloaders(
        data_dir=args.data_dir, batch_size=64, 
        train_split=0.7, val_split=0.15, max_length=3.0, sample_rate=16000
    )
    noise_aug = NoiseAugmentor(sample_rate=16000, seed=args.seed)
    
    # Find Models
    ckpt_dir = Path(args.checkpoint_dir)
    models = {}
    for name in ['mel_mlp', 'mfcc_mlp']:
        matches = list(ckpt_dir.rglob(f"*{name}*/**/best_model.pth"))
        if matches:
            best = sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            models[name] = str(best)
    
    if not models:
        logger.error("No models found!")
        return

    results = {}
    noise_types = args.noise_types.split(',')
    snr_levels = [int(x) for x in args.snr.split(',')]
    
    for name, path in models.items():
        logger.info(f"\nProcessing {name}...")
        try:
            # PASS NAME HERE to help inference
            model = load_feature_extractor_model(path, name, device)
            
            model_res = {}
            
            # Helper to extract all embeddings
            def get_embeddings(n_type='clean', snr=0):
                all_emb, all_lbl = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        if len(batch) >= 2:
                            audio, label = batch[0], batch[1]
                            
                            if n_type != 'clean':
                                audio_np = audio.numpy()
                                noisy = noise_aug.add_noise(audio_np, n_type, snr)
                                audio = torch.from_numpy(noisy).float()
                            
                            audio = audio.to(device)
                            emb = model(audio)
                            all_emb.append(emb.cpu())
                            all_lbl.append(label)
                return torch.cat(all_emb), torch.cat(all_lbl)

            # 1. Clean Baseline
            logger.info("  Testing Clean...")
            emb, lbl = get_embeddings('clean')
            pos, neg, gap = compute_similarity_stats(emb, lbl)
            logger.info(f"    Clean Gap: {gap:.4f} (Pos: {pos:.2f}, Neg: {neg:.2f})")
            model_res['clean'] = {'gap': gap, 'pos': pos, 'neg': neg}
            
            # 2. Noise Tests
            for n_type in noise_types:
                logger.info(f"  Testing {n_type}...")
                model_res[n_type] = {}
                for snr in snr_levels:
                    emb, lbl = get_embeddings(n_type, snr)
                    pos, neg, gap = compute_similarity_stats(emb, lbl)
                    logger.info(f"    SNR {snr}dB: Gap: {gap:.4f}")
                    model_res[n_type][snr] = {'gap': gap, 'pos': pos, 'neg': neg}
            
            results[name] = model_res
            
        except Exception as e:
            logger.error(f"Failed to test {name}: {e}")
            import traceback
            traceback.print_exc()

    # Save Results
    with open(out_dir / 'verification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {out_dir / 'verification_results.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2')
    parser.add_argument('--checkpoint_dir', default='./outputs/experiment1_unified')
    parser.add_argument('--output_dir', default='./outputs/experiment2_verification')
    parser.add_argument('--noise_types', default='gaussian,babble,cafe,street')
    parser.add_argument('--snr', default='20,15,10,5,0')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    run_experiment(parser.parse_args())