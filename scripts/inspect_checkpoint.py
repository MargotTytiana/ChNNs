#!/usr/bin/env python3
"""Inspect checkpoint structure to understand model architecture."""

import pickle
import torch
import sys
from pathlib import Path

def inspect_checkpoint(checkpoint_path: str):
    """Inspect checkpoint structure."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("Loaded with torch.load")
    except:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print("Loaded with pickle")
    
    # Show top-level keys
    if isinstance(checkpoint, dict):
        print(f"\nTop-level keys: {list(checkpoint.keys())}")
        
        # Show config if available
        if 'config' in checkpoint:
            print(f"\nConfig:")
            config = checkpoint['config']
            if isinstance(config, dict):
                for k, v in config.items():
                    print(f"  {k}: {v}")
        
        # Check 'model' key
        if 'model' in checkpoint:
            model_obj = checkpoint['model']
            print(f"\n'model' type: {type(model_obj)}")
            
            if isinstance(model_obj, dict):
                print(f"'model' is dict with {len(model_obj)} keys:")
                for i, (k, v) in enumerate(model_obj.items()):
                    if hasattr(v, 'shape'):
                        print(f"  {i+1}. {k}: {v.shape}")
                    else:
                        print(f"  {i+1}. {k}: {type(v)}")
            elif hasattr(model_obj, 'state_dict'):
                print("'model' is nn.Module")
                state_dict = model_obj.state_dict()
                print(f"state_dict has {len(state_dict)} keys:")
                for i, (k, v) in enumerate(state_dict.items()):
                    print(f"  {i+1}. {k}: {v.shape}")
        
        # Show model state dict keys
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\nmodel_state_dict keys ({len(state_dict)}):")
            for i, k in enumerate(sorted(state_dict.keys())):
                shape = state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'N/A'
                print(f"  {i+1}. {k}: {shape}")
        
        # Show metrics
        if 'metrics' in checkpoint:
            print(f"\nMetrics: {checkpoint['metrics']}")
        
        # Show additional_info
        if 'additional_info' in checkpoint:
            print(f"\nAdditional info: {checkpoint['additional_info']}")
        
        # Show best metric info
        if 'best_metric' in checkpoint:
            print(f"\nBest metric: {checkpoint['best_metric']}")
        if 'best_epoch' in checkpoint:
            print(f"Best epoch: {checkpoint['best_epoch']}")
        if 'epoch' in checkpoint:
            print(f"Saved at epoch: {checkpoint['epoch']}")
            
        # Show experiment state if available
        if 'experiment_state' in checkpoint:
            exp_state = checkpoint['experiment_state']
            print(f"\nExperiment state keys: {list(exp_state.keys())}")
            if 'best_model_state' in exp_state:
                best_state = exp_state['best_model_state']
                if isinstance(best_state, dict):
                    print(f"best_model_state has {len(best_state)} keys:")
                    for i, (k, v) in enumerate(list(best_state.items())[:10]):
                        if hasattr(v, 'shape'):
                            print(f"  {i+1}. {k}: {v.shape}")
                        else:
                            print(f"  {i+1}. {k}: {type(v)}")
                    if len(best_state) > 10:
                        print(f"  ... and {len(best_state)-10} more keys")
    else:
        print(f"Checkpoint type: {type(checkpoint)}")

def search_for_checkpoints(base_dir: str):
    """Search for all checkpoint files."""
    base = Path(base_dir)
    patterns = ['*.pkl', '*.pth', '*.pt']
    
    all_files = []
    for pattern in patterns:
        all_files.extend(base.rglob(pattern))
    
    print(f"\nFound {len(all_files)} checkpoint files:")
    for f in sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f} ({size_mb:.2f} MB)")
    
    return all_files

if __name__ == "__main__":
    # Default checkpoint path
    default_path = "outputs/experiment1_comparison/chaotic_hybrid/exp1_chaotic_hybrid/checkpoints/exp_20251130_185504/checkpoint_epoch_0000_20251201_195739.pkl"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--search':
            search_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
            files = search_for_checkpoints(search_dir)
            if files and len(sys.argv) <= 2:
                # Inspect the largest file
                largest = max(files, key=lambda x: x.stat().st_size)
                print(f"\nInspecting largest file: {largest}")
                inspect_checkpoint(str(largest))
        else:
            checkpoint_path = sys.argv[1]
            inspect_checkpoint(checkpoint_path)
    else:
        inspect_checkpoint(default_path)