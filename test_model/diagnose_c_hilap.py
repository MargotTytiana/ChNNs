#!/usr/bin/env python3
"""
Comprehensive Diagnostic Script for C-HiLAP Model Issues

This script checks:
1. Checkpoint contents and structure
2. Model configuration consistency
3. Dataset and label mapping
4. Model output distributions
5. Embedding quality analysis
6. Classification accuracy (direct)
7. Layer-by-layer activation analysis
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

# =============================================================================
# 1. Project Setup
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

# =============================================================================
# 2. Checkpoint Analysis
# =============================================================================
def analyze_checkpoint(ckpt_path):
    """Analyze checkpoint file contents."""
    print("\n" + "=" * 70)
    print("1. CHECKPOINT ANALYSIS")
    print("=" * 70)
    print(f"Path: {ckpt_path}")
    
    # Try loading
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print("✓ Loaded with torch.load")
    except:
        try:
            with open(ckpt_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print("✓ Loaded with pickle")
        except Exception as e:
            print(f"✗ Failed to load: {e}")
            return None
    
    # Analyze structure
    print(f"\nCheckpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Keys: {list(checkpoint.keys())}")
        
        # Check config
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"\n[Config Contents]")
            important_keys = ['num_speakers', 'embedding_dim', 'chaotic_system', 
                           'speaker_embedding_dim', 'classifier_type', 'data_dir']
            for key in important_keys:
                if key in config:
                    print(f"  {key}: {config[key]}")
                elif isinstance(config, dict):
                    # Search nested
                    for k, v in config.items():
                        if isinstance(v, dict) and key in v:
                            print(f"  {key}: {v[key]} (in {k})")
        
        # Check state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        print(f"\n[State Dict Analysis]")
        print(f"Total keys: {len(state_dict)}")
        
        # Find classifier layer
        classifier_info = []
        for key, value in state_dict.items():
            if 'classifier' in key.lower() and 'weight' in key.lower():
                if hasattr(value, 'shape'):
                    classifier_info.append((key, value.shape))
                    
        if classifier_info:
            print(f"\n[Classifier Layers Found]")
            for key, shape in classifier_info:
                print(f"  {key}: {shape}")
                if len(shape) == 2:
                    print(f"    → Output dim (num_speakers): {shape[0]}")
                    print(f"    → Input dim (embedding): {shape[1]}")
        
        # Check for training info
        if 'epoch' in checkpoint:
            print(f"\n[Training Info]")
            print(f"  Epoch: {checkpoint.get('epoch')}")
            print(f"  Best val acc: {checkpoint.get('best_val_acc', 'N/A')}")
            print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
            
        return checkpoint
    else:
        print(f"Unexpected checkpoint type: {type(checkpoint)}")
        return checkpoint


# =============================================================================
# 3. Dataset Analysis
# =============================================================================
def analyze_dataset(data_dir):
    """Analyze dataset structure and labels."""
    print("\n" + "=" * 70)
    print("2. DATASET ANALYSIS")
    print("=" * 70)
    print(f"Data dir: {data_dir}")
    
    from data.dataset_loader import create_speaker_dataloaders
    
    train_loader, val_loader, test_loader = create_speaker_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        train_split=0.7,
        val_split=0.15,
        max_length=3.0,
        sample_rate=16000
    )
    
    print(f"\n[Dataset Statistics]")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Get unique labels
    all_labels = []
    for batch in test_loader:
        if len(batch) >= 2:
            all_labels.extend(batch[1].numpy().tolist())
    
    unique_labels = sorted(set(all_labels))
    print(f"\n[Label Analysis]")
    print(f"  Unique labels in test: {len(unique_labels)}")
    print(f"  Label range: [{min(unique_labels)}, {max(unique_labels)}]")
    print(f"  Labels: {unique_labels[:10]}..." if len(unique_labels) > 10 else f"  Labels: {unique_labels}")
    
    # Check label distribution
    from collections import Counter
    label_counts = Counter(all_labels)
    print(f"\n[Label Distribution]")
    for label, count in sorted(label_counts.items())[:5]:
        print(f"  Label {label}: {count} samples")
    if len(label_counts) > 5:
        print(f"  ... ({len(label_counts)} total labels)")
    
    return train_loader, val_loader, test_loader, unique_labels


# =============================================================================
# 4. Model Loading and Analysis
# =============================================================================
def load_and_analyze_model(ckpt_path, config, device='cpu'):
    """Load model and analyze its structure."""
    print("\n" + "=" * 70)
    print("3. MODEL ANALYSIS")
    print("=" * 70)
    
    from models.chaotic_network import ChaoticSpeakerRecognitionNetwork
    
    # Extract config
    if isinstance(config, dict):
        model_config = {
            'sample_rate': config.get('sample_rate', 16000),
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
            'num_speakers': config.get('num_speakers', 251),
            'classifier_type': config.get('classifier_type', 'cosine'),
            'device': device
        }
    else:
        model_config = {'device': device, 'num_speakers': 251}
    
    print(f"[Model Config]")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Create model
    print(f"\n[Creating Model...]")
    model = ChaoticSpeakerRecognitionNetwork(**model_config)
    
    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device) if ckpt_path.endswith('.pth') else pickle.load(open(ckpt_path, 'rb'))
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ Weights loaded (strict=True)")
    except RuntimeError as e:
        print(f"⚠ Strict loading failed: {e}")
        model.load_state_dict(state_dict, strict=False)
        print("✓ Weights loaded (strict=False)")
    
    model.to(device)
    model.eval()
    
    # Analyze model structure
    print(f"\n[Model Structure]")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Check classifier
    if hasattr(model, 'classifier'):
        print(f"\n[Classifier Info]")
        print(f"  Type: {type(model.classifier)}")
        if hasattr(model.classifier, 'weight'):
            print(f"  Weight shape: {model.classifier.weight.shape}")
        elif hasattr(model.classifier, 'fc'):
            print(f"  FC weight shape: {model.classifier.fc.weight.shape}")
    
    return model, model_config


# =============================================================================
# 5. Forward Pass Analysis
# =============================================================================
def analyze_forward_pass(model, test_loader, device='cpu', num_batches=3):
    """Analyze model outputs in detail."""
    print("\n" + "=" * 70)
    print("4. FORWARD PASS ANALYSIS")
    print("=" * 70)
    
    model.eval()
    
    all_outputs = []
    all_labels = []
    all_embeddings = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_batches:
                break
                
            if len(batch) >= 2:
                audio, labels = batch[0], batch[1]
            else:
                continue
            
            audio = audio.to(device)
            labels = labels.to(device)
            
            print(f"\n[Batch {i+1}]")
            print(f"  Input shape: {audio.shape}")
            print(f"  Input range: [{audio.min():.4f}, {audio.max():.4f}]")
            print(f"  Labels: {labels[:5].tolist()}...")
            
            # Forward pass
            try:
                outputs = model(audio)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    print(f"  Output is tuple, length: {len(outputs)}")
                elif isinstance(outputs, dict):
                    print(f"  Output is dict, keys: {outputs.keys()}")
                    logits = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))
                else:
                    logits = outputs
                
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"  Logits mean: {logits.mean():.4f}, std: {logits.std():.4f}")
                
                # Check for NaN/Inf
                if torch.isnan(logits).any():
                    print("  ⚠ WARNING: NaN values in logits!")
                if torch.isinf(logits).any():
                    print("  ⚠ WARNING: Inf values in logits!")
                
                # Predictions
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                print(f"  Predictions: {preds[:5].tolist()}...")
                print(f"  Max prob: {probs.max(dim=-1)[0][:5].tolist()}")
                
                # Check if predictions are always the same
                unique_preds = torch.unique(preds)
                print(f"  Unique predictions: {len(unique_preds)} ({unique_preds[:10].tolist()}...)")
                
                # Accuracy
                correct = (preds == labels).sum().item()
                total = labels.size(0)
                print(f"  Batch accuracy: {correct}/{total} = {correct/total*100:.2f}%")
                
                all_outputs.append(logits.cpu())
                all_labels.append(labels.cpu())
                
            except Exception as e:
                print(f"  ✗ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
    
    if all_outputs:
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        print(f"\n[Overall Statistics]")
        print(f"  Total samples analyzed: {all_outputs.shape[0]}")
        
        # Check output distribution
        print(f"\n[Output Distribution per Class]")
        for class_idx in range(min(5, all_outputs.shape[1])):
            class_logits = all_outputs[:, class_idx]
            print(f"  Class {class_idx}: mean={class_logits.mean():.4f}, std={class_logits.std():.4f}")
        
        return all_outputs, all_labels
    
    return None, None


# =============================================================================
# 6. Classification Accuracy Test
# =============================================================================
def test_classification_accuracy(model, test_loader, device='cpu'):
    """Test direct classification accuracy."""
    print("\n" + "=" * 70)
    print("5. CLASSIFICATION ACCURACY TEST")
    print("=" * 70)
    
    model.eval()
    correct = 0
    total = 0
    
    label_correct = defaultdict(int)
    label_total = defaultdict(int)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
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
                logits = outputs.get('logits', list(outputs.values())[0])
            else:
                logits = outputs
            
            preds = torch.argmax(logits, dim=-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                label_total[label] += 1
                if pred == label:
                    label_correct[label] += 1
                all_preds.append(pred)
                all_labels.append(label)
    
    overall_acc = correct / total if total > 0 else 0
    print(f"\n[Overall Accuracy]")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {overall_acc*100:.2f}%")
    
    print(f"\n[Per-Class Accuracy]")
    for label in sorted(label_total.keys())[:10]:
        acc = label_correct[label] / label_total[label] if label_total[label] > 0 else 0
        print(f"  Class {label}: {label_correct[label]}/{label_total[label]} = {acc*100:.2f}%")
    
    # Confusion analysis
    from collections import Counter
    pred_counter = Counter(all_preds)
    print(f"\n[Prediction Distribution]")
    print(f"  Unique predictions: {len(pred_counter)}")
    most_common = pred_counter.most_common(5)
    for pred, count in most_common:
        print(f"  Pred {pred}: {count} times ({count/len(all_preds)*100:.1f}%)")
    
    # Check for collapsed predictions
    if len(pred_counter) < 5:
        print(f"\n⚠ WARNING: Model appears to be collapsing predictions to few classes!")
        print(f"  Only {len(pred_counter)} unique predictions out of {len(set(all_labels))} possible classes")
    
    return overall_acc, label_correct, label_total


# =============================================================================
# 7. Embedding Quality Analysis
# =============================================================================
def analyze_embedding_quality(model, test_loader, device='cpu'):
    """Analyze the quality of speaker embeddings."""
    print("\n" + "=" * 70)
    print("6. EMBEDDING QUALITY ANALYSIS")
    print("=" * 70)
    
    model.eval()
    
    # Try to get embeddings before classifier
    embeddings = []
    labels_list = []
    
    # Hook to capture embeddings
    embedding_output = []
    
    def hook_fn(module, input, output):
        embedding_output.append(output.detach().cpu())
    
    # Find the layer before classifier
    hook_registered = False
    if hasattr(model, 'speaker_embedding'):
        handle = model.speaker_embedding.register_forward_hook(hook_fn)
        hook_registered = True
        print("Hook registered on: speaker_embedding")
    elif hasattr(model, 'embedding_layer'):
        handle = model.embedding_layer.register_forward_hook(hook_fn)
        hook_registered = True
        print("Hook registered on: embedding_layer")
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) >= 2:
                audio, labels = batch[0], batch[1]
            else:
                continue
            
            audio = audio.to(device)
            _ = model(audio)
            
            if embedding_output:
                embeddings.append(embedding_output[-1])
                embedding_output.clear()
            
            labels_list.append(labels)
    
    if hook_registered:
        handle.remove()
    
    if embeddings:
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        print(f"\n[Embedding Statistics]")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        
        # Check for collapsed embeddings
        embedding_std_per_dim = embeddings.std(dim=0)
        collapsed_dims = (embedding_std_per_dim < 0.01).sum().item()
        print(f"\n[Embedding Health]")
        print(f"  Collapsed dimensions (std < 0.01): {collapsed_dims}/{embeddings.shape[1]}")
        
        # Compute intra-class and inter-class distances
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        diag_mask = torch.eye(len(labels), dtype=torch.bool)
        
        pos_mask = label_matrix & (~diag_mask)
        neg_mask = ~label_matrix
        
        if pos_mask.sum() > 0:
            pos_sim = sim_matrix[pos_mask].mean().item()
            neg_sim = sim_matrix[neg_mask].mean().item()
            gap = pos_sim - neg_sim
            
            print(f"\n[Similarity Analysis]")
            print(f"  Intra-class similarity (same speaker): {pos_sim:.4f}")
            print(f"  Inter-class similarity (diff speaker): {neg_sim:.4f}")
            print(f"  Gap: {gap:.4f}")
            
            if gap < 0.05:
                print(f"\n⚠ WARNING: Very low similarity gap!")
                print(f"  This indicates embeddings are NOT speaker-discriminative")
        
        return embeddings, labels
    else:
        print("Could not capture embeddings")
        return None, None


# =============================================================================
# 8. Main Diagnostic Function
# =============================================================================
def run_full_diagnosis(ckpt_path, data_dir, device='auto'):
    """Run complete diagnostic analysis."""
    print("=" * 70)
    print("C-HiLAP MODEL COMPREHENSIVE DIAGNOSIS")
    print("=" * 70)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Data: {data_dir}")
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Checkpoint Analysis
    checkpoint = analyze_checkpoint(ckpt_path)
    if checkpoint is None:
        return
    
    config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    
    # 2. Dataset Analysis
    train_loader, val_loader, test_loader, unique_labels = analyze_dataset(data_dir)
    
    # Check for mismatch
    num_speakers_config = config.get('num_speakers', 251)
    num_speakers_data = len(unique_labels)
    
    print("\n" + "=" * 70)
    print("⚠ CRITICAL CHECK: Speaker Count Mismatch")
    print("=" * 70)
    print(f"  Config num_speakers: {num_speakers_config}")
    print(f"  Data unique labels: {num_speakers_data}")
    print(f"  Label range in data: [0, {max(unique_labels)}]")
    
    if num_speakers_config != num_speakers_data:
        print(f"\n  ⚠ MISMATCH DETECTED!")
        print(f"  Model was trained for {num_speakers_config} speakers")
        print(f"  But test data has {num_speakers_data} speakers")
        
        if max(unique_labels) >= num_speakers_config:
            print(f"\n  ✗ CRITICAL: Test labels exceed model output dimension!")
            print(f"    Max label: {max(unique_labels)}, Model outputs: {num_speakers_config}")
    
    # 3. Load and Analyze Model
    model, model_config = load_and_analyze_model(ckpt_path, config, device)
    
    # 4. Forward Pass Analysis
    outputs, labels = analyze_forward_pass(model, test_loader, device)
    
    # 5. Classification Accuracy
    accuracy, label_correct, label_total = test_classification_accuracy(model, test_loader, device)
    
    # 6. Embedding Quality
    embeddings, emb_labels = analyze_embedding_quality(model, test_loader, device)
    
    # Final Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    issues = []
    
    if num_speakers_config != num_speakers_data:
        issues.append(f"Speaker count mismatch: config={num_speakers_config}, data={num_speakers_data}")
    
    if accuracy < 0.1:
        issues.append(f"Very low accuracy: {accuracy*100:.2f}%")
    
    if embeddings is not None:
        collapsed_dims = (embeddings.std(dim=0) < 0.01).sum().item()
        if collapsed_dims > embeddings.shape[1] * 0.5:
            issues.append(f"Embedding collapse: {collapsed_dims}/{embeddings.shape[1]} dims collapsed")
    
    if issues:
        print("\n[Issues Found]")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✓ No major issues detected")
    
    print("\n[Recommendations]")
    if num_speakers_config != num_speakers_data:
        print("  1. Retrain model with correct num_speakers matching your data")
        print("  2. Or use the same dataset that was used for training")
    if accuracy < 0.5:
        print("  3. Check if model checkpoint is from a completed training")
        print("  4. Verify training was successful (check training logs)")


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose C-HiLAP Model Issues')
    parser.add_argument('--checkpoint', '-c', required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', '-d', 
                        default='/scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2')
    parser.add_argument('--device', default='auto')
    
    args = parser.parse_args()
    
    run_full_diagnosis(args.checkpoint, args.data_dir, args.device)