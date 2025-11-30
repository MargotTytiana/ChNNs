#!/usr/bin/env python3
"""
Diagnostic Script for Chaotic Speaker Recognition Network

This script helps identify why the model accuracy is ~4% instead of expected 90%+

Run this script in your training environment to diagnose issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# ============================================================================
# DIAGNOSIS 1: Check if gradients are flowing properly
# ============================================================================

def diagnose_gradient_flow(model, sample_input, sample_labels):
    """Check if gradients are properly flowing through the network."""
    print("\n" + "="*70)
    print("DIAGNOSIS 1: Gradient Flow Analysis")
    print("="*70)
    
    model.train()
    model.zero_grad()
    
    # Forward pass
    try:
        logits = model(sample_input, labels=sample_labels)
        loss = F.cross_entropy(logits, sample_labels)
        loss.backward()
    except Exception as e:
        print(f"[ERROR] Forward/backward pass failed: {e}")
        return False
    
    # Check gradients for each module
    gradient_issues = []
    zero_grad_modules = []
    none_grad_modules = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                none_grad_modules.append(name)
            elif param.grad.abs().max() == 0:
                zero_grad_modules.append(name)
    
    if none_grad_modules:
        print(f"\n[CRITICAL] Modules with NO gradients ({len(none_grad_modules)}):")
        for name in none_grad_modules[:10]:
            print(f"  - {name}")
        if len(none_grad_modules) > 10:
            print(f"  ... and {len(none_grad_modules) - 10} more")
        gradient_issues.append("no_gradients")
    
    if zero_grad_modules:
        print(f"\n[WARNING] Modules with ZERO gradients ({len(zero_grad_modules)}):")
        for name in zero_grad_modules[:10]:
            print(f"  - {name}")
        if len(zero_grad_modules) > 10:
            print(f"  ... and {len(zero_grad_modules) - 10} more")
        gradient_issues.append("zero_gradients")
    
    # Check specific critical modules
    critical_modules = ['feature_projection', 'speaker_embedding', 'classifier']
    print(f"\n[INFO] Gradient statistics for critical modules:")
    for name, param in model.named_parameters():
        for critical in critical_modules:
            if critical in name and param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()
                print(f"  {name}: norm={grad_norm:.6f}, max={grad_max:.6f}")
    
    if not gradient_issues:
        print("\n[OK] Gradient flow appears normal")
        return True
    else:
        print(f"\n[PROBLEM] Gradient issues detected: {gradient_issues}")
        return False


# ============================================================================
# DIAGNOSIS 2: Check feature extraction quality
# ============================================================================

def diagnose_feature_extraction(model, sample_input):
    """Check if chaotic features are being extracted properly."""
    print("\n" + "="*70)
    print("DIAGNOSIS 2: Feature Extraction Analysis")
    print("="*70)
    
    model.eval()
    issues = []
    
    with torch.no_grad():
        try:
            logits, intermediates = model(sample_input, return_intermediates=True)
        except Exception as e:
            print(f"[ERROR] Forward pass with intermediates failed: {e}")
            return False
    
    # Check phase space
    if 'phase_space' in intermediates:
        ps = intermediates['phase_space']
        print(f"\n[Phase Space]")
        print(f"  Shape: {ps.shape}")
        print(f"  Range: [{ps.min():.4f}, {ps.max():.4f}]")
        print(f"  Std: {ps.std():.4f}")
        
        if ps.std() < 1e-6:
            print("  [CRITICAL] Phase space has no variance!")
            issues.append("phase_space_no_variance")
    
    # Check chaotic features
    if 'chaotic_features' in intermediates:
        cf = intermediates['chaotic_features']
        print(f"\n[Chaotic Features]")
        print(f"  Shape: {cf.shape}")
        print(f"  Range: [{cf.min():.4f}, {cf.max():.4f}]")
        print(f"  Std: {cf.std():.4f}")
        print(f"  NaN count: {torch.isnan(cf).sum().item()}")
        print(f"  Inf count: {torch.isinf(cf).sum().item()}")
        
        # Check if features are distinguishable between samples
        if cf.shape[0] > 1:
            pairwise_dist = torch.cdist(cf, cf)
            off_diag_mask = ~torch.eye(cf.shape[0], dtype=bool, device=cf.device)
            avg_dist = pairwise_dist[off_diag_mask].mean().item()
            print(f"  Avg pairwise distance: {avg_dist:.6f}")
            
            if avg_dist < 0.01:
                print("  [CRITICAL] Features are nearly identical across samples!")
                issues.append("identical_chaotic_features")
    
    # Check chaotic trajectories
    if 'chaotic_trajectories' in intermediates:
        ct = intermediates['chaotic_trajectories']
        print(f"\n[Chaotic Trajectories]")
        print(f"  Shape: {ct.shape}")
        print(f"  Range: [{ct.min():.4f}, {ct.max():.4f}]")
        print(f"  Std: {ct.std():.4f}")
        
        # Check trajectory diversity
        ct_flat = ct.view(ct.shape[0], -1)
        if ct_flat.shape[0] > 1:
            traj_dist = torch.cdist(ct_flat, ct_flat)
            off_diag_mask = ~torch.eye(ct_flat.shape[0], dtype=bool, device=ct_flat.device)
            avg_traj_dist = traj_dist[off_diag_mask].mean().item()
            print(f"  Avg trajectory distance: {avg_traj_dist:.6f}")
            
            if avg_traj_dist < 0.1:
                print("  [CRITICAL] Trajectories are too similar!")
                issues.append("similar_trajectories")
    
    # Check pooled features
    if 'pooled_features' in intermediates:
        pf = intermediates['pooled_features']
        print(f"\n[Pooled Features]")
        print(f"  Shape: {pf.shape}")
        print(f"  Range: [{pf.min():.4f}, {pf.max():.4f}]")
        print(f"  Std per dim (first 10): {pf.std(dim=0)[:10].tolist()}")
        
        # Check if pooled features vary
        if pf.shape[0] > 1:
            pf_dist = torch.cdist(pf, pf)
            off_diag_mask = ~torch.eye(pf.shape[0], dtype=bool, device=pf.device)
            avg_pf_dist = pf_dist[off_diag_mask].mean().item()
            print(f"  Avg pooled feature distance: {avg_pf_dist:.6f}")
            
            if avg_pf_dist < 0.001:
                print("  [CRITICAL] Pooled features are nearly identical!")
                issues.append("identical_pooled_features")
    
    # Check speaker embeddings
    if 'speaker_embeddings' in intermediates:
        se = intermediates['speaker_embeddings']
        print(f"\n[Speaker Embeddings]")
        print(f"  Shape: {se.shape}")
        print(f"  Norm mean: {se.norm(dim=1).mean():.4f}")
        print(f"  Norm std: {se.norm(dim=1).std():.4f}")
        
        if se.shape[0] > 1:
            se_dist = torch.cdist(se, se)
            off_diag_mask = ~torch.eye(se.shape[0], dtype=bool, device=se.device)
            avg_se_dist = se_dist[off_diag_mask].mean().item()
            print(f"  Avg embedding distance: {avg_se_dist:.6f}")
            
            if avg_se_dist < 0.001:
                print("  [CRITICAL] Speaker embeddings are nearly identical!")
                issues.append("identical_embeddings")
    
    # Check logits
    print(f"\n[Logits]")
    print(f"  Shape: {logits.shape}")
    print(f"  Range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
    max_prob = probs.max(dim=1)[0].mean()
    print(f"  Max probability (avg): {max_prob:.4f}")
    print(f"  Entropy (avg): {entropy:.4f}")
    
    if not issues:
        print("\n[OK] Feature extraction appears normal")
        return True
    else:
        print(f"\n[PROBLEM] Feature extraction issues: {issues}")
        return False


# ============================================================================
# DIAGNOSIS 3: Check for gradient disconnection in chaotic features
# ============================================================================

def diagnose_chaotic_feature_gradient(model, sample_input, sample_labels):
    """Check if the chaotic feature extraction breaks gradient flow."""
    print("\n" + "="*70)
    print("DIAGNOSIS 3: Chaotic Feature Gradient Disconnection Check")
    print("="*70)
    
    model.train()
    
    # Check if extract_chaotic_features uses .detach() or .numpy()
    # This is a common issue - MLSA/RQA often use numpy operations
    
    print("\n[INFO] Checking extract_chaotic_features method...")
    
    if hasattr(model, 'extract_chaotic_features'):
        import inspect
        source = inspect.getsource(model.extract_chaotic_features)
        
        issues = []
        if '.cpu()' in source and '.numpy()' in source:
            issues.append("Uses .cpu().numpy() - breaks gradient!")
        if '.detach()' in source:
            issues.append("Uses .detach() - may break gradient")
        if 'torch.no_grad()' in source:
            issues.append("Uses no_grad() - breaks gradient")
        
        if issues:
            print(f"  [CRITICAL] Found gradient-breaking operations:")
            for issue in issues:
                print(f"    - {issue}")
            print("\n  This is likely THE ROOT CAUSE of poor training!")
            print("  The chaotic features are computed using numpy,")
            print("  which completely disconnects the gradient flow.")
            return False
        else:
            print("  [OK] No obvious gradient disconnection found in source")
    
    # Check if feature_projection receives gradients
    print("\n[INFO] Testing gradient flow through feature_projection...")
    
    if hasattr(model, 'feature_projection'):
        model.zero_grad()
        logits = model(sample_input, labels=sample_labels)
        loss = F.cross_entropy(logits, sample_labels)
        loss.backward()
        
        has_grad = False
        for name, param in model.feature_projection.named_parameters():
            if param.grad is not None and param.grad.abs().max() > 0:
                has_grad = True
                print(f"  {name}: grad_max = {param.grad.abs().max():.6f}")
        
        if not has_grad:
            print("  [CRITICAL] feature_projection receives no gradients!")
            print("  This confirms gradient disconnection at chaotic features.")
            return False
        else:
            print("  [OK] feature_projection receives gradients")
    
    return True


# ============================================================================
# DIAGNOSIS 4: Check data distribution and labels
# ============================================================================

def diagnose_data_distribution(train_loader, num_speakers):
    """Check if data distribution is causing issues."""
    print("\n" + "="*70)
    print("DIAGNOSIS 4: Data Distribution Analysis")
    print("="*70)
    
    # Collect label statistics
    label_counts = {}
    total_samples = 0
    
    for batch_idx, (audio, labels) in enumerate(train_loader):
        for label in labels.tolist():
            label_counts[label] = label_counts.get(label, 0) + 1
            total_samples += 1
        
        if batch_idx >= 10:  # Sample first 10 batches
            break
    
    print(f"\n[INFO] Label distribution (first 10 batches):")
    print(f"  Total samples: {total_samples}")
    print(f"  Unique labels: {len(label_counts)}")
    print(f"  Label range: [{min(label_counts.keys())}, {max(label_counts.keys())}]")
    print(f"  Expected speakers: {num_speakers}")
    
    if max(label_counts.keys()) >= num_speakers:
        print(f"  [CRITICAL] Label {max(label_counts.keys())} >= num_speakers {num_speakers}!")
        print("  This will cause indexing errors or wrong predictions!")
        return False
    
    # Check for imbalance
    counts = list(label_counts.values())
    if len(counts) > 1:
        imbalance_ratio = max(counts) / min(counts)
        print(f"  Class imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 10:
            print("  [WARNING] Severe class imbalance detected!")
    
    print("\n[OK] Data distribution appears acceptable")
    return True


# ============================================================================
# DIAGNOSIS 5: Learning rate and optimization issues
# ============================================================================

def diagnose_optimization(model, optimizer, sample_input, sample_labels):
    """Check for optimization issues."""
    print("\n" + "="*70)
    print("DIAGNOSIS 5: Optimization Analysis")
    print("="*70)
    
    model.train()
    
    # Check initial loss
    with torch.no_grad():
        logits = model(sample_input, labels=sample_labels)
        initial_loss = F.cross_entropy(logits, sample_labels).item()
    
    num_classes = logits.shape[1]
    expected_random_loss = np.log(num_classes)
    
    print(f"\n[INFO] Loss analysis:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Expected random guess loss (log({num_classes})): {expected_random_loss:.4f}")
    
    if initial_loss > expected_random_loss * 1.5:
        print("  [WARNING] Loss is higher than random - model may be unstable")
    elif abs(initial_loss - expected_random_loss) < 0.1:
        print("  [INFO] Loss close to random - model hasn't learned yet")
    
    # Try a few optimization steps
    print(f"\n[INFO] Testing optimization (5 steps)...")
    losses = [initial_loss]
    
    for step in range(5):
        optimizer.zero_grad()
        logits = model(sample_input, labels=sample_labels)
        loss = F.cross_entropy(logits, sample_labels)
        loss.backward()
        
        # Check gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        optimizer.step()
        losses.append(loss.item())
        
        if step == 0:
            print(f"    Step {step}: loss={loss.item():.4f}, grad_norm={total_norm:.4f}")
    
    print(f"    Final loss: {losses[-1]:.4f}")
    
    if losses[-1] >= losses[0]:
        print("  [WARNING] Loss not decreasing - optimization may be stuck")
        if total_norm < 1e-6:
            print("  [CRITICAL] Gradient norm is near zero - no learning happening!")
            return False
    else:
        print("  [OK] Loss is decreasing normally")
    
    return True


# ============================================================================
# DIAGNOSIS 6: Model architecture issues
# ============================================================================

def diagnose_architecture(model):
    """Check for architecture issues."""
    print("\n" + "="*70)
    print("DIAGNOSIS 6: Architecture Analysis")
    print("="*70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[INFO] Model statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    if hasattr(model, 'num_speakers'):
        params_per_speaker = total_params / model.num_speakers
        print(f"  Parameters per speaker: {params_per_speaker:.0f}")
        
        if params_per_speaker < 1000:
            print("  [WARNING] Very few parameters per speaker - model may underfit")
    
    # Check specific component dimensions
    print(f"\n[INFO] Component dimensions:")
    
    if hasattr(model, 'speaker_embedding_dim'):
        print(f"  Speaker embedding dim: {model.speaker_embedding_dim}")
    
    if hasattr(model, 'embedding_hidden_dims'):
        print(f"  Hidden dims: {model.embedding_hidden_dims}")
    
    if hasattr(model, 'classifier'):
        if hasattr(model.classifier, 'classifier'):
            for i, layer in enumerate(model.classifier.classifier):
                if isinstance(layer, nn.Linear):
                    print(f"  Classifier layer {i}: {layer.in_features} -> {layer.out_features}")
    
    return True


# ============================================================================
# MAIN DIAGNOSTIC RUNNER
# ============================================================================

def run_full_diagnosis(model, train_loader, optimizer, device, num_speakers):
    """Run all diagnostics."""
    print("\n" + "="*70)
    print("CHAOTIC SPEAKER RECOGNITION - FULL DIAGNOSIS")
    print("="*70)
    
    # Get sample batch
    sample_batch = next(iter(train_loader))
    sample_input, sample_labels = sample_batch
    sample_input = sample_input.to(device)
    sample_labels = sample_labels.to(device)
    
    print(f"\nSample batch: audio shape = {sample_input.shape}, labels = {sample_labels[:8].tolist()}")
    
    results = {}
    
    # Run all diagnostics
    results['architecture'] = diagnose_architecture(model)
    results['data'] = diagnose_data_distribution(train_loader, num_speakers)
    results['features'] = diagnose_feature_extraction(model, sample_input)
    results['gradient_flow'] = diagnose_gradient_flow(model, sample_input, sample_labels)
    results['chaotic_gradient'] = diagnose_chaotic_feature_gradient(model, sample_input, sample_labels)
    results['optimization'] = diagnose_optimization(model, optimizer, sample_input, sample_labels)
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        print("\n" + "="*70)
        print("RECOMMENDED FIXES")
        print("="*70)
        
        if not results.get('chaotic_gradient', True) or not results.get('gradient_flow', True):
            print("""
[FIX 1] GRADIENT DISCONNECTION (Most likely cause)
--------------------------------------------------
The chaotic feature extraction uses numpy operations which break gradient flow.

Solution: Make the feature projection layer do most of the learning:

1. In chaotic_network.py, modify extract_chaotic_features to add noise:
   
   def extract_chaotic_features(self, phase_space_data):
       # ... existing code ...
       chaotic_features = torch.from_numpy(features_np).float().to(device)
       
       # Add learnable noise to restore gradient flow
       if self.training:
           noise = torch.randn_like(chaotic_features) * 0.01
           chaotic_features = chaotic_features + noise
       
       return chaotic_features

2. Or use a differentiable approximation of MLSA/RQA

3. Or increase the capacity of post-extraction learnable layers
""")
        
        if not results.get('features', True):
            print("""
[FIX 2] IDENTICAL FEATURES
--------------------------
Features are not distinguishing between speakers.

Solution:
1. Check if MLSA/RQA extraction is actually working (not returning fallback values)
2. Increase audio segment length (currently 3s might be too short)
3. Add data augmentation to create more feature variation
4. Check if phase space reconstruction parameters are appropriate
""")
    
    return all_passed


if __name__ == "__main__":
    print("This script should be run in your training environment.")
    print("Import and call run_full_diagnosis() with your model and data.")
    print("\nExample usage:")
    print("""
    from diagnose_chaotic_training import run_full_diagnosis
    
    # After setting up your experiment
    experiment.setup()
    
    # Run diagnosis
    passed = run_full_diagnosis(
        model=experiment.model,
        train_loader=experiment.train_loader,
        optimizer=experiment.optimizer,
        device=experiment.device,
        num_speakers=experiment.config['num_speakers']
    )
    """)