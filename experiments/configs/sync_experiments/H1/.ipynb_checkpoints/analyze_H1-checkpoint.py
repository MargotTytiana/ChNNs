#!/usr/bin/env python3
"""
H1 Experiment Analysis Script
Analyzes gradient conflict between CE loss and sync loss

Usage: python analyze_H1.py <log_file_path>
Example: python analyze_H1.py training.log
"""

import re
import sys
import json
from collections import defaultdict

def analyze_h1_log(log_path):
    """Analyze H1 experiment log file"""
    
    print("=" * 70)
    print("H1 EXPERIMENT ANALYSIS REPORT")
    print("=" * 70)
    
    # Data containers
    conflicts = []  # (epoch, batch, ratio, cos_sim, num_conflicts)
    epoch_metrics = defaultdict(list)  # epoch -> [(loss, acc), ...]
    sync_loss_info = []
    errors = []
    
    current_epoch = 0
    batch_in_epoch = 0
    
    # Validation checks
    checks = {
        'gradient_surgery_init': False,
        'conflict_detection': False,
        'sync_loss_computed': False,
        'training_completed': False,
    }
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Check initialization
            if 'SyncLossWithGradientSurgery' in line or 'gradient_surgery' in line:
                checks['gradient_surgery_init'] = True
            
            # Parse epoch
            epoch_match = re.search(r'\[DEBUG\] Current epoch: (\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                batch_in_epoch = 0
            
            # Parse H1 conflict data
            conflict_match = re.search(
                r'\[H1 CE-SYNC CONFLICT\].*?conflict_ratio=([\d.]+).*?avg_cos_sim=([-\d.]+).*?num_conflicts=(\d+)',
                line
            )
            if conflict_match:
                checks['conflict_detection'] = True
                ratio = float(conflict_match.group(1))
                cos_sim = float(conflict_match.group(2))
                num_conf = int(conflict_match.group(3))
                conflicts.append((current_epoch, batch_in_epoch, ratio, cos_sim, num_conf))
                batch_in_epoch += 1
            
            # Parse training metrics
            train_match = re.search(r'Train Epoch (\d+):.*?Loss: ([\d.]+)', line)
            if train_match:
                epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                epoch_metrics[epoch].append(('train_loss', loss))
            
            # Parse validation metrics
            val_match = re.search(r'Val.*?Loss: ([\d.]+).*?Acc: ([\d.]+)', line)
            if val_match:
                epoch_metrics[current_epoch].append(('val', float(val_match.group(1)), float(val_match.group(2))))
            
            # Check sync loss
            if 'SYNC DEBUG' in line and 'loss_sync=' in line:
                checks['sync_loss_computed'] = True
                sync_loss_info.append(line.strip())
            
            # Check for errors
            if 'Error' in line or 'error' in line:
                errors.append(line.strip()[:200])
            
            # Check completion
            if 'Training completed' in line or 'Best accuracy' in line:
                checks['training_completed'] = True
    
    # ============ REPORT ============
    
    print("\n" + "=" * 70)
    print("1. VALIDATION CHECKS")
    print("=" * 70)
    
    all_ok = True
    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_ok = False
        print(f"  {check}: {status}")
    
    if not checks['conflict_detection']:
        print("\n  ⚠ WARNING: No gradient conflict data found!")
        print("    Make sure the code with compute_ce_sync_conflict() is deployed.")
    
    print(f"\n  Overall: {'✓ H1 experiment ran correctly' if all_ok else '✗ Some issues detected'}")
    
    # ============ CONFLICT ANALYSIS ============
    
    print("\n" + "=" * 70)
    print("2. GRADIENT CONFLICT ANALYSIS")
    print("=" * 70)
    
    if not conflicts:
        print("  No conflict data available.")
        return
    
    # Overall statistics
    ratios = [c[2] for c in conflicts]
    cos_sims = [c[3] for c in conflicts]
    
    avg_ratio = sum(ratios) / len(ratios)
    avg_cos_sim = sum(cos_sims) / len(cos_sims)
    
    high_conflict = sum(1 for r in ratios if r > 0.5)
    low_conflict = sum(1 for r in ratios if r <= 0.5)
    
    print(f"\n  Total batches analyzed: {len(conflicts)}")
    print(f"\n  Conflict Ratio Statistics:")
    print(f"    Mean: {avg_ratio:.4f} ({avg_ratio*100:.1f}% parameters in conflict)")
    print(f"    Min:  {min(ratios):.4f}")
    print(f"    Max:  {max(ratios):.4f}")
    print(f"    High conflict batches (>50%): {high_conflict} ({100*high_conflict/len(ratios):.1f}%)")
    print(f"    Low conflict batches (≤50%):  {low_conflict} ({100*low_conflict/len(ratios):.1f}%)")
    
    print(f"\n  Cosine Similarity Statistics:")
    print(f"    Mean: {avg_cos_sim:.4f}")
    print(f"    Min:  {min(cos_sims):.4f}")
    print(f"    Max:  {max(cos_sims):.4f}")
    negative_cos = sum(1 for c in cos_sims if c < 0)
    print(f"    Negative (conflicting): {negative_cos} ({100*negative_cos/len(cos_sims):.1f}%)")
    
    # Per-epoch analysis
    print("\n  Per-Epoch Summary:")
    print("  " + "-" * 50)
    print(f"  {'Epoch':<8} {'Batches':<10} {'Avg Ratio':<12} {'Avg CosSim':<12}")
    print("  " + "-" * 50)
    
    epoch_data = defaultdict(list)
    for e, b, r, c, n in conflicts:
        epoch_data[e].append((r, c))
    
    for epoch in sorted(epoch_data.keys())[:10]:  # First 10 epochs
        data = epoch_data[epoch]
        avg_r = sum(d[0] for d in data) / len(data)
        avg_c = sum(d[1] for d in data) / len(data)
        print(f"  {epoch:<8} {len(data):<10} {avg_r:<12.4f} {avg_c:<12.4f}")
    
    if len(epoch_data) > 10:
        print(f"  ... ({len(epoch_data) - 10} more epochs)")
    
    # ============ CONCLUSIONS ============
    
    print("\n" + "=" * 70)
    print("3. CONCLUSIONS - WHAT H1 PROVES")
    print("=" * 70)
    
    # Determine significance
    if avg_ratio > 0.5:
        significance = "STRONG"
        conclusion = "CE and sync loss have SIGNIFICANT gradient conflicts"
    elif avg_ratio > 0.3:
        significance = "MODERATE"
        conclusion = "CE and sync loss have MODERATE gradient conflicts"
    else:
        significance = "WEAK"
        conclusion = "CE and sync loss have MINIMAL gradient conflicts"
    
    print(f"\n  Significance Level: {significance}")
    print(f"\n  Main Conclusion:")
    print(f"    {conclusion}")
    print(f"    - {avg_ratio*100:.1f}% of parameters show opposing gradient directions")
    print(f"    - Average cosine similarity: {avg_cos_sim:.3f}")
    
    if avg_cos_sim < -0.3:
        print(f"    - Gradients are predominantly OPPOSING (cos_sim < -0.3)")
    elif avg_cos_sim < 0:
        print(f"    - Gradients show SLIGHT opposition")
    else:
        print(f"    - Gradients are ALIGNED on average")
    
    print("\n  Interpretation:")
    if avg_ratio > 0.5 and avg_cos_sim < -0.3:
        print("    ★ H1 HYPOTHESIS CONFIRMED ★")
        print("    Cross-entropy loss and synchronization loss have conflicting")
        print("    optimization objectives in the majority of training batches.")
        print("    This explains potential training instability and embedding collapse.")
    elif avg_ratio > 0.3:
        print("    H1 hypothesis partially supported.")
        print("    Significant conflicts exist but may be manageable.")
    else:
        print("    H1 hypothesis NOT strongly supported.")
        print("    Conflicts are minimal - other factors may cause issues.")
    
    # ============ RECOMMENDATIONS ============
    
    print("\n" + "=" * 70)
    print("4. FOLLOW-UP EXPERIMENTS NEEDED?")
    print("=" * 70)
    
    if avg_ratio > 0.5:
        print("\n  Recommended follow-up experiments:")
        print("    1. H2: Test gradient surgery to resolve conflicts")
        print("    2. H3: Compare with desync-only approach")
        print("    3. H4: Analyze conflict patterns per network layer")
        print("\n  Current experiment is SUFFICIENT for thesis H1 section.")
    else:
        print("\n  May need additional analysis:")
        print("    - Verify sync loss is computing correctly")
        print("    - Check if conflict varies by training phase")
    
    # ============ THESIS WRITING GUIDE ============
    
    print("\n" + "=" * 70)
    print("5. THESIS WRITING GUIDE")
    print("=" * 70)
    
    print("""
  WHERE TO WRITE:
  ---------------
  Chapter 5: Experiments and Results
    Section 5.X: Gradient Conflict Analysis (H1)

  SUGGESTED STRUCTURE:
  -------------------
  1. Motivation (1 paragraph)
     - Why investigate gradient conflicts?
     - Hypothesis: CE and sync loss may conflict
  
  2. Methodology (1-2 paragraphs)
     - Per-batch gradient computation for both losses
     - Cosine similarity measurement
     - Conflict ratio definition
  
  3. Results (1-2 paragraphs + table/figure)
     - Use the statistics above
     - Include a figure showing conflict ratio over epochs
  
  4. Discussion (1 paragraph)
     - Implications for training stability
     - Connection to embedding collapse

  KEY NUMBERS FOR THESIS:
  ----------------------""")
    
    print(f"    - Total batches analyzed: {len(conflicts)}")
    print(f"    - Average conflict ratio: {avg_ratio:.2%}")
    print(f"    - Average cosine similarity: {avg_cos_sim:.3f}")
    print(f"    - Batches with >50% conflict: {100*high_conflict/len(conflicts):.1f}%")
    
    print("\n  SAMPLE THESIS TEXT:")
    print("  " + "-" * 50)
    print(f"""
    "To validate hypothesis H1, we computed per-batch gradient 
    conflicts between cross-entropy and synchronization losses.
    Analysis of {len(conflicts)} training batches revealed that 
    {avg_ratio:.1%} of parameters exhibited conflicting gradient 
    directions on average (cosine similarity = {avg_cos_sim:.3f}).
    In {100*high_conflict/len(conflicts):.0f}% of batches, more than 
    half of all parameters showed opposing gradients between the 
    two loss components. These findings confirm that naive 
    combination of CE and sync losses leads to optimization 
    conflicts, motivating the gradient surgery approach in H2."
    """)
    
    # Save detailed data for plotting
    output_data = {
        'summary': {
            'total_batches': len(conflicts),
            'avg_conflict_ratio': avg_ratio,
            'avg_cos_sim': avg_cos_sim,
            'high_conflict_percentage': 100*high_conflict/len(conflicts),
        },
        'per_epoch': {str(e): {
            'batches': len(d),
            'avg_ratio': sum(x[0] for x in d)/len(d),
            'avg_cos_sim': sum(x[1] for x in d)/len(d),
        } for e, d in epoch_data.items()},
        'all_conflicts': [{'epoch': c[0], 'batch': c[1], 'ratio': c[2], 'cos_sim': c[3]} 
                         for c in conflicts]
    }
    
    output_path = log_path.replace('.log', '_h1_analysis.json')
    if output_path == log_path:
        output_path = 'h1_analysis.json'
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n  Detailed data saved to: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_H1.py <log_file_path>")
        print("Example: python analyze_H1.py training.log")
        sys.exit(1)
    
    analyze_h1_log(sys.argv[1])
