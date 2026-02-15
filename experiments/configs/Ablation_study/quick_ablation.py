"""
Quick Ablation Study Runner
============================

Simplified script for quick ablation experiments.
This version uses fewer epochs for faster iteration.

Usage:
    cd /home/claude
    python quick_ablation.py
"""

import sys
sys.path.insert(0, '/mnt/project')

from ablation_study import AblationExperiment

def main():
    print("\n" + "="*70)
    print("QUICK ABLATION STUDY")
    print("="*70 + "\n")
    
    # Configuration
    config_path = '/mnt/project/chaotic_config.yaml'
    output_dir = '/mnt/user-data/outputs/ablation_study'
    epochs = 30  # Reduced for faster testing
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}\n")
    
    # Create and run experiment
    ablation_exp = AblationExperiment(
        config_path=config_path,
        output_dir=output_dir
    )
    
    # Run all ablations
    results = ablation_exp.run_all_ablations(
        epochs=epochs,
        device=device
    )
    
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETED")
    print(f"Results saved to: {output_dir}")
    print("="*70 + "\n")
    
    # Show quick summary
    print("\nQuick Results:")
    print("-" * 50)
    for ablation_type, result in results.items():
        acc = result.get('test_accuracy', 0.0)
        print(f"{ablation_type:<30}: {acc:.2%}")
    print("-" * 50)


if __name__ == '__main__':
    main()
