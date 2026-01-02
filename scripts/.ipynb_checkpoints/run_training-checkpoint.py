#!/usr/bin/env python3
"""
C-HiLAP Training Script

Usage examples:
    # Basic training with default config
    python run_training.py
    
    # Training with YAML config
    python run_training.py --config chaotic_config_full.yaml
    
    # Training with preset
    python run_training.py --config chaotic_config_full.yaml --preset high_performance
    
    # Quick debug run
    python run_training.py --debug
    
    # With PLDA classifier
    python run_training.py --config chaotic_config_full.yaml --classifier plda
"""

import argparse
import sys
from pathlib import Path
import os
def fix_imports():
    current_file = Path(__file__).resolve()
    model_dir = current_file.parent.parent  # experiments -> Model
    paths = [
        str(model_dir),
        str(model_dir/'experiments'), 
        str(model_dir/'models'),
        str(model_dir/'features'),
        str(model_dir/'data'),
        str(model_dir/'utils')
    ]
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    return model_dir

MODEL_DIR = fix_imports()

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def parse_args():
    parser = argparse.ArgumentParser(
        description='C-HiLAP Chaotic Neural Network Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--preset', '-p',
        type=str,
        choices=['fast_debug', 'baseline', 'high_performance', 'robust'],
        default=None,
        help='Configuration preset to use'
    )

    parser.add_argument(
        '--preset_sync', '-s',
        type=str,
        choices=['experiment_A_full_sync', 'experiment_B_no_sync', 'experiment_C_sync_only', 'experiment_D_ce_baseline'],
        default=None,
        help='Configuration preset_sync to use'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Quick debug run with minimal settings'
    )
    
    parser.add_argument(
        '--classifier',
        type=str,
        choices=['linear', 'cosine', 'angular', 'plda'],
        default=None,
        help='Override classifier type'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=None,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default=None,
        help='Override data directory'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./experiments/outputs',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda, cuda:0, etc.)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Loss function toggles
    parser.add_argument(
        '--no-stability',
        action='store_true',
        help='Disable Lyapunov stability loss'
    )
    
    parser.add_argument(
        '--no-sync',
        action='store_true',
        help='Disable phase synchronization loss'
    )
    
    parser.add_argument(
        '--no-adversarial',
        action='store_true',
        help='Disable adversarial training'
    )
    
    parser.add_argument(
        '--bifurcation',
        action='store_true',
        help='Enable bifurcation control module'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("C-HiLAP: Chaotic Hierarchical Attractor Propagation")
    print("Speaker Recognition with Chaos Theory")
    print("=" * 70 + "\n")
    
    # ============ Load Configuration ============
    if args.debug:
        # Quick debug configuration
        from config_loader import get_fast_debug_config
        config = get_fast_debug_config()
        print("[MODE] Debug mode - using minimal settings")
        
    elif args.config:
        # Load from YAML file
        from config_loader import create_experiment_config
        config = create_experiment_config(
            args.config,
            preset=args.preset
        )
        
    else:
        # Use default configuration
        from config_loader import get_default_config
        config = get_default_config()
        print("[MODE] Using default configuration")
    
    # ============ Apply Command Line Overrides ============
    if args.classifier:
        config['classifier_type'] = args.classifier
        print(f"[OVERRIDE] Classifier: {args.classifier}")
        
    if args.epochs:
        config['num_epochs'] = args.epochs
        print(f"[OVERRIDE] Epochs: {args.epochs}")
        
    if args.batch_size:
        config['batch_size'] = args.batch_size
        print(f"[OVERRIDE] Batch size: {args.batch_size}")
        
    if args.data_dir:
        config['data_dir'] = args.data_dir
        print(f"[OVERRIDE] Data dir: {args.data_dir}")
        
    if args.device != 'auto':
        config['device'] = args.device
        
    config['seed'] = args.seed
    
    # Loss function toggles
    if args.no_stability:
        config['stability_loss'] = {'enabled': False}
        print("[OVERRIDE] Stability loss: DISABLED")
        
    if args.no_sync:
        config['sync_loss'] = {'enabled': False}
        print("[OVERRIDE] Sync loss: DISABLED")
        
    if args.no_adversarial:
        config['adversarial_training'] = {'enabled': False}
        print("[OVERRIDE] Adversarial training: DISABLED")
        
    if args.bifurcation:
        config['use_bifurcation_control'] = True
        print("[OVERRIDE] Bifurcation control: ENABLED")
    
    # ============ Initialize and Run Experiment ============
    print("\n" + "-" * 70)
    print("Initializing experiment...")
    print("-" * 70 + "\n")
    
    try:
        from chaotic_experiment import ChaoticExperiment
        
        experiment = ChaoticExperiment(
            config=config,
            experiment_name=config.get('experiment_name', 'chaotic_experiment'),
            output_dir=args.output_dir,
            device=config.get('device', 'auto'),
            seed=config.get('seed', 42)
        )
        
        # Setup experiment (create model, dataloaders, optimizer, etc.)
        print("Setting up experiment components...")
        experiment.setup()
        
        # Run training
        print("\nStarting training...\n")
        num_epochs = config.get('num_epochs', 100)
        experiment.train(num_epochs=num_epochs)
        
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print("=" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training stopped by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()