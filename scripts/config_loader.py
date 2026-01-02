"""
Configuration Loader for C-HiLAP Chaotic Neural Network.

Loads YAML configuration and converts it to the format expected by
ChaoticExperiment and ChaoticSpeakerRecognitionNetwork.

Usage:
    from config_loader import load_config, create_experiment_config
    
    # Load full config
    config = load_config('chaotic_config_full.yaml')
    
    # Or create experiment-ready config
    experiment_config = create_experiment_config('chaotic_config_full.yaml')
    experiment = ChaoticExperiment(experiment_config)
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def apply_preset(config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """Apply a preset configuration on top of base config."""
    if 'presets' not in config or preset_name not in config['presets']:
        print(f"[WARNING] Preset '{preset_name}' not found, using base config")
        return config
    
    preset = config['presets'][preset_name]
    
    def deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override into base."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
    
    merged = deep_merge(config, preset)
    print(f"[CONFIG] Applied preset: {preset_name}")
    return merged


def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested YAML config to flat dictionary expected by ChaoticExperiment.
    
    Converts:
        model:
          chaotic_embedding:
            system_type: "lorenz"
    To:
        chaotic_system: "lorenz"
    """
    flat = {}
    
    # ============ Data Configuration ============
    if 'data' in config:
        data = config['data']
        flat['data_dir'] = data.get('data_dir', './data')
        flat['train_split'] = data.get('train_split', 0.7)
        flat['val_split'] = data.get('val_split', 0.15)
        flat['num_workers'] = data.get('num_workers', 4)
    
    # ============ Audio Configuration ============
    if 'audio' in config:
        audio = config['audio']
        flat['sample_rate'] = audio.get('sample_rate', 16000)
        flat['max_audio_length'] = audio.get('max_audio_length', 3.0)
        flat['frame_length'] = audio.get('frame_length', 400)
        flat['hop_length'] = audio.get('hop_length', 160)
    
    # ============ Model Configuration ============
    if 'model' in config:
        model = config['model']
        flat['model_type'] = model.get('type', 'full_chaotic')
        flat['num_speakers'] = model.get('num_speakers', 251)
        
        # Phase space
        if 'phase_space' in model:
            ps = model['phase_space']
            flat['embedding_dim'] = ps.get('embedding_dim', 10)
            flat['delay_method'] = ps.get('delay_method', 'autocorr')
        
        # Chaotic features
        if 'chaotic_features' in model:
            cf = model['chaotic_features']
            if 'mlsa' in cf:
                flat['mlsa_scales'] = cf['mlsa'].get('scales', 5)
            if 'rqa' in cf:
                flat['rqa_radius_ratio'] = cf['rqa'].get('radius_ratio', 0.1)
        
        # Chaotic embedding
        if 'chaotic_embedding' in model:
            ce = model['chaotic_embedding']
            flat['chaotic_system'] = ce.get('system_type', 'lorenz')
            flat['evolution_time'] = ce.get('evolution_time', 0.5)
            flat['time_step'] = ce.get('time_step', 0.01)
            flat['coupling_strength'] = ce.get('coupling_strength', 1.0)
            flat['noise_level'] = ce.get('noise_level', 0.001)
        
        # Attractor pooling
        if 'attractor_pooling' in model:
            flat['pooling_type'] = model['attractor_pooling'].get('type', 'comprehensive')
        
        # Speaker embedding
        if 'speaker_embedding' in model:
            se = model['speaker_embedding']
            flat['speaker_embedding_dim'] = se.get('dim', 256)
            flat['embedding_hidden_dims'] = se.get('hidden_dims', [512, 256, 128])
        
        # Classifier
        if 'classifier' in model:
            clf = model['classifier']
            flat['classifier_type'] = clf.get('type', 'linear')
            flat['temperature'] = clf.get('temperature', 30.0)
            flat['margin'] = clf.get('margin', 0.35)
            
            if 'plda' in clf:
                flat['plda_latent_dim'] = clf['plda'].get('latent_dim', 128)
                flat['plda_length_norm'] = clf['plda'].get('use_length_norm', True)
    
    # ============ Optional Modules ============
    if 'optional_modules' in config:
        opt = config['optional_modules']
        
        # Bifurcation control
        if 'bifurcation_control' in opt:
            flat['use_bifurcation_control'] = opt['bifurcation_control'].get('enabled', False)
        
        # Differentiable features
        if 'differentiable_features' in opt:
            flat['differentiable_features'] = opt['differentiable_features']
    
    # ============ Loss Configuration ============
    if 'loss' in config:
        loss = config['loss']
        
        # Stability loss
        if 'stability' in loss:
            stab = loss['stability']
            flat['stability_loss'] = {
                'enabled': stab.get('enabled', True),
                'target_lyapunov_range': stab.get('target_lyapunov_range', [0.1, 2.0]),
                'trajectory_bound': stab.get('trajectory_bound', 50.0),
                'stability_weight': stab.get('weight', 0.1),
                'diversity_weight': stab.get('diversity_weight', 0.05),
                'collapse_threshold': stab.get('collapse_threshold', 0.1)
            }
        
        # Synchronization loss
        if 'synchronization' in loss:
            sync = loss['synchronization']
            flat['sync_loss'] = {
                'enabled': sync.get('enabled', True),
                'sync_weight': sync.get('sync_weight', 0.1),
                'desync_weight': sync.get('desync_weight', 0.1),
                'margin': sync.get('margin', 1.0),
                'sample_ratio': sync.get('sample_ratio', 0.3)
            }
        
        # Adversarial training
        if 'adversarial' in loss:
            adv = loss['adversarial']
            flat['adversarial_training'] = {
                'enabled': adv.get('enabled', True),
                'perturbation_scale': adv.get('perturbation_scale', 0.1),
                'parameter_noise': adv.get('parameter_noise', True),
                'trajectory_noise': adv.get('trajectory_noise', True),
                'adversarial_weight': adv.get('weight', 0.3),
                'warmup_epochs': adv.get('warmup_epochs', 5)
            }
    
    # ============ Training Configuration ============
    if 'training' in config:
        train = config['training']
        flat['batch_size'] = train.get('batch_size', 32)
        flat['num_epochs'] = train.get('num_epochs', 100)
        flat['gradient_clipping'] = train.get('gradient_clipping', 1.0)
        
        # Optimizer
        if 'optimizer' in train:
            opt = train['optimizer']
            flat['learning_rate'] = opt.get('learning_rate', 0.0005)
            flat['weight_decay'] = opt.get('weight_decay', 0.0001)
            flat['optimizer'] = {
                'type': opt.get('type', 'adamw'),
                'params': {
                    'betas': opt.get('betas', [0.9, 0.999]),
                    'eps': opt.get('eps', 1e-8),
                    'weight_decay': opt.get('weight_decay', 0.0001)
                }
            }
        
        # Scheduler
        if 'scheduler' in train:
            sched = train['scheduler']
            flat['scheduler'] = {
                'type': sched.get('type', 'cosine'),
                'params': {
                    'T_max': sched.get('T_max', train.get('num_epochs', 100)),
                    'eta_min': sched.get('eta_min', 1e-6)
                }
            }
        
        # Early stopping
        if 'early_stopping' in train:
            es = train['early_stopping']
            flat['early_stopping'] = {
                'patience': es.get('patience', 30)
            }
    
    # ============ Hardware Configuration ============
    if 'hardware' in config:
        hw = config['hardware']
        flat['device'] = hw.get('device', 'auto')
        flat['seed'] = hw.get('seed', 42)
    
    return flat


def load_config(
    filepath: Union[str, Path],
    preset: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        filepath: Path to YAML config file
        preset: Optional preset name to apply
        
    Returns:
        Raw configuration dictionary
    """
    config = load_yaml(filepath)
    
    # Check for preset in config or argument
    if preset is None and 'preset' in config:
        preset = config['preset']
    
    if preset:
        config = apply_preset(config, preset)
    
    return config


def create_experiment_config(
    filepath: Union[str, Path],
    preset: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load YAML config and convert to ChaoticExperiment format.
    
    Args:
        filepath: Path to YAML config file
        preset: Optional preset name to apply
        overrides: Optional dictionary of values to override
        
    Returns:
        Flattened configuration ready for ChaoticExperiment
        
    Example:
        config = create_experiment_config(
            'chaotic_config_full.yaml',
            preset='high_performance',
            overrides={'batch_size': 64, 'num_epochs': 200}
        )
        experiment = ChaoticExperiment(config)
    """
    # Load and optionally apply preset
    raw_config = load_config(filepath, preset)
    
    # Flatten to experiment format
    flat_config = flatten_config(raw_config)
    
    # Apply overrides
    if overrides:
        flat_config.update(overrides)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Model type: {flat_config.get('model_type', 'full_chaotic')}")
    print(f"Chaotic system: {flat_config.get('chaotic_system', 'lorenz')}")
    print(f"Num speakers: {flat_config.get('num_speakers', 251)}")
    print(f"Batch size: {flat_config.get('batch_size', 32)}")
    print(f"Epochs: {flat_config.get('num_epochs', 100)}")
    print(f"Classifier: {flat_config.get('classifier_type', 'linear')}")
    print("-" * 60)
    print("Loss functions enabled:")
    print(f"  - Stability: {flat_config.get('stability_loss', {}).get('enabled', False)}")
    print(f"  - Sync: {flat_config.get('sync_loss', {}).get('enabled', False)}")
    print(f"  - Adversarial: {flat_config.get('adversarial_training', {}).get('enabled', False)}")
    print("=" * 60 + "\n")
    
    return flat_config


def save_config(config: Dict[str, Any], filepath: Union[str, Path]):
    """Save configuration to YAML file."""
    filepath = Path(filepath)
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"[CONFIG] Saved configuration to: {filepath}")


# ============================================================
# Quick Start Functions
# ============================================================

def get_default_config() -> Dict[str, Any]:
    """Get default configuration without loading from file."""
    return {
        # Data
        'data_dir': './data',
        'train_split': 0.7,
        'val_split': 0.15,
        'num_workers': 4,
        
        # Audio
        'sample_rate': 16000,
        'max_audio_length': 3.0,
        'frame_length': 400,
        'hop_length': 160,
        
        # Model
        'model_type': 'full_chaotic',
        'num_speakers': 251,
        'embedding_dim': 10,
        'delay_method': 'autocorr',
        'mlsa_scales': 5,
        'rqa_radius_ratio': 0.1,
        'chaotic_system': 'lorenz',
        'evolution_time': 0.5,
        'time_step': 0.01,
        'coupling_strength': 1.0,
        'noise_level': 0.001,
        'pooling_type': 'comprehensive',
        'speaker_embedding_dim': 256,
        'embedding_hidden_dims': [512, 256, 128],
        'classifier_type': 'linear',
        'temperature': 30.0,
        'margin': 0.35,
        
        # Optional modules
        'use_bifurcation_control': False,
        
        # Training
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.0005,
        'weight_decay': 0.0001,
        'gradient_clipping': 1.0,
        
        # Loss functions
        'stability_loss': {
            'enabled': True,
            'target_lyapunov_range': [0.1, 2.0],
            'trajectory_bound': 50.0,
            'stability_weight': 0.1,
            'diversity_weight': 0.05,
            'collapse_threshold': 0.1
        },
        'sync_loss': {
            'enabled': True,
            'sync_weight': 0.1,
            'desync_weight': 0.1,
            'margin': 1.0,
            'sample_ratio': 0.3
        },
        'adversarial_training': {
            'enabled': True,
            'perturbation_scale': 0.1,
            'parameter_noise': True,
            'trajectory_noise': True,
            'adversarial_weight': 0.3,
            'warmup_epochs': 5
        },
        
        # Hardware
        'device': 'auto',
        'seed': 42
    }


def get_fast_debug_config() -> Dict[str, Any]:
    """Get minimal config for quick testing."""
    config = get_default_config()
    config.update({
        'batch_size': 8,
        'num_epochs': 5,
        'embedding_hidden_dims': [128, 64],
        'adversarial_training': {'enabled': False},
        'sync_loss': {'enabled': False},
        'stability_loss': {'enabled': True, 'stability_weight': 0.05}
    })
    return config


# ============================================================
# Main: Test configuration loading
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Test 1: Get default config
    print("\n[TEST 1] Default configuration:")
    default_config = get_default_config()
    print(f"Keys: {list(default_config.keys())[:10]}...")
    
    # Test 2: Load from YAML if file exists
    yaml_path = Path(__file__).parent / "chaotic_config_full.yaml"
    if yaml_path.exists():
        print(f"\n[TEST 2] Loading from: {yaml_path}")
        config = create_experiment_config(yaml_path)
        print(f"Loaded {len(config)} configuration keys")
    else:
        print(f"\n[TEST 2] YAML file not found: {yaml_path}")
    
    # Test 3: Fast debug config
    print("\n[TEST 3] Fast debug configuration:")
    debug_config = get_fast_debug_config()
    print(f"Batch size: {debug_config['batch_size']}")
    print(f"Epochs: {debug_config['num_epochs']}")
    
    print("\n✓ Configuration loader tests completed!")
