#!/usr/bin/env python3
"""
Test checkpoint using the EXACT same evaluation function from training code.
This ensures 100% consistency with training evaluation.
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# ============================================================
# STEP 1: Setup paths
# ============================================================
# Project root is where the Model directory is
PROJECT_ROOT = Path("/scratch/project_2003370/yueyao/Model")
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
CONFIG_DIR = EXPERIMENT_DIR / "configs" / "base_chaotic"

# Add to Python path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

print("="*60)
print("TEST WITH TRAINING EVALUATION FUNCTION")
print("="*60)
print(f"Project root: {PROJECT_ROOT}")
print(f"Experiment dir: {EXPERIMENT_DIR}")
print(f"Config dir: {CONFIG_DIR}")
print()

# ============================================================
# STEP 2: Import training code
# ============================================================
try:
    from chaotic_experiment import ChaoticExperiment
    from models.chaotic_network import ChaoticSpeakerRecognitionNetwork
    print("✓ Successfully imported ChaoticExperiment")
    print("✓ Successfully imported ChaoticSpeakerRecognitionNetwork")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print("\nTrying alternative import...")
    try:
        sys.path.insert(0, str(CONFIG_DIR))
        from chaotic_experiment import ChaoticExperiment
        from models.chaotic_network import ChaoticSpeakerRecognitionNetwork
        print("✓ Successfully imported ChaoticExperiment (alternative path)")
        print("✓ Successfully imported ChaoticSpeakerRecognitionNetwork")
    except ImportError as e2:
        print(f"✗ Failed again: {e2}")
        print("\nPlease check that files exist")
        sys.exit(1)

print()

# ============================================================
# STEP 3: Load training configuration
# ============================================================
CONFIG_FILE = CONFIG_DIR / "base_chaotic.yaml"

print(f"Loading config from: {CONFIG_FILE}")

if not CONFIG_FILE.exists():
    print(f"✗ Config file not found!")
    print(f"  Expected: {CONFIG_FILE}")
    print("\nSearching for yaml files...")
    yaml_files = list(CONFIG_DIR.glob("*.yaml"))
    if yaml_files:
        print(f"  Found {len(yaml_files)} yaml files:")
        for f in yaml_files:
            print(f"    - {f.name}")
        CONFIG_FILE = yaml_files[0]
        print(f"\n  Using: {CONFIG_FILE}")
    else:
        print("  No yaml files found!")
        sys.exit(1)

with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

print("✓ Config loaded successfully")
print()

# ============================================================
# STEP 4: Update paths for your dataset
# ============================================================
DATASET_PATH = "/scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2"
CHECKPOINT_PATH = "/scratch/project_2003370/yueyao/Model/experiments/configs/base_chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20260122_131044/checkpoint_epoch_0092.pt"

print("Updating config with your paths...")
print(f"  Dataset: {DATASET_PATH}")
print(f"  Checkpoint: {CHECKPOINT_PATH}")

# 同时更新所有可能的数据路径配置键
config['data_dir'] = DATASET_PATH
config['dataset_path'] = DATASET_PATH

if 'data' not in config:
    config['data'] = {}
config['data']['dataset_path'] = DATASET_PATH

if 'dataset' not in config:
    config['dataset'] = {}
config['dataset']['path'] = DATASET_PATH
config['dataset']['num_speakers'] = 26
config['dataset']['min_samples_per_speaker'] = 2
config['dataset']['max_samples_per_speaker'] = 30

# 训练相关配置
if 'training' not in config:
    config['training'] = {}
config['training']['num_epochs'] = 0
config['training']['epochs'] = 0

# 硬件配置
if 'hardware' not in config:
    config['hardware'] = {}
config['hardware']['device'] = 'cpu'

# 添加所有ChaoticExperiment必需的配置
required_configs = {
    'num_speakers': 26,
    'batch_size': 32,
    'chaotic_system': 'lorenz',
    'sample_rate': 16000,
    'embedding_dim': 10,
    'speaker_embedding_dim': 256,
    'classifier_type': 'cosine',
    'evolution_time': 0.5,
    'time_step': 0.01,
    'pooling_type': 'comprehensive',
    'mlsa_scales': 5,
    'rqa_radius_ratio': 0.1,
    'frame_length': 400,
    'hop_length': 160,
    'max_audio_length': 3.0,
    'min_audio_length': 0.5
}
config.update(required_configs)
config['data_dir'] = DATASET_PATH
config['dataset_path'] = DATASET_PATH

print("✓ Config updated")
print()

# ============================================================
# STEP 5: Create experiment and load checkpoint
# ============================================================
print("Creating ChaoticExperiment...")
try:
    experiment = ChaoticExperiment(config=config)

    experiment.criterion = torch.nn.CrossEntropyLoss()
    
    print("✓ ChaoticExperiment created successfully with CrossEntropyLoss")
except Exception as e:
    print(f"✗ Failed to create experiment: {e}")
    sys.exit(1)

    
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 初始化模型
print("Initializing model...")
try:
    experiment.model = ChaoticSpeakerRecognitionNetwork(
        num_speakers=26,
        embedding_dim=10,
        speaker_embedding_dim=256,
        embedding_hidden_dims=[512, 256, 128],
        classifier_type='cosine',
        device='cpu'
    )
    print("✓ Model created")
    
    # 初始化数据加载器
    experiment.create_dataloaders()
    # 手动创建test_loader
    from torch.utils.data import DataLoader
    from data.dataset_loader import create_chaotic_speaker_dataset
    
    if experiment.test_loader is None:
        # 重新加载数据集
        dataset = create_chaotic_speaker_dataset(
            dataset_path=DATASET_PATH,
            min_samples_per_speaker=2,
            max_samples_per_speaker=30,
            target_num_speakers=26
        )
        splits = dataset.create_data_splits(0.7, 0.15, 0.15)
        test_dataset = splits['test'].create_pytorch_dataset()


        def collate_fn(batch):
            """Fix audio to 3 seconds (48000 samples @ 16kHz)"""
            audios, labels = zip(*batch)
            
            target_length = 48000  # 3秒 @ 16kHz
            
            fixed_audios = []
            for audio in audios:
                if audio.shape[0] > target_length:
                    # 裁剪
                    audio = audio[:target_length]
                elif audio.shape[0] < target_length:
                    # Padding
                    padding = torch.zeros(target_length - audio.shape[0])
                    audio = torch.cat([audio, padding])
                fixed_audios.append(audio)
            
            return torch.stack(fixed_audios), torch.tensor(labels)
        
        experiment.test_loader = DataLoader(
            test_dataset, 
            batch_size=32, 
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn  # 添加这个
        )

    print("✓ Dataloaders initialized")
    
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Loading checkpoint...")
print(f"  Path: {CHECKPOINT_PATH}")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"✗ Checkpoint not found!")
    print(f"  Looking for: {CHECKPOINT_PATH}")
    sys.exit(1)

try:
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    # Print checkpoint info
    print("\nCheckpoint information:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Metrics: {checkpoint.get('metrics', 'N/A')}")
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        experiment.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✓ Model weights loaded")
    else:
        print("✗ No model_state_dict in checkpoint")
        sys.exit(1)
    
    # Move model to CPU and set eval mode
    experiment.model = experiment.model.cpu()
    experiment.model.eval()
    experiment.device = 'cpu'
    
    # 强制所有子模块到CPU
    for module in experiment.model.modules():
        module.to('cpu')
    
    print("✓ Model ready for evaluation (on CPU)")
    
except Exception as e:
    print(f"✗ Failed to load checkpoint: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================
# STEP 6: Run evaluation using training code's test function
# ============================================================
print("="*60)
print("RUNNING EVALUATION (using training code's test() function)")
print("="*60)
print()

try:
    # This uses the EXACT same evaluation logic as during training!
    test_metrics = experiment.test()
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name:20s}: {metric_value:.4f}")
        else:
            print(f"{metric_name:20s}: {metric_value}")
    
    print()
    
    # Highlight the accuracy
    if 'accuracy' in test_metrics:
        accuracy = test_metrics['accuracy']
        print("="*60)
        print(f"ACCURACY: {accuracy*100:.2f}%")
        print("="*60)
        
        # Compare with training
        print("\nComparison:")
        print(f"  Training accuracy: 94.92%")
        print(f"  Current test:      {accuracy*100:.2f}%")
        
        if abs(accuracy - 0.9492) < 0.05:
            print("\n✓ Accuracy matches training! Checkpoint is good.")
        else:
            print("\n⚠ Accuracy differs from training significantly.")
            print("  This might indicate:")
            print("  1. Different test set")
            print("  2. Checkpoint loaded incorrectly")
            print("  3. Dataset mismatch")
    
except Exception as e:
    print(f"\n✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "="*60)
    print("ATTEMPTING MANUAL EVALUATION")
    print("="*60)
    
    # Fallback: manual evaluation
    try:
        from torch.utils.data import DataLoader
        
        print("\nManual evaluation using test_loader...")
        
        experiment.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(experiment.test_loader):
                # Move batch to CPU
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    audio, labels = batch
                    audio = audio.cpu()
                    labels = labels.cpu()
                    
                    # Forward pass
                    outputs = experiment.model(audio, labels=None, return_intermediates=False)
                    
                    # Get predictions
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    _, predicted = torch.max(outputs, 1)
                    
                    # Update metrics
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
                    
                    if (batch_idx + 1) % 10 == 0:
                        current_acc = 100.0 * total_correct / total_samples
                        print(f"  Batch {batch_idx+1}: accuracy so far = {current_acc:.2f}%")
        
        final_accuracy = 100.0 * total_correct / total_samples
        print(f"\n✓ Manual evaluation completed")
        print(f"  Total samples: {total_samples}")
        print(f"  Correct: {total_correct}")
        print(f"  Accuracy: {final_accuracy:.2f}%")
        
    except Exception as e2:
        print(f"✗ Manual evaluation also failed: {e2}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)