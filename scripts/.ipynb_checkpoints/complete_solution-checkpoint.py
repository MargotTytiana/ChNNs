# complete_solution.py
import torch
import pickle
import os
import sys

def complete_solution():
    """完整的解决方案：测试并修复检查点"""
    print("="*60)
    print("COMPLETE CHECKPOINT SOLUTION")
    print("="*60)
    
    # 定义路径
    original_path = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_epoch_0000_20251030_092020.pkl"
    compatible_path = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_compatible.pth"
    
    # 步骤1: 检查文件是否存在
    print("\n1. Checking checkpoint file...")
    if not os.path.exists(original_path):
        print(f"❌ Checkpoint not found: {original_path}")
        
        # 查找可能的检查点文件
        base_dir = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/"
        if os.path.exists(base_dir):
            print(f"Looking for checkpoints in: {base_dir}")
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(('.pkl', '.pth', '.pt')):
                        full_path = os.path.join(root, file)
                        size = os.path.getsize(full_path)
                        print(f"  Found: {full_path} ({size} bytes)")
        return False
    
    print(f"✅ Checkpoint found: {original_path}")
    print(f"   Size: {os.path.getsize(original_path)} bytes")
    
    # 步骤2: 测试加载
    print("\n2. Testing checkpoint loading...")
    checkpoint = None
    try:
        # 先尝试 torch.load
        checkpoint = torch.load(original_path, map_location='cpu', weights_only=False)
        print("✅ Loaded with torch.load")
    except Exception as e1:
        print(f"❌ torch.load failed: {e1}")
        try:
            # 再尝试 pickle
            with open(original_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print("✅ Loaded with pickle")
        except Exception as e2:
            print(f"❌ pickle.load failed: {e2}")
            return False
    
    if checkpoint is None:
        print("❌ Could not load checkpoint with any method")
        return False
    
    print(f"✅ Checkpoint keys: {checkpoint.keys()}")
    
    # 步骤3: 提取模型状态
    print("\n3. Extracting model state...")
    model_state_dict = None
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        print("✓ Found model_state_dict")
    elif 'model' in checkpoint:
        model_state_dict = checkpoint['model']
        print("✓ Found model (converting to model_state_dict)")
    else:
        print("❌ No model state found in checkpoint")
        return False
    
    print(f"✓ Model state dict has {len(model_state_dict)} keys")
    
    # 步骤4: 创建兼容检查点
    print("\n4. Creating compatible checkpoint...")
    compatible_checkpoint = {
        'model_state_dict': model_state_dict,
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', checkpoint.get('best_val_accuracy', 0.0)),
        'best_epoch': checkpoint.get('best_epoch', 0),
        'training_stats': checkpoint.get('training_stats', {})
    }
    
    # 添加优化器和调度器状态（如果存在）
    if 'optimizer_state_dict' in checkpoint:
        compatible_checkpoint['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
        print("✓ Added optimizer_state_dict")
    
    if 'scheduler_state_dict' in checkpoint:
        compatible_checkpoint['scheduler_state_dict'] = checkpoint['scheduler_state_dict']
        print("✓ Added scheduler_state_dict")
    
    # 保存
    torch.save(compatible_checkpoint, compatible_path)
    print(f"✅ Saved compatible checkpoint: {compatible_path}")
    
    # 步骤5: 验证新检查点
    print("\n5. Verifying new checkpoint...")
    try:
        test_checkpoint = torch.load(compatible_path, map_location='cpu')
        print(f"✅ Verification successful")
        print(f"   Keys: {test_checkpoint.keys()}")
        print(f"   Epoch: {test_checkpoint.get('epoch', 'N/A')}")
        print(f"   Best metric: {test_checkpoint.get('best_metric', 'N/A')}")
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = complete_solution()
    
    print("\n" + "="*60)
    if success:
        print("🎉 COMPLETE SOLUTION SUCCESSFUL!")
        print("\nNext steps:")
        print("1. Use the training command below")
        print("2. Make sure train_chaotic.py has the fixed checkpoint loading code")
        print("\nTraining command:")
        print("python train_chaotic.py --system lorenz --epochs 100 --save_models --model_types full_chaotic --data_dir /scratch/project_2003370/yueyao/dataset/train-clean-100/LibriSpeech/train-clean-100 --resume outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_compatible.pth")
    else:
        print("💥 COMPLETE SOLUTION FAILED!")
        print("\nTroubleshooting:")
        print("1. Check if the checkpoint file exists")
        print("2. Verify file permissions")
        print("3. Try manual inspection with: python -c \"import pickle; import torch; data = pickle.load(open('outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_epoch_0000_20251030_092020.pkl', 'rb')); print(data.keys())\"")