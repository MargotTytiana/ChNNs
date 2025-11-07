# simple_checkpoint_test.py
import torch
import pickle
import os

def simple_test():
    """简单的检查点测试，不使用命令行参数"""
    # 直接指定文件路径
    filepath = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_epoch_0000_20251030_092020.pkl"
    
    print(f"Testing checkpoint: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ Checkpoint file does not exist: {filepath}")
        
        # 列出目录内容帮助调试
        dir_path = os.path.dirname(filepath)
        if os.path.exists(dir_path):
            print(f"Contents of {dir_path}:")
            for item in os.listdir(dir_path):
                print(f"  {item}")
        return False
    
    print(f"File size: {os.path.getsize(filepath)} bytes")
    
    # 方法1: 尝试 torch.load
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        print("✅ Successfully loaded with torch.load")
        print(f"Keys: {checkpoint.keys()}")
        
        # 显示重要信息
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'best_metric' in checkpoint:
            print(f"Best metric: {checkpoint['best_metric']}")
        if 'model_state_dict' in checkpoint:
            print(f"Model state dict keys: {len(checkpoint['model_state_dict'])}")
        
        return True
    except Exception as e:
        print(f"❌ torch.load failed: {e}")
    
    # 方法2: 尝试 pickle
    try:
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        print("✅ Successfully loaded with pickle")
        print(f"Keys: {checkpoint.keys()}")
        return True
    except Exception as e:
        print(f"❌ pickle.load failed: {e}")
    
    return False

if __name__ == "__main__":
    success = simple_test()
    
    if success:
        print(f"\n🎉 Checkpoint test successful!")
    else:
        print(f"\n💥 Checkpoint test failed!")