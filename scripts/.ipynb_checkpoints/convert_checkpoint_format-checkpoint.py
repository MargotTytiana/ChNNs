# convert_checkpoint_format.py
import torch
import pickle
import os
import sys

def convert_to_standard_format(old_checkpoint_path, new_checkpoint_path):
    """将旧格式检查点转换为标准格式"""
    try:
        # 加载旧检查点
        with open(old_checkpoint_path, 'rb') as f:
            old_checkpoint = pickle.load(f)
        
        print("Old checkpoint keys:", old_checkpoint.keys())
        
        # 创建标准格式检查点
        new_checkpoint = {
            'epoch': old_checkpoint.get('epoch', 0),
            'model_state_dict': old_checkpoint.get('model', {}),
            'optimizer_state_dict': None,  # 旧格式可能没有
            'scheduler_state_dict': None,  # 旧格式可能没有
            'best_val_accuracy': old_checkpoint.get('metrics', {}).get('best_val_accuracy', 0.0),
            'best_val_loss': old_checkpoint.get('metrics', {}).get('best_val_loss', float('inf')),
            'best_epoch': old_checkpoint.get('metrics', {}).get('best_epoch', 0),
            'training_stats': old_checkpoint.get('additional_info', {}),
            'timestamp': old_checkpoint.get('timestamp', 'unknown')
        }
        
        # 保存为标准格式
        torch.save(new_checkpoint, new_checkpoint_path, pickle_protocol=2)
        print(f"✅ Converted checkpoint saved: {new_checkpoint_path}")
        
        # 验证转换
        test_checkpoint = torch.load(new_checkpoint_path, map_location='cpu')
        print("New checkpoint keys:", test_checkpoint.keys())
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) > 1:
        old_path = sys.argv[1]
    else:
        old_path = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_epoch_0000_20251030_092020.pkl"
    
    if len(sys.argv) > 2:
        new_path = sys.argv[2]
    else:
        new_path = old_path.replace('.pkl', '_standard.pth')
    
    print(f"Converting checkpoint format...")
    print(f"From: {old_path}")
    print(f"To: {new_path}")
    
    success = convert_to_standard_format(old_path, new_path)
    
    if success:
        print(f"\n🎉 Conversion successful!")
        print(f"Now you can use the converted checkpoint with:")
        print(f"python train_chaotic.py ... --resume {new_path}")
    else:
        print(f"\n💥 Conversion failed!")

if __name__ == "__main__":
    main()