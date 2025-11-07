# ultimate_checkpoint_fix.py
import torch
import pickle
import os
import sys

def ultimate_checkpoint_solution():
    """终极检查点解决方案"""
    print("="*70)
    print("ULTIMATE CHECKPOINT RECOVERY SOLUTION")
    print("="*70)
    
    # 定义所有可能的检查点路径
    checkpoint_paths = [
        "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_epoch_0000_20251030_092020.pkl",
        "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251026_015146/checkpoint_epoch_0000_20251027_103724.pkl"
    ]
    
    # 找到实际存在的检查点
    available_checkpoints = []
    for path in checkpoint_paths:
        if os.path.exists(path):
            available_checkpoints.append(path)
            print(f"✅ Found: {path} ({os.path.getsize(path)} bytes)")
    
    if not available_checkpoints:
        print("❌ No checkpoint files found!")
        return None
    
    # 使用第一个可用的检查点
    original_path = available_checkpoints[0]
    ultimate_path = original_path.replace('.pkl', '_ultimate.pth')
    
    print(f"\nUsing checkpoint: {original_path}")
    
    try:
        # 步骤1: 加载原始检查点
        print("\n1. Loading original checkpoint...")
        with open(original_path, 'rb') as f:
            original_checkpoint = pickle.load(f)
        
        print(f"   Original keys: {original_checkpoint.keys()}")
        
        # 步骤2: 提取模型状态
        print("\n2. Extracting model state...")
        if 'model' not in original_checkpoint:
            print("❌ No 'model' key in checkpoint")
            return None
        
        original_model_state = original_checkpoint['model']
        print(f"   Model keys: {len(original_model_state)}")
        
        # 步骤3: 创建当前模型以获取架构
        print("\n3. Creating current model architecture...")
        try:
            from chaotic_network import ChaoticSpeakerRecognitionNetwork
            current_model = ChaoticSpeakerRecognitionNetwork(
                num_speakers=251,
                device='cpu'
            )
            current_model_state = current_model.state_dict()
            print(f"   Current model keys: {len(current_model_state)}")
        except Exception as e:
            print(f"❌ Failed to create current model: {e}")
            return None
        
        # 步骤4: 智能参数迁移
        print("\n4. Migrating parameters intelligently...")
        migrated_state = {}
        match_stats = {
            'exact': 0,
            'shape_match': 0,
            'partial': 0,
            'missed': 0
        }
        
        for current_key, current_tensor in current_model_state.items():
            matched = False
            
            # 策略1: 精确匹配
            if current_key in original_model_state:
                if original_model_state[current_key].shape == current_tensor.shape:
                    migrated_state[current_key] = original_model_state[current_key]
                    match_stats['exact'] += 1
                    matched = True
                    print(f"   ✓ Exact: {current_key}")
            
            # 策略2: 形状匹配但键名不同
            if not matched:
                for orig_key, orig_tensor in original_model_state.items():
                    if (orig_tensor.shape == current_tensor.shape and 
                        any(part in orig_key for part in current_key.split('.'))):
                        migrated_state[current_key] = orig_tensor
                        match_stats['shape_match'] += 1
                        matched = True
                        print(f"   ≈ Shape: {orig_key} -> {current_key}")
                        break
            
            # 策略3: 部分匹配（子字符串）
            if not matched:
                for orig_key in original_model_state.keys():
                    common_parts = set(orig_key.split('.')) & set(current_key.split('.'))
                    if len(common_parts) >= 2:  # 至少有2个共同部分
                        if original_model_state[orig_key].shape == current_tensor.shape:
                            migrated_state[current_key] = original_model_state[orig_key]
                            match_stats['partial'] += 1
                            matched = True
                            print(f"   ~ Partial: {orig_key} -> {current_key}")
                            break
            
            # 策略4: 无法匹配，使用随机初始化
            if not matched:
                migrated_state[current_key] = current_tensor
                match_stats['missed'] += 1
                print(f"   × Missed: {current_key}")
        
        # 步骤5: 创建终极检查点
        print("\n5. Creating ultimate checkpoint...")
        ultimate_checkpoint = {
            'model_state_dict': migrated_state,
            'epoch': original_checkpoint.get('epoch', 0),
            'best_metric': original_checkpoint.get('metrics', {}).get('best_val_accuracy', 0.0),
            'best_epoch': original_checkpoint.get('epoch', 0),
            'migration_stats': match_stats,
            'total_parameters': sum(p.numel() for p in current_model.parameters()),
            'loaded_parameters': sum(p.numel() for p in migrated_state.values() if hasattr(p, 'numel')),
            'original_checkpoint': original_path
        }
        
        # 保存终极检查点
        torch.save(ultimate_checkpoint, ultimate_path)
        print(f"✅ Ultimate checkpoint saved: {ultimate_path}")
        
        # 步骤6: 验证
        print("\n6. Verifying ultimate checkpoint...")
        test_checkpoint = torch.load(ultimate_path, map_location='cpu')
        print(f"   Verified keys: {test_checkpoint.keys()}")
        
        # 加载到模型测试
        try:
            current_model.load_state_dict(test_checkpoint['model_state_dict'], strict=False)
            print("✅ Model loading test: SUCCESS (non-strict)")
        except Exception as e:
            print(f"⚠️  Model loading test: {e}")
        
        # 输出统计信息
        print("\n" + "="*50)
        print("MIGRATION STATISTICS")
        print("="*50)
        total_matched = match_stats['exact'] + match_stats['shape_match'] + match_stats['partial']
        total_keys = len(current_model_state)
        match_rate = total_matched / total_keys
        
        print(f"Exact matches: {match_stats['exact']}/{total_keys}")
        print(f"Shape matches: {match_stats['shape_match']}/{total_keys}") 
        print(f"Partial matches: {match_stats['partial']}/{total_keys}")
        print(f"Missed keys: {match_stats['missed']}/{total_keys}")
        print(f"Overall match rate: {match_rate:.1%}")
        
        if match_rate >= 0.7:
            print("🎉 EXCELLENT: High compatibility, should work well!")
        elif match_rate >= 0.3:
            print("👍 GOOD: Moderate compatibility, should provide good initialization")
        else:
            print("⚠️  LOW: Limited compatibility, consider training from scratch")
        
        return ultimate_path
        
    except Exception as e:
        print(f"❌ Ultimate solution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result_path = ultimate_checkpoint_solution()
    
    print("\n" + "="*70)
    if result_path:
        print("🎉 ULTIMATE SOLUTION SUCCESSFUL!")
        print(f"\nUse this checkpoint:")
        print(f"python train_chaotic.py ... --resume {result_path}")
        print(f"\nTraining command:")
        print(f"python train_chaotic.py --system lorenz --epochs 100 --save_models --model_types full_chaotic --data_dir /scratch/project_2003370/yueyao/dataset/train-clean-100/LibriSpeech/train-clean-100 --resume {result_path}")
    else:
        print("💥 ULTIMATE SOLUTION FAILED!")
        print("\nRecommendation: Train from scratch with adjusted learning rate")