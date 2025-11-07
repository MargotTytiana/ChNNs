# enhanced_fix_checkpoint.py
import torch
import pickle
import os
import sys
from pathlib import Path

def analyze_model_mismatch():
    """分析模型不匹配的具体原因"""
    print("🔍 Analyzing model architecture mismatch...")
    
    # 原始检查点路径
    original_path = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_epoch_0000_20251030_092020.pkl"
    compatible_path = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_enhanced.pth"
    
    if not os.path.exists(original_path):
        print(f"❌ Original checkpoint not found: {original_path}")
        return False
    
    try:
        # 加载原始检查点
        print("1. Loading original checkpoint...")
        with open(original_path, 'rb') as f:
            original = pickle.load(f)
        
        print(f"   Original checkpoint keys: {list(original.keys())}")
        
        # 提取模型状态
        if 'model' not in original:
            print("❌ No model state found in checkpoint")
            return False
        
        checkpoint_state = original['model']
        print(f"   Checkpoint model parameters: {len(checkpoint_state)}")
        print(f"   First 5 keys in checkpoint: {list(checkpoint_state.keys())[:5]}")
        
        # 尝试导入当前模型架构
        print("2. Loading current model architecture...")
        try:
            # 添加模型路径到系统路径
            model_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(model_dir))
            
            from experiments.chaotic_experiment import ChaoticExperiment
            from utils.reproducibility import set_seed
            
            # 创建当前模型实例
            set_seed(42)
            experiment = ChaoticExperiment(
                config={
                    'chaotic_system': 'lorenz',
                    'model_type': 'full_chaotic', 
                    'num_speakers': 251,
                    'batch_size': 32,
                    'device': 'cpu'
                },
                experiment_name="temp_model",
                output_dir="./temp",
                device='cpu'
            )
            experiment.setup()
            
            current_model = experiment.model
            current_state = current_model.state_dict()
            print(f"   Current model parameters: {len(current_state)}")
            print(f"   First 5 keys in current model: {list(current_state.keys())[:5]}")
            
            # 分析键匹配情况
            print("3. Analyzing key matching...")
            matched_keys = []
            unmatched_keys = []
            
            for key in current_state.keys():
                if key in checkpoint_state:
                    if current_state[key].shape == checkpoint_state[key].shape:
                        matched_keys.append(key)
                    else:
                        print(f"   Shape mismatch for {key}: current {current_state[key].shape} vs checkpoint {checkpoint_state[key].shape}")
                        unmatched_keys.append(key)
                else:
                    unmatched_keys.append(key)
            
            print(f"   Matched keys: {len(matched_keys)}")
            print(f"   Unmatched keys: {len(unmatched_keys)}")
            
            if matched_keys:
                print(f"   Example matched keys: {matched_keys[:3]}")
            
            # 创建增强的检查点
            print("4. Creating enhanced checkpoint...")
            enhanced_checkpoint = {
                'model_state_dict': checkpoint_state,
                'current_model_state_dict': current_state,  # 保存当前模型状态供参考
                'epoch': original.get('epoch', 0),
                'best_metric': original.get('metrics', {}).get('best_val_accuracy', 0.0),
                'best_epoch': original.get('epoch', 0),
                'matched_keys': matched_keys,
                'unmatched_keys': unmatched_keys,
                'checkpoint_info': {
                    'total_params_checkpoint': sum(p.numel() for p in checkpoint_state.values()),
                    'total_params_current': sum(p.numel() for p in current_state.values()),
                    'matched_ratio': len(matched_keys) / len(current_state) if current_state else 0
                }
            }
            
            # 保存增强检查点
            torch.save(enhanced_checkpoint, compatible_path)
            print(f"✅ Enhanced checkpoint saved: {compatible_path}")
            
            # 验证
            test_checkpoint = torch.load(compatible_path, map_location='cpu')
            print(f"✅ Verification successful")
            print(f"   Matched ratio: {test_checkpoint['checkpoint_info']['matched_ratio']:.1%}")
            print(f"   Checkpoint params: {test_checkpoint['checkpoint_info']['total_params_checkpoint']:,}")
            print(f"   Current params: {test_checkpoint['checkpoint_info']['total_params_current']:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to analyze current model: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Failed to create enhanced checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_fallback_checkpoint():
    """创建回退方案：只保存训练状态，不加载模型参数"""
    print("\n🔄 Creating fallback checkpoint...")
    
    original_path = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_epoch_0000_20251030_092020.pkl"
    fallback_path = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_fallback.pth"
    
    try:
        with open(original_path, 'rb') as f:
            original = pickle.load(f)
        
        # 只保存训练状态，不保存模型参数
        fallback_checkpoint = {
            'epoch': original.get('epoch', 0),
            'best_metric': original.get('metrics', {}).get('best_val_accuracy', 0.0),
            'best_epoch': original.get('epoch', 0),
            'training_state': {
                'optimizer': original.get('optimizer', {}),
                'scheduler': original.get('scheduler', {}),
                'metrics': original.get('metrics', {})
            },
            'note': 'Fallback checkpoint - model parameters not loaded due to architecture mismatch'
        }
        
        torch.save(fallback_checkpoint, fallback_path)
        print(f"✅ Fallback checkpoint saved: {fallback_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create fallback checkpoint: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Enhanced Checkpoint Recovery Tool")
    print("=" * 50)
    
    # 首先尝试详细分析
    success = analyze_model_mismatch()
    
    if not success:
        print("\n⚠️  Detailed analysis failed, trying fallback...")
        success = create_fallback_checkpoint()
    
    if success:
        print("\n🎉 RECOVERY OPTIONS:")
        print("1. If architecture analysis shows good match (>50%):")
        print("   python train_chaotic.py --system lorenz --epochs 100 --resume outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_enhanced.pth")
        
        print("\n2. If architecture mismatch is severe:")
        print("   python train_chaotic.py --system lorenz --epochs 100 --resume outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_fallback.pth")
        
        print("\n3. Start from scratch (recommended if mismatch > 80%):")
        print("   python train_chaotic.py --system lorenz --epochs 100")
        
    else:
        print("\n💥 All recovery attempts failed!")
        print("Recommendation: Start training from scratch")
        print("python train_chaotic.py --system lorenz --epochs 100")