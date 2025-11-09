import torch
import os
import sys
from pathlib import Path

def fix_imports():
    current_file = Path(__file__).resolve()
    model_dir = current_file.parent.parent  # models -> Model
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

from chaotic_network import ChaoticSpeakerRecognitionNetwork

def debug_classifier_mismatch():
    """
    专门调试分类器参数不匹配问题
    """
    print("🔍 开始分类器参数不匹配调试...")
    
    # 检查点路径
    checkpoint_path = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251103_134109/checkpoint_epoch_0000_20251104_074641.pkl"
    
    # 1. 加载检查点
    print("\n1. 加载检查点...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✅ 检查点加载成功")
        
        # 获取模型状态字典
        if 'model_state_dict' in checkpoint:
            checkpoint_sd = checkpoint['model_state_dict']
            print("📋 使用 model_state_dict")
        elif 'model' in checkpoint:
            checkpoint_sd = checkpoint['model']
            print("📋 使用 model")
        else:
            checkpoint_sd = checkpoint
            print("📋 使用整个检查点作为状态字典")
            
        print(f"检查点参数数量: {len(checkpoint_sd)}")
        
    except Exception as e:
        print(f"❌ 检查点加载失败: {e}")
        return
    
    # 2. 分析检查点中的分类器参数
    print("\n2. 分析检查点分类器参数...")
    classifier_keys = [k for k in checkpoint_sd.keys() if 'classifier' in k]
    print(f"检查点中分类器相关键: {len(classifier_keys)}")
    for key in classifier_keys:
        print(f"  {key}: {checkpoint_sd[key].shape}")
        if 'weight' in key:
            checkpoint_classifier_shape = checkpoint_sd[key].shape
            print(f"    → 检查点分类器形状: {checkpoint_classifier_shape}")
            print(f"    → 检查点分类器 speaker数量: {checkpoint_classifier_shape[0]}")
    
    # 3. 创建当前模型
    print("\n3. 创建当前模型...")
    try:
        # 根据检查点推断正确的speaker数量
        if 'classifier.weight' in checkpoint_sd:
            num_speakers_checkpoint = checkpoint_sd['classifier.weight'].shape[0]
            print(f"📊 检查点中的speaker数量: {num_speakers_checkpoint}")
        else:
            num_speakers_checkpoint = 251  # 默认值
        
        # 创建模型 - 使用检查点中的speaker数量
        model_kwargs = {
            'num_speakers': num_speakers_checkpoint,  # 关键：使用检查点中的数量
            'embedding_dim': 10,
            'mlsa_scales': 5,
            'rqa_radius_ratio': 0.1,
            'evolution_time': 0.5,
            'time_step': 0.01,
            'coupling_strength': 1.0,
            'noise_level': 0.001,
            'pooling_type': 'comprehensive',
            'speaker_embedding_dim': 128,
            'classifier_type': 'cosine',
            'device': 'cpu'
        }
        
        model = ChaoticSpeakerRecognitionNetwork(**model_kwargs)
        current_sd = model.state_dict()
        print(f"✅ 当前模型创建成功")
        print(f"当前模型参数数量: {len(current_sd)}")
        print(f"当前模型 speaker数量: {model.num_speakers}")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return
    
    # 4. 详细比较分类器参数
    print("\n4. 分类器参数详细比较...")
    
    # 检查分类器权重
    if 'classifier.weight' in checkpoint_sd and 'classifier.weight' in current_sd:
        checkpoint_weight = checkpoint_sd['classifier.weight']
        current_weight = current_sd['classifier.weight']
        
        print(f"检查点分类器权重: {checkpoint_weight.shape}")
        print(f"当前模型分类器权重: {current_weight.shape}")
        
        if checkpoint_weight.shape == current_weight.shape:
            print("✅ 分类器权重形状匹配!")
        else:
            print("❌ 分类器权重形状不匹配!")
            print(f"  检查点: {checkpoint_weight.shape}")
            print(f"  当前: {current_weight.shape}")
            
            # 尝试修复
            if checkpoint_weight.shape[0] != current_weight.shape[0]:
                print(f"  Speaker数量不匹配: 检查点={checkpoint_weight.shape[0]}, 当前={current_weight.shape[0]}")
    
    # 5. 尝试不同的加载策略
    print("\n5. 尝试不同的加载策略...")
    
    # 策略1: 严格模式
    print("策略1: 严格模式加载")
    try:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_sd, strict=True)
        print(f"✅ 严格模式加载成功")
        print(f"  缺失的键: {len(missing_keys)}")
        print(f"  意外的键: {len(unexpected_keys)}")
    except Exception as e:
        print(f"❌ 严格模式加载失败: {e}")
    
    # 策略2: 非严格模式
    print("\n策略2: 非严格模式加载")
    try:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_sd, strict=False)
        print(f"✅ 非严格模式加载成功")
        print(f"  缺失的键: {len(missing_keys)}")
        if missing_keys:
            print("  具体缺失键:")
            for key in missing_keys:
                print(f"    - {key}")
        print(f"  意外的键: {len(unexpected_keys)}")
        if unexpected_keys:
            print("  具体意外键:")
            for key in unexpected_keys:
                print(f"    - {key}")
    except Exception as e:
        print(f"❌ 非严格模式也失败: {e}")
    
    # 策略3: 手动修复分类器参数
    print("\n策略3: 手动修复分类器参数")
    try:
        # 创建修复后的状态字典
        fixed_sd = current_sd.copy()
        
        # 复制所有匹配的参数
        loaded_count = 0
        for key in checkpoint_sd.keys():
            if key in fixed_sd and checkpoint_sd[key].shape == fixed_sd[key].shape:
                fixed_sd[key] = checkpoint_sd[key]
                loaded_count += 1
            elif key == 'classifier.weight' and key in fixed_sd:
                # 特殊处理分类器权重
                checkpoint_weight = checkpoint_sd[key]
                current_weight = fixed_sd[key]
                
                min_speakers = min(checkpoint_weight.shape[0], current_weight.shape[0])
                min_features = min(checkpoint_weight.shape[1], current_weight.shape[1])
                
                # 复制匹配的部分
                fixed_sd[key][:min_speakers, :min_features] = checkpoint_weight[:min_speakers, :min_features]
                print(f"  🔧 部分复制分类器权重: {min_speakers} speakers, {min_features} features")
                loaded_count += 1
            else:
                print(f"  ⚠️ 跳过不匹配的参数: {key}")
        
        # 加载修复后的状态字典
        model.load_state_dict(fixed_sd, strict=True)
        print(f"✅ 手动修复加载成功")
        print(f"  成功加载参数: {loaded_count}/{len(checkpoint_sd)}")
        
    except Exception as e:
        print(f"❌ 手动修复失败: {e}")
    
    # 6. 生成修复建议
    print("\n6. 修复建议...")
    
    if 'classifier.weight' in checkpoint_sd:
        checkpoint_speakers = checkpoint_sd['classifier.weight'].shape[0]
        current_speakers = model.num_speakers
        
        if checkpoint_speakers != current_speakers:
            print(f"🔧 主要问题: Speaker数量不匹配")
            print(f"   检查点: {checkpoint_speakers} speakers")
            print(f"   当前模型: {current_speakers} speakers")
            print(f"   建议: 创建模型时设置 num_speakers={checkpoint_speakers}")
            
    # 7. 测试修复后的模型
    print("\n7. 测试修复后的模型...")
    try:
        # 创建测试输入
        batch_size = 2
        audio_length = 16000  # 1秒音频
        test_audio = torch.randn(batch_size, audio_length)
        
        # 前向传播测试
        with torch.no_grad():
            logits = model(test_audio)
            print(f"✅ 前向传播测试成功")
            print(f"  输入形状: {test_audio.shape}")
            print(f"  输出logits形状: {logits.shape}")
            print(f"  预测speaker数量: {logits.shape[1]}")
            
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")

if __name__ == "__main__":
    debug_classifier_mismatch()