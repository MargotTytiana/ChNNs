# final_model_analyzer.py
import torch
import pickle
import os
import sys
import re
from pathlib import Path
import pandas as pd
from collections import defaultdict

def setup_imports_correctly():
    """正确设置导入路径"""
    # 获取当前脚本的绝对路径
    current_file = Path(__file__).resolve()
    
    # 尝试不同的路径设置
    possible_paths = [
        current_file.parent,  # 当前目录
        current_file.parent.parent,  # 父目录
        current_file.parent.parent.parent,  # 祖父目录
    ]
    
    for path in possible_paths:
        model_path = path / 'models'
        experiments_path = path / 'experiments'
        
        if model_path.exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            if str(model_path) not in sys.path:
                sys.path.insert(0, str(model_path))
            if str(experiments_path) not in sys.path:
                sys.path.insert(0, str(experiments_path))
            print(f"✅ Added to path: {path}")
            return path
    
    print("❌ Could not find model directory")
    return None

def analyze_checkpoint_correctly(checkpoint_path):
    """正确分析检查点结构"""
    print(f"\n🔍 Correctly analyzing checkpoint: {checkpoint_path}")
    
    try:
        # 加载检查点
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
        
        # 检查点结构分析
        model_data = checkpoint.get('model', {})
        print(f"📦 Model data type: {type(model_data)}")
        print(f"📦 Model data keys: {list(model_data.keys()) if isinstance(model_data, dict) else 'N/A'}")
        
        # 提取真正的模型状态字典
        model_state_dict = None
        if isinstance(model_data, dict) and 'model_state_dict' in model_data:
            model_state_dict = model_data['model_state_dict']
            print("✅ Found model_state_dict in model data")
        elif isinstance(model_data, dict) and any('weight' in key for key in model_data.keys()):
            # 如果model_data本身看起来像状态字典
            model_state_dict = model_data
            print("✅ Model data appears to be state dict")
        else:
            print("❌ Could not find model state dict")
            return None
        
        # 分析模型参数
        param_analysis = {}
        total_params = 0
        
        for key, value in model_state_dict.items():
            if isinstance(value, torch.Tensor):
                param_analysis[key] = {
                    'shape': list(value.shape),
                    'numel': value.numel(),
                    'dtype': str(value.dtype)
                }
                total_params += value.numel()
            else:
                print(f"⚠️  Key '{key}' is not a tensor: {type(value)}")
        
        checkpoint_info = {
            'model_state_dict': model_state_dict,
            'param_analysis': param_analysis,
            'total_params': total_params,
            'num_layers': len(param_analysis),
            'param_keys': list(param_analysis.keys())
        }
        
        print(f"📊 Checkpoint analysis:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Number of layers: {len(param_analysis)}")
        if param_analysis:
            print(f"   Parameter keys (first 5): {list(param_analysis.keys())[:5]}")
        
        return checkpoint_info
        
    except Exception as e:
        print(f"❌ Failed to analyze checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_model_with_direct_import():
    """通过直接导入创建模型"""
    print(f"\n🔍 Creating model with direct import...")
    
    # 设置正确的路径
    base_path = setup_imports_correctly()
    if not base_path:
        return None
    
    try:
        # 直接导入模型类
        print("🔧 Attempting direct imports...")
        
        # 尝试从 chaotic_network 导入
        try:
            from models.chaotic_network import ChaoticSpeakerRecognitionNetwork
            print("✅ Successfully imported ChaoticSpeakerRecognitionNetwork")
            
            config = {
                'chaotic_system': 'lorenz',
                'num_speakers': 251,
                'speaker_embedding_dim': 128,
                'embedding_hidden_dims': [64, 32],
                'pooling_type': 'comprehensive'
            }
            model = ChaoticSpeakerRecognitionNetwork(config)
            
            state_dict = model.state_dict()
            param_analysis = {}
            total_params = 0
            
            for key, tensor in state_dict.items():
                param_analysis[key] = {
                    'shape': list(tensor.shape),
                    'numel': tensor.numel(),
                    'dtype': str(tensor.dtype)
                }
                total_params += tensor.numel()
            
            model_info = {
                'model': model,
                'model_state_dict': state_dict,
                'param_analysis': param_analysis,
                'total_params': total_params,
                'num_layers': len(param_analysis),
                'model_class': 'ChaoticSpeakerRecognitionNetwork',
                'param_keys': list(param_analysis.keys())
            }
            
            print(f"✅ Successfully created ChaoticSpeakerRecognitionNetwork")
            print(f"   Parameters: {total_params:,}")
            print(f"   Layers: {len(param_analysis)}")
            print(f"   Example keys: {list(param_analysis.keys())[:3]}")
            
            return model_info
            
        except ImportError as e:
            print(f"❌ Failed to import ChaoticSpeakerRecognitionNetwork: {e}")
        
        # 尝试其他可能的模型类
        model_classes = [
            ('models.hybrid_models', 'TraditionalChaoticHybrid'),
            ('models.hybrid_models', 'ChaoticMLPHybrid'), 
            ('models.mlp_classifier', 'MLPNetwork'),
        ]
        
        for module_path, class_name in model_classes:
            try:
                module = __import__(module_path, fromlist=[class_name])
                model_class = getattr(module, class_name)
                print(f"✅ Successfully imported {class_name}")
                
                config = {'num_speakers': 251}
                model = model_class(config)
                
                state_dict = model.state_dict()
                param_analysis = {}
                total_params = 0
                
                for key, tensor in state_dict.items():
                    param_analysis[key] = {
                        'shape': list(tensor.shape),
                        'numel': tensor.numel(),
                        'dtype': str(tensor.dtype)
                    }
                    total_params += tensor.numel()
                
                model_info = {
                    'model': model,
                    'model_state_dict': state_dict,
                    'param_analysis': param_analysis,
                    'total_params': total_params,
                    'num_layers': len(param_analysis),
                    'model_class': class_name,
                    'param_keys': list(param_analysis.keys())
                }
                
                print(f"✅ Successfully created {class_name}")
                print(f"   Parameters: {total_params:,}")
                print(f"   Layers: {len(param_analysis)}")
                
                return model_info
                
            except ImportError as e:
                print(f"❌ Failed to import {class_name}: {e}")
            except Exception as e:
                print(f"❌ Failed to create {class_name}: {e}")
        
        return None
        
    except Exception as e:
        print(f"❌ Failed in model creation: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_simple_model_for_analysis():
    """创建一个简单模型用于分析"""
    print(f"\n🔍 Creating simple model for analysis...")
    
    try:
        import torch.nn as nn
        
        # 创建一个简单的测试模型来验证流程
        class SimpleTestModel(nn.Module):
            def __init__(self, num_speakers=251):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(230, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64)
                )
                self.classifier = nn.Linear(64, num_speakers)
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = SimpleTestModel()
        state_dict = model.state_dict()
        
        param_analysis = {}
        total_params = 0
        for key, tensor in state_dict.items():
            param_analysis[key] = {
                'shape': list(tensor.shape),
                'numel': tensor.numel(),
                'dtype': str(tensor.dtype)
            }
            total_params += tensor.numel()
        
        model_info = {
            'model': model,
            'model_state_dict': state_dict,
            'param_analysis': param_analysis,
            'total_params': total_params,
            'num_layers': len(param_analysis),
            'model_class': 'SimpleTestModel',
            'param_keys': list(param_analysis.keys())
        }
        
        print(f"✅ Created SimpleTestModel for analysis")
        print(f"   Parameters: {total_params:,}")
        print(f"   Layers: {len(param_analysis)}")
        print(f"   Parameter keys: {list(param_analysis.keys())}")
        
        return model_info
        
    except Exception as e:
        print(f"❌ Failed to create simple model: {e}")
        return None

def analyze_parameter_structure(checkpoint_info, model_info):
    """分析参数结构对比"""
    print(f"\n📊 Analyzing parameter structure...")
    
    checkpoint_params = checkpoint_info['param_analysis']
    current_params = model_info['param_analysis']
    
    print(f"🔍 Checkpoint parameter structure:")
    for key in list(checkpoint_params.keys())[:10]:  # 显示前10个
        print(f"   {key}: {checkpoint_params[key]['shape']}")
    
    print(f"🔍 Current model parameter structure:")
    for key in list(current_params.keys())[:10]:
        print(f"   {key}: {current_params[key]['shape']}")
    
    # 分析匹配情况
    checkpoint_keys = set(checkpoint_params.keys())
    current_keys = set(current_params.keys())
    
    exact_matches = checkpoint_keys & current_keys
    checkpoint_only = checkpoint_keys - current_keys
    current_only = current_keys - checkpoint_keys
    
    print(f"\n📈 Structure analysis:")
    print(f"   Exact matches: {len(exact_matches)}")
    print(f"   Checkpoint only: {len(checkpoint_only)}")
    print(f"   Current model only: {len(current_only)}")
    
    if checkpoint_only:
        print(f"   Checkpoint-only keys (first 5): {list(checkpoint_only)[:5]}")
    if current_only:
        print(f"   Current-only keys (first 5): {list(current_only)[:5]}")
    
    # 生成详细的对比数据
    comparison_data = []
    
    # 检查点参数
    for key in checkpoint_keys:
        status = "✅ Exact Match" if key in exact_matches else "❌ Not Found"
        checkpoint_shape = str(checkpoint_params[key]['shape'])
        current_shape = str(current_params[key]['shape']) if key in current_params else "N/A"
        
        comparison_data.append({
            'Parameter Key': key,
            'Source': 'Checkpoint',
            'Status': status,
            'Checkpoint Shape': checkpoint_shape,
            'Current Shape': current_shape,
            'Suggested Action': find_specific_mapping(key, current_keys)
        })
    
    # 当前模型特有参数
    for key in current_only:
        comparison_data.append({
            'Parameter Key': key,
            'Source': 'Current Model Only', 
            'Status': '❌ Not in Checkpoint',
            'Checkpoint Shape': 'N/A',
            'Current Shape': str(current_params[key]['shape']),
            'Suggested Action': 'New parameter in current model'
        })
    
    return pd.DataFrame(comparison_data)

def find_specific_mapping(checkpoint_key, current_keys):
    """为特定键找到映射建议"""
    
    # 分析键的结构
    parts = checkpoint_key.split('.')
    
    # 常见的映射模式
    mapping_patterns = [
        # 特征提取层
        (r'^features\.', 'chaotic_features.'),
        (r'^conv\.', 'features.conv.'),
        (r'^backbone\.', 'feature_extractor.'),
        
        # 分类器层
        (r'^classifier\.', 'speaker_classifier.'),
        (r'^fc\.', 'classifier.'),
        (r'^linear\.', 'classifier.'),
        
        # 嵌入层
        (r'^embedding\.', 'speaker_embedding.'),
        (r'^embed\.', 'embedding.'),
        
        # 参数类型
        (r'\.weight$', '.weight'),
        (r'\.bias$', '.bias'),
        (r'\.running_mean$', '.running_mean'),
        (r'\.running_var$', '.running_var'),
    ]
    
    # 尝试应用映射模式
    for pattern, replacement in mapping_patterns:
        potential_key = re.sub(pattern, replacement, checkpoint_key)
        if potential_key in current_keys:
            return f"Rename to: {potential_key}"
    
    # 基于参数名称的模糊匹配
    param_name = parts[-1] if parts else ""
    for current_key in current_keys:
        current_parts = current_key.split('.')
        if current_parts[-1] == param_name:
            return f"Possible match: {current_key}"
    
    return "No obvious mapping"

def generate_specific_recommendations(df, checkpoint_info, model_info):
    """生成具体的修改建议"""
    print(f"\n💡 Generating specific recommendations...")
    
    recommendations = []
    
    # 分析不匹配的模式
    not_found_df = df[df['Status'] == '❌ Not Found']
    
    # 按层级分组
    layer_patterns = defaultdict(list)
    for _, row in not_found_df.iterrows():
        key = row['Parameter Key']
        parts = key.split('.')
        if len(parts) > 1:
            layer_name = parts[0]
            layer_patterns[layer_name].append(key)
    
    # 为每个层级生成具体建议
    for layer_name, keys in layer_patterns.items():
        if keys:
            # 分析这个层级的模式
            sample_key = keys[0]
            parts = sample_key.split('.')
            
            if len(parts) >= 2:
                recommendation = {
                    'Layer': layer_name,
                    'Issue': f"Layer '{layer_name}' not found in current model",
                    'Affected Parameters': len(keys),
                    'Specific Action': f"Rename layer '{layer_name}' in current model to match checkpoint",
                    'Example Parameters': keys[:3],
                    'Code Change Example': f"# In model definition, change:\n# self.{layer_name} = ...\n# to match checkpoint structure"
                }
                recommendations.append(recommendation)
    
    # 总体建议
    exact_matches = len(df[df['Status'] == '✅ Exact Match'])
    total_checkpoint = len(df[df['Source'] == 'Checkpoint'])
    match_rate = exact_matches / total_checkpoint if total_checkpoint > 0 else 0
    
    if match_rate == 0:
        recommendations.append({
            'Layer': 'ALL',
            'Issue': 'Complete architecture mismatch',
            'Affected Parameters': 'All',
            'Specific Action': 'Major model restructuring or use different model class',
            'Example Parameters': ['All parameters'],
            'Code Change Example': '# Consider using a completely different model architecture\n# or implementing parameter mapping logic'
        })
    elif match_rate < 0.3:
        recommendations.append({
            'Layer': 'MULTIPLE',
            'Issue': 'Severe layer naming mismatch',
            'Affected Parameters': 'Most',
            'Specific Action': 'Rename multiple layers in model definition',
            'Example Parameters': ['Multiple layers'],
            'Code Change Example': '# Rename layers in models/chaotic_network.py\n# to match checkpoint parameter names'
        })
    
    return recommendations

def save_final_analysis(df, recommendations, checkpoint_info, model_info, output_dir):
    """保存最终分析结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存Excel报告
    excel_path = output_dir / 'final_architecture_analysis.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Parameter Comparison', index=False)
        
        # 建议表
        rec_df = pd.DataFrame(recommendations)
        rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
    
    # 保存详细的文本报告
    report_path = output_dir / 'final_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("FINAL MODEL ARCHITECTURE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SUMMARY:\n")
        exact_matches = len(df[df['Status'] == '✅ Exact Match'])
        total_checkpoint = len(df[df['Source'] == 'Checkpoint'])
        match_rate = exact_matches / total_checkpoint if total_checkpoint > 0 else 0
        
        f.write(f"  Checkpoint Parameters: {checkpoint_info['total_params']:,}\n")
        f.write(f"  Current Model Parameters: {model_info['total_params']:,}\n")
        f.write(f"  Parameter Match Rate: {match_rate:.1%}\n")
        f.write(f"  Current Model Class: {model_info['model_class']}\n\n")
        
        f.write("KEY FINDINGS:\n")
        if match_rate == 0:
            f.write("  🚨 COMPLETE MISMATCH: No parameters match between checkpoint and current model\n")
            f.write("  This suggests either:\n")
            f.write("  1. The wrong model class is being used\n")
            f.write("  2. The model architecture has changed completely\n")
            f.write("  3. There's a version mismatch\n")
        else:
            f.write(f"  {exact_matches}/{total_checkpoint} parameters match exactly\n\n")
        
        f.write("CHECKPOINT PARAMETER STRUCTURE:\n")
        for key in list(checkpoint_info['param_analysis'].keys())[:15]:
            shape = checkpoint_info['param_analysis'][key]['shape']
            f.write(f"  {key}: {shape}\n")
        
        f.write("\nCURRENT MODEL PARAMETER STRUCTURE:\n")
        for key in list(model_info['param_analysis'].keys())[:15]:
            shape = model_info['param_analysis'][key]['shape']
            f.write(f"  {key}: {shape}\n")
        
        f.write("\nSPECIFIC ACTIONS NEEDED:\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec['Layer']}: {rec['Issue']}\n")
            f.write(f"   Action: {rec['Specific Action']}\n")
            if 'Code Change Example' in rec:
                f.write(f"   Code: {rec['Code Change Example']}\n")
            f.write("\n")
    
    print(f"✅ Final analysis saved to: {output_dir}")
    print(f"   📊 Excel file: {excel_path}")
    print(f"   📝 Report: {report_path}")

def main():
    """主分析函数"""
    print("🔧 Final Model Architecture Analyzer")
    print("=" * 50)
    
    # 设置检查点路径
    checkpoint_path = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251103_134109/checkpoint_epoch_0000_20251104_074641.pkl"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # 步骤1: 正确分析检查点
    checkpoint_info = analyze_checkpoint_correctly(checkpoint_path)
    if not checkpoint_info:
        print("❌ Could not analyze checkpoint")
        return
    
    # 步骤2: 创建模型
    model_info = create_model_with_direct_import()
    if not model_info:
        print("⚠️  Could not import project models, creating simple test model")
        model_info = create_simple_model_for_analysis()
    
    if not model_info:
        print("❌ Could not create any model for analysis")
        return
    
    # 步骤3: 分析参数结构
    comparison_df = analyze_parameter_structure(checkpoint_info, model_info)
    
    # 步骤4: 生成建议
    recommendations = generate_specific_recommendations(comparison_df, checkpoint_info, model_info)
    
    # 步骤5: 保存最终分析
    output_dir = "final_architecture_analysis"
    save_final_analysis(comparison_df, recommendations, checkpoint_info, model_info, output_dir)
    
    # 显示关键结果
    print(f"\n🎯 FINAL RESULTS:")
    exact_matches = len(comparison_df[comparison_df['Status'] == '✅ Exact Match'])
    total_checkpoint = len(comparison_df[comparison_df['Source'] == 'Checkpoint'])
    match_rate = exact_matches / total_checkpoint if total_checkpoint > 0 else 0
    
    print(f"   Model Class: {model_info['model_class']}")
    print(f"   Parameter Match Rate: {match_rate:.1%}")
    
    if match_rate == 0:
        print("   🚨 CRITICAL: Complete architecture mismatch")
        print("   💡 You need to:")
        print("      1. Identify the correct model class used in the checkpoint")
        print("      2. Modify your current model to match that architecture")
        print("      3. Or retrain from scratch with the current architecture")
    else:
        print(f"   🔧 Focus on renaming {total_checkpoint - exact_matches} parameters")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Check the detailed report in: {output_dir}/")
    print(f"   2. Modify the model files based on the specific recommendations")
    print(f"   3. Test the parameter loading with the modified model")

if __name__ == "__main__":
    main()