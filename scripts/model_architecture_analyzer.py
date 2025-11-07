# model_architecture_analyzer.py
import torch
import pickle
import os
import sys
import re
from pathlib import Path
import pandas as pd
from collections import defaultdict

def setup_imports():
    """设置导入路径"""
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

def find_all_model_classes():
    """查找项目中所有的模型类定义"""
    model_dir = setup_imports()
    model_files = []
    
    # 查找所有可能的模型文件
    search_paths = [
        model_dir / 'models',
        model_dir / 'experiments',
        model_dir
    ]
    
    model_classes = {}
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        for py_file in search_path.rglob('*.py'):
            if py_file.name.startswith('__'):
                continue
                
            try:
                # 读取文件内容分析模型类
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找类定义
                class_matches = re.findall(r'class\s+(\w+)\s*\(\s*nn\.Module\s*\):', content)
                for class_name in class_matches:
                    if class_name not in ['Module', 'Sequential']:  # 排除基础类
                        relative_path = py_file.relative_to(model_dir)
                        model_classes[class_name] = {
                            'file': str(relative_path),
                            'full_path': str(py_file),
                            'class_name': class_name
                        }
                        print(f"📁 Found model class: {class_name} in {relative_path}")
                        
            except Exception as e:
                print(f"⚠️  Error reading {py_file}: {e}")
    
    return model_classes

def analyze_checkpoint_architecture(checkpoint_path):
    """分析检查点中的模型架构"""
    print(f"\n🔍 Analyzing checkpoint: {checkpoint_path}")
    
    try:
        # 尝试不同方式加载检查点
        if checkpoint_path.endswith('.pkl'):
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
        
        # 提取模型状态字典
        model_state_dict = None
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            print("📦 Using model_state_dict")
        elif 'model' in checkpoint:
            model_state_dict = checkpoint['model']
            print("📦 Using model key")
        elif all(not k.startswith(('epoch', 'best_', 'training_', 'config')) 
                for k in checkpoint.keys()):
            model_state_dict = checkpoint
            print("📦 Checkpoint is pure model state")
        else:
            print("❌ Could not find model state in checkpoint")
            return None
        
        # 分析参数
        param_analysis = {}
        total_params = 0
        
        for key, tensor in model_state_dict.items():
            param_analysis[key] = {
                'shape': list(tensor.shape),
                'numel': tensor.numel(),
                'dtype': str(tensor.dtype),
                'key_pattern': analyze_key_pattern(key)
            }
            total_params += tensor.numel()
        
        checkpoint_info = {
            'model_state_dict': model_state_dict,
            'param_analysis': param_analysis,
            'total_params': total_params,
            'num_layers': len(param_analysis),
            'checkpoint_keys': list(checkpoint.keys()),
            'param_keys': list(param_analysis.keys())
        }
        
        print(f"📊 Checkpoint analysis:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Number of layers: {len(param_analysis)}")
        print(f"   Parameter keys (first 5): {list(param_analysis.keys())[:5]}")
        
        return checkpoint_info
        
    except Exception as e:
        print(f"❌ Failed to analyze checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_key_pattern(key):
    """分析参数键的模式"""
    patterns = {
        'weight': 'weight' in key,
        'bias': 'bias' in key,
        'running_mean': 'running_mean' in key,
        'running_var': 'running_var' in key,
        'num_batches_tracked': 'num_batches_tracked' in key,
        'embedding': 'embedding' in key,
        'classifier': 'classifier' in key,
        'features': 'features' in key,
        'chaotic': 'chaotic' in key,
        'speaker': 'speaker' in key
    }
    
    # 提取层级信息
    parts = key.split('.')
    hierarchy = '.'.join(parts[:-1]) if len(parts) > 1 else 'root'
    param_name = parts[-1]
    
    return {
        'hierarchy': hierarchy,
        'param_name': param_name,
        'depth': len(parts) - 1,
        'patterns': patterns
    }

def create_current_model_analysis():
    """创建当前模型并分析其架构"""
    print(f"\n🔍 Analyzing current model architecture...")
    
    try:
        # 尝试导入模型创建函数
        from experiments.chaotic_experiment import ChaoticExperiment
        
        # 创建实验实例（使用与训练相同的配置）
        config = {
            'chaotic_system': 'lorenz',
            'model_type': 'full_chaotic',
            'num_speakers': 251,
            'batch_size': 32,
            'speaker_embedding_dim': 128,
            'embedding_hidden_dims': [64, 32],
            'pooling_type': 'comprehensive',
            'device': 'cpu'
        }
        
        experiment = ChaoticExperiment(
            config=config,
            experiment_name="architecture_analysis",
            output_dir="./temp_analysis",
            device='cpu'
        )
        
        experiment.setup()
        
        current_model = experiment.model
        current_state = current_model.state_dict()
        
        # 分析当前模型参数
        current_analysis = {}
        total_params = 0
        
        for key, tensor in current_state.items():
            current_analysis[key] = {
                'shape': list(tensor.shape),
                'numel': tensor.numel(),
                'dtype': str(tensor.dtype),
                'key_pattern': analyze_key_pattern(key)
            }
            total_params += tensor.numel()
        
        current_info = {
            'model': current_model,
            'model_state_dict': current_state,
            'param_analysis': current_analysis,
            'total_params': total_params,
            'num_layers': len(current_analysis),
            'param_keys': list(current_analysis.keys()),
            'model_class': current_model.__class__.__name__,
            'model_config': config
        }
        
        print(f"📊 Current model analysis:")
        print(f"   Model class: {current_model.__class__.__name__}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Number of layers: {len(current_analysis)}")
        print(f"   Parameter keys (first 5): {list(current_analysis.keys())[:5]}")
        
        return current_info
        
    except Exception as e:
        print(f"❌ Failed to create current model: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_comparison_table(checkpoint_info, current_info):
    """生成详细的对比表格"""
    print(f"\n📊 Generating comparison table...")
    
    checkpoint_params = checkpoint_info['param_analysis']
    current_params = current_info['param_analysis']
    
    # 创建对比数据
    comparison_data = []
    
    # 首先添加检查点中的所有参数
    for key, checkpoint_data in checkpoint_params.items():
        match_status = "❌ Not Found"
        current_shape = "N/A"
        shape_match = "N/A"
        suggested_mapping = ""
        
        if key in current_params:
            current_data = current_params[key]
            current_shape = str(current_data['shape'])
            shape_match = "✅" if checkpoint_data['shape'] == current_data['shape'] else "❌ Shape Mismatch"
            match_status = "✅ Exact Match" if shape_match == "✅" else "⚠️ Key Match"
        else:
            # 尝试找到可能的映射
            suggested_mapping = find_suggested_mapping(key, list(current_params.keys()))
        
        comparison_data.append({
            'Parameter Key': key,
            'Source': 'Checkpoint',
            'Shape': str(checkpoint_data['shape']),
            'Current Shape': current_shape,
            'Match Status': match_status,
            'Shape Match': shape_match,
            'Suggested Mapping': suggested_mapping,
            'Num Elements': checkpoint_data['numel']
        })
    
    # 添加当前模型中有但检查点中没有的参数
    for key, current_data in current_params.items():
        if key not in checkpoint_params:
            comparison_data.append({
                'Parameter Key': key,
                'Source': 'Current Model',
                'Shape': str(current_data['shape']),
                'Current Shape': str(current_data['shape']),
                'Match Status': "❌ Not in Checkpoint",
                'Shape Match': "N/A",
                'Suggested Mapping': "",
                'Num Elements': current_data['numel']
            })
    
    # 创建DataFrame
    df = pd.DataFrame(comparison_data)
    
    # 计算统计信息
    exact_matches = len(df[df['Match Status'] == '✅ Exact Match'])
    key_matches_shape_mismatch = len(df[df['Match Status'] == '⚠️ Key Match'])
    not_found = len(df[df['Match Status'] == '❌ Not Found'])
    not_in_checkpoint = len(df[df['Match Status'] == '❌ Not in Checkpoint'])
    
    total_checkpoint_params = sum(1 for item in comparison_data if item['Source'] == 'Checkpoint')
    total_current_params = len(current_params)
    
    print(f"\n📈 Match Statistics:")
    print(f"   Exact matches: {exact_matches}/{total_checkpoint_params} ({exact_matches/max(1,total_checkpoint_params)*100:.1f}%)")
    print(f"   Key matches (shape mismatch): {key_matches_shape_mismatch}")
    print(f"   Not found in current model: {not_found}")
    print(f"   Current model parameters not in checkpoint: {not_in_checkpoint}")
    
    return df

def find_suggested_mapping(checkpoint_key, current_keys):
    """为检查点键找到建议的映射"""
    
    # 常见的重命名模式
    rename_patterns = [
        (r'^features\.', 'chaotic_features.'),
        (r'^classifier\.', 'speaker_classifier.'),
        (r'^embedding\.', 'speaker_embedding.'),
        (r'^backbone\.', 'feature_extractor.'),
        (r'^fc\.', 'classifier.'),
        (r'^conv\.', 'features.conv.'),
        (r'\.weight$', '.weight'),
        (r'\.bias$', '.bias'),
    ]
    
    # 尝试直接重命名
    for pattern, replacement in rename_patterns:
        potential_key = re.sub(pattern, replacement, checkpoint_key)
        if potential_key in current_keys:
            return f"→ {potential_key}"
    
    # 尝试基于参数名称匹配
    key_parts = checkpoint_key.split('.')
    param_name = key_parts[-1]  # 参数名称（weight, bias等）
    
    for current_key in current_keys:
        current_parts = current_key.split('.')
        if current_parts[-1] == param_name:
            # 参数名称匹配，但路径不同
            return f"→ {current_key} (param name match)"
    
    # 尝试基于层级深度匹配
    checkpoint_depth = len(key_parts)
    for current_key in current_keys:
        current_depth = len(current_key.split('.'))
        if current_depth == checkpoint_depth and key_parts[-1] == current_key.split('.')[-1]:
            return f"→ {current_key} (depth match)"
    
    return "No obvious mapping found"

def generate_recommendations(df, checkpoint_info, current_info):
    """生成具体的修改建议"""
    print(f"\n💡 Generating recommendations...")
    
    recommendations = []
    
    # 分析不匹配的模式
    not_found_keys = df[df['Match Status'] == '❌ Not Found']['Parameter Key'].tolist()
    shape_mismatch_keys = df[df['Match Status'] == '⚠️ Key Match']['Parameter Key'].tolist()
    
    # 按层级分组不匹配的键
    hierarchy_groups = defaultdict(list)
    for key in not_found_keys:
        pattern = analyze_key_pattern(key)
        hierarchy_groups[pattern['hierarchy']].append(key)
    
    # 生成层级修改建议
    for hierarchy, keys in hierarchy_groups.items():
        if hierarchy:  # 非根层级
            recommendations.append({
                'type': 'RENAME_LAYER',
                'description': f"Rename layer hierarchy '{hierarchy}'",
                'affected_keys': keys[:3],  # 显示前3个受影响的键
                'suggestion': f"Consider renaming '{hierarchy}' to match current model structure"
            })
    
    # 分析形状不匹配
    for key in shape_mismatch_keys:
        checkpoint_shape = df[df['Parameter Key'] == key]['Shape'].iloc[0]
        current_shape = df[df['Parameter Key'] == key]['Current Shape'].iloc[0]
        
        recommendations.append({
            'type': 'SHAPE_MISMATCH',
            'description': f"Shape mismatch for '{key}'",
            'details': f"Checkpoint: {checkpoint_shape}, Current: {current_shape}",
            'suggestion': "Adjust layer dimensions in model definition"
        })
    
    # 总体建议
    match_rate = len(df[df['Match Status'] == '✅ Exact Match']) / len(df[df['Source'] == 'Checkpoint'])
    
    if match_rate < 0.1:
        recommendations.append({
            'type': 'ARCHITECTURE_OVERHAUL',
            'description': 'Complete architecture mismatch',
            'suggestion': 'Consider creating a new model adapter or retraining from scratch'
        })
    elif match_rate < 0.5:
        recommendations.append({
            'type': 'PARAMETER_MAPPING',
            'description': 'Significant parameter mapping needed',
            'suggestion': 'Implement parameter mapping in model loading logic'
        })
    else:
        recommendations.append({
            'type': 'MINOR_ADJUSTMENTS',
            'description': 'Most parameters match with minor adjustments needed',
            'suggestion': 'Focus on renaming mismatched layers'
        })
    
    return recommendations

def save_analysis_results(df, recommendations, checkpoint_info, current_info, output_dir):
    """保存分析结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细的对比表格
    excel_path = output_dir / 'model_architecture_comparison.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 主对比表
        df.to_excel(writer, sheet_name='Parameter Comparison', index=False)
        
        # 按匹配状态分组
        match_summary = df[df['Source'] == 'Checkpoint'].groupby('Match Status').size().reset_index()
        match_summary.columns = ['Match Status', 'Count']
        match_summary.to_excel(writer, sheet_name='Match Summary', index=False)
        
        # 建议表
        rec_df = pd.DataFrame(recommendations)
        rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
    
    # 保存文本报告
    report_path = output_dir / 'architecture_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("MODEL ARCHITECTURE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CHECKPOINT INFORMATION:\n")
        f.write(f"  Total parameters: {checkpoint_info['total_params']:,}\n")
        f.write(f"  Number of layers: {checkpoint_info['num_layers']}\n")
        f.write(f"  Checkpoint keys: {checkpoint_info['checkpoint_keys']}\n\n")
        
        f.write("CURRENT MODEL INFORMATION:\n")
        f.write(f"  Model class: {current_info['model_class']}\n")
        f.write(f"  Total parameters: {current_info['total_params']:,}\n")
        f.write(f"  Number of layers: {current_info['num_layers']}\n\n")
        
        f.write("MATCH STATISTICS:\n")
        exact_matches = len(df[df['Match Status'] == '✅ Exact Match'])
        total_checkpoint = len(df[df['Source'] == 'Checkpoint'])
        f.write(f"  Exact matches: {exact_matches}/{total_checkpoint} ({exact_matches/max(1,total_checkpoint)*100:.1f}%)\n\n")
        
        f.write("KEY RECOMMENDATIONS:\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec['type']}: {rec['description']}\n")
            f.write(f"   Suggestion: {rec['suggestion']}\n\n")
    
    print(f"✅ Analysis results saved to: {output_dir}")
    print(f"   📊 Excel comparison: {excel_path}")
    print(f"   📝 Text report: {report_path}")

def main():
    """主分析函数"""
    print("🔧 Model Architecture Analyzer")
    print("=" * 50)
    
    # 设置检查点路径
    checkpoint_path = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251103_134109/checkpoint_epoch_0000_20251104_074641.pkl"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # 步骤1: 查找所有模型类
    print("\n📁 STEP 1: Finding all model classes...")
    model_classes = find_all_model_classes()
    print(f"   Found {len(model_classes)} model classes")
    
    # 步骤2: 分析检查点架构
    checkpoint_info = analyze_checkpoint_architecture(checkpoint_path)
    if not checkpoint_info:
        return
    
    # 步骤3: 分析当前模型架构
    current_info = create_current_model_analysis()
    if not current_info:
        return
    
    # 步骤4: 生成对比表格
    comparison_df = generate_comparison_table(checkpoint_info, current_info)
    
    # 步骤5: 生成建议
    recommendations = generate_recommendations(comparison_df, checkpoint_info, current_info)
    
    # 步骤6: 保存结果
    output_dir = "model_architecture_analysis"
    save_analysis_results(comparison_df, recommendations, checkpoint_info, current_info, output_dir)
    
    # 步骤7: 显示关键发现
    print(f"\n🎯 KEY FINDINGS:")
    exact_matches = len(comparison_df[comparison_df['Match Status'] == '✅ Exact Match'])
    total_checkpoint = len(comparison_df[comparison_df['Source'] == 'Checkpoint'])
    match_rate = exact_matches / max(1, total_checkpoint)
    
    print(f"   Parameter match rate: {match_rate:.1%}")
    
    if match_rate < 0.1:
        print("   🚨 CRITICAL: Complete architecture mismatch")
        print("   💡 Recommendation: Major model restructuring needed")
    elif match_rate < 0.5:
        print("   ⚠️  WARNING: Significant architecture differences")
        print("   💡 Recommendation: Parameter mapping implementation needed")
    else:
        print("   ✅ GOOD: Mostly compatible architectures")
        print("   💡 Recommendation: Minor adjustments and renaming needed")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Review the detailed comparison in: {output_dir}/")
    print(f"   2. Implement the recommended changes in model files")
    print(f"   3. Test parameter loading with the modified models")

if __name__ == "__main__":
    main()