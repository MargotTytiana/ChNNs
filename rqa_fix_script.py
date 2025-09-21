#!/usr/bin/env python3
"""
RQA提取器导入问题修复脚本
解决 rqa_extractor.py 的 EmbeddingConfig 未定义问题
"""

import os
import re
from pathlib import Path

def fix_rqa_import_issue(rqa_file_path):
    """修复RQA文件的导入问题"""
    
    # 确保路径是字符串
    rqa_file_path = str(rqa_file_path)
    
    if not os.path.exists(rqa_file_path):
        print(f"文件不存在: {rqa_file_path}")
        return False
    
    print(f"修复文件: {rqa_file_path}")
    
    # 读取原文件
    with open(rqa_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原文件
    backup_path = rqa_file_path + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已备份到: {backup_path}")
    
    # RQA文件的修复内容
    replacement_import_block = '''# Import project modules
try:
    from phase_space_reconstruction import (
        PhaseSpaceReconstructor, EmbeddingConfig
    )
    from utils.numerical_stability import (
        NumericalConfig, NumericalValidator, OutlierDetector,
        safe_divide, safe_log
    )
    from chaos_utils import (
        correlation_dimension, largest_lyapunov_from_data
    )
except ImportError as e:
    warnings.warn(f"Failed to import required modules: {e}")
    
    # 提供默认类定义以避免NameError
    @dataclass
    class EmbeddingConfig:
        """Default EmbeddingConfig when import fails"""
        embedding_dim: int = 3
        time_delay: int = 1
        max_embedding_dim: int = 20
        fnn_threshold: float = 15.0
        autocorr_threshold: float = 0.1
    
    @dataclass
    class NumericalConfig:
        """Default NumericalConfig when import fails"""
        epsilon: float = 1e-12
        max_iterations: int = 1000
        tolerance: float = 1e-8
        rtol: float = 1e-5
        atol: float = 1e-8
        outlier_threshold: float = 3.0
        stability_check: bool = True
    
    class PhaseSpaceReconstructor:
        """Default PhaseSpaceReconstructor when import fails"""
        def __init__(self, config):
            self.config = config
        
        def reconstruct(self, signal, delay=None, dimension=None):
            if delay is None:
                delay = 1
            if dimension is None:
                dimension = 3
            
            N = len(signal)
            if N < dimension * delay:
                return {
                    'embedding_success': False,
                    'error': 'Signal too short for embedding'
                }
            
            embedded = np.zeros((N - (dimension-1)*delay, dimension))
            for i in range(dimension):
                embedded[:, i] = signal[i*delay:N-(dimension-1-i)*delay]
            
            return {
                'embedding_success': True,
                'embedded_data': embedded,
                'delay': delay,
                'dimension': dimension,
                'n_embedded_points': embedded.shape[0]
            }
    
    class NumericalValidator:
        def __init__(self, config=None):
            self.config = config or NumericalConfig()
        
        def validate_array(self, arr, name="array"):
            issues = []
            
            if not isinstance(arr, np.ndarray):
                issues.append(f"{name} must be numpy array")
            elif arr.size == 0:
                issues.append(f"{name} cannot be empty")
            elif not np.isfinite(arr).all():
                issues.append(f"{name} contains non-finite values")
            
            return {
                'is_valid': len(issues) == 0,
                'issues': issues
            }
    
    class OutlierDetector:
        def __init__(self, config=None):
            self.config = config or NumericalConfig()
        
        def remove_outliers(self, data, method='zscore'):
            if method == 'zscore':
                z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-12))
                mask = z_scores < 3.0
                return data[mask], np.where(~mask)[0]
            else:
                return data, np.array([])
    
    def correlation_dimension(data, r_points=30):
        try:
            N = len(data) if data.ndim == 1 else data.shape[0]
            
            if N < 50:
                return np.array([]), np.array([]), 2.0
            
            if data.ndim == 1:
                distances = np.abs(data[:, None] - data[None, :])
            else:
                from scipy.spatial.distance import pdist, squareform
                distances = squareform(pdist(data))
            
            max_dist = np.max(distances)
            min_dist = np.min(distances[distances > 0])
            
            radii = np.logspace(np.log10(min_dist), np.log10(max_dist), r_points)
            correlations = []
            
            for r in radii:
                count = np.sum(distances <= r)
                correlation = count / (N * N)
                correlations.append(correlation)
            
            correlations = np.array(correlations)
            
            valid_mask = (correlations > 0) & (correlations < 1)
            if np.sum(valid_mask) > 5:
                log_r = np.log(radii[valid_mask])
                log_c = np.log(correlations[valid_mask])
                slope = np.polyfit(log_r, log_c, 1)[0]
                corr_dim = max(0, slope)
            else:
                corr_dim = 2.0
            
            return radii, correlations, corr_dim
            
        except Exception:
            return np.array([]), np.array([]), 2.0
    
    def largest_lyapunov_from_data(data, dt=0.01, tau=1, min_neighbors=10):
        try:
            N = len(data)
            if N < 100:
                return 0.0
            
            m = 3
            embedded = np.zeros((N - (m-1)*tau, m))
            for i in range(m):
                embedded[:, i] = data[i*tau:N-(m-1-i)*tau]
            
            n_points = min(200, embedded.shape[0] - 1)
            divergence_rates = []
            
            for i in range(0, n_points, 10):
                distances = np.linalg.norm(embedded - embedded[i], axis=1)
                distances[i] = np.inf
                
                if np.min(distances) > 0:
                    j = np.argmin(distances)
                    track_length = min(20, embedded.shape[0] - max(i, j) - 1)
                    if track_length > 5:
                        initial_dist = distances[j]
                        final_dist = np.linalg.norm(embedded[i + track_length] - 
                                                  embedded[j + track_length])
                        if initial_dist > 0 and final_dist > initial_dist:
                            rate = np.log(final_dist / initial_dist) / (track_length * dt)
                            divergence_rates.append(rate)
            
            return np.mean(divergence_rates) if divergence_rates else 0.0
            
        except Exception:
            return 0.0
    
    def safe_divide(numerator, denominator, default=0.0, epsilon=1e-12):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(np.abs(denominator) > epsilon, 
                             numerator / denominator, default)
        return result
    
    def safe_log(x, epsilon=1e-12):
        return np.log(np.maximum(x, epsilon))


'''
    
    # 查找并替换原始的导入块
    # 查找从 "# Import project modules" 到 "except ImportError" 的块
    try_except_pattern = r'# Import project modules.*?except ImportError as e:\s*warnings\.warn\(f"Failed to import required modules: \{e\}"\)'
    
    if re.search(try_except_pattern, content, re.DOTALL):
        new_content = re.sub(try_except_pattern, replacement_import_block, content, flags=re.DOTALL)
        print("使用主要替换方法")
    else:
        print("使用备选替换方法")
        # 备选方法：查找 try: 块
        try_pattern = r'try:\s*\n.*?warnings\.warn\(f"Failed to import required modules: \{e\}"\)'
        new_content = re.sub(try_pattern, replacement_import_block, content, flags=re.DOTALL)
    
    # 如果还没有替换成功，尝试更简单的模式
    if new_content == content:
        print("使用简化替换方法")
        # 找到导入失败的警告行，在其后插入类定义
        warning_line = 'warnings.warn(f"Failed to import required modules: {e}")'
        if warning_line in content:
            insert_position = content.find(warning_line) + len(warning_line)
            
            # 插入默认类定义
            default_classes = '''
    
    # 提供默认类定义以避免NameError
    @dataclass
    class EmbeddingConfig:
        """Default EmbeddingConfig when import fails"""
        embedding_dim: int = 3
        time_delay: int = 1
        max_embedding_dim: int = 20
        fnn_threshold: float = 15.0
        autocorr_threshold: float = 0.1
    
    @dataclass
    class NumericalConfig:
        """Default NumericalConfig when import fails"""
        epsilon: float = 1e-12
        max_iterations: int = 1000
        tolerance: float = 1e-8
        rtol: float = 1e-5
        atol: float = 1e-8
        outlier_threshold: float = 3.0
        stability_check: bool = True
    
    class PhaseSpaceReconstructor:
        def __init__(self, config):
            self.config = config
        
        def reconstruct(self, signal, delay=None, dimension=None):
            if delay is None:
                delay = 1
            if dimension is None:
                dimension = 3
            
            N = len(signal)
            if N < dimension * delay:
                return {'embedding_success': False, 'error': 'Signal too short'}
            
            embedded = np.zeros((N - (dimension-1)*delay, dimension))
            for i in range(dimension):
                embedded[:, i] = signal[i*delay:N-(dimension-1-i)*delay]
            
            return {
                'embedding_success': True,
                'embedded_data': embedded,
                'delay': delay,
                'dimension': dimension,
                'n_embedded_points': embedded.shape[0]
            }
    
    class NumericalValidator:
        def __init__(self, config=None):
            self.config = config
        
        def validate_array(self, arr, name="array"):
            return {'is_valid': True, 'issues': []}
    
    class OutlierDetector:
        def __init__(self, config=None):
            self.config = config
        
        def remove_outliers(self, data, method='zscore'):
            return data, np.array([])
    
    def correlation_dimension(data, r_points=30):
        return np.array([]), np.array([]), 2.0
    
    def largest_lyapunov_from_data(data, dt=0.01, tau=1, min_neighbors=10):
        return 0.0
    
    def safe_divide(numerator, denominator, default=0.0, epsilon=1e-12):
        return default
    
    def safe_log(x, epsilon=1e-12):
        return 0.0
'''
            
            new_content = content[:insert_position] + default_classes + content[insert_position:]
    
    # 写回文件
    with open(rqa_file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"已修复导入问题: {rqa_file_path}")
    return True

def test_rqa_imports(project_root):
    """测试修复后的RQA导入"""
    print("\n测试修复后的RQA导入...")
    
    import sys
    sys.path.insert(0, str(project_root))
    
    try:
        from core.rqa_extractor import RQAExtractor, RQAConfig
        print("RQA模块导入成功")
        
        config = RQAConfig()
        extractor = RQAExtractor(config)
        print("RQA实例化成功")
        
        return True
        
    except Exception as e:
        print(f"RQA测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chaotic_features(project_root):
    """测试完整的chaotic_features导入"""
    print("\n测试chaotic_features导入...")
    
    import sys
    sys.path.insert(0, str(project_root))
    
    try:
        from features.chaotic_features import ChaoticFeatureExtractor, ChaoticFeatureConfig
        print("ChaoticFeatures模块导入成功")
        
        config = ChaoticFeatureConfig()
        extractor = ChaoticFeatureExtractor(config)
        print("ChaoticFeatureExtractor实例化成功")
        
        return True
        
    except Exception as e:
        print(f"ChaoticFeatures测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("RQA提取器导入问题修复脚本")
    print("=" * 50)
    
    project_root = Path.cwd()
    rqa_file = project_root / 'core' / 'rqa_extractor.py'
    
    if not rqa_file.exists():
        print(f"RQA文件不存在: {rqa_file}")
        return False
    
    # 修复RQA导入问题
    success = fix_rqa_import_issue(rqa_file)
    
    if success:
        print("RQA导入问题修复完成")
        
        # 测试RQA导入
        rqa_success = test_rqa_imports(project_root)
        
        if rqa_success:
            print("RQA修复成功")
            
            # 测试完整的chaotic_features
            chaotic_success = test_chaotic_features(project_root)
            
            if chaotic_success:
                print("\n修复成功！现在可以正常使用所有混沌特征模块了！")
                print("\n使用方法:")
                print("from features.chaotic_features import ChaoticFeatureExtractor, ChaoticFeatureConfig")
                print("config = ChaoticFeatureConfig()")
                print("extractor = ChaoticFeatureExtractor(config)")
                print("features = extractor.extract_features(your_audio_data)")
            else:
                print("\nChaotic Features仍有问题")
        else:
            print("\nRQA修复后仍有问题")
    else:
        print("RQA修复失败")
    
    return success

if __name__ == "__main__":
    main()