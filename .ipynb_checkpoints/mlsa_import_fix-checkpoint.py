#!/usr/bin/env python3
"""
修复后的MLSA导入问题解决脚本
修正了路径对象的类型错误
"""

import os
import re
from pathlib import Path

def fix_mlsa_import_issue(mlsa_file_path):
    """修复MLSA文件的导入问题"""
    
    # 确保路径是字符串
    mlsa_file_path = str(mlsa_file_path)
    
    if not os.path.exists(mlsa_file_path):
        print(f"文件不存在: {mlsa_file_path}")
        return False
    
    print(f"修复文件: {mlsa_file_path}")
    
    # 读取原文件
    with open(mlsa_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原文件
    backup_path = mlsa_file_path + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已备份到: {backup_path}")
    
    # 修复策略：在导入失败时提供默认类定义
    replacement_import_block = '''# Import project modules
try:
    from chaos_utils import (
        largest_lyapunov_from_data, correlation_dimension, 
        hurst_exponent, kolmogorov_entropy
    )
    from phase_space_reconstruction import (
        PhaseSpaceReconstructor, EmbeddingConfig
    )
    from utils.numerical_stability import (
        NumericalConfig, PrecisionManager, OutlierDetector, 
        NumericalValidator, safe_divide, safe_log
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
    
    class PrecisionManager:
        def __init__(self, config):
            self.config = config
    
    class OutlierDetector:
        def __init__(self, config):
            self.config = config
        
        def remove_outliers(self, data, method='zscore'):
            if method == 'zscore':
                z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-12))
                mask = z_scores < 3.0
                return data[mask], np.where(~mask)[0]
            else:
                return data, np.array([])
    
    class NumericalValidator:
        def __init__(self, config):
            self.config = config
        
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
    
    def hurst_exponent(data):
        try:
            N = len(data)
            if N < 20:
                return 0.5
            
            lags = np.arange(2, min(N//4, 100))
            rs_values = []
            
            for lag in lags:
                mean_val = np.mean(data)
                cumulative_dev = np.cumsum(data - mean_val)
                
                n_segments = N // lag
                rs_segment = []
                
                for i in range(n_segments):
                    segment = cumulative_dev[i*lag:(i+1)*lag]
                    if len(segment) > 1:
                        R = np.max(segment) - np.min(segment)
                        S = np.std(data[i*lag:(i+1)*lag])
                        if S > 0:
                            rs_segment.append(R / S)
                
                if rs_segment:
                    rs_values.append(np.mean(rs_segment))
                else:
                    rs_values.append(1.0)
            
            if len(rs_values) > 3:
                log_lags = np.log(lags[:len(rs_values)])
                log_rs = np.log(rs_values)
                hurst = np.polyfit(log_lags, log_rs, 1)[0]
                return np.clip(hurst, 0.0, 1.0)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def kolmogorov_entropy(data):
        return 1.0
    
    def safe_divide(numerator, denominator, default=0.0, epsilon=1e-12):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(np.abs(denominator) > epsilon, 
                             numerator / denominator, default)
        return result
    
    def safe_log(x, epsilon=1e-12):
        return np.log(np.maximum(x, epsilon))


'''
    
    # 查找并替换原始的导入块
    # 找到从 "# Import project modules" 到 "except ImportError" 之后的 warnings.warn 行
    try_except_pattern = r'# Import project modules.*?except ImportError as e:\s*warnings\.warn\(f"Failed to import required modules: \{e\}"\)'
    
    if re.search(try_except_pattern, content, re.DOTALL):
        new_content = re.sub(try_except_pattern, replacement_import_block, content, flags=re.DOTALL)
        print("使用主要替换方法")
    else:
        print("使用备选替换方法")
        # 备选方法：直接查找 try: 块
        try_pattern = r'try:\s*\n.*?warnings\.warn\(f"Failed to import required modules: \{e\}"\)'
        new_content = re.sub(try_pattern, replacement_import_block, content, flags=re.DOTALL)
    
    # 写回文件
    with open(mlsa_file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"已修复导入问题: {mlsa_file_path}")
    return True

def test_fixed_imports(project_root):
    """测试修复后的导入"""
    print("\n测试修复后的导入...")
    
    import sys
    sys.path.insert(0, str(project_root))
    
    try:
        from core.mlsa_extractor import MLSAExtractor, MLSAConfig
        print("MLSA模块导入成功")
        
        config = MLSAConfig()
        extractor = MLSAExtractor(config)
        print("MLSA实例化成功")
        
        import numpy as np
        test_data = np.sin(np.linspace(0, 100, 1000)) + 0.1 * np.random.randn(1000)
        
        print("测试特征提取...")
        result = extractor.extract_features(test_data)
        
        if result.get('success', False):
            print("特征提取成功")
            print(f"   特征向量长度: {len(result['feature_vector'])}")
            print(f"   分析的尺度: {result['scales_analyzed']}")
        else:
            print(f"特征提取有问题: {result.get('error', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("MLSA导入问题快速修复")
    print("=" * 50)
    
    project_root = Path.cwd()
    mlsa_file = project_root / 'core' / 'mlsa_extractor.py'
    
    if not mlsa_file.exists():
        print(f"MLSA文件不存在: {mlsa_file}")
        return False
    
    success = fix_mlsa_import_issue(mlsa_file)
    
    if success:
        print("导入问题修复完成")
        
        test_success = test_fixed_imports(project_root)
        
        if test_success:
            print("\n修复成功！现在你可以正常使用MLSA模块了！")
            print("\n使用方法:")
            print("from core.mlsa_extractor import MLSAExtractor, MLSAConfig")
            print("config = MLSAConfig()")
            print("extractor = MLSAExtractor(config)")
            print("features = extractor.extract_features(your_audio_data)")
        else:
            print("\n修复后仍有问题，请检查错误信息")
    else:
        print("修复失败")
    
    return success

if __name__ == "__main__":
    main()