# chaos_utils_debug_patch.py
"""
混沌工具调试补丁
在现有代码基础上添加调试功能
"""

import numpy as np
from scipy import stats

def debug_hurst_exponent(signal, max_window=None, debug=True):
    """
    调试版Hurst指数计算
    添加详细的调试信息和数值稳定性处理
    """
    if debug:
        print(f"🔍 Hurst调试 - 信号长度: {len(signal)}, 范围: [{signal.min():.6f}, {signal.max():.6f}]")
    
    # 检查信号质量
    if len(signal) < 50:
        if debug:
            print("⚠️  信号长度不足，返回默认值0.5")
        return 0.5
    
    signal_std = np.std(signal)
    if signal_std < 1e-10:
        if debug:
            print("⚠️  信号方差接近0，返回默认值0.5")
        return 0.5
    
    try:
        n = len(signal)
        if max_window is None:
            max_window = n // 4
        
        # 确保信号是numpy数组
        signal_np = np.asarray(signal, dtype=np.float64)
        
        # 数值稳定性：去除异常值
        signal_np = np.clip(signal_np, 
                           np.percentile(signal_np, 1), 
                           np.percentile(signal_np, 99))
        
        windows = []
        rs_values = []
        
        # 使用更稳健的窗口选择
        window_sizes = np.linspace(10, min(max_window, n//3), 20, dtype=int)
        window_sizes = window_sizes[window_sizes >= 10]
        
        for window_size in window_sizes:
            segments = n // window_size
            if segments < 2:
                continue
                
            r_s_list = []
            for i in range(segments):
                segment = signal_np[i*window_size:(i+1)*window_size]
                if len(segment) < 10:
                    continue
                    
                # 计算重标极差
                mean_segment = np.mean(segment)
                cumulative_deviation = np.cumsum(segment - mean_segment)
                r = np.max(cumulative_deviation) - np.min(cumulative_deviation)
                s = np.std(segment)
                
                if s > 1e-10 and r > 1e-10:  # 避免数值问题
                    r_s_list.append(r / s)
            
            if len(r_s_list) > 0:
                windows.append(window_size)
                rs_values.append(np.mean(r_s_list))
        
        if len(windows) < 3:
            if debug:
                print("❌ 有效窗口数不足，返回默认值0.5")
            return 0.5
        
        # 对数变换和线性拟合
        log_windows = np.log(windows)
        log_rs = np.log(rs_values)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_windows, log_rs)
        
        if debug:
            print(f"  Hurst指数: {slope:.4f}, R²: {r_value**2:.4f}, 窗口数: {len(windows)}")
        
        # 检查拟合质量
        if r_value**2 < 0.7:
            if debug:
                print("⚠️  拟合质量较差，使用截断值")
            return max(0.1, min(0.9, slope))  # 限制在合理范围
        else:
            return slope
            
    except Exception as e:
        if debug:
            print(f"❌ Hurst计算错误: {e}")
        return 0.5

def validate_chaotic_features(features, feature_names=None):
    """
    验证混沌特征质量
    """
    print("🎯 混沌特征质量验证:")
    
    if isinstance(features, list):
        features = np.array(features)
    
    print(f"  特征形状: {features.shape}")
    print(f"  特征范围: [{features.min():.6f}, {features.max():.6f}]")
    print(f"  特征均值: {features.mean():.6f} ± {features.std():.6f}")
    
    # 检查异常值
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    
    print(f"  异常值 - NaN: {nan_count}, Inf: {inf_count}")
    
    # 检查特征方差
    if features.ndim > 1:
        feature_vars = features.var(axis=0)
        constant_features = (feature_vars < 1e-10).sum()
        print(f"  常数特征: {constant_features}/{len(feature_vars)}")
        
        # 如果有特征名称，显示有问题的特征
        if feature_names is not None and constant_features > 0:
            problematic = [feature_names[i] for i, var in enumerate(feature_vars) if var < 1e-10]
            print(f"  有问题的特征: {problematic}")
    
    return nan_count == 0 and inf_count == 0
