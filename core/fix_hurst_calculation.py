# fix_hurst_calculation.py
import numpy as np
from scipy.stats import linregress
import warnings

def test_hurst_fixes():
    """测试Hurst计算修复"""
    print("🧪 测试Hurst计算修复...")
    
    # 测试不同长度的序列
    test_cases = [
        np.random.randn(50),   # 太短
        np.random.randn(100),  # 边界情况
        np.random.randn(1000), # 正常长度
        np.random.randn(5000)  # 长序列
    ]
    
    for i, signal in enumerate(test_cases):
        print(f"\n测试案例 {i+1}: 序列长度={len(signal)}")
        
        try:
            hurst = robust_hurst_exponent(signal)
            print(f"  Hurst指数: {hurst:.3f}")
        except Exception as e:
            print(f"  ❌ 计算失败: {e}")

if __name__ == "__main__":
    test_hurst_fixes()