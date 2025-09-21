#!/usr/bin/env python3
"""
自动修复脚本 - 由导入诊断工具生成
"""

import sys
import os
from pathlib import Path

def fix_imports():
    """修复导入问题"""
    
    # 1. 设置正确的Python路径
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print(f"✅ 添加项目根目录到路径: {project_root}")
    
    # 2. 创建缺失的__init__.py文件
    dirs_need_init = ['core', 'features', 'utils', 'models', 'experiments']
    
    for dir_name in dirs_need_init:
        dir_path = project_root / dir_name
        init_file = dir_path / '__init__.py'
        
        if dir_path.exists() and not init_file.exists():
            init_file.touch()
            print(f"✅ 创建 {dir_name}/__init__.py")
    
    # 3. 测试导入
    try:
        # 在这里添加你的导入测试
        print("🧪 测试导入...")
        
        # 示例测试 - 替换为你的实际模块
        # from core.mlsa_extractor import MLSAExtractor
        # print("✅ 核心模块导入成功")
        
    except ImportError as e:
        print(f"❌ 导入仍然失败: {e}")
        return False
    
    print("🎉 修复完成!")
    return True

if __name__ == "__main__":
    fix_imports()
