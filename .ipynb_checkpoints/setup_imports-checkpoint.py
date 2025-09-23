# setup_imports.py
"""
项目导入设置工具
确保所有模块能够正确导入
"""

import os
import sys
from pathlib import Path

def setup_project_imports():
    """设置项目导入路径"""
    # 获取项目根目录
    current_file = Path(__file__).resolve()
    
    # 假设这个文件在项目根目录
    project_root = current_file.parent
    
    # 如果当前文件在Model子目录中，需要向上一级
    if current_file.parent.name == 'Model':
        project_root = current_file.parent.parent
        
    # 添加路径
    paths_to_add = [
        str(project_root),
        str(project_root / 'Model'),
        str(project_root / 'Model' / 'core'),
        str(project_root / 'Model' / 'features'),
        str(project_root / 'Model' / 'models'),
        str(project_root / 'Model' / 'experiments'),
        str(project_root / 'Model' / 'evaluation'),
        str(project_root / 'Model' / 'data'),
        str(project_root / 'Model' / 'utils'),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print(f"Project root: {project_root}")
    print(f"Added {len(paths_to_add)} paths to sys.path")
    
    return project_root

if __name__ == "__main__":
    setup_project_imports()