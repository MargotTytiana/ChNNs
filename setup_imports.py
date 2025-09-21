# setup_imports.py
import os
import sys
from pathlib import Path

def setup_project_imports():
    """仅设置项目导入路径，不进行验证"""
    project_root = Path(__file__).parent.absolute()
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print(f"项目根目录已添加到路径: {project_root}")
    return True  # 不进行导入验证

if __name__ == "__main__":
    setup_project_imports()