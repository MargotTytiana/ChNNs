#!/usr/bin/env python3
"""
Python导入问题诊断和修复工具
用于解决C-HiLAP项目的模块导入问题

使用方法：
1. 将此文件放在项目根目录 (/scratch/project_2003370/yueyao/Model/)
2. 运行: python import_diagnostic.py
3. 根据诊断结果修复问题
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

class ImportDiagnostic:
    """导入问题诊断工具"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.issues = []
        self.solutions = []
    
    def diagnose_all(self):
        """运行完整诊断"""
        print("🔍 Python导入问题诊断工具")
        print("=" * 50)
        
        self.check_python_environment()
        self.check_project_structure()
        self.check_python_path()
        self.check_init_files()
        self.check_module_imports()
        self.check_circular_imports()
        
        self.print_summary()
        self.provide_solutions()
    
    def check_python_environment(self):
        """检查Python环境"""
        print("\n📋 检查Python环境...")
        
        print(f"Python版本: {sys.version}")
        print(f"Python可执行文件: {sys.executable}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"项目根目录: {self.project_root}")
        
        # 检查关键包
        required_packages = ['numpy', 'scipy']
        for pkg in required_packages:
            try:
                importlib.import_module(pkg)
                print(f"✅ {pkg} 已安装")
            except ImportError:
                print(f"❌ {pkg} 未安装")
                self.issues.append(f"缺少必需包: {pkg}")
    
    def check_project_structure(self):
        """检查项目结构"""
        print("\n📁 检查项目结构...")
        
        expected_dirs = ['core', 'features', 'utils']
        expected_files = [
            'core/mlsa_extractor.py',
            'core/rqa_extractor.py',
            'core/phase_space_reconstruction.py',
            'features/chaotic_features.py',
            'utils/numerical_stability.py'
        ]
        
        # 检查目录
        for dir_name in expected_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                print(f"✅ 目录存在: {dir_name}")
            else:
                print(f"❌ 目录缺失: {dir_name}")
                self.issues.append(f"缺少目录: {dir_name}")
        
        # 检查文件
        for file_path in expected_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"✅ 文件存在: {file_path}")
            else:
                print(f"❌ 文件缺失: {file_path}")
                self.issues.append(f"缺少文件: {file_path}")
    
    def check_python_path(self):
        """检查Python路径设置"""
        print("\n🛤️  检查Python路径...")
        
        print("当前sys.path:")
        for i, path in enumerate(sys.path):
            print(f"  {i}: {path}")
        
        # 检查项目根目录是否在路径中
        project_str = str(self.project_root)
        if project_str in sys.path:
            print(f"✅ 项目根目录在sys.path中: {project_str}")
        else:
            print(f"❌ 项目根目录不在sys.path中: {project_str}")
            self.issues.append("项目根目录不在sys.path中")
    
    def check_init_files(self):
        """检查__init__.py文件"""
        print("\n📄 检查__init__.py文件...")
        
        # 检查所有目录的__init__.py
        for dir_path in self.project_root.rglob('*'):
            if dir_path.is_dir() and not dir_path.name.startswith('.'):
                init_file = dir_path / '__init__.py'
                rel_path = dir_path.relative_to(self.project_root)
                
                if init_file.exists():
                    print(f"✅ {rel_path}/__init__.py 存在")
                    
                    # 检查__init__.py内容
                    try:
                        with open(init_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        if content:
                            print(f"   📝 有内容 ({len(content)} 字符)")
                        else:
                            print(f"   📝 空文件")
                    except Exception as e:
                        print(f"   ❌ 读取错误: {e}")
                else:
                    print(f"❌ {rel_path}/__init__.py 缺失")
                    self.issues.append(f"缺少__init__.py: {rel_path}")
    
    def check_module_imports(self):
        """检查模块导入"""
        print("\n🔗 检查模块导入...")
        
        # 测试各种导入方式
        test_imports = [
            "core",
            "core.mlsa_extractor",
            "utils",
            "utils.numerical_stability",
            "features",
            "features.chaotic_features"
        ]
        
        # 临时添加项目根目录到路径
        original_path = sys.path.copy()
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        try:
            for module_name in test_imports:
                try:
                    module = importlib.import_module(module_name)
                    print(f"✅ 导入成功: {module_name}")
                    if hasattr(module, '__file__'):
                        print(f"   📍 位置: {module.__file__}")
                except ImportError as e:
                    print(f"❌ 导入失败: {module_name}")
                    print(f"   💬 错误: {e}")
                    self.issues.append(f"导入失败: {module_name} - {e}")
                except Exception as e:
                    print(f"❌ 其他错误: {module_name}")
                    print(f"   💬 错误: {e}")
                    self.issues.append(f"导入错误: {module_name} - {e}")
        finally:
            sys.path = original_path
    
    def check_circular_imports(self):
        """检查循环导入"""
        print("\n🔄 检查循环导入...")
        
        # 分析Python文件中的import语句
        python_files = list(self.project_root.rglob('*.py'))
        imports_map = {}
        
        for py_file in python_files:
            if py_file.name.startswith('__'):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 简单的import语句提取
                imports = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('from ') and ' import ' in line:
                        # from xxx import yyy
                        module = line.split('from ')[1].split(' import ')[0].strip()
                        imports.append(module)
                    elif line.startswith('import '):
                        # import xxx
                        module = line.split('import ')[1].split()[0].strip()
                        imports.append(module)
                
                rel_path = py_file.relative_to(self.project_root)
                imports_map[str(rel_path)] = imports
                
            except Exception as e:
                print(f"   ⚠️  无法分析文件: {py_file} - {e}")
        
        # 检查可能的循环依赖
        print("   文件导入依赖:")
        for file_path, imports in imports_map.items():
            if imports:
                print(f"   📄 {file_path}:")
                for imp in imports[:5]:  # 只显示前5个
                    print(f"     → {imp}")
                if len(imports) > 5:
                    print(f"     ... 还有 {len(imports) - 5} 个导入")
    
    def print_summary(self):
        """打印诊断总结"""
        print("\n" + "=" * 50)
        print("📊 诊断总结")
        print("=" * 50)
        
        if not self.issues:
            print("🎉 未发现明显问题！")
        else:
            print(f"❌ 发现 {len(self.issues)} 个问题:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
    
    def provide_solutions(self):
        """提供解决方案"""
        print("\n🔧 解决方案")
        print("=" * 50)
        
        if not self.issues:
            print("✅ 项目结构看起来正常，尝试以下通用解决方案:")
        
        solutions = [
            "1. 使用绝对导入路径",
            "2. 重新初始化Python环境",
            "3. 创建独立模块避免依赖",
            "4. 检查文件编码问题",
            "5. 验证Python包安装"
        ]
        
        for solution in solutions:
            print(f"   {solution}")
        
        # 生成修复脚本
        self.generate_fix_script()
    
    def generate_fix_script(self):
        """生成自动修复脚本"""
        fix_script = '''#!/usr/bin/env python3
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
'''
        
        fix_file = self.project_root / 'fix_imports.py'
        with open(fix_file, 'w', encoding='utf-8') as f:
            f.write(fix_script)
        
        print(f"\n📜 已生成自动修复脚本: {fix_file}")
        print("   运行命令: python fix_imports.py")

def main():
    """主函数"""
    print("请输入项目根目录路径 (回车使用当前目录):")
    project_path = input().strip()
    
    if not project_path:
        project_path = os.getcwd()
    
    diagnostic = ImportDiagnostic(project_path)
    diagnostic.diagnose_all()

if __name__ == "__main__":
    main()