# test_imports.py
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
model_dir = project_root / "Model"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(model_dir))

print("测试导入...")

try:
    from data.dataset_loader import create_speaker_dataloaders
    print(f"✓ create_speaker_dataloaders 导入成功: {create_speaker_dataloaders}")
except ImportError as e:
    print(f"✗ create_speaker_dataloaders 导入失败: {e}")

try:
    from experiments.baseline_experiment import BaselineExperiment
    print("✓ BaselineExperiment 导入成功")
except ImportError as e:
    print(f"✗ BaselineExperiment 导入失败: {e}")

print("导入测试完成")