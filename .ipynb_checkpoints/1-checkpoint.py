# test_data_path.py
from pathlib import Path

# 从当前脚本位置计算LibriSpeech路径
script_dir = Path(__file__).parent
if script_dir.name == "Model":
    project_root = script_dir.parent
else:
    project_root = script_dir  # 如果在project根目录运行

librispeech_path = project_root / "dataset" / "train-clean-100" / "LibriSpeech" / "train-clean-100"

print(f"脚本目录: {script_dir}")
print(f"项目根目录: {project_root}")
print(f"LibriSpeech路径: {librispeech_path}")
print(f"路径是否存在: {librispeech_path.exists()}")

if librispeech_path.exists():
    flac_files = list(librispeech_path.rglob("*.flac"))
    print(f"找到 {len(flac_files)} 个FLAC文件")
    if flac_files:
        print(f"示例文件: {flac_files[0]}")
else:
    print("LibriSpeech数据不存在，请检查数据集是否正确解压到指定位置")