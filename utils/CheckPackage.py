import os
import subprocess

search_dir = r"P:\PycharmProjects"  # 修改为你的目录

def is_venv(path):
    return os.path.isdir(os.path.join(path, "Scripts")) and os.path.isfile(os.path.join(path, "Scripts", "activate"))

def list_packages(venv_path):
    pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
    try:
        result = subprocess.run([pip_path, "list"], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        return f"Error: {e}"

for root, dirs, files in os.walk(search_dir):
    if is_venv(root):
        print(f"Virtual environment found: {root}")
        print("Installed packages:")
        print(list_packages(root))
        print("-" * 40)

