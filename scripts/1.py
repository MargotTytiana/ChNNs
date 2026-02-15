import torch
import pickle
import sys
import os
from pathlib import Path

# 获取当前脚本的绝对路径
current_file = Path(__file__).resolve()
# 获取项目根目录 (假设结构是 Model/scripts/1.py，所以向上两级是 Model 目录)
project_root = current_file.parent.parent

# 将项目根目录加入 Python 搜索路径
sys.path.insert(0, str(project_root))

# --- 下面才是你原来的导入代码 ---
from models.chaotic_network import ChaoticSpeakerRecognitionNetwork

# 你的 C-HiLAP 模型路径
CKPT_PATH = "/scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H2/H2c/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20260114_142457/checkpoint_epoch_0000_20260114_143911.pkl"

def check_keys():
    # 1. 加载 Checkpoint
    print("Loading Checkpoint...")
    with open(CKPT_PATH, 'rb') as f:
        ckpt = pickle.load(f)
    
    if 'model' in ckpt:
        ckpt_state = ckpt['model']
    else:
        ckpt_state = ckpt
        
    ckpt_keys = set(ckpt_state.keys())
    print(f"Checkpoint contains {len(ckpt_keys)} keys.")
    print(f"Example Checkpoint Keys: {list(ckpt_keys)[:3]}")

    # 2. 初始化模型 (使用默认参数，主要看结构)
    print("\nInitializing Model...")
    model = ChaoticSpeakerRecognitionNetwork(
        sample_rate=16000,
        embedding_dim=10,
        mlsa_scales=5,
        rqa_radius_ratio=0.1,
        chaotic_system='lorenz',
        evolution_time=0.5,
        pooling_type='comprehensive',
        speaker_embedding_dim=256,
        num_speakers=251
    )
    model_keys = set(model.state_dict().keys())
    print(f"Model expects {len(model_keys)} keys.")
    print(f"Example Model Keys: {list(model_keys)[:3]}")

    # 3. 对比差异
    print("\n--- DIAGNOSIS ---")
    common_keys = ckpt_keys.intersection(model_keys)
    missing_in_ckpt = model_keys - ckpt_keys
    missing_in_model = ckpt_keys - model_keys
    
    print(f"Matching Keys: {len(common_keys)}")
    print(f"Missing in Checkpoint: {len(missing_in_ckpt)}")
    print(f"Missing in Model: {len(missing_in_model)}")
    
    if len(common_keys) == 0:
        print("\nCRITICAL: NO KEYS MATCH! The model is random initialized.")
        print("Possible reason: Checkpoint keys might have a prefix like 'module.' or 'model.'")
        
        # 尝试查看前缀
        sample_ckpt = list(ckpt_keys)[0]
        sample_model = list(model_keys)[0]
        print(f"\nCompare:\nCKPT:  {sample_ckpt}\nMODEL: {sample_model}")

if __name__ == "__main__":
    check_keys()