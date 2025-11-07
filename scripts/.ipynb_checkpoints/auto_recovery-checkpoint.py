# auto_recovery.py
import os
import subprocess
import sys

def auto_recovery():
    """自动恢复训练流程"""
    print("🤖 AUTO RECOVERY SYSTEM ACTIVATED")
    print("="*50)
    
    # 步骤1: 运行终极检查点修复
    print("\n1. Running ultimate checkpoint fix...")
    result = subprocess.run([sys.executable, "ultimate_checkpoint_fix.py"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Ultimate fix failed, trying alternative approach...")
        # 备用方案：创建简单模型检查点
        subprocess.run([sys.executable, "create_simple_model_checkpoint.py"])
        checkpoint_path = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/model_simple.pth"
    else:
        # 从输出中提取生成的检查点路径
        for line in result.stdout.split('\n'):
            if '--resume' in line and '.pth' in line:
                checkpoint_path = line.split('--resume')[-1].strip()
                break
        else:
            checkpoint_path = "outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_ultimate.pth"
    
    print(f"✅ Using checkpoint: {checkpoint_path}")
    
    # 步骤2: 验证检查点
    print("\n2. Verifying checkpoint...")
    if not os.path.exists(checkpoint_path):
        print("❌ Checkpoint not created, training from scratch")
        checkpoint_path = None
    
    # 步骤3: 启动训练
    print("\n3. Starting training...")
    cmd = [
        sys.executable, "train_chaotic.py",
        "--system", "lorenz",
        "--epochs", "100", 
        "--save_models",
        "--model_types", "full_chaotic",
        "--data_dir", "/scratch/project_2003370/yueyao/dataset/train-clean-100/LibriSpeech/train-clean-100"
    ]
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        cmd.extend(["--resume", checkpoint_path])
        print(f"🚀 Starting training with checkpoint: {checkpoint_path}")
    else:
        print("🚀 Starting training from scratch")
    
    print(f"\nCommand: {' '.join(cmd)}")
    
    # 执行训练命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = auto_recovery()
    
    if success:
        print("\n🎉 AUTO RECOVERY COMPLETED SUCCESSFULLY!")
    else:
        print("\n💥 AUTO RECOVERY FAILED!")
        print("Please check the error messages above.")