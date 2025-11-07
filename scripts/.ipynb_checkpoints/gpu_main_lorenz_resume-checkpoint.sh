#!/bin/bash
#SBATCH --job-name=esc51_resume
#SBATCH --account=project_2003370
#SBATCH --output=out_esc_lorenz_resume_2.txt
#SBATCH --error=err_esc_lorenz_resume_2.txt
#SBATCH --partition=gpusmall
#SBATCH --time=1-12:00:00
#SBATCH --begin=now
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --nodes=1
##SBATCH --gpus=1
#SBATCH --gres=gpu:a100:1


module load pytorch

echo "================================================"
echo "ENHANCED CHAOTIC NETWORK TRAINING RECOVERY"
echo "================================================"
echo "Start time: $(date)"
echo "================================================"

echo "1. Running enhanced checkpoint analysis..."
python fix_checkpoint.py

echo "2. Checking analysis results..."
ENHANCED_PATH="outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_enhanced.pth"
FALLBACK_PATH="outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251029_151635/checkpoint_fallback.pth"

if [ -f "$ENHANCED_PATH" ]; then
    echo "✅ Enhanced checkpoint available"
    python -c "
import torch
checkpoint = torch.load('$ENHANCED_PATH', map_location='cpu')
matched_ratio = checkpoint['checkpoint_info']['matched_ratio']
print(f'Model parameter match ratio: {matched_ratio:.1%}')
exit(0 if matched_ratio > 0.5 else 1)
"
    if [ $? -eq 0 ]; then
        RESUME_ARG="--resume $ENHANCED_PATH"
        echo "🔄 Using enhanced checkpoint (good match)"
    else
        RESUME_ARG="--resume $FALLBACK_PATH"
        echo "⚠️  Using fallback checkpoint (poor match)"
    fi
elif [ -f "$FALLBACK_PATH" ]; then
    RESUME_ARG="--resume $FALLBACK_PATH"
    echo "🔄 Using fallback checkpoint"
else
    RESUME_ARG=""
    echo "🔄 No compatible checkpoint found, starting from scratch"
fi

echo "3. Starting training..."
echo "Command: python train_chaotic.py --system lorenz --epochs 100 --save_models --model_types full_chaotic --data_dir /scratch/project_2003370/yueyao/dataset/train-clean-100/LibriSpeech/train-clean-100 $RESUME_ARG"

python train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/train-clean-100/LibriSpeech/train-clean-100 \
    $RESUME_ARG

echo "================================================"
echo "Training completed at: $(date)"
echo "================================================"