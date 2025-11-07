#!/bin/bash
#SBATCH --job-name=esc51_continue
#SBATCH --account=project_2003370
#SBATCH --output=out_esc_lorenz_continue.txt
#SBATCH --error=err_esc_lorenz_continue.txt
#SBATCH --partition=gpusmall
#SBATCH --time=1-12:00:00
#SBATCH --begin=now
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1

module load pytorch

echo "================================================"
echo "CONTINUING CHAOTIC NETWORK TRAINING"
echo "================================================"
echo "Start time: $(date)"
echo "================================================"

LATEST_CHECKPOINT="outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251103_134109/checkpoint_epoch_0000_20251104_074641.pkl"

if [ -f "$LATEST_CHECKPOINT" ]; then
    echo "✅ Found latest checkpoint: $LATEST_CHECKPOINT"
    RESUME_ARG="--resume $LATEST_CHECKPOINT"
    echo "🔄 Continuing from checkpoint (epoch 10+)"
else
    echo "⚠️  Latest checkpoint not found, checking for any checkpoint..."
    CHECKPOINT_DIR="outputs/chaotic/experiments/chaotic_lorenz_full_chaotic_run_0/checkpoints/exp_20251103_134109"
    if [ -d "$CHECKPOINT_DIR" ]; then
        LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/*.pkl 2>/dev/null | head -1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            RESUME_ARG="--resume $LATEST_CHECKPOINT"
            echo "🔄 Using checkpoint: $LATEST_CHECKPOINT"
        else
            RESUME_ARG=""
            echo "🔄 No checkpoint found, starting from scratch"
        fi
    else
        RESUME_ARG=""
        echo "🔄 No checkpoint directory found, starting from scratch"
    fi
fi

echo "2. Continuing training..."
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