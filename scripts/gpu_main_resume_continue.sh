#!/bin/bash
#SBATCH --job-name=esc51_continue
#SBATCH --account=project_2003370
#SBATCH --output=out_esc_lorenz_continue_1120.txt
#SBATCH --error=err_esc_lorenz_continue_1120.txt  
#SBATCH --partition=gpusmall
#SBATCH --time=1-12:00:00
#SBATCH --begin=now
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1

module load pytorch

python train_chaotic.py \
    --config experiments/configs/chaotic_config.yaml \
    --system lorenz \
    --epochs 100 \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/train-clean-100/LibriSpeech/train-clean-100