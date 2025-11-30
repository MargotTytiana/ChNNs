#!/bin/bash

# SBATCH --job-name=esc51
# SBATCH --account=project_2003370
# SBATCH --output=out_chua.txt
# SBATCH --error=err_chua.txt
# SBATCH --partition=gpusmall
# SBATCH --time=1-12:00:00
# SBATCH --begin=now
# SBATCH --cpus-per-task=20
# SBATCH --mem-per-cpu=16000
# SBATCH --nodes=1
# #SBATCH --gpus=1
# SBATCH --gres=gpu:a100:1


# #module use /appl/local/csc/modulefiles/

module load pytorch


python train_chaotic.py \
    --system chua \
    --epochs 100 \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/train-clean-100/LibriSpeech/train-clean-100

