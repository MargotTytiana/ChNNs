#!/bin/bash
#SBATCH --job-name=esc50
#SBATCH --account=project_20251009
#SBATCH --output=out_esc_main_freeze_sgd.txt
#SBATCH --error=err_esc_main_freeze_sgd.txt
#SBATCH --partition=small-g
#SBATCH --time=72:00:00
#SBATCH --begin=now

#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --nodes=1
##SBATCH --gpus=1
#SBATCH --gres=gpu:v100:1

##module use /appl/local/csc/modulefiles/
module load pytorch

python train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --no_save_models \
    --no_analyze_dynamics \
    --data_dir /scratch/project_2003370/yueyao/dataset/train-clean-100/LibriSpeech/train-clean-100
