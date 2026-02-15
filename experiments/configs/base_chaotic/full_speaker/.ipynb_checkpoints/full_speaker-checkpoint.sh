#!/bin/bash

#SBATCH --job-name=esc51
#SBATCH --account=project_2003370
#SBATCH --output=out_251.txt
#SBATCH --error=err_251.txt
#SBATCH --partition=gpusmall
#SBATCH --time=1-12:00:00
#SBATCH --begin=now
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --nodes=1
##SBATCH --gpus=1
#SBATCH --gres=gpu:a100:1


##module use /appl/local/csc/modulefiles/

module load pytorch


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/base_chaotic/full_speaker \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/train-clean-100/LibriSpeech/train-clean-100 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/base_chaotic/full_speaker/base_config.yaml