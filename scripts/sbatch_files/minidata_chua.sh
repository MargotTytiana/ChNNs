#!/bin/bash
#-*- coding: utf-8 -*-
#SBATCH --job-name=esc51_minidata_chua
#SBATCH --account=project_2003370
#SBATCH --output=err_minidata_chua.txt
#SBATCH --error=out_minidata_chua.txt  
#SBATCH --partition=gpusmall
#SBATCH --time=1-12:00:00
#SBATCH --begin=now
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1

module load pytorch

python train_chaotic.py \
    --system chua \
    --epochs 100 \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2
