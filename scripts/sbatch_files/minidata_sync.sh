#!/bin/bash
#-*- coding: utf-8 -*-
#SBATCH --job-name=esc51_minidata_sync
#SBATCH --account=project_2003370
#SBATCH --output=err_minidata_sync.txt
#SBATCH --error=out_minidata_sync.txt  
#SBATCH --partition=gpusmall
#SBATCH --time=1-12:00:00
#SBATCH --begin=now
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1

module load pytorch


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/F \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_F_no_adversarial.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H2a \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H2a_small_weight.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H2b \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H2b_tiny_weight.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H2c \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H2c_0.05.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H2d \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H2d_0.05.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H3a \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H3a_full_sampling.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H3b \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H3b_half_sampling.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H3c \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H3c_10_sampling.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H4a \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H4a_warmup_20.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H4b \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H4b_warmup_50.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H4c \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H4c_progressive.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H5a \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H5a_cosine_distance.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H5b \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H5b_manhattan.yaml


python /scratch/project_2003370/yueyao/Model/scripts/train_chaotic.py \
    --system lorenz \
    --epochs 100 \
    --batch_size 32 \
    --output_dir  /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/H5c \
    --save_models \
    --model_types full_chaotic \
    --data_dir /scratch/project_2003370/yueyao/dataset/mini_librispeech/LibriSpeech/dev-clean-2 \
    --config /scratch/project_2003370/yueyao/Model/experiments/configs/sync_experiments/experiment_H5c_euclidean.yaml







