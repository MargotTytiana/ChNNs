#!/bin/bash
#-*- coding: utf-8 -*-
#SBATCH --job-name=esc51_com
#SBATCH --account=project_2003370
#SBATCH --partition=gpusmall
#SBATCH --time=1-12:00:00
#SBATCH --begin=now
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1

module load pytorch

python run_chaotic_experiments.py --exp 3
