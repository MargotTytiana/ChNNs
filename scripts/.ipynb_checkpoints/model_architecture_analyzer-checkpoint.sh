#!/bin/bash
#SBATCH --job-name=esc51_Ana
#SBATCH --account=project_2003370
#SBATCH --output=model_architecture_analyzer.txt
#SBATCH --error=err_model_architecture_analyzer.txt
#SBATCH --partition=gpusmall
#SBATCH --time=1-12:00:00
#SBATCH --begin=now
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1

module load pytorch

python model_architecture_analyzer.py
