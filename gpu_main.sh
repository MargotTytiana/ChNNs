#!/bin/bash
#SBATCH --job-name=esc50
#SBATCH --account=project_462000765
#SBATCH --output=out_esc_main_freeze_sgd.txt
#SBATCH --error=err_esc_main_freeze_sgd.txt
#SBATCH --partition=small-g
#SBATCH --time=48:00:00
#SBATCH --begin=now

#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --nodes=1
##SBATCH --gpus=1
#SBATCH --gres=gpu:v100:1

##module use /appl/local/csc/modulefiles/
module load pytorch

python pytorch/main.py train --holdout_fold=1 --model_type="Transfer_Cnn14" --loss_type=clip_nll --augmentation='none' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda --num_workers 16 --freeze_base

