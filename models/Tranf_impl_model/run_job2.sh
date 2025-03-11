#!/usr/bin/env bash

#SBATCH --job-name=RunningModelERC
#SBATCH --gres=gpu:1
#SBATCH --output=./selfatt_withT_withemo_withencod.out
#SBATCH --error=./selfatt_withT_withemo_withencod.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=1
srun python3 ./main.py
