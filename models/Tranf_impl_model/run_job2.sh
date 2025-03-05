#!/usr/bin/env bash

#SBATCH --job-name=RunningModelERC
#SBATCH --gres=gpu:1
#SBATCH --output=./selfatt_withoutT.out
#SBATCH --error=./selfatt_wihtoutT.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=1
srun python3 ./main.py
