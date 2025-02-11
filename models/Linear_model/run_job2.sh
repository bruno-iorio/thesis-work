#!/usr/bin/env bash

#SBATCH --job-name=RunningModelERC
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./output6.out
#SBATCH --error=./error6.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
srun python3 ./main.py
