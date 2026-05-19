#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=gpu_students
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

mkdir -p logs

echo "Job started on $(hostname) at $(date)"
nvidia-smi

poetry run python main.py

echo "Job finished at $(date)"
