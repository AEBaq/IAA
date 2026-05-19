#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu_students
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

mkdir -p logs

echo "Job started on $(hostname) at $(date)"

# Modifier --checkpoint si besoin
poetry run python main.py --eval-only

echo "Job finished at $(date)"
