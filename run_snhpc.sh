#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=pytorch_test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4  # For DataLoader's num_workers

echo "=== SLURM ENV ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"

# Load conda (adjust module name if needed)
module load devel/miniforge/24.11.0-python-3.12

# Activate env (assuming default location)
conda activate hpcenv

echo "Conda env: $CONDA_DEFAULT_ENV"

# Debug script location
echo "Running from: $(pwd)"

# Navigate to repo (if not already there)
cd ~/hpc-project

# # Run script (paths are now relative to repo root)
python src/milestones/milestone4.py

# # Run with unbuffered output
# PYTHONUNBUFFERED=1 python src/pythontest.py
