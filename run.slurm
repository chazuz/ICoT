#!/bin/bash
#SBATCH --job-name=physbench_cpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=NvidiaAll
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/w/webere/big/STRUCTURAL

# DO NOT request memory or GPUs
# DO NOT set --gres

echo "Job started at $(date)"
echo "Running on $(hostname)"

# fix TMPDIR issue (cluster bug, safe global fallback)
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

# activate venv
source venv/bin/activate

export PYTHONPATH=$PWD:$PYTHONPATH

# force CPU mode (IMPORTANT)
export CUDA_VISIBLE_DEVICES=""

# run script
echo "Running: $1"
python -u "$1" "${@:2}"

echo "Job finished at $(date)"
