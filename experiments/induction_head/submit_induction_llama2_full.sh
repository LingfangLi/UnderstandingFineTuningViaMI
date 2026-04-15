#!/bin/bash -l
# Detect induction heads in Llama-2-7B FULL fine-tuning models for all 6
# canonical tasks (twitter excluded). Single SLURM job - tasks run sequentially
# inside the script because each task has a different model to load.
#
# Optional env vars:
#   TASKS=task1,task2   restrict to a subset of tasks
#   RERUN_BASE=1        force rerun of the base pretrained model
#
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o detect_induction_llama2_full_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 12:00:00

set -euo pipefail

PROJECT_ROOT="/users/sglli24/UnderstandingFineTuningViaMI"
SCRIPT_PATH="${PROJECT_ROOT}/experiments/induction_head/detect_induction_head_llama2_full.py"

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "========================================================="
echo "SLURM job $SLURM_JOB_ID started at $(date)"
echo "Node = $(hostname)"
echo "TASKS env  = ${TASKS:-<all>}"
echo "RERUN_BASE = ${RERUN_BASE:-0}"
echo "========================================================="

date_start=$(date +%s)

python -u "$SCRIPT_PATH"

date_end=$(date +%s)
sec=$((date_end - date_start))
hh=$((sec / 3600)); mm=$(((sec % 3600) / 60)); ss=$((sec % 60))

echo "========================================================="
echo "Finished at $(date)  |  wall clock: ${hh}h ${mm}m ${ss}s"
echo "========================================================="
