#!/bin/bash -l
# Aggregate the 18 text-complexity full-FT eval JSONs into summary tables.
# Designed to be submitted with --dependency=afterok on all 9 eval jobs so it
# fires automatically when the last eval finishes.
#
# CPU-only - aggregation just reads JSON and writes CSV/TeX. No GPU needed.
#
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o tc_aggregate_%j.out
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 30:00

set -euo pipefail

PROJECT_ROOT="/users/sglli24/UnderstandingFineTuningViaMI"
SCRIPT="${PROJECT_ROOT}/experiments/text_complexity/fine_tune/aggregate_eval_results.py"

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

echo "========================================================="
echo "Aggregation job $SLURM_JOB_ID started at $(date)"
echo "Node = $(hostname)"
echo "========================================================="

python -u "$SCRIPT"

echo "========================================================="
echo "Done at $(date)"
echo "========================================================="
