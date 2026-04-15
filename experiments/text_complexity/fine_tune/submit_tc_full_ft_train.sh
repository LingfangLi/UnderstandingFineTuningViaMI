#!/bin/bash
# Submit 6 SLURM jobs for Llama-2-7B full fine-tuning on lexical-complexity
# subsets: 3 tasks (yelp, squad, tatoeba) x 2 subsets (simple, complex).
#
# Twitter is intentionally excluded.
#
# Usage: bash submit_tc_full_ft_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKER="${SCRIPT_DIR}/tc_full_ft_train_worker.sh"

TASKS=(yelp squad tatoeba)
SUBSETS=(simple complex)

for task in "${TASKS[@]}"; do
    for subset in "${SUBSETS[@]}"; do
        job_name="tc_${task}_${subset}"
        sbatch \
            --export=ALL,TASK="${task}",EXPERIMENT_TYPE="${subset}" \
            --job-name="${job_name}" \
            --output="tc_train_${task}_${subset}_%j.out" \
            "${WORKER}"
    done
done

echo
echo "Submitted 6 jobs. Check status with:  squeue -u \$USER"
