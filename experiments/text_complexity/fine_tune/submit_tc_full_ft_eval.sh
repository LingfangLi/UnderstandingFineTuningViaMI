#!/bin/bash
# Submit 9 SLURM jobs for text_complexity full FT evaluation.
#
# 6 FT eval jobs:  (yelp, squad, tatoeba) x (train_simple, train_complex)
#                  Each runs the model on BOTH simple and complex test subsets,
#                  producing 2 JSONs per job (12 FT eval JSONs in total).
# 3 base eval jobs: one per task (yelp, squad, tatoeba), each runs base
#                   pretrained Llama-2-7B on BOTH test subsets, producing
#                   2 JSONs per job (6 base eval JSONs in total).
#
# Grand total: 18 JSONs in results_full/.
#
# Twitter is intentionally excluded.
#
# Usage: bash submit_tc_full_ft_eval.sh
#
# IMPORTANT: only run this AFTER submit_tc_full_ft_train.sh has finished
#            producing all 6 model directories. The worker resolves model
#            paths via glob, so missing dirs will fail fast.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKER="${SCRIPT_DIR}/tc_full_ft_eval_worker.sh"

TASKS=(yelp squad tatoeba)
SUBSETS=(simple complex)

# 6 FT eval jobs
for task in "${TASKS[@]}"; do
    for subset in "${SUBSETS[@]}"; do
        sbatch \
            --export=ALL,TASK="${task}",TRAIN_SUBSET="${subset}" \
            --job-name="ev_${task}_${subset}" \
            --output="tc_eval_${task}_train_${subset}_%j.out" \
            "${WORKER}"
    done
done

# 3 base eval jobs (one per task)
for task in "${TASKS[@]}"; do
    sbatch \
        --export=ALL,TASK="${task}",IS_BASE_MODEL=1 \
        --job-name="ev_${task}_base" \
        --output="tc_eval_${task}_base_%j.out" \
        "${WORKER}"
done

echo
echo "Submitted 9 jobs (6 FT + 3 base). Check status with:  squeue -u \$USER"
echo "Results will be written to: ${SCRIPT_DIR}/results_full/"
