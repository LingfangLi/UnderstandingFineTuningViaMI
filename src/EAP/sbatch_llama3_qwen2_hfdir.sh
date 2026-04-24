#!/bin/bash -l
# Run Llama-3.2 + Qwen2 cross-task EAP v2 (source = /mnt/data1/.../fine_tuned_models/)
# on the gpu-a100-cs partition.
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o /users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/sbatch_llama3_qwen2_hfdir_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH --job-name=llama3qwen2_eap_v2

set -uo pipefail

PROJECT_ROOT="/users/sglli24/UnderstandingFineTuningViaMI"

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "========================================================="
echo "SLURM job $SLURM_JOB_ID on $(hostname) started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================================="

start_ts=$(date +%s)

echo; echo "===== Llama-3.2 v2 (source=fine_tuned_models) ====="
bash "${PROJECT_ROOT}/src/EAP/run_llama3_all_edges_v2.sh" 2>&1 \
    | tee -a "${PROJECT_ROOT}/output/EAP_edges/llama3_cross_task_hfdir/_run.log" \
    | grep -E "^\[|===|ERROR" | tail -80

echo; echo "===== Qwen2 v2 (source=fine_tuned_models) ====="
bash "${PROJECT_ROOT}/src/EAP/run_qwen2_all_edges_v2.sh" 2>&1 \
    | tee -a "${PROJECT_ROOT}/output/EAP_edges/qwen2_cross_task_hfdir/_run.log" \
    | grep -E "^\[|===|ERROR" | tail -80

end_ts=$(date +%s)
sec=$((end_ts-start_ts)); hh=$((sec/3600)); mm=$(((sec%3600)/60)); ss=$((sec%60))
echo; echo "========================================================="
echo "Finished at $(date)  |  wall clock: ${hh}h ${mm}m ${ss}s"
echo "  llama3_cross_task_hfdir/: $(ls ${PROJECT_ROOT}/output/EAP_edges/llama3_cross_task_hfdir 2>/dev/null | wc -l) files"
echo "  qwen2_cross_task_hfdir/:  $(ls ${PROJECT_ROOT}/output/EAP_edges/qwen2_cross_task_hfdir 2>/dev/null | wc -l) files"
echo "========================================================="
