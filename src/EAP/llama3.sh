#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o unified_eap_llama3_final.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-h100,gpu-a100-cs,gpu-a100-lowbig,gpu-l40s
#SBATCH -N 1
#SBATCH -t 1-00:00:00

# Load environment
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

# CUDA memory settings
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export CUDA_LAUNCH_BLOCKING=1

echo ========================================================= 
echo SLURM job: submitted date = $(date)
date_start=$(date +%s)
hostname
echo ========================================================= 

# Paths
MODEL_DIR="/mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models"
DATA_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/corrupted_data"
SCRIPT_PATH="/users/sglli24/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py"

# Main loop
for ft_path in "$MODEL_DIR"/*; do
    filename=$(basename "$ft_path")
    
    # Skip non-llama3.2 models
    if [[ "$filename" != llama3.2-* ]]; then
        continue
    fi

    # Skip sst2 tasks
    if [[ "$filename" == *-sst2 ]]; then
        echo "Skipping sst2: $filename"
        continue
    fi

    base_model="meta-llama/Llama-3.2-1B"
    model_name="llama3.2"
    task_name="${filename#llama3.2-}"
    
    current_data_path="${DATA_DIR}/${task_name}_corrupted.csv"

    if [ ! -f "$current_data_path" ]; then
        echo "Error: data file not found: $current_data_path"
        continue
    fi

    echo "--------------------------------------------------"
    echo "Processing: $filename"
    echo "   Base Model: $base_model"
    echo "   Task Name : $task_name"
    echo "--------------------------------------------------"

    python "$SCRIPT_PATH" \
        --task "$task_name" \
        --base_model_name "$base_model" \
        --model_name "$model_name" \
        --data_path "$current_data_path" \
        --ft_model_path "$ft_path"

    echo "Done: $filename"
    echo ""

done

echo ----------------- 
echo Job output ends 
date_end=$(date +%s)
seconds=$((date_end-date_start))
minutes=$((seconds/60))
hours=$((minutes/60))
echo Total run time : $hours Hours $minutes Minutes