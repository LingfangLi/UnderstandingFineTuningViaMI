#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o unified_eap_sst2_full-%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-h100,gpu-a100-cs,gpu-v100,gpu-l40s
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
echo ----------------- 
hostname

# Paths
MODEL_DIR="/mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models"
DATA_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/corrupted_data"
SCRIPT_PATH="/users/sglli24/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py"
OUTPUT_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges"

# Main loop
for ft_path in "$MODEL_DIR"/*; do
    filename=$(basename "$ft_path")
    
    if [ -d "$ft_path" ]; then
        name_no_ext="$filename"
    else
        name_no_ext="${filename%.*}"
    fi

    base_model=""
    task_name=""
    model_name=""

    # Filter: only process sst2 tasks
    if [[ "$name_no_ext" != *-sst2 ]]; then
        # echo " ������ sst2 ����: $name_no_ext"
        continue
    fi

    # Determine model type and extract task
    if [[ "$name_no_ext" == qwen2-* ]]; then
        base_model="Qwen/Qwen2-0.5B"
        model_name="qwen2"
        task_name="sst2"
    elif [[ "$name_no_ext" == gpt2-* ]]; then
        base_model="gpt2"
        model_name="gpt2"
        task_name="sst2"
    elif [[ "$name_no_ext" == llama2-* ]]; then
        base_model="meta-llama/Llama-2-7b-hf"
        model_name="llama2"
        task_name="sst2"
    elif [[ "$name_no_ext" == llama3.2-* ]]; then
        base_model="meta-llama/Llama-3.2-1B"
        model_name="llama3.2"
        task_name="sst2"
    else
        echo "Error: unrecognized model format: $filename"
        continue
    fi
   
    # Set data path
    current_data_path="${DATA_DIR}/${task_name}_corrupted.csv"

    if [ ! -f "$current_data_path" ]; then
        echo "Error: data file not found: $current_data_path"
        continue
    fi

    echo "--------------------------------------------------"
    echo "Processing: $name_no_ext"
    echo "  Base Model: $base_model"
    echo "--------------------------------------------------"

    # Run finetuned mode
    echo ">>> [1/2] Running FINETUNED mode..."
    python "$SCRIPT_PATH" \
        --task "$task_name" \
        --base_model_name "$base_model" \
        --model_name "$model_name" \
        --data_path "$current_data_path" \
        --ft_model_path "$ft_path" \
        --output_dir "$OUTPUT_DIR" \
        --mode "finetuned"

    # Run pretrained mode (skip if already exists)
    pretrained_file="${OUTPUT_DIR}/pretrained/${model_name}_${task_name}_pretrained_edges.csv"
    
    if [ -f "$pretrained_file" ]; then
        echo ">>> [2/2] Pretrained edges already exist at $pretrained_file. Skipping."
    else
        echo ">>> [2/2] Running PRETRAINED mode..."
        python "$SCRIPT_PATH" \
            --task "$task_name" \
            --base_model_name "$base_model" \
            --model_name "$model_name" \
            --data_path "$current_data_path" \
            --output_dir "$OUTPUT_DIR" \
            --mode "pretrained"
    fi

    echo "Done: $filename"
    echo ""

done

echo ----------------- 
echo Job output ends 
date_end=$(date +%s)
echo Total run time : $((date_end-date_start)) seconds