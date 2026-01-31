#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o cross_task_llama2-sst2_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs,gpu-h100,gpu-a100-lowbig,gpu-a-lowsmall,gpu-l40s
#SBATCH -N 1
#SBATCH -t 1-00:00:00

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
pip install tabulate
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export CUDA_LAUNCH_BLOCKING=1

# Paths
# Fine-tuned model directory
MODEL_DIR="<MODEL_STORAGE>/fine-tuning-project-1/old_version_finetuned_models"
# Corrupted data directory
DATA_DIR="<PROJECT_ROOT>/output/corrupted_data"
# Script path
SCRIPT_PATH="<PROJECT_ROOT>/src/EAP/eap_unified.py"
# Output directory for cross-task edges
OUTPUT_DIR="<PROJECT_ROOT>/output/EAP_edges/cross_task_edges"

# All task names
ALL_TASKS=("yelp" "sst2" "coqa" "squad" "kde4" "tatoeba")

mkdir -p "$OUTPUT_DIR"

# Main loop
for ft_path in "$MODEL_DIR"/*; do
    filename=$(basename "$ft_path")
    name_no_ext="${filename%.*}"
    
    # if [[ "$name_no_ext" != "llama2-yelp" ]]; then
    #     # �������ļ�����������׺��汾�ţ��� llama2-yelp-v1��������ʹ�� [[ "$name_no_ext" != *"llama2-yelp"* ]]
    #     continue
    # fi
    # ---------------------------------------------
 
    # Determine model type
    base_model=""
    model_short=""
    model_train_task=""

    if [[ "$name_no_ext" == gpt2-* ]]; then
        base_model="gpt2"
        model_short="gpt2"
        model_train_task="${name_no_ext#gpt2-}"
    elif [[ "$name_no_ext" == qwen2-* ]]; then
        base_model="Qwen/Qwen2-0.5B"
        model_short="qwen2"
        model_train_task="${name_no_ext#qwen2-}"
    elif [[ "$name_no_ext" == llama2-* ]]; then
        base_model="meta-llama/Llama-2-7b-hf"
        model_short="llama2"
        model_train_task="${name_no_ext#llama2-}"
    elif [[ "$name_no_ext" == llama3.2-* ]]; then
        base_model="meta-llama/Llama-3.2-1B"
        model_short="llama3.2"
        model_train_task="${name_no_ext#llama3.2-}"
    else
        echo "Error: unknown model format: $filename"
        continue
    fi

    # Model-level filter
    if [[ "$model_short" != "qwen2" ]]; then
        echo "Skipping model type: $model_short (Not in target list)"
        continue
    fi
    
    #if [[ "$model_train_task" != "sst2" ]]; then
    #    echo "  [Skip] model_train_task ($model_train_task) matches model task. Skipping."
    #    continue
    #    fi
        
    # --------------------------------

    echo "=================================================="
    echo "Processing Model: $model_short | Trained on: $model_train_task"
    echo "=================================================="

    # Inner loop: run on all other tasks' data
    for data_task in "${ALL_TASKS[@]}"; do

        # Skip if data task matches model training task
        if [[ "$data_task" == "$model_train_task" ]]; then
            echo "  [Skip] Data task ($data_task) matches model task. Skipping."
            continue
        fi
        
        #if [[ "$model_short" == "llama2" ]]; then
        #    if [[ "$data_task" != "squad" && "$data_task" != "coqa" ]]; then
                 # echo "  [Skip] Llama2 filter: skipping $data_task" 
        #         continue
         #   fi
        #fi
        # ---------------------------------------

        # Set data path
        current_data_path="${DATA_DIR}/${data_task}_corrupted.csv"
        if [ ! -f "$current_data_path" ]; then
            echo "  [Error] Data not found: $current_data_path"
            continue
        fi

        # Custom model name for descriptive output filenames
        custom_model_name="${model_short}_Finetuned-${model_train_task}_Corrupted-Data"

        echo "  -> Running on Data: $data_task"
        
        python "$SCRIPT_PATH" \
            --mode finetuned \
            --task "$data_task" \
            --base_model_name "$base_model" \
            --model_name "$custom_model_name" \
            --data_path "$current_data_path" \
            --ft_model_path "$ft_path" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size 1

    done
done

echo "Cross-task generation finished."