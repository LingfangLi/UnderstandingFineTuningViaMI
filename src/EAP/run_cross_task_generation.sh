#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o cross_task_llama2_yelp_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs,gpu-h100
#gpu-a100-lowbig,gpu-a-lowsmall,gpu-l40s
#SBATCH -N 1
#SBATCH -t 1-00:00:00

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export CUDA_LAUNCH_BLOCKING=1

# ================= 配置路径 =================
# 微调模型所在的文件夹
MODEL_DIR="/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models"
# Corrupted 数据所在的文件夹
DATA_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/corrupted_data"
# Python 脚本路径
SCRIPT_PATH="/users/sglli24/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py"
# 新的输出目录：存放交叉任务生成的 Edge
OUTPUT_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/cross_task_edges"

# 定义所有任务列表
ALL_TASKS=("yelp" "sst2" "coqa" "squad" "kde4" "tatoeba")

mkdir -p "$OUTPUT_DIR"

# ================= 循环逻辑 =================
for ft_path in "$MODEL_DIR"/*; do
    filename=$(basename "$ft_path")
    name_no_ext="${filename%.*}"
    
    if [[ "$name_no_ext" != "llama2-yelp" ]]; then
        # 如果你的文件名有其他后缀或版本号（如 llama2-yelp-v1），可以使用 [[ "$name_no_ext" != *"llama2-yelp"* ]]
        continue
    fi

    # --- 解析模型与原任务 ---
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
        echo "跳过未知格式: $filename"
        continue
    fi

    echo "=================================================="
    echo "Processing Model: $model_short | Trained on: $model_train_task"
    echo "=================================================="

    # --- 内部循环：遍历所有数据集 ---
    for data_task in "${ALL_TASKS[@]}"; do
        
        # 1. 如果数据任务 == 模型训练任务，跳过 (之前已经算过了)
        if [[ "$data_task" == "$model_train_task" ]]; then
            echo "  [Skip] Data task ($data_task) matches model task. Skipping."
            continue
        fi

        # 2. 构造数据路径
        current_data_path="${DATA_DIR}/${data_task}_corrupted.csv"
        if [ ! -f "$current_data_path" ]; then
            echo "  [Error] Data not found: $current_data_path"
            continue
        fi

        # 3. 构造特殊的 model_name 以便生成区分度高的文件名
        # 结果文件名将是: {model_short}_finetuned_{model_train_task}_{data_task}_finetuned_edges.csv
        # 例如: gpt2_finetuned_yelp_sst2_finetuned_edges.csv
        # 含义: GPT2 (Yelp Finetuned) running on SST2
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