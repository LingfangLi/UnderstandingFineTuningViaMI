#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o unified_eap_sst2_full-%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-h100,gpu-a100-cs,gpu-v100,gpu-l40s
#SBATCH -N 1
#SBATCH -t 1-00:00:00

# 加载环境
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export CUDA_LAUNCH_BLOCKING=1

echo ========================================================= 
echo SLURM job: submitted date = $(date)
date_start=$(date +%s)
echo ----------------- 
hostname

# ================= 配置路径 =================
MODEL_DIR="/mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models"
DATA_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/corrupted_data"
SCRIPT_PATH="/users/sglli24/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py"
OUTPUT_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges"

# ================= 循环逻辑 =================
for ft_path in "$MODEL_DIR"/*; do
    filename=$(basename "$ft_path")
    
    # === 修改处 ===
    if [ -d "$ft_path" ]; then
        # 目录：不切后缀
        name_no_ext="$filename"
    else
        # 文件：切去 .pt 后缀
        name_no_ext="${filename%.*}"
    fi
    # =============

    # 初始化变量
    base_model=""
    task_name=""
    model_name=""

    # 过滤: 只跑 sst2 任务
    if [[ "$name_no_ext" != *-sst2 ]]; then
        # echo " 跳过非 sst2 任务: $name_no_ext"
        continue
    fi

    # --- 自动判定模型和提取任务名 ---
    if [[ "$name_no_ext" == qwen2-* ]]; then
        base_model="Qwen/Qwen2-0.5B"
        model_name="qwen2"
        # 假设命名格式 qwen2-sst2 -> task is sst2
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
        echo "?? 跳过无法识别的模型格式: $filename"
        continue
    fi
   
    # 3. 构建 Data Path
    current_data_path="${DATA_DIR}/${task_name}_corrupted.csv"

    if [ ! -f "$current_data_path" ]; then
        echo "? 错误: 数据文件不存在: $current_data_path"
        continue
    fi

    echo "--------------------------------------------------"
    echo "处理任务: $name_no_ext"
    echo "  Base Model: $base_model"
    echo "--------------------------------------------------"

    # ========================================================
    # 步骤 A: 运行微调模型 (Finetuned Mode)
    # ========================================================
    echo ">>> [1/2] Running FINETUNED mode..."
    python "$SCRIPT_PATH" \
        --task "$task_name" \
        --base_model_name "$base_model" \
        --model_name "$model_name" \
        --data_path "$current_data_path" \
        --ft_model_path "$ft_path" \
        --output_dir "$OUTPUT_DIR" \
        --mode "finetuned"

    # ========================================================
    # 步骤 B: 运行预训练模型 (Pretrained Mode)
    # ========================================================
    # 检查是否已经跑过该 Base Model + Task 的组合，避免重复跑
    # Python脚本保存路径示例: output_dir/pretrained/llama2_sst2_pretrained_edges.csv
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

    echo "完成: $filename"
    echo ""

done

echo ----------------- 
echo Job output ends 
date_end=$(date +%s)
# ... (时间计算代码保持不变) ...
echo Total run time : $((date_end-date_start)) seconds