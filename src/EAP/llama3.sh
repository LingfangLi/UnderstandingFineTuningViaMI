#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o unified_eap_llama3_final.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-h100,gpu-a100-cs,gpu-a100-lowbig,gpu-l40s
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
hostname
echo ========================================================= 

# ================= 配置路径 =================
MODEL_DIR="/mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models"
DATA_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/corrupted_data"
SCRIPT_PATH="/users/sglli24/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py"

# ================= 循环逻辑 =================
for ft_path in "$MODEL_DIR"/*; do
    # 1. 获取文件夹名称 (例如: llama3.2-kde4)
    filename=$(basename "$ft_path")
    
    # -----------------------------------------------------------
    # 核心筛选逻辑 (基于你的截图)
    # -----------------------------------------------------------
    
    # [筛选1] 如果名字不是以 llama3.2 开头，直接跳过
    # 注意：这里直接用 filename，不要做任何裁剪
    if [[ "$filename" != llama3.2-* ]]; then
        continue
    fi

    # [筛选2] 如果包含 sst2，跳过 (按需保留)
    if [[ "$filename" == *-sst2 ]]; then
        echo " [跳过] sst2 任务: $filename"
        continue
    fi

    # -----------------------------------------------------------
    # 参数配置
    # -----------------------------------------------------------
    base_model="meta-llama/Llama-3.2-1B"
    model_name="llama3.2"
    
    # 提取任务名：使用 Bash 字符串删除前缀功能
    # ${filename#llama3.2-} 意思是：删掉开头的 "llama3.2-"，剩下的就是任务名
    # 例如: llama3.2-kde4 -> kde4
    task_name="${filename#llama3.2-}"
    
    # -----------------------------------------------------------
    # 检查数据 & 运行
    # -----------------------------------------------------------
    current_data_path="${DATA_DIR}/${task_name}_corrupted.csv"

    if [ ! -f "$current_data_path" ]; then
        echo " ? [错误] 数据文件未找到，跳过: $current_data_path"
        continue
    fi

    echo "--------------------------------------------------"
    echo "正在运行: $filename"
    echo "   Base Model: $base_model"
    echo "   Task Name : $task_name"
    echo "--------------------------------------------------"

    python "$SCRIPT_PATH" \
        --task "$task_name" \
        --base_model_name "$base_model" \
        --model_name "$model_name" \
        --data_path "$current_data_path" \
        --ft_model_path "$ft_path"

    echo "完成: $filename"
    echo ""

done

echo ----------------- 
echo Job output ends 
date_end=$(date +%s)
seconds=$((date_end-date_start))
minutes=$((seconds/60))
hours=$((minutes/60))
echo Total run time : $hours Hours $minutes Minutes