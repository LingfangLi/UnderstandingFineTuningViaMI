#!/bin/bash -l
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o unified_eap_loop_sst2-%j.out
# Request 40 cores on 1 node
#SBATCH --gres=gpu:1
#SBATCH -p gpu-h100,gpu-a100-cs,gpu-v100,gpu-l40s
##SBATCH -p gpu-h100
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
echo ========================================================= 
echo Job output begins 
echo ----------------- 
hostname

# ================= 配置路径 =================
# 1. 微调模型所在的目录 (根据你的图片路径)
MODEL_DIR="/mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models"

# 2. 数据集所在的目录 (CSV文件)
DATA_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/corrupted_data"

# 3. Python 脚本路径
SCRIPT_PATH="/users/sglli24/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py"

# ================= 循环逻辑 =================
# 遍历目录下的所有文件和文件夹
for ft_path in "$MODEL_DIR"/*; do
    # 获取文件名 (例如: llama3.2-tatoeba.pt 或 llama2-kde4)
    filename=$(basename "$ft_path")
    
    # 去掉文件后缀 (针对 .pt 文件)，文件夹名字不受影响
    # 结果: llama3.2-tatoeba.pt -> llama3.2-tatoeba
    name_no_ext="${filename%.*}"
    
    # 初始化变量
    base_model=""
    task_name=""
    model_name=""
    if [["$name_no_ext" != *-sst2 ]]; then
        echo " 跳过非 sst2 任务: $filename"
        continue
    fi
    # --- 自动判定模型和提取任务名 ---
    if [[ "$name_no_ext" == qwen2-* ]]; then
        # 匹配 llama3.2 (例如 llama3.2-tatoeba)
        base_model="Qwen/Qwen2-0.5B"
        model_name="qwen2"
        #task_name="${name_no_ext#qwen2-}" 
    elif [[ "$name_no_ext" == gpt2-* ]]; then
        # 匹配 gpt2 (例如 gpt2-yelp)
        base_model="gpt2"
        model_name="gpt2"
        # 提取 gpt2- 之后的部分作为 task_name
        task_name="${name_no_ext#gpt2-}"
        
    elif [[ "$name_no_ext" == llama2-* ]]; then
        suffix="${name_no_ext#llama2-}"
         # 排除 tatoeba 和 yelp
        #if [[ "$suffix" == "tatoeba" || "$suffix" == "yelp" ]]; then
         #   echo "跳过任务: $name_no_ext"
        #    continue
        #fi
        
        # 匹配 llama2 (例如 llama2-kde4)
        base_model="meta-llama/Llama-2-7b-hf"
        model_name="llama2"
        task_name="${name_no_ext#llama2-}"
        
    elif [[ "$name_no_ext" == llama3.2-* ]]; then
        # 匹配 llama3.2 (例如 llama3.2-tatoeba)
        base_model="meta-llama/Llama-3.2-1B"
        model_name="llama3.2"
        task_name="${name_no_ext#llama3.2-}"
      
    else
        echo "?? 跳过无法识别的模型格式: $filename"
        continue
    fi
  

    # --- 构建 Data Path ---
    # 假设 CSV 命名格式为: {task_name}_corrupted.csv
    current_data_path="${DATA_DIR}/${task_name}_corrupted.csv"

    # 检查数据文件是否存在
    if [ ! -f "$current_data_path" ]; then
        echo "? 错误: 数据文件不存在，跳过此任务: $current_data_path"
        continue
    fi

    # --- 运行 Python 脚本 ---
    echo "--------------------------------------------------"
    echo "正在运行: $filename"
    echo "   Base Model: $base_model"
    echo "   Task Name : $task_name"
    echo "   Data Path : $current_data_path"
    echo "--------------------------------------------------"

    # 假设你的 Python 代码接收以下参数: 
    # --ft_model_path, --base_model_name, --data_path
    # 如果你的参数名称不同，请在下面修改
    
    python "$SCRIPT_PATH" \
        --task "$task_name"\
        --base_model_name "$base_model" \
        --model_name "$model_name"\
        --data_path "$current_data_path"
        #--ft_model_path "$ft_path" \
    echo "完成: $filename"
    echo ""

done

echo ----------------- 
echo Job output ends 
date_end=$(date +%s)
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo ========================================================= 
echo SLURM job: finished date = $(date) 
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================

