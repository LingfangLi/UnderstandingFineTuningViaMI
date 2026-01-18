#!/bin/bash -l
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o unified_eap_loop_qwen%j.out
# Request 40 cores on 1 node
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs,gpu-h100,gpu-a100-lowbig,gpu-a-lowsmall,gpu-l40s
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
MODEL_DIR="/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models/" #"/mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models"
DATA_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/corrupted_data"
SCRIPT_PATH="/users/sglli24/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py"

# ================= 循环逻辑 =================
for ft_path in "$MODEL_DIR"/*; do
    # 获取文件名
    filename=$(basename "$ft_path")
    name_no_ext="${filename%.*}"
    
    # 初始化变量
    base_model=""
    task_name=""
    model_name=""

    # --- 1. 修复：增加空格 ---
    #if [[ "$name_no_ext" == *-sst2 ]]; then
    #    echo " 跳过非 sst2 任务: $filename"
    #    continue
    #fi

    # --- 2. 修复：取消注释并整理逻辑 ---
    #if [[ "$name_no_ext" == qwen2-* ]]; then
     #   base_model="Qwen/Qwen2-0.5B"
     #   model_name="qwen2"
      #  task_name="${name_no_ext#qwen2-}" 
        
    if [[ "$name_no_ext" == gpt2-* ]]; then
        base_model="gpt2"
        model_name="gpt2"
        task_name="${name_no_ext#gpt2-}"
        
    #elif [[ "$name_no_ext" == llama2-* ]]; then
        # 提取后缀
    #    suffix="${name_no_ext#llama2-}"
        # 如果需要排除特定任务，可以在这里加判断，否则正常提取
    #    base_model="meta-llama/Llama-2-7b-hf"
    #    model_name="llama2"
    #    task_name="$suffix"
        
    elif [[ "$name_no_ext" == llama3.2-* ]]; then
        base_model="meta-llama/Llama-3.2-1B"
        model_name="llama3.2"
        task_name="${name_no_ext#llama3.2-}"
      
    else
        echo "?? 跳过无法识别的模型格式: $filename"
        continue
    fi
  
  
    # -----------------------------------------------------------
    # 1. 【核心修复】过滤逻辑：只允许 llama3.2 通过
    # -----------------------------------------------------------
    # 如果文件名 不是 以 llama3.2 开头，直接跳过进入下一次循环
    #if [[ "$name_no_ext" != Qwen2-* ]]; then
        # 这是一个静默跳过，不打印日志以免刷屏
    #    continue
    #fi

    # -----------------------------------------------------------
    # 2. 是否跳过 sst2 (保留你之前的逻辑)
    # -----------------------------------------------------------
    #if [[ "$name_no_ext" == *-sst2 ]]; then
    #    echo ">> 跳过 sst2 任务: $filename"
    #    continue
    #fi

    # -----------------------------------------------------------
    # 3. 设置 Llama 3.2 专属参数 (因为前面已经过滤了，这里必然是 llama3.2)
    # -----------------------------------------------------------
    #base_model="meta-llama/Llama-2-7b-hf" #"meta-llama/Llama-3.2-1B"
    #model_name="llama2" #"llama3.2"
    
    # 提取任务名：去掉前缀 "llama3.2-"
    #task_name="${name_no_ext#llama2-}"

    #temp_name="${name_no_ext#Qwen2-0.5B_}"
    
    # 第二步: 去掉后缀 "_best" -> 得到 "yelp"
    #task_name="${temp_name%_best}"
    
    #base_model="Qwen/Qwen2-0.5B"
    #model_name="qwen2"
    
    # --- 构建 Data Path ---
    current_data_path="${DATA_DIR}/${task_name}_corrupted.csv"
    
    # --- 调试信息：打印出来看看路径对不对 ---
    echo "正在检查任务: $task_name"
    echo "  -> 数据路径: $current_data_path"

    if [ ! -f "$current_data_path" ]; then
        echo "  [错误] 数据文件不存在，跳过: $current_data_path"
        echo ""
        continue
    fi
    

    # --- 运行 Python 脚本 ---
    echo "--------------------------------------------------"
    echo "正在运行: $filename"
    echo "   Base Model: $base_model"
    echo "   Task Name : $task_name"
    echo "   Data Path : $current_data_path"
    echo "--------------------------------------------------"

    # --- 3. 修复：必须传入 ft_model_path，否则 Python 脚本不执行分析 ---
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
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo ========================================================= 
echo SLURM job: finished date = $(date) 
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================

