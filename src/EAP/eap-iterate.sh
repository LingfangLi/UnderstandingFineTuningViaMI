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

# Load environment
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

# CUDA memory settings
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export CUDA_LAUNCH_BLOCKING=1

echo ========================================================= 
echo SLURM job: submitted date = $(date)
date_start=$(date +%s)
echo ========================================================= 
echo Job output begins 
echo ----------------- 
hostname

# Paths
MODEL_DIR="/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models/" #"/mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models"
DATA_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/corrupted_data"
SCRIPT_PATH="/users/sglli24/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py"

# Main loop
for ft_path in "$MODEL_DIR"/*; do
    filename=$(basename "$ft_path")
    name_no_ext="${filename%.*}"
    
    # Initialize variables
    base_model=""
    task_name=""
    model_name=""

    #if [[ "$name_no_ext" == *-sst2 ]]; then
    #    echo " ������ sst2 ����: $filename"
    #    continue
    #fi

    #if [[ "$name_no_ext" == qwen2-* ]]; then
     #   base_model="Qwen/Qwen2-0.5B"
     #   model_name="qwen2"
      #  task_name="${name_no_ext#qwen2-}" 
        
    if [[ "$name_no_ext" == gpt2-* ]]; then
        base_model="gpt2"
        model_name="gpt2"
        task_name="${name_no_ext#gpt2-}"
        
    #elif [[ "$name_no_ext" == llama2-* ]]; then
        # ��ȡ��׺
    #    suffix="${name_no_ext#llama2-}"
        # �����Ҫ�ų��ض����񣬿�����������жϣ�����������ȡ
    #    base_model="meta-llama/Llama-2-7b-hf"
    #    model_name="llama2"
    #    task_name="$suffix"
        
    elif [[ "$name_no_ext" == llama3.2-* ]]; then
        base_model="meta-llama/Llama-3.2-1B"
        model_name="llama3.2"
        task_name="${name_no_ext#llama3.2-}"
      
    else
        echo "Error: unrecognized model format: $filename"
        continue
    fi
  
  
    # -----------------------------------------------------------
    # 1. �������޸��������߼���ֻ���� llama3.2 ͨ��
    # -----------------------------------------------------------
    # ����ļ��� ���� �� llama3.2 ��ͷ��ֱ������������һ��ѭ��
    #if [[ "$name_no_ext" != Qwen2-* ]]; then
        # ����һ����Ĭ����������ӡ��־����ˢ��
    #    continue
    #fi

    # -----------------------------------------------------------
    # 2. �Ƿ����� sst2 (������֮ǰ���߼�)
    # -----------------------------------------------------------
    #if [[ "$name_no_ext" == *-sst2 ]]; then
    #    echo ">> ���� sst2 ����: $filename"
    #    continue
    #fi

    # -----------------------------------------------------------
    # 3. ���� Llama 3.2 ר������ (��Ϊǰ���Ѿ������ˣ������Ȼ�� llama3.2)
    # -----------------------------------------------------------
    #base_model="meta-llama/Llama-2-7b-hf" #"meta-llama/Llama-3.2-1B"
    #model_name="llama2" #"llama3.2"
    
    # ��ȡ��������ȥ��ǰ׺ "llama3.2-"
    #task_name="${name_no_ext#llama2-}"

    #temp_name="${name_no_ext#Qwen2-0.5B_}"
    
    # �ڶ���: ȥ����׺ "_best" -> �õ� "yelp"
    #task_name="${temp_name%_best}"
    
    #base_model="Qwen/Qwen2-0.5B"
    #model_name="qwen2"
    
    # Set data path
    current_data_path="${DATA_DIR}/${task_name}_corrupted.csv"
    
    echo "Processing task: $task_name"
    echo "  Data path: $current_data_path"

    if [ ! -f "$current_data_path" ]; then
        echo "  [Error] Data file not found: $current_data_path"
        echo ""
        continue
    fi
    

    # Run EAP script
    echo "--------------------------------------------------"
    echo "Processing: $filename"
    echo "   Base Model: $base_model"
    echo "   Task Name : $task_name"
    echo "   Data Path : $current_data_path"
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
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo ========================================================= 
echo SLURM job: finished date = $(date) 
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================

