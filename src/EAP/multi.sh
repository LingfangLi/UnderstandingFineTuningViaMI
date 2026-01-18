#!/bin/bash -l
#SBATCH -J EAP_ModelParallel
#SBATCH -o logs/eap_mp_%A_%a.out
#SBATCH -e logs/eap_mp_%A_%a.err
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -p gpu-a100-dacdt           
#SBATCH -N 1                    
#SBATCH --gres=gpu:2             
                                 
#SBATCH --cpus-per-task=16       
#SBATCH -t 1-00:00:00



module purge
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ========================================================= 
echo SLURM job: submitted date = date 
date_start=$(date +%s)
echo ========================================================= 
echo Job output begins 
echo ----------------- 
echo

DATA_DIR="/users/yangwr/Lingfang/UnderstandingFineTuningViaMI/output/corrupted_data"

if [ -z "$TASK_NAME" ]; then
    TASK_NAME="squad"
fi

echo "Running Single Task: $TASK_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

current_data_path="${DATA_DIR}/${TASK_NAME}_corrupted.csv"


python /users/yangwr/Lingfang/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py \
  --task $TASK_NAME \
  --model_name "llama2" \
  --mode "pretrained" \
  --data_path $current_data_path \
  --output_dir "/users/yangwr/Lingfang/UnderstandingFineTuningViaMI/output/EAP_edges/pretrained/" \
  --batch_size 1 

echo --------------- 
echo Job output ends 
date_end=$(date +%s)
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo ========================================================= 
echo SLURM job: finished date =$(date) 
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================