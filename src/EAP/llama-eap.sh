#!/bin/bash -l
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL


#SBATCH -o unified_eap_llama3.2-tatoeba%j.out

# Request 40 cores on 1 node
#SBATCH --gres=gpu:1
#SBATCH -p  gpu-a100-cs
#SBTACH -N 1
#SBATCH -n 16

##SBATCH -t 3-00:00:00


module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
#pip install cmapy  
#conda install -c conda-forge graphviz pygraphviz
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

#install eap pack for mt eap
#git clone https://github.com/hannamw/EAP-positional.git
#cd EAP-positional
#git checkout tutorial
#pip install -e .

echo ========================================================= 
echo SLURM job: submitted date = date 
date_start=$(date +%s)
echo ========================================================= 
echo Job output begins 
echo ----------------- 
echo
hostname
# $SLURM_NTASKS is defined automatically as the number of processes in the
# parallel environment.
export CUDA_LAUNCH_BLOCKING=1


#python -u /users/sglli24/fine-tuning-project/find_attention_change_both_wise_v2.py << EOF
#1
#EOF

#python /users/sglli24/UnderstandingFineTuningViaMI/src/EAP/LlaMA/Qlora_LlaMA2_EAP_sentiment.py
#python /users/sglli24/UnderstandingFineTuningViaMI/src/EAP/LlaMA/Qlora_LlaMA2_eap_unified.py
python /users/sglli24/UnderstandingFineTuningViaMI/src/EAP/LlaMA/eap_unified.py

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