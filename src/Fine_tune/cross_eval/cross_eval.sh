#!/bin/bash -l
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL


#SBATCH -o cross_eval_gpt2-llama3-model-sst2-task%j.out


# Request 40 cores on 1 node
#SBATCH --gres=gpu:1
##SBATCH -p  gpu-a100-cs
#SBATCH -p gpu-a100-cs
##,gpu-a-lowsmall,gpu-a100-lowbig,gpu-l40s
#SBTACH -N 1

#SBATCH -t 1-00:00:00


module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
#pip install pygraphviz cmapy
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


#python -u <PROJECT_ROOT>/fine-tuning-project/find_attention_change_both_wise_v2.py << EOF
#1
#EOF

#python <PROJECT_ROOT>/src/Fine_tune/cross_eval/universal_cross_eval.py
python <PROJECT_ROOT>/src/Fine_tune/cross_eval/gpt2-llama3-cross-eval.py
#python <PROJECT_ROOT>/src/Fine_tune/cross_eval/llama2_cross_eval.py
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