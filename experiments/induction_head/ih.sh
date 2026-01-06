#!/bin/bash -l
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL

##SBATCH -o detect_induction_head_sst2_finetuned_%j.out
##SBATCH -o check_k_value_induction_head_all_npy_%j.out
#SBATCH -o induction_head_important_edges_overlap%j.out

# Request 40 cores on 1 node
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a-lowsmall
##SBTACH -N 1

##SBATCH -t 3-00:00:00


module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
#pip install cmapy  
#conda install -c conda-forge graphviz pygraphviz
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

#pip install pandas seaborn matplotlib 
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

#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/induction_head/detect_induction_head.py
#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/induction_head/check_k_value_of_induction_head.py
python /users/sglli24/UnderstandingFineTuningViaMI/experiments/induction_head/analyze_head_top_edges_overlap.py
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