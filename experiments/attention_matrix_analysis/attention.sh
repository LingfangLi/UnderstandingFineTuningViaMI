#!/bin/bash -l
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL

##SBATCH -o attention_kl_llama2-sst2_%j.out
#SBATCH -o attention_figures_all_in_one%j.out

# Request 40 cores on 1 node
#SBATCH --gres=gpu:1
#SBATCH -p  gpu-h100,gpu-a100-cs,gpu-a100-lowbig,gpu-a-lowsmall,gpu-l40s,gpu-v100
##SBATCH --exclude=gpu08
##SBATCH -p lowpriority  
#SBTACH -N 1

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


#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/attention_matrix_analysis/measure_attention_kl.py
#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/attention_matrix_analysis/kl_visualize_heatmap.py
python /users/sglli24/UnderstandingFineTuningViaMI/experiments/attention_matrix_analysis/kl_visulize_heatmap_all.py
#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/attention_matrix_analysis/old_measure_attention_kl.py
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