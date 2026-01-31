#!/bin/bash -l
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL

##SBATCH -o qwen2-senti-complex-train%j.out
#SBATCH -o qwen2-senti-complex-simple-test_%j.out

# Request 40 cores on 1 node
#SBATCH --gres=gpu:1
#SBATCH -pgpu-h100,gpu-a100-cs,gpu-a100-lowbig,gpu-a-lowsmall,gpu-l40s,gpu-v100

##SBTACH -N 1

#SBATCH -t 1-00:00:00


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

#python <PROJECT_ROOT>/experiments/text_complexity/fine_tune/qwen2_sentiment_train.py
python <PROJECT_ROOT>/experiments/text_complexity/fine_tune/Old-Qwen2-yelp-eval.py
#python <PROJECT_ROOT>/experiments/text_complexity/fine_tune/qwen2_qa_train.py
#python <PROJECT_ROOT>/experiments/text_complexity/fine_tune/qwen2_mt_train.py
#python <PROJECT_ROOT>/experiments/text_complexity/fine_tune/qwen-qa-eval.py
#python <PROJECT_ROOT>/experiments/text_complexity/fine_tune/qwen2-mt-eval.py

#python <PROJECT_ROOT>/experiments/text_complexity/fine_tune/llama2_mt_train.py
#python <PROJECT_ROOT>/experiments/text_complexity/fine_tune/llama2_sentiment_train.py
#python <PROJECT_ROOT>/experiments/text_complexity/fine_tune/llama2_qa_train.py

#python <PROJECT_ROOT>/experiments/text_complexity/fine_tune/llama2-qa-eval.py
#python <PROJECT_ROOT>/experiments/text_complexity/fine_tune/llama2-mt-eval.py
#python <PROJECT_ROOT>/experiments/text_complexity/fine_tune/llama-sentiment-eval.py
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