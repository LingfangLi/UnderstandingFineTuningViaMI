#!/bin/bash -l
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL


##SBATCH -o EVAL-old-llama3-sst2-yelp-data%j.out
#SBATCH -o eval-old-llama2-sst2-yelp-data%j.out
##SBATCH -o test_old-base-qwen2-sst2-1.2tep-50Mtoken%j.out
##SBATCH -o test_qwen-full-tune-sst2-%j.out

##SBATCH -o test_pretrained_llama2_yelp%j.out
##SBATCH -o train_llama2-full-tune-sst2-%j.out
##SBATCH -o test_pretrained_llama2-full-tune-sst2-%j.out

##SBATCH -o test_pretrained_gpt2-sst2-%j.out

##SBATCH -o train_llama3-full-tune-sst2-%j.out
##SBATCH -o test_llama3-full-tune-sst2-%j.out
##SBATCH -o test_pretrained_llama3-sst2-%j.out

# Request 40 cores on 1 node
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a-lowsmall
##gpu-a100-cs
## 
##
##SBATCH -p gpu-a100-cs
#SBTACH -N 1

##SBATCH -t 10:00:00


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

#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/Llama2-7b-yelp-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/LlaMA2-eval.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/llama2-7b-sst2-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/llama2-sst2-eval.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/old-llama2-sst2-eval.py
python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/old_llama2_yelp_eval.py

#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/old_llama2_sst2.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/old_llama2_yelp.py

#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/CLM_universe_finetune.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/sst2-eval.py

#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/Qwen2-0.5b-yelp-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/Qwen2-eval.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/qwen2-sst2-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/qwen2-sst2-eval.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/Old-Qwen2-yelp-eval.py

#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/old-gpt2-yelp-eval.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/gpt2-sst2-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/GPT2-eval.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/gpt2-yelp-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/gpt2-yelp-eval.py

#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/llama3.2-sst2-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/llama3.2-yelp-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/llama3.2-yelp-eval.py
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