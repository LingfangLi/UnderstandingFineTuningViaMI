#!/bin/bash -l
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL


##SBATCH -o train_old-llama2-squad%j.out

#SBATCH -o test-old-llama3-sst2-coqa%j.out

##SBATCH -o test-old-llama2-yelp-coqa%j.out


# Request 40 cores on 1 node
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a-lowsmall
##SBATCH --exclude=gpu08 gpu-a100-cs, gpu-h100,gpu-a100-lowbig,
#,gpu-l40s,gpu-l40s-low
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
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/gpt2-squad-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/gpt2-coqa-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/gpt2-squad-eval.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/gpt2-coqa-eval.py
python <PROJECT_ROOT>/src/Fine_tune/Question_answering/old-gpt2-qa-eval.py

#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/CLM_universe_finetune.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/qwen-qa-eval.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/old_llama2_coqa.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/old_llama2_squad.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/old_llama2_qa_eval.py

#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/llama3.2-squad-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/llama3.2-coqa-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/llama3.2-squad-eval.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/llama3.2-coqa-eval.py

#python <PROJECT_ROOT>/src/Fine_tune/Sentiment_classification/Llama2-7b-yelp-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/LlaMA2-7b-squad-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/LlaMA2-squad-eval.py

#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/LlaMA2-7b-coqa-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/LlaMA2-coqa-eval.py

#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/Qwen2-0.5b-squad-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/Qwen2-squad-eval.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/Qwen2-0.5b-coqa-data.py
#python <PROJECT_ROOT>/src/Fine_tune/Question_answering/Qwen2-coqa-eval.py
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