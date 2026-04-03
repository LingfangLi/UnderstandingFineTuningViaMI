#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o llama2-full-ft-%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=128G

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo =========================================================
echo "SLURM job: submitted date = $(date)"
date_start=$(date +%s)
echo =========================================================
echo "Job output begins"
echo -----------------
echo
hostname
nvidia-smi

# ============================================================
# Llama2-7B Full Fine-Tuning
# ============================================================

# --- Sentiment Classification ---
python /users/sglli24/UnderstandingFineTuningViaMI/src/Fine_tune/Sentiment_classification/Llama2-7b-yelp-full.py
python /users/sglli24/UnderstandingFineTuningViaMI/src/Fine_tune/Sentiment_classification/llama2-7b-sst2-full.py

# --- Question Answering ---
#python /users/sglli24/UnderstandingFineTuningViaMI/src/Fine_tune/Question_answering/LlaMA2-7b-squad-full.py
#python /users/sglli24/UnderstandingFineTuningViaMI/src/Fine_tune/Question_answering/LlaMA2-7b-coqa-full.py

# --- Machine Translation ---
#python /users/sglli24/UnderstandingFineTuningViaMI/src/Fine_tune/Machine_translation/LlaMA2-7b-kde4-full.py
#python /users/sglli24/UnderstandingFineTuningViaMI/src/Fine_tune/Machine_translation/LlaMA2-7b-tatoeba-full.py

echo ---------------
echo "Job output ends"
date_end=$(date +%s)
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================
echo "SLURM job: finished date = $(date)"
echo "Total run time : $hours Hours $minutes Minutes $seconds Seconds"
echo =========================================================
