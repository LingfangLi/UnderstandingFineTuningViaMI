#!/bin/bash -l
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL


#SBATCH -o llama3-mt-sample-distance%j.out
##SBATCH -o llama2-qa-sample_filter%j.out

##SBATCH -o llama3-yelp-barchart%j.out
# Request 40 cores on 1 node
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs,gpu-h100,gpu-a100-lowbig,gpu-a-lowsmall,gpu-l40s,gpu-v100
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

#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/PCA_yelp_samples_filter_dcreased_distance.py
#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/PCA_Squad_samples_filter_dcreased_distance.py
python  /users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/PCA_KDE4_samples_filter_decreased_distance.py

#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/qa_fitler_samples_f1.py
#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/qa-llama2-filter-f1.py
#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/filter_samples_bleu.py

#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/filter_sample_sentiment.py
#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/plot_results_from_json.py

#python -u /users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/llama2_PCA_distance.py

#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/universal_barchart.py

#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/check_llama3_key.py

#python /users/sglli24/UnderstandingFineTuningViaMI/experiments/Layerwise_Representation_Distance_Analysis/check.py
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