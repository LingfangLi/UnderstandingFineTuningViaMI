#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o overlap_analysis_unified_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs,gpu-a-lowsmall,gpu-a100-lowbig,gpu-v100,gpu-l40s
#SBATCH -t 1-00:00:00

# Load environment
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
pip install tabulate pandas

# Configuration parameters
# Arg 1: run mode (task1, task2, all)
RUN_MODE=${1:-task1}
# Arg 2: target model (gpt2, llama2, qwen2, llama3.2, all)
FILTER_MODEL=${2:-qwen2}
# Arg 3: target source task (sst2, yelp, coqa, ..., all)
FILTER_TASK=${3:-all}

echo "=================================================="
echo "Config:"
echo "  Mode:   $RUN_MODE"
echo "  Model:  $FILTER_MODEL"
echo "  Task:   $FILTER_TASK"
echo "=================================================="

# Paths
SCRIPT_PATH="/users/sglli24/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py"
BASE_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges"

CROSS_TASK_DIR="${BASE_DIR}/cross_task_edges"
STD_FT_DIR="${BASE_DIR}/old-version-finetuned"
PRETRAINED_DIR="${BASE_DIR}/pretrained"
SUMMARY_DIR="${CROSS_TASK_DIR}/summary_tables"

# Output directories
OUT_OVERLAP_T1="${CROSS_TASK_DIR}/overlap_cross_vs_target_ft"
OUT_OVERLAP_T2="${CROSS_TASK_DIR}/overlap_source_ft_vs_target_pt"

mkdir -p "$OUT_OVERLAP_T1"
mkdir -p "$OUT_OVERLAP_T2"
mkdir -p "$SUMMARY_DIR"

# Log file setup
# Master log (append across runs)
MASTER_LOG="${SUMMARY_DIR}/raw_data_log.csv"

# Per-run incremental log
INCREMENTAL_LOG="${SUMMARY_DIR}/temp_run_${SLURM_JOB_ID}_log.csv"

# Initialize incremental log
echo "Model_Arch,Type,Source_Task,Target_Task,Count" > "$INCREMENTAL_LOG"

# Initialize master log if not exists
if [ ! -f "$MASTER_LOG" ]; then
    echo "Model_Arch,Type,Source_Task,Target_Task,Count" > "$MASTER_LOG"
fi

# All possible tasks for diagonal analysis
ALL_TASKS_LIST=("coqa" "kde4" "squad" "sst2" "tatoeba" "yelp")

# Cross-task analysis
echo ">> Starting Cross-Task Analysis..."

for cross_file in "$CROSS_TASK_DIR"/*_finetuned_edges.csv; do
    [ -e "$cross_file" ] || continue
    
    filename=$(basename "$cross_file")
    IFS='_' read -r -a parts <<< "${filename%.csv}"
    
    model_arch="${parts[0]}"      # e.g., llama2
    raw_source="${parts[1]}"      # e.g., Finetuned-sst2
    source_task="${raw_source#Finetuned-}" 
    target_task="${parts[3]}"     # e.g., yelp

    # Filter by model
    if [[ "$FILTER_MODEL" != "all" && "$model_arch" != "$FILTER_MODEL" ]]; then continue; fi
    # Filter by source task
    if [[ "$FILTER_TASK" != "all" ]]; then
        if [[ "${source_task,,}" != "${FILTER_TASK,,}" ]]; then continue; fi
    fi

    echo "Processing: $model_arch | $source_task -> $target_task"

    # --- Task 1: Cross vs Target FT ---
    if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "task1" ]]; then
        std_ft_file="${STD_FT_DIR}/${model_arch}_${target_task}_finetuned_edges.csv"
        if [ -f "$std_ft_file" ]; then
            output=$(python "$SCRIPT_PATH" --mode compare --output_dir "$OUT_OVERLAP_T1" --edge_file_1 "$cross_file" --edge_file_2 "$std_ft_file")
            count=$(echo "$output" | grep "Intersection (Common):" | awk -F': ' '{print $2}')
            if [[ -n "$count" ]]; then
                echo "${model_arch},T1_Cross_TargetFT,${source_task},${target_task},${count}" >> "$INCREMENTAL_LOG"
            fi
        fi
    fi

    # --- Task 2: Source FT vs Target PT ---
    if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "task2" ]]; then
        source_ft_file="${STD_FT_DIR}/${model_arch}_${source_task}_finetuned_edges.csv"
        target_pt_file="${PRETRAINED_DIR}/${model_arch}_${target_task}_pretrained_edges.csv"
        if [ -f "$source_ft_file" ] && [ -f "$target_pt_file" ]; then
            output=$(python "$SCRIPT_PATH" --mode compare --output_dir "$OUT_OVERLAP_T2" --edge_file_1 "$source_ft_file" --edge_file_2 "$target_pt_file")
            count=$(echo "$output" | grep "Intersection (Common):" | awk -F': ' '{print $2}')
            if [[ -n "$count" ]]; then
                echo "${model_arch},T2_SourceFT_TargetPT,${source_task},${target_task},${count}" >> "$INCREMENTAL_LOG"
            fi
        fi
    fi
done

# Diagonal analysis (self-overlap)
echo "--------------------------------------------------"
echo ">> Checking Diagonal (Self-Overlap) for selected filters..."

# Determine target model list
if [[ "$FILTER_MODEL" == "all" ]]; then
    TARGET_MODELS=("gpt2" "llama2" "qwen2" "llama3.2")
else
    TARGET_MODELS=("$FILTER_MODEL")
fi

# Determine target task list
if [[ "$FILTER_TASK" == "all" ]]; then
    TARGET_TASKS=("${ALL_TASKS_LIST[@]}")
else
    TARGET_TASKS=("$FILTER_TASK")
fi

for arch in "${TARGET_MODELS[@]}"; do
    for task in "${TARGET_TASKS[@]}"; do
        
        # Only run diagonal for task2 or all modes
        if [[ "$RUN_MODE" != "task1" ]]; then
            
            source_ft_file="${STD_FT_DIR}/${arch}_${task}_finetuned_edges.csv"
            target_pt_file="${PRETRAINED_DIR}/${arch}_${task}_pretrained_edges.csv"

            if [ -f "$source_ft_file" ] && [ -f "$target_pt_file" ]; then
                # echo "  Diagonal Check: $arch - $task"
                output=$(python "$SCRIPT_PATH" --mode compare --output_dir "$OUT_OVERLAP_T2" --edge_file_1 "$source_ft_file" --edge_file_2 "$target_pt_file")
                count=$(echo "$output" | grep "Intersection (Common):" | awk -F': ' '{print $2}')
                
                if [[ -n "$count" ]]; then
                    echo "${arch},T2_SourceFT_TargetPT,${task},${task},${count}" >> "$INCREMENTAL_LOG"
                    echo "  [Diagonal] $arch: $task vs $task -> $count"
                fi
            fi
        fi
    done
done

# Merge data and generate summary tables
echo "--------------------------------------------------"
echo ">> Merging Data and Updating Master Log..."

python3 - <<EOF
# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys

# Paths passed from Bash
master_log_path = "${MASTER_LOG}"
incremental_log_path = "${INCREMENTAL_LOG}"
output_dir = "${SUMMARY_DIR}"
top_k = 400

try:
    # 1. Load Master Log
    if os.path.exists(master_log_path):
        df_master = pd.read_csv(master_log_path)
    else:
        df_master = pd.DataFrame(columns=['Model_Arch', 'Type', 'Source_Task', 'Target_Task', 'Count'])

    # 2. Load Incremental Log
    if os.path.exists(incremental_log_path):
        df_new = pd.read_csv(incremental_log_path)
    else:
        df_new = pd.DataFrame()

    if not df_new.empty:
        print(f"Merging {len(df_new)} new records into Master Log...")
        
        # 3. Concatenate
        df_combined = pd.concat([df_master, df_new], ignore_index=True)
        
        # 4. Remove Duplicates (keep the last entry / newest run)
        # We identify a unique record by Arch, Type, Source, and Target
        df_combined = df_combined.drop_duplicates(
            subset=['Model_Arch', 'Type', 'Source_Task', 'Target_Task'], 
            keep='last'
        )
        
        # 5. Save back to Master Log
        df_combined.to_csv(master_log_path, index=False)
        print(f"Master Log updated. Total records: {len(df_combined)}")
        
        # 6. Generate Summary Tables
        df_t1 = df_combined[df_combined['Type'] == 'T1_Cross_TargetFT']
        df_t2 = df_combined[df_combined['Type'] == 'T2_SourceFT_TargetPT']

        def save_pivot(dataframe, type_name):
            if dataframe.empty:
                return
            
            # Pivot: Rows=(Arch, Source), Cols=Target
            pivot_count = dataframe.pivot_table(
                index=['Model_Arch', 'Source_Task'], 
                columns='Target_Task', 
                values='Count'
            )
            
            # Save Count Matrix
            count_file = os.path.join(output_dir, f"Summary_{type_name}_Overlap_Counts.csv")
            pivot_count.to_csv(count_file)
            
            # Save Percentage Matrix
            pivot_pct = (pivot_count / top_k) * 100
            pct_file = os.path.join(output_dir, f"Summary_{type_name}_Overlap_Percentages.csv")
            pivot_pct.to_csv(pct_file)
            print(f"Tables saved for {type_name}")

        save_pivot(df_t1, "Task1_Cross_vs_TargetFT")
        save_pivot(df_t2, "Task2_SourceFT_vs_TargetPT")

    else:
        print("No new data found in incremental log.")

except Exception as e:
    print(f"Error in Python merge script: {e}")
EOF

# Clean up temporary file
rm "$INCREMENTAL_LOG"

echo "=================================================="
echo "All Done. Results saved to $MASTER_LOG"
echo "=================================================="