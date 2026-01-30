#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o overlap_analysis_llama_patch_%j.out
#SBATCH --gres=gpu:1 
#SBATCH -p gpu-a100-cs,gpu-a-lowsmall,gpu-a100-lowbig,gpu-v100,gpu-l40s
#SBATCH -t 1-00:00:00

# Load environment
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

# Mode selection
RUN_MODE=${1:-all}

# Paths
SCRIPT_PATH="/users/sglli24/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py"
CROSS_TASK_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/cross_task_edges"
STD_FT_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/old-version-finetuned"
PRETRAINED_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/pretrained"

# Output directories
OUT_OVERLAP_T1="${CROSS_TASK_DIR}/overlap_cross_vs_target_ft"
OUT_OVERLAP_T2="${CROSS_TASK_DIR}/overlap_source_ft_vs_target_pt"
SUMMARY_DIR="${CROSS_TASK_DIR}/task1_summary_tables"

mkdir -p "$OUT_OVERLAP_T1"
mkdir -p "$OUT_OVERLAP_T2"
mkdir -p "$SUMMARY_DIR"

# Log file setup
TEMP_LOG="${SUMMARY_DIR}/v2_raw_data_log.csv"
NEW_RESULTS_LOG="${SUMMARY_DIR}/incremental_llama_qwen_patch.csv"
echo "Model_Arch,Type,Source_Task,Target_Task,Count" > "$NEW_RESULTS_LOG"

echo "Starting Targeted Overlap Analysis (Llama2-SST2 & Diagonals)..."

# Process cross-task files (targeted: llama2-sst2)

for cross_file in "$CROSS_TASK_DIR"/*_finetuned_edges.csv; do
    [ -e "$cross_file" ] || continue
    
    filename=$(basename "$cross_file")
    IFS='_' read -r -a parts <<< "${filename%.csv}"
    
    model_arch="${parts[0]}"      # e.g., llama2
    raw_source="${parts[1]}"      # e.g., Finetuned-sst2
    source_task="${raw_source#Finetuned-}"
    target_task="${parts[3]}"     # e.g., yelp

    # Filter: only process llama2 with sst2 as source task
    if [[ "$model_arch" == "llama2" && ("$source_task" == "sst2" || "$source_task" == "SST2") ]]; then
        echo ">> Processing Targeted Pair: $model_arch | Source($source_task) -> Target($target_task)"
        
        # Task 1: Cross-Result vs Target-FT
        if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "task1" ]]; then
            std_ft_file="${STD_FT_DIR}/${model_arch}_${target_task}_finetuned_edges.csv"
            if [ -f "$std_ft_file" ]; then
                output=$(python "$SCRIPT_PATH" --mode compare --output_dir "$OUT_OVERLAP_T1" --edge_file_1 "$cross_file" --edge_file_2 "$std_ft_file")
                count=$(echo "$output" | grep "Intersection (Common):" | awk -F': ' '{print $2}')
                if [[ -n "$count" ]]; then
                    echo "${model_arch},T1_Cross_TargetFT,${source_task},${target_task},${count}" >> "$NEW_RESULTS_LOG"
                fi
            fi
        fi

        # Task 2: Source-FT vs Target-PT
        if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "task2" ]]; then
            source_ft_file="${STD_FT_DIR}/${model_arch}_${source_task}_finetuned_edges.csv"
            target_pt_file="${PRETRAINED_DIR}/${model_arch}_${target_task}_pretrained_edges.csv"
            if [ -f "$source_ft_file" ] && [ -f "$target_pt_file" ]; then
                output=$(python "$SCRIPT_PATH" --mode compare --output_dir "$OUT_OVERLAP_T2" --edge_file_1 "$source_ft_file" --edge_file_2 "$target_pt_file")
                count=$(echo "$output" | grep "Intersection (Common):" | awk -F': ' '{print $2}')
                if [[ -n "$count" ]]; then
                    echo "${model_arch},T2_SourceFT_TargetPT,${source_task},${target_task},${count}" >> "$NEW_RESULTS_LOG"
                fi
            fi
        fi
    fi
done

# Diagonal analysis (llama2, qwen2, llama3.2)

echo "--------------------------------------------------"
echo "Running Diagonal Analysis for Llama2, Qwen2, Llama3.2..."

ALL_TASKS=("coqa" "kde4" "squad" "sst2" "tatoeba" "yelp")
TARGET_MODELS=("llama2" "qwen2" "llama3.2")

for arch in "${TARGET_MODELS[@]}"; do
    for task in "${ALL_TASKS[@]}"; do
        source_ft_file="${STD_FT_DIR}/${arch}_${task}_finetuned_edges.csv"
        target_pt_file="${PRETRAINED_DIR}/${arch}_${task}_pretrained_edges.csv"

        if [ -f "$source_ft_file" ] && [ -f "$target_pt_file" ]; then
            output=$(python "$SCRIPT_PATH" --mode compare --output_dir "$OUT_OVERLAP_T2" --edge_file_1 "$source_ft_file" --edge_file_2 "$target_pt_file")
            count=$(echo "$output" | grep "Intersection (Common):" | awk -F': ' '{print $2}')
            
            if [[ -n "$count" ]]; then
                echo "${arch},T2_SourceFT_TargetPT,${task},${task},${count}" >> "$NEW_RESULTS_LOG"
                echo "   [Diagonal] $arch: FT-${task} vs PT-${task} -> $count"
            fi
        fi
    done
done

# Merge results and backup

echo "--------------------------------------------------"
echo "Merging results..."

if [ -f "$TEMP_LOG" ]; then
    echo "Appending new results to $TEMP_LOG..."
    # Exclude header line, append to master log
    tail -n +2 "$NEW_RESULTS_LOG" >> "$TEMP_LOG"
    echo "Success: Data merged into $TEMP_LOG."
else
    echo "Original log $TEMP_LOG not found. New results saved in $NEW_RESULTS_LOG only."
fi

# Generate summary tables

python3 - <<EOF
import pandas as pd
import os

log_file = "${TEMP_LOG}"
if not os.path.exists(log_file):
    log_file = "${NEW_RESULTS_LOG}" # Fall back to new results log if merge failed

out_dir = "${SUMMARY_DIR}"
top_k = 400

if os.path.exists(log_file):
    df = pd.read_csv(log_file)
    if not df.empty:
        # Remove duplicates (keep latest)
        df = df.drop_duplicates(subset=['Model_Arch', 'Type', 'Source_Task', 'Target_Task'], keep='last')
        
        df_t1 = df[df['Type'] == 'T1_Cross_TargetFT']
        df_t2 = df[df['Type'] == 'T2_SourceFT_TargetPT']

        def save_tables(dataframe, type_name):
            if dataframe.empty: return
            pivot_count = dataframe.pivot_table(index=['Model_Arch', 'Source_Task'], columns='Target_Task', values='Count')
            pivot_count.to_csv(os.path.join(out_dir, f"Summary_{type_name}_Overlap_Counts.csv"))
            pivot_pct = (pivot_count / top_k) * 100
            pivot_pct.to_csv(os.path.join(out_dir, f"Summary_{type_name}_Overlap_Percentages.csv"))

        save_tables(df_t1, "Task1_Cross_vs_TargetFT")
        save_tables(df_t2, "Task2_SourceFT_vs_TargetPT")
        print("Summary tables updated.")
EOF

echo "Process Complete. Incremental results: $NEW_RESULTS_LOG"