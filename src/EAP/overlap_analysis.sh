#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o overlap_analysis_pre-ft%j.out
#SBATCH --gres=gpu:1 
#SBATCH -p gpu-a100-cs,gpu-v100,gpu-l40s
#SBATCH -t 1-00:00:00

# Load environment
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
pip install tabulate
# Mode selection
# Arg 1: run mode (task1, task2, all)
RUN_MODE=${1:-task2}

echo "=================================================="
echo "Current Run Mode: $RUN_MODE"
if [[ "$RUN_MODE" == "task1" ]]; then
    echo "  -> Only running Task 1: Cross vs Target FT"
elif [[ "$RUN_MODE" == "task2" ]]; then
    echo "  -> Only running Task 2: Source FT vs Target PT"
else
    echo "  -> Running BOTH Tasks"
fi
echo "=================================================="

# Paths
SCRIPT_PATH="/users/sglli24/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py"
CROSS_TASK_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/cross_task_edges"
STD_FT_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/old-version-finetuned"
PRETRAINED_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/pretrained"

# Output directories for detailed overlap CSVs
OUT_OVERLAP_T1="${CROSS_TASK_DIR}/overlap_cross_vs_target_ft"
OUT_OVERLAP_T2="${CROSS_TASK_DIR}/overlap_source_ft_vs_target_pt"

# Summary tables directory
SUMMARY_DIR="${CROSS_TASK_DIR}/summary_tables"

mkdir -p "$OUT_OVERLAP_T1"
mkdir -p "$OUT_OVERLAP_T2"
mkdir -p "$SUMMARY_DIR"

# Temporary log for this run (overwritten each time)
TEMP_LOG="${SUMMARY_DIR}/${RUN_MODE}_raw_data_log.csv"
echo "Model_Arch,Type,Source_Task,Target_Task,Count" > "$TEMP_LOG"

echo "Starting Overlap Analysis..."

# Process cross-task files

for cross_file in "$CROSS_TASK_DIR"/*_finetuned_edges.csv; do
    [ -e "$cross_file" ] || continue
    
    filename=$(basename "$cross_file")
    
    # Parse filename
    IFS='_' read -r -a parts <<< "${filename%.csv}"
    
    model_arch="${parts[0]}"      # gpt2
    raw_source="${parts[1]}"      # Finetuned-Yelp
    source_task="${raw_source#Finetuned-}" # Yelp
    target_task="${parts[3]}"     # sst2

    echo ">> Pair: Source($source_task) - Target($target_task)"

    # Task 1: Cross-result vs target fine-tuned
    if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "task1" ]]; then
        std_ft_file="${STD_FT_DIR}/${model_arch}_${target_task}_finetuned_edges.csv"

        if [ -f "$std_ft_file" ]; then
            output=$(python "$SCRIPT_PATH" \
                --mode compare \
                --output_dir "$OUT_OVERLAP_T1" \
                --edge_file_1 "$cross_file" \
                --edge_file_2 "$std_ft_file")
            
            count=$(echo "$output" | grep "Intersection (Common):" | awk -F': ' '{print $2}')
            
            if [[ -n "$count" ]]; then
                echo "${model_arch},T1_Cross_TargetFT,${source_task},${target_task},${count}" >> "$TEMP_LOG"
            else
                echo "   [T1] Error extracting count."
            fi
        else
            echo "   [T1] Missing Target FT file: $(basename "$std_ft_file")"
        fi
    fi

    # Task 2: Source fine-tuned vs target pretrained
    if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "task2" ]]; then
        source_ft_file="${STD_FT_DIR}/${model_arch}_${source_task}_finetuned_edges.csv"
        target_pt_file="${PRETRAINED_DIR}/${model_arch}_${target_task}_pretrained_edges.csv"

        if [ -f "$source_ft_file" ] && [ -f "$target_pt_file" ]; then
            output=$(python "$SCRIPT_PATH" \
                --mode compare \
                --output_dir "$OUT_OVERLAP_T2" \
                --edge_file_1 "$source_ft_file" \
                --edge_file_2 "$target_pt_file")
                
            count=$(echo "$output" | grep "Intersection (Common):" | awk -F': ' '{print $2}')
            
            if [[ -n "$count" ]]; then
                echo "${model_arch},T2_SourceFT_TargetPT,${source_task},${target_task},${count}" >> "$TEMP_LOG"
                echo "   [T2] Checked: FT-${source_task} vs PT-${target_task} -> $count"
            else
                echo "   [T2] Error extracting count."
            fi
        else
            echo "   [T2] Missing files (Source FT or Target PT)."
        fi
    fi

done

# Diagonal analysis (self-overlap)

echo "--------------------------------------------------"
echo "Running Diagonal Analysis (Self-Correction)..."

# All tasks and architectures
ALL_TASKS=("coqa" "kde4" "squad" "sst2" "tatoeba" "yelp")
CURRENT_ARCH="gpt2" 

for task in "${ALL_TASKS[@]}"; do
    # Compute FT vs PT overlap for same task
    source_ft_file="${STD_FT_DIR}/${CURRENT_ARCH}_${task}_finetuned_edges.csv"
    target_pt_file="${PRETRAINED_DIR}/${CURRENT_ARCH}_${task}_pretrained_edges.csv"

    if [ -f "$source_ft_file" ] && [ -f "$target_pt_file" ]; then
        output=$(python "$SCRIPT_PATH" \
            --mode compare \
            --output_dir "$OUT_OVERLAP_T2" \
            --edge_file_1 "$source_ft_file" \
            --edge_file_2 "$target_pt_file")
        
        count=$(echo "$output" | grep "Intersection (Common):" | awk -F': ' '{print $2}')
        
        if [[ -n "$count" ]]; then
            # Write to log (same format as main loop)
            echo "${CURRENT_ARCH},T2_SourceFT_TargetPT,${task},${task},${count}" >> "$TEMP_LOG"
            echo "   [Diagonal] Checked: FT-${task} vs PT-${task} -> $count"
        fi
    else
        echo "   [Diagonal] Missing files for ${task} (Source FT or Target PT)"
    fi
done

echo "--------------------------------------------------"
echo "Generating Summary Tables..."

# Generate summary tables

python3 - <<EOF
# -*- coding: utf-8 -*-
import pandas as pd
import os

log_file = "${TEMP_LOG}"
out_dir = "${SUMMARY_DIR}"
top_k = 400

if os.path.exists(log_file):
    df = pd.read_csv(log_file)
    if not df.empty:
        df_t1 = df[df['Type'] == 'T1_Cross_TargetFT']
        df_t2 = df[df['Type'] == 'T2_SourceFT_TargetPT']

        def save_tables(dataframe, type_name):
            if dataframe.empty: 
                print(f"Skipping {type_name} (No data found in this run).")
                return
            
            # Count Matrix
            pivot_count = dataframe.pivot_table(
                index=['Model_Arch', 'Source_Task'], 
                columns='Target_Task', 
                values='Count'
            )
            count_path = os.path.join(out_dir, f"Summary_{type_name}_Overlap_Counts.csv")
            pivot_count.to_csv(count_path)
            
            # Percent Matrix
            pivot_pct = (pivot_count / top_k) * 100
            pct_path = os.path.join(out_dir, f"Summary_{type_name}_Overlap_Percentages.csv")
            pivot_pct.to_csv(pct_path)
            print(f"Saved tables for {type_name}")

        print("--- Check Task 1 Data ---")
        save_tables(df_t1, "Task1_Cross_vs_TargetFT")
        
        print("--- Check Task 2 Data ---")
        save_tables(df_t2, "Task2_SourceFT_vs_TargetPT")
    else:
        print("Log file is empty.")
else:
    print("Log file missing.")
EOF

echo "Process Complete."