#!/bin/bash

# Paths
SCRIPT_PATH="<PROJECT_ROOT>/src/EAP/eap_unified.py"
CROSS_TASK_DIR="<PROJECT_ROOT>/output/EAP_edges/cross_task_edges"
STD_FT_DIR="<PROJECT_ROOT>/output/EAP_edges/old-version-finetuned"
PRETRAINED_DIR="<PROJECT_ROOT>/output/EAP_edges/pretrained"

# Output directories
OUT_OVERLAP_T2="${CROSS_TASK_DIR}/overlap_source_ft_vs_target_pt"
SUMMARY_DIR="${CROSS_TASK_DIR}/summary_tables"
# Append to existing master log
TEMP_LOG="${SUMMARY_DIR}/raw_data_log.csv"

# Verify master log exists
if [ ! -f "$TEMP_LOG" ]; then
    echo "Error: master log not found at $TEMP_LOG"
    echo "Run the main overlap script first to generate initial data."
    exit 1
fi

echo "=================================================="
echo "Running ONLY Diagonal Analysis (Appending to existing log)"
echo "=================================================="

# Task and architecture lists
ALL_TASKS=("coqa" "kde4" "squad" "sst2" "tatoeba" "yelp")

ALL_ARCHS=("gpt2" "llama3.2" "qwen2" "llama2")

# Diagonal overlap computation
for CURRENT_ARCH in "${ALL_ARCHS[@]}"; do
    echo "--------------------------------------------------"
    echo "Current Model: ${CURRENT_ARCH}"
    echo "--------------------------------------------------"
    
    for task in "${ALL_TASKS[@]}"; do
        source_ft_file="${STD_FT_DIR}/${CURRENT_ARCH}_${task}_finetuned_edges.csv"
        target_pt_file="${PRETRAINED_DIR}/${CURRENT_ARCH}_${task}_pretrained_edges.csv"
    
        echo ">> Checking Diagonal: ${task}..."
    
        if [ -f "$source_ft_file" ] && [ -f "$target_pt_file" ]; then
            output=$(python "$SCRIPT_PATH" \
                --mode compare \
                --output_dir "$OUT_OVERLAP_T2" \
                --edge_file_1 "$source_ft_file" \
                --edge_file_2 "$target_pt_file")
            
            count=$(echo "$output" | grep "Intersection (Common):" | awk -F': ' '{print $2}')
            
            if [[ -n "$count" ]]; then
                # Append to master log
                echo "${CURRENT_ARCH},T2_SourceFT_TargetPT,${task},${task},${count}" >> "$TEMP_LOG"
                echo "   Result: $count"
            else
                echo "   Error extracting count."
            fi
        else
            echo "   Missing files for ${task}"
        fi
    done
done

echo "--------------------------------------------------"
echo "Regenerating Summary Tables (Merging old and new data)..."

# Regenerate summary tables
python3 - <<EOF
# -*- coding: utf-8 -*-
import pandas as pd
import os

log_file = "${TEMP_LOG}"
out_dir = "${SUMMARY_DIR}"
top_k = 400

if os.path.exists(log_file):
    # Handle encoding issues with replace strategy
    try:
        df = pd.read_csv(log_file, encoding='utf-8', encoding_errors='replace')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(log_file, encoding='latin1')
        except:
            print("can't load table")
            df = pd.DataFrame()

    if not df.empty:
        df_t2 = df[df['Type'] == 'T2_SourceFT_TargetPT']
        
        if not df_t2.empty:
            # 1. Count Matrix
            pivot_count = df_t2.pivot_table(
                index=['Model_Arch', 'Source_Task'], 
                columns='Target_Task', 
                values='Count'
            )
            count_path = os.path.join(out_dir, "Summary_Task2_SourceFT_vs_TargetPT_Overlap_Counts.csv")
            pivot_count.to_csv(count_path)
            
            # 2. Percent Matrix
            pivot_pct = (pivot_count / top_k) * 100
            pct_path = os.path.join(out_dir, "Summary_Task2_SourceFT_vs_TargetPT_Overlap_Percentages.csv")
            pivot_pct.to_csv(pct_path)
            
            print("Tables updated with diagonal data.")
            print(f"Output: {pct_path}")
        else:
            print("No Task 2 data in log.")
    else:
        print("Log file empty or unreadable.")
else:
    print("Log file not found.")
EOF