#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o overlap_analysis_%j.out
#SBATCH --gres=gpu:1 
#SBATCH -N 1
#SBATCH -t 02:00:00

# 加载环境
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

# ================= 模式选择 =================
# 获取第一个命令行参数，如果没有提供，默认为 "all"
# 可选值: "task1", "task2", "all"
RUN_MODE=${1:-all}

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

# ================= 配置路径 =================
SCRIPT_PATH="/users/sglli24/UnderstandingFineTuningViaMI/src/EAP/eap_unified.py"
CROSS_TASK_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/cross_task_edges"
STD_FT_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/old-version-finetuned"
PRETRAINED_DIR="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges/pretrained"

# 1. 保存详细 Overlap 边文件的目录
OUT_OVERLAP_T1="${CROSS_TASK_DIR}/overlap_cross_vs_target_ft"
OUT_OVERLAP_T2="${CROSS_TASK_DIR}/overlap_source_ft_vs_target_pt"

# 2. 保存汇总表的目录
SUMMARY_DIR="${CROSS_TASK_DIR}/summary_tables"

mkdir -p "$OUT_OVERLAP_T1"
mkdir -p "$OUT_OVERLAP_T2"
mkdir -p "$SUMMARY_DIR"

# 创建临时日志 (注意：每次运行都会覆盖旧日志，保证数据纯净)
TEMP_LOG="${SUMMARY_DIR}/raw_data_log.csv"
echo "Model_Arch,Type,Source_Task,Target_Task,Count" > "$TEMP_LOG"

echo "Starting Overlap Analysis..."

# ========================================================
# 遍历 Cross-Task 文件
# ========================================================

for cross_file in "$CROSS_TASK_DIR"/*_finetuned_edges.csv; do
    [ -e "$cross_file" ] || continue
    
    filename=$(basename "$cross_file")
    
    # --- 解析文件名 ---
    IFS='_' read -r -a parts <<< "${filename%.csv}"
    
    model_arch="${parts[0]}"      # gpt2
    raw_source="${parts[1]}"      # Finetuned-Yelp
    source_task="${raw_source#Finetuned-}" # Yelp
    target_task="${parts[3]}"     # sst2

    echo ">> Pair: Source($source_task) - Target($target_task)"

    # ====================================================
    # Task 1: Cross-Result vs Target-FT
    # 控制开关：只有模式为 'all' 或 'task1' 时运行
    # ====================================================
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

    # ====================================================
    # Task 2: Source-FT vs Target-PT
    # 控制开关：只有模式为 'all' 或 'task2' 时运行
    # ====================================================
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

echo "--------------------------------------------------"
echo "Generating Summary Tables..."

# ========================================================
# Python 生成矩阵表
# ========================================================

python3 - <<EOF
import pandas as pd
import os

log_file = "${TEMP_LOG}"
out_dir = "${SUMMARY_DIR}"
top_k = 400

if os.path.exists(log_file):
    df = pd.read_csv(log_file)
    if not df.empty:
        # 分离两种比较类型
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

        # 无论模式如何，只要 DataFrame 有数据就会保存，没有数据就会跳过
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