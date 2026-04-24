#!/bin/bash
# Run GPT-2 (new full-FT version) EAP and save ALL edges (no top-K threshold).
# Covers: 6 pretrained + 6 own-task FT + 30 cross-task FT = 42 runs total.
#
# Usage:  bash src/EAP/run_gpt2_all_edges.sh
# Outputs under: output/EAP_edges/gpt2_all_edges/
set -uo pipefail

PROJECT_ROOT="/users/sglli24/UnderstandingFineTuningViaMI"
MODEL_DIR="/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model"
DATA_DIR="${PROJECT_ROOT}/output/corrupted_data"
SCRIPT="${PROJECT_ROOT}/src/EAP/eap_unified.py"
OUT_DIR="${PROJECT_ROOT}/output/EAP_edges/gpt2_all_edges"
mkdir -p "$OUT_DIR"

# GPT-2 full-FT checkpoints (new version)
declare -A GPT2_FT
GPT2_FT[yelp]="${MODEL_DIR}/gpt2-small-yelp-full-ft-20260415-232443"
GPT2_FT[sst2]="${MODEL_DIR}/gpt2-sst2-full-ft-20251205-172809"
GPT2_FT[squad]="${MODEL_DIR}/gpt2-small-squad-full-ft-20260105-230037"
GPT2_FT[coqa]="${MODEL_DIR}/gpt2-small-COQA-full-ft-20260105-230716"
GPT2_FT[kde4]="${MODEL_DIR}/gpt2-small-kde4-full-ft-20260106-204426/gpt2-small-kde4-full-ft-20260106-204426"
GPT2_FT[tatoeba]="${MODEL_DIR}/gpt2-small-tatoeba-full-ft-20260106-224923"

TASKS=(yelp sst2 squad coqa kde4 tatoeba)

run_one() {
    local label="$1"; shift
    echo
    echo "===================================================="
    echo "[$(date '+%H:%M:%S')] $label"
    echo "----------------------------------------------------"
    "$@"
    local rc=$?
    echo "[$(date '+%H:%M:%S')] done rc=$rc"
    return $rc
}

# ---------- 1) Pretrained base model, 6 corrupted tasks ----------
for t in "${TASKS[@]}"; do
    out="${OUT_DIR}/gpt2_${t}_pretrained_edges.csv"
    if [ -f "$out" ]; then
        echo "[skip] exists: $out"; continue
    fi
    run_one "pretrained | data=$t" \
        python "$SCRIPT" \
            --task "$t" \
            --model_name gpt2 \
            --mode pretrained \
            --base_model_name gpt2 \
            --data_path "${DATA_DIR}/${t}_corrupted.csv" \
            --top_k -1 \
            --batch_size 1 \
            --output_dir "$OUT_DIR"
    # pretrained mode writes to $OUT_DIR/pretrained/..., flatten into $OUT_DIR
    src="${OUT_DIR}/pretrained/gpt2_${t}_pretrained_edges.csv"
    [ -f "$src" ] && mv "$src" "$out"
done
rmdir "${OUT_DIR}/pretrained" 2>/dev/null || true

# ---------- 2) Fine-tuned cross-task (ft_task x test_task) ----------
# When ft_task == test_task, save as own-task name; else cross-task name.
TMP_DIR="${OUT_DIR}/_tmp"
mkdir -p "$TMP_DIR"

for ft_task in "${TASKS[@]}"; do
    ft_path="${GPT2_FT[$ft_task]}"
    if [ ! -d "$ft_path" ]; then
        echo "[Error] FT dir missing: $ft_path"; continue
    fi
    for test_task in "${TASKS[@]}"; do
        if [ "$ft_task" = "$test_task" ]; then
            out="${OUT_DIR}/gpt2_${test_task}_finetuned_edges.csv"
        else
            out="${OUT_DIR}/gpt2_Finetuned-${ft_task}_Corrupted-Data_${test_task}_finetuned_edges.csv"
        fi
        if [ -f "$out" ]; then
            echo "[skip] exists: $out"; continue
        fi
        run_one "FT=$ft_task | data=$test_task" \
            python "$SCRIPT" \
                --task "$test_task" \
                --model_name gpt2 \
                --mode finetuned \
                --base_model_name gpt2 \
                --ft_model_path "$ft_path" \
                --data_path "${DATA_DIR}/${test_task}_corrupted.csv" \
                --top_k -1 \
                --batch_size 1 \
                --output_dir "$TMP_DIR"
        src="${TMP_DIR}/gpt2_${test_task}_finetuned_edges.csv"
        if [ -f "$src" ]; then
            mv "$src" "$out"
        else
            echo "[Error] expected output missing: $src"
        fi
    done
done
rmdir "$TMP_DIR" 2>/dev/null || true

echo
echo "===================================================="
echo "All done. Files under: $OUT_DIR"
ls "$OUT_DIR" | wc -l
