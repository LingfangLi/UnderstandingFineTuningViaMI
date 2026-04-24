#!/bin/bash
# Run Qwen2-0.5B (Pipeline B full-FT) EAP and save ALL edges (top_k=-1).
# Covers: 6 pretrained + 6 own-task FT + 30 cross-task FT = 42 runs.
# Output: output/EAP_edges/qwen2_all_edges/
set -uo pipefail

PROJECT_ROOT="/users/sglli24/UnderstandingFineTuningViaMI"
MODEL_DIR="/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model"
DATA_DIR="${PROJECT_ROOT}/output/corrupted_data"
SCRIPT="${PROJECT_ROOT}/src/EAP/eap_unified.py"
OUT_DIR="${PROJECT_ROOT}/output/EAP_edges/qwen2_all_edges"
mkdir -p "$OUT_DIR"

declare -A QWEN2_FT
QWEN2_FT[yelp]="${MODEL_DIR}/qwen2-0.5b-yelp-full-ft-20251124-204027"
QWEN2_FT[sst2]="${MODEL_DIR}/qwen2-0.5b-sst2-full-20251209-1054"
QWEN2_FT[squad]="${MODEL_DIR}/qwen2-0.5b-squad-full-20251125-165024"
QWEN2_FT[coqa]="${MODEL_DIR}/qwen2-0.5b-coqa-full-20251125-182058"
QWEN2_FT[kde4]="${MODEL_DIR}/qwen2-kde4-tech-trans-full-20251125-165755"
QWEN2_FT[tatoeba]="${MODEL_DIR}/qwen2-0.5b-tatoeba-en-fr-20251125-165129"

TASKS=(yelp sst2 squad coqa kde4 tatoeba)
BASE_MODEL="Qwen/Qwen2-0.5B"
MODEL_NAME="qwen2"

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

# 1) Pretrained
for t in "${TASKS[@]}"; do
    out="${OUT_DIR}/${MODEL_NAME}_${t}_pretrained_edges.csv"
    [ -f "$out" ] && { echo "[skip] $(basename $out)"; continue; }
    run_one "pretrained | data=$t" \
        python "$SCRIPT" \
            --task "$t" --model_name "$MODEL_NAME" --mode pretrained \
            --base_model_name "$BASE_MODEL" \
            --data_path "${DATA_DIR}/${t}_corrupted.csv" \
            --top_k -1 --batch_size 1 --output_dir "$OUT_DIR"
    src="${OUT_DIR}/pretrained/${MODEL_NAME}_${t}_pretrained_edges.csv"
    [ -f "$src" ] && mv "$src" "$out"
done
rmdir "${OUT_DIR}/pretrained" 2>/dev/null || true

# 2) FT cross-task + own-task
TMP_DIR="${OUT_DIR}/_tmp"; mkdir -p "$TMP_DIR"
for ft_task in "${TASKS[@]}"; do
    ft_path="${QWEN2_FT[$ft_task]}"
    [ ! -d "$ft_path" ] && { echo "[ERROR] missing: $ft_path"; continue; }
    for test_task in "${TASKS[@]}"; do
        if [ "$ft_task" = "$test_task" ]; then
            out="${OUT_DIR}/${MODEL_NAME}_${test_task}_finetuned_edges.csv"
        else
            out="${OUT_DIR}/${MODEL_NAME}_Finetuned-${ft_task}_Corrupted-Data_${test_task}_finetuned_edges.csv"
        fi
        [ -f "$out" ] && { echo "[skip] $(basename $out)"; continue; }
        run_one "FT=$ft_task | data=$test_task" \
            python "$SCRIPT" \
                --task "$test_task" --model_name "$MODEL_NAME" --mode finetuned \
                --base_model_name "$BASE_MODEL" --ft_model_path "$ft_path" \
                --data_path "${DATA_DIR}/${test_task}_corrupted.csv" \
                --top_k -1 --batch_size 1 --output_dir "$TMP_DIR"
        src="${TMP_DIR}/${MODEL_NAME}_${test_task}_finetuned_edges.csv"
        [ -f "$src" ] && mv "$src" "$out" || echo "[ERROR] missing: $src"
    done
done
rmdir "$TMP_DIR" 2>/dev/null || true

echo "===================================================="
echo "All done. Files: $(ls $OUT_DIR | wc -l)"
