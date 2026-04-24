#!/bin/bash
# Run GPT-2 cross-task EAP using Pipeline B-style HF checkpoints at
# /mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models/gpt2-{task}/.
# This matches the ckpt source that `finetuned/gpt2_*.csv` on GitHub came from
# (SFTTrainer-saved HF dirs), providing the missing Pipeline B cross-task edges.
#
# Saves top-400 (standard) to match existing `cross_task_edges/` format.
# Output: output/EAP_edges/gpt2_cross_task_hfdir/
set -uo pipefail

PROJECT_ROOT="/users/sglli24/UnderstandingFineTuningViaMI"
MODEL_DIR="/mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models"
DATA_DIR="${PROJECT_ROOT}/output/corrupted_data"
SCRIPT="${PROJECT_ROOT}/src/EAP/eap_unified.py"
OUT_DIR="${PROJECT_ROOT}/output/EAP_edges/gpt2_cross_task_hfdir"
mkdir -p "$OUT_DIR"

declare -A GPT2_FT
GPT2_FT[yelp]="${MODEL_DIR}/gpt2-yelp"
GPT2_FT[sst2]="${MODEL_DIR}/gpt2-sst2"
GPT2_FT[squad]="${MODEL_DIR}/gpt2-squad"
GPT2_FT[coqa]="${MODEL_DIR}/gpt2-coqa"
GPT2_FT[kde4]="${MODEL_DIR}/gpt2-kde4"
GPT2_FT[tatoeba]="${MODEL_DIR}/gpt2-tatoeba"

TASKS=(yelp sst2 squad coqa kde4 tatoeba)
BASE_MODEL="gpt2"
MODEL_NAME="gpt2"
TOP_K=400  # match existing cross_task_edges/ convention

run_one() {
    local label="$1"; shift
    echo
    echo "===================================================="
    echo "[$(date '+%H:%M:%S')] $label"
    "$@"
    echo "[$(date '+%H:%M:%S')] done rc=$?"
}

# 1) Pretrained base (6 tasks)
for t in "${TASKS[@]}"; do
    out="${OUT_DIR}/${MODEL_NAME}_${t}_pretrained_edges.csv"
    [ -f "$out" ] && { echo "[skip] $(basename $out)"; continue; }
    run_one "pretrained | data=$t" \
        python "$SCRIPT" \
            --task "$t" --model_name "$MODEL_NAME" --mode pretrained \
            --base_model_name "$BASE_MODEL" \
            --data_path "${DATA_DIR}/${t}_corrupted.csv" \
            --top_k $TOP_K --batch_size 1 --output_dir "$OUT_DIR"
    src="${OUT_DIR}/pretrained/${MODEL_NAME}_${t}_pretrained_edges.csv"
    [ -f "$src" ] && mv "$src" "$out"
done
rmdir "${OUT_DIR}/pretrained" 2>/dev/null || true

# 2) FT cross-task + own-task (36 combos: 6 own + 30 cross)
TMP_DIR="${OUT_DIR}/_tmp"; mkdir -p "$TMP_DIR"
for ft_task in "${TASKS[@]}"; do
    ft_path="${GPT2_FT[$ft_task]}"
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
                --top_k $TOP_K --batch_size 1 --output_dir "$TMP_DIR"
        src="${TMP_DIR}/${MODEL_NAME}_${test_task}_finetuned_edges.csv"
        [ -f "$src" ] && mv "$src" "$out" || echo "[ERROR] expected output missing: $src"
    done
done
rmdir "$TMP_DIR" 2>/dev/null || true

echo
echo "===================================================="
echo "All done. Files: $(ls $OUT_DIR | wc -l) in $OUT_DIR"
