#!/bin/bash
# Qwen2-0.5B cross-task EAP v2:
# Source = /mnt/data1/.../fine_tuned_models/ (user's canonical "best" ckpts,
# matches GitHub `finetuned/` source). top_k=400 to match GitHub convention.
set -uo pipefail

PROJECT_ROOT="/users/sglli24/UnderstandingFineTuningViaMI"
MODEL_DIR="/mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models"
DATA_DIR="${PROJECT_ROOT}/output/corrupted_data"
SCRIPT="${PROJECT_ROOT}/src/EAP/eap_unified.py"
OUT_DIR="${PROJECT_ROOT}/output/EAP_edges/qwen2_cross_task_hfdir"
mkdir -p "$OUT_DIR"

declare -A FT
FT[yelp]="${MODEL_DIR}/qwen2-yelp"
FT[sst2]="${MODEL_DIR}/qwen2-sst2"
FT[squad]="${MODEL_DIR}/qwen2-squad"
FT[coqa]="${MODEL_DIR}/qwen2-coqa"
FT[kde4]="${MODEL_DIR}/qwen2-kde4"
FT[tatoeba]="${MODEL_DIR}/qwen2-tatoeba"

TASKS=(yelp sst2 squad coqa kde4 tatoeba)
BASE_MODEL="Qwen/Qwen2-0.5B"
MODEL_NAME="qwen2"
TOP_K=400

log() { echo "[$(date '+%H:%M:%S')] $*"; }

for t in "${TASKS[@]}"; do
    out="${OUT_DIR}/${MODEL_NAME}_${t}_pretrained_edges.csv"
    [ -f "$out" ] && { log "[skip] $(basename $out)"; continue; }
    log "===== pretrained | data=$t ====="
    python "$SCRIPT" \
        --task "$t" --model_name "$MODEL_NAME" --mode pretrained \
        --base_model_name "$BASE_MODEL" \
        --data_path "${DATA_DIR}/${t}_corrupted.csv" \
        --top_k $TOP_K --batch_size 1 --output_dir "$OUT_DIR"
    src="${OUT_DIR}/pretrained/${MODEL_NAME}_${t}_pretrained_edges.csv"
    [ -f "$src" ] && mv "$src" "$out"
done
rmdir "${OUT_DIR}/pretrained" 2>/dev/null || true

TMP_DIR="${OUT_DIR}/_tmp"; mkdir -p "$TMP_DIR"
for ft_task in "${TASKS[@]}"; do
    ft_path="${FT[$ft_task]}"
    [ ! -d "$ft_path" ] && { log "[ERROR] missing: $ft_path"; continue; }
    for test_task in "${TASKS[@]}"; do
        if [ "$ft_task" = "$test_task" ]; then
            out="${OUT_DIR}/${MODEL_NAME}_${test_task}_finetuned_edges.csv"
        else
            out="${OUT_DIR}/${MODEL_NAME}_Finetuned-${ft_task}_Corrupted-Data_${test_task}_finetuned_edges.csv"
        fi
        [ -f "$out" ] && { log "[skip] $(basename $out)"; continue; }
        log "===== FT=$ft_task | data=$test_task ====="
        python "$SCRIPT" \
            --task "$test_task" --model_name "$MODEL_NAME" --mode finetuned \
            --base_model_name "$BASE_MODEL" --ft_model_path "$ft_path" \
            --data_path "${DATA_DIR}/${test_task}_corrupted.csv" \
            --top_k $TOP_K --batch_size 1 --output_dir "$TMP_DIR"
        src="${TMP_DIR}/${MODEL_NAME}_${test_task}_finetuned_edges.csv"
        [ -f "$src" ] && mv "$src" "$out" || log "[ERROR] missing: $src"
    done
done
rmdir "$TMP_DIR" 2>/dev/null || true

log "ALL DONE. Files: $(ls $OUT_DIR | wc -l) in $OUT_DIR"
