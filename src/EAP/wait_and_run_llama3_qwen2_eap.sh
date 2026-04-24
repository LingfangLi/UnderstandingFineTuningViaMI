#!/bin/bash
# Queue: wait until Qwen2 lexical-complexity LC background job finishes
# (detected by absence of its runner python process), then run Llama-3.2
# all-edges EAP followed by Qwen2 all-edges EAP.
set -uo pipefail

PROJECT_ROOT="/users/sglli24/UnderstandingFineTuningViaMI"
LOG_DIR="${PROJECT_ROOT}/output/EAP_edges"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Wait for the LC aggregate to appear (the LAST step of run_qwen2_tc.sh prints
# "ALL DONE" and then exits). We detect completion by presence of
# results_full_qwen2/summary.tex (generated only by the aggregator).
RESULTS_SENTINEL="${PROJECT_ROOT}/experiments/text_complexity/fine_tune/results_full_qwen2/summary.tex"
JSON_DIR="${PROJECT_ROOT}/experiments/text_complexity/fine_tune/results_full_qwen2"

log "Waiting for Qwen2 LC pipeline to finish ..."
log "Sentinel file: $RESULTS_SENTINEL"

# Safety: also detect failure mode (no new file for > 30 min -> assume dead)
last_activity=$(date +%s)
prev_json_count=0
while [ ! -f "$RESULTS_SENTINEL" ]; do
    sleep 60
    cur=$(ls "$JSON_DIR"/*.json 2>/dev/null | wc -l)
    if [ "$cur" -ne "$prev_json_count" ]; then
        log "LC progress: $cur eval JSONs produced"
        prev_json_count=$cur
        last_activity=$(date +%s)
    fi
    # bail if no activity for 45 min AND no new JSON (could be stuck in train)
    # — skip this timeout if fewer than 6 JSONs (still training phase)
done

log "Sentinel appeared. LC pipeline is done."

# Now run EAP jobs sequentially
source /opt/apps/pkg/tools/miniforge3/25.3.0_python3.12.10/etc/profile.d/conda.sh 2>/dev/null || true
module load miniforge3/25.3.0-python3.12.10 2>/dev/null || true
source activate MI-FineTune

log "===== Starting Llama-3.2 all-edges EAP ====="
bash "${PROJECT_ROOT}/src/EAP/run_llama3_all_edges.sh" \
    2>&1 | tee -a "${LOG_DIR}/llama3_all_edges/_run.log" | grep -E "^\[|Error|====" | tail -50
log "Llama-3.2 done. Files: $(ls ${LOG_DIR}/llama3_all_edges 2>/dev/null | wc -l)"

log "===== Starting Qwen2 all-edges EAP ====="
bash "${PROJECT_ROOT}/src/EAP/run_qwen2_all_edges.sh" \
    2>&1 | tee -a "${LOG_DIR}/qwen2_all_edges/_run.log" | grep -E "^\[|Error|====" | tail -50
log "Qwen2 done. Files: $(ls ${LOG_DIR}/qwen2_all_edges 2>/dev/null | wc -l)"

log "ALL EAP BATCHES COMPLETE."
