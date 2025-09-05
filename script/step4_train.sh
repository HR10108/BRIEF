#!/usr/bin/env bash
CONFIG_FILE="${1:-./BRIEF_config.sh}"
source "$CONFIG_FILE"

# Override NUM_TRAIN_EPOCHS for final training step
export NUM_TRAIN_EPOCHS=4
LOG_FILE=$(log_file "$JOB_NAME_FINAL")
bash "$FINAL_SCRIPT_PATH" \
  "$DATA_FILE" \
  "$MODEL_TRAIN_PATH" \
  "$JOB_NAME_FINAL" \
  "$RESULTS_FILE" \
  "$OUTPUT_PATH_FINAL" \
  "$SEED" \
  "$MAX_LENGTH" \
  "$CORESET_FILE" \
  "1.0" \
  > "$LOG_FILE"