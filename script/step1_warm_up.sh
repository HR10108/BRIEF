#!/usr/bin/env bash
CONFIG_FILE="${1:-./BRIEF_config.sh}"
source "$CONFIG_FILE"

# Override NUM_TRAIN_EPOCHS for warm-up step
export NUM_TRAIN_EPOCHS=1

echo "Beginning warm-up training for job: ${JOB_NAME_WARM}"
LOG_FILE=$(log_file "$JOB_NAME_WARM")

bash "$WARM_SCRIPT_PATH" \
  "$DATA_FILE" \
  "$MODEL_TRAIN_PATH" \
  "$JOB_NAME_WARM" \
  "$RESULTS_FILE" \
  "$WARM_OUTPUT_PATH" \
  "$SEED" \
  "$MAX_LENGTH" \
  "$SAMPLE_RATIO" \
  > "$LOG_FILE"
