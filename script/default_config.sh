#!/usr/bin/env bash

# ---------- Global environment ----------
# Determine GPUs: use GPUS env or default to all
if [ -n "$GPUS" ]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  # Count the number of GPUs specified
  IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
  export NUM_GPUS=${#GPU_ARRAY[@]}
else
  unset CUDA_VISIBLE_DEVICES
  export NUM_GPUS=$(nvidia-smi -L | wc -l)
fi
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES:-all} (${NUM_GPUS} GPUs)"
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

# change to the working directory(../)
cd "$(dirname "$0")/.."

# ---------- Common paths ----------
SEED=42
MAX_LENGTH=2048
BASE_DIR="."
DATA_FILE="${BASE_DIR}/data/tulu3_mix.jsonl" # path to the training data
MODEL_TRAIN_PATH="${BASE_DIR}/../cache/Llama3-8B-Base"
RESULTS_FILE="None"           
LOG_DIR="log"                 # all logs under ./log
mkdir -p "${LOG_DIR}"

# ---------- Warm‑up (step‑1) ----------
WARM_JOB="5-warmup-tulu3"
JOB_NAME_WARM="Llama3-8B-${WARM_JOB}"
SAMPLE_RATIO=0.05
mkdir -p "${BASE_DIR}/output/model"
WARM_OUTPUT_PATH="${BASE_DIR}/output/model/${JOB_NAME_WARM}"
WARM_SCRIPT_PATH="${BASE_DIR}/brief/scripts/lora_train_unified.sh"

# ---------- Gradient extraction (step‑2) ----------
GRADIENT_TYPE="adam"
INFO_TYPE="knif"
DIM=8192                     # embedding / projection dimension, reused later
MODEL_PATH_GRAD="${WARM_OUTPUT_PATH}"
# Search for the latest model checkpoint
LATEST_MODEL=$(ls -t "${MODEL_PATH_GRAD}" | grep -E 'checkpoint-[0-9]+$' | head -n 1)
if [ -z "$LATEST_MODEL" ]; then
  echo "No model checkpoint found in ${MODEL_PATH_GRAD}"
else
  echo "Latest model checkpoint found: ${LATEST_MODEL}"
fi
MODEL_PATH_GRAD="${MODEL_PATH_GRAD}/${LATEST_MODEL}"
OUTPUT_PATH_GRAD="${BASE_DIR}/output/grads/${JOB_NAME_WARM}/"
GRAD_SCRIPT_PATH="${BASE_DIR}/brief/scripts/get_train_lora_grads.sh"

# ---------- BRIEF coreset selection (step‑3) ----------
MODEL_BRIEF="${JOB_NAME_WARM}"
GRADIENT_PATH="${OUTPUT_PATH_GRAD}"
PROPORTIONS="0.05"  # subset size fractions (0~1)
DEVICE="cpu"        # 'cpu' or 'cuda' 
BATCH_SIZE=32
GRAD_TYPE="unormalized"    # 'orig' or 'unormalized'
SAVE_DIR="${BASE_DIR}/output/coreset/${MODEL_BRIEF}/"
BRIEF_PYTHON_PATH="${BASE_DIR}/brief/coreset/run_BRIEF.py"
ENABLE_PLOT="true"          # set to "true" to enable plotting
PREFIX=""                    # prefix for output files
ENABLE_AUTO_SEARCH="true"   # set to "true" to automatically search for optimal alpha
SEARCH_PRECISION="0.005"     # precision threshold for alpha search
MAX_ITERATIONS="15"          # maximum iterations for alpha search

# ---------- Final training on coreset (step‑4) ----------
TRAIN_RATIO="${PROPORTIONS}"
FINAL_JOB="gpt4-BRIEF-${TRAIN_RATIO}"
JOB_NAME_FINAL="Llama3-8B-${FINAL_JOB}"
# Dynamic coreset file path - will be set by step3_BRIEF.sh based on actual output
CORESET_CONFIG_FILE="${BASE_DIR}/coreset_path.conf"
if [ -f "$CORESET_CONFIG_FILE" ]; then
    source "$CORESET_CONFIG_FILE"
    echo "Loaded coreset file: $CORESET_FILE"
else
    CORESET_FILE=""  # Will be determined after BRIEF selection
fi
OUTPUT_PATH_FINAL="${BASE_DIR}/output/model/${JOB_NAME_FINAL}"
FINAL_SCRIPT_PATH="${BASE_DIR}/brief/scripts/lora_train_unified.sh"

# ---------- Helpers ----------
# Usage: LOG_FILE=$(log_file "$JOB_NAME")
log_file() {
  local job_name="$1"
  echo "${LOG_DIR}/${job_name}.log"
}
