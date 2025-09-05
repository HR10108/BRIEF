#!/usr/bin/env bash
set -euo pipefail

child_pid=0
cleanup(){
  if [[ $child_pid -ne 0 ]]; then
    echo "Received signal, terminating training (PID=$child_pid)…"
    kill "$child_pid" 2>/dev/null || true
  fi
  exit
}
trap cleanup SIGINT SIGTERM

source brief/scripts/base_training_args.sh

# Parse arguments - supports both warmup and coreset training
train_files=$1
model_path=$2
job_name=$3
results_file=$4
output_dir=$5
seed=$6
max_length=$7

# Optional arguments for different training modes
if [ $# -eq 8 ]; then
    # Warmup training mode: 8th argument is sample_ratio
    sample_ratio=$8
    coreset_file=""
    echo "=== Warmup Training Mode ==="
elif [ $# -eq 9 ]; then
    # Coreset training mode: 8th argument is coreset_file, 9th is ignored (for compatibility)
    coreset_file=$8
    sample_ratio=1.0  # Not used in coreset mode
    echo "=== Coreset Training Mode ==="
    echo "Coreset file: $coreset_file"
else
    echo "Error: Invalid number of arguments"
    echo "Usage for warmup: $0 train_files model_path job_name results_file output_dir seed max_length sample_ratio"
    echo "Usage for coreset: $0 train_files model_path job_name results_file coreset_file output_dir seed max_length [ignored]"
    exit 1
fi

mkdir -p "$output_dir"

echo "Model path: $model_path"
echo "Save output directory: $output_dir"

echo $base_training_args
nvidia-smi

LOG_FILE="$output_dir/train.log"

ID=$RANDOM
PORT=$((12000 + RANDOM % 20000)) 
export header="torchrun --nproc_per_node $NUM_GPUS --nnodes 1 --master_port $PORT \
-m brief.train.train"

# Build command based on training mode
cmd_args="$header $base_training_args \
    --model_name_or_path $model_path \
    --output_dir $output_dir \
    --train_files $train_files \
    --seed $seed \
    --max_seq_length $max_length"

# Add mode-specific arguments
if [ -n "$coreset_file" ]; then
    # Coreset training mode
    cmd_args="$cmd_args --coreset_file $coreset_file"
else
    # Warmup training mode
    cmd_args="$cmd_args --percentage $sample_ratio"
fi

bash -c "\
  $cmd_args \
  2>&1 | tee \"$LOG_FILE\"" &
child_pid=$!

echo "🚀 Training started (PID=$child_pid), logs written to $LOG_FILE"
wait "$child_pid"
exit_code=$?

echo "Training process (PID=$child_pid) has exited with code $exit_code"
exit $exit_code