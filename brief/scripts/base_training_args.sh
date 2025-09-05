#!/bin/bash

TARGET_EFF_BATCH_SIZE=128
PER_DEVICE_BATCH_SIZE=2      

# Enable CPU offloading for even larger models
export FSDP_CPU_RAM_EFFICIENT_LOADING=true
export FLASH_ATTENTION_USE_TRITON=1 
if [ -n "${GPUS:-}" ]; then
  IFS=',' read -ra GPU_LIST <<< "${GPUS:-}"
  export CUDA_VISIBLE_DEVICES="${GPUS:-}"
  NUM_GPUS=${#GPU_LIST[@]}
else
  unset CUDA_VISIBLE_DEVICES
  NUM_GPUS=$(nvidia-smi -L | wc -l)
fi
export NUM_GPUS
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES:-all} (${NUM_GPUS} GPUs detected)"


GRAD_ACC_STEPS=$(( TARGET_EFF_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NUM_GPUS) ))

if [ "$GRAD_ACC_STEPS" -lt 1 ]; then
  GRAD_ACC_STEPS=1
fi
echo "Gradient Accumulation Steps set to: $GRAD_ACC_STEPS (Effective BS = $TARGET_EFF_BATCH_SIZE)"

ID=$RANDOM
PORT=$((12000 + RANDOM % 20000))
export header="torchrun --nproc_per_node $NUM_GPUS --nnodes 1 --master_port $PORT -m brief.train.train"

export base_training_args="--do_train True \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--logging_steps 5 \
--save_strategy epoch \
--num_train_epochs ${NUM_TRAIN_EPOCHS:-4} \
--bf16 True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--optim adamw_torch \
--percentage 1.0 \
--save_total_limit 4 \
--report_to swanlab \
--use_flash_attn False \
--lora True \
--lora_r 128 \
--lora_alpha 512 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--learning_rate 5e-06 \
--per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
--gradient_accumulation_steps $GRAD_ACC_STEPS"