#!/usr/bin/env python
# coding=utf-8

import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Tuple

import datasets
import torch
import torch.distributed as dist
import transformers
from packaging import version
import torch.nn.functional as F
import numpy as np

# --- NEW: Validate Flash‑Attention 2 availability ---------------------------
try:
    import flash_attn  # noqa: F401
except ImportError as err:
    raise ImportError(
        "Flash‑Attention 2 is required but not installed.\n"
        "Install it with:\n"
        "  pip install --upgrade flash-attn --no-build-isolation\n\n"
        "See https://github.com/Dao-AILab/flash-attention for details"
    ) from err

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    set_seed,
)

from brief.get_grads.get_training_dataset import get_training_dataset
from brief.train.data_arguments import DataArguments, get_data_statistics
from brief.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from brief.train.training_arguments import TrainingArguments
import swanlab
from transformers import TrainerCallback


# ---------------------------------------------------------------------------
# Extend TrainingArguments with a simple flag so the feature can be toggled
# ---------------------------------------------------------------------------


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

IGNORE_INDEX = -100


def add_weights_to_dataset(dataset, weights):
    """Add weights to dataset for coreset training."""
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Weights size: {len(weights)}")

    def add_weight(example, idx):
        example["weight"] = weights[idx]
        return example

    dataset = dataset.map(add_weight, with_indices=True)
    logger.info(f"Dataset with weights: {dataset[0]['weight']}")
    return dataset



class WeightedTrainer(Trainer):
    """Custom trainer that supports weighted loss for coreset training."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        IGNORE_INDEX = -100
        labels = inputs.pop("labels")  # (B, L)
        weights = inputs.pop("weight", None)  # (B,) or None

        outputs = model(**inputs)
        logits = outputs.logits.float()
        vocab_size = logits.size(-1)

        shift_logits = logits[..., :-1, :].contiguous()  # (B, L-1, V)
        shift_labels = labels[..., 1:].contiguous()  # (B, L-1)

        token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )  # (B*(L-1),)
        token_loss = token_loss.view(shift_labels.size())  # (B, L-1)

        valid_mask = shift_labels.ne(IGNORE_INDEX)  # (B, L-1)
        token_loss = token_loss * valid_mask  # mask invalid

        if weights is not None:
            token_loss = token_loss * weights.unsqueeze(1)  # broadcast

        total_token_loss = token_loss.sum()
        if num_items_in_batch is None:
            raise ValueError(
                "num_items_in_batch must be provided for loss computation."
            )

        loss = total_token_loss / num_items_in_batch

        return (loss, outputs) if return_outputs else loss



def load_coreset(coreset_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the coreset file and return the subset and weights."""
    coreset_data = np.load(coreset_file)
    subset = coreset_data["subset"]
    weights = coreset_data["weights"]
    return subset, weights


def validate_dataset_lengths(dataset, max_length, dataset_name="dataset"):
    """Validate that dataset doesn't have sequences longer than max_length"""
    long_sequences = 0
    total_sequences = len(dataset)
    for i, example in enumerate(dataset):
        actual_length = len(example["input_ids"])
        if actual_length > max_length:
            long_sequences += 1
            if long_sequences <= 5:
                logger.warning(
                    f"{dataset_name} example {i} has length {actual_length} > {max_length}"
                )
    if long_sequences > 0:
        logger.warning(
            f"{dataset_name}: {long_sequences}/{total_sequences} sequences exceed max_length"
        )
    else:
        logger.info(f"{dataset_name}: All sequences are ≤ max_length ({max_length})")




def main():
    swanlab_api_key = os.getenv("SWANLAB_API_KEY")
    if swanlab_api_key:
        swanlab.login(api_key=swanlab_api_key, save=True)

    # Parse CLI / JSON arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16‑bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=True
    )
    # ValueError: You are attempting to perform batched generation with padding_side='right' this may lead to unexpected behaviour for Flash Attention version of Qwen3. Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input. 
    if (model_args.model_name_or_path.find("Qwen") != -1
            and tokenizer.padding_side == "right"):
        logger.warning(
            "Qwen model detected with right padding. Setting padding side to left for compatibility with Flash Attention."
        )
        tokenizer.padding_side = "left"
    # Determine if we're doing coreset training
    use_coreset = hasattr(data_args, "coreset_file") and data_args.coreset_file

    if use_coreset:
        logger.info(f"Loading Coreset file: {data_args.coreset_file}")
        subset, weights = load_coreset(data_args.coreset_file)
        logger.info(f"Loaded Coreset with {len(subset)} samples.")
        weights = weights / np.mean(weights)  # Normalize weights
        logger.info(f"Normalized weights: {weights[:10]}...")
        logger.info(f"Subset indices: {subset[:10]}...")

    # Load training dataset
    train_dataset = get_training_dataset(
        data_args.train_files,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        sample_percentage=data_args.percentage,
        seed=data_args.sample_data_seed,
    )

    if use_coreset:
        # Filter dataset using coreset subset and add weights
        logger.info(f"Filtering the dataset based on the Coreset subset...")
        logger.info(f"Original dataset size: {len(train_dataset)}")
        train_dataset = train_dataset.select(subset)
        logger.info(f"Filtered dataset size: {len(train_dataset)}")

        train_dataset = add_weights_to_dataset(train_dataset, weights)

        required_columns = ["input_ids", "labels", "attention_mask", "weight"]
        cols_to_remove = [
            col for col in train_dataset.column_names if col not in required_columns
        ]
        train_dataset = train_dataset.remove_columns(cols_to_remove)
        logger.info(f"Sample from train dataset: {train_dataset[0]}")

    # Initialize model
    pretrained_kwargs = {
        "torch_dtype": model_args.torch_dtype,
        "trust_remote_code": True,
    }

    if training_args.use_flash_attn:
        pretrained_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Initialising model with Flash‑Attention 2 kernels …")
    else:
        logger.info("Initialising model with *standard* attention …")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **pretrained_kwargs,
    )
    model.to(training_args.device)
    add_padding_to_tokenizer(tokenizer)

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info("Applied LoRA to model.")
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    get_data_statistics(train_dataset)

    if "dataset" in train_dataset.features:
        train_dataset = train_dataset.remove_columns(["dataset", "id", "messages"])

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}")

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    # Validate training dataset lengths
    validate_dataset_lengths(
        train_dataset, data_args.max_seq_length, "Training dataset"
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        print(model)
    elif not dist.is_initialized():
        print(model)

    if use_coreset:
        trainer_cls = WeightedTrainer
    else:
        trainer_cls = Trainer

    # compile the model
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model, padding="longest"),
    )


    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # remove the full model in the end to save space, only adapter is needed
    if isinstance(model, PeftModel):
        pytorch_model_path = os.path.join(
            training_args.output_dir, "pytorch_model_fsdp.bin"
        )
        os.remove(pytorch_model_path) if os.path.exists(pytorch_model_path) else None


if __name__ == "__main__":
    main()
