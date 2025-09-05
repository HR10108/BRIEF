import contextlib
from functools import partial
from typing import List, Union

import numpy as np
import torch
from datasets import load_dataset
import multiprocessing
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase
import json
from datasets import Dataset, Features, Sequence, Value
import os

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_training_dataset(
    train_files: List[str], tokenizer, max_seq_length, sample_percentage=1.0, seed=0, is_get_grads=False
):
    """get training dataset with a specified seed"""
    raw_datasets = load_raw_dataset(
        train_files, sample_percentage=sample_percentage, seed=seed
    )
    lm_datasets = encode_data(raw_datasets, tokenizer, max_seq_length,is_get_grads=is_get_grads)

    # Add dataset length statistics
    lengths = []
    truncated_count = 0
    for example in lm_datasets:
        length = len(example["input_ids"])
        lengths.append(length)
        if length == max_seq_length:
            truncated_count += 1

    if lengths:
        import numpy as np

        print(f"Dataset length statistics:")
        print(f"  Total examples: {len(lengths)}")
        print(f"  Average length: {np.mean(lengths):.1f}")
        print(f"  Max length: {max(lengths)}")
        print(f"  Min length: {min(lengths)}")
        print(
            f"  Potentially truncated: {truncated_count} ({100*truncated_count/len(lengths):.1f}%)"
        )

    return lm_datasets


def load_raw_dataset(
    train_files: Union[List[str], str], sample_size=None, sample_percentage=1.0, seed=0
):
    """load raw dataset"""
    if isinstance(train_files, str):
        if os.path.isdir(train_files):
            print(f"Loading huggingface dataset from directory: {train_files}")
            processed_datasets = load_dataset(train_files)["train"]
        else:
            train_files = [train_files]
            processed_datasets = load_dataset(
                "json",
                data_files=train_files,
            )["train"]
    else:
        if len(train_files) == 1 and os.path.isdir(train_files[0]):
            print(f"Loading huggingface dataset from directory: {train_files[0]}")
            processed_datasets = load_dataset(train_files[0])["train"]
        else:
            processed_datasets = load_dataset(
                "json",
                data_files=train_files,
            )["train"]
    
    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        print("Load dataset without sampling.")
        return processed_datasets  # not shuffle

    with temp_seed(seed):
        index = np.random.permutation(len(processed_datasets))[:sample_size]

    sampled_dataset = processed_datasets.select(index)

    return sampled_dataset


def encode_data(
    raw_datasets,
    tokenizer,
    max_seq_length,
    processing_num_workers=10,
    overwrite_cache=False,
    func_name="encode_with_messages_format",
    is_get_grads=False
):
    """encode data with the specified tokenizer and the chat format."""
    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    encode_function = get_encode_function(
        raw_datasets, tokenizer, max_seq_length, func_name,is_get_grads
    )
    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    return lm_datasets


def get_encode_function(
    raw_datasets, tokenizer, max_seq_length, func="encode_with_messages_format",is_get_grads=False
):
    cols = raw_datasets.column_names

    if "prompt" in cols and "completion" in cols:
        if is_get_grads:
            raise ValueError(
                "The dataset has 'prompt' and 'completion' fields, but is_get_grads is True. "
                "This function is not compatible with get_grads."
            )
        return partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            is_get_grads=is_get_grads,
        )

    if "instruction" in cols and "output" in cols:
        raise ValueError(
            "The dataset has 'instruction' and 'output' fields")

    if "messages" in cols:
        if func == "encode_with_messages_format":
            encode_func = encode_with_messages_format
        else:
            if is_get_grads:
                raise ValueError(
                    "The dataset has 'messages' field, but is_get_grads is True. "
                    "This function is not compatible with get_grads."
                )
            encode_func = encode_with_messages_format_with_llama2_chat
        return partial(
            encode_func,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            is_get_grads=is_get_grads,
        )

    raise ValueError("The dataset does not have the required fields for encoding. ")



def encode_with_prompt_completion_format(example, tokenizer, max_seq_length,is_get_grads):
    """
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L238

    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example[
        "completion"
    ].startswith((" ", "\n", "\t")):
        example_text = example["prompt"] + " " + example["completion"]
    else:
        example_text = example["prompt"] + example["completion"]
    example_text = example_text + tokenizer.eos_token

    # Check original length before truncation
    original_length = len(
        tokenizer(example_text, return_tensors="pt", truncation=False).input_ids[0]
    )
    was_truncated = original_length > max_seq_length

    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    tokenized_prompt = tokenizer(
        example["prompt"],
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )

    # Ensure prompt length doesn't exceed actual input length after truncation
    prompt_length = min(tokenized_prompt.input_ids.shape[1], input_ids.shape[1])
    # mask the prompt part for avoiding loss
    labels[:, :prompt_length] = -100

    # Warn if completion was truncated
    if was_truncated and prompt_length >= input_ids.shape[1]:
        import warnings

        warnings.warn(
            f"Input was truncated and may have lost completion tokens. "
            f"Original length: {original_length}, Max length: {max_seq_length}"
        )

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }

def encode_with_messages_format(example, tokenizer, max_seq_length, is_get_grads):
    if not is_get_grads:
        return _encode_with_messages_format(example, tokenizer, max_seq_length)

    base_dict = _encode_with_messages_format(example, tokenizer, max_seq_length)

    assistant_msg = None
    for m in example["messages"]:
        if m["role"] == "assistant":
            assistant_msg = m["content"].strip()
            break
    if assistant_msg is None:
        raise ValueError("No assistant message found in example.")

    y_text = assistant_msg + tokenizer.eos_token
    y_tok = tokenizer(
        y_text,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    y_input_ids = y_tok.input_ids                        # (1, L)
    y_attention_mask = torch.ones_like(y_input_ids)      # (1, L)
    y_labels = y_input_ids.clone()                       

    return {
        "input_ids":        base_dict["input_ids"],           # (seq_len,)
        "attention_mask":   base_dict["attention_mask"],      # (seq_len,)
        "labels":           base_dict["labels"],              # (seq_len,)
        "y_labels":         y_labels.flatten(),               # (y_len,)
        "y_input_ids":      y_input_ids.flatten(),            # (y_len,)
        "y_attention_mask": y_attention_mask.flatten(),       # (y_len,)
    }





def _encode_with_messages_format(example, tokenizer, max_seq_length):
    """
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L264C1-L322C1

    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    example_text = concat_messages(messages, tokenizer)

    # Check original length before truncation
    original_length = len(
        tokenizer(example_text, return_tensors="pt", truncation=False).input_ids[0]
    )
    was_truncated = original_length > max_seq_length

    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                partial_messages = concat_messages(messages[:message_idx], tokenizer)
                message_start_idx = len(
                    tokenizer(
                        partial_messages,
                        return_tensors="pt",
                        max_length=max_seq_length,
                        truncation=True,
                    ).input_ids[0]
                )

            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # here we also ignore the role of the assistant
                messages_so_far = (
                    concat_messages(messages[: message_idx + 1], tokenizer)
                    + "<|assistant|>\n"
                )
            else:
                messages_so_far = concat_messages(
                    messages[: message_idx + 1], tokenizer
                )

            message_end_idx = len(
                tokenizer(
                    messages_so_far,
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids[0]
            )

            # Ensure indices don't exceed actual sequence length after truncation
            message_start_idx = min(message_start_idx, input_ids.shape[1])
            message_end_idx = min(message_end_idx, input_ids.shape[1])

            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    # Warn if assistant responses were truncated
    if was_truncated:
        import warnings

        warnings.warn(
            f"Messages were truncated and may have lost assistant responses. "
            f"Original length: {original_length}, Max length: {max_seq_length}"
        )

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += (
                "<|assistant|>\n"
                + message["content"].strip()
                + tokenizer.eos_token
                + "\n"
            )
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text


def encode_with_messages_format_with_llama2_chat(example, tokenizer, max_seq_length,is_get_grads):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(
        messages,
    ):
        B_INST, E_INST = "[INST]", "[/INST]"
        bos = "<s>"
        eos = "</s>"
        formatted_text = ""
        for message in messages:
            if message["role"] == "user":
                formatted_text += (
                    bos + f"{B_INST} {(message['content']).strip()} {E_INST}"
                )
            elif message["role"] == "assistant":
                formatted_text += f" {(message['content'])} " + eos
            else:
                raise ValueError(
                    "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"]
                    )
                )
        formatted_text = formatted_text[len(bos) :]
        return formatted_text

    example_text = _concat_messages(messages).strip()
    print(example_text)
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if messages[message_idx + 1]["role"] == "assistant":
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # When getting gradients, we only do this single batch process
        collate_fn=data_collator,
        shuffle=False,
    )
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader

