import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from functools import partial
import torch
import argparse


def concat_messages(messages, tokenizer):
    """Concatenate messages with role delimiters."""
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
            raise ValueError(f"Invalid role: {message['role']}")
    return message_text


def check_truncation_for_dataset(example, tokenizer, max_seq_length):
    """Check if example would be truncated, detect multi‑turn, and add metadata."""
    try:
        messages = example["messages"]
        if len(messages) == 0:
            example.update(
                {
                    "was_truncated": True,
                    "original_length": 0,
                    "is_multi_turn": False,
                }
            )
            return example

        # Detect multi‑turn: more than one user→assistant exchange
        user_cnt = sum(m["role"] == "user" for m in messages)
        assistant_cnt = sum(m["role"] == "assistant" for m in messages)
        example["is_multi_turn"] = (user_cnt > 1) or (assistant_cnt > 1)

        example_text = concat_messages(messages, tokenizer)

        tokenized = tokenizer(example_text, return_tensors="pt", truncation=False)
        original_length = len(tokenized.input_ids[0])

        example["was_truncated"] = original_length > max_seq_length
        example["original_length"] = original_length
        return example
    except Exception as e:
        print(f"Error processing example: {e}")
        example.update(
            {
                "was_truncated": True,
                "original_length": 0,
                "is_multi_turn": False,
            }
        )
        return example


def filter_truncated_dataset_optimized(
    *,
    dataset_path: str,
    tokenizer_name: str,
    max_seq_length: int,
    output_path: str,
    num_proc: int | None = None,
    batch_size: int = 1000,
    filter_multi_turn: bool = False,
):
    """Filter examples longer than `max_seq_length`.
    Optionally keep only single‑turn dialogues when `filter_multi_turn` is True.
    """

    # Set number of processes
    if num_proc is None:
        num_proc = os.cpu_count() or 1

    print(f"Using {num_proc} processes for parallel processing")

    # Load tokenizer
    print(f"Loading tokenizer from: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    if os.path.isdir(dataset_path):
        dataset = load_dataset(dataset_path)["train"]
    else:
        dataset = load_dataset("json", data_files=dataset_path)["train"]

    print(f"Total examples in dataset: {len(dataset)}")
    # Check if the id column is unique
    if "id" in dataset.column_names:
        # dataset["id"] is a list
        ids = dataset["id"]
        if len(ids) != len(set(ids)):
            print("Warning: Found duplicate IDs in the dataset. This may cause issues.")
            print(f"Total IDs: {len(ids)}, Unique IDs: {len(set(ids))}")
    # Add truncation & multi‑turn check
    print("Checking examples...")
    check_func = partial(
        check_truncation_for_dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    dataset_with_meta = dataset.map(
        check_func,
        num_proc=num_proc,
        desc="Annotating",
        batch_size=batch_size,
    )

    # Filter examples
    print(
        "Filtering truncated "
        + ("and multi‑turn" if filter_multi_turn else "")
        + " examples..."
    )

    def _keep(example):
        keep = (not example["was_truncated"]) and (example["original_length"] > 0)
        if filter_multi_turn:
            keep = keep and (not example["is_multi_turn"])
        return keep

    filtered_dataset = dataset_with_meta.filter(
        _keep, num_proc=num_proc, desc="Filtering"
    )

    # Collect statistics
    lengths = [l for l in dataset_with_meta["original_length"] if l > 0]
    truncated_count = sum(dataset_with_meta["was_truncated"])
    multi_turn_total = sum(dataset_with_meta["is_multi_turn"])
    multi_turn_kept = sum(filtered_dataset["is_multi_turn"]) if filter_multi_turn else multi_turn_total

    # Save
    print(f"\nSaving {len(filtered_dataset)} examples to: {output_path}")

    filtered_dataset = filtered_dataset.remove_columns(
        ["was_truncated", "original_length", "is_multi_turn"]
    )
    filtered_dataset.to_json(
        output_path, orient="records", lines=True, force_ascii=False
    )

    # Stats
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Original dataset size: {len(dataset)}")
    print(f"Valid examples: {len(lengths)}")
    print(f"Truncated examples: {truncated_count} ({truncated_count/len(lengths):.1%})")
    print(f"Multi‑turn dialogues: {multi_turn_total} ({multi_turn_total/len(dataset):.1%})")
    if filter_multi_turn:
        removed_multi_turn = multi_turn_total - multi_turn_kept
        print(
            f"Multi‑turn removed: {removed_multi_turn} ({removed_multi_turn/len(dataset):.1%})"
        )
    print(f"Kept examples: {len(filtered_dataset)} ({len(filtered_dataset)/len(lengths):.1%})")

    if lengths:
        print(f"\nLength distribution (in tokens):")
        print(f"  Mean: {np.mean(lengths):.1f}")
        print(f"  Median: {np.median(lengths):.1f}")
        print(f"  Std: {np.std(lengths):.1f}")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  95th percentile: {np.percentile(lengths,95):.1f}")
        print(f"  99th percentile: {np.percentile(lengths,99):.1f}")

    print("=" * 50)

    # Estimate output size
    if len(filtered_dataset) > 0:
        sample = filtered_dataset.select(range(min(100, len(filtered_dataset))))
        avg_size = np.mean([len(json.dumps(ex["messages"])) for ex in sample])
        est_mb = (avg_size * len(filtered_dataset)) / (1024 * 1024)
        print(f"Estimated output file size: {est_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Filter dataset by sequence length and optional single‑turn dialogues."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Directory or JSONL file containing the dataset.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="",
        help="Tokenizer name or path.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length to keep.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the filtered dataset (jsonl).",
    )
    parser.add_argument(
        "--filter_multi_turn",
        action="store_true",
        help="If set, keep only single‑turn (one user + one assistant) dialogues.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of CPU processes to use (default: all cores).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for the mapping step.",
    )

    args = parser.parse_args()

    if args.output_path is None:
        base_name = os.path.basename(os.path.abspath(args.dataset_path.rstrip('/')))
        suffix = "_single_turn" if args.filter_multi_turn else ""
        args.output_path = f"{base_name}_filtered_{args.max_seq_length}{suffix}.jsonl"

    filter_truncated_dataset_optimized(
        dataset_path=args.dataset_path,
        tokenizer_name=args.tokenizer_name,
        max_seq_length=args.max_seq_length,
        output_path=args.output_path,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        filter_multi_turn=args.filter_multi_turn,
    )


if __name__ == "__main__":
    main()
