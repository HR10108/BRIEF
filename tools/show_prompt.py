import numpy as np
import random
from pprint import pprint
from brief.get_grads.get_training_dataset import load_raw_dataset, concat_messages
from transformers import AutoTokenizer

def show_sample_prompt(
    train_files,
    tokenizer,
    max_seq_length,
    sample_percentage=1.0,
    seed=0,
):
    """
    Randomly view the appearance after concatenating the prompt.
    
    Args:
        train_files (str | list[str]): Parameters passed to `load_raw_dataset`
        tokenizer (transformers.PreTrainedTokenizerBase): Your tokenizer
        max_seq_length (int): Keep consistent with training time for comparison of truncation
        sample_percentage (float): Adjust this parameter if you only want to sample a subset
        seed (int): Control randomness
    """
    # 1. Load / sample
    raw_dataset = load_raw_dataset(
        train_files,
        sample_percentage=sample_percentage,
        seed=seed
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    
    # 2. Randomly select a sample
    np.random.seed(seed)
    idx = np.random.randint(0, len(raw_dataset))
    example = raw_dataset[idx]
    
    # 3. Concatenate Prompt according to field format
    if {"prompt", "completion"}.issubset(raw_dataset.column_names):
        # ---- prompt + completion format ----
        # Keep completely consistent with encode_with_prompt_completion_format
        if not example["prompt"].endswith((" ", "\n", "\t")) and not example["completion"].startswith((" ", "\n", "\t")):
            prompt_text = example["prompt"] + " " + example["completion"]
        else:
            prompt_text = example["prompt"] + example["completion"]
        prompt_text += tokenizer.eos_token
    elif "messages" in raw_dataset.column_names:
        # ---- Chat messages format ----
        prompt_text = concat_messages(example["messages"], tokenizer)
    else:
        raise ValueError("Current dataset fields are not within the script's support range.")
    
    # 4. Print results
    print(f"\n=== Original sample (index {idx}) ===")
    pprint(example)
    
    print("\n=== Concatenated Prompt ===")
    print(prompt_text)
    
    # 5. Display token length to see if it exceeds max_seq_length
    tokenized = tokenizer(prompt_text, return_tensors="pt", truncation=False)
    print(f"\nTokenized length: {tokenized.input_ids.shape[1]} / {max_seq_length}")

# —— Usage example ——
show_sample_prompt(
    train_files="",
    tokenizer="",
    max_seq_length=2048,
    sample_percentage=1.0,
    seed=42,
)
