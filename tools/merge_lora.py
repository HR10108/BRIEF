"""
batch_merge_lora.py
Sequentially merge multiple LoRA adapters to the same base model and save them separately
Dependencies:
    pip install transformers==4.41.0 peft==0.11.1 accelerate safetensors
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== 1. Global Configuration =====
# Base model (remains unchanged)
BASE_MODEL_NAME_OR_PATH = (
    ""
)

# List of LoRA adapters to batch merge (add/adjust as needed)
LORA_NAMES = [
    "",

]

# Root directory (consistent with original script)
LORA_ADAPTER_DIR = (
    ""
)
MERGED_SAVE_ROOT = (
    ""
)

# ===== 2. Pre-load tokenizer (once is enough) =====
print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME_OR_PATH,
    trust_remote_code=True,
)



# ===== 3. Process each LoRA individually =====
for lora_name in LORA_NAMES:
    print(f"\n================  Starting to process {lora_name}  ================\n")

    lora_adapter_path = f"{LORA_ADAPTER_DIR}/{lora_name}"
    merged_save_path = f"{MERGED_SAVE_ROOT}/{lora_name}-merged"
    if (lora_adapter_path.find("Qwen") != -1
        and tokenizer.padding_side == "right"):
        print(
            "Qwen model detected with right padding. Setting padding side to left for compatibility with Flash Attention."
        )
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # 3.1 Load base model
    print("🔄 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME_OR_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        base_model.resize_token_embeddings(len(tokenizer))
    # 3.2 Mount LoRA adapter
    print("🔄 Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # 3.3 Merge LoRA into model
    print("🪄 Merging LoRA weights (merge_and_unload)...")
    model = model.merge_and_unload()  # Return merged pure model

    # 3.4 Save merged model & tokenizer
    print(f"💾 Saving to {merged_save_path}...")
    model.save_pretrained(merged_save_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_save_path)

    print(f"✅ {lora_name} merge completed!\nYou can directly use {merged_save_path} for inference.")

print("\n🎉 All LoRA adapters have been merged successfully!")
