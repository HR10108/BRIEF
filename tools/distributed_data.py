#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distributed_data.py
1. split : Split json/jsonl data sequentially into N parts and write to shards/shard_k.jsonl
           Also add "sample_id" field to each sample for subsequent gradient alignment
2. merge : Merge grads calculated by multiple machines back in sample_id order
           Generate two sets of files: all_orig.pt (L2-normalized) and all_unormalized.pt
Usage:
    python distributed_data.py split \
        --train_file ../data/tulu3_mix_filtered_2048_single_turn.jsonl \
        --n_shards 8 \
        --out_dir ../output/grads/shards/

    python distributed_data.py merge \
        --grads_root grads/ \
        --dims 8192 4096 \
        --out_dir merged/
"""
import argparse, json, os, glob, math, shutil, sys
from typing import List, Dict
from pathlib import Path

import torch
from torch.nn.functional import normalize
from tqdm import tqdm


# ============== Part-1  Data Splitting ================================
def _load_examples(fp: str):
    """Load .json or .jsonl as List[dict]"""
    exs: List[Dict] = []
    with open(fp, "r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if first.lstrip().startswith("{"):  # jsonl
            for line in f:
                if line.strip():
                    exs.append(json.loads(line))
        else:  # regular json
            exs = json.load(f)
    return exs


def split_dataset(train_file: str, n_shards: int, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    examples = _load_examples(train_file)
    total = len(examples)
    shard_sz = math.ceil(total / n_shards)
    print(f"Total samples {total}, average per shard ≈ {shard_sz}")

    for i in range(n_shards):
        beg, end = i * shard_sz, min((i + 1) * shard_sz, total)
        shard = examples[beg:end]
        if not shard:
            continue
        # Add global sample_id to each sample
        for local_idx, ex in enumerate(shard):
            ex["sample_id"] = beg + local_idx  # 0-based
        shard_fp = Path(out_dir) / f"shard_{i}.jsonl"
        with open(shard_fp, "w", encoding="utf-8") as f:
            for ex in shard:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Written to {shard_fp}  ({len(shard)} items)")


# ============== Part-2  Multi-machine Result Merging =============================
def _collect_grads_single_dim(kind_dir: str) -> torch.Tensor:
    """
    kind_dir = '…/sft/dim8192'
    Read grads-*.pt files and concatenate them after sorting by filename numbers
    """
    files = sorted(
        glob.glob(os.path.join(kind_dir, "grads-*.pt")),
        key=lambda p: int(Path(p).stem.split("-")[1]),
    )
    parts = [torch.load(f, map_location="cpu") for f in files]
    return torch.cat(parts, dim=0) if parts else torch.empty(0, 0)


def merge_grads(grads_root: str, dims: List[int], out_dir: str):
    """
    grads_root/
        shard_0/sft/dim8192/grads-*.pt
        …
    Merge rule: Concatenate in order of shard_0, shard_1... (since order was not shuffled during splitting)

    Save hierarchy: out_dir/{kind}/dim{d}/{all_unormalized.pt, all_orig.pt}
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    shard_dirs = sorted([d for d in Path(grads_root).iterdir() if d.is_dir()])

    for kind in ["sft", "kn_loss", "if_loss"]:
        for dim in dims:
            cat_list = []
            for sd in shard_dirs:
                kind_dim_dir = sd / kind / f"dim{dim}"
                if not kind_dim_dir.exists():
                    print(f"⚠️  Skip {kind_dim_dir}, does not exist")
                    continue
                cat_list.append(_collect_grads_single_dim(str(kind_dim_dir)))
            if not cat_list:
                print(f"❌ No {kind} dim{dim} gradients found")
                continue

            merged = torch.cat(cat_list, dim=0)

            # —— Ensure save directory is consistent with read hierarchy ——
            save_dir = Path(out_dir) / kind / f"dim{dim}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # 1) Unnormalized
            un_fp = save_dir / "all_unormalized.pt"
            torch.save(merged, un_fp)

            # 2) L2-normalized
            norm = normalize(merged, dim=1)
            no_fp = save_dir / "all_orig.pt"
            torch.save(norm, no_fp)

            print(f"✅ {kind} dim{dim}  ->  {tuple(merged.shape)}   saved to {save_dir}/")


# =================== CLI Entry Point ===================================
def main():
    parser = argparse.ArgumentParser("distributed_data helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="Split training dataset")
    sp.add_argument("--train_file", required=True)
    sp.add_argument("--n_shards", type=int, required=True)
    sp.add_argument("--out_dir", required=True)

    mp = sub.add_parser("merge", help="Merge multi-machine gradients")
    mp.add_argument(
        "--grads_root",
        required=True,
        help="Parent directory of outputs from each machine (contains shard_0/, shard_1/...)",
    )
    mp.add_argument(
        "--dims", type=int, nargs="+", required=True, help="Should match training settings, e.g. 8192 4096"
    )
    mp.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    if args.cmd == "split":
        split_dataset(args.train_file, args.n_shards, args.out_dir)
    elif args.cmd == "merge":
        merge_grads(args.grads_root, args.dims, args.out_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
