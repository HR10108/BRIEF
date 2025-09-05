#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============

Select coresets with BRIEF from pre-computed gradient tensors
and (optionally) plot error curves.

Example
-------
python run_BRIEF.py \
    --grad_dir /data/shenchaoyuan/BRIEF/grads/Llama3-8B-5-warmup-tulu3-128-2048-5e-6-lora \
    --proj_dim 8192 \
    --device cpu \
    --output_dir ./Llama_un_auto/ \
    --plot \
    --prefix auto15 \
    --grad_type unormalized \
    --search_precision 0.005 \
    --max_iterations 15 \
    --auto_search

"""

import os
import math
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------ Optional plotting ------------------------------ #
def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

# ------------------------------ Data loading ------------------------------ #
KIND2DIR = {"sft": "sft", "kn_loss": "kn_loss", "if_loss": "if_loss"}


def load_grad_block(root: str, kind: str, dim: int, device: str,grad_type: str) -> torch.Tensor:
    """
    Load gradient tensor (N, dim) from SuperFiltering directory.

    Parameters
    ----------
    root : str
        root directory containing sft/kn_loss/if_loss sub-dirs
    kind : str
        'sft' | 'kn_loss' | 'if_loss'
    dim : int
        feature dimension
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    torch.Tensor
        (N, dim) float32 tensor on ``device``
    """
    logging.info("Loading %s gradients, type=%s", kind, grad_type)
    path = os.path.join(root, KIND2DIR[kind], f"dim{dim}", f"all_{grad_type}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    grad = torch.load(path, map_location=device).float()
    # check if there is nan or inf
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        logging.warning("Gradient tensor contains NaN or Inf values. This may affect selection quality.")
    return grad


# ------------------------------ DummyModel ------------------------------ #
class DummyModel(torch.nn.Module):
    """
    Placeholder model so BRIEF works with if_convex=True.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, 1, bias=False)
        self._emb_dim = input_dim

    def forward(self, x, **kwargs):
        return self.fc(x), x

    def get_embedding_dim(self):
        return self._emb_dim


# ------------------------------ BRIEFStrategy Classes ------------------------------ #
from cords.selectionstrategies.SL.craigstrategy import CRAIGStrategy


class MultiGradBRIEFStrategy(CRAIGStrategy):

    def __init__(
        self,
        kn_loss_means: torch.Tensor,
        if_loss_means: torch.Tensor,
        alpha: float,
        shuffle_indices: torch.Tensor,
        batch_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if kn_loss_means.shape != if_loss_means.shape:
            raise ValueError("kn_loss_means and if_loss_means shape mismatch")
        self.kn_loss_means = kn_loss_means
        self.if_loss_means = if_loss_means
        self.alpha = alpha
        self.shuffle_indices = shuffle_indices
        self.batch_size = batch_size
        logging.info(
            "MultiGradBRIEFStrategy initialized with batch_size=%d, selection_type=%s",
            self.batch_size,
            self.selection_type,
        )

    def compute_score(self, model_params, idxs):
        if self.selection_type == "PerBatch":
            kn_loss = self.kn_loss_means
            if_loss = self.if_loss_means
        else:  # PerClass
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.to(self.kn_loss_means.device)
            kn_loss = self.kn_loss_means[idxs]
            if_loss = self.if_loss_means[idxs]

        N = kn_loss.size(0)
        chunk_size = min(50000, N)  # Adjustable chunk size to balance memory and speed
        
        # Initialize distance matrix (using float16 can further save memory)
        dist = torch.zeros((N, N), device='cpu', dtype=torch.float32)
        
        # Compute distance matrix in chunks
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            for j in range(0, N, chunk_size):
                end_j = min(j + chunk_size, N)
                
                # Calculate distance for current chunk
                d_kn_loss_chunk = torch.cdist(kn_loss[i:end_i], kn_loss[j:end_j], p=2)
                d_if_loss_chunk = torch.cdist(if_loss[i:end_i], if_loss[j:end_j], p=2)
                
                # Combine distances and store
                if self.alpha == 0:
                    dist_chunk = d_kn_loss_chunk
                elif self.alpha == 1:
                    dist_chunk = d_if_loss_chunk
                else:
                    dist_chunk = d_kn_loss_chunk / self.alpha + d_if_loss_chunk / (1.0 - self.alpha)
                dist[i:end_i, j:end_j] = dist_chunk.cpu()
                
                # Immediately free GPU memory
                del d_kn_loss_chunk, d_if_loss_chunk, dist_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.N = dist.size(0)
        self.dist_mat = dist
        self.const = torch.max(self.dist_mat).item()
        self.dist_mat = (self.const - self.dist_mat).numpy()
        
        logging.info(
            "Computed distance matrix with shape %s using chunked computation",
            self.dist_mat.shape
        )

    def select(self, budget, model_params):
        shuffled_indices, gammas = super().select(budget, model_params)

        original_indices = [
            self.shuffle_indices[idx].item() for idx in shuffled_indices
        ]
        return original_indices, gammas
    
    def select_multi_budget(self, budget_ratios, model_params):
        """Wrap parent's select_multi_budget to return original indices"""
        results = super().select_multi_budget(budget_ratios, model_params)
        
        # Convert shuffled indices back to original indices
        converted_results = {}
        for ratio, (shuffled_indices, gammas) in results.items():
            original_indices = [
                self.shuffle_indices[idx].item() for idx in shuffled_indices
            ]
            converted_results[ratio] = (original_indices, gammas)
        
        return converted_results

    def reset(self):
        self.dist_mat = None
        self.N = 0
        self.const = 0.0


class SingleGradBRIEFStrategy(CRAIGStrategy):

    def __init__(
        self,
        grad_means: torch.Tensor,
        shuffle_indices: torch.Tensor,
        batch_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grad_means = grad_means
        self.shuffle_indices = shuffle_indices
        self.batch_size = batch_size
        logging.info(
            "SingleGradBRIEFStrategy initialized with batch_size=%d, selection_type=%s",
            self.batch_size,
            self.selection_type,
        )

    def compute_score(self, model_params, idxs):
        if self.selection_type == "PerBatch":
            grads = self.grad_means
        else:  # PerClass
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.to(self.grad_means.device)
            grads = self.grad_means[idxs]

        N = grads.size(0)
        chunk_size = min(50000, N)  # Adjustable chunk size to balance memory and speed
        
        # Initialize distance matrix (using float16 can further save memory)
        dist = torch.zeros((N, N), device='cpu', dtype=torch.float32)
        
        # Compute distance matrix in chunks
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            for j in range(0, N, chunk_size):
                end_j = min(j + chunk_size, N)
                
                # Calculate distance for current chunk
                dist_chunk = torch.cdist(grads[i:end_i], grads[j:end_j], p=2)
                
                dist[i:end_i, j:end_j] = dist_chunk.cpu()
                
                # Immediately free GPU memory
                del dist_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.N = dist.size(0)
        self.dist_mat = dist
        self.const = torch.max(self.dist_mat).item()
        self.dist_mat = (self.const - self.dist_mat).numpy()
        
        logging.info(
            "Computed distance matrix with shape %s using chunked computation",
            self.dist_mat.shape
        )

    def select(self, budget, model_params):
        shuffled_indices, gammas = super().select(budget, model_params)

        original_indices = [
            self.shuffle_indices[idx].item() for idx in shuffled_indices
        ]
        return original_indices, gammas
    
    def select_multi_budget(self, budget_ratios, model_params):
        """Wrap parent's select_multi_budget to return original indices"""
        results = super().select_multi_budget(budget_ratios, model_params)
        
        # Convert shuffled indices back to original indices
        converted_results = {}
        for ratio, (shuffled_indices, gammas) in results.items():
            original_indices = [
                self.shuffle_indices[idx].item() for idx in shuffled_indices
            ]
            converted_results[ratio] = (original_indices, gammas)
        
        return converted_results

    def reset(self):
        self.dist_mat = None
        self.N = 0
        self.const = 0.0


# ------------------------------ Alpha automatic search ------------------------------ #
def golden_section_search_alpha(
    brief: MultiGradBRIEFStrategy,
    model_params,
    grads_kn_loss: torch.Tensor,
    grads_if_loss: torch.Tensor,
    proportion: float,
    precision: float = 0.01,
    max_iterations: int = 10,
    device: str = "cpu"
) -> tuple:
    """
    Use golden section search to find optimal alpha that minimizes error_kn_loss + error_if_loss
    
    Parameters
    ----------
    brief : MultiGradBRIEFStrategy
        BRIEF strategy instance
    model_params : dict
        Model parameters
    grads_kn_loss : torch.Tensor
        kn_loss gradient tensor
    grads_if_loss : torch.Tensor  
        if_loss gradient tensor
    proportion : float
        Selection proportion
    precision : float
        Search precision threshold
    max_iterations : int
        Maximum iterations
    device : str
        Computing device
        
    Returns
    -------
    tuple
        (optimal_alpha, min_error_sum, search_history)
    """
    logging.info("Starting golden section search for optimal alpha")
    
    # Golden section ratio
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    
    # Initial search interval [0, 1]
    tol = precision
    a, b = 0, 1
    
    # Initial interior points
    x1 = a + resphi * (b - a)  
    x2 = b - resphi * (b - a)
    
    search_history = []
    
    def evaluate_alpha(alpha_val):
        """Evaluate error_kn_loss + error_if_loss for given alpha"""
        brief.reset()
        brief.alpha = alpha_val
        
        # Select samples
        selected_indices, _ = brief.select(int(len(grads_kn_loss) * proportion), model_params)
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        
        # Calculate errors
        error_kn_loss = compute_coreset_error(grads_kn_loss, selected_indices, device=device, chunk_size=16384)
        error_if_loss = compute_coreset_error(grads_if_loss, selected_indices, device=device, chunk_size=16384)
        error_sum = error_kn_loss + error_if_loss
        
        logging.info("Alpha=%.4f: error_kn_loss=%.6f, error_if_loss=%.6f, sum=%.6f", 
                    alpha_val, error_kn_loss, error_if_loss, error_sum)
        
        return error_sum, error_kn_loss, error_if_loss
    
    # Calculate function values at initial points
    f1_sum, f1_kn_loss, f1_if_loss = evaluate_alpha(x1)
    f2_sum, f2_kn_loss, f2_if_loss = evaluate_alpha(x2) 
    
    search_history.append((x1, f1_sum, f1_kn_loss, f1_if_loss))
    search_history.append((x2, f2_sum, f2_kn_loss, f2_if_loss))
    
    iteration = 0
    while abs(b - a) > tol and iteration < max_iterations:
        iteration += 1
        logging.info("Golden search iteration %d: interval [%.4f, %.4f], width=%.4f", 
                    iteration, a, b, b - a)
        
        if f1_sum < f2_sum:
            b = x2
            x2 = x1
            f2_sum = f1_sum
            x1 = a + resphi * (b - a)
            f1_sum, f1_kn_loss, f1_if_loss = evaluate_alpha(x1)
            search_history.append((x1, f1_sum, f1_kn_loss, f1_if_loss))
        else:
            a = x1  
            x1 = x2
            f1_sum = f2_sum
            x2 = b - resphi * (b - a)
            f2_sum, f2_kn_loss, f2_if_loss = evaluate_alpha(x2)
            search_history.append((x2, f2_sum, f2_kn_loss, f2_if_loss))
    
    # Return optimal solution
    if f1_sum < f2_sum:
        optimal_alpha = x1
        min_error_sum = f1_sum
    else:
        optimal_alpha = x2
        min_error_sum = f2_sum
    
    logging.info("Golden section search completed after %d iterations", iteration)
    logging.info("Optimal alpha=%.4f with error_sum=%.6f", optimal_alpha, min_error_sum)
    
    return optimal_alpha, min_error_sum, search_history


# ------------------------------ Random selection baseline ------------------------------ #
def random_selection_baseline(
    N: int,
    proportions: list,
    random_ratio: float = 1.0,
    seed: int = 42
) -> dict:
    """
    Random selection baseline, randomly select specified proportion from r% of total space
    
    Parameters
    ----------
    N : int
        Total number of samples
    proportions : list
        List of selection proportions
    random_ratio : float
        Proportion to randomly select from total (0-1)
    seed : int
        Random seed
    
    Returns
    -------
    dict
        {proportion: selected_indices}
    """
    np.random.seed(seed)
    
    # Calculate number of samples to select from total population
    candidate_size = int(N * random_ratio)
    candidate_indices = np.random.choice(N, candidate_size, replace=False)
    
    results = {}
    for p in proportions:
        # Select specified proportion from candidate samples
        select_size = int(N * p)
        if select_size > candidate_size:
            logging.warning(f"Requested size {select_size} > candidate size {candidate_size}, using all candidates")
            selected = candidate_indices
        else:
            selected = np.random.choice(candidate_indices, select_size, replace=False)
        results[p] = selected
    
    return results


# ------------------------------ Error calculation ------------------------------ #
def compute_coreset_error(
    data: torch.Tensor,
    sel_indices: np.ndarray,
    device: str = "cpu",
    chunk_size: int = 16384,
) -> float:
    """
    Average minimum Euclidean distance from all points to the chosen coreset.
    """
    # Ensure using specified device
    data_device = data.to(device).float()
    sel = data_device[torch.as_tensor(sel_indices, dtype=torch.long, device=device)]
    N = data_device.size(0)
    total_dist = 0.0
    
    # Use torch.cuda.device context manager to ensure operations on correct device
    if device.startswith("cuda"):
        device_id = int(device.split(":")[-1]) if ":" in device else 0
        with torch.cuda.device(device_id):
            for start in tqdm.tqdm(range(0, N, chunk_size), desc=f"Computing distances on {device}"):
                end = min(start + chunk_size, N)
                chunk = data_device[start:end]
                dists = torch.cdist(chunk, sel, p=2)
                mins, _ = torch.min(dists, dim=1)
                total_dist += mins.sum().item()
    else:
        for start in tqdm.tqdm(range(0, N, chunk_size), desc=f"Computing distances on {device}"):
            end = min(start + chunk_size, N)
            chunk = data_device[start:end]
            dists = torch.cdist(chunk, sel, p=2)
            mins, _ = torch.min(dists, dim=1)
            total_dist += mins.sum().item()
    
    return total_dist / N


# Helper function for parallel error computation
def compute_error_wrapper(grads, idxs, device, chunk_size, error_type):
    """Wrapper function for parallel execution"""
    error = compute_coreset_error(grads, idxs, device=device, chunk_size=chunk_size)
    return error_type, error


# ------------------------------ Main process ------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select coresets with BRIEF (PerClass) and optionally plot errors"
    )
    parser.add_argument("--grad_dir", required=True, help="gradient root dir")
    parser.add_argument("--proj_dim", type=int, default=8192, help="vector dim")
    parser.add_argument("--alpha", type=float, default=0.5, help="default α")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--proportions",
        type=float,
        nargs="+",
        default=[0.15],
        help="subset size fractions (0~1)",
    )
    parser.add_argument(
        "--output_dir", default="brief_outputs", help="directory to save indices & plots"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="enable plotting (saved to --output_dir)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="enable parallel error computation on multiple GPUs",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="shuffle gradients before selection",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="prefix for output files (optional)",
    )
    parser.add_argument(
        "--selection_type",
        choices=[ "PerClass"],
        default="PerClass",
        help="BRIEF selection granularity",
    )
    parser.add_argument(
        "--map_file",
        type=str,
        default=None,
        help="(PerClass only) .npy file mapping sample idx → class id. If omitted, defaults to a single-class mapping.",
    )
    parser.add_argument(
        "--grad_only",
        choices=["sft"],
        default=None,
        help="Run in grad-only mode, using only SFT gradients (CRAIG)"
    )
    parser.add_argument(
        "--grad_type",
        choices=["orig","unormalized"],
        default="orig",
        help="type of gradients to use (orig or unormalized)"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        default=False,
        help="use random selection baseline instead of BRIEF"
    )
    parser.add_argument(
        "-r", "--random_ratio",
        type=float,
        default=1.0,
        help="ratio of indices to randomly select for baseline (0-1, default 1.0)"
    )
    parser.add_argument(
        "--auto_search",
        action="store_true",
        default=False,
        help="automatically search for optimal alpha that minimizes error_kn_loss + error_if_loss"
    )
    parser.add_argument(
        "--search_precision",
        type=float,
        default=0.01,
        help="precision threshold for alpha search (default 0.01)"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="maximum iterations for alpha search (default 10)"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
    )
    logger = logging.getLogger("BRIEF")
    logger.info("Args: %s", args)
    args.sft_only = not (args.grad_only == None)
    # Check GPU availability
    if args.parallel and not args.sft_only:
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            logger.warning(f"Only {gpu_count} GPU(s) available. Falling back to sequential computation.")
            args.parallel = False
        else:
            logger.info(f"Found {gpu_count} GPUs. Will use cuda:0 and cuda:1 for parallel computation.")

    # Random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check running mode
    if args.random:
        logger.info("Running in random selection baseline mode with ratio=%.3f", args.random_ratio)
        if not (0.0 < args.random_ratio <= 1.0):
            parser.error("--random_ratio must be in (0, 1]")
    elif args.auto_search:
        logger.info("Running in auto search mode for optimal alpha")
        if args.sft_only:
            parser.error("--auto_search requires multi-gradient mode (remove --grad_only)")
        if args.selection_type != "PerClass":
            parser.error("--auto_search currently only supports PerClass selection")

    # 1. Load gradients
    if args.sft_only:
        logger.info(f"Grad-only mode: Loading {args.grad_only} gradients from %s", args.grad_dir)
        grads_sft = load_grad_block(args.grad_dir, args.grad_only, args.proj_dim, args.device,args.grad_type)
        N = grads_sft.shape[0]
        logger.info(f"{args.grad_only} gradient tensor shape: %s", tuple(grads_sft.shape))
        logger.info("Loading kn_loss & if_loss gradients from %s", args.grad_dir)
        grads_kn_loss = load_grad_block(args.grad_dir, "kn_loss", args.proj_dim, args.device,args.grad_type)
        grads_if_loss = load_grad_block(args.grad_dir, "if_loss", args.proj_dim, args.device,args.grad_type)
    else:
        logger.info("Loading kn_loss & if_loss gradients from %s", args.grad_dir)
        grads_kn_loss = load_grad_block(args.grad_dir, "kn_loss", args.proj_dim, args.device,args.grad_type)
        grads_if_loss = load_grad_block(args.grad_dir, "if_loss", args.proj_dim, args.device,args.grad_type)
        if grads_kn_loss.shape != grads_if_loss.shape:
            raise ValueError("kn_loss and if_loss gradient tensors shape mismatch")
        N = grads_kn_loss.shape[0]
        logger.info("Gradient tensor shape: %s", tuple(grads_kn_loss.shape))

    # 2. Shuffle indices & data (+ optional class mapping)
    if args.shuffle:
        shuffle_indices = torch.randperm(N)
        logger.info("Shuffling indices")
    else:
        shuffle_indices = torch.arange(N, dtype=torch.long)
        logger.info("Not shuffling indices")
    logger.info("Using %d indices", N)
    
    if args.sft_only:
        grads_sft_shuffled = grads_sft[shuffle_indices]
    else:
        grads_kn_loss_shuffled = grads_kn_loss[shuffle_indices]
        grads_if_loss_shuffled = grads_if_loss[shuffle_indices]

    # 3. Generate labels
    if args.selection_type == "PerClass":
        # If no map_file is provided, default to a single-class mapping of length N
        if args.map_file is None:
            logger.warning(
                "PerClass mode without --map_file: defaulting to a single-class mapping (all zeros), N=%d",
                N,
            )
            mapping = np.zeros(N, dtype=np.int64)
        else:
            mapping = np.load(args.map_file)
            if len(mapping) != N:
                raise ValueError(
                    f"mapping length {len(mapping)} doesn't match gradient count {N}"
                )
            # Ensure integer dtype
            if mapping.dtype not in (np.int64, np.int32, np.int16, np.int8, np.uint8):
                mapping = mapping.astype(np.int64)

        mapping_shuffled = mapping[shuffle_indices.cpu().numpy()]
        labels = torch.from_numpy(mapping_shuffled).long()
        num_classes = int(mapping.max()) + 1  # will be 1 when defaulting to single-class
    else:
        labels = torch.zeros(N, dtype=torch.long)
        num_classes = 1
    
    if args.sft_only:
        dataset = TensorDataset(grads_sft_shuffled, labels)
    else:
        dataset = TensorDataset(grads_kn_loss_shuffled, labels)
    trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    valloader = trainloader  # unused in convex branch

    # 4. Dummy model & loss
    dummy_model = DummyModel(args.proj_dim)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    # 5. Pre-compute batch means
    if args.selection_type == "PerBatch":
        # Original logic: take mean of each mini-batch
        batch_size = args.batch_size
        if args.sft_only:
            sft_means = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                sft_means.append(grads_sft_shuffled[start:end].mean(dim=0))
            sft_means = torch.stack(sft_means).to(args.device)
        else:
            kn_loss_means, if_loss_means = [], []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                kn_loss_means.append(grads_kn_loss_shuffled[start:end].mean(dim=0))
                if_loss_means.append(grads_if_loss_shuffled[start:end].mean(dim=0))
            kn_loss_means = torch.stack(kn_loss_means).to(args.device)
            if_loss_means = torch.stack(if_loss_means).to(args.device)
    else:  # PerClass
        batch_size = 1  # Each sample treated as a "point"
        if args.sft_only:
            sft_means = grads_sft_shuffled.to(args.device)
        else:
            kn_loss_means = grads_kn_loss_shuffled.to(args.device)
            if_loss_means = grads_if_loss_shuffled.to(args.device)

    # Handle random selection mode
    if args.random:
        logger.info("Using random selection baseline")
        
        # Execute random selection
        random_results = random_selection_baseline(N, args.proportions, args.random_ratio, args.seed)
        
        # Process results for each proportion
        for p in args.proportions:
            idxs = random_results[p]
            
            # Save indices
            out_path = os.path.join(
                args.output_dir,
                f"{args.prefix}_random_r{int(args.random_ratio*100)}_p{int(p*100)}_idx.npz",
            )
            np.savez(out_path, subset=idxs, weights=np.ones(len(idxs)))
            logger.info("Saved %d random indices -> %s", len(idxs), out_path)
            
            # Calculate errors - compute kn_loss and if_loss errors over entire space
            if args.sft_only:
                # Calculate errors on kn_loss and if_loss
                error_kn_loss = compute_coreset_error(
                    grads_kn_loss, idxs, device=args.device, chunk_size=32768
                )
                error_if_loss = compute_coreset_error(
                    grads_if_loss, idxs, device=args.device, chunk_size=32768
                )
            else:
                # Calculate errors on kn_loss and if_loss (parallel or sequential)
                if args.parallel:
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        future_kn_loss = executor.submit(
                            compute_error_wrapper, grads_kn_loss, idxs, "cuda:0", 32768, "kn_loss"
                        )
                        future_if_loss = executor.submit(
                            compute_error_wrapper, grads_if_loss, idxs, "cuda:1", 32768, "if_loss"
                        )
                        
                        results_error = {}
                        for future in as_completed([future_kn_loss, future_if_loss]):
                            error_type, error_value = future.result()
                            results_error[error_type] = error_value
                        
                        error_kn_loss = results_error["kn_loss"]
                        error_if_loss = results_error["if_loss"]
                else:
                    error_kn_loss = compute_coreset_error(
                        grads_kn_loss, idxs, device="cpu", chunk_size=32768
                    )
                    error_if_loss = compute_coreset_error(
                        grads_if_loss, idxs, device="cpu", chunk_size=32768
                    )
            
            logger.info(
                "Random baseline p=%.2f%%, r=%.2f%% | error_kn_loss=%.6f | error_if_loss=%.6f", 
                p * 100, args.random_ratio * 100, error_kn_loss, error_if_loss
            )
            
            # Save errors to CSV
            csv_path = os.path.join(
                args.output_dir, f"{args.prefix}_random_r{int(args.random_ratio*100)}_error_p{int(p*100)}.csv"
            )
            with open(csv_path, "w") as f:
                f.write("proportion,random_ratio,error_kn_loss,error_if_loss\n")
                f.write(f"{p:.2f},{args.random_ratio:.2f},{error_kn_loss:.6f},{error_if_loss:.6f}\n")
            logger.info("Saved random baseline error to %s", csv_path)
        
        logger.info("Random selection baseline completed.")
        return
    
    # Create corresponding BRIEF strategy
    if args.sft_only:
        brief = SingleGradBRIEFStrategy(
            grad_means=sft_means,
            shuffle_indices=shuffle_indices,
            batch_size=batch_size,
            trainloader=trainloader,
            valloader=valloader,
            model=dummy_model,
            loss=loss_fn,
            device=args.device,
            num_classes=num_classes,
            linear_layer=False,
            if_convex=True,
            selection_type=args.selection_type,
            logger=logger,
            optimizer="lazy",
        )
    else:
        brief = MultiGradBRIEFStrategy(
            kn_loss_means=kn_loss_means,
            if_loss_means=if_loss_means,
            alpha=args.alpha,
            shuffle_indices=shuffle_indices,
            batch_size=batch_size,
            trainloader=trainloader,
            valloader=valloader,
            model=dummy_model,
            loss=loss_fn,
            device=args.device,
            num_classes=num_classes,
            linear_layer=False,
            if_convex=True,
            selection_type=args.selection_type,
            logger=logger,
            optimizer="lazy",
        )
    
    model_params = dummy_model.state_dict()

    if args.plot:
        plt = import_matplotlib()

    # ============ Main optimization: Use select_multi_budget for PerClass ============ #
    if args.selection_type == "PerClass":
        logger.info("Using optimized select_multi_budget for PerClass selection")
        
        if args.sft_only:
            # Single gradient mode: no alpha sweep
            logger.info("Grad-only mode: Selecting multiple budgets simultaneously")
            
            # Sort proportions in descending order (required by select_multi_budget)
            sorted_proportions = sorted(args.proportions, reverse=True)
            
            # Single call to select_multi_budget
            results = brief.select_multi_budget(sorted_proportions, model_params)
            
            # Process results for each proportion
            for p in args.proportions:
                idxs, gammas = results[p]
                idxs = np.asarray(idxs, dtype=np.int64)
                
                # Save indices
                out_path = os.path.join(
                    args.output_dir,
                    f"{args.prefix}_sft_only_p{int(p*100)}_idx.npz",
                )
                np.savez(out_path, subset=idxs, weights=gammas)
                logger.info("Saved %d indices -> %s", len(idxs), out_path)
                
                # Compute error
                error_if_loss = compute_coreset_error(
                    grads_if_loss, idxs, device=args.device, chunk_size=32768
                )
                error_kn_loss = compute_coreset_error(
                    grads_kn_loss, idxs, device=args.device, chunk_size=32768
                )
                logger.info("p=%.2f%% | error_kn_loss=%.6f | error_if_loss=%.6f", p * 100, error_kn_loss, error_if_loss)
                
                # Save error to file
                csv_path = os.path.join(
                    args.output_dir, f"{args.prefix}_sft_only_error_p{int(p*100)}.csv"
                )
                with open(csv_path, "w") as f:
                    f.write("proportion,error_kn_loss,error_if_loss\n")
                    f.write(f"{p:.2f},{error_kn_loss:.6f},{error_if_loss:.6f}\n")
                logger.info("Saved error to %s", csv_path)
                
        else:
            # Multi-gradient mode 
            if args.auto_search:
                # Automatic search for optimal alpha mode
                logger.info("Starting automatic search for optimal alpha")
                
                # Process each proportion
                for p in args.proportions:
                    logger.info("Searching optimal alpha for proportion %.2f%%", p * 100)
                    
                    # Execute golden section search
                    optimal_alpha, min_error_sum, search_history = golden_section_search_alpha(
                        brief=brief,
                        model_params=model_params,
                        grads_kn_loss=grads_kn_loss,
                        grads_if_loss=grads_if_loss,
                        proportion=p,
                        precision=args.search_precision,
                        max_iterations=args.max_iterations,
                        device=args.device
                    )
                    
                    # Use optimal alpha for final selection
                    brief.reset()
                    brief.alpha = optimal_alpha
                    idxs, gammas = brief.select(int(len(grads_kn_loss) * p), model_params)
                    idxs = np.asarray(idxs, dtype=np.int64)
                    
                    # Save indices of optimal result
                    out_path = os.path.join(
                        args.output_dir,
                        f"{args.prefix}_auto_alpha{int(optimal_alpha*1000)}_p{int(p*100)}_idx.npz",
                    )
                    np.savez(out_path, subset=idxs, weights=gammas)
                    logger.info("Saved optimal selection (%d indices) -> %s", len(idxs), out_path)
                    
                    # Calculate final errors
                    error_kn_loss = compute_coreset_error(grads_kn_loss, idxs, device=args.device, chunk_size=32768)
                    error_if_loss = compute_coreset_error(grads_if_loss, idxs, device=args.device, chunk_size=32768)
                    error_sum = error_kn_loss + error_if_loss
                    
                    logger.info("Optimal result: p=%.2f%%, α*=%.4f | error_kn_loss=%.6f | error_if_loss=%.6f | sum=%.6f", 
                               p * 100, optimal_alpha, error_kn_loss, error_if_loss, error_sum)
                    
                    # Save search history and results
                    search_csv_path = os.path.join(
                        args.output_dir, f"{args.prefix}_auto_search_p{int(p*100)}.csv"
                    )
                    with open(search_csv_path, "w") as f:
                        f.write("alpha,error_sum,error_kn_loss,error_if_loss\n")
                        for alpha_val, err_sum, err_kn_loss, err_if_loss in search_history:
                            f.write(f"{alpha_val:.6f},{err_sum:.6f},{err_kn_loss:.6f},{err_if_loss:.6f}\n")
                    
                    # Save optimal results
                    result_csv_path = os.path.join(
                        args.output_dir, f"{args.prefix}_auto_optimal_p{int(p*100)}.csv"
                    )
                    with open(result_csv_path, "w") as f:
                        f.write("proportion,optimal_alpha,min_error_sum,error_kn_loss,error_if_loss\n")
                        f.write(f"{p:.2f},{optimal_alpha:.6f},{error_sum:.6f},{error_kn_loss:.6f},{error_if_loss:.6f}\n")
                    
                    logger.info("Saved search history to %s", search_csv_path)
                    logger.info("Saved optimal result to %s", result_csv_path)
                    
                    # If plotting is enabled, create search process plot
                    if args.plot:
                        plt = import_matplotlib()
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        alphas = [h[0] for h in search_history]
                        error_sums = [h[1] for h in search_history]
                        error_kn_losss = [h[2] for h in search_history]  
                        error_if_losss = [h[3] for h in search_history]
                        
                        ax.scatter(alphas, error_sums, c='red', s=50, alpha=0.7, label='error_sum (search points)')
                        ax.scatter(alphas, error_kn_losss, c='blue', s=30, alpha=0.7, label='error_kn_loss')
                        ax.scatter(alphas, error_if_losss, c='green', s=30, alpha=0.7, label='error_if_loss')
                        
                        # Mark optimal point
                        ax.scatter(optimal_alpha, min_error_sum, c='red', s=100, marker='*', 
                                  label=f'Optimal α={optimal_alpha:.4f}')
                        
                        ax.set_xlabel("Alpha")
                        ax.set_ylabel("Error")
                        ax.set_title(f"Alpha Search Process (p={p*100:.0f}%)")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        plot_path = os.path.join(args.output_dir, 
                                               f"{args.prefix}_auto_search_p{int(p*100)}.png")
                        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        logger.info("Saved search plot -> %s", plot_path)
                
                logger.info("Auto search completed for all proportions")
                return
            
            else:
                # Original alpha sweep mode
                alpha_list = [0,1]
            
            # Sort proportions in descending order
            sorted_proportions = sorted(args.proportions, reverse=True)
            
            # Store all results for plotting
            all_results = {p: {'alphas': [], 'errs_kn_loss': [], 'errs_if_loss': [], 
                               'errs_sum': [], 'errs_weighted': []} for p in args.proportions}
            
            for alpha in alpha_list:
                brief.reset()
                brief.alpha = alpha
                
                logger.info("Selecting multiple budgets with α=%.3f", alpha)
                
                # Single call to select_multi_budget for all proportions
                results = brief.select_multi_budget(sorted_proportions, model_params)
                
                # Process results for each proportion
                for p in args.proportions:
                    idxs, gammas = results[p]
                    idxs = np.asarray(idxs, dtype=np.int64)
                    
                    # Save indices
                    out_path = os.path.join(
                        args.output_dir,
                        f"{args.prefix}_alpha{int(alpha*1000)}_p{int(p*100)}_idx.npz",
                    )
                    np.savez(out_path, subset=idxs, weights=gammas)
                    logger.info("Saved %d indices -> %s", len(idxs), out_path)
                    
                    # Compute errors (parallel or sequential)
                    if args.parallel:
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            future_kn_loss = executor.submit(
                                compute_error_wrapper, grads_kn_loss, idxs, "cuda:0", 32768, "kn_loss"
                            )
                            future_if_loss = executor.submit(
                                compute_error_wrapper, grads_if_loss, idxs, "cuda:1", 32768, "if_loss"
                            )
                            
                            results_error = {}
                            for future in as_completed([future_kn_loss, future_if_loss]):
                                error_type, error_value = future.result()
                                results_error[error_type] = error_value
                            
                            error_kn_loss = results_error["kn_loss"]
                            error_if_loss = results_error["if_loss"]
                    else:
                        error_kn_loss = compute_coreset_error(
                            grads_kn_loss, idxs, device="cpu", chunk_size=32768
                        )
                        error_if_loss = compute_coreset_error(
                            grads_if_loss, idxs, device="cpu", chunk_size=32768
                        )
                    
                    logger.info(
                        "p=%.2f%%, α=%.3f | error_kn_loss=%.6f | error_if_loss=%.6f", 
                        p * 100, alpha, error_kn_loss, error_if_loss
                    )
                    
                    # Store for plotting and CSV
                    all_results[p]['alphas'].append(alpha)
                    all_results[p]['errs_kn_loss'].append(error_kn_loss)
                    all_results[p]['errs_if_loss'].append(error_if_loss)
                    all_results[p]['errs_sum'].append(error_kn_loss + error_if_loss)
                    all_results[p]['errs_weighted'].append(alpha * error_kn_loss + (1 - alpha) * error_if_loss)
            
            # Save CSV and plot for each proportion
            for p in args.proportions:
                res = all_results[p]
                
                # Save CSV
                csv_path = os.path.join(
                    args.output_dir, f"{args.prefix}_errors_p{int(p*100)}.csv"
                )
                with open(csv_path, "w") as f:
                    f.write("alpha,error_kn_loss,error_if_loss,error_sum,error_weighted\n")
                    for i in range(len(res['alphas'])):
                        f.write(f"{res['alphas'][i]:.3f},{res['errs_kn_loss'][i]:.6f},"
                               f"{res['errs_if_loss'][i]:.6f},{res['errs_sum'][i]:.6f},"
                               f"{res['errs_weighted'][i]:.6f}\n")
                logger.info("Saved errors to %s", csv_path)
                
                # Plot if requested
                if args.plot:
                    # FIGURE 1: error_kn_loss vs error_if_loss
                    fig1, ax1 = plt.subplots(figsize=(6, 5))
                    ax1.plot(res['errs_if_loss'], res['errs_kn_loss'], marker="o", linestyle="-")
                    for x, y, a in zip(res['errs_if_loss'], res['errs_kn_loss'], res['alphas']):
                        ax1.annotate(f"{a:.3f}", (x, y), textcoords="offset points", xytext=(3, 3))
                    ax1.set_xlabel("error_if_loss")
                    ax1.set_ylabel("error_kn_loss")
                    ax1.set_title(f"Subset {p*100:.0f}% – error_kn_loss vs error_if_loss")
                    ax1.grid(True)
                    fig1.tight_layout()
                    path1 = os.path.join(args.output_dir, f"{args.prefix}_plot1_p{int(p*100)}.png")
                    fig1.savefig(path1, dpi=300)
                    plt.close(fig1)
                    logger.info("Saved plot -> %s", path1)

                    # FIGURE 2: α curves
                    fig2, ax2 = plt.subplots(figsize=(7, 5))
                    ax2.plot(res['alphas'], res['errs_kn_loss'], marker="o", label="error_kn_loss")
                    ax2.plot(res['alphas'], res['errs_if_loss'], marker="s", label="error_if_loss")
                    ax2.set_xlabel("alpha")
                    ax2.set_ylabel("error")
                    ax2.grid(True)
                    ax2.legend(loc="upper left")

                    ax3 = ax2.twinx()
                    ax3.plot(res['alphas'], res['errs_sum'], marker="^", linestyle="--", 
                            label="error_kn_loss + error_if_loss")
                    ax3.plot(
                        res['alphas'],
                        res['errs_weighted'],
                        marker="v",
                        linestyle="--",
                        label="alpha*error_kn_loss + (1-alpha)*error_if_loss",
                    )
                    ax3.set_ylabel("combined error")

                    # combine legends
                    lines, labels = ax2.get_legend_handles_labels()
                    lines2, labels2 = ax3.get_legend_handles_labels()
                    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

                    ax2.set_title(f"Subset {p*100:.0f}% – error curves vs alpha")
                    fig2.tight_layout()
                    path2 = os.path.join(args.output_dir, f"{args.prefix}_plot2_p{int(p*100)}.png")
                    fig2.savefig(path2, dpi=300)
                    plt.close(fig2)
                    logger.info("Saved plot -> %s", path2)
    else:
        raise ValueError("Unsupported selection type: %s" % args.selection_type)
    if args.plot and not args.sft_only:
        import matplotlib.cm as cm
        
        # Total plot: error_kn_loss vs error_if_loss for different budgets
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        colors = cm.tab10(np.arange(len(all_results)) / len(all_results))
        for i, p in enumerate(sorted(all_results.keys())):
            res = all_results[p]
            ax3.plot(res['errs_if_loss'], res['errs_kn_loss'], 'o-', color=colors[i], label=f"budget {p*100:.0f}%")
            for j in range(len(res['alphas'])):
                ax3.annotate(f"{res['alphas'][j]:.1f}", (res['errs_if_loss'][j], res['errs_kn_loss'][j]), xytext=(5,5), textcoords="offset points")
        ax3.set_xlabel("error_if_loss")
        ax3.set_ylabel("error_kn_loss")
        ax3.set_title("error_kn_loss vs error_if_loss across different budgets")
        ax3.legend()
        ax3.grid(True)
        fig3.tight_layout()
        path3 = os.path.join(args.output_dir, f"{args.prefix}_summary_kn_loss_if_loss.png")
        fig3.savefig(path3, dpi=300)
        plt.close(fig3)
        logger.info("Saved summary plot -> %s", path3)
        
        # Other summary plots
        # Error_kn_loss vs proportion for different alphas
        alpha_list = all_results[list(all_results.keys())[0]]['alphas']  # Assume same alphas across p
        sorted_ps = sorted(all_results.keys())
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        for j, alpha in enumerate(alpha_list):
            errs_kn_loss = [all_results[p]['errs_kn_loss'][j] for p in sorted_ps]
            ax4.plot(sorted_ps, errs_kn_loss, 'o-', label=f"alpha={alpha:.3f}")
        ax4.set_xlabel("proportion")
        ax4.set_ylabel("error_kn_loss")
        ax4.set_title("kn_loss vs Budget for different alphas")
        ax4.legend()
        ax4.grid(True)
        fig4.tight_layout()
        path4 = os.path.join(args.output_dir, f"{args.prefix}_error_kn_loss_vs_budget.png")
        fig4.savefig(path4, dpi=300)
        plt.close(fig4)
        logger.info("Saved %s", path4)
        
        # Error_if_loss vs proportion for different alphas
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        for j, alpha in enumerate(alpha_list):
            errs_if_loss = [all_results[p]['errs_if_loss'][j] for p in sorted_ps]
            ax5.plot(sorted_ps, errs_if_loss, 'o-', label=f"alpha={alpha:.3f}")
        ax5.set_xlabel("proportion")
        ax5.set_ylabel("error_if_loss")
        ax5.set_title("Error if_loss vs Budget for different alphas")
        ax5.legend()
        ax5.grid(True)
        fig5.tight_layout()
        path5 = os.path.join(args.output_dir, f"{args.prefix}_error_if_loss_vs_budget.png")
        fig5.savefig(path5, dpi=300)
        plt.close(fig5)
        logger.info("Saved %s", path5)
        
        # Combined error (min/mean sum) vs proportion
        fig6, ax6 = plt.subplots(figsize=(7, 5))
        min_sums = [min(all_results[p]['errs_sum']) for p in sorted_ps]
        mean_sums = [sum(all_results[p]['errs_sum']) / len(all_results[p]['errs_sum']) for p in sorted_ps]
        ax6.plot(sorted_ps, min_sums, 'o-', label="min (error_kn_loss + error_if_loss) over alpha")
        ax6.plot(sorted_ps, mean_sums, 'o-', label="mean (error_kn_loss + error_if_loss) over alpha")
        ax6.set_xlabel("proportion")
        ax6.set_ylabel("combined error")
        ax6.set_title("Combined Error vs Budget")
        ax6.legend()
        ax6.grid(True)
        fig6.tight_layout()
        path6 = os.path.join(args.output_dir, f"{args.prefix}_combined_error_vs_budget.png")
        fig6.savefig(path6, dpi=300)
        plt.close(fig6)
        logger.info("Saved %s", path6)
    logger.info("BRIEF coreset selection completed.")


if __name__ == "__main__":
    main()