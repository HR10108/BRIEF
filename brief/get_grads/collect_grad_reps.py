import json
import os
from hashlib import md5
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from peft import PeftModel
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import RobertaModel
import os
import json
import itertools
from typing import List, Optional, Dict

import torch
from tqdm import tqdm


def prepare_batch(batch, device=torch.device("cuda:0")):
    """Move the batch to the device."""
    for key in batch:
        batch[key] = batch[key].to(device)


def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """
    Retrieve the highest index for which the data (either representation or gradients) has been stored.

    Args:
        output_dir (str): The output directory.
        prefix (str, optional): The prefix of the files, [reps | grads]. Defaults to "reps".

    Returns:
        int: The maximum representation index, or -1 if no index is found.
    """

    files = [file for file in os.listdir(output_dir) if file.startswith(prefix)]
    index = [
        int(file.split(".")[0].split("-")[1]) for file in files
    ]  # e.g., output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1


def get_output(
    model,
    weights: Iterable[Tensor],
    buffers: Iterable[Tensor],
    input_ids=None,
    attention_mask=None,
    labels=None,
) -> Tensor:
    logits = model(
        weights, buffers, *(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
    ).logits
    labels = labels.unsqueeze(0)
    loss_fct = F.cross_entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)
    )
    return loss


def get_trak_projector(device: torch.device):
    """Get trak projectors (see https://github.com/MadryLab/trak for details)"""
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(
            torch.zeros(8, 1_000, device=device), 512, 0, num_sms
        )
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def get_number_of_params(model):
    """Make sure that only lora parameters require gradients in peft models."""
    if isinstance(model, PeftModel):
        names = [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and "lora" not in n
        ]
        assert len(names) == 0
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params


def obtain_gradients(model, batch):
    """obtain gradients."""
    loss = model(**batch).loss
    loss.backward()
    vectorized_grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    )
    return vectorized_grads, loss.item()


def obtain_sign_gradients(model, batch):
    """obtain gradients with sign."""
    loss = model(**batch).loss
    loss.backward()

    # Instead of concatenating the gradients, concatenate their signs
    vectorized_grad_signs = torch.cat(
        [torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None]
    )

    return vectorized_grad_signs, loss.item()


def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    """obtain gradients with adam optimizer states."""
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    loss = model(**batch).loss
    loss.backward()

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None]
    )

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads**2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads, loss.item()


def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in range(len(names))])
    avg_sq = torch.cat(
        [optimizer_state[n]["exp_avg_sq"].view(-1) for n in range(len(names))]
    )
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


def collect_grads(
    dataloader,
    model,
    output_dir,
    proj_dim: List[int] = [8192],
    adam_optimizer_state: Optional[dict] = None,
    gradient_type: str = "adam",
    max_samples: Optional[int] = None,
):
    """
    Collects gradients from the model during evaluation and saves them to disk.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        output_dir (str): The directory where the gradients will be saved.
        proj_dim List[int]: The dimensions of the target projectors. Each dimension will be saved in a separate folder.
        gradient_type (str): The type of gradients to collect. [adam | sign | sgd]
        adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states.
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 16  # batch size for the projectors
    torch.random.manual_seed(0)  # set the random seed for torch

    project_interval = 16  # project every 16 batches
    save_interval = 160  # save every 160 batches

    def _project(current_full_grads, projected_grads):
        current_full_grads = torch.stack(current_full_grads).to(torch.float16)
        for i, projector in enumerate(projectors):
            current_projected_grads = projector.project(
                current_full_grads, model_id=model_id
            )
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

    def _save(projected_grads, output_dirs):
        for dim in proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])

            output_dir = output_dirs[dim]
            outfile = os.path.join(output_dir, f"grads-{count}.pt")
            torch.save(projected_grads[dim], outfile)
            print(f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
            projected_grads[dim] = []

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model)

    # never made it work sadly
    # fmodel, params, buffers = make_functional_with_buffers(model)
    # grads_loss = torch.func.grad(get_output, has_aux=False, argnums=1)

    # initialize a project for each target projector dimension
    projectors = []
    for dim in proj_dim:
        proj = projector(
            grad_dim=number_of_params,
            proj_dim=dim,
            seed=0,
            proj_type=ProjectionType.rademacher,
            device=device,
            dtype=dtype,
            block_size=block_size,
            max_batch_size=projector_batch_size,
        )
        projectors.append(proj)

    count = 0

    # set up a output directory for each dimension
    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)

    # max index for each dimension
    max_index = min(get_max_saved_index(output_dirs[dim], "grads") for dim in proj_dim)

    # projected_gradients
    full_grads = []  # full gradients
    projected_grads = {dim: [] for dim in proj_dim}  # projected gradients

    for batch in tqdm(dataloader, total=len(dataloader)):
        prepare_batch(batch)
        count += 1

        if count <= max_index:
            print("skipping count", count)
            continue

        if gradient_type == "adam":
            if count == 1:
                print("Using Adam gradients")
            vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
        elif gradient_type == "sign":
            if count == 1:
                print("Using Sign gradients")
            vectorized_grads = obtain_sign_gradients(model, batch)
        else:
            if count == 1:
                print("Using SGD gradients")
            vectorized_grads = obtain_gradients(model, batch)

        # add the gradients to the full_grads
        full_grads.append(vectorized_grads)
        model.zero_grad()

        if count % project_interval == 0:
            _project(full_grads, projected_grads)
            full_grads = []

        if count % save_interval == 0:
            _save(projected_grads, output_dirs)

        if max_samples is not None and count == max_samples:
            break

    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []

    for dim in proj_dim:
        _save(projected_grads, output_dirs)

    torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir = output_dirs[dim]
        merge_and_normalize_info(output_dir, prefix="grads")
        merge_info(output_dir, prefix="grads")

    print("Finished")


def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """Merge and normalize the representations and gradients into a single file."""
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}."
    )


def merge_info(output_dir: str, prefix="reps"):
    """Merge the representations and gradients into a single file without normalization."""
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        merged_data.append(data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}."
    )


def collect_reps(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    output_dir: str,
    max_samples: Optional[int] = None,
):
    """
    Collects representations from a dataloader using a given model and saves them to the output directory.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.
        model (torch.nn.Module): The model used to compute the representations.
        output_dir (str): The directory where the representations will be saved.
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    all_reps = []
    count = 0
    save_interval = 160  # save every 160 batches

    device = next(model.parameters()).device  # only works for single gpu
    max_index = get_max_saved_index(output_dir, prefix="reps")

    for batch in tqdm(dataloader):
        count += 1
        if count <= max_index:
            print("skipping count", count)
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            if isinstance(model, RobertaModel):
                reps = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                ).pooler_output
            else:
                hidden_states = model(
                    input_ids,
                    labels=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                ).hidden_states
                ids = torch.arange(len(input_ids), device=input_ids.device)
                pos = attention_mask.sum(dim=1) - 1
                reps = hidden_states[-1][ids, pos]

            all_reps.append(reps.cpu())
            if count % save_interval == 0:
                all_reps = torch.cat(all_reps)
                outfile = os.path.join(output_dir, f"reps-{count}.pt")
                torch.save(all_reps, outfile)
                all_reps = []
                print(f"Saving {outfile}")

            if max_samples is not None and count >= max_samples:
                break

    if len(all_reps) > 0:
        all_reps = torch.cat(all_reps)
        outfile = os.path.join(output_dir, f"reps-{count}.pt")
        torch.save(all_reps, outfile)
        print(f"Saving {outfile}")

    torch.cuda.empty_cache()
    merge_and_normalize_info(output_dir, prefix="reps")

    print("Finished")


def get_loss(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    output_dir: str,
):
    """Get the loss of the model on the given dataset."""
    total_loss = 0
    total_tokens = 0
    for batch in tqdm(dataloader):
        prepare_batch(batch)
        num_token = (batch["labels"] != -100).sum()
        with torch.inference_mode():
            loss = model(**batch).loss * num_token
        total_loss += loss.item()
        total_tokens += num_token.item()

    print(f"Loss: {total_loss / total_tokens}")
    result = {"num_tokens": total_tokens, "loss": (total_loss / total_tokens)}
    with open(os.path.join(output_dir, "loss.txt"), "w") as f:
        f.write(json.dumps(result, indent=4))


def get_knif(
    dataloader,
    model,
    output_dir,
    proj_dim: List[int] = [8192],
    *,
    gradient_type: str = "adam",
    adam_optimizer_state: Optional[dict] = None,
    max_samples: Optional[int] = None,
    resume: bool = True,
):

    # ---------------- 0) constants & bookkeeping ----------------
    model_id = 0
    block_size = 128
    projector_batch_size = 16
    project_interval = 16
    save_interval = 160

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    torch.random.manual_seed(0)

    # Adam state scaffolding --------------------------------------------------
    if gradient_type == "adam":
        assert adam_optimizer_state is not None, "Adam state is required"
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    # ---------------- 1) read (or init) checkpoint ---------------------------
    ckpt_path = os.path.join(output_dir, "progress.json")
    if resume and os.path.isfile(ckpt_path):
        with open(ckpt_path, "r") as fp:
            ckpt: Dict = json.load(fp)
        start_step: int = ckpt.get("count", 0)
        loss_sum: Dict[str, float] = ckpt.get(
            "loss_sum", {"sft": 0.0, "kn_loss": 0.0, "if_loss": 0.0}
        )
        tok_sum: Dict[str, int] = ckpt.get("tok_sum", {"sft": 0, "kn_loss": 0, "if_loss": 0})
        print(f"[RESUME] Loaded checkpoint from step {start_step}")
    else:
        start_step = 0
        loss_sum = {"sft": 0.0, "kn_loss": 0.0, "if_loss": 0.0}
        tok_sum = {"sft": 0, "kn_loss": 0, "if_loss": 0}

    # ---------------- 2) build projectors & dirs -----------------------------
    number_of_params = get_number_of_params(model)
    projector_cls = get_trak_projector(device)

    projectors = {kind: [] for kind in ["sft", "kn_loss", "if_loss"]}
    for dim in proj_dim:
        for kind in projectors:
            proj = projector_cls(
                grad_dim=number_of_params,
                proj_dim=dim,
                seed=0,
                proj_type=ProjectionType.rademacher,
                device=device,
                dtype=dtype,
                block_size=block_size,
                max_batch_size=projector_batch_size,
            )
            projectors[kind].append(proj)

    output_dirs = {
        kind: {dim: os.path.join(output_dir, kind, f"dim{dim}") for dim in proj_dim}
        for kind in ["sft", "kn_loss", "if_loss"]
    }
    for kind in output_dirs:
        for dim_dir in output_dirs[kind].values():
            os.makedirs(dim_dir, exist_ok=True)

    # ---------------- 3) internal helpers ------------------------------------
    def _project(kind, current_full_grads, projected):
        stacked = torch.stack(current_full_grads).to(torch.float16)
        for i, proj in enumerate(projectors[kind]):
            cur_proj = proj.project(stacked, model_id=model_id)
            projected[kind][proj_dim[i]].append(cur_proj.cpu())

    def _flush(kind, projected, step):
        for dim in proj_dim:
            if not projected[kind][dim]:
                continue
            data = torch.cat(projected[kind][dim])
            outfile = os.path.join(output_dirs[kind][dim], f"grads-{step}.pt")
            torch.save(data, outfile)
            print(f"[{kind}] Saved {outfile} {tuple(data.shape)}", flush=True)
            projected[kind][dim].clear()

    def _save_ckpt(step):
        ckpt_data = {
            "count": step,
            "loss_sum": loss_sum,
            "tok_sum": tok_sum,
        }
        tmp_path = ckpt_path + ".tmp"
        with open(tmp_path, "w") as fp:
            json.dump(ckpt_data, fp)
        os.replace(tmp_path, ckpt_path)  # atomic

    # ---------------- 4) main loop -------------------------------------------
    full_grads = {k: [] for k in ["sft", "kn_loss", "if_loss"]}
    projected_gr = {k: {d: [] for d in proj_dim} for k in ["sft", "kn_loss", "if_loss"]}

    # Create iterator that skips already‑processed samples
    remaining_total = None
    if hasattr(dataloader, "__len__"):
        remaining_total = max(len(dataloader) - start_step, 0)
    dataloader_iter = itertools.islice(dataloader, start_step, None)

    count = start_step
    for batch in tqdm(dataloader_iter, total=remaining_total, initial=start_step):
        if max_samples is not None and count >= max_samples:
            break

        # prepare_batch(batch)
        count += 1

        # ---- (a) SFT grads ---------------------------------------------------
        sft_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        prepare_batch(sft_batch)
        if gradient_type == "adam":
            g_sft, loss_sft = obtain_gradients_with_adam(model, sft_batch, m, v)
        elif gradient_type == "sign":
            g_sft, loss_sft = obtain_sign_gradients(model, sft_batch)
        else:
            g_sft, loss_sft = obtain_gradients(model, sft_batch)
        full_grads["sft"].append(g_sft)
        sft_tokens = (sft_batch["labels"] != -100).sum().item()
        loss_sum["sft"] += loss_sft * sft_tokens
        tok_sum["sft"] += sft_tokens
        model.zero_grad()

        # ---- (b) kn_loss grads ----------------------------------------------------
        kn_loss_batch = {
            "input_ids": batch["y_input_ids"],
            "attention_mask": batch["y_attention_mask"],
            "labels": batch["y_labels"],
        }
        prepare_batch(kn_loss_batch)

        if gradient_type == "adam":
            g_kn_loss, loss_kn_loss = obtain_gradients_with_adam(model, kn_loss_batch, m, v)
        elif gradient_type == "sign":
            g_kn_loss, loss_kn_loss = obtain_sign_gradients(model, kn_loss_batch)
        else:
            g_kn_loss, loss_kn_loss = obtain_gradients(model, kn_loss_batch)
        full_grads["kn_loss"].append(g_kn_loss)
        kn_loss_tokens = (kn_loss_batch["labels"] != -100).sum().item()
        loss_sum["kn_loss"] += loss_kn_loss * kn_loss_tokens
        tok_sum["kn_loss"] += kn_loss_tokens
        model.zero_grad()

        # ---- (c) if_loss = SFT – kn_loss ---------------------------------------------
        g_if_loss = g_sft - g_kn_loss
        full_grads["if_loss"].append(g_if_loss)
        loss_if_loss = loss_sft - loss_kn_loss
        loss_sum["if_loss"] += loss_if_loss * sft_tokens
        tok_sum["if_loss"] += sft_tokens

        # ---------- projection & saving --------------------------------------
        if count % project_interval == 0:
            for kind in ["sft", "kn_loss", "if_loss"]:
                _project(kind, full_grads[kind], projected_gr)
                full_grads[kind].clear()

        if count % save_interval == 0:
            for kind in ["sft", "kn_loss", "if_loss"]:
                _flush(kind, projected_gr, count)
            _save_ckpt(count)

    # ---------------- 5) drain remaining work -------------------------------
    for kind in ["sft", "kn_loss", "if_loss"]:
        if full_grads[kind]:
            _project(kind, full_grads[kind], projected_gr)
        _flush(kind, projected_gr, count)

    _save_ckpt(count)  # final checkpoint – marks completion
    torch.cuda.empty_cache()

    # ---------------- 6) merge & write metrics ------------------------------
    for kind in ["sft", "kn_loss", "if_loss"]:
        for dim in proj_dim:
            dim_dir = output_dirs[kind][dim]
            merge_and_normalize_info(dim_dir, prefix="grads")
            merge_info(dim_dir, prefix="grads")

    for kind in ["sft", "kn_loss", "if_loss"]:
        avg_loss = loss_sum[kind] / max(tok_sum[kind], 1)
        info = {"num_tokens": tok_sum[kind], "loss": avg_loss}
        out_path = os.path.join(output_dir, f"{kind}_loss.txt")
        with open(out_path, "w") as fp:
            json.dump(info, fp, indent=4)
        print(f"[{kind}] avg loss = {avg_loss:.6f}  → {out_path}")

    print("Finished KN&IF gradient collection ✅")
