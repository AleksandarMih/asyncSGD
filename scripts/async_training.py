"""
async_training.py — library for Hogwild!-style asynchronous SGD training runs.

Spawn strategy: spawn-per-epoch.
Each global epoch spawns N worker processes, all workers join, then the main
process runs update_bn + evaluate. Chosen over spawn-once-with-barriers because:
  - Error propagation is trivial (check p.exitcode after each p.join()).
  - No BrokenBarrierError edge cases if a worker dies.
  - Spawn overhead (~0.2–0.5 s per process) is <3 % of total CPU training time.

Limitation: momentum buffers are re-initialised each epoch (workers are freshly
spawned). With MOMENTUM=0.0 (the Phase 1 default) this has no effect.

Reference for multiprocessing mechanics: scripts/train_par_baseline.py (not imported).

Import train_one_run_async and evaluate from sweep scripts; do not run directly.
"""

import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import _train_transform, _test_transform, get_loaders
from src.model import resnet20
from seq_training import evaluate  # no circular dep: seq_training never imports async_training

# ---------------------------------------------------------------------------
# Default hyperparameters — import and optionally override in sweep scripts
# ---------------------------------------------------------------------------

EPOCHS = 100
LR = 0.1
BATCH_SIZE = 128
MOMENTUM = 0.0
WEIGHT_DECAY = 1e-4
DROPOUT = 0.0
SEEDS = [42, 123, 456]
DATA_ROOT = "data"


# ---------------------------------------------------------------------------
# Internal dataset helpers (mirrors train_par_baseline.py:111-138)
# ---------------------------------------------------------------------------

def _get_train_subset(rank: int, num_workers: int, data_root: str) -> torch.utils.data.Subset:
    """Contiguous shard of CIFAR-10 training set for worker `rank`."""
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=_train_transform,
    )
    n = len(dataset)
    per_worker = n // num_workers
    start = rank * per_worker
    end = start + per_worker if rank < num_workers - 1 else n
    return torch.utils.data.Subset(dataset, list(range(start, end)))


def _get_bn_train_loader(data_root: str) -> torch.utils.data.DataLoader:
    """Full training set loader for update_bn (shuffle=False, num_workers=0)."""
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=_train_transform,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Worker function — must be a module-level def for pickling under spawn
# ---------------------------------------------------------------------------

def _worker(
    rank: int,
    model: nn.Module,
    batch_size: int,
    num_workers: int,
    momentum: float,
    weight_decay: float,
    lr: float,
    data_root: str,
    seed: int,
    epoch_losses: mp.Array,   # "d" array, length num_workers — loss sum per worker
    epoch_counts: mp.Array,   # "l" array, length num_workers — sample count per worker
    counter: mp.Value,        # "i" shared batch counter for logging
    lock: mp.Lock,
) -> None:
    """One Hogwild! worker: trains one local epoch over its data shard."""
    # Limit intra-op threads to avoid N×cores oversubscription (mirrors train_par_baseline.py:199)
    torch.set_num_threads(1)
    # Distinct DataLoader shuffle order per worker (mirrors train_par_baseline.py:201)
    torch.manual_seed(seed + rank)

    subset = _get_train_subset(rank, num_workers, data_root)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # no nested multiprocessing inside a spawned process
        pin_memory=False,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=False,
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    running_loss = 0.0
    n_samples = 0

    for inputs, targets in loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

        with lock:
            counter.value += 1

        # Sample-weighted accumulation (mirrors train_par_baseline.py:262)
        running_loss += loss.item() * inputs.size(0)
        n_samples += inputs.size(0)

    # Write this worker's epoch stats into the shared arrays (indexed by rank)
    epoch_losses[rank] = running_loss
    epoch_counts[rank] = n_samples


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_one_run_async(
    seed: int,
    num_workers: int,
    momentum: float,
    weight_decay: float,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,   # accepted for API parity; Hogwild! always runs on CPU
    data_root: str,
) -> pd.DataFrame:
    """Train a single Hogwild! run and return per-epoch metrics as a DataFrame.

    Note: runs are *not* bitwise reproducible despite seeding — Hogwild!
    concurrent writes to shared memory introduce non-determinism. Seeds are
    used for dataset partitioning and weight initialisation only.
    """
    cpu = torch.device("cpu")

    # Reproducibility (non-determinism from async writes acknowledged)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = resnet20(dropout=dropout)
    model.share_memory()   # move all tensors into POSIX shared memory

    _, test_loader = get_loaders(data_root, batch_size=256, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    records = []
    for epoch in range(1, epochs + 1):
        # Per-epoch shared arrays for collecting worker stats
        epoch_losses = mp.Array("d", num_workers)
        epoch_counts = mp.Array("l", num_workers)
        counter = mp.Value("i", 0)
        lock = mp.Lock()

        processes = []
        for rank in range(num_workers):
            p = mp.Process(
                target=_worker,
                args=(
                    rank, model, batch_size, num_workers,
                    momentum, weight_decay, lr, data_root, seed,
                    epoch_losses, epoch_counts, counter, lock,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                # Kill remaining workers before re-raising so the process pool
                # doesn't hang if the main process exits.
                for q in processes:
                    if q.is_alive():
                        q.terminate()
                raise RuntimeError(
                    f"Worker process pid={p.pid} exited with code {p.exitcode} "
                    f"at epoch {epoch}."
                )

        # Fix BatchNorm running_mean/running_var corrupted by concurrent writes.
        # Called in main process after all workers have joined (race-free).
        torch.optim.swa_utils.update_bn(_get_bn_train_loader(data_root), model)

        # Train loss: sample-weighted average across all workers
        total_loss = sum(epoch_losses[r] for r in range(num_workers))
        total_samples = sum(epoch_counts[r] for r in range(num_workers))
        train_loss = total_loss / total_samples if total_samples > 0 else float("nan")

        # Evaluate in main process (no concurrent writes at this point)
        test_loss, test_acc, test_error = evaluate(model, test_loader, cpu, criterion)

        # Weight L2 norm (conv/linear weights only, excludes BN params and biases)
        # Same convention as seq_training.py:116-118
        weight_norm = torch.sqrt(
            sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad and p.dim() >= 2)
        ).item()

        records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_error": test_error,
            "weight_norm": weight_norm,
        })

    return pd.DataFrame(records)
