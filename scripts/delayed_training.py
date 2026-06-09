"""
delayed_training.py — library for delayed-gradient (FIFO) SGD training runs.

FIFO mechanism mirrors scripts/train_delayed_sgd.py (reference, not imported).
Import train_one_run_delayed and evaluate from sweep scripts; do not run directly.
"""

import collections
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import get_loaders
from src.model import resnet20
from seq_training import evaluate  # no circular dep: seq_training never imports delayed_training

# ---------------------------------------------------------------------------
# Default hyperparameters — import and optionally override in sweep scripts
# ---------------------------------------------------------------------------

EPOCHS = 100
LR = 0.1
BATCH_SIZE = 128
WEIGHT_DECAY = 0.0
DROPOUT = 0.0
SEEDS = [42, 123, 456]
DATA_ROOT = "data"
NUM_WORKERS = 2


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def train_one_run_delayed(
    seed: int,
    tau: int,
    momentum: float,
    weight_decay: float,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    data_root: str,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Train a single ResNet-20 run with FIFO-delayed gradients; return per-epoch metrics.

    The grad_buffer deque persists across epoch boundaries so delay is truly
    τ batches regardless of where epoch boundaries fall (mirrors train_delayed_sgd.py:185).
    τ=0 reduces to standard SGD: the snapshot is appended and immediately popped.

    Returns:
        df_epoch: per-epoch DataFrame with columns epoch, train_loss, test_loss,
            test_acc, test_error, weight_norm, mean_grad_sq.
        grad_sq_per_step: 1-D float64 array; each entry is ‖stale_grad_t‖² for one
            optimizer step (conv/linear weights only).  Length ≈ epochs*steps_per_epoch;
            the first τ batches (buffer fill) contribute no entries.
        w0_norm: scalar ‖w₀‖ computed before training begins.
    """
    # Reproducibility (mirrors train_delayed_sgd.py:163-165)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, test_loader = get_loaders(data_root, batch_size, NUM_WORKERS)
    model = resnet20(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=False,
    )

    # Initial weight norm for theoretical bound (before any gradient step)
    w0_norm = float(np.sqrt(sum(
        p.pow(2).sum().item() for p in model.parameters() if p.requires_grad and p.dim() >= 2
    )))

    # Buffer persists across epoch boundaries (mirrors train_delayed_sgd.py:185)
    grad_buffer: collections.deque = collections.deque()

    grad_sq_per_step: list[float] = []

    records = []
    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        running_loss = 0.0
        n_batches = 0
        epoch_grad_sq: list[float] = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()

            # Snapshot current gradients (mirrors train_delayed_sgd.py:114-118)
            # .detach().clone() is mandatory: without it every deque entry aliases
            # the same storage and reads as zero after the next zero_grad().
            snapshot = [
                p.grad.detach().clone() if p.grad is not None else None
                for p in model.parameters()
            ]
            grad_buffer.append(snapshot)

            # Apply oldest snapshot once buffer depth exceeds τ (mirrors train_delayed_sgd.py:120-128)
            if len(grad_buffer) > tau:
                stale = grad_buffer.popleft()
                # Squared L2 norm of stale gradient (conv/linear weights only, same filter as weight_norm)
                grad_sq = sum(
                    g.pow(2).sum().item()
                    for p, g in zip(model.parameters(), stale)
                    if g is not None and p.dim() >= 2
                )
                grad_sq_per_step.append(grad_sq)
                epoch_grad_sq.append(grad_sq)
                for p, g in zip(model.parameters(), stale):
                    p.grad = g
                optimizer.step()

            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / n_batches

        # --- evaluate ---
        test_loss, test_acc, test_error = evaluate(model, test_loader, device, criterion)

        # --- weight L2 norm (conv/linear weights only, excludes BN params and biases) ---
        weight_norm = torch.sqrt(
            sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad and p.dim() >= 2)
        ).item()

        mean_grad_sq = float(np.mean(epoch_grad_sq)) if epoch_grad_sq else float("nan")

        records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_error": test_error,
            "weight_norm": weight_norm,
            "mean_grad_sq": mean_grad_sq,
        })

    return pd.DataFrame(records), np.array(grad_sq_per_step, dtype=np.float64), w0_norm
