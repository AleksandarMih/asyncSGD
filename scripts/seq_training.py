"""
seq_training.py — library for sequential (single-device) training runs.

Import train_one_run and evaluate from sweep scripts; do not run directly.
"""

import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_loaders
from src.model import resnet20

# ---------------------------------------------------------------------------
# Default hyperparameters — import and optionally override in sweep scripts
# ---------------------------------------------------------------------------

EPOCHS = 100
LR = 0.1
BATCH_SIZE = 128
MOMENTUM = 0
SEEDS = [42, 123, 456]
DATA_ROOT = "data"
NUM_WORKERS = 2


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    """Return (test_loss, test_acc, test_error) over loader. Acc/error in [0, 1]."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item() * inputs.size(0)
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
            n += inputs.size(0)
    test_loss = total_loss / n
    test_acc = correct / n
    test_error = 1.0 - test_acc
    return test_loss, test_acc, test_error


def train_one_run(
    seed: int,
    l2_lambda: float,
    dropout: float,
    momentum: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    data_root: str,
) -> pd.DataFrame:
    """Train a single ResNet-20 run and return per-epoch metrics as a DataFrame."""
    # Reproducibility
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
        weight_decay=l2_lambda,
        nesterov=False,
    )

    records = []
    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
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

        records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_error": test_error,
            "weight_norm": weight_norm,
        })

    return pd.DataFrame(records)
