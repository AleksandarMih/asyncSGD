"""
Step 4: Delay-momentum theory experiments (Mitliagkas et al. 2014).

Three experiments, all sharing this script via --experiment label:

  Exp A — Equivalence test
    Theory: delay τ induces implicit momentum β_impl ≈ (τ-1)/τ.
    Test: does (τ=k, β=0) converge like (τ=0, β=(k-1)/k)?
    Usage: run pairs manually or via sweep_step4.sh.

  Exp B — Compensation
    If delay adds β_impl, subtract it from β_target so total ≈ β_target.
    β_explicit = max(0, β_target − (τ−1)/τ)
    Usage: python scripts/train.py --tau 5 --compensate --beta-target 0.9

  Exp C — LR scaling
    Does scaling lr by 1/τ or 1/√τ restore convergence under large delays?
    Usage: python scripts/train.py --tau 10 --lr-scale 0.1 --experiment C

Two delay regimes:
  fifo (default): deterministic FIFO buffer with fixed lag --tau.
  geometric: τ_t ~ Geom(1/M)−1 per step, matching the Mitliagkas M/M/1 model.
    Use --delay-type geometric --M <workers>. E[τ] = M−1.

Gradient alignment is always tracked: each epoch logs the cosine similarity between
the fresh gradient (just computed) and the stale gradient being applied.

CSV columns:
  tau, momentum, seed, epoch, train_loss, train_acc, test_acc,
  lr_scale, experiment, grad_angle, schedule, beta_applied, bound_offset,
  delay_type, M
"""

import argparse
import collections
import csv
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import resnet20
from src.data import _train_transform, _test_transform


# ---------------------------------------------------------------------------
# Shared default hyperparameters — importable by sweep scripts
# ---------------------------------------------------------------------------

EPOCHS      = 100
LR          = 0.1
BATCH_SIZE  = 128
MOMENTUM    = 0.0
DROPOUT     = 0.0
SEEDS       = [42, 123, 456]
DATA_ROOT   = "data"
NUM_WORKERS = 2


# ---------------------------------------------------------------------------
# Custom optimizer — supports negative momentum (PyTorch SGD rejects β < 0)
# ---------------------------------------------------------------------------

class SGDMomentum(torch.optim.Optimizer):
    """SGD with momentum for any β ∈ ℝ, including negative values.

    Update rule (identical to torch.optim.SGD with dampening=0, nesterov=False):
      v_t = β · v_{t-1} + g_t
      w_t = w_{t-1} − lr · v_t

    Negative β damps the velocity and flips its sign each step, effectively
    acting as a high-pass filter on gradients — the theoretical compensation
    for excess implicit momentum from gradient delay (Mitliagkas 2014, §VI).
    """
    def __init__(self, params, lr: float, momentum: float, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform one SGD+momentum update over all parameter groups.

        Args:
            closure: optional callable that re-evaluates the model and returns the loss.

        Returns:
            loss returned by closure if provided, else None.
        """
        for group in self.param_groups:
            lr       = group["lr"]
            beta     = group["momentum"]
            wd       = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p.data, alpha=wd)
                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data)
                v = state["v"]
                v.mul_(beta).add_(g)
                p.data.add_(v, alpha=-lr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the delayed-gradient training script.

    Returns:
        argparse.Namespace with all hyperparameter and experiment configuration fields.
    """
    p = argparse.ArgumentParser(
        description="Step 4 delayed-gradient experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── shared with step3 ──────────────────────────────────────────────────
    p.add_argument("--tau",            type=int,   default=0)
    p.add_argument("--momentum",       type=float, default=0.0,
                   help="explicit SGD momentum β (overridden by --compensate)")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--lr",             type=float, default=0.1,
                   help="base learning rate (multiplied by --lr-scale)")
    p.add_argument("--lr-milestones",  type=int,   nargs="+", default=[30, 40])
    p.add_argument("--lr-gamma",       type=float, default=0.1)
    p.add_argument("--batch-size",     type=int,   default=128)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--data-root",      type=str,   default="data")
    # ── new in step4 ───────────────────────────────────────────────────────
    p.add_argument("--lr-scale",       type=float, default=1.0,
                   help="multiply base lr by this factor (Exp C: try 1/τ or 1/√τ)")
    p.add_argument("--compensate",     action="store_true",
                   help="set β = max(0, --beta-target − (τ−1)/τ)  [Exp B]")
    p.add_argument("--beta-target",    type=float, default=0.9,
                   help="desired effective momentum when using --compensate")
    p.add_argument("--experiment",     type=str,   default="A",
                   choices=["A", "B", "C", "D", "E", "F", "G", "H"],
                   help="label written to CSV (does not change training logic)")
    p.add_argument("--schedule",       type=str,   default="fixed",
                   choices=["fixed", "at_bound", "below_bound", "above_bound", "neg_ramp"],
                   help="Exp G/H: adaptive momentum schedule derived from Liu et al. bound")
    p.add_argument("--bound-offset",   type=float, default=0.3,
                   help="Exp H: additive offset above Liu bound for above_bound schedule")
    p.add_argument("--out-dir",        type=str,   default="outputs/det_theory")
    # ── geometric delay (new) ──────────────────────────────────────────────
    p.add_argument("--delay-type",     type=str,   default="fifo",
                   choices=["fifo", "geometric"],
                   help="fifo: fixed deterministic delay --tau; "
                        "geometric: τ_t ~ Geom(1/M)−1, E[τ]=M−1")
    p.add_argument("--M",              type=int,   default=1,
                   help="simulated workers for geometric delay; E[τ]=M−1 (ignored for fifo)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def implicit_momentum(tau: int) -> float:
    """Compute the implicit momentum induced by gradient delay (Mitliagkas 2014).

    Args:
        tau: gradient delay in steps.

    Returns:
        Implicit momentum value (τ−1)/τ, or 0.0 when tau=0.
    """
    return (tau - 1) / tau if tau > 0 else 0.0


def compensated_momentum(tau: int, beta_target: float) -> float:
    """Compute the explicit momentum that compensates for implicit delay momentum.

    Args:
        tau: gradient delay in steps.
        beta_target: desired effective total momentum.

    Returns:
        max(0, beta_target − implicit_momentum(tau)), rounded to 6 decimal places.
    """
    return max(0.0, round(beta_target - implicit_momentum(tau), 6))


def liu_bound_beta(tau: int, lr: float) -> float:
    """Compute the safe momentum ceiling from Liu et al. NeurIPS 2018.

    Args:
        tau: gradient delay in steps.
        lr: current learning rate η.

    Returns:
        µ_safe = 1 − √(τ·η), the largest µ satisfying τ ≲ (1−µ)²/η.
    """
    return 1.0 - math.sqrt(max(tau * lr, 0.0))


def compute_scheduled_beta(tau: int, current_lr: float, schedule: str,
                           initial_lr: float, bound_offset: float = 0.3) -> float:
    """Return the momentum to apply this epoch according to the named schedule.

    Args:
        tau: gradient delay in steps.
        current_lr: learning rate in use this epoch.
        schedule: one of 'at_bound', 'below_bound', 'above_bound', 'neg_ramp'.
        initial_lr: base learning rate before any decay (used by 'neg_ramp').
        bound_offset: additive offset above the Liu bound for 'above_bound'.

    Returns:
        Momentum scalar for this epoch.
    """
    liu = liu_bound_beta(tau, current_lr)
    if schedule == "at_bound":
        return liu
    elif schedule == "below_bound":
        return liu - 0.3
    elif schedule == "above_bound":
        return min(0.95, liu + bound_offset)
    elif schedule == "neg_ramp":
        if abs(current_lr - initial_lr) < 1e-9:   # phase 1: LR has not yet decayed
            return -((tau - 1) / tau)
        return liu
    return liu  # fallback


def get_loaders(data_root: str, batch_size: int, num_workers: int = 2):
    """Build CIFAR-10 train and test DataLoaders.

    Args:
        data_root: path to the CIFAR-10 data directory (downloaded if absent).
        batch_size: mini-batch size for the training loader (test loader uses 256).
        num_workers: DataLoader worker processes.

    Returns:
        (train_loader, test_loader) tuple of torch DataLoader objects.
    """
    train_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=_train_transform)
    test_ds  = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_test_transform)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def make_csv_path(out_dir: str, args: argparse.Namespace) -> str:
    """Construct the output CSV file path from the experiment configuration.

    Args:
        out_dir: directory where the CSV will be written.
        args: parsed argument namespace; uses experiment, delay_type, tau, M,
              momentum, lr_scale, schedule, bound_offset, and seed.

    Returns:
        Path string for the CSV output file.
    """
    if args.delay_type == "geometric":
        if args.experiment == "G":
            name = (f"expG_geo_M{args.M}_schedule{args.schedule}_seed{args.seed}.csv")
        elif args.experiment == "H":
            name = (f"expH_geo_M{args.M}_offset{args.bound_offset}_seed{args.seed}.csv")
        elif args.experiment == "C":
            name = (f"expC_geo_M{args.M}"
                    f"_lrscale{args.lr_scale}_beta{args.momentum}_seed{args.seed}.csv")
        else:
            name = (f"exp{args.experiment}_geo_M{args.M}"
                    f"_beta{args.momentum}_seed{args.seed}.csv")
    elif args.experiment == "C":
        name = (f"exp{args.experiment}_tau{args.tau}"
                f"_lrscale{args.lr_scale}_beta{args.momentum}_seed{args.seed}.csv")
    elif args.experiment == "G":
        name = (f"expG_tau{args.tau}_schedule{args.schedule}_seed{args.seed}.csv")
    elif args.experiment == "H":
        name = (f"expH_tau{args.tau}_offset{args.bound_offset}_seed{args.seed}.csv")
    else:
        name = (f"exp{args.experiment}_tau{args.tau}"
                f"_beta{args.momentum}_seed{args.seed}.csv")
    return os.path.join(out_dir, name)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    grad_buffer: collections.deque,
    tau: int,
    device: torch.device,
    delay_type: str = "fifo",
    M: int = 1,
    rng: np.random.Generator = None,
) -> tuple[float, float, float]:
    """
    One epoch of delayed-gradient SGD with gradient-alignment tracking.

    For delay_type='fifo': deterministic FIFO buffer with fixed lag tau.
    For delay_type='geometric': tau_t ~ Geom(1/M)−1 per step; grad_buffer
      stores the last min(5M, step) snapshots, indexed oldest→newest (left→right).
      The sampled delay indexes into this history.

    Args:
        model: the neural network being trained.
        loader: training DataLoader.
        optimizer: SGD optimizer whose parameter gradients will be replaced with stale ones.
        grad_buffer: persistent deque of gradient snapshots shared across epochs.
        tau: FIFO delay depth in steps (ignored for geometric mode).
        device: torch device for data and model.
        delay_type: 'fifo' for deterministic lag; 'geometric' for random per-step delay.
        M: number of simulated workers; E[τ_t] = M−1 (geometric mode only).
        rng: numpy Generator for geometric delay sampling.

    Returns:
        (avg_train_loss, train_accuracy_pct, avg_grad_angle) — avg_grad_angle is
        the mean cosine similarity between the fresh and applied stale gradient;
        nan when no stale gradients were applied.
    """
    model.train()
    total_loss = correct = total = 0
    angle_accum: list[float] = []

    max_buf = 5 * M if delay_type == "geometric" else (tau + 1)

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss   = F.cross_entropy(output, target)
        loss.backward()

        snapshot = [p.grad.detach().clone() if p.grad is not None else None
                    for p in model.parameters()]
        grad_buffer.append(snapshot)

        # Keep buffer bounded
        while len(grad_buffer) > max_buf:
            grad_buffer.popleft()

        if delay_type == "geometric":
            # Sample per-step delay; τ_t=0 means apply the current gradient.
            tau_t = int(rng.geometric(p=1.0 / M)) - 1   # E[tau_t] = M−1
            tau_t = min(tau_t, len(grad_buffer) - 1)     # cap to available history
            # Buffer indexed oldest→newest; -(tau_t+1) gives tau_t steps back.
            stale = grad_buffer[-(tau_t + 1)]

            if tau_t > 0:
                fresh_vec = torch.cat([g.flatten() for g in snapshot if g is not None])
                stale_vec = torch.cat([g.flatten() for g in stale   if g is not None])
                cos_sim   = F.cosine_similarity(
                    fresh_vec.unsqueeze(0), stale_vec.unsqueeze(0)).item()
                angle_accum.append(cos_sim)

            for p, g in zip(model.parameters(), stale):
                p.grad = g
            optimizer.step()

        else:  # fifo — original deterministic behaviour
            if len(grad_buffer) > tau:
                stale = grad_buffer[0]  # peek before pop

                if tau > 0:
                    fresh_vec = torch.cat([g.flatten() for g in snapshot if g is not None])
                    stale_vec = torch.cat([g.flatten() for g in stale   if g is not None])
                    cos_sim   = F.cosine_similarity(
                        fresh_vec.unsqueeze(0), stale_vec.unsqueeze(0)).item()
                    angle_accum.append(cos_sim)

                stale = grad_buffer.popleft()
                for p, g in zip(model.parameters(), stale):
                    p.grad = g
                optimizer.step()

        total_loss += loss.item() * target.size(0)
        correct    += output.argmax(dim=1).eq(target).sum().item()
        total      += target.size(0)

    avg_loss    = total_loss / total
    accuracy    = 100.0 * correct / total
    avg_angle   = float("nan") if not angle_accum else sum(angle_accum) / len(angle_accum)
    return avg_loss, accuracy, avg_angle


def evaluate(model: torch.nn.Module, loader, device: torch.device) -> float:
    """Evaluate top-1 classification accuracy on a DataLoader.

    Args:
        model: network to evaluate (set to eval mode internally).
        loader: DataLoader over the evaluation set.
        device: torch device for data and model.

    Returns:
        Top-1 accuracy as a percentage in [0, 100].
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            correct += model(data).argmax(dim=1).eq(target).sum().item()
            total   += target.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Library: richer evaluation — returns (test_loss, test_acc, test_error)
# ---------------------------------------------------------------------------

def evaluate_full(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> tuple[float, float, float]:
    """Evaluate loss and accuracy over a DataLoader.

    Args:
        model: network to evaluate (set to eval mode internally).
        loader: DataLoader over the evaluation set.
        device: torch device for data and model.
        criterion: loss function module (e.g. CrossEntropyLoss).

    Returns:
        (test_loss, test_acc, test_error) where test_acc and test_error are in
        [0, 1] and test_error = 1 − test_acc.
    """
    model.eval()
    total_loss = correct = n = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            total_loss += criterion(out, target).item() * data.size(0)
            correct += out.argmax(dim=1).eq(target).sum().item()
            n += data.size(0)
    acc = correct / n
    return total_loss / n, acc, 1.0 - acc


# ---------------------------------------------------------------------------
# Library: sequential SGD (importable from sweep scripts)
# ---------------------------------------------------------------------------

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
    num_workers: int = 2,
) -> pd.DataFrame:
    """Train a single ResNet-20 run with standard SGD and return per-epoch metrics.

    Args:
        seed: random seed for reproducibility.
        l2_lambda: L2 weight-decay coefficient.
        dropout: dropout probability applied inside ResNet-20.
        momentum: SGD momentum β.
        epochs: number of training epochs.
        lr: learning rate.
        batch_size: mini-batch size.
        device: torch device for training.
        data_root: path to the CIFAR-10 data directory.
        num_workers: DataLoader worker processes.

    Returns:
        DataFrame with columns epoch, train_loss, test_loss, test_acc,
        test_error, weight_norm — one row per epoch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, test_loader = get_loaders(data_root, batch_size, num_workers)
    model = resnet20(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum,
        weight_decay=l2_lambda, nesterov=False,
    )

    records = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for inputs, targets in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / n_batches

        test_loss, test_acc, test_error = evaluate_full(model, test_loader, device, criterion)
        weight_norm = float(torch.stack(
            [p.pow(2).sum() for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        ).sum().sqrt())
        records.append({
            "epoch": epoch, "train_loss": train_loss,
            "test_loss": test_loss, "test_acc": test_acc,
            "test_error": test_error, "weight_norm": weight_norm,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Library: FIFO delayed-gradient SGD (importable from sweep scripts)
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
    num_workers: int = 2,
    delay_type: str = "fifo",
    M: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Train ResNet-20 with delayed gradients; return per-epoch metrics.

    The grad_buffer deque persists across epoch boundaries so delay is truly
    τ batches regardless of where epoch boundaries fall.
    τ=0 reduces to standard SGD.

    delay_type='fifo'      — deterministic FIFO queue; lag = τ batches.
    delay_type='geometric' — τ_t ~ Geom(1/M)−1 per step; sliding window of 5M.
                             Set M = τ + 1 so E[τ_t] = τ matches the FIFO nominal.

    Args:
        seed: random seed for reproducibility.
        tau: FIFO delay depth in batches; ignored when delay_type='geometric'.
        momentum: SGD momentum β.
        weight_decay: L2 regularisation coefficient.
        dropout: dropout probability applied inside ResNet-20.
        epochs: number of training epochs.
        lr: learning rate.
        batch_size: mini-batch size.
        device: torch device for training.
        data_root: path to the CIFAR-10 data directory.
        num_workers: DataLoader worker processes.
        delay_type: 'fifo' or 'geometric'.
        M: number of simulated workers for geometric delay; E[τ_t] = M−1.
        rng: optional numpy Generator; created from seed if None.

    Returns:
        df_epoch: per-epoch DataFrame with columns epoch, train_loss, test_loss,
            test_acc, test_error, weight_norm, mean_grad_sq.
        grad_sq_per_step: 1-D float64 array of ‖stale_grad_t‖² per optimizer step
            (conv/linear weights only). Length ≈ epochs×steps_per_epoch.
        w0_norm: ‖w₀‖ computed before training begins.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if rng is None:
        rng = np.random.default_rng(seed)

    train_loader, test_loader = get_loaders(data_root, batch_size, num_workers)
    model = resnet20(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum,
        weight_decay=weight_decay, nesterov=False,
    )

    w0_norm = float(np.sqrt(sum(
        p.pow(2).sum().item()
        for p in model.parameters() if p.requires_grad and p.dim() >= 2
    )))

    grad_buffer: collections.deque = collections.deque()
    grad_sq_per_step: list[float] = []
    records = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        epoch_grad_sq: list[float] = []
        for inputs, targets in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            snapshot = [
                p.grad.detach().clone() if p.grad is not None else None
                for p in model.parameters()
            ]
            grad_buffer.append(snapshot)
            if delay_type == "geometric":
                while len(grad_buffer) > 5 * M:
                    grad_buffer.popleft()
                tau_t = int(rng.geometric(p=1.0 / M)) - 1
                tau_t = min(tau_t, len(grad_buffer) - 1)
                stale = grad_buffer[-(tau_t + 1)]
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
            else:  # fifo
                if len(grad_buffer) > tau:
                    stale = grad_buffer.popleft()
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

        test_loss, test_acc, test_error = evaluate_full(model, test_loader, device, criterion)
        weight_norm = float(torch.stack(
            [p.pow(2).sum() for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        ).sum().sqrt())
        mean_grad_sq = float(np.mean(epoch_grad_sq)) if epoch_grad_sq else float("nan")
        records.append({
            "epoch": epoch, "train_loss": train_loss,
            "test_loss": test_loss, "test_acc": test_acc,
            "test_error": test_error, "weight_norm": weight_norm,
            "mean_grad_sq": mean_grad_sq,
        })

    return pd.DataFrame(records), np.array(grad_sq_per_step, dtype=np.float64), w0_norm


# ---------------------------------------------------------------------------
# Library: Hogwild! async SGD — helpers and main function
# ---------------------------------------------------------------------------

def _get_train_subset(rank: int, num_workers: int, data_root: str) -> torch.utils.data.Subset:
    """Return a contiguous shard of the CIFAR-10 training set for one Hogwild! worker.

    Args:
        rank: worker index (0-based).
        num_workers: total number of workers; determines shard size.
        data_root: path to the CIFAR-10 data directory.

    Returns:
        torch.utils.data.Subset covering this worker's portion of the training set.
    """
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=_train_transform,
    )
    n = len(dataset)
    per_worker = n // num_workers
    start = rank * per_worker
    end = start + per_worker if rank < num_workers - 1 else n
    return torch.utils.data.Subset(dataset, list(range(start, end)))


def _get_bn_train_loader(data_root: str) -> torch.utils.data.DataLoader:
    """Return a DataLoader over the full CIFAR-10 training set for BatchNorm re-estimation.

    Args:
        data_root: path to the CIFAR-10 data directory.

    Returns:
        DataLoader with batch_size=256, shuffle=False, num_workers=0.
    """
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=_train_transform,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False,
    )


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
    epoch_losses,
    epoch_counts,
    counter,
    lock,
) -> None:
    """One Hogwild! worker: trains one local epoch over its data shard.

    Writes directly to the shared-memory model without locks (Hogwild! semantics).
    Results are accumulated into the shared epoch_losses / epoch_counts arrays.

    Args:
        rank: worker index used for data sharding and seeding.
        model: shared-memory ResNet-20; parameters are updated in-place.
        batch_size: mini-batch size.
        num_workers: total worker count (for sharding).
        momentum: SGD momentum β.
        weight_decay: L2 regularisation coefficient.
        lr: learning rate.
        data_root: path to the CIFAR-10 data directory.
        seed: base random seed; worker uses seed + rank.
        epoch_losses: shared array; element [rank] receives this worker's total loss.
        epoch_counts: shared array; element [rank] receives this worker's sample count.
        counter: shared step counter incremented after each batch.
        lock: multiprocessing lock protecting counter updates.

    Returns:
        None (results written to shared arrays).
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed + rank)

    subset = _get_train_subset(rank, num_workers, data_root)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum,
        weight_decay=weight_decay, nesterov=False,
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
        running_loss += loss.item() * inputs.size(0)
        n_samples += inputs.size(0)
    epoch_losses[rank] = running_loss
    epoch_counts[rank] = n_samples


def train_one_run_async(
    seed: int,
    num_workers: int,
    momentum: float,
    weight_decay: float,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    data_root: str,
) -> pd.DataFrame:
    """Train a single Hogwild! run and return per-epoch metrics as a DataFrame.

    Spawns num_workers parallel processes each epoch; they write concurrently to
    a shared-memory model without locks (Hogwild! semantics). BatchNorm statistics
    are re-estimated after each epoch via update_bn.

    Note: runs are *not* bitwise reproducible despite seeding — Hogwild!
    concurrent writes to shared memory introduce non-determinism.
    `device` is accepted for API parity; Hogwild! always runs on CPU.

    Args:
        seed: random seed for model initialisation and worker seeding.
        num_workers: number of parallel Hogwild! worker processes.
        momentum: SGD momentum β.
        weight_decay: L2 regularisation coefficient.
        dropout: dropout probability applied inside ResNet-20.
        epochs: number of training epochs.
        lr: learning rate.
        batch_size: mini-batch size per worker.
        device: accepted for API parity; training always runs on CPU.
        data_root: path to the CIFAR-10 data directory.

    Returns:
        DataFrame with columns epoch, train_loss, test_loss, test_acc,
        test_error, weight_norm — one row per epoch.
    """
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    cpu = torch.device("cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = resnet20(dropout=dropout)
    model.share_memory()

    _, test_loader = get_loaders(data_root, batch_size=256, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    records = []
    for epoch in range(1, epochs + 1):
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
                for q in processes:
                    if q.is_alive():
                        q.terminate()
                raise RuntimeError(
                    f"Worker process pid={p.pid} exited with code {p.exitcode} "
                    f"at epoch {epoch}."
                )

        torch.optim.swa_utils.update_bn(_get_bn_train_loader(data_root), model)

        total_loss = sum(epoch_losses[r] for r in range(num_workers))
        total_samples = sum(epoch_counts[r] for r in range(num_workers))
        train_loss = total_loss / total_samples if total_samples > 0 else float("nan")

        test_loss, test_acc, test_error = evaluate_full(model, test_loader, cpu, criterion)
        weight_norm = float(torch.stack(
            [p.pow(2).sum() for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        ).sum().sqrt())
        records.append({
            "epoch": epoch, "train_loss": train_loss,
            "test_loss": test_loss, "test_acc": test_acc,
            "test_error": test_error, "weight_norm": weight_norm,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run a single delayed-gradient training experiment from CLI arguments.

    Resolves effective momentum (compensated or fixed), initialises the model,
    optimizer, LR scheduler, and gradient buffer, then trains for --epochs epochs,
    writing per-epoch metrics to a CSV file under --out-dir.
    """
    args = parse_args()

    # For geometric delay, τ_eff = E[τ] = M−1, used by compensation / schedule helpers.
    if args.delay_type == "geometric":
        args.tau = args.M - 1

    # Resolve the actual momentum to use.
    if args.compensate:
        args.momentum = compensated_momentum(args.tau, args.beta_target)

    effective_lr = args.lr * args.lr_scale

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # Separate numpy RNG for geometric sampling (decoupled from torch seed stream).
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  exp={args.experiment}  delay={args.delay_type}  "
          f"tau={args.tau}  M={args.M}  momentum={args.momentum}  "
          f"lr_scale={args.lr_scale}  seed={args.seed}")

    train_loader, test_loader = get_loaders(args.data_root, args.batch_size)
    model     = resnet20().to(device)
    optimizer = SGDMomentum(
        model.parameters(), lr=effective_lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)

    grad_buffer = collections.deque()
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = make_csv_path(args.out_dir, args)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "momentum", "seed", "epoch",
                         "train_loss", "train_acc", "test_acc",
                         "lr_scale", "experiment", "grad_angle",
                         "schedule", "beta_applied", "bound_offset",
                         "delay_type", "M"])

        for epoch in range(1, args.epochs + 1):
            beta_applied = optimizer.param_groups[0]["momentum"]  # used this epoch
            train_loss, train_acc, grad_angle = train_one_epoch(
                model, train_loader, optimizer, grad_buffer, args.tau, device,
                delay_type=args.delay_type, M=args.M, rng=rng)
            scheduler.step()
            test_acc = evaluate(model, test_loader, device)

            if args.schedule != "fixed":
                current_lr = optimizer.param_groups[0]["lr"]
                new_beta = compute_scheduled_beta(
                    args.tau, current_lr, args.schedule, args.lr, args.bound_offset)
                optimizer.param_groups[0]["momentum"] = new_beta

            angle_str = "" if math.isnan(grad_angle) else f"{grad_angle:.4f}"
            writer.writerow([
                args.tau, args.momentum, args.seed, epoch,
                f"{train_loss:.4f}", f"{train_acc:.2f}", f"{test_acc:.2f}",
                args.lr_scale, args.experiment, angle_str,
                args.schedule, f"{beta_applied:.4f}", args.bound_offset,
                args.delay_type, args.M,
            ])
            f.flush()

            print(
                f"[exp={args.experiment} delay={args.delay_type} "
                f"τ={args.tau} M={args.M} sched={args.schedule} "
                f"β_applied={beta_applied:.3f} s={args.seed}] "
                f"epoch {epoch:3d}/{args.epochs}  "
                f"loss={train_loss:.4f}  train={train_acc:.1f}%  "
                f"test={test_acc:.1f}%  "
                f"angle={angle_str or 'n/a':>7}",
                flush=True,
            )

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
