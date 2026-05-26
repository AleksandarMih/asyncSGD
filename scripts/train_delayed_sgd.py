"""
Delayed-gradient SGD — controlled τ × β experiment (Steps 2 & 3).

Simulates gradient staleness via a per-worker FIFO buffer of depth τ:
  - τ=0  →  standard SGD (buffer fires immediately, no delay)
  - τ=k  →  gradient computed at step t is applied at step t+k

Single process, deterministic, exact delay — suitable for a controlled
academic sweep over (τ, β) pairs.

Usage:
  python scripts/train_delayed_sgd.py --tau 0  --momentum 0.9 --seed 42
  python scripts/train_delayed_sgd.py --tau 10 --momentum 0.0 --seed 42
  bash scripts/sweep_step3.sh   # runs all 45 (τ, β, seed) combinations
"""

import argparse
import collections
import csv
import os
import sys

import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import resnet20
from src.data import _train_transform, _test_transform


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delayed-gradient SGD sweep (Steps 2 & 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tau",           type=int,   default=0,
                   help="gradient delay buffer depth (0 = standard SGD)")
    p.add_argument("--momentum",      type=float, default=0.9,
                   help="SGD momentum β")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=0.1)
    p.add_argument("--lr-milestones", type=int,   nargs="+", default=[30, 40],
                   metavar="E")
    p.add_argument("--lr-gamma",      type=float, default=0.1)
    p.add_argument("--batch-size",    type=int,   default=128)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--data-root",     type=str,   default="data")
    p.add_argument("--out-dir",       type=str,   default="outputs/step3")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_loaders(data_root: str, batch_size: int):
    train_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=_train_transform,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_test_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, test_loader


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
) -> tuple[float, float]:
    """
    One epoch of delayed-gradient SGD.

    grad_buffer persists across epoch boundaries — a snapshot taken at the
    last batch of epoch N may be applied at the first batch of epoch N+1.

    Returns (avg_train_loss, train_accuracy_pct).
    """
    model.train()
    total_loss = correct = total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Snapshot current gradients — .detach().clone() is mandatory:
        # without it every entry in the deque aliases the same storage
        # and reads as zero after the next zero_grad().
        snapshot = [
            p.grad.detach().clone() if p.grad is not None else None
            for p in model.parameters()
        ]
        grad_buffer.append(snapshot)

        # Apply the oldest snapshot once the buffer has more than τ entries.
        # When τ=0: 1 > 0 is true immediately → standard SGD.
        # When τ=k: wait until k+1 entries accumulate, then apply the oldest
        #           (which is exactly k steps stale).
        if len(grad_buffer) > tau:
            stale = grad_buffer.popleft()
            for p, g in zip(model.parameters(), stale):
                p.grad = g
            optimizer.step()

        total_loss += loss.item() * target.size(0)
        correct    += output.argmax(dim=1).eq(target).sum().item()
        total      += target.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: torch.nn.Module, loader, device: torch.device) -> float:
    """Returns test accuracy (0–100)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            preds = model(data).argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total   += target.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  tau={args.tau}  momentum={args.momentum}  seed={args.seed}")

    train_loader, test_loader = get_loaders(args.data_root, args.batch_size)

    model     = resnet20().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_gamma,
    )

    grad_buffer = collections.deque()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(
        args.out_dir,
        f"tau{args.tau}_beta{args.momentum}_seed{args.seed}.csv",
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "momentum", "seed", "epoch",
                         "train_loss", "train_acc", "test_acc"])

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, grad_buffer, args.tau, device,
            )
            scheduler.step()
            test_acc = evaluate(model, test_loader, device)

            writer.writerow([
                args.tau, args.momentum, args.seed, epoch,
                f"{train_loss:.4f}", f"{train_acc:.2f}", f"{test_acc:.2f}",
            ])
            f.flush()

            print(
                f"[tau={args.tau} β={args.momentum} s={args.seed}] "
                f"epoch {epoch:3d}/{args.epochs}  "
                f"loss={train_loss:.4f}  train={train_acc:.1f}%  "
                f"test={test_acc:.1f}%  lr={scheduler.get_last_lr()[0]:.4f}",
                flush=True,
            )

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
