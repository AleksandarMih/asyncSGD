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
    Usage: python scripts/train_step4.py --tau 5 --compensate --beta-target 0.9

  Exp C — LR scaling
    Does scaling lr by 1/τ or 1/√τ restore convergence under large delays?
    Usage: python scripts/train_step4.py --tau 10 --lr-scale 0.1 --experiment C

Gradient alignment is always tracked: each epoch logs the cosine similarity between
the fresh gradient (just computed) and the stale gradient (τ steps old, about to be
applied). This shows mechanistically why high β + high τ diverges.

CSV columns (extends step3 schema):
  tau, momentum, seed, epoch, train_loss, train_acc, test_acc,
  lr_scale, experiment, grad_angle
"""

import argparse
import collections
import csv
import math
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
                   choices=["A", "B", "C", "D", "E", "F"],
                   help="label written to CSV (does not change training logic)")
    p.add_argument("--out-dir",        type=str,   default="outputs/step4")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def implicit_momentum(tau: int) -> float:
    """Mitliagkas approximation: delay τ induces implicit momentum (τ-1)/τ."""
    return (tau - 1) / tau if tau > 0 else 0.0


def compensated_momentum(tau: int, beta_target: float) -> float:
    """β_explicit = max(0, β_target − β_implicit) so total ≈ β_target."""
    return max(0.0, round(beta_target - implicit_momentum(tau), 6))


def get_loaders(data_root: str, batch_size: int):
    train_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=_train_transform)
    test_ds  = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_test_transform)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True)
    return train_loader, test_loader


def make_csv_path(out_dir: str, args: argparse.Namespace) -> str:
    if args.experiment == "C":
        name = (f"exp{args.experiment}_tau{args.tau}"
                f"_lrscale{args.lr_scale}_beta{args.momentum}_seed{args.seed}.csv")
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
) -> tuple[float, float, float]:
    """
    One epoch of delayed-gradient SGD with gradient-alignment tracking.

    Returns (avg_train_loss, train_accuracy_pct, avg_grad_angle).
    avg_grad_angle is the mean cosine similarity between the fresh gradient
    and the stale gradient being applied; nan when τ=0 (no staleness).
    """
    model.train()
    total_loss = correct = total = 0
    angle_accum: list[float] = []

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss   = F.cross_entropy(output, target)
        loss.backward()

        snapshot = [p.grad.detach().clone() if p.grad is not None else None
                    for p in model.parameters()]
        grad_buffer.append(snapshot)

        if len(grad_buffer) > tau:
            stale = grad_buffer[0]  # peek before pop

            # Gradient alignment: cosine similarity between fresh and stale.
            # A value near 1 means the stale gradient still points in a useful
            # direction; near 0 or negative means staleness has rotated it away.
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
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            correct += model(data).argmax(dim=1).eq(target).sum().item()
            total   += target.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve the actual momentum to use.
    if args.compensate:
        args.momentum = compensated_momentum(args.tau, args.beta_target)

    effective_lr = args.lr * args.lr_scale

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  exp={args.experiment}  tau={args.tau}  "
          f"momentum={args.momentum}  lr_scale={args.lr_scale}  seed={args.seed}")

    train_loader, test_loader = get_loaders(args.data_root, args.batch_size)
    model     = resnet20().to(device)
    optimizer = torch.optim.SGD(
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
                         "lr_scale", "experiment", "grad_angle"])

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, grad_angle = train_one_epoch(
                model, train_loader, optimizer, grad_buffer, args.tau, device)
            scheduler.step()
            test_acc = evaluate(model, test_loader, device)

            angle_str = "" if math.isnan(grad_angle) else f"{grad_angle:.4f}"
            writer.writerow([
                args.tau, args.momentum, args.seed, epoch,
                f"{train_loss:.4f}", f"{train_acc:.2f}", f"{test_acc:.2f}",
                args.lr_scale, args.experiment, angle_str,
            ])
            f.flush()

            print(
                f"[exp={args.experiment} τ={args.tau} β={args.momentum} "
                f"lr_scale={args.lr_scale} s={args.seed}] "
                f"epoch {epoch:3d}/{args.epochs}  "
                f"loss={train_loss:.4f}  train={train_acc:.1f}%  "
                f"test={test_acc:.1f}%  "
                f"angle={angle_str or 'n/a':>7}",
                flush=True,
            )

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
