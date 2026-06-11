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
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import resnet20
from src.data import _train_transform, _test_transform


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
    """Mitliagkas approximation: delay τ induces implicit momentum (τ-1)/τ."""
    return (tau - 1) / tau if tau > 0 else 0.0


def compensated_momentum(tau: int, beta_target: float) -> float:
    """β_explicit = max(0, β_target − β_implicit) so total ≈ β_target."""
    return max(0.0, round(beta_target - implicit_momentum(tau), 6))


def liu_bound_beta(tau: int, lr: float) -> float:
    """µ_safe ceiling from Liu et al. NeurIPS 2018: τ ≲ (1−µ)²/η → µ = 1−√(τ·η)."""
    return 1.0 - math.sqrt(max(tau * lr, 0.0))


def compute_scheduled_beta(tau: int, current_lr: float, schedule: str,
                           initial_lr: float, bound_offset: float = 0.3) -> float:
    """Return the momentum to apply this epoch according to the named schedule."""
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

    Returns (avg_train_loss, train_accuracy_pct, avg_grad_angle).
    avg_grad_angle is the mean cosine similarity between the fresh gradient
    and the stale gradient being applied; nan when no staleness.
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
