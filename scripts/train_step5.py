"""
Step 5: Mitliagkas equivalence in the true HogWild! setting.

Mitliagkas et al. (2014, Theorem 1) proved that k asynchronous workers with
no explicit momentum (β=0) induce an implicit momentum of µ_S = (k−1)/k,
equivalent to running a single sequential worker with β_explicit = (k−1)/k.

This script tests that prediction:

  --mode hogwild     k workers share the same model in memory; each processes
                     its 1/k data shard every round; workers never synchronise
                     during a round (true HogWild! asynchrony). β_explicit=0.

  --mode sequential  1 worker, full dataset, β_explicit = (k−1)/k.
                     The Mitliagkas prediction: should converge like hogwild.

For evaluation, hogwild mode adds an epoch-level barrier so the main process
can evaluate the shared model after every round. Workers are truly async within
a round; the barrier only exists between rounds.

CSV columns:
  mode, num_workers, momentum, seed, epoch, train_loss, test_acc

Usage (run via sweep_step5.sh or individually):
  python scripts/train_step5.py --mode hogwild    --num-workers 4 --seed 42
  python scripts/train_step5.py --mode sequential --num-workers 4 --seed 42
"""

import argparse
import csv
import os
import sys

import torch
import torch.multiprocessing as mp
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
        description="Step 5 — Mitliagkas equivalence: HogWild! vs sequential",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",        required=True, choices=["hogwild", "sequential"])
    p.add_argument("--num-workers", type=int,   default=4,
                   help="k: number of HogWild! workers (both modes use this to set β)")
    p.add_argument("--momentum",    type=float, default=None,
                   help="override β (defaults: 0.0 for hogwild, (k−1)/k for sequential)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=0.1)
    p.add_argument("--lr-milestones", type=int, nargs="+", default=[30, 40])
    p.add_argument("--lr-gamma",    type=float, default=0.1)
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--weight-decay",type=float, default=1e-4)
    p.add_argument("--data-root",   type=str,   default="data")
    p.add_argument("--out-dir",     type=str,   default="outputs/step5")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mitliagkas_beta(k: int) -> float:
    """Predicted implicit momentum from k HogWild! workers: (k−1)/k."""
    return (k - 1) / k if k > 1 else 0.0


def get_data_shard(rank: int, num_workers: int, data_root: str, batch_size: int):
    """Strided 1/k data shard for HogWild! worker `rank`."""
    full_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=_train_transform)
    indices = list(range(rank, len(full_ds), num_workers))
    subset  = torch.utils.data.Subset(full_ds, indices)
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=0)


def get_full_train_loader(data_root: str, batch_size: int):
    ds = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=_train_transform)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0)


def get_test_loader(data_root: str):
    ds = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=_test_transform)
    return torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=False, num_workers=0)


def evaluate(model: torch.nn.Module, test_loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            correct += model(data).argmax(1).eq(target).sum().item()
            total   += target.size(0)
    model.train()
    return 100.0 * correct / total


def make_csv_path(out_dir: str, args: argparse.Namespace) -> str:
    return os.path.join(
        out_dir,
        f"step5_{args.mode}_k{args.num_workers}_beta{args.momentum}_seed{args.seed}.csv",
    )


# ---------------------------------------------------------------------------
# HogWild! worker (runs in a child process)
# ---------------------------------------------------------------------------

def _hogwild_worker(
    rank: int,
    model: torch.nn.Module,
    args: argparse.Namespace,
    epoch_barrier,   # all workers call this at end of each round
    eval_barrier,    # main releases workers after evaluation
    loss_queue,      # workers push (rank, epoch, avg_loss)
) -> None:
    """
    Pure HogWild!: no locks, no gradient communication.  The shared-memory
    model is read and written concurrently by all workers.  The two barriers
    are only used between rounds so the main process can evaluate the model.
    """
    torch.manual_seed(args.seed + rank)
    loader    = get_data_shard(rank, args.num_workers, args.data_root, args.batch_size)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)

    for epoch in range(args.epochs):
        total_loss = total = 0
        for data, target in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()           # writes directly to shared params — no lock
            total_loss += loss.item() * target.size(0)
            total      += target.size(0)
        scheduler.step()

        loss_queue.put((rank, epoch, total_loss / total))
        epoch_barrier.wait()   # tell main: this worker finished round `epoch`
        eval_barrier.wait()    # wait for main to finish evaluating before continuing


# ---------------------------------------------------------------------------
# HogWild! training loop (runs in the main process)
# ---------------------------------------------------------------------------

def train_hogwild(
    model: torch.nn.Module,
    args: argparse.Namespace,
    test_loader,
    writer,
) -> None:
    loss_queue    = mp.Queue()
    epoch_barrier = mp.Barrier(args.num_workers + 1)  # k workers + main
    eval_barrier  = mp.Barrier(args.num_workers + 1)

    procs = [
        mp.Process(
            target=_hogwild_worker,
            args=(rank, model, args, epoch_barrier, eval_barrier, loss_queue),
        )
        for rank in range(args.num_workers)
    ]
    for p in procs:
        p.start()

    for epoch in range(args.epochs):
        epoch_barrier.wait()  # block until all workers finish this round

        # Workers are now blocked at eval_barrier; safe to evaluate.
        losses   = sorted([loss_queue.get() for _ in range(args.num_workers)],
                          key=lambda x: x[0])
        avg_loss = sum(t[2] for t in losses) / args.num_workers
        test_acc = evaluate(model, test_loader)

        writer.writerow([
            "hogwild", args.num_workers, args.momentum, args.seed,
            epoch + 1, f"{avg_loss:.4f}", f"{test_acc:.2f}",
        ])
        print(
            f"[hogwild k={args.num_workers} β={args.momentum} s={args.seed}] "
            f"epoch {epoch+1:3d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  test={test_acc:.1f}%",
            flush=True,
        )
        eval_barrier.wait()  # release workers to start next round

    for p in procs:
        p.join()


# ---------------------------------------------------------------------------
# Sequential training loop (runs in the main process)
# ---------------------------------------------------------------------------

def train_sequential(
    model: torch.nn.Module,
    args: argparse.Namespace,
    test_loader,
    writer,
) -> None:
    torch.manual_seed(args.seed)
    loader    = get_full_train_loader(args.data_root, args.batch_size)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)

    for epoch in range(args.epochs):
        model.train()
        total_loss = total = 0
        for data, target in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.size(0)
            total      += target.size(0)
        scheduler.step()

        avg_loss = total_loss / total
        test_acc = evaluate(model, test_loader)

        writer.writerow([
            "sequential", args.num_workers, args.momentum, args.seed,
            epoch + 1, f"{avg_loss:.4f}", f"{test_acc:.2f}",
        ])
        print(
            f"[sequential k={args.num_workers} β={args.momentum} s={args.seed}] "
            f"epoch {epoch+1:3d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  test={test_acc:.1f}%",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve default momentum.
    if args.momentum is None:
        args.momentum = 0.0 if args.mode == "hogwild" else mitliagkas_beta(args.num_workers)

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Download dataset in main process to avoid parallel-download races.
    torchvision.datasets.CIFAR10(root=args.data_root, train=True,  download=True)
    torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True)

    model = resnet20()
    if args.mode == "hogwild":
        model.share_memory()

    test_loader = get_test_loader(args.data_root)
    csv_path    = make_csv_path(args.out_dir, args)

    print(
        f"mode={args.mode}  k={args.num_workers}  β={args.momentum}  "
        f"seed={args.seed}  epochs={args.epochs}",
        flush=True,
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "num_workers", "momentum", "seed",
                         "epoch", "train_loss", "test_acc"])

        if args.mode == "hogwild":
            train_hogwild(model, args, test_loader, writer)
        else:
            train_sequential(model, args, test_loader, writer)

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
