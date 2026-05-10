"""
HogWild!-style asynchronous parallel SGD for ResNet20 on CIFAR-10.

Based on:
  Niu et al., "HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic
  Gradient Descent", NeurIPS 2011. https://arxiv.org/abs/1106.5730

Closely following the structure of the official PyTorch HogWild! MNIST example:
  https://github.com/pytorch/examples/tree/main/mnist_hogwild

Usage:
  python scripts/train_par_baseline.py --num-processes 4
  python scripts/train_par_baseline.py --num-processes 2 --epochs 50 --lr 0.05
"""

# ============================================================
# Section 1: Imports and argument parsing
# ============================================================

import argparse
import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision

# Make `src/` importable when the script is run from the project root or from
# the scripts/ subdirectory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import resnet20
from src.data import _train_transform, _test_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="HogWild! async parallel SGD — ResNet20 on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Parallelism ────────────────────────────────────────────────────────
    parser.add_argument(
        "--num-processes", type=int, default=4, metavar="N",
        help="number of parallel worker processes to spawn",
    )

    # ── Training schedule ─────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="training epochs per worker (each epoch covers that worker's data shard)")
    parser.add_argument("--batch-size", type=int, default=128, metavar="B",
                        help="mini-batch size per worker")
    parser.add_argument("--lr", type=float, default=0.1, metavar="LR",
                        help="initial SGD learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M",
                        help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, metavar="WD",
                        help="SGD L2 weight decay")
    parser.add_argument(
        "--lr-milestones", type=int, nargs="+", default=[50, 75],
        metavar="E",
        help="local epochs at which each worker decays its LR by --lr-gamma",
    )
    parser.add_argument("--lr-gamma", type=float, default=0.1, metavar="G",
                        help="LR multiplicative decay factor at each milestone")

    # ── Misc ──────────────────────────────────────────────────────────────
    parser.add_argument("--data-root", type=str, default="data",
                        help="path to CIFAR-10 dataset root directory")
    parser.add_argument("--seed", type=int, default=42, metavar="S",
                        help="base random seed (each worker gets seed + rank)")
    parser.add_argument("--log-interval", type=int, default=50, metavar="N",
                        help="print a training status line every N batches")
    parser.add_argument("--save", action="store_true",
                        help="save evaluation metrics to outputs/cifar10_par_baseline_<N>.csv")

    return parser.parse_args()


# ============================================================
# Section 2: Model instantiation and share_memory()
# ============================================================

def build_shared_model():
    """
    Instantiate ResNet20 and move all parameter tensors into OS shared memory.

    share_memory() calls share_memory_() on every parameter and buffer tensor,
    which replaces their backing storage with a POSIX shared-memory segment
    (shm_open on Linux/macOS). Any process that receives a handle to this
    object — via pickle across a mp.Process boundary — maps the exact same
    physical pages. This is the core mechanism behind HogWild!: all workers
    read and write the same weight values without explicit synchronisation.
    """
    model = resnet20()
    model.share_memory()
    return model


# ============================================================
# Section 3: Dataset loading and partitioning across workers
# ============================================================

def get_train_subset(rank: int, num_processes: int, data_root: str):
    """
    Return a non-overlapping Subset of CIFAR-10 training data for this worker.

    The 50 000 training images are split into `num_processes` contiguous shards.
    Worker `rank` owns shard `rank`; the last worker absorbs any remainder so
    every sample is used exactly once per full pass through all workers.
    """
    full_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=_train_transform,
    )
    n = len(full_dataset)
    per_worker = n // num_processes
    start = rank * per_worker
    end = start + per_worker if rank < num_processes - 1 else n
    return torch.utils.data.Subset(full_dataset, list(range(start, end)))


def get_test_loader(data_root: str, batch_size: int = 256):
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=_test_transform,
    )
    return torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )


# ============================================================
# Section 4: Worker training function — async SGD
# ============================================================

def train_worker(rank: int, model, args, counter, lock):
    """
    One HogWild! worker.

    Each worker:
      1. Builds a DataLoader over its private data shard.
      2. Creates a local SGD optimizer that holds references to the *shared*
         parameter tensors — the same tensors that every other worker's
         optimizer also points to.
      3. Loops over its shard for `args.epochs` epochs, doing the standard
         forward → cross-entropy → backward → step cycle with no inter-worker
         synchronisation at any point.

    Gradient staleness
    ------------------
    The sequence inside the training loop is:

        optimizer.zero_grad()          # zeroes this worker's .grad buffers
        output  = model(data)          # forward: READS shared params
        loss    = cross_entropy(...)
        loss.backward()                # backward: READS shared params again,
                                       #           WRITES .grad buffers
        optimizer.step()               # WRITES updates into shared params

    Between zero_grad() and step(), every other worker is concurrently
    executing the same sequence on the same shared parameter storage.  The
    gradient ∇L(w_t) computed in backward() is therefore applied to a weight
    vector w_{t+Δ} that may have already moved by Δ concurrent updates from
    other workers.  This is classical gradient staleness.  HogWild! proves
    that, under sparsity and a Lipschitz-smooth loss, the algorithm still
    converges — and in practice it often converges nearly as fast as
    synchronous SGD while being trivially parallelisable.

    BatchNorm note
    --------------
    running_mean and running_var inside each BatchNorm layer are updated
    during the forward pass without locking.  Concurrent writes from multiple
    workers produce noisy statistics.  This is a known limitation of applying
    HogWild! to architectures with shared BatchNorm buffers.
    """
    torch.manual_seed(args.seed + rank)  # distinct shuffle order per worker

    subset = get_train_subset(rank, args.num_processes, args.data_root)
    # num_workers=0 inside a spawned process avoids nested multiprocessing
    # issues (DataLoader would try to fork again, hitting OS limits or deadlocks).
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # The optimizer holds references to the shared parameter tensors.
    # There is no gradient accumulation buffer shared across workers —
    # each worker maintains its own .grad attributes on the shared parameters,
    # which itself can cause races on the .grad tensors if momentum is used.
    # SGD with momentum=0 would be fully lock-free; with momentum > 0 the
    # momentum buffer is per-optimizer (local to this worker), so it is safe.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Each worker runs its own LR schedule keyed on local epoch count.
    # Because one local epoch covers only 1/num_processes of the full dataset,
    # "local epoch 50" corresponds to "50/num_processes effective full epochs".
    # Adjust --lr-milestones accordingly if you want globally consistent drops.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma,
    )

    model.train()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()

            # ── FORWARD (reads shared params) ──────────────────────────────
            output = model(data)

            loss = F.cross_entropy(output, target)

            # ── BACKWARD (reads shared params again to compute gradients) ──
            # Staleness happens here: other workers have been updating the
            # shared params since the forward pass began.
            loss.backward()

            # ── ASYNC UPDATE (writes into shared param storage, no lock) ───
            optimizer.step()

            # Track total gradient steps across all workers for reporting.
            with lock:
                counter.value += 1

            epoch_loss += loss.item() * target.size(0)
            epoch_correct += output.argmax(dim=1).eq(target).sum().item()
            epoch_total += target.size(0)

            if batch_idx % args.log_interval == 0:
                print(
                    f"[Worker {rank:2d}] Epoch {epoch:3d}/{args.epochs} "
                    f"batch {batch_idx:4d}/{len(loader)}  "
                    f"loss {loss.item():.4f}",
                    flush=True,
                )

        scheduler.step()

        avg_loss = epoch_loss / epoch_total if epoch_total else float("nan")
        train_acc = 100.0 * epoch_correct / epoch_total if epoch_total else float("nan")
        print(
            f"[Worker {rank:2d}] Epoch {epoch:3d}/{args.epochs} done — "
            f"avg loss: {avg_loss:.4f}  train acc: {train_acc:.2f}%  "
            f"lr: {scheduler.get_last_lr()[0]:.5f}",
            flush=True,
        )

    print(f"[Worker {rank:2d}] Finished.", flush=True)


# ============================================================
# Section 5: Evaluation on test set (run in main process after join)
# ============================================================

_CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def evaluate(model, args):
    model.eval()
    test_loader = get_test_loader(args.data_root, batch_size=256)

    all_targets, all_preds = [], []
    with torch.no_grad():
        for data, target in test_loader:
            preds = model(data).argmax(dim=1)
            all_targets.append(target.numpy())
            all_preds.append(preds.numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    acc = 100.0 * accuracy_score(y_true, y_pred)
    f1  = 100.0 * f1_score(y_true, y_pred, average="macro")

    print(f"\n[Evaluation]")
    print(f"  accuracy : {acc:.2f}%")
    print(f"  macro F1 : {f1:.2f}%")

    if args.save:
        os.makedirs("outputs", exist_ok=True)
        tag = args.num_processes

        csv_path = f"outputs/cifar10_par_baseline_{tag}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["num_processes", args.num_processes])
            writer.writerow(["accuracy", f"{acc:.4f}"])
            writer.writerow(["macro_f1", f"{f1:.4f}"])
        print(f"  metrics saved to {csv_path}")

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=range(len(_CIFAR10_CLASSES)),
            yticks=range(len(_CIFAR10_CLASSES)),
            xticklabels=_CIFAR10_CLASSES,
            yticklabels=_CIFAR10_CLASSES,
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion matrix ({args.num_processes} workers)",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for i in range(len(_CIFAR10_CLASSES)):
            for j in range(len(_CIFAR10_CLASSES)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        png_path = f"outputs/cifar10_par_baseline_{tag}_cm.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  confusion matrix saved to {png_path}")

    return acc, f1


# ============================================================
# Section 6: Entry point — set start method, spawn workers, evaluate
# ============================================================

if __name__ == "__main__":
    # `spawn` is required because:
    #  (a) It is safe on macOS (which bans fork-after-import of most frameworks).
    #  (b) It avoids DataLoader file-descriptor inheritance bugs.
    #  (c) It keeps CUDA safety (not used here, but good practice).
    # With `spawn` each child gets a fresh Python interpreter; the shared-memory
    # model is passed by pickling the object header — the actual tensor data
    # lives in shared memory and is accessed directly by all processes.
    mp.set_start_method("spawn")

    args = parse_args()
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("HogWild! Async Parallel SGD — ResNet20 on CIFAR-10")
    print("=" * 60)
    print(f"  workers      : {args.num_processes}")
    print(f"  epochs/worker: {args.epochs}")
    print(f"  batch size   : {args.batch_size}")
    print(f"  lr           : {args.lr}  momentum: {args.momentum}  wd: {args.weight_decay}")
    print(f"  lr milestones: {args.lr_milestones}  gamma: {args.lr_gamma}")
    print(f"  data root    : {args.data_root}")
    print("=" * 60)

    # Download dataset in the main process to avoid parallel-download races
    # when multiple workers all call torchvision.datasets.CIFAR10 simultaneously.
    torchvision.datasets.CIFAR10(root=args.data_root, train=True,  download=True)
    torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True)

    # ── Build the shared model ─────────────────────────────────────────────
    model = build_shared_model()
    print(f"\nResNet20 in shared memory — "
          f"{sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Shared step counter and its lock (used only for logging, not for training)
    counter = mp.Value("i", 0)
    lock    = mp.Lock()

    # ── Spawn one worker process per requested CPU worker ──────────────────
    # Each process receives the model handle; because the underlying tensors
    # are in shared memory, every worker reads and writes the same weights.
    processes = []
    t_start = time.time()

    for rank in range(args.num_processes):
        p = mp.Process(
            target=train_worker,
            args=(rank, model, args, counter, lock),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - t_start
    print(
        f"\nAll {args.num_processes} workers finished in {elapsed:.1f}s  "
        f"({counter.value:,} total async gradient steps)"
    )

    # ── Evaluate in the main process (no concurrent writes) ────────────────
    evaluate(model, args)
