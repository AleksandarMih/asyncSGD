"""
Multi-seed, multi-worker-count experiment for HogWild! async parallel SGD.

Sweeps every integer worker count in [min_workers, max_workers], trains with
5 fixed seeds per count, aggregates test-set metrics, and writes:
  - outputs/summary_drop_<d>_mom_<m>.csv
  - outputs/{error,accuracy,macro_f1}_vs_num_workers_drop_<d>_mom_<m>.pdf

Usage:
  python scripts/train_par_experiment.py --min-workers 1 --max-workers 8
"""

import argparse
import copy
import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.multiprocessing as mp
import torchvision

# Make both src/ and scripts/ importable regardless of CWD.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPTS_DIR, ".."))  # src/
sys.path.insert(0, _SCRIPTS_DIR)                      # train_par_baseline

from train_par_baseline import (
    build_shared_model,
    get_bn_train_loader,
    get_test_loader,
    train_worker,
)


SEEDS = [0, 1, 2, 3, 4]


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="HogWild! experiment: sweep worker counts × seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--min-workers", type=int, default=1, metavar="N",
                        help="smallest worker count to sweep (inclusive)")
    parser.add_argument("--max-workers", type=int, default=8, metavar="N",
                        help="largest worker count to sweep (inclusive)")

    parser.add_argument("--epochs",       type=int,   default=100,       metavar="N")
    parser.add_argument("--batch-size",   type=int,   default=128,       metavar="B")
    parser.add_argument("--lr",           type=float, default=0.1,       metavar="LR")
    parser.add_argument("--momentum",     type=float, default=0.9,       metavar="M")
    parser.add_argument("--weight-decay", type=float, default=1e-4,      metavar="WD")
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=[50, 75], metavar="E")
    parser.add_argument("--lr-gamma",     type=float, default=0.1,       metavar="G")
    parser.add_argument("--data-root",    type=str,   default="data")
    parser.add_argument(
        "--dropout", type=float, default=0.0, metavar="D",
        help="dropout rate (label/filename only — not wired into the architecture)",
    )

    return parser.parse_args()


# ============================================================
# Training helpers
# ============================================================

def _quiet_train_worker(rank, model, args, counter, lock):
    """Wrapper that silences stdout/stderr inside each spawned worker process."""
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        train_worker(rank, model, args, counter, lock)


def _run_one(num_workers, seed, args):
    """
    Train a fresh shared model with num_workers HogWild! workers under seed.
    Returns the trained model.
    """
    torch.manual_seed(seed)
    model = build_shared_model()

    # Build a per-run args namespace reusing all hyper-parameters from the
    # sweep args, but with the fields train_worker expects.
    run_args = copy.copy(args)
    run_args.num_processes = num_workers
    run_args.seed          = seed
    run_args.log_interval  = 10 ** 9   # effectively disables per-batch prints
    run_args.save          = False     # evaluation/saving is handled by the sweep

    counter = mp.Value("i", 0)
    lock    = mp.Lock()

    processes = []
    for rank in range(num_workers):
        p = mp.Process(
            target=_quiet_train_worker,
            args=(rank, model, run_args, counter, lock),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    return model


# ============================================================
# Evaluation
# ============================================================

def _evaluate(model, data_root):
    """
    Evaluate model on the CIFAR-10 test set.
    Returns (top-1 accuracy %, classification error %, macro F1 %).
    """
    model.eval()
    loader = get_test_loader(data_root, batch_size=256)
    all_targets, all_preds = [], []
    with torch.no_grad():
        for data, target in loader:
            preds = model(data).argmax(dim=1)
            all_targets.append(target.numpy())
            all_preds.append(preds.numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    acc   = 100.0 * accuracy_score(y_true, y_pred)
    error = 100.0 - acc
    f1    = 100.0 * f1_score(y_true, y_pred, average="macro")
    return acc, error, f1


# ============================================================
# Output helpers
# ============================================================

def _file_tag(args):
    """Shared suffix for all output filenames."""
    return f"drop_{args.dropout}_mom_{args.momentum}"


def _write_csv(results, args, out_dir):
    """
    Write/overwrite the summary CSV.
    Column order: num_workers, mean_accuracy, std_accuracy,
                  mean_error, std_error, mean_macro_f1, std_macro_f1.
    """
    path = os.path.join(out_dir, f"summary_{_file_tag(args)}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "num_workers",
            "mean_accuracy", "std_accuracy",
            "mean_error",    "std_error",
            "mean_macro_f1", "std_macro_f1",
        ])
        for nw in sorted(results):
            mean_acc, std_acc, mean_err, std_err, mean_f1, std_f1 = results[nw]
            writer.writerow([
                nw,
                f"{mean_acc:.4f}", f"{std_acc:.4f}",
                f"{mean_err:.4f}", f"{std_err:.4f}",
                f"{mean_f1:.4f}",  f"{std_f1:.4f}",
            ])
    return path


def _plot_metrics(results, args, out_dir):
    """
    Save three PDFs — one each for classification error, accuracy, macro F1.
    Each shows mean across worker counts with a ±1 std shaded band.
    """
    worker_counts = sorted(results)
    tag = _file_tag(args)

    # (metric_key, y-axis label, index into results tuple for mean, index for std)
    specs = [
        ("error",    "Classification Error (%)", 2, 3),
        ("accuracy", "Top-1 Accuracy (%)",       0, 1),
        ("macro_f1", "Macro F1 (%)",              4, 5),
    ]

    paths = []
    for metric_key, ylabel, mean_idx, std_idx in specs:
        means = np.array([results[nw][mean_idx] for nw in worker_counts])
        stds  = np.array([results[nw][std_idx]  for nw in worker_counts])
        stem  = f"{metric_key}_vs_num_workers_{tag}"

        fig, ax = plt.subplots()
        ax.plot(worker_counts, means, marker="o", color="steelblue")
        ax.fill_between(worker_counts, means - stds, means + stds,
                        alpha=0.25, color="steelblue")
        ax.set_xlabel("Number of workers")
        ax.set_ylabel(ylabel)
        ax.set_xticks(worker_counts)
        ax.set_title(stem)
        fig.tight_layout()

        path = os.path.join(out_dir, f"{stem}.pdf")
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    return paths


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # spawn is called once here; the outer sweep loop must not call it again.
    mp.set_start_method("spawn")

    args = parse_args()

    out_dir = os.path.join(_SCRIPTS_DIR, "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Download dataset once in the main process to avoid parallel-download races.
    torchvision.datasets.CIFAR10(root=args.data_root, train=True,  download=True)
    torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True)

    # results[num_workers] = (mean_acc, std_acc, mean_err, std_err, mean_f1, std_f1)
    # std computed with ddof=1 (sample std) over 5 runs.
    results = {}

    for num_workers in range(args.min_workers, args.max_workers + 1):
        run_metrics = []  # (acc, error, f1) per seed

        for seed in SEEDS:
            print(f"[workers={num_workers}, seed={seed}]  training...", flush=True)
            t0 = time.time()
            model = _run_one(num_workers, seed, args)
            # Recompute BN running stats race-free in the main process before
            # evaluation.  Buffers only — no weights or gradients touched.
            torch.optim.swa_utils.update_bn(
                get_bn_train_loader(args.data_root), model
            )
            acc, error, f1 = _evaluate(model, args.data_root)
            run_metrics.append((acc, error, f1))
            print(
                f"[workers={num_workers}, seed={seed}]  "
                f"acc={acc:.2f}%  error={error:.2f}%  F1={f1:.2f}%  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

        accs   = np.array([m[0] for m in run_metrics])
        errors = np.array([m[1] for m in run_metrics])
        f1s    = np.array([m[2] for m in run_metrics])

        # ddof=1: sample std (unbiased) over the 5 seeds
        mean_acc, std_acc = accs.mean(),   accs.std(ddof=1)
        mean_err, std_err = errors.mean(), errors.std(ddof=1)
        mean_f1,  std_f1  = f1s.mean(),    f1s.std(ddof=1)

        results[num_workers] = (mean_acc, std_acc, mean_err, std_err, mean_f1, std_f1)

        print(
            f"\n[workers={num_workers}]  "
            f"acc={mean_acc:.2f}±{std_acc:.2f}%  "
            f"error={mean_err:.2f}±{std_err:.2f}%  "
            f"F1={mean_f1:.2f}±{std_f1:.2f}%\n",
            flush=True,
        )

        # Overwrite CSV after each worker count so a late crash doesn't lose data.
        csv_path = _write_csv(results, args, out_dir)
        print(f"  CSV updated → {csv_path}", flush=True)

    plot_paths = _plot_metrics(results, args, out_dir)
    for p in plot_paths:
        print(f"  Plot saved  → {p}", flush=True)
