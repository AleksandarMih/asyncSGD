"""
sweep_workers.py — Hogwild! async-SGD sweep over number of workers for ResNet-20 on CIFAR-10.

Phase 1: N-worker sweep at fixed regulariser hyperparameters to test whether
async parallelism acts as an implicit regulariser. The generalization gap plot
(gen_gap_workersweep.png) is the primary scientific output.

Run:  python scripts/regularization_sweeps/sweep_workers.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPTS_DIR, ".."))
sys.path.insert(0, _SCRIPTS_DIR)

from train import (
    BATCH_SIZE,
    DATA_ROOT,
    DROPOUT,
    EPOCHS,
    LR,
    MOMENTUM,
    SEEDS,
    train_one_run_async,
)

WEIGHT_DECAY = 1e-4

# ---------------------------------------------------------------------------
# Sweep hyperparameters — edit here
# ---------------------------------------------------------------------------

NUM_WORKERS_LIST = [1, 2, 4, 8]

OUT_DIR = os.path.join(_SCRIPTS_DIR, "..", "..", "outputs", "async_experiments")
PLOT_DIR = os.path.join(OUT_DIR, "plots")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_float(x: float) -> str:
    """Format a float for use in filenames (0.0 → '0.0', 0.0001 → '1e-04')."""
    if x == 0.0:
        return "0.0"
    return f"{x:.0e}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary(summary_csv_path: str, save_dir: str) -> None:
    """Two-subplot figure: test_acc and test_error vs num_workers (linear x-axis)."""
    df = pd.read_csv(summary_csv_path, index_col="num_workers")
    workers = df.index.to_numpy(dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, ylabel, title in [
        (axes[0], "test_acc",   "Top-1 accuracy", "Test accuracy vs N workers"),
        (axes[1], "test_error", "Top-1 error",    "Test error vs N workers"),
    ]:
        mean = df[f"mean_{metric}"].to_numpy()
        std  = df[f"std_{metric}"].to_numpy()
        ax.plot(workers, mean, marker="o", linewidth=1.5)
        ax.fill_between(workers, mean - std, mean + std, alpha=0.25)
        ax.set_xticks(workers)
        ax.set_xlabel("Number of workers N")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Worker sweep  |  dropout={DROPOUT}  wd={WEIGHT_DECAY}  momentum={MOMENTUM}  epochs={EPOCHS}",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "summary_workersweep.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_weight_norm(traj_csv_path: str, save_dir: str) -> None:
    """One curve per num_workers: mean weight-norm over epochs with ±std fill."""
    df = pd.read_csv(traj_csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n_workers, grp in df.groupby("num_workers"):
        epochs = grp["epoch"].to_numpy()
        mean   = grp["mean_weight_norm"].to_numpy()
        std    = grp["std_weight_norm"].to_numpy()
        line, = ax.plot(epochs, mean, linewidth=1.5, label=f"N={n_workers}")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=line.get_color())

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight L2 norm  ‖w‖")
    ax.set_title(f"Weight norm over training  |  dropout={DROPOUT}  wd={WEIGHT_DECAY}  momentum={MOMENTUM}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "weight_norm_workersweep.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_generalization_gap(traj_csv_path: str, save_dir: str) -> None:
    """Generalization gap (test_loss − train_loss) per epoch, one curve per N.

    The gap shrinking with larger N would be evidence of async-as-regularizer.
    Std propagated as sqrt(std_test_loss² + std_train_loss²) assuming independence.
    """
    df = pd.read_csv(traj_csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n_workers, grp in df.groupby("num_workers"):
        epochs    = grp["epoch"].to_numpy()
        gap_mean  = grp["mean_test_loss"].to_numpy() - grp["mean_train_loss"].to_numpy()
        gap_std   = np.sqrt(grp["std_test_loss"].to_numpy() ** 2 + grp["std_train_loss"].to_numpy() ** 2)
        line, = ax.plot(epochs, gap_mean, linewidth=1.5, label=f"N={n_workers}")
        ax.fill_between(epochs, gap_mean - gap_std, gap_mean + gap_std,
                        alpha=0.15, color=line.get_color())

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Generalization gap  (test loss − train loss)")
    ax.set_title(
        f"Generalization gap vs epoch  |  dropout={DROPOUT}  wd={WEIGHT_DECAY}  momentum={MOMENTUM}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "gen_gap_workersweep.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Hogwild! requires shared CPU memory — do not use CUDA here.
    device = torch.device("cpu")
    print("Device: cpu (Hogwild! requires shared memory)")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Collect results: (num_workers, seed) -> DataFrame
    results: dict[tuple[int, int], pd.DataFrame] = {}

    for n_workers in NUM_WORKERS_LIST:
        for seed in SEEDS:
            df = train_one_run_async(
                seed=seed,
                num_workers=n_workers,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
                dropout=DROPOUT,
                epochs=EPOCHS,
                lr=LR,
                batch_size=BATCH_SIZE,
                device=device,
                data_root=DATA_ROOT,
            )
            results[(n_workers, seed)] = df
            final = df.iloc[-1]
            print(
                f"[N={n_workers}, seed={seed}]  "
                f"final test_acc={final['test_acc']:.4f}  "
                f"test_loss={final['test_loss']:.4f}  "
                f"‖w‖={final['weight_norm']:.2f}"
            )

    # ------------------------------------------------------------------
    # Build summary CSV (final-epoch stats, one row per num_workers)
    # ------------------------------------------------------------------
    summary_rows = []
    for n_workers in NUM_WORKERS_LIST:
        finals = pd.DataFrame([results[(n_workers, s)].iloc[-1] for s in SEEDS])
        row: dict = {"num_workers": n_workers}
        for col in ("test_loss", "test_acc", "test_error"):
            row[f"mean_{col}"] = finals[col].mean()
            row[f"std_{col}"]  = finals[col].std(ddof=1)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).set_index("num_workers")

    summary_name = (
        f"test_async_d_{fmt_float(DROPOUT)}_wd_{fmt_float(WEIGHT_DECAY)}"
        f"_m_{fmt_float(MOMENTUM)}_workersweep.csv"
    )
    summary_path = os.path.join(OUT_DIR, summary_name)
    summary_df.to_csv(summary_path)
    print(f"Saved {summary_path}")

    # ------------------------------------------------------------------
    # Build trajectory CSV (mean/std per (num_workers, epoch))
    # ------------------------------------------------------------------
    traj_rows = []
    metric_cols = ["train_loss", "test_loss", "test_acc", "test_error", "weight_norm"]
    for n_workers in NUM_WORKERS_LIST:
        all_dfs = pd.concat(
            [results[(n_workers, s)].assign(seed=s) for s in SEEDS]
        )
        for epoch, grp in all_dfs.groupby("epoch"):
            row = {"num_workers": n_workers, "epoch": epoch}
            for col in metric_cols:
                row[f"mean_{col}"] = grp[col].mean()
                row[f"std_{col}"]  = grp[col].std(ddof=1)
            traj_rows.append(row)
    traj_df = pd.DataFrame(traj_rows)

    traj_name = (
        f"traj_async_d_{fmt_float(DROPOUT)}_wd_{fmt_float(WEIGHT_DECAY)}"
        f"_m_{fmt_float(MOMENTUM)}_workersweep.csv"
    )
    traj_path = os.path.join(OUT_DIR, traj_name)
    traj_df.to_csv(traj_path, index=False)
    print(f"Saved {traj_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_summary(summary_path, PLOT_DIR)
    plot_weight_norm(traj_path, PLOT_DIR)
    plot_generalization_gap(traj_path, PLOT_DIR)


if __name__ == "__main__":
    main()
