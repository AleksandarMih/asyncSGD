"""
sweep_dropout.py — dropout-rate sweep for ResNet-20 on CIFAR-10.

Run:  python scripts/regularization_sweeps/sweep_dropout.py
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
    EPOCHS,
    LR,
    MOMENTUM,
    SEEDS,
    train_one_run,
)

# ---------------------------------------------------------------------------
# Style — matches make_figures.py
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL1 = 3.5   # single-column width (inches)
COL2 = 7.0   # double-column width (inches)

# ---------------------------------------------------------------------------
# Sweep hyperparameters — edit here
# ---------------------------------------------------------------------------

DROPOUT_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
L2_LAMBDA = 0.0

OUT_DIR = os.path.join(_SCRIPTS_DIR, "..", "..", "outputs", "seq_experiments")
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
    """Two-subplot figure: test_acc and test_error vs dropout rate (linear x-axis)."""
    df = pd.read_csv(summary_csv_path, index_col="dropout")

    dropout_vals = df.index.to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5))

    for ax, metric, ylabel, title in [
        (axes[0], "test_acc",   "Top-1 accuracy", "Test accuracy vs dropout rate"),
        (axes[1], "test_error", "Top-1 error",    "Test error vs dropout rate"),
    ]:
        mean = df[f"mean_{metric}"].to_numpy()
        std  = df[f"std_{metric}"].to_numpy()
        ax.plot(dropout_vals, mean, marker="o", linewidth=1.2)
        ax.fill_between(dropout_vals, mean - std, mean + std, alpha=0.25)
        ax.set_xlabel("Dropout rate (p)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(
        f"Dropout sweep  |  L2={fmt_float(L2_LAMBDA)}  momentum={MOMENTUM}  epochs={EPOCHS}",
        fontsize=8,
    )
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"summary_dropoutsweep_m_{fmt_float(MOMENTUM)}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_weight_norm(traj_csv_path: str, save_dir: str) -> None:
    """One curve per dropout rate: mean weight-norm over epochs with ±std fill."""
    df = pd.read_csv(traj_csv_path)

    fig, ax = plt.subplots(figsize=(COL2, 3.2))
    for dropout, grp in df.groupby("dropout"):
        epochs = grp["epoch"].to_numpy()
        mean   = grp["mean_weight_norm"].to_numpy()
        std    = grp["std_weight_norm"].to_numpy()
        label  = f"p={fmt_float(dropout)}"
        ax.plot(epochs, mean, label=label, linewidth=1.2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight L2 norm  ‖w‖")
    ax.set_title(
        f"Weight norm over training  |  L2={fmt_float(L2_LAMBDA)}  momentum={MOMENTUM}"
    )
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"weight_norm_dropoutsweep_m_{fmt_float(MOMENTUM)}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Collect results: {(dropout, seed): DataFrame}
    results: dict[tuple[float, int], pd.DataFrame] = {}

    for dropout in DROPOUT_RATES:
        for seed in SEEDS:
            df = train_one_run(
                seed=seed,
                l2_lambda=L2_LAMBDA,
                dropout=dropout,
                momentum=MOMENTUM,
                epochs=EPOCHS,
                lr=LR,
                batch_size=BATCH_SIZE,
                device=device,
                data_root=DATA_ROOT,
            )
            results[(dropout, seed)] = df
            final = df.iloc[-1]
            print(
                f"[dropout={fmt_float(dropout)}, seed={seed}]  "
                f"final test_acc={final['test_acc']:.4f}  "
                f"test_loss={final['test_loss']:.4f}  "
                f"‖w‖={final['weight_norm']:.2f}"
            )

    # ------------------------------------------------------------------
    # Build summary CSV (final-epoch stats, one row per dropout rate)
    # ------------------------------------------------------------------
    summary_rows = []
    for dropout in DROPOUT_RATES:
        finals = pd.DataFrame([results[(dropout, s)].iloc[-1] for s in SEEDS])
        row = {"dropout": dropout}
        for col in ("test_loss", "test_acc", "test_error"):
            row[f"mean_{col}"] = finals[col].mean()
            row[f"std_{col}"]  = finals[col].std(ddof=1)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).set_index("dropout")

    summary_name = f"test_seq_l2_{fmt_float(L2_LAMBDA)}_m_{fmt_float(MOMENTUM)}_dropoutsweep.csv"
    summary_path = os.path.join(OUT_DIR, summary_name)
    summary_df.to_csv(summary_path)
    print(f"Saved {summary_path}")

    # ------------------------------------------------------------------
    # Build trajectory CSV (mean/std per (dropout, epoch))
    # ------------------------------------------------------------------
    traj_rows = []
    metric_cols = ["train_loss", "test_loss", "test_acc", "test_error", "weight_norm"]
    for dropout in DROPOUT_RATES:
        all_dfs = pd.concat(
            [results[(dropout, s)].assign(seed=s) for s in SEEDS]
        )
        for epoch, grp in all_dfs.groupby("epoch"):
            row = {"dropout": dropout, "epoch": epoch}
            for col in metric_cols:
                row[f"mean_{col}"] = grp[col].mean()
                row[f"std_{col}"]  = grp[col].std(ddof=1)
            traj_rows.append(row)
    traj_df = pd.DataFrame(traj_rows)

    traj_name = f"traj_seq_l2_{fmt_float(L2_LAMBDA)}_m_{fmt_float(MOMENTUM)}_dropoutsweep.csv"
    traj_path = os.path.join(OUT_DIR, traj_name)
    traj_df.to_csv(traj_path, index=False)
    print(f"Saved {traj_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_summary(summary_path, PLOT_DIR)
    plot_weight_norm(traj_path, PLOT_DIR)


if __name__ == "__main__":
    main()
