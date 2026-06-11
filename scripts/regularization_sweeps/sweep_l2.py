"""
sweep_l2.py — L2 / weight-decay sweep for ResNet-20 on CIFAR-10.

Run:  python scripts/regularization_sweeps/sweep_l2.py
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
# Sweep hyperparameters — edit here
# ---------------------------------------------------------------------------

L2_LAMBDAS = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
DROPOUT = 0.0

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
    """Two-subplot figure: test_acc and test_error vs l2_lambda (log x-axis)."""
    df = pd.read_csv(summary_csv_path, index_col="l2_lambda")

    lambdas = df.index.to_numpy(dtype=float)
    # Map λ=0 to a small dummy x-value so it sits left of the log axis
    _ZERO_POS = 1e-6
    x_plot = np.where(lambdas == 0.0, _ZERO_POS, lambdas)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, ylabel, title in [
        (axes[0], "test_acc",   "Top-1 accuracy",   "Test accuracy vs L2 lambda"),
        (axes[1], "test_error", "Top-1 error",       "Test error vs L2 lambda"),
    ]:
        mean = df[f"mean_{metric}"].to_numpy()
        std  = df[f"std_{metric}"].to_numpy()
        ax.semilogx(x_plot, mean, marker="o", linewidth=1.5)
        ax.fill_between(x_plot, mean - std, mean + std, alpha=0.25)
        # Replace dummy tick with "0" label
        ticks = sorted(set(x_plot.tolist()))
        tick_labels = ["0" if t == _ZERO_POS else fmt_float(t) for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("L2 lambda (λ)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"L2 sweep  |  dropout={DROPOUT}  momentum={MOMENTUM}  epochs={EPOCHS}",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"summary_l2sweep_m_{fmt_float(MOMENTUM)}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_weight_norm(traj_csv_path: str, save_dir: str) -> None:
    """One curve per l2_lambda: mean weight-norm over epochs with ±std fill."""
    df = pd.read_csv(traj_csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    for lam, grp in df.groupby("l2_lambda"):
        epochs = grp["epoch"].to_numpy()
        mean   = grp["mean_weight_norm"].to_numpy()
        std    = grp["std_weight_norm"].to_numpy()
        label  = f"λ={fmt_float(lam)}"
        ax.plot(epochs, mean, label=label, linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight L2 norm  ‖w‖")
    ax.set_title(
        f"Weight norm over training  |  dropout={DROPOUT}  momentum={MOMENTUM}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"weight_norm_l2sweep_m_{fmt_float(MOMENTUM)}.png")
    fig.savefig(out_path, dpi=150)
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

    # Collect results: {(l2_lambda, seed): DataFrame}
    results: dict[tuple[float, int], pd.DataFrame] = {}

    for lam in L2_LAMBDAS:
        for seed in SEEDS:
            df = train_one_run(
                seed=seed,
                l2_lambda=lam,
                dropout=DROPOUT,
                momentum=MOMENTUM,
                epochs=EPOCHS,
                lr=LR,
                batch_size=BATCH_SIZE,
                device=device,
                data_root=DATA_ROOT,
            )
            results[(lam, seed)] = df
            final = df.iloc[-1]
            print(
                f"[λ={fmt_float(lam)}, seed={seed}]  "
                f"final test_acc={final['test_acc']:.4f}  "
                f"test_loss={final['test_loss']:.4f}  "
                f"‖w‖={final['weight_norm']:.2f}"
            )

    # ------------------------------------------------------------------
    # Build summary CSV (final-epoch stats, one row per l2_lambda)
    # ------------------------------------------------------------------
    summary_rows = []
    for lam in L2_LAMBDAS:
        finals = pd.DataFrame([results[(lam, s)].iloc[-1] for s in SEEDS])
        row = {"l2_lambda": lam}
        for col in ("test_loss", "test_acc", "test_error"):
            row[f"mean_{col}"] = finals[col].mean()
            row[f"std_{col}"]  = finals[col].std(ddof=1)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).set_index("l2_lambda")

    summary_name = f"test_seq_d_{fmt_float(DROPOUT)}_m_{fmt_float(MOMENTUM)}_l2sweep.csv"
    summary_path = os.path.join(OUT_DIR, summary_name)
    summary_df.to_csv(summary_path)
    print(f"Saved {summary_path}")

    # ------------------------------------------------------------------
    # Build trajectory CSV (mean/std per (l2_lambda, epoch))
    # ------------------------------------------------------------------
    traj_rows = []
    metric_cols = ["train_loss", "test_loss", "test_acc", "test_error", "weight_norm"]
    for lam in L2_LAMBDAS:
        all_dfs = pd.concat(
            [results[(lam, s)].assign(seed=s) for s in SEEDS]
        )
        for epoch, grp in all_dfs.groupby("epoch"):
            row = {"l2_lambda": lam, "epoch": epoch}
            for col in metric_cols:
                row[f"mean_{col}"] = grp[col].mean()
                row[f"std_{col}"]  = grp[col].std(ddof=1)
            traj_rows.append(row)
    traj_df = pd.DataFrame(traj_rows)

    traj_name = f"traj_seq_d_{fmt_float(DROPOUT)}_m_{fmt_float(MOMENTUM)}_l2sweep.csv"
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
