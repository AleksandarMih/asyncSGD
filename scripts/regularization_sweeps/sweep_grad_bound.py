"""
sweep_grad_bound.py — sweep delay τ and overlay the theoretical gradient-norm bound.

Run:  python scripts/regularization_sweeps/sweep_grad_bound.py
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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
    SEEDS,
    train_one_run_delayed,
)

# ---------------------------------------------------------------------------
# Sweep hyperparameters — edit here
# ---------------------------------------------------------------------------

TAUS         = [0, 1, 2, 4, 8, 10]
MOMENTUM     = 0.0
WEIGHT_DECAY = 0.0
DROPOUT      = 0.0
QUEUE        = "GEOMETRIC"          # "FIFO" or "GEOMETRIC"
QTAG         = QUEUE.lower()   # used in all output filenames

OUT_DIR  = os.path.join(_SCRIPTS_DIR, "..", "..", "outputs", "grad_bound")
PLOT_DIR = os.path.join(OUT_DIR, "plots")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _delay_label(tau) -> str:
    return f"M={int(tau)+1}" if QUEUE == "GEOMETRIC" else f"τ={tau}"


def plot_grad_bound(
    traj_csv_path: str,
    save_dir: str,
    w0_mean: float,
    G: float,
    steps_per_epoch: int,
) -> None:
    """Two-subplot figure: weight norm and mean squared gradient norm vs epoch."""
    df = pd.read_csv(traj_csv_path)

    fig, (ax_wn, ax_gn) = plt.subplots(1, 2, figsize=(12, 5))

    for tau, grp in df.groupby("tau"):
        epochs = grp["epoch"].to_numpy()
        label = _delay_label(tau)

        # Left: weight norm
        mean_wn = grp["mean_weight_norm"].to_numpy()
        std_wn  = grp["std_weight_norm"].to_numpy()
        ax_wn.plot(epochs, mean_wn, label=label, linewidth=1.5)
        ax_wn.fill_between(epochs, mean_wn - std_wn, mean_wn + std_wn, alpha=0.15)

        # Right: mean squared gradient norm
        mean_gs = grp["mean_mean_grad_sq"].to_numpy()
        std_gs  = grp["std_mean_grad_sq"].to_numpy()
        ax_gn.plot(epochs, mean_gs, label=label, linewidth=1.5)
        ax_gn.fill_between(epochs, mean_gs - std_gs, mean_gs + std_gs, alpha=0.15)

    # Theoretical bound overlay on left subplot
    bound_epochs = np.arange(0, EPOCHS + 1)
    bound_t      = bound_epochs * steps_per_epoch  # SGD steps
    bound_y      = w0_mean + LR * G * bound_t
    ax_wn.plot(bound_epochs, bound_y, color="black", linestyle="--", linewidth=1.5,
               label="Bound: ‖w₀‖ + η G t")

    ax_wn.set_xlabel("Epoch")
    ax_wn.set_ylabel("Weight L2 norm  ‖w‖")
    ax_wn.set_title("Weight norm with theoretical bound  |  β=0  wd=0")
    ax_wn.legend(fontsize=8)
    ax_wn.grid(True, alpha=0.3)

    ax_gn.set_xlabel("Epoch")
    ax_gn.set_ylabel("Mean ‖∇f‖²")
    ax_gn.set_title("Per-step squared gradient norm  |  β=0  wd=0")
    ax_gn.legend(fontsize=8)
    ax_gn.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"grad_bound_{QTAG}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_weight_norm(
    df: pd.DataFrame,
    ax: Axes,
    bound_y: np.ndarray | None,
    bound_label: str | None,
) -> None:
    """Single-axes weight-norm plot, optionally with a bound overlay."""
    for tau, grp in df.groupby("tau"):
        epochs  = grp["epoch"].to_numpy()
        mean_wn = grp["mean_weight_norm"].to_numpy()
        std_wn  = grp["std_weight_norm"].to_numpy()
        ax.plot(epochs, mean_wn, label=_delay_label(tau), linewidth=1.5)
        ax.fill_between(epochs, mean_wn - std_wn, mean_wn + std_wn, alpha=0.15)

    if bound_y is not None:
        epochs_all = np.arange(0, int(df["epoch"].max()) + 1)
        ax.plot(epochs_all, bound_y, color="black", linestyle="--",
                linewidth=1.5, label=bound_label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight L2 norm  ‖w‖")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Collect results: {(tau, seed): (df_epoch, grad_sq_per_step, w0_norm)}
    results: dict[tuple[int, int], tuple[pd.DataFrame, np.ndarray, float]] = {}

    for tau in TAUS:
        for seed in SEEDS:
            df, grad_sq, w0_norm = train_one_run_delayed(
                seed=seed,
                tau=tau,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
                dropout=DROPOUT,
                epochs=EPOCHS,
                lr=LR,
                batch_size=BATCH_SIZE,
                device=device,
                data_root=DATA_ROOT,
                delay_type=("geometric" if QUEUE == "GEOMETRIC" else "fifo"),
                M=(tau + 1),
            )
            results[(tau, seed)] = (df, grad_sq, w0_norm)
            final = df.iloc[-1]
            print(
                f"[τ={tau}, seed={seed}]  "
                f"final test_acc={final['test_acc']:.4f}  "
                f"test_loss={final['test_loss']:.4f}  "
                f"‖w‖={final['weight_norm']:.2f}  "
                f"mean_grad²={final['mean_grad_sq']:.4f}"
            )

    # ------------------------------------------------------------------
    # Estimate G from τ=0 runs (95th-percentile of per-step ‖grad‖²)
    # ------------------------------------------------------------------
    tau0_all_grad_sq = np.concatenate([results[(0, s)][1] for s in SEEDS])
    g_sq_95 = float(np.percentile(tau0_all_grad_sq, 95))
    G = float(np.sqrt(g_sq_95))
    print(f"\nEstimated G = {G:.6f}  (95th-pct grad² = {g_sq_95:.6f})")

    # steps_per_epoch: exact for τ=0 (every batch applies a gradient)
    steps_per_epoch = len(results[(0, SEEDS[0])][1]) // EPOCHS
    w0_mean = float(np.mean([results[(0, s)][2] for s in SEEDS]))
    print(f"steps_per_epoch = {steps_per_epoch},  ‖w₀‖ mean = {w0_mean:.4f},  η = {LR}")

    # ------------------------------------------------------------------
    # Save G estimate artifact
    # ------------------------------------------------------------------
    g_est_path = os.path.join(OUT_DIR, f"G_estimate_{QTAG}.txt")
    with open(g_est_path, "w") as fh:
        fh.write(f"G             = {G}\n")
        fh.write(f"g_sq_95th_pct = {g_sq_95}\n")
        fh.write(f"w0_norm_mean  = {w0_mean}\n")
        fh.write(f"steps_per_epoch = {steps_per_epoch}\n")
        fh.write(f"lr            = {LR}\n")
    print(f"Saved {g_est_path}")

    # ------------------------------------------------------------------
    # Build summary CSV (final-epoch stats, one row per τ)
    # ------------------------------------------------------------------
    summary_rows = []
    for tau in TAUS:
        finals = pd.DataFrame([results[(tau, s)][0].iloc[-1] for s in SEEDS])
        row: dict = {"tau": tau}
        for col in ("test_loss", "test_acc", "test_error", "weight_norm", "mean_grad_sq"):
            row[f"mean_{col}"] = finals[col].mean()
            row[f"std_{col}"]  = finals[col].std(ddof=1)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).set_index("tau")

    summary_path = os.path.join(OUT_DIR, f"summary_grad_bound_{QTAG}.csv")
    summary_df.to_csv(summary_path)
    print(f"Saved {summary_path}")

    # ------------------------------------------------------------------
    # Build trajectory CSV (mean/std per (τ, epoch))
    # ------------------------------------------------------------------
    metric_cols = [
        "train_loss", "test_loss", "test_acc", "test_error",
        "weight_norm", "mean_grad_sq",
    ]
    traj_rows = []
    for tau in TAUS:
        all_dfs = pd.concat(
            [results[(tau, s)][0].assign(seed=s) for s in SEEDS]
        )
        for epoch, grp in all_dfs.groupby("epoch"):
            row = {"tau": tau, "epoch": epoch}
            for col in metric_cols:
                row[f"mean_{col}"] = grp[col].mean()
                row[f"std_{col}"]  = grp[col].std(ddof=1)
            traj_rows.append(row)
    traj_df = pd.DataFrame(traj_rows)

    traj_path = os.path.join(OUT_DIR, f"traj_grad_bound_{QTAG}.csv")
    traj_df.to_csv(traj_path, index=False)
    print(f"Saved {traj_path}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot_grad_bound(traj_path, PLOT_DIR, w0_mean=w0_mean, G=G, steps_per_epoch=steps_per_epoch)

    # ------------------------------------------------------------------
    # Three focused weight-norm plots (no bound / log bound / epoch bound)
    # ------------------------------------------------------------------
    epochs_all = np.arange(0, EPOCHS + 1)
    t_steps    = epochs_all * steps_per_epoch

    # 1. No bound
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_weight_norm(traj_df, ax, bound_y=None, bound_label=None)
    ax.set_title("Weight norm vs epoch  |  β=0  wd=0")
    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"weight_norm_no_bound_{QTAG}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # 2. Log bound: ‖w₀‖ + η·G·log(1 + t)
    bound_log = w0_mean + LR * G * np.log1p(t_steps)
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_weight_norm(traj_df, ax, bound_y=bound_log,
                     bound_label="Bound: ‖w₀‖ + η G log(1+t)")
    ax.set_title("Weight norm — log bound  |  β=0  wd=0")
    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"weight_norm_bound_log_{QTAG}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # 3. Epoch bound: ‖w₀‖ + η·G·epoch
    bound_epoch = w0_mean + LR * G * epochs_all
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_weight_norm(traj_df, ax, bound_y=bound_epoch,
                     bound_label="Bound: ‖w₀‖ + η G · epoch")
    ax.set_title("Weight norm — epoch bound  |  β=0  wd=0")
    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"weight_norm_bound_epoch_{QTAG}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
