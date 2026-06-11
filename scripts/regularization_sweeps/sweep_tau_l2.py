"""
sweep_tau_l2.py — delayed-gradient SGD sweep over (τ, weight_decay) for ResNet-20 on CIFAR-10.

Run:  python scripts/regularization_sweeps/sweep_tau_l2.py
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
    SEEDS,
    train_one_run_delayed,
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

MOMENTUM     = 0.0
QUEUE        = "GEOMETRIC"          # "FIFO" or "GEOMETRIC"
QTAG         = QUEUE.lower()   # used in all output filenames
SWEEP_TAG    = "l2"            # identifies the outer swept variable in filenames

# ---------------------------------------------------------------------------
# Sweep hyperparameters — edit here
# ---------------------------------------------------------------------------

TAUS = [0, 1, 2, 4, 6, 8, 10]
WEIGHT_DECAYS = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

OUT_DIR = os.path.join(_SCRIPTS_DIR, "..", "..", "outputs", "delayed_experiments")
PLOT_DIR = os.path.join(OUT_DIR, "plots")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _delay_label(tau) -> str:
    return f"M={int(tau)+1}" if QUEUE == "GEOMETRIC" else f"τ={tau}"


def fmt_float(x: float) -> str:
    """Format a float for use in filenames (0.0 → '0.0', 0.0001 → '1e-04')."""
    if x == 0.0:
        return "0.0"
    return f"{x:.0e}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary(summary_csv_paths: dict[float, str], save_dir: str) -> None:
    """Single 2-subplot figure: test_acc and test_error vs τ, one line per λ."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5))

    x_label = "Number of workers M" if QUEUE == "GEOMETRIC" else "Delay τ (batches)"
    titles  = (
        ("Test accuracy vs workers M",  "Test error vs workers M")
        if QUEUE == "GEOMETRIC" else
        ("Test accuracy vs delay τ", "Test error vs delay τ")
    )

    for wd, csv_path in sorted(summary_csv_paths.items()):
        df = pd.read_csv(csv_path, index_col="tau")
        taus = df.index.to_numpy(dtype=float)
        x_vals = taus + 1 if QUEUE == "GEOMETRIC" else taus
        label = f"λ={fmt_float(wd)}"

        for ax, metric in [
            (axes[0], "test_acc"),
            (axes[1], "test_error"),
        ]:
            mean = df[f"mean_{metric}"].to_numpy()
            std  = df[f"std_{metric}"].to_numpy()
            line, = ax.plot(x_vals, mean, marker="o", linewidth=1.2, label=label)
            ax.fill_between(x_vals, mean - std, mean + std, alpha=0.2, color=line.get_color())

    for ax, ylabel, title in [
        (axes[0], "Top-1 accuracy", titles[0]),
        (axes[1], "Top-1 error",    titles[1]),
    ]:
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(
        f"τ sweep  |  dropout={DROPOUT}  momentum={MOMENTUM}  epochs={EPOCHS}",
        fontsize=8,
    )
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"summary_{SWEEP_TAG}_tausweep_{QTAG}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_weight_norm(traj_csv_paths: dict[float, str], save_dir: str) -> None:
    """One figure per λ: weight norm over epochs, one curve per τ (mean ± std fill)."""
    os.makedirs(save_dir, exist_ok=True)
    for wd, csv_path in sorted(traj_csv_paths.items()):
        df = pd.read_csv(csv_path)

        fig, ax = plt.subplots(figsize=(COL2, 3.2))
        for tau, grp in df.groupby("tau"):
            epochs = grp["epoch"].to_numpy()
            mean   = grp["mean_weight_norm"].to_numpy()
            std    = grp["std_weight_norm"].to_numpy()
            line, = ax.plot(epochs, mean, linewidth=1.2, label=_delay_label(tau))
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=line.get_color())

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weight L2 norm  ‖w‖")
        ax.set_title(f"Weight norm over training  |  λ={fmt_float(wd)}  dropout={DROPOUT}  momentum={MOMENTUM}")
        ax.legend()
        ax.grid(True, linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        out_path = os.path.join(save_dir, f"weight_norm_{SWEEP_TAG}_tausweep_wd_{fmt_float(wd)}_{QTAG}.png")
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

    # Collect results: (wd, tau, seed) -> DataFrame
    results: dict[tuple[float, int, int], pd.DataFrame] = {}

    for wd in WEIGHT_DECAYS:
        for tau in TAUS:
            for seed in SEEDS:
                df, _, _ = train_one_run_delayed(
                    seed=seed,
                    tau=tau,
                    momentum=MOMENTUM,
                    weight_decay=wd,
                    dropout=DROPOUT,
                    epochs=EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    device=device,
                    data_root=DATA_ROOT,
                    delay_type=("geometric" if QUEUE == "GEOMETRIC" else "fifo"),
                    M=(tau + 1),
                )
                results[(wd, tau, seed)] = df
                final = df.iloc[-1]
                print(
                    f"[τ={tau}, λ={fmt_float(wd)}, seed={seed}]  "
                    f"final test_acc={final['test_acc']:.4f}  "
                    f"test_loss={final['test_loss']:.4f}  "
                    f"‖w‖={final['weight_norm']:.2f}"
                )

    summary_paths: dict[float, str] = {}
    traj_paths: dict[float, str] = {}
    metric_cols = ["train_loss", "test_loss", "test_acc", "test_error", "weight_norm"]

    for wd in WEIGHT_DECAYS:
        # ------------------------------------------------------------------
        # Summary CSV (final-epoch stats, one row per τ)
        # ------------------------------------------------------------------
        summary_rows = []
        for tau in TAUS:
            finals = pd.DataFrame([results[(wd, tau, s)].iloc[-1] for s in SEEDS])
            row: dict = {"tau": tau}
            for col in ("test_loss", "test_acc", "test_error"):
                row[f"mean_{col}"] = finals[col].mean()
                row[f"std_{col}"]  = finals[col].std(ddof=1)
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows).set_index("tau")

        summary_name = (
            f"test_delayed_d_{fmt_float(DROPOUT)}_wd_{fmt_float(wd)}"
            f"_m_{fmt_float(MOMENTUM)}_{SWEEP_TAG}_tausweep_{QTAG}.csv"
        )
        summary_path = os.path.join(OUT_DIR, summary_name)
        summary_df.to_csv(summary_path)
        print(f"Saved {summary_path}")
        summary_paths[wd] = summary_path

        # ------------------------------------------------------------------
        # Trajectory CSV (mean/std per (τ, epoch))
        # ------------------------------------------------------------------
        traj_rows = []
        for tau in TAUS:
            all_dfs = pd.concat(
                [results[(wd, tau, s)].assign(seed=s) for s in SEEDS]
            )
            for epoch, grp in all_dfs.groupby("epoch"):
                row = {"tau": tau, "epoch": epoch}
                for col in metric_cols:
                    row[f"mean_{col}"] = grp[col].mean()
                    row[f"std_{col}"]  = grp[col].std(ddof=1)
                traj_rows.append(row)
        traj_df = pd.DataFrame(traj_rows)

        traj_name = (
            f"traj_delayed_d_{fmt_float(DROPOUT)}_wd_{fmt_float(wd)}"
            f"_m_{fmt_float(MOMENTUM)}_{SWEEP_TAG}_tausweep_{QTAG}.csv"
        )
        traj_path = os.path.join(OUT_DIR, traj_name)
        traj_df.to_csv(traj_path, index=False)
        print(f"Saved {traj_path}")
        traj_paths[wd] = traj_path

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_summary(summary_paths, PLOT_DIR)
    plot_weight_norm(traj_paths, PLOT_DIR)


if __name__ == "__main__":
    main()
