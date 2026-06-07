"""
sweep_tau_beta.py — delayed-gradient SGD sweep over (τ, β) for ResNet-20 on CIFAR-10.

Run:  python scripts/sweep_tau_beta.py
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

from delayed_training import (
    BATCH_SIZE,
    DATA_ROOT,
    DROPOUT,
    EPOCHS,
    LR,
    SEEDS,
    WEIGHT_DECAY,
    train_one_run_delayed,
)

# ---------------------------------------------------------------------------
# Sweep hyperparameters — edit here
# ---------------------------------------------------------------------------

TAUS = [0, 1, 2, 4, 8, 16, 32]
MOMENTUMS = [0.0, 0.9]

OUT_DIR = os.path.join(_SCRIPTS_DIR, "..", "outputs", "delayed_experiments")
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

def plot_summary(summary_csv_paths: dict[float, str], save_dir: str) -> None:
    """Single 2-subplot figure: test_acc and test_error vs τ, one line per β."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for beta, csv_path in sorted(summary_csv_paths.items()):
        df = pd.read_csv(csv_path, index_col="tau")
        taus = df.index.to_numpy(dtype=float)
        label = f"β={fmt_float(beta)}"

        for ax, metric, ylabel, title in [
            (axes[0], "test_acc",   "Top-1 accuracy", "Test accuracy vs delay τ"),
            (axes[1], "test_error", "Top-1 error",    "Test error vs delay τ"),
        ]:
            mean = df[f"mean_{metric}"].to_numpy()
            std  = df[f"std_{metric}"].to_numpy()
            line, = ax.plot(taus, mean, marker="o", linewidth=1.5, label=label)
            ax.fill_between(taus, mean - std, mean + std, alpha=0.2, color=line.get_color())

    for ax, ylabel, title in [
        (axes[0], "Top-1 accuracy", "Test accuracy vs delay τ"),
        (axes[1], "Top-1 error",    "Test error vs delay τ"),
    ]:
        ax.set_xlabel("Delay τ (batches)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"τ sweep  |  dropout={DROPOUT}  weight_decay={WEIGHT_DECAY}  epochs={EPOCHS}",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "summary_tausweep.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_weight_norm(traj_csv_paths: dict[float, str], save_dir: str) -> None:
    """One figure per β: weight norm over epochs, one curve per τ (mean ± std fill)."""
    os.makedirs(save_dir, exist_ok=True)
    for beta, csv_path in sorted(traj_csv_paths.items()):
        df = pd.read_csv(csv_path)

        fig, ax = plt.subplots(figsize=(8, 5))
        for tau, grp in df.groupby("tau"):
            epochs = grp["epoch"].to_numpy()
            mean   = grp["mean_weight_norm"].to_numpy()
            std    = grp["std_weight_norm"].to_numpy()
            line, = ax.plot(epochs, mean, linewidth=1.5, label=f"τ={tau}")
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=line.get_color())

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weight L2 norm  ‖w‖")
        ax.set_title(f"Weight norm over training  |  β={fmt_float(beta)}  dropout={DROPOUT}  wd={WEIGHT_DECAY}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(save_dir, f"weight_norm_tausweep_m_{fmt_float(beta)}.png")
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

    # Collect results: (beta, tau, seed) -> DataFrame
    results: dict[tuple[float, int, int], pd.DataFrame] = {}

    for beta in MOMENTUMS:
        for tau in TAUS:
            for seed in SEEDS:
                df = train_one_run_delayed(
                    seed=seed,
                    tau=tau,
                    momentum=beta,
                    weight_decay=WEIGHT_DECAY,
                    dropout=DROPOUT,
                    epochs=EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    device=device,
                    data_root=DATA_ROOT,
                )
                results[(beta, tau, seed)] = df
                final = df.iloc[-1]
                print(
                    f"[τ={tau}, β={fmt_float(beta)}, seed={seed}]  "
                    f"final test_acc={final['test_acc']:.4f}  "
                    f"test_loss={final['test_loss']:.4f}  "
                    f"‖w‖={final['weight_norm']:.2f}"
                )

    summary_paths: dict[float, str] = {}
    traj_paths: dict[float, str] = {}
    metric_cols = ["train_loss", "test_loss", "test_acc", "test_error", "weight_norm"]

    for beta in MOMENTUMS:
        # ------------------------------------------------------------------
        # Summary CSV (final-epoch stats, one row per τ)
        # ------------------------------------------------------------------
        summary_rows = []
        for tau in TAUS:
            finals = pd.DataFrame([results[(beta, tau, s)].iloc[-1] for s in SEEDS])
            row: dict = {"tau": tau}
            for col in ("test_loss", "test_acc", "test_error"):
                row[f"mean_{col}"] = finals[col].mean()
                row[f"std_{col}"]  = finals[col].std(ddof=1)
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows).set_index("tau")

        summary_name = (
            f"test_delayed_d_{fmt_float(DROPOUT)}_wd_{fmt_float(WEIGHT_DECAY)}"
            f"_m_{fmt_float(beta)}_tausweep.csv"
        )
        summary_path = os.path.join(OUT_DIR, summary_name)
        summary_df.to_csv(summary_path)
        print(f"Saved {summary_path}")
        summary_paths[beta] = summary_path

        # ------------------------------------------------------------------
        # Trajectory CSV (mean/std per (τ, epoch))
        # ------------------------------------------------------------------
        traj_rows = []
        for tau in TAUS:
            all_dfs = pd.concat(
                [results[(beta, tau, s)].assign(seed=s) for s in SEEDS]
            )
            for epoch, grp in all_dfs.groupby("epoch"):
                row = {"tau": tau, "epoch": epoch}
                for col in metric_cols:
                    row[f"mean_{col}"] = grp[col].mean()
                    row[f"std_{col}"]  = grp[col].std(ddof=1)
                traj_rows.append(row)
        traj_df = pd.DataFrame(traj_rows)

        traj_name = (
            f"traj_delayed_d_{fmt_float(DROPOUT)}_wd_{fmt_float(WEIGHT_DECAY)}"
            f"_m_{fmt_float(beta)}_tausweep.csv"
        )
        traj_path = os.path.join(OUT_DIR, traj_name)
        traj_df.to_csv(traj_path, index=False)
        print(f"Saved {traj_path}")
        traj_paths[beta] = traj_path

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_summary(summary_paths, PLOT_DIR)
    plot_weight_norm(traj_paths, PLOT_DIR)


if __name__ == "__main__":
    main()
