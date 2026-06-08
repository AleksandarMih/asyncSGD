"""
Aggregate Step 3 results and produce three plots.

Run after sweep_det_grid.sh completes:
  python scripts/analyze_det_grid.py

Plots saved to outputs/det_grid/:
  plot_A_acc_vs_tau.png       — final test accuracy vs τ, one line per β
  plot_B_loss_curves_tau10.png — training loss over epochs at τ=10, by β
  plot_C_momentum_benefit.png  — acc(β=0.9) − acc(β=0) vs τ  [headline result]
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_csvs(in_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(in_dir, "tau*_beta*_seed*.csv")))
    if not paths:
        raise FileNotFoundError(f"No result CSVs found in {in_dir!r}. "
                                "Run sweep_det_grid.sh first.")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["tau"]      = df["tau"].astype(int)
    df["momentum"] = df["momentum"].astype(float)
    df["seed"]     = df["seed"].astype(int)
    df["epoch"]    = df["epoch"].astype(int)
    print(f"Loaded {len(paths)} files, {len(df)} rows total.")
    return df


def compute_final_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of final-epoch test accuracy, grouped by (tau, momentum)."""
    final = df[df["epoch"] == df["epoch"].max()].copy()
    stats = (
        final.groupby(["tau", "momentum"])["test_acc"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    return stats


# ---------------------------------------------------------------------------
# Plot A — final accuracy vs τ
# ---------------------------------------------------------------------------

def plot_A(stats: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    for beta, grp in stats.groupby("momentum"):
        grp = grp.sort_values("tau")
        ax.plot(grp["tau"], grp["mean"], marker="o", label=f"β={beta}")
        ax.fill_between(
            grp["tau"],
            grp["mean"] - grp["std"],
            grp["mean"] + grp["std"],
            alpha=0.2,
        )

    ax.set_xlabel("Gradient delay τ")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Final test accuracy vs delay, by momentum")
    ax.set_xticks([0, 2, 5, 10, 20])
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_A_acc_vs_tau.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot B — training loss curves at τ=10
# ---------------------------------------------------------------------------

def plot_B(df: pd.DataFrame, out_dir: str, tau: int = 10) -> None:
    sub = df[df["tau"] == tau]
    if sub.empty:
        print(f"Warning: no data for τ={tau}, skipping Plot B.")
        return

    curves = (
        sub.groupby(["momentum", "epoch"])["train_loss"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for beta, grp in curves.groupby("momentum"):
        grp = grp.sort_values("epoch")
        ax.plot(grp["epoch"], grp["mean"], label=f"β={beta}")
        ax.fill_between(
            grp["epoch"],
            grp["mean"] - grp["std"],
            grp["mean"] + grp["std"],
            alpha=0.2,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title(f"Training loss curves at τ={tau}, by momentum")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"plot_B_loss_curves_tau{tau}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot C — three-line accuracy vs τ  (headline result)
# ---------------------------------------------------------------------------

def plot_C(stats: pd.DataFrame, out_dir: str) -> None:
    BETAS  = [0.0, 0.5, 0.9]
    COLORS = {"0.0": "#2166ac", "0.5": "#f4a261", "0.9": "#d62728"}
    LABELS = {"0.0": "β = 0 (no momentum)", "0.5": "β = 0.5", "0.9": "β = 0.9 (high momentum)"}

    eta = 0.1  # base LR — used to draw Liu bound thresholds

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for beta in BETAS:
        key  = f"{beta:.1f}"
        grp  = stats[stats["momentum"] == beta].sort_values("tau")
        if grp.empty:
            continue
        taus = grp["tau"].values
        mean = grp["mean"].values
        std  = grp["std"].fillna(0).values
        col  = COLORS[key]

        ax.plot(taus, mean, marker="o", color=col, linewidth=2,
                markersize=6, label=LABELS[key], zorder=3)
        ax.fill_between(taus, mean - std, mean + std,
                        color=col, alpha=0.15, zorder=2)

        # Annotate final value at τ=20
        ax.annotate(f"{mean[-1]:.1f}%", xy=(taus[-1], mean[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=col, va="center")

    ax.set_xlabel("Gradient delay τ", fontsize=11)
    ax.set_ylabel("Final test accuracy (%)", fontsize=11)
    ax.set_title("Momentum collapses under delay\n"
                 "β = 0.9 best at τ = 0, worst at τ ≥ 5", fontsize=11)
    ax.set_xticks([0, 2, 5, 10, 20])
    ax.set_ylim(bottom=max(0, stats["mean"].min() - 8))

    # Liu bound threshold: τ_unsafe = (1−β)²/η  — drawn after ylim is set
    y0 = ax.get_ylim()[0]
    for beta, col in [(0.5, COLORS["0.5"]), (0.9, COLORS["0.9"])]:
        tau_thresh = (1 - beta) ** 2 / eta
        if tau_thresh < max(stats["tau"]):
            ax.axvline(tau_thresh, color=col, linestyle=":", linewidth=1.2,
                       alpha=0.7, zorder=1)
            ax.text(tau_thresh + 0.2, y0 + 2,
                    "Liu\nlimit", fontsize=6.5, color=col,
                    va="bottom", ha="left", linespacing=1.2)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "plot_C_momentum_benefit.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(stats: pd.DataFrame) -> None:
    pivot = stats.pivot(index="tau", columns="momentum", values="mean")
    pivot.columns = [f"β={c}" for c in pivot.columns]
    pivot.index.name = "τ"
    print("\nMean test accuracy (%) — averaged over seeds:\n")
    print(pivot.round(2).to_string())
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",  default="outputs/det_grid")
    p.add_argument("--out-dir", default="outputs/plots")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df    = load_all_csvs(args.in_dir)
    stats = compute_final_stats(df)

    print_summary(stats)
    plot_A(stats, args.out_dir)
    plot_B(df,    args.out_dir)
    plot_C(stats, args.out_dir)


if __name__ == "__main__":
    main()
