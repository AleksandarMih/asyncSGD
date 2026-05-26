"""
Aggregate Step 3 results and produce three plots.

Run after sweep_step3.sh completes:
  python scripts/analyze_step3.py

Plots saved to outputs/step3/:
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
                                "Run sweep_step3.sh first.")
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
# Plot C — momentum benefit vs τ  (headline result)
# ---------------------------------------------------------------------------

def plot_C(stats: pd.DataFrame, out_dir: str) -> None:
    pivot = stats.pivot(index="tau", columns="momentum", values="mean")

    if 0.9 not in pivot.columns or 0.0 not in pivot.columns:
        print("Warning: β=0.0 or β=0.9 missing from data, skipping Plot C.")
        return

    delta = pivot[0.9] - pivot[0.0]
    taus  = delta.index.tolist()
    vals  = delta.values

    colors = ["steelblue" if v >= 0 else "tomato" for v in vals]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(taus, vals, width=1.5, color=colors, edgecolor="k", linewidth=0.7)
    ax.axhline(0, color="k", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Gradient delay τ")
    ax.set_ylabel("Δ test accuracy  (β=0.9 − β=0)")
    ax.set_title("Momentum benefit vs delay\n(positive = momentum helps)")
    ax.set_xticks(taus)
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
    p.add_argument("--in-dir",  default="outputs/step3")
    p.add_argument("--out-dir", default="outputs/step3")
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
