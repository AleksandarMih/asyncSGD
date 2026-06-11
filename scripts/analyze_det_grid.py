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

COL1, COL2 = 3.5, 7.0  # single / double column width (inches)
plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "lines.linewidth":  1.2,
    "axes.linewidth":   0.7,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})


def _save(fig, path: str) -> None:
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")


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
    fig, ax = plt.subplots(figsize=(COL2, 2.8))
    for beta, grp in stats.groupby("momentum"):
        grp = grp.sort_values("tau")
        ax.plot(grp["tau"], grp["mean"], marker="o", label=f"β={beta}")
        ax.fill_between(grp["tau"],
                        grp["mean"] - grp["std"],
                        grp["mean"] + grp["std"],
                        alpha=0.2)
    ax.set_xlabel("Gradient delay τ")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Final test accuracy vs delay, by momentum")
    ax.set_xticks([0, 2, 5, 10, 20])
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_A_acc_vs_tau.png")
    _save(fig, path)
    plt.close(fig)


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

    fig, ax = plt.subplots(figsize=(COL2, 2.8))
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
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot C — three-line accuracy vs τ  (headline result)
# ---------------------------------------------------------------------------

def plot_C(stats: pd.DataFrame, out_dir: str) -> None:
    BETAS  = [0.0, 0.5, 0.9]
    COLORS = {"0.0": "#2166ac", "0.5": "#f4a261", "0.9": "#d62728"}
    LABELS = {"0.0": "β = 0 (no momentum)", "0.5": "β = 0.5", "0.9": "β = 0.9 (high momentum)"}

    eta = 0.1  # base LR — used to draw Liu bound thresholds

    fig, ax = plt.subplots(figsize=(COL2, 3.0))

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
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot Q1a — learning curves (test accuracy vs epoch) for β=0 and β=0.9
# ---------------------------------------------------------------------------

def plot_Q1_curves(df: pd.DataFrame, out_dir: str) -> None:
    TAUS   = [0, 2, 5, 10, 20]
    BETAS  = [0.0, 0.9]
    TITLES = {0.0: "β = 0 (no momentum)", 0.9: "β = 0.9 (high momentum)"}

    cmap   = plt.cm.Blues
    colors = {tau: cmap(0.35 + 0.65 * i / (len(TAUS) - 1)) for i, tau in enumerate(TAUS)}

    curves = (
        df.groupby(["tau", "momentum", "epoch"])["test_acc"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0), sharey=True)

    for ax, beta in zip(axes, BETAS):
        sub = curves[curves["momentum"] == beta]
        for tau in TAUS:
            grp = sub[sub["tau"] == tau].sort_values("epoch")
            if grp.empty:
                continue
            col = colors[tau]
            ax.plot(grp["epoch"], grp["mean"], color=col, linewidth=1.8,
                    label=f"τ={tau}", zorder=3)
            ax.fill_between(grp["epoch"],
                            grp["mean"] - grp["std"].fillna(0),
                            grp["mean"] + grp["std"].fillna(0),
                            color=col, alpha=0.15, zorder=2)

        for milestone in [30, 40]:
            ax.axvline(milestone, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_title(TITLES[beta], fontsize=11)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.9)

    axes[0].set_ylabel("Test accuracy (%)", fontsize=10)
    fig.suptitle("Convergence curves: test accuracy vs epoch (deterministic delay)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_Q1_convergence_curves.png")
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot Q1b — convergence speed metrics: epochs-to-threshold + normalised AUC
# ---------------------------------------------------------------------------

def plot_Q1_convergence_metrics(df: pd.DataFrame, out_dir: str) -> None:
    TAUS       = [0, 2, 5, 10, 20]
    THRESHOLDS = [80.0, 85.0]
    BETA       = 0.0

    curves = (
        df[df["momentum"] == BETA]
        .groupby(["tau", "epoch"])["test_acc"]
        .mean()
        .reset_index()
    )

    # --- epochs-to-threshold ---
    epochs_to = {}
    for tau in TAUS:
        grp  = curves[curves["tau"] == tau].sort_values("epoch")
        row  = {}
        for thr in THRESHOLDS:
            hit = grp[grp["test_acc"] >= thr]
            row[thr] = int(hit["epoch"].iloc[0]) if not hit.empty else None
        epochs_to[tau] = row

    # --- normalised AUC ---
    auc_vals = {}
    for tau in TAUS:
        grp = curves[curves["tau"] == tau].sort_values("epoch")
        try:
            _trapz = np.trapezoid
        except AttributeError:
            _trapz = lambda y, x: sum((y[i]+y[i+1])*(x[i+1]-x[i])/2 for i in range(len(y)-1))
        auc_vals[tau] = _trapz(grp["test_acc"].values, grp["epoch"].values)
    auc_norm = {tau: auc_vals[tau] / auc_vals[0] for tau in TAUS}

    # --- plot ---
    x      = np.arange(len(TAUS))
    width  = 0.35
    colors_thr = ["#4393c3", "#2166ac"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 3.0))

    for i, thr in enumerate(THRESHOLDS):
        vals = []
        dnr_pos = []
        for j, tau in enumerate(TAUS):
            v = epochs_to[tau][thr]
            if v is None:
                vals.append(0)
                dnr_pos.append(j)
            else:
                vals.append(v)
        bars = ax1.bar(x + (i - 0.5) * width, vals, width,
                       label=f"≥{thr:.0f}%", color=colors_thr[i], alpha=0.85)
        for j in dnr_pos:
            ax1.text(x[j] + (i - 0.5) * width, 3, "DNR",
                     ha="center", va="bottom", fontsize=8, color="crimson",
                     fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"τ={t}" for t in TAUS])
    ax1.set_ylabel("First epoch ≥ threshold")
    ax1.set_title("Epochs to reach accuracy threshold\n(β=0, DNR = did not reach)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    auc_y = [auc_norm[t] for t in TAUS]
    bar_colors = [plt.cm.Blues(0.35 + 0.65 * i / (len(TAUS) - 1)) for i in range(len(TAUS))]
    ax2.bar(x, auc_y, color=bar_colors, alpha=0.85)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="τ=0 baseline")
    for j, (tau, v) in enumerate(zip(TAUS, auc_y)):
        ax2.text(j, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"τ={t}" for t in TAUS])
    ax2.set_ylabel("Normalised AUC (relative to τ=0)")
    ax2.set_title("Training efficiency: area under learning curve\n(β=0, normalised to no-delay baseline)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "plot_Q1_convergence_metrics.png")
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot Q1c — train-test gap vs epoch (generalisation check, appendix)
# ---------------------------------------------------------------------------

def plot_Q1_gap(df: pd.DataFrame, out_dir: str) -> None:
    TAUS = [0, 2, 5, 10, 20]
    BETA = 0.0

    sub = df[df["momentum"] == BETA].copy()
    sub["gap"] = sub["train_acc"] - sub["test_acc"]

    curves = (
        sub.groupby(["tau", "epoch"])["gap"]
        .mean()
        .reset_index()
    )

    cmap   = plt.cm.Blues
    colors = {tau: cmap(0.35 + 0.65 * i / (len(TAUS) - 1)) for i, tau in enumerate(TAUS)}

    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    for tau in TAUS:
        grp = curves[curves["tau"] == tau].sort_values("epoch")
        ax.plot(grp["epoch"], grp["gap"], color=colors[tau], linewidth=1.8,
                label=f"τ={tau}")

    for milestone in [30, 40]:
        ax.axvline(milestone, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Train acc − Test acc (%)", fontsize=10)
    ax.set_title("Generalisation gap vs epoch (β=0, deterministic delay)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_Q1_train_test_gap.png")
    _save(fig, path)
    plt.close(fig)


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
    plot_Q1_curves(df,                  args.out_dir)
    plot_Q1_convergence_metrics(df,     args.out_dir)
    plot_Q1_gap(df,                     args.out_dir)


if __name__ == "__main__":
    main()
