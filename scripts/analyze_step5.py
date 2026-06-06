"""
Analyze Step 5 results — Mitliagkas equivalence in the HogWild! setting.

Run after sweep_step5.sh completes:
  python scripts/analyze_step5.py

Plots saved to outputs/step5/:
  plot_K_equivalence.png   — per-k panel: HogWild! vs sequential loss + accuracy curves
  plot_L_final_acc.png     — grouped bar chart of final test accuracy across k and mode
"""

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

def load_all(in_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(in_dir, "step5_*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No step5 CSVs in {in_dir!r}. Run sweep_step5.sh first.")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["num_workers"] = df["num_workers"].astype(int)
    df["momentum"]    = df["momentum"].astype(float)
    df["seed"]        = df["seed"].astype(int)
    df["epoch"]       = df["epoch"].astype(int)
    df["train_loss"]  = df["train_loss"].astype(float)
    df["test_acc"]    = df["test_acc"].astype(float)
    print(f"Loaded {len(paths)} files, {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# Plot K — per-k equivalence curves (headline result)
# ---------------------------------------------------------------------------

def plot_K(df: pd.DataFrame, out_dir: str) -> None:
    """
    One column per k value, two rows: train loss and test accuracy.
    Solid = HogWild! (β=0); dashed = sequential (β=(k-1)/k).
    Shaded bands = ±1 std across seeds.

    If the Mitliagkas prediction holds, solid and dashed lines should overlap.
    """
    ks = sorted(df["num_workers"].unique())
    fig, axes = plt.subplots(2, len(ks), figsize=(5 * len(ks), 8), sharey="row")

    color_hw  = "steelblue"
    color_seq = "tomato"

    for col, k in enumerate(ks):
        sub = df[df["num_workers"] == k]
        beta_seq = round((k - 1) / k, 6)

        for row, (metric, ylabel) in enumerate([
            ("train_loss", "Train loss"),
            ("test_acc",   "Test accuracy (%)"),
        ]):
            ax = axes[row, col]
            for mode, color, ls, label in [
                ("hogwild",    color_hw,  "-",  f"HogWild! k={k}, β=0"),
                ("sequential", color_seq, "--", f"Sequential β={beta_seq:.3g}"),
            ]:
                grp = (sub[sub["mode"] == mode]
                       .groupby("epoch")[metric]
                       .agg(mean="mean", std="std")
                       .reset_index()
                       .sort_values("epoch"))
                ax.plot(grp["epoch"], grp["mean"], color=color, ls=ls, label=label)
                ax.fill_between(
                    grp["epoch"],
                    grp["mean"] - grp["std"],
                    grp["mean"] + grp["std"],
                    color=color, alpha=0.15,
                )

            if row == 0:
                ax.set_title(f"k = {k}  (β_pred = {beta_seq:.3g})", fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)

    fig.suptitle(
        "Mitliagkas equivalence: HogWild! (β=0) vs sequential (β=(k−1)/k)\n"
        "Overlapping lines = equivalence holds",
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_K_equivalence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot L — final accuracy bar chart
# ---------------------------------------------------------------------------

def plot_L(df: pd.DataFrame, out_dir: str) -> None:
    """
    Grouped bars: groups = k values, bars = (HogWild!, sequential).
    Shows whether the final accuracy of both modes converges.
    """
    max_epoch = df["epoch"].max()
    final = df[df["epoch"] == max_epoch]

    stats = (
        final.groupby(["num_workers", "mode"])["test_acc"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    ks     = sorted(stats["num_workers"].unique())
    modes  = ["hogwild", "sequential"]
    colors = {"hogwild": "steelblue", "sequential": "tomato"}
    x      = np.arange(len(ks))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, mode in enumerate(modes):
        sub = stats[stats["mode"] == mode].set_index("num_workers")
        means = [sub.loc[k, "mean"] if k in sub.index else float("nan") for k in ks]
        stds  = [sub.loc[k, "std"]  if k in sub.index else 0.0           for k in ks]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, color=colors[mode],
                      label=mode, edgecolor="k", linewidth=0.6)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    ecolor="k", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title(f"Final accuracy at epoch {max_epoch}\n"
                 "HogWild! (β=0) vs sequential (β=(k−1)/k)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_L_final_acc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    max_epoch = df["epoch"].max()
    final = df[df["epoch"] == max_epoch]
    tbl = (
        final.groupby(["num_workers", "mode"])["test_acc"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    tbl["result"] = tbl.apply(lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}", axis=1)
    pivot = tbl.pivot(index="num_workers", columns="mode", values="result")
    pivot.index.name = "k"
    print("\nFinal test accuracy (mean ± std over seeds):\n")
    print(pivot.to_string())
    print()

    # Gap column: |hogwild − sequential| mean
    hw  = tbl[tbl["mode"] == "hogwild"].set_index("num_workers")["mean"]
    seq = tbl[tbl["mode"] == "sequential"].set_index("num_workers")["mean"]
    print("Accuracy gap  |HogWild! − sequential|:")
    for k in sorted(hw.index):
        if k in seq.index:
            print(f"  k={k}: {abs(hw[k] - seq[k]):.2f} pp")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",  default="outputs/step5")
    p.add_argument("--out-dir", default="outputs/step5")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_all(args.in_dir)

    print_summary(df)
    plot_K(df, args.out_dir)
    plot_L(df, args.out_dir)


if __name__ == "__main__":
    main()
