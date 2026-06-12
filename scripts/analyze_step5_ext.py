"""
Unified analysis of Ilia's Step-5 data (outputs/step5/) combined with the
β=0 baseline + LR-rescue extension (outputs/step5_ext/).

Produces:
  plot_step5_grid.png/pdf      — final accuracy vs τ for β=0 and β=0.9
  plot_lrs_rescue.png/pdf      — LR-rescue: accuracy vs lrs for τ=64/K=65
  plot_lrs_rescue_sweep.png/pdf— rescue efficiency across τ ∈ {4,16,64}

Usage:
  python3 scripts/analyze_step5_ext.py
  python3 scripts/analyze_step5_ext.py --out-dir outputs/plots
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

COL1, COL2 = 3.5, 7.0
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

STEP5_DIR     = "outputs/step5"
STEP5_EXT_DIR = "outputs/step5_ext"


def _parse_step5_filename(path: str) -> dict:
    """Extract metadata from Ilia's step5 filename convention."""
    fname = os.path.basename(path)
    # e.g. step5_fifo_tau64_lrs0.125_beta0.9_seed42.csv
    #      step5_geometric_K65_c4_lrs0.015625_beta0.9_seed42.csv
    info = {}
    m = re.search(r"_seed(\d+)", fname)
    info["seed"] = int(m.group(1)) if m else None
    m = re.search(r"_beta([\d.]+)", fname)
    info["momentum"] = float(m.group(1)) if m else None
    m = re.search(r"_lrs([\d.]+)", fname)
    info["lr_scale"] = float(m.group(1)) if m else 1.0
    if "_fifo_" in fname or "_fifo" in fname:
        info["dist"] = "fifo"
        m = re.search(r"_tau(\d+)", fname)
        info["tau"] = int(m.group(1)) if m else None
        info["M"]   = None
    else:
        info["dist"] = "geometric"
        m = re.search(r"_K(\d+)", fname)
        info["M"]   = int(m.group(1)) if m else None
        info["tau"] = info["M"] - 1 if info["M"] is not None else None
    return info


def load_step5() -> pd.DataFrame:
    """Load Ilia's original step5 CSVs (β=0.9 + τ=64/K=65 LR rescue)."""
    paths = glob.glob(os.path.join(STEP5_DIR, "*.csv"))
    paths = [p for p in paths if "_summary" not in p]
    if not paths:
        print(f"[warn] no CSVs in {STEP5_DIR!r}")
        return pd.DataFrame()
    frames = []
    for p in paths:
        meta = _parse_step5_filename(p)
        df   = pd.read_csv(p)
        for k, v in meta.items():
            if k not in df.columns:
                df[k] = v
            elif df[k].isna().all():
                df[k] = v
        # ensure lr_scale column exists
        if "lr_scale" not in df.columns:
            df["lr_scale"] = meta.get("lr_scale", 1.0)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(f"[step5]     loaded {len(paths)} files, {len(out)} rows")
    return out


def _parse_ext_filename(path: str) -> dict:
    """Extract metadata from step5_ext filename produced by train.py."""
    fname = os.path.basename(path)
    # expA_tau4_beta0.0_seed42.csv
    # expC_tau16_lrscale0.25_beta0.9_seed42.csv
    # expA_geo_M5_beta0.0_seed42.csv
    # expC_geo_M17_lrscale0.25_beta0.0_seed42.csv
    info = {}
    m = re.search(r"_seed(\d+)", fname)
    info["seed"] = int(m.group(1)) if m else None
    m = re.search(r"_beta([\d.]+)", fname)
    info["momentum"] = float(m.group(1)) if m else None
    m = re.search(r"_lrscale([\d.]+)", fname)
    info["lr_scale"] = float(m.group(1)) if m else 1.0
    if "_geo_" in fname:
        info["dist"] = "geometric"
        m = re.search(r"_M(\d+)", fname)
        info["M"]   = int(m.group(1)) if m else None
        info["tau"] = info["M"] - 1 if info["M"] is not None else None
    else:
        info["dist"] = "fifo"
        m = re.search(r"_tau(\d+)", fname)
        info["tau"] = int(m.group(1)) if m else None
        info["M"]   = None
    return info


def load_step5_ext() -> pd.DataFrame:
    """Load new β=0 baseline + full LR rescue CSVs from step5_ext."""
    paths = glob.glob(os.path.join(STEP5_EXT_DIR, "*.csv"))
    paths = [p for p in paths if "_summary" not in p]
    if not paths:
        print(f"[warn] no CSVs in {STEP5_EXT_DIR!r} — run sweep_ilia_beta0.sh and sweep_lrs_rescue.sh first")
        return pd.DataFrame()
    frames = []
    for p in paths:
        meta = _parse_ext_filename(p)
        df   = pd.read_csv(p)
        for k, v in meta.items():
            if k not in df.columns:
                df[k] = v
        if "lr_scale" not in df.columns:
            df["lr_scale"] = meta.get("lr_scale", 1.0)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(f"[step5_ext] loaded {len(paths)} files, {len(out)} rows")
    return out


def merge_data(s5: pd.DataFrame, s5e: pd.DataFrame) -> pd.DataFrame:
    """Combine step5 and step5_ext into one normalised frame."""
    frames = [f for f in [s5, s5e] if not f.empty]
    if not frames:
        raise RuntimeError("No data loaded from either step5 or step5_ext.")
    df = pd.concat(frames, ignore_index=True)
    df["tau"]      = pd.to_numeric(df["tau"],      errors="coerce").astype("Int64")
    df["momentum"] = pd.to_numeric(df["momentum"], errors="coerce")
    df["seed"]     = pd.to_numeric(df["seed"],      errors="coerce").astype("Int64")
    df["epoch"]    = pd.to_numeric(df["epoch"],     errors="coerce").astype("Int64")
    df["lr_scale"] = pd.to_numeric(df["lr_scale"],  errors="coerce").fillna(1.0)
    df["M"]        = pd.to_numeric(df.get("M", pd.Series(dtype=float)), errors="coerce").astype("Int64")
    return df


def final_acc(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    last = df.groupby(groupby, dropna=False)["epoch"].transform("max")
    fin  = df[df["epoch"] == last].copy()
    agg  = (fin.groupby(groupby, dropna=False)["test_acc"]
               .agg(mean="mean", std="std")
               .reset_index())
    return agg


# ---------------------------------------------------------------------------
# Plot: final accuracy vs τ / E[τ]  (β=0 vs β=0.9, FIFO vs Geo)
# ---------------------------------------------------------------------------

def plot_step5_grid(df: pd.DataFrame, out_dir: str) -> None:
    """
    Two-panel: left = β=0, right = β=0.9.
    Each panel has two lines: FIFO (solid) and Geometric (dashed).
    Only rows without LR scaling (lr_scale ≈ 1.0).
    """
    base = df[df["lr_scale"].round(4) == 1.0].copy()
    if base.empty:
        print("[skip] plot_step5_grid: no lr_scale=1 data")
        return

    BETAS   = [0.0, 0.9]
    TAUS    = [0, 4, 16, 64]
    col_det = "#2166ac"
    col_geo = "#d62728"

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.8), sharey=True)

    for ax, beta in zip(axes, BETAS):
        # FIFO
        fifo = base[(base["dist"] == "fifo") & (base["momentum"].round(2) == beta)]
        if not fifo.empty:
            agg = final_acc(fifo, ["tau"])
            agg = agg[agg["tau"].isin(TAUS)].sort_values("tau")
            ax.errorbar(agg["tau"], agg["mean"], yerr=agg["std"].fillna(0),
                        marker="o", color=col_det, linestyle="-",
                        capsize=3, label="FIFO")

        # Geometric (x-axis = E[τ] = M−1)
        geo = base[(base["dist"] == "geometric") & (base["momentum"].round(2) == beta)]
        if not geo.empty:
            agg = final_acc(geo, ["M"])
            agg["etau"] = agg["M"].astype(int) - 1
            agg = agg[agg["etau"].isin(TAUS)].sort_values("etau")
            ax.errorbar(agg["etau"], agg["mean"], yerr=agg["std"].fillna(0),
                        marker="s", color=col_geo, linestyle="--",
                        capsize=3, label="Geometric")

        ax.set_title(f"β = {beta}")
        ax.set_xlabel("Delay τ (or E[τ] for geo)")
        ax.set_xticks(TAUS)
        ax.set_ylim(0, 95)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Final test accuracy (%)")
    fig.suptitle("Effect of gradient delay on final accuracy", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_step5_grid.png")
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: LR rescue at τ=64 / K=65  (headline figure)
# ---------------------------------------------------------------------------

def plot_lrs_rescue(df: pd.DataFrame, out_dir: str) -> None:
    """
    x-axis: lr_scale (log scale) — values tested: 1, 1/√64≈0.125, 1/64≈0.0156
    y-axis: final test accuracy
    Four lines: FIFO β=0, FIFO β=0.9, Geo β=0, Geo β=0.9
    Only τ=64 (FIFO) and K=65/E[τ]=64 (geo).
    """
    fifo_t64 = df[(df["dist"] == "fifo") & (df["tau"] == 64)].copy()
    geo_k65  = df[(df["dist"] == "geometric") & (df["M"] == 65)].copy()

    if fifo_t64.empty and geo_k65.empty:
        print("[skip] plot_lrs_rescue: no τ=64 / K=65 data")
        return

    configs = [
        (fifo_t64, 0.0, "#2166ac", "-",  "FIFO β=0"),
        (fifo_t64, 0.9, "#6baed6", "--", "FIFO β=0.9"),
        (geo_k65,  0.0, "#d62728", "-",  "Geo β=0"),
        (geo_k65,  0.9, "#fc8d59", "--", "Geo β=0.9"),
    ]

    fig, ax = plt.subplots(figsize=(COL1 * 1.6, 2.8))

    for sub, beta, col, ls, label in configs:
        grp = sub[sub["momentum"].round(2) == beta]
        if grp.empty:
            continue
        agg = final_acc(grp, ["lr_scale"])
        agg = agg.sort_values("lr_scale")
        ax.plot(agg["lr_scale"], agg["mean"], marker="o",
                color=col, linestyle=ls, label=label)
        ax.fill_between(agg["lr_scale"],
                        agg["mean"] - agg["std"].fillna(0),
                        agg["mean"] + agg["std"].fillna(0),
                        color=col, alpha=0.15)

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("LR scale (η → η × lrs)")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title("LR rescue at τ = 64  (FIFO) / E[τ] = 64  (Geo)")
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_lrs_rescue.png")
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: LR rescue efficiency across τ ∈ {4, 16, 64}
# ---------------------------------------------------------------------------

def plot_lrs_rescue_sweep(df: pd.DataFrame, out_dir: str) -> None:
    """
    x-axis: τ (or E[τ])
    y-axis: accuracy gain = acc(best_lrs) − acc(lrs=1)
    Two lines: β=0 and β=0.9
    One subplot per distribution (FIFO / Geo).
    """
    TAUS = [4, 16, 64]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.8), sharey=True)

    for ax, dist, key, label in [
        (axes[0], "fifo",      "tau", "FIFO"),
        (axes[1], "geometric", "M",   "Geometric"),
    ]:
        sub = df[df["dist"] == dist].copy()
        if sub.empty:
            ax.set_title(f"{label} — no data")
            continue

        for beta, col, ls in [(0.0, "#2166ac", "-"), (0.9, "#d62728", "--")]:
            gains, x_vals = [], []
            for tau in TAUS:
                if key == "tau":
                    rows = sub[(sub["tau"] == tau) & (sub["momentum"].round(2) == beta)]
                else:
                    M = tau + 1
                    rows = sub[(sub["M"] == M) & (sub["momentum"].round(2) == beta)]
                if rows.empty:
                    continue
                agg   = final_acc(rows, ["lr_scale"])
                base  = agg[agg["lr_scale"].round(4) == 1.0]["mean"].values
                best  = agg["mean"].max()
                if len(base) == 0:
                    continue
                gains.append(best - base[0])
                x_vals.append(tau)
            if x_vals:
                ax.plot(x_vals, gains, marker="o", color=col, linestyle=ls,
                        label=f"β={beta}")

        ax.set_title(label)
        ax.set_xlabel("τ  (or E[τ] for geo)")
        ax.set_xticks(TAUS)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Accuracy gain from LR rescue (%)")
    fig.suptitle("LR rescue efficiency across delays", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_lrs_rescue_sweep.png")
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="outputs/plots")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    s5  = load_step5()
    s5e = load_step5_ext()
    df  = merge_data(s5, s5e)
    print(f"Combined: {len(df)} rows, "
          f"dist={df['dist'].unique().tolist()}, "
          f"beta={sorted(df['momentum'].dropna().unique().round(2).tolist())}, "
          f"lr_scale={sorted(df['lr_scale'].dropna().unique().round(4).tolist())}")

    plot_step5_grid(df, args.out_dir)
    plot_lrs_rescue(df, args.out_dir)
    plot_lrs_rescue_sweep(df, args.out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
