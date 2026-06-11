"""
Analyze and compare deterministic vs geometric delay results.

Generates three figures:
  plot_geo_step3_comparison.png  — accuracy vs E[τ] for det vs geo, per β
  plot_geo_equivalence.png       — Mitliagkas Exp A: does geo equivalence hold?
  plot_geo_adaptive.png          — Exp G adaptive schedule: det vs geo

Usage:
  python3 scripts/analyze_geo.py
  python3 scripts/analyze_geo.py --out-dir outputs/figures
"""

import argparse
import math
import os
import glob
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Config
# ---------------------------------------------------------------------------

STEP3_DET_DIR = "outputs/det_grid"
GEO_GRID_DIR   = "outputs/geo_grid"
GEO_LARGE_DIR  = "outputs/geo_large"
GEO_THEORY_DIR = "outputs/geo_theory"
STEP4_DIR     = "outputs/det_theory"

BETAS = [0.0, 0.5, 0.9]
SEEDS = [42, 123, 456]

# M → E[τ] = M−1 mapping (must match sweep values)
M_TO_ETAU = {1: 0, 3: 2, 6: 5, 11: 10, 21: 20}
ETAU_LIST  = [0, 2, 5, 10, 20]

COLORS = {0.0: "#2196F3", 0.5: "#4CAF50", 0.9: "#F44336"}
LABEL_DET = "Deterministic"
LABEL_GEO = "Geometric"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_step3_det() -> pd.DataFrame:
    """Load all deterministic step3 CSVs."""
    files = glob.glob(os.path.join(STEP3_DET_DIR, "tau*_beta*_seed*.csv"))
    if not files:
        print(f"[warn] no files in {STEP3_DET_DIR}")
        return pd.DataFrame()
    frames = []
    for f in files:
        df = pd.read_csv(f)
        # step3 CSVs may not have delay_type column
        if "delay_type" not in df.columns:
            df["delay_type"] = "fifo"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_step3_geo() -> pd.DataFrame:
    """Load all geometric CSVs from geo_grid and geo_theory."""
    files = glob.glob(os.path.join(GEO_GRID_DIR, "exp*_geo_M*_beta*_seed*.csv"))
    if not files:
        print(f"[warn] no files in {GEO_GRID_DIR}")
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def final_test_acc(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    """Return mean ± std of final-epoch test_acc over seeds."""
    last_epoch = df.groupby(groupby)["epoch"].transform("max")
    final = df[df["epoch"] == last_epoch].copy()
    agg = (
        final.groupby(groupby)["test_acc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = groupby + ["mean_acc", "std_acc"]
    return agg


# ---------------------------------------------------------------------------
# Plot 1: Det vs Geo accuracy vs E[τ] for each β
# ---------------------------------------------------------------------------

def plot_step3_comparison(det: pd.DataFrame, geo: pd.DataFrame, out_dir: str):
    if det.empty and geo.empty:
        print("[skip] plot_geo_step3_comparison: no data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(COL2 * 2, 2.8), sharey=True)
    fig.suptitle("Deterministic vs Geometric Delay: Final Test Accuracy",
                 fontsize=13, fontweight="bold")

    for ax, beta in zip(axes, BETAS):
        ax.set_title(f"β = {beta}")
        ax.set_xlabel("E[τ] (average staleness)")
        ax.set_ylabel("Test Accuracy (%)" if beta == 0.0 else "")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ETAU_LIST)

        # Deterministic
        if not det.empty:
            sub = det[det["momentum"].round(2) == round(beta, 2)]
            if not sub.empty:
                agg = final_test_acc(sub, ["tau"])
                agg = agg.sort_values("tau")
                ax.errorbar(
                    agg["tau"], agg["mean_acc"], yerr=agg["std_acc"],
                    marker="o", linewidth=2, capsize=4,
                    color=COLORS[beta], linestyle="-",
                    label=LABEL_DET,
                )

        # Geometric
        if not geo.empty:
            sub = geo[geo["momentum"].round(2) == round(beta, 2)]
            if not sub.empty:
                agg = final_test_acc(sub, ["M"])
                agg = agg.sort_values("M")
                agg["etau"] = agg["M"] - 1
                ax.errorbar(
                    agg["etau"], agg["mean_acc"], yerr=agg["std_acc"],
                    marker="s", linewidth=2, capsize=4,
                    color=COLORS[beta], linestyle="--",
                    label=LABEL_GEO,
                )

        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(out_dir, "plot_geo_step3_comparison.png")
    _save(fig, out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Mitliagkas equivalence (Exp A det vs Exp A geo)
# ---------------------------------------------------------------------------

def plot_equivalence(geo: pd.DataFrame, det4: pd.DataFrame, out_dir: str):
    """
    Compare:
      - Geo M=3 β=0  vs  Fifo τ=0 β=0.667 (µ_S = (3-1)/3)
      - Geo M=6 β=0  vs  Fifo τ=0 β=0.833 (µ_S = (6-1)/6)
    """
    if geo.empty:
        print("[skip] plot_geo_equivalence: no geo data")
        return

    geo_a = geo[(geo["delay_type"] == "geometric") & (geo["momentum"].round(2) == 0.0)]
    if det4.empty:
        det_a = pd.DataFrame()
    else:
        det_a = det4[(det4["delay_type"] == "fifo") & (det4["tau"] == 0)]

    pairs = [(3, (3-1)/3), (6, (6-1)/6)]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.8), sharey=False)
    fig.suptitle("Mitliagkas Equivalence Test: Geometric Delay vs Implicit Momentum",
                 fontsize=12, fontweight="bold")

    for ax, (M, mu_s) in zip(axes, pairs):
        ax.set_title(f"M={M}  (E[τ]={M-1},  µ_S={(M-1)/M:.3f})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Accuracy (%)")
        ax.grid(True, alpha=0.3)

        sub_geo = geo_a[geo_a["M"] == M]
        if not sub_geo.empty:
            for seed, grp in sub_geo.groupby("seed"):
                ax.plot(grp["epoch"], grp["test_acc"],
                        color="#2196F3", alpha=0.35, linewidth=1)
            mean_g = sub_geo.groupby("epoch")["test_acc"].mean()
            ax.plot(mean_g.index, mean_g.values,
                    color="#2196F3", linewidth=2.5,
                    label=f"Geo M={M}, β=0  (stochastic async)")

        if not det_a.empty:
            mu_s_rounded = round(mu_s, 6)
            sub_det = det_a[det_a["momentum"].round(4).between(mu_s_rounded - 0.01,
                                                                mu_s_rounded + 0.01)]
            if not sub_det.empty:
                for seed, grp in sub_det.groupby("seed"):
                    ax.plot(grp["epoch"], grp["test_acc"],
                            color="#FF9800", alpha=0.35, linewidth=1)
                mean_d = sub_det.groupby("epoch")["test_acc"].mean()
                ax.plot(mean_d.index, mean_d.values,
                        color="#FF9800", linewidth=2.5, linestyle="--",
                        label=f"Fifo τ=0, β={mu_s:.3f}  (equivalent)")

        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(out_dir, "plot_geo_equivalence.png")
    _save(fig, out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Adaptive schedule det vs geo (Exp G)
# ---------------------------------------------------------------------------

def plot_adaptive(det: pd.DataFrame, geo: pd.DataFrame, out_dir: str):
    SCHEDULES = ["at_bound", "below_bound", "above_bound", "neg_ramp"]
    SCHED_COLORS = {
        "at_bound":    "#2196F3",
        "below_bound": "#4CAF50",
        "above_bound": "#F44336",
        "neg_ramp":    "#9C27B0",
    }

    if "schedule" not in det.columns and "schedule" not in geo.columns:
        print("[skip] plot_geo_adaptive: no schedule column found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(COL2 * 2, 2.8), sharey=True)
    fig.suptitle("Adaptive Liu-Bound Schedule: Deterministic vs Geometric Delay",
                 fontsize=12, fontweight="bold")

    ms_to_plot = [3, 6, 11]
    for ax, M in zip(axes, ms_to_plot):
        etau = M - 1
        ax.set_title(f"M={M}  (E[τ]={etau})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Accuracy (%)" if M == 3 else "")
        ax.grid(True, alpha=0.3)

        # β=0 reference lines (no momentum, no schedule)
        if not det.empty:
            sub0_det = det[(det["tau"] == etau)
                           & (det["momentum"].round(2) == 0.0)
                           & (det.get("delay_type", "fifo") == "fifo")
                           & (~det.get("schedule", pd.Series(["fixed"] * len(det))).isin(
                               ["at_bound", "below_bound", "above_bound", "neg_ramp"]))]
            if not sub0_det.empty:
                mean0_det = sub0_det.groupby("epoch")["test_acc"].mean()
                ax.plot(mean0_det.index, mean0_det.values,
                        color="black", linewidth=1.5, linestyle="-", alpha=0.5,
                        label="β=0 det")

        if not geo.empty:
            sub0_geo = geo[(geo["M"] == M)
                           & (geo["momentum"].round(2) == 0.0)
                           & (geo["delay_type"] == "geometric")
                           & (~geo.get("schedule", pd.Series(["fixed"] * len(geo))).isin(
                               ["at_bound", "below_bound", "above_bound", "neg_ramp"]))]
            if not sub0_geo.empty:
                mean0_geo = sub0_geo.groupby("epoch")["test_acc"].mean()
                ax.plot(mean0_geo.index, mean0_geo.values,
                        color="black", linewidth=2, linestyle="--", alpha=0.7,
                        label="β=0 geo")

        for sched in SCHEDULES:
            color = SCHED_COLORS[sched]

            # Deterministic at same τ=etau
            if not det.empty and "schedule" in det.columns:
                sub = det[(det["tau"] == etau) & (det["schedule"] == sched)
                          & (det.get("delay_type", "fifo") == "fifo")]
                if not sub.empty:
                    mean_d = sub.groupby("epoch")["test_acc"].mean()
                    ax.plot(mean_d.index, mean_d.values,
                            color=color, linewidth=1.8, linestyle="-", alpha=0.6,
                            label=f"{sched} det")

            # Geometric at M
            if not geo.empty and "schedule" in geo.columns:
                sub = geo[(geo["M"] == M) & (geo["schedule"] == sched)
                          & (geo["delay_type"] == "geometric")]
                if not sub.empty:
                    mean_g = sub.groupby("epoch")["test_acc"].mean()
                    ax.plot(mean_g.index, mean_g.values,
                            color=color, linewidth=2.5, linestyle="--",
                            label=f"{sched} geo")

        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    out = os.path.join(out_dir, "plot_geo_adaptive.png")
    _save(fig, out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot geo-L2 — Exp G geo: final accuracy bar chart (mirrors det plot_L2)
# ---------------------------------------------------------------------------

SCHED_COLORS = {
    "at_bound":    "#e6194b",
    "below_bound": "#f58231",
    "above_bound": "#3cb44b",
    "neg_ramp":    "#4363d8",
}
SCHED_LABELS = {
    "at_bound":    "at_bound",
    "below_bound": "below_bound",
    "above_bound": "above_bound",
    "neg_ramp":    "neg_ramp",
}

def plot_geo_L2(geo: pd.DataFrame, det: pd.DataFrame, out_dir: str) -> None:
    """
    Grouped bar chart of final test accuracy under geometric delay:
    one group per M value, one bar per schedule + β=0 and β=0.9 baselines.
    Matches the layout of plot_L2_adaptive_summary for easy comparison.
    """
    Ms = [6, 11, 21]   # E[τ] = 5, 10, 20 — matches deterministic τ grid
    schedules = ["at_bound", "below_bound", "above_bound", "neg_ramp"]

    geo_g = geo[geo.get("schedule", pd.Series(dtype=str)).isin(schedules)] \
        if "schedule" in geo.columns else pd.DataFrame()

    if geo_g.empty:
        print("plot_geo_L2: no Exp G geo data, skipping.")
        return

    bar_labels = schedules + ["β=0 (geo)", "β=0.9 (geo)"]
    bar_colors = [SCHED_COLORS[s] for s in schedules] + ["#aaaaaa", "#666666"]
    n_bars  = len(bar_labels)
    x       = np.arange(len(Ms))
    width   = 0.13
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

    fig, ax = plt.subplots(figsize=(COL2, 3.0))

    for i, (label, color) in enumerate(zip(bar_labels, bar_colors)):
        means, stds = [], []
        for M in Ms:
            if label in ("β=0 (geo)", "β=0.9 (geo)"):
                beta = 0.0 if label == "β=0 (geo)" else 0.9
                sub = geo[(geo["M"] == M)
                          & (geo["momentum"].round(2) == beta)
                          & (~geo.get("schedule", pd.Series(dtype=str)).isin(schedules))]
            else:
                sub = geo_g[(geo_g["M"] == M) & (geo_g["schedule"] == label)]
            final = sub[sub["epoch"] == sub["epoch"].max()]["test_acc"] \
                if not sub.empty else pd.Series(dtype=float)
            means.append(final.mean() if not final.empty else float("nan"))
            stds.append(final.std()   if len(final) > 1  else 0.0)

        bars = ax.bar(x + offsets[i], means, width,
                      yerr=stds, capsize=3,
                      label=SCHED_LABELS.get(label, label),
                      color=color, alpha=0.85)
        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.4,
                        f"{mean:.1f}", ha="center", va="bottom", fontsize=6)

    # No-delay ceiling from deterministic τ=0, β=0.9
    no_delay = det[(det["tau"] == 0) & (det["momentum"].round(2) == 0.9)]
    if not no_delay.empty:
        ceiling = no_delay[no_delay["epoch"] == no_delay["epoch"].max()]["test_acc"].mean()
        ax.axhline(ceiling, color="black", linestyle="--", linewidth=1.4,
                   label=f"τ=0, β=0.9 no-delay ceiling ({ceiling:.1f}%)")
        ax.text(x[-1] + 0.55, ceiling + 0.3, f"{ceiling:.1f}%",
                fontsize=7, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([f"M={M}  (E[τ]={M-1})" for M in Ms])
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title(
        "Exp G (geometric delay): Final accuracy by M and momentum schedule\n"
        "(dashed line = τ=0 no-delay ceiling; β=0 and β=0.9 geo baselines shown)"
    )
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()

    out = os.path.join(out_dir, "plot_geo_L2_adaptive_summary.png")
    _save(fig, out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Q1 geo: learning curves (test accuracy vs epoch)
# ---------------------------------------------------------------------------

def plot_Q1_curves_geo(geo: pd.DataFrame, out_dir: str) -> None:
    BETAS_PLOT = [0.0, 0.9]
    TITLES = {0.0: "β = 0 (no momentum)", 0.9: "β = 0.9 (high momentum)"}
    Ms = [1, 3, 6, 11, 21]   # E[τ] = 0, 2, 5, 10, 20

    # grid data only (no adaptive schedules)
    grid = geo[~geo.get("schedule", pd.Series(dtype=str)).isin(
        ["at_bound", "below_bound", "above_bound", "neg_ramp"])].copy()
    if grid.empty:
        print("[skip] plot_Q1_curves_geo: no grid data")
        return

    cmap   = plt.cm.Blues
    colors = {M: cmap(0.35 + 0.65 * i / (len(Ms) - 1)) for i, M in enumerate(Ms)}

    curves = (
        grid.groupby(["M", "momentum", "epoch"])["test_acc"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.8), sharey=True)
    for ax, beta in zip(axes, BETAS_PLOT):
        sub = curves[curves["momentum"].round(2) == beta]
        for M in Ms:
            grp = sub[sub["M"] == M].sort_values("epoch")
            if grp.empty:
                continue
            col   = colors[M]
            etau  = M - 1
            ax.plot(grp["epoch"], grp["mean"], color=col, linewidth=1.8,
                    label=f"E[τ]={etau} (M={M})", zorder=3)
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
    fig.suptitle("Convergence curves: test accuracy vs epoch (geometric delay)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_Q1_geo_convergence_curves.png")
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Q1 geo: convergence speed metrics (epochs-to-threshold + AUC)
# ---------------------------------------------------------------------------

def plot_Q1_convergence_metrics_geo(geo: pd.DataFrame, out_dir: str) -> None:
    Ms         = [1, 3, 6, 11, 21]
    THRESHOLDS = [80.0, 85.0]
    BETA       = 0.0

    grid = geo[~geo.get("schedule", pd.Series(dtype=str)).isin(
        ["at_bound", "below_bound", "above_bound", "neg_ramp"])].copy()
    if grid.empty:
        print("[skip] plot_Q1_convergence_metrics_geo: no grid data")
        return

    curves = (
        grid[grid["momentum"].round(2) == BETA]
        .groupby(["M", "epoch"])["test_acc"]
        .mean()
        .reset_index()
    )

    epochs_to = {}
    for M in Ms:
        grp = curves[curves["M"] == M].sort_values("epoch")
        row = {}
        for thr in THRESHOLDS:
            hit = grp[grp["test_acc"] >= thr]
            row[thr] = int(hit["epoch"].iloc[0]) if not hit.empty else None
        epochs_to[M] = row

    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = lambda y, x: sum((y[i]+y[i+1])*(x[i+1]-x[i])/2 for i in range(len(y)-1))

    auc_vals = {}
    for M in Ms:
        grp = curves[curves["M"] == M].sort_values("epoch")
        auc_vals[M] = _trapz(grp["test_acc"].values, grp["epoch"].values)
    auc_norm = {M: auc_vals[M] / auc_vals[1] for M in Ms}   # normalise to M=1 (E[τ]=0)

    x      = np.arange(len(Ms))
    width  = 0.35
    xlabels = [f"E[τ]={M-1}" for M in Ms]
    colors_thr = ["#4393c3", "#2166ac"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.8))

    for i, thr in enumerate(THRESHOLDS):
        vals    = []
        dnr_pos = []
        for j, M in enumerate(Ms):
            v = epochs_to[M][thr]
            if v is None:
                vals.append(0)
                dnr_pos.append(j)
            else:
                vals.append(v)
        ax1.bar(x + (i - 0.5) * width, vals, width,
                label=f"≥{thr:.0f}%", color=colors_thr[i], alpha=0.85)
        for j in dnr_pos:
            ax1.text(x[j] + (i - 0.5) * width, 3, "DNR",
                     ha="center", va="bottom", fontsize=8, color="crimson",
                     fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels)
    ax1.set_ylabel("First epoch ≥ threshold")
    ax1.set_title("Epochs to reach accuracy threshold\n(β=0, geometric delay, DNR = did not reach)",
                  fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    bar_colors = [plt.cm.Blues(0.35 + 0.65 * i / (len(Ms) - 1)) for i in range(len(Ms))]
    auc_y = [auc_norm[M] for M in Ms]
    ax2.bar(x, auc_y, color=bar_colors, alpha=0.85)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
                label="E[τ]=0 baseline")
    for j, v in enumerate(auc_y):
        ax2.text(j, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels)
    ax2.set_ylabel("Normalised AUC (relative to E[τ]=0)")
    ax2.set_title("Training efficiency: area under learning curve\n(β=0, geometric delay)",
                  fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "plot_Q1_geo_convergence_metrics.png")
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stability boundary: final accuracy vs E[τ] across full M range
# ---------------------------------------------------------------------------

def plot_geo_stability_boundary(geo_all: pd.DataFrame, out_dir: str) -> None:
    """
    Final test accuracy vs E[τ] for all M values (grid + large sweep combined),
    one line per β. Shows where geometric delay causes convergence failure.
    """
    grid = geo_all[~geo_all.get("schedule", pd.Series(dtype=str)).isin(
        ["at_bound", "below_bound", "above_bound", "neg_ramp"])].copy()
    if grid.empty:
        print("[skip] plot_geo_stability_boundary: no data")
        return

    BETAS_PLOT = [0.0, 0.5, 0.9]
    COLORS_B   = {0.0: "#2166ac", 0.5: "#f4a261", 0.9: "#d62728"}
    LABELS_B   = {0.0: "β = 0", 0.5: "β = 0.5", 0.9: "β = 0.9"}

    agg = final_test_acc(grid, ["M", "momentum"])
    agg["etau"] = agg["M"] - 1

    fig, ax = plt.subplots(figsize=(COL2, 3.0))

    for beta in BETAS_PLOT:
        sub = agg[agg["momentum"].round(2) == beta].sort_values("etau")
        if sub.empty:
            continue
        col = COLORS_B[beta]
        ax.plot(sub["etau"], sub["mean_acc"], marker="o", color=col,
                linewidth=2, markersize=6, label=LABELS_B[beta], zorder=3)
        ax.fill_between(sub["etau"],
                        sub["mean_acc"] - sub["std_acc"].fillna(0),
                        sub["mean_acc"] + sub["std_acc"].fillna(0),
                        color=col, alpha=0.15, zorder=2)
        # annotate final point
        last = sub.iloc[-1]
        ax.annotate(f"{last['mean_acc']:.1f}%",
                    xy=(last["etau"], last["mean_acc"]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=col, va="center")

    # mark random-chance floor
    ax.axhline(10.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(agg["etau"].max() * 0.02, 10.6, "random (10%)",
            fontsize=8, color="gray")

    ax.set_xlabel("E[τ] (average staleness)", fontsize=11)
    ax.set_ylabel("Final test accuracy (%)", fontsize=11)
    ax.set_title("Geometric delay stability boundary\n"
                 "Finding E[τ] where geometric delay causes convergence failure",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "plot_geo_stability_boundary.png")
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Q1 geo: train-test gap (appendix)
# ---------------------------------------------------------------------------

def plot_Q1_gap_geo(geo: pd.DataFrame, out_dir: str) -> None:
    Ms   = [1, 3, 6, 11, 21]
    BETA = 0.0

    grid = geo[~geo.get("schedule", pd.Series(dtype=str)).isin(
        ["at_bound", "below_bound", "above_bound", "neg_ramp"])].copy()
    if grid.empty:
        print("[skip] plot_Q1_gap_geo: no grid data")
        return

    sub      = grid[grid["momentum"].round(2) == BETA].copy()
    sub["gap"] = sub["train_acc"] - sub["test_acc"]
    curves   = sub.groupby(["M", "epoch"])["gap"].mean().reset_index()

    cmap   = plt.cm.Blues
    colors = {M: cmap(0.35 + 0.65 * i / (len(Ms) - 1)) for i, M in enumerate(Ms)}

    fig, ax = plt.subplots(figsize=(COL2, 2.8))
    for M in Ms:
        grp = curves[curves["M"] == M].sort_values("epoch")
        ax.plot(grp["epoch"], grp["gap"], color=colors[M], linewidth=1.8,
                label=f"E[τ]={M-1} (M={M})")

    for milestone in [30, 40]:
        ax.axvline(milestone, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Train acc − Test acc (%)", fontsize=10)
    ax.set_title("Generalisation gap vs epoch (β=0, geometric delay)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_Q1_geo_train_test_gap.png")
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze det vs geometric delay results")
    p.add_argument("--out-dir", default="outputs/plots",
                   help="directory to write output PNGs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading deterministic step3 data ...")
    det = load_step3_det()
    print(f"  {len(det)} rows from {STEP3_DET_DIR}")

    print("Loading geometric step3 data ...")
    geo = load_step3_geo()
    print(f"  {len(geo)} rows from {GEO_GRID_DIR} + {GEO_THEORY_DIR}")

    print("Loading det_theory fifo data (for equivalence test) ...")
    files4 = glob.glob(os.path.join(STEP4_DIR, "expA_tau*_beta*_seed*.csv"))
    if files4:
        det4 = pd.concat([pd.read_csv(f) for f in files4], ignore_index=True)
        if "delay_type" not in det4.columns:
            det4["delay_type"] = "fifo"
    else:
        det4 = pd.DataFrame()
    print(f"  {len(det4)} rows from {STEP4_DIR}")

    # Load large-M stability sweep if available
    print("Loading geometric large-M data ...")
    geo_large_files = glob.glob(os.path.join(GEO_LARGE_DIR, "exp*_geo_M*_beta*_seed*.csv"))
    if geo_large_files:
        geo_large = pd.concat([pd.read_csv(f) for f in geo_large_files], ignore_index=True)
        print(f"  {len(geo_large)} rows from {GEO_LARGE_DIR}")
        geo_all = pd.concat([geo, geo_large], ignore_index=True)
    else:
        print(f"  [none] {GEO_LARGE_DIR} not yet available")
        geo_large = pd.DataFrame()
        geo_all = geo.copy()

    # Merge geo expG data into geo df if present
    geo_g_files = glob.glob(os.path.join(GEO_THEORY_DIR, "expG_geo_M*_schedule*_seed*.csv"))
    if geo_g_files:
        geo_g = pd.concat([pd.read_csv(f) for f in geo_g_files], ignore_index=True)
        geo = pd.concat([geo, geo_g], ignore_index=True)

    # Merge det expG data
    det_g_files = glob.glob(os.path.join(STEP4_DIR, "expG_tau*_schedule*_seed*.csv"))
    if det_g_files:
        det_g = pd.concat([pd.read_csv(f) for f in det_g_files], ignore_index=True)
        if "delay_type" not in det_g.columns:
            det_g["delay_type"] = "fifo"
        det = pd.concat([det, det_g], ignore_index=True)

    plot_step3_comparison(det, geo, args.out_dir)
    plot_equivalence(geo, det4, args.out_dir)
    plot_adaptive(det, geo, args.out_dir)
    plot_geo_L2(geo, det, args.out_dir)
    plot_Q1_curves_geo(geo,                     args.out_dir)
    plot_Q1_convergence_metrics_geo(geo,         args.out_dir)
    plot_Q1_gap_geo(geo,                         args.out_dir)
    plot_geo_stability_boundary(geo_all,         args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
