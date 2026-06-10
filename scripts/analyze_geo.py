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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STEP3_DET_DIR = "outputs/det_grid"
GEO_GRID_DIR  = "outputs/geo_grid"
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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
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
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
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
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
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
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


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

    fig, ax = plt.subplots(figsize=(10, 5))

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
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


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

    print("\nDone.")


if __name__ == "__main__":
    main()
