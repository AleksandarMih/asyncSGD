"""
Async-SGD experiment figures: ResNet-20 on CIFAR-10.
Produces Fig 1–5 as PDF + PNG at 300 dpi.
"""

import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Style
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

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_DIR = Path("outputs/step5")

def parse_filename(path):
    name = Path(path).stem
    # dist
    dist = "fifo" if "fifo" in name else "geometric"
    # K
    m = re.search(r"_K(\d+)", name)
    K = int(m.group(1)) if m else 0
    # tau
    m = re.search(r"_tau(\d+)", name)
    tau = int(m.group(1)) if m else 0
    # lrs
    m = re.search(r"_lrs([\d.]+)", name)
    lrs = float(m.group(1)) if m else 1.0
    # seed
    m = re.search(r"_seed(\d+)", name)
    seed = int(m.group(1)) if m else -1
    return dist, K, tau, lrs, seed

frames = []
for path in sorted(glob.glob(str(DATA_DIR / "step5_*.csv"))):
    if "summary" in path:
        continue
    dist, K, tau, lrs, seed = parse_filename(path)
    df = pd.read_csv(path)
    df["dist"] = dist
    df["K_nom"] = K
    df["tau_nom"] = tau
    df["lrs"] = lrs
    df["seed"] = seed
    frames.append(df)

all_data = pd.concat(frames, ignore_index=True)

# Canonical staleness label: for fifo it's tau; for geometric it's K-1
all_data["stale_nom"] = np.where(
    all_data["dist"] == "fifo",
    all_data["tau_nom"],
    all_data["K_nom"] - 1,
)

print(f"Loaded {len(frames)} runs, {len(all_data)} rows total.")

# ---------------------------------------------------------------------------
# Divergence check
# ---------------------------------------------------------------------------
last_epoch = all_data.groupby(["dist", "K_nom", "tau_nom", "lrs", "seed"]).last().reset_index()

print("\n=== Final test_acc table (all runs) ===")
print(f"{'dist':<12} {'K':>4} {'tau':>4} {'lrs':>10} {'seed':>6} | {'test_acc':>8} {'flag'}")
print("-" * 60)
diverged = []
for _, row in last_epoch.sort_values(["dist", "tau_nom", "K_nom", "lrs", "seed"]).iterrows():
    flag = "  <-- DIVERGED" if row["test_acc"] < 15.0 else ""
    print(f"{row['dist']:<12} {row['K_nom']:>4} {row['tau_nom']:>4} {row['lrs']:>10.6f} {row['seed']:>6} | {row['test_acc']:>8.2f}{flag}")
    if row["test_acc"] < 15.0:
        diverged.append(row)

print(f"\nTotal diverged runs: {len(diverged)}")

# divergence rate per (dist, stale_nom, lrs)
div_rate = last_epoch.copy()
div_rate["diverged"] = div_rate["test_acc"] < 15.0
div_summary = (div_rate.groupby(["dist", "stale_nom", "lrs"])["diverged"]
               .agg(["sum", "count"])
               .rename(columns={"sum": "n_div", "count": "n_total"})
               .reset_index())
div_summary["rate"] = div_summary["n_div"] / div_summary["n_total"]
print("\n=== Divergence rate per cell ===")
print(div_summary.to_string(index=False))

# ---------------------------------------------------------------------------
# Helper: get empirical stale_mean at last epoch for a group
# ---------------------------------------------------------------------------
def get_empirical_staleness(data, dist, stale_nom, lrs=1.0):
    """Return mean of stale_mean across seeds at the last epoch."""
    mask = (
        (data["dist"] == dist) &
        (data["stale_nom"] == stale_nom) &
        (data["lrs"] == lrs)
    )
    sub = data[mask]
    last = sub.groupby("seed").last()
    return last["stale_mean"].mean()

# ---------------------------------------------------------------------------
# Fig 1 — final test error vs empirical mean staleness  (lrs=1.0 only)
# ---------------------------------------------------------------------------
fig1_data = last_epoch[last_epoch["lrs"] == 1.0].copy()

# Group to get mean ± SE across seeds
def summarise_group(grp):
    vals = grp["test_acc"].values
    return pd.Series({
        "mean_test_acc": vals.mean(),
        "se_test_acc": vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0,
        "emp_stale": grp["stale_mean"].mean(),
    })

fig1_summary = (fig1_data
                .groupby(["dist", "stale_nom"])
                .apply(summarise_group)
                .reset_index())
fig1_summary["test_error"] = 100.0 - fig1_summary["mean_test_acc"]

fifo_s = fig1_summary[fig1_summary["dist"] == "fifo"].sort_values("emp_stale")
geo_s  = fig1_summary[fig1_summary["dist"] == "geometric"].sort_values("emp_stale")

fig, ax = plt.subplots(figsize=(COL1, 2.5))

ax.errorbar(fifo_s["emp_stale"], fifo_s["test_error"],
            yerr=fifo_s["se_test_acc"],
            fmt="o-", color="#1f77b4", label="FIFO (deterministic)",
            capsize=3, linewidth=1.2, markersize=4)
ax.errorbar(geo_s["emp_stale"], geo_s["test_error"],
            yerr=geo_s["se_test_acc"],
            fmt="s--", color="#d62728", label="Geometric (K workers)",
            capsize=3, linewidth=1.2, markersize=4)

ax.set_xlabel("Empirical mean staleness (last epoch)")
ax.set_ylabel("Test error (%) at epoch 50")
ax.set_xscale("symlog", linthresh=1)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([0, 4, 16, 64])
ax.legend(loc="upper left")
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.set_title("Fig 1 — Final test error vs. mean staleness")

# divergence note
div_note_lines = []
for _, row in div_summary[div_summary["lrs"] == 1.0].iterrows():
    if row["n_div"] > 0:
        div_note_lines.append(
            f"{row['dist']} stale≈{row['stale_nom']:.0f}: {row['n_div']}/{row['n_total']} diverged"
        )
if div_note_lines:
    ax.annotate("Diverged runs: " + "; ".join(div_note_lines),
                xy=(0.5, -0.22), xycoords="axes fraction",
                ha="center", fontsize=5.5, color="#555555")

fig.tight_layout()
fig.savefig(OUT / "fig1_test_error_vs_staleness.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1_test_error_vs_staleness.png", bbox_inches="tight")
plt.close(fig)
print("Saved Fig 1")

# ---------------------------------------------------------------------------
# Fig 2 — test_acc vs epoch, one panel per staleness level (lrs=1.0)
# ---------------------------------------------------------------------------
LEVELS = [0, 4, 16, 64]   # nominal staleness values shown as panels

fig2_data = all_data[all_data["lrs"] == 1.0].copy()

n_panels = len(LEVELS)
fig, axes = plt.subplots(1, n_panels, figsize=(COL2, 2.3), sharey=True)

COLORS = {"fifo": "#1f77b4", "geometric": "#d62728"}
LINES  = {"fifo": "-",       "geometric": "--"}
LABELS = {"fifo": "FIFO",    "geometric": "Geometric"}

for ax, level in zip(axes, LEVELS):
    if level == 0:
        # tau=0 reference only (fifo)
        sub = fig2_data[(fig2_data["dist"] == "fifo") & (fig2_data["stale_nom"] == 0)]
        pivot = sub.pivot_table(index="epoch", columns="seed", values="test_acc")
        m = pivot.mean(axis=1)
        s = pivot.std(axis=1)
        ax.plot(m.index, m.values, color=COLORS["fifo"], linestyle=LINES["fifo"],
                label="FIFO τ=0")
        ax.fill_between(m.index, m - s, m + s, color=COLORS["fifo"], alpha=0.15)
        ax.set_title("τ = 0\n(sync reference)")
    else:
        for dist in ["fifo", "geometric"]:
            sub = fig2_data[(fig2_data["dist"] == dist) & (fig2_data["stale_nom"] == level)]
            if sub.empty:
                continue
            pivot = sub.pivot_table(index="epoch", columns="seed", values="test_acc")
            m = pivot.mean(axis=1)
            s = pivot.std(axis=1)
            ax.plot(m.index, m.values, color=COLORS[dist], linestyle=LINES[dist],
                    label=LABELS[dist])
            ax.fill_between(m.index, m - s, m + s, color=COLORS[dist], alpha=0.15)
        nom_label = f"τ = {level}" if True else ""
        ax.set_title(f"Staleness ≈ {level}")

    ax.set_xlabel("Epoch")
    ax.set_xlim(1, 50)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    if ax == axes[0]:
        ax.set_ylabel("Test accuracy (%)")

# unified legend on last panel
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.05), frameon=True)
fig.suptitle("Fig 2 — Test accuracy vs. epoch by staleness level (lr-scale = 1.0)",
             y=1.02, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig2_test_acc_vs_epoch.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig2_test_acc_vs_epoch.png", bbox_inches="tight")
plt.close(fig)
print("Saved Fig 2")

# ---------------------------------------------------------------------------
# Fig 3 — LR rescue at worst delay (fifo τ=64, geometric K=65)
# ---------------------------------------------------------------------------
LRS_VALUES = [1.0, 0.125, 0.015625]
LRS_LABELS = {1.0: "lr-scale 1.0", 0.125: "lr-scale 1/√64", 0.015625: "lr-scale 1/64"}
LRS_COLORS = {1.0: "#1f77b4", 0.125: "#ff7f0e", 0.015625: "#2ca02c"}
LRS_LINES  = {1.0: "-",       0.125: "--",       0.015625: ":"}

# reference: fifo tau=0 lrs=1.0
ref_data = all_data[(all_data["dist"] == "fifo") & (all_data["stale_nom"] == 0) & (all_data["lrs"] == 1.0)]

fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5), sharey=True)
DIST_TITLES = {"fifo": "FIFO τ = 64", "geometric": "Geometric K = 65"}
WORST = {"fifo": 64, "geometric": 64}  # stale_nom for worst case

for ax, dist in zip(axes, ["fifo", "geometric"]):
    # reference ceiling
    ref_pivot = ref_data.pivot_table(index="epoch", columns="seed", values="test_acc")
    rm = ref_pivot.mean(axis=1)
    rs = ref_pivot.std(axis=1)
    ax.plot(rm.index, rm.values, color="black", linestyle="-", linewidth=1.0,
            label="τ=0 / sync (ceiling)")
    ax.fill_between(rm.index, rm - rs, rm + rs, color="black", alpha=0.10)

    for lrs in LRS_VALUES:
        sub = all_data[
            (all_data["dist"] == dist) &
            (all_data["stale_nom"] == WORST[dist]) &
            (all_data["lrs"] == lrs)
        ]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="epoch", columns="seed", values="test_acc")
        m = pivot.mean(axis=1)
        s = pivot.std(axis=1)
        ax.plot(m.index, m.values,
                color=LRS_COLORS[lrs], linestyle=LRS_LINES[lrs],
                label=LRS_LABELS[lrs])
        ax.fill_between(m.index, m - s, m + s, color=LRS_COLORS[lrs], alpha=0.15)

    ax.set_title(DIST_TITLES[dist])
    ax.set_xlabel("Epoch")
    ax.set_xlim(1, 50)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    if ax == axes[0]:
        ax.set_ylabel("Test accuracy (%)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.08), frameon=True)
fig.suptitle("Fig 3 — LR rescue at worst delay (staleness ≈ 64)", y=1.02, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig3_lr_rescue.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig3_lr_rescue.png", bbox_inches="tight")
plt.close(fig)
print("Saved Fig 3")

# ---------------------------------------------------------------------------
# Fig 4 — Empirical staleness histograms (appendix)
# ---------------------------------------------------------------------------
# For geometric runs we have per-epoch stale_mean; use those as the distribution.
# For fifo we show a vertical line (point mass at tau).
GEO_Ks = [5, 17, 65]
GEO_COLORS = {5: "#9467bd", 17: "#8c564b", 65: "#e377c2"}

fig, axes = plt.subplots(1, len(GEO_Ks), figsize=(COL2, 2.2))

for ax, K in zip(axes, GEO_Ks):
    sub = all_data[(all_data["dist"] == "geometric") &
                   (all_data["K_nom"] == K) &
                   (all_data["lrs"] == 1.0)]
    stale_vals = sub["stale_mean"].dropna().values
    if len(stale_vals):
        ax.hist(stale_vals, bins=30, density=True,
                color=GEO_COLORS[K], alpha=0.7, edgecolor="white", linewidth=0.3,
                label=f"K={K} (geometric)")
    theo_mean = K - 1
    ax.axvline(theo_mean, color="black", linestyle="--", linewidth=1.0,
               label=f"Theoretical mean = {theo_mean}")

    # fifo tau = tau matching K-1
    tau_match = K - 1
    ax.axvline(tau_match, color="#1f77b4", linestyle=":", linewidth=1.2,
               label=f"FIFO τ={tau_match} (point mass)")

    ax.set_title(f"K = {K}")
    ax.set_xlabel("Empirical mean staleness\n(per epoch)")
    if ax == axes[0]:
        ax.set_ylabel("Density")
    ax.legend(fontsize=5.5)
    ax.grid(True, linewidth=0.3, alpha=0.4)

fig.suptitle("Fig 4 (Appendix) — Empirical staleness distributions", y=1.02, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig4_staleness_histograms.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig4_staleness_histograms.png", bbox_inches="tight")
plt.close(fig)
print("Saved Fig 4")

# ---------------------------------------------------------------------------
# Fig 5 — grad_angle vs epoch (appendix)
# ---------------------------------------------------------------------------
fig5_data = all_data[all_data["lrs"] == 1.0].copy()

# one curve per (dist, stale_nom), skip tau=0 geometric (doesn't exist)
CONFIGS = [
    ("fifo", 0,  "#1f77b4", "-",  "FIFO τ=0"),
    ("fifo", 4,  "#aec7e8", "-",  "FIFO τ=4"),
    ("fifo", 16, "#6baed6", "-",  "FIFO τ=16"),
    ("fifo", 64, "#08519c", "-",  "FIFO τ=64"),
    ("geometric", 4,  "#fc9272", "--", "Geo K=5  (stale≈4)"),
    ("geometric", 16, "#ef3b2c", "--", "Geo K=17 (stale≈16)"),
    ("geometric", 64, "#99000d", "--", "Geo K=65 (stale≈64)"),
]

fig, ax = plt.subplots(figsize=(COL1, 2.5))
for dist, stale_nom, color, ls, label in CONFIGS:
    sub = fig5_data[(fig5_data["dist"] == dist) & (fig5_data["stale_nom"] == stale_nom)]
    if sub.empty:
        continue
    # grad_angle may be NaN in tau=0 runs (sync)
    pivot = sub.pivot_table(index="epoch", columns="seed", values="grad_angle")
    m = pivot.mean(axis=1).dropna()
    s = pivot.std(axis=1).reindex(m.index).fillna(0)
    ax.plot(m.index, m.values, color=color, linestyle=ls, label=label, linewidth=1.0)
    ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.12)

ax.set_xlabel("Epoch")
ax.set_ylabel("Gradient angle (rad)")
ax.set_xlim(1, 50)
ax.legend(loc="upper right", ncol=1, fontsize=5.5)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.set_title("Fig 5 (Appendix) — Gradient angle vs. epoch (lr-scale = 1.0)")
fig.tight_layout()
fig.savefig(OUT / "fig5_grad_angle.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig5_grad_angle.png", bbox_inches="tight")
plt.close(fig)
print("Saved Fig 5")

print(f"\nAll figures written to {OUT.resolve()}/")
