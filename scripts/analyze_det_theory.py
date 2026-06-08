"""
Step 4 analysis — delay-momentum theory experiments.

Loads step3 data (baseline) + step4 data (new runs) and produces five plots
in narrative order:

  plot_D  — Exp A: equivalence test  (delay τ vs implicit momentum)
  plot_E  — Exp B: compensation      (does subtracting β_impl help?)
  plot_F  — Exp C: LR scaling        (1/τ and 1/√τ rules)
  plot_G  — Gradient alignment       (cos-sim between fresh and stale grads)
  plot_H  — Convergence speed        (epochs to 80% — summary across all exps)

Usage (from repo root):
  python scripts/analyze_det_theory.py
  python scripts/analyze_det_theory.py --step3-dir outputs/det_grid \\
                                   --step4-dir outputs/det_theory \\
                                   --out-dir   outputs/plots
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

def load_step3(step3_dir: str) -> pd.DataFrame:
    """Load all step3 CSVs; tag with experiment='step3', lr_scale=1.0."""
    paths = sorted(glob.glob(os.path.join(step3_dir, "tau*_beta*_seed*.csv")))
    if not paths:
        raise FileNotFoundError(f"No step3 CSVs in {step3_dir!r}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["experiment"] = "step3"
    df["lr_scale"]   = 1.0
    df["grad_angle"] = float("nan")
    print(f"[step3] Loaded {len(paths)} files, {len(df)} rows.")
    return df


def load_step4(step4_dir: str) -> pd.DataFrame:
    """Load all step4 CSVs; parse grad_angle as float (empty string → nan)."""
    paths = sorted(glob.glob(os.path.join(step4_dir, "exp*_tau*_seed*.csv")))
    if not paths:
        raise FileNotFoundError(f"No step4 CSVs in {step4_dir!r}")
    frames = []
    for p in paths:
        sub = pd.read_csv(p, dtype={"grad_angle": str})
        sub["grad_angle"] = pd.to_numeric(sub["grad_angle"], errors="coerce")
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    # Back-fill columns added in Exp G so older experiment rows stay valid.
    if "schedule" not in df.columns:
        df["schedule"] = "fixed"
    else:
        df["schedule"] = df["schedule"].fillna("fixed")
    if "beta_applied" not in df.columns:
        df["beta_applied"] = df["momentum"]
    else:
        df["beta_applied"] = pd.to_numeric(df["beta_applied"], errors="coerce")
        df["beta_applied"] = df["beta_applied"].fillna(df["momentum"])
    if "bound_offset" not in df.columns:
        df["bound_offset"] = float("nan")
    else:
        df["bound_offset"] = pd.to_numeric(df["bound_offset"], errors="coerce")
    print(f"[step4] Loaded {len(paths)} files, {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def implicit_momentum(tau: int) -> float:
    return (tau - 1) / tau if tau > 0 else 0.0


def final_stats(df: pd.DataFrame, groupby: list) -> pd.DataFrame:
    """Mean ± std of final-epoch test_acc, grouped by given columns."""
    final = df[df["epoch"] == df["epoch"].max()].copy()
    return (final.groupby(groupby)["test_acc"]
            .agg(mean="mean", std="std")
            .reset_index())


def mean_curves(df: pd.DataFrame, groupby: list) -> pd.DataFrame:
    """Mean ± std of train_loss per epoch, grouped by given columns."""
    return (df.groupby(groupby + ["epoch"])["train_loss"]
            .agg(mean="mean", std="std")
            .reset_index())


def epochs_to_threshold(df: pd.DataFrame, threshold: float = 80.0) -> pd.DataFrame:
    """
    First epoch where test_acc >= threshold for each (tau, momentum, seed,
    experiment, lr_scale) combination.  NaN = never reached (DNR).
    """
    reached = df[df["test_acc"] >= threshold]
    first   = (reached.groupby(["tau", "momentum", "seed", "experiment", "lr_scale"])
               ["epoch"].min().rename("epochs_to_thresh").reset_index())
    return first


# ---------------------------------------------------------------------------
# Plot D — Exp A: Equivalence test
# ---------------------------------------------------------------------------

def plot_D(df: pd.DataFrame, out_dir: str) -> None:
    """
    2×2 grid. Each cell compares training loss for one equivalence pair:
      solid  = (τ=k, β=0)         — delay only, no explicit momentum
      dashed = (τ=0, β=(k−1)/k)   — pure momentum, no delay
    If the curves overlap, delay τ ≈ implicit momentum.
    """
    pairs = [
        (2,  0.5,  "τ=2, β=0  vs  τ=0, β=0.5"),
        (5,  0.8,  "τ=5, β=0  vs  τ=0, β=0.8"),
        (10, 0.9,  "τ=10, β=0 vs  τ=0, β=0.9"),
        (20, 0.95, "τ=20, β=0 vs  τ=0, β=0.95"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    fig.suptitle("Exp A: Is delay τ equivalent to implicit momentum? (Mitliagkas 2014)",
                 fontsize=11)

    for ax, (tau_k, beta_impl, title) in zip(axes.flat, pairs):
        # Delayed side: (τ=k, β=0) — may come from step3 or step4
        delay_df = df[(df["tau"] == tau_k) & (np.isclose(df["momentum"], 0.0))]
        # Momentum side: (τ=0, β=β_impl) — may come from step3 or step4
        mom_df   = df[(df["tau"] == 0)    & (np.isclose(df["momentum"], beta_impl))]

        for sub, label, ls in [(delay_df, f"τ={tau_k}, β=0 (delay)", "-"),
                                (mom_df,   f"τ=0, β={beta_impl} (momentum)", "--")]:
            if sub.empty:
                ax.set_title(title + "\n[data missing]")
                continue
            curves = (sub.groupby("epoch")["train_loss"]
                      .agg(mean="mean", std="std").reset_index()
                      .sort_values("epoch"))
            ax.plot(curves["epoch"], curves["mean"], ls=ls, label=label)
            ax.fill_between(curves["epoch"],
                            curves["mean"] - curves["std"],
                            curves["mean"] + curves["std"], alpha=0.15)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss")
        ax.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(out_dir, "plot_D_equivalence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot E — Exp B: Compensation
# ---------------------------------------------------------------------------

def plot_E(df: pd.DataFrame, out_dir: str) -> None:
    """
    4 subplots (one per τ∈{2,5,10,20}). Each shows three bars:
      β=0.9 (step3, no compensation)  — tomato
      β=0.0 (step3, no momentum)      — lightgray
      β_comp (step4 exp B)             — steelblue
    Error bars are std over seeds.
    """
    taus = [2, 5, 10, 20]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    fig.suptitle("Exp B: Compensation — subtract implicit momentum from β",
                 fontsize=11)

    for ax, tau in zip(axes, taus):
        beta_comp = max(0.0, round(0.9 - implicit_momentum(tau), 6))

        bars = [
            ("β=0.9\n(step3)",  df[(df["tau"] == tau) & np.isclose(df["momentum"], 0.9)],  "tomato"),
            ("β=0.0\n(step3)",  df[(df["tau"] == tau) & np.isclose(df["momentum"], 0.0)
                                   & (df["experiment"] == "step3")],                         "lightgray"),
            (f"β_comp={beta_comp:.2f}\n(compensated)",
             df[(df["tau"] == tau) & np.isclose(df["momentum"], beta_comp)
                & (df["experiment"] == "B")],                                                "steelblue"),
        ]

        x     = np.arange(len(bars))
        means = []
        stds  = []
        for _, sub, _ in bars:
            final = sub[sub["epoch"] == sub["epoch"].max()] if not sub.empty else sub
            means.append(final["test_acc"].mean() if not final.empty else 0.0)
            stds.append( final["test_acc"].std()  if len(final) > 1  else 0.0)

        colors = [c for _, _, c in bars]
        labels = [l for l, _, _ in bars]
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="k",
               linewidth=0.7, capsize=4, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"τ={tau}", fontsize=10)
        ax.set_xlabel("")
        if tau == taus[0]:
            ax.set_ylabel("Test accuracy (%)")

    fig.tight_layout()
    path = os.path.join(out_dir, "plot_E_compensation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot F — Exp C: LR scaling
# ---------------------------------------------------------------------------

def plot_F(df: pd.DataFrame, out_dir: str) -> None:
    """
    3 subplots (τ∈{5,10,20}). Each: bars for lr_scale∈{1.0, 1/τ, 1/√τ}.
    A dashed reference line shows τ=0 accuracy (no delay, no scaling).
    """
    taus = [5, 10, 20]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    fig.suptitle("Exp C: LR scaling — which rule compensates for staleness? (β=0)",
                 fontsize=11)

    # τ=0 reference: no delay, β=0
    ref_acc = (df[(df["tau"] == 0) & np.isclose(df["momentum"], 0.0)]
               .query("epoch == epoch.max()")["test_acc"].mean())

    for ax, tau in zip(axes, taus):
        scales = {
            "1  (baseline)": 1.0,
            f"1/τ = {1/tau:.2f}":    round(1 / tau,       6),
            f"1/√τ = {1/tau**0.5:.3f}": round(1 / tau**0.5, 6),
        }
        means, stds, labels = [], [], []
        for label, scale in scales.items():
            # lr_scale=1.0 at high τ with β=0 may come from step3
            if np.isclose(scale, 1.0):
                sub = df[(df["tau"] == tau) & np.isclose(df["momentum"], 0.0)
                         & (df["experiment"].isin(["step3", "C"]))]
            else:
                sub = df[(df["tau"] == tau) & np.isclose(df["lr_scale"], scale)
                         & (df["experiment"] == "C")]
            final = sub[sub["epoch"] == sub["epoch"].max()] if not sub.empty else sub
            means.append(final["test_acc"].mean() if not final.empty else 0.0)
            stds.append( final["test_acc"].std()  if len(final) > 1  else 0.0)
            labels.append(label)

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color="steelblue", edgecolor="k",
               linewidth=0.7, capsize=4, width=0.5, alpha=0.85)
        ax.axhline(ref_acc, color="k", linestyle="--", linewidth=0.9,
                   label=f"τ=0 ref ({ref_acc:.1f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"τ={tau}", fontsize=10)
        ax.legend(fontsize=7)
        if tau == taus[0]:
            ax.set_ylabel("Test accuracy (%)")

    fig.tight_layout()
    path = os.path.join(out_dir, "plot_F_lr_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot G — Gradient alignment
# ---------------------------------------------------------------------------

def plot_G(df4: pd.DataFrame, out_dir: str) -> None:
    """
    Top: heatmap of mean final-epoch grad_angle (rows=τ, cols=β).
    Bottom: grad_angle vs epoch for τ=10 at β∈{0.0, 0.5, 0.9}.
    Shows how quickly stale gradients become misaligned as τ and β increase.
    """
    # Only rows with actual grad_angle data (τ>0, step4 runs)
    sub = df4[df4["grad_angle"].notna() & (df4["tau"] > 0)].copy()
    if sub.empty:
        print("Warning: no grad_angle data found, skipping Plot G.")
        return

    fig, (ax_heat, ax_line) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Gradient alignment: cosine similarity (fresh vs stale gradient)",
                 fontsize=11)

    # ── heatmap ──
    final_angle = (sub[sub["epoch"] == sub["epoch"].max()]
                   .groupby(["tau", "momentum"])["grad_angle"].mean().unstack())
    taus  = sorted(final_angle.index.tolist())
    betas = sorted(final_angle.columns.tolist())
    data  = [[final_angle.loc[t, b] if b in final_angle.columns else float("nan")
              for b in betas] for t in taus]
    im = ax_heat.imshow(data, aspect="auto", vmin=-1, vmax=1, cmap="RdYlGn")
    ax_heat.set_xticks(range(len(betas)))
    ax_heat.set_xticklabels([f"β={b}" for b in betas])
    ax_heat.set_yticks(range(len(taus)))
    ax_heat.set_yticklabels([f"τ={t}" for t in taus])
    ax_heat.set_title("Mean grad angle at final epoch\n(green=aligned, red=opposed)")
    for i, t in enumerate(taus):
        for j, b in enumerate(betas):
            val = data[i][j]
            if not np.isnan(val):
                ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                             fontsize=8, color="black")
    plt.colorbar(im, ax=ax_heat, label="cos similarity")

    # ── line chart for τ=10 ──
    tau_line = 10
    for beta, grp in sub[sub["tau"] == tau_line].groupby("momentum"):
        curves = (grp.groupby("epoch")["grad_angle"]
                  .agg(mean="mean", std="std").reset_index()
                  .sort_values("epoch"))
        ax_line.plot(curves["epoch"], curves["mean"], label=f"β={beta}")
        ax_line.fill_between(curves["epoch"],
                             curves["mean"] - curves["std"],
                             curves["mean"] + curves["std"], alpha=0.15)
    ax_line.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax_line.set_xlabel("Epoch")
    ax_line.set_ylabel("Cosine similarity")
    ax_line.set_title(f"Grad alignment over training (τ={tau_line})")
    ax_line.legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "plot_G_grad_angle.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot H — Convergence speed (summary)
# ---------------------------------------------------------------------------

def plot_H(df: pd.DataFrame, out_dir: str, threshold: float = 80.0) -> None:
    """
    Grouped bar chart: epochs to reach {threshold}% test accuracy.
    Groups = τ values; within each group, bars = key conditions from all experiments.
    DNR (did not reach) shown as hatched bar at max_epochs+2.
    Ties together all three experiments in one summary view.
    """
    max_epochs = int(df["epoch"].max())
    dnr_height = max_epochs + 2

    # Conditions to compare: (label, experiment-filter, color)
    def select(tau, mom, exp=None, lr_scale=None):
        mask = (df["tau"] == tau) & np.isclose(df["momentum"], mom)
        if exp is not None:
            mask &= df["experiment"] == exp
        if lr_scale is not None:
            mask &= np.isclose(df["lr_scale"], lr_scale)
        return df[mask]

    taus = [0, 2, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Convergence speed: epochs to reach {threshold:.0f}% test accuracy",
                 fontsize=11)

    bar_width = 0.18
    conditions = [
        ("β=0\n(step3)",       "lightgray"),
        ("β=0.9\n(step3)",     "tomato"),
        ("β_comp\n(Exp B)",    "steelblue"),
        ("lr=1/√τ\n(Exp C)",   "mediumseagreen"),
    ]
    offsets = np.linspace(-(len(conditions)-1)/2, (len(conditions)-1)/2, len(conditions))

    speed_df = epochs_to_threshold(df, threshold)

    for i, tau in enumerate(taus):
        beta_comp = max(0.0, round(0.9 - implicit_momentum(tau), 6))
        lr_sqrt   = round(1 / max(tau, 1)**0.5, 6)

        subs = [
            select(tau, 0.0, exp="step3"),
            select(tau, 0.9, exp="step3"),
            select(tau, beta_comp, exp="B") if tau in (2, 5) else select(tau, 0.0, exp="step3"),
            select(tau, 0.0, exp="C", lr_scale=lr_sqrt) if tau in (5, 10, 20) else select(tau, 0.0, exp="step3"),
        ]

        for j, (sub, (label, color)) in enumerate(zip(subs, conditions)):
            if sub.empty:
                height, err, hatch = dnr_height, 0, "//"
            else:
                merged = speed_df.merge(
                    sub[["tau", "momentum", "seed", "experiment", "lr_scale"]].drop_duplicates(),
                    on=["tau", "momentum", "seed", "experiment", "lr_scale"], how="inner")
                vals   = merged["epochs_to_thresh"].dropna()
                dnr    = len(merged) - len(vals)
                height = vals.mean() if len(vals) > 0 else dnr_height
                err    = vals.std()  if len(vals) > 1 else 0.0
                hatch  = "//" if dnr > 0 else ""

            x = i + offsets[j] * bar_width * 1.1
            ax.bar(x, height, width=bar_width, color=color, edgecolor="k",
                   linewidth=0.7, hatch=hatch, yerr=err, capsize=3,
                   label=label if i == 0 else "")

    ax.axhline(dnr_height, color="k", linestyle=":", linewidth=0.7, label="DNR bound")
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([f"τ={t}" for t in taus])
    ax.set_ylabel("Epochs to reach threshold")
    ax.set_ylim(0, dnr_height + 3)
    ax.legend(loc="upper left", fontsize=8, ncol=len(conditions) + 1)

    fig.tight_layout()
    path = os.path.join(out_dir, "plot_H_convergence_speed.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot J — Exp E: Calibration sweep
# ---------------------------------------------------------------------------

def plot_J(df: pd.DataFrame, out_dir: str) -> None:
    """
    For each τ ∈ {2, 5, 10, 20}, overlay all τ=0 training loss curves (one per β)
    against the (τ=k, β=0) reference curve.  Then find the best-matching β* for
    each τ by minimising MSE between the two mean loss curves, and plot β* vs τ
    alongside the theoretical Mitliagkas formulas τ/(τ+1) and (τ-1)/τ.

    If the Mitliagkas equivalence holds empirically, β* should follow one of
    those formulas.  If not, the gap between β* and the theoretical lines
    quantifies how far off the theory is in our deterministic delay setting.
    """
    taus_ref = [2, 5, 10, 20]

    # Collect all τ=0 runs across every experiment (step3, A, E …)
    tau0 = df[df["tau"] == 0].copy()
    all_betas = sorted(tau0["momentum"].unique())

    if tau0.empty or len(all_betas) < 3:
        print("Warning: not enough τ=0 data for Plot J (run Exp E first).")
        return

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "Exp E: Calibration — which β at τ=0 best matches (τ=k, β=0)?",
        fontsize=11,
    )

    # Colour map: low β = blue, high β = red
    cmap   = plt.cm.coolwarm
    colors = {b: cmap(i / max(len(all_betas) - 1, 1))
              for i, b in enumerate(all_betas)}

    # ── Top row: 4 overlay panels ──────────────────────────────────────────
    axes_top = [fig.add_subplot(2, 4, i + 1) for i in range(4)]
    best_betas: dict[int, float] = {}

    for ax, tau_k in zip(axes_top, taus_ref):
        # Reference curve: (τ=k, β=0)
        ref = df[(df["tau"] == tau_k) & np.isclose(df["momentum"], 0.0)]
        ref_curve = (ref.groupby("epoch")["train_loss"]
                     .mean().reset_index().sort_values("epoch"))

        ax.plot(ref_curve["epoch"], ref_curve["train_loss"],
                color="black", linewidth=2.5, label=f"τ={tau_k}, β=0 (target)",
                zorder=10)

        # All τ=0 curves + MSE computation
        best_mse, best_b = np.inf, 0.0
        for beta in all_betas:
            sub = tau0[np.isclose(tau0["momentum"], beta)]
            curve = (sub.groupby("epoch")["train_loss"]
                     .mean().reset_index().sort_values("epoch"))
            if curve.empty:
                continue

            # Align on common epochs before computing MSE
            merged = ref_curve.merge(curve, on="epoch",
                                     suffixes=("_ref", "_tau0"))
            if merged.empty:
                continue
            mse = np.mean((merged["train_loss_ref"] - merged["train_loss_tau0"]) ** 2)
            if mse < best_mse:
                best_mse, best_b = mse, beta

            ax.plot(curve["epoch"], curve["train_loss"],
                    color=colors[beta], linewidth=0.9, alpha=0.7,
                    label=f"β={beta}" if tau_k == taus_ref[0] else None)

        best_betas[tau_k] = best_b

        # Highlight the best-matching τ=0 curve
        best_sub = tau0[np.isclose(tau0["momentum"], best_b)]
        best_curve = (best_sub.groupby("epoch")["train_loss"]
                      .mean().reset_index().sort_values("epoch"))
        ax.plot(best_curve["epoch"], best_curve["train_loss"],
                color=colors[best_b], linewidth=2.5, linestyle="--",
                label=f"best match β*={best_b}")

        ax.set_title(f"τ={tau_k} target\nbest match β*={best_b}", fontsize=9)
        ax.set_xlabel("Epoch")
        if tau_k == taus_ref[0]:
            ax.set_ylabel("Train loss")

    # Shared legend for top row (β colour scale)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=min(all_betas),
                                                  vmax=max(all_betas)))
    sm.set_array([])
    fig.colorbar(sm, ax=axes_top, label="β  (τ=0 runs)", shrink=0.6,
                 pad=0.02)

    # ── Bottom centre: β* vs τ with theoretical formulas ──────────────────
    ax_bot = fig.add_subplot(2, 1, 2)
    tau_vals = np.array(taus_ref)
    beta_star = np.array([best_betas[t] for t in taus_ref])

    ax_bot.plot(tau_vals, beta_star, "ko-", linewidth=2, markersize=7,
                label="Empirical β* (best MSE match)")

    # Theoretical lines
    tau_range = np.linspace(1, 21, 200)
    ax_bot.plot(tau_range, (tau_range - 1) / tau_range, "b--", linewidth=1.4,
                label="(τ−1)/τ  [used in Exp A]")
    ax_bot.plot(tau_range, tau_range / (tau_range + 1), "r:",  linewidth=1.4,
                label="τ/(τ+1)  [Mitliagkas k workers]")

    ax_bot.set_xlabel("Gradient delay τ")
    ax_bot.set_ylabel("β*  (best-matching momentum at τ=0)")
    ax_bot.set_title(
        "Does β* follow a theoretical formula?\n"
        "Overlap = Mitliagkas equivalence holds; gap = it does not",
        fontsize=9,
    )
    ax_bot.set_xlim(0, 22)
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "plot_J_calibration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot K  — Exp F: accuracy vs β (full range) per τ, with theory prediction
# ---------------------------------------------------------------------------

def plot_K(df: pd.DataFrame, out_dir: str) -> None:
    """
    Exp F: for each τ, show mean final test accuracy vs β over the full range
    [−0.5, 0.9], combining new negative-β runs with existing step3 baselines.

    A vertical dashed line marks β_optimal = −(τ−1)/τ from compensation theory.
    A horizontal dashed line marks the τ=0 ceiling (β=0.9, no delay).
    If the theory holds, accuracy should peak near the vertical marker.
    """
    taus = [5, 10, 20]
    sub_f  = df[df["experiment"] == "F"]
    sub_s3 = df[(df["experiment"] == "step3") & (df["tau"].isin(taus))]
    sub    = pd.concat([sub_f, sub_s3], ignore_index=True)

    if sub_f.empty:
        print("plot_K: no Exp F data found, skipping.")
        return

    # τ=0 ceiling: β=0.9, no delay (from step3)
    ceil_df = df[(df["experiment"] == "step3") & (df["tau"] == 0)
                 & (df["momentum"] == 0.9)]
    ceil_val = (ceil_df[ceil_df["epoch"] == ceil_df["epoch"].max()]["test_acc"]
                .mean() if not ceil_df.empty else None)

    fig, axes = plt.subplots(1, len(taus), figsize=(5 * len(taus), 4), sharey=True)

    for ax, tau in zip(axes, taus):
        grp_data = sub[sub["tau"] == tau]
        stats = final_stats(grp_data, ["momentum"]).sort_values("momentum")

        ax.errorbar(stats["momentum"], stats["mean"], yerr=stats["std"],
                    fmt="o-", color="steelblue", capsize=4, linewidth=1.5,
                    markersize=5, label="mean ± std")

        beta_opt = -((tau - 1) / tau)
        ax.axvline(beta_opt, color="tomato", linestyle="--", linewidth=1.2,
                   label=f"β_opt = {beta_opt:.2f}")

        if ceil_val is not None:
            ax.axhline(ceil_val, color="gray", linestyle=":", linewidth=1,
                       label=f"τ=0 ceiling ({ceil_val:.1f}%)")

        ax.set_xlabel("Explicit momentum β")
        ax.set_title(f"τ = {tau}  (β_opt ≈ {beta_opt:.2f})")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Final test accuracy (%)")
    fig.suptitle(
        "Exp F: does negative β compensate implicit delay momentum?\n"
        "Red dashed = theory prediction β_opt = −(τ−1)/τ",
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_K_negative_momentum.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot L — Exp G: Adaptive momentum schedule (Liu et al. bound)
# ---------------------------------------------------------------------------

_SCHED_COLORS = {
    "at_bound":    "steelblue",
    "below_bound": "seagreen",
    "above_bound": "tomato",
    "neg_ramp":    "darkorchid",
}
_SCHED_LABELS = {
    "at_bound":    "at bound  µ=1−√(τη)",
    "below_bound": "below bound  µ−0.3",
    "above_bound": "above bound  µ+0.3",
    "neg_ramp":    "neg→ramp  (Mitliagkas+Liu)",
}


def plot_L(df: pd.DataFrame, out_dir: str) -> None:
    """
    Exp G: adaptive momentum schedule.

    Figure L1 (plot_L1_adaptive_curves.png): 2 rows × 3 cols (τ=5, 10, 20).
      Top row    — momentum trajectory (β_applied vs epoch, mean across seeds).
      Bottom row — test accuracy curves (mean ± std band per schedule +
                   dashed baselines β=0 and β=0.9 from step3).

    Figure L2 (plot_L2_adaptive_summary.png): grouped bar chart of final
      test accuracy with ±std error bars — one group per τ, one bar per
      schedule + 2 baselines.
    """
    taus = [5, 10, 20]
    schedules = ["at_bound", "below_bound", "above_bound", "neg_ramp"]

    sub_g  = df[df["experiment"] == "G"]
    if sub_g.empty:
        print("plot_L: no Exp G data found, skipping.")
        return

    sub_s3 = df[(df["experiment"] == "step3") & (df["tau"].isin(taus))]

    # ── Figure L1: momentum trajectory + accuracy curves ──────────────────
    fig, axes = plt.subplots(2, len(taus), figsize=(5 * len(taus), 8),
                             sharex="row")

    for col, tau in enumerate(taus):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        g_tau  = sub_g[sub_g["tau"] == tau]

        # Top: momentum trajectory
        for sched in schedules:
            g_s = g_tau[g_tau["schedule"] == sched]
            if g_s.empty:
                continue
            traj = (g_s.groupby("epoch")["beta_applied"]
                    .mean().reset_index())
            ax_top.plot(traj["epoch"], traj["beta_applied"],
                        color=_SCHED_COLORS[sched],
                        label=_SCHED_LABELS[sched], linewidth=1.5)
        ax_top.axhline(0.0, color="gray",  linestyle="--", linewidth=0.8, alpha=0.6)
        ax_top.axhline(0.9, color="gray",  linestyle=":",  linewidth=0.8, alpha=0.6)
        ax_top.axvline(30,  color="black", linestyle=":",  linewidth=0.6, alpha=0.4)
        ax_top.axvline(40,  color="black", linestyle=":",  linewidth=0.6, alpha=0.4)
        ax_top.set_ylabel("β applied")
        ax_top.set_title(f"τ = {tau}  — momentum trajectory")
        ax_top.legend(fontsize=6, loc="lower right")
        ax_top.set_ylim(-1.05, 1.05)

        # Bottom: test accuracy curves
        for sched in schedules:
            g_s = g_tau[g_tau["schedule"] == sched]
            if g_s.empty:
                continue
            curves = (g_s.groupby("epoch")["test_acc"]
                      .agg(mean="mean", std="std").reset_index())
            ax_bot.plot(curves["epoch"], curves["mean"],
                        color=_SCHED_COLORS[sched],
                        label=_SCHED_LABELS[sched], linewidth=1.5)
            ax_bot.fill_between(curves["epoch"],
                                curves["mean"] - curves["std"],
                                curves["mean"] + curves["std"],
                                color=_SCHED_COLORS[sched], alpha=0.15)

        # Step3 baselines
        for beta, ls, lbl in [(0.0, "--", "β=0  (fixed)"),
                               (0.9, ":",  "β=0.9 (fixed)")]:
            base = sub_s3[(sub_s3["tau"] == tau) & (sub_s3["momentum"] == beta)]
            if base.empty:
                continue
            bc = (base.groupby("epoch")["test_acc"]
                  .agg(mean="mean", std="std").reset_index())
            ax_bot.plot(bc["epoch"], bc["mean"], color="gray",
                        linestyle=ls, linewidth=1.2, label=lbl, alpha=0.8)

        ax_bot.set_xlabel("Epoch")
        ax_bot.set_ylabel("Test accuracy (%)")
        ax_bot.set_title(f"τ = {tau}  — accuracy")
        ax_bot.legend(fontsize=6)

    fig.suptitle(
        "Exp G: Adaptive momentum schedule (Liu et al. NeurIPS 2018 bound)\n"
        "µ_safe(η) = 1−√(τ·η)  |  vertical dotted lines = LR decay milestones",
        fontsize=10,
    )
    fig.tight_layout()
    p1 = os.path.join(out_dir, "plot_L1_adaptive_curves.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved: {p1}")

    # ── Figure L2: final accuracy bar chart ───────────────────────────────
    bar_labels  = schedules + ["β=0", "β=0.9"]
    bar_colors  = [_SCHED_COLORS[s] for s in schedules] + ["#aaaaaa", "#666666"]
    n_bars = len(bar_labels)
    x = np.arange(len(taus))
    width = 0.13
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    for i, (label, color) in enumerate(zip(bar_labels, bar_colors)):
        means, stds = [], []
        for tau in taus:
            if label in ("β=0", "β=0.9"):
                beta = 0.0 if label == "β=0" else 0.9
                sub = sub_s3[(sub_s3["tau"] == tau) & (sub_s3["momentum"] == beta)]
            else:
                sub = sub_g[(sub_g["tau"] == tau) & (sub_g["schedule"] == label)]
            final = sub[sub["epoch"] == sub["epoch"].max()]["test_acc"] if not sub.empty else pd.Series(dtype=float)
            means.append(final.mean() if not final.empty else float("nan"))
            stds.append(final.std()  if len(final) > 1  else 0.0)

        bars = ax2.bar(x + offsets[i], means, width,
                       yerr=stds, capsize=3,
                       label=_SCHED_LABELS.get(label, label),
                       color=color, alpha=0.85)
        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                         f"{mean:.1f}", ha="center", va="bottom", fontsize=6)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"τ = {tau}" for tau in taus])
    ax2.set_ylabel("Final test accuracy (%)")

    # τ=0 no-delay ceiling
    no_delay = df[(df["experiment"] == "step3") & (df["tau"] == 0) & (df["momentum"] == 0.9)]
    if not no_delay.empty:
        ceiling = no_delay[no_delay["epoch"] == no_delay["epoch"].max()]["test_acc"].mean()
        ax2.axhline(ceiling, color="black", linestyle="--", linewidth=1.4,
                    label=f"τ=0, β=0.9 (no-delay ceiling, {ceiling:.1f}%)")
        ax2.text(x[-1] + 0.55, ceiling + 0.3, f"{ceiling:.1f}%", fontsize=7, va="bottom")

    ax2.set_title(
        "Exp G: Final accuracy by τ and momentum schedule\n"
        "(dashed line = τ=0 no-delay ceiling; β=0 and β=0.9 baselines shown with delay)"
    )
    ax2.legend(fontsize=7, loc="lower left")
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "plot_L2_adaptive_summary.png")
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"Saved: {p2}")


# ---------------------------------------------------------------------------
# Plot M — Exp H: Above-bound offset sweep (sweet spot)
# ---------------------------------------------------------------------------

def plot_M(df: pd.DataFrame, out_dir: str) -> None:
    """
    Exp H: sweep the additive offset δ in µ = min(0.95, µ_liu + δ).

    Combines Exp G anchor points (δ=0.0 from at_bound, δ=0.3 from above_bound)
    with Exp H new runs (δ=0.1, 0.5, 0.7) for τ ∈ {5, 10, 20}.

    Figure M (plot_M_above_bound_sweep.png): 1 row × 3 cols (τ=5, 10, 20).
    X-axis: δ value.  Y-axis: mean final test_acc ± std.
    Horizontal dashed lines for β=0 and β=0.9 fixed baselines.
    """
    taus = [5, 10, 20]

    sub_h  = df[df["experiment"] == "H"]
    if sub_h.empty:
        print("plot_M: no Exp H data found, skipping.")
        return

    sub_g  = df[df["experiment"] == "G"]
    sub_s3 = df[(df["experiment"] == "step3") & (df["tau"].isin(taus))]

    fig, axes = plt.subplots(1, len(taus), figsize=(4.5 * len(taus), 4), sharey=False)

    for ax, tau in zip(axes, taus):
        # Collect (offset, mean, std) from all sources
        points = []

        # Exp G anchor: at_bound → δ=0.0
        g_at = sub_g[(sub_g["tau"] == tau) & (sub_g["schedule"] == "at_bound")]
        if not g_at.empty:
            final = g_at[g_at["epoch"] == g_at["epoch"].max()]["test_acc"]
            points.append((0.0, final.mean(), final.std()))

        # Exp G anchor: above_bound → δ=0.3
        g_ab = sub_g[(sub_g["tau"] == tau) & (sub_g["schedule"] == "above_bound")]
        if not g_ab.empty:
            final = g_ab[g_ab["epoch"] == g_ab["epoch"].max()]["test_acc"]
            points.append((0.3, final.mean(), final.std()))

        # Exp H: one point per bound_offset value
        for offset, grp in sub_h[sub_h["tau"] == tau].groupby("bound_offset"):
            final = grp[grp["epoch"] == grp["epoch"].max()]["test_acc"]
            points.append((float(offset), final.mean(), final.std()))

        if not points:
            continue

        points.sort(key=lambda x: x[0])
        offsets_x = [p[0] for p in points]
        means      = [p[1] for p in points]
        stds       = [p[2] for p in points]

        ax.errorbar(offsets_x, means, yerr=stds, fmt="o-", color="steelblue",
                    capsize=4, linewidth=1.5, markersize=5, label="adaptive (above_bound)")

        # Mark the empirical peak
        peak_idx = int(np.argmax(means))
        ax.axvline(offsets_x[peak_idx], color="steelblue", linestyle="--",
                   linewidth=1.0, alpha=0.6, label=f"peak δ={offsets_x[peak_idx]:.1f}")

        # Fixed baselines
        for beta, ls, lbl, col in [(0.0, "--", "β=0 (fixed)", "gray"),
                                    (0.9, ":",  "β=0.9 (fixed)", "tomato")]:
            base = sub_s3[(sub_s3["tau"] == tau) & (sub_s3["momentum"] == beta)]
            if not base.empty:
                val = base[base["epoch"] == base["epoch"].max()]["test_acc"].mean()
                ax.axhline(val, color=col, linestyle=ls, linewidth=1.1,
                           alpha=0.8, label=f"{lbl} ({val:.1f}%)")

        ax.set_xlabel("Offset δ above Liu bound")
        ax.set_ylabel("Final test accuracy (%)")
        ax.set_title(f"τ = {tau}  — sweet spot for µ = min(0.95, µ_liu + δ)")
        ax.legend(fontsize=7)
        ax.set_xticks(offsets_x)

    fig.suptitle(
        "Exp H: How far above the Liu bound can we push momentum?\n"
        "δ=0 is the Liu bound; larger δ = more momentum than theory allows",
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "plot_M_above_bound_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    final = df[df["epoch"] == df["epoch"].max()]
    stats = (final.groupby(["experiment", "tau", "momentum"])["test_acc"]
             .agg(mean="mean", std="std").round(2))
    print("\nMean test accuracy (%) by experiment / τ / β:\n")
    print(stats.to_string())
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--step3-dir", default="outputs/det_grid")
    p.add_argument("--step4-dir", default="outputs/det_theory")
    p.add_argument("--out-dir",   default="outputs/plots")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df3 = load_step3(args.step3_dir)
    df4 = load_step4(args.step4_dir)
    df  = pd.concat([df3, df4], ignore_index=True)

    print_summary(df)

    # Narrative order: theory → compensation → fix → mechanism → summary → calibration
    plot_D(df,  args.out_dir)
    plot_E(df,  args.out_dir)
    plot_F(df,  args.out_dir)
    plot_G(df4, args.out_dir)   # grad_angle only exists in step4 CSVs
    plot_H(df,  args.out_dir)
    plot_J(df,  args.out_dir)   # no-op if Exp E not yet run
    plot_K(df,  args.out_dir)   # no-op if Exp F not yet run
    plot_L(df,  args.out_dir)   # no-op if Exp G not yet run
    plot_M(df,  args.out_dir)   # no-op if Exp H not yet run


if __name__ == "__main__":
    main()
