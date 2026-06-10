"""
plot_l2.py — weight L2 norm vs epoch, with optional bound overlays.

Reads:
  outputs/grad_bound/traj_grad_bound.csv   (produced by sweep_grad_bound.py)
  outputs/grad_bound/G_estimate.txt        (produced by sweep_grad_bound.py)

Produces three PNGs in outputs/grad_bound/plots/:
  weight_norm_no_bound.png       — actual ‖w‖ only
  weight_norm_bound_log.png      — actual ‖w‖ + bound ‖w₀‖ + η·G·log(1+t)
  weight_norm_bound_epoch.png    — actual ‖w‖ + bound ‖w₀‖ + η·G·epoch

Run:  python scripts/plot_l2.py
"""

import os

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(_SCRIPTS_DIR, "..", "outputs", "grad_bound")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

CSV_PATH     = os.path.join(OUT_DIR, "traj_grad_bound.csv")
G_EST_PATH   = os.path.join(OUT_DIR, "G_estimate.txt")


def load_g_estimate(path: str) -> dict[str, float]:
    params: dict[str, float] = {}
    with open(path) as fh:
        for line in fh:
            if "=" in line:
                key, val = line.split("=", 1)
                params[key.strip()] = float(val.strip())
    return params


def plot_weight_norm(
    df: pd.DataFrame,
    ax: Axes,
    bound_y: np.ndarray | None,
    bound_label: str | None,
) -> None:
    for tau, grp in df.groupby("tau"):
        epochs  = grp["epoch"].to_numpy()
        mean_wn = grp["mean_weight_norm"].to_numpy()
        std_wn  = grp["std_weight_norm"].to_numpy()
        ax.plot(epochs, mean_wn, label=f"τ={tau}", linewidth=1.5)
        ax.fill_between(epochs, mean_wn - std_wn, mean_wn + std_wn, alpha=0.15)

    if bound_y is not None:
        epochs_all = np.arange(0, int(df["epoch"].max()) + 1)
        ax.plot(epochs_all, bound_y, color="black", linestyle="--",
                linewidth=1.5, label=bound_label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight L2 norm  ‖w‖")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> None:
    df     = pd.read_csv(CSV_PATH)
    params = load_g_estimate(G_EST_PATH)

    G               = params["G"]
    w0_mean         = params["w0_norm_mean"]
    lr              = params["lr"]
    steps_per_epoch = int(params["steps_per_epoch"])
    max_epoch       = int(df["epoch"].max())
    epochs_all      = np.arange(0, max_epoch + 1)

    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 1. No bound ---
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_weight_norm(df, ax, bound_y=None, bound_label=None)
    ax.set_title("Weight norm vs epoch  |  β=0  wd=0")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "weight_norm_no_bound.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # --- 2. Log bound: ‖w₀‖ + η·G·log(1 + t),  t = epoch * steps_per_epoch ---
    t_log   = epochs_all * steps_per_epoch
    bound_log = w0_mean + lr * G * np.log1p(t_log)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_weight_norm(df, ax, bound_y=bound_log,
                     bound_label="Bound: ‖w₀‖ + η G log(1+t)")
    ax.set_title("Weight norm — log bound  |  β=0  wd=0")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "weight_norm_bound_log.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # --- 3. Epoch bound: ‖w₀‖ + η·G·epoch ---
    bound_epoch = w0_mean + lr * G * epochs_all

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_weight_norm(df, ax, bound_y=bound_epoch,
                     bound_label="Bound: ‖w₀‖ + η G · epoch")
    ax.set_title("Weight norm — epoch bound  |  β=0  wd=0")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "weight_norm_bound_epoch.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
