"""Print convergence metrics for Q1 report numbers."""
import glob
import numpy as np
import pandas as pd

paths = sorted(glob.glob("outputs/det_grid/tau*_beta*_seed*.csv"))
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
df["tau"]      = df["tau"].astype(int)
df["momentum"] = df["momentum"].astype(float)

TAUS = [0, 2, 5, 10, 20]
BETA = 0.0

curves = (df[df["momentum"] == BETA]
          .groupby(["tau", "epoch"])["test_acc"]
          .mean().reset_index())

print("=== Epochs to threshold (β=0) ===")
for tau in TAUS:
    grp = curves[curves["tau"] == tau].sort_values("epoch")
    for thr in [80.0, 85.0]:
        hit = grp[grp["test_acc"] >= thr]
        e = int(hit["epoch"].iloc[0]) if not hit.empty else "DNR"
        print(f"  tau={tau}, >={thr:.0f}%: epoch {e}")

print()
print("=== Normalised AUC (β=0) ===")
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = lambda y, x: sum((y[i]+y[i+1])*(x[i+1]-x[i])/2 for i in range(len(y)-1))
auc0 = None
for tau in TAUS:
    grp = curves[curves["tau"] == tau].sort_values("epoch")
    a = _trapz(grp["test_acc"].values, grp["epoch"].values)
    if auc0 is None:
        auc0 = a
    print(f"  tau={tau}: AUC={a:.1f}, norm={a/auc0:.3f}")
