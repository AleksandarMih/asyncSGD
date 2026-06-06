#!/usr/bin/env bash
# Step 4 experiment sweep — 30 new training runs total.
#
# Experiments (details in train_step4.py docstring):
#   Exp A — Equivalence test  (6 new runs; τ=k,β=0 side reused from step3)
#   Exp B — Compensation      (6 new runs; τ=10/20 comp→β=0 reused from step3)
#   Exp C — LR scaling        (18 new runs; lr_scale=1.0 reused from step3)
#
# After all runs complete:
#   python scripts/analyze_step4.py
#
# Exp D (conditional) is commented out below. Fill in SCALE_STAR from the
# Exp C analysis before uncommenting.
#
# Estimated time: ~30s/run on GPU → ≈ 15 min total for Exp A–C.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS=(42 123 456)
TRAIN="python3 scripts/train_step4.py"
COMMON="--epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
        --batch-size 128 --weight-decay 1e-4 --data-root data
        --out-dir outputs/step4"

total=30
run=0

# ---------------------------------------------------------------------------
# Exp A — Equivalence test
#
# Claim (Mitliagkas 2014): delay τ adds implicit momentum ≈ (τ-1)/τ.
# Each τ pairs a delayed run (τ=k, β=0) with a pure-momentum run (τ=0, β=(k-1)/k).
#
#   τ=2  ↔  β=0.5   (already in step3)
#   τ=5  ↔  β=0.8   ← new run
#   τ=10 ↔  β=0.9   (already in step3)
#   τ=20 ↔  β=0.95  ← new run
#
# The (τ=k, β=0) side already exists in step3; analyze_step4.py loads it there.
# Only the missing pure-momentum equivalents are run here.
# ---------------------------------------------------------------------------
echo "=== Exp A: Equivalence test (6 new runs) ==="
for beta in 0.8 0.95; do
    for seed in "${SEEDS[@]}"; do
        run=$(( run + 1 ))
        echo ""
        echo "--- Run ${run}/${total}: ExpA  tau=0  beta=${beta}  seed=${seed} ---"
        # shellcheck disable=SC2086
        $TRAIN --tau 0 --momentum "$beta" --seed "$seed" \
               --experiment A $COMMON
    done
done

# ---------------------------------------------------------------------------
# Exp B — Compensation
#
# If delay adds β_impl = (τ-1)/τ, set β_explicit = max(0, 0.9 − β_impl)
# so the total effective momentum ≈ 0.9.
#
#   τ=2  → β_comp = 0.4   ← new run
#   τ=5  → β_comp = 0.1   ← new run
#   τ=10 → β_comp = 0.0   (already in step3 as τ=10, β=0.0)
#   τ=20 → β_comp = 0.0   (already in step3 as τ=20, β=0.0)
#
# Baselines (β=0 and β=0.9 at each τ) come from step3 in the analysis.
# ---------------------------------------------------------------------------
echo ""
echo "=== Exp B: Compensation (6 new runs) ==="
for tau in 2 5; do
    beta_comp=$(python3 -c "tau=$tau; print(max(0.0, round(0.9-(tau-1)/tau, 6)))")
    for seed in "${SEEDS[@]}"; do
        run=$(( run + 1 ))
        echo ""
        echo "--- Run ${run}/${total}: ExpB  tau=${tau}  beta_comp=${beta_comp}  seed=${seed} ---"
        # shellcheck disable=SC2086
        $TRAIN --tau "$tau" --momentum "$beta_comp" --seed "$seed" \
               --experiment B $COMMON
    done
done

# ---------------------------------------------------------------------------
# Exp C — LR scaling
#
# Does scaling the learning rate by 1/τ or 1/√τ restore convergence?
# Tested at β=0 to isolate LR from momentum effects.
# lr_scale=1.0 (no scaling) is already in step3; only the two new scales run here.
# ---------------------------------------------------------------------------
echo ""
echo "=== Exp C: LR scaling (18 new runs) ==="
for tau in 5 10 20; do
    for scale_expr in "1/tau" "1/tau**0.5"; do
        lr_scale=$(python3 -c "import math; tau=$tau; print(round($scale_expr, 6))")
        for seed in "${SEEDS[@]}"; do
            run=$(( run + 1 ))
            echo ""
            echo "--- Run ${run}/${total}: ExpC  tau=${tau}  lr_scale=${lr_scale}  seed=${seed} ---"
            # shellcheck disable=SC2086
            $TRAIN --tau "$tau" --momentum 0.0 --lr-scale "$lr_scale" \
                   --seed "$seed" --experiment C $COMMON
        done
    done
done

# ---------------------------------------------------------------------------
# Exp E — Calibration sweep
#
# Empirically find which β at τ=0 best reproduces the training dynamics of
# each (τ=k, β=0) run.  If the Mitliagkas equivalence holds, the best-match
# β* should follow β* ≈ τ/(τ+1).  If not, it reveals the actual relationship
# (or absence thereof) between delay and implicit momentum in our setting.
#
# Already available at τ=0 (reused in analysis, not re-run):
#   step3: β ∈ {0.0, 0.5, 0.9}
#   step4 expA: β ∈ {0.8, 0.95}
#
# New runs: β ∈ {0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.85, 0.99}
# ---------------------------------------------------------------------------
E_BETAS=(0.1 0.2 0.3 0.4 0.6 0.7 0.85 0.99)
e_total=$(( ${#E_BETAS[@]} * ${#SEEDS[@]} ))
e_run=0

echo ""
echo "=== Exp E: Calibration sweep (${e_total} new runs) ==="
for beta in "${E_BETAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        e_run=$(( e_run + 1 ))
        echo ""
        echo "--- Exp E run ${e_run}/${e_total}: tau=0  beta=${beta}  seed=${seed} ---"
        # shellcheck disable=SC2086
        $TRAIN --tau 0 --momentum "$beta" --seed "$seed" \
               --experiment E $COMMON
    done
done

echo ""
echo "All $((total + e_total)) Exp A–E runs complete. Results in outputs/step4/"
echo "Next: python scripts/analyze_step4.py"
