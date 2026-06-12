#!/usr/bin/env bash
# Exp F — Negative explicit momentum sweep.
#
# Theory (Mitliagkas 2014): delay τ adds implicit momentum ≈ (τ-1)/τ.
# To push effective momentum toward zero, explicit β should be negative:
#   β_optimal ≈ −(τ−1)/τ
#
# Tested values:
#   τ=5  → β_opt ≈ −0.80   |  sweep: −0.5, −0.3, −0.1
#   τ=10 → β_opt ≈ −0.90   |  sweep: −0.5, −0.3, −0.1
#   τ=20 → β_opt ≈ −0.95   |  sweep: −0.5, −0.3, −0.1
#
# Baselines (β=0, 0.5, 0.9) come from existing step3 data — not re-run here.
# Results are written to outputs/det_theory/ alongside Exp A–E data.
#
# After completion:
#   python scripts/analyze_det_theory.py   →  produces plot_K_negative_momentum.png

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS=(42 123 456)
F_TAUS=(5 10 20)
F_BETAS=(-0.5 -0.3 -0.1)
TRAIN="python3 scripts/train.py"
COMMON="--epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
        --batch-size 128 --weight-decay 1e-4 --data-root data
        --out-dir outputs/det_theory"

total=$(( ${#F_TAUS[@]} * ${#F_BETAS[@]} * ${#SEEDS[@]} ))
run=0

echo "=== Exp F: Negative momentum (${total} runs) ==="

for tau in "${F_TAUS[@]}"; do
    beta_opt=$(python3 -c "tau=$tau; print(f'-{(tau-1)/tau:.2f}')")
    echo ""
    echo "--- τ=${tau}  β_optimal≈${beta_opt} ---"

    for beta in "${F_BETAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run=$(( run + 1 ))
            echo ""
            echo "--- Run ${run}/${total}: ExpF  tau=${tau}  beta=${beta}  seed=${seed} ---"
            # shellcheck disable=SC2086
            $TRAIN --tau "$tau" --momentum "$beta" --seed "$seed" \
                   --experiment F $COMMON
        done
    done
done

echo ""
echo "All ${total} Exp F runs complete. Results in outputs/det_theory/"
echo "Next: python scripts/analyze_det_theory.py"
