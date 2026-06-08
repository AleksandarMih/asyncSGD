#!/usr/bin/env bash
# Exp H — Above-bound offset sweep: finding the sweet spot.
#
# Exp G showed that above_bound (µ = µ_liu + 0.3) wins at every τ.
# This sweep asks: how far above the Liu bound can we go before accuracy
# degrades?  δ=0.0 (at_bound) and δ=0.3 (above_bound) already exist in
# Exp G.  Only τ ∈ {10, 20} are tested — the delay regime where the bound
# matters and the above_bound effect was large (3.5pp and 6pp respectively).
#
# New offsets: δ ∈ {0.1, 0.5, 0.7}  → 18 new runs (~1.5h on GPU)
# Combined with Exp G anchor points: 5 points per τ for plot_M.
#
# After completion: python3 scripts/analyze_det_theory.py → plot_M_above_bound_sweep.png

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAUS=(10 20)
OFFSETS=(0.1 0.5 0.7)
SEEDS=(42 123 456)
TRAIN="python3 scripts/train.py"
COMMON="--epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
        --batch-size 128 --weight-decay 1e-4 --data-root data
        --out-dir outputs/det_theory"

total=$(( ${#TAUS[@]} * ${#OFFSETS[@]} * ${#SEEDS[@]} ))
run=0

mkdir -p outputs/det_theory

echo "=== Exp H: Above-bound offset sweep (${total} runs) ==="

for tau in "${TAUS[@]}"; do
    echo ""
    echo "--- τ=${tau} ---"
    for offset in "${OFFSETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run=$(( run + 1 ))
            echo ""
            echo "--- Run ${run}/${total}: ExpH  tau=${tau}  offset=${offset}  seed=${seed} ---"
            # shellcheck disable=SC2086
            $TRAIN --tau "$tau" --schedule above_bound --bound-offset "$offset" \
                   --seed "$seed" --experiment H $COMMON
        done
    done
done

echo ""
echo "All ${total} Exp H runs complete. Results in outputs/det_theory/"
echo "Next: python3 scripts/analyze_det_theory.py"
