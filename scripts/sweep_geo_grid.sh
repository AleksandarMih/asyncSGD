#!/usr/bin/env bash
# Step 3 — Geometric delay baseline: M×β×seed sweep.
#
# Geometric delay: τ_t ~ Geom(1/M)−1, E[τ] = M−1.
# M values are chosen so E[τ] matches the deterministic τ values in step3:
#   M=1  → E[τ]=0   (no delay, baseline)
#   M=3  → E[τ]=2
#   M=6  → E[τ]=5
#   M=11 → E[τ]=10
#   M=21 → E[τ]=20
#
# This allows direct comparison of deterministic vs geometric delay at the same
# average staleness, testing whether delay distribution type changes momentum dynamics.
#
# 45 runs × ~30 s GPU ≈ 23 min total.
# Output: outputs/geo_grid/
# After completion: python3 scripts/analyze_geo.py

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MS=(1 3 6 11 21)
BETAS=(0.0 0.5 0.9)
SEEDS=(42 123 456)
TRAIN="python3 scripts/train.py"
COMMON="--delay-type geometric
        --experiment A
        --epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
        --batch-size 128 --weight-decay 1e-4 --data-root data
        --out-dir outputs/geo_grid"

total=$(( ${#MS[@]} * ${#BETAS[@]} * ${#SEEDS[@]} ))
run=0

mkdir -p outputs/geo_grid

echo "=== Step 3 Geometric: M×β×seed sweep (${total} runs) ==="
echo "E[τ] = M−1 matches deterministic τ ∈ {0,2,5,10,20}"

for M in "${MS[@]}"; do
    echo ""
    echo "--- M=${M}  (E[τ]=$(( M - 1 ))) ---"
    for beta in "${BETAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run=$(( run + 1 ))
            echo ""
            echo "=== Run ${run}/${total}: M=${M}  beta=${beta}  seed=${seed} ==="
            # shellcheck disable=SC2086
            $TRAIN --M "$M" --momentum "$beta" --seed "$seed" $COMMON
        done
    done
done

echo ""
echo "All ${total} runs complete. Results in outputs/geo_grid/"
echo "Next: python3 scripts/analyze_geo.py"
