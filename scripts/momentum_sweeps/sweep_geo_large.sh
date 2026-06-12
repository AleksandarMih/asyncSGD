#!/usr/bin/env bash
# Geometric delay stability sweep — extended M range.
#
# Goal: find the breaking point for β=0 under geometric delay.
# M=21 (E[τ]=20) is still stable; sweep M ∈ {31,51,101,201,501} to locate
# where geometric delay causes clear convergence failure.
#
# 45 runs × ~30 s GPU ≈ 23 min total.
# Output: outputs/geo_large/
# After completion: python3 scripts/analyze_geo.py

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MS=(31 51 101 201 501)
BETAS=(0.0 0.5 0.9)
SEEDS=(42 123 456)
TRAIN="python3 scripts/train.py"
COMMON="--delay-type geometric
        --experiment A
        --epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
        --batch-size 128 --weight-decay 1e-4 --data-root data
        --out-dir outputs/geo_large"

total=$(( ${#MS[@]} * ${#BETAS[@]} * ${#SEEDS[@]} ))
run=0

mkdir -p outputs/geo_large

echo "=== Geometric large-M sweep: find stability boundary (${total} runs) ==="
echo "M ∈ {31,51,101,201,501}  →  E[τ] ∈ {30,50,100,200,500}"

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
echo "All ${total} runs complete. Results in outputs/geo_large/"
echo "Next: python3 scripts/analyze_geo.py"
