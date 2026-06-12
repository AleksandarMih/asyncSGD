#!/usr/bin/env bash
# Exp A (Geometric) — Mitliagkas equivalence test under stochastic delay.
#
# Mitliagkas (2014) claim: M async workers with geometric staleness (M/M/1 queue)
# induce implicit momentum µ_S = (M−1)/M. So training with geometric delay M and
# β=0 should match training with τ=0 and β=(M−1)/M.
#
# Under deterministic delay the equivalence fails (confirmed by Liu Remark 7).
# This sweep tests whether the equivalence holds when delay is actually geometric,
# i.e. when the theoretical preconditions are satisfied.
#
# Pairs tested (M | geometric β=0) vs (τ=0 | β=µ_S):
#   M=3:  µ_S=0.667  →  compare expA_geo_M3_beta0.0 vs expA_tau0_beta0.666667
#   M=6:  µ_S=0.833  →  compare expA_geo_M6_beta0.0 vs expA_tau0_beta0.833333
#
# Runs: 2 M values × 2 conditions × 3 seeds = 12 runs
# Note: τ=0, β=µ_S conditions reuse existing step3 data if available,
#       or re-run with fifo delay-type here for clean labeling.
#
# 12 runs × ~30 s GPU ≈ 6 min total.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS=(42 123 456)
TRAIN="python3 scripts/train.py"
COMMON="--experiment A
        --epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
        --batch-size 128 --weight-decay 1e-4 --data-root data"

total=$(( 2 * 2 * ${#SEEDS[@]} ))
run=0

mkdir -p outputs/geo_grid outputs/det_theory

echo "=== Exp A Geometric: Mitliagkas equivalence test (${total} runs) ==="

for M in 3 6; do
    mu_s_num=$(( M - 1 ))
    mu_s=$(python3 -c "print(f'{(${M}-1)/${M}:.6f}')")
    echo ""
    echo "--- M=${M}: µ_S = (M−1)/M = ${mu_s} ---"

    # Condition 1: geometric delay M, β=0 (should exhibit implicit µ_S)
    for seed in "${SEEDS[@]}"; do
        run=$(( run + 1 ))
        echo ""
        echo "=== Run ${run}/${total}: geo M=${M}  beta=0.0  seed=${seed} ==="
        # shellcheck disable=SC2086
        $TRAIN --delay-type geometric --M "$M" --momentum 0.0 --seed "$seed" \
               --out-dir outputs/geo_grid $COMMON
    done

    # Condition 2: no delay, β=µ_S (the theoretically equivalent run)
    for seed in "${SEEDS[@]}"; do
        run=$(( run + 1 ))
        echo ""
        echo "=== Run ${run}/${total}: fifo tau=0  beta=${mu_s}  seed=${seed} ==="
        # shellcheck disable=SC2086
        $TRAIN --delay-type fifo --tau 0 --momentum "$mu_s" --seed "$seed" \
               --out-dir outputs/det_theory $COMMON
    done
done

echo ""
echo "All ${total} Exp A geometric runs complete."
echo "Geo results: outputs/geo_grid/  |  Fifo baseline: outputs/det_theory/"
echo "Next: python3 scripts/analyze_geo.py"
