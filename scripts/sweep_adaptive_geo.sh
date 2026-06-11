#!/usr/bin/env bash
# Exp G (Geometric) — Adaptive Liu-bound schedule under stochastic delay.
#
# The Liu (2018) bound τ ≲ (1−µ)²/η was derived for deterministic fixed delay.
# Under geometric delay with E[τ] = M−1, substituting E[τ] into the bound gives
# µ_safe = 1 − √(E[τ]·η). This is an approximation — we test it empirically.
#
# The adaptive schedule adjusts µ at each LR decay milestone (tracking the bound).
# If the approximation is valid, schedules that respect µ_safe(E[τ]) should still
# avoid divergence even though individual τ_t may spike above E[τ].
#
# M values and equivalent E[τ]:
#   M=3  → E[τ]=2   → µ_safe(η=0.1)≈0.553, µ_safe(η=0.01)≈0.900
#   M=6  → E[τ]=5   → µ_safe(η=0.1)≈0.293, µ_safe(η=0.01)≈0.776
#   M=11 → E[τ]=10  → µ_safe(η=0.1)≈0.000, µ_safe(η=0.01)≈0.684
#
# Schedules: at_bound, below_bound, above_bound (stress test), neg_ramp
# 36 runs × ~30 s GPU ≈ 18 min total.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MS=(3 6 11)
SCHEDULES=(at_bound below_bound above_bound neg_ramp)
SEEDS=(42 123 456)
TRAIN="python3 scripts/train.py"
COMMON="--delay-type geometric
        --experiment G
        --epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
        --batch-size 128 --weight-decay 1e-4 --data-root data
        --out-dir outputs/geo_theory"

total=$(( ${#MS[@]} * ${#SCHEDULES[@]} * ${#SEEDS[@]} ))
run=0

mkdir -p outputs/geo_theory

echo "=== Exp G Geometric: adaptive schedule under stochastic delay (${total} runs) ==="

for M in "${MS[@]}"; do
    echo ""
    echo "--- M=${M}  (E[τ]=$(( M - 1 ))) ---"
    for sched in "${SCHEDULES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run=$(( run + 1 ))
            echo ""
            echo "=== Run ${run}/${total}: M=${M}  schedule=${sched}  seed=${seed} ==="
            # shellcheck disable=SC2086
            $TRAIN --M "$M" --schedule "$sched" --seed "$seed" $COMMON
        done
    done
done

echo ""
echo "All ${total} Exp G geometric runs complete. Results in outputs/geo_theory/"
echo "Next: python3 scripts/analyze_geo.py"
