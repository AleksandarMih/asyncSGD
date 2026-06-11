#!/usr/bin/env bash
# Extend Ilia's Step 5 grid with β=0 baseline.
#
# Ilia ran τ ∈ {0,4,16,64} × β=0.9 only. This sweep adds β=0 at the same
# τ values (FIFO) and the matching geometric K values, so we can compare
# the two momentum regimes at the same delays.
#
# FIFO:      τ ∈ {0,4,16,64} × β ∈ {0.0} × 3 seeds = 12 runs
# Geometric: K ∈ {5,17,65}   × β ∈ {0.0} × 3 seeds =  9 runs
# Total: 21 runs × ~30s ≈ 11 min
# Output: outputs/step5_ext/

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS=(42 123 456)
TRAIN="python3 scripts/train.py"
COMMON_FLAGS="--epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
              --batch-size 128 --weight-decay 1e-4 --data-root data
              --out-dir outputs/step5_ext"

mkdir -p outputs/step5_ext

total=$(( 4 * 1 * 3 + 3 * 1 * 3 ))
run=0

echo "=== Ilia β=0 extension: FIFO + geometric (${total} runs) ==="

# ── FIFO ──────────────────────────────────────────────────────────────
echo ""
echo "--- FIFO τ ∈ {0,4,16,64}, β=0 ---"
for tau in 0 4 16 64; do
    for seed in "${SEEDS[@]}"; do
        run=$(( run + 1 ))
        echo "=== Run ${run}/${total}: FIFO τ=${tau} β=0 seed=${seed} ==="
        # shellcheck disable=SC2086
        $TRAIN --delay-type fifo --tau "$tau" --momentum 0.0 --seed "$seed" \
               --experiment A $COMMON_FLAGS
    done
done

# ── Geometric ─────────────────────────────────────────────────────────
echo ""
echo "--- Geometric K ∈ {5,17,65}, β=0 ---"
for M in 5 17 65; do
    for seed in "${SEEDS[@]}"; do
        run=$(( run + 1 ))
        echo "=== Run ${run}/${total}: Geo M=${M} (E[τ]=$(( M - 1 ))) β=0 seed=${seed} ==="
        # shellcheck disable=SC2086
        $TRAIN --delay-type geometric --M "$M" --momentum 0.0 --seed "$seed" \
               --experiment A $COMMON_FLAGS
    done
done

echo ""
echo "All ${total} runs complete. Results in outputs/step5_ext/"
