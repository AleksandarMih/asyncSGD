#!/usr/bin/env bash
# LR rescue sweep: test η → η/√τ and η/τ at collapsed delay values.
#
# Theory: gradient variance scales with τ, so rescaling η by 1/√τ (or 1/τ)
# should restore convergence. We test both β=0 and β=0.9 to see whether
# LR correction interacts with momentum choice.
#
# FIFO      τ ∈ {4,16,64} × β ∈ {0.0,0.9} × lrs ∈ {1/√τ, 1/τ} × seed=42 = 12 runs
# Geometric K ∈ {5,17,65}  × β ∈ {0.0,0.9} × lrs ∈ {1/√K, 1/K} × seed=42 = 12 runs
# Total: 24 runs × ~30s ≈ 12 min
# Output: outputs/step5_ext/

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEED=42
TRAIN="python3 scripts/train.py"
COMMON_FLAGS="--epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
              --batch-size 128 --weight-decay 1e-4 --data-root data
              --out-dir outputs/step5_ext"

mkdir -p outputs/step5_ext

# lrs = 1/√τ and 1/τ for each τ value
declare -A LRS_SQRT=( [4]="0.5" [16]="0.25" [64]="0.125" )
declare -A LRS_INV=(  [4]="0.25" [16]="0.0625" [64]="0.015625" )

total=$(( 3 * 2 * 2 + 3 * 2 * 2 ))
run=0

echo "=== LR rescue sweep: FIFO + geometric (${total} runs) ==="

# ── FIFO ──────────────────────────────────────────────────────────────
echo ""
echo "--- FIFO τ ∈ {4,16,64}, β ∈ {0,0.9}, lrs ∈ {1/√τ, 1/τ} ---"
for tau in 4 16 64; do
    for beta in 0.0 0.9; do
        for lrs in "${LRS_SQRT[$tau]}" "${LRS_INV[$tau]}"; do
            run=$(( run + 1 ))
            echo "=== Run ${run}/${total}: FIFO τ=${tau} β=${beta} lrs=${lrs} ==="
            # shellcheck disable=SC2086
            $TRAIN --delay-type fifo --tau "$tau" --momentum "$beta" \
                   --lr-scale "$lrs" --seed "$SEED" \
                   --experiment C $COMMON_FLAGS
        done
    done
done

# ── Geometric ─────────────────────────────────────────────────────────
echo ""
echo "--- Geometric K ∈ {5,17,65}, β ∈ {0,0.9}, lrs ∈ {1/√K, 1/K} ---"
for M in 5 17 65; do
    for beta in 0.0 0.9; do
        for lrs in "${LRS_SQRT[$M]}" "${LRS_INV[$M]}"; do
            run=$(( run + 1 ))
            echo "=== Run ${run}/${total}: Geo M=${M} β=${beta} lrs=${lrs} ==="
            # shellcheck disable=SC2086
            $TRAIN --delay-type geometric --M "$M" --momentum "$beta" \
                   --lr-scale "$lrs" --seed "$SEED" \
                   --experiment C $COMMON_FLAGS
        done
    done
done

echo ""
echo "All ${total} runs complete. Results in outputs/step5_ext/"
