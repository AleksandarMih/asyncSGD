#!/usr/bin/env bash
# Geo-only relaunch of lrs rescue sweep.
# The original sweep_lrs_rescue.sh keyed lrs arrays by {4,16,64} but looped
# over M∈{5,17,65}, causing empty lrs → argparse crash on every geo run.
# This script hardcodes the correct M→lrs mapping.
#
# Geometric K∈{5,17,65} (E[τ]≈{4,16,64}) × β∈{0,0.9} × lrs∈{1/√E[τ], 1/E[τ]} × seed=42
# Total: 12 runs × ~30s ≈ 6 min
# Output: outputs/step5_ext/expC_geo_M{M}_lrscale{lrs}_beta{beta}_seed42.csv

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED=42
TRAIN="python3 scripts/train.py"
COMMON_FLAGS="--epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
              --batch-size 128 --weight-decay 1e-4 --data-root data
              --out-dir outputs/step5_ext"

mkdir -p outputs/step5_ext

# M → (lrs_sqrt, lrs_inv)  using E[τ]=M-1≈{4,16,64}
declare -A LRS_SQRT=( [5]="0.5" [17]="0.25" [65]="0.125" )
declare -A LRS_INV=(  [5]="0.25" [17]="0.0625" [65]="0.015625" )

total=$(( 3 * 2 * 2 ))
run=0

echo "=== LR rescue sweep (geo only): ${total} runs ==="

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
echo "All ${total} geo runs complete. Results in outputs/step5_ext/"
