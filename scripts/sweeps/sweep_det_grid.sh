#!/usr/bin/env bash
# Runs all 45 (τ, β, seed) combinations for the Step 2 & 3 experiment.
# β=0.0 column = Step 2 (delay only).  All three columns = Step 3 (delay × momentum).
#
# Usage (from repo root or scripts/):
#   bash scripts/sweep_det_grid.sh
#
# Estimated time: ~30s/run on GPU, ~3–5 min/run on CPU.
# Output: outputs/det_grid/tau{τ}_beta{β}_seed{seed}.csv  (one file per run)

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAUS=(0 2 5 10 20)
BETAS=(0.0 0.5 0.9)
SEEDS=(42 123 456)

total=$(( ${#TAUS[@]} * ${#BETAS[@]} * ${#SEEDS[@]} ))
run=0

for tau in "${TAUS[@]}"; do
    for beta in "${BETAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run=$(( run + 1 ))
            echo ""
            echo "=== Run ${run}/${total}: tau=${tau}  beta=${beta}  seed=${seed} ==="
            python scripts/train_delayed_sgd.py \
                --tau          "$tau"  \
                --momentum     "$beta" \
                --seed         "$seed" \
                --epochs       50      \
                --lr           0.1     \
                --lr-milestones 30 40  \
                --lr-gamma     0.1     \
                --batch-size   128     \
                --weight-decay 1e-4    \
                --data-root    data    \
                --out-dir      outputs/det_grid
        done
    done
done

echo ""
echo "All ${total} runs complete. Results in outputs/det_grid/"
echo "Run:  python scripts/analyze_det_grid.py"
