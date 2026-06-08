#!/usr/bin/env bash
# Exp F extended — push negative momentum further at τ=20.
#
# Original sweep (sweep_neg_momentum.sh) tested β ∈ {-0.5, -0.3, -0.1} × τ ∈ {5,10,20}.
# Results showed a clear monotone gain at τ=20 (β=-0.5 → +11pp over β=0),
# with no plateau — theory predicts β_opt ≈ -(τ-1)/τ = -0.95 at τ=20.
# τ=5 and τ=10 showed no signal, so only τ=20 is extended here.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS=(42 123 456)
F_BETAS=(-0.7 -0.9)
TRAIN="python3 scripts/train.py"
COMMON="--epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
        --batch-size 128 --weight-decay 1e-4 --data-root data
        --out-dir outputs/det_theory"

total=$(( ${#F_BETAS[@]} * ${#SEEDS[@]} ))
run=0

echo "=== Exp F2: Extended negative momentum at τ=20 (${total} runs) ==="

for beta in "${F_BETAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run=$(( run + 1 ))
        echo ""
        echo "--- Run ${run}/${total}: ExpF  tau=20  beta=${beta}  seed=${seed} ---"
        # shellcheck disable=SC2086
        $TRAIN --tau 20 --momentum "$beta" --seed "$seed" \
               --experiment F $COMMON
    done
done

echo ""
echo "All ${total} Exp F2 runs complete. Results in outputs/det_theory/"
echo "Next: python3 scripts/analyze_det_theory.py"
