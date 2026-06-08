#!/usr/bin/env bash
# Exp G — Adaptive momentum schedule derived from Liu et al. NeurIPS 2018.
#
# Theory: τ ≲ (1−µ)²/η → µ_safe(η) = 1 − √(τ·η).
# Since η decays at LR milestones, µ_safe changes throughout training.
# Standard training violates the bound in the early high-LR phase; this
# experiment schedules µ to track the bound.
#
# Four schedules tested per τ (concrete values in notes/related_papers.md):
#   at_bound    — track Liu bound exactly each phase
#   below_bound — conservative: bound − 0.3
#   above_bound — stress test: bound + 0.3 (control: exceeds bound, should hurt)
#   neg_ramp    — Mitliagkas phase 1 (negative), Liu bound phases 2-3
#
# Baselines (β=0, β=0.9) come from step3 data — not re-run here.
#
# 36 runs × ~30s GPU ≈ 18 min total.
# After completion: python3 scripts/analyze_det_theory.py

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAUS=(5 10 20)
SCHEDULES=(at_bound below_bound above_bound neg_ramp)
SEEDS=(42 123 456)
TRAIN="python3 scripts/train.py"
COMMON="--epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
        --batch-size 128 --weight-decay 1e-4 --data-root data
        --out-dir outputs/det_theory"

total=$(( ${#TAUS[@]} * ${#SCHEDULES[@]} * ${#SEEDS[@]} ))
run=0

mkdir -p outputs/det_theory

echo "=== Exp G: Adaptive momentum schedule (${total} runs) ==="

for tau in "${TAUS[@]}"; do
    echo ""
    echo "--- τ=${tau} ---"
    for sched in "${SCHEDULES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run=$(( run + 1 ))
            echo ""
            echo "--- Run ${run}/${total}: ExpG  tau=${tau}  schedule=${sched}  seed=${seed} ---"
            # shellcheck disable=SC2086
            $TRAIN --tau "$tau" --schedule "$sched" --seed "$seed" \
                   --experiment G $COMMON
        done
    done
done

echo ""
echo "All ${total} Exp G runs complete. Results in outputs/det_theory/"
echo "Next: python3 scripts/analyze_det_theory.py"
