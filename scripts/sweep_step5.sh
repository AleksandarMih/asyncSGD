#!/usr/bin/env bash
# Step 5 sweep — Mitliagkas equivalence in the true HogWild! setting.
#
# Theory (Mitliagkas 2014): k async workers with β=0 ≡ 1 worker with β=(k−1)/k.
#
# For each k ∈ {2, 4, 8} and each seed, we run a matched pair:
#   hogwild mode    — k workers, β=0
#   sequential mode — 1 worker,  β=(k−1)/k
#
# All 18 runs are launched in background (&) so they run in parallel.
# CPU budget: k=2→3, k=4→5, k=8→9 CPUs per hogwild run (workers+main);
#             sequential runs use 1 CPU each.
# Total CPUs needed if all run simultaneously: ~60.  Request ≥64 on RunAI.
#
# Estimated wall-clock time (server CPU, ResNet-20/CIFAR-10):
#   hogwild   k=8: ~12 min  (50 rounds × ~14s per round, workers in parallel)
#   sequential:    ~100 min (50 epochs × ~2 min on a modern CPU core)
#   → ~100 min total wall-clock with all runs in parallel.
#
# After all runs complete:
#   python scripts/analyze_step5.py

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS=(42 123 456)
KS=(2 4 8)
TRAIN="python3 scripts/train_step5.py"
COMMON="--epochs 50 --lr 0.1 --lr-milestones 30 40 --lr-gamma 0.1
        --batch-size 128 --weight-decay 1e-4 --data-root data
        --out-dir outputs/step5"

mkdir -p outputs/step5
pids=()

echo "=== Step 5: Launching all runs in parallel ==="
echo ""

for k in "${KS[@]}"; do
    beta_seq=$(python3 -c "print(round(($k-1)/$k, 6))")

    for seed in "${SEEDS[@]}"; do
        echo "--- hogwild     k=${k}  β=0.0       seed=${seed} ---"
        # shellcheck disable=SC2086
        $TRAIN --mode hogwild --num-workers "$k" --seed "$seed" $COMMON \
            > "outputs/step5/log_hogwild_k${k}_seed${seed}.txt" 2>&1 &
        pids+=($!)

        echo "--- sequential  k=${k}  β=${beta_seq}  seed=${seed} ---"
        # shellcheck disable=SC2086
        $TRAIN --mode sequential --num-workers "$k" --seed "$seed" $COMMON \
            > "outputs/step5/log_sequential_k${k}_seed${seed}.txt" 2>&1 &
        pids+=($!)
    done
done

echo ""
echo "All ${#pids[@]} runs launched. Waiting for completion..."
echo "(logs in outputs/step5/log_*.txt)"
echo ""

# Wait for all background jobs and collect exit codes.
fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "WARNING: process $pid failed" >&2
        fail=1
    fi
done

if [[ $fail -eq 0 ]]; then
    echo "All runs complete. Results in outputs/step5/"
    echo "Next: python scripts/analyze_step5.py"
else
    echo "Some runs failed — check logs in outputs/step5/" >&2
    exit 1
fi
