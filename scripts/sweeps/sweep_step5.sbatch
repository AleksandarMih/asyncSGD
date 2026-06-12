#!/bin/bash
#SBATCH --job-name=step5-async
#SBATCH --array=0-32%8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --time=03:30:00
#SBATCH --output=logs/%x_%a.out

set -euo pipefail

module load gcc python cuda
source ~/venvs/asyncsgd/bin/activate

cd "$SLURM_SUBMIT_DIR"

cfg=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" configs_step5.txt)
read -r DIST K TAU LRS MOM SEED <<< "$cfg"

echo "task=$SLURM_ARRAY_TASK_ID dist=$DIST K=$K tau=$TAU lrs=$LRS mom=$MOM seed=$SEED"

srun python scripts/train_step5.py \
    --dist "$DIST" \
    --K "$K" \
    --tau "$TAU" \
    --lr-scale "$LRS" \
    --momentum "$MOM" \
    --seed "$SEED" \
    --epochs 50 \
    --lr 0.1 \
    --batch-size 128 \
    --weight-decay 1e-4 \
    --c-mean 4 \
    --data-root /scratch/izar/badanin/cifar10 \
    --out-dir outputs/step5
