#!/bin/bash
# configs_step5.txt: DIST  K  TAU  LRSCALE  MOM  SEED
> configs_step5.txt
MOM=0.9

for seed in 42 123 456; do

  # Exp 1: convergence
  for tau in 0 4 16 64; do
    echo "fifo 0 $tau 1.0 $MOM $seed" >> configs_step5.txt
  done
  for K in 5 17 65; do
    echo "geometric $K 0 1.0 $MOM $seed" >> configs_step5.txt
  done

  # Exp 2: LR rescue at worst delay
  for lrs in 0.125 0.015625; do
    echo "fifo 0 64 $lrs $MOM $seed" >> configs_step5.txt
    echo "geometric 65 0 $lrs $MOM $seed" >> configs_step5.txt
  done

done

n=$(wc -l < configs_step5.txt)
echo "Generated $n configs (expected 33)."
