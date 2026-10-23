#!/usr/bin/env bash
# One GPU per train seed. Each GPU runs every arm for its own seed.
#   ./run_parallel.sh 1 2 3          # all parts
#   SEEDS="1 2 3 4 5" ./run_parallel.sh 1 2 3
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEEDS=${SEEDS:-"1 2 3 4 5"}
[[ $# -gt 0 ]] || { echo "usage: ./run_parallel.sh <parts or arms>"; exit 1; }

gpu=0
for s in $SEEDS; do
  echo "GPU $gpu -> train seed $s"
  CUDA_VISIBLE_DEVICES=$gpu TRAIN_SEEDS="$s" NO_AGG=1 \
    ./run_sweep.sh "$@" > "sweep_gpu${gpu}_seed${s}.log" 2>&1 &
  gpu=$((gpu + 1))
done

wait
echo "all GPUs finished, aggregating"
python aggregate_sweep.py
