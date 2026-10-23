#!/usr/bin/env bash
# Multi-train-seed sweep: train N models per arm, generate M sets per model.
#
#   ./run_sweep.sh 1                       # to run all of part 1
#   ./run_sweep.sh 1 2 3                   # to run everything
#   ./run_sweep.sh part2_arm_full          # to run one arm
#   TRAIN_SEEDS="1 2 3" N_SETS=5 ./run_sweep.sh 2
#

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_SEEDS=${TRAIN_SEEDS:-"1 2 3 4 5"}   # unit of replication for the training statistics. use 5
N_SETS=${N_SETS:-3}                       # generation sets per trained model. use 3
GEN_SEED_BASE=${GEN_SEED_BASE:-42}        # gen seeds are BASE..BASE+N_SETS-1, reused across train seeds
MAX_STEPS=${MAX_STEPS:-20000}
RUNS_DIR=${RUNS_DIR:-./runs_sweep}
OUT_DIR=${OUT_DIR:-./generated_sweep}
WANDB_MODE=${WANDB_MODE:-offline}         # online | offline | disabled
PRUNE_CKPT=${PRUNE_CKPT:-1}               # 1 = delete non-final checkpoints after training

# name | train_csv | col | tokenizer_dir | rep
ARMS=(
"part1_arm_a|/data/input//part1_arm_a_train_1K.csv|safe|/data/tokenizers/tokenizer_safe|safe"
"part1_arm_b|/data/input/part1_arm_b_train_25K.csv|safe|/data/tokenizers/tokenizer_safe|safe"
"part1_arm_c|/data/input/part1_arm_c_train_50K.csv|safe|/data/tokenizers/tokenizer_safe|safe"
"part2_arm_atom|/data/input/part2_arm_atom_train.csv|safe|/data/tokenizers/tokenizer_safe|safe"
"part2_arm_dot|/data/input/part2_arm_dot_train.csv|safe|/data/tokenizers/tokenizer_safe|safe"
"part2_arm_full|/data/input/part2_arm_full_train.csv|safe|/data/tokenizers/tokenizer_safe|safe"
"part3_arm_safe|/data/input/part3_arm_safe_train.csv|safe|/data/tokenizers/tokenizer_safe|safe"
"part3_arm_smiles|/data/input/part3_arm_smiles_train.csv|smiles|/data/tokenizers/tokenizer_smiles|smiles"
)

[[ $# -gt 0 ]] || { grep '^#' "$0" | sed 's/^# \?//'; exit 1; }

selected=()
for want in "$@"; do
  for spec in "${ARMS[@]}"; do
    nm=${spec%%|*}
    if [[ "$want" =~ ^[123]$ && "$nm" == part${want}_* ]] || [[ "$nm" == "$want" ]]; then
      selected+=("$spec")
    fi
  done
done
[[ ${#selected[@]} -gt 0 ]] || { echo "no arms matched: $*" >&2; exit 1; }

export WANDB_MODE
mkdir -p "$RUNS_DIR" "$OUT_DIR" train_logs

echo "arms=${#selected[@]}  train_seeds=($TRAIN_SEEDS)  n_sets=$N_SETS"
for spec in "${selected[@]}"; do
  IFS='|' read -r name csv col tokdir rep <<< "$spec"
  [[ -f "$csv" ]] || { echo "MISSING $csv -- skipping $name" >&2; continue; }

  for s in $TRAIN_SEEDS; do
    tag="${name}_seed${s}"
    outdir="$RUNS_DIR/$name/seed$s"
    ckpt="$outdir/checkpoint-$MAX_STEPS"

    if [[ -d "$ckpt" ]]; then
      echo "[$tag] checkpoint exists, skipping train"
    else
      echo "[$tag] training -> train_logs/$tag.log"
      mkdir -p "$outdir"
      cat > "$outdir/config.yaml" <<YAML
WANDB_ENTITY: "KaushalMindLab"
WANDB_PROJECT: fragopt_sweep
RUN_NAME: $tag
WANDB_RUN_ID: $tag
SEED: $s
TRAIN_CSV: $csv
VALID_CSV: /data/input/valid_200.csv
COL: $col
TOKENIZER_DIR: $tokdir
OUTPUT_DIR: $outdir
HIDDEN_SIZE: 384
NUM_LAYERS: 5
NUM_HEADS: 8
MAX_LEN: 140
BATCH_SIZE: 64
LR: 0.0005
WEIGHT_DECAY: 0.01
WARMUP_RATIO: 0.01
MAX_STEPS: $MAX_STEPS
EVAL_STEPS: 500
LOG_STEPS: 50
BF16: true
YAML
      CONFIG="$outdir/config.yaml" python 5_pretrain.py > "train_logs/$tag.log" 2>&1
      if [[ "$PRUNE_CKPT" == "1" ]]; then
        find "$outdir" -maxdepth 1 -type d -name 'checkpoint-*' ! -name "checkpoint-$MAX_STEPS" -exec rm -rf {} +
      fi
    fi

    if [[ -f "$OUT_DIR/${tag}_metrics_per_set.csv" ]]; then
      echo "[$tag] metrics exist, skipping generation"
    else
      echo "[$tag] generating $N_SETS sets"
      ARM_NAME="$tag" MODEL_DIR="$ckpt" TRAIN_CSV="$csv" REP="$rep" \
      OUT_DIR="$OUT_DIR" N_SETS="$N_SETS" GEN_SEED_BASE="$GEN_SEED_BASE" \
        python sweep_generate.py > "$OUT_DIR/$tag.log" 2>&1 \
        || echo "[$tag] GENERATION FAILED -- see $OUT_DIR/$tag.log" >&2
    fi
  done
done

echo "aggregating"
OUT_DIR="$OUT_DIR" python aggregate_sweep.py
