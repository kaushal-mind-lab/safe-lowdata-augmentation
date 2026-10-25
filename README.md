# What to randomize matters: a controlled study of volume, axis and representation in low-data molecular generation
Code and data for a controlled study of randomized SAFE augmentation for a small decoder-only OPT chemical language model trained from 1,000 seed molecules (GuacaMol). Three comparisons, all with the same architecture, objective, optimizer and number of steps (20,000) with only the training corpus (and, in Family C, its tokenizer) changes.

| Family | Question | Arms (training strings) |
|---|---|---|
| A | Does augmentation volume help? | safe-1k (1,000), safe-25k (25,000), safe-50k (50,000) |
| B | Which randomization axis matters? | atom, dot (fragment order), full: 32,152 each |
| C | SAFE or SMILES randomization? | smiles-rand, safe-rand: 49,209 each |

Each arm is trained from 5 seeds and each model generates 3 sets of 10,000 molecules. Metrics: validity, uniqueness, novelty, yield, genuine yield, seed recovery, internal diversity. Statistics use training runs as the unit of replication.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Pipeline

Run in numeric order.

| Step | Script | Purpose |
|---|---|---|
| 1 | `1_get_safe_samples.ipynb` | Select seed and validation molecules, encode as SAFE |
| 2 | `2_augment_data.py` | Randomize SAFE/SMILES (atom, dot, full) |
| 3 | `3_balanced_subsample.py` | Match corpus sizes across arms |
| 4 | `4_train_tokenizer.py` | Train SAFE and SMILES tokenizers |
| 5 | `5_pretrain.py` | Train OPT (`CONFIG=<yaml> python 5_pretrain.py`) |
| 6 | `6_generate.py` | Sample, decode and score molecules |
| 7 | `aggregate_sweep.py` | Run-level and arm-level tables |
| 8 | `8_statistical_tests.py` | Welch ANOVA, Games-Howell, Welch t, Levene |

`run_parallel.sh` runs steps 5-7 for all arms, one GPU per training seed.

## Data

- `data/input/`: training corpora (`part1_*`, `part2_*`, `part3_*`) and `valid_200.csv`
- `data/tokenizers/`: SAFE and SMILES tokenizers
- `data/results/`: per-set and run-level metrics, statistical outputs per family
