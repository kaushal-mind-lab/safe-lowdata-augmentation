"""Aggregate the sweep into the three tables the paper needs.

  sweep_all_sets.csv    one row per (arm, train_seed, gen_set)   -- raw
  sweep_run_level.csv   one row per (arm, train_seed)            -- FEED THIS TO THE STATS
  sweep_table.csv       mean +/- SD over train seeds             -- the manuscript tables
  sweep_variance.csv    between-run vs within-run SD             -- the appendix sentence

The unit of replication is the TRAINING RUN. Generation sets are averaged within
a run first; the SD in sweep_table.csv is over train seeds, never over all sets.
"""
import glob
import os
import re

import pandas as pd

OUT_DIR = os.environ.get("OUT_DIR", "./generated_sweep")
KEYS = ["arm", "train_seed", "gen_set"]

frames = []
for p in sorted(glob.glob(os.path.join(OUT_DIR, "*_metrics_per_set.csv"))):
    base = os.path.basename(p)[: -len("_metrics_per_set.csv")]
    m = re.match(r"^(.*)_seed(\d+)$", base)
    if not m:
        print(f"skipping unrecognised {base}")
        continue
    d = pd.read_csv(p)
    d.insert(0, "arm", m.group(1))
    d.insert(1, "train_seed", int(m.group(2)))
    d.insert(2, "gen_set", range(1, len(d) + 1))
    frames.append(d)

if not frames:
    raise SystemExit(f"no *_metrics_per_set.csv in {OUT_DIR}")

sets = pd.concat(frames, ignore_index=True)
METRICS = [c for c in sets.columns if c not in KEYS]

run = sets.groupby(["arm", "train_seed"], as_index=False)[METRICS].mean()
table = run.groupby("arm")[METRICS].agg(["mean", "std"])
between = run.groupby("arm")[METRICS].std()
within = sets.groupby(["arm", "train_seed"])[METRICS].std().groupby("arm").mean()

sets.to_csv(os.path.join(OUT_DIR, "sweep_all_sets.csv"), index=False)
run.to_csv(os.path.join(OUT_DIR, "sweep_run_level.csv"), index=False)
table.to_csv(os.path.join(OUT_DIR, "sweep_table.csv"))
pd.concat({"between_run": between, "within_run": within}, axis=1).to_csv(
    os.path.join(OUT_DIR, "sweep_variance.csv"))

n = run.groupby("arm").size()
print(f"arms={len(n)}  train seeds per arm: {n.min()}-{n.max()}  "
      f"gen sets total={len(sets)}\n")
show = ["validity", "uniqueness", "novelty", "yield", "genuine_yield", "seed_recovery"]
show = [c for c in show if c in METRICS]
print(table[show].round(4).to_string())
print(f"\n-> {OUT_DIR}/sweep_run_level.csv is the input for the Welch/Games-Howell tests")
