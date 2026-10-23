"""
Purpose: create a balanced subsample of the three datasets (full, dot, atom) such that each molecule has the same number of rows in each dataset. The number of rows per molecule is capped at the minimum number of rows for that molecule across the three datasets. Then, do the same for SMILES vs SAFE i.e., part 3. Comment FILES dict accordingly. See equations 1 and 2 in paper.
"""

import numpy as np, pandas as pd

FILES = {"full": "/data/temporary/input_train_49K_full.csv",
         "dot":  "/data/temporary/train_38K_dot.csv",
         "atom": "/data/temporary/train_37K_atom.csv"}
# FILES = {"full": "./data/train_49K_full.csv",
#          "smiles":  "./data/train_smiles_N50.csv",}
SEED = 42

dfs = {m: pd.read_csv(p) for m, p in FILES.items()}
gs  = {m: {i: g for i, g in df.groupby("mol_id")} for m, df in dfs.items()}
ids = sorted(set.intersection(*[set(g) for g in gs.values()]))     # molecules present in all three

k = {i: min(len(gs[m][i]) for m in FILES) for i in ids}            # per-mol cap = smallest axis
tot = sum(k.values())
kk = np.array(list(k.values()))
print(f"matched total = {tot} rows over {len(ids)} molecules | per-mol min={kk.min()} median={int(np.median(kk))} max={kk.max()}")

for m in FILES:
    out = pd.concat([gs[m][i].sample(n=k[i], random_state=SEED) for i in ids]) \
            .sample(frac=1, random_state=SEED).reset_index(drop=True)
    out.to_csv(f"/data/input/part3_arm_{m}_train.csv", index=False)
    print(f"{m}: {len(out)} rows")