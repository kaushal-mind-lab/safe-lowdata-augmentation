"""
Code to create a randomized SAFE dataset for training the tokenizer. The randomization can be done along three axes: fragment order, atom order, or both. The output is a CSV file with the randomized SAFE strings and their corresponding canonical SMILES.
For part 1, use MODE="full" and N=25 and N=50 for part1 training data
For part 2, use MODE="full", MODE="dot", and MODE="atom" with N=50 for part2 training data and subsample later with balanced_subsample.py
For part 3, see the second block after SAFE
"""

#----For SAFE (part 1 and part 2)


import random
import numpy as np
import pandas as pd
from rdkit import RDLogger
import safe
RDLogger.DisableLog("rdApp.*")

INP = "/data/input/part1_arm_a_train_1K.csv"      # input csv with a SMILES column
OUT = "/data/input/part1_arm_b_train_25K.csv"     # output csv - run once with N=25, then again with N=50 for arm c
SMILES_COL = "canonical_smiles"
N = 50                             # randomized SAFE variants per molecule
MODE = "full"                      # "full" = dot+atom | "dot" = fragment-order only | "atom" = atom-order only
SEED = 42                          # base seed (reproducible)


def encode_one(smi, mode, seed):
    # one SAFE string for the requested randomization axis (None on failure)
    try:
        if mode == "dot":                                    # fragment order only; atoms stay canonical
            f = safe.encode(smi, slicer="brics", canonical=True).split(".")
            random.Random(seed).shuffle(f)
            return ".".join(f)
        s = safe.encode(smi, slicer="brics", canonical=False, randomize=True, seed=seed)
        if mode == "atom":                                   # atom order only; freeze fragment order by sorting
            s = ".".join(sorted(s.split(".")))
        return s                                             # mode == "full": dot + atom, as safe returns it
    except Exception:
        return None


def seed_string(smi, mode):
    # canonical SAFE in the same fragment-order convention the mode uses, kept "by design"
    try:
        f = safe.encode(smi, slicer="brics", canonical=True).split(".")
        return ".".join(sorted(f)) if mode == "atom" else ".".join(f)
    except Exception:
        return None


def variants(smi, n, seed0, mode):
    out = set()
    s0 = seed_string(smi, mode)
    if s0:
        out.add(s0)
    for k in range(n * 50):        # oversample; unique variants are capped by the molecule's axis multiplicity
        s = encode_one(smi, mode, seed0 + k)
        if s:
            out.add(s)
        if len(out) >= n:
            break
    return list(out)


df = pd.read_csv(INP)
rows, counts = [], []
for i, smi in enumerate(df[SMILES_COL]):
    v = variants(smi, N, SEED + i * N * 3, MODE)     # disjoint seed block per molecule
    counts.append(len(v))
    rows += [{"safe": s, "mol_id": i, "canonical_smiles": smi} for s in v]

out = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
out.to_csv(OUT, index=False, header=True)

c = np.array(counts)
print(f"[{MODE}] {len(df)} molecules -> {len(out)} rows -> {OUT}")
print(f"unique variants/molecule: min={c.min()} median={int(np.median(c))} max={c.max()} "
      f"mean={c.mean():.1f} | reached N={int((c >= N).sum())}/{len(c)}")   # effective augmentation factor



#----For SMILES (part 3)


INP = "/data/input/part1_arm_a_train_1K.csv"            # input csv with a SMILES column (the 1000 seed molecules)
OUT = "/data/input/train_smiles_N50_unmatched.csv"    # output csv (cols: smiles, mol_id, canonical_smiles)
SMILES_COL = "canonical_smiles"
N = 50                                 # randomized SMILES variants per molecule
SEED = 42


def randomize(smi, seed):
    # restricted randomized SMILES: shuffle atom order, write non-canonical (Arus-Pous style)
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    idx = list(range(m.GetNumAtoms()))
    np.random.RandomState(seed).shuffle(idx)
    return Chem.MolToSmiles(Chem.RenumberAtoms(m, idx), canonical=False)


def variants(smi, n, seed0):
    m = Chem.MolFromSmiles(smi)
    out = {Chem.MolToSmiles(m, canonical=True)} if m else set()   # canonical seed, by design
    for k in range(n * 50):
        s = randomize(smi, seed0 + k)
        if s:
            out.add(s)
        if len(out) >= n:
            break
    return list(out)


df = pd.read_csv(INP)
rows, counts = [], []
for i, smi in enumerate(df[SMILES_COL]):
    v = variants(smi, N, SEED + i * N * 3)     # disjoint seed block per molecule
    counts.append(len(v))
    rows += [{"smiles": s, "mol_id": i, "canonical_smiles": smi} for s in v]

out = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
out.to_csv(OUT, index=False)

c = np.array(counts)
print(f"{len(df)} molecules -> {len(out)} rows -> {OUT}")
print(f"unique variants/molecule: min={c.min()} median={int(np.median(c))} max={c.max()} "
      f"mean={c.mean():.1f} | reached N={int((c >= N).sum())}/{len(c)}")
