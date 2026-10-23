import contextlib
import glob
import os
import sys
import traceback
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import safe
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
RDLogger.DisableLog("rdApp.*")

# ---- params ----
# (name, model_dir, train_csv, rep)   rep: "safe" = decode SAFE to SMILES | "smiles" = model emits SMILES directly
ARMS = [
    ("part1_arm_a",      "./runs/part1_arm_a/checkpoint-20000",      "/data/input/part1_arm_a_train_1K.csv",   "safe"),
    ("part1_arm_b",      "./runs/part1_arm_b/checkpoint-20000",      "/data/input/part1_arm_b_train_25K.csv",  "safe"),
    ("part1_arm_c",      "./runs/part1_arm_c/checkpoint-20000",      "/data/input/part1_arm_c_train_50K.csv",  "safe"),
    ("part2_arm_atom",   "./runs/part2_arm_atom/checkpoint-20000",   "/data/input/part2_arm_atom_train.csv",   "safe"),
    ("part2_arm_dot",    "./runs/part2_arm_dot/checkpoint-20000",    "/data/input/part2_arm_dot_train.csv",    "safe"),
    ("part2_arm_full",   "./runs/part2_arm_full/checkpoint-20000",   "/data/input/part2_arm_full_train.csv",   "safe"),
    ("part3_arm_safe",   "./runs/part3_arm_safe/checkpoint-20000",   "/data/input/part3_arm_safe_train.csv",   "safe"),
    ("part3_arm_smiles", "./runs/part3_arm_smiles/checkpoint-20000", "/data/input/part3_arm_smiles_train.csv", "smiles"),
]
VALID_CSV = "/data/input/valid_200.csv"
SMILES_COL = "canonical_smiles"   # seed-molecule column present in every TRAIN_CSV
OUT_DIR = "./generated"           # every output lands here as <arm>_<file>; logs as <arm>.log
SKIP_DONE = False                 # True skips arms that already have <arm>_metrics_summary.csv in OUT_DIR
NN_THRESH = 0.7                   # nearest-seed Tanimoto >= this = near-duplicate; < this = "genuine" novelty
N_SETS = 5
N_PER_SET = 10_000
BATCH = 500
MAX_LEN = 140
TEMPERATURE = 0.9
TOP_K = 0
TOP_P = 1.0
SEED = 42
# ----------------

device = "cuda" if torch.cuda.is_available() else "cpu"
NAN = float("nan")


def out(name, fname):  # ./generated/part1_arm_a_novel.csv
    return os.path.join(OUT_DIR, f"{name}_{fname}")


@contextlib.contextmanager
def log_to(path):  # send fd 1 and fd 2 (python, C libs, logging) to a file, then restore; == `&> path`
    sys.stdout.flush(); sys.stderr.flush()
    old_out, old_err = os.dup(1), os.dup(2)
    f = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(f, 1); os.dup2(f, 2)
    try:
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(old_out, 1); os.dup2(old_err, 2)
        for fd in (f, old_out, old_err):
            os.close(fd)


def canonicalize(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(m) if m else None


def decode(s, rep):  # generated string -> parseable canonical SMILES (may be multi-component), or None
    if rep == "smiles":
        return canonicalize(s)
    try:
        return canonicalize(safe.decode(s, canonical=True, ignore_errors=True))
    except Exception:
        return None


def largest_fragment(smi):  # the delivered molecule: largest connected component of a decode (salt-strip)
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    return Chem.MolToSmiles(max(frags, key=lambda x: x.GetNumHeavyAtoms())) if frags else None


@torch.no_grad()
def generate(model, tok, n, seed):
    set_seed(seed)
    res = []
    while len(res) < n:
        b = min(BATCH, n - len(res))
        ids = torch.full((b, 1), tok.bos_token_id, dtype=torch.long, device=device)
        gen = model.generate(
            ids, attention_mask=torch.ones_like(ids), max_length=MAX_LEN,
            do_sample=True, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        )
        res += tok.batch_decode(gen, skip_special_tokens=True)
    return res[:n]


def morgan(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def mean_pairwise(fps):
    tot, cnt = 0.0, 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        tot += sum(sims); cnt += len(sims)
    return tot / cnt if cnt else NAN


def diversity(smis):  # internal diversity over molecules and Murcko scaffolds (duplicates included, as in MOSES)
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
    mol_fps = [morgan(m) for m in mols]
    scaf_fps = []
    for m in mols:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        if sc is not None and sc.GetNumAtoms() > 0:      # skip acyclic (empty scaffold)
            scaf_fps.append(morgan(sc))
    return 1 - mean_pairwise(mol_fps), 1 - mean_pairwise(scaf_fps)


def nn_to_seeds(smis, seed_fps):  # smiles -> max Tanimoto to any training seed
    d = {}
    for s in smis:
        m = Chem.MolFromSmiles(s)
        d[s] = max(DataStructs.BulkTanimotoSimilarity(morgan(m), seed_fps)) if m else NAN
    return d


def load_refs(train_csv):  # seed molecules, seeds+validation, seed fingerprints
    ref = pd.read_csv(train_csv)
    assert SMILES_COL in ref.columns, f"{SMILES_COL} not in {train_csv}: {list(ref.columns)}"
    train_mols = set(filter(None, (canonicalize(s) for s in ref[SMILES_COL].dropna())))
    val = pd.read_csv(VALID_CSV)
    tv_mols = train_mols | set(filter(None, (canonicalize(s) for s in val[SMILES_COL].dropna())))
    seed_fps = [morgan(m) for m in (Chem.MolFromSmiles(s) for s in train_mols) if m]
    return train_mols, tv_mols, seed_fps


def metrics(gen_list, rep, train_mols, tv_mols, seed_fps):
    n = len(gen_list)
    parsed = [decode(s, rep) for s in gen_list]                       # may be multi-component
    delivered = [largest_fragment(p) if p else None for p in parsed]  # one molecule per sample, aligned with gen_list
    mols = [m for m in delivered if m]                                # the valid, delivered molecules
    u = set(mols)                                                     # distinct among them
    novel_tr = u - train_mols                                         # novel vs training seeds
    novel = u - tv_mols                                               # novel vs training + validation (the yield set)

    nn = nn_to_seeds(novel, seed_fps)                                 # nearest training seed, per novel molecule
    nn_v = np.array([nn[x] for x in novel])
    has = len(nn_v) > 0
    d_mol, d_scaf = diversity(mols)
    dn_mol, dn_scaf = diversity(sorted(novel))

    m = {
        "n_valid": len(mols),
        "validity": len(mols) / n,
        "uniqueness": len(u) / max(len(mols), 1),
        "novelty": len(novel_tr) / max(len(u), 1),
        "novelty_tv": len(novel) / max(len(u), 1),
        "yield": len(novel),                                              # novel+unique+valid per N_PER_SET
        "genuine_yield": int((nn_v < NN_THRESH).sum()),                   # ... with nearest-seed Tanimoto < NN_THRESH
        "seed_recovery": len(u & train_mols) / max(len(train_mols), 1),   # fraction of seeds regenerated exactly
        #"nn_median": float(np.median(nn_v)) if has else NAN,
        #"near_dup_frac": float((nn_v >= NN_THRESH).mean()) if has else NAN,
        #"nn_ge_0.9_frac": float((nn_v >= 0.9).mean()) if has else NAN,
        #"intdiv_mol": d_mol,
        #"intdiv_scaffold": d_scaf,
        "intdiv_novel_mol": dn_mol,
        "intdiv_novel_scaffold": dn_scaf,
    }
    return m, parsed, delivered, sorted(novel), nn


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)


def run_arm(name, model_dir, train_csv, rep):
    print(f"arm={name} rep={rep}\n  model={model_dir}\n  train={train_csv}\n  device={device}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device).eval()
    train_mols, tv_mols, seed_fps = load_refs(train_csv)
    print(f"  seeds={len(train_mols)} seeds+valid={len(tv_mols)}", flush=True)

    rows = []
    for k in range(N_SETS):
        gen = generate(model, tok, N_PER_SET, SEED + k)
        m, parsed, delivered, novel, nn = metrics(gen, rep, train_mols, tv_mols, seed_fps)
        rows.append(m)
        print(f"set {k + 1}:")
        for key, v in m.items():
            print(f"  {key:24s} {fmt(v)}")
        if k == 0:
            pd.DataFrame({"raw": gen, "decoded": parsed, "smiles": delivered,
                          "multi_component": [bool(p) and "." in p for p in parsed]}
                         ).to_csv(out(name, "gen_set.csv"), index=False)
            pd.DataFrame({"smiles": novel, "nn_tanimoto": [nn[s] for s in novel]}
                         ).to_csv(out(name, "novel.csv"), index=False)
            print(f"  saved {out(name, 'gen_set.csv')} ({len(gen)} rows), "
                  f"{out(name, 'novel.csv')} ({len(novel)} molecules)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out(name, "metrics_per_set.csv"), index=False)   # per-set values, for the statistical tests
    summ = pd.DataFrame({"mean": df.mean(), "std": df.std()})
    summ.to_csv(out(name, "metrics_summary.csv"))
    print(f"\nmean +/- std over {N_SETS} sets:")
    for key in df.columns:
        print(f"  {key:24s} {summ.loc[key, 'mean']:.4f} +/- {summ.loc[key, 'std']:.4f}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_combined():  # one tidy table of every arm's per-set metrics, for the statistical tests
    frames = []
    for p in sorted(glob.glob(os.path.join(OUT_DIR, "*_metrics_per_set.csv"))):
        arm = os.path.basename(p)[:-len("_metrics_per_set.csv")]
        d = pd.read_csv(p)
        d.insert(0, "arm", arm)
        d.insert(1, "set", range(1, len(d) + 1))
        frames.append(d)
    if frames:
        path = os.path.join(OUT_DIR, "all_arms_metrics.csv")
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        print(f"combined {len(frames)} arms -> {path}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, model_dir, train_csv, rep in ARMS:
        if SKIP_DONE and os.path.exists(out(name, "metrics_summary.csv")):
            print(f"[{name}] already done, skipping", flush=True)
            continue
        log = os.path.join(OUT_DIR, f"{name}.log")
        print(f"[{name}] running -> {log}", flush=True)
        with log_to(log):
            try:
                run_arm(name, model_dir, train_csv, rep)
            except Exception:
                traceback.print_exc()          # lands in the arm's log; move on to the next arm
        print(f"[{name}] finished (check log for errors)", flush=True)
    write_combined()