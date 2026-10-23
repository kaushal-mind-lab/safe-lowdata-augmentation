"""
utils.py  (trimmed + fixed for decoder-only causal LM)
======================================================
Kept, verified routines from the FragBERTa utils; dropped MLM/prefix-specific
ones. Changes from the original:
  * build_tokenizer      -> loads any saved fast tokenizer (AutoTokenizer),
                            not RobertaTokenizerFast specifically.
  * tokenize_function    -> truncate only, NO max_length padding
                            (dynamic padding is done by the data collator).
  * convert_smiles_to_safe -> BRICS, NO prefix (prefix-free, per spec).
  * removed  brics_or_random_slicer  (prepended BRICS./MMPA. prefixes).
  * get_total_steps      -> number of processes (accelerate) instead of manual GPUs.
"""
import math
import os

import yaml
from rdkit import Chem
from transformers import AutoTokenizer

import safe


def time_elapsed(t1, t2):
    """Return elapsed time between t1 and t2 as a human string."""
    elapsed = abs(t2 - t1)
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    return "time elapsed: " + ", ".join(parts)


def get_total_steps(num_procs, n_samples, batch_size, grad_accum, epochs):
    """Estimate total optimizer steps (for logging only)."""
    steps_per_epoch = math.ceil(n_samples / (batch_size * max(num_procs, 1) * grad_accum))
    return steps_per_epoch * epochs


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config YAML not found at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def convert_smiles_to_safe(smiles, slicer="brics"):
    """SMILES -> SAFE (BRICS, prefix-free). Returns None on failure."""
    try:
        return safe.encode(smiles, slicer=slicer, canonical=True, ignore_stereo=True)
    except Exception:
        return None


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def build_tokenizer(tokenizer_dir):
    """Load the saved fast tokenizer; ensure a pad token exists; right-pad for training."""
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    tok.padding_side = "right"
    return tok


def tokenize_function(examples, tokenizer, max_len, col_name):
    """Tokenize a batch; truncate only. Dynamic padding is handled by the collator."""
    return tokenizer(
        examples[col_name],
        truncation=True,
        max_length=max_len,
    )


def save_updated_config(original_cfg, output_dir, total_runtime, perplexity,
                        total_params, trainable_params, train_steps):
    new_cfg = dict(original_cfg)
    new_cfg["TOTAL_RUNTIME"] = str(total_runtime)
    new_cfg["FINAL_EVAL_PERPLEXITY"] = float(perplexity)
    new_cfg["SAVED_AT"] = new_cfg.get("OUTPUT_DIR_BEST", output_dir)
    new_cfg["TOTAL_PARAMS"] = int(total_params)
    new_cfg["TRAINABLE_PARAMS"] = int(trainable_params)
    new_cfg["TRAINING_STEPS"] = int(train_steps)
    save_path = os.path.join(output_dir, "config_final.yaml")
    with open(save_path, "w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)
    print(f"[config] saved updated config -> {save_path}")
