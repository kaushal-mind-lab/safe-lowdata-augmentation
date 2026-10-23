"""
Code to train an atom-level tokenizer for SMILES and then SAFE. 
"""

import pandas as pd
from tokenizers import Tokenizer, Regex, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast


# ---- params ----
encoding="smiles"

if encoding=="smiles":
    TRAIN_FROM = "/data/input/part3_arm_smiles_train.csv"
    COL = "smiles" #--not canonical_smiles
    MODEL_MAX_LEN = 256
    OUT = "/data/tokenizers/tokenizer_smiles"
elif encoding=="safe":
    TRAIN_FROM = "/data/input/part3_arm_safe_train.csv" #--combine all safe files from different arms and parts to be comprehensive + validation file
    COL = "safe"
    MODEL_MAX_LEN = 256
    OUT = "/data/tokenizers/tokenizer_safe"
# ----------------

SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]

# atom-level split: bracket atom | 2-letter atom | ring number | organic atom | bond/branch/dot = 1 token
ATOM_REGEX = r"\[[^\]]+\]|Br|Cl|%\(\d+\)|%\d{2}|[0-9]|[BCNOPSFIbcnops]|\(|\)|\.|=|#|-|\+|/|\\|:|~|@|\*"

corpus = pd.read_csv(TRAIN_FROM)[COL].dropna().astype(str).tolist()

tok = Tokenizer(models.WordLevel(unk_token="<unk>"))     # WordLevel = no merges -> no CCCC chains
tok.pre_tokenizer = pre_tokenizers.Split(Regex(ATOM_REGEX), behavior="isolated")
tok.decoder = decoders.Fuse()                            # concatenate tokens back (exact round-trip)

trainer = trainers.WordLevelTrainer(special_tokens=SPECIALS, min_frequency=1)
tok.train_from_iterator(corpus, trainer=trainer)

tok.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[("<s>", tok.token_to_id("<s>")), ("</s>", tok.token_to_id("</s>"))],
)

hf = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
)
hf.model_max_length = MODEL_MAX_LEN
hf.padding_side = "right"
hf.save_pretrained(OUT)

s = corpus[0]
ids = hf(s)["input_ids"]
print(f"vocab={hf.vocab_size} -> {OUT}")
print(f"example tokens: {hf.convert_ids_to_tokens(ids)[:24]}")
print(f"roundtrip exact: {hf.decode(ids, skip_special_tokens=True) == s}")


VALID_FROM = "/data/input/valid_200.csv"
# ---- unknown-token check ----
vocab = set(hf.get_vocab())

def unk_check(name, path):
    strings = pd.read_csv(path)[COL].dropna().astype(str).tolist()
    bad = {}
    for s in strings:
        for piece, _ in hf.backend_tokenizer.pre_tokenizer.pre_tokenize_str(s):
            if piece not in vocab:
                bad[piece] = bad.get(piece, 0) + 1
    print(f"[{name}] {len(strings)} strings | unknown occurrences: {sum(bad.values())} | distinct: {len(bad)}")
    if bad:
        print("   unknown tokens:", dict(sorted(bad.items(), key=lambda kv: -kv[1])))

unk_check("train", TRAIN_FROM)
if VALID_FROM:
    unk_check("valid", VALID_FROM)


# ---- token-length stats ----
import numpy as np

def length_stats(name, path):
    strings = pd.read_csv(path)[COL].dropna().astype(str).tolist()
    lens = np.array([len(hf(s)["input_ids"]) for s in strings])   # includes <s> and </s>
    vals, counts = np.unique(lens, return_counts=True)
    mode = vals[counts.argmax()]
    print(f"[{name}] tokens/seq  mean={lens.mean():.1f}  median={np.median(lens):.0f}  "
          f"mode={mode}  min={lens.min()}  max={lens.max()}  std={lens.std():.1f}")

length_stats("train", TRAIN_FROM)
if VALID_FROM:
    length_stats("valid", VALID_FROM)