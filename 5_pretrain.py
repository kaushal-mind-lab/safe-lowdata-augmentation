import os
import math
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import yaml
import pandas as pd
import wandb
from accelerate import PartialState
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    OPTConfig,
    OPTForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

cfg = yaml.safe_load(open(os.environ.get("CONFIG", "config_pretrain.yaml")))
set_seed(cfg["SEED"])
is_main = PartialState().is_main_process

# tokenizer
tok = AutoTokenizer.from_pretrained(cfg["TOKENIZER_DIR"])
tok.padding_side = "right"
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "<pad>"})

# data: separate train/valid files, SAFE column -> tokenize (truncate only; collator pads)
def load(path):
    df = pd.read_csv(path)[[cfg["COL"]]].dropna()
    df.columns = ["text"]
    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds.map(
        lambda b: tok(b["text"], truncation=True, max_length=cfg["MAX_LEN"]),
        batched=True,
        remove_columns=["text"],
    )
train_ds = load(cfg["TRAIN_CSV"])
valid_ds = load(cfg["VALID_CSV"])

# model is decoder-only OPT, causal LM
model = OPTForCausalLM(
    OPTConfig(
        vocab_size=len(tok),
        hidden_size=cfg["HIDDEN_SIZE"],
        num_hidden_layers=cfg["NUM_LAYERS"],
        num_attention_heads=cfg["NUM_HEADS"],
        ffn_dim=4 * cfg["HIDDEN_SIZE"],
        max_position_embeddings=cfg["MAX_LEN"],
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
)

if is_main:
    wandb.init(entity=cfg.get("WANDB_ENTITY", None),
               project=cfg["WANDB_PROJECT"],
               name=cfg["RUN_NAME"],
               config=cfg,
               resume="allow",
               id=cfg.get("WANDB_RUN_ID", None),
               )
    print(f"vocab={len(tok)} | params={sum(p.numel() for p in model.parameters()):,} "
          f"| train={len(train_ds)} valid={len(valid_ds)}")

args = TrainingArguments(
    output_dir=cfg["OUTPUT_DIR"],
    run_name=cfg["RUN_NAME"],
    max_steps=cfg["MAX_STEPS"],                 # SAME for both arms -> matched compute
    per_device_train_batch_size=cfg["BATCH_SIZE"],
    per_device_eval_batch_size=cfg["BATCH_SIZE"],
    learning_rate=cfg["LR"],
    weight_decay=cfg["WEIGHT_DECAY"],
    warmup_ratio=cfg["WARMUP_RATIO"],
    eval_strategy="steps",
    eval_steps=cfg["EVAL_STEPS"],
    save_steps=cfg["EVAL_STEPS"],
    save_total_limit=2,
    logging_steps=cfg["LOG_STEPS"],
    bf16=cfg["BF16"],
    report_to="wandb" if is_main else "none",
    seed=cfg["SEED"],
)
if not os.path.exists(cfg["OUTPUT_DIR"]):
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),   # causal LM (not MLM)
)
trainer.train()

if is_main:
    best = os.path.join(cfg["OUTPUT_DIR"], "best")
    trainer.save_model(best)
    tok.save_pretrained(best)
    m = trainer.evaluate()
    print(f"eval_loss={m['eval_loss']:.4f} | ppl={math.exp(m['eval_loss']):.4f} | saved -> {best}")
    wandb.finish()