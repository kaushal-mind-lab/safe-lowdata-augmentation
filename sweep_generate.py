"""Generate + score ONE (arm, train-seed) run by reusing 5_generate.py.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "6_generate.py")

spec = importlib.util.spec_from_file_location("gen_impl", SRC)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)          # its work is under __main__, so nothing runs on import

gen.OUT_DIR   = os.environ.get("OUT_DIR", "./generated_sweep")
gen.N_SETS    = int(os.environ.get("N_SETS", 3))
gen.SEED      = int(os.environ.get("GEN_SEED_BASE", 42))   # same gen seeds for every train seed
gen.N_PER_SET = int(os.environ.get("N_PER_SET", 10_000))

os.makedirs(gen.OUT_DIR, exist_ok=True)
gen.run_arm(
    os.environ["ARM_NAME"],
    os.environ["MODEL_DIR"],
    os.environ["TRAIN_CSV"],
    os.environ.get("REP", "safe"),
)
