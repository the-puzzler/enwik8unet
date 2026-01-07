# config.py
import os

# Paths
#WORK_DIR = "runs/enwik8_unet"  #commented when running base
DATA_PATH = "data/enwik8"   # raw enwik8 file (100MB) named "enwik8" (no extension)

# Repro
SEED = 1337

# Device / precision
DEVICE = "cuda"
USE_AMP = True            # mixed precision
USE_COMPILE = True        # torch.compile (PyTorch 2.x)

# Model
VOCAB_SIZE = 256          # byte-level
DIM = 512
NUM_HEADS = 8
MLP_RATIO = 4.0
DROPOUT = 0.1
WINDOW_SIZES = [4, 4, 2, 2]  # block_size must be divisible by product(WINDOW_SIZES)
ROPE_MAX_SEQ_LEN = 2048      # must be >= BLOCK_SIZE (matches your model default)

# Training
BLOCK_SIZE = 1024
BATCH_SIZE = 32 #16 < 64    #128 for unet
GRAD_ACCUM = 8    #2 for unet

MAX_STEPS = 100_000
WARMUP_STEPS = 2_000

LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
CLIP_GRAD_NORM = 1.0

# Logging / eval / checkpointing
LOG_INTERVAL = 50
EVAL_INTERVAL = 1000
EVAL_ITERS = 100
CKPT_INTERVAL = 2000           # save latest every N steps
CKPT_SNAPSHOT_INTERVAL = 10000 # also save runs/enwik8_unet/ckpt_step_XXXXXX.pt (0 to disable)


#For baseline
WORK_DIR = "runs/enwik8_baseline"
NUM_LAYERS = 10