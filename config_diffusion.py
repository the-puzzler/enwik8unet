# config_diffusion.py
import os

# Paths
DATA_PATH = "data/enwik8"  # raw enwik8 file (100MB) named "enwik8" (no extension)
WORK_DIR = "runs/enwik8_diffusion"

# Repro
SEED = 1337

# Device / precision
DEVICE = "cuda"
USE_AMP = True            # mixed precision
USE_COMPILE = True        # torch.compile (PyTorch 2.x)

# Model
VOCAB_SIZE = 256          # byte-level
INPUT_DIM = 256           # learned byte embedding dim (x0 / xt space)
DIM = 512                 # transformer width (internal)
OUT_DIM = INPUT_DIM       # predicts x0 in same space
NUM_HEADS = 8
MLP_RATIO = 4.0
DROPOUT = 0.1
WINDOW_SIZES = [4, 4, 2, 2]  # block_size must be divisible by product(WINDOW_SIZES)
ROPE_MAX_SEQ_LEN = 2048      # must be >= BLOCK_SIZE (matches model default in unet_transformer)

# Training
BLOCK_SIZE = 1024
BATCH_SIZE = 32
GRAD_ACCUM = 8

MAX_STEPS = 100_000
WARMUP_STEPS = 2_000

LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
CLIP_GRAD_NORM = 1.0

# Diffusion / loss
SIGREG_WEIGHT = 0.1
SIGREG_KNOTS = 17
DECODE_TAU = 0.1  # temperature for softmax(cos(pred, emb)/tau) when reporting CE/bpb

# Logging / eval / checkpointing
LOG_INTERVAL = 50
EVAL_INTERVAL = 1000
EVAL_ITERS = 100
CKPT_INTERVAL = 2000
CKPT_SNAPSHOT_INTERVAL = 10000  # 0 to disable
