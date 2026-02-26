# config.py
import os

# Paths
#WORK_DIR = "runs/enwik8_unet"  #commented when running base
DATA_PATH = "data/enwik8"   # raw enwik8 file (100MB) named "enwik8" (no extension)
TOKENIZER_TYPE = "byte_bpe"     # "byte" or "byte_bpe"
TOKENIZED_DATA_PATH = "data/enwik8_bpe_tokens.npy"  # used when TOKENIZER_TYPE="byte_bpe"

# Repro
SEED = 1337

# Device / precision
DEVICE = "cuda"
USE_AMP = False     # mixed precision
USE_COMPILE = True     # torch.compile (PyTorch 2.x); fixed padded shape helps compilation
AMP_DTYPE = "float16"  # "float16" or "bfloat16" (bf16 is more stable on supported GPUs)
NAN_DEBUG = True       # skip/log non-finite train steps
MAX_CONSEC_BAD_STEPS = 32  # abort run if too many consecutive non-finite train steps

# Model
VOCAB_SIZE = 2048        # byte-level
DIM = 512
NUM_HEADS = 8
MLP_RATIO = 4
DROPOUT = 0.1
WINDOW_SIZES = [4, 4, 2, 2, 8]  # variable-length: no divisibility requirement
ROPE_MAX_SEQ_LEN = 2048      # must be >= BLOCK_SIZE (matches your model default)
NUM_CODES = 0           # >0 enables codebook at bottleneck (discrete indices -> embeddings)

# Training
BLOCK_SIZE = 1024

# Variable-length training (single-length-per-batch)
VAR_LEN_ENABLE = True
# Probability of forcing full-length batches at BLOCK_SIZE.
VAR_LEN_FULL_PROB = 0.5
# Otherwise sample seq_len uniformly in [VAR_LEN_MIN, VAR_LEN_MAX].
VAR_LEN_MIN = 2
VAR_LEN_MAX = 1024
# Fixed-shape padded batches: always feed [B, BLOCK_SIZE] and mask padded tokens.
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 2
BATCH_SIZE = 32 #16 < 64    #128 for unet
GRAD_ACCUM = 8    #2 for unet? didnt do this last tme

MAX_STEPS = 200_000
WARMUP_STEPS = 2_000

LR = 1.5e-4 #3e-4 may cause overflow with amp.
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


# Autoencoder run output directory
WORK_DIR = "runs/enwik8_ae"
