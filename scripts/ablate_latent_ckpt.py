import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import config_ae as C
from unet_autoencoder import UNetAutoEncoder


LN2 = math.log(2.0)

# Edit these and just run: `python3 ablate_latent_ckpt.py`
# Defaults match `config_ae.py`.
DATA_PATH = C.DATA_PATH
CKPT_PATH = os.path.join(C.WORK_DIR, "ckpt_latest.pt")
SPLIT = "val"  # "train" | "val" | "test"
ITERS = C.EVAL_ITERS

BATCH_SIZE = C.BATCH_SIZE
BLOCK_SIZE = C.BLOCK_SIZE

# Ablations
COMBOS = [
    ("baseline", "none", "none", "final"),
    ("bottleneck_zero", "zero", "none", "final"),
    ("bottleneck_random", "random", "none", "final"),
    ("kept_shuffle_final", "none", "shuffle", "final"),
    ("kept_shuffle_all", "none", "shuffle", "all"),
]
# Each combo is: (name, BOTTLENECK, KEPT, KEPT_LEVELS)

device = torch.device("cpu")

if not os.path.isfile(CKPT_PATH):
    raise FileNotFoundError(f"Missing checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
cfg = ckpt.get("config", {}) or {}

vocab_size = int(cfg.get("VOCAB_SIZE", 256))
dim = int(cfg.get("DIM", 512))
num_heads = int(cfg.get("NUM_HEADS", 8))
mlp_ratio = float(cfg.get("MLP_RATIO", 4))
dropout = float(cfg.get("DROPOUT", 0.1))
window_sizes = list(cfg.get("WINDOW_SIZES", C.WINDOW_SIZES))

block_size = int(BLOCK_SIZE)
batch_size = int(BATCH_SIZE)

ws_prod = 1
for w in window_sizes:
    ws_prod *= int(w)
if block_size % ws_prod != 0:
    raise ValueError(f"BLOCK_SIZE={block_size} must be divisible by product(WINDOW_SIZES)={ws_prod}.")

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"Missing {DATA_PATH}.\n"
        "Download the raw enwik8 file (100MB) and place it there (typically named 'enwik8')."
    )
data = np.memmap(DATA_PATH, dtype=np.uint8, mode="r")
n = len(data)
n_train = int(n * 0.9)
n_val = int(n * 0.05)
train_arr = data[:n_train]
val_arr = data[n_train : n_train + n_val]
test_arr = data[n_train + n_val :]
split_arr = {"train": train_arr, "val": val_arr, "test": test_arr}[SPLIT]

model = UNetAutoEncoder(
    vocab_size=vocab_size,
    dim=dim,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    dropout=dropout,
    window_sizes=window_sizes,
).to(device)
state = ckpt.get("model", ckpt)

if not isinstance(state, dict):
    raise TypeError(f"Expected checkpoint state_dict dict, got: {type(state)}")

if any(k.startswith("_orig_mod.") for k in state.keys()):
    state = {k[len("_orig_mod.") :]: v for k, v in state.items()}
if any(k.startswith("module.") for k in state.keys()):
    state = {k[len("module.") :]: v for k, v in state.items()}

try:
    model.load_state_dict(state, strict=True)
except RuntimeError as e:
    keys = list(state.keys())
    head = ", ".join(keys[:10])
    raise RuntimeError(
        "Checkpoint doesn't match UNetAutoEncoder weights. "
        "Double-check CKPT_PATH points at the AE run (not baseline), and whether the checkpoint was saved from a different model.\n"
        f"CKPT_PATH={CKPT_PATH}\n"
        f"first_keys={head}"
    ) from e
model.eval()

results = {}
for name, bottleneck_mode, kept_mode, kept_levels in COMBOS:
    results[name] = {
        "bottleneck": bottleneck_mode,
        "kept": kept_mode,
        "kept_levels": kept_levels,
        "loss_sum": 0.0,
        "acc_sum": 0.0,
        "n": 0,
    }

with torch.no_grad():
    for _ in range(int(ITERS)):
        max_start = len(split_arr) - block_size
        starts = np.random.randint(0, max_start, size=(batch_size,))
        x_tokens = np.stack([split_arr[s : s + block_size].astype(np.int64) for s in starts], axis=0)
        x_tokens = torch.from_numpy(x_tokens).to(device)

        for name, bottleneck_mode, kept_mode, kept_levels in COMBOS:
            x = model.token_emb(x_tokens)
            x = model.dropout(x)

            n_levels = len(model.downsample_layers)
            for level_idx, (encoder_block, downsample) in enumerate(zip(model.encoder_blocks, model.downsample_layers)):
                x = encoder_block(x, mask=None)

                do_kept = kept_mode != "none" and (kept_levels == "all" or level_idx == n_levels - 1)
                if do_kept:
                    window_size = int(downsample.window_size)
                    if x.size(1) % window_size != 0:
                        raise ValueError(f"seq_len={x.size(1)} must be divisible by window_size={window_size}")

                    kept_idx = torch.arange(0, x.size(1), window_size, device=x.device)
                    kept = x.index_select(1, kept_idx)  # [B, S/window, D]
                    if kept_mode == "shuffle":
                        perm = torch.randperm(kept.size(1), device=x.device)
                        kept = kept.index_select(1, perm)
                    elif kept_mode == "random":
                        kept = torch.randn_like(kept)
                    elif kept_mode == "zero":
                        kept = torch.zeros_like(kept)
                    else:
                        raise ValueError(f"unknown kept mode: {kept_mode}")

                    x = x.clone()
                    x.index_copy_(1, kept_idx, kept)

                x = downsample(x)

            if bottleneck_mode == "zero":
                x = torch.zeros_like(x)
            elif bottleneck_mode == "random":
                x = torch.randn_like(x)
            elif bottleneck_mode == "none":
                pass
            else:
                raise ValueError(f"unknown bottleneck mode: {bottleneck_mode}")

            x = model.bottleneck(x, mask=None)
            for upsample, decoder_block in zip(model.upsample_layers, model.decoder_blocks):
                x = upsample(x)
                x = decoder_block(x, mask=None)
            x = model.norm(x)
            logits = model.head(x)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x_tokens.view(-1))
            pred = logits.argmax(dim=-1)
            acc = (pred == x_tokens).float().mean()

            results[name]["loss_sum"] += float(loss.item())
            results[name]["acc_sum"] += float(acc.item())
            results[name]["n"] += 1

print(f"split={SPLIT} iters={ITERS} batch={batch_size} block={block_size}")
for name, _, _, _ in COMBOS:
    r = results[name]
    loss_nats = r["loss_sum"] / max(1, r["n"])
    bpb = loss_nats / LN2
    acc = r["acc_sum"] / max(1, r["n"])
    print("----")
    print(f"name={name} bottleneck={r['bottleneck']} kept={r['kept']} kept_levels={r['kept_levels']}")
    print(f"loss_nats={loss_nats:.6f} bpb={bpb:.6f} acc={acc:.6f}")
