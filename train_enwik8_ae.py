import os
import json
import math
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_autoencoder import UNetAutoEncoder
import config_ae as C

LN2 = math.log(2.0)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def human_num(n: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:,.2f}{unit}"
        n /= 1000
    return f"{n:,.2f}P"


def save_checkpoint(path, payload: dict):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)



def load_token_stream() -> np.ndarray:
    tok_type = str(getattr(C, "TOKENIZER_TYPE", "byte")).lower()
    if tok_type == "byte_bpe":
        path = str(getattr(C, "TOKENIZED_DATA_PATH", ""))
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing tokenized stream at {path}.\n"
                "Build it first with scripts/byte_bpe.py encode, or switch TOKENIZER_TYPE to 'byte'."
            )
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D token stream in {path}, got shape={arr.shape}")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"Expected integer dtype in {path}, got {arr.dtype}")
        return arr

    data_path = str(getattr(C, "DATA_PATH", ""))
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Missing {data_path}.\n"
            "Download the raw enwik8 file (100MB) and place it there (typically named 'enwik8')."
        )
    return np.memmap(data_path, dtype=np.uint8, mode="r")


def make_splits(data: np.memmap, train_frac=0.9, val_frac=0.05):
    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]
    return train, val, test


class TextLogger:
    def __init__(self, work_dir: str):
        ensure_dir(work_dir)
        self.log_path = os.path.join(work_dir, "train.log")
        self.csv_path = os.path.join(work_dir, "metrics.csv")

        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write("time,step,split,loss_nats,bpb,lr,tok_per_sec\n")

    def log_line(self, msg: str):
        line = f"[{now_str()}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_metric(self, step: int, split: str, loss_nats: float, bpb: float, lr: float, tok_per_sec: float):
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{now_str()},{step},{split},{loss_nats:.6f},{bpb:.6f},{lr:.8e},{tok_per_sec:.2f}\n")


def sample_batch(data_arr: np.ndarray, batch_size: int, seq_len: int, device: torch.device):
    max_start = len(data_arr) - seq_len
    if max_start <= 0:
        raise ValueError(f"seq_len={seq_len} too long for data length={len(data_arr)}")
    starts = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([data_arr[s:s+seq_len].astype(np.int64) for s in starts], axis=0)
    x = torch.from_numpy(x).to(device, non_blocking=True)
    return x


def sample_batch_padded(
    data_arr: np.ndarray,
    batch_size: int,
    content_len: int,
    block_size: int,
    pad_token_id: int,
    eos_token_id: int,
    device: torch.device,
):
    # Build fixed-size sequences as: [content ...][EOS][PAD ...]
    if content_len <= 0:
        raise ValueError("content_len must be > 0")
    if content_len >= block_size:
        raise ValueError(f"content_len={content_len} must be < block_size={block_size} (need EOS slot)")

    x = sample_batch(data_arr, batch_size, content_len, device)
    x_pad = torch.full((batch_size, block_size), int(pad_token_id), dtype=x.dtype, device=device)
    x_pad[:, :content_len] = x
    x_pad[:, content_len] = int(eos_token_id)

    # Valid for loss/attention: content tokens + EOS, but not PAD tail.
    valid_mask = torch.zeros((batch_size, block_size), dtype=torch.bool, device=device)
    valid_mask[:, : content_len + 1] = True
    return x_pad, valid_mask


def make_key_padding_attn_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    # valid_mask: [B, S] -> mask [B, 1, 1, S], where True means "key is visible".
    return valid_mask[:, None, None, :]


def choose_seq_len() -> int:
    # Returns content length; EOS is appended separately in the batch builder.
    if not getattr(C, "VAR_LEN_ENABLE", False):
        return max(1, int(C.BLOCK_SIZE) - 1)

    block_size = int(C.BLOCK_SIZE)
    rope_max = int(C.ROPE_MAX_SEQ_LEN)
    min_len = int(getattr(C, "VAR_LEN_MIN", 64))
    max_len = int(getattr(C, "VAR_LEN_MAX", block_size))
    min_len = max(1, min_len)
    max_len = min(max_len, rope_max, block_size - 1)
    if min_len > max_len:
        raise ValueError(
            f"Invalid variable-length range: VAR_LEN_MIN={min_len} > VAR_LEN_MAX={max_len} "
            f"(after ROPE_MAX_SEQ_LEN={rope_max} clamp)."
        )
    full_prob = float(getattr(C, "VAR_LEN_FULL_PROB", 0.5))

    r = random.random()
    if r < full_prob:
        return max_len
    return random.randint(min_len, max_len)


def get_amp_dtype():
    name = str(getattr(C, "AMP_DTYPE", "float16")).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16

@torch.no_grad()
def estimate_loss_and_bpb(model, data_arr, block_size, batch_size, eval_iters, device, mask, use_amp: bool):
    model.eval()
    losses = []
    amp_dtype = get_amp_dtype()
    for _ in range(eval_iters):
        seq_len = choose_seq_len() if getattr(C, "VAR_LEN_ENABLE", False) else int(block_size)
        x, valid_mask = sample_batch_padded(
            data_arr,
            batch_size,
            seq_len,
            int(block_size),
            int(getattr(C, "PAD_TOKEN_ID", 0)),
            int(getattr(C, "EOS_TOKEN_ID", 2)),
            device,
        )
        attn_mask = make_key_padding_attn_mask(valid_mask)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(x, mask=attn_mask)
            per_tok = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                x.view(-1),
                reduction="none",
            ).view_as(x)
            denom = valid_mask.sum().clamp_min(1)
            loss = (per_tok * valid_mask).sum() / denom
        losses.append(loss.item())
    mean_loss = float(np.mean(losses))     # nats
    mean_bpb = mean_loss / LN2             # bits per byte (base-2)
    model.train()
    return mean_loss, mean_bpb

def get_lr(step: int):
    if step < C.WARMUP_STEPS:
        return C.LR * step / max(1, C.WARMUP_STEPS)
    # cosine decay to MIN_LR
    t = (step - C.WARMUP_STEPS) / max(1, C.MAX_STEPS - C.WARMUP_STEPS)
    t = min(max(t, 0.0), 1.0)
    return C.MIN_LR + 0.5 * (C.LR - C.MIN_LR) * (1.0 + math.cos(math.pi * t))

# -----------------------------
# Main
# -----------------------------

ensure_dir(C.WORK_DIR)
set_seed(C.SEED)

if C.DEVICE != "cuda" or not torch.cuda.is_available():
    raise RuntimeError("CUDA not available (or config DEVICE != 'cuda').")

device = torch.device("cuda")

if C.BLOCK_SIZE > C.ROPE_MAX_SEQ_LEN:
    raise ValueError(
        f"BLOCK_SIZE={C.BLOCK_SIZE} exceeds ROPE_MAX_SEQ_LEN={C.ROPE_MAX_SEQ_LEN}.\n"
        "Either lower BLOCK_SIZE or increase RoPE max_seq_len in your model."
    )

logger = TextLogger(C.WORK_DIR)

pad_token_id = int(getattr(C, "PAD_TOKEN_ID", 0))
eos_token_id = int(getattr(C, "EOS_TOKEN_ID", 2))
if not (0 <= pad_token_id < int(C.VOCAB_SIZE)):
    raise ValueError(f"PAD_TOKEN_ID={pad_token_id} must be in [0, {int(C.VOCAB_SIZE)-1}]")
if not (0 <= eos_token_id < int(C.VOCAB_SIZE)):
    raise ValueError(f"EOS_TOKEN_ID={eos_token_id} must be in [0, {int(C.VOCAB_SIZE)-1}]")
if pad_token_id == eos_token_id:
    raise ValueError("PAD_TOKEN_ID and EOS_TOKEN_ID must be different.")

# Save config snapshot
cfg_dump = {k: getattr(C, k) for k in dir(C) if k.isupper()}
with open(os.path.join(C.WORK_DIR, "config_snapshot.json"), "w", encoding="utf-8") as f:
    json.dump(cfg_dump, f, indent=2)

# Data
data = load_token_stream()
train_arr, val_arr, test_arr = make_splits(data, train_frac=0.9, val_frac=0.05)
logger.log_line(
    f"Loaded token stream ({getattr(C, 'TOKENIZER_TYPE', 'byte')}, dtype={data.dtype}): "
    f"{len(data):,} | train {len(train_arr):,} | val {len(val_arr):,} | test {len(test_arr):,}"
)

# Model
raw_model = UNetAutoEncoder(
    vocab_size=C.VOCAB_SIZE,
    dim=C.DIM,
    num_heads=C.NUM_HEADS,
    mlp_ratio=C.MLP_RATIO,
    dropout=C.DROPOUT,
    window_sizes=C.WINDOW_SIZES,
    num_codes=getattr(C, "NUM_CODES", 0),
).to(device)



n_params = sum(p.numel() for p in raw_model.parameters())
logger.log_line(f"Model params: {n_params:,}")

# Optionally compile for faster training. Keep a handle to the uncompiled module
# so checkpoint state_dict keys remain stable across compiled/uncompiled runs.
model = torch.compile(raw_model) if C.USE_COMPILE else raw_model

# Ckpt paths
ckpt_latest = os.path.join(C.WORK_DIR, "ckpt_latest.pt")
ckpt_best = os.path.join(C.WORK_DIR, "ckpt_best.pt")

# Resume if latest exists
step = 0
best_val_bpb = float("inf")
ckpt = None
if os.path.isfile(ckpt_latest):
    ckpt = load_checkpoint(ckpt_latest, map_location="cpu")
    step = int(ckpt.get("step", 0))
    best_val_bpb = float(ckpt.get("best_val_bpb", best_val_bpb))
    logger.log_line(f"Resumed from {ckpt_latest} at step={step} best_val_bpb={best_val_bpb:.4f}")
else:
    logger.log_line("No ckpt_latest found; starting fresh.")

if ckpt is not None and "model" in ckpt:
    sd = ckpt["model"]
    # Backward compatibility: checkpoints saved from torch.compile may prefix keys with "_orig_mod."
    if isinstance(sd, dict) and any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}
    # Backward compatibility: DataParallel prefixes keys with "module."
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    raw_model.load_state_dict(sd)

# Optim
optimizer = torch.optim.AdamW(
    raw_model.parameters(),
    lr=C.LR,
    betas=C.BETAS,
    weight_decay=C.WEIGHT_DECAY,
)
scaler = torch.amp.GradScaler("cuda", enabled=C.USE_AMP) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=C.USE_AMP)

if ckpt is not None:
    optimizer.load_state_dict(ckpt["optimizer"])
    if C.USE_AMP and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])


model.train()
t0 = time.time()
last_log_time = t0
tokens_seen = 0
tokens_seen_at_last_log = 0
amp_dtype = get_amp_dtype()
nan_debug = bool(getattr(C, "NAN_DEBUG", True))
max_consec_bad = int(getattr(C, "MAX_CONSEC_BAD_STEPS", 32))
consec_bad_steps = 0

logger.log_line("Starting training loop.")
logger.log_line(f"AMP enabled={bool(C.USE_AMP)} dtype={amp_dtype} compile={bool(C.USE_COMPILE)}")

while step < C.MAX_STEPS:
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    bad_step = False
    bad_reason = ""
    bad_seq_len = None
    for micro in range(C.GRAD_ACCUM):
        seq_len = choose_seq_len()
        x, valid_mask = sample_batch_padded(
            train_arr,
            C.BATCH_SIZE,
            seq_len,
            int(C.BLOCK_SIZE),
            pad_token_id,
            eos_token_id,
            device,
        )
        attn_mask = make_key_padding_attn_mask(valid_mask)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=C.USE_AMP):
            logits = model(x, mask=attn_mask)
            per_tok = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                x.view(-1),
                reduction="none",
            ).view_as(x)
            denom = valid_mask.sum().clamp_min(1)
            loss = (per_tok * valid_mask).sum() / denom
            loss = loss / C.GRAD_ACCUM

        if not torch.isfinite(loss):
            bad_step = True
            bad_seq_len = int(seq_len)
            if nan_debug:
                finite_frac = float(torch.isfinite(logits).float().mean().item())
                logit_absmax = float(logits.detach().abs().max().item())
                bad_reason = (
                    f"non-finite loss micro={micro} seq_len={seq_len} "
                    f"logits_finite_frac={finite_frac:.6f} logits_absmax={logit_absmax:.3e}"
                )
            else:
                bad_reason = f"non-finite loss micro={micro} seq_len={seq_len}"
            break

        scaler.scale(loss).backward()
        accum_loss += loss.item()
        tokens_seen += int(valid_mask.sum().item())

    if bad_step:
        optimizer.zero_grad(set_to_none=True)
        logger.log_line(f"[nan] step {step} skipped: {bad_reason}")
        consec_bad_steps += 1
        if consec_bad_steps >= max_consec_bad:
            raise RuntimeError(
                f"Aborting after {consec_bad_steps} consecutive non-finite steps at step={step}. "
                "Resume from last clean checkpoint after adjusting stability settings."
            )
        continue

    if C.CLIP_GRAD_NORM and C.CLIP_GRAD_NORM > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.CLIP_GRAD_NORM)

    grads_finite = True
    for p in raw_model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            grads_finite = False
            break
    if not grads_finite:
        optimizer.zero_grad(set_to_none=True)
        logger.log_line(f"[nan] step {step} skipped: non-finite gradient(s)")
        consec_bad_steps += 1
        if consec_bad_steps >= max_consec_bad:
            raise RuntimeError(
                f"Aborting after {consec_bad_steps} consecutive non-finite steps at step={step}. "
                "Resume from last clean checkpoint after adjusting stability settings."
            )
        continue

    consec_bad_steps = 0
    scaler.step(optimizer)
    scaler.update()

    # Logging
    if step % C.LOG_INTERVAL == 0:
        now = time.time()
        dt = now - last_log_time
        tok_per_sec = (tokens_seen - tokens_seen_at_last_log) / max(1e-9, dt)

        train_loss_nats = accum_loss
        train_bpb = train_loss_nats / LN2

        logger.log_line(
            f"step {step} | lr {lr:.2e} | loss(nats) {train_loss_nats:.4f} | bpb {train_bpb:.4f} | tok/s {human_num(tok_per_sec)}"
        )
        logger.log_metric(step, "train", train_loss_nats, train_bpb, lr, tok_per_sec)
        last_log_time = now
        tokens_seen_at_last_log = tokens_seen

    # Eval
    if step > 0 and step % C.EVAL_INTERVAL == 0:
        val_loss, val_bpb = estimate_loss_and_bpb(
            model, val_arr, C.BLOCK_SIZE, C.BATCH_SIZE, C.EVAL_ITERS, device, None, C.USE_AMP
        )
        logger.log_line(f"[eval] step {step} | val_loss(nats) {val_loss:.4f} | val_bpb {val_bpb:.4f}")
        logger.log_metric(step, "val", val_loss, val_bpb, lr, 0.0)

        # Save best
        if val_bpb < best_val_bpb:
            best_val_bpb = val_bpb
            payload = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if C.USE_AMP else None,
                "step": step,
                "best_val_bpb": best_val_bpb,
                "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
            }
            save_checkpoint(ckpt_best, payload)
            logger.log_line(f"[ckpt] saved BEST: {ckpt_best} (best_val_bpb={best_val_bpb:.4f})")

    # Save latest
    if step > 0 and step % C.CKPT_INTERVAL == 0:
        payload = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if C.USE_AMP else None,
            "step": step,
            "best_val_bpb": best_val_bpb,
            "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
        }
        save_checkpoint(ckpt_latest, payload)
        logger.log_line(f"[ckpt] saved latest: {ckpt_latest}")

    # Optional snapshot
    if C.CKPT_SNAPSHOT_INTERVAL and C.CKPT_SNAPSHOT_INTERVAL > 0 and step > 0 and step % C.CKPT_SNAPSHOT_INTERVAL == 0:
        snap = os.path.join(C.WORK_DIR, f"ckpt_step_{step:06d}.pt")
        payload = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if C.USE_AMP else None,
            "step": step,
            "best_val_bpb": best_val_bpb,
            "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
        }
        save_checkpoint(snap, payload)
        logger.log_line(f"[ckpt] saved snapshot: {snap}")

    step += 1

# Final save
payload = {
    "model": raw_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scaler": scaler.state_dict() if C.USE_AMP else None,
    "step": step,
    "best_val_bpb": best_val_bpb,
    "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
}
save_checkpoint(ckpt_latest, payload)
logger.log_line(f"Training done. Final ckpt: {ckpt_latest}")
