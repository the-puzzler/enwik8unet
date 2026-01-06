# train_enwik8.py
import os
import math
import time
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from unet_transformer import UNetTransformer

LN2 = math.log(2.0)

# -----------------------------
# Utils
# -----------------------------
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

def product(xs):
    p = 1
    for x in xs:
        p *= int(x)
    return p

def _unwrap_model(m):
    """Return the underlying nn.Module (handles torch.compile wrapping)."""
    return getattr(m, "_orig_mod", m)

# -----------------------------
# Data
# -----------------------------
def load_enwik8_memmap(data_path: str) -> np.memmap:
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
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test

def get_causal_mask(seq_len: int, device: torch.device):
    # shape [1, 1, S, S]; 1=allow, 0=block (matches your attention masked_fill(mask==0))
    m = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.uint8))
    return m[None, None, :, :]

def sample_batch(data_arr: np.ndarray, batch_size: int, block_size: int, device: torch.device):
    max_start = len(data_arr) - (block_size + 1)
    starts = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([data_arr[s:s+block_size].astype(np.int64) for s in starts], axis=0)
    y = np.stack([data_arr[s+1:s+block_size+1].astype(np.int64) for s in starts], axis=0)
    x = torch.from_numpy(x).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    return x, y

@torch.no_grad()
def estimate_loss_and_bpb(model, data_arr, block_size, batch_size, eval_iters, device, mask, use_amp: bool):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = sample_batch(data_arr, batch_size, block_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x, mask=mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    mean_loss = float(np.mean(losses))     # nats
    mean_bpb = mean_loss / LN2             # bits per byte (base-2)
    model.train()
    return mean_loss, mean_bpb

# -----------------------------
# LR schedule
# -----------------------------
def get_lr(step: int):
    if step < C.WARMUP_STEPS:
        return C.LR * step / max(1, C.WARMUP_STEPS)
    # cosine decay to MIN_LR
    t = (step - C.WARMUP_STEPS) / max(1, C.MAX_STEPS - C.WARMUP_STEPS)
    t = min(max(t, 0.0), 1.0)
    return C.MIN_LR + 0.5 * (C.LR - C.MIN_LR) * (1.0 + math.cos(math.pi * t))

# -----------------------------
# Logging
# -----------------------------
class TextLogger:
    def __init__(self, work_dir: str):
        ensure_dir(work_dir)
        self.log_path = os.path.join(work_dir, "train.log")
        self.csv_path = os.path.join(work_dir, "metrics.csv")

        # Write headers if new
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

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(C.WORK_DIR)
    set_seed(C.SEED)

    if C.DEVICE != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA not available (or config DEVICE != 'cuda').")

    device = torch.device("cuda")

    # Validate block size constraints for your architecture
    ws_prod = product(C.WINDOW_SIZES)
    if C.BLOCK_SIZE % ws_prod != 0:
        raise ValueError(f"BLOCK_SIZE={C.BLOCK_SIZE} must be divisible by product(WINDOW_SIZES)={ws_prod}.")
    if C.BLOCK_SIZE > C.ROPE_MAX_SEQ_LEN:
        raise ValueError(
            f"BLOCK_SIZE={C.BLOCK_SIZE} exceeds ROPE_MAX_SEQ_LEN={C.ROPE_MAX_SEQ_LEN}.\n"
            "Either lower BLOCK_SIZE or increase RoPE max_seq_len in your model."
        )

    logger = TextLogger(C.WORK_DIR)

    # Save config snapshot
    cfg_dump = {k: getattr(C, k) for k in dir(C) if k.isupper()}
    with open(os.path.join(C.WORK_DIR, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)

    # Data
    data = load_enwik8_memmap(C.DATA_PATH)
    train_arr, val_arr, test_arr = make_splits(data, train_frac=0.9, val_frac=0.05)
    logger.log_line(
        f"Loaded enwik8 bytes: {len(data):,} | train {len(train_arr):,} | val {len(val_arr):,} | test {len(test_arr):,}"
    )

    # Model
    model = UNetTransformer(
        vocab_size=C.VOCAB_SIZE,
        dim=C.DIM,
        num_heads=C.NUM_HEADS,
        mlp_ratio=C.MLP_RATIO,
        dropout=C.DROPOUT,
        window_sizes=C.WINDOW_SIZES,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.log_line(f"Model params: {n_params:,}")

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
        model.load_state_dict(ckpt["model"])
    if C.USE_COMPILE:
        model = torch.compile(model)

    # Optim
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=C.LR,
        betas=C.BETAS,
        weight_decay=C.WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=C.USE_AMP) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=C.USE_AMP)

    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        if C.USE_AMP and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

    # Fixed causal mask (BLOCK_SIZE fixed)
    causal_mask = get_causal_mask(C.BLOCK_SIZE, device)

    model.train()
    t0 = time.time()
    last_log_time = t0
    tokens_seen = 0
    tokens_seen_at_last_log = 0

    logger.log_line("Starting training loop.")

    while step < C.MAX_STEPS:
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(C.GRAD_ACCUM):
            x, y = sample_batch(train_arr, C.BATCH_SIZE, C.BLOCK_SIZE, device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=C.USE_AMP):
                logits = model(x, mask=causal_mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / C.GRAD_ACCUM

            scaler.scale(loss).backward()
            accum_loss += loss.item()
            tokens_seen += x.numel()

        if C.CLIP_GRAD_NORM and C.CLIP_GRAD_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.CLIP_GRAD_NORM)

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
                model, val_arr, C.BLOCK_SIZE, C.BATCH_SIZE, C.EVAL_ITERS, device, causal_mask, C.USE_AMP
            )
            logger.log_line(f"[eval] step {step} | val_loss(nats) {val_loss:.4f} | val_bpb {val_bpb:.4f}")
            logger.log_metric(step, "val", val_loss, val_bpb, lr, 0.0)

            # Save best
            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
                payload = {
                    "model": model.state_dict(),
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
                "model": model.state_dict(),
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
                "model": model.state_dict(),
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
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if C.USE_AMP else None,
        "step": step,
        "best_val_bpb": best_val_bpb,
        "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
    }
    save_checkpoint(ckpt_latest, payload)
    logger.log_line(f"Training done. Final ckpt: {ckpt_latest}")


if __name__ == "__main__":
    main()
