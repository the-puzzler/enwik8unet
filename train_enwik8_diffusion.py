# train_enwik8_diffusion.py
import os
import math
import time
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import config_diffusion as C
from diffusion_unet_transformer import TextDiffusionUNetTransformer
from sigreg import SIGReg


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
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]
    return train, val, test


def sample_tokens(data_arr: np.ndarray, batch_size: int, block_size: int, device: torch.device):
    max_start = len(data_arr) - block_size
    starts = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([data_arr[s : s + block_size].astype(np.int64) for s in starts], axis=0)
    x = torch.from_numpy(x).to(device, non_blocking=True)
    return x


# -----------------------------
# Diffusion schedule (cosine, continuous t in [0,1])
# -----------------------------
def alpha_sigma(t: torch.Tensor):
    # t: [B] in [0,1]
    half_pi = 0.5 * math.pi
    a = torch.cos(t * half_pi)
    s = torch.sin(t * half_pi)
    return a, s


class LossAwareTSampler:
    def __init__(self, num_bins: int, ema_beta: float, sample_power: float):
        self.num_bins = int(num_bins)
        self.ema_beta = float(ema_beta)
        self.sample_power = float(sample_power)
        self.ema = torch.ones(self.num_bins, dtype=torch.float32)

    def to(self, device: torch.device):
        self.ema = self.ema.to(device)
        return self

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        probs = torch.clamp(self.ema, min=1e-8).pow(self.sample_power)
        probs = probs / probs.sum()
        bins = torch.multinomial(probs, num_samples=int(batch_size), replacement=True)
        u = torch.rand(int(batch_size), device=device)
        t = (bins.to(device) + u) / float(self.num_bins)
        return t.clamp_(0.0, 1.0)

    @torch.no_grad()
    def update(self, t: torch.Tensor, per_example_loss: torch.Tensor):
        # t: [B], per_example_loss: [B] (float)
        t = t.detach().float().clamp(0.0, 1.0)
        per_example_loss = per_example_loss.detach().float()
        bins = torch.clamp((t * self.num_bins).long(), min=0, max=self.num_bins - 1)
        # EMA update per bin (vectorized via scatter add/count)
        sums = torch.zeros_like(self.ema)
        counts = torch.zeros_like(self.ema)
        sums.scatter_add_(0, bins, per_example_loss)
        counts.scatter_add_(0, bins, torch.ones_like(per_example_loss))
        means = sums / torch.clamp(counts, min=1.0)
        touched = counts > 0
        self.ema[touched] = self.ema[touched] * self.ema_beta + means[touched] * (1.0 - self.ema_beta)


def l2_decode_xent_nats(pred: torch.Tensor, tokens: torch.Tensor, emb_weight: torch.Tensor, tau: float) -> torch.Tensor:
    # Gaussian classifier in embedding space:
    #   p(token=i | pred) ∝ exp(-||pred - e_i||^2 / (2*tau^2))
    pred_f = pred.float()
    emb_f = emb_weight.float()
    pred_sq = (pred_f * pred_f).sum(dim=-1, keepdim=True)  # [B,S,1]
    emb_sq = (emb_f * emb_f).sum(dim=-1)  # [V]
    dot = pred_f @ emb_f.t()  # [B,S,V]
    dist2 = pred_sq + emb_sq - 2.0 * dot
    logits = -dist2 / (2.0 * (float(tau) ** 2))
    return F.cross_entropy(logits.view(-1, logits.size(-1)), tokens.view(-1))


def cosine_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = F.normalize(pred.float(), p=2, dim=-1)
    target_n = F.normalize(target.float(), p=2, dim=-1)
    return (1.0 - (pred_n * target_n).sum(dim=-1)).mean()


@torch.no_grad()
def estimate_loss(model, token_emb, sigreg, data_arr, block_size, batch_size, eval_iters, device, use_amp: bool):
    model.eval()
    token_emb.eval()
    total_losses = []
    mse_losses = []
    cosine_losses = []
    sigreg_losses = []
    sigreg_raw_terms = []
    xent_losses = []
    bpb_vals = []
    sigreg_raw = sigreg(token_emb.weight)
    sigreg_loss = float(C.SIGREG_WEIGHT) * sigreg_raw
    for _ in range(eval_iters):
        tokens = sample_tokens(data_arr, batch_size, block_size, device)
        t = torch.rand(tokens.size(0), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            x0 = token_emb(tokens)
            eps = torch.randn_like(x0)
            a, s = alpha_sigma(t)
            a = a[:, None, None].to(dtype=x0.dtype)
            s = s[:, None, None].to(dtype=x0.dtype)
            xt = a * x0 + s * eps
            pred = model(xt, t, mask=None)
            mse_loss = F.mse_loss(pred, x0)
            cos_loss = cosine_distance(pred, x0)
            total_loss = mse_loss + sigreg_loss
            xent_nats = l2_decode_xent_nats(pred, tokens, token_emb.weight, C.DECODE_TAU)
        total_losses.append(total_loss.item())
        mse_losses.append(mse_loss.item())
        cosine_losses.append(cos_loss.item())
        sigreg_losses.append(float(sigreg_loss))
        sigreg_raw_terms.append(sigreg_raw.item())
        xent_losses.append(xent_nats.item())
        bpb_vals.append((xent_nats / LN2).item())
    model.train()
    token_emb.train()
    return (
        float(np.mean(total_losses)),
        float(np.mean(mse_losses)),
        float(np.mean(cosine_losses)),
        float(np.mean(sigreg_losses)),
        float(np.mean(sigreg_raw_terms)),
        float(np.mean(xent_losses)),
        float(np.mean(bpb_vals)),
    )


# -----------------------------
# LR schedule
# -----------------------------
def get_lr(step: int):
    if step < C.WARMUP_STEPS:
        return C.LR * step / max(1, C.WARMUP_STEPS)
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
        desired_header = (
            "time,step,split,total_loss,mse_loss,cosine_loss,sigreg_loss,sigreg_raw,xent_nats,bpb,"
            "uniform_total_loss,uniform_mse_loss,uniform_cosine_loss,uniform_xent_nats,uniform_bpb,"
            "lr,tok_per_sec\n"
        )
        if os.path.isfile(self.csv_path):
            with open(self.csv_path, "r", encoding="utf-8") as f:
                first = f.readline()
            if first != desired_header:
                self.csv_path = os.path.join(work_dir, "metrics_v4.csv")
        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write(desired_header)

    def log_line(self, msg: str):
        line = f"[{now_str()}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_metric(
        self,
        step: int,
        split: str,
        total_loss: float,
        mse_loss: float,
        cosine_loss: float,
        sigreg_loss: float,
        sigreg_raw: float,
        xent_nats: float,
        bpb: float,
        uniform_total_loss: float,
        uniform_mse_loss: float,
        uniform_cosine_loss: float,
        uniform_xent_nats: float,
        uniform_bpb: float,
        lr: float,
        tok_per_sec: float,
    ):
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{now_str()},{step},{split},{total_loss:.6f},{mse_loss:.6f},{cosine_loss:.6f},{sigreg_loss:.6f},{sigreg_raw:.6f},{xent_nats:.6f},{bpb:.6f},"
                f"{uniform_total_loss:.6f},{uniform_mse_loss:.6f},{uniform_cosine_loss:.6f},{uniform_xent_nats:.6f},{uniform_bpb:.6f},"
                f"{lr:.8e},{tok_per_sec:.2f}\n"
            )


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(C.WORK_DIR)
    set_seed(C.SEED)

    if C.DEVICE != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA not available (or config DEVICE != 'cuda').")

    device = torch.device("cuda")

    ws_prod = product(C.WINDOW_SIZES)
    if C.BLOCK_SIZE % ws_prod != 0:
        raise ValueError(f"BLOCK_SIZE={C.BLOCK_SIZE} must be divisible by product(WINDOW_SIZES)={ws_prod}.")
    if C.BLOCK_SIZE > C.ROPE_MAX_SEQ_LEN:
        raise ValueError(f"BLOCK_SIZE={C.BLOCK_SIZE} exceeds ROPE_MAX_SEQ_LEN={C.ROPE_MAX_SEQ_LEN}.")

    logger = TextLogger(C.WORK_DIR)

    cfg_dump = {k: getattr(C, k) for k in dir(C) if k.isupper()}
    with open(os.path.join(C.WORK_DIR, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)

    data = load_enwik8_memmap(C.DATA_PATH)
    train_arr, val_arr, test_arr = make_splits(data, train_frac=0.9, val_frac=0.05)
    logger.log_line(
        f"Loaded enwik8 bytes: {len(data):,} | train {len(train_arr):,} | val {len(val_arr):,} | test {len(test_arr):,}"
    )

    token_emb = nn.Embedding(C.VOCAB_SIZE, C.INPUT_DIM).to(device)
    model = TextDiffusionUNetTransformer(
        dim=C.DIM,
        num_heads=C.NUM_HEADS,
        mlp_ratio=C.MLP_RATIO,
        dropout=C.DROPOUT,
        window_sizes=C.WINDOW_SIZES,
        input_dim=C.INPUT_DIM,
        out_dim=C.OUT_DIM,
    ).to(device)
    sigreg = SIGReg(knots=C.SIGREG_KNOTS).to(device)
    t_sampler = None
    if getattr(C, "T_SAMPLER", "uniform") == "loss_aware":
        t_sampler = LossAwareTSampler(C.T_BINS, C.T_EMA_BETA, C.T_SAMPLE_POWER).to(device)

    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in token_emb.parameters())
    logger.log_line(f"Model+embed params: {n_params:,}")

    ckpt_latest = os.path.join(C.WORK_DIR, "ckpt_latest.pt")
    ckpt_best = os.path.join(C.WORK_DIR, "ckpt_best.pt")

    step = 0
    best_val_loss = float("inf")
    ckpt = None
    if os.path.isfile(ckpt_latest):
        ckpt = load_checkpoint(ckpt_latest, map_location="cpu")
        step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        logger.log_line(f"Resumed from {ckpt_latest} at step={step} best_val_loss={best_val_loss:.6f}")
    else:
        logger.log_line("No ckpt_latest found; starting fresh.")

    if ckpt is not None:
        token_emb.load_state_dict(ckpt["token_emb"])
        model.load_state_dict(ckpt["model"])
        if t_sampler is not None and ckpt.get("t_sampler_ema") is not None:
            t_sampler.ema.copy_(ckpt["t_sampler_ema"].to(device))

    if C.USE_COMPILE:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        list(token_emb.parameters()) + list(model.parameters()),
        lr=C.LR,
        betas=C.BETAS,
        weight_decay=C.WEIGHT_DECAY,
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=C.USE_AMP)
        if hasattr(torch, "amp")
        else torch.cuda.amp.GradScaler(enabled=C.USE_AMP)
    )

    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        if C.USE_AMP and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

    model.train()
    token_emb.train()

    t0 = time.time()
    last_log_time = t0
    tokens_seen = 0
    tokens_seen_at_last_log = 0

    logger.log_line("Starting diffusion training loop.")

    while step < C.MAX_STEPS:
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_total = 0.0
        accum_mse = 0.0
        accum_cosine = 0.0
        accum_sigreg_loss = 0.0
        accum_sigreg_raw = 0.0
        accum_xent_nats = 0.0
        accum_bpb = 0.0
        accum_u_total = 0.0
        accum_u_mse = 0.0
        accum_u_cosine = 0.0
        accum_u_xent = 0.0
        accum_u_bpb = 0.0
        uniform_count = 0

        for _ in range(C.GRAD_ACCUM):
            tokens = sample_tokens(train_arr, C.BATCH_SIZE, C.BLOCK_SIZE, device)
            if t_sampler is not None and step >= getattr(C, "T_WARMUP_T_STEPS", 0):
                if torch.rand((), device=device).item() < float(getattr(C, "T_UNIFORM_MIX", 0.0)):
                    t = torch.rand(tokens.size(0), device=device)
                    is_uniform = True
                else:
                    t = t_sampler.sample(tokens.size(0), device=device)
                    is_uniform = False
            else:
                t = torch.rand(tokens.size(0), device=device)
                is_uniform = True

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=C.USE_AMP):
                x0 = token_emb(tokens)
                eps = torch.randn_like(x0)
                a, s = alpha_sigma(t)
                a = a[:, None, None].to(dtype=x0.dtype)
                s = s[:, None, None].to(dtype=x0.dtype)
                xt = a * x0 + s * eps
                pred = model(xt, t, mask=None)
                per_ex_mse = F.mse_loss(pred.float(), x0.float(), reduction="none").mean(dim=(1, 2))
                mse_loss = per_ex_mse.mean()
                cos_loss = cosine_distance(pred, x0)
                xent_nats = l2_decode_xent_nats(pred, tokens, token_emb.weight, C.DECODE_TAU)
            if t_sampler is not None and step >= getattr(C, "T_WARMUP_T_STEPS", 0):
                t_sampler.update(t, per_ex_mse)
            sigreg_raw = sigreg(token_emb.weight)
            sigreg_loss = float(C.SIGREG_WEIGHT) * sigreg_raw
            total_loss = mse_loss + sigreg_loss
            loss = total_loss / C.GRAD_ACCUM

            scaler.scale(loss).backward()
            accum_total += total_loss.item()
            accum_mse += mse_loss.item()
            accum_cosine += cos_loss.item()
            accum_sigreg_loss += float(sigreg_loss)
            accum_sigreg_raw += sigreg_raw.item()
            accum_xent_nats += xent_nats.item()
            accum_bpb += (xent_nats / LN2).item()
            if is_uniform:
                uniform_count += 1
                accum_u_total += total_loss.item()
                accum_u_mse += mse_loss.item()
                accum_u_cosine += cos_loss.item()
                accum_u_xent += xent_nats.item()
                accum_u_bpb += (xent_nats / LN2).item()
            tokens_seen += tokens.numel()

        if C.CLIP_GRAD_NORM and C.CLIP_GRAD_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(token_emb.parameters()) + list(_unwrap_model(model).parameters()), C.CLIP_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        if step % C.LOG_INTERVAL == 0:
            now = time.time()
            dt = now - last_log_time
            tok_per_sec = (tokens_seen - tokens_seen_at_last_log) / max(1e-9, dt)
            train_total = accum_total / C.GRAD_ACCUM
            train_mse = accum_mse / C.GRAD_ACCUM
            train_cosine = accum_cosine / C.GRAD_ACCUM
            train_sigreg_loss = accum_sigreg_loss / C.GRAD_ACCUM
            train_sigreg_raw = accum_sigreg_raw / C.GRAD_ACCUM
            train_xent_nats = accum_xent_nats / C.GRAD_ACCUM
            train_bpb = accum_bpb / C.GRAD_ACCUM
            u_denom = max(1, uniform_count)
            u_total = accum_u_total / u_denom
            u_mse = accum_u_mse / u_denom
            u_cosine = accum_u_cosine / u_denom
            u_xent = accum_u_xent / u_denom
            u_bpb = accum_u_bpb / u_denom
            logger.log_line(
                f"step {step} | lr {lr:.2e} | total {train_total:.6f} | mse {train_mse:.6f} | cosine {train_cosine:.6f} | sigreg {train_sigreg_loss:.6f} | xent {train_xent_nats:.6f} | bpb {train_bpb:.6f} | "
                f"uniform(total/mse/xent) {u_total:.6f}/{u_mse:.6f}/{u_xent:.6f} | tok/s {human_num(tok_per_sec)}"
            )
            logger.log_metric(
                step,
                "train",
                train_total,
                train_mse,
                train_cosine,
                train_sigreg_loss,
                train_sigreg_raw,
                train_xent_nats,
                train_bpb,
                u_total,
                u_mse,
                u_cosine,
                u_xent,
                u_bpb,
                lr,
                tok_per_sec,
            )
            last_log_time = now
            tokens_seen_at_last_log = tokens_seen

        if step > 0 and step % C.EVAL_INTERVAL == 0:
            val_total, val_mse, val_cosine, val_sigreg_loss, val_sigreg_raw, val_xent_nats, val_bpb = estimate_loss(
                model, token_emb, sigreg, val_arr, C.BLOCK_SIZE, C.BATCH_SIZE, C.EVAL_ITERS, device, C.USE_AMP
            )
            logger.log_line(
                f"[eval] step {step} | val_total {val_total:.6f} | val_mse {val_mse:.6f} | val_cosine {val_cosine:.6f} | val_sigreg {val_sigreg_loss:.6f} | val_xent {val_xent_nats:.6f} | val_bpb {val_bpb:.6f}"
            )
            logger.log_metric(
                step,
                "val",
                val_total,
                val_mse,
                val_cosine,
                val_sigreg_loss,
                val_sigreg_raw,
                val_xent_nats,
                val_bpb,
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                lr,
                0.0,
            )

            if val_total < best_val_loss:
                best_val_loss = val_total
                payload = {
                    "token_emb": token_emb.state_dict(),
                    "model": _unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if C.USE_AMP else None,
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
                }
                save_checkpoint(ckpt_best, payload)
                logger.log_line(f"[ckpt] saved BEST: {ckpt_best} (best_val_loss={best_val_loss:.6f})")

        if step > 0 and step % C.CKPT_INTERVAL == 0:
            payload = {
                "token_emb": token_emb.state_dict(),
                "model": _unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if C.USE_AMP else None,
                "step": step,
                "best_val_loss": best_val_loss,
                "t_sampler_ema": t_sampler.ema.detach().cpu() if t_sampler is not None else None,
                "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
            }
            save_checkpoint(ckpt_latest, payload)
            logger.log_line(f"[ckpt] saved latest: {ckpt_latest}")

        if (
            C.CKPT_SNAPSHOT_INTERVAL
            and C.CKPT_SNAPSHOT_INTERVAL > 0
            and step > 0
            and step % C.CKPT_SNAPSHOT_INTERVAL == 0
        ):
            snap = os.path.join(C.WORK_DIR, f"ckpt_step_{step:06d}.pt")
            payload = {
                "token_emb": token_emb.state_dict(),
                "model": _unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if C.USE_AMP else None,
                "step": step,
                "best_val_loss": best_val_loss,
                "t_sampler_ema": t_sampler.ema.detach().cpu() if t_sampler is not None else None,
                "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
            }
            save_checkpoint(snap, payload)
            logger.log_line(f"[ckpt] saved snapshot: {snap}")

        step += 1

    payload = {
        "token_emb": token_emb.state_dict(),
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if C.USE_AMP else None,
        "step": step,
        "best_val_loss": best_val_loss,
        "t_sampler_ema": t_sampler.ema.detach().cpu() if t_sampler is not None else None,
        "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
    }
    save_checkpoint(ckpt_latest, payload)
    logger.log_line(f"Training done. Final ckpt: {ckpt_latest}")


if __name__ == "__main__":
    main()
