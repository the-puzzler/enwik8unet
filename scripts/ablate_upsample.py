"""
Mini ablation for UNet upsampling mode on enwik8.

Runs short, matched training jobs for:
  1) interpolation upsampling
  2) learned expansion upsampling

Then reports validation bpb so you can compare quickly without full training.
"""

import argparse
import json
import math
import os
import random
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unet_transformer import UNetTransformer


LN2 = math.log(2.0)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_enwik8_memmap(data_path: str) -> np.memmap:
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Missing dataset file: {data_path}")
    return np.memmap(data_path, dtype=np.uint8, mode="r")


def make_splits(data: np.memmap, train_frac=0.9, val_frac=0.05):
    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    return train, val


def sample_batch(data_arr: np.ndarray, batch_size: int, block_size: int, device: torch.device):
    max_start = len(data_arr) - (block_size + 1)
    starts = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([data_arr[s:s + block_size].astype(np.int64) for s in starts], axis=0)
    y = np.stack([data_arr[s + 1:s + block_size + 1].astype(np.int64) for s in starts], axis=0)
    x = torch.from_numpy(x).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    return x, y


def get_causal_mask(seq_len: int, device: torch.device):
    m = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.uint8))
    return m[None, None, :, :]


@torch.no_grad()
def estimate_val_bpb(model, val_arr, block_size, batch_size, eval_iters, device, mask, use_amp: bool):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = sample_batch(val_arr, batch_size, block_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x, mask=mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    mean_loss = float(np.mean(losses))
    return mean_loss / LN2


def run_short_train(
    mode: str,
    train_arr,
    val_arr,
    device: torch.device,
    steps: int,
    eval_every: int,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    grad_accum: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    weight_decay: float,
    betas,
    clip_grad_norm: float,
    dim: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    window_sizes,
    use_amp: bool,
):
    model = UNetTransformer(
        vocab_size=256,
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        window_sizes=window_sizes,
        upsample_mode=mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    mask = get_causal_mask(block_size, device)

    best_val_bpb = float("inf")
    start = time.time()

    for step in range(steps):
        if step < warmup_steps:
            cur_lr = lr * step / max(1, warmup_steps)
        else:
            t = (step - warmup_steps) / max(1, steps - warmup_steps)
            t = min(max(t, 0.0), 1.0)
            cur_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * t))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = sample_batch(train_arr, batch_size, block_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x, mask=mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum
            scaler.scale(loss).backward()

        if clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % eval_every == 0 or step == steps - 1:
            val_bpb = estimate_val_bpb(
                model=model,
                val_arr=val_arr,
                block_size=block_size,
                batch_size=batch_size,
                eval_iters=eval_iters,
                device=device,
                mask=mask,
                use_amp=use_amp,
            )
            best_val_bpb = min(best_val_bpb, val_bpb)
            print(f"[{mode}] step={step + 1}/{steps} val_bpb={val_bpb:.5f} best={best_val_bpb:.5f}")

    elapsed = time.time() - start
    return {
        "mode": mode,
        "best_val_bpb": best_val_bpb,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/enwik8")
    parser.add_argument("--out", default="runs/ablate_upsample/results.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--repeats", type=int, default=1, help="number of repeated runs with incrementing seeds")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[4, 4, 2])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be >= 1")

    ws_prod = 1
    for w in args.window_sizes:
        ws_prod *= int(w)
    if args.block_size % ws_prod != 0:
        raise ValueError(
            f"block-size ({args.block_size}) must be divisible by product(window-sizes) ({ws_prod})."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if args.device not in ("cuda", "cpu"):
        raise ValueError("--device must be 'cuda' or 'cpu'")
    device = torch.device(args.device)

    use_amp = device.type == "cuda"
    set_seed(args.seed)

    data = load_enwik8_memmap(args.data_path)
    train_arr, val_arr = make_splits(data, train_frac=0.9, val_frac=0.05)

    common = dict(
        train_arr=train_arr,
        val_arr=val_arr,
        device=device,
        steps=args.steps,
        eval_every=args.eval_every,
        eval_iters=args.eval_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        clip_grad_norm=args.clip_grad_norm,
        dim=args.dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        window_sizes=args.window_sizes,
        use_amp=use_amp,
    )

    runs = []
    for rep in range(args.repeats):
        run_seed = args.seed + rep
        print(f"\n=== Repeat {rep + 1}/{args.repeats} | seed={run_seed} ===")
        per_mode = []
        for mode in ("interp", "expand"):
            set_seed(run_seed)
            per_mode.append(run_short_train(mode=mode, **common))
        interp = next(r for r in per_mode if r["mode"] == "interp")
        expand = next(r for r in per_mode if r["mode"] == "expand")
        runs.append(
            {
                "repeat": rep + 1,
                "seed": run_seed,
                "results": per_mode,
                "delta_best_val_bpb_expand_minus_interp": expand["best_val_bpb"] - interp["best_val_bpb"],
            }
        )

    interp_vals = []
    expand_vals = []
    delta_vals = []
    for run in runs:
        by_mode = {r["mode"]: r["best_val_bpb"] for r in run["results"]}
        i = by_mode["interp"]
        e = by_mode["expand"]
        interp_vals.append(i)
        expand_vals.append(e)
        delta_vals.append(e - i)

    def mean_std(vals):
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return mean, std

    interp_mean, interp_std = mean_std(interp_vals)
    expand_mean, expand_std = mean_std(expand_vals)
    delta_mean, delta_std = mean_std(delta_vals)

    summary = {
        "config": vars(args),
        "runs": runs,
        "aggregate": {
            "n": len(runs),
            "interp_best_val_bpb_mean": interp_mean,
            "interp_best_val_bpb_std": interp_std,
            "expand_best_val_bpb_mean": expand_mean,
            "expand_best_val_bpb_std": expand_std,
            "delta_expand_minus_interp_mean": delta_mean,
            "delta_expand_minus_interp_std": delta_std,
        },
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(f"interp best val bpb: {interp_mean:.5f} +- {interp_std:.5f}")
    print(f"expand best val bpb: {expand_mean:.5f} +- {expand_std:.5f}")
    print(f"expand - interp: {delta_mean:+.5f} +- {delta_std:.5f}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
