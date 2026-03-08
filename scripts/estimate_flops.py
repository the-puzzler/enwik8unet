"""
No-args model profiler.

What it does:
1) Reads config snapshots from UNet and baseline run directories (including baseline_small if present).
2) Runs short simulated train benchmarks using each run's batch/seq/grad_accum.
3) Runs inference benchmarks using each run's batch/seq.
4) Saves CSV/JSON + plots and prints a compact summary.
"""

import csv
import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

import config as C
from baseline_transformer import BaselineTransformer
from unet_transformer import UNetTransformer


# ---------------
# Fixed settings
# ---------------
RUN_SPECS = {
    "unet": {
        "model_type": "unet",
        "candidates": ["runs/enwik8_unet", "runs/enwik8_unet_done"],
    },
    "baseline": {
        "model_type": "baseline",
        "candidates": ["runs/enwik8_baseline"],
    },
    "baseline_small": {
        "model_type": "baseline",
        "candidates": ["runs/enwik8_baseline_small"],
    },
}
MODEL_ORDER = ("unet", "baseline_small", "baseline")
MODEL_COLORS = {
    "unet": "#1f77b4",
    "baseline": "#ff7f0e",
    "baseline_small": "#2ca02c",
}
MODEL_LABEL_Y_OFFSETS = {
    "unet": 0.026,
    "baseline": -0.073,
    "baseline_small": -0.030,
}
OUT_DIR = "profiling"
TEST_SCORES_CSV = os.path.join(OUT_DIR, "test_scores.csv")

TRAIN_WARMUP_STEPS = 2
TRAIN_BENCH_STEPS = 5
INFER_WARMUP_ITERS = 8
INFER_BENCH_ITERS = 40
TRAIN_FLOP_MULTIPLIER = 3.0  # proxy: fwd+bwd+optimizer ~= 3x forward FLOPs
ENABLE_COMPILE = False  # keep profiler stable and avoid compile-time memory spikes
INFER_BATCH_SIZE = 1


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pick_run_dir(candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def load_config_snapshot(work_dir: str):
    path = os.path.join(work_dir, "config_snapshot.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cfg_get(cfg, key, default):
    return cfg.get(key, default)


def fill_defaults(cfg):
    use_amp = cfg_get(cfg, "USE_AMP", C.USE_AMP)
    return {
        "MODEL_TYPE": str(cfg_get(cfg, "MODEL_TYPE", C.MODEL_TYPE)).lower(),
        "BATCH_SIZE": int(cfg_get(cfg, "BATCH_SIZE", C.BATCH_SIZE)),
        "GRAD_ACCUM": int(cfg_get(cfg, "GRAD_ACCUM", C.GRAD_ACCUM)),
        "BLOCK_SIZE": int(cfg_get(cfg, "BLOCK_SIZE", C.BLOCK_SIZE)),
        "DIM": int(cfg_get(cfg, "DIM", C.DIM)),
        "NUM_HEADS": int(cfg_get(cfg, "NUM_HEADS", C.NUM_HEADS)),
        "MLP_RATIO": float(cfg_get(cfg, "MLP_RATIO", C.MLP_RATIO)),
        "NUM_LAYERS": int(cfg_get(cfg, "NUM_LAYERS", getattr(C, "NUM_LAYERS", 12))),
        "WINDOW_SIZES": list(cfg_get(cfg, "WINDOW_SIZES", C.WINDOW_SIZES)),
        "DROPOUT": float(cfg_get(cfg, "DROPOUT", C.DROPOUT)),
        "USE_AMP": bool(use_amp),
        "USE_COMPILE": bool(cfg_get(cfg, "USE_COMPILE", C.USE_COMPILE)),
    }


def attn_flops(batch, seq_len, dim, num_heads):
    d_head = dim // num_heads
    qkv = 6 * batch * seq_len * dim * dim
    out = 2 * batch * seq_len * dim * dim
    attn = 4 * batch * num_heads * seq_len * seq_len * d_head
    return qkv + out + attn


def mlp_flops(batch, seq_len, dim, mlp_ratio):
    hidden = int(dim * mlp_ratio)
    return 6 * batch * seq_len * dim * hidden


def block_flops(batch, seq_len, dim, num_heads, mlp_ratio):
    return attn_flops(batch, seq_len, dim, num_heads) + mlp_flops(batch, seq_len, dim, mlp_ratio)


def estimate_forward_flops(model_type: str, cfg: dict, batch: int, seq_len: int):
    dim = cfg["DIM"]
    num_heads = cfg["NUM_HEADS"]
    mlp_ratio = cfg["MLP_RATIO"]
    if model_type == "baseline":
        per_block = block_flops(batch, seq_len, dim, num_heads, mlp_ratio)
        return per_block * cfg["NUM_LAYERS"]

    total = 0
    seq = seq_len
    for w in cfg["WINDOW_SIZES"]:
        total += block_flops(batch, seq, dim, num_heads, mlp_ratio)
        seq = seq // int(w)
    total += block_flops(batch, seq, dim, num_heads, mlp_ratio)
    for w in reversed(cfg["WINDOW_SIZES"]):
        seq = seq * int(w)
        total += block_flops(batch, seq, dim, num_heads, mlp_ratio)
    return total


def get_causal_mask(seq_len: int, device: torch.device):
    m = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.uint8))
    return m[None, None, :, :]


def build_model(model_type: str, cfg: dict, device: torch.device):
    if model_type == "unet":
        model = UNetTransformer(
            vocab_size=C.VOCAB_SIZE,
            dim=cfg["DIM"],
            num_heads=cfg["NUM_HEADS"],
            mlp_ratio=cfg["MLP_RATIO"],
            dropout=cfg["DROPOUT"],
            window_sizes=cfg["WINDOW_SIZES"],
        )
    elif model_type == "baseline":
        model = BaselineTransformer(
            vocab_size=C.VOCAB_SIZE,
            dim=cfg["DIM"],
            num_heads=cfg["NUM_HEADS"],
            mlp_ratio=cfg["MLP_RATIO"],
            dropout=cfg["DROPOUT"],
            num_layers=cfg["NUM_LAYERS"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model.to(device)


def benchmark_train(model, cfg: dict, model_type: str, bench_batch: int):
    device = torch.device("cuda")
    batch = bench_batch
    seq_len = cfg["BLOCK_SIZE"]
    grad_accum = cfg["GRAD_ACCUM"]
    use_amp = cfg["USE_AMP"]
    mask = get_causal_mask(seq_len, device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_amp)

    x = torch.randint(0, C.VOCAB_SIZE, (batch, seq_len), device=device)
    y = torch.randint(0, C.VOCAB_SIZE, (batch, seq_len), device=device)

    for _ in range(TRAIN_WARMUP_STEPS):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x, mask=mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / grad_accum
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(TRAIN_BENCH_STEPS):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x, mask=mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / grad_accum
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    step_s = dt / TRAIN_BENCH_STEPS
    step_ms = step_s * 1000.0
    tokens_per_step = batch * seq_len * grad_accum
    tokens_per_s = tokens_per_step / max(step_s, 1e-9)
    fwd_flops_per_step = estimate_forward_flops(model_type, cfg, batch, seq_len) * grad_accum
    train_flops_proxy = fwd_flops_per_step * TRAIN_FLOP_MULTIPLIER
    achieved_tflops_proxy = (train_flops_proxy / max(step_s, 1e-9)) / 1e12
    peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    return {
        "batch": batch,
        "config_batch": cfg["BATCH_SIZE"],
        "seq_len": seq_len,
        "grad_accum": grad_accum,
        "step_ms": step_ms,
        "tokens_per_s": tokens_per_s,
        "fwd_flops_per_step_tflop": fwd_flops_per_step / 1e12,
        "train_flops_proxy_tflop": train_flops_proxy / 1e12,
        "achieved_tflops_proxy": achieved_tflops_proxy,
        "peak_alloc_mib": peak_alloc_mib,
        "peak_reserved_mib": peak_reserved_mib,
    }


@torch.no_grad()
def benchmark_inference(model, cfg: dict, model_type: str, bench_batch: int):
    device = torch.device("cuda")
    batch = bench_batch
    seq_len = cfg["BLOCK_SIZE"]
    use_amp = cfg["USE_AMP"]
    mask = get_causal_mask(seq_len, device)
    x = torch.randint(0, C.VOCAB_SIZE, (batch, seq_len), device=device)
    model.eval()

    for _ in range(INFER_WARMUP_ITERS):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _ = model(x, mask=mask)
    torch.cuda.synchronize()

    # Single-forward memory traffic measurement (actual allocator bytes in one pass).
    torch.cuda.empty_cache()
    sf_start_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    sf_start_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    sf_stats_before = torch.cuda.memory_stats(device)
    sf_total_alloc_before = sf_stats_before.get("allocated_bytes.all.allocated", 0)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        _ = model(x, mask=mask)
    torch.cuda.synchronize()
    sf_peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    sf_peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    sf_stats_after = torch.cuda.memory_stats(device)
    sf_total_alloc_after = sf_stats_after.get("allocated_bytes.all.allocated", sf_total_alloc_before)
    single_forward_alloc_delta_mib = max(0.0, (sf_total_alloc_after - sf_total_alloc_before) / (1024 ** 2))
    single_forward_peak_alloc_delta_mib = max(0.0, sf_peak_alloc_mib - sf_start_alloc)
    single_forward_peak_reserved_delta_mib = max(0.0, sf_peak_reserved_mib - sf_start_reserved)

    # Multi-iter throughput benchmark.
    torch.cuda.empty_cache()
    start_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    start_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(INFER_BENCH_ITERS):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _ = model(x, mask=mask)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    iter_s = dt / INFER_BENCH_ITERS
    iter_ms = iter_s * 1000.0
    tokens_per_s = (batch * seq_len) / max(iter_s, 1e-9)
    fwd_flops_per_iter = estimate_forward_flops(model_type, cfg, batch, seq_len)
    achieved_tflops = (fwd_flops_per_iter / max(iter_s, 1e-9)) / 1e12
    peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    peak_alloc_delta_mib = max(0.0, peak_alloc_mib - start_alloc)
    peak_reserved_delta_mib = max(0.0, peak_reserved_mib - start_reserved)

    return {
        "batch": batch,
        "config_batch": cfg["BATCH_SIZE"],
        "seq_len": seq_len,
        "iter_ms": iter_ms,
        "tokens_per_s": tokens_per_s,
        "fwd_flops_per_iter_tflop": fwd_flops_per_iter / 1e12,
        "achieved_tflops": achieved_tflops,
        "start_alloc_mib": start_alloc,
        "start_reserved_mib": start_reserved,
        "peak_alloc_mib": peak_alloc_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "peak_alloc_delta_mib": peak_alloc_delta_mib,
        "peak_reserved_delta_mib": peak_reserved_delta_mib,
        "single_forward_alloc_delta_mib": single_forward_alloc_delta_mib,
        "single_forward_peak_alloc_delta_mib": single_forward_peak_alloc_delta_mib,
        "single_forward_peak_reserved_delta_mib": single_forward_peak_reserved_delta_mib,
    }


def save_rows_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_test_scores(path):
    scores = {}
    if not os.path.isfile(path):
        return scores
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model")
            if not model:
                continue
            try:
                scores[model] = {
                    "test_bpb": float(row.get("test_bpb", "nan")),
                    "test_loss_nats": float(row.get("test_loss_nats", "nan")),
                    "eval_iters": int(row.get("eval_iters", "0")),
                    "batch_size": int(row.get("batch_size", "0")),
                }
            except ValueError:
                continue
    return scores


def load_val_rows(metrics_csv_path):
    rows = []
    if not os.path.isfile(metrics_csv_path):
        return rows
    with open(metrics_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            if row[2] != "val":
                continue
            rows.append((int(row[1]), float(row[4])))
    rows.sort(key=lambda x: x[0])
    return rows


def build_val_flops_curves(runs):
    curves = {}
    for run_name, meta in runs.items():
        cfg = meta["cfg"]
        run_dir = meta["run_dir"]
        metrics_csv = os.path.join(run_dir, "metrics.csv")
        val_rows = load_val_rows(metrics_csv)
        flops_per_sample = estimate_forward_flops(meta["model_type"], cfg, batch=1, seq_len=cfg["BLOCK_SIZE"])
        xs = []
        ys = []
        steps = []
        for step, bpb in val_rows:
            cum_fwd = step * cfg["GRAD_ACCUM"] * cfg["BATCH_SIZE"] * flops_per_sample
            xs.append(cum_fwd / 1e15)  # PFLOPs
            ys.append(bpb)
            steps.append(step)
        curves[run_name] = {
            "run_dir": run_dir,
            "metrics_csv": metrics_csv,
            "steps": np.array(steps, dtype=np.int64),
            "x_pflops": np.array(xs, dtype=np.float64),
            "y_bpb": np.array(ys, dtype=np.float64),
            "flops_per_sample": flops_per_sample,
        }
    return curves


def save_val_flops_csv(path, curves):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "step", "cum_forward_pflops", "val_bpb"])
        for model_name in MODEL_ORDER:
            if model_name not in curves:
                continue
            c = curves[model_name]
            for step, x, y in zip(c["steps"], c["x_pflops"], c["y_bpb"]):
                w.writerow([model_name, int(step), f"{x:.8f}", f"{y:.8f}"])


def plot_val_vs_flops(curves, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
    endpoints = []
    for model_name in MODEL_ORDER:
        if model_name not in curves:
            continue
        color = MODEL_COLORS.get(model_name, None)
        c = curves[model_name]
        if len(c["x_pflops"]) == 0:
            continue
        best_idx = int(np.argmin(c["y_bpb"]))
        x = c["x_pflops"][: best_idx + 1]
        y = c["y_bpb"][: best_idx + 1]
        ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.8, label=model_name, color=color)
        x_last = float(x[-1])
        y_last = float(y[-1])
        step_last = int(c["steps"][best_idx])
        ax.plot([x_last], [y_last], marker="o", markersize=2.5, color=color)
        endpoints.append((model_name, color, x_last, y_last, step_last))
    ax.set_xlim(left=0.0)
    for i, (model_name, color, x_last, y_last, step_last) in enumerate(endpoints):
        ax.hlines(y_last, 0.0, x_last, colors=color, linestyles=":", linewidth=1.0, alpha=0.9)
        y_offset = MODEL_LABEL_Y_OFFSETS.get(model_name, 0.016 if i % 2 == 0 else -0.016)
        ax.text(
            -0.04,
            y_last + y_offset,
            f"{model_name}: {y_last:.4f} (step {step_last})",
            fontsize=9,
            color=color,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            ha="right",
            va="center",
        )
    ax.set_title("Validation bpb vs Cumulative Forward FLOPs")
    ax.set_xlabel("Cumulative forward FLOPs (PFLOPs)")
    ax.set_ylabel("Validation bpb")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_val_vs_step(curves, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
    endpoints = []
    for model_name in MODEL_ORDER:
        if model_name not in curves:
            continue
        color = MODEL_COLORS.get(model_name, None)
        c = curves[model_name]
        if len(c["steps"]) == 0:
            continue
        best_idx = int(np.argmin(c["y_bpb"]))
        x = c["steps"][: best_idx + 1]
        y = c["y_bpb"][: best_idx + 1]
        ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.8, label=model_name, color=color)
        x_last = int(x[-1])
        y_last = float(y[-1])
        ax.plot([x_last], [y_last], marker="o", markersize=2.5, color=color)
        endpoints.append((model_name, color, x_last, y_last))
    ax.set_xlim(left=0)
    for i, (model_name, color, x_last, y_last) in enumerate(endpoints):
        ax.hlines(y_last, 0, x_last, colors=color, linestyles=":", linewidth=1.0, alpha=0.9)
        y_offset = MODEL_LABEL_Y_OFFSETS.get(model_name, 0.016 if i % 2 == 0 else -0.016)
        ax.text(
            -0.04,
            y_last + y_offset,
            f"{model_name}: {y_last:.4f} (step {x_last})",
            fontsize=9,
            color=color,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            ha="right",
            va="center",
        )
    ax.set_title("Validation bpb vs Step")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation bpb")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_train_bars(train_rows, out_path):
    labels = [r["model"] for r in train_rows]
    step_ms = [r["step_ms"] for r in train_rows]
    toks = [r["tokens_per_s"] for r in train_rows]
    compute = [r["fwd_flops_per_step_tflop"] for r in train_rows]
    mem = [r["peak_alloc_mib"] for r in train_rows]
    test_bpb = [r.get("test_bpb", float("nan")) for r in train_rows]

    fig, axs = plt.subplots(1, 5, figsize=(20, 4.5))
    axs[0].bar(labels, step_ms)
    axs[0].set_title("Train Step Latency")
    axs[0].set_ylabel("ms/optimizer step")
    axs[1].bar(labels, toks)
    axs[1].set_title("Train Speed")
    axs[1].set_ylabel("tokens/s")
    axs[2].bar(labels, compute)
    axs[2].set_title("Train Compute")
    axs[2].set_ylabel("forward TFLOPs/step (est)")
    axs[3].bar(labels, mem)
    axs[3].set_title("Train Peak Memory")
    axs[3].set_ylabel("MiB (allocated)")
    axs[4].bar(labels, test_bpb)
    axs[4].set_title("Test bpb")
    axs[4].set_ylabel("bpb (lower is better)")
    for ax in axs:
        ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_infer_bars(infer_rows, out_path):
    labels = [r["model"] for r in infer_rows]
    memory = [r["single_forward_alloc_delta_mib"] for r in infer_rows]
    compute = [r["fwd_flops_per_iter_tflop"] for r in infer_rows]
    speed = [r["tokens_per_s"] for r in infer_rows]
    params_m = [r["param_count"] / 1e6 for r in infer_rows]
    best_val_bpb = [r["best_val_bpb"] for r in infer_rows]
    test_bpb = [r.get("test_bpb", float("nan")) for r in infer_rows]

    fig, axs = plt.subplots(1, 6, figsize=(24, 4.5))
    axs[0].bar(labels, memory)
    axs[0].set_title("Inference Memory (Single Forward Alloc)")
    axs[0].set_ylabel("MiB allocated in one forward")
    axs[1].bar(labels, compute)
    axs[1].set_title("Inference Compute")
    axs[1].set_ylabel("forward TFLOPs/iter (est)")
    axs[2].bar(labels, speed)
    axs[2].set_title("Inference Speed")
    axs[2].set_ylabel("tokens/s")
    axs[3].bar(labels, params_m)
    axs[3].set_title("Parameter Count")
    axs[3].set_ylabel("Millions")
    axs[4].bar(labels, best_val_bpb)
    axs[4].set_title("Best Validation bpb")
    axs[4].set_ylabel("bpb (lower is better)")
    axs[5].bar(labels, test_bpb)
    axs[5].set_title("Test bpb")
    axs[5].set_ylabel("bpb (lower is better)")
    for ax in axs:
        ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_compute_normalized_metrics(infer_rows, out_path):
    labels = [r["model"] for r in infer_rows]

    # Best val bpb normalized by inference compute per batch-1 forward (lower is better).
    bpb_per_infer_tflop = []
    test_bpb_per_infer_tflop = []
    for r in infer_rows:
        fwd_tflop = float(r["fwd_flops_per_iter_tflop"])
        best_bpb = float(r["best_val_bpb"])
        test_bpb = float(r.get("test_bpb", float("nan")))
        if fwd_tflop <= 0:
            bpb_per_infer_tflop.append(float("nan"))
            test_bpb_per_infer_tflop.append(float("nan"))
        else:
            bpb_per_infer_tflop.append(best_bpb / fwd_tflop)
            test_bpb_per_infer_tflop.append(test_bpb / fwd_tflop)

    # Inference allocator churn per forward TFLOP/iter (lower is better)
    mem_per_tflop = []
    # Inference throughput per forward TFLOP/iter (higher is better)
    speed_per_tflop = []
    for r in infer_rows:
        fwd_tflop = float(r["fwd_flops_per_iter_tflop"])
        if fwd_tflop <= 0:
            mem_per_tflop.append(float("nan"))
            speed_per_tflop.append(float("nan"))
        else:
            mem_per_tflop.append(float(r["single_forward_alloc_delta_mib"]) / fwd_tflop)
            speed_per_tflop.append(float(r["tokens_per_s"]) / fwd_tflop)

    fig, axs = plt.subplots(1, 4, figsize=(21, 4.8))
    axs[0].bar(labels, bpb_per_infer_tflop)
    axs[0].set_title("Val bpb / Compute")
    axs[0].set_ylabel("best val bpb per inference TFLOP (batch=1)")

    axs[1].bar(labels, test_bpb_per_infer_tflop)
    axs[1].set_title("Test bpb / Compute")
    axs[1].set_ylabel("test bpb per inference TFLOP (batch=1)")

    axs[2].bar(labels, mem_per_tflop)
    axs[2].set_title("Inference Memory / Compute")
    axs[2].set_ylabel("MiB one-forward alloc per forward TFLOP")

    axs[3].bar(labels, speed_per_tflop)
    axs[3].set_title("Inference Speed / Compute")
    axs[3].set_ylabel("tokens/s per forward TFLOP")

    for ax in axs:
        ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_single_metric_bar(labels, values, title, ylabel, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.6))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_individual_inference_metrics(infer_rows, out_dir):
    labels = [r["model"] for r in infer_rows]
    test_bpb_per_compute = []
    speed_per_compute = []
    for r in infer_rows:
        fwd_tflop = float(r["fwd_flops_per_iter_tflop"])
        if fwd_tflop <= 0:
            test_bpb_per_compute.append(float("nan"))
            speed_per_compute.append(float("nan"))
        else:
            test_bpb_per_compute.append(float(r.get("test_bpb", float("nan"))) / fwd_tflop)
            speed_per_compute.append(float(r["tokens_per_s"]) / fwd_tflop)

    plot_single_metric_bar(
        labels,
        [r.get("test_bpb", float("nan")) for r in infer_rows],
        "Test bpb",
        "bpb (lower is better)",
        os.path.join(out_dir, "metric_test_bpb.png"),
    )
    plot_single_metric_bar(
        labels,
        test_bpb_per_compute,
        "Test bpb / Compute",
        "test bpb per inference TFLOP (batch=1)",
        os.path.join(out_dir, "metric_test_bpb_per_compute.png"),
    )
    plot_single_metric_bar(
        labels,
        speed_per_compute,
        "Inference Speed / Compute",
        "tokens/s per forward TFLOP",
        os.path.join(out_dir, "metric_infer_speed_per_compute.png"),
    )
    plot_single_metric_bar(
        labels,
        [r["single_forward_alloc_delta_mib"] for r in infer_rows],
        "Inference Memory",
        "MiB allocated in one forward",
        os.path.join(out_dir, "metric_infer_memory.png"),
    )
    plot_single_metric_bar(
        labels,
        [r["fwd_flops_per_iter_tflop"] for r in infer_rows],
        "Inference Compute",
        "forward TFLOPs/iter (est)",
        os.path.join(out_dir, "metric_infer_compute.png"),
    )
    plot_single_metric_bar(
        labels,
        [r["tokens_per_s"] for r in infer_rows],
        "Inference Speed",
        "tokens/s",
        os.path.join(out_dir, "metric_infer_speed.png"),
    )
    plot_single_metric_bar(
        labels,
        [r["param_count"] / 1e6 for r in infer_rows],
        "Parameter Count",
        "Millions",
        os.path.join(out_dir, "metric_param_count.png"),
    )


def run_with_auto_batch(kind: str, model, cfg: dict, model_type: str):
    batch = int(cfg["BATCH_SIZE"])
    while batch >= 1:
        try:
            if kind == "train":
                return benchmark_train(model, cfg, model_type, bench_batch=batch)
            return benchmark_inference(model, cfg, model_type, bench_batch=batch)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch //= 2
    raise RuntimeError(f"{model_type} {kind} profiling failed even at batch=1 due to OOM.")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for profiling.")

    ensure_dir(OUT_DIR)
    device = torch.device("cuda")

    runs = {}
    for name in MODEL_ORDER:
        spec = RUN_SPECS[name]
        run_dir = pick_run_dir(spec["candidates"])
        if run_dir is None:
            continue
        cfg_raw = load_config_snapshot(run_dir)
        cfg = fill_defaults(cfg_raw)
        cfg["MODEL_TYPE"] = spec["model_type"]
        runs[name] = {"run_dir": run_dir, "cfg": cfg, "model_type": spec["model_type"]}

    if not runs:
        raise SystemExit("No run directories found for profiling.")

    print("Profiling runs:")
    for k in MODEL_ORDER:
        if k not in runs:
            continue
        r = runs[k]
        c = r["cfg"]
        print(
            f"  {k}: {r['run_dir']} | batch={c['BATCH_SIZE']} seq={c['BLOCK_SIZE']} "
            f"grad_accum={c['GRAD_ACCUM']} amp={c['USE_AMP']} compile={c['USE_COMPILE']}"
        )

    train_rows = []
    infer_rows = []
    for run_name in MODEL_ORDER:
        if run_name not in runs:
            continue
        cfg = runs[run_name]["cfg"]
        model_type = runs[run_name]["model_type"]
        model = build_model(model_type, cfg, device)
        param_count = sum(p.numel() for p in model.parameters())
        if ENABLE_COMPILE and cfg["USE_COMPILE"]:
            model = torch.compile(model)

        train_stats = run_with_auto_batch("train", model, cfg, model_type)
        train_rows.append({"model": run_name, "param_count": param_count, **train_stats})

        infer_stats = benchmark_inference(model, cfg, model_type, bench_batch=INFER_BATCH_SIZE)
        infer_rows.append({"model": run_name, "param_count": param_count, **infer_stats})

        del model
        torch.cuda.empty_cache()

    train_csv = os.path.join(OUT_DIR, "train_profile.csv")
    infer_csv = os.path.join(OUT_DIR, "inference_profile.csv")
    summary_json = os.path.join(OUT_DIR, "profile_summary.json")
    train_plot = os.path.join(OUT_DIR, "train_profile_bars.png")
    infer_plot = os.path.join(OUT_DIR, "inference_profile_bars.png")
    normalized_plot = os.path.join(OUT_DIR, "compute_normalized_metrics.png")
    val_flops_csv = os.path.join(OUT_DIR, "val_vs_flops.csv")
    val_flops_plot = os.path.join(OUT_DIR, "val_vs_flops.png")
    val_step_plot = os.path.join(OUT_DIR, "val_vs_step.png")

    curves = build_val_flops_curves(runs)
    test_scores = load_test_scores(TEST_SCORES_CSV)
    for row in infer_rows:
        c = curves[row["model"]]
        row["best_val_bpb"] = float(np.min(c["y_bpb"])) if len(c["y_bpb"]) > 0 else float("nan")
        t = test_scores.get(row["model"])
        if t is not None:
            row["test_bpb"] = t["test_bpb"]
            row["test_loss_nats"] = t["test_loss_nats"]
            row["test_eval_iters"] = t["eval_iters"]
            row["test_batch_size"] = t["batch_size"]
        else:
            row["test_bpb"] = float("nan")
            row["test_loss_nats"] = float("nan")
            row["test_eval_iters"] = 0
            row["test_batch_size"] = 0
    for row in train_rows:
        t = test_scores.get(row["model"])
        if t is not None:
            row["test_bpb"] = t["test_bpb"]
            row["test_loss_nats"] = t["test_loss_nats"]
            row["test_eval_iters"] = t["eval_iters"]
            row["test_batch_size"] = t["batch_size"]
        else:
            row["test_bpb"] = float("nan")
            row["test_loss_nats"] = float("nan")
            row["test_eval_iters"] = 0
            row["test_batch_size"] = 0

    save_rows_csv(train_csv, list(train_rows[0].keys()), train_rows)
    save_rows_csv(infer_csv, list(infer_rows[0].keys()), infer_rows)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "settings": {
                    "train_warmup_steps": TRAIN_WARMUP_STEPS,
                    "train_bench_steps": TRAIN_BENCH_STEPS,
                    "infer_warmup_iters": INFER_WARMUP_ITERS,
                    "infer_bench_iters": INFER_BENCH_ITERS,
                    "train_flop_multiplier": TRAIN_FLOP_MULTIPLIER,
                },
                "runs": {k: {"run_dir": v["run_dir"], "cfg": v["cfg"]} for k, v in runs.items()},
                "train": train_rows,
                "inference": infer_rows,
            },
            f,
            indent=2,
        )

    plot_train_bars(train_rows, train_plot)
    plot_infer_bars(infer_rows, infer_plot)
    plot_compute_normalized_metrics(infer_rows, normalized_plot)
    plot_individual_inference_metrics(infer_rows, OUT_DIR)
    save_val_flops_csv(val_flops_csv, curves)
    plot_val_vs_flops(curves, val_flops_plot)
    plot_val_vs_step(curves, val_step_plot)

    print("\nTrain profile (simulated, config-matched):")
    for r in train_rows:
        print(
            f"  {r['model']}: step={r['step_ms']:.2f} ms | tok/s={r['tokens_per_s']:.0f} | "
            f"fwd_tf/step={r['fwd_flops_per_step_tflop']:.2f} | mem={r['peak_alloc_mib']:.1f} MiB "
            f"(batch {r['batch']} from cfg {r['config_batch']})"
        )

    print("\nInference profile (config-matched):")
    for r in infer_rows:
        print(
            f"  {r['model']}: iter={r['iter_ms']:.2f} ms | tok/s={r['tokens_per_s']:.0f} | "
            f"fwd_tf/iter={r['fwd_flops_per_iter_tflop']:.2f} | "
            f"mem_one_fwd_alloc={r['single_forward_alloc_delta_mib']:.1f} MiB "
            f"(one_fwd_peak_delta={r['single_forward_peak_alloc_delta_mib']:.1f}) "
            f"(batch {r['batch']} from cfg {r['config_batch']})"
        )

    print("\nFinal val points (from val_vs_flops):")
    for model_name in MODEL_ORDER:
        if model_name not in curves:
            continue
        c = curves[model_name]
        if len(c["x_pflops"]) == 0:
            print(f"  {model_name}: no val rows found")
            continue
        print(
            f"  {model_name}: val_bpb={float(c['y_bpb'][-1]):.6f} "
            f"at {float(c['x_pflops'][-1]):.3f} PF (step {int(c['steps'][-1])})"
        )
    if test_scores:
        print("\nLoaded test scores:")
        for model_name in MODEL_ORDER:
            if model_name not in runs:
                continue
            t = test_scores.get(model_name)
            if t is None:
                print(f"  {model_name}: missing in {TEST_SCORES_CSV}")
                continue
            print(
                f"  {model_name}: test_bpb={t['test_bpb']:.6f}, "
                f"loss={t['test_loss_nats']:.6f}, eval_iters={t['eval_iters']}, batch={t['batch_size']}"
            )
    else:
        print(f"\nNo test score file found at {TEST_SCORES_CSV}. Run scripts/eval_test_bpb.py --model both first.")

    print("\nSaved outputs:")
    print(f"  {train_csv}")
    print(f"  {infer_csv}")
    print(f"  {summary_json}")
    print(f"  {train_plot}")
    print(f"  {infer_plot}")
    print(f"  {normalized_plot}")
    print(f"  {os.path.join(OUT_DIR, 'metric_test_bpb.png')}")
    print(f"  {os.path.join(OUT_DIR, 'metric_test_bpb_per_compute.png')}")
    print(f"  {os.path.join(OUT_DIR, 'metric_infer_speed_per_compute.png')}")
    print(f"  {os.path.join(OUT_DIR, 'metric_infer_memory.png')}")
    print(f"  {os.path.join(OUT_DIR, 'metric_infer_compute.png')}")
    print(f"  {os.path.join(OUT_DIR, 'metric_infer_speed.png')}")
    print(f"  {os.path.join(OUT_DIR, 'metric_param_count.png')}")
    print(f"  {val_flops_csv}")
    print(f"  {val_flops_plot}")
    print(f"  {val_step_plot}")


if __name__ == "__main__":
    main()
