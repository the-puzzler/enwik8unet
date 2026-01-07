"""
Estimate theoretical FLOPs per forward pass for baseline and UNet models.

This is a rough matmul-based estimate (QKV, attention, MLP) and ignores
norms, dropout, and elementwise ops.
"""

import argparse
import json
import os

import config as C


def load_config_snapshot(work_dir: str):
    path = os.path.join(work_dir, "config_snapshot.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cfg_get(cfg, key, default):
    return cfg.get(key, default)


def attn_flops(batch, seq_len, dim, num_heads):
    d_head = dim // num_heads
    # QKV projections + output projection
    qkv = 6 * batch * seq_len * dim * dim
    out = 2 * batch * seq_len * dim * dim
    # Attention scores and weighted sum
    attn = 4 * batch * num_heads * seq_len * seq_len * d_head
    return qkv + out + attn


def mlp_flops(batch, seq_len, dim, mlp_ratio):
    hidden = int(dim * mlp_ratio)
    # SwiGLU uses two input projections and one output projection
    return 6 * batch * seq_len * dim * hidden


def block_flops(batch, seq_len, dim, num_heads, mlp_ratio):
    return attn_flops(batch, seq_len, dim, num_heads) + mlp_flops(batch, seq_len, dim, mlp_ratio)


def kv_cache_bytes(batch, seq_len, dim, bytes_per_elem):
    # Keys + values per token
    return 2 * batch * seq_len * dim * bytes_per_elem


def estimate_baseline(cfg):
    batch = cfg["BATCH_SIZE"]
    seq_len = cfg["BLOCK_SIZE"]
    dim = cfg["DIM"]
    num_heads = cfg["NUM_HEADS"]
    mlp_ratio = cfg["MLP_RATIO"]
    num_layers = cfg["NUM_LAYERS"]
    per_block = block_flops(batch, seq_len, dim, num_heads, mlp_ratio)
    total = per_block * num_layers
    bytes_per_elem = cfg["BYTES_PER_ELEM"]
    per_block_kv = kv_cache_bytes(batch, seq_len, dim, bytes_per_elem)
    return total, {
        "per_block": per_block,
        "num_layers": num_layers,
        "seq_len": seq_len,
        "per_block_kv": per_block_kv,
        "sum_kv": per_block_kv * num_layers,
    }


def estimate_unet(cfg):
    batch = cfg["BATCH_SIZE"]
    seq_len = cfg["BLOCK_SIZE"]
    dim = cfg["DIM"]
    num_heads = cfg["NUM_HEADS"]
    mlp_ratio = cfg["MLP_RATIO"]
    window_sizes = cfg["WINDOW_SIZES"]
    bytes_per_elem = cfg["BYTES_PER_ELEM"]

    total = 0
    max_kv = 0
    sum_kv = 0
    seq = seq_len
    for w in window_sizes:
        total += block_flops(batch, seq, dim, num_heads, mlp_ratio)
        kv = kv_cache_bytes(batch, seq, dim, bytes_per_elem)
        max_kv = max(max_kv, kv)
        sum_kv += kv
        seq = seq // w
    # bottleneck
    total += block_flops(batch, seq, dim, num_heads, mlp_ratio)
    kv = kv_cache_bytes(batch, seq, dim, bytes_per_elem)
    max_kv = max(max_kv, kv)
    sum_kv += kv
    # decoder
    for w in reversed(window_sizes):
        seq = seq * w
        total += block_flops(batch, seq, dim, num_heads, mlp_ratio)
        kv = kv_cache_bytes(batch, seq, dim, bytes_per_elem)
        max_kv = max(max_kv, kv)
        sum_kv += kv

    return total, {
        "num_blocks": 2 * len(window_sizes) + 1,
        "seq_len": seq_len,
        "max_kv": max_kv,
        "sum_kv": sum_kv,
    }


def fmt_gflops(x):
    return x / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir-unet", default="runs/enwik8_unet_done", help="UNet run directory")
    parser.add_argument("--work-dir-baseline", default="runs/enwik8_baseline", help="Baseline run directory")
    args = parser.parse_args()

    unet_cfg = load_config_snapshot(args.work_dir_unet)
    baseline_cfg = load_config_snapshot(args.work_dir_baseline)

    # Fall back to config.py defaults when snapshot missing.
    def fill_defaults(cfg):
        use_amp = cfg_get(cfg, "USE_AMP", C.USE_AMP)
        bytes_per_elem = 2 if use_amp else 4
        return {
            "BATCH_SIZE": cfg_get(cfg, "BATCH_SIZE", C.BATCH_SIZE),
            "BLOCK_SIZE": cfg_get(cfg, "BLOCK_SIZE", C.BLOCK_SIZE),
            "DIM": cfg_get(cfg, "DIM", C.DIM),
            "NUM_HEADS": cfg_get(cfg, "NUM_HEADS", C.NUM_HEADS),
            "MLP_RATIO": cfg_get(cfg, "MLP_RATIO", C.MLP_RATIO),
            "NUM_LAYERS": cfg_get(cfg, "NUM_LAYERS", getattr(C, "NUM_LAYERS", 12)),
            "WINDOW_SIZES": cfg_get(cfg, "WINDOW_SIZES", C.WINDOW_SIZES),
            "BYTES_PER_ELEM": bytes_per_elem,
        }

    unet_cfg = fill_defaults(unet_cfg)
    baseline_cfg = fill_defaults(baseline_cfg)
    unet_cfg["BATCH_SIZE"] = 1
    baseline_cfg["BATCH_SIZE"] = 1

    unet_total, unet_meta = estimate_unet(unet_cfg)
    base_total, base_meta = estimate_baseline(baseline_cfg)

    print("Approx FLOPs per forward pass (matmul-based, excludes norms/elemwise):")
    print(f"UNet:     {fmt_gflops(unet_total):.2f} GFLOPs (blocks={unet_meta['num_blocks']}, seq={unet_meta['seq_len']}, batch=1)")
    print(f"Baseline: {fmt_gflops(base_total):.2f} GFLOPs (layers={base_meta['num_layers']}, seq={base_meta['seq_len']}, batch=1)")
    if base_total > 0:
        print(f"Baseline / UNet ratio: {base_total / unet_total:.2f}x")

    print("Approx KV-cache memory for inference (keys+values):")
    print(f"UNet peak block: {unet_meta['max_kv'] / (1024**2):.2f} MiB")
    print(f"UNet sum blocks: {unet_meta['sum_kv'] / (1024**2):.2f} MiB")
    print(f"Base per block:  {base_meta['per_block_kv'] / (1024**2):.2f} MiB")
    print(f"Base sum blocks: {base_meta['sum_kv'] / (1024**2):.2f} MiB")


if __name__ == "__main__":
    main()
