"""
Plot validation bpb versus cumulative training compute for UNet and baseline runs.

This is a post-hoc, compute-aligned comparison using:
  cumulative_compute = step * grad_accum * batch_size * forward_flops_per_sample

It reads each run's `metrics.csv` and `config_snapshot.json`.
"""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import config as C

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
MODEL_ORDER = ("unet", "baseline", "baseline_small")
MODEL_COLORS = {"unet": "#1f77b4", "baseline": "#ff7f0e", "baseline_small": "#2ca02c"}


def cfg_get(cfg, key, default):
    return cfg.get(key, default)


def load_config_snapshot(work_dir: str):
    path = os.path.join(work_dir, "config_snapshot.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_run_dir(candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def fill_defaults(cfg):
    return {
        "MODEL_TYPE": cfg_get(cfg, "MODEL_TYPE", C.MODEL_TYPE),
        "BATCH_SIZE": int(cfg_get(cfg, "BATCH_SIZE", C.BATCH_SIZE)),
        "GRAD_ACCUM": int(cfg_get(cfg, "GRAD_ACCUM", C.GRAD_ACCUM)),
        "BLOCK_SIZE": int(cfg_get(cfg, "BLOCK_SIZE", C.BLOCK_SIZE)),
        "DIM": int(cfg_get(cfg, "DIM", C.DIM)),
        "NUM_HEADS": int(cfg_get(cfg, "NUM_HEADS", C.NUM_HEADS)),
        "MLP_RATIO": float(cfg_get(cfg, "MLP_RATIO", C.MLP_RATIO)),
        "NUM_LAYERS": int(cfg_get(cfg, "NUM_LAYERS", getattr(C, "NUM_LAYERS", 12))),
        "WINDOW_SIZES": list(cfg_get(cfg, "WINDOW_SIZES", C.WINDOW_SIZES)),
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


def estimate_baseline_forward_flops_per_sample(cfg):
    per_block = block_flops(1, cfg["BLOCK_SIZE"], cfg["DIM"], cfg["NUM_HEADS"], cfg["MLP_RATIO"])
    return per_block * cfg["NUM_LAYERS"]


def estimate_unet_forward_flops_per_sample(cfg):
    total = 0
    seq = cfg["BLOCK_SIZE"]
    for w in cfg["WINDOW_SIZES"]:
        total += block_flops(1, seq, cfg["DIM"], cfg["NUM_HEADS"], cfg["MLP_RATIO"])
        seq = seq // int(w)
    total += block_flops(1, seq, cfg["DIM"], cfg["NUM_HEADS"], cfg["MLP_RATIO"])
    for w in reversed(cfg["WINDOW_SIZES"]):
        seq = seq * int(w)
        total += block_flops(1, seq, cfg["DIM"], cfg["NUM_HEADS"], cfg["MLP_RATIO"])
    return total


def load_val_metrics(metrics_path):
    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            if row[2] != "val":
                continue
            step = int(row[1])
            bpb = float(row[4])
            rows.append((step, bpb))
    rows.sort(key=lambda x: x[0])
    return rows


def build_compute_curve(work_dir, force_model_type=None):
    cfg_raw = load_config_snapshot(work_dir)
    cfg = fill_defaults(cfg_raw)
    if force_model_type is not None:
        model_type = force_model_type
    else:
        model_type = str(cfg["MODEL_TYPE"]).lower()

    if model_type == "unet":
        flops_per_sample = estimate_unet_forward_flops_per_sample(cfg)
    elif model_type == "baseline":
        flops_per_sample = estimate_baseline_forward_flops_per_sample(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    metrics_path = os.path.join(work_dir, "metrics.csv")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    val_rows = load_val_metrics(metrics_path)
    if not val_rows:
        raise ValueError(f"No val rows found in {metrics_path}")

    steps = []
    compute = []
    bpb = []
    for step, b in val_rows:
        c = step * cfg["GRAD_ACCUM"] * cfg["BATCH_SIZE"] * flops_per_sample
        steps.append(int(step))
        compute.append(float(c))
        bpb.append(float(b))

    return {
        "steps": np.array(steps, dtype=np.int64),
        "compute": np.array(compute, dtype=np.float64),
        "bpb": np.array(bpb, dtype=np.float64),
        "flops_per_sample": float(flops_per_sample),
        "batch_size": cfg["BATCH_SIZE"],
        "grad_accum": cfg["GRAD_ACCUM"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet-dir", default=None, help="UNet run directory (optional override)")
    parser.add_argument("--baseline-dir", default=None, help="Baseline run directory (optional override)")
    parser.add_argument("--baseline-small-dir", default=None, help="Baseline-small run directory (optional override)")
    parser.add_argument("--out", default="compute_aligned_bpb.png", help="output plot path")
    parser.add_argument("--out-csv", default="compute_aligned_bpb.csv", help="output aligned CSV path")
    args = parser.parse_args()

    overrides = {
        "unet": args.unet_dir,
        "baseline": args.baseline_dir,
        "baseline_small": args.baseline_small_dir,
    }

    curves = {}
    for name in MODEL_ORDER:
        work_dir = overrides[name] if overrides[name] is not None else pick_run_dir(RUN_SPECS[name]["candidates"])
        if work_dir is None:
            print(f"Skipping {name}: no run directory found")
            continue
        curves[name] = build_compute_curve(work_dir, force_model_type=RUN_SPECS[name]["model_type"])

    if "unet" not in curves:
        raise SystemExit("UNet run is required for compute-aligned comparison.")

    grid_parts = [curves["unet"]["compute"]]
    for name in MODEL_ORDER:
        if name == "unet" or name not in curves:
            continue
        grid_parts.append(curves[name]["compute"])
    grid = np.unique(np.concatenate(grid_parts))

    interps = {}
    for name, c in curves.items():
        x = c["compute"]
        y = c["bpb"]
        yi = np.interp(grid, x, y)
        yi[(grid < x.min()) | (grid > x.max())] = np.nan
        interps[name] = yi

    # Save aligned table.
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["compute_pflops", "unet_bpb"]
        for name in MODEL_ORDER:
            if name == "unet" or name not in curves:
                continue
            header += [f"{name}_bpb", f"{name}_minus_unet"]
        w.writerow(header)

        for i, c in enumerate(grid):
            row = [f"{c/1e15:.8f}"]
            u = interps["unet"][i]
            row.append("" if np.isnan(u) else f"{u:.8f}")
            for name in MODEL_ORDER:
                if name == "unet" or name not in curves:
                    continue
                v = interps[name][i]
                v_str = "" if np.isnan(v) else f"{v:.8f}"
                if np.isnan(u) or np.isnan(v):
                    d_str = ""
                else:
                    d_str = f"{(v-u):.8f}"
                row += [v_str, d_str]
            w.writerow(row)

    # Plot raw points and aligned curves.
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for name in MODEL_ORDER:
        if name not in curves:
            continue
        color = MODEL_COLORS[name]
        c = curves[name]
        ax.plot(c["compute"] / 1e15, c["bpb"], "o", ms=3, alpha=0.35, label=f"{name} val (raw)", color=color)
        ax.plot(grid / 1e15, interps[name], "-", lw=2, label=f"{name} (interp)", color=color)
    ax.set_xlabel("Cumulative forward compute (PFLOPs, proxy)")
    ax.set_ylabel("Validation bpb")
    ax.set_title("Compute-Aligned Validation bpb")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    print("Saved:")
    print(f"  plot: {args.out}")
    print(f"  csv:  {args.out_csv}")
    print("Run summary:")
    for name in MODEL_ORDER:
        if name not in curves:
            continue
        c = curves[name]
        print(
            f"  {name}: "
            f"flops/sample={c['flops_per_sample']/1e9:.2f} GF, "
            f"batch={c['batch_size']}, grad_accum={c['grad_accum']}"
        )


if __name__ == "__main__":
    main()
