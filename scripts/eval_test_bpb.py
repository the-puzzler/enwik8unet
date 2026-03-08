"""
Evaluate test-set bpb for a trained model checkpoint.

Usage:
  python eval_test_bpb.py --work-dir runs/enwik8_unet_done --ckpt ckpt_best.pt
"""

import argparse
import csv
import json
import os

import torch

import config as C
from train_enwik8 import (
    load_enwik8_memmap,
    make_splits,
    get_causal_mask,
    estimate_loss_and_bpb,
    load_checkpoint,
)
from unet_transformer import UNetTransformer
from baseline_transformer import BaselineTransformer


def _strip_orig_mod(sd: dict):
    """Handle state_dicts saved from torch.compile (keys prefixed with _orig_mod.)."""
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


def load_config_snapshot(work_dir: str):
    path = os.path.join(work_dir, "config_snapshot.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default=C.WORK_DIR, help="run directory with checkpoints (single-model mode)")
    parser.add_argument("--ckpt", default="ckpt_best.pt", help="checkpoint filename or path (single-model mode)")
    parser.add_argument("--work-dir-unet", default="runs/enwik8_unet", help="UNet run directory (both-mode)")
    parser.add_argument("--work-dir-baseline", default="runs/enwik8_baseline", help="Baseline run directory (both-mode)")
    parser.add_argument("--work-dir-baseline-small", default="runs/enwik8_baseline_small", help="Baseline-small run directory (both-mode)")
    parser.add_argument("--ckpt-unet", default="ckpt_best.pt", help="UNet checkpoint filename/path (both-mode)")
    parser.add_argument("--ckpt-baseline", default="ckpt_best.pt", help="Baseline checkpoint filename/path (both-mode)")
    parser.add_argument("--ckpt-baseline-small", default="ckpt_best.pt", help="Baseline-small checkpoint filename/path (both-mode)")
    parser.add_argument("--eval-iters", type=int, default=C.EVAL_ITERS, help="number of eval batches")
    parser.add_argument("--batch-size", type=int, default=None, help="override eval batch size (default: from config snapshot)")
    parser.add_argument("--out-csv", default="profiling/test_scores.csv", help="output CSV path for test results")
    parser.add_argument("--out-json", default="profiling/test_scores.json", help="output JSON path for test results")
    parser.add_argument(
        "--model",
        choices=["unet", "baseline", "baseline_small", "both"],
        default="both",
        help="model architecture",
    )
    args = parser.parse_args()

    def resolve_ckpt(work_dir: str, ckpt_name_or_path: str):
        ckpt_path = ckpt_name_or_path
        if not os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(work_dir, ckpt_name_or_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        return ckpt_path

    results = []

    def eval_one(model_name: str, work_dir: str, ckpt_name_or_path: str):
        cfg = load_config_snapshot(work_dir)

        def cfg_get(key, default):
            return cfg.get(key, default)

        ckpt_path = resolve_ckpt(work_dir, ckpt_name_or_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = load_enwik8_memmap(cfg_get("DATA_PATH", C.DATA_PATH))
        _, _, test_arr = make_splits(data, train_frac=0.9, val_frac=0.05)

        model_type = "baseline" if model_name == "baseline_small" else model_name

        if model_type == "unet":
            model = UNetTransformer(
                vocab_size=cfg_get("VOCAB_SIZE", C.VOCAB_SIZE),
                dim=cfg_get("DIM", C.DIM),
                num_heads=cfg_get("NUM_HEADS", C.NUM_HEADS),
                mlp_ratio=cfg_get("MLP_RATIO", C.MLP_RATIO),
                dropout=cfg_get("DROPOUT", C.DROPOUT),
                window_sizes=cfg_get("WINDOW_SIZES", C.WINDOW_SIZES),
            ).to(device)
        else:
            model = BaselineTransformer(
                vocab_size=cfg_get("VOCAB_SIZE", C.VOCAB_SIZE),
                dim=cfg_get("DIM", C.DIM),
                num_heads=cfg_get("NUM_HEADS", C.NUM_HEADS),
                mlp_ratio=cfg_get("MLP_RATIO", C.MLP_RATIO),
                dropout=cfg_get("DROPOUT", C.DROPOUT),
                num_layers=cfg_get("NUM_LAYERS", 12),
            ).to(device)

        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        state = _strip_orig_mod(ckpt["model"])
        model.load_state_dict(state)

        block_size = cfg_get("BLOCK_SIZE", C.BLOCK_SIZE)
        batch_size = args.batch_size if args.batch_size is not None else cfg_get("BATCH_SIZE", C.BATCH_SIZE)
        mask = get_causal_mask(block_size, device)
        loss, bpb = estimate_loss_and_bpb(
            model,
            test_arr,
            block_size,
            batch_size,
            args.eval_iters,
            device,
            mask,
            cfg_get("USE_AMP", C.USE_AMP),
        )
        print(f"[{model_name}] work_dir={work_dir} ckpt={os.path.basename(ckpt_path)}")
        print(f"[{model_name}] Test loss (nats): {loss:.6f}")
        print(f"[{model_name}] Test bpb: {bpb:.6f}")
        results.append(
            {
                "model": model_name,
                "work_dir": work_dir,
                "ckpt": os.path.basename(ckpt_path),
                "eval_iters": int(args.eval_iters),
                "batch_size": int(batch_size),
                "test_loss_nats": float(loss),
                "test_bpb": float(bpb),
            }
        )

    if args.model == "both":
        eval_one("unet", args.work_dir_unet, args.ckpt_unet)
        eval_one("baseline", args.work_dir_baseline, args.ckpt_baseline)
        eval_one("baseline_small", args.work_dir_baseline_small, args.ckpt_baseline_small)
    else:
        eval_one(args.model, args.work_dir, args.ckpt)

    if results:
        out_csv_dir = os.path.dirname(args.out_csv)
        if out_csv_dir:
            os.makedirs(out_csv_dir, exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["model", "work_dir", "ckpt", "eval_iters", "batch_size", "test_loss_nats", "test_bpb"],
            )
            w.writeheader()
            for row in results:
                w.writerow(row)

        out_json_dir = os.path.dirname(args.out_json)
        if out_json_dir:
            os.makedirs(out_json_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"Saved test results CSV: {args.out_csv}")
        print(f"Saved test results JSON: {args.out_json}")


if __name__ == "__main__":
    main()
