"""
Evaluate test-set bpb for a trained model checkpoint.

Usage:
  python eval_test_bpb.py --work-dir runs/enwik8_unet_done --ckpt ckpt_best.pt
"""

import argparse
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
    parser.add_argument("--work-dir", default=C.WORK_DIR, help="run directory with checkpoints")
    parser.add_argument("--ckpt", default="ckpt_best.pt", help="checkpoint filename or path")
    parser.add_argument("--eval-iters", type=int, default=C.EVAL_ITERS, help="number of eval batches")
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE, help="batch size for eval")
    parser.add_argument("--model", choices=["unet", "baseline"], default="unet", help="model architecture")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(args.work_dir, args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    cfg = load_config_snapshot(args.work_dir)
    def cfg_get(key, default):
        return cfg.get(key, default)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_enwik8_memmap(cfg_get("DATA_PATH", C.DATA_PATH))
    _, _, test_arr = make_splits(data, train_frac=0.9, val_frac=0.05)

    if args.model == "unet":
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

    mask = get_causal_mask(cfg_get("BLOCK_SIZE", C.BLOCK_SIZE), device)
    loss, bpb = estimate_loss_and_bpb(
        model,
        test_arr,
        cfg_get("BLOCK_SIZE", C.BLOCK_SIZE),
        args.batch_size,
        args.eval_iters,
        device,
        mask,
        cfg_get("USE_AMP", C.USE_AMP),
    )

    print(f"Test loss (nats): {loss:.6f}")
    print(f"Test bpb: {bpb:.6f}")


if __name__ == "__main__":
    main()
