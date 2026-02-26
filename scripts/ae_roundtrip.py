#!/usr/bin/env python3
"""
Roundtrip text/file bytes through a trained UNetAutoEncoder using a local HF tokenizer.

Flow:
input bytes -> latin-1 text -> tokenizer ids -> AE -> argmax ids -> tokenizer decode

This script is intentionally local-tokenizer only (no remote/pretrained lookup).
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from tokenizers import Tokenizer

from unet_autoencoder import UNetAutoEncoder


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_run_config(run_dir: str, ckpt_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if ckpt_payload is not None and isinstance(ckpt_payload.get("config"), dict):
        return dict(ckpt_payload["config"])
    snap = os.path.join(run_dir, "config_snapshot.json")
    if os.path.isfile(snap):
        return _read_json(snap)
    raise FileNotFoundError(f"Could not find config in checkpoint payload, and {snap} does not exist.")


def _choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Requested --device cuda but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"Unknown --device: {device_arg}")


def _read_input_bytes(args: argparse.Namespace) -> bytes:
    if args.text is not None:
        return args.text.encode(args.text_encoding, errors=args.text_errors)
    if args.text_file is not None:
        with open(args.text_file, "rb") as f:
            return f.read()
    if args.stdin:
        return sys.stdin.buffer.read()
    raise SystemExit("Provide one of: --text, --text-file, or --stdin")


def _bytes_to_latin1_text(b: bytes) -> str:
    return b.decode("latin-1")


def _load_local_tokenizer(tokenizer_dir: str) -> Tokenizer:
    tok_json = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.isfile(tok_json):
        raise FileNotFoundError(f"Missing tokenizer file: {tok_json}")
    return Tokenizer.from_file(tok_json)


def load_model_from_run(run_dir: str, ckpt_name: str, device: torch.device) -> Tuple[UNetAutoEncoder, Dict[str, Any]]:
    ckpt_path = ckpt_name if os.path.isabs(ckpt_name) else os.path.join(run_dir, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = _load_run_config(run_dir, ckpt)

    model = UNetAutoEncoder(
        vocab_size=int(cfg["VOCAB_SIZE"]),
        dim=int(cfg["DIM"]),
        num_heads=int(cfg["NUM_HEADS"]),
        mlp_ratio=int(cfg["MLP_RATIO"]),
        dropout=float(cfg["DROPOUT"]),
        window_sizes=[int(x) for x in cfg["WINDOW_SIZES"]],
        num_codes=int(cfg.get("NUM_CODES", 0)),
    ).to(device)

    sd = ckpt.get("model", ckpt)
    if not isinstance(sd, dict):
        raise TypeError(f"Expected checkpoint state_dict dict, got: {type(sd)}")
    sd = _strip_known_prefixes(sd)

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg


@dataclass(frozen=True)
class RoundtripResult:
    input_ids: List[int]
    output_ids: List[int]
    output_ids_trimmed: List[int]
    latent_shapes: List[Tuple[int, int, int]]


@torch.no_grad()
def _run_chunk(
    model: UNetAutoEncoder,
    chunk_ids: List[int],
    block_size: int,
    pad_token_id: int,
    device: torch.device,
) -> Tuple[List[int], Tuple[int, int, int]]:
    l = len(chunk_ids)
    if l <= 0:
        return [], (0, 0, 0)
    if l > block_size:
        raise ValueError(f"chunk len {l} exceeds block_size {block_size}")

    if l == block_size:
        x = torch.tensor(chunk_ids, dtype=torch.long, device=device).unsqueeze(0)
        valid_mask = torch.ones((1, block_size), dtype=torch.bool, device=device)
    else:
        x = torch.full((1, block_size), int(pad_token_id), dtype=torch.long, device=device)
        x[0, :l] = torch.tensor(chunk_ids, dtype=torch.long, device=device)
        valid_mask = torch.zeros((1, block_size), dtype=torch.bool, device=device)
        valid_mask[:, :l] = True

    attn_mask = valid_mask[:, None, None, :]
    hooked: List[torch.Tensor] = []

    def _hook(_m, _i, out):
        hooked.append(out.detach())

    h = model.bottleneck.register_forward_hook(_hook)
    try:
        logits = model(x, mask=attn_mask)
    finally:
        h.remove()

    y = logits.argmax(dim=-1).squeeze(0).tolist()[:l]
    latent_shape = tuple(int(v) for v in (hooked[-1].shape if hooked else torch.zeros(0).shape))
    return y, latent_shape


@torch.no_grad()
def roundtrip_ids(
    model: UNetAutoEncoder,
    input_ids: List[int],
    block_size: int,
    pad_token_id: int,
    eos_token_id: int,
    trim_at_eos: bool,
    device: torch.device,
) -> RoundtripResult:
    out: List[int] = []
    latent_shapes: List[Tuple[int, int, int]] = []

    for i in range(0, len(input_ids), block_size):
        chunk = input_ids[i : i + block_size]
        y, latent_shape = _run_chunk(
            model=model,
            chunk_ids=chunk,
            block_size=block_size,
            pad_token_id=pad_token_id,
            device=device,
        )
        out.extend(y)
        latent_shapes.append(latent_shape)

    out_trimmed = list(out)
    if trim_at_eos:
        try:
            eos_pos = out_trimmed.index(int(eos_token_id))
            out_trimmed = out_trimmed[:eos_pos]
        except ValueError:
            pass

    return RoundtripResult(
        input_ids=input_ids,
        output_ids=out,
        output_ids_trimmed=out_trimmed,
        latent_shapes=latent_shapes,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Roundtrip through AE with local HF tokenizer.")
    p.add_argument("run_dir", help="Path to run directory (e.g. runs/enwik8_ae_bpe)")
    p.add_argument("--tokenizer-dir", required=True, help="Directory containing tokenizer.json")
    p.add_argument("--ckpt", default="ckpt_best.pt", help="Checkpoint filename inside run_dir")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Text to roundtrip")
    g.add_argument("--text-file", help="File to roundtrip (bytes as-is)")
    g.add_argument("--stdin", action="store_true", help="Read bytes from stdin")

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--block-size", type=int, default=None, help="Override BLOCK_SIZE from run config")
    p.add_argument("--pad-token-id", type=int, default=None, help="Override PAD_TOKEN_ID from run config")
    p.add_argument("--eos-token-id", type=int, default=None, help="Override EOS_TOKEN_ID from run config")
    p.add_argument("--no-trim-eos", action="store_true", help="Do not trim output at first EOS")

    p.add_argument("--text-encoding", default="utf-8")
    p.add_argument("--text-errors", default="strict")
    p.add_argument("--print-token-stats", action="store_true")
    p.add_argument("--print-latent-shapes", action="store_true")

    args = p.parse_args(argv)

    device = _choose_device(args.device)
    tokenizer = _load_local_tokenizer(args.tokenizer_dir)
    model, cfg = load_model_from_run(args.run_dir, args.ckpt, device=device)

    block_size = int(args.block_size) if args.block_size is not None else int(cfg["BLOCK_SIZE"])
    pad_token_id = int(args.pad_token_id) if args.pad_token_id is not None else int(cfg.get("PAD_TOKEN_ID", 0))
    eos_token_id = int(args.eos_token_id) if args.eos_token_id is not None else int(cfg.get("EOS_TOKEN_ID", 2))
    vocab_size = int(cfg["VOCAB_SIZE"])

    raw_bytes = _read_input_bytes(args)
    text_for_tokenizer = _bytes_to_latin1_text(raw_bytes)
    input_ids = tokenizer.encode(text_for_tokenizer).ids

    if not input_ids:
        print("No tokens produced by tokenizer for the given input.")
        return 0

    bad = [t for t in input_ids if t < 0 or t >= vocab_size]
    if bad:
        raise SystemExit(
            f"Tokenizer/model vocab mismatch: got token IDs outside [0, {vocab_size-1}], "
            f"example={bad[:10]}"
        )

    res = roundtrip_ids(
        model=model,
        input_ids=input_ids,
        block_size=block_size,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        trim_at_eos=not bool(args.no_trim_eos),
        device=device,
    )

    if args.print_token_stats:
        n_ref = min(len(res.input_ids), len(res.output_ids_trimmed))
        eq = sum(1 for a, b in zip(res.input_ids, res.output_ids_trimmed) if a == b)
        print(
            f"equal_tokens={eq}/{n_ref} ({(100.0*eq/max(1,n_ref)):.2f}%) "
            f"| in_len={len(res.input_ids)} out_len={len(res.output_ids_trimmed)}"
        )

    if args.print_latent_shapes:
        uniq = sorted(set(res.latent_shapes))
        print(f"latent_shapes={uniq}")

    out_text = tokenizer.decode(res.output_ids_trimmed)
    print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
