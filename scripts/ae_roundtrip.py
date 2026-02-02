#!/usr/bin/env python3
"""
ae_roundtrip.py

Roundtrip bytes through a trained UNetAutoEncoder: encode -> decode -> argmax.

Key points for your setup:
- Your WINDOW_SIZES product is 4*4*2*2*8*2 = 1024 == BLOCK_SIZE.
- With this Downsample, inputs MUST be length 1024 (or a multiple, chunked).
- The model has never seen a dedicated PAD token; so we avoid "pad tokens".
  Instead, we fill the remainder of the last block with *real enwik8 bytes*
  (default), or a fallback strategy (repeat-last / spaces).

Usage examples:
  uv run python3 ae_roundtrip.py runs/enwik8_ae_bottle2 \
      --device cpu --text 'Welcome to this messaage' --print-latent-shape

  uv run python3 ae_roundtrip.py runs/enwik8_ae_bottle2 \
      --device cpu --text 'Welcome to this messaage' --fill-mode repeat

  uv run python3 ae_roundtrip.py runs/enwik8_ae_bottle2 \
      --device cpu --text-file some.bin --print-bytes

Notes:
- Exact reconstruction is not expected (your AE is lossy).
- For VOCAB_SIZE=250, bytes 250..255 are OOV by default; choose --oov-mode.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from unet_autoencoder import UNetAutoEncoder


# ----------------------------
# Utilities
# ----------------------------

def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _product(xs: List[int]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


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


def _read_text_args(args: argparse.Namespace) -> bytes:
    if args.text is not None:
        return args.text.encode(args.text_encoding, errors=args.text_errors)
    if args.text_file is not None:
        with open(args.text_file, "rb") as f:
            return f.read()
    if args.stdin:
        return sys.stdin.buffer.read()
    raise SystemExit("Provide one of: --text, --text-file, or --stdin")


def _bytes_to_tokens(
    b: bytes,
    vocab_size: int,
    oov_mode: str,
    oov_token: int,
) -> List[int]:
    toks = list(b)
    if all(t < vocab_size for t in toks):
        return toks

    if oov_mode == "error":
        bad = sorted({t for t in toks if t >= vocab_size})
        raise ValueError(
            f"Input contains byte values outside vocab_size={vocab_size}: {bad[:20]}"
            + (" (truncated)" if len(bad) > 20 else "")
            + ". Use --oov-mode replace/mod to proceed."
        )

    if oov_mode == "replace":
        if not (0 <= oov_token < vocab_size):
            raise ValueError(f"--oov-token must be in [0, {vocab_size - 1}] for replace mode.")
        return [t if t < vocab_size else oov_token for t in toks]

    if oov_mode == "mod":
        return [t % vocab_size for t in toks]

    raise ValueError(f"Unknown --oov-mode: {oov_mode}")


def _chunk(tokens: List[int], block_size: int) -> List[List[int]]:
    return [tokens[i : i + block_size] for i in range(0, len(tokens), block_size)]


# ----------------------------
# Model encode/decode
# ----------------------------

@torch.no_grad()
def encode_tokens(model: UNetAutoEncoder, tokens: torch.Tensor) -> torch.Tensor:
    x = model.token_emb(tokens)
    x = model.dropout(x)
    for encoder_block, downsample in zip(model.encoder_blocks, model.downsample_layers):
        x = encoder_block(x, mask=None)
        x = downsample(x)
    if getattr(model, "codebook", None) is not None:
        x = model.codebook(x)
    else:
        x = model.bottleneck(x, mask=None)
    return x


@torch.no_grad()
def decode_latent(model: UNetAutoEncoder, latent: torch.Tensor) -> torch.Tensor:
    x = latent
    for upsample, decoder_block in zip(model.upsample_layers, model.decoder_blocks):
        x = upsample(x)
        x = decoder_block(x, mask=None)
    x = model.norm(x)
    return model.head(x)


# ----------------------------
# Filling / padding strategy (no PAD token)
# ----------------------------

class Enwik8Filler:
    """
    Supplies in-distribution bytes from the enwik8 file for "padding".
    Uses a memmap so it is cheap and doesn't load the whole file.
    """
    def __init__(self, data_path: str, seed: int = 1337):
        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"Missing {data_path}. Download raw enwik8 and place it there."
            )
        self.data = np.memmap(data_path, dtype=np.uint8, mode="r")
        self.n = int(len(self.data))
        self.rng = np.random.RandomState(seed)

    def sample_tokens(self, n: int, vocab_size: int, mode: str) -> List[int]:
        """
        mode:
          - "enwik8": take a contiguous slice from enwik8 (random start)
          - "enwik8_random": take random positions from enwik8 (iid-ish)
        """
        if n <= 0:
            return []
        if self.n <= 0:
            return [0] * n

        if mode == "enwik8":
            start = int(self.rng.randint(0, max(1, self.n - n)))
            arr = self.data[start : start + n]
        elif mode == "enwik8_random":
            idx = self.rng.randint(0, self.n, size=(n,))
            arr = self.data[idx]
        else:
            raise ValueError(f"unknown enwik8 filler mode: {mode}")

        # Map bytes -> tokens in [0, vocab_size-1] safely.
        # This matches the idea of your --oov-mode mod, but only for filler bytes.
        return [int(x) % vocab_size for x in arr.tolist()]


def _fill_to_block(
    blk: List[int],
    block_size: int,
    vocab_size: int,
    fill_mode: str,
    filler: Optional[Enwik8Filler],
) -> Tuple[List[int], int]:
    """
    Return (blk_full, blk_len) where blk_full is exactly block_size long.
    This avoids a dedicated PAD token by filling with plausible bytes/tokens.
    """
    blk_len = len(blk)
    if blk_len == 0:
        return [], 0
    if blk_len > block_size:
        raise ValueError("Internal error: blk longer than block_size (chunking bug).")
    if blk_len == block_size:
        return blk, blk_len

    need = block_size - blk_len

    if fill_mode in ("enwik8", "enwik8_random"):
        if filler is None:
            raise ValueError(f"--fill-mode {fill_mode} requires DATA_PATH to exist.")
        tail = filler.sample_tokens(need, vocab_size=vocab_size, mode=fill_mode)
        return blk + tail, blk_len

    if fill_mode == "repeat":
        last = blk[-1]
        return blk + [last] * need, blk_len

    if fill_mode == "space":
        space = 32 % vocab_size
        return blk + [space] * need, blk_len

    raise ValueError(f"Unknown --fill-mode: {fill_mode}")


# ----------------------------
# Roundtrip
# ----------------------------

@dataclass(frozen=True)
class RoundtripResult:
    input_bytes: bytes
    output_bytes: bytes
    exact: bool
    num_equal: int
    num_total: int
    latent_shape: Tuple[int, int, int]


def roundtrip_text(
    model: UNetAutoEncoder,
    text_bytes: bytes,
    vocab_size: int,
    block_size: int,
    window_sizes: List[int],
    oov_mode: str,
    oov_token: int,
    fill_mode: str,
    data_path: Optional[str],
    seed: int,
    device: torch.device,
) -> RoundtripResult:
    ws_prod = _product([int(x) for x in window_sizes])

    # With your current Downsample stack, you effectively need seq_len divisible by ws_prod.
    # In your config ws_prod==block_size==1024, so we enforce that by filling the last block to block_size.
    if block_size % ws_prod != 0:
        raise ValueError(f"block_size={block_size} must be divisible by product(window_sizes)={ws_prod}.")

    tokens = _bytes_to_tokens(text_bytes, vocab_size=vocab_size, oov_mode=oov_mode, oov_token=oov_token)
    orig_len = len(tokens)
    blocks = _chunk(tokens, block_size=block_size)

    filler = None
    if fill_mode in ("enwik8", "enwik8_random"):
        if data_path is None:
            raise ValueError("--fill-mode enwik8/enwik8_random needs a DATA_PATH from config or --data-path")
        filler = Enwik8Filler(data_path=data_path, seed=seed)

    out: List[int] = []
    latent_shape = (0, 0, 0)

    for blk in blocks:
        blk_full, blk_len = _fill_to_block(
            blk=blk,
            block_size=block_size,
            vocab_size=vocab_size,
            fill_mode=fill_mode,
            filler=filler,
        )
        if blk_len == 0:
            continue

        x = torch.tensor(blk_full, dtype=torch.long, device=device).unsqueeze(0)  # [1, 1024]
        latent = encode_tokens(model, x)
        logits = decode_latent(model, latent)
        y = logits.argmax(dim=-1).squeeze(0).tolist()

        # Only count/return the bytes that correspond to real input for this block.
        out.extend(y[:blk_len])
        latent_shape = tuple(int(v) for v in latent.shape)  # last block

    # Safety crop
    out = out[:orig_len]
    out_bytes = bytes(int(t) for t in out)

    num_equal = sum(1 for a, b in zip(text_bytes, out_bytes) if a == b)
    exact = (text_bytes == out_bytes)
    return RoundtripResult(
        input_bytes=text_bytes,
        output_bytes=out_bytes,
        exact=exact,
        num_equal=num_equal,
        num_total=len(text_bytes),
        latent_shape=latent_shape,
    )


# ----------------------------
# Load model
# ----------------------------

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


# ----------------------------
# Main
# ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Roundtrip bytes through a trained UNetAutoEncoder (encode -> decode).")
    p.add_argument("run_dir", help="Path to run directory (e.g. runs/enwik8_ae_bottle2)")
    p.add_argument("--ckpt", default="ckpt_best.pt", help="Checkpoint filename inside run_dir (default: ckpt_best.pt)")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Text to roundtrip (encoded with --text-encoding)")
    g.add_argument("--text-file", help="File to roundtrip (bytes are used as-is)")
    g.add_argument("--stdin", action="store_true", help="Read bytes from stdin")

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--block-size", type=int, default=None, help="Override BLOCK_SIZE from run config")
    p.add_argument("--data-path", type=str, default=None, help="Override DATA_PATH from run config (for fill-mode enwik8)")
    p.add_argument("--seed", type=int, default=1337, help="Seed for enwik8 filler sampling")

    p.add_argument(
        "--fill-mode",
        choices=["enwik8", "enwik8_random", "repeat", "space"],
        default="enwik8",
        help="How to fill the remainder of the last block to block_size WITHOUT a pad token.",
    )

    p.add_argument(
        "--oov-mode",
        choices=["error", "replace", "mod"],
        default="error",
        help="What to do if input bytes contain values >= VOCAB_SIZE (e.g. VOCAB_SIZE=250 means bytes 250..255 are OOV).",
    )
    p.add_argument("--oov-token", type=int, default=32, help="Replacement token when --oov-mode replace")

    p.add_argument("--text-encoding", default="utf-8", help="Encoding used for --text (default: utf-8)")
    p.add_argument("--text-errors", default="strict", help="Error mode for --text encoding (default: strict)")

    p.add_argument("--print-latent-shape", action="store_true", help="Print the bottleneck latent shape")
    p.add_argument("--print-bytes", action="store_true", help="Print raw bytes (repr) instead of decoding to text")

    args = p.parse_args(argv)

    device = _choose_device(args.device)
    model, cfg = load_model_from_run(args.run_dir, args.ckpt, device=device)

    block_size = int(args.block_size) if args.block_size is not None else int(cfg["BLOCK_SIZE"])
    vocab_size = int(cfg["VOCAB_SIZE"])
    window_sizes = [int(x) for x in cfg["WINDOW_SIZES"]]
    data_path = str(args.data_path) if args.data_path is not None else str(cfg.get("DATA_PATH", "")) or None

    # Helpful one-liner for debugging divisibility / expected latent length
    ws_prod = _product(window_sizes)
    if block_size % ws_prod != 0:
        raise SystemExit(f"Config mismatch: BLOCK_SIZE={block_size} not divisible by ws_prod={ws_prod}.")

    text_bytes = _read_text_args(args)

    res = roundtrip_text(
        model=model,
        text_bytes=text_bytes,
        vocab_size=vocab_size,
        block_size=block_size,
        window_sizes=window_sizes,
        oov_mode=str(args.oov_mode),
        oov_token=int(args.oov_token),
        fill_mode=str(args.fill_mode),
        data_path=data_path,
        seed=int(args.seed),
        device=device,
    )

    if args.print_latent_shape:
        print(f"latent_shape={res.latent_shape}")

    print(f"exact={res.exact} equal_bytes={res.num_equal}/{res.num_total}")

    if args.print_bytes:
        print("input_bytes =", repr(res.input_bytes))
        print("output_bytes=", repr(res.output_bytes))
        return 0

    out_text = res.output_bytes.decode(args.text_encoding, errors="replace")
    print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
