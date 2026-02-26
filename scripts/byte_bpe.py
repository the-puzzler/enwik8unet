#!/usr/bin/env python3
"""
Hugging Face tokenizer pipeline for enwik8.

Supports two workflows:
1) Train a byte-level BPE tokenizer on enwik8 using HF tokenizers.
2) Encode enwik8 into token IDs with either:
   - a local tokenizer from --tokenizer-dir, or
   - a pretrained HF tokenizer from --pretrained-name (e.g., gpt2).

Examples:
  # Train local byte-level BPE tokenizer
  uv run python scripts/byte_bpe.py train \
    --data-path data/enwik8 \
    --tokenizer-dir data/hf_enwik8_bpe \
    --vocab-size 2048 \
    --min-frequency 2

  # Encode with local tokenizer
  uv run python scripts/byte_bpe.py encode \
    --tokenizer-dir data/hf_enwik8_bpe \
    --data-path data/enwik8 \
    --out-path data/enwik8_bpe_tokens.npy

  # Encode with pretrained GPT-2 tokenizer
  uv run python scripts/byte_bpe.py encode \
    --pretrained-name gpt2 \
    --data-path data/enwik8 \
    --out-path data/enwik8_gpt2_tokens.npy
"""

import argparse
import os
from pathlib import Path

import numpy as np
from tokenizers import ByteLevelBPETokenizer, Tokenizer


def _load_text_from_bytes(path: str) -> str:
    # latin-1 gives a 1:1 byte->unicode mapping (0..255), preserving raw bytes.
    return Path(path).read_bytes().decode("latin-1")


def train_cmd(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"Missing data file: {args.data_path}")

    os.makedirs(args.tokenizer_dir, exist_ok=True)

    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[args.data_path],
        vocab_size=int(args.vocab_size),
        min_frequency=int(args.min_frequency),
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    )

    # Save tokenizer artifacts in standard HF tokenizers format.
    tok.save_model(args.tokenizer_dir)
    tok._tokenizer.save(os.path.join(args.tokenizer_dir, "tokenizer.json"))

    print(f"Saved tokenizer artifacts to: {args.tokenizer_dir}")


def _load_tokenizer(args: argparse.Namespace) -> Tokenizer:
    if args.pretrained_name:
        return Tokenizer.from_pretrained(args.pretrained_name)

    if not args.tokenizer_dir:
        raise ValueError("Provide either --tokenizer-dir or --pretrained-name")

    tok_json = os.path.join(args.tokenizer_dir, "tokenizer.json")
    if not os.path.isfile(tok_json):
        raise FileNotFoundError(
            f"Missing {tok_json}. Run train first or pass --pretrained-name."
        )
    return Tokenizer.from_file(tok_json)


def encode_cmd(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"Missing data file: {args.data_path}")

    tokenizer = _load_tokenizer(args)
    text = _load_text_from_bytes(args.data_path)

    enc = tokenizer.encode(text)
    ids = np.asarray(enc.ids, dtype=np.int64)

    max_id = int(ids.max()) if ids.size else 0
    if max_id <= np.iinfo(np.uint16).max:
        out = ids.astype(np.uint16, copy=False)
    else:
        out = ids.astype(np.uint32, copy=False)

    out_path = args.out_path
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, out)

    print(f"Saved token IDs to: {out_path}")
    print(f"input_bytes={len(text):,} output_tokens={len(out):,} ratio={len(out)/max(1,len(text)):.4f}")
    print(f"dtype={out.dtype} max_id={max_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train byte-level BPE tokenizer")
    p_train.add_argument("--data-path", default="data/enwik8", help="raw enwik8 path")
    p_train.add_argument("--tokenizer-dir", required=True, help="output tokenizer directory")
    p_train.add_argument("--vocab-size", type=int, default=2048)
    p_train.add_argument("--min-frequency", type=int, default=2)
    p_train.set_defaults(func=train_cmd)

    p_encode = sub.add_parser("encode", help="encode raw bytes with HF tokenizer")
    p_encode.add_argument("--data-path", default="data/enwik8", help="raw enwik8 path")
    p_encode.add_argument("--out-path", required=True, help="output .npy token IDs")
    p_encode.add_argument("--tokenizer-dir", default=None, help="directory with tokenizer.json")
    p_encode.add_argument("--pretrained-name", default=None, help="HF pretrained tokenizer name, e.g. gpt2")
    p_encode.set_defaults(func=encode_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
