#!/usr/bin/env bash
set -euo pipefail

python eval_test_bpb.py --work-dir runs/enwik8_unet_done --ckpt ckpt_best.pt --model unet
python eval_test_bpb.py --work-dir runs/enwik8_baseline --ckpt ckpt_best.pt --model baseline
