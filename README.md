## enwik8 UNet Language Model (Simple Hierarchical Transformer)

This repo explores a simple UNet-style autoregressive language model on **enwik8** and compares it against a size-matched dense Transformer baseline.

The goal was not to build a complex tokenizer-inside-the-model system, but to test whether a **fixed, simple hierarchical compression/expansion pattern** can keep quality close to a standard Transformer while reducing compute and memory.

In this setup, we achieve about **2.7x lower single-forward memory allocation with no meaningful performance loss** in the compute-matched comparison.

For broader results and deeper analysis on this direction, see the main blog post: https://the-puzzler.github.io/share/tokens-who-needs-them.html

## Contents

- [Motivation](#motivation)
- [Model Design](#model-design)
- [Curves](#curves)
- [Results](#results)
- [Quickstart](#quickstart)

<p align="center">
  <img src="umap.png" alt="UNet bottleneck UMAP" width="48%" />
  <img src="word_cloud.png" alt="UNet bottleneck word clouds" width="48%" />
</p>
<p align="center"><em>
Left: 2D UMAP of bottleneck embeddings from random test spans. Right: cluster-enriched word clouds from those same bottleneck clusters.
</em></p>

## Motivation

The core idea is to introduce a contracting-expanding pathway in a causal language model:

- Downsample sequence representations in stages.
- Process a compressed bottleneck.
- Upsample back to full length for next-token prediction.

This gives a UNet-style compute pattern where much of the expensive processing happens at shorter sequence lengths.

## Model Design

### Baseline Transformer

- Byte-level vocab (`256`)
- `dim=512`, `heads=8`, `SwiGLU`, `RMSNorm`, `RoPE`
- 10 layers

### Simple UNet Transformer

- Same core block choices as baseline (`dim=512`, `heads=8`, `SwiGLU`, `RMSNorm`, `RoPE`)
- Hierarchical sequence scales with window sizes: `[4, 4, 2, 2]`
- Fixed embedding dimension through encoder/decoder
- Causal downsampling via first-token window representative
- Learned upsampling via projection + reshape
- UNet skip connections across matching scales

Architecture schematic:

![UNet Architecture](arch.png)

### Practical comparison (approx)

- Params: baseline ~42M, UNet ~43M
- FLOPs per forward: baseline ~107 GFLOPs, UNet ~27 GFLOPs
- KV-cache memory (inference): baseline ~20 MB, UNet ~5 MB

## Training Setup

- Dataset: `enwik8` (byte-level)
- Hardware: NVIDIA A10 24GB
- Baseline run: 100k steps, batch 32, grad accum 8
- UNet run: 100k steps, batch 128, grad accum 2
- Baseline Small run: 100k steps, batch 32, grad accum 6

## Curves

![Validation bpb vs FLOPs](profiling/val_vs_flops.png)

## Results

![Inference Profile Bars](profiling/inference_profile_bars.png)

- Final test bpb (baseline): **1.1679**
- Final test bpb (baseline_small): **1.1768**
- Final test bpb (UNet): **1.2090**

So the main result is:

- **Quality:** near-parity with dense Transformer at this scale.
- **Efficiency:** substantial memory reduction, with about **2.7x lower single-forward memory allocation** than the compute-matched baseline (`baseline_small`).

## Bottleneck Embedding Analysis

The UNet bottleneck behaved like a meaningful latent space in downstream analysis:

- Random test spans were embedded at the bottleneck.
- UMAP + HDBSCAN revealed coherent topical clusters.
- Cluster-level term enrichment and word clouds showed interpretable themes (for example, technical/security-heavy spans, historical conflict clusters, citation/link-heavy text neighborhoods, and generic prose background clusters).

The bottleneck embeddings capture real semantic information with clear themes per cluster. For example, cluster 25 is clearly on religious topics. Meanwhile, clusters 22 and 21 are mathematical and 9 is computing.

This indicates the compressed representation is not only computationally useful, but also semantically structured.

UMAP projection of bottleneck embeddings:

![UNet Bottleneck UMAP](umap.png)

Cluster-enriched word cloud view:

![UNet Bottleneck Word Cloud](word_cloud.png)

## Takeaway

A simple hierarchical UNet-style Transformer can stay competitive with a dense baseline on enwik8 while being much cheaper in estimated compute/memory. At this scale, it does not beat the baseline on bpb, but it demonstrates a strong quality-efficiency tradeoff and useful latent structure in the bottleneck.

## Quickstart

### 1) Install dependencies

```bash
uv sync
```

### 2) Download enwik8 data

This project expects the raw file at `data/enwik8` (no extension).

```bash
mkdir -p data
cd data
wget http://mattmahoney.net/dc/enwik8.zip
unzip -o enwik8.zip
cd ..
```

### 3) Run training

Model selection is controlled by `MODEL_TYPE` in [config.py](/mnt/mnemo9/mpelus/experiments/enwik8unet/config.py):

- `MODEL_TYPE = "baseline"`
- `MODEL_TYPE = "unet"`

Also set `WORK_DIR` to the output folder you want for that run.

Start training:

```bash
uv run python train_enwik8.py
```

### 4) Evaluate a trained checkpoint

Baseline:

```bash
uv run python scripts/eval_test_bpb.py \
  --work-dir runs/enwik8_baseline \
  --ckpt ckpt_best.pt \
  --model baseline
```

UNet:

```bash
uv run python scripts/eval_test_bpb.py \
  --work-dir runs/enwik8_unet \
  --ckpt ckpt_best.pt \
  --model unet
```
