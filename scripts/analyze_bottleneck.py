"""
Extract bottleneck representations from a trained UNetTransformer on enwik8,
run UMAP + HDBSCAN clustering, and save plots/wordclouds for quick inspection.

Outputs (saved under runs/enwik8_unet/analysis by default):
- bottleneck_vectors.npy : pooled bottleneck embeddings [N, dim]
- bottleneck_labels.npy  : cluster labels from HDBSCAN (-1 = noise)
- spans.npy              : raw byte spans sampled for analysis [N, block_size]
- umap_2d.npy            : 2D UMAP projection [N, 2]
- umap_clusters.png      : scatter plot colored by cluster
- wordcloud_<label>.png  : word clouds per cluster (basic byte->char heuristic)

Note: Requires extra packages: umap-learn, hdbscan, wordcloud, matplotlib.
"""

import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import config as C
from train_enwik8 import load_enwik8_memmap, make_splits, get_causal_mask, load_checkpoint
from unet_transformer import UNetTransformer


def _strip_orig_mod(sd: dict):
    """Handle state_dicts saved from torch.compile (keys prefixed with _orig_mod.)."""
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


def sample_spans(data_arr: np.ndarray, num_spans: int, block_size: int):
    max_start = len(data_arr) - block_size
    starts = np.random.randint(0, max_start, size=(num_spans,))
    spans = np.stack([data_arr[s : s + block_size].astype(np.int64) for s in starts], axis=0)
    return torch.from_numpy(spans)


def collect_bottlenecks(model, spans, mask, device, batch_size: int = 1):
    model.eval()
    pooled_list = []
    for start in range(0, spans.size(0), batch_size):
        end = start + batch_size
        batch = spans[start:end].to(device)
        bottleneck_outputs = []

        def hook(_, __, output):
            bottleneck_outputs.append(output.detach().cpu())

        handle = model.bottleneck.register_forward_hook(hook)
        with torch.no_grad():
            _ = model(batch, mask=mask)
        handle.remove()

        assert len(bottleneck_outputs) == 1
        feats = bottleneck_outputs[0]       # [b, S, D]
        pooled_list.append(feats.mean(dim=1))  # -> [b, D]

    return torch.cat(pooled_list, dim=0)


def plot_umap(umap_xy, labels, out_path):
    plt.figure(figsize=(8, 6))
    palette = plt.cm.get_cmap("tab20")
    unique = sorted(set(labels))
    for lab in unique:
        mask = labels == lab
        color = palette(lab % palette.N) if lab >= 0 else (0.6, 0.6, 0.6, 0.4)
        plt.scatter(umap_xy[mask, 0], umap_xy[mask, 1], s=8, alpha=0.7, label=str(lab), color=color)
    plt.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_wordclouds(spans_np, labels, out_dir):
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud not installed; skipping wordcloud generation.")
        return

    unique = sorted(set(labels))
    for lab in unique:
        if lab < 0:
            continue  # skip noise
        mask = labels == lab
        if mask.sum() == 0:
            continue
        # Build text corpus for this cluster from bytes -> chars (ascii-ish)
        bytes_flat = spans_np[mask].flatten()
        chars = []
        for b in bytes_flat:
            if 32 <= b < 127:  # printable ASCII
                chars.append(chr(b))
            else:
                chars.append("�")
        text = "".join(chars)
        if not text:
            continue
        wc = WordCloud(width=800, height=400, max_words=200, background_color="white").generate(text)
        out_path = os.path.join(out_dir, f"wordcloud_{lab}.png")
        wc.to_file(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-spans", type=int, default=1000, help="number of random test spans to sample")
    parser.add_argument("--work-dir", type=str, default=C.WORK_DIR, help="run directory with checkpoints")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint (defaults to ckpt_best.pt)")
    parser.add_argument("--out-dir", type=str, default=None, help="where to save analysis outputs")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size for forward passes when collecting bottlenecks")
    args = parser.parse_args()

    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    data = load_enwik8_memmap(C.DATA_PATH)
    _, _, test_arr = make_splits(data, train_frac=0.9, val_frac=0.05)

    model = UNetTransformer(
        vocab_size=C.VOCAB_SIZE,
        dim=C.DIM,
        num_heads=C.NUM_HEADS,
        mlp_ratio=C.MLP_RATIO,
        dropout=C.DROPOUT,
        window_sizes=C.WINDOW_SIZES,
    ).to(device)

    ckpt_path = args.ckpt or os.path.join(args.work_dir, "ckpt_best.pt")
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    state = _strip_orig_mod(ckpt["model"])
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {ckpt_path}")

    mask = get_causal_mask(C.BLOCK_SIZE, device)

    spans = sample_spans(test_arr, args.num_spans, C.BLOCK_SIZE)
    pooled = collect_bottlenecks(model, spans, mask, device, batch_size=args.batch_size)  # [N, D]

    out_dir = args.out_dir or os.path.join(args.work_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    pooled_np = pooled.numpy()
    np.save(os.path.join(out_dir, "bottleneck_vectors.npy"), pooled_np)
    np.save(os.path.join(out_dir, "spans.npy"), spans.numpy())

    try:
        import umap
    except ImportError:
        print("umap-learn not installed; skipping UMAP/clustering. Install with `pip install umap-learn`.")
        return
    try:
        import hdbscan
    except ImportError:
        print("hdbscan not installed; skipping clustering. Install with `pip install hdbscan`.")
        return

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42)
    umap_xy = reducer.fit_transform(pooled_np)
    np.save(os.path.join(out_dir, "umap_2d.npy"), umap_xy)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10, metric="euclidean")
    labels = clusterer.fit_predict(umap_xy)
    np.save(os.path.join(out_dir, "bottleneck_labels.npy"), labels)

    plot_umap(umap_xy, labels, os.path.join(out_dir, "umap_clusters.png"))

    generate_wordclouds(spans.numpy(), labels, out_dir)

    print(f"Saved analysis outputs to {out_dir}")


if __name__ == "__main__":
    main()
