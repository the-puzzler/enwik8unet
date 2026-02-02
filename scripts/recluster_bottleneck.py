"""
Recluster existing bottleneck embeddings without re-running the model.

Loads bottleneck_vectors.npy (and optionally umap_2d.npy/spans.npy) from
runs/enwik8_unet/analysis, recomputes UMAP (optional), runs HDBSCAN with
user-specified parameters, and saves new labels/plots/wordclouds.

Requires: umap-learn, hdbscan, matplotlib. wordcloud is optional.
"""

import os
import re
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt


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

    eps = 1e-6
    min_in_docs = 2
    min_in_frac = 0.1

    stopwords = {
        "the", "and", "of", "to", "a", "in", "for", "is", "on", "that", "with", "as", "by",
        "be", "are", "this", "from", "or", "an", "at", "it", "was", "not", "which", "have",
    }
    html_noise = {
        "quot", "amp", "nbsp", "lt", "gt", "br", "href", "http", "www", "com", "org",
        "html", "htm", "xml", "img", "src", "ref", "table", "tr", "td", "th", "span",
        "div", "font", "style", "class", "align", "center", "left", "right", "nowrap",
        "colspan", "rowspan", "width", "height", "align", "title", "body", "head",
        "ccffcc", "cccccc",
        "bgcolor", "cellspacing", "cellpadding", "border", "valign", "wikitable",
        "1px", "1em", "png", "jpg", "thumb",
        "redirect", "category", "external", "links", "references", "isbn", "pdf",
        "edu", "gov", "php", "index", "archive",
    }

    tsv_lines = []
    wc_images = []
    wc_labels = []

    total_docs = spans_np.shape[0]
    global_doc_counts = {}
    cluster_doc_counts = {}
    cluster_sizes = {}

    for span, lab in zip(spans_np, labels):
        decoded = "".join(chr(b) if 32 <= b < 127 else " " for b in span)
        decoded = re.sub(r"<[^>]+>", " ", decoded)
        decoded = decoded.lower()
        tokens = re.split(r"[^a-z0-9]+", decoded)
        tokens = {t for t in tokens if len(t) > 2 and t not in stopwords and t not in html_noise}
        if not tokens:
            continue
        for t in tokens:
            global_doc_counts[t] = global_doc_counts.get(t, 0) + 1
        if lab < 0:
            continue
        if lab not in cluster_doc_counts:
            cluster_doc_counts[lab] = {}
            cluster_sizes[lab] = 0
        cluster_sizes[lab] += 1
        for t in tokens:
            cluster_doc_counts[lab][t] = cluster_doc_counts[lab].get(t, 0) + 1

    unique = sorted(cluster_doc_counts.keys())
    for lab in unique:
        in_docs = cluster_sizes.get(lab, 0)
        out_docs = total_docs - in_docs
        if in_docs == 0 or out_docs <= 0:
            continue
        counts = {}
        rows = []
        for word, in_count in cluster_doc_counts[lab].items():
            out_count = global_doc_counts.get(word, 0) - in_count
            p_in = in_count / in_docs
            p_out = out_count / out_docs
            score = math.log((p_in + eps) / (p_out + eps))
            if score <= 0:
                continue
            if in_count < min_in_docs or p_in < min_in_frac:
                continue
            counts[word] = score
            rows.append((word, score, p_in, p_out, in_count, out_count))

        if not counts:
            continue

        tsv_lines.append(f"# Cluster {lab} (n={in_docs})\nword\tscore\tp_in\tp_out\tin_docs\tout_docs\n")
        for word, score, p_in, p_out, in_count, out_count in sorted(rows, key=lambda r: r[1], reverse=True):
            tsv_lines.append(f"{word}\t{score:.4f}\t{p_in:.3f}\t{p_out:.3f}\t{in_count}\t{out_count}\n")
        tsv_lines.append("\n")

        wc = WordCloud(width=800, height=400, max_words=200, background_color="white").generate_from_frequencies(counts)
        wc_images.append(wc.to_array())
        wc_labels.append(lab)

    if tsv_lines:
        tsv_path = os.path.join(out_dir, "wordcloud_terms.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.writelines(tsv_lines)
        print(f"Saved word cloud term counts to {tsv_path}")

    if wc_images:
        n = len(wc_images)
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        if rows == 1 and cols == 1:
            axes = [axes]
        axes = [axes] if rows == 1 else axes
        axes = axes.flatten() if hasattr(axes, "flatten") else axes

        for idx, (img, lab) in enumerate(zip(wc_images, wc_labels)):
            ax = axes[idx]
            ax.imshow(img)
            ax.set_title(f"Cluster {lab}")
            ax.axis("off")

        for idx in range(len(wc_images), rows * cols):
            axes[idx].axis("off")

        out_path = os.path.join(out_dir, "wordcloud_grid.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved word cloud grid to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=str, default="runs/enwik8_unet/analysis", help="where bottleneck_vectors.npy lives")
    parser.add_argument("--recompute-umap", action="store_true", help="recompute UMAP instead of using existing umap_2d.npy if present")
    parser.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--min-cluster-size", type=int, default=10, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min-samples", type=int, default=5, help="HDBSCAN min_samples")
    parser.add_argument("--metric", type=str, default="euclidean", help="HDBSCAN metric")
    parser.add_argument("--spans-path", type=str, default=None, help="optional .npy containing spans [N, block_size] for wordclouds (defaults to analysis_dir/spans.npy)")
    parser.add_argument("--suffix", type=str, default="", help="optional suffix for output filenames")
    args = parser.parse_args()

    suffix = f"_{args.suffix}" if args.suffix else ""

    vec_path = os.path.join(args.analysis_dir, "bottleneck_vectors.npy")
    if not os.path.isfile(vec_path):
        raise FileNotFoundError(f"Missing {vec_path}. Run analyze_bottleneck.py first.")
    vectors = np.load(vec_path)
    print(f"Loaded vectors: {vectors.shape}")

    # UMAP
    umap_path = os.path.join(args.analysis_dir, "umap_2d.npy")
    if args.recompute_umap or not os.path.isfile(umap_path):
        try:
            import umap
        except ImportError:
            raise SystemExit("umap-learn not installed. Install with `pip install umap-learn`.")
        reducer = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric="cosine",
            random_state=42,
        )
        umap_xy = reducer.fit_transform(vectors)
        np.save(os.path.join(args.analysis_dir, f"umap_2d{suffix or ''}.npy"), umap_xy)
    else:
        umap_xy = np.load(umap_path)
        print(f"Loaded existing UMAP: {umap_xy.shape}")

    # HDBSCAN
    try:
        import hdbscan
    except ImportError:
        raise SystemExit("hdbscan not installed. Install with `pip install hdbscan`.")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
    )
    labels = clusterer.fit_predict(umap_xy)
    np.save(os.path.join(args.analysis_dir, f"bottleneck_labels{suffix or ''}.npy"), labels)
    print(f"Clusters found: {len(set(labels))} (including -1 noise)")

    plot_umap(umap_xy, labels, os.path.join(args.analysis_dir, f"umap_clusters{suffix or ''}.png"))

    spans_path = args.spans_path or os.path.join(args.analysis_dir, "spans.npy")
    if os.path.isfile(spans_path):
        spans = np.load(spans_path)
        generate_wordclouds(spans, labels, args.analysis_dir)
    else:
        print(f"No spans file found at {spans_path}; skipping wordclouds.")


if __name__ == "__main__":
    main()
