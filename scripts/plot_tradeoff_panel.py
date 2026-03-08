"""
2x2 small-multiples tradeoff panel vs UNet reference.

Panels (all relative to reference model, default: unet):
1) Inference compute difference (%)
2) Performance difference (%) using 2^(delta_bpb)-1
3) Single-forward memory difference (%)
4) Train compute-to-best-val difference (%)

Bars are shown for baseline_small and baseline; the reference model is the 0% line.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt


COMPARE_MODELS = ("baseline_small", "baseline")
BAR_COLORS = {"baseline_small": "#1f77b4", "baseline": "#1f77b4"}


def load_inference(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            m = r.get("model")
            if not m:
                continue
            out[m] = {
                "infer_tflop": float(r["fwd_flops_per_iter_tflop"]),
                "test_bpb": float(r["test_bpb"]),
                "single_forward_mem_mib": float(r["single_forward_alloc_delta_mib"]),
            }
    return out


def load_compute_to_best_val(path):
    best = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            m = r.get("model")
            if not m:
                continue
            bpb = float(r["val_bpb"])
            c_pf = float(r["cum_forward_pflops"])
            prev = best.get(m)
            if prev is None or bpb < prev["val_bpb"]:
                best[m] = {"val_bpb": bpb, "compute_pf": c_pf}
    return {m: d["compute_pf"] for m, d in best.items()}


def rel_pct(value, ref):
    return ((value - ref) / ref) * 100.0


def perf_pct(test_bpb_model, test_bpb_ref):
    # Positive means model is better than ref (lower bpb).
    return (2.0 ** (test_bpb_ref - test_bpb_model) - 1.0) * 100.0


def add_bar_values(ax, bars):
    y0, y1 = ax.get_ylim()
    yr = max(abs(y1 - y0), 1e-9)
    offset = max(0.01, 0.006 * yr)
    for b in bars:
        h = b.get_height()
        x = b.get_x() + b.get_width() / 2
        va = "bottom" if h >= 0 else "top"
        y = h + (offset if h >= 0 else -offset)
        ax.text(x, y, f"{h:+.2f}%", ha="center", va=va, fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-csv", default="profiling/inference_profile.csv")
    parser.add_argument("--val-flops-csv", default="profiling/val_vs_flops.csv")
    parser.add_argument("--reference-model", default="unet")
    parser.add_argument("--out", default="profiling/tradeoff_panel_vs_unet.png")
    args = parser.parse_args()

    inf = load_inference(args.inference_csv)
    best_pf = load_compute_to_best_val(args.val_flops_csv)

    ref = args.reference_model
    if ref not in inf:
        raise SystemExit(f"Reference model '{ref}' missing from inference CSV.")

    if ref not in best_pf:
        raise SystemExit(f"Reference model '{ref}' missing from val-vs-flops CSV.")

    models = [m for m in COMPARE_MODELS if m in inf and m in best_pf]
    if not models:
        raise SystemExit("No comparison models found.")

    compute_diff = [rel_pct(inf[m]["infer_tflop"], inf[ref]["infer_tflop"]) for m in models]
    perf_diff = [perf_pct(inf[m]["test_bpb"], inf[ref]["test_bpb"]) for m in models]
    mem_diff = [rel_pct(inf[m]["single_forward_mem_mib"], inf[ref]["single_forward_mem_mib"]) for m in models]
    train_compute_diff = [rel_pct(best_pf[m], best_pf[ref]) for m in models]

    x = list(range(len(models)))
    colors = [BAR_COLORS.get(m, "#888888") for m in models]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("Compute Diff (%)", compute_diff, "Inference compute per forward"),
        ("Performance Diff (%)", perf_diff, "2^Δbpb - 1 (test)"),
        ("Memory Diff (%)", mem_diff, "Single-forward memory allocation"),
        ("Train Compute-to-Best Diff (%)", train_compute_diff, "PFLOPs to best val"),
    ]

    for ax, (title, vals, subtitle) in zip(axs.ravel(), panels):
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8)
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
        ax.set_xticks(x, models)
        ax.set_title(title)
        ax.set_ylabel("% vs UNet")
        ax.text(0.01, 0.96, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=8, color="0.35")
        ax.grid(True, axis="y", alpha=0.25)
        add_bar_values(ax, bars)

    fig.suptitle("Tradeoff Panel vs UNet Reference", fontsize=14, y=0.99)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=220)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
