"""
Plot train/val loss and bpb from a metrics.csv file.

Usage:
  python plot_metrics.py --metrics runs/enwik8_unet/metrics.csv --out metrics_plot.png
"""

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def load_metrics(path):
    rows = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            step = int(row[1])
            split = row[2]
            loss = float(row[3])
            bpb = float(row[4])
            rows[split].append((step, loss, bpb))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="path to metrics.csv")
    parser.add_argument("--out", default="metrics_plot.png", help="output png path")
    args = parser.parse_args()

    data = load_metrics(args.metrics)
    train = sorted(data.get("train", []), key=lambda x: x[0])
    val = sorted(data.get("val", []), key=lambda x: x[0])

    if not train and not val:
        raise SystemExit("No train/val rows found in metrics file.")

    fig, ax_bpb = plt.subplots(1, 1, figsize=(10, 5))

    if train:
        steps, _, bpb = zip(*train)
        ax_bpb.plot(steps, bpb, label="train", color="#1f77b4")
    if val:
        steps, _, bpb = zip(*val)
        ax_bpb.plot(steps, bpb, label="val", color="#ff7f0e")

    ax_bpb.set_title("Bits per byte (bpb)")
    ax_bpb.set_ylabel("bpb")
    ax_bpb.set_xlabel("step")
    ax_bpb.set_ylim(0, 3)
    ax_bpb.grid(True, alpha=0.3)
    ax_bpb.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
