#!/usr/bin/env python
import sys

sys.dont_write_bytecode = True

import argparse
import math
import os
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from diffusion_unet_transformer import TextDiffusionUNetTransformer, cosine_sigreg_loss
from sigreg import SIGReg


LN2 = math.log(2.0)


def load_enwik8_memmap(data_path: str) -> np.memmap:
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Missing {data_path}.\n"
            "Download the raw enwik8 file (100MB) and place it there (typically named 'enwik8')."
        )
    return np.memmap(data_path, dtype=np.uint8, mode="r")


def make_splits(data: np.memmap, train_frac=0.9, val_frac=0.05):
    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]
    return train, val, test


def sample_tokens(data_arr: np.ndarray, batch_size: int, block_size: int, device: torch.device):
    max_start = len(data_arr) - block_size
    starts = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([data_arr[s : s + block_size].astype(np.int64) for s in starts], axis=0)
    return torch.from_numpy(x).to(device, non_blocking=True)


def alpha_sigma(t: torch.Tensor):
    half_pi = 0.5 * math.pi
    a = torch.cos(t * half_pi)
    s = torch.sin(t * half_pi)
    return a, s


@dataclass
class Metrics:
    total: float = 0.0
    cosine: float = 0.0
    sigreg_loss: float = 0.0
    sigreg_raw: float = 0.0
    xent_nats: float = 0.0
    bpb: float = 0.0
    true_logit: float = 0.0
    max_neg_logit: float = 0.0
    margin: float = 0.0
    n: int = 0  # number of batches aggregated

    def add(self, other: "Metrics"):
        self.total += other.total
        self.cosine += other.cosine
        self.sigreg_loss += other.sigreg_loss
        self.sigreg_raw += other.sigreg_raw
        self.xent_nats += other.xent_nats
        self.bpb += other.bpb
        self.true_logit += other.true_logit
        self.max_neg_logit += other.max_neg_logit
        self.margin += other.margin
        self.n += other.n

    def mean(self) -> "Metrics":
        denom = max(1, self.n)
        return Metrics(
            total=self.total / denom,
            cosine=self.cosine / denom,
            sigreg_loss=self.sigreg_loss / denom,
            sigreg_raw=self.sigreg_raw / denom,
            xent_nats=self.xent_nats / denom,
            bpb=self.bpb / denom,
            true_logit=self.true_logit / denom,
            max_neg_logit=self.max_neg_logit / denom,
            margin=self.margin / denom,
            n=denom,
        )


def compute_logits_l2(pred: torch.Tensor, emb_weight: torch.Tensor, tau: float) -> torch.Tensor:
    # Gaussian classifier in embedding space:
    #   p(token=i | pred) ∝ exp(-||pred - e_i||^2 / (2*tau^2))
    pred_f = pred.float()
    emb_f = emb_weight.float()
    pred_sq = (pred_f * pred_f).sum(dim=-1, keepdim=True)  # [...,1]
    emb_sq = (emb_f * emb_f).sum(dim=-1)  # [V]
    dot = pred_f @ emb_f.t()  # [...,V]
    dist2 = pred_sq + emb_sq - 2.0 * dot
    return -dist2 / (2.0 * (float(tau) ** 2))


@torch.no_grad()
def embedding_nn_cosine_stats(emb_weight: torch.Tensor) -> dict:
    emb_n = nn.functional.normalize(emb_weight.float(), p=2, dim=-1)
    sims = emb_n @ emb_n.t()
    sims.fill_diagonal_(-float("inf"))
    nn_sims = sims.max(dim=1).values
    nn_sims_sorted = nn_sims.sort().values
    n = nn_sims_sorted.numel()

    def pct(p: float) -> float:
        if n == 0:
            return float("nan")
        idx = int(round((p / 100.0) * (n - 1)))
        return float(nn_sims_sorted[idx])

    return {
        "mean": float(nn_sims.mean()),
        "min": float(nn_sims.min()),
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "max": float(nn_sims.max()),
    }


def decode_tokens_cosine(x: torch.Tensor, emb_weight: torch.Tensor) -> torch.Tensor:
    logits = compute_logits_l2(x, emb_weight, tau=1.0)
    return logits.argmax(dim=-1)


def decode_tokens_cosine_sample(x: torch.Tensor, emb_weight: torch.Tensor, tau: float) -> torch.Tensor:
    logits = compute_logits_l2(x, emb_weight, tau=float(tau))
    probs = logits.softmax(dim=-1)
    flat = probs.view(-1, probs.size(-1))
    samples = torch.multinomial(flat, num_samples=1).squeeze(1)
    return samples.view(probs.shape[:-1])


@torch.no_grad()
def ddim_denoise_x0(
    *,
    model: nn.Module,
    xt: torch.Tensor,
    t_start: float,
    t_end: float,
    steps: int,
    use_amp: bool,
) -> torch.Tensor:
    # Deterministic DDIM-style update where model predicts x0.
    # xt: [B,S,D]
    if steps < 2:
        raise ValueError("steps must be >= 2")

    device = xt.device
    t_vals = torch.linspace(float(t_start), float(t_end), steps, device=device)

    x = xt
    for i in range(steps - 1):
        t = t_vals[i].expand(x.size(0))
        t_next = t_vals[i + 1].expand(x.size(0))

        a, s = alpha_sigma(t)
        a_next, s_next = alpha_sigma(t_next)
        a = a[:, None, None].to(dtype=x.dtype)
        s = s[:, None, None].to(dtype=x.dtype)
        a_next = a_next[:, None, None].to(dtype=x.dtype)
        s_next = s_next[:, None, None].to(dtype=x.dtype)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            x0_hat = model(x, t, mask=None)

        eps_hat = (x - a * x0_hat) / (s + 1e-8)
        x = a_next * x0_hat + s_next * eps_hat

    return x


def _bytes_preview(tokens: torch.Tensor, limit: int = 256) -> str:
    b = bytes(tokens[:limit].tolist())
    try:
        return b.decode("utf-8", errors="replace")
    except Exception:
        return repr(b)


def _save_line_plot_png(values: list[float], path: str, *, title: str, x_label: str, y_label: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not values:
        return
    # Force a non-interactive backend and attempt to use a writable config/cache dir.
    mpl_dir = os.path.join(os.path.dirname(path) or ".", ".mplconfig")
    try:
        os.makedirs(mpl_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
        os.environ.setdefault("TMPDIR", mpl_dir)
    except Exception:
        pass
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(values, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def ddim_denoise_x0_with_mse_curve(
    *,
    model: nn.Module,
    xt: torch.Tensor,
    x0: torch.Tensor,
    t_start: float,
    t_end: float,
    steps: int,
    use_amp: bool,
) -> tuple[torch.Tensor, list[float]]:
    if steps < 2:
        raise ValueError("steps must be >= 2")

    device = xt.device
    t_vals = torch.linspace(float(t_start), float(t_end), steps, device=device)
    x = xt
    mse_curve: list[float] = []

    for i in range(steps - 1):
        t = t_vals[i].expand(x.size(0))
        t_next = t_vals[i + 1].expand(x.size(0))

        a, s = alpha_sigma(t)
        a_next, s_next = alpha_sigma(t_next)
        a = a[:, None, None].to(dtype=x.dtype)
        s = s[:, None, None].to(dtype=x.dtype)
        a_next = a_next[:, None, None].to(dtype=x.dtype)
        s_next = s_next[:, None, None].to(dtype=x.dtype)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            x0_hat = model(x, t, mask=None)

        mse_curve.append(float(((x0_hat.float() - x0.float()) ** 2).mean()))

        eps_hat = (x - a * x0_hat) / (s + 1e-8)
        x = a_next * x0_hat + s_next * eps_hat

    return x, mse_curve


@torch.no_grad()
def eval_batches(
    *,
    model: nn.Module,
    token_emb: nn.Embedding,
    sigreg: nn.Module,
    sigreg_weight: float,
    data_arr: np.ndarray,
    device: torch.device,
    batch_size: int,
    block_size: int,
    batches: int,
    use_amp: bool,
    t_mode: str,
    t_value: float | None,
    tau: float,
) -> Metrics:
    model.eval()
    token_emb.eval()

    out = Metrics()

    sigreg_raw = sigreg(token_emb.weight)
    sigreg_loss = float(sigreg_weight) * sigreg_raw

    for _ in range(batches):
        tokens = sample_tokens(data_arr, batch_size, block_size, device)

        if t_mode == "uniform":
            t = torch.rand(tokens.size(0), device=device)
        elif t_mode == "fixed":
            t = torch.full((tokens.size(0),), float(t_value), device=device)
        else:
            raise ValueError(f"bad t_mode={t_mode}")

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            x0 = token_emb(tokens)
            eps = torch.randn_like(x0)
            a, s = alpha_sigma(t)
            a = a[:, None, None].to(dtype=x0.dtype)
            s = s[:, None, None].to(dtype=x0.dtype)
            xt = a * x0 + s * eps
            pred = model(xt, t, mask=None)
            cosine_loss = cosine_sigreg_loss(pred, x0)
            total_loss = cosine_loss + sigreg_loss

        logits = compute_logits_l2(pred, token_emb.weight, tau=float(tau)).view(-1, token_emb.num_embeddings)
        targets = tokens.view(-1)

        xent_nats = nn.functional.cross_entropy(logits, targets)
        bpb = xent_nats / LN2

        true_logits = logits.gather(1, targets[:, None]).squeeze(1)
        logits_neg = logits.clone()
        logits_neg[torch.arange(logits_neg.size(0), device=logits_neg.device), targets] = -float("inf")
        max_neg = logits_neg.max(dim=1).values

        out.add(
            Metrics(
                total=float(total_loss),
                cosine=float(cosine_loss),
                sigreg_loss=float(sigreg_loss),
                sigreg_raw=float(sigreg_raw),
                xent_nats=float(xent_nats),
                bpb=float(bpb),
                true_logit=float(true_logits.mean()),
                max_neg_logit=float(max_neg.mean()),
                margin=float((true_logits - max_neg).mean()),
                n=1,
            )
        )

    return out.mean()


def _load_cfg_from_work_dir(work_dir: str):
    snap = os.path.join(work_dir, "config_snapshot.json")
    if os.path.isfile(snap):
        with open(snap, "r", encoding="utf-8") as f:
            return json.load(f)
    import config_diffusion as C

    return {k: getattr(C, k) for k in dir(C) if k.isupper()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="runs/enwik8_diffusion")
    ap.add_argument("--ckpt", default=None, help="Defaults to <work-dir>/ckpt_latest.pt")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--taus", default="1.0,0.3,0.1,0.03")
    ap.add_argument("--t-buckets", default="0.0,0.1,0.3,0.5,0.7,0.9,1.0")
    ap.add_argument("--denoise-demo", action="store_true", help="Generate from pure noise (no ground-truth sequence).")
    ap.add_argument("--reconstruct-demo", action="store_true", help="Denoise a noised real sequence (reconstruction test).")
    ap.add_argument("--decode", default="sample", choices=["sample", "argmax"])
    ap.add_argument("--decode-tau", type=float, default=None, help="Overrides config DECODE_TAU for sampling decode.")
    ap.add_argument("--denoise-steps", type=int, default=64)
    ap.add_argument("--denoise-t-noise", type=float, default=1.0, help="(reconstruct-demo) noise level used to create xt from x0.")
    ap.add_argument("--print-bytes", type=int, default=256)
    ap.add_argument(
        "--reconstruct-mse-png",
        default="reconstruct_mse.png",
        help="Output PNG path for reconstruction MSE plot (default: reconstruct_mse.png in --work-dir).",
    )
    ap.add_argument("--no-reconstruct-mse-png", action="store_true", help="Disable saving reconstruction MSE PNG.")
    args = ap.parse_args()

    cfg = _load_cfg_from_work_dir(args.work_dir)
    ckpt_path = args.ckpt or os.path.join(args.work_dir, "ckpt_latest.pt")

    if cfg.get("DEVICE", "cuda") == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    batch_size = int(args.batch_size or cfg["BATCH_SIZE"])
    block_size = int(args.block_size or cfg["BLOCK_SIZE"])
    use_amp = (not args.no_amp) and (device.type == "cuda") and bool(cfg.get("USE_AMP", True))

    data = load_enwik8_memmap(cfg["DATA_PATH"])
    train_arr, val_arr, test_arr = make_splits(data, train_frac=0.9, val_frac=0.05)
    split_arr = {"train": train_arr, "val": val_arr, "test": test_arr}[args.split]

    token_emb = nn.Embedding(int(cfg["VOCAB_SIZE"]), int(cfg["INPUT_DIM"])).to(device)
    model = TextDiffusionUNetTransformer(
        dim=int(cfg["DIM"]),
        num_heads=int(cfg["NUM_HEADS"]),
        mlp_ratio=float(cfg["MLP_RATIO"]),
        dropout=float(cfg["DROPOUT"]),
        window_sizes=cfg["WINDOW_SIZES"],
        input_dim=int(cfg["INPUT_DIM"]),
        out_dim=int(cfg["OUT_DIM"]),
    ).to(device)
    sigreg = SIGReg(knots=int(cfg.get("SIGREG_KNOTS", 17))).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    token_emb.load_state_dict(ckpt["token_emb"])
    model.load_state_dict(ckpt["model"])

    sigreg_weight = float(cfg.get("SIGREG_WEIGHT", 0.0))
    t_buckets = [float(x) for x in args.t_buckets.split(",") if x.strip()]
    taus = [float(x) for x in args.taus.split(",") if x.strip()]

    print(f"ckpt: {ckpt_path}")
    print(f"split: {args.split} | device: {device.type} | amp: {use_amp}")
    print(f"batch_size={batch_size} block_size={block_size} batches={args.batches}")
    print(f"sigreg_weight={sigreg_weight} | sigreg_knots={cfg.get('SIGREG_KNOTS', 17)}")
    print()

    tau_decode = float(args.decode_tau if args.decode_tau is not None else cfg.get("DECODE_TAU", 0.1))

    emb_stats = embedding_nn_cosine_stats(token_emb.weight)
    print("[embedding nn cosine] (for each token, max cosine to any other token)")
    print(
        f"mean={emb_stats['mean']:.4f} min={emb_stats['min']:.4f} "
        f"p50={emb_stats['p50']:.4f} p90={emb_stats['p90']:.4f} p99={emb_stats['p99']:.4f} max={emb_stats['max']:.4f}"
    )
    print()

    if args.denoise_demo:
        print("[denoise demo] (generate from pure noise)")
        with torch.no_grad():
            x = torch.randn(1, block_size, int(cfg["INPUT_DIM"]), device=device)
            x_hat = ddim_denoise_x0(
                model=model,
                xt=x,
                t_start=1.0,
                t_end=0.0,
                steps=int(args.denoise_steps),
                use_amp=use_amp,
            )
            pred_tokens = (
                decode_tokens_cosine_sample(x_hat, token_emb.weight, tau=tau_decode)
                if args.decode == "sample"
                else decode_tokens_cosine(x_hat, token_emb.weight)
            )

        print(
            f"steps={args.denoise_steps} | decode={args.decode} tau={tau_decode}"
        )
        print("pred: " + _bytes_preview(pred_tokens[0].cpu(), limit=args.print_bytes))
        print()

    if args.reconstruct_demo:
        print("[reconstruct demo] (denoise a noised real sequence)")
        tokens = sample_tokens(split_arr, batch_size=1, block_size=block_size, device=device)
        with torch.no_grad():
            x0 = token_emb(tokens)
            t_noise = torch.full((1,), float(args.denoise_t_noise), device=device)
            a, s = alpha_sigma(t_noise)
            a = a[:, None, None].to(dtype=x0.dtype)
            s = s[:, None, None].to(dtype=x0.dtype)
            xt = a * x0 + s * torch.randn_like(x0)

            x_hat, mse_curve = ddim_denoise_x0_with_mse_curve(
                model=model,
                xt=xt,
                x0=x0,
                t_start=float(args.denoise_t_noise),
                t_end=0.0,
                steps=int(args.denoise_steps),
                use_amp=use_amp,
            )
            pred_tokens = (
                decode_tokens_cosine_sample(x_hat, token_emb.weight, tau=tau_decode)
                if args.decode == "sample"
                else decode_tokens_cosine(x_hat, token_emb.weight)
            )
            acc = (pred_tokens == tokens).float().mean().item()

        print(
            f"steps={args.denoise_steps} t_noise={args.denoise_t_noise} | decode={args.decode} tau={tau_decode} | token_acc={acc*100:.2f}%"
        )
        print("gt : " + _bytes_preview(tokens[0].cpu(), limit=args.print_bytes))
        print("pred: " + _bytes_preview(pred_tokens[0].cpu(), limit=args.print_bytes))
        if not args.no_reconstruct_mse_png:
            out_png = args.reconstruct_mse_png
            if not os.path.isabs(out_png):
                out_png = os.path.join(args.work_dir, out_png)
            _save_line_plot_png(
                mse_curve,
                out_png,
                title="Reconstruction MSE vs DDIM steps",
                x_label="step",
                y_label="mse(x0_hat, x0)",
            )
            print(f"saved: {out_png}")
        print()

    # Fixed-t diagnostics at a default tau (use config if present).
    tau0 = float(cfg.get("DECODE_TAU", 0.1))
    print(f"[fixed t] tau={tau0} (L2 logits)")
    for t in t_buckets:
        m = eval_batches(
            model=model,
            token_emb=token_emb,
            sigreg=sigreg,
            sigreg_weight=sigreg_weight,
            data_arr=split_arr,
            device=device,
            batch_size=batch_size,
            block_size=block_size,
            batches=args.batches,
            use_amp=use_amp,
            t_mode="fixed",
            t_value=t,
            tau=tau0,
        )
        print(
            f"t={t:>4.2f} | xent={m.xent_nats:.4f} | bpb={m.bpb:.4f} | cosine={m.cosine:.4f} | "
            f"true_logit={m.true_logit:.4f} | max_neg={m.max_neg_logit:.4f} | margin={m.margin:.4f}"
        )

    print()

    # Tau sweep at uniform t.
    print("[tau sweep] (t ~ Uniform(0,1))")
    for tau in taus:
        m = eval_batches(
            model=model,
            token_emb=token_emb,
            sigreg=sigreg,
            sigreg_weight=sigreg_weight,
            data_arr=split_arr,
            device=device,
            batch_size=batch_size,
            block_size=block_size,
            batches=args.batches,
            use_amp=use_amp,
            t_mode="uniform",
            t_value=None,
            tau=tau,
        )
        print(
            f"tau={tau:<5g} | xent={m.xent_nats:.4f} | bpb={m.bpb:.4f} | cosine={m.cosine:.4f} | "
            f"true_logit={m.true_logit:.4f} | max_neg={m.max_neg_logit:.4f} | margin={m.margin:.4f}"
        )


if __name__ == "__main__":
    main()
