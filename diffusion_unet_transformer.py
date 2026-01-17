import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the same building blocks as unet_transformer for parity.
from unet_transformer import RMSNorm, TransformerBlock, Upsample


class DownsampleMean(nn.Module):
    """Downsample by mean-pooling over non-overlapping windows."""

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = int(window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size, seq_len // self.window_size, self.window_size, dim)
        return x.mean(dim=2)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for (continuous or discrete) timesteps."""

    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        self.dim = int(dim)
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [batch] float or int
        half = self.dim // 2
        t = t.float()
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return F.pad(emb, (0, self.dim - emb.shape[-1]))


class TextDiffusionUNetTransformer(nn.Module):
    """
    U-Net-style transformer for continuous (diffusion) text representations.

    - Full attention by default (pass mask=None).
    - No embedding->vocab head; returns continuous predictions for MSE-style losses.
    - Expects continuous inputs (no token IDs).
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        window_sizes=(2, 2, 2, 2),
        *,
        input_dim: int | None = None,
        out_dim: int | None = None,
        time_embed_dim: int | None = None,
        time_mlp_mult: int = 4,
        downsample_factory=DownsampleMean,
    ):
        super().__init__()

        self.model_dim = int(dim)
        self.window_sizes = list(window_sizes)
        self.num_levels = len(self.window_sizes)

        self.input_dim = int(input_dim) if input_dim is not None else int(dim)
        self.out_dim = int(out_dim) if out_dim is not None else self.input_dim

        self.in_proj = nn.Linear(self.input_dim, self.model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        time_embed_dim = int(time_embed_dim) if time_embed_dim is not None else self.model_dim
        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_mlp_mult * self.model_dim, bias=True),
            nn.SiLU(),
            nn.Linear(time_mlp_mult * self.model_dim, self.model_dim, bias=True),
        )

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        for window_size in self.window_sizes:
            self.encoder_blocks.append(TransformerBlock(self.model_dim, num_heads, mlp_ratio, dropout))
            self.downsample_layers.append(downsample_factory(window_size))

        # Bottleneck
        self.bottleneck = TransformerBlock(self.model_dim, num_heads, mlp_ratio, dropout)

        # Decoder
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for window_size in reversed(self.window_sizes):
            self.upsample_layers.append(Upsample(self.model_dim, window_size))
            self.decoder_blocks.append(TransformerBlock(self.model_dim, num_heads, mlp_ratio, dropout))

        # Output projection (continuous)
        self.norm = RMSNorm(self.model_dim)
        self.out_proj = nn.Linear(self.model_dim, self.out_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Stabilize diffusion-style training: start with near-zero output.
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: embeddings [B,S,input_dim]
            timesteps: [B] discrete or continuous time input
            mask: Optional attention mask in the same format as unet_transformer.MultiHeadAttention

        Returns:
            pred: [B,S,out_dim] continuous prediction (e.g. eps, v, or x0 target)
        """
        x = self.in_proj(x)
        x = self.dropout(x)

        t_emb = self.time_mlp(self.time_emb(timesteps)).unsqueeze(1)  # [B,1,model_dim]

        # Encoder + skips
        skip_connections = []
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = x + t_emb
            x = encoder_block(x, mask=mask)
            skip_connections.append(x)
            x = downsample(x)

        # Bottleneck
        x = x + t_emb
        x = self.bottleneck(x, mask=mask)

        # Decoder
        for upsample, decoder_block, skip in zip(self.upsample_layers, self.decoder_blocks, reversed(skip_connections)):
            x = upsample(x)
            x = x + skip
            x = x + t_emb
            x = decoder_block(x, mask=mask)

        x = self.norm(x)
        return self.out_proj(x)


class ZeroSigReg(nn.Module):
    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        return proj.new_zeros(())


def mse_sigreg_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sigreg: nn.Module = ZeroSigReg(),
    sigreg_weight: float = 0.0,
) -> torch.Tensor:
    pred_n = F.normalize(pred, p=2, dim=-1)
    target_n = F.normalize(target, p=2, dim=-1)
    cosine_loss = (1.0 - (pred_n * target_n).sum(dim=-1)).mean()
    return cosine_loss + float(sigreg_weight) * sigreg(pred.float())


cosine_sigreg_loss = mse_sigreg_loss


if __name__ == "__main__":
    # Simple smoke test (full attention, continuous output).
    model = TextDiffusionUNetTransformer(
        dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        window_sizes=(2, 2, 2),
        input_dim=256,
        out_dim=256,
    )
    x = torch.randn(2, 128, 256)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t, mask=None)
    print(out.shape)
