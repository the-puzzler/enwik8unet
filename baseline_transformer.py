import torch
import torch.nn as nn

# Reuse the same components as unet_transformer to ensure parity
from unet_transformer import RMSNorm, TransformerBlock


class BaselineTransformer(nn.Module):
    """
    A vanilla stacked transformer baseline for comparison with UNetTransformer.

    Uses the exact same building blocks (RMSNorm, rotary MHA, SwiGLU MLP) as
    unet_transformer.py, but without the downsample/upsample path.
    """

    def __init__(self, vocab_size, dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1, num_layers=12):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )

        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying (matches UNetTransformer)
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        # x: [batch, seq_len]
        x = self.token_emb(x)
        x = self.dropout(x)

        for block in self.layers:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.head(x)
        return logits


if __name__ == "__main__":
    # Simple smoke test
    model = BaselineTransformer(
        vocab_size=256,
        dim=512,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        num_layers=12,
    )
    dummy = torch.randint(0, 256, (2, 1024))
    mask = torch.tril(torch.ones(1, 1, 1024, 1024, dtype=torch.uint8))
    out = model(dummy, mask=mask)
    print(out.shape)
