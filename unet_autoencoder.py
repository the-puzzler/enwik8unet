import torch
import torch.nn as nn
from torch.autograd import Function
from unet_transformer import Upsample, Downsample, TransformerBlock, RMSNorm


class _STEQuantize(Function):
    @staticmethod
    def forward(ctx, x, q):
        return q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through gradients to both inputs so the encoder path and the
        # codebook embeddings can learn from reconstruction loss without extra terms.
        return grad_output, grad_output


class CodebookQuantizer(nn.Module):
    def __init__(self, num_codes: int, dim: int):
        super().__init__()
        self.num_codes = int(num_codes)
        self.dim = int(dim)
        self.codebook = nn.Embedding(self.num_codes, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        b, s, d = x.shape
        if d != self.dim:
            raise ValueError(f"Codebook dim mismatch: x has dim={d}, codebook dim={self.dim}")

        x_flat = x.reshape(b * s, d)
        # Distance in fp32 for stability (esp under AMP).
        x_f = x_flat.float()
        e_f = self.codebook.weight.float()  # [K, D]

        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x·e
        x2 = (x_f * x_f).sum(dim=1, keepdim=True)          # [N, 1]
        e2 = (e_f * e_f).sum(dim=1).unsqueeze(0)           # [1, K]
        xe = x_f @ e_f.t()                                 # [N, K]
        dist = x2 + e2 - 2.0 * xe

        codes = dist.argmin(dim=1)                          # [N]
        q_flat = self.codebook(codes)                       # [N, D]
        q = q_flat.view(b, s, d).type_as(x)

        return _STEQuantize.apply(x, q)


class UNetAutoEncoder(nn.Module):
     
    def __init__(self, vocab_size, dim=512, num_heads=8, mlp_ratio=4, 
                 dropout=0.1, window_sizes=[2, 2, 2, 2], num_codes: int = 0):
       
        super().__init__()
        
        self.dim = dim
        self.window_sizes = window_sizes
        self.num_levels = len(window_sizes)
        self.use_codebook = int(num_codes) > 0
        
        # Input embedding
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Encoder blocks and downsampling
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i, window_size in enumerate(window_sizes):
            self.encoder_blocks.append(
                TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            )
            self.downsample_layers.append(Downsample(window_size))
        
        # Bottleneck
        self.bottleneck = TransformerBlock(dim, num_heads, mlp_ratio, dropout)
        self.codebook = CodebookQuantizer(num_codes, dim) if self.use_codebook else None
        
        # Decoder blocks and upsampling
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for i, window_size in enumerate(reversed(window_sizes)):
            self.upsample_layers.append(Upsample(dim, window_size))
            self.decoder_blocks.append(
                TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            )
        
        # Output
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        

    def forward(self, x, mask=None):
        # Token embeddings
        x = self.token_emb(x)
        x = self.dropout(x)
  
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x, mask)
            
            x = downsample(x)
        
        # Bottleneck
        if self.codebook is not None:
            x = self.codebook(x)
        else:
            x = self.bottleneck(x, mask)
        
        # Decoder path with skip connections
        for upsample, decoder_block in zip(
            self.upsample_layers, 
            self.decoder_blocks, 
        ):
            x = upsample(x)
            x = decoder_block(x, mask)
        
        # Output
        x = self.norm(x)
        logits = self.head(x)
        
        return logits
