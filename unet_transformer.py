import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - more stable than LayerNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - better than learned positional embeddings"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute for efficiency
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x, seq_len):
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to queries and keys"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    """Multi-head attention with modern improvements"""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project and split into Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Apply rotary embeddings
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # Downsampling changes seq_len, so slice the causal mask to match
            mask = mask[..., :seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.out_proj(out)

class SwiGLU(nn.Module):
    """SwiGLU activation - performs better than ReLU/GELU"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture"""
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm with residual connections
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class Downsample(nn.Module):
    """Downsample by averaging windows of tokens"""
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
    
    def forward(self, x):
        # x: [batch, seq_len, dim]
        batch_size, seq_len, dim = x.shape
        assert seq_len % self.window_size == 0, f"Seq len {seq_len} must be divisible by window size {self.window_size}"
        
        # Reshape to [batch, num_windows, window_size, dim] and keep earliest token in each window to stay causal
        x = x.view(batch_size, seq_len // self.window_size, self.window_size, dim)
        x = x[:, :, 0, :]
        
        return x

class Upsample(nn.Module):
    """Upsample by learned expansion - each token generates multiple tokens"""
    def __init__(self, dim, expansion_factor):
        super().__init__()
        self.expansion_factor = expansion_factor
        # Project to expansion_factor * dim, then reshape
        self.proj = nn.Linear(dim, expansion_factor * dim, bias=False)
    
    def forward(self, x):
        # x: [batch, seq_len, dim]
        batch_size, seq_len, dim = x.shape
        
        # Project and reshape to expand
        x = self.proj(x)  # [batch, seq_len, expansion_factor * dim]
        x = x.view(batch_size, seq_len * self.expansion_factor, dim)
        
        return x

class UNetTransformer(nn.Module):
    """U-Net style transformer with hierarchical token processing"""
    def __init__(self, vocab_size, dim=512, num_heads=8, mlp_ratio=4, 
                 dropout=0.1, window_sizes=[2, 2, 2, 2]):
        """
        Args:
            vocab_size: Size of vocabulary
            dim: Model dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            window_sizes: List of downsampling factors for each encoder level
                         e.g., [2, 2, 4] means: 512 -> 256 -> 128 -> 32
        """
        super().__init__()
        
        self.dim = dim
        self.window_sizes = window_sizes
        self.num_levels = len(window_sizes)
        
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
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, mask=None):
        # Token embeddings
        x = self.token_emb(x)
        x = self.dropout(x)
        
        # Encoder path with skip connections
        skip_connections = []
        
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = encoder_block(x, mask)
            skip_connections.append(x)  # Save for skip connection
            x = downsample(x)
        
        # Bottleneck
        x = self.bottleneck(x, mask)
        
        # Decoder path with skip connections
        for upsample, decoder_block, skip in zip(
            self.upsample_layers, 
            self.decoder_blocks, 
            reversed(skip_connections)
        ):
            x = upsample(x)
            x = x + skip  # Add skip connection (residual style)
            x = decoder_block(x, mask)
        
        # Output
        x = self.norm(x)
        logits = self.head(x)
        
        return logits
