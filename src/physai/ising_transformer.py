"""
IsingTransformer: Minimal autoregressive Transformer for learning Ising model distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IsingTransformer(nn.Module):
    def __init__(self, d_model: int = 128, n_head: int = 8, n_layer: int = 6, block_size: int = 31):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model
        self.n_head = n_head
        
        self.tok_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        b, t = idx.shape
        
        # Token + position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        if targets is None:
            return logits
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class TransformerBlock(nn.Module):
    """Single transformer block with causal self-attention"""
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, 
            n_head, 
            dropout=0.0, 
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
    
    def forward(self, x):
        # Pre-norm + attention with causal mask
        b, t, d = x.shape
        # Lower triangular = True (visible), upper = False (masked)
        causal_mask = torch.tril(
            torch.ones(t, t, device=x.device, dtype=torch.bool), 
            diagonal=0
        )
        
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            need_weights=False
        )
        x = x + attn_out
        
        # Pre-norm + MLP
        x = x + self.mlp(self.ln2(x))
        
        return x
