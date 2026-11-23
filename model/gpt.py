"""Small decoder-only Transformer language model implemented with PyTorch."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, seq_len, self.n_heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=self.attn_dropout, training=self.training)
            y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.resid_dropout(self.out(y))


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TesseraGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 3,
                 n_heads: int = 4, max_seq_len: int = 128, dropout: float = 0.05):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        if idx.ndim != 2:
            raise ValueError("input token IDs must have shape [batch, sequence]")
        seq_len = idx.shape[1]
        if seq_len > self.max_seq_len:
            idx = idx[:, -self.max_seq_len:]
            if targets is not None:
                targets = targets[:, -self.max_seq_len:]
            seq_len = idx.shape[1]
        positions = torch.arange(seq_len, device=idx.device)
        x = self.dropout(self.token_embedding(idx) + self.position_embedding(positions)[None, :, :])
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
