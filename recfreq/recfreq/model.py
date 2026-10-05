"""Small pre-LN transformer with optional MLPs and several positional encodings (plan §6).

pe:
  alibi_learn  per-head learnable slope lambda >= 0, logit bias -lambda * (i - j)
  alibi_fixed  standard ALiBi slopes (not trained)
  learned_abs  learned absolute position embedding
  rope         rotary position embedding on queries and keys
  nope         no positional information beyond the causal mask
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab: int
    max_len: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp: bool = False
    pe: str = "alibi_learn"
    init_std: float = 0.02


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.cfg = cfg
        self.H, self.dh = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        if cfg.pe == "alibi_learn":
            # log-uniform init in [1/64, 1/4], parametrized as log(lambda)
            lo, hi = math.log(1 / 64), math.log(1 / 4)
            self.log_slope = nn.Parameter(torch.linspace(lo, hi, self.H))
        elif cfg.pe == "alibi_fixed":
            self.register_buffer("fixed_slope", torch.tensor([2 ** (-8 * (h + 1) / self.H) for h in range(self.H)]))

    def slopes(self) -> torch.Tensor | None:
        if self.cfg.pe == "alibi_learn":
            return self.log_slope.exp()
        if self.cfg.pe == "alibi_fixed":
            return self.fixed_slope
        return None

    def bias(self, T: int, device) -> torch.Tensor:
        i = torch.arange(T, device=device)
        dist = (i[:, None] - i[None, :]).float()
        causal = torch.where(dist >= 0, 0.0, float("-inf"))
        sl = self.slopes()
        if sl is None:
            return causal.expand(1, 1, T, T)
        return (-sl.view(-1, 1, 1) * dist.clamp_min(0) + causal).unsqueeze(0)

    def rope(self, x: torch.Tensor) -> torch.Tensor:
        """Rotary embedding on (B, H, T, dh), base 10000, rotate-half convention."""
        T, half = x.shape[2], self.dh // 2
        inv = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
        ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv[None]
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., :half].float(), x[..., half:].float()
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)

    def forward(self, x, return_attn=False, head_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        if self.cfg.pe == "rope":
            q, k = self.rope(q), self.rope(k)
        bias = self.bias(T, x.device).to(q.dtype)
        if return_attn or head_mask is not None:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh) + bias
            attn = scores.float().softmax(-1).to(q.dtype)
            o = attn @ v
            if head_mask is not None:
                o = o * head_mask.view(1, -1, 1, 1).to(o.dtype)
        else:
            attn = None
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        o = o.transpose(1, 2).reshape(B, T, C)
        return self.out(o), attn


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg, layer)
        self.mlp = None
        if cfg.mlp:
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(), nn.Linear(4 * cfg.d_model, cfg.d_model)
            )

    def forward(self, x, return_attn=False, head_mask=None):
        a, attn = self.attn(self.ln1(x), return_attn, head_mask)
        x = x + a
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x, attn


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model) if cfg.pe == "learned_abs" else None
        self.blocks = nn.ModuleList([Block(cfg, l) for l in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.apply(self._init)
        # GPT-2 style: scale residual-writing projections by depth
        for n, p in self.named_parameters():
            if n.endswith("attn.out.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=self.cfg.init_std)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, return_attn=False, head_masks=None):
        B, T = idx.shape
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(T, device=idx.device))
        attns = []
        for l, blk in enumerate(self.blocks):
            hm = None if head_masks is None else head_masks.get(l)
            x, a = blk(x, return_attn, hm)
            attns.append(a)
        logits = self.unembed(self.ln_f(x))
        return (logits, attns) if return_attn else logits

    def slopes(self) -> list[list[float]] | None:
        out = []
        for blk in self.blocks:
            s = blk.attn.slopes()
            if s is None:
                return None
            out.append(s.detach().float().cpu().tolist())
        return out
