"""Switch-typo key-value streams (plan §5.1).

Sequence layout: BOS, k_0, y_0, k_1, y_1, ..., k_{N-1}, y_{N-1}  (length 1 + 2N).
Token ids: BOS = 0, keys = 1..K, values = K+1..K+V.
The model's logits at the position of k_t predict y_t, so value targets sit at
positions 2t+1 (inputs) -> 2t+2 (targets).

Each key k has a hidden value z_k(t) in {0..V-1}. At t = 0 it is uniform; at each
later pair step it is independently redrawn uniformly with probability h.
Observed y_t = z_{k_t}(t) w.p. 1-eps, else a typo:
  structured: T(z) = (z + 1) mod V
  uniform:    a uniformly random value != z
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaskConfig:
    K: int = 16
    V: int = 64
    n_pairs: int = 256
    h: float = 0.0
    eps: float = 0.0
    typo: str = "structured"  # or "uniform"
    key_dist: str = "zipf"  # or "uniform"
    zipf_alpha: float = 1.0

    @property
    def vocab(self) -> int:
        return 1 + self.K + self.V

    @property
    def seq_len(self) -> int:
        return 1 + 2 * self.n_pairs

    def key_probs(self) -> torch.Tensor:
        if self.key_dist == "uniform":
            return torch.full((self.K,), 1.0 / self.K)
        r = torch.arange(1, self.K + 1, dtype=torch.float64)
        p = r ** (-self.zipf_alpha)
        return (p / p.sum()).float()


def sample_keys(cfg: TaskConfig, B: int, g: torch.Generator, device) -> torch.Tensor:
    """(B, N) key indices in 0..K-1; Zipf ranks are permuted per sequence."""
    p = cfg.key_probs().to(device)
    ranks = torch.multinomial(p.expand(B, -1), cfg.n_pairs, replacement=True, generator=g)
    perm = torch.argsort(torch.rand(B, cfg.K, device=device, generator=g), dim=1)
    return torch.gather(perm, 1, ranks)


def hidden_values(cfg: TaskConfig, B: int, g: torch.Generator, device) -> torch.Tensor:
    """(B, K, N) hidden value of every key at every pair step."""
    K, N, V = cfg.K, cfg.n_pairs, cfg.V
    draws = torch.randint(0, V, (B, K, N), device=device, generator=g)
    flags = torch.rand(B, K, N, device=device, generator=g) < cfg.h
    flags[:, :, 0] = True  # initial value
    t = torch.arange(N, device=device).expand(B, K, N)
    last = torch.where(flags, t, torch.zeros_like(t)).cummax(dim=2).values
    return torch.gather(draws, 2, last)


def emit(cfg: TaskConfig, z: torch.Tensor, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the typo channel to hidden values z (any shape). Returns (y, is_typo)."""
    V = cfg.V
    typo = torch.rand(z.shape, device=z.device, generator=g) < cfg.eps
    if cfg.typo == "structured":
        alt = (z + 1) % V
    else:
        alt = (z + torch.randint(1, V, z.shape, device=z.device, generator=g)) % V
    return torch.where(typo, alt, z), typo


def to_tokens(cfg: TaskConfig, keys: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    B, N = keys.shape
    tok = torch.zeros(B, 1 + 2 * N, dtype=torch.long, device=keys.device)
    tok[:, 1::2] = keys + 1
    tok[:, 2::2] = vals + 1 + cfg.K
    return tok


def sample_batch(cfg: TaskConfig, B: int, g: torch.Generator, device) -> dict:
    keys = sample_keys(cfg, B, g, device)
    z_all = hidden_values(cfg, B, g, device)
    z = torch.gather(z_all, 1, keys.unsqueeze(1)).squeeze(1)  # (B, N)
    y, typo = emit(cfg, z, g)
    return {"tokens": to_tokens(cfg, keys, y), "keys": keys, "vals": y, "z": z, "typo": typo}


def first_occurrence(keys: torch.Tensor, K: int) -> torch.Tensor:
    """(B, N) bool: True where the key has not appeared earlier in the sequence."""
    onehot = torch.nn.functional.one_hot(keys, K)
    seen_before = (onehot.cumsum(1) - onehot).gather(2, keys.unsqueeze(2)).squeeze(2)
    return seen_before == 0
