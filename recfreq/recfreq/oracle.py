"""Exact Bayes predictor for switch-typo KV streams (plan §5.2).

Keys are independent HMMs over V hidden values. Between two sightings of a key
that are g pair steps apart, the hidden value survives w.p. s = (1-h)^g and is
otherwise uniform, so the prior is s*b + (1-s)/V. The predictive is over the
*observed* token y (typo channel applied).
"""
from __future__ import annotations

import math

import torch

from .data import TaskConfig


def emission_matrix(cfg: TaskConfig, device, dtype=torch.float64) -> torch.Tensor:
    """E[z, y] = P(y | z)."""
    V, eps = cfg.V, cfg.eps
    eye = torch.eye(V, device=device, dtype=dtype)
    if cfg.typo == "structured":
        return (1 - eps) * eye + eps * torch.roll(eye, shifts=1, dims=1)  # y = z+1
    return (1 - eps) * eye + eps * (1 - eye) / (V - 1)


@torch.no_grad()
def bayes_predictive(cfg: TaskConfig, keys: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    """keys, vals: (B, N) ints. Returns (B, N, V) float64 with P(y_t = v | history)."""
    B, N = keys.shape
    V, K, device = cfg.V, cfg.K, keys.device
    E = emission_matrix(cfg, device)
    belief = torch.full((B, K, V), 1.0 / V, device=device, dtype=torch.float64)
    last = torch.full((B, K), -1, device=device, dtype=torch.long)
    out = torch.empty(B, N, V, device=device, dtype=torch.float64)
    ar = torch.arange(B, device=device)
    log1mh = math.log1p(-cfg.h) if cfg.h < 1 else -math.inf
    for t in range(N):
        k = keys[:, t]
        b = belief[ar, k]  # (B, V)
        gap = (t - last[ar, k]).to(torch.float64)
        s = torch.where(last[ar, k] >= 0, torch.exp(gap * log1mh), torch.zeros_like(gap)).unsqueeze(1)
        prior = s * b + (1 - s) / V
        pred = prior @ E
        out[:, t] = pred
        post = prior * E[:, vals[:, t]].T
        post = post / post.sum(1, keepdim=True).clamp_min(1e-300)
        belief[ar, k] = post
        last[ar, k] = t
    return out


def bayes_metrics(pred: torch.Tensor, vals: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    p = pred.gather(2, vals.unsqueeze(2)).squeeze(2).clamp_min(1e-300)
    ce = -p.log()
    acc = (pred.argmax(2) == vals).double()
    if mask is None:
        mask = torch.ones_like(vals, dtype=torch.bool)
    m = mask.double()
    return {"ce": float((ce * m).sum() / m.sum()), "acc": float((acc * m).sum() / m.sum())}


def critical_gap(V: int, h: float, eps: float, n_w: int) -> float:
    """Prop. 2 boundary G* (in pair steps), structured typos; inf if never, 0 if always."""
    if eps == 0:
        return 0.0
    if h == 0:
        return math.inf
    r = ((1 - eps) / eps) ** n_w / V  # s/(1-s) at the boundary
    s = r / (1 + r)
    return math.log(s) / math.log1p(-h)
