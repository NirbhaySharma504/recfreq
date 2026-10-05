"""Controlled conflict probes (plan §5.4).

For a target key tau that never appears in the background, place at the end of a
full-length sequence:
    n_c copies of (tau, c), `sp` pairs apart
    gap of G pairs between the last c and the first w
    n_w copies of (tau, w), w = T(c), `sp` pairs apart
    query (tau, ?) `sp` pairs after the last w, at the final pair
and read log p(c) - log p(w) from the model and the Bayes oracle.
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import pandas as pd
import torch

from .data import TaskConfig, sample_batch, to_tokens
from .oracle import bayes_predictive

NC = (1, 2, 4, 8)
NW = (1, 2, 3, 4)
GAPS = (2, 4, 8, 16, 32, 64, 128)


@torch.no_grad()
def probe_batch(task: TaskConfig, n_c: int, n_w: int, G: int, B: int, g: torch.Generator, device, sp: int = 4,
                w_shift: int = 1):
    """w_shift = 1: recent value w = T(c) = c + 1 (the recent copies are the typo-variant).
    w_shift = -1: w = c - 1, so the old copies are the typo-variant of the recent value (swapped layout)."""
    N = task.n_pairs
    bg = sample_batch(replace(task, K=task.K - 1), B, g, device)
    tau = torch.randint(0, task.K, (B, 1), device=device, generator=g)
    keys = bg["keys"] + (bg["keys"] >= tau).long()  # background never uses tau
    vals = bg["vals"].clone()
    c = torch.randint(0, task.V, (B,), device=device, generator=g)
    if task.typo == "structured":
        w = (c + w_shift) % task.V
    else:
        w = (c + torch.randint(1, task.V, (B,), device=device, generator=g)) % task.V
    q = N - 1
    w_pos = [q - sp * j for j in range(1, n_w + 1)]
    first_w = min(w_pos)
    c_pos = [first_w - G - sp * i for i in range(n_c)]
    assert min(c_pos) >= 0, "probe does not fit in context"
    keys[:, [q] + w_pos + c_pos] = tau
    vals[:, w_pos] = w[:, None]
    vals[:, c_pos] = c[:, None]
    vals[:, q] = c  # placeholder; the prediction at q does not see it
    return to_tokens(task, keys, vals), keys, vals, c, w


@torch.no_grad()
def key_predictive(task: TaskConfig, pos: list[int], obs: torch.Tensor, q: int) -> torch.Tensor:
    """Bayes predictive over y at pair q for a single key observed at pairs `pos` (sorted)
    with values obs (B, len(pos)). Same filter as oracle.bayes_predictive, restricted to one key."""
    from .oracle import emission_matrix
    B, V = obs.shape[0], task.V
    E = emission_matrix(task, obs.device)
    b = torch.full((B, V), 1.0 / V, device=obs.device, dtype=torch.float64)
    last = None
    for i, t in enumerate(list(pos) + [q]):
        s = 0.0 if last is None else (1 - task.h) ** (t - last)
        prior = s * b + (1 - s) / V
        if i == len(pos):
            return prior @ E
        post = prior * E[:, obs[:, i]].T
        b = post / post.sum(1, keepdim=True).clamp_min(1e-300)
        last = t


@torch.no_grad()
def conflict_batch(task: TaskConfig, B: int, g: torch.Generator, device, chunks: int = 1, sp: int = 4) -> dict:
    """Probe-shaped training sequences with random (n_c, n_w, G). The query value is sampled
    from the exact Bayes predictive, so Bayes stays the optimal predictor; only the query counts."""
    toks, vals_all = [], []
    for B_i in [B // chunks + (1 if i < B % chunks else 0) for i in range(chunks)]:
        if B_i == 0:
            continue
        n_c = int(torch.randint(1, 9, (1,), generator=g, device=device))
        n_w = int(torch.randint(1, 5, (1,), generator=g, device=device))
        G = 2 ** int(1 + 7 * torch.rand(1, generator=g, device=device).item())  # 2..128
        tok, keys, vals, c, w = probe_batch(task, n_c, n_w, G, B_i, g, device, sp)
        tau = keys[:, -1:]
        q = task.n_pairs - 1
        pos = (keys[0] == tau[0]).nonzero().flatten().tolist()[:-1]  # same layout for the whole chunk
        pred = key_predictive(task, pos, vals[:, pos], q)
        y = torch.multinomial(pred.float(), 1, generator=g).squeeze(1)
        vals[:, q] = y
        toks.append(to_tokens(task, keys, vals))
        vals_all.append(vals)
    return {"tokens": torch.cat(toks), "vals": torch.cat(vals_all)}


@torch.no_grad()
def run_probes(model, task: TaskConfig, device, fills: int = 256, seed: int = 777, sp: int = 4) -> pd.DataFrame:
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    rows = []
    qtok = 1 + 2 * (task.n_pairs - 1)
    for n_c, n_w, G in itertools.product(NC, NW, GAPS):
        tok, keys, vals, c, w = probe_batch(task, n_c, n_w, G, fills, g, device, sp)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(tok)[:, qtok, 1 + task.K : 1 + task.K + task.V].float()
        lp = lg.log_softmax(-1)
        bp = bayes_predictive(task, keys, vals)[:, -1].clamp_min(1e-300).log()
        ar = torch.arange(fills, device=device)
        lo_m = lp[ar, c] - lp[ar, w]
        lo_b = bp[ar, c] - bp[ar, w]
        am = lp.argmax(-1)
        rows.append({
            "n_c": n_c, "n_w": n_w, "G": G,
            "lo_model": lo_m.mean().item(), "lo_model_sd": lo_m.std().item(),
            "lo_bayes": lo_b.mean().item(),
            "p_cw_model": (lp[ar, c].exp() + lp[ar, w].exp()).mean().item(),
            "pick_w_model": (am == w).float().mean().item(),
            "pick_c_model": (am == c).float().mean().item(),
            "pick_w_bayes": (lo_b < 0).double().mean().item(),
            "agree": ((lo_m < 0) == (lo_b < 0)).float().mean().item(),
        })
    model.train()
    return pd.DataFrame(rows)
