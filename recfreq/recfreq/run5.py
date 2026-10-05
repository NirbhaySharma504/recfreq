"""Run 5 (plan §16): how do slow copy heads compute their Bayes-shaped old/recent attention split?

5A  combination ablations of layer-0 heads (all non-previous-token heads together, per path; pairs).
5B  score decomposition in copy heads: score_j = content (q·k_j / sqrt(d)) + distance (ALiBi) ; the content part
    is split by upstream source (token embedding incl. absolute position, each layer-0 head, LayerNorm bias), once
    on the key side (full query · key part) and once on the query side (query part · full key). LayerNorm is
    linearized at the actual residual, so the parts sum exactly to the content score.

python -m recfreq.run5 results/runs --out figs_run5 --run4 figs_run4 --shard 0 --nshards 3
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import first_occurrence, sample_batch
from .mech import load_model
from .paths import forward_paths, layer0, run_condition
from .probes import GAPS, NC, NW, probe_batch


@torch.no_grad()
def decompose_scores(model, tok, heads, qi, copy_pos):
    """For layer-1 heads in `heads`: per source component, the content score contribution at each copy position
    (key side and query side), the distance bias, and the true pre-softmax score (for a sanity check)."""
    x0, O, _ = layer0(model, tok)
    x1 = x0 + O.sum(1)
    blk = model.blocks[1]
    at = blk.attn
    B, T, C = x1.shape
    H0 = O.shape[1]
    sd = (x1.var(-1, unbiased=False, keepdim=True) + blk.ln1.eps).sqrt()
    g, bvec = blk.ln1.weight, blk.ln1.bias
    comps = {"emb": x0, **{f"L0H{h}": O[:, h] for h in range(H0)}}
    W = at.qkv.weight
    Wq, Wk = W[:C], W[C:2 * C]
    pos = torch.tensor(copy_pos, device=tok.device)

    def proj(u, Wm, idx):  # normalized component at positions idx, projected and split into heads: (B, n, H, dh)
        un = ((u[:, idx] - u[:, idx].mean(-1, keepdim=True)) / sd[:, idx]) * g
        return (un @ Wm.T).view(B, len(idx) if idx.dim() else 1, at.H, at.dh)

    def rope_at(x, positions):  # x: (B, n, H, dh); rotate with absolute positions
        if at.cfg.pe != "rope":
            return x
        half = at.dh // 2
        inv = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
        ang = positions.float()[:, None] * inv[None]
        cos, sin = ang.cos()[None, :, None], ang.sin()[None, :, None]
        x1_, x2_ = x[..., :half], x[..., half:]
        return torch.cat([x1_ * cos - x2_ * sin, x1_ * sin + x2_ * cos], -1)

    qidx = torch.tensor([qi], device=tok.device)
    bias_q = (bvec @ Wq.T).view(1, 1, at.H, at.dh).expand(B, 1, at.H, at.dh)
    bias_k = (bvec @ Wk.T).view(1, 1, at.H, at.dh).expand(B, len(copy_pos), at.H, at.dh)
    qparts = {k: rope_at(proj(u, Wq, qidx), qidx) for k, u in comps.items()}
    qparts["lnb"] = rope_at(bias_q, qidx)
    kparts = {k: rope_at(proj(u, Wk, pos), pos) for k, u in comps.items()}
    kparts["lnb"] = rope_at(bias_k, pos)
    q_full = sum(qparts.values())  # (B, 1, H, dh)
    k_full = sum(kparts.values())  # (B, n, H, dh)
    scale = 1 / math.sqrt(at.dh)
    out = {}
    slopes = at.slopes()
    for h in heads:
        kside = {k: (q_full[:, :, h] * v[:, :, h]).sum(-1) * scale for k, v in kparts.items()}  # (B, n)
        qside = {k: (v[:, :, h] * k_full[:, :, h]).sum(-1) * scale for k, v in qparts.items()}
        content = (q_full[:, :, h] * k_full[:, :, h]).sum(-1) * scale
        dist = (-(slopes[h] * (qi - pos).float())).expand(B, -1) if slopes is not None else torch.zeros_like(content)
        out[h] = {"kside": kside, "qside": qside, "content": content, "dist": dist}
    return out


@torch.no_grad()
def run_decomposition(model, task, heads, prev, fills=32):
    dev = next(model.parameters()).device
    qp = task.n_pairs - 1
    qi = 1 + 2 * qp
    rows, err = [], 0.0
    for lay, ws in (("sp4", 1), ("swap4", -1)):
        g = torch.Generator(device=dev).manual_seed(9)
        for n_c, n_w, G in itertools.product(NC, NW, GAPS):
            tok, keys, vals, c, w = probe_batch(task, n_c, n_w, G, fills, g, dev, sp=4, w_shift=ws)
            w_pos = [qp - 4 * j for j in range(1, n_w + 1)]
            c_pos = [qp - 4 * n_w - G - 4 * i for i in range(n_c)]
            vpos = [2 + 2 * p for p in c_pos + w_pos]
            dec = decompose_scores(model, tok, heads, qi, vpos)
            # sanity: content + distance must reproduce the true attention split between old and recent copies
            _, a1, _ = forward_paths(model, tok, return_attn=True)
            for h in heads:
                d = dec[h]
                sc = d["content"] + d["dist"]
                oc, ow = slice(0, n_c), slice(n_c, n_c + n_w)
                split_scores = torch.logsumexp(sc[:, oc], 1) - torch.logsumexp(sc[:, ow], 1)
                A = a1[:, h, qi][:, vpos]
                split_true = (A[:, oc].sum(1).clamp_min(1e-30).log() - A[:, ow].sum(1).clamp_min(1e-30).log())
                ok = (A[:, oc].sum(1) > 1e-12) & (A[:, ow].sum(1) > 1e-12)  # skip float-underflow cases
                if ok.any():
                    err = max(err, float((split_scores - split_true)[ok].abs().max()))
                rec = {"layout": lay, "head": h, "n_c": n_c, "n_w": n_w, "G": G,
                       "split": split_true.mean().item(),
                       "content_lse": (torch.logsumexp(d["content"][:, oc], 1) -
                                       torch.logsumexp(d["content"][:, ow], 1)).mean().item(),
                       "content_mean": (d["content"][:, oc].mean(1) - d["content"][:, ow].mean(1)).mean().item(),
                       "dist_mean": (d["dist"][:, oc].mean(1) - d["dist"][:, ow].mean(1)).mean().item()}
                for side in ("kside", "qside"):
                    for comp, v in d[side].items():
                        tag = "prev" if comp == f"L0H{prev}" else comp
                        rec[f"{side}.{tag}"] = (v[:, oc].mean(1) - v[:, ow].mean(1)).mean().item()
                rows.append(rec)
    return pd.DataFrame(rows), err


def main_one(d, out, run4, dev):
    name = os.path.basename(d)
    if os.path.exists(os.path.join(out, f"{name}.json")):
        return
    model, task = load_model(d, dev)
    if model.cfg.n_layers != 2 or model.cfg.mlp:
        return
    meta4 = json.load(open(os.path.join(run4, f"{name}.json")))
    prev = int(np.argmax(meta4["prev_score"]))
    h4 = pd.read_csv(os.path.join(run4, f"{name}.heads.csv"))
    base = h4[(h4.cond == "base") & (h4["mass"] > 0.5)].copy()
    slopes = model.blocks[1].attn.slopes()
    base["decay"] = [float(slopes[int(h)]) * 2 if slopes is not None else float(b) for h, b in zip(base["head"], base.b_d)]
    slow = base[base.decay <= base.decay.median()]["head"].astype(int).tolist()
    copy_heads = base["head"].astype(int).tolist()
    # ---- 5A: combination ablations -------------------------------------------------------------------------
    g = torch.Generator(device=dev).manual_seed(4321)
    ev = sample_batch(task, 128, g, dev)
    rep = ~first_occurrence(ev["keys"], task.K)
    _, O, _ = layer0(model, ev["tokens"][:64])
    means = O.mean((0, 2))
    H0 = model.cfg.n_heads
    others = [h for h in range(H0) if h != prev]
    conds = [("base", [])] + [(f"nonprev_{p}", [(h, p) for h in others]) for p in ("all", "q", "k", "v")]
    if H0 <= 4:
        conds += [(f"pair{a}{b}_all", [(a, "all"), (b, "all")]) for a, b in itertools.combinations(others, 2)]
    surfs, heads, losses = [], [], {}
    for cname, abl in conds:
        ces = []
        for i in range(0, ev["tokens"].shape[0], 32):
            lg = forward_paths(model, ev["tokens"][i:i + 32], abl, means)[:, 1::2, 1 + task.K:1 + task.K + task.V]
            ces.append(F.cross_entropy(lg.float().transpose(1, 2), ev["vals"][i:i + 32], reduction="none"))
        ce = torch.cat(ces)
        losses[cname] = float((ce * rep).sum() / rep.sum())
        s, hd = run_condition(model, task, abl, means)
        surfs.append(s.assign(cond=cname))
        heads.extend({**r, "cond": cname} for r in hd)
    pd.concat(surfs).to_parquet(os.path.join(out, f"{name}.surf.parquet"))
    pd.DataFrame(heads).to_csv(os.path.join(out, f"{name}.heads.csv"), index=False)
    # ---- 5B: score decomposition in copy heads --------------------------------------------------------------
    dec, err = run_decomposition(model, task, copy_heads, prev)
    dec["slow"] = dec["head"].isin(slow)
    dec.to_parquet(os.path.join(out, f"{name}.dec.parquet"))
    cfg = json.load(open(os.path.join(d, "config.json")))
    json.dump({"run": name, "prev": prev, "slow_heads": slow, "copy_heads": copy_heads, "losses": losses,
               "decomp_err": err, **{k: cfg.get(k) for k in ("h", "eps", "pe", "seed", "conflict_frac", "d_model",
                                                            "steps")}},
              open(os.path.join(out, f"{name}.json"), "w"))
    print(f"{name:<52} prev {prev} slow {slow} decomp err {err:.1e}  CE base {losses['base']:.3f} "
          f"nonprev_all {losses['nonprev_all']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--out", default="figs_run5")
    ap.add_argument("--run4", default="figs_run4")
    ap.add_argument("--only", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    names = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(a.run4, "*.json")))
    if a.only:
        names = [n for n in names if n.startswith(a.only)]
    for n in names[a.shard::a.nshards]:
        main_one(os.path.join(a.runs_dir, n), a.out, a.run4, dev)


if __name__ == "__main__":
    main()
