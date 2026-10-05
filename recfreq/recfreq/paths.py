"""Run 4A: path ablation of layer-0 heads in 2-layer attention-only models.

Each layer-0 head's output is replaced by its mean (over ordinary data) on one path into layer 1:
  q   - the residual read by layer-1 queries
  k   - the residual read by layer-1 keys
  v   - the residual read by layer-1 values
  all - everywhere (including the direct path to the logits)
For every condition we record repeat-sighting loss, probe surfaces (original and swapped layout), and how
the layer-1 heads' attention to each copy depends on distance, recency, typo-variant status and duplicates.

python -m recfreq.paths results/runs --exp E1,E2a,E2b,E2c,E3 --out figs_run4 --shard 0 --nshards 3
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
from .probes import GAPS, NC, NW, probe_batch

LAYOUTS = {"sp4": (4, 1), "swap4": (4, -1)}


@torch.no_grad()
def layer0(model, tok):
    blk = model.blocks[0]
    at = blk.attn
    x0 = model.tok(tok)
    if model.pos is not None:
        x0 = x0 + model.pos(torch.arange(tok.shape[1], device=tok.device))
    B, T, C = x0.shape
    q, k, v = at.qkv(blk.ln1(x0)).view(B, T, 3, at.H, at.dh).permute(2, 0, 3, 1, 4)
    if at.cfg.pe == "rope":
        q, k = at.rope(q), at.rope(k)
    a = ((q @ k.transpose(-1, -2)) / math.sqrt(at.dh) + at.bias(T, x0.device)).softmax(-1)
    o = a @ v  # (B,H,T,dh)
    Wo = at.out.weight.view(C, at.H, at.dh)
    O = torch.einsum("bhtd,chd->bhtc", o, Wo)  # per-head output written to the residual
    return x0, O, a


@torch.no_grad()
def forward_paths(model, tok, ablate=(), means=None, return_attn=False):
    """ablate: list of (head, path) with path in {q, k, v, all}; means: (H, C) mean output of each layer-0 head."""
    x0, O, a0 = layer0(model, tok)
    x1 = x0 + O.sum(1)
    xs = {p: x1 for p in ("q", "k", "v", "res")}
    for h, p in ablate:
        delta = means[h].view(1, 1, -1) - O[:, h]
        for path in (("q", "k", "v", "res") if p == "all" else (p,)):
            xs[path] = xs[path] + delta
    blk = model.blocks[1]
    at = blk.attn
    B, T, C = x1.shape
    W = at.qkv.weight
    q = (blk.ln1(xs["q"]) @ W[:C].T).view(B, T, at.H, at.dh).transpose(1, 2)
    k = (blk.ln1(xs["k"]) @ W[C:2 * C].T).view(B, T, at.H, at.dh).transpose(1, 2)
    v = (blk.ln1(xs["v"]) @ W[2 * C:].T).view(B, T, at.H, at.dh).transpose(1, 2)
    if at.cfg.pe == "rope":
        q, k = at.rope(q), at.rope(k)
    a1 = ((q @ k.transpose(-1, -2)) / math.sqrt(at.dh) + at.bias(T, x1.device)).softmax(-1)
    out = at.out((a1 @ v).transpose(1, 2).reshape(B, T, C))
    logits = model.unembed(model.ln_f(xs["res"] + out))
    return (logits, a1, a0) if return_attn else logits


def regress_attention(rows: dict) -> dict:
    """Within-sequence regression of log attention on distance, recency, typo-variant status, duplicates."""
    la, d, rec, var, lns, grp = (np.concatenate(rows[k]) for k in ("la", "d", "recent", "variant", "lnsame", "grp"))
    df = pd.DataFrame({"la": la, "d": d, "recent": rec, "variant": var, "lnsame": lns, "grp": grp})
    g = df.groupby("grp")
    for c in ("la", "d", "recent", "variant", "lnsame"):
        df[c + "_c"] = df[c] - g[c].transform("mean")
    X = df[["d_c", "recent_c", "variant_c", "lnsame_c"]].to_numpy()
    coef, *_ = np.linalg.lstsq(X, df.la_c.to_numpy(), rcond=None)
    ss = (df.la_c ** 2).sum()
    return {"b_d": -coef[0], "b_recent": coef[1], "b_variant": coef[2], "b_lnsame": coef[3],
            "r2": float(1 - ((df.la_c.to_numpy() - X @ coef) ** 2).sum() / ss)}


@torch.no_grad()
def run_condition(model, task, ablate, means, fills=32, attn_fills=16):
    dev = next(model.parameters()).device
    qp = task.n_pairs - 1
    qi, lo_k = 1 + 2 * qp, 1 + task.K
    H1 = model.cfg.n_heads
    surf, mass = [], np.zeros(H1)
    att = {h: {k: [] for k in ("la", "d", "recent", "variant", "lnsame", "grp")} for h in range(H1)}
    gid = 0
    ncell = 0
    for lay, (sp, ws) in LAYOUTS.items():
        g = torch.Generator(device=dev).manual_seed(5)
        for n_c, n_w, G in itertools.product(NC, NW, GAPS):
            tok, keys, vals, c, w = probe_batch(task, n_c, n_w, G, fills, g, dev, sp=sp, w_shift=ws)
            logits, a1, _ = forward_paths(model, tok, ablate, means, return_attn=True)
            lp = logits[:, qi, lo_k:lo_k + task.V].float().log_softmax(-1)
            ar = torch.arange(fills, device=dev)
            surf.append({"layout": lay, "n_c": n_c, "n_w": n_w, "G": G, "lo": (lp[ar, c] - lp[ar, w]).mean().item()})
            w_pos = [qp - sp * j for j in range(1, n_w + 1)]
            c_pos = [qp - sp * n_w - G - sp * i for i in range(n_c)]
            pos = c_pos + w_pos
            vpos = torch.tensor([2 + 2 * p for p in pos], device=dev)
            A = a1[:attn_fills, :, qi][:, :, vpos].clamp_min(1e-12).cpu().numpy()  # (F, H, ncopies)
            if lay == "sp4":
                mass += A.sum(-1).mean(0)
                ncell += 1
            recent = np.array([0] * n_c + [1] * n_w)
            variant = recent if ws == 1 else 1 - recent
            nsame = np.array([n_c] * n_c + [n_w] * n_w)
            dist = np.array([qp - p for p in pos], float)
            for h in range(H1):
                for f in range(attn_fills):
                    att[h]["la"].append(np.log(A[f, h])); att[h]["d"].append(dist)
                    att[h]["recent"].append(recent); att[h]["variant"].append(variant)
                    att[h]["lnsame"].append(np.log(nsame)); att[h]["grp"].append(np.full(len(pos), gid + f))
            gid += attn_fills
    heads = []
    for h in range(H1):
        r = regress_attention(att[h])
        r.update({"head": h, "mass": float(mass[h] / ncell)})
        heads.append(r)
    return pd.DataFrame(surf), heads


@torch.no_grad()
def main_one(d: str, out: str, dev):
    name = os.path.basename(d)
    if os.path.exists(os.path.join(out, f"{name}.json")):
        return
    model, task = load_model(d, dev)
    if model.cfg.n_layers != 2 or model.cfg.mlp:
        return
    g = torch.Generator(device=dev).manual_seed(4321)
    ev = sample_batch(task, 128, g, dev)
    rep = ~first_occurrence(ev["keys"], task.K)
    # sanity: the rebuilt forward pass must reproduce the model
    ref = model(ev["tokens"][:16])
    mine = forward_paths(model, ev["tokens"][:16])
    err = float((ref - mine).abs().max())
    assert err < 1e-3, f"path forward mismatch {err}"
    # mean output of each layer-0 head, and previous-token score
    _, O, a0 = layer0(model, ev["tokens"][:64])
    means = O.mean((0, 2))
    T = a0.shape[-1]
    idx = torch.arange(1, T, device=dev)
    prev = a0[:, :, idx, idx - 1].mean((0, 2)).tolist()
    H0 = model.cfg.n_heads
    conds = [("base", [])] + [(f"h{h}_{p}", [(h, p)]) for h in range(H0) for p in ("q", "k", "v", "all")]
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
    cfg = json.load(open(os.path.join(d, "config.json")))
    pd.concat(surfs).to_parquet(os.path.join(out, f"{name}.surf.parquet"))
    pd.DataFrame(heads).to_csv(os.path.join(out, f"{name}.heads.csv"), index=False)
    json.dump({"run": name, "prev_score": prev, "losses": losses, "forward_err": err,
               **{k: cfg.get(k) for k in ("h", "eps", "pe", "seed", "conflict_frac", "d_model", "steps")}},
              open(os.path.join(out, f"{name}.json"), "w"))
    pt = int(np.argmax(prev))
    print(f"{name:<52} prev-token head {pt} ({prev[pt]:.2f})  repeat-CE base {losses['base']:.3f}  "
          f"k-ablate prev {losses[f'h{pt}_k']:.3f}  err {err:.1e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--exp", default="E1,E2a,E2b,E2c,E3")
    ap.add_argument("--out", default="figs_run4")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    dirs = sorted(d for e in a.exp.split(",") for d in glob.glob(os.path.join(a.runs_dir, f"{e}_*")))
    keep = []
    for d in dirs:
        c = json.load(open(os.path.join(d, "config.json")))
        if c["h"] > 0 and c["eps"] > 0 and c["layers"] == 2 and not c["mlp"] and os.path.exists(os.path.join(d, "done.json")):
            keep.append(d)
    for d in keep[a.shard::a.nshards]:
        main_one(d, a.out, dev)


if __name__ == "__main__":
    main()
