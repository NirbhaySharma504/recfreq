"""Mechanism on the conflict probes: exact direct-logit attribution and the induction head's attention kernel.

For each run and probe cell, the logit difference L = logit(c) - logit(w) at the query is split exactly into
contributions of the embeddings, every attention head and every MLP (final LayerNorm linearized at the actual
residual, so the parts sum to L). For the induction head we also record its attention to the old copies (A_c)
and to the recent copies (A_w), and its attention weight on each copy as a function of distance.

python -m recfreq.mech results/runs --exp E1,E2a,E2c --out figs_mech
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

from .data import TaskConfig
from .model import ModelConfig, Transformer
from .probes import GAPS, NC, NW, probe_batch


def load_model(run_dir: str, device):
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location=device, weights_only=False)
    mcfg = ModelConfig(**ck["mcfg"])
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, TaskConfig(**ck["task"])


@torch.no_grad()
def decompose(model: Transformer, tok: torch.Tensor, qi: int, du: torch.Tensor):
    """Exact split of the logit difference at position qi along unembedding direction du (B,C).
    Returns dict of (B,) tensors, plus attention patterns (per layer, (B,H,T)) from qi."""
    x = model.tok(tok)
    if model.pos is not None:
        x = x + model.pos(torch.arange(tok.shape[1], device=tok.device))
    parts = {"embed": x[:, qi].clone()}
    attn_q, head_vals = [], []
    for l, blk in enumerate(model.blocks):
        at = blk.attn
        B, T, C = x.shape
        h_in = blk.ln1(x)
        q, k, v = at.qkv(h_in).view(B, T, 3, at.H, at.dh).permute(2, 0, 3, 1, 4)
        if at.cfg.pe == "rope":
            q, k = at.rope(q), at.rope(k)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(at.dh) + at.bias(T, x.device)
        a = scores.softmax(-1)  # (B,H,T,T)
        o = a @ v  # (B,H,T,dh)
        Wo = at.out.weight  # (C, C), columns grouped by head
        out_total = torch.zeros_like(x)
        for hd in range(at.H):
            sl = slice(hd * at.dh, (hd + 1) * at.dh)
            oh = o[:, hd] @ Wo[:, sl].T  # (B,T,C)
            parts[f"L{l}H{hd}"] = oh[:, qi].clone()
            out_total = out_total + oh
        # per-source-position value vectors through W_O for each head at the query, for kernel analysis
        head_vals.append(torch.einsum("bhtd,chd->bhtc", v, Wo.view(C, at.H, at.dh)))  # (B,H,T,C)
        attn_q.append(a[:, :, qi])  # (B,H,T)
        x = x + out_total
        if blk.mlp is not None:
            m = blk.mlp(blk.ln2(x))
            parts[f"MLP{l}"] = m[:, qi].clone()
            x = x + m
    xf = x[:, qi]
    mu, sd = xf.mean(-1, keepdim=True), (xf.var(-1, unbiased=False, keepdim=True) + model.ln_f.eps).sqrt()
    g, bvec = model.ln_f.weight, model.ln_f.bias
    out = {}
    for name, u in parts.items():
        out[name] = (((u - u.mean(-1, keepdim=True)) / sd) * g * du).sum(-1)
    out["ln_bias"] = (bvec * du).sum(-1)
    return out, attn_q, head_vals, (mu, sd)


@torch.no_grad()
def run_one(run_dir: str, device, fills: int = 64, seed: int = 99):
    """Per probe cell: true logit difference, its exact decomposition, and for every head in layers >= 1 its
    direct contribution, its attention mass on the old (A_c) and recent (A_w) copies, and per-copy weights."""
    model, task = load_model(run_dir, device)
    g = torch.Generator(device=device).manual_seed(seed)
    qp = task.n_pairs - 1
    qi = 1 + 2 * qp
    lo_k = 1 + task.K
    heads = [(l, h) for l in range(1, model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    rows, kern = [], []
    ar = torch.arange(fills, device=device)
    for n_c, n_w, G in itertools.product(NC, NW, GAPS):
        tok, keys, vals, c, w = probe_batch(task, n_c, n_w, G, fills, g, device)
        Wu = model.unembed.weight
        du = Wu[lo_k + c] - Wu[lo_k + w]
        parts, attn_q, head_vals, (mu, sd) = decompose(model, tok, qi, du)
        logits = model(tok)[:, qi]
        L_true = logits[ar, lo_k + c] - logits[ar, lo_k + w]
        L_sum = sum(parts.values())
        w_pos = [qp - 4 * j for j in range(1, n_w + 1)]
        c_pos = [qp - 4 * n_w - G - 4 * i for i in range(n_c)]
        vc, vw = [2 + 2 * p for p in c_pos], [2 + 2 * p for p in w_pos]
        proj = lambda u: (((u - u.mean(-1, keepdim=True)) / sd) * model.ln_f.weight * du).sum(-1)
        rec = {"n_c": n_c, "n_w": n_w, "G": G, "L": L_true.mean().item(), "L_sd": L_true.std().item(),
               "recon_err": (L_sum - L_true).abs().max().item(), "L_absmax": L_true.abs().max().item()}
        for name, v in parts.items():
            rec["dla." + name] = v.mean().item()
        for l, h in heads:
            a = attn_q[l][:, h]
            contrib = a.unsqueeze(-1) * head_vals[l][:, h]
            tag = f"L{l}H{h}"
            rec[f"{tag}.A_c"] = a[:, vc].sum(1).mean().item()
            rec[f"{tag}.A_w"] = a[:, vw].sum(1).mean().item()
            rec[f"{tag}.D_cw"] = (proj(contrib[:, vc].sum(1)) + proj(contrib[:, vw].sum(1))).mean().item()
            for pos_list, typ in ((c_pos, "c"), (w_pos, "w")):
                for p in pos_list:
                    wt = a[:16, 2 + 2 * p].tolist()
                    kern.extend({"head": tag, "n_c": n_c, "n_w": n_w, "G": G, "fill": b, "type": typ,
                                 "d": qp - p, "a": x} for b, x in enumerate(wt))
        rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(kern), task, model


def kernel_fit(k: pd.DataFrame) -> dict:
    """Within-sequence fit of log attention weight on distance (exponential) or log distance (power law),
    plus a recent-copy (w) offset."""
    k = k[k.a > 0].copy()
    k["la"], k["ld"], k["is_w"] = np.log(k.a), np.log(k.d), (k.type == "w").astype(float)
    grp = k.groupby(["n_c", "n_w", "G", "fill"])
    for col in ("la", "d", "ld", "is_w"):
        k[col + "_c"] = k[col] - grp[col].transform("mean")
    out = {}
    ss = (k.la_c ** 2).sum()
    for nm, cols in (("exp", ["d_c"]), ("pow", ["ld_c"]), ("exp_w", ["d_c", "is_w_c"]), ("pow_w", ["ld_c", "is_w_c"])):
        X = k[cols].to_numpy()
        coef, *_ = np.linalg.lstsq(X, k.la_c.to_numpy(), rcond=None)
        out[f"kern_{nm}_r2"] = float(1 - ((k.la_c.to_numpy() - X @ coef) ** 2).sum() / ss)
        out[f"kern_{nm}_coef"] = float(-coef[0])
        if len(cols) == 2:
            out[f"kern_{nm}_wbonus"] = float(coef[1])
    return out


def summarize(df: pd.DataFrame, kern: pd.DataFrame, model) -> tuple[dict, list[dict]]:
    L = df.L.to_numpy()
    var = L.var()
    comps = [c for c in df.columns if c.startswith("dla.")]
    shares = {c[4:]: float(np.cov(df[c], L, bias=True)[0, 1] / var) for c in comps}
    out = {"recon_err": float(df.recon_err.max()), "L_absmax": float(df.L_absmax.max()), "shares": shares,
           "top": max(shares, key=shares.get)}
    head_rows = []
    slopes = model.slopes()
    for tag in sorted({c.split(".")[0] for c in df.columns if c.startswith("L") and ".A_c" in c}):
        D = df[f"dla.{tag}"].to_numpy()
        A_c, A_w = df[f"{tag}.A_c"].to_numpy(), df[f"{tag}.A_w"].to_numpy()
        r = {"head": tag, "share": shares[tag], "mass_on_copies": float((A_c + A_w).mean()),
             "cw_part": float(np.cov(df[f"{tag}.D_cw"], D, bias=True)[0, 1] / max(D.var(), 1e-12))}
        x = A_c - A_w  # Prop. 1: direct contribution = gamma * (A_c - A_w) + const
        X = np.stack([x, np.ones_like(x)], 1)
        coef, *_ = np.linalg.lstsq(X, D, rcond=None)
        r["prop1_gamma"] = float(coef[0])
        r["prop1_r2"] = float(1 - ((X @ coef - D) ** 2).mean() / max(D.var(), 1e-12))
        r.update(kernel_fit(kern[kern["head"] == tag]))
        l, h = int(tag[1]), int(tag[3])
        r["alibi_slope_per_pair"] = None if slopes is None else 2 * slopes[l][h]
        head_rows.append(r)
    copy_heads = [r for r in head_rows if r["share"] >= 0.1]
    out["n_copy_heads"] = len(copy_heads)
    out["copy_heads_share"] = float(sum(r["share"] for r in copy_heads))
    # Prop. 1 at the model level: predict the logit difference from the copy heads' measured attention alone,
    # L ~ sum_h gamma_h (A_c^h - A_w^h) + const; also with all heads in layers >= 1
    for nm, hs in (("copy", [r["head"] for r in copy_heads]), ("all", [r["head"] for r in head_rows])):
        if not hs:
            out[f"prop1_model_r2_{nm}"] = float("nan")
            continue
        X = np.stack([df[f"{t}.A_c"] - df[f"{t}.A_w"] for t in hs] + [np.ones(len(df))], 1)
        coef, *_ = np.linalg.lstsq(X, L, rcond=None)
        out[f"prop1_model_r2_{nm}"] = float(1 - ((X @ coef - L) ** 2).mean() / var)
    return out, head_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--exp", default="E1,E2a,E2c")
    ap.add_argument("--out", default="figs_mech")
    ap.add_argument("--fills", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = sorted(d for e in a.exp.split(",") for d in glob.glob(os.path.join(a.runs_dir, f"{e}_*")))
    rows, head_rows = [], []
    for d in dirs:
        cfg = json.load(open(os.path.join(d, "config.json")))
        if cfg["h"] == 0 or cfg["eps"] == 0 or not os.path.exists(os.path.join(d, "done.json")):
            continue
        df, kern, task, model = run_one(d, device, a.fills)
        name = os.path.basename(d)
        df.to_parquet(os.path.join(a.out, f"{name}.cells.parquet"))
        s, heads = summarize(df, kern, model)
        meta = {"run": name, "h": cfg["h"], "eps": cfg["eps"], "layers": cfg["layers"], "mlp": cfg["mlp"],
                "pe": cfg["pe"], "conflict_frac": cfg.get("conflict_frac", 0), "seed": cfg["seed"]}
        rows.append({**meta, **{k: v for k, v in s.items() if k != "shares"},
                     **{"share." + k: v for k, v in s["shares"].items()}})
        head_rows.extend({**meta, **hr} for hr in heads)
        cp = [hr for hr in heads if hr["share"] >= 0.1]
        desc = "  ".join(f"{hr['head']}:{hr['share']:.2f}/P1 {hr['prop1_r2']:.2f}/exp {hr['kern_exp_r2']:.2f}"
                         f"/pow {hr['kern_pow_r2']:.2f}" for hr in cp)
        print(f"{name:<50} recon {s['recon_err']:.1e}/{s['L_absmax']:.0f}  copy heads {s['copy_heads_share']:.2f}  {desc}",
              flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "mech.csv"), index=False)
    pd.DataFrame(head_rows).to_csv(os.path.join(a.out, "mech_heads.csv"), index=False)
    print("->", os.path.join(a.out, "mech.csv"))


if __name__ == "__main__":
    main()
