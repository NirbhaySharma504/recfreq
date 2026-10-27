"""Plan §19.14 round 8 (H-mass): copy-head attention mass on old vs recent copies, and a mass-matching boost.

Cells: n_c in {1, 2, 4, 8}, n_w 1-6, G up to 224 (where the layout fits). Per cell: lo_model, lo_bayes, and for each
layer-1 copy head (induction score > 0.9, else the best one) the log attention mass from the query key token to the
old-copy pairs (key + value tokens) and to the recent-copy pairs (logsumexp of the pre-softmax scores, so far copies
do not underflow), averaged over fills: lc{h}, lw{h}.

Boost (n_c = 2): for n_w = k in {2, 3, 4} at gap G, the matching n_w = 1 cell is at gap G + 4(k - 1), so the old
copies sit at the same distances. The n_w = 1 cell is run twice on the same fills, plain and with beta_h added to copy head h's scores at the single recent copy's tokens, with
beta_h = (lw - lc in the real k-copy cell) - (lw - lc in the n_w = 1 cell), so each head's log mass ratio matches
the k-copy cell. Record lo under the boost.

Writes mass.parquet and mass_boost.parquet into each run dir.
python -m recfreq.mass_probe results/runs --match E1_,E2b_,E5a_,E6_,E11_,E2a_,E5r_,E12_ --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os

import pandas as pd
import torch

from .mech import load_model
from .probes import NC, probe_batch
from .verify.theory import exact_lo

NW = (1, 2, 3, 4, 5, 6)
GS = (2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224)


def cols(pos):
    return [t for p in pos for t in (1 + 2 * p, 2 + 2 * p)]


@torch.no_grad()
def cell(model, task, heads, n_c, n_w, G, fills, seed, device, beta=None):
    g = torch.Generator(device=device).manual_seed(seed)
    tok, keys, vals, c, w = probe_batch(task, n_c, n_w, G, fills, g, device)
    N, T = task.n_pairs, tok.shape[1]
    q, qtok = N - 1, 1 + 2 * (N - 1)
    w_pos = [q - 4 * j for j in range(1, n_w + 1)]
    c_pos = [min(w_pos) - G - 4 * i for i in range(n_c)]
    att = model.blocks[1].attn
    store = {}
    hook = att.register_forward_hook(lambda m, inp, out: store.__setitem__("x", inp[0]))
    if beta is not None:
        eb = torch.zeros(1, att.H, T, T, device=device)
        for h, b in zip(heads, beta):
            eb[0, h, qtok, cols(w_pos)] = b
        att.extra_bias = eb
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(tok)
        S = query_scores(att, store["x"].float(), qtok)  # (B, H, T), pre-softmax, extra_bias included
    finally:
        att.extra_bias = None
        hook.remove()
    lp = lg[:, qtok, 1 + task.K: 1 + task.K + task.V].float().log_softmax(-1)
    ar = torch.arange(fills, device=device)
    out = {"n_c": n_c, "n_w": n_w, "G": G, "lo_model": (lp[ar, c] - lp[ar, w]).mean().item()}
    lse = S.logsumexp(-1)
    for h in heads:
        out[f"lc{h}"] = (S[:, h, cols(c_pos)].logsumexp(-1) - lse[:, h]).mean().item()
        out[f"lw{h}"] = (S[:, h, cols(w_pos)].logsumexp(-1) - lse[:, h]).mean().item()
    return out


def query_scores(att, x, qtok):
    """Pre-softmax scores of the query token's row in attention module att, recomputed in float32."""
    B, T, _ = x.shape
    q, k, _ = torch.nn.functional.linear(x, att.qkv.weight.float()).view(B, T, 3, att.H, att.dh).permute(2, 0, 3, 1, 4)
    if att.cfg.pe == "rope":
        q, k = att.rope(q), att.rope(k)
    bias = att.bias(T, x.device).float()
    if getattr(att, "extra_bias", None) is not None:
        bias = bias + att.extra_bias.float()
    return (q[:, :, qtok:qtok + 1] @ k.transpose(-1, -2)).squeeze(2) / math.sqrt(att.dh) + bias[:, :, qtok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--match", default="E6_")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--fills", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
        name = os.path.basename(d)
        if not name.startswith(tuple(a.match.split(","))) or os.path.exists(os.path.join(d, "mass_boost.parquet")):
            continue
        if not (os.path.exists(os.path.join(d, "ckpt.pt")) and os.path.exists(os.path.join(d, "done.json"))):
            continue
        dj = json.load(open(os.path.join(d, "done.json")))
        ar = dj["args"]
        if str(ar["seed"]) not in a.seeds.split(",") or ar["layers"] != 2 or ar["mlp"] or ar["V"] != 64 \
                or ar.get("pe") not in ("alibi_learn", "rope"):
            continue
        L1 = [x for x in dj["heads"] if x["layer"] == 1]
        heads = [x["head"] for x in L1 if x["ind_score"] > 0.9] or [max(L1, key=lambda x: x["ind_score"])["head"]]
        model, task = load_model(d, dev)
        rows = []
        for i, (n_c, n_w, G) in enumerate(itertools.product(NC, NW, GS)):
            if 4 * n_w + G + 4 * (n_c - 1) > task.n_pairs - 1:
                continue
            r = cell(model, task, heads, n_c, n_w, G, a.fills, 1000 + i, dev)
            r["lo_bayes"] = exact_lo(task.h, task.eps, n_c, n_w, G)
            rows.append(r)
        C = pd.DataFrame(rows)
        C.assign(heads=",".join(map(str, heads))).to_parquet(os.path.join(d, "mass.parquet"))
        idx = C.set_index(["n_c", "n_w", "G"])
        brows = []
        for k, G in itertools.product((2, 3, 4), GS):
            G1 = G + 4 * (k - 1)
            if (2, k, G) not in idx.index:
                continue
            real = idx.loc[(2, k, G)]
            seed = 5000 + 100 * k + G  # the unboosted and boosted n_w = 1 cells share their fills
            one = cell(model, task, heads, 2, 1, G1, a.fills, seed, dev)
            beta = [(real[f"lw{h}"] - real[f"lc{h}"]) - (one[f"lw{h}"] - one[f"lc{h}"]) for h in heads]
            r = cell(model, task, heads, 2, 1, G1, a.fills, seed, dev, beta=beta)
            brows.append({"k": k, "G": G, "G1": G1, "lo_boost": r["lo_model"], "lo_one": one["lo_model"],
                          "lo_real": real.lo_model, "beta_mean": sum(beta) / len(beta),
                          **{f"lr{h}_boost": r[f"lw{h}"] - r[f"lc{h}"] for h in heads},
                          **{f"lr{h}_real": real[f"lw{h}"] - real[f"lc{h}"] for h in heads}})
        pd.DataFrame(brows).to_parquet(os.path.join(d, "mass_boost.parquet"))
        print("done", name, len(heads), "copy heads", flush=True)


if __name__ == "__main__":
    main()
