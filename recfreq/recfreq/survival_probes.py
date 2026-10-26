"""Plan §19 B1: survival probes and a wider conflict-probe window, on trained checkpoints.

Survival probe: a fresh key tau is shown n (1 or 2, 4 pairs apart) times with value c, the last one g pairs before
the query; ordinary background. Under the switch-typo model with any survival curve s(g), the predictive mass on
{c-1, c, c+1} is M = s + 3(1-s)/V whatever eps is (structured typos), so s_hat = (M - 3/V) / (1 - 3/V) reads off the
model's belief that the old value is still current. The same is computed for the exact Bayes predictive.

Wide probes: the usual conflict cells plus G in {192, 224} and n_w in {5, 6}, wherever the layout fits in N pairs.

python -m recfreq.survival_probes results/runs --match E1_,E2b_,E5a_,E5r_,E2a_,E6_ --device cuda
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import pandas as pd
import torch

from .mech import load_model
from .probes import NC, NW, GAPS, key_predictive, probe_batch
from .verify.theory import exact_lo

DISTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 248)


@torch.no_grad()
def survival(model, task, device, fills=256, seed=99):
    g = torch.Generator(device=device).manual_seed(seed)
    V, N = task.V, task.n_pairs
    q, qtok = N - 1, 1 + 2 * (N - 1)
    rows = []
    for n, d in itertools.product((1, 2), DISTS):
        # reuse probe_batch with n_w = 0 recent copies: build by hand
        from .data import sample_batch, to_tokens
        from dataclasses import replace
        bg = sample_batch(replace(task, K=task.K - 1), fills, g, device)
        tau = torch.randint(0, task.K, (fills, 1), device=device, generator=g)
        keys = bg["keys"] + (bg["keys"] >= tau).long()
        vals = bg["vals"].clone()
        c = torch.randint(0, V, (fills,), device=device, generator=g)
        pos = [q - d - 4 * i for i in range(n)][::-1]
        if min(pos) < 0:
            continue
        keys[:, pos + [q]] = tau
        vals[:, pos] = c[:, None]
        vals[:, q] = c
        tok = to_tokens(task, keys, vals)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(tok)[:, qtok, 1 + task.K: 1 + task.K + V].float()
        p = lg.softmax(-1).double()
        ar = torch.arange(fills, device=device)
        idx = torch.stack([(c - 1) % V, c, (c + 1) % V], 1)
        M = p.gather(1, idx).sum(1)
        pb = key_predictive(task, pos, vals[:, pos], q)
        Mb = pb.gather(1, idx).sum(1)
        f = lambda m: ((m - 3 / V) / (1 - 3 / V)).clamp(1e-6, 1 - 1e-6)
        rows.append({"n": n, "d": d, "s_model": f(M).mean().item(), "s_model_med": f(M).median().item(),
                     "logit_s_model": torch.logit(f(M)).mean().item(),
                     "s_bayes": f(Mb).mean().item(), "logit_s_bayes": torch.logit(f(Mb)).mean().item(),
                     "p_c_model": p[ar, c].mean().item(), "p_c1_model": p[ar, (c + 1) % V].mean().item()})
    return pd.DataFrame(rows)


@torch.no_grad()
def wide(model, task, device, fills=256, seed=777):
    g = torch.Generator(device=device).manual_seed(seed)
    N, qtok = task.n_pairs, 1 + 2 * (task.n_pairs - 1)
    rows = []
    for n_c, n_w, G in itertools.product(NC, NW + (5, 6), GAPS + (192, 224)):
        if (n_w <= 4 and G <= 128) or 4 * n_w + G + 4 * (n_c - 1) > N - 1:
            continue
        tok, keys, vals, c, w = probe_batch(task, n_c, n_w, G, fills, g, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(tok)[:, qtok, 1 + task.K: 1 + task.K + task.V].float()
        lp = lg.log_softmax(-1)
        ar = torch.arange(fills, device=device)
        lo_b = exact_lo(task.h, task.eps, n_c, n_w, G) if task.V == 64 else float("nan")
        rows.append({"n_c": n_c, "n_w": n_w, "G": G, "lo_model": (lp[ar, c] - lp[ar, w]).mean().item(),
                     "lo_model_sd": (lp[ar, c] - lp[ar, w]).std().item(), "lo_bayes": lo_b})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--match", default="E1_,E2b_,E5a_,E5r_,E2a_,E6_")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--redo", type=int, default=0)
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
        name = os.path.basename(d)
        if not any(name.startswith(m) for m in a.match.split(",")):
            continue
        if not (os.path.exists(os.path.join(d, "ckpt.pt")) and os.path.exists(os.path.join(d, "done.json"))):
            continue
        args = json.load(open(os.path.join(d, "done.json")))["args"]
        if str(args["seed"]) not in a.seeds.split(","):
            continue
        if args["layers"] != 2 or args["mlp"] or args.get("typo", "structured") != "structured" or args["h"] <= 0 or args["eps"] <= 0:
            continue
        if not a.redo and os.path.exists(os.path.join(d, "survival.parquet")) and os.path.exists(os.path.join(d, "probes_wide.parquet")):
            continue
        model, task = load_model(d, dev)
        survival(model, task, dev).to_parquet(os.path.join(d, "survival.parquet"))
        wide(model, task, dev).to_parquet(os.path.join(d, "probes_wide.parquet"))
        print("done", name, flush=True)


if __name__ == "__main__":
    main()
