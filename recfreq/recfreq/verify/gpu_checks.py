"""GPU audit checks (run after the E4 training finishes).

ood        2.2  probes are not out of distribution: loss on the probe sequences' background positions vs ordinary
                data, and copy-head attention mass on copies, probes vs ordinary data (20 models).
precision  1.3  stored probe surfaces were scored under bf16 autocast; rescore in fp32 (12 models) and compare the
                fitted (h', eps') and R².
slope      5.2  scale the ALiBi slopes of all late heads by s in {0.25, 0.5, 0.71, 1, 1.41, 2, 4} (all ALiBi models
                with probes): fitted h' and ordinary-data loss per s; elasticity k from log h' = k log s + c.

python -m recfreq.verify.gpu_checks ood|precision|slope --out verify_out
"""
from __future__ import annotations

import argparse
import copy
import glob
import itertools
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data import first_occurrence, sample_batch
from ..layouts import CELLS, LAY4, BayesFree, r2
from ..mech import load_model
from ..probes import GAPS, NC, NW, probe_batch
from ..train import value_logits

R = "results/runs"


def pick(pattern_list):
    out = []
    for p in pattern_list:
        out += sorted(glob.glob(os.path.join(R, p)))
    return [d for d in out if os.path.exists(os.path.join(d, "ckpt.pt"))]


@torch.no_grad()
def surface(model, task, fills=128, seed=777, autocast=False, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    qi, lo_k = 1 + 2 * (task.n_pairs - 1), 1 + task.K
    ys = []
    for n_c, n_w, G in itertools.product(NC, NW, GAPS):
        tok, keys, vals, c, w = probe_batch(task, n_c, n_w, G, fills, g, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            lg = model(tok)[:, qi, lo_k:lo_k + task.V].float()
        lp = lg.log_softmax(-1)
        ar = torch.arange(fills, device=dev)
        ys.append((lp[ar, c] - lp[ar, w]).mean().item())
    return np.array(ys)


@torch.no_grad()
def ood(out, dev):
    rows = []
    dirs = pick(["E1_h0.01_e0.1_*_s0", "E1_h0.03_e0.2_*_s0", "E2a_h0.01_e0.1_*_s0", "E2a_h0.03_e0.2_*_s0",
                 "E2b_h0.003_e0.05_*_s0", "E2b_h0.03_e0.3_*_s0"])[:20]
    for d in dirs:
        model, task = load_model(d, dev)
        g = torch.Generator(device=dev).manual_seed(55)
        ev = sample_batch(task, 64, g, dev)
        lg = value_logits(task, model(ev["tokens"])).float()
        ce = F.cross_entropy(lg.transpose(1, 2), ev["vals"], reduction="none")
        rep = ~first_occurrence(ev["keys"], task.K)
        base_ce = float(ce[rep].mean())
        pce, n = 0.0, 0
        for n_c, n_w, G in [(2, 1, 16), (4, 2, 32), (8, 3, 64), (1, 4, 8)]:
            tok, keys, vals, c, w = probe_batch(task, n_c, n_w, G, 64, g, dev)
            lg = value_logits(task, model(tok)).float()
            ce = F.cross_entropy(lg.transpose(1, 2), vals, reduction="none")
            tau = keys[:, -1:]
            bg = (keys != tau) & ~first_occurrence(keys, task.K)  # background repeat sightings only
            pce += float(ce[bg].sum()); n += int(bg.sum())
        rows.append({"run": os.path.basename(d), "repeat_ce_ordinary": base_ce, "repeat_ce_probe_background": pce / n})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df["diff"] = df.repeat_ce_probe_background - df.repeat_ce_ordinary
    df.to_csv(os.path.join(out, "ood.csv"), index=False)
    print(f"probe-background minus ordinary repeat-sighting loss: median {df['diff'].median():+.4f} nats, "
          f"max |diff| {df['diff'].abs().max():.4f}")


@torch.no_grad()
def precision(out, dev):
    bf = BayesFree(LAY4)
    dirs = []
    for p in ["E1_h0.01_e0.1_L2_alibi_learn_attn_s*", "E1_h0.03_e0.2_L2_alibi_learn_mlp_s*",
              "E1_h0.01_e0.1_L3_alibi_learn_attn_s*", "E2a_h0.01_e0.1_L2_rope_attn_s*",
              "E2a_h0.03_e0.2_L2_learned_abs_attn_s*", "E2b_h0.003_e0.1_L2_alibi_learn_attn_s*"]:
        dirs += pick([p])[:2]
    m = np.ones(len(CELLS), bool)
    rows = []
    for d in dirs:
        model, task = load_model(d, dev)
        y16 = surface(model, task, fills=256, autocast=True, dev=dev)
        y32 = surface(model, task, fills=256, autocast=False, dev=dev)
        p16, p32 = bf.fit(y16, m), bf.fit(y32, m)
        rows.append({"run": os.path.basename(d), "max_abs_diff": float(np.abs(y16 - y32).max()),
                     "mean_abs_diff": float(np.abs(y16 - y32).mean()), "h16": p16["h"], "h32": p32["h"],
                     "e16": p16["eps"], "e32": p32["eps"],
                     "r2_16": r2(BayesFree.predict(LAY4, p16), y16), "r2_32": r2(BayesFree.predict(LAY4, p32), y32)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "precision.csv"), index=False)
    ok = df[(df.h16 > 0) & (df.h32 > 0)]
    print(f"bf16 vs fp32: mean |LO diff| median {df.mean_abs_diff.median():.4f}; h' ratio median "
          f"{np.median(ok.h32 / ok.h16):.3f} (range {np.min(ok.h32 / ok.h16):.3f}-{np.max(ok.h32 / ok.h16):.3f}); "
          f"eps' ratio median {np.median(df.e32 / df.e16):.3f}")


@torch.no_grad()
def slope(out, dev):
    bf = BayesFree(LAY4)
    m = np.ones(len(CELLS), bool)
    scales = (0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0)
    rows = []
    for d in sorted(glob.glob(os.path.join(R, "E*_*"))):
        cfg = json.load(open(os.path.join(d, "config.json")))
        if cfg["pe"] != "alibi_learn" or cfg["h"] == 0 or cfg["eps"] == 0 or not os.path.exists(os.path.join(d, "ckpt.pt")):
            continue
        if cfg.get("typo", "structured") != "structured":
            continue
        model, task = load_model(d, dev)
        g = torch.Generator(device=dev).manual_seed(4321)
        ev = sample_batch(task, 128, g, dev)
        rep = ~first_occurrence(ev["keys"], task.K)
        for s in scales:
            m2 = copy.deepcopy(model)
            for blk in m2.blocks[1:]:
                blk.attn.log_slope.data += math.log(s)
            lg = value_logits(task, m2(ev["tokens"])).float()
            ce = F.cross_entropy(lg.transpose(1, 2), ev["vals"], reduction="none")
            y = surface(m2, task, fills=64, dev=dev)
            p = bf.fit(y, m) if y.var() > 0.02 else {"h": np.nan, "eps": np.nan}
            rows.append({"run": os.path.basename(d), "h": cfg["h"], "eps": cfg["eps"], "layers": cfg["layers"],
                         "mlp": cfg["mlp"], "scale": s, "h_eff": p["h"], "eps_eff": p["eps"],
                         "ce_rep": float(ce[rep].mean())})
            del m2
        print(os.path.basename(d), [round(r["h_eff"], 4) for r in rows[-len(scales):]], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "slope_sweep.csv"), index=False)
    ks, r2s = [], []
    for run, g in df.groupby("run"):
        g = g[(g.h_eff > 0) & g.h_eff.notna()]
        if len(g) >= 5:
            x, yv = np.log(g.scale), np.log(g.h_eff)
            k, c0 = np.polyfit(x, yv, 1)
            ks.append(k)
            r2s.append(1 - ((yv - (k * x + c0)) ** 2).sum() / ((yv - yv.mean()) ** 2).sum())
    ks, r2s = np.array(ks), np.array(r2s)
    print(f"elasticity k: median {np.median(ks):.3f} [IQR {np.quantile(ks, .25):.3f}-{np.quantile(ks, .75):.3f}], "
          f"per-model R² median {np.median(r2s):.3f}; pre-registered: R² > 0.9 and k in [0.7, 1.3]")
    base = df[df.scale == 1.0].set_index("run").ce_rep
    curv = df.assign(d_ce=df.ce_rep - df.run.map(base)).groupby("scale").d_ce.median()
    print("median repeat-sighting loss change vs scale:\n" + curv.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["ood", "precision", "slope"])
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    {"ood": ood, "precision": precision, "slope": slope}[a.check](a.out, dev)


if __name__ == "__main__":
    main()
