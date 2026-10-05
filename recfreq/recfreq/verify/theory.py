"""Audit 3.2: Prop. 2 against an independent exact Bayes computation (fresh code, not oracle.py / layouts.py).

For the probe layout (n_c old copies and n_w recent copies 4 pairs apart, query 4 pairs after the last recent
copy, structured typos, V = 64) compute the exact switch-over gap G* by bisection and compare it with the
Prop. 2 formula; check n_c-independence and the per-copy weight log((1-eps)/eps).

python -m recfreq.verify.theory --out verify_out
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

V = 64


def exact_lo(h, eps, n_c, n_w, G, sp=4):
    """log P(y=c)/P(y=w) at the query; independent forward filter over the probe key's observations."""
    obs = [(sp * n_w + G + sp * i, 0) for i in range(n_c)][::-1] + [(sp * j, 1) for j in range(n_w, 0, -1)]
    # observations ordered oldest first; values: 0 = c, 1 = w = c + 1
    b = np.full(V, 1.0 / V)
    prev = None
    for d, y in obs:
        if prev is not None:
            s = (1 - h) ** (prev - d)
            b = s * b + (1 - s) / V
        like = np.zeros(V)
        like[y] += 1 - eps          # z = y emitted correctly
        like[(y - 1) % V] += eps    # z = y - 1 emitted as the typo y
        b = b * like
        b = b / b.sum()
        prev = d
    s = (1 - h) ** prev
    prior = s * b + (1 - s) / V
    p_c = (1 - eps) * prior[0] + eps * prior[V - 1]
    p_w = (1 - eps) * prior[1] + eps * prior[0]
    return math.log(p_c) - math.log(p_w)


def exact_gstar(h, eps, n_w, n_c=2, gmax=4000.0):
    lo = lambda G: exact_lo(h, eps, n_c, n_w, G)
    if lo(0.0) < 0:
        return 0.0
    if lo(gmax) > 0:
        return math.inf
    a, b = 0.0, gmax
    for _ in range(60):
        m = (a + b) / 2
        if lo(m) > 0:
            a = m
        else:
            b = m
    return (a + b) / 2


def formula(h, eps, n_w):
    r = ((1 - eps) / eps) ** n_w / V
    s = r / (1 + r)
    return math.log(s) / math.log1p(-h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for h in (0.001, 0.003, 0.01, 0.03, 0.1):
        for eps in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4):
            for n_w in (1, 2, 3, 4):
                ge, gf = exact_gstar(h, eps, n_w), formula(h, eps, n_w)
                rows.append({"h": h, "eps": eps, "n_w": n_w, "G_exact": ge, "G_formula": gf,
                             "hG": h * gf, "ratio": ge / gf if 0 < ge < math.inf and gf > 0 else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "theory.csv"), index=False)
    ok = df.dropna(subset=["ratio"])
    ok = ok[(ok.G_formula >= 4)]  # below a few pairs the probe's own spacing dominates
    print(f"{len(ok)} (h, eps, n_w) cases with G* >= 4 pairs")
    for lim in (0.1, 0.3, 1.0, 3.0):
        sub = ok[ok.hG <= lim]
        print(f"  h*G* <= {lim}: n={len(sub)}, |exact/formula - 1| median {np.median(np.abs(sub.ratio - 1)):.3f}, "
              f"max {np.max(np.abs(sub.ratio - 1)):.3f}")
    # n_c independence
    worst = 0.0
    for h in (0.003, 0.01, 0.03):
        for eps in (0.05, 0.1, 0.2, 0.3):
            for n_w in (1, 2, 3):
                for G in (4, 16, 64):
                    v = [exact_lo(h, eps, n_c, n_w, G) for n_c in (2, 4, 8, 16)]
                    worst = max(worst, max(v) - min(v))
    print(f"n_c independence (n_c = 2..16): largest change in log-odds over the grid = {worst:.4f} nats")
    # per-copy weight: d log-odds / d n_w at fixed gap, small h
    dev = []
    for eps in (0.05, 0.1, 0.2, 0.3):
        v = [exact_lo(1e-5, eps, 2, n_w, 8) for n_w in (1, 2, 3, 4)]
        dev.append(np.mean(np.diff(v)) + math.log((1 - eps) / eps))
    print(f"per-copy weight at h = 1e-5: mean slope + log((1-eps)/eps) = {np.round(dev, 4).tolist()} (0 = exact)")


if __name__ == "__main__":
    main()
