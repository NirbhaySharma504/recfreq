"""Plan §19 round 2, hypothesis H-support: models are calibrated where ordinary training data contains the
probe's situation, and the miscalibration is extrapolation into situations training (almost) never shows.

For each (h, eps) setting, count in generator data how often a query's key history ends with a run of exactly
n_w sightings of w preceded by a sighting of c = w - 1 (the probe's conflict, any spacing), binned by the gap G
between that c and the first w (log2 bins matching the probe gaps). Rate = events per query position.
Then relate it, cell by cell, to the trained models' deviation from Bayes, |LO_model - LO_bayes|.

python -m recfreq.verify.support --out verify_out
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from ..data import TaskConfig, sample_batch
from ..probes import GAPS

BINS = np.array(GAPS, float)


def counts(h, eps, B=4000, seed=5, **task):
    cfg = TaskConfig(h=h, eps=eps, **task)
    bt = sample_batch(cfg, B, torch.Generator().manual_seed(seed), "cpu")
    keys, vals = bt["keys"].numpy(), bt["vals"].numpy()
    V = cfg.V
    C = np.zeros((4, len(BINS)))  # n_w 1..4 x G bin
    for b in range(B):
        for k in range(cfg.K):
            t = np.flatnonzero(keys[b] == k)
            if len(t) < 3:
                continue
            y = vals[b, t]
            for i in range(2, len(t)):  # query at sighting i, history = sightings < i
                w = y[i - 1]
                n = 1
                while n < i and y[i - 1 - n] == w:
                    n += 1
                if n > 4 or n >= i:
                    continue
                if y[i - 1 - n] != (w - 1) % V:
                    continue
                G = t[i - n] - t[i - 1 - n]
                j = int(np.argmin(np.abs(np.log2(BINS) - np.log2(G))))
                C[n - 1, j] += 1
    return C / (B * cfg.n_pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--runs", default="results/runs")
    a = ap.parse_args()
    F = pd.read_csv(os.path.join(a.out, "law_scope_all.csv"))
    F = F[(F.pe == "alibi_learn") & (F.V == 64) & (F.K == 16) & (F.N == 256) & (F.key_dist == "zipf") & (F.typo == "structured")]
    rows = []
    for (h, e), g in F.groupby(["h", "eps"]):
        C = counts(h, e)
        ps = [pd.read_parquet(os.path.join(a.runs, r, "probes.parquet")) for r in g.run]
        P = pd.concat(ps).groupby(["n_c", "n_w", "G"])[["lo_model", "lo_bayes"]].mean().reset_index()
        P = P[P.n_c == 2]
        for r in P.itertuples():
            rows.append({"h": h, "eps": e, "n_w": r.n_w, "G": r.G, "rate": C[r.n_w - 1, list(GAPS).index(r.G)],
                         "lo_model": r.lo_model, "lo_bayes": r.lo_bayes, "dev": abs(r.lo_model - r.lo_bayes)})
        print(h, e, "conflict rate per query by n_w:", np.round(C.sum(1), 6).tolist(), flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "support.csv"), index=False)
    D["lr"] = np.log10(D.rate + 1e-7)
    print("\nSpearman(dev, log rate) over all cells:", round(D[["dev", "lr"]].corr("spearman").iloc[0, 1], 3))
    print("within-setting Spearman, median:", round(D.groupby(["h", "eps"]).apply(lambda d: d[["dev", "lr"]].corr("spearman").iloc[0, 1]).median(), 3))
    D["bin"] = pd.cut(D.lr, [-8, -6, -5, -4, -3, -2, 0])
    print(D.groupby("bin", observed=True).dev.agg(["count", "median", "mean"]).round(3))


if __name__ == "__main__":
    main()
