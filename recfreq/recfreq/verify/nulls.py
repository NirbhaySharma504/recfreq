"""Audit 4.3 (pre-registered, plan §17): is Bayes(h', eps') a generic fitter of smooth surfaces?

1. generic smooth monotone surfaces: lo = A*tanh(a0 + a1*log2 G + a2*n_w + a3*log2 n_c) + B, signs as in the
   models (more gap and more recent copies push toward w; more old copies toward c), plus probe-level noise;
2. the trained models' surfaces with the 112 cells shuffled.
Bayes(h', eps') must fit (1) clearly worse than it fits the models (median R² < 0.80) and (2) hardly at all (< 0.2).

python -m recfreq.verify.nulls --n 300 --out verify_out
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from ..layouts import CELLS, LAY4, BayesFree, fit_counter, r2

NOISE = 0.04
_BF = None
CELL = np.array(CELLS, float)


def _init():
    global _BF
    _BF = BayesFree(LAY4)


def generic(seed):
    rng = np.random.default_rng(seed)
    A = rng.uniform(1, 5)
    a0 = rng.uniform(-1, 4)
    a1 = -rng.uniform(0.1, 1.5)   # more gap -> toward w
    a2 = -rng.uniform(0.1, 1.5)   # more recent copies -> toward w
    a3 = rng.uniform(0.0, 1.0)    # more old copies -> toward c
    B = rng.uniform(-1, 1)
    y = A * np.tanh(a0 + a1 * np.log2(CELL[:, 2]) + a2 * CELL[:, 1] + a3 * np.log2(CELL[:, 0])) + B
    return y + rng.normal(0, NOISE, len(y))


def fit_both(y):
    m = np.ones(len(y), bool)
    p = _BF.fit(y, m)
    return r2(BayesFree.predict(LAY4, p), y), r2(LAY4.counter(fit_counter(LAY4, y, m)), y)


def work(job):
    kind, payload = job
    if kind == "generic":
        y = generic(payload)
    else:
        name, y, seed = payload
        y = np.random.default_rng(seed).permutation(y)
    if y.var() < 0.05:
        return None
    rb, rc = fit_both(y)
    return {"kind": kind, "r2_bayes": rb, "r2_counter": rc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [("generic", s) for s in range(a.n)]
    for pq in sorted(glob.glob(os.path.join(a.runs, "E*_*", "probes.parquet"))):
        d = pd.read_parquet(pq).set_index(["n_c", "n_w", "G"]).loc[CELLS]
        y = d.lo_model.to_numpy()
        if y.var() >= 0.05:
            jobs.append(("shuffled", (os.path.basename(os.path.dirname(pq)), y, 0)))
    with mp.Pool(a.procs, initializer=_init) as pool:
        res = [r for r in pool.map(work, jobs) if r is not None]
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(a.out, "nulls.csv"), index=False)
    print(df.groupby("kind")[["r2_bayes", "r2_counter"]].describe(percentiles=[.1, .5, .9]).round(3).T.to_string())
    g = df[df.kind == "generic"].r2_bayes.median()
    s = df[df.kind == "shuffled"].r2_bayes.median()
    print(f"\npre-registered: generic median R² {g:.3f} (pass if < 0.80); shuffled median R² {s:.3f} (pass if < 0.2)")


if __name__ == "__main__":
    main()
