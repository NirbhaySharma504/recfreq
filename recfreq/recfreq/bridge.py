"""Is 'Bayes-shaped with fitted (h', eps')' informative, or does any sum of counters look like that?

Simulate probe surfaces from random single counters and random sums of 2 or 3 normalized exponential
counters (the mechanism we measured), add cell-level noise like the models', and score how well each
description family fits them. Compare with the families' fits to the trained models.

python -m recfreq.bridge --n 200 --out figs_run3/bridge.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

from .layouts import LAY4, BayesFree, fit_counter, fit_counter2, r2

_BF = None


def simulate(kind: str, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    k = {"single": 1, "sum2": 2, "sum3": 3}[kind]
    kappa = float(rng.uniform(0, 1)) if rng.random() < 0.5 else 0.0
    y = np.zeros(len(LAY4.dists))
    lams = []
    for _ in range(k):
        lam = float(np.exp(rng.uniform(np.log(0.01), np.log(0.8))))
        gamma = float(rng.uniform(1, 5))
        z0 = float(np.exp(rng.uniform(np.log(0.01), np.log(3))))
        y += LAY4._norm_counter(lam, gamma, kappa, z0)
        lams.append(lam)
    return y, {"kind": kind, "lams": lams, "kappa": kappa}


def work(args):
    kind, seed, noise = args
    rng = np.random.default_rng(seed)
    y, meta = simulate(kind, rng)
    y = y + rng.normal(0, noise, size=y.shape)
    if y.var() < 0.05:
        return None
    m = np.ones(len(y), bool)
    pb = _BF.fit(y, m)
    out = {"kind": kind, "seed": seed, "var": float(y.var()), "kappa": meta["kappa"],
           "lam_min": min(meta["lams"]), "lam_max": max(meta["lams"]),
           "r2_bayes_free": r2(BayesFree.predict(LAY4, pb), y), "h_eff": pb["h"], "eps_eff": pb["eps"],
           "r2_counter": r2(LAY4.counter(fit_counter(LAY4, y, m)), y),
           "r2_counter2": r2(LAY4.counter2(fit_counter2(LAY4, y, m)), y)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--noise", type=float, default=0.1, help="cell-mean noise sd, similar to the models' probes")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default="figs_run3/bridge.csv")
    a = ap.parse_args()
    global _BF
    _BF = BayesFree(LAY4)
    jobs = [(kind, 1000 * i + j, a.noise) for i, kind in enumerate(["single", "sum2", "sum3"]) for j in range(a.n)]
    with mp.Pool(a.procs) as pool:
        res = [r for r in pool.map(work, jobs) if r is not None]
    df = pd.DataFrame(res)
    df.to_csv(a.out, index=False)
    print(df.groupby("kind")[["r2_bayes_free", "r2_counter", "r2_counter2"]].describe(percentiles=[.1, .5, .9])
          .round(3).T.to_string())


if __name__ == "__main__":
    main()
