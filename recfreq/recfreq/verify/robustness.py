"""Audit 3.4: robustness of the Bayes(h', eps') fits and of the laws built on them.

1. second optimizer: finer (h', eps') grid (61 x 60, plus h' = 0) and Nelder-Mead refinement; compare with fits.csv.
2. wild residual bootstrap (Rademacher signs on the fit residuals, 30 reps) per model: CI of h', eps'.
3. sign of the fitted scale b.
4. seed-resampled CIs of the laws log h' = alpha + beta log h and logit eps' = gamma + delta logit eps
   (2-layer ALiBi grid), and of the per-copy law.

python -m recfreq.verify.robustness --out verify_out
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..layouts import CELLS, LAY4, BayesFree, r2

_BF = None
_FINE = None


def _init():
    global _BF, _FINE
    _BF = BayesFree(LAY4)
    hg = np.concatenate([[0.0], np.exp(np.linspace(np.log(1e-4), np.log(0.5), 60))])
    eg = np.exp(np.linspace(np.log(1e-3), np.log(0.49), 60))
    H, E = np.meshgrid(hg, eg, indexing="ij")
    _FINE = (H.ravel(), E.ravel(), LAY4.bayes(H.ravel(), E.ravel()))


def fine_fit(y):
    H, E, T = _FINE
    xm = T.mean(1, keepdims=True)
    b = ((T - xm) * (y - y.mean())).sum(1) / np.clip(((T - xm) ** 2).sum(1), 1e-12, None)
    a = y.mean() - b * xm[:, 0]
    sse = ((a[:, None] + b[:, None] * T - y) ** 2).sum(1)
    g = int(np.argmin(sse))
    if H[g] == 0:
        return {"h": 0.0, "eps": E[g], "sse": sse[g]}

    def obj(t):
        x = LAY4.bayes([math.exp(t[0])], [1 / (1 + math.exp(-t[1]))])[0]
        A = np.stack([np.ones_like(x), x], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return ((A @ coef - y) ** 2).sum()

    r = minimize(obj, [math.log(H[g]), math.log(E[g] / (1 - E[g]))], method="Nelder-Mead",
                 options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 400})
    return {"h": math.exp(r.x[0]), "eps": 1 / (1 + math.exp(-r.x[1])), "sse": r.fun}


def work(job):
    run, y, reps = job
    m = np.ones(len(y), bool)
    p = _BF.fit(y, m)
    yh = BayesFree.predict(LAY4, p)
    ff = fine_fit(y)
    res = y - yh
    rng = np.random.default_rng(0)
    hs, es = [], []
    for _ in range(reps):
        yb = yh + res * rng.choice([-1.0, 1.0], size=len(y))
        pb = _BF.fit(yb, m)
        hs.append(pb["h"]); es.append(pb["eps"])
    hs, es = np.array(hs), np.array(es)
    return {"run": run, "h": p["h"], "eps": p["eps"], "b": p["b"], "r2": r2(yh, y),
            "fine_h": ff["h"], "fine_eps": ff["eps"],
            "h_lo": np.quantile(hs, 0.05), "h_hi": np.quantile(hs, 0.95),
            "eps_lo": np.quantile(es, 0.05), "eps_hi": np.quantile(es, 0.95)}


def logit(x):
    return np.log(x / (1 - x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="figs_fit/fits.csv")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f = pd.read_csv(a.fit)
    f = f[~f.arch.str.contains("uniform")]
    jobs = []
    for r in f.run:
        d = pd.read_parquet(os.path.join(a.runs, r, "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS]
        jobs.append((r, d.lo_model.to_numpy(), a.reps))
    with mp.Pool(a.procs, initializer=_init) as pool:
        df = pd.DataFrame(pool.map(work, jobs))
    df = df.merge(f[["run", "arch", "h", "eps", "exp"]].rename(columns={"h": "true_h", "eps": "true_eps"}), on="run")
    df.to_csv(os.path.join(a.out, "robustness.csv"), index=False)
    ok = df[(df.h > 0) & (df.fine_h > 0)]
    print(f"models: {len(df)}")
    print(f"1. second optimizer: |log(h'_fine / h')| median {np.median(np.abs(np.log(ok.fine_h / ok.h))):.3f}, "
          f"within 25% in {(np.abs(np.log(ok.fine_h / ok.h)) < math.log(1.25)).mean():.2f}; "
          f"eps' within 25% in {(np.abs(np.log(df.fine_eps / df.eps)) < math.log(1.25)).mean():.2f}")
    print(f"2. bootstrap 90% interval width for h' (ratio hi/lo): median {np.median(ok.h_hi / ok.h_lo.clip(1e-6)):.2f}; "
          f"for eps': median {np.median(df.eps_hi / df.eps_lo):.2f}")
    print(f"   h' interval excludes the true h in {(df.h_lo > df.true_h).mean():.2f} of models (h' > h significantly)")
    print(f"3. fitted scale b > 0 in {(df.b > 0).mean():.2f} of models (min b {df.b.min():.3f})")
    g = df[(df.arch == "L2") & df.exp.isin(["E1", "E2b"])]
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(2000):
        s = g.groupby(["true_h", "true_eps"]).sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))
        sm = s.groupby(["true_h", "true_eps"])[["h", "eps"]].median().reset_index()
        sm = sm[sm.h > 0]
        beta, alpha = np.polyfit(np.log(sm.true_h), np.log(sm.h), 1)
        delta, gamma = np.polyfit(logit(sm.true_eps), logit(sm.eps), 1)
        boots.append((math.exp(alpha), beta, gamma, delta))
    b = np.array(boots)
    q = lambda c: f"{np.median(b[:, c]):.2f} [{np.quantile(b[:, c], .025):.2f}, {np.quantile(b[:, c], .975):.2f}]"
    print(f"4. laws (seed-resampled, 95% CI): h' = {q(0)} * h^{q(1)};  logit eps' = {q(2)} + {q(3)} * logit eps")


if __name__ == "__main__":
    main()
