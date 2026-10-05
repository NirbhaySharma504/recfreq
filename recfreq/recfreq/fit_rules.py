"""Which decision rule do trained models implement? (Prop. 1 vs Prop. 2 fits on probe surfaces.)

For every run with probes, fit four families to the model's mean log p(c)/p(w) over the
112 probe cells (n_c x n_w x G) and score them on held-out gaps (leave-one-G-out):

  bayes_true   a + b * LO_Bayes(h, eps)                               2 params
  bayes_free   a + b * LO_Bayes(h', eps')  with h', eps' fitted         4 params
  counter_exp  delta + gamma * (S_c - (1-kappa) S_w) / (S_c + S_w + Z0),
               S = sum over copies of exp(-lam * d)                    5 params
  counter_pow  same with S = sum d^(-alpha)                             5 params

d is the distance in pairs from the query to each copy. counter_* is Prop. 1 in its
probability (normalized-attention) form; kappa lets a copied w also vote for c, which a
head can learn under structured typos.

python -m recfreq.fit_rules results/runs --exp E1,E2a,E2b,E2c,E2u --out figs_fit
"""
from __future__ import annotations

import argparse
import itertools
import math
import multiprocessing as mp
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .analyze import load
from .probes import GAPS, NC, NW

warnings.filterwarnings("ignore")
V, N, SP = 64, 256, 4
CELLS = list(itertools.product(NC, NW, GAPS))


def layout(n_c: int, n_w: int, G: int, sp: int = SP):
    """Distances (pairs) from the query to the c copies and w copies, as in probes.probe_batch."""
    w_d = np.array([sp * j for j in range(1, n_w + 1)], float)
    first_w = sp * n_w
    c_d = np.array([first_w + G + sp * i for i in range(n_c)], float)
    return c_d, w_d


LAYOUTS = [layout(*c) for c in CELLS]
# padded distance matrices for vectorized counters: (cells, max copies); padding gets distance inf
DC = np.full((len(CELLS), max(NC)), np.inf)
DW = np.full((len(CELLS), max(NW)), np.inf)
for _k, (_c, _w) in enumerate(LAYOUTS):
    DC[_k, : len(_c)] = _c
    DW[_k, : len(_w)] = _w


def bayes_lo_grid(hs: np.ndarray, es: np.ndarray, typo: str = "structured") -> np.ndarray:
    """Exact Bayes log p(c)/p(w) at the query for every (h, eps) pair in hs, es (same length) and
    every probe cell. Returns (len(hs), len(CELLS)). c = 0, w = 1 (= T(c) for structured typos)."""
    P = len(hs)
    eye = np.eye(V)
    out = np.empty((P, len(CELLS)))
    hs, es = np.asarray(hs, float)[:, None], np.asarray(es, float)[:, None]
    # emission columns we need: P(y | z) for y in {0, 1}
    if typo == "structured":
        def em(y):  # y = z w.p. 1-eps, y = z+1 w.p. eps
            col = np.zeros(V); col[y] = 1.0
            sh = np.zeros(V); sh[(y - 1) % V] = 1.0
            return (1 - es) * col[None] + es * sh[None]
    else:
        def em(y):
            col = np.zeros(V); col[y] = 1.0
            return (1 - es) * col[None] + es * (1 - col[None]) / (V - 1)
    E0, E1 = em(0), em(1)
    for k, (c_d, w_d) in enumerate(LAYOUTS):
        obs = sorted([(d, 0) for d in c_d] + [(d, 1) for d in w_d], key=lambda t: -t[0])  # oldest first
        b = np.full((P, V), 1.0 / V)
        prev = None
        for d, y in obs:
            if prev is not None:
                s = (1 - hs) ** (prev - d)
                b = s * b + (1 - s) / V
            b = b * (E0 if y == 0 else E1)
            b /= b.sum(1, keepdims=True)
            prev = d
        s = (1 - hs) ** prev  # query is at distance 0
        prior = s * b + (1 - s) / V
        pc = (prior * E0).sum(1)  # P(y=0) = sum_z prior(z) P(0|z)
        pw = (prior * E1).sum(1)
        out[:, k] = np.log(np.clip(pc, 1e-300, None)) - np.log(np.clip(pw, 1e-300, None))
    return out


def counter_pred(theta: np.ndarray, kernel: str) -> np.ndarray:
    lp, gamma, kappa, lz, delta = theta
    p, z0 = math.exp(lp), math.exp(lz)
    if kernel == "exp":
        sc, sw = np.exp(-p * DC).sum(1), np.exp(-p * DW).sum(1)
    else:
        sc, sw = (DC ** -p).sum(1), (DW ** -p).sum(1)  # inf ** -p = 0 for the padding
    return delta + gamma * (sc - (1 - kappa) * sw) / (sc + sw + z0)


def additive_pred(theta: np.ndarray) -> np.ndarray:
    """Unnormalized evidence sum with separate forgetting for old and recent copies:
    delta + g_c * sum_c exp(-lam_c d) - g_w * sum_w exp(-lam_w d)."""
    lc, lw, gc, gw, delta = theta
    return delta + gc * np.exp(-math.exp(lc) * DC).sum(1) - gw * np.exp(-math.exp(lw) * DW).sum(1)


def fit_additive(y: np.ndarray, mask: np.ndarray):
    best = None
    for lc0, lw0 in itertools.product(np.log([0.005, 0.03, 0.15]), np.log([0.005, 0.03, 0.15])):
        x0 = np.array([lc0, lw0, 1.0, 1.0, 0.0])
        r = least_squares(lambda t: (additive_pred(t) - y)[mask], x0,
                          bounds=([-9, -9, -50, -50, -30], [3, 3, 50, 50, 30]), max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def fit_counter(y: np.ndarray, mask: np.ndarray, kernel: str):
    best = None
    lps = np.log([0.005, 0.02, 0.06, 0.2, 0.6]) if kernel == "exp" else np.log([0.3, 1.0, 2.0, 4.0, 8.0])
    for lp0, lz0, k0 in itertools.product(lps, np.log([0.01, 0.3, 3.0]), [0.0, 0.7]):
        x0 = np.array([lp0, 5.0, k0, lz0, 0.0])
        try:
            r = least_squares(lambda t: (counter_pred(t, kernel) - y)[mask], x0,
                              bounds=([-9, -60, -2, -12, -20], [3, 60, 2, 8, 20]), max_nfev=400)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def fit_affine(x: np.ndarray, y: np.ndarray, mask: np.ndarray):
    A = np.stack([np.ones(mask.sum()), x[mask]], 1)
    coef, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
    return coef


class BayesFree:
    """Grid over (h', eps'), affine fit per grid point, then local refinement."""

    def __init__(self, typo: str):
        self.typo = typo
        # h' = 0 is included explicitly: with uniform typos the Bayes rule is discontinuous at h = 0
        self.hg = np.concatenate([[0.0], np.exp(np.linspace(np.log(1e-4), np.log(0.5), 36))])
        self.eg = np.exp(np.linspace(np.log(1e-3), np.log(0.49), 36))
        H, E = np.meshgrid(self.hg, self.eg, indexing="ij")
        self.H, self.E = H.ravel(), E.ravel()
        self.table = bayes_lo_grid(self.H, self.E, typo)

    def fit(self, y: np.ndarray, mask: np.ndarray):
        # closed-form affine least squares for every grid point at once
        X, Y = self.table[:, mask], y[mask]
        xm, ym = X.mean(1, keepdims=True), Y.mean()
        b = ((X - xm) * (Y - ym)).sum(1) / np.clip(((X - xm) ** 2).sum(1), 1e-12, None)
        a = ym - b * xm[:, 0]
        sse = ((a[:, None] + b[:, None] * X - Y) ** 2).sum(1)
        g = int(np.argmin(sse))
        best = (float(sse[g]), (self.H[g], self.E[g], a[g], b[g]))
        h0, e0, a0, b0 = best[1]
        fix_h0 = h0 == 0.0  # stay exactly at h' = 0 and refine the rest

        def unpack(t):
            return (0.0 if fix_h0 else math.exp(t[0])), 1 / (1 + math.exp(-t[1]))

        def resid(t):
            h, e = unpack(t)
            return (t[2] + t[3] * bayes_lo_grid([h], [e], self.typo)[0] - y)[mask]

        t0 = np.array([math.log(1e-4) if fix_h0 else math.log(h0), math.log(e0 / (1 - e0)), a0, b0])
        lo = [math.log(1e-4) - 1e-9 if fix_h0 else math.log(1e-5), -9, -30, -30]
        hi = [math.log(1e-4) + 1e-9 if fix_h0 else math.log(0.7), 0, 30, 30]
        try:
            r = least_squares(resid, t0, bounds=(lo, hi), max_nfev=200)
            t = r.x if 2 * r.cost <= best[0] else t0
        except Exception:
            t = t0
        return np.append(t, float(fix_h0))

    def predict(self, t):
        h = 0.0 if t[4] else math.exp(t[0])
        x = bayes_lo_grid([h], [1 / (1 + math.exp(-t[1]))], self.typo)[0]
        return t[2] + t[3] * x


def evaluate_run(y: np.ndarray, h: float, eps: float, typo: str, bfree: BayesFree):
    """Fit all families on all cells and with leave-one-G-out; return metrics and parameters."""
    Gs = np.array([c[2] for c in CELLS])
    x_true = bayes_lo_grid([h], [eps], typo)[0]
    full = np.ones(len(CELLS), bool)
    fams = {
        "bayes_true": (lambda m: fit_affine(x_true, y, m), lambda p: p[0] + p[1] * x_true),
        "bayes_free": (lambda m: bfree.fit(y, m), bfree.predict),
        "counter_exp": (lambda m: fit_counter(y, m, "exp"), lambda p: counter_pred(p, "exp")),
        "counter_pow": (lambda m: fit_counter(y, m, "pow"), lambda p: counter_pred(p, "pow")),
        "additive": (lambda m: fit_additive(y, m), additive_pred),
    }
    res = {}
    vy = y.var()
    for name, (fit, pred) in fams.items():
        p_full = fit(full)
        yhat = pred(p_full)
        cv = np.empty(len(CELLS))
        for G in GAPS:
            m = Gs != G
            cv[~m] = pred(fit(m))[~m]
        sig = np.abs(y) > 0.25
        res[name] = {
            "r2": 1 - ((yhat - y) ** 2).mean() / vy,
            "rmse": float(np.sqrt(((yhat - y) ** 2).mean())),
            "cv_rmse": float(np.sqrt(((cv - y) ** 2).mean())),
            "cv_r2": 1 - ((cv - y) ** 2).mean() / vy,
            "sign_agree": float((np.sign(yhat[sig]) == np.sign(y[sig])).mean()) if sig.any() else float("nan"),
            "params": [float(v) for v in p_full],
            "yhat": yhat,
        }
    return res, x_true


_BFREE: dict = {}


def _work(args):
    h, eps, typo, y = args
    return evaluate_run(y, h, eps, typo, _BFREE[typo])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("runs_dir")
    ap.add_argument("--exp", default="E1,E2a,E2b,E2c,E2u")
    ap.add_argument("--out", default="figs_fit")
    ap.add_argument("--min_var", type=float, default=0.05, help="skip runs whose probe surface is flat")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load(a.runs_dir, a.exp)
    bfree = {t: BayesFree(t) for t in ("structured", "uniform")}
    rows, curves = [], {}
    # sanity: our numpy Bayes must reproduce the oracle values stored in the probes
    worst = 0.0
    for r in runs:
        if "probes" not in r:
            continue
        typo = "uniform" if "uniform" in r["arch"] else "structured"
        pr = r["probes"].set_index(["n_c", "n_w", "G"]).loc[CELLS]
        worst = max(worst, float(np.abs(bayes_lo_grid([r["h"]], [r["eps"]], typo)[0] - pr.lo_bayes.to_numpy()).max()))
    print(f"max |numpy Bayes - oracle| over all runs and cells: {worst:.2e}")
    jobs = []
    for r in runs:
        if "probes" not in r:
            continue
        y = r["probes"].set_index(["n_c", "n_w", "G"]).loc[CELLS].lo_model.to_numpy()
        if y.var() >= a.min_var:
            jobs.append((r, y))
    print(f"{len(jobs)} runs with a non-flat probe surface (of {len(runs)})", flush=True)
    global _BFREE
    _BFREE = bfree
    with mp.Pool(a.procs) as pool:
        results = pool.map(_work, [(r["h"], r["eps"], "uniform" if "uniform" in r["arch"] else "structured", y)
                                   for r, y in jobs])
    for (r, y), (res, x_true) in zip(jobs, results):
        row = {"run": r["run"], "exp": r["exp"], "h": r["h"], "eps": r["eps"], "arch": r["arch"], "seed": r["seed"],
               "lambda_hat": r["lambda_hat"], "var_y": float(y.var())}
        for fam, m in res.items():
            for k in ("r2", "rmse", "cv_rmse", "cv_r2", "sign_agree"):
                row[f"{fam}.{k}"] = m[k]
        pe = res["counter_exp"]["params"]
        row.update({"cexp.lam": math.exp(pe[0]), "cexp.gamma": pe[1], "cexp.kappa": pe[2], "cexp.Z0": math.exp(pe[3])})
        pp = res["counter_pow"]["params"]
        row.update({"cpow.alpha": math.exp(pp[0]), "cpow.kappa": pp[2]})
        pa = res["additive"]["params"]
        row.update({"add.lam_c": math.exp(pa[0]), "add.lam_w": math.exp(pa[1]), "add.g_c": pa[2], "add.g_w": pa[3]})
        pb = res["bayes_free"]["params"]
        row.update({"bfree.h": 0.0 if pb[4] else math.exp(pb[0]), "bfree.eps": 1 / (1 + math.exp(-pb[1])),
                    "bfree.offset": pb[2], "bfree.scale": pb[3]})
        row["btrue.scale"] = res["bayes_true"]["params"][1]
        rows.append(row)
        curves[r["run"]] = {"y": y, "bayes": x_true, **{f: res[f]["yhat"] for f in res}}
        print(f"{r['run']:<52} cvRMSE  Btrue {row['bayes_true.cv_rmse']:.3f}  Bfree {row['bayes_free.cv_rmse']:.3f}  "
              f"Cexp {row['counter_exp.cv_rmse']:.3f}  Cpow {row['counter_pow.cv_rmse']:.3f}  "
              f"Add {row['additive.cv_rmse']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "fits.csv"), index=False)
    np.savez_compressed(os.path.join(a.out, "curves.npz"),
                        **{f"{k}|{f}": v for k, d in curves.items() for f, v in d.items()})
    print(f"\n{len(df)} runs fitted -> {a.out}/fits.csv")


if __name__ == "__main__":
    main()
