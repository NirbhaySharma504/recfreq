"""Plan §19.6: is the miscalibration in time or in evidence?

The survival probes say a single old copy is believed as Bayes would believe it (the clock is right). If that is the
whole story for time, then Bayes with the TRUE h and a fitted eps' (plus the usual a + b scale) should describe the
probe surface almost as well as the free fit (h', eps', a, b). The reverse restriction (true eps, fitted h') tells
how much the free fit's h' is needed. In-sample RMSE on the 112 standard probe cells for:
  free    h', eps', a, b        (BayesFree, as in the law fits)
  clock   h fixed, eps', a, b   ("right clock, mis-weighted evidence")
  evid    eps fixed, h', a, b   ("right evidence, wrong clock")
  true    h, eps fixed, a, b
  clockV  h fixed, V_eff, eps', a, b  (the value prior 1/V shrunk instead of the clock sped up; V_eff on a grid)

python -m recfreq.verify.clock_fixed --out verify_out
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..layouts import CELLS, BayesFree, Lay
from .evidence_weight import DEFAULT


VGRID = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]


def affine_rmse(x, y):
    A = np.stack([np.ones_like(x), x], 1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.sqrt(((A @ c - y) ** 2).mean())), c


def best_1d(f, lo, hi, n=60):
    """Minimise f over [lo, hi]: coarse grid, then bounded refinement around the best grid point."""
    g = np.linspace(lo, hi, n)
    v = [f(t) for t in g]
    i = int(np.argmin(v))
    r = minimize_scalar(f, bounds=(g[max(i - 1, 0)], g[min(i + 1, n - 1)]), method="bounded")
    return (r.x, r.fun) if r.fun < v[i] else (g[i], v[i])


def fit_model(y, h, eps, lay, bf):
    sig = lambda t: 1 / (1 + np.exp(-t))
    out = {}
    p = bf.fit(y, np.ones(len(y), bool))
    out["rmse_free"] = float(np.sqrt(((BayesFree.predict(lay, p) - y) ** 2).mean()))
    out["h_free"], out["eps_free"] = p["h"], p["eps"]
    f = lambda t: affine_rmse(lay.bayes([h], [sig(t)])[0], y)[0]
    t, v = best_1d(f, np.log(1e-3 / (1 - 1e-3)), np.log(0.49 / 0.51))
    out["rmse_clock"], out["eps_clock"] = v, float(sig(t))
    out["b_clock"] = float(affine_rmse(lay.bayes([h], [sig(t)])[0], y)[1][1])
    f = lambda t: affine_rmse(lay.bayes([float(np.exp(t))], [eps])[0], y)[0]
    t, v = best_1d(f, np.log(1e-5), np.log(0.7))
    out["rmse_evid"], out["h_evid"] = v, float(np.exp(t))
    out["rmse_true"] = affine_rmse(lay.bayes([h], [eps])[0], y)[0]
    best = (np.inf, None, None)
    for V in VGRID:
        lv = Lay(lay.sp, lay.swap, V)
        f = lambda t: affine_rmse(lv.bayes([h], [sig(t)])[0], y)[0]
        t, v = best_1d(f, np.log(1e-3 / (1 - 1e-3)), np.log(0.49 / 0.51), n=30)
        if v < best[0]:
            best = (v, V, float(sig(t)))
    out["rmse_clockV"], out["V_eff"], out["eps_clockV"] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--prefixes", default="E1_,E2b_,E5a_,E6_,E2a_,E5r_")
    a = ap.parse_args()
    lay = Lay(4, False, 64)
    bf = BayesFree(lay)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.runs, "*/done.json"))):
        name = f.split("/")[-2]
        if not name.startswith(tuple(a.prefixes.split(","))):
            continue
        r = json.load(open(f))["args"]
        if r["layers"] != 2 or r["mlp"] or r.get("typo", "structured") != "structured" or r["h"] <= 0 or r["eps"] <= 0:
            continue
        if any(r.get(k, v) != v for k, v in {**DEFAULT, "conflict_frac": 0.0}.items()):
            continue
        y = pd.read_parquet(os.path.join(os.path.dirname(f), "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS]
        y = y.lo_model.to_numpy()
        if y.var() < 1e-3:
            continue
        rows.append({"run": name, "pe": r["pe"], "h": r["h"], "eps": r["eps"], "sd_y": float(y.std()),
                     **fit_model(y, r["h"], r["eps"], lay, bf)})
        print(name, {k: round(v, 3) for k, v in rows[-1].items() if k.startswith("rmse")}, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "clock_fixed.csv"), index=False)
    for k in ("clock", "evid", "true", "clockV"):
        D[f"x_{k}"] = D[f"rmse_{k}"] / D.rmse_free
    pd.set_option("display.width", 220)
    print("\nRMSE relative to the free fit, median per (pe, h):")
    print(D.groupby(["pe", "h"])[["x_clock", "x_evid", "x_true", "x_clockV", "V_eff"]].median().round(2).to_string())
    print("\noverall medians:", D.groupby("pe")[["x_clock", "x_evid", "x_true", "x_clockV"]].median().round(2).to_dict("index"))


if __name__ == "__main__":
    main()
