"""Plan §18 post-hoc: under uniform typos, is "counters beat Bayes(h', eps')" an artifact of capping eps' at 0.5?

The Bayes-free fits cap eps' at 0.5, which is right for structured typos (above 0.5 the typo variant would be
more likely than the true value) but not for uniform typos, where any eps < (V-1)/V is a valid channel.
Refit every uniform-typo model with eps' allowed up to 0.95 and compare leave-one-G-out RMSE with the counters.

python -m recfreq.verify.uniform_typo_check
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .. import fit_rules
from ..layouts import CELLS, Lay, r2
from ..probes import GAPS


def fit_wide(lay, y, mask, table, H, E):
    X, Y = table[:, mask], y[mask]
    xm, ym = X.mean(1, keepdims=True), Y.mean()
    b = ((X - xm) * (Y - ym)).sum(1) / np.clip(((X - xm) ** 2).sum(1), 1e-12, None)
    a = ym - b * xm[:, 0]
    sse = ((a[:, None] + b[:, None] * X - Y) ** 2).sum(1)
    g = int(np.argmin(sse))
    h0 = H[g]
    fix0 = h0 == 0.0

    def unpack(t):
        return (0.0 if fix0 else math.exp(t[0])), 1 / (1 + math.exp(-t[1]))

    def resid(t):
        h, e = unpack(t)
        return (t[2] + t[3] * lay.bayes([h], [e], "uniform")[0] - y)[mask]

    t0 = np.array([math.log(1e-4) if fix0 else math.log(h0), math.log(E[g] / (1 - E[g])), a[g], b[g]])
    lo = [t0[0] - 1e-9 if fix0 else math.log(1e-5), -9, -30, -30]
    hi = [t0[0] + 1e-9 if fix0 else math.log(0.7), math.log(0.95 / 0.05), 30, 30]
    r = least_squares(resid, t0, bounds=(lo, hi), max_nfev=200)
    t = r.x if 2 * r.cost <= sse[g] else t0
    h, e = unpack(t)
    return {"h": h, "eps": e, "a": t[2], "b": t[3]}


def main():
    lay = Lay(4, False, 64)
    hg = np.concatenate([[0.0], np.exp(np.linspace(np.log(1e-4), np.log(0.5), 30))])
    eg = 1 / (1 + np.exp(-np.linspace(math.log(1e-3 / (1 - 1e-3)), math.log(0.95 / 0.05), 40)))
    Hm, Em = np.meshgrid(hg, eg, indexing="ij")
    H, E = Hm.ravel(), Em.ravel()
    table = lay.bayes(H, E, "uniform")
    Gs = np.array([c[2] for c in CELLS])
    rows = []
    for done in sorted(glob.glob("results/runs/*/done.json")):
        a = json.load(open(done))["args"]
        if a.get("typo") != "uniform" or a.get("V", 64) != 64:
            continue
        d = os.path.dirname(done)
        y = pd.read_parquet(os.path.join(d, "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS].lo_model.to_numpy()
        res = {}
        for name, fit, pred in (
            ("bayes_wide", lambda m: fit_wide(lay, y, m, table, H, E), lambda p: p["a"] + p["b"] * lay.bayes([p["h"]], [p["eps"]], "uniform")[0]),
            ("counter_exp", lambda m: fit_rules.fit_counter(y, m, "exp"), lambda p: fit_rules.counter_pred(p, "exp")),
            ("counter_pow", lambda m: fit_rules.fit_counter(y, m, "pow"), lambda p: fit_rules.counter_pred(p, "pow"))):
            p = fit(np.ones(len(CELLS), bool))
            cv = np.empty(len(CELLS))
            for G in GAPS:
                m = Gs != G
                cv[~m] = pred(fit(m))[~m]
            res[name] = (p, r2(pred(p), y), float(np.sqrt(((cv - y) ** 2).mean())))
        pb = res["bayes_wide"][0]
        rows.append({"run": os.path.basename(d), "h": a["h"], "eps": a["eps"], "layers": a["layers"],
                     "h_fit": pb["h"], "eps_fit": pb["eps"], "b_fit": pb["b"],
                     "bayes_wide_r2": res["bayes_wide"][1], "bayes_wide_cv": res["bayes_wide"][2],
                     "cexp_cv": res["counter_exp"][2], "cpow_cv": res["counter_pow"][2]})
    df = pd.DataFrame(rows)
    df["bayes_wins"] = df.bayes_wide_cv < df[["cexp_cv", "cpow_cv"]].min(1)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string())
    print(f"\nBayes (eps' up to 0.95) beats both counters in {df.bayes_wins.sum()}/{len(df)} uniform-typo models")
    df.to_csv("verify_out/uniform_typo_check.csv", index=False)


if __name__ == "__main__":
    main()
