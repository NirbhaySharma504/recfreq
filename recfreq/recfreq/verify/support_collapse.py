"""Plan §19.2 R2-P2: does the support model fitted on the old settings predict the new (E6) models' deviations?

Model: dev = cell fixed effect (n_w, G) + beta * log10(rate + 1e-7), fitted on the 19 old settings in
verify_out/support.csv (n_c = 2 cells; rate = generator rate of the probe's conflict configuration per query).
Prediction for each E6 seed-0..k model cell: the old fixed effect + the old beta times the E6 setting's own rate.
Pre-registered pass: Spearman(predicted, observed) >= 0.6 over all E6 cells, and beta refitted on E6 alone
(with its own cell effects) within +-50% of -1.22.

python -m recfreq.verify.support_collapse --exp E6
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..probes import GAPS
from .support import counts


def fit_fe(D):
    """Least squares dev ~ cell dummies + beta * lr; returns (fixed effects by cell, beta)."""
    cells = sorted(set(zip(D.n_w, D.G)))
    idx = {c: i for i, c in enumerate(cells)}
    X = np.zeros((len(D), len(cells) + 1))
    X[np.arange(len(D)), [idx[c] for c in zip(D.n_w, D.G)]] = 1
    X[:, -1] = D.lr
    coef, *_ = np.linalg.lstsq(X, D.dev.values, rcond=None)
    return {c: coef[i] for c, i in idx.items()}, coef[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E6")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    old = pd.read_csv(os.path.join(a.out, "support.csv"))
    old["lr"] = np.log10(old.rate + 1e-7)
    fe, beta = fit_fe(old)
    print(f"old settings: beta = {beta:.3f} (n = {len(old)} cells)")
    rows, rates = [], {}
    for f in sorted(glob.glob(os.path.join(a.runs, f"{a.exp}_*/done.json"))):
        r = json.load(open(f))["args"]
        if r.get("pe") != "alibi_learn" or r["layers"] != 2 or r["mlp"]:
            continue
        key = (r["h"], r["eps"])
        if key not in rates:
            rates[key] = counts(*key)
        C = rates[key]
        P = pd.read_parquet(os.path.join(os.path.dirname(f), "probes.parquet"))
        for x in P[P.n_c == 2].itertuples():
            rows.append({"run": f.split("/")[-2], "h": key[0], "eps": key[1], "n_w": x.n_w, "G": x.G,
                         "rate": C[x.n_w - 1, list(GAPS).index(x.G)], "dev": abs(x.lo_model - x.lo_bayes)})
    D = pd.DataFrame(rows)
    D["lr"] = np.log10(D.rate + 1e-7)
    D["pred"] = [fe[(n, g)] + beta * l for n, g, l in zip(D.n_w, D.G, D.lr)]
    rho = spearmanr(D.pred, D.dev).statistic
    _, beta_new = fit_fe(D)
    D.to_csv(os.path.join(a.out, f"support_collapse_{a.exp}.csv"), index=False)
    print(f"{a.exp}: {D.run.nunique()} models, {len(D)} cells")
    print(f"R2-P2a Spearman(predicted, observed dev) = {rho:.3f} (criterion >= 0.6) -> {'PASS' if rho >= 0.6 else 'FAIL'}")
    ok = 0.5 * 1.22 <= -beta_new <= 1.5 * 1.22
    print(f"R2-P2b beta refitted on {a.exp} = {beta_new:.3f} (criterion -0.61 .. -1.83) -> {'PASS' if ok else 'FAIL'}")
    print("baseline: Spearman(cell effect alone, dev) =", round(spearmanr([fe[(n, g)] for n, g in zip(D.n_w, D.G)], D.dev).statistic, 3))


if __name__ == "__main__":
    main()
