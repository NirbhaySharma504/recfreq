"""Plan §19.2 R2-P1 and §19.3 R2-P5: conflict dose (E7) against deviation from Bayes, beliefs and evidence weight.

Settings C = (h 0.01, eps 0.1) and D = (h 0.03, eps 0.2); doses f = 0 (E1), 0.2 (E2c), 0.01/0.03/0.1/0.4 (E7).
R2-P1 (each setting, over the six doses):
  a. seed-mean median over cells of |LO_model - LO_bayes| falls with f (Spearman <= -0.8)
  b. h'/h at f = 0.4 below h'/h at f = 0
  c. |logit eps' - logit eps| at f = 0.4 below its value at f = 0
R2-P5: per f, fit log G* = c0 + C1 log(1/h) + C2 log(1/r_B) on C and D pooled (crossings, standard probes,
n_c = 2); C2 rises with f (Spearman >= 0.8) and C2(0.4) > 1.5 C2(0).

python -m recfreq.verify.dose
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .law_scope import fit_one
from .natural_law import log_rb
from .switch_points import crossing

SETTINGS = {"C": (0.01, 0.1), "D": (0.03, 0.2)}


def collect(runs="results/runs"):
    rows = []
    for f in sorted(glob.glob(os.path.join(runs, "*/done.json"))):
        name = f.split("/")[-2]
        if not name.startswith(("E1_", "E2c_", "E7_")):
            continue
        a = json.load(open(f))["args"]
        if a["layers"] != 2 or a["mlp"] or a.get("pe") != "alibi_learn":
            continue
        s = [k for k, v in SETTINGS.items() if (a["h"], a["eps"]) == v]
        if not s or any(a.get(k, v) != v for k, v in dict(V=64, K=16, n_pairs=256, steps=20000).items()):
            continue
        d = os.path.dirname(f)
        P = pd.read_parquet(os.path.join(d, "probes.parquet"))
        r = {"run": name, "set": s[0], "h": a["h"], "eps": a["eps"], "f": a.get("conflict_frac", 0.0),
             "seed": a["seed"], "dev": float(np.median(np.abs(P.lo_model - P.lo_bayes)))}
        P2 = P[P.n_c == 2]
        for nw in (1, 2, 3, 4):
            x = P2[P2.n_w == nw].sort_values("G")
            r[f"G{nw}"] = crossing(x.G.values.astype(float), x.lo_model.values)
        fit = fit_one({"run": name, "dir": d, "h": a["h"], "eps": a["eps"], "V": 64, "typo": "structured"})
        r.update({"h_fit": fit.get("h_fit"), "eps_fit": fit.get("eps_fit"), "b_fit": fit.get("b_fit")})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    cache = "verify_out/dose_models.csv"
    D = collect()
    D.to_csv(cache, index=False)
    lg = lambda p: np.log(p / (1 - p))
    D["h_ratio"] = D.h_fit / D.h
    D["eps_err"] = np.abs(lg(D.eps_fit) - lg(D.eps))
    pd.set_option("display.width", 200)
    T = D.groupby(["set", "f"]).agg(n=("run", "size"), dev=("dev", "mean"), h_ratio=("h_ratio", "median"),
                                    eps_fit=("eps_fit", "median"), eps_err=("eps_err", "median"))
    print(T.round(3).to_string())
    for s in SETTINGS:
        t = T.loc[s]
        rho = spearmanr(t.index, t.dev).statistic
        print(f"R2-P1 {s}: a Spearman(f, dev) = {rho:.2f} -> {'PASS' if rho <= -0.8 else 'FAIL'};  "
              f"b h'/h {t.h_ratio.loc[0.4]:.2f} vs {t.h_ratio.loc[0.0]:.2f} -> {'PASS' if t.h_ratio.loc[0.4] < t.h_ratio.loc[0.0] else 'FAIL'};  "
              f"c eps err {t.eps_err.loc[0.4]:.2f} vs {t.eps_err.loc[0.0]:.2f} -> {'PASS' if t.eps_err.loc[0.4] < t.eps_err.loc[0.0] else 'FAIL'}")
    rows = []
    for f, g in D.groupby("f"):
        X, y = [], []
        for r in g.itertuples():
            for nw in (1, 2, 3, 4):
                G = getattr(r, f"G{nw}")
                if np.isfinite(G):
                    X.append([1, math.log(1 / r.h), -log_rb(r.eps, nw)]); y.append(math.log(G))
        X, y = np.array(X), np.array(y)
        c = np.linalg.lstsq(X, y, rcond=None)[0] if len(y) >= 4 else [np.nan] * 3
        rows.append({"f": f, "crossings": len(y), "c0": c[0], "C1": c[1], "C2": c[2]})
    S = pd.DataFrame(rows).set_index("f")
    print(S.round(3).to_string())
    rho = spearmanr(S.index, S.C2).statistic
    ok = rho >= 0.8 and S.C2.loc[0.4] > 1.5 * S.C2.loc[0.0]
    print(f"R2-P5: Spearman(f, C2) = {rho:.2f}; C2(0.4) / C2(0) = {S.C2.loc[0.4] / S.C2.loc[0.0]:.2f} -> {'PASS' if ok else 'FAIL'}")
    S.to_csv("verify_out/dose_natural.csv")


if __name__ == "__main__":
    main()
