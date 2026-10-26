"""Plan §19 round 2: the models' behaviour in the probes' own coordinates.

For every 2-layer ALiBi model at the default task, the switch gap G*(n_c = 2, n_w) where the model's log-odds cross 0
(log-linear interpolation between probe gaps), next to the exact Bayes G* for the same (h, eps).
Two summary coordinates per model:
  time scale  T = G*(n_w = 2)                       (where two recent copies win)
  spacing     D = mean over n_w of -d log G* / d n_w (how far one extra recent copy moves the switch point)
python -m recfreq.verify.switch_points --out verify_out
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from .theory import exact_gstar


def crossing(G, y):
    k = np.flatnonzero((y[:-1] > 0) & (y[1:] <= 0))
    if len(k) == 0:
        return np.nan
    i = k[0]
    t = y[i] / (y[i] - y[i + 1])
    return float(np.exp(np.log(G[i]) + t * (np.log(G[i + 1]) - np.log(G[i]))))


def model_rows(runs, prefixes=("E1_", "E2b_", "E5a_", "E6_", "E7_", "E2c_")):
    rows = []
    for f in sorted(glob.glob(os.path.join(runs, "*/done.json"))):
        name = f.split("/")[-2]
        if not name.startswith(prefixes):
            continue
        a = json.load(open(f))["args"]
        if a["layers"] != 2 or a["mlp"] or a.get("typo", "structured") != "structured" or a["h"] <= 0 or a["eps"] <= 0:
            continue
        if a.get("V", 64) != 64 or a.get("K", 16) != 16 or a.get("n_pairs", 256) != 256 or a.get("key_dist", "zipf") != "zipf" \
                or a.get("d_model", 128) != 128 or a.get("steps", 20000) != 20000 or a.get("lr", 1e-3) != 1e-3 or a.get("batch", 128) != 128:
            continue
        P = pd.read_parquet(os.path.join(os.path.dirname(f), "probes.parquet"))
        r = {"run": name, "h": a["h"], "eps": a["eps"], "pe": a["pe"], "cf": a.get("conflict_frac", 0.0), "seed": a["seed"]}
        for nw in (1, 2, 3, 4):
            x = P[(P.n_c == 2) & (P.n_w == nw)].sort_values("G")
            r[f"G{nw}"] = crossing(x.G.values.astype(float), x.lo_model.values)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    M = model_rows(a.runs)
    B = {}
    for h, e in set(zip(M.h, M.eps)):
        B[(h, e)] = [exact_gstar(h, e, nw, 2, gmax=1e6) for nw in (1, 2, 3, 4)]
    for nw in (1, 2, 3, 4):
        M[f"B{nw}"] = [B[(h, e)][nw - 1] for h, e in zip(M.h, M.eps)]
    lg = lambda c: np.log(M[c].where(M[c] > 0))
    M["D_model"] = -(pd.concat([lg(f"G{n+1}") - lg(f"G{n}") for n in (1, 2, 3)], axis=1)).mean(1)
    M["D_bayes"] = -(pd.concat([lg(f"B{n+1}") - lg(f"B{n}") for n in (1, 2, 3)], axis=1)).mean(1)
    M["D_corr"] = [math.log((1 - e) / e) for e in M.eps]
    M.to_csv(os.path.join(a.out, "switch_points.csv"), index=False)
    pd.set_option("display.width", 250)
    S = M[(M.pe == "alibi_learn") & (M.cf == 0)].groupby(["h", "eps"])[["G1", "B1", "G2", "B2", "G3", "B3", "G4", "B4", "D_model", "D_bayes", "D_corr"]].median()
    print(S.round(2).to_string())
    s = S.dropna(subset=["G2", "B2"])
    s = s[(s.B2 > 0) & np.isfinite(s.B2)]
    c = np.polyfit(np.log(s.B2), np.log(s.G2), 1)
    print(f"\ntime scale: log G*_model(n_w=2) = {c[1]:.2f} + {c[0]:.3f} log G*_bayes(n_w=2)   (n = {len(s)} settings)")
    d = S.dropna(subset=["D_model", "D_bayes"])
    print(f"spacing: median D_model / D_bayes = {np.median(d.D_model / d.D_bayes):.2f};  D_model vs log((1-eps)/eps): "
          f"slope {np.polyfit(d.D_corr, d.D_model, 1)[0]:.3f}")


if __name__ == "__main__":
    main()
