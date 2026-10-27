"""Plan §19.10 R5-P3: evidence weight w under slope surgery (surgery.parquet from recfreq.slope_surgery).

w per condition in clock units, as in verify/evidence_weight.py: crossings G*(n_w) at n_c = 2 over the standard
and wide windows, exact Bayes G* at the same (h, eps, n_w), slope of log(h G*_model) on log(h G*_Bayes).

python -m recfreq.verify.surgery_score --match E6_
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .switch_points import crossing
from .theory import exact_gstar


def weight(S, h, eps):
    g = {}
    for nw, d in S.groupby("n_w"):
        d = d.sort_values("G")
        g[nw] = crossing(d.G.values.astype(float), d.lo_model.values)
    nws = [n for n in sorted(g) if np.isfinite(g[n])]
    b = {n: exact_gstar(h, eps, n, 2, gmax=1e7) for n in nws}
    nws = [n for n in nws if 0 < b[n] < np.inf]
    if len(nws) < 2:
        return np.nan, np.nan
    w, al = np.polyfit(np.log([h * b[n] for n in nws]), np.log([h * g[n] for n in nws]), 1)
    return w, float(np.exp(al))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--match", default="E6_")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.runs, f"{a.match}*/surgery.parquet"))):
        d = os.path.dirname(f)
        r = json.load(open(os.path.join(d, "done.json")))["args"]
        S = pd.read_parquet(f)
        row = {"run": os.path.basename(d), "h": r["h"], "eps": r["eps"]}
        for cond, s in S.groupby("cond"):
            row[f"w_{cond}"], row[f"x_{cond}"] = weight(s, r["h"], r["eps"])
        rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(f"verify_out/surgery_{a.match.strip('_')}.csv", index=False)
    pd.set_option("display.width", 200)
    print(D.round(3).to_string())
    for cond, lo, hi, up in (("x2.0", 1.3, 2.2, True), ("x0.5", 0.45, 0.8, False)):
        r = (D[f"w_{cond}"] / D.w_base).dropna()
        frac = (r > 1).mean() if up else (r < 1).mean()
        med = r.median()
        ok = lo <= med <= hi and frac >= 0.8
        print(f"R5-P3 {cond}: median w ratio {med:.2f} (criterion {lo}-{hi}); direction right in {frac:.0%} "
              f"of {len(r)} models -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
