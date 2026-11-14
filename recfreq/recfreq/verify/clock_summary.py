"""Clock check: is each model's single-sighting survival belief s_model(d) equal to exact Bayes's s_bayes(d)?

Reads every results/runs/*/survival.parquet (written by recfreq.survival_probes), takes the per-model median over
distances d of |s_model - s_bayes| (one old copy, n = 1), and summarises by experiment and position encoding.

python -m recfreq.verify.clock_summary --out verify_out
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.runs, "*", "survival.parquet"))):
        d = os.path.dirname(f)
        args = json.load(open(os.path.join(d, "done.json")))["args"]
        s = pd.read_parquet(f)
        s1 = s[s.n == 1]
        err = (s1.s_model - s1.s_bayes).abs()
        rows.append({"run": os.path.basename(d), "exp": os.path.basename(d).split("_")[0], "pe": args["pe"],
                     "h": args["h"], "eps": args["eps"], "cf": args.get("conflict_frac", 0.0),
                     "med_abs_err": float(err.median()), "max_abs_err": float(err.max()), "n_dist": len(s1)})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "clock_summary.csv"), index=False)
    print(f"models with survival probes: {len(D)}")
    print(f"per-model median |s_model - s_bayes|: max {D.med_abs_err.max():.3f}, median {D.med_abs_err.median():.3f}")
    print(D.groupby(["exp", "pe"]).med_abs_err.agg(["count", "median", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
