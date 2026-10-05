"""Audit 2.3 (reproducibility) and 2.4 (hyperparameters), pre-registered in plan §17.

Needs: python -m recfreq.fit_rules results/runs --exp E4r,E4h --out figs_fit_E4

python -m recfreq.verify.e4 --out verify_out
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .registry import CELLS, my_cliff, probes

R = "results/runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_e4", default="figs_fit_E4/fits.csv")
    ap.add_argument("--fit", default="figs_fit/fits.csv")
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    f4 = pd.read_csv(a.fit_e4)
    f0 = pd.read_csv(a.fit)
    pd.set_option("display.width", 250)

    print("=== 2.3 reproducibility ===")
    rows = []
    for r in f4[f4.run.str.startswith("E4r")].run:
        cfg = json.load(open(os.path.join(R, r, "config.json")))
        # the original run with the same configuration and seed
        pre = "E2a" if cfg["pe"] == "rope" else "E1"
        orig = r.replace("E4r", pre)
        e4 = f4.set_index("run").loc[r]
        rec = {"run": r, "seed": cfg["seed"], "h_new": e4["bfree.h"], "eps_new": e4["bfree.eps"],
               "excess_new": json.load(open(os.path.join(R, r, "done.json")))["final"]["excess"]}
        if os.path.exists(os.path.join(R, orig, "done.json")):
            o = f0.set_index("run").loc[orig]
            rec.update({"orig": orig, "h_orig": o["bfree.h"], "eps_orig": o["bfree.eps"],
                        "excess_orig": json.load(open(os.path.join(R, orig, "done.json")))["final"]["excess"],
                        "surface_corr": float(np.corrcoef(probes(r).lo_model, probes(orig).lo_model)[0, 1])})
        else:  # new seed: compare with the range of the original seeds
            sib = f0[f0.run.str.match(orig.rsplit("_s", 1)[0].replace(".", r"\.") + r"_s\d$")]
            rec.update({"orig_h_min": sib["bfree.h"].min(), "orig_h_max": sib["bfree.h"].max(),
                        "orig_eps_min": sib["bfree.eps"].min(), "orig_eps_max": sib["bfree.eps"].max()})
        rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(a.out, "e4_repro.csv"), index=False)
    print(d.round(4).to_string(index=False))
    same = d[d.get("orig").notna()] if "orig" in d else d.iloc[:0]
    if len(same):
        ok = ((same.h_new / same.h_orig - 1).abs() <= 0.15) & ((same.eps_new / same.eps_orig - 1).abs() <= 0.15) & \
             (same.surface_corr > 0.95)
        print(f"same-seed retrains within 15% on h' and eps' with surface corr > 0.95: {int(ok.sum())} of {len(same)}")
    new = d[d.get("orig").isna()] if "orig" in d else d
    if len(new):
        inr = (new.h_new >= 0.7 * new.orig_h_min) & (new.h_new <= 1.3 * new.orig_h_max)
        print(f"new seeds with h' inside the original seeds' range +-30%: {int(inr.sum())} of {len(new)}")

    print("\n=== 2.4 hyperparameters ===")
    h = f4[f4.run.str.startswith("E4h")].copy()
    h["pe"] = np.where(h.arch.str.contains("rope"), "RoPE", "ALiBi")
    h["variant"] = h.arch.str.extract(r"(lr[\d.e-]+|b\d+)")[0]
    h["ratio"] = h["bfree.h"] / h.h
    h["best_counter_cv"] = h[["counter_exp.cv_r2", "counter_pow.cv_r2", "additive.cv_r2"]].max(1)
    h["bayes_wins"] = h["bayes_free.cv_r2"] > h.best_counter_cv
    h["cliff_ratio"] = [my_cliff(probes(r), "lo_model") / my_cliff(probes(r), "lo_bayes") for r in h.run]
    h.to_csv(os.path.join(a.out, "e4_hparams.csv"), index=False)
    print(h.groupby(["pe", "variant", "h"])[["ratio", "bayes_free.cv_r2", "best_counter_cv", "cliff_ratio"]]
          .median().round(3).to_string())
    al = h[h.pe == "ALiBi"]
    print(f"\npre-registered: ALiBi h'/h > 2 in every run: {bool((al.ratio > 2).all())} (min {al.ratio.min():.2f}); "
          f"Bayes beats best counter held-out in {h.bayes_wins.mean():.2f} of {len(h)} runs (pass if >= 0.90)")
    print(f"against: any setting with h'/h <= 1.5: {bool((h.ratio <= 1.5).any())}")


if __name__ == "__main__":
    main()
