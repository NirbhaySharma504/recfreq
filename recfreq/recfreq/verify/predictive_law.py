"""Audit 5.1 (pre-registered, plan §17): predict a trained model's conflict behaviour from its training (h, eps).

Leave-one-setting-out over the 12 grid settings (2-layer ALiBi; E1 points C, D + E2b): fit
    log h' = alpha + beta log h,   logit eps' = gamma + delta logit eps,   a, b = medians
on 11 settings (fitted (h', eps', a, b) per model from figs_fit/fits.csv), then predict every model of the
held-out setting: lo = a + b * Bayes(h'(h), eps'(eps)). Score R² against the stored probe surface.
Baselines: true Bayes with a, b fitted in-sample on the held-out surface (generous); true Bayes with the
training medians a, b; the mean training surface; a counter with median training parameters.

python -m recfreq.verify.predictive_law --out verify_out
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from ..layouts import CELLS, LAY4, fit_affine, fit_counter, r2


def logit(x):
    return np.log(x / (1 - x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="figs_fit/fits.csv")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f = pd.read_csv(a.fit)
    f = f[(f.arch == "L2") & f.exp.isin(["E1", "E2b"]) & (f.h > 0) & (f.eps > 0)].copy()
    f["a"] = f["bfree.offset"]
    f["b"] = f["bfree.scale"]
    surf = {}
    for r in f.run:
        d = pd.read_parquet(os.path.join(a.runs, r, "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS]
        surf[r] = d.lo_model.to_numpy()
    settings = sorted({(h, e) for h, e in zip(f.h, f.eps)})
    print(f"{len(f)} models in {len(settings)} settings")
    # per-model counter fits (for the counter baseline)
    cpar = {r: fit_counter(LAY4, surf[r], np.ones(len(CELLS), bool)) for r in f.run}
    rows = []
    for (h, e) in settings:
        tr = f[~((f.h == h) & (f.eps == e))]
        te = f[(f.h == h) & (f.eps == e)]
        # laws fitted on the setting medians of the 11 training settings
        sm = tr.groupby(["h", "eps"])[["bfree.h", "bfree.eps", "a", "b"]].median().reset_index()
        sm = sm[sm["bfree.h"] > 0]
        beta, alpha = np.polyfit(np.log(sm.h), np.log(sm["bfree.h"]), 1)
        delta, gamma = np.polyfit(logit(sm.eps), logit(sm["bfree.eps"]), 1)
        a_m, b_m = sm.a.median(), sm.b.median()
        hp = math.exp(alpha + beta * math.log(h))
        ep = 1 / (1 + math.exp(-(gamma + delta * logit(e))))
        pred = a_m + b_m * LAY4.bayes([hp], [ep])[0]
        xtrue = LAY4.bayes([h], [e])[0]
        mean_surf = np.mean([surf[r] for r in tr.run], 0)
        cmed = np.median(np.stack([cpar[r] for r in tr.run]), 0)
        for r in te.run:
            y = surf[r]
            ab = fit_affine(xtrue, y, np.ones(len(y), bool))
            rows.append({"run": r, "h": h, "eps": e, "pred_h": hp, "pred_eps": ep,
                         "fit_h": float(te.set_index("run").loc[r, "bfree.h"]),
                         "fit_eps": float(te.set_index("run").loc[r, "bfree.eps"]),
                         "r2_law": r2(pred, y),
                         "r2_true_bayes_insample": r2(ab[0] + ab[1] * xtrue, y),
                         "r2_true_bayes_median_ab": r2(a_m + b_m * xtrue, y),
                         "r2_mean_surface": r2(mean_surf, y),
                         "r2_counter_median": r2(LAY4.counter(cmed), y)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "predictive_law.csv"), index=False)
    pd.set_option("display.width", 220)
    print(df.groupby(["h", "eps"])[["pred_h", "fit_h", "pred_eps", "fit_eps", "r2_law", "r2_true_bayes_insample",
                                    "r2_true_bayes_median_ab", "r2_mean_surface", "r2_counter_median"]]
          .median().round(3).to_string())
    med = df[[c for c in df.columns if c.startswith("r2_")]].median()
    print("\nmedian over held-out models:\n" + med.round(3).to_string())
    print(f"\npre-registered: law median R² {med.r2_law:.3f} (pass if >= 0.75) and above true Bayes with in-sample "
          f"scale/offset ({med.r2_true_bayes_insample:.3f}): {med.r2_law > med.r2_true_bayes_insample}")
    print(f"law beats true-Bayes(in-sample a, b) in {(df.r2_law > df.r2_true_bayes_insample).mean():.2f} of models; "
          f"beats the mean surface in {(df.r2_law > df.r2_mean_surface).mean():.2f}")


if __name__ == "__main__":
    main()
