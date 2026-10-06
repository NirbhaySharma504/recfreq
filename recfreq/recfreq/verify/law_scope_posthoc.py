"""Plan §18 post-hoc (labelled as such): why does the frozen law fail where it fails?

For every run in verify_out/law_scope_all.csv (baselines and E5):
- rmse_law: absolute error of the frozen law (R² is unstable when a model's surface is nearly flat)
- r2_law_ab: the law's beliefs (h', eps') with scale/offset a, b fitted to the model itself; compare r2_true_ab,
  true Bayes with the same two free parameters. Separates "wrong beliefs" from "wrong output scale".
- r2_cross / r2_cross_ab: the law with cross terms (log h' also linear in logit eps, logit eps' also linear in
  log h), fitted on the original 12 settings only, so its predictions for E5 runs are still out of sample.

python -m recfreq.verify.law_scope_posthoc --out verify_out
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from ..layouts import CELLS, Lay, fit_affine, r2
from .law_scope import A_MED, B_MED, law
from .law_stress import fit_law, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="verify_out/law_scope_all.csv")
    ap.add_argument("--fit", default="figs_fit/fits.csv")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    F = pd.read_csv(a.csv)
    base, _ = load(a.fit, a.runs)
    cross, ch, ce = fit_law(base, cross=True)
    print(f"cross-term law (fitted on the 12 original settings): log h' = {ch[0]:.3f} + {ch[1]:.3f} log h + {ch[2]:.3f} "
          f"logit eps;  logit eps' = {ce[0]:.3f} + {ce[1]:.3f} logit eps + {ce[2]:.3f} log h")
    rows = []
    full = np.ones(len(CELLS), bool)
    for r in F.itertuples():
        y = pd.read_parquet(os.path.join(a.runs, r.run, "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS]
        y = y.lo_model.to_numpy()
        lay = Lay(4, False, int(r.V))
        hp, ep = law(r.h, r.eps)
        xl = lay.bayes([hp], [ep], r.typo)[0]
        hc, ec, a_m, b_m = cross(r.h, r.eps)
        xc = lay.bayes([hc], [ec], r.typo)[0]
        xt = lay.bayes([r.h], [r.eps], r.typo)[0]
        fit_ab = lambda x: (lambda c: c[0] + c[1] * x)(fit_affine(x, y, full))
        pl = A_MED + B_MED * xl
        rows.append({"run": r.run, "exp": r.exp, "h": r.h, "eps": r.eps, "N": r.N, "V": r.V, "K": r.K,
                     "key_dist": r.key_dist, "typo": r.typo, "pe": r.pe, "sd_y": float(y.std()),
                     "rmse_law": float(np.sqrt(((pl - y) ** 2).mean())), "r2_law": r2(pl, y),
                     "r2_law_ab": r2(fit_ab(xl), y), "r2_true_ab": r2(fit_ab(xt), y),
                     "r2_cross": r2(A_MED + B_MED * xc, y), "r2_cross_ab": r2(fit_ab(xc), y),
                     "h_cross": hc, "eps_cross": ec, "b_fit": getattr(r, "b_fit")})
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(a.out, "law_scope_posthoc.csv"), index=False)
    pd.set_option("display.width", 220)
    cols = ["sd_y", "rmse_law", "r2_law", "r2_law_ab", "r2_true_ab", "r2_cross", "r2_cross_ab", "b_fit"]
    for name, d in (("baseline grid (E1+E2b)", P[P.exp.isin(["E1", "E2b"]) & (P.pe == "alibi_learn") & (P.V == 64)
                                                    & (P.K == 16) & (P.N == 256) & (P.key_dist == "zipf")
                                                    & (P.typo == "structured")]),
                    ("E5a new (h, eps)", P[P.exp == "E5a"])):
        if len(d):
            print(f"\n{name}: medians per setting")
            print(d.groupby(["h", "eps"])[cols].median().round(3).to_string())
            print("  overall median:", d[cols].median().round(3).to_dict())
    for e in ("E5n", "E5d", "E5r"):
        d = P[P.exp == e]
        if len(d):
            g = ["N", "h", "eps"] if e == "E5n" else ["V", "K", "key_dist", "typo", "h", "eps"] if e == "E5d" else ["h", "eps"]
            print(f"\n{e}: medians")
            print(d.groupby(g)[cols].median().round(3).to_string())


if __name__ == "__main__":
    main()
