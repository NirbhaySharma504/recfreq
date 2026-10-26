"""Plan §19.3: the natural (switch-point) law, frozen 27 Oct 02:55 IST before any E6 result, and its tests R2-P3.

  log G*_model(n_w) = C0 + C1 log(1/h) + C2 log(1/r_B),
  r_B = [R (1 + rho^2) (1-eps)/(1-2 eps) - 1] / V,  R = ((1-eps)/eps)^n_w,  rho = eps/(1-eps)   (n_c = 2)

python -m recfreq.verify.natural_law --exp E6
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from ..layouts import CELLS, Lay
from .law_scope import A_MED, B_MED, law
from .switch_points import crossing, model_rows

C0, C1, C2 = -0.739, 0.933, 0.327
GAPS = np.array([2, 4, 8, 16, 32, 64, 128], float)


def log_rb(eps, nw, V=64, nc=2):
    R, rho = ((1 - eps) / eps) ** nw, eps / (1 - eps)
    return math.log(max((R * (1 + rho ** nc) * (1 - eps) / (1 - 2 * eps) - 1) / V, 1e-12))


def g_natural(h, eps, nw, V=64):
    return math.exp(C0 + C1 * math.log(1 / h) - C2 * log_rb(eps, nw, V))


def g_frozen_law(h, eps, nw, lay=Lay(4, False, 64)):
    hp, ep = law(h, eps)
    x = A_MED + B_MED * lay.bayes([hp], [ep])[0]
    idx = [i for i, c in enumerate(CELLS) if c[0] == 2 and c[1] == nw]
    return crossing(GAPS, x[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--exp", default="E6")
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    M = model_rows(a.runs, prefixes=(a.exp + "_",))
    M = M[(M.pe == "alibi_learn") & (M.cf == 0)]
    rows = []
    for r in M.itertuples():
        for nw in (1, 2, 3, 4):
            obs = getattr(r, f"G{nw}")
            gn, gl = g_natural(r.h, r.eps, nw), g_frozen_law(r.h, r.eps, nw)
            rows.append({"run": r.run, "h": r.h, "eps": r.eps, "nw": nw, "G_obs": obs, "G_nat": gn, "G_law": gl,
                         "obs_in": not np.isnan(obs), "nat_in": 2 <= gn <= 128})
    D = pd.DataFrame(rows)
    D["err_nat"] = np.abs(np.log(D.G_obs / D.G_nat))
    D["err_law"] = np.abs(np.log(D.G_obs / D.G_law))
    D.to_csv(os.path.join(a.out, f"natural_law_{a.exp}.csv"), index=False)
    inn = D[D.obs_in]
    both = inn.dropna(subset=["err_law"])
    print(f"{len(M)} {a.exp} models, {len(D)} (model, n_w) pairs, {len(inn)} with a crossing inside [2, 128]")
    print(f"R2-P3a natural law median |log err| = {inn.err_nat.median():.3f} (criterion <= 0.30) -> {'PASS' if inn.err_nat.median() <= 0.30 else 'FAIL'}")
    acc = (D.obs_in == D.nat_in).mean()
    print(f"R2-P3b inside/outside window predicted correctly: {acc:.2f} (criterion >= 0.80) -> {'PASS' if acc >= 0.8 else 'FAIL'}")
    print(f"R2-P3c on {len(both)} crossings where the frozen law also crosses: natural {both.err_nat.median():.3f} vs frozen law "
          f"{both.err_law.median():.3f} -> {'PASS' if both.err_nat.median() < both.err_law.median() else 'FAIL'}"
          f"  (frozen law has no crossing for {inn.err_law.isna().sum()} of the observed ones)")
    pd.set_option("display.width", 220)
    print(D.groupby(["h", "eps", "nw"])[["G_obs", "G_nat", "G_law"]].median().unstack("nw").round(1).to_string())


if __name__ == "__main__":
    main()
