"""Plan §19.14 round 8 (H-mass): score mass.parquet / mass_boost.parquet from recfreq.mass_probe.

x = mean over copy heads of (lc - lw), the log attention mass ratio old : new at the query.
R8-P1  per model R2 of lo_model on x vs R2 of lo_model on lo_bayes; median R2_mass >= 0.8 and R2_mass > R2_bayes in
       >= 75% of models.
R8-P2  f = (lo_boost - lo_one) / (lo_real - lo_one) over boost pairs with |lo_real - lo_one| >= 0.3; median over
       models of the per-model median f in [0.8, 1.25].
R8-P3  A = x(n_c 2, n_w 1, G 2) - x at G*_1 (crossing of lo_model at n_c 2, n_w 1; x interpolated in log G);
       Spearman(A, w) <= -0.6 and Spearman(log h, A) <= -0.6.
Primary set: ordinary 2-layer ALiBi models at the default task from E1, E2b, E5a, E6 (eps > 0).

python -m recfreq.verify.mass_score
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .switch_points import crossing

DEFAULT = dict(K=16, n_pairs=256, V=64, key_dist="zipf", typo="structured", steps=20000)


def r2(y, x):
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1] ** 2) if m.sum() > 3 else np.nan


def one(d):
    a = json.load(open(os.path.join(d, "done.json")))["args"]
    if a["eps"] <= 0 or any(a.get(k, v) != v for k, v in DEFAULT.items()):
        return None
    C = pd.read_parquet(os.path.join(d, "mass.parquet"))
    B = pd.read_parquet(os.path.join(d, "mass_boost.parquet"))
    heads = C.heads.iloc[0].split(",")
    C["x"] = np.mean([C[f"lc{h}"] - C[f"lw{h}"] for h in heads], axis=0)
    r = {"run": os.path.basename(d), "exp": os.path.basename(d).split("_")[0], "pe": a["pe"], "h": a["h"],
         "eps": a["eps"], "cf": a.get("conflict_frac", 0.0), "seed": a["seed"], "n_heads": len(heads),
         "r2_mass": r2(C.lo_model.values, C.x.values), "r2_bayes": r2(C.lo_model.values, C.lo_bayes.values),
         "gain": np.polyfit(C.x, C.lo_model, 1)[0]}
    B = B[(B.lo_real - B.lo_one).abs() >= 0.3]
    f = (B.lo_boost - B.lo_one) / (B.lo_real - B.lo_one)
    r.update({"n_boost": len(f), "f_med": f.median() if len(f) else np.nan})
    s = C[(C.n_c == 2) & (C.n_w == 1)].sort_values("G")
    g1 = crossing(s.G.values.astype(float), s.lo_model.values)
    r["G1"] = g1
    if np.isfinite(g1) and 2 <= g1 <= 224:
        r["A"] = s.x.iloc[0] - np.interp(np.log(g1), np.log(s.G.values), s.x.values)
    else:
        r["A"] = np.nan
    return r


def main():
    rows = [one(os.path.dirname(f)) for f in sorted(glob.glob("results/runs/*/mass_boost.parquet"))]
    D = pd.DataFrame([r for r in rows if r])
    W = pd.read_csv("verify_out/evidence_weight.csv")[["run", "w"]]
    D = D.merge(W, on="run", how="left")
    D.to_csv("verify_out/mass_score.csv", index=False)
    pd.set_option("display.width", 220)
    prim = D[(D.pe == "alibi_learn") & (D.cf == 0) & D.exp.isin(["E1", "E2b", "E5a", "E6"])]
    print(f"primary set: {len(prim)} ALiBi models")
    m, frac = prim.r2_mass.median(), (prim.r2_mass > prim.r2_bayes).mean()
    print(f"R8-P1: median R2_mass {m:.3f} (R2_bayes {prim.r2_bayes.median():.3f}); R2_mass > R2_bayes in {frac:.0%} "
          f"-> {'PASS' if m >= 0.8 and frac >= 0.75 else 'FAIL'}")
    fm = prim.f_med.median()
    print(f"R8-P2: median f {fm:.2f} over {prim.f_med.notna().sum()} models (IQR {prim.f_med.quantile(0.25):.2f}-"
          f"{prim.f_med.quantile(0.75):.2f}) -> {'PASS' if 0.8 <= fm <= 1.25 else 'FAIL'}")
    p = prim.dropna(subset=["A", "w"])
    s1 = spearmanr(p.A, p.w).statistic
    q = prim.dropna(subset=["A"])
    s2 = spearmanr(np.log(q.h), q.A).statistic
    print(f"R8-P3: Spearman(A, w) {s1:.2f} (n = {len(p)}); Spearman(log h, A) {s2:.2f} (n = {len(q)}) -> "
          f"{'PASS' if s1 <= -0.6 and s2 <= -0.6 else 'FAIL'}")
    for name, sub in (("RoPE", D[D.pe == "rope"]), ("E12 conflict", D[D.exp == "E12"])):
        print(f"\n{name} (exploratory, {len(sub)} models): median R2_mass {sub.r2_mass.median():.3f}, "
              f"R2_bayes {sub.r2_bayes.median():.3f}, median f {sub.f_med.median():.2f}")
    print("\nby h (primary):")
    print(prim.groupby("h")[["r2_mass", "r2_bayes", "f_med", "A", "w", "gain"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
