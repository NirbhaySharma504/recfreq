"""Plan §18 S7: stress tests of Law 1 on the existing 2-layer ALiBi grid (CPU only, criteria fixed in §18).

Law 1:  log h' = alpha + beta log h,  logit eps' = gamma + delta logit eps,  surface = a + b Bayes(h', eps').
1. leave-one-h-level-out and leave-one-eps-level-out (extrapolation in h and eps; interpolation for middle levels)
2. separability: cross terms (logit eps in the h' law, log h in the eps' law), compared by leave-one-setting-out R²
3. constrained refits: fix eps' at the law value and refit h' (and vice versa) to see whether the eps-dependence
   of h' at fixed h is real or the h'-eps' trade-off ridge
4. noise ceiling: predict each model's surface by the mean surface of the other seeds at the same setting
5. decision agreement: sign of the law-predicted surface vs the model's on cells with |y| > 0.25

python -m recfreq.verify.law_stress --out verify_out
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..layouts import CELLS, LAY4, r2


def logit(x):
    return np.log(x / (1 - x))


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def load(fit_csv, runs_dir):
    f = pd.read_csv(fit_csv)
    f = f[(f.arch == "L2") & f.exp.isin(["E1", "E2b"]) & (f.h > 0) & (f.eps > 0)].copy()
    f["a"], f["b"] = f["bfree.offset"], f["bfree.scale"]
    surf = {}
    for r in f.run:
        d = pd.read_parquet(os.path.join(runs_dir, r, "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS]
        surf[r] = d.lo_model.to_numpy()
    return f, surf


def fit_law(tr, cross=False):
    """Fit the law on the setting medians of the training models. Returns a predictor (h, eps) -> (h', eps', a, b)."""
    sm = tr.groupby(["h", "eps"])[["bfree.h", "bfree.eps", "a", "b"]].median().reset_index()
    sm = sm[sm["bfree.h"] > 0]
    lh, le = np.log(sm.h.to_numpy()), logit(sm.eps.to_numpy())
    yh, ye = np.log(sm["bfree.h"].to_numpy()), logit(sm["bfree.eps"].to_numpy())
    if cross:
        Xh = np.stack([np.ones_like(lh), lh, le], 1)
        Xe = np.stack([np.ones_like(le), le, lh], 1)
    else:
        Xh = np.stack([np.ones_like(lh), lh], 1)
        Xe = np.stack([np.ones_like(le), le], 1)
    ch = np.linalg.lstsq(Xh, yh, rcond=None)[0]
    ce = np.linalg.lstsq(Xe, ye, rcond=None)[0]
    a_m, b_m = sm.a.median(), sm.b.median()

    def pred(h, e):
        xh = [1, math.log(h)] + ([logit(e)] if cross else [])
        xe = [1, logit(e)] + ([math.log(h)] if cross else [])
        return math.exp(np.dot(ch, xh)), float(sigmoid(np.dot(ce, xe))), a_m, b_m
    return pred, ch, ce


def score(pred, te, surf):
    out = []
    for r, h, e in zip(te.run, te.h, te.eps):
        hp, ep, a_m, b_m = pred(h, e)
        yhat = a_m + b_m * LAY4.bayes([hp], [ep])[0]
        y = surf[r]
        sig = np.abs(y) > 0.25
        out.append({"run": r, "h": h, "eps": e, "r2": r2(yhat, y),
                    "sign_agree": float((np.sign(yhat[sig]) == np.sign(y[sig])).mean())})
    return out


def holdout(f, surf, key, cross=False):
    rows = []
    for lvl in sorted(f[key].unique()):
        pred, *_ = fit_law(f[f[key] != lvl], cross)
        for s in score(pred, f[f[key] == lvl], surf):
            rows.append({**s, "held": f"{key}={lvl}"})
    return pd.DataFrame(rows)


def loso(f, surf, cross=False):
    rows = []
    for (h, e), te in f.groupby(["h", "eps"]):
        pred, *_ = fit_law(f[~((f.h == h) & (f.eps == e))], cross)
        rows += score(pred, te, surf)
    return pd.DataFrame(rows)


def constrained_fit(y, fixed: str, value: float):
    """Bayes fit with eps' (or h') fixed; free: the other parameter and the affine a, b."""
    def model(t):
        if fixed == "eps":
            return t[1] + t[2] * LAY4.bayes([math.exp(t[0])], [value])[0]
        return t[1] + t[2] * LAY4.bayes([value], [float(sigmoid(t[0]))])[0]
    best = None
    starts = np.log([0.003, 0.01, 0.03, 0.1, 0.3]) if fixed == "eps" else logit(np.array([0.05, 0.15, 0.3, 0.45]))
    lo, hi = ([math.log(1e-5), -30, -30], [math.log(0.7), 30, 30]) if fixed == "eps" else ([-9, -30, -30], [0, 30, 30])
    for s in starts:
        r = least_squares(lambda t: model(t) - y, [s, 0.0, 1.0], bounds=(lo, hi), max_nfev=300)
        if best is None or r.cost < best.cost:
            best = r
    t = best.x
    p = math.exp(t[0]) if fixed == "eps" else float(sigmoid(t[0]))
    return p, r2(model(t), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="figs_fit/fits.csv")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f, surf = load(a.fit, a.runs)
    pd.set_option("display.width", 200)
    print(f"{len(f)} models, {f.groupby(['h', 'eps']).ngroups} settings\n")

    pred_all, ch, ce = fit_law(f)
    print(f"law on all 12 settings: h' = {math.exp(ch[0]):.3f} h^{ch[1]:.3f};  logit eps' = {ce[0]:.3f} + {ce[1]:.3f} logit eps")
    insample = pd.DataFrame(score(pred_all, f, surf))
    lo = loso(f, surf)

    # 1. level hold-outs
    print("\n1. level hold-outs (criterion: median R² >= 0.75 for each held-out level)")
    lv = pd.concat([holdout(f, surf, "h"), holdout(f, surf, "eps")])
    t1 = lv.groupby("held").agg(n=("r2", "size"), r2_median=("r2", "median"), r2_min=("r2", "min"),
                                 sign_agree=("sign_agree", "median"))
    t1["pass"] = t1.r2_median >= 0.75
    print(t1.round(3).to_string())
    lv.to_csv(os.path.join(a.out, "law_stress_levels.csv"), index=False)

    # 2. separability
    lx = loso(f, surf, cross=True)
    _, chx, cex = fit_law(f, cross=True)
    d = lx.r2.median() - lo.r2.median()
    print(f"\n2. separability: LOSO median R² simple {lo.r2.median():.3f}, with cross terms {lx.r2.median():.3f} "
          f"(diff {d:+.3f}; keep simple law if < 0.03 -> {'keep' if d < 0.03 else 'cross terms needed'})")
    print(f"   cross-term law: log h' = {chx[0]:.2f} + {chx[1]:.3f} log h + {chx[2]:.3f} logit eps;  "
          f"logit eps' = {cex[0]:.2f} + {cex[1]:.3f} logit eps + {cex[2]:.3f} log h")
    sm = f.groupby(["h", "eps"])[["bfree.h", "bfree.eps"]].median().reset_index()
    res_h = np.log(sm["bfree.h"]) - (ch[0] + ch[1] * np.log(sm.h))
    res_e = logit(sm["bfree.eps"]) - (ce[0] + ce[1] * logit(sm.eps))
    print(f"   corr(h' law residual, logit eps) = {np.corrcoef(res_h, logit(sm.eps))[0, 1]:+.2f};  "
          f"corr(eps' law residual, log h) = {np.corrcoef(res_e, np.log(sm.h))[0, 1]:+.2f}")
    print(f"   R² of the h' law on log setting medians {1 - res_h.var() / np.log(sm['bfree.h']).var():.3f}; "
          f"of the eps' law on logit medians {1 - res_e.var() / logit(sm['bfree.eps']).var():.3f}")

    # 3. constrained refits: is the eps-dependence of h' at fixed h real, or the h'-eps' ridge?
    rows = []
    for r, h, e in zip(f.run, f.h, f.eps):
        hp_law, ep_law, *_ = pred_all(h, e)
        h_c, r2_c = constrained_fit(surf[r], "eps", ep_law)
        rows.append({"run": r, "h": h, "eps": e, "h_free": float(f.set_index("run").loc[r, "bfree.h"]),
                     "h_eps_fixed": h_c, "r2_free": float(f.set_index("run").loc[r, "bayes_free.r2"]),
                     "r2_eps_fixed": r2_c})
    cf = pd.DataFrame(rows)
    cf.to_csv(os.path.join(a.out, "law_stress_constrained.csv"), index=False)
    g = cf.groupby(["h", "eps"])[["h_free", "h_eps_fixed", "r2_free", "r2_eps_fixed"]].median()
    print("\n3. h' refitted with eps' fixed at the law value (setting medians)")
    print(g.round(4).to_string())
    for h, gg in g.groupby(level=0):
        print(f"   h = {h}: spread of h' across eps  free x{gg.h_free.max() / gg.h_free.min():.2f}  "
              f"eps'-fixed x{gg.h_eps_fixed.max() / gg.h_eps_fixed.min():.2f}")
    print(f"   median R² loss from fixing eps': {(cf.r2_free - cf.r2_eps_fixed).median():.3f}")

    # 4. noise ceiling
    rows = []
    for (h, e), te in f.groupby(["h", "eps"]):
        for r in te.run:
            others = [surf[o] for o in te.run if o != r]
            rows.append({"run": r, "h": h, "eps": e, "r2_seed_ceiling": r2(np.mean(others, 0), surf[r])})
    nc = pd.DataFrame(rows)
    m = nc.merge(lo[["run", "r2"]].rename(columns={"r2": "r2_loso"}), on="run") \
          .merge(insample[["run", "r2"]].rename(columns={"r2": "r2_law_insample"}), on="run") \
          .merge(f[["run", "bayes_free.r2"]].rename(columns={"bayes_free.r2": "r2_own_fit"}), on="run")
    m.to_csv(os.path.join(a.out, "law_stress_ceiling.csv"), index=False)
    print("\n4. noise ceiling (medians over models)")
    print(f"   own Bayes(h', eps') fit {m.r2_own_fit.median():.3f} | other seeds' mean surface {m.r2_seed_ceiling.median():.3f}"
          f" | law in-sample {m.r2_law_insample.median():.3f} | law leave-one-setting-out {m.r2_loso.median():.3f}")
    print(f"   law LOSO as a fraction of the seed ceiling: {(m.r2_loso / m.r2_seed_ceiling).median():.2f}")
    print(m.groupby(["h", "eps"])[["r2_own_fit", "r2_seed_ceiling", "r2_law_insample", "r2_loso"]].median().round(3).to_string())

    # 5. decisions
    print(f"\n5. decision agreement (sign, cells with |y| > 0.25): law in-sample {insample.sign_agree.median():.3f}; "
          f"LOSO {lo.sign_agree.median():.3f}")


if __name__ == "__main__":
    main()
