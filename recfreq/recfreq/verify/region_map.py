"""Plan §19 B3: where does the frozen law work? Criterion pre-registered in §19:
(i) error   law RMSE <= 1.25 x RMSE of the models' own Bayes(h', eps') fit (median over seeds);
(ii) measurable  under the law's (h', eps') the exact Bayes switch point (n_c = 2) lies inside G in [2, 128]
     for at least 2 of the 4 n_w.
Uses 2-layer ALiBi models at the default task (V 64, K 16, N 256, Zipf keys, structured typos) from E1, E2b, E5a, E6.

python -m recfreq.verify.region_map --out verify_out --figs figs_region
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..layouts import CELLS, Lay
from . import law_scope
from .law_scope import A_MED, B_MED, law

law_scope.EXPS = ["E1", "E2b", "E5a", "E6"]
NW_IDX = {nw: [i for i, c in enumerate(CELLS) if c[0] == 2 and c[1] == nw] for nw in (1, 2, 3, 4)}
GRID = [(h, e) for h in (0.003, 0.01, 0.03) for e in (0.05, 0.1, 0.2, 0.3)]


def measurable(lay, hp, ep):
    x = lay.bayes([hp], [ep])[0]
    n = 0
    for nw, idx in NW_IDX.items():
        v = x[idx]  # G ascending
        n += int(v[0] > 0 and v[-1] < 0)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--figs", default="figs_region")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.figs, exist_ok=True)
    cache = os.path.join(a.out, "law_scope_fits.csv")
    R = law_scope.runs(a.runs)
    R = R[(R.pe == "alibi_learn") & (R.V == 64) & (R.K == 16) & (R.N == 256) & (R.key_dist == "zipf") & (R.typo == "structured")]
    R = R[~R.run.str.contains("_cf|_d256|_st|_lr|_b256")]
    old = pd.read_csv(cache)
    todo = R[~R.run.isin(old.run)]
    print(f"{len(R)} runs, {len(todo)} new fits", flush=True)
    if len(todo):
        with mp.Pool(a.procs) as pool:
            new = pool.map(law_scope.fit_one, todo.to_dict("records"))
        old = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
        old.to_csv(cache, index=False)
    F = R.merge(old, on="run")
    lay = Lay(4, False, 64)
    rows = []
    for r in F.itertuples():
        y = pd.read_parquet(os.path.join(a.runs, r.run, "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS].lo_model.to_numpy()
        hp, ep = law(r.h, r.eps)
        pl = A_MED + B_MED * lay.bayes([hp], [ep])[0]
        if np.isnan(getattr(r, "h_fit", np.nan)):
            po = np.full_like(y, y.mean())
        else:
            po = r.a_fit + r.b_fit * lay.bayes([r.h_fit], [r.eps_fit])[0]
        rows.append({"run": r.run, "exp": r.exp, "h": r.h, "eps": r.eps, "sd_y": float(y.std()),
                     "rmse_law": float(np.sqrt(((pl - y) ** 2).mean())), "rmse_own": float(np.sqrt(((po - y) ** 2).mean())),
                     "h_fit": r.h_fit, "eps_fit": r.eps_fit, "b_fit": r.b_fit, "r2_law": r.r2_law if hasattr(r, "r2_law") else np.nan,
                     "r2_own": getattr(r, "_14", np.nan)})
    D = pd.DataFrame(rows)
    D["ratio"] = D.rmse_law / D.rmse_own
    S = D.groupby(["h", "eps"]).agg(n=("run", "size"), exp=("exp", "first"), sd_y=("sd_y", "median"),
                                    rmse_law=("rmse_law", "median"), rmse_own=("rmse_own", "median"),
                                    ratio=("ratio", "median"), h_fit=("h_fit", "median"), eps_fit=("eps_fit", "median"),
                                    b_fit=("b_fit", "median")).reset_index()
    S["h_law"], S["eps_law"] = zip(*[law(h, e) for h, e in zip(S.h, S.eps)])
    S["n_meas"] = [measurable(lay, hl, el) for hl, el in zip(S.h_law, S.eps_law)]
    S["measurable"] = S.n_meas >= 2
    S["works"] = (S.ratio <= 1.25) & S.measurable
    S["in_fit_grid"] = [(h, e) in GRID for h, e in zip(S.h, S.eps)]
    S.to_csv(os.path.join(a.out, "region_map.csv"), index=False)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string())
    print(f"\nworks: {S.works.sum()}/{len(S)};  not measurable: {(~S.measurable).sum()};  error criterion fails: {(S.ratio > 1.25).sum()}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    lx, ly = np.log10(S.h), np.log(S.eps / (1 - S.eps))
    sc = ax[0].scatter(lx, ly, c=np.log2(S.ratio), cmap="RdYlGn_r", vmin=-0.5, vmax=2, s=140,
                       edgecolors=np.where(S.measurable, "k", "none"), linewidths=1.2)
    for x0, y0, m in zip(lx, ly, S.measurable):
        if not m:
            ax[0].scatter([x0], [y0], marker="x", c="k", s=40)
    ax[0].add_patch(plt.Rectangle((math.log10(0.003), math.log(0.05 / 0.95)), math.log10(10), math.log(0.3 / 0.7) - math.log(0.05 / 0.95),
                                  fill=False, ls="--", ec="grey"))
    cb = fig.colorbar(sc, ax=ax[0]); cb.set_label("log2(law RMSE / own-fit RMSE)")
    ax[0].set_xlabel("log10 h"); ax[0].set_ylabel("logit eps"); ax[0].set_title("Where the frozen law works (x: not measurable)")
    for e, g in S.groupby("eps"):
        if len(g) >= 3:
            ax[1].plot(g.h, g.h_fit / g.h, "o-", label=f"eps={e}")
    hh = np.logspace(-3.6, -0.8, 50)
    ax[1].plot(hh, 1.453 * hh ** 0.736 / hh, "k--", label="law")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("h"); ax[1].set_ylabel("h'/h"); ax[1].legend(fontsize=7)
    for h, g in S.groupby("h"):
        if len(g) >= 3:
            ax[2].plot(g.eps, g.eps_fit, "o-", label=f"h={h}")
    ee = np.linspace(0.01, 0.45, 50)
    ax[2].plot(ee, [law(0.01, e)[1] for e in ee], "k--", label="law"); ax[2].plot(ee, ee, ":", c="grey")
    ax[2].set_xlabel("eps"); ax[2].set_ylabel("eps'"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.figs, "region_map.png"), dpi=130)


if __name__ == "__main__":
    main()
