"""Verdicts for run 4A (path ablation of layer-0 heads), against plan §15.

python -m recfreq.summarize4 --run4 figs_run4
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from .layouts import CELLS, LAY4, BayesFree, fit_counter, r2

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
_BF = None


def swap_nc_slope(s: pd.DataFrame) -> float:
    """Mean over (n_w, G) of the slope of swapped-layout LO against log2 n_c."""
    g = s[s.layout == "swap4"]
    sl = []
    for _, c in g.groupby(["n_w", "G"]):
        sl.append(np.polyfit(np.log2(c.n_c), c.lo, 1)[0])
    return float(np.mean(sl))


def fit_gap(y: np.ndarray) -> dict:
    m = np.ones(len(y), bool)
    p = _BF.fit(y, m)
    rb = r2(BayesFree.predict(LAY4, p), y)
    rc = r2(LAY4.counter(fit_counter(LAY4, y, m)), y)
    return {"r2_bayes": rb, "r2_counter": rc, "gap": rb - rc, "h_eff": p["h"]}


def per_model(path: str) -> list[dict]:
    meta = json.load(open(path))
    name = meta["run"]
    base_dir = os.path.dirname(path)
    heads = pd.read_csv(os.path.join(base_dir, f"{name}.heads.csv"))
    surf = pd.read_parquet(os.path.join(base_dir, f"{name}.surf.parquet"))
    prev = int(np.argmax(meta["prev_score"]))
    base_h = heads[heads.cond == "base"].set_index("head")
    copy = base_h[base_h["mass"] > 0.5].index.tolist()
    w = base_h.loc[copy, "mass"]

    def discount(cond):
        hh = heads[heads.cond == cond].set_index("head").loc[copy]
        return float(np.average(hh.b_variant, weights=w))

    d0 = discount("base")
    sb = surf[surf.cond == "base"]
    y0 = sb[sb.layout == "sp4"].set_index(["n_c", "n_w", "G"]).loc[CELLS].lo.to_numpy()
    base_fit = fit_gap(y0)
    rows = []
    for cond in sorted(surf.cond.unique()):
        if cond == "base":
            continue
        h, p = cond[1:].split("_")
        h = int(h)
        s = surf[surf.cond == cond]
        y = s[s.layout == "sp4"].set_index(["n_c", "n_w", "G"]).loc[CELLS].lo.to_numpy()
        dc = discount(cond)
        row = {"run": name, "cond": cond, "head": h, "path": p, "is_prev": h == prev,
               "pe": meta["pe"], "conflict_frac": meta["conflict_frac"], "d_model": meta.get("d_model", 128),
               "steps": meta.get("steps", 20000),
               "d_ce": meta["losses"][cond] - meta["losses"]["base"],
               "discount_base": d0, "discount": dc,
               "discount_removed": (1 - dc / d0) if d0 < -0.05 else np.nan,
               "swap_slope_base": swap_nc_slope(sb), "swap_slope": swap_nc_slope(s),
               "gap_base": base_fit["gap"], "h_eff_base": base_fit["h_eff"]}
        if p in ("q", "k") and not row["is_prev"] and y.var() > 0.02:
            f = fit_gap(y)
            row.update({"gap": f["gap"], "h_eff": f["h_eff"]})
        rows.append(row)
    return rows


def _init():
    global _BF
    _BF = BayesFree(LAY4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run4", default="figs_run4")
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.run4, "*.json")))
    with mp.Pool(a.procs, initializer=_init) as pool:
        rows = [r for rs in pool.map(per_model, paths) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.run4, "paths_summary.csv"), index=False)
    df["arch"] = np.where(df.pe == "rope", "RoPE", np.where(df.pe == "learned_abs", "learned abs", "ALiBi"))
    df.loc[df.conflict_frac > 0, "arch"] += " enr"
    df.loc[df.d_model > 128, "arch"] += " d256"
    df.loc[df.steps > 20000, "arch"] += " 60k"
    n = df.run.nunique()
    print(f"models: {n}")

    print("\n=== A1 (sanity): previous-token head, key path ===")
    a1 = df[df.is_prev & (df.path == "k")]
    print(f"repeat-sighting loss rises by > 1 nat in {(a1.d_ce > 1).mean():.2f} of {len(a1)} models "
          f"(median rise {a1.d_ce.median():.2f})")

    print("\n=== A2: best single non-previous-token head on the query or key path ===")
    qk = df[(~df.is_prev) & df.path.isin(["q", "k"]) & df.discount_removed.notna()]
    for label, sub in (("any", qk), ("model still works (loss +< 0.5 nat)", qk[qk.d_ce < 0.5])):
        best = sub.loc[sub.groupby("run").discount_removed.idxmax()]
        ok = (best.discount_removed >= 0.5)
        print(f"[{label}] models where one head removes >= 50% of the discount: {ok.mean():.2f} of {len(best)} "
              f"(median best removal {best.discount_removed.median():.2f})")
        print("   which path:", best[ok].path.value_counts().to_dict())
        print(best.groupby("arch").apply(lambda d: pd.Series({"n": len(d), "pass": (d.discount_removed >= 0.5).mean(),
                                                                "median_removed": d.discount_removed.median(),
                                                                "median_d_ce": d.d_ce.median()}),
                                         include_groups=False).round(2).to_string())
    print("\nmedian discount removed, by path, best non-previous head per model:")
    bp = df[(~df.is_prev) & df.discount_removed.notna()].groupby(["run", "path"]).discount_removed.max().unstack()
    print(bp.median().round(2).to_string())

    print("\n=== A3: does removing the discount make behaviour counter-like? ===")
    best = qk.loc[qk.groupby("run").discount_removed.idxmax()]
    best = best[best.discount_removed >= 0.5]
    if len(best):
        best = best.assign(d_slope=best.swap_slope - best.swap_slope_base, d_gap=best.gap - best.gap_base)
        print(f"{len(best)} models: swapped-layout n_c slope rises by >= 0.2 in {(best.d_slope >= 0.2).mean():.2f} "
              f"(median change {best.d_slope.median():+.3f}); Bayes-minus-counter R² gap shrinks in "
              f"{(best.d_gap < 0).mean():.2f} (median change {best.d_gap.median():+.3f})")
    print("\nall single-head ablations: largest swapped-layout n_c slope increase per model (any path, model still works):")
    w = df[(df.d_ce < 0.5)].assign(d_slope=lambda d: d.swap_slope - d.swap_slope_base)
    bb = w.loc[w.groupby("run").d_slope.idxmax()]
    print(f"median {bb.d_slope.median():+.3f}; >= 0.2 in {(bb.d_slope >= 0.2).mean():.2f} of models; paths: "
          f"{bb.path.value_counts().to_dict()}")
    print("\nbase swapped-layout n_c slope (Bayes: 0):", round(df.groupby("run").swap_slope_base.first().median(), 3))


if __name__ == "__main__":
    main()
