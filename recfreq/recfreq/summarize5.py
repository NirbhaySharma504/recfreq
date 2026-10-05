"""Verdicts for run 5 (plan §16).

python -m recfreq.summarize5 --run5 figs_run5
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


def _init():
    global _BF
    _BF = BayesFree(LAY4)


def gap_of(s: pd.DataFrame) -> dict:
    y = s[s.layout == "sp4"].set_index(["n_c", "n_w", "G"]).loc[CELLS].lo.to_numpy()
    if y.var() < 0.02:
        return {"gap": np.nan, "h_eff": np.nan, "r2_bayes": np.nan}
    m = np.ones(len(y), bool)
    p = _BF.fit(y, m)
    rb = r2(BayesFree.predict(LAY4, p), y)
    rc = r2(LAY4.counter(fit_counter(LAY4, y, m)), y)
    sw = s[s.layout == "swap4"]
    sl = np.mean([np.polyfit(np.log2(c.n_c), c.lo, 1)[0] for _, c in sw.groupby(["n_w", "G"])])
    return {"gap": rb - rc, "h_eff": p["h"], "r2_bayes": rb, "swap_slope": sl}


def part_a(path: str) -> list[dict]:
    meta = json.load(open(path))
    name = meta["run"]
    surf = pd.read_parquet(path.replace(".json", ".surf.parquet"))
    rows = []
    for cond, s in surf.groupby("cond"):
        rows.append({"run": name, "cond": cond, "pe": meta["pe"], "d_model": meta.get("d_model", 128),
                     "steps": meta.get("steps", 20000), "conflict_frac": meta["conflict_frac"],
                     "d_ce": meta["losses"][cond] - meta["losses"]["base"], **gap_of(s)})
    return rows


def part_b(path: str) -> list[dict]:
    meta = json.load(open(path))
    dec = pd.read_parquet(path.replace(".json", ".dec.parquet"))
    dec = dec[(dec.layout == "sp4") & dec.slow & (dec.G <= 64)]
    prev = meta["prev"]
    rows = []
    for h, d in dec.groupby("head"):
        c = d.groupby("G").mean(numeric_only=True)
        out = {"run": meta["run"], "head": h, "pe": meta["pe"],
               "content_range": float(c.content_mean.max() - c.content_mean.min()),
               "content_lse_range": float(c.content_lse.max() - c.content_lse.min()),
               "content_trend": float(np.polyfit(np.log2(c.index), c.content_mean, 1)[0]),
               "split_range": float(c.split.max() - c.split.min()),
               "dist_range": float(c.dist_mean.max() - c.dist_mean.min())}
        tot = c.content_mean
        var = tot.var(ddof=0)
        for side in ("kside", "qside"):
            cols = [k for k in c.columns if k.startswith(side + ".")]
            groups = {"emb": [f"{side}.emb"], "prev": [f"{side}.prev"], "lnb": [f"{side}.lnb"],
                      "other0": [k for k in cols if k.startswith(f"{side}.L0H")]}
            for gname, gcols in groups.items():
                comp = c[gcols].sum(1) if gcols else pd.Series(0.0, index=c.index)
                out[f"{side}.share.{gname}"] = float(np.cov(comp, tot, bias=True)[0, 1] / var) if var > 1e-9 else np.nan
        rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run5", default="figs_run5")
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.run5, "*.json")))
    with mp.Pool(a.procs, initializer=_init) as pool:
        A = pd.DataFrame([r for rs in pool.map(part_a, paths) for r in rs])
    B = pd.DataFrame([r for p in paths for r in part_b(p)])
    A.to_csv(os.path.join(a.run5, "run5_A.csv"), index=False)
    B.to_csv(os.path.join(a.run5, "run5_B.csv"), index=False)
    for d in (A, B):
        d["arch"] = d.pe.map({"alibi_learn": "ALiBi", "rope": "RoPE", "learned_abs": "learned abs"})
    print(f"models: {A.run.nunique()}")

    print("\n=== 5A: combination ablations of layer-0 heads ===")
    base = A[A.cond == "base"].set_index("run")
    A = A.join(base[["gap", "h_eff", "swap_slope"]], on="run", rsuffix="_base")
    na = A[A.cond == "nonprev_all"]
    works = na[na.d_ce < 1]
    print(f"5A-1: induction still works (loss + < 1 nat) with all non-previous-token layer-0 heads removed: "
          f"{len(works)} of {len(na)} models ({len(works) / len(na):.2f}); median loss change {na.d_ce.median():+.3f}")
    gone = (works.gap < 0.05)
    print(f"5A-2: Bayes advantage (R² gap) falls below 0.05 in {gone.mean():.2f} of {len(works)} working models; "
          f"median gap base {works.gap_base.median():.3f} -> {works.gap.median():.3f}")
    print("\nby condition (median over models):")
    print(A.groupby("cond")[["d_ce", "gap_base", "gap", "r2_bayes", "h_eff_base", "h_eff", "swap_slope"]].median()
          .round(3).to_string())
    print("\nnonprev_all by architecture:")
    print(na.groupby("arch")[["d_ce", "gap_base", "gap", "r2_bayes"]].median().round(3).to_string())

    print("\n=== 5B: score decomposition in slow copy heads (normal layout, G = 2..64) ===")
    mb = B.groupby("run").median(numeric_only=True)
    mb["arch"] = B.groupby("run").arch.first()
    p1 = (mb.content_range > 1)
    print(f"5B-1: content part of the old-minus-recent split varies by > 1 nat over G in {p1.mean():.2f} of {len(mb)} "
          f"models (median range {mb.content_range.median():.2f}; LSE version {mb.content_lse_range.median():.2f}; "
          f"distance part range {mb.dist_range.median():.2f})")
    print("   direction: content split rises with G (compensates distance) in "
          f"{(mb.content_trend > 0).mean():.2f} of models")
    l0 = mb["kside.share.prev"] + mb["kside.share.other0"]
    print(f"5B-2: key side, share of the G-dependence from layer-0 head outputs: median {l0.median():.2f}; "
          f">= 0.5 in {(l0 >= 0.5).mean():.2f} of models")
    print("\nkey-side and query-side shares (median over models), by architecture:")
    cols = [c for c in mb.columns if ".share." in c]
    print(mb.groupby("arch")[cols].median().round(2).to_string())
    print("\n(RoPE caveat: position enters through rotation of every component, so 'emb' vs layer-0 is confounded.)")


if __name__ == "__main__":
    main()
