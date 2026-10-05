"""Post-hoc analyses after run 4 (NOT pre-registered; hypotheses formed after seeing one model).

P1  For every copy head in 2-layer attention-only models: is the head's split of attention between old and
    recent copies Bayes-shaped or counter-shaped, and does that depend on how fast the head fades?
P2  ALiBi copy heads: how far is the measured split from what the head's own ALiBi slope predicts, as a function
    of the gap G, in the normal and the swapped layout? (Hypothesis: a typo-variant offset that fades with G.)
P3  Path ablation: when the layer-0 head that carries the typo-variant signal is removed from the key path,
    what happens to the model's effective switch rate h'?

python -m recfreq.posthoc4 --mech figs_mech --run3 figs_run3 --run4 figs_run4 --out figs_run4
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .layouts import CELLS, LAY4, BayesFree, fit_counter, r2

pd.set_option("display.width", 250)
_BF = None


def _init():
    global _BF
    _BF = BayesFree(LAY4)


def p1_model(name: str) -> list[dict]:
    mc = pd.read_parquet(os.path.join(A.mech, f"{name}.cells.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS]
    H = pd.read_csv(os.path.join(A.mech, "mech_heads.csv"))
    H = H[(H.run == name) & (H.mass_on_copies > 0.5)]
    m = np.ones(len(CELLS), bool)
    rows = []
    for _, h in H.iterrows():
        t = h["head"]
        fr = (mc[f"{t}.A_c"] / (mc[f"{t}.A_c"] + mc[f"{t}.A_w"])).clip(1e-4, 1 - 1e-4).to_numpy()
        lg = np.log(fr / (1 - fr))
        if lg.var() < 0.05:
            continue
        p = _BF.fit(lg, m)
        rb = r2(BayesFree.predict(LAY4, p), lg)
        rc = r2(LAY4.counter(fit_counter(LAY4, lg, m)), lg)
        decay = h["alibi_slope_per_pair"] if not pd.isna(h["alibi_slope_per_pair"]) else h["kern_exp_coef"]
        rows.append({"run": name, "head": t, "pe": h["pe"], "share": h["share"], "decay": decay,
                     "split_r2_bayes": rb, "split_r2_counter": rc, "split_h": p["h"], "split_eps": p["eps"]})
    return rows


def p2_model(name: str) -> list[dict]:
    att = pd.read_parquet(os.path.join(A.run3, f"{name}.attn.parquet"))
    H = pd.read_csv(os.path.join(A.mech, "mech_heads.csv"))
    H = H[(H.run == name) & (H.mass_on_copies > 0.5) & H.alibi_slope_per_pair.notna()].set_index("head")
    rows = []
    for (t, lay), k in att.groupby(["head", "layout"]):
        if t not in H.index:
            continue
        lam = H.loc[t, "alibi_slope_per_pair"]
        g = k.groupby(["n_c", "n_w", "G", "fill"])
        obs = g.apply(lambda d: np.log(d.a[d.recent == 0].sum() + 1e-12) - np.log(d.a[d.recent == 1].sum() + 1e-12),
                      include_groups=False)
        ker = g.apply(lambda d: np.log(np.exp(-lam * d.d[d.recent == 0]).sum()) -
                      np.log(np.exp(-lam * d.d[d.recent == 1]).sum()), include_groups=False)
        off = (obs - ker).groupby(level="G").median()
        for G, v in off.items():
            rows.append({"run": name, "head": t, "layout": lay, "G": G, "offset": v, "slope": lam})
    return rows


A = None


def _init_all(a):
    global A
    A = a
    _init()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default="figs_mech")
    ap.add_argument("--run3", default="figs_run3")
    ap.add_argument("--run4", default="figs_run4")
    ap.add_argument("--out", default="figs_run4")
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    H = pd.read_csv(os.path.join(a.mech, "mech_heads.csv"))
    two = H[(H.layers == 2) & (H.mlp == 0) & (H.h > 0) & (H.eps > 0)].run.unique().tolist()
    with mp.Pool(a.procs, initializer=_init_all, initargs=(a,)) as pool:
        p1 = pd.DataFrame([r for rs in pool.map(p1_model, two) for r in rs])
        p2 = pd.DataFrame([r for rs in pool.map(p2_model, [n for n in two
                                                           if os.path.exists(os.path.join(a.run3, f"{n}.attn.parquet"))])
                           for r in rs])
    p1.to_csv(os.path.join(a.out, "posthoc_p1.csv"), index=False)
    p2.to_csv(os.path.join(a.out, "posthoc_p2.csv"), index=False)

    print("=== P1 (post-hoc): is each copy head's attention split Bayes-shaped or counter-shaped? ===")
    p1["bayes_like"] = p1.split_r2_bayes - p1.split_r2_counter
    p1["pe_l"] = p1.pe.map({"alibi_learn": "ALiBi", "rope": "RoPE", "learned_abs": "learned abs"})
    print(p1.groupby("pe_l").agg(heads=("head", "size"), r2_bayes=("split_r2_bayes", "median"),
                                 r2_counter=("split_r2_counter", "median"),
                                 share_bayes_like=("bayes_like", lambda x: (x > 0).mean())).round(3).to_string())
    rhos = []
    for _, g in p1.groupby("run"):
        if len(g) >= 3:
            rhos.append(spearmanr(g.decay, g.bayes_like).correlation)
    print(f"within-model rank correlation (decay vs Bayes-likeness): median {np.nanmedian(rhos):+.2f}, "
          f"negative in {np.mean(np.array(rhos) < 0):.2f} of {len(rhos)} models (negative = slow heads more Bayes-like)")
    al = p1[p1.pe == "alibi_learn"].copy()
    al["speed"] = pd.qcut(al.decay, 3, labels=["slow", "mid", "fast"])
    print(al.groupby("speed", observed=True)[["split_r2_bayes", "split_r2_counter", "bayes_like"]].median().round(3)
          .to_string())

    print("\n=== P2 (post-hoc): ALiBi copy heads, measured old/recent split minus own-kernel prediction, by gap ===")
    print("(positive = more attention on the old copies than the head's ALiBi slope predicts)")
    p2["speed"] = pd.qcut(p2.slope, 3, labels=["slow", "mid", "fast"])
    print(p2.pivot_table(index=["layout", "speed"], columns="G", values="offset", aggfunc="median",
                         observed=True).round(2).to_string())

    print("\n=== P3 (post-hoc): effective switch rate when the typo-variant signal is removed on the key path ===")
    ps = os.path.join(a.run4, "paths_summary.csv")
    if os.path.exists(ps):
        s = pd.read_csv(ps)
        qk = s[(~s.is_prev) & s.path.isin(["q", "k"]) & s.discount_removed.notna() & (s.d_ce < 0.5)]
        best = qk.loc[qk.groupby("run").discount_removed.idxmax()]
        best = best[best.discount_removed >= 0.5].dropna(subset=["h_eff"])
        ratio = best.h_eff / best.h_eff_base
        print(f"{len(best)} models: h' after / before = median {ratio.median():.2f} "
              f"(IQR {ratio.quantile(.25):.2f}–{ratio.quantile(.75):.2f}); h' rises in {(ratio > 1).mean():.2f}")


if __name__ == "__main__":
    main()
