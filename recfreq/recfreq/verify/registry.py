"""Audit 3.1: claims registry. Every headline number in the report, recomputed from the raw per-run files with
freshly written code (own boundary / cliff / aggregation code; fits re-used only where they are themselves
tested by test_recovery.py and re-checked by robustness.py). Prints reported vs recomputed and a verdict.

python -m recfreq.verify.registry --out verify_out
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

R = "results/runs"
CELLS = [(nc, nw, G) for nc in (1, 2, 4, 8) for nw in (1, 2, 3, 4) for G in (2, 4, 8, 16, 32, 64, 128)]


def probes(run):
    return pd.read_parquet(os.path.join(R, run, "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS]


def done(run):
    return json.load(open(os.path.join(R, run, "done.json")))


def runs(prefix):
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(R, prefix)))


# ---- fresh implementations ---------------------------------------------------------------------------------
def my_gstar(p, col, n_c, n_w):
    """Gap where the curve first crosses zero, linear interpolation in log2 G (own code)."""
    c = p.xs((n_c, n_w), level=("n_c", "n_w"))[col]
    G = np.array(c.index, float)
    v = c.to_numpy()
    if v[0] < 0:
        return 0.0
    for i in range(1, len(v)):
        if v[i] < 0:
            t = v[i - 1] / (v[i - 1] - v[i])
            return float(2 ** (math.log2(G[i - 1]) + t * (math.log2(G[i]) - math.log2(G[i - 1]))))
    return math.inf


def my_cliff(p, col):
    out = []
    for n_c in (2, 4, 8):
        for n_w in (1, 2, 3, 4):
            v = p.xs((n_c, n_w), level=("n_c", "n_w"))[col].to_numpy()
            out.append(np.max(v[:-1] - v[1:]))
    return float(np.mean(out))


ROWS = []


def check(cid, what, reported, value, tol, note=""):
    if isinstance(reported, (tuple, list)):
        ok = all(abs(r - v) <= tol for r, v in zip(reported, value))
        rep, val = ", ".join(f"{x:g}" for x in reported), ", ".join(f"{x:.4g}" for x in value)
    else:
        ok = abs(reported - value) <= tol
        rep, val = f"{reported:g}", f"{value:.4g}"
    ROWS.append({"id": cid, "claim": what, "reported": rep, "recomputed": val, "tol": tol,
                 "verdict": "ok" if ok else "MISMATCH", "note": note})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # R1 gate
    g = [done(r)["final"] for r in runs("E0_*")]
    check("R1", "gate: repeat-sighting accuracy (both seeds)", (1.0, 1.0), [x["acc_rep"] for x in g], 1e-6)
    check("R1b", "gate: |excess| below 1e-4 nats", 0, max(abs(x["excess"]) for x in g), 1e-4)

    # R2 run-1 excess range over (point, architecture) means
    ex = pd.DataFrame([{"run": r, **done(r)["args"], "excess": done(r)["final"]["excess"]} for r in runs("E1_*")])
    m = ex.groupby(["h", "eps", "layers", "mlp"]).excess.mean()
    check("R2", "run 1: excess loss range over points x architectures (min, max)", (0.0004, 0.021),
          (m.min(), m.max()), 0.0006)

    # R3 cliff, run-1/2 table (point C and D; L2, L2+MLP, L3 ALiBi), model vs Bayes
    def cliff_mean(prefix, col):
        return np.mean([my_cliff(probes(r), col) for r in runs(prefix)])
    check("R3", "cliff at C: Bayes, L2, L2+MLP, L3", (0.90, 3.14, 2.96, 1.59),
          (cliff_mean("E1_h0.01_e0.1_L2_alibi_learn_attn_s*", "lo_bayes"),
           cliff_mean("E1_h0.01_e0.1_L2_alibi_learn_attn_s*", "lo_model"),
           cliff_mean("E1_h0.01_e0.1_L2_alibi_learn_mlp_s*", "lo_model"),
           cliff_mean("E1_h0.01_e0.1_L3_alibi_learn_attn_s*", "lo_model")), 0.02)
    check("R3b", "cliff at D: Bayes, L2, L2+MLP, L3", (1.37, 2.25, 2.36, 1.62),
          (cliff_mean("E1_h0.03_e0.2_L2_alibi_learn_attn_s*", "lo_bayes"),
           cliff_mean("E1_h0.03_e0.2_L2_alibi_learn_attn_s*", "lo_model"),
           cliff_mean("E1_h0.03_e0.2_L2_alibi_learn_mlp_s*", "lo_model"),
           cliff_mean("E1_h0.03_e0.2_L3_alibi_learn_attn_s*", "lo_model")), 0.02)
    check("R3c", "cliff RoPE / learned-abs L2 at C", (1.61, 2.20),
          (cliff_mean("E2a_h0.01_e0.1_L2_rope_attn_s*", "lo_model"),
           cliff_mean("E2a_h0.01_e0.1_L2_learned_abs_attn_s*", "lo_model")), 0.02)

    # R4 switch-over gaps at C, n_c = 2 (L2 ALiBi): model mean and Bayes
    gs = np.array([[my_gstar(probes(r), "lo_model", 2, nw) for nw in (1, 2, 3)]
                   for r in runs("E1_h0.01_e0.1_L2_alibi_learn_attn_s*")])
    check("R4", "G* at C, L2 (n_w = 1, 2, 3), mean over seeds", (78.3, 38.5, 20.0), gs.mean(0), 0.6)
    pb = probes(runs("E1_h0.01_e0.1_L2_alibi_learn_attn_s*")[0])
    check("R4b", "Bayes G* at C (n_w = 2, 3)", (51.6, 6.87),
          (my_gstar(pb, "lo_bayes", 2, 2), my_gstar(pb, "lo_bayes", 2, 3)), 0.1)
    # Bayes per-copy weight at h = 0.003, eps = 0.05 (measured 2.948)
    pbb = probes(runs("E2b_h0.003_e0.05_L2_alibi_learn_attn_s*")[0])
    gb = [my_gstar(pbb, "lo_bayes", 2, nw) for nw in (1, 2, 3, 4)]
    v = [math.log(x) - math.log(y) for x, y in zip(gb, gb[1:]) if 0 < x < math.inf and 0 < y < math.inf]
    check("R4c", "measured Bayes per-copy weight at h=0.003, eps=0.05", 2.948, float(np.mean(v)), 0.005)

    # R5 learned slope table (strongest induction head, layer >= 1), L2 ALiBi
    def lam(prefix):
        out = []
        for r in runs(prefix):
            hd = pd.DataFrame(done(r)["heads"])
            hd = hd[hd.layer >= 1]
            out.append(hd.loc[hd.ind_score.idxmax(), "slope"])
        return float(np.mean(out))
    check("R5", "learned slope L2: A, C, D, B", (0.002, 0.078, 0.115, 0.572),
          (lam("E1_h0.0_e0.2_L2_alibi_learn_attn_s*"), lam("E1_h0.01_e0.1_L2_alibi_learn_attn_s*"),
           lam("E1_h0.03_e0.2_L2_alibi_learn_attn_s*"), lam("E1_h0.03_e0.0_L2_alibi_learn_attn_s*")), 0.002)

    # R6 rule fits (fits.csv): win counts and median held-out R²
    f = pd.read_csv("figs_fit/fits.csv")
    s = f[~f.arch.str.contains("uniform")]
    fam = ["bayes_true", "bayes_free", "counter_exp", "counter_pow", "additive"]
    best = s[[x + ".cv_r2" for x in fam]].idxmax(1).str.split(".").str[0]
    check("R6", "Bayes(h', eps') best held-out description: count of 84", 75, int((best == "bayes_free").sum()), 0)
    med = s.groupby("arch")[["bayes_free.cv_r2", "counter_exp.cv_r2"]].median()
    check("R6b", "median held-out R² range, Bayes(h', eps') (min, max over arch)", (0.845, 0.954),
          (med["bayes_free.cv_r2"].min(), med["bayes_free.cv_r2"].max()), 0.002)
    check("R6c", "median held-out R² range, counter (min, max over arch)", (0.371, 0.742),
          (med["counter_exp.cv_r2"].min(), med["counter_exp.cv_r2"].max()), 0.002)

    # R7 laws on the 2-layer ALiBi grid (setting medians)
    gr = s[(s.arch == "L2") & s.exp.isin(["E1", "E2b"])]
    sm = gr.groupby(["h", "eps"])[["bfree.h", "bfree.eps"]].median().reset_index()
    sm = sm[sm["bfree.h"] > 0]
    b1, a1 = np.polyfit(np.log(sm.h), np.log(sm["bfree.h"]), 1)
    lg = lambda x: np.log(x / (1 - x))
    b2, a2 = np.polyfit(lg(sm.eps), lg(sm["bfree.eps"]), 1)
    check("R7", "h' law: coefficient, exponent", (1.45, 0.74), (math.exp(a1), b1), 0.02)
    check("R7b", "eps' law: intercept, slope", (-0.35, 0.44), (a2, b2), 0.02)
    ratio = sm["bfree.h"] / sm.h
    check("R7c", "h'/h range on the grid (min, max)", (2.8, 14), (ratio.min(), ratio.max()), 0.3)

    # R8 bridge simulation and models (in-sample)
    br = pd.read_csv("figs_run3/bridge.csv")
    check("R8", "random counter sums: median R² Bayes(h', eps'), counter", (0.74, 0.99),
          (br.r2_bayes_free.median(), br.r2_counter.median()), 0.02)
    check("R8b", "trained models: median in-sample R² Bayes(h', eps'), counter", (0.94, 0.72),
          (s["bayes_free.r2"].median(), s["counter_exp.r2"].median()), 0.02)

    # R9 layouts (generalize.csv)
    gz = pd.read_csv("figs_run3/generalize.csv")
    new = gz[gz.layout.isin(["sp2", "sp8"])].copy()
    new["best_counter"] = new[["r2.counter", "r2.counter2", "r2.additive"]].max(1)
    pairs = gz[gz.layout != "sp4"].copy()
    pairs["best_counter_mae"] = pairs[["mae.counter", "mae.counter2", "mae.additive"]].min(1)
    win = new["r2.bayes_free"] > new.best_counter  # spacing 2 and 8: 84 models x 2 = 168 pairs (swapped: R9c)
    check("R9", "spacing 2/8: Bayes(h', eps') beats best counter, share of 168 pairs", 0.90, float(win.mean()), 0.01,
          f"n = {len(new)}")
    check("R9b", "median R² on spacing 2, 8", (0.925, 0.930),
          tuple(new.groupby("layout")["r2.bayes_free"].median().reindex(["sp2", "sp8"])), 0.002)
    sw = pairs[pairs.layout == "swap4"]
    check("R9c", "swapped layout: Bayes(h', eps') lower error share", 0.94,
          float((sw["mae.bayes_free"] < sw.best_counter_mae).mean()), 0.01)
    ok = new[(new.fit_h > 0) & (new.refit_h > 0)]
    within = ((ok.refit_h / ok.fit_h).between(0.5, 2) & (ok.refit_eps / ok.fit_eps).between(0.5, 2)).mean()
    check("R9d", "refit (h', eps') within 2x", 0.97, float(within), 0.01)

    # R10 slope surgery (causal.csv)
    c = pd.read_csv("figs_run3/causal.csv")
    al = c[c.pe == "alibi_learn"].pivot_table(index="run", columns="cond", values="h_eff")
    al = al[["slope_copy_x0.5", "base", "slope_copy_x2.0"]].dropna()
    mono = ((al["slope_copy_x0.5"] < al["base"]) & (al["base"] < al["slope_copy_x2.0"])).sum()
    check("R10", "slope surgery: monotone models of 57", 56, int(mono), 0)
    check("R10b", "slope surgery: median h' x0.5, x1, x2", (0.030, 0.055, 0.109), tuple(al.median()), 0.002)
    ce = c[c.pe == "alibi_learn"].pivot_table(index="run", columns="cond", values="ce_rep")
    check("R10c", "slope surgery: median loss change x0.5, x2 (nats)", (0.01, 0.02),
          ((ce["slope_copy_x0.5"] - ce["base"]).median(), (ce["slope_copy_x2.0"] - ce["base"]).median()), 0.006)
    b = c[c.cond == "base"].set_index("run").mean_abs_lo
    alll = c[c.cond == "ablate_all_late"].set_index("run").mean_abs_lo
    check("R10d", "ablating all late heads: median |LO| reduction", 0.96, float((1 - alll / b).median()), 0.01)

    # R11 test 6 (scale)
    e3 = pd.read_csv("figs_fit_E3/fits.csv")
    e3["variant"] = np.where(e3.arch.str.contains("d256"), "d256", "60k")
    base = f[f.arch.isin(["L2", "L2 rope"]) & f.exp.isin(["E1", "E2a"])].assign(variant="base")
    d = pd.concat([base, e3])
    d = d[((d.h == 0.01) & (d.eps == 0.1)) | ((d.h == 0.03) & (d.eps == 0.2))]
    d["pe"] = np.where(d.arch.str.contains("rope"), "rope", "alibi")
    mm = d.groupby(["h", "pe", "variant"])["bfree.h"].median().unstack()
    r_al = (mm.loc[(slice(None), "alibi"), "d256"] / mm.loc[(slice(None), "alibi"), "d256"].index.get_level_values(0))
    r_ro = (mm.loc[(slice(None), "rope"), "d256"] / mm.loc[(slice(None), "rope"), "d256"].index.get_level_values(0))
    check("R11", "width 256: h'/h ALiBi (min, max)", (2.9, 5.3), (r_al.min(), r_al.max()), 0.1)
    check("R11b", "width 256: h'/h RoPE (min, max)", (1.3, 1.5), (r_ro.min(), r_ro.max()), 0.1)
    imp = mm["base"] / mm["60k"]
    check("R11c", "3x training: h' change factor (min, max)", (0.76, 1.17), (imp.min(), imp.max()), 0.02)

    # R12 mechanism
    mc = pd.read_csv("figs_mech/mech.csv")
    H = pd.read_csv("figs_mech/mech_heads.csv")
    A = H[(H.share >= 0.1) & (H.pe == "alibi_learn")].dropna(subset=["alibi_slope_per_pair"])
    check("R12", "ALiBi copy heads: n, r(kernel decay, slope)", (124, 0.990),
          (len(A), np.corrcoef(A.kern_exp_coef, A.alibi_slope_per_pair)[0, 1]), 0.002)
    att = mc[(mc.mlp == 0)].copy()
    att["arch"] = att.layers.astype(str) + att.pe + att.conflict_frac.astype(str)
    md = att.groupby("arch").prop1_model_r2_copy.median()
    check("R12b", "decision from measured attention: median R² range over attention-only arch", (0.90, 0.99),
          (md.min(), md.max()), 0.01)

    out = pd.DataFrame(ROWS)
    out.to_csv(os.path.join(a.out, "registry.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(out[["id", "claim", "reported", "recomputed", "verdict"]].to_string(index=False))
    print(f"\n{(out.verdict == 'ok').sum()} ok, {(out.verdict != 'ok').sum()} mismatches")


if __name__ == "__main__":
    main()
