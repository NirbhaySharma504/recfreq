"""Verdicts for the run-3 tests, each against its pre-registered prediction (plan §14).

python -m recfreq.summarize3 --dir figs_run3 --mech figs_mech --fit figs_fit
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def arch(d: pd.DataFrame) -> pd.Series:
    return ("L" + d.layers.astype(int).astype(str) + np.where(d.mlp == 1, "+MLP", "") +
            np.where(d.pe != "alibi_learn", " " + d.pe.astype(str), "") +
            np.where(d.conflict_frac > 0, " enr", "") +
            np.where(d.run.str.startswith("E2b") | d.run.str.startswith("E1"), "", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="figs_run3")
    ap.add_argument("--mech", default="figs_mech")
    a = ap.parse_args()
    c = pd.read_csv(os.path.join(a.dir, "causal.csv"))
    c["arch"] = arch(c)
    base = c[c.cond == "base"].set_index("run")
    c = c.join(base[["mean_abs_lo", "mean_lo", "h_eff", "ce_rep"] + [f"Gstar_nw{n}" for n in (1, 2, 3)]],
               on="run", rsuffix="_base")
    c["lo_reduction"] = 1 - c.mean_abs_lo / c.mean_abs_lo_base
    c["d_mean_lo"] = c.mean_lo - c.mean_lo_base
    c["d_ce_rep"] = c.ce_rep - c.ce_rep_base
    print(f"runs: {c.run.nunique()}")

    print("\n=== Test 1 (pre-registered): ablate copy heads vs control heads ===")
    print("prediction: copy -> |LO| falls > 70%; control -> |LO| changes < 20%")
    t = c[c.cond.isin(["ablate_copy", "ablate_control", "ablate_all_late"])]
    print(t.groupby(["arch", "cond"])[["lo_reduction", "d_ce_rep", "corr_base"]].median().round(3).unstack("cond").to_string())
    cp = t[t.cond == "ablate_copy"].lo_reduction
    ct = t[t.cond == "ablate_control"].lo_reduction
    print(f"copy: median reduction {cp.median():.2f}, share of runs > 0.70: {(cp > 0.7).mean():.2f}")
    print(f"control: median |change| {ct.abs().median():.2f}, share of runs with |change| < 0.20: {(ct.abs() < 0.2).mean():.2f}")
    al = t[t.cond == "ablate_all_late"].lo_reduction
    print(f"post-hoc, all late heads: median reduction {al.median():.2f}, share > 0.70: {(al > 0.7).mean():.2f}")

    print("\n=== Post-hoc: single-head ablations vs the head's attention decay ===")
    print("counter-model prediction: removing a fast-decay head shifts the decision toward the old value (mean LO up)")
    H = pd.read_csv(os.path.join(a.mech, "mech_heads.csv"))[["run", "head", "kern_exp_coef", "kern_exp_r2", "share",
                                                             "mass_on_copies", "alibi_slope_per_pair"]]
    s = c[c.cond.str.match(r"ablate_L\dH\d")].copy()
    s["head"] = s.cond.str.replace("ablate_", "")
    s = s.merge(H, on=["run", "head"], how="left")
    s["decay"] = s.alibi_slope_per_pair.fillna(s.kern_exp_coef)
    s = s[s.mass_on_copies > 0.5]
    rows = []
    for ar, g in s.groupby("arch"):
        # within each run, rank heads by decay and correlate with the shift they cause when removed
        rhos = [spearmanr(gg.decay, gg.d_mean_lo).correlation for _, gg in g.groupby("run") if len(gg) >= 3]
        rows.append({"arch": ar, "heads": len(g), "runs": g.run.nunique(),
                     "pooled_spearman": spearmanr(g.decay, g.d_mean_lo).correlation,
                     "median_within_run_spearman": np.nanmedian(rhos) if rhos else np.nan,
                     "share_within_run_positive": np.mean(np.array(rhos) > 0) if rhos else np.nan})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== Test 2 (pre-registered): slope surgery (ALiBi) ===")
    print("prediction: x2 -> h' up and G* down; x0.5 -> h' down and G* up; monotone in all seeds")
    for tag in ("copy", "late"):
        g = c[c.cond.isin([f"slope_{tag}_x0.5", "base", f"slope_{tag}_x2.0"]) & (c.pe == "alibi_learn")]
        if g.empty:
            continue
        p = g.pivot_table(index="run", columns="cond", values="h_eff")
        lo_, hi_ = f"slope_{tag}_x0.5", f"slope_{tag}_x2.0"
        p = p.dropna()
        mono_h = ((p[lo_] < p["base"]) & (p["base"] < p[hi_])).mean()
        q = g.pivot_table(index="run", columns="cond", values="Gstar_nw2").dropna()
        fin = q.replace(np.inf, np.nan).dropna()
        mono_g = ((fin[lo_] > fin["base"]) & (fin["base"] > fin[hi_])).mean() if len(fin) else np.nan
        print(f"[{tag} heads] runs {len(p)}: h' monotone in {mono_h:.2f} of runs; median h' x0.5/base/x2 = "
              f"{p[lo_].median():.3f} / {p['base'].median():.3f} / {p[hi_].median():.3f}; "
              f"G*(n_w=2) monotone in {mono_g:.2f} of {len(fin)} runs with finite G*")
        ce = g.pivot_table(index="run", columns="cond", values="ce_rep")
        print(f"    repeat-sighting loss median x0.5/base/x2 = {ce[lo_].median():.4f} / {ce['base'].median():.4f} / "
              f"{ce[hi_].median():.4f}")

    print("\n=== Test 3 (pre-registered): new layouts, descriptions fitted on the original layout only ===")
    print("prediction: Bayes(h', eps') predicts new layouts better than the best counter; refit h', eps' within 2x")
    gz = pd.read_csv(os.path.join(a.dir, "generalize.csv"))
    gz["arch"] = arch(gz)
    print("max |numpy Bayes - torch oracle| on new layouts:", gz.bayes_check.max())
    fams = ["bayes_true", "bayes_free", "counter", "counter2", "additive"]
    print("R² (sp2, sp8; the original sp4 is in-sample):")
    print(gz[gz.layout != "swap4"].groupby("layout")[[f"r2.{f}" for f in fams]].median().round(3).to_string())
    print("mean absolute error, all layouts (swap4: Bayes is constant there, so R² is undefined):")
    print(gz.groupby("layout")[[f"mae.{f}" for f in fams]].median().round(3).to_string())
    sw = gz[gz.layout == "swap4"]
    print(f"swap4: model LO mean {sw.y_mean.median():.2f} (sd across cells {sw.y_sd.median():.2f}); "
          f"true Bayes {sw.bayes_mean.median():.2f}; predicted sd across cells: " +
          ", ".join(f"{f} {sw[f'predsd.{f}'].median():.2f}" for f in fams))
    new = gz[gz.layout.isin(["sp2", "sp8"])].copy()
    new["best_counter"] = new[["r2.counter", "r2.counter2", "r2.additive"]].max(1)
    print(f"Bayes(h', eps') beats the best counter description in {(new['r2.bayes_free'] > new.best_counter).mean():.2f} "
          f"of {len(new)} (run, layout) pairs")
    print(new.groupby("arch").apply(lambda d: pd.Series({"bayes_free": d["r2.bayes_free"].median(),
                                                          "best_counter": d.best_counter.median(),
                                                          "win_rate": (d["r2.bayes_free"] > d.best_counter).mean()}))
          .round(3).to_string())
    swm = sw.copy()
    swm["best_counter_mae"] = swm[["mae.counter", "mae.counter2", "mae.additive"]].min(1)
    print(f"swap4: Bayes(h', eps') has lower error than the best counter in "
          f"{(swm['mae.bayes_free'] < swm.best_counter_mae).mean():.2f} of {len(swm)} runs")
    new["h_ratio"] = new.refit_h / new.fit_h
    new["eps_ratio"] = new.refit_eps / new.fit_eps
    ok = new[(new.fit_h > 0) & (new.refit_h > 0)]
    within = ((ok.h_ratio.between(0.5, 2)) & (ok.eps_ratio.between(0.5, 2))).mean()
    print(f"refit (h', eps') within 2x of the original-layout fit: {within:.2f} of {len(ok)}")
    print(ok.groupby("layout")[["h_ratio", "eps_ratio"]].median().round(2).to_string())

    print("\n=== Test 4: what besides distance drives copy-head attention? ===")
    at = pd.read_csv(os.path.join(a.dir, "attention.csv"))
    at = at[at.mass > 0.5]
    at["arch"] = arch(at.merge(c[["run", "layers", "mlp", "pe", "conflict_frac"]].drop_duplicates(), on="run"))
    at = at.merge(c[["run", "layers", "mlp", "pe", "conflict_frac"]].drop_duplicates(), on="run")
    at["arch"] = arch(at)
    print(at.groupby("arch")[["r2_d", "r2_d_rec_var", "r2_full", "b_d", "b_recent", "b_variant", "b_lnsame"]]
          .median().round(3).to_string())
    print("sign consistency (share of heads with coefficient > 0):")
    print(at.groupby("arch")[["b_recent", "b_variant", "b_lnsame"]].agg(lambda x: (x > 0).mean()).round(2).to_string())
    print("per layout, recent-block offset beyond distance (sp4: recent = typo-variant; swap4: old = typo-variant):")
    print(at.groupby("arch")[["b_recent_sp4", "b_recent_swap4", "b_d_sp4", "b_d_swap4"]].median().round(3).to_string())
    print("share of heads where the recent-block offset has the same sign in both layouts (distance-like):",
          round(float((np.sign(at.b_recent_sp4) == np.sign(at.b_recent_swap4)).mean()), 2))


if __name__ == "__main__":
    main()
