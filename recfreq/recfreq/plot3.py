"""Figures for run 3: slope surgery, per-head ablation vs decay, layout generalization, bridge simulation, scale.

python -m recfreq.plot3 --run3 figs_run3 --mech figs_mech --fit figs_fit --fit_e3 figs_fit_E3 --out figs_paper
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .layouts import CELLS, LAY4, Lay


def fig_surgery(run3: str, mech: str, out: str):
    c = pd.read_csv(os.path.join(run3, "causal.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    a = c[(c.pe == "alibi_learn")].pivot_table(index="run", columns="cond", values="h_eff")
    a = a[["slope_copy_x0.5", "base", "slope_copy_x2.0"]].dropna()
    for _, r in a.iterrows():
        ax.plot([0.5, 1, 2], r.values, "-", color="#1f4fb8", alpha=0.25, lw=0.8)
    ax.plot([0.5, 1, 2], a.median().values, "o-", color="k", lw=2, label="median")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([0.5, 1, 2], ["×0.5", "×1", "×2"])
    ax.set_xlabel("copy heads' ALiBi slope"); ax.set_ylabel("fitted effective switch rate h′")
    mono = ((a["slope_copy_x0.5"] < a["base"]) & (a["base"] < a["slope_copy_x2.0"])).mean()
    ax.set_title(f"slope surgery moves h′ ({len(a)} ALiBi models, monotone in {mono:.0%})", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1]
    l = c[(c.pe == "alibi_learn")].pivot_table(index="run", columns="cond", values="ce_rep")
    l = l[["slope_copy_x0.5", "base", "slope_copy_x2.0", "ablate_copy", "ablate_all_late"]].dropna()
    d = l.sub(l["base"], axis=0)
    ax.boxplot([d["slope_copy_x0.5"], d["slope_copy_x2.0"], d["ablate_copy"], d["ablate_all_late"]],
               tick_labels=["slope ×0.5", "slope ×2", "ablate copy\nheads", "ablate all\nlate heads"], showfliers=False)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_ylabel("change in repeat-sighting loss (nats)")
    ax.set_title("changing the forgetting rate barely costs loss;\nremoving the heads costs a lot", fontsize=9)
    ax = axes[2]
    H = pd.read_csv(os.path.join(mech, "mech_heads.csv"))[["run", "head", "alibi_slope_per_pair", "kern_exp_coef",
                                                           "mass_on_copies"]]
    s = c[c.cond.str.match(r"ablate_L\dH\d") & (c.layers == 2) & (c.mlp == 0) & (c.pe == "alibi_learn")].copy()
    base = c[c.cond == "base"].set_index("run").mean_lo
    s["head"] = s.cond.str.replace("ablate_", "")
    s = s.merge(H, on=["run", "head"])
    s = s[s.mass_on_copies > 0.5]
    s["d_lo"] = s.mean_lo - s.run.map(base)
    ax.scatter(s.alibi_slope_per_pair, s.d_lo, s=10, c="#1f4fb8", alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("head's ALiBi slope (per pair)"); ax.set_ylabel("shift in mean log p(c)/p(w) when removed")
    ax.set_title(f"L2 · ALiBi: removing fast heads favours the old value\n({len(s)} heads, {s.run.nunique()} models)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "run3_causal.png"), dpi=130)
    plt.close(fig)


def fig_layouts(run3: str, out: str):
    g = pd.read_csv(os.path.join(run3, "generalize.csv"))
    fams = [("bayes_true", "Bayes, true", "#6b5aa8"), ("bayes_free", "Bayes, fitted h′, ε′", "#1f4fb8"),
            ("counter", "counter", "#e45756"), ("counter2", "2-counter sum", "#f58518"),
            ("additive", "additive", "#9aa5b1")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax = axes[0]
    lays = ["sp2", "sp8", "swap4"]
    w = 0.16
    for i, (f, lab, col) in enumerate(fams):
        q = g.groupby("layout")[f"mae.{f}"].median().reindex(lays)
        ax.bar(np.arange(3) + (i - 2) * w, q, w, color=col, label=lab)
    ax.set_xticks(range(3), ["spacing 2", "spacing 8", "swapped"])
    ax.set_yscale("log")
    ax.set_ylabel("median absolute error (log-odds)")
    ax.set_title("fitted on the original layout only, predicting new layouts (84 models)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    run = "E1_h0.01_e0.1_L2_alibi_learn_attn_s0"
    s = pd.read_parquet(os.path.join(run3, f"{run}.surf.parquet"))
    sw = s[(s.layout == "swap4") & (s.cond == "base")].set_index(["n_c", "n_w", "G"]).loc[CELLS]
    cells = np.array(CELLS)
    m = cells[:, 1] == 1
    for nc, col in zip((1, 2, 4, 8), ("#9aa5b1", "#4c78a8", "#f58518", "#e45756")):
        mm = m & (cells[:, 0] == nc)
        ax.plot(cells[mm, 2], sw.lo.to_numpy()[mm], "o-", color=col, ms=4, label=f"model, n_c={nc}")
    ax.plot(cells[m & (cells[:, 0] == 2), 2], sw.lo_bayes.to_numpy()[m & (cells[:, 0] == 2)], "k--", label="Bayes")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("gap G (pairs)"); ax.set_ylabel("log p(c)/p(w), n_w = 1")
    ax.set_title("swapped layout (old copies are typo-variants of the recent value):\n"
                 "model and Bayes always pick the recent value; a counter would not", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "run3_layouts.png"), dpi=130)
    plt.close(fig)


def fig_bridge(run3: str, fit: str, out: str):
    b = pd.read_csv(os.path.join(run3, "bridge.csv"))
    f = pd.read_csv(os.path.join(fit, "fits.csv"))
    f = f[~f.arch.str.contains("uniform")]
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [b[b.kind == "single"].r2_bayes_free, b[b.kind == "sum2"].r2_bayes_free, b[b.kind == "sum3"].r2_bayes_free,
            f["bayes_free.r2"]]
    data2 = [b[b.kind == "single"].r2_counter, b[b.kind == "sum2"].r2_counter, b[b.kind == "sum3"].r2_counter,
             f["counter_exp.r2"]]
    pos = np.arange(4)
    bp1 = ax.boxplot(data, positions=pos - 0.18, widths=0.3, showfliers=False, patch_artist=True)
    bp2 = ax.boxplot(data2, positions=pos + 0.18, widths=0.3, showfliers=False, patch_artist=True)
    for p in bp1["boxes"]:
        p.set_facecolor("#9db4ea")
    for p in bp2["boxes"]:
        p.set_facecolor("#f2a9a3")
    ax.set_xticks(pos, ["random\ncounter", "random sum\nof 2", "random sum\nof 3", "trained\nmodels (84)"])
    ax.set_ylabel("in-sample R²")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["fit by Bayes(h′, ε′)", "fit by one counter"], fontsize=8)
    ax.set_title("random counter sums look like counters; trained models look like Bayes", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "run3_bridge.png"), dpi=130)
    plt.close(fig)


def fig_scale(fit: str, fit_e3: str, out: str):
    base = pd.read_csv(os.path.join(fit, "fits.csv"))
    base = base[base.arch.isin(["L2", "L2 rope"]) & base.exp.isin(["E1", "E2a"])]
    base["variant"] = "width 128, 20k steps"
    e3 = pd.read_csv(os.path.join(fit_e3, "fits.csv"))
    e3["variant"] = np.where(e3.arch.str.contains("d256"), "width 256, 20k steps", "width 128, 60k steps")
    d = pd.concat([base, e3])
    d = d[((d.h == 0.01) & (d.eps == 0.1)) | ((d.h == 0.03) & (d.eps == 0.2))]
    d["pe"] = np.where(d.arch.str.contains("rope"), "RoPE", "ALiBi")
    d["ratio"] = d["bfree.h"] / d.h
    order = ["width 128, 20k steps", "width 128, 60k steps", "width 256, 20k steps"]
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = {"ALiBi": "#1f4fb8", "RoPE": "#e45756"}
    mk = {0.01: "o", 0.03: "s"}
    for (pe, h), g in d.groupby(["pe", "h"]):
        m = g.groupby("variant").ratio.median().reindex(order)
        x = np.arange(3) + (0.06 if pe == "RoPE" else -0.06)
        ax.plot(x, m.values, marker=mk[h], color=cols[pe], label=f"{pe}, {'C' if h == 0.01 else 'D'}")
        for i, v in enumerate(order):
            pts = g[g.variant == v].ratio
            ax.scatter(np.full(len(pts), x[i]), pts, s=8, color=cols[pe], alpha=0.4)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.axhline(1.5, color="k", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(range(3), order)
    ax.set_ylabel("h′ / true h  (1 = calibrated)")
    ax.set_title("more training does not fix the switch rate; width helps RoPE most", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "run3_scale.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run3", default="figs_run3")
    ap.add_argument("--mech", default="figs_mech")
    ap.add_argument("--fit", default="figs_fit")
    ap.add_argument("--fit_e3", default="figs_fit_E3")
    ap.add_argument("--out", default="figs_paper")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_surgery(a.run3, a.mech, a.out)
    fig_layouts(a.run3, a.out)
    fig_bridge(a.run3, a.fit, a.out)
    fig_scale(a.fit, a.fit_e3, a.out)
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
