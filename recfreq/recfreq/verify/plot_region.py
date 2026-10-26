"""Plan §19 figures: survival curves (time belief), natural law (switch points), region map in two coordinate systems.

python -m recfreq.verify.plot_region --figs figs_region
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .natural_law import g_frozen_law, g_natural
from .switch_points import model_rows

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, ORANGE, GREEN, GREY = "#16213A", "#2B59C3", "#D9822B", "#2E8B57", "#9AA3B5"


def survival_fig(runs, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    picks = [("E2b_h0.003_e0.1_L2_alibi_learn_attn_s0", "h = 0.003, ε = 0.1"),
             ("E1_h0.01_e0.1_L2_alibi_learn_attn_s0", "h = 0.01, ε = 0.1"),
             ("E2b_h0.03_e0.1_L2_alibi_learn_attn_s0", "h = 0.03, ε = 0.1")]
    for ax, (run, title) in zip(axs, picks):
        f = os.path.join(runs, run, "survival.parquet")
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        d = d[d.n == 1]
        ax.plot(d.d, d.s_bayes, "--", c=INK, lw=2, label="exact Bayes")
        ax.plot(d.d, d.s_model, "o-", c=BLUE, lw=2, label="ALiBi model")
        rope = glob.glob(os.path.join(runs, run.replace("alibi_learn", "rope").replace("E2b_", "E5r_").replace("E1_", "E2a_"), "survival.parquet"))
        if rope:
            r = pd.read_parquet(rope[0]); r = r[r.n == 1]
            ax.plot(r.d, r.s_model, "s-", c=ORANGE, lw=1.5, ms=4, label="RoPE model")
        ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel("distance to the single old copy (pairs)")
    axs[0].set_ylabel("belief that the old value still holds"); axs[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "survival.png"), dpi=140); plt.close(fig)


def natural_fig(runs, out):
    M = model_rows(runs, prefixes=("E1_", "E2b_", "E5a_", "E6_"))
    M = M[(M.pe == "alibi_learn") & (M.cf == 0)]
    rows = []
    for r in M.itertuples():
        for nw in (1, 2, 3, 4):
            g = getattr(r, f"G{nw}")
            if np.isnan(g):
                continue
            rows.append({"new": r.run.startswith("E6_"), "nw": nw, "obs": g, "nat": g_natural(r.h, r.eps, nw),
                         "law": g_frozen_law(r.h, r.eps, nw)})
    D = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, col, name in ((axs[0], "law", "frozen law (Bayes with h′, ε′)"), (axs[1], "nat", "natural law (switch points)")):
        for new, c, lab in ((False, GREY, "settings used to fit"), (True, BLUE, "new settings (E6)")):
            d = D[D.new == new]
            ax.scatter(d[col], d.obs, s=18, c=c, alpha=0.8, label=f"{lab} (n={len(d)})")
        ax.plot([1, 300], [1, 300], c=INK, lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1.5, 300); ax.set_ylim(1.5, 300)
        ax.set_xlabel(f"predicted switch gap G*"); ax.set_title(name)
        d = D[D.new].dropna(subset=[col])
        if len(d):
            e = np.median(np.abs(np.log(d.obs / d[col])))
            ax.text(0.04, 0.92, f"new settings: median error ×{math.exp(e):.2f}", transform=ax.transAxes)
    axs[0].set_ylabel("model's switch gap G* (pairs)"); axs[0].legend(frameon=False, loc="lower right", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "natural_law.png"), dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--figs", default="figs_region")
    a = ap.parse_args()
    os.makedirs(a.figs, exist_ok=True)
    survival_fig(a.runs, a.figs)
    natural_fig(a.runs, a.figs)


if __name__ == "__main__":
    main()
