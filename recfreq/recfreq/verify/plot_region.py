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

plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, ORANGE, GREEN, GREY = "#16213A", "#2B59C3", "#D9822B", "#2E8B57", "#9AA3B5"


def survival_fig(runs, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    picks = [("E6_h0.0003_e0.1_L2_alibi_learn_attn_s0", "h = 0.0003, ε = 0.1"),
             ("E1_h0.01_e0.1_L2_alibi_learn_attn_s0", "h = 0.01, ε = 0.1"),
             ("E6_h0.1_e0.1_L2_alibi_learn_attn_s0", "h = 0.1, ε = 0.1")]
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
        ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel("distance to the old copy (pairs)")
        if rope: ax.legend(frameon=False, loc="lower left")
    axs[0].set_ylabel("belief: old value still holds"); axs[0].legend(frameon=False, loc="lower left")
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


def region_slide_fig(csv, out):
    """Slide figure: (a) where the frozen law works; (b) h'/h against h; (c) eps' against h. From region_map.csv."""
    from .law_scope import law
    R = pd.read_csv(csv)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axs[0]
    ax.add_patch(plt.Rectangle((0.003, 0.05), 0.027, 0.25, fill=False, ls="--", ec=INK, lw=1.2))
    ref = R[R.in_fit_grid].ratio.median()
    m = R[R.measurable]
    sc = ax.scatter(m.h, m.eps, c=np.log2(m.ratio.clip(1, 8)), cmap="RdYlGn_r", vmin=0, vmax=3, s=110,
                    edgecolor=INK, lw=0.6)
    d = R[~R.measurable]
    ax.scatter(d.h, d.eps, marker="x", c=GREY, s=70)
    ax.text(0.00026, 0.255, "× not\nmeasurable", color="#5A6377", fontsize=11)
    cb = fig.colorbar(sc, ax=ax, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["×1", "×2", "×4", "≥×8"]); cb.set_label("law error ÷ model's own Bayes fit")
    ax.set_xscale("log"); ax.set_xlabel("switch rate h"); ax.set_ylabel("typo rate ε")
    ax.set_title(f"(a) law error (box: fitted range, ×{ref:.1f} there)")
    hs = np.exp(np.linspace(np.log(2e-4), np.log(0.2), 50))
    for e, c, m in ((0.1, BLUE, "o"), (0.2, ORANGE, "s")):
        d = R[np.isclose(R.eps, e)].sort_values("h")
        axs[1].plot(d.h, d.h_fit / d.h, m, c=c, ms=7, label=f"models, ε = {e}")
        axs[2].plot(d.h, d.eps_fit, m + "-", c=c, ms=7, label=f"models, ε = {e}")
        axs[2].axhline(law(0.01, e)[1], c=c, ls="--", lw=1.5, alpha=0.8)
        axs[2].axhline(e, c=c, ls=":", lw=1, alpha=0.6)
    axs[1].plot(hs, [law(h, 0.1)[0] / h for h in hs], "-", c=INK, lw=1.5, label="frozen law")
    axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("switch rate h")
    axs[1].set_ylabel("believed / true switch rate  h′/h"); axs[1].set_title("(b) h′ follows the law over 2 decades")
    axs[1].legend(frameon=False, fontsize=10)
    axs[2].set_xscale("log"); axs[2].set_xlabel("switch rate h"); axs[2].set_ylabel("believed typo rate ε′")
    axs[2].set_title("(c) ε′ drifts with h (law: dashed)")
    for ax in axs[1:]:
        ax.axvspan(0.003, 0.03, color="#E6E8EE", zorder=-1)
    fig.tight_layout(); fig.savefig(os.path.join(out, "region_slide.png"), dpi=150); plt.close(fig)


def mech_fig(out, sw="verify_out/slope_vs_w.csv", surg="verify_out/surgery_E6.csv"):
    """(a) trained copy-head slope vs h; (b) evidence weight vs slope; (c) slope surgery on unseen models."""
    from matplotlib.ticker import NullFormatter
    from scipy.stats import spearmanr
    M = pd.read_csv(sw)
    o = M[(M.cf == 0)].dropna(subset=["lam"])
    fig, axs = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 1, 0.75]})
    ax = axs[0]
    ax.scatter(o.h, o.lam, c=BLUE, s=30, alpha=0.7, edgecolor="white", lw=0.5)
    rho = spearmanr(np.log(o.h), np.log(o.lam)).statistic
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("switch rate h"); ax.set_ylabel("copy-head ALiBi slope λ")
    ax.set_title(f"(a) the clock sets the slope (ρ = {rho:.2f})")
    ax = axs[1]
    d = o.dropna(subset=["w"])
    d = d[d.eps <= 0.3]
    rho = spearmanr(d.lam, d.w).statistic
    ax.scatter(d.lam, d.w, c=BLUE, s=30, alpha=0.8, edgecolor="white", lw=0.5, label=f"trained (ε ≤ 0.3), ρ = {rho:.2f}")
    c = M[M.cf > 0].dropna(subset=["w", "lam"])
    ax.scatter(c.lam, c.w, c=ORANGE, marker="s", s=36, label="trained with extra conflicts")
    ax.axhline(1, ls="--", c=INK, lw=1.2); ax.text(0.031, 1.05, "Bayes", color=INK)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("copy-head ALiBi slope λ"); ax.set_ylabel("evidence weight w")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.03, 0.1, 0.3]); ax.set_xticklabels(["0.03", "0.1", "0.3"])
    ax.set_yticks([0.1, 0.3, 1]); ax.set_yticklabels(["0.1", "0.3", "1"])
    ax.set_title("(b) the slope tracks the evidence weight"); ax.legend(frameon=False, fontsize=11, loc="lower right")
    ax = axs[2]
    S = pd.read_csv(surg)
    for i, (cond, lab) in enumerate((("w_x0.5", "×0.5"), (None, "×1"), ("w_x2.0", "×2"))):
        if cond is None:
            ax.plot(i, 1, "o", c=INK, ms=9)
            continue
        r = (S[cond] / S.w_base).dropna()
        q = r.quantile([0.25, 0.5, 0.75]).values
        ax.plot([i, i], [q[0], q[2]], c=BLUE, lw=6, alpha=0.4)
        ax.plot(i, q[1], "o", c=BLUE, ms=10)
        ax.text(i + 0.12, q[1], f"{q[1]:.2f}\n(n = {len(r)})", va="center", fontsize=12)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["×0.5", "×1", "×2"]); ax.set_xlim(-0.4, 2.7)
    ax.set_xlabel("slope surgery on 25 unseen models"); ax.set_ylabel("w ÷ w unedited (median, IQR)")
    ax.set_title("(c) editing the slope moves w")
    fig.tight_layout(); fig.savefig(os.path.join(out, "mech_slope.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--figs", default="figs_region")
    ap.add_argument("--region-csv", default="verify_out/region_map.csv")
    a = ap.parse_args()
    os.makedirs(a.figs, exist_ok=True)
    survival_fig(a.runs, a.figs)
    natural_fig(a.runs, a.figs)
    if os.path.exists(a.region_csv):
        region_slide_fig(a.region_csv, a.figs)
    if os.path.exists("verify_out/surgery_E6.csv"):
        mech_fig(a.figs)


if __name__ == "__main__":
    main()
