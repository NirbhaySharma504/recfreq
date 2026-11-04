"""Plan §20.2: figure for the validation sweep, and the evidence weight w on the new models (exploratory).

python -m recfreq.verify.plot_validity --out verify_out --figs figs_validity
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evidence_weight import model_table
from .validity import BAND

INK, BLUE, ORANGE, GREY = "#16213A", "#2B59C3", "#D9822B", "#8A93A6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--figs", default="figs_validity")
    ap.add_argument("--runs", default="results/runs")
    a = ap.parse_args()
    os.makedirs(a.figs, exist_ok=True)
    W = model_table(a.runs, ("E13_", "E14_"))
    W.to_csv(os.path.join(a.out, "evidence_weight_E13_E14.csv"), index=False)
    old = pd.read_csv(os.path.join(a.out, "evidence_weight.csv"))
    old = old[(old.pe == "alibi_learn") & (old.cf == 0) & (old.K == 16) & (old.N == 256)]
    w13 = W[W.exp == "E13"]
    print("E13 evidence weight w by h (median over eps and seeds):")
    print(w13.groupby("h").w.agg(["median", "count"]).round(3).to_string())
    from scipy.stats import spearmanr
    d = w13.dropna(subset=["w"])
    print(f"E13: Spearman(log h, w) = {spearmanr(np.log(d.h), d.w).statistic:.2f} (n = {len(d)}); "
          f"median w {d.w.median():.3f}, IQR {d.w.quantile(.25):.3f}-{d.w.quantile(.75):.3f}")

    S = pd.read_csv(os.path.join(a.out, "validity_settings.csv"))
    B = pd.read_csv(os.path.join(a.out, "validity_baseline_settings.csv"))
    s13 = S[S.exp == "E13"]
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    lg = lambda e: np.log(e / (1 - e))
    x0, x1 = math.log10(BAND["h"][0]), math.log10(BAND["h"][1])
    y0, y1 = lg(BAND["eps"][0]), lg(BAND["eps"][1])
    ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", ec=GREY, lw=1.5))
    ok = s13.ratio_own <= 1.51
    ax[0].scatter(np.log10(B.h), lg(B.eps), s=40, c=np.where(B.ratio_own <= 1.51, "#BFD0F0", "#F2C9A0"), marker="s",
                  label="earlier settings")
    ax[0].scatter(np.log10(s13.h[ok]), lg(s13.eps[ok]), s=110, c=BLUE, edgecolor="white", label="new: passes (fit ratio ≤ 1.51)")
    ax[0].scatter(np.log10(s13.h[~ok]), lg(s13.eps[~ok]), s=110, c=ORANGE, edgecolor="white", label="new: fails")
    ticks = [0.01, 0.05, 0.1, 0.2, 0.3, 0.45]
    ax[0].set_yticks([lg(t) for t in ticks]); ax[0].set_yticklabels([str(t) for t in ticks])
    ax[0].set_xticks([-3.5, -3, -2.5, -2, -1.5, -1]); ax[0].set_xticklabels(["3e-4", "1e-3", "3e-3", "0.01", "0.03", "0.1"])
    ax[0].set_xlabel("switch rate h"); ax[0].set_ylabel("typo rate ε")
    ax[0].set_title("24 new settings, 3 seeds each (dashed: band)")
    ax[0].legend(frameon=False, fontsize=10, loc="lower left")
    M = pd.read_csv(os.path.join(a.out, "validity_models.csv"))
    m13 = M[(M.exp == "E13") & (M.var_y >= 0.05)]
    ax[1].hist(m13.r2_law, bins=np.linspace(0.2, 1, 33), color=BLUE)
    ax[1].axvline(m13.r2_law.median(), c=INK, ls="--")
    ax[1].text(m13.r2_law.median() - 0.01, ax[1].get_ylim()[1] * 0.9, f"median {m13.r2_law.median():.2f}", ha="right")
    ax[1].set_xlabel("frozen-law R² per new model"); ax[1].set_ylabel("models")
    ax[1].set_title("72 new models, law never refitted")
    order = np.argsort(s13.h.values)
    xs = np.arange(len(s13))
    ax[2].bar(xs - 0.2, s13.noise.values[order], 0.4, color=GREY, label="seed noise (sd per cell)")
    ax[2].bar(xs + 0.2, s13.sys.values[order], 0.4, color=ORANGE, label="law bias (seed noise removed)")
    ax[2].set_xticks(xs[::3]); ax[2].set_xticklabels([f"{h:g}" for h in s13.h.values[order][::3]], rotation=45)
    ax[2].set_xlabel("h of the setting"); ax[2].set_ylabel("nats (RMS over 112 probe cells)")
    ax[2].set_title("The law's error is a reproducible bias")
    ax[2].legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.figs, "validity.png"), dpi=140)

    fig, axw = plt.subplots(figsize=(7.5, 5))
    for d_, c, lab in ((old, GREY, "earlier ALiBi models"), (w13, BLUE, "E13 (new)")):
        d_ = d_.dropna(subset=["w"])
        axw.scatter(d_.h, d_.w, s=26, c=c, alpha=0.7, label=lab)
    axw.axhline(1, ls="--", c=INK); axw.set_xscale("log"); axw.set_xlabel("switch rate h"); axw.set_ylabel("evidence weight w")
    axw.legend(frameon=False); fig.tight_layout(); fig.savefig(os.path.join(a.figs, "evidence_weight_E13.png"), dpi=140)
    print("figures written to", a.figs)


if __name__ == "__main__":
    main()
