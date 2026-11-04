"""Slide figures for deck 3 (plan §20): one panel per file, in the deck's colours.

python -m recfreq.verify.plot_slides3 --out verify_out --figs figs_validity/slides
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

from .validity import BAND

INK, BLUE, ORANGE, GREEN, GREY, PALE = "#16213A", "#2B59C3", "#D9822B", "#2E8B57", "#8A93A6", "#C9D2E3"
plt.rcParams.update({"font.size": 17, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "font.family": "DejaVu Sans"})


def lg(e):
    return np.log(np.asarray(e) / (1 - np.asarray(e)))


def band_map(S, B, path):
    s = S[S.exp == "E13"]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    x0, x1 = math.log10(BAND["h"][0]), math.log10(BAND["h"][1])
    y0, y1 = lg(BAND["eps"][0]), lg(BAND["eps"][1])
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, fc="#EEF2FA", ls="--", ec=GREY, lw=1.8, zorder=0))
    ax.text(x0 + 0.03, y1 + 0.08, "band under test", color=GREY, fontsize=15)
    ok_b = B.ratio_own <= 1.51
    ax.scatter(np.log10(B.h[ok_b]), lg(B.eps[ok_b]), s=46, marker="s", c=PALE, label="earlier settings: within 1.51×")
    ax.scatter(np.log10(B.h[~ok_b]), lg(B.eps[~ok_b]), s=46, marker="s", c="#F2D3B3", label="earlier settings: worse")
    ok = s.ratio_own <= 1.51
    ax.scatter(np.log10(s.h[ok]), lg(s.eps[ok]), s=150, c=BLUE, edgecolor="white", lw=1.2, label=f"new, passes ({ok.sum()})")
    ax.scatter(np.log10(s.h[~ok]), lg(s.eps[~ok]), s=150, c=ORANGE, edgecolor="white", lw=1.2, label=f"new, fails ({(~ok).sum()})")
    ticks = [0.01, 0.05, 0.1, 0.2, 0.3, 0.45]
    ax.set_yticks(lg(ticks)); ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xticks([-3.5, -3, -2.5, -2, -1.5, -1]); ax.set_xticklabels(["0.0003", "0.001", "0.003", "0.01", "0.03", "0.1"])
    ax.set_xlabel("switch rate h"); ax.set_ylabel("typo rate ε")
    ax.legend(frameon=False, fontsize=13, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def r2_hist(M, path):
    m = M[(M.exp == "E13") & (M.var_y >= 0.05)]
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    bins = np.linspace(0.2, 1, 33)
    flat = (m.h == 0.001) & (m.eps == 0.25)
    ax.hist([m.r2_law[~flat], m.r2_law[flat]], bins=bins, stacked=True, color=[BLUE, PALE],
            label=["72 new models", "(0.001, 0.25): nearly flat surfaces"])
    med = m.r2_law.median()
    ax.axvline(med, c=INK, ls="--", lw=1.5)
    ax.text(med - 0.015, ax.get_ylim()[1] * 0.93, f"median {med:.2f}", ha="right", color=INK)
    ax.axvline(0.7, c=GREY, ls=":", lw=1.5)
    ax.text(0.69, ax.get_ylim()[1] * 0.6, f"{(m.r2_law >= 0.7).mean():.0%} ≥ 0.7", ha="right", color=GREY, fontsize=15)
    ax.set_xlabel("R² of the frozen law, per new model"); ax.set_ylabel("models")
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def bias_noise(S, path):
    s = S[S.exp == "E13"].sort_values("h")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = np.arange(len(s))
    ax.bar(xs - 0.2, s.noise, 0.4, color=GREY, label="seed noise: how much two seeds differ (per-cell sd)")
    ax.bar(xs + 0.2, s.sys, 0.4, color=ORANGE, label="law bias: law vs. seed mean, noise removed")
    ax.set_xticks(xs[::2]); ax.set_xticklabels([f"{h:g}" for h in s.h.values[::2]], rotation=40)
    ax.set_xlabel("switch rate h of each of the 24 new settings"); ax.set_ylabel("nats (RMS over 112 probe cells)")
    ax.axhline(s.sys.median(), c=ORANGE, ls="--", lw=1.2); ax.axhline(s.noise.median(), c=GREY, ls="--", lw=1.2)
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    ax.set_ylim(0, 1.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def robustness(M, path):
    m = M[(M.exp == "E14") & (M.var_y >= 0.05)].copy()
    m["variant"] = m.variant.fillna("")
    be = m[m.variant == ""].err.median()
    order = [("", "default (3 seeds)"), ("lr=0.0005", "lr ÷2"), ("lr=0.002", "lr ×2"), ("batch=256", "batch ×2"),
             ("steps=40000", "steps ×2"), ("d_model=64", "width 64"), ("d_model=256", "width 256"),
             ("mlp=1", "MLPs on*"), ("layers=3", "3 layers*")]
    vals = [m[m.variant == v].err.median() / be for v, _ in order]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    cols = [INK] + [GREEN if v <= 1.25 else ORANGE for v in vals[1:]]
    ax.barh(range(len(order))[::-1], vals, color=cols, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, len(order) - 1 - i, f"×{v:.2f}", va="center", fontsize=15, color=INK)
    ax.axvline(1.25, c=GREY, ls="--", lw=1.5)
    ax.text(1.26, len(order) - 0.6, "registered limit ×1.25", color=GREY, fontsize=14)
    ax.set_yticks(range(len(order))[::-1]); ax.set_yticklabels([lab for _, lab in order])
    ax.set_xlim(0, 1.85); ax.set_xlabel("law error ÷ default recipe's (median over 4 points × 2 seeds)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evidence(out, path):
    W = pd.read_csv(os.path.join(out, "evidence_weight_E13_E14.csv"))
    old = pd.read_csv(os.path.join(out, "evidence_weight.csv"))
    old = old[(old.pe == "alibi_learn") & (old.cf == 0) & (old.K == 16) & (old.N == 256)].dropna(subset=["w"])
    w = W[W.exp == "E13"].dropna(subset=["w"])
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.axvspan(BAND["h"][0], BAND["h"][1], color="#EEF2FA", zorder=0)
    ax.scatter(old.h, old.w, s=30, c=PALE, label="earlier ALiBi models", zorder=2)
    ax.scatter(w.h, w.w, s=46, c=BLUE, edgecolor="white", lw=0.6, label="72 new models (E13)", zorder=3)
    g = w.groupby("h").w.median()
    ax.plot(g.index, g.values, c=BLUE, lw=2, zorder=3)
    ax.axhline(1, ls="--", c=INK, lw=1.4); ax.text(0.13, 1.03, "Bayes", ha="right", color=INK)
    ax.set_xscale("log"); ax.set_ylim(0, 1.6)
    ax.set_xlabel("switch rate h"); ax.set_ylabel("evidence weight w (÷ Bayes)")
    ax.legend(frameon=False, fontsize=14, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--figs", default="figs_validity/slides")
    a = ap.parse_args()
    os.makedirs(a.figs, exist_ok=True)
    S = pd.read_csv(os.path.join(a.out, "validity_settings.csv"))
    B = pd.read_csv(os.path.join(a.out, "validity_baseline_settings.csv"))
    M = pd.read_csv(os.path.join(a.out, "validity_models.csv"))
    band_map(S, B, os.path.join(a.figs, "band_map.png"))
    r2_hist(M, os.path.join(a.figs, "r2_hist.png"))
    bias_noise(S, os.path.join(a.figs, "bias_noise.png"))
    robustness(M, os.path.join(a.figs, "robustness.png"))
    evidence(a.out, os.path.join(a.figs, "evidence_e13.png"))
    print("written:", sorted(os.listdir(a.figs)))


if __name__ == "__main__":
    main()
