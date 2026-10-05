"""Figures for the rule-identification (fit_rules) and mechanism (mech) analyses.

python -m recfreq.plot_fits --fit figs_fit --mech figs_mech --out figs_paper
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fit_rules import CELLS

FAM = [("bayes_true", "Bayes, true h, ε", "#6b5aa8"), ("bayes_free", "Bayes, fitted h′, ε′", "#1f4fb8"),
       ("counter_exp", "Prop. 1 counter (exp)", "#e45756"), ("counter_pow", "counter (power law)", "#f58518"),
       ("additive", "additive evidence", "#9aa5b1")]


def arch_of(d: pd.DataFrame) -> pd.Series:
    return ("L" + d.layers.astype(str) + np.where(d.mlp == 1, "+MLP", "") +
            np.where(d.pe != "alibi_learn", " " + d.pe.str.replace("learned_abs", "abs"), "") +
            np.where(d.conflict_frac > 0, " enr", ""))


def fig_examples(fit_dir: str, out: str):
    cur = np.load(os.path.join(fit_dir, "curves.npz"))
    runs = [("E1_h0.01_e0.1_L2_alibi_learn_attn_s0", "L2 · ALiBi · C"), ("E2a_h0.01_e0.1_L3_rope_attn_s0", "L3 · RoPE · C"),
            ("E1_h0.03_e0.2_L2_alibi_learn_attn_s0", "L2 · ALiBi · D")]
    cells = np.array(CELLS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    cols = {1: "#4c78a8", 2: "#f58518", 3: "#e45756"}
    for ax, (run, title) in zip(axes, runs):
        for nw in (1, 2, 3):
            m = (cells[:, 0] == 2) & (cells[:, 1] == nw)
            G = cells[m, 2]
            ax.plot(G, cur[f"{run}|y"][m], "o", color=cols[nw], ms=5, label=f"model, n_w={nw}")
            ax.plot(G, cur[f"{run}|bayes_free"][m], "-", color=cols[nw], lw=1.6)
            ax.plot(G, cur[f"{run}|counter_exp"][m], ":", color=cols[nw], lw=1.6)
            ax.plot(G, cur[f"{run}|bayes"][m], "--", color="k", lw=0.8, alpha=0.6)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log", base=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("gap G (pairs)")
    axes[0].set_ylabel("log p(c)/p(w), n_c = 2")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Dots: model. Solid: Bayes with fitted h′, ε′. Dotted: Prop. 1 counter fit. Dashed: true Bayes.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fit_examples.png"), dpi=130)
    plt.close(fig)


def fig_quality(fit_dir: str, out: str):
    d = pd.read_csv(os.path.join(fit_dir, "fits.csv"))
    d = d[~d.arch.str.contains("uniform")]
    order = ["L2", "L2+mlp", "L3", "L2 rope", "L3 rope", "L2 learned_abs", "L3 learned_abs", "L2 enriched0.2",
             "L3 enriched0.2"]
    fig, ax = plt.subplots(figsize=(13, 3.8))
    w = 0.16
    for i, (f, lab, col) in enumerate(FAM):
        q = d.groupby("arch")[f + ".cv_r2"].quantile([0.25, 0.5, 0.75]).unstack().reindex(order)
        x = np.arange(len(order)) + (i - 2) * w
        ax.bar(x, q[0.5].clip(lower=0), w, color=col, label=lab)
        ax.errorbar(x, q[0.5].clip(lower=0), yerr=[(q[0.5] - q[0.25]).clip(lower=0), (q[0.75] - q[0.5]).clip(lower=0)],
                    fmt="none", ecolor="k", lw=0.7, capsize=1.5)
    n = d.groupby("arch").size().reindex(order)
    ax.set_xticks(np.arange(len(order)), [f"{o}\n(n={k})" for o, k in zip(order, n)], fontsize=8)
    ax.set_ylabel("held-out R² (leave one gap out)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fit_quality.png"), dpi=130)
    plt.close(fig)


def fig_effective(fit_dir: str, out: str):
    d = pd.read_csv(os.path.join(fit_dir, "fits.csv"))
    d = d[(d.h > 0) & (d.eps > 0)]
    g = d.groupby(["h", "eps", "arch"])[["bfree.h", "bfree.eps"]].median().reset_index()
    L2 = g[g.arch == "L2"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epscol = {0.05: "#4c78a8", 0.1: "#f58518", 0.2: "#e45756", 0.3: "#54a24b"}
    ax = axes[0]
    for e, c in epscol.items():
        s = L2[L2.eps == e]
        ax.plot(s.h, s["bfree.h"], "o-", color=c, label=f"ε = {e}")
    xs = np.array([0.002, 0.05])
    ax.plot(xs, xs, "k--", lw=0.8, label="calibrated")
    b = np.polyfit(np.log(L2.h), np.log(L2["bfree.h"]), 1)
    ax.plot(xs, np.exp(b[1]) * xs ** b[0], "k-", lw=0.8, alpha=0.5, label=f"fit: h′ = {np.exp(b[1]):.2f} h^{b[0]:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true switch rate h"); ax.set_ylabel("effective h′ (fitted)")
    ax.legend(fontsize=7)
    ax = axes[1]
    hcol = {0.003: "#4c78a8", 0.01: "#f58518", 0.03: "#e45756"}
    for h, c in hcol.items():
        s = L2[L2.h == h]
        ax.plot(s.eps, s["bfree.eps"], "o-", color=c, label=f"h = {h}")
    xs = np.array([0.04, 0.32])
    ax.plot(xs, xs, "k--", lw=0.8, label="calibrated")
    ax.set_xlabel("true typo rate ε"); ax.set_ylabel("effective ε′ (fitted)")
    ax.legend(fontsize=7)
    fig.suptitle("L2 · ALiBi, median of 3 seeds per setting: models act as Bayes with h′ > h and ε′ pulled toward ~0.25",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "effective_params.png"), dpi=130)
    plt.close(fig)


def fig_mech(mech_dir: str, out: str):
    H = pd.read_csv(os.path.join(mech_dir, "mech_heads.csv"))
    m = pd.read_csv(os.path.join(mech_dir, "mech.csv"))
    C = H[(H.share >= 0.1) & (H.pe == "alibi_learn")].dropna(subset=["alibi_slope_per_pair"])
    run = "E1_h0.01_e0.1_L2_alibi_learn_attn_s0"
    cells = pd.read_parquet(os.path.join(mech_dir, f"{run}.cells.parquet"))
    hs = H[(H.run == run) & (H.share >= 0.1)]["head"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    ax.scatter(C.alibi_slope_per_pair, C.kern_exp_coef, s=10, c="#1f4fb8", alpha=0.6)
    lim = [0, max(C.alibi_slope_per_pair.max(), C.kern_exp_coef.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    r = np.corrcoef(C.alibi_slope_per_pair, C.kern_exp_coef)[0, 1]
    ax.set_title(f"attention decay = learned ALiBi slope\n{len(C)} copy heads, r = {r:.3f}", fontsize=9)
    ax.set_xlabel("learned ALiBi slope (per pair)"); ax.set_ylabel("fitted decay of attention vs distance")
    ax = axes[1]
    for t, col in zip(hs, ["#e45756", "#4c78a8", "#54a24b"]):
        x = cells[f"{t}.A_c"] - cells[f"{t}.A_w"]
        ax.scatter(x, cells[f"dla.{t}"], s=10, color=col, label=t)
    ax.set_xlabel("head attention: old copies − recent copies")
    ax.set_ylabel("head's direct effect on logit(c) − logit(w)")
    ax.set_title("each copy head is linear in its attention (Prop. 1)\nL2 · ALiBi · C, seed 0", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[2]
    X = np.stack([cells[f"{t}.A_c"] - cells[f"{t}.A_w"] for t in hs] + [np.ones(len(cells))], 1)
    coef, *_ = np.linalg.lstsq(X, cells.L, rcond=None)
    ax.scatter(X @ coef, cells.L, s=10, c="#1f4fb8")
    lim = [cells.L.min() - 0.3, cells.L.max() + 0.3]
    ax.plot(lim, lim, "k--", lw=0.8)
    r2 = 1 - ((X @ coef - cells.L) ** 2).mean() / cells.L.var()
    ax.set_xlabel("prediction from copy heads' attention")
    ax.set_ylabel("model logit(c) − logit(w)")
    ax.set_title(f"model decision from measured attention, R² = {r2:.3f}\n(median over 84 models: 0.90–0.99)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mechanism.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="figs_fit")
    ap.add_argument("--mech", default="figs_mech")
    ap.add_argument("--out", default="figs_paper")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_examples(a.fit, a.out)
    fig_quality(a.fit, a.out)
    fig_effective(a.fit, a.out)
    fig_mech(a.mech, a.out)
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
