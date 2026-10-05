"""Figures for plan §18 (law scope) from verify_out/law_scope_all.csv.

python -m recfreq.verify.plot_scope --out figs_scope
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

from ..layouts import CELLS, Lay
from .law_scope import A_MED, B_MED, law

INTERIOR = {(0.005, 0.07), (0.02, 0.15), (0.007, 0.25)}


def logit(x):
    return np.log(x / (1 - x))


def group(r):
    if r.exp in ("E1", "E2b"):
        return "fit set (12 settings)"
    if r.exp == "E5a":
        return "new, interpolation" if (r.h, r.eps) in INTERIOR else "new, extrapolation"
    if r.exp == "E5n":
        return f"context N={r.N}"
    if r.exp == "E5d":
        return "task variant"
    return "RoPE"


COL = {"fit set (12 settings)": "#9aa3b2", "new, interpolation": "#2B59C3", "new, extrapolation": "#E07A2F",
       "context N=192": "#7a4fbf", "context N=512": "#2f8f5b", "task variant": "#c8473a"}


def fig_pred_vs_fit(F, out):
    F = F[F.pe == "alibi_learn"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for g, d in F.groupby("grp"):
        if g not in COL:
            continue
        m = d.groupby(["h", "eps", "N", "V", "K", "key_dist", "typo"])[["h_law", "h_fit", "eps_law", "eps_fit"]].median()
        ax[0].scatter(m.h_law, m.h_fit.clip(1e-4), s=36, c=COL[g], label=g, alpha=0.9, edgecolor="white", lw=0.5)
        ax[1].scatter(m.eps_law, m.eps_fit, s=36, c=COL[g], label=g, alpha=0.9, edgecolor="white", lw=0.5)
    for a_, lo, hi in ((ax[0], 3e-3, 0.4), (ax[1], 0.05, 0.45)):
        a_.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax[0].plot([3e-3, 0.4], [6e-3, 0.8], ":", c="0.5", lw=0.8)
    ax[0].plot([3e-3, 0.4], [1.5e-3, 0.2], ":", c="0.5", lw=0.8)
    ax[0].set(xscale="log", yscale="log", xlabel="law prediction h′ = 1.45 h^0.74", ylabel="fitted h′ (median of seeds)",
              title="believed switch rate (dotted: ×2)")
    ax[1].set(xlabel="law prediction ε′", ylabel="fitted ε′ (median of seeds)", title="believed typo rate")
    ax[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Law 1 predictions for settings it was not fitted on (2-layer ALiBi)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scope_pred_vs_fit.png"), dpi=160)
    plt.close(fig)


def fig_r2(F, out):
    rows = []
    for g, d in F.groupby("grp"):
        rows.append((g, d.r2_law.median(), d.r2_law.quantile(0.25), d.r2_law.quantile(0.75),
                     d["bfree.r2"].median(), len(d)))
    t = pd.DataFrame(rows, columns=["grp", "med", "q1", "q3", "own", "n"])
    order = ["fit set (12 settings)", "new, interpolation", "new, extrapolation", "context N=192", "context N=512",
             "task variant", "RoPE"]
    t = t.set_index("grp").reindex([o for o in order if o in t.grp.values]).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4))
    y = np.arange(len(t))[::-1]
    ax.errorbar(t.med, y, xerr=[t.med - t.q1, t.q3 - t.med], fmt="o", c="#2B59C3", label="frozen law (no refit)")
    ax.scatter(t.own, y, marker="|", s=200, c="k", label="own Bayes(h′, ε′) fit (ceiling for the form)")
    ax.axvline(0.75, c="#E07A2F", ls="--", lw=1, label="pre-registered threshold")
    ax.set_yticks(y, [f"{g}  (n={n})" for g, n in zip(t.grp, t.n)])
    ax.set(xlabel="R² of the predicted probe surface (median, IQR over models)", xlim=(-0.2, 1.0))
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scope_r2.png"), dpi=160)
    plt.close(fig)


def fig_context(F, out):
    d = F[(F.pe == "alibi_learn") & (F.V == 64) & (F.K == 16) & (F.key_dist == "zipf") & (F.typo == "structured")
          & F.exp.isin(["E1", "E2b", "E5n"])]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    cols = {192: "#7a4fbf", 256: "#16213A", 512: "#2f8f5b"}
    for N, g in d.groupby("N"):
        hs = g[g.eps == 0.1].groupby("h").h_fit.median()
        es = g[g.h == 0.01].groupby("eps").eps_fit.median()
        if len(hs) >= 2:
            b = np.polyfit(np.log(hs.index), np.log(hs.values), 1)[0]
            ax[0].plot(hs.index, hs.values, "o-", c=cols.get(N, "0.5"), label=f"N = {N} (slope {b:.2f})")
        if len(es) >= 2:
            ax[1].plot(es.index, es.values, "o-", c=cols.get(N, "0.5"), label=f"N = {N}")
    x = np.array([0.003, 0.03])
    ax[0].plot(x, x, "k--", lw=0.8, label="calibrated")
    ax[0].set(xscale="log", yscale="log", xlabel="true switch rate h (ε = 0.1)", ylabel="fitted h′", title="h′ vs h")
    ax[1].plot([0.05, 0.3], [0.05, 0.3], "k--", lw=0.8)
    ax[1].set(xlabel="true typo rate ε (h = 0.01)", ylabel="fitted ε′", title="ε′ vs ε")
    for a_ in ax:
        a_.legend(fontsize=8)
    fig.suptitle("Context length (pairs per sequence)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scope_context.png"), dpi=160)
    plt.close(fig)


def fig_rope(F, out):
    d = F[(F.V == 64) & (F.K == 16) & (F.N == 256) & (F.key_dist == "zipf") & (F.typo == "structured")
          & F.exp.isin(["E1", "E2b", "E2a", "E5r"])]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for pe, mk in (("alibi_learn", "o"), ("rope", "s")):
        g = d[d.pe == pe].groupby(["h", "eps"])[["h_fit", "eps_fit"]].median().reset_index()
        for e, c in zip((0.05, 0.1, 0.2, 0.3), ("#2B59C3", "#E07A2F", "#c8473a", "#2f8f5b")):
            s = g[g.eps == e]
            ax[0].plot(s.h, s.h_fit.clip(1e-4), mk + ("-" if pe == "rope" else ":"), c=c,
                       label=f"{'RoPE' if pe == 'rope' else 'ALiBi'} ε={e}")
        for h, c in zip((0.003, 0.01, 0.03), ("#2B59C3", "#E07A2F", "#c8473a")):
            s = g[g.h == h]
            ax[1].plot(s.eps, s.eps_fit, mk + ("-" if pe == "rope" else ":"), c=c,
                       label=f"{'RoPE' if pe == 'rope' else 'ALiBi'} h={h}")
    ax[0].plot([0.003, 0.03], [0.003, 0.03], "k--", lw=0.8)
    ax[0].set(xscale="log", yscale="log", xlabel="true h", ylabel="fitted h′", title="h′ (solid RoPE, dotted ALiBi)")
    ax[1].plot([0.05, 0.3], [0.05, 0.3], "k--", lw=0.8)
    ax[1].set(xlabel="true ε", ylabel="fitted ε′", title="ε′")
    ax[0].legend(fontsize=6.5, ncol=2)
    ax[1].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scope_rope.png"), dpi=160)
    plt.close(fig)


def fig_surfaces(F, runs_dir, out):
    d = F[F.exp == "E5a"]
    settings = sorted({(h, e) for h, e in zip(d.h, d.eps)})
    if not settings:
        return
    fig, axs = plt.subplots(2, math.ceil(len(settings) / 2), figsize=(3.3 * math.ceil(len(settings) / 2), 6.2),
                            squeeze=False)
    lay = Lay(4, False, 64)
    idx = [i for i, c in enumerate(CELLS) if c[0] == 2]
    for ax, (h, e) in zip(axs.ravel(), settings):
        ys = []
        for r in d[(d.h == h) & (d.eps == e)].run:
            p = pd.read_parquet(os.path.join(runs_dir, r, "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS]
            ys.append(p.lo_model.to_numpy())
        y = np.mean(ys, 0)
        hp, ep = law(h, e)
        pred = A_MED + B_MED * lay.bayes([hp], [ep])[0]
        tb = lay.bayes([h], [e])[0]
        for nw, c in zip((1, 2, 3, 4), ("#9aa3b2", "#2B59C3", "#E07A2F", "#c8473a")):
            ii = [i for i in idx if CELLS[i][1] == nw]
            G = [CELLS[i][2] for i in ii]
            ax.plot(G, y[ii], "o-", c=c, ms=3, lw=1.4, label=f"model n_w={nw}")
            ax.plot(G, pred[ii], "--", c=c, lw=1)
            ax.plot(G, tb[ii], ":", c=c, lw=0.8)
        ax.axhline(0, c="k", lw=0.5)
        ax.set_xscale("log", base=2)
        kind = "interp." if (h, e) in INTERIOR else "extrap."
        ax.set_title(f"h={h}, ε={e} ({kind})", fontsize=9)
    for ax in axs.ravel()[len(settings):]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=6.5)
    fig.suptitle("New settings, n_c = 2: model (solid, seed mean) vs frozen law (dashed) vs true Bayes (dotted)", fontsize=10)
    fig.supxlabel("gap G (pairs)", fontsize=9)
    fig.supylabel("log p(c)/p(w)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scope_surfaces.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="verify_out/law_scope_all.csv")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="figs_scope")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    F = pd.read_csv(a.csv)
    F["grp"] = F.apply(group, axis=1)
    fig_pred_vs_fit(F, a.out)
    fig_r2(F, a.out)
    fig_context(F, a.out)
    fig_rope(F, a.out)
    fig_surfaces(F, a.runs, a.out)
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
