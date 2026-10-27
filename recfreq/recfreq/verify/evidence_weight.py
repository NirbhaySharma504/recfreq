"""Plan §19.6: per-model evidence weight in the probes' own coordinates.

For each 2-layer model at the default task, find the switch gap G*(n_c = 2, n_w) where its log-odds cross 0, using
the standard probes (G <= 128, n_w 1-4) and, where present, the wide-window probes (G 192/224, n_w 5/6). Then
  spacing D = -slope of log G* against n_w (least squares over every n_w that crosses; at least 2 needed)
  evidence weight w = D_model / D_bayes, D_bayes from the exact Bayes G* at the same (h, eps) and n_w values.
w = 1: each extra recent copy moves the switch point as much as for Bayes; w < 1: recent copies count for less.
Also the time scale: T = G*_model / G*_bayes at the smallest n_w that crosses.

python -m recfreq.verify.evidence_weight --out verify_out --figs figs_region
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .switch_points import crossing
from .theory import exact_gstar

DEFAULT = dict(V=64, K=16, n_pairs=256, key_dist="zipf", d_model=128, steps=20000, lr=1e-3, batch=128)


def curves(run_dir):
    """{n_w: (G, lo_model)} at n_c = 2, standard and wide probes merged."""
    P = pd.read_parquet(os.path.join(run_dir, "probes.parquet"))
    w = os.path.join(run_dir, "probes_wide.parquet")
    if os.path.exists(w):
        P = pd.concat([P, pd.read_parquet(w)])
    P = P[P.n_c == 2].groupby(["n_w", "G"]).lo_model.mean().reset_index()
    return {nw: (d.G.values.astype(float), d.lo_model.values) for nw, d in P.groupby("n_w")}


def model_table(runs, prefixes, allow=("K", "n_pairs")):
    rows = []
    for f in sorted(glob.glob(os.path.join(runs, "*/done.json"))):
        name = f.split("/")[-2]
        if not name.startswith(prefixes):
            continue
        a = json.load(open(f))["args"]
        if a["layers"] != 2 or a["mlp"] or a.get("typo", "structured") != "structured" or a["h"] <= 0 or a["eps"] <= 0:
            continue
        if any(a.get(k, v) != v for k, v in DEFAULT.items() if k not in allow):
            continue
        cv = curves(os.path.dirname(f))
        g = {nw: crossing(G, y) for nw, (G, y) in cv.items()}
        nws = [nw for nw in sorted(g) if np.isfinite(g[nw])]
        r = {"run": name, "exp": name.split("_")[0], "h": a["h"], "eps": a["eps"], "pe": a["pe"],
             "cf": a.get("conflict_frac", 0.0), "K": a.get("K", 16), "N": a.get("n_pairs", 256), "seed": a["seed"],
             "n_cross": len(nws), "nw_min": nws[0] if nws else np.nan}
        for nw in range(1, 7):
            r[f"G{nw}"] = g.get(nw, np.nan)
        b = {nw: exact_gstar(a["h"], a["eps"], nw, 2, gmax=1e7) for nw in nws}
        nws = [nw for nw in nws if 0 < b[nw] < np.inf]
        if nws:
            r["T"] = g[nws[0]] / b[nws[0]]
        if len(nws) >= 2:
            # clock units x = h G* (expected switches in the gap): log x_model = alpha + w log x_bayes
            xb = np.log([a["h"] * b[n] for n in nws])
            xm = np.log([a["h"] * g[n] for n in nws])
            r["w"], r["alpha"] = np.polyfit(xb, xm, 1)
            r["x_ref"] = float(np.exp(r["alpha"]))  # model's switch point where Bayes's is x = 1
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--figs", default="")
    a = ap.parse_args()
    M = model_table(a.runs, ("E1_", "E2b_", "E5a_", "E5r_", "E2a_", "E6_", "E7_", "E8_", "E9_"))
    M.to_csv(os.path.join(a.out, "evidence_weight.csv"), index=False)
    pd.set_option("display.width", 220)
    base = M[(M.cf == 0) & (M.K == 16) & (M.N == 256)]
    print("evidence weight w by (pe, h), median over eps and seeds (n models):")
    print(base.groupby(["pe", "h"]).agg(w=("w", "median"), n=("w", "count"), x_ref=("x_ref", "median"), T=("T", "median")).round(3).to_string())
    from scipy.stats import spearmanr
    for pe, d in base.dropna(subset=["w"]).groupby("pe"):
        if len(d) >= 5:
            r = spearmanr(np.log(d.h), d.w)
            print(f"{pe}: Spearman(log h, w) = {r.statistic:.2f} (n = {len(d)})")
    if a.figs:
        plot(M, a.figs)


def plot(M, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False})
    INK, BLUE, ORANGE = "#16213A", "#2B59C3", "#D9822B"
    d = M[(M.cf == 0) & (M.K == 16) & (M.N == 256)].dropna(subset=["w"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for pe, c, m, lab in (("alibi_learn", BLUE, "o", "ALiBi models"), ("rope", ORANGE, "s", "RoPE models")):
        x = d[d.pe == pe]
        ax.scatter(x.h, x.w, c=c, marker=m, s=34, alpha=0.75, label=lab, edgecolor="white", lw=0.5)
        g = x.groupby("h").w.agg(["median", "count"])
        g = g[g["count"] >= 2]["median"]
        if pe == "alibi_learn":
            ax.plot(g.index, g.values, "-", c=c, lw=2)
    ax.axhline(1, ls="--", c=INK, lw=1.5)
    ax.text(0.12, 1.03, "Bayes", color=INK, ha="right")
    ax.axvspan(0.003, 0.03, color="#E6E8EE", zorder=-1)
    ax.text(0.0034, 0.03, "law fitted here", color="#5A6377", fontsize=13)
    ax.set_xscale("log"); ax.set_xlabel("switch rate h"); ax.set_ylim(0, 1.6)
    ax.set_ylabel("evidence weight w (÷ Bayes)")
    ax.legend(frameon=False, loc="upper left", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(out, "evidence_weight.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
