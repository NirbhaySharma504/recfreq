"""Aggregate runs: in-distribution fit, induction-head slope, n_c-sensitivity, boundaries, plots.

python -m recfreq.analyze results/runs --exp E1 --out figs
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

from .oracle import critical_gap


def nc_slope(df: pd.DataFrame, col: str) -> float:
    """Mean over (n_w, G) cells of the least-squares slope of `col` on log2(n_c), n_c >= 2."""
    d = df[df.n_c >= 2]
    slopes = []
    for _, cell in d.groupby(["n_w", "G"]):
        x = np.log2(cell.n_c.to_numpy())
        slopes.append(np.polyfit(x, cell[col].to_numpy(), 1)[0])
    return float(np.mean(slopes))


def nc_step(df: pd.DataFrame, col: str, a: int, b: int) -> float:
    """Mean over (n_w, G) cells of col(n_c=b) - col(n_c=a)."""
    p = df.pivot_table(index=["n_w", "G"], columns="n_c", values=col)
    return float((p[b] - p[a]).mean())


def boundary(df: pd.DataFrame, col: str, n_c: int, n_w: int) -> float:
    """Gap G* where mean log-odds `col` first crosses 0, interpolated in log2 G.
    0 if already < 0 at the smallest gap, inf if never within the probed gaps."""
    d = df[(df.n_c == n_c) & (df.n_w == n_w)].sort_values("G")
    G, lo = np.log2(d.G.to_numpy(float)), d[col].to_numpy()
    if lo[0] < 0:
        return 0.0
    for i in range(1, len(lo)):
        if lo[i] < 0:
            x = G[i - 1] + (G[i] - G[i - 1]) * lo[i - 1] / (lo[i - 1] - lo[i])
            return float(2 ** x)
    return float("inf")


def max_drop(df: pd.DataFrame, col: str) -> float:
    """Cliff sharpness: largest fall in `col` between neighbouring gaps (x2 apart),
    averaged over the (n_c >= 2, n_w) curves."""
    drops = []
    for _, g in df[df.n_c >= 2].groupby(["n_c", "n_w"]):
        lo = g.sort_values("G")[col].to_numpy()
        drops.append((lo[:-1] - lo[1:]).max())
    return float(np.mean(drops))


def per_copy(gstars: list[float]) -> float:
    """Mean of ln G*(n_w) - ln G*(n_w + 1) over consecutive n_w with both boundaries inside the probe range."""
    v = [math.log(a) - math.log(b) for a, b in zip(gstars, gstars[1:]) if 0 < a < math.inf and 0 < b < math.inf]
    return float(np.mean(v)) if v else float("nan")


def arch_label(a: dict) -> str:
    s = f"L{a['layers']}{'+mlp' if a['mlp'] else ''}"
    if a.get("pe", "alibi_learn") != "alibi_learn":
        s += f" {a['pe']}"
    if a.get("typo", "structured") != "structured":
        s += f" {a['typo']}-typo"
    if a.get("conflict_frac", 0):
        s += f" enriched{a['conflict_frac']}"
    if a.get("d_model", 128) != 128:
        s += f" d{a['d_model']}"
    if a.get("steps", 20000) != 20000:
        s += f" {a['steps'] // 1000}k"
    if a.get("lr", 1e-3) != 1e-3:
        s += f" lr{a['lr']}"
    if a.get("batch", 128) != 128:
        s += f" b{a['batch']}"
    for k, d in (("V", 64), ("K", 16), ("n_pairs", 256)):
        if a.get(k, d) != d:
            s += f" {'N' if k == 'n_pairs' else k}{a[k]}"
    if a.get("key_dist", "zipf") != "zipf":
        s += f" {a['key_dist']}-keys"
    return s


def load(runs_dir: str, exp: str) -> list[dict]:
    runs = []
    paths = sorted(p for e in exp.split(",") for p in glob.glob(os.path.join(runs_dir, f"{e}_*", "done.json")))
    for done in paths:
        d = os.path.dirname(done)
        s = json.load(open(done))
        a = s["args"]
        heads = pd.DataFrame(s["heads"])
        late = heads[heads.layer >= 1]
        ind = late.loc[late.ind_score.idxmax()]
        r = {"run": os.path.basename(d), "exp": os.path.basename(d).split("_")[0], "h": a["h"], "eps": a["eps"],
             "arch": arch_label(a), "seed": a["seed"],
             **{k: s["final"][k] for k in ["excess", "acc_rep", "bayes_acc_rep", "ce", "bayes_ce"]},
             "ind_layer": int(ind["layer"]), "ind_head": int(ind["head"]), "ind_score": ind["ind_score"],
             "lambda_hat": float("nan") if ind["slope"] is None else ind["slope"]}
        if "gate_pass" in s:
            r["gate_pass"] = s["gate_pass"]
        pq = os.path.join(d, "probes.parquet")
        if os.path.exists(pq):
            pr = pd.read_parquet(pq)
            r["probes"] = pr
            r["nc_slope_model"] = nc_slope(pr, "lo_model")
            r["nc_slope_bayes"] = nc_slope(pr, "lo_bayes")
            r["agree"] = pr.agree.mean()
            for who in ("model", "bayes"):
                r[f"nc1to2_{who}"] = nc_step(pr, f"lo_{who}", 1, 2)
                r[f"nc2to4_{who}"] = nc_step(pr, f"lo_{who}", 2, 4)
                r[f"nc4to8_{who}"] = nc_step(pr, f"lo_{who}", 4, 8)
                r[f"maxdrop_{who}"] = max_drop(pr, f"lo_{who}")
                gs = [boundary(pr, f"lo_{who}", 2, nw) for nw in (1, 2, 3, 4)]
                for nw, gv in zip((1, 2, 3, 4), gs):
                    r[f"Gstar_{who}_nw{nw}"] = gv
                r[f"percopy_{who}"] = per_copy(gs)
        runs.append(r)
    return runs


def plot_curves(runs: list[dict], out: str, n_w: int = 2):
    pts = sorted({(r["h"], r["eps"]) for r in runs})
    archs = sorted({r["arch"] for r in runs})
    fig, axes = plt.subplots(len(pts), len(archs), figsize=(4 * len(archs), 3.2 * len(pts)), squeeze=False,
                             sharex=True)
    colors = {1: "#9aa5b1", 2: "#4c78a8", 4: "#f58518", 8: "#e45756"}
    for i, pt in enumerate(pts):
        for j, arch in enumerate(archs):
            ax = axes[i][j]
            rs = [r for r in runs if (r["h"], r["eps"]) == pt and r["arch"] == arch and "probes" in r]
            if not rs:
                ax.set_visible(False)
                continue
            allp = pd.concat([r["probes"] for r in rs])
            m = allp.groupby(["n_c", "n_w", "G"]).mean(numeric_only=True).reset_index()
            for nc in (1, 2, 4, 8):
                d = m[(m.n_c == nc) & (m.n_w == n_w)]
                ax.plot(d.G, d.lo_model, "-o", ms=3, color=colors[nc], label=f"model n_c={nc}")
                ax.plot(d.G, d.lo_bayes, "--", color=colors[nc], alpha=0.6)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xscale("log", base=2)
            ax.set_title(f"h={pt[0]}, eps={pt[1]} | {arch}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"log p(c)/p(w), n_w={n_w}")
            if i == len(pts) - 1:
                ax.set_xlabel("gap G (pairs)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Probe log-odds: model (solid) vs Bayes (dashed); mean over seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"probe_curves_nw{n_w}.png"), dpi=130)
    plt.close(fig)


def plot_nc_slope(summ: pd.DataFrame, out: str):
    g = summ.groupby(["h", "eps", "arch"]).agg(m=("nc_slope_model", "mean"), s=("nc_slope_model", "std"),
                                                 b=("nc_slope_bayes", "mean")).reset_index()
    pts = sorted({(r.h, r.eps) for r in g.itertuples()})
    archs = sorted(g.arch.unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(pts) * len(archs) / 2 + 2, 3.4))
    w = 0.8 / (len(archs) + 1)
    for k, arch in enumerate(archs):
        d = g[g.arch == arch].set_index(["h", "eps"]).reindex(pts)
        ax.bar(np.arange(len(pts)) + k * w, d.m, w, yerr=d.s, label=arch, capsize=2)
    d = g.groupby(["h", "eps"]).b.mean().reindex(pts)
    ax.bar(np.arange(len(pts)) + len(archs) * w, d, w, label="Bayes", color="k", alpha=0.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(pts)) + w * len(archs) / 2)
    ax.set_xticklabels([f"h={h}\neps={e}" for h, e in pts], fontsize=8)
    ax.set_ylabel("d log p(c)/p(w) / d log2 n_c")
    ax.set_title("n_c-sensitivity (counter > 0, Bayes ~ 0)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "nc_slope.png"), dpi=130)
    plt.close(fig)


def plot_grid(summ: pd.DataFrame, out: str, arch: str = "L2"):
    """(h, eps) heatmaps for one architecture: learned slope, per-copy evidence weight, cliff sharpness."""
    d = summ[summ.arch == arch].groupby(["h", "eps"]).mean(numeric_only=True).reset_index()
    d = d[(d.h > 0) | (d.eps > 0)]
    if d.h.nunique() < 2 or d.eps.nunique() < 2:
        return
    panels = [("lambda_hat", "learned slope λ̂ (per token)"), ("percopy_model", "per-copy weight, model (nats)"),
              ("percopy_bayes", "per-copy weight, Bayes (nats)"), ("maxdrop_model", "cliff: max drop / doubling, model"),
              ("maxdrop_bayes", "cliff: max drop / doubling, Bayes")]
    hs, es = sorted(d.h.unique()), sorted(d.eps.unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.2))
    for ax, (col, title) in zip(axes, panels):
        M = np.full((len(hs), len(es)), np.nan)
        for r in d.itertuples():
            M[hs.index(r.h), es.index(r.eps)] = getattr(r, col)
        ax.imshow(M, cmap="viridis", aspect="auto", origin="lower")
        for i in range(len(hs)):
            for j in range(len(es)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_xticks(range(len(es)), [str(e) for e in es], fontsize=7)
        ax.set_yticks(range(len(hs)), [str(h) for h in hs], fontsize=7)
        ax.set_xlabel("eps", fontsize=8)
        ax.set_ylabel("h", fontsize=8)
        ax.set_title(title, fontsize=8)
    fig.suptitle(f"{arch}: mean over seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"grid_{arch.replace(' ', '_')}.png"), dpi=130)
    plt.close(fig)


def fmt(agg: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in agg.columns.get_level_values(0)]
    mean, sd = agg.xs("mean", axis=1, level=1)[cols], agg.xs("std", axis=1, level=1)[cols]
    return mean.round(3).astype(str) + " ± " + sd.round(3).astype(str)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs_dir")
    p.add_argument("--exp", default="E1", help="comma-separated experiment prefixes")
    p.add_argument("--out", default="figs")
    p.add_argument("--curves", type=int, default=1)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load(a.runs_dir, a.exp)
    if not runs:
        raise SystemExit("no finished runs")
    tag = a.exp.replace(",", "+")
    summ = pd.DataFrame([{k: v for k, v in r.items() if k != "probes"} for r in runs])
    summ.to_csv(os.path.join(a.out, f"{tag}_runs.csv"), index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 50)
    print(f"{len(runs)} finished runs: {sorted(summ.exp.unique())}")
    if "nc1to2_model" in summ:
        num = [c for c in summ.columns if summ[c].dtype.kind in "fi" and c not in ("h", "eps", "seed")]
        agg = summ.groupby(["h", "eps", "arch"])[num].agg(["mean", "std"])
        agg[("n_seeds", "mean")] = summ.groupby(["h", "eps", "arch"]).size()
        print("\n== fit, slope, cliff, evidence weights (mean ± sd over seeds) ==")
        t = fmt(agg, ["excess", "lambda_hat", "agree", "maxdrop_model", "maxdrop_bayes", "percopy_model",
                      "percopy_bayes", "nc1to2_model", "nc1to2_bayes", "nc4to8_model", "nc4to8_bayes"])
        t.insert(0, "seeds", summ.groupby(["h", "eps", "arch"]).size())
        print(t.to_string())
        print("\n== switch-over gap G* at n_c = 2 (pairs; 0 = before G=2, inf = after G=128) ==")
        print(fmt(agg, [f"Gstar_{w}_nw{n}" for n in (1, 2, 3, 4) for w in ("model", "bayes")]).to_string())
        print("\ntheory G* for structured typos (n_w=1..4):")
        for h, e in sorted({(r["h"], r["eps"]) for r in runs}):
            print(f"  h={h} eps={e}:", [round(critical_gap(64, h, e, n), 1) for n in (1, 2, 3, 4)])
        if a.curves:
            for nw in (1, 2, 3):
                plot_curves(runs, a.out, n_w=nw)
        for arch in sorted(summ.arch.unique()):
            plot_grid(summ, a.out, arch)
        print("figures ->", a.out)


if __name__ == "__main__":
    main()
