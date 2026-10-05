"""Analysis of run-3 outputs: causal ablations, slope surgery, layout generalization, and what the
copy heads' attention depends on besides distance.

python -m recfreq.analyze3 --run3 figs_run3 --runs results/runs --mech figs_mech --out figs_run3
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from .analyze import boundary
from .layouts import CELLS, LAY4, BayesFree, Lay, fit_additive, fit_affine, fit_counter, fit_counter2, r2

LAYS = {"sp4": LAY4, "sp2": Lay(2), "sp8": Lay(8), "swap4": Lay(4, True)}
_BF: dict = {}


def cells_vec(df: pd.DataFrame, col: str = "lo") -> np.ndarray:
    return df.set_index(["n_c", "n_w", "G"]).loc[CELLS][col].to_numpy()


def bf_fit(lay: str, y: np.ndarray) -> dict:
    p = _BF[lay].fit(y, np.ones(len(y), bool))
    p["r2"] = r2(BayesFree.predict(LAYS[lay], p), y)
    return p


# ---------------------------------------------------------------------------------------------------------------
def causal_one(name: str) -> list[dict]:
    meta = json.load(open(os.path.join(A.run3, f"{name}.json")))
    s = pd.read_parquet(os.path.join(A.run3, f"{name}.surf.parquet"))
    s4 = s[s.layout == "sp4"]
    base = cells_vec(s4[s4.cond == "base"])
    rows = []
    for cond, g in s4.groupby("cond"):
        y = cells_vec(g)
        p = bf_fit("sp4", y) if y.var() > 0.02 else {"h": np.nan, "eps": np.nan, "r2": np.nan}
        gst = {f"Gstar_nw{nw}": boundary(g, "lo", 2, nw) for nw in (1, 2, 3)}
        rows.append({"run": name, "cond": cond, **gst, "mean_abs_lo": float(np.abs(y).mean()), "mean_lo": float(y.mean()),
                     "corr_base": float(np.corrcoef(y, base)[0, 1]) if y.std() > 0 else np.nan,
                     "h_eff": p["h"], "eps_eff": p["eps"], "bf_r2": p["r2"],
                     "ce_rep": meta["losses"][cond]["ce_rep"], "ce": meta["losses"][cond]["ce"],
                     **{k: meta[k] for k in ("h", "eps", "layers", "mlp", "pe", "seed", "conflict_frac")},
                     "copy_heads": str(meta["copy_heads"]), "control_heads": str(meta["control_heads"])})
    return rows


def general_one(name: str) -> list[dict]:
    """Fit every family on the original layout (stored 256-fill probes), predict the new layouts unchanged."""
    meta = json.load(open(os.path.join(A.run3, f"{name}.json")))
    pr = pd.read_parquet(os.path.join(A.runs, name, "probes.parquet")).rename(columns={"lo_model": "lo"})
    y4 = cells_vec(pr)
    m = np.ones(len(y4), bool)
    h, e = meta["h"], meta["eps"]
    fits = {"bayes_true": fit_affine(LAY4.bayes([h], [e])[0], y4, m), "bayes_free": _BF["sp4"].fit(y4, m),
            "counter": fit_counter(LAY4, y4, m), "counter2": fit_counter2(LAY4, y4, m),
            "additive": fit_additive(LAY4, y4, m)}

    def pred(fam, lay):
        p = fits[fam]
        if fam == "bayes_true":
            return p[0] + p[1] * lay.bayes([h], [e])[0]
        if fam == "bayes_free":
            return BayesFree.predict(lay, p)
        return {"counter": lay.counter, "counter2": lay.counter2, "additive": lay.additive}[fam](p)

    s = pd.read_parquet(os.path.join(A.run3, f"{name}.surf.parquet"))
    rows = []
    for lay_name in ("sp4", "sp2", "sp8", "swap4"):
        if lay_name == "sp4":
            y = y4
        else:
            g = s[(s.layout == lay_name) & (s.cond == "base")]
            y = cells_vec(g)
            bayes_chk = float(np.abs(cells_vec(g, "lo_bayes") - LAYS[lay_name].bayes([h], [e])[0]).max())
        row = {"run": name, "layout": lay_name, **{k: meta[k] for k in ("h", "eps", "layers", "mlp", "pe", "seed",
                                                                         "conflict_frac")},
               "y_mean": float(y.mean()), "y_sd": float(y.std()),
               "bayes_mean": float(LAYS[lay_name].bayes([h], [e])[0].mean())}
        for fam in fits:
            yh = pred(fam, LAYS[lay_name])
            row[f"r2.{fam}"] = r2(yh, y)
            row[f"mae.{fam}"] = float(np.abs(yh - y).mean())
            row[f"predsd.{fam}"] = float(yh.std())
        if lay_name != "sp4":
            row["bayes_check"] = bayes_chk
            p = bf_fit(lay_name, y)  # refit on this layout alone, to compare (h', eps')
            row.update({"refit_h": p["h"], "refit_eps": p["eps"], "refit_r2": p["r2"]})
        row.update({"fit_h": fits["bayes_free"]["h"], "fit_eps": fits["bayes_free"]["eps"]})
        rows.append(row)
    return rows


def attention_one(name: str) -> list[dict]:
    """Within-sequence regression of log attention on distance, recency, typo-variant status and the number of
    copies sharing this copy's value, per late head, pooled over the original and swapped layouts."""
    att = pd.read_parquet(os.path.join(A.run3, f"{name}.attn.parquet"))
    heads = pd.read_csv(os.path.join(A.mech, "mech_heads.csv"))
    heads = heads[heads.run == name].set_index("head")
    rows = []
    for hd, k in att.groupby("head"):
        k = k[k.a > 1e-8].copy()
        k["la"] = np.log(k.a)
        k["nsame"] = np.where(k.recent == 1, k.n_w, k.n_c)
        k["lnsame"] = np.log(k.nsame)
        grp = k.groupby(["layout", "n_c", "n_w", "G", "fill"])
        for c in ("la", "d", "recent", "variant", "lnsame"):
            k[c + "_c"] = k[c] - grp[c].transform("mean")
        ss = (k.la_c ** 2).sum()
        out = {"run": name, "head": hd, "share": heads.loc[hd, "share"] if hd in heads.index else np.nan,
               "mass": heads.loc[hd, "mass_on_copies"] if hd in heads.index else np.nan, "n": len(k)}
        for tag, cols in (("d", ["d_c"]), ("d_rec_var", ["d_c", "recent_c", "variant_c"]),
                          ("full", ["d_c", "recent_c", "variant_c", "lnsame_c"])):
            X = k[cols].to_numpy()
            coef, *_ = np.linalg.lstsq(X, k.la_c.to_numpy(), rcond=None)
            out[f"r2_{tag}"] = float(1 - ((k.la_c.to_numpy() - X @ coef) ** 2).sum() / ss)
            if tag == "full":
                out.update({"b_d": -coef[0], "b_recent": coef[1], "b_variant": coef[2], "b_lnsame": coef[3]})
        # per layout: is the recent block above or below the distance kernel? In sp4 the recent copies are the
        # typo-variants; in swap4 the old copies are. Same sign in both = distance effect; flipped sign = content.
        for lay, kk in k.groupby("layout"):
            X = kk[["d_c", "recent_c"]].to_numpy()
            coef, *_ = np.linalg.lstsq(X, kk.la_c.to_numpy(), rcond=None)
            out[f"b_recent_{lay}"] = coef[1]
            out[f"b_d_{lay}"] = -coef[0]
        rows.append(out)
    return rows


A = None


def _init(args):
    global A, _BF
    A = args
    _BF = {k: BayesFree(v) for k, v in LAYS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run3", default="figs_run3")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--mech", default="figs_mech")
    ap.add_argument("--out", default="figs_run3")
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    names = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(a.run3, "*.json")))
    print(len(names), "runs with run-3 outputs", flush=True)
    with mp.Pool(a.procs, initializer=_init, initargs=(a,)) as pool:
        causal = [r for rs in pool.map(causal_one, names) for r in rs]
        gen = [r for rs in pool.map(general_one, names) for r in rs]
        att = [r for rs in pool.map(attention_one, names) for r in rs]
    pd.DataFrame(causal).to_csv(os.path.join(a.out, "causal.csv"), index=False)
    pd.DataFrame(gen).to_csv(os.path.join(a.out, "generalize.csv"), index=False)
    pd.DataFrame(att).to_csv(os.path.join(a.out, "attention.csv"), index=False)
    print("->", a.out)


if __name__ == "__main__":
    main()
