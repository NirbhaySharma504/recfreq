"""Run 4B: does a content-aware counter close the gap between mechanism and behaviour?

B1  content-aware counter, 8 fitted parameters:
      weight of a copy = exp(-lam*d + r*recent + rho*variant) * nsame^eta
      LO = delta + gamma * (S_c - (1-kappa) S_w) / (S_c + S_w + Z0)         (normal layout; variant = recent copies)
      LO = delta + gamma * ((1-kappa) S_c - S_w) / (S_c + S_w + Z0)         (swapped layout; variant = old copies)
B2  mechanism-built: every layer-1 head that attends mainly to the copies uses its *measured* kernel
      (b_d, b_recent, b_variant, b_lnsame from run 4A, base condition) and its measured gain (mech prop1_gamma);
      only a global scale, an offset and kappa are fitted.
Both are fitted on the original layout (stored 256-fill probes) and scored on spacing 2, spacing 8 and the
swapped layout (run-3 surfaces), next to Bayes(h', eps') from run 3.

python -m recfreq.content_counter --run4 figs_run4 --run3 figs_run3 --mech figs_mech --out figs_run4
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .layouts import CELLS, Lay, r2

LAYS = {"sp4": Lay(4), "sp2": Lay(2), "sp8": Lay(8), "swap4": Lay(4, True)}


def feats(lay: Lay):
    """Per cell and copy: distance, recent flag, variant flag, log number of same-value copies (padded)."""
    nc = np.array([c[0] for c in CELLS], float)[:, None]
    nw = np.array([c[1] for c in CELLS], float)[:, None]
    var_c, var_w = (1.0, 0.0) if lay.swap else (0.0, 1.0)
    return {"DC": lay.DC, "DW": lay.DW, "lnc": np.log(nc), "lnw": np.log(nw), "var_c": var_c, "var_w": var_w,
            "swap": lay.swap}


FEATS = {k: feats(v) for k, v in LAYS.items()}


def kernel_sums(f, b_d, b_rec, b_var, b_ln):
    Sc = np.exp(-b_d * f["DC"] + b_var * f["var_c"] + b_ln * f["lnc"]).sum(1)
    Sw = np.exp(-b_d * f["DW"] + b_rec + b_var * f["var_w"] + b_ln * f["lnw"]).sum(1)
    return Sc, Sw


def readout(Sc, Sw, kappa, z0, swap):
    num = (1 - kappa) * Sc - Sw if swap else Sc - (1 - kappa) * Sw
    return num / (Sc + Sw + z0)


def b1_pred(t, lay_name):
    lam, r, rho, eta, kappa, lz, gamma, delta = t
    f = FEATS[lay_name]
    Sc, Sw = kernel_sums(f, math.exp(lam), r, rho, eta)
    return delta + gamma * readout(Sc, Sw, kappa, math.exp(lz), f["swap"])


def fit_b1(y):
    best = None
    for lam, r, rho, kap in itertools.product(np.log([0.01, 0.06, 0.3]), [0.0, 1.0], [0.0, -0.5], [0.0, 0.7]):
        x0 = np.array([lam, r, rho, 0.0, kap, math.log(0.3), 4.0, 0.0])
        try:
            res = least_squares(lambda t: b1_pred(t, "sp4") - y, x0,
                                bounds=([-9, -6, -6, -3, -2, -12, -60, -20], [3, 12, 6, 3, 2, 8, 60, 20]), max_nfev=800)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    return best.x


def b2_pred(t, lay_name, heads):
    scale, delta, kappa = t
    f = FEATS[lay_name]
    tot = 0.0
    for h in heads:
        Sc, Sw = kernel_sums(f, h["b_d"], h["b_recent"], h["b_variant"], h["b_lnsame"])
        z0 = (Sc + Sw) * (1 - h["mass"]) / max(h["mass"], 1e-3)  # attention left on non-copy positions
        tot = tot + h["gamma"] * readout(Sc, Sw, kappa, z0, f["swap"])
    return delta + scale * tot


def surfaces(name, run3):
    pr = pd.read_parquet(os.path.join(A.runs, name, "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS]
    out = {"sp4": pr.lo_model.to_numpy()}
    s = pd.read_parquet(os.path.join(run3, f"{name}.surf.parquet"))
    for lay in ("sp2", "sp8", "swap4"):
        g = s[(s.layout == lay) & (s.cond == "base")].set_index(["n_c", "n_w", "G"]).loc[CELLS]
        out[lay] = g.lo.to_numpy()
    return out


def work(name):
    ys = surfaces(name, A.run3)
    row = {"run": name}
    t1 = fit_b1(ys["sp4"])
    row.update({f"b1.{k}": v for k, v in zip(["lam", "r", "rho", "eta", "kappa", "lz", "gamma", "delta"], t1)})
    heads = pd.read_csv(os.path.join(A.run4, f"{name}.heads.csv"))
    heads = heads[(heads.cond == "base") & (heads["mass"] > 0.5)]
    mh = pd.read_csv(os.path.join(A.mech, "mech_heads.csv"))
    mh = mh[mh.run == name].set_index("head").prop1_gamma
    hl = []
    for _, h in heads.iterrows():
        tag = f"L1H{int(h['head'])}"
        if tag in mh.index:
            hl.append({**h.to_dict(), "gamma": float(mh[tag])})
    row["b2.n_heads"] = len(hl)
    # post-hoc diagnostic: how well does each head's fitted kernel reproduce its *measured* attention fraction on
    # the old copies (from the mech decomposition, same sp4 cells)? If poorly, B2's failure is about the kernel.
    cells_p = os.path.join(A.mech, f"{name}.cells.parquet")
    if hl and os.path.exists(cells_p):
        mc = pd.read_parquet(cells_p).set_index(["n_c", "n_w", "G"]).loc[CELLS]
        f = FEATS["sp4"]
        r2s = []
        for h in hl:
            tag = f"L1H{int(h['head'])}"
            Sc, Sw = kernel_sums(f, h["b_d"], h["b_recent"], h["b_variant"], h["b_lnsame"])
            pred = Sc / (Sc + Sw)
            meas = (mc[f"{tag}.A_c"] / (mc[f"{tag}.A_c"] + mc[f"{tag}.A_w"])).to_numpy()
            r2s.append(r2(pred, meas))
        row["b2.kernel_frac_r2_median"] = float(np.median(r2s))
        # and the same readout fed with the measured fractions instead of the kernel (3 fitted parameters)
        meas_fr = [((mc[f"L1H{int(h['head'])}.A_c"]) / (mc[f"L1H{int(h['head'])}.A_c"] + mc[f"L1H{int(h['head'])}.A_w"])
                    ).to_numpy() for h in hl]
        X = np.stack([h["gamma"] * (2 * fr - 1) for h, fr in zip(hl, meas_fr)], 1).sum(1)
        A_ = np.stack([np.ones_like(X), X], 1)
        coef, *_ = np.linalg.lstsq(A_, ys["sp4"], rcond=None)
        row["b2.measured_attention_r2_sp4"] = r2(A_ @ coef, ys["sp4"])
    t2 = None
    if hl:
        best = None
        for k0 in (0.0, 0.5, 1.0):
            res = least_squares(lambda t: b2_pred(t, "sp4", hl) - ys["sp4"], np.array([1.0, 0.0, k0]),
                                bounds=([-20, -20, -2], [20, 20, 2]))
            if best is None or res.cost < best.cost:
                best = res
        t2 = best.x
    for lay, y in ys.items():
        p1 = b1_pred(t1, lay)
        row[f"r2.b1.{lay}"], row[f"mae.b1.{lay}"] = r2(p1, y), float(np.abs(p1 - y).mean())
        if t2 is not None:
            p2 = b2_pred(t2, lay, hl)
            row[f"r2.b2.{lay}"], row[f"mae.b2.{lay}"] = r2(p2, y), float(np.abs(p2 - y).mean())
    return row


A = None


def _init(a):
    global A
    A = a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run4", default="figs_run4")
    ap.add_argument("--run3", default="figs_run3")
    ap.add_argument("--mech", default="figs_mech")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="figs_run4")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    names = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(a.run4, "*.json")))
    names = [n for n in names if os.path.exists(os.path.join(a.run3, f"{n}.surf.parquet"))]
    print(len(names), "models with run-3 and run-4 data", flush=True)
    with mp.Pool(a.procs, initializer=_init, initargs=(a,)) as pool:
        rows = pool.map(work, names)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "content_counter.csv"), index=False)
    gz = pd.read_csv(os.path.join(a.run3, "generalize.csv")).set_index(["run", "layout"])
    for lay in ("sp4", "sp2", "sp8", "swap4"):
        df[f"r2.bf.{lay}"] = [gz.loc[(r, lay), "r2.bayes_free"] for r in df.run]
        df[f"mae.bf.{lay}"] = [gz.loc[(r, lay), "mae.bayes_free"] for r in df.run]
        df[f"r2.c2.{lay}"] = [gz.loc[(r, lay), "r2.counter2"] for r in df.run]
        df[f"mae.c2.{lay}"] = [gz.loc[(r, lay), "mae.counter2"] for r in df.run]
    df.to_csv(os.path.join(a.out, "content_counter.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\nmedian R² (sp4 in-sample; sp2, sp8 held out):")
    print(pd.DataFrame({m: [df[f"r2.{m}.{l}"].median() for l in ("sp4", "sp2", "sp8")] for m in ("bf", "b1", "b2", "c2")},
                       index=["sp4", "sp2", "sp8"]).round(3).to_string())
    print("\nmedian absolute error:")
    print(pd.DataFrame({m: [df[f"mae.{m}.{l}"].median() for l in ("sp4", "sp2", "sp8", "swap4")]
                        for m in ("bf", "b1", "b2", "c2")}, index=["sp4", "sp2", "sp8", "swap4"]).round(3).to_string())
    gap = np.median([df[f"r2.bf.{l}"].median() - df[f"r2.b1.{l}"].median() for l in ("sp2", "sp8")])
    ratio = (df["mae.b1.swap4"] / df["mae.bf.swap4"]).median()
    print(f"\nB1 pre-registered: R² gap to Bayes(h', eps') on sp2/sp8 = {gap:.3f} (pass if <= 0.03); "
          f"swap4 error ratio = {ratio:.2f} (pass if <= 1.5)")
    b2 = np.median([df[f"r2.b2.{l}"].median() for l in ("sp2", "sp8")])
    print(f"B2 pre-registered: median R² on sp2/sp8 = {b2:.3f} (pass if >= 0.8; against if < 0.6)")
    if "b2.kernel_frac_r2_median" in df:
        print("B2 post-hoc diagnostic: fitted kernel vs measured attention fraction, median R² "
              f"{df['b2.kernel_frac_r2_median'].median():.3f}; same readout with measured attention (sp4) R² "
              f"{df['b2.measured_attention_r2_sp4'].median():.3f}")
    print("\nB1 fitted content terms (median): ",
          df[["b1.r", "b1.rho", "b1.eta", "b1.kappa"]].median().round(3).to_dict())


if __name__ == "__main__":
    main()
