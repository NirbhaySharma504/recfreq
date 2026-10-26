"""Plan §19 A1–A2: Prop. 2 redone, and each approximation checked against the exact oracle.

Probe: n_c old copies of c, a gap of G pairs, n_w recent copies of w = c+1 (all 4 pairs apart), query 4 pairs later;
structured typos (y = z w.p. 1-eps, y = z+1 w.p. eps), V values, switch = uniform redraw with prob. h per pair.

With s = (1-h)^G, R = ((1-eps)/eps)^n_w and rho = eps/(1-eps), the two paths that matter give the posterior odds
    P(z=c) / P(z=w) = (1 + sV/(1-s)) / (R (1 + rho^n_c))
  - "1 +":           a redraw can land on c again, so "no change" has prior s + (1-s)/V, not s;
  - "(1 + rho^n_c)": after a switch, the old copies may also have come from z = c-1 (all typos);
and the probes measure the observed-token odds p(y=c)/p(y=w) = (1-eps)P_c / ((1-eps)P_w + eps P_c), which is 0 at
P_c/P_w = (1-eps)/(1-2eps).  Boundary:
    P2 (corrected):   s/(1-s) = [R (1 + rho^n_c) (1-eps)/(1-2eps) - 1] / V
    P1 (paper, exact s):         s/(1-s) = R / V
    P0 (paper, small hG):        h G = V / R
Left out of P2: switches inside the recent block or between the last copy and the query (O(h * 4 n_w)).

python -m recfreq.verify.prop2_exact --out verify_out
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from .theory import V, exact_gstar, exact_lo


def g_from_r(r, h):
    if r <= 0:
        return 0.0
    return math.log(r / (1 + r)) / math.log1p(-h)


def p0(h, eps, n_w, n_c=2):
    return V / (h * ((1 - eps) / eps) ** n_w)


def p1(h, eps, n_w, n_c=2):
    return g_from_r(((1 - eps) / eps) ** n_w / V, h)


def p2(h, eps, n_w, n_c=2):
    R, rho = ((1 - eps) / eps) ** n_w, eps / (1 - eps)
    return g_from_r((R * (1 + rho ** n_c) * (1 - eps) / (1 - 2 * eps) - 1) / V, h)


def lo_p2(h, eps, n_c, n_w, G):
    """P2's log-odds at the query (posterior odds mapped to observed tokens)."""
    s = (1 - h) ** G
    R, rho = ((1 - eps) / eps) ** n_w, eps / (1 - eps)
    o = (1 + s * V / (1 - s)) / (R * (1 + rho ** n_c))
    pc = o / (1 + o)
    return math.log((1 - eps) * pc) - math.log((1 - eps) * (1 - pc) + eps * pc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    rows = []
    for h in (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.06, 0.1, 0.2):
        for eps in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45):
            for n_c in (1, 2, 8):
                for n_w in range(1, 7):
                    ge = exact_gstar(h, eps, n_w, n_c, gmax=1e5)
                    rows.append({"h": h, "eps": eps, "n_c": n_c, "n_w": n_w, "G_exact": ge,
                                 "G_P0": p0(h, eps, n_w), "G_P1": p1(h, eps, n_w), "G_P2": p2(h, eps, n_w, n_c)})
    df = pd.DataFrame(rows)
    for k in ("P0", "P1", "P2"):
        df[f"err_{k}"] = np.abs(np.log(df[f"G_{k}"].clip(lower=1e-9) / df.G_exact.clip(lower=1e-9)))
    df.to_csv(os.path.join(a.out, "prop2_exact.csv"), index=False)
    ok = df[(df.G_exact >= 4) & np.isfinite(df.G_exact)]
    print(f"{len(ok)} cases with exact switch point G* >= 4 pairs (n_c in 1,2,8; n_w 1..6)")
    print("median |log(G_formula / G_exact)|, i.e. 0.10 = 10% off; and share of cases within 10%")
    for name, d in (("all", ok), ("n_c >= 2", ok[ok.n_c >= 2]), ("n_c = 1", ok[ok.n_c == 1]),
                    ("eps <= 0.2", ok[ok.eps <= 0.2]), ("eps >= 0.3", ok[ok.eps >= 0.3]),
                    ("hG* <= 0.3", ok[ok.h * ok.G_exact <= 0.3]), ("hG* > 1", ok[ok.h * ok.G_exact > 1])):
        print(f"  {name:12s} n={len(d):4d}  " + "  ".join(
            f"{k}: {d[f'err_{k}'].median():.3f} ({(d[f'err_{k}'] < math.log(1.1)).mean():.0%})" for k in ("P0", "P1", "P2")))
    print("\nworst P2 cases:")
    print(ok.sort_values("err_P2").tail(8)[["h", "eps", "n_c", "n_w", "G_exact", "G_P1", "G_P2", "err_P2"]].round(3).to_string())
    # log-odds curve check on the actual probe cells
    errs = []
    for h in (0.003, 0.01, 0.03):
        for eps in (0.05, 0.1, 0.2, 0.3):
            for n_c in (1, 2, 4, 8):
                for n_w in (1, 2, 3, 4):
                    for G in (2, 4, 8, 16, 32, 64, 128):
                        errs.append(abs(lo_p2(h, eps, n_c, n_w, G) - exact_lo(h, eps, n_c, n_w, G)))
    print(f"\nP2 log-odds vs exact on the probe cells of the 12-setting grid: median |diff| {np.median(errs):.3f}, "
          f"90th pct {np.quantile(errs, 0.9):.3f}, max {np.max(errs):.3f} nats")


if __name__ == "__main__":
    main()
