"""Plan §19.8 R5-P1 and §19.10 R5-P2: E10 (80k steps) against the 20k-step references.

w as in verify/evidence_weight.py (standard + wide probes). lambda = geometric-mean ALiBi slope of the layer-1
heads with induction score > 0.9 (done.json). References (pre-registered): h 0.0003 -> E6 s0; h 0.001 -> E5a
(median of 3 seeds); h 0.01 -> E1 (median of 3 seeds); all eps = 0.1.
R5-P1: w(80k)/w(20k) >= 1.5 at both low h and above the h = 0.01 ratio (H-signal); both in [0.8, 1.25] -> converged.
R5-P2: w ratio within a factor 1.5 of (lambda ratio)^1.22.

python -m recfreq.verify.long_train
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .evidence_weight import model_table

REF = {0.0003: "E6_", 0.001: "E5a_", 0.01: "E1_"}


def lam(run):
    H = json.load(open(f"results/runs/{run}/done.json"))["heads"]
    s = [h["slope"] for h in H if h["layer"] == 1 and h["ind_score"] > 0.9]
    return float(np.exp(np.mean(np.log(s)))) if s else np.nan


def main():
    M = model_table("results/runs", ("E1_", "E5a_", "E6_", "E10_"), allow=("K", "n_pairs", "steps"))
    M = M[(M.pe == "alibi_learn") & (M.cf == 0) & (M.eps == 0.1) & (M.K == 16) & (M.N == 256) & M.h.isin(list(REF))]
    M["lam"] = [lam(r) for r in M.run]
    rows = []
    for h, pre in REF.items():
        ref = M[(M.h == h) & M.run.str.startswith(pre)]
        if pre == "E6_":
            ref = ref[ref.seed == 0]
        new = M[(M.h == h) & M.run.str.startswith("E10_")]
        if new.empty:
            continue
        rw, rl = ref.w.median(), ref.lam.median()
        nw, nl = new.w.median(), new.lam.median()
        rows.append({"h": h, "w20k": rw, "w80k": nw, "w_ratio": nw / rw, "lam20k": rl, "lam80k": nl,
                     "lam_ratio": nl / rl, "pred_w_ratio": (nl / rl) ** 1.22,
                     "x_ref20k": ref.x_ref.median(), "x_ref80k": new.x_ref.median()})
    D = pd.DataFrame(rows).set_index("h")
    pd.set_option("display.width", 200)
    print(D.round(3).to_string())
    lo = D.loc[[0.0003, 0.001], "w_ratio"]
    sig = (lo >= 1.5).all() and (lo > D.loc[0.01, "w_ratio"]).all()
    conv = lo.between(0.8, 1.25).all()
    print(f"R5-P1: low-h w ratios {lo.round(2).tolist()} (h = 0.01: {D.loc[0.01, 'w_ratio']:.2f}) -> "
          f"{'PASS (H-signal)' if sig else 'converged (no H-signal)' if conv else 'FAIL (neither outcome)'}")
    f = D.w_ratio / D.pred_w_ratio
    print(f"R5-P2: observed / predicted w ratio {f.round(2).tolist()} -> {'PASS' if ((f <= 1.5) & (f >= 1 / 1.5)).all() else 'FAIL'}")
    D.to_csv("verify_out/long_train.csv")


if __name__ == "__main__":
    main()
