"""Plan §19.5 R3-P1 (E8, context length) and §19.6 R3-P3 (E9, number of keys) at the low-h edge.

Statistic: geometric mean over seeds and over n_w in {3, 4} of the model's switch gap G* (n_c = 2), read from the
standard probes plus the wide-window probes where present. Compared with the default task (N = 256, K = 16):
E6 at h = 0.0003 and E5a at h = 0.001, both eps = 0.1.
  R3-P1 (H-ctx): N = 512 / N = 256 >= 1.3 at both h; no context effect if both ratios lie in [0.8, 1.25].
  R3-P3 (H-gap): K = 32 / K = 16 >= 1.3 and K = 8 / K = 16 <= 0.8 at both h; H-ctx predicts both in [0.8, 1.25].

python -m recfreq.verify.context_gap
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evidence_weight import model_table


def gstat(M):
    g = M[["G3", "G4"]].to_numpy().ravel()
    g = g[np.isfinite(g)]
    return (float(np.exp(np.log(g).mean())) if len(g) else np.nan), len(g)


def main():
    M = model_table("results/runs", ("E5a_", "E6_", "E8_", "E9_"), allow=("K", "n_pairs"))
    M = M[(M.pe == "alibi_learn") & (M.cf == 0) & (M.eps == 0.1) & M.h.isin([0.0003, 0.001])]
    pd.set_option("display.width", 200)
    print(M[["run", "h", "K", "N", "seed", "G1", "G2", "G3", "G4", "G5", "G6"]].round(1).to_string())
    rows = []
    for (h, K, N), d in M.groupby(["h", "K", "N"]):
        g, n = gstat(d)
        rows.append({"h": h, "K": K, "N": N, "models": len(d), "crossings": n, "G34": g})
    S = pd.DataFrame(rows)
    print(S.round(2).to_string())
    ref = {h: S[(S.h == h) & (S.K == 16) & (S.N == 256)].G34.iloc[0] for h in S.h.unique()}
    S["ratio"] = [g / ref[h] for g, h in zip(S.G34, S.h)]
    for h in sorted(ref):
        s = S[S.h == h].set_index(["K", "N"]).ratio
        print(f"h = {h}: N512/N256 = {s.get((16, 512), np.nan):.2f}; K32/K16 = {s.get((32, 256), np.nan):.2f}; "
              f"K8/K16 = {s.get((8, 256), np.nan):.2f}; K64/K16 = {s.get((64, 256), np.nan):.2f}")
    S.to_csv("verify_out/context_gap.csv", index=False)


if __name__ == "__main__":
    main()
