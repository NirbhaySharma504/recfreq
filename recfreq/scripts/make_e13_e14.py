"""Plan §20: write configs/E13.json (validation sweep) and configs/E14*.json (recipe robustness).

E13: 24 (h, eps) points by Latin hypercube in (log10 h, logit eps) over the band h 0.001-0.045, eps 0.05-0.25
     (scipy LatinHypercube, seed 20261104), 2 significant figures; 3 seeds; default recipe.
E14: 4 fixed points inside the band, off every earlier grid. Default recipe with 3 seeds (E14_base), and one change
     at a time with 2 seeds: d_model 64 / 256, lr 5e-4 / 2e-3, batch 256, 40k steps (E14_hp); mlp on, 3 layers (E14_arch).

python scripts/make_e13_e14.py
"""
import json
import math
import os

import numpy as np
from scipy.stats import qmc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = {"steps": 20000, "mlp": 0, "pe": "alibi_learn", "layers": 2}


def sig2(x):
    return float(f"{x:.2g}")


lo = np.array([math.log10(0.001), math.log(0.05 / 0.95)])
hi = np.array([math.log10(0.045), math.log(0.25 / 0.75)])
X = qmc.scale(qmc.LatinHypercube(d=2, seed=20261104).random(24), lo, hi)
pts = [{"h": sig2(10 ** a), "eps": sig2(1 / (1 + math.exp(-b)))} for a, b in X]
pts.sort(key=lambda p: (p["h"], p["eps"]))
assert len({(p["h"], p["eps"]) for p in pts}) == 24
E14_PTS = [{"h": 0.0015, "eps": 0.08}, {"h": 0.004, "eps": 0.18}, {"h": 0.012, "eps": 0.12}, {"h": 0.025, "eps": 0.22}]

cfgs = {
    "E13": {"exp": "E13", "base": BASE, "grid": {"seed": [0, 1, 2], "point": pts}},
    "E14_base": {"exp": "E14", "base": BASE, "grid": {"seed": [0, 1, 2], "point": E14_PTS}},
    "E14_hp": {"exp": "E14", "base": BASE, "grid": {
        "seed": [0, 1], "change": [{"d_model": 64}, {"d_model": 256}, {"lr": 5e-4}, {"lr": 2e-3}, {"batch": 256},
                                   {"steps": 40000}], "point": E14_PTS}},
    "E14_arch": {"exp": "E14", "base": BASE, "grid": {"seed": [0, 1], "change": [{"mlp": 1}, {"layers": 3}],
                                                      "point": E14_PTS}},
}
for name, c in cfgs.items():
    with open(os.path.join(ROOT, "configs", f"{name}.json"), "w") as f:
        json.dump(c, f, indent=1)
print("E13 points:", [(p["h"], p["eps"]) for p in pts])
