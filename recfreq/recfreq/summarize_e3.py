"""Test 6 (pre-registered): does more training or width remove the miscalibration?

Compares the fitted (h', eps') and fit quality of E3 models (60k steps at width 128; width 256 x 8 heads at
20k steps) with the matching 20k-step width-128 models from E1 (ALiBi) and E2a (RoPE).

python -m recfreq.summarize_e3 --fit figs_fit --fit_e3 figs_fit_E3
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="figs_fit")
    ap.add_argument("--fit_e3", default="figs_fit_E3")
    a = ap.parse_args()
    base = pd.read_csv(os.path.join(a.fit, "fits.csv"))
    base = base[base.arch.isin(["L2", "L2 rope"]) & base.exp.isin(["E1", "E2a"])]
    base["variant"] = "d128 20k"
    e3 = pd.read_csv(os.path.join(a.fit_e3, "fits.csv"))
    e3["variant"] = np.where(e3.arch.str.contains("d256"), "d256 20k", "d128 60k")
    for d in (base, e3):
        d["pe"] = np.where(d.arch.str.contains("rope"), "rope", "alibi")
    df = pd.concat([base, e3])
    df = df[((df.h == 0.01) & (df.eps == 0.1)) | ((df.h == 0.03) & (df.eps == 0.2))]
    cols = ["bfree.h", "bfree.eps", "bayes_free.cv_r2", "bayes_true.cv_r2", "counter_exp.cv_r2"]
    t = df.groupby(["h", "eps", "pe", "variant"])[cols].agg(["median", "min", "max", "size"])
    print(t.round(3).to_string())
    print("\npre-registered: miscalibration persists if h' moves toward h by less than 2x; "
          "'mainly under-training' if h' reaches within 1.5x of h")
    m = df.groupby(["h", "eps", "pe", "variant"])["bfree.h"].median().unstack("variant")
    m["true_h"] = m.index.get_level_values("h")
    for v in ("d128 60k", "d256 20k"):
        if v in m:
            m[f"{v}: improvement x"] = m["d128 20k"] / m[v]
            m[f"{v}: h'/h"] = m[v] / m.true_h
    m["d128 20k: h'/h"] = m["d128 20k"] / m.true_h
    print(m.round(3).to_string())


if __name__ == "__main__":
    main()
