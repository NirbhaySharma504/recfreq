"""Plan §19 round 2 (post hoc): does copy-head attention register extra recent copies?

Run-3 stored layer-1 attention on the sp4 probes. For every head and model: the share of copy attention on the recent
copies, A_w / (A_c + A_w), at n_c = 2. Sensitivity = slope of logit(share) per extra recent copy (n_w 1 -> 4),
averaged over short gaps G in {2, 4, 8}, where Bayes says each copy should matter most. Exact Bayes reference:
the posterior log-odds change per copy, log((1-eps)/eps).
python -m recfreq.verify.attn_evidence
"""
import glob
import math

import numpy as np
import pandas as pd


def main():
    rows = []
    for f in sorted(glob.glob("figs_run3/*_L2_*attn_s*.attn.parquet")):
        run = f.split("/")[-1].replace(".attn.parquet", "")
        pe = "rope" if "rope" in run else "alibi" if "alibi" in run else "abs"
        eps = float(run.split("_")[2][1:])
        d = pd.read_parquet(f)
        d = d[(d.layout == "sp4") & (d.n_c == 2) & d.G.isin([2, 4, 8])]
        t = d.groupby(["head", "n_w", "G", "fill", "recent"]).a.sum().unstack("recent").fillna(0)
        t["mass"] = t[0] + t[1]
        t["share"] = (t[1] / t.mass.clip(lower=1e-9)).clip(1e-3, 1 - 1e-3)
        g = t.reset_index().groupby(["head", "n_w"]).agg(share=("share", "mean"), mass=("mass", "mean")).reset_index()
        for h, x in g.groupby("head"):
            if x.mass.mean() < 0.2:  # not a copy head on these probes
                continue
            lg = np.log(x.share / (1 - x.share))
            slope = np.polyfit(x.n_w, lg, 1)[0]
            rows.append({"run": run, "pe": pe, "eps": eps, "head": h, "mass": x.mass.mean(), "slope": slope,
                         "bayes_per_copy": math.log((1 - eps) / eps)})
    D = pd.DataFrame(rows)
    D["ratio"] = D.slope / D.bayes_per_copy
    pd.set_option("display.width", 200)
    print(D.groupby(["pe", "eps"])[["slope", "bayes_per_copy", "ratio"]].median().round(3))
    m = D.groupby(["pe", "run"]).apply(lambda x: np.average(x.ratio, weights=x.mass)).groupby("pe").agg(["median", "min", "max", "count"])
    print("\nper model, mass-weighted mean over copy heads of (attention slope per copy / Bayes per copy):")
    print(m.round(3))
    D.to_csv("verify_out/attn_evidence.csv", index=False)


if __name__ == "__main__":
    main()
