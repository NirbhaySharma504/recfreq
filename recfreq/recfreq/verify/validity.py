"""Plan §20: does the frozen law hold across a stated band, and with what variability?

The yardstick is a replicate: another training run at the same setting with a different seed. At a setting with
seeds i = 1..n (surfaces y_i over the 112 probe cells, law prediction p):
  rep   = median over pairs i<j of RMSE(y_i, y_j)                  how far two runs of the same recipe differ
  err   = median over i of RMSE(y_i, p)                             how far the law is from a run
  ratio = err / rep                                                 law vs. a replicate (1 = as good as rerunning)
  noise = sqrt(mean over cells of the across-seed variance)         seed sd per cell
  sys   = sqrt(max(0, mean((ybar - p)^2) - noise^2 / n))            law bias after removing seed noise from ybar
Plus the believed parameters from each model's own Bayes(h', eps') fit (law_scope.fit_one):
  dlh   = log10(h_fit / h_law),  dle = logit(eps_fit) - logit(eps_law).

python -m recfreq.verify.validity baseline --out verify_out      (existing models; numbers that feed §20)
python -m recfreq.verify.validity score --out verify_out         (E13 / E14 after training; grades §20)
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from ..layouts import CELLS, Lay
from . import law_scope
from .law_scope import A_MED, B_MED, law, logit

BAND = {"h": (0.001, 0.045), "eps": (0.05, 0.25)}
FIT_GRID = {(h, e) for h in (0.003, 0.01, 0.03) for e in (0.05, 0.1, 0.2, 0.3)}
DEFAULT = dict(V=64, K=16, n_pairs=256, key_dist="zipf", typo="structured", layers=2, mlp=0, pe="alibi_learn",
               d_model=128, steps=20000, lr=1e-3, batch=128, conflict_frac=0.0)


def in_band(h, e):
    return BAND["h"][0] <= h <= BAND["h"][1] and BAND["eps"][0] <= e <= BAND["eps"][1]


def variant(a: dict) -> str:
    """Name of the departure from the default recipe ('' if default)."""
    d = [f"{k}={a.get(k, v)}" for k, v in DEFAULT.items() if a.get(k, v) != v]
    return ",".join(d)


def collect(runs_dir, prefixes):
    out = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "*", "done.json"))):
        d = os.path.dirname(f)
        name = os.path.basename(d)
        if not name.startswith(prefixes) or not os.path.exists(os.path.join(d, "probes.parquet")):
            continue
        a = json.load(open(f))["args"]
        if a["h"] <= 0 or a["eps"] <= 0:
            continue
        out.append({"run": name, "exp": name.split("_")[0], "h": a["h"], "eps": a["eps"], "seed": a["seed"],
                    "variant": variant(a), "V": a.get("V", 64), "typo": a.get("typo", "structured"), "dir": d})
    return pd.DataFrame(out)


def fits(R, cache, procs):
    old = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame({"run": []})
    todo = R[~R.run.isin(old.run)]
    if len(todo):
        print(f"fitting {len(todo)} new models", flush=True)
        with mp.Pool(procs) as pool:
            new = pool.map(law_scope.fit_one, todo.to_dict("records"))
        old = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
        old.to_csv(cache, index=False)
    return R.merge(old[["run", "h_fit", "eps_fit", "a_fit", "b_fit", "var_y"]], on="run", how="left")


def surface(d):
    return pd.read_parquet(os.path.join(d, "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS].lo_model.to_numpy()


def per_model(F):
    lays = {}
    rows = []
    for r in F.itertuples():
        lay = lays.setdefault(r.V, Lay(4, False, r.V))
        y = surface(r.dir)
        hp, ep = law(r.h, r.eps)
        p = A_MED + B_MED * lay.bayes([hp], [ep], r.typo)[0]
        row = r._asdict()
        row.update({"err": float(np.sqrt(((y - p) ** 2).mean())), "sd_y": float(y.std()), "h_law": hp, "eps_law": ep,
                    "dlh": float(np.log10(r.h_fit / hp)) if np.isfinite(r.h_fit) else np.nan,
                    "dle": float(logit(r.eps_fit) - logit(ep)) if np.isfinite(r.eps_fit) else np.nan})
        row["r2_law"] = 1 - row["err"] ** 2 / y.var()
        if np.isfinite(r.h_fit):
            po = r.a_fit + r.b_fit * lay.bayes([r.h_fit], [r.eps_fit], r.typo)[0]
            row["own"] = float(np.sqrt(((y - po) ** 2).mean()))
        else:
            row["own"] = float(y.std())
        row["ratio_own"] = row["err"] / row["own"]
        row["_y"], row["_p"] = y, p
        rows.append(row)
    return pd.DataFrame(rows)


def per_setting(M, keys=("variant", "h", "eps")):
    rows = []
    for k, g in M.groupby(list(keys)):
        Y = np.stack(g._y.to_list())
        p = g._p.iloc[0]
        n = len(Y)
        row = dict(zip(keys, k))
        row.update({"n": n, "exp": g.exp.iloc[0], "err": float(np.median(g.err)), "sd_y": float(np.median(g.sd_y)),
                    "dlh": float(np.nanmedian(g.dlh)) if g.dlh.notna().any() else np.nan,
                    "dle": float(np.nanmedian(g.dle)) if g.dle.notna().any() else np.nan,
                    "ratio_own": float(np.median(g.ratio_own)), "r2_law": float(np.median(g.r2_law)),
                    "dlh_sd": float(g.dlh.std()) if g.dlh.notna().sum() >= 2 else np.nan,
                    "dle_sd": float(g.dle.std()) if g.dle.notna().sum() >= 2 else np.nan,
                    "in_band": in_band(row["h"], row["eps"]), "in_fit_grid": (row["h"], row["eps"]) in FIT_GRID})
        if n >= 2:
            row["rep"] = float(np.median([np.sqrt(((Y[i] - Y[j]) ** 2).mean()) for i, j in itertools.combinations(range(n), 2)]))
            noise2 = float(Y.var(0, ddof=1).mean())
            row["noise"] = np.sqrt(noise2)
            row["sys"] = float(np.sqrt(max(0.0, ((Y.mean(0) - p) ** 2).mean() - noise2 / n)))
            row["ratio"] = row["err"] / row["rep"]
        rows.append(row)
    return pd.DataFrame(rows)


def q(x, ps=(5, 25, 50, 75, 90, 95)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f"p{p}": round(float(np.percentile(x, p)), 3) for p in ps} | {"n": len(x)}


def baseline(a):
    R = collect(a.runs, ("E1_", "E2b_", "E5a_", "E6_"))
    R = R[R.variant == ""]
    F = fits(R, os.path.join(a.out, "law_scope_fits.csv"), a.procs)
    M = per_model(F)
    S = per_setting(M)
    S.to_csv(os.path.join(a.out, "validity_baseline_settings.csv"), index=False)
    M.drop(columns=["_y", "_p"]).to_csv(os.path.join(a.out, "validity_baseline_models.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())
    for lab, m in (("all in-band", S.in_band), ("in-band, out of fitting grid", S.in_band & ~S.in_fit_grid),
                   ("in fitting grid", S.in_fit_grid), ("outside band", ~S.in_band)):
        s = S[m]
        print(f"\n== {lab}: {len(s)} settings")
        for c in ("ratio", "ratio_own", "r2_law", "err", "rep", "noise", "sys", "dlh", "dle", "dlh_sd", "dle_sd"):
            print(f"  {c:9s}", q(s[c]))
        print(f"  ratio_own <= 1.51: {(s.ratio_own <= 1.51).sum()}/{len(s)};  sys > noise: {(s.sys > s.noise).sum()}/{s.sys.notna().sum()}")
    Mb = M[[in_band(h, e) for h, e in zip(M.h, M.eps)]]
    print("\n== in-band models (per model):")
    for c in ("err", "dlh", "dle", "r2_law", "ratio_own"):
        print(f"  {c:9s}", q(Mb[c], (5, 10, 25, 50, 75, 90, 95)))
    print(f"  |dlh| <= 0.3: {(Mb.dlh.abs() <= 0.3).mean():.2f};  |dle| <= 0.6: {(Mb.dle.abs() <= 0.6).mean():.2f};  "
          f"r2_law >= 0.7: {(Mb.r2_law >= 0.7).mean():.2f}")


def score(a):
    R = collect(a.runs, ("E13_", "E14_"))
    F = fits(R, os.path.join(a.out, "law_scope_fits.csv"), a.procs)
    M = per_model(F)
    S = per_setting(M)
    S.to_csv(os.path.join(a.out, "validity_settings.csv"), index=False)
    M.drop(columns=["_y", "_p"]).to_csv(os.path.join(a.out, "validity_models.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())
    grade(M, S)


def verdict(ok):
    return "PASS" if ok else "FAIL"


def grade(M, S):
    """§20.1, graded exactly as registered."""
    flat = M[M.var_y < 0.05]
    if len(flat):
        print("flat surfaces (excluded):", ", ".join(flat.run))
    M = M[M.var_y >= 0.05]
    m13, s13 = M[M.exp == "E13"], S[S.exp == "E13"]
    print(f"\n==== E13: {len(m13)} models, {len(s13)} settings")
    if len(m13):
        r2, er = m13.r2_law, m13.err
        p1 = r2.median() >= 0.80 and (r2 >= 0.7).mean() >= 0.85
        print(f"P1  median R² {r2.median():.3f} (>= 0.80); share >= 0.7: {(r2 >= 0.7).mean():.2f} (>= 0.85)  {verdict(p1)}")
        p2 = er.median() <= 0.85 and er.quantile(0.9) <= 1.1
        print(f"P2  median RMSE {er.median():.3f} (<= 0.85); p90 {er.quantile(0.9):.3f} (<= 1.1)  {verdict(p2)}")
        a, b = (m13.dlh.abs() <= 0.3).mean(), (m13.dle.abs() <= 0.6).mean()
        p3 = a >= 0.90 and b >= 0.85
        print(f"P3  |dlh| <= 0.3: {a:.2f} (>= 0.90); |dle| <= 0.6: {b:.2f} (>= 0.85)  {verdict(p3)}")
        k = int((s13.ratio_own <= 1.51).sum())
        p4 = k >= 17
        print(f"P4  fit ratio <= 1.51 at {k}/{len(s13)} settings (>= 17)  {verdict(p4)}")
        sn = (s13.sys > s13.noise).mean()
        p5 = sn >= 0.8 and s13.ratio.median() >= 1.5 and s13.dlh_sd.median() <= 0.05
        print(f"P5  sys > noise at {sn:.2f} (>= 0.8); median replicate ratio {s13.ratio.median():.2f} (>= 1.5); "
              f"median dlh sd {s13.dlh_sd.median():.3f} (<= 0.05)  {verdict(p5)}")
        print(f"HEADLINE (P1-P4): band {'HOLDS' if p1 and p2 and p3 and p4 else 'DOES NOT HOLD'}")
        print("settings failing P4:", s13[s13.ratio_own > 1.51][["h", "eps", "ratio_own"]].round(2).to_dict("records"))
        for c in ("r2_law", "err", "dlh", "dle", "ratio_own"):
            print(f"  per-model {c:9s}", q(m13[c], (5, 10, 25, 50, 75, 90, 95)))
        for c in ("ratio", "rep", "noise", "sys", "dlh_sd", "dle_sd"):
            print(f"  setting   {c:9s}", q(s13[c]))
    m14 = M[M.exp == "E14"]
    if len(m14):
        print(f"\n==== E14: {len(m14)} models")
        base = m14[m14.variant == ""]
        if not len(base):
            return
        be = base.err.median()
        print(f"E14_base: {len(base)} models, median RMSE {be:.3f}, |dlh| {base.dlh.abs().median():.3f}, |dle| {base.dle.abs().median():.3f}")
        HP = ["d_model=64", "d_model=256", "lr=0.0005", "lr=0.002", "batch=256", "steps=40000"]
        n_ok = 0
        for v, g in m14[m14.variant != ""].groupby("variant"):
            ok = g.err.median() <= 1.25 * be and g.dlh.abs().median() <= 0.3 and g.dle.abs().median() <= 0.6
            hp = v in HP
            n_ok += int(ok and hp)
            print(f"  {v:14s} n={len(g):2d}  RMSE {g.err.median():.3f} (x{g.err.median() / be:.2f})  |dlh| {g.dlh.abs().median():.3f}  "
                  f"|dle| {g.dle.abs().median():.3f}  R² {g.r2_law.median():.3f}  {'transfers' if ok else 'does not transfer'}"
                  f"{'' if hp else '  (arch, exploratory)'}")
        done = sum(v in set(m14.variant) for v in HP)
        print(f"Q1  {n_ok}/{done} hp variants transfer (>= 4 of 6)  {verdict(n_ok >= 4) if done == 6 else 'incomplete'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "score"])
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    {"baseline": baseline, "score": score}[a.mode](a)


if __name__ == "__main__":
    main()
