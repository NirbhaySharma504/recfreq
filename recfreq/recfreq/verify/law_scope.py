"""Plan §18 (run 6): does Law 1 predict models trained in new settings, and how far does it reach?

Fits every run of E5a/E5n/E5d/E5r plus the baselines (E1 + E2b 2-layer ALiBi grid, E2a 2-layer RoPE at C, D)
with one pipeline (Bayes(h', eps') for the run's own V and typo type; exp and pow counters; leave-one-G-out CV),
caches the fits, then grades S1-S6 against the frozen law.

python -m recfreq.verify.law_scope --out verify_out [--procs 6]
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

from .. import fit_rules
from ..layouts import CELLS, BayesFree, Lay, fit_affine, r2
from ..probes import GAPS

# frozen in plan §18 (point estimates from the 12 setting medians, 36 models)
ALPHA, BETA, GAMMA, DELTA, A_MED, B_MED = 1.453, 0.736, -0.345, 0.438, 0.215, 1.433
MIN_VAR = 0.05
EXPS = ["E1", "E2b", "E2a", "E5a", "E5n", "E5d", "E5r"]


def logit(x):
    return np.log(x / (1 - x))


def law(h, e):
    return ALPHA * h ** BETA, float(1 / (1 + math.exp(-(GAMMA + DELTA * logit(e)))))


def runs(runs_dir):
    out = []
    for e in EXPS:
        for d in sorted(glob.glob(os.path.join(runs_dir, f"{e}_*"))):
            if not (os.path.exists(os.path.join(d, "done.json")) and os.path.exists(os.path.join(d, "probes.parquet"))):
                continue
            a = json.load(open(os.path.join(d, "done.json")))["args"]
            if a["h"] <= 0 or a["eps"] <= 0 or a["layers"] != 2 or a["mlp"]:
                continue
            if e == "E2a" and a["pe"] != "rope":
                continue
            out.append({"run": os.path.basename(d), "exp": e, "h": a["h"], "eps": a["eps"], "seed": a["seed"],
                        "pe": a["pe"], "V": a.get("V", 64), "K": a.get("K", 16), "N": a.get("n_pairs", 256),
                        "key_dist": a.get("key_dist", "zipf"), "typo": a.get("typo", "structured"), "dir": d})
    return pd.DataFrame(out)


_BF: dict = {}


def _bfree(V, typo):
    if (V, typo) not in _BF:
        _BF[(V, typo)] = BayesFree(Lay(4, False, V), typo)
    return _BF[(V, typo)]


def fit_one(r: dict) -> dict:
    y = pd.read_parquet(os.path.join(r["dir"], "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS]
    lo_b = y.lo_bayes.to_numpy()
    y = y.lo_model.to_numpy()
    lay = Lay(4, False, r["V"])
    out = {"run": r["run"], "var_y": float(y.var()),
           "bayes_check": float(np.abs(lay.bayes([r["h"]], [r["eps"]], r["typo"])[0] - lo_b).max())}
    if y.var() < MIN_VAR:
        return out
    bf = _bfree(r["V"], r["typo"])
    Gs = np.array([c[2] for c in CELLS])
    full = np.ones(len(CELLS), bool)
    fams = {
        "bfree": (lambda m: bf.fit(y, m), lambda p: BayesFree.predict(lay, p, r["typo"])),
        "cexp": (lambda m: fit_rules.fit_counter(y, m, "exp"), lambda p: fit_rules.counter_pred(p, "exp")),
        "cpow": (lambda m: fit_rules.fit_counter(y, m, "pow"), lambda p: fit_rules.counter_pred(p, "pow")),
    }
    for name, (fit, pred) in fams.items():
        p = fit(full)
        cv = np.empty(len(CELLS))
        for G in GAPS:
            m = Gs != G
            cv[~m] = pred(fit(m))[~m]
        out[f"{name}.r2"] = r2(pred(p), y)
        out[f"{name}.cv_rmse"] = float(np.sqrt(((cv - y) ** 2).mean()))
        if name == "bfree":
            out.update({"h_fit": p["h"], "eps_fit": p["eps"], "a_fit": p["a"], "b_fit": p["b"]})
    hp, ep = law(r["h"], r["eps"])
    out.update({"h_law": hp, "eps_law": ep,
                "r2_law": r2(A_MED + B_MED * lay.bayes([hp], [ep], r["typo"])[0], y)})
    x = lay.bayes([r["h"]], [r["eps"]], r["typo"])[0]
    ab = fit_affine(x, y, full)
    out["r2_true_bayes"] = r2(ab[0] + ab[1] * x, y)
    return out


def med(df, cols):
    return df.groupby(["h", "eps"])[cols].median()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "law_scope_fits.csv")
    R = runs(a.runs)
    old = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame({"run": []})
    todo = R[~R.run.isin(old.run)]
    print(f"{len(R)} runs found, {len(todo)} to fit", flush=True)
    if len(todo):
        with mp.Pool(a.procs) as pool:
            new = pool.map(fit_one, todo.to_dict("records"))
        old = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
        old.to_csv(cache, index=False)
    F = R.merge(old, on="run")
    print(f"max |layouts Bayes - stored oracle| over all runs: {F.bayes_check.max():.1e}")
    flat = F[F.var_y < MIN_VAR]
    if len(flat):
        print("flat surfaces (excluded from R²):", ", ".join(flat.run))
    F = F[F.var_y >= MIN_VAR].copy()
    F["bayes_wins"] = F["bfree.cv_rmse"] < F[["cexp.cv_rmse", "cpow.cv_rmse"]].min(1)
    pd.set_option("display.width", 220)
    std = (F.V == 64) & (F.K == 16) & (F.N == 256) & (F.key_dist == "zipf") & (F.typo == "structured")
    base = F[F.exp.isin(["E1", "E2b"]) & std & (F.pe == "alibi_learn")]
    print(f"\nbaseline (E1+E2b L2 ALiBi, this pipeline): {len(base)} models; law R² median {base.r2_law.median():.3f}; "
          f"Bayes wins {base.bayes_wins.mean():.2f}")

    def setting_table(df):
        t = df.groupby(["h", "eps"]).agg(n=("run", "size"), h_law=("h_law", "first"), h_fit=("h_fit", "median"),
                                         eps_law=("eps_law", "first"), eps_fit=("eps_fit", "median"),
                                         r2_law=("r2_law", "median"), r2_own=("bfree.r2", "median"),
                                         r2_true_bayes=("r2_true_bayes", "median"), bayes_wins=("bayes_wins", "mean"))
        t["h_ratio"] = t.h_fit / t.h_law
        t["logit_err"] = (logit(t.eps_fit) - logit(t.eps_law)).abs()
        return t

    # S1 / S2
    e5a = F[F.exp == "E5a"]
    if len(e5a):
        t = setting_table(e5a)
        interior = {(0.005, 0.07), (0.02, 0.15), (0.007, 0.25)}
        t["kind"] = ["interp" if k in interior else "extrap" for k in t.index]
        t["law_holds"] = t.r2_law >= 0.75
        t["S1_pass"] = t.law_holds & (t.h_ratio.between(0.5, 2)) & (t.logit_err < 0.5)
        print("\nS1/S2: new (h, eps), frozen law\n" + t.round(3).to_string())
        ti = t[t.kind == "interp"]
        print(f"S1 (interpolation): {'PASS' if len(ti) == 3 and ti.S1_pass.all() else 'FAIL/incomplete'} "
              f"({ti.S1_pass.sum()}/{len(ti)} settings)")
        te = t[t.kind == "extrap"]
        print(f"S2 (extrapolation): law holds at {te.law_holds.sum()}/{len(te)}: "
              + ", ".join(f"{k}: {'holds' if v else 'fails'}" for k, v in te.law_holds.items()))

    # S3
    new = F[F.exp.str.startswith("E5")]
    if len(new):
        print(f"\nS3 (form): Bayes(h', eps') beats both counters on leave-one-G-out RMSE in "
              f"{new.bayes_wins.sum()}/{len(new)} = {new.bayes_wins.mean():.2f} of new runs (pass if >= 0.80): "
              f"{'PASS' if new.bayes_wins.mean() >= 0.8 else 'FAIL'}")
        print(new.groupby("exp").bayes_wins.agg(["sum", "size"]).to_string())

    # S4: context length
    ctx = F[((F.exp == "E5n") | (F.exp.isin(["E1", "E2b"]) & std & (F.pe == "alibi_learn")))
            & (F.K == 16) & (F.V == 64) & (F.typo == "structured") & (F.key_dist == "zipf")]
    sweep = {(0.003, 0.1), (0.01, 0.1), (0.03, 0.1), (0.01, 0.05), (0.01, 0.2), (0.01, 0.3)}
    ctx = ctx[[(h, e) in sweep for h, e in zip(ctx.h, ctx.eps)]]
    if (ctx.N != 256).any():
        print("\nS4: context length")
        t = ctx.groupby(["N", "h", "eps"]).agg(n=("run", "size"), h_fit=("h_fit", "median"),
                                               eps_fit=("eps_fit", "median"), r2_law=("r2_law", "median"),
                                               bayes_wins=("bayes_wins", "mean"))
        print(t.round(4).to_string())
        beta, deltas, ratio = {}, {}, {}
        for N, g in t.groupby(level=0):
            g = g.droplevel(0)
            hs = [k for k in [(0.003, 0.1), (0.01, 0.1), (0.03, 0.1)] if k in g.index]
            if len(hs) == 3:
                beta[N] = np.polyfit(np.log([k[0] for k in hs]), np.log(g.loc[hs].h_fit), 1)[0]
            es = [k for k in [(0.01, 0.05), (0.01, 0.1), (0.01, 0.2), (0.01, 0.3)] if k in g.index]
            if len(es) == 4:
                deltas[N] = np.polyfit(logit(np.array([k[1] for k in es])), logit(g.loc[es].eps_fit), 1)[0]
            if (0.003, 0.1) in g.index:
                ratio[N] = g.loc[(0.003, 0.1)].h_fit
            print(f"  N={N}: law R² median {ctx[ctx.N == N].r2_law.median():.3f}; beta_N {beta.get(N, float('nan')):.3f}; "
                  f"delta_N {deltas.get(N, float('nan')):.3f}; h'(0.003, 0.1) {ratio.get(N, float('nan')):.4f}")
        if 512 in beta and 256 in beta and 512 in ratio:
            r512 = ratio[512] / ratio[256]
            sup = (r512 <= 0.77) and (beta[512] >= beta[256] + 0.1)
            ref = (beta[512] <= beta[256] + 0.03) and (r512 > 0.9)
            print(f"  H_N: h'(N=512)/h'(N=256) = {r512:.2f}; beta_512 - beta_256 = {beta[512] - beta[256]:+.3f} -> "
                  f"{'SUPPORTED' if sup else 'REFUTED' if ref else 'INCONCLUSIVE'}")
            if 192 in beta and 192 in ratio:
                print(f"  N=192: h' ratio {ratio[192] / ratio[256]:.2f} (H_N: >= 1.1); beta_192 - beta_256 = "
                      f"{beta[192] - beta[256]:+.3f} (H_N: <= 0)")

    # S5: task dimensions
    e5d = F[F.exp == "E5d"]
    if len(e5d):
        def variant(r):
            for k, d in (("V", 64), ("K", 16)):
                if r[k] != d:
                    return f"{k}={r[k]}"
            if r["key_dist"] != "zipf":
                return "uniform keys"
            return "uniform typos"
        e5d = e5d.assign(variant=e5d.apply(variant, axis=1))
        bm = base.groupby(["h", "eps"]).h_fit.median()
        rows = []
        for (v, h, e), g in e5d.groupby(["variant", "h", "eps"]):
            rows.append({"variant": v, "h": h, "eps": e, "n": len(g), "h_fit": g.h_fit.median(),
                         "h_base": bm.get((h, e), np.nan), "eps_fit": g.eps_fit.median(),
                         "r2_law": g.r2_law.median(), "r2_own": g["bfree.r2"].median(),
                         "bayes_wins": g.bayes_wins.mean()})
        t = pd.DataFrame(rows)
        t["h_ratio"] = t.h_fit / t.h_base
        print("\nS5: task dimensions (baseline = V64 K16 N256 Zipf structured, same pipeline)\n" + t.round(3).to_string())
        for v, g in t.groupby("variant"):
            ok = len(g) == 2 and (g.r2_law >= 0.75).all() and g.h_ratio.between(1 / 1.5, 1.5).all()
            print(f"  {v}: {'constants TRANSFER' if ok else 'constants do not transfer'} "
                  f"(law R² {', '.join(f'{x:.2f}' for x in g.r2_law)}; h' ratio {', '.join(f'{x:.2f}' for x in g.h_ratio)})")

    # S6: RoPE
    rope = F[(F.pe == "rope") & std]
    if rope.exp.eq("E5r").any():
        t = setting_table(rope)
        alibi = setting_table(base)
        t["h_over_h_rope"] = t.h_fit / t.index.get_level_values(0)
        t["h_over_h_alibi"] = alibi.h_fit.reindex(t.index) / t.index.get_level_values(0)
        print("\nS6: RoPE grid\n" + t.round(3).to_string())
        sm = t[t.h_fit > 0]
        lh, yh = np.log(sm.index.get_level_values(0)), np.log(sm.h_fit)
        le, ye = logit(sm.index.get_level_values(1).to_numpy()), logit(sm.eps_fit.to_numpy())
        bh, ah = np.polyfit(lh, yh, 1)
        de, ge = np.polyfit(le, ye, 1)
        r2h = 1 - ((yh - (ah + bh * lh)) ** 2).mean() / yh.var()
        r2e = 1 - ((ye - (ge + de * le)) ** 2).mean() / ye.var()
        print(f"  RoPE law: h' = {math.exp(ah):.3f} h^{bh:.3f} (R² {r2h:.3f});  logit eps' = {ge:.3f} + {de:.3f} logit eps "
              f"(R² {r2e:.3f})")
        # leave-one-setting-out with RoPE's own constants
        rr = rope.assign(a=rope.a_fit, b=rope.b_fit)
        sc = []
        for (h, e), te in rr.groupby(["h", "eps"]):
            tr = rr[~((rr.h == h) & (rr.eps == e))]
            s = tr.groupby(["h", "eps"])[["h_fit", "eps_fit", "a", "b"]].median().reset_index()
            s = s[s.h_fit > 0]
            b1, a1 = np.polyfit(np.log(s.h), np.log(s.h_fit), 1)
            d1, g1 = np.polyfit(logit(s.eps), logit(s.eps_fit), 1)
            hp = math.exp(a1 + b1 * math.log(h))
            ep = 1 / (1 + math.exp(-(g1 + d1 * logit(e))))
            pred = s.a.median() + s.b.median() * Lay(4, False, 64).bayes([hp], [ep])[0]
            for r in te.run:
                y = pd.read_parquet(os.path.join(a.runs, r, "probes.parquet")).set_index(["n_c", "n_w", "G"]).loc[CELLS]
                sc.append(r2(pred, y.lo_model.to_numpy()))
        lo = float(np.median(sc)) if sc else float("nan")
        n_lower = int((t.h_over_h_rope < t.h_over_h_alibi).sum())
        print(f"  RoPE leave-one-setting-out median R² {lo:.3f}; RoPE h'/h below ALiBi at {n_lower}/{len(t)} settings")
        ok = r2h >= 0.8 and r2e >= 0.8 and lo >= 0.75
        print(f"  S6: {'PASS (same form)' if ok else 'FAIL'}  [h' law R² {r2h:.2f}, eps' law R² {r2e:.2f}, LOSO {lo:.2f}]"
              f"; frozen ALiBi law on RoPE models: R² median {rope.r2_law.median():.3f}")
    F.to_csv(os.path.join(a.out, "law_scope_all.csv"), index=False)


if __name__ == "__main__":
    main()
