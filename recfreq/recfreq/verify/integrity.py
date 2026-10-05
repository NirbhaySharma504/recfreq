"""Audit 2.1: run integrity. Convergence, gate, seeds, eval sets.

python -m recfreq.verify.integrity results/runs --out verify_out
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--out", default="verify_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(a.runs_dir, "E*_*"))):
        name = os.path.basename(d)
        if not os.path.exists(os.path.join(d, "done.json")):
            rows.append({"run": name, "complete": False})
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        log = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
        steps = cfg["steps"]
        ex = {r["step"]: r["excess"] for r in log}
        last = ex[max(ex)]
        at75 = ex[min(ex, key=lambda s: abs(s - 0.75 * steps))]
        rows.append({"run": name, "exp": name.split("_")[0], "complete": True, "steps": steps, "seed": cfg["seed"],
                     "h": cfg["h"], "eps": cfg["eps"], "excess_75": at75, "excess_final": last,
                     "rel_improve_last25": (at75 - last) / max(at75, 1e-9), "abs_improve_last25": at75 - last,
                     "acc_rep": json.load(open(os.path.join(d, "done.json")))["final"]["acc_rep"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "integrity.csv"), index=False)
    print(f"runs: {len(df)}, complete: {int(df.complete.sum())}")
    still = df[(df.rel_improve_last25 > 0.10) & (df.abs_improve_last25 > 0.002)]
    print(f"still improving in the last 25% of training (>10% relative and >0.002 nats): {len(still)}")
    if len(still):
        print(still[["run", "excess_75", "excess_final"]].to_string(index=False))
    print("median relative improvement over the last 25% of steps, by experiment:")
    print(df.groupby("exp").rel_improve_last25.median().round(3).to_string())
    g = df[df.exp == "E0"]
    print(f"gate runs: {len(g)}; repeat-sighting accuracy {g.acc_rep.tolist()}; final excess {g.excess_final.round(6).tolist()}")
    dup = df.groupby(["exp", "h", "eps", "seed"]).size()
    print("configs sharing a seed within an experiment (expected: one per architecture):",
          int((dup > 1).sum()), "groups")


if __name__ == "__main__":
    main()
