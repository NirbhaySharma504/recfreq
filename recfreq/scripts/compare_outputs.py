"""Compare two verify_out/ directories CSV by CSV (e.g. the committed one against a fresh reproduce_analysis.sh run).

python scripts/compare_outputs.py REF_DIR NEW_DIR [--rtol 1e-6]

For each CSV present in both: same columns and row count, and every numeric value equal within rtol (relative) or
1e-9 (absolute); text columns must match exactly. Prints one line per file and a final count.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("new")
ap.add_argument("--rtol", type=float, default=1e-6)
a = ap.parse_args()

same = diff = 0
for f in sorted(os.listdir(a.new)):
    if not f.endswith(".csv") or not os.path.exists(os.path.join(a.ref, f)):
        continue
    R, N = pd.read_csv(os.path.join(a.ref, f)), pd.read_csv(os.path.join(a.new, f))
    if "run" in R.columns and "run" in N.columns:  # row order can differ (parallel fits, caches): match by run
        extra = len(set(N.run) - set(R.run))
        if extra and set(R.run) <= set(N.run):
            print(f"note  {f}: {extra} runs added since the reference was made; comparing the {len(R)} shared runs")
            N = N[N.run.isin(R.run)]
        key = ["run"] + [c for c in R.columns if c != "run" and not pd.api.types.is_numeric_dtype(R[c])]
        R, N = R.sort_values(key).reset_index(drop=True), N.sort_values(key).reset_index(drop=True)
    problem = ""
    if list(R.columns) != list(N.columns) or len(R) != len(N):
        problem = f"shape/columns differ: ref {R.shape}, new {N.shape}"
    else:
        worst = 0.0
        for c in R.columns:
            r, n = R[c], N[c]
            if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(n):
                rv, nv = r.to_numpy(float), n.to_numpy(float)
                ok = np.isclose(rv, nv, rtol=a.rtol, atol=1e-9, equal_nan=True)
                if not ok.all():
                    rel = np.nanmax(np.abs(rv - nv)[~ok] / np.maximum(np.abs(rv[~ok]), 1e-12))
                    worst = max(worst, rel)
                    problem = f"column {c}: {int((~ok).sum())} values differ (max rel {rel:.2g})"
            elif not (r.astype(str) == n.astype(str)).all():
                problem = f"text column {c} differs"
    if problem:
        diff += 1
        print(f"DIFF  {f}: {problem}")
    else:
        same += 1
        print(f"same  {f}")
print(f"\n{same} identical, {diff} different")
sys.exit(1 if diff else 0)
