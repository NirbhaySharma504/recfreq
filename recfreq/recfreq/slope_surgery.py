"""Plan §19.10 R5-P3: rescale the ALiBi slopes of all layer-1 heads (x0.5, x2) on trained checkpoints and read the
conflict-probe log-odds at n_c = 2, n_w 1-6, G in the standard and wide windows. Eval only; writes surgery.parquet
into each run dir (columns: cond, n_w, G, lo_model).

python -m recfreq.slope_surgery results/runs --match E6_ --seeds 0
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os

import pandas as pd
import torch

from .mech import load_model
from .probes import GAPS, probe_batch

SCALES = {"base": 1.0, "x0.5": 0.5, "x2.0": 2.0}


@torch.no_grad()
def surface(model, task, device, fills=128, seed=555):
    g = torch.Generator(device=device).manual_seed(seed)
    qtok = 1 + 2 * (task.n_pairs - 1)
    rows = []
    for n_w in range(1, 7):
        for G in GAPS + (192, 224):
            if 4 * n_w + G + 4 > task.n_pairs - 1:
                continue
            tok, keys, vals, c, w = probe_batch(task, 2, n_w, G, fills, g, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lg = model(tok)[:, qtok, 1 + task.K: 1 + task.K + task.V].float()
            lp = lg.log_softmax(-1)
            ar = torch.arange(fills, device=device)
            rows.append({"n_w": n_w, "G": G, "lo_model": (lp[ar, c] - lp[ar, w]).mean().item()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--match", default="E6_")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
        name = os.path.basename(d)
        if not any(name.startswith(m) for m in a.match.split(",")):
            continue
        if not (os.path.exists(os.path.join(d, "ckpt.pt")) and os.path.exists(os.path.join(d, "done.json"))):
            continue
        args = json.load(open(os.path.join(d, "done.json")))["args"]
        if str(args["seed"]) not in a.seeds.split(",") or args.get("pe") != "alibi_learn" or args["layers"] != 2:
            continue
        if os.path.exists(os.path.join(d, "surgery.parquet")):
            continue
        model, task = load_model(d, dev)
        out = []
        for cond, s in SCALES.items():
            m2 = copy.deepcopy(model)
            for h in range(m2.cfg.n_heads):
                m2.blocks[1].attn.log_slope.data[h] += math.log(s)
            out.append(surface(m2, task, dev).assign(cond=cond))
            del m2
        pd.concat(out).to_parquet(os.path.join(d, "surgery.parquet"))
        print("done", name, flush=True)


if __name__ == "__main__":
    main()
