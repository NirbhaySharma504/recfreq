"""Plan §19 C1 (round 2): attention surrogate of the copy heads, trained like the transformers.

Each of M heads looks at every earlier sighting j of the queried key (distance d_j in pairs) and at a sink:
    attention logit  -lam_m * d_j  for copies,  sig_m for the sink   (softmax over copies + sink)
and adds to the value logits
    gamma_m * sum_j a_mj * ( 1[y_j = v] + kp_m 1[y_j + 1 = v] + km_m 1[y_j - 1 = v] ).
Prediction = softmax over the V values of the summed logits. This is Prop. 1 with normalisation, i.e. the
measured mechanism (logit(c) - logit(w) = sum_m gamma_m (A_c - A_w)) with pure ALiBi distance scores.
Parameters are fitted by cross-entropy on ordinary generator data; then the surrogate is probed (112 cells) and
fitted with layouts.BayesFree like the transformers, and its single-sighting survival curve is read off.

python -m recfreq.satt --out verify_out --heads 1,3 --procs 7
"""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from .data import TaskConfig, sample_batch
from .layouts import CELLS, BayesFree, Lay
from .surrogate import settings

DISTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 248)


def unpack(th, M):
    t = th.view(M, 5)
    lam = torch.exp(t[:, 0])  # per pair
    return lam, t[:, 1], t[:, 2], t[:, 3], t[:, 4]


def logits_seq(th, M, keys, vals, V):
    """(B, N, V) value logits for every position of ordinary sequences."""
    B, N = keys.shape
    lam, sig, gam, kp, km = unpack(th, M)
    t = torch.arange(N)
    d = (t[:, None] - t[None, :]).to(torch.float32)  # (N, N) query t, copy j
    same = (keys[:, :, None] == keys[:, None, :]) & (t[:, None] > t[None, :])[None]  # (B, N, N)
    oh = torch.nn.functional.one_hot(vals, V).float()  # (B, N, V)
    ohp, ohm = oh.roll(1, -1), oh.roll(-1, -1)  # value y+1, y-1
    out = torch.zeros(B, N, V)
    for m in range(M):
        s = torch.where(same, -lam[m] * d, torch.tensor(-1e9))
        z = torch.cat([s, sig[m].expand(B, N, 1)], -1)
        a = torch.softmax(z, -1)[..., :N]
        out = out + gam[m] * (a @ oh + kp[m] * (a @ ohp) + km[m] * (a @ ohm))
    return out


def lo_layout(th, M, c_d, w_d, V):
    """log p(c) - log p(w) at a query with copies of c at distances c_d and of w = c+1 at w_d."""
    lam, sig, gam, kp, km = unpack(torch.as_tensor(th, dtype=torch.float64), M)
    d = torch.tensor(list(c_d) + list(w_d), dtype=torch.float64)
    y = torch.tensor([0] * len(c_d) + [1] * len(w_d))
    L = torch.zeros(V, dtype=torch.float64)
    for m in range(M):
        z = torch.cat([-lam[m] * d, sig[m].view(1)])
        a = torch.softmax(z, 0)[:-1]
        for aj, yj in zip(a, y):
            L[yj] += gam[m] * aj
            L[(yj + 1) % V] += gam[m] * kp[m] * aj
            L[(yj - 1) % V] += gam[m] * km[m] * aj
    lp = torch.log_softmax(L, 0)
    return float(lp[0] - lp[1])


def survival(th, M, V):
    out = []
    for d in DISTS:
        lam, sig, gam, kp, km = unpack(torch.as_tensor(th, dtype=torch.float64), M)
        L = torch.zeros(V, dtype=torch.float64)
        for m in range(M):
            a = torch.softmax(torch.stack([-lam[m] * d, sig[m]]), 0)[0]
            L[0] += gam[m] * a; L[1] += gam[m] * kp[m] * a; L[V - 1] += gam[m] * km[m] * a
        p = torch.softmax(L, 0)
        Mm = float(p[0] + p[1] + p[V - 1])
        out.append((Mm - 3 / V) / (1 - 3 / V))
    return out


def init(M, h):
    rows = []
    for m in range(M):
        lam = 0.05 * (4 ** m) * (h / 0.01) ** 0.4
        rows.append([math.log(lam), 0.0, 6.0, 0.1, 0.0])
    return torch.tensor(rows, dtype=torch.float32).flatten()


def fit_one(job):
    torch.set_num_threads(1)
    cfg = TaskConfig(**job["task"])
    M = job["M"]
    g = torch.Generator().manual_seed(4321)
    bt = sample_batch(cfg, job.get("B", 256), g, "cpu")
    th = init(M, cfg.h).requires_grad_(True)
    opt = torch.optim.Adam([th], lr=0.05)
    for it in range(job.get("iters", 400)):
        if it % 100 == 0 and it:  # fresh data every 100 steps; the final fit uses a fixed batch
            bt = sample_batch(cfg, job.get("B", 256), g, "cpu")
        opt.zero_grad()
        lg = logits_seq(th, M, bt["keys"], bt["vals"], cfg.V)
        loss = torch.nn.functional.cross_entropy(lg.transpose(1, 2), bt["vals"])
        loss.backward()
        opt.step()
    opt2 = torch.optim.LBFGS([th], lr=1, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt2.zero_grad()
        l = torch.nn.functional.cross_entropy(logits_seq(th, M, bt["keys"], bt["vals"], cfg.V).transpose(1, 2), bt["vals"])
        l.backward()
        return l
    opt2.step(closure)
    thd = th.detach().double()
    with torch.no_grad():
        ce = float(torch.nn.functional.cross_entropy(logits_seq(th, M, bt["keys"], bt["vals"], cfg.V).transpose(1, 2), bt["vals"]))
    lay = Lay(4, False, cfg.V)
    y = np.array([lo_layout(thd, M, c, w, cfg.V) for c, w in lay.dists])
    p = BayesFree(lay, cfg.typo).fit(y, np.ones(len(CELLS), bool))
    lam = torch.exp(thd.view(M, 5)[:, 0]).numpy()
    out = dict(job["task"])
    out.update({"name": job["name"], "group": job["group"], "M": M, "ce": ce, "th": json.dumps(np.round(thd.numpy(), 4).tolist()),
                "lam_min": float(lam.min()), "lam_max": float(lam.max()), "h_fit": p["h"], "eps_fit": p["eps"],
                "a_fit": p["a"], "b_fit": p["b"], "surv": json.dumps(np.round(survival(thd, M, cfg.V), 4).tolist())})
    np.save(os.path.join(job["out"], "satt_surfaces", f"{job['name']}_M{M}.npy"), y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--heads", default="1,3")
    ap.add_argument("--only", default="")
    ap.add_argument("--iters", type=int, default=400)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "satt_surfaces"), exist_ok=True)
    jobs = []
    for M in map(int, a.heads.split(",")):
        for grp, t in settings():
            if a.only and grp not in a.only.split(","):
                continue
            jobs.append({"task": t, "name": "_".join(f"{k}{v}" for k, v in t.items()), "group": grp, "out": a.out,
                         "M": M, "iters": a.iters})
    rows = []
    with ProcessPoolExecutor(a.procs) as ex:
        for r in ex.map(fit_one, jobs):
            rows.append(r)
            print(r["name"], "M", r["M"], {k: round(r[k], 4) for k in ("ce", "lam_min", "lam_max", "h_fit", "eps_fit", "b_fit")}, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, f"satt_{a.heads.replace(',', '')}.csv"), index=False)


if __name__ == "__main__":
    main()
