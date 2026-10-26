"""Plan §19 C1: constrained-Bayes surrogates. Hypothesis H-lin: the models run the Bayes filter, but their sense of
time is distorted, because an ALiBi head can only make the log-odds "the old value is still current" linear in distance.

The surrogate is the exact forward filter of oracle.py with the survival s(g) = (1-h)^g replaced by a parametric curve:
  bayes:  s = (1 - h~)^g                (contains the truth; sanity check, must recover h~ = h)
  lin:    s = sigmoid(c0 - lam * g)     (one ALiBi head: attention to the copy vs a sink, linear in distance)
  loglin: s = sigmoid(c0 - lam * log g) (log-odds linear in log distance, the shape Bayes has for small h g)
The curve's parameters and the typo rate eps~ are fitted by cross-entropy on ordinary training data from the
generator (the models' training loss), then the surrogate is run on the 112 conflict probes and fitted with
layouts.BayesFree, exactly as the transformers are.

python -m recfreq.surrogate --out verify_out --procs 6
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

KINDS = ("bayes", "lin", "loglin")


def surv_t(kind, th, g):
    if kind == "bayes":
        return torch.exp(g * torch.log1p(-torch.sigmoid(th[0])))
    if kind == "lin":
        return torch.sigmoid(th[0] - torch.exp(th[1]) * g)
    if kind == "loglin":
        return torch.sigmoid(th[0] - torch.exp(th[1]) * torch.log(g))
    raise ValueError(kind)


def surv_np(kind, th):
    th = [float(t) for t in th]
    sig = lambda x: 1 / (1 + np.exp(-x))
    if kind == "bayes":
        return lambda d: (1 - sig(th[0])) ** np.asarray(d, float)
    if kind == "lin":
        return lambda d: sig(th[0] - math.exp(th[1]) * np.asarray(d, float))
    return lambda d: sig(th[0] - math.exp(th[1]) * np.log(np.asarray(d, float)))


def filter_ce(cfg: TaskConfig, keys, vals, kind, th, te):
    """Mean CE over all value positions of the filter with survival curve (kind, th) and typo rate sigmoid(te)."""
    B, N = keys.shape
    V, K = cfg.V, cfg.K
    eps = torch.sigmoid(te)
    eye = torch.eye(V, dtype=torch.float64)
    if cfg.typo == "structured":
        E = (1 - eps) * eye + eps * torch.roll(eye, 1, 1)
    else:
        E = (1 - eps) * eye + eps * (1 - eye) / (V - 1)
    belief = torch.full((B, K, V), 1.0 / V, dtype=torch.float64)
    last = torch.full((B, K), -1, dtype=torch.long)
    ar = torch.arange(B)
    ce = torch.zeros((), dtype=torch.float64)
    for t in range(N):
        k = keys[:, t]
        b = belief[ar, k]
        seen = last[ar, k] >= 0
        gap = (t - last[ar, k]).clamp_min(1).to(torch.float64)
        s = torch.where(seen, surv_t(kind, th, gap), torch.zeros_like(gap)).unsqueeze(1)
        prior = s * b + (1 - s) / V
        lik = E[:, vals[:, t]].T  # (B, V): P(y_t | z)
        p = (prior * lik).sum(1)
        ce = ce - torch.log(p.clamp_min(1e-300)).sum()
        post = prior * lik
        belief = belief.clone()
        belief[ar, k] = post / post.sum(1, keepdim=True)
        last = last.clone()
        last[ar, k] = t
    return ce / (B * N)


def init(kind, h):
    if kind == "bayes":
        return [math.log(h / (1 - h))]
    if kind == "lin":
        return [math.log(1 / (h * 8)), math.log(h * 2)]  # rough: log-odds ~ -log(h g) near g = 8
    return [-math.log(h), 0.0]


def fit_one(job):
    torch.set_num_threads(1)
    cfg = TaskConfig(**job["task"])
    g = torch.Generator().manual_seed(job.get("seed", 1234))
    bt = sample_batch(cfg, job.get("B", 512), g, "cpu")
    out = dict(job["task"])
    lay = Lay(4, False, cfg.V)
    bf = BayesFree(lay, cfg.typo)
    for kind in job.get("kinds", KINDS):
        th = torch.tensor(init(kind, cfg.h), dtype=torch.float64, requires_grad=True)
        te = torch.tensor(math.log(cfg.eps / (1 - cfg.eps)), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th, te], lr=1, max_iter=60, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = filter_ce(cfg, bt["keys"], bt["vals"], kind, th, te)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            ce = float(filter_ce(cfg, bt["keys"], bt["vals"], kind, th, te))
        e_hat = float(torch.sigmoid(te.detach()))
        y = lay.bayes([0.0], [e_hat], cfg.typo, surv=surv_np(kind, th.detach().numpy()))[0]
        p = bf.fit(y, np.ones(len(CELLS), bool))
        out.update({f"{kind}.ce": ce, f"{kind}.th": json.dumps([round(float(t), 5) for t in th.detach()]),
                    f"{kind}.eps_hat": e_hat, f"{kind}.h_fit": p["h"], f"{kind}.eps_fit": p["eps"],
                    f"{kind}.a_fit": p["a"], f"{kind}.b_fit": p["b"]})
        np.save(os.path.join(job["out"], "surfaces", f"{job['name']}_{kind}.npy"), y)
    with torch.no_grad():  # the true filter's CE on the same batch
        out["true.ce"] = float(filter_ce(cfg, bt["keys"], bt["vals"], "bayes",
                                         torch.tensor([math.log(cfg.h / (1 - cfg.h))], dtype=torch.float64),
                                         torch.tensor(math.log(cfg.eps / (1 - cfg.eps)), dtype=torch.float64)))
    out["name"] = job["name"]
    return out


def settings():
    """Every ALiBi L2 setting with trained models (run 1-6) or planned (E6)."""
    pts = [(h, e) for h in (0.003, 0.01, 0.03) for e in (0.05, 0.1, 0.2, 0.3)]
    pts += [(0.005, 0.07), (0.02, 0.15), (0.007, 0.25), (0.001, 0.1), (0.06, 0.1), (0.01, 0.02), (0.01, 0.4)]
    e6 = [(0.0003, 0.1), (0.002, 0.1), (0.045, 0.1), (0.1, 0.1), (0.15, 0.1),
          (0.0003, 0.2), (0.001, 0.2), (0.002, 0.2), (0.045, 0.2), (0.06, 0.2), (0.1, 0.2), (0.15, 0.2),
          (0.01, 0.01), (0.01, 0.35), (0.01, 0.45), (0.003, 0.01), (0.003, 0.02), (0.003, 0.35), (0.003, 0.4),
          (0.003, 0.45), (0.03, 0.01), (0.03, 0.02), (0.03, 0.35), (0.03, 0.4), (0.03, 0.45)]
    out = [("base", dict(h=h, eps=e)) for h, e in pts] + [("E6", dict(h=h, eps=e)) for h, e in e6]
    for N in (192, 512):
        for h in (0.003, 0.01, 0.03):
            out.append(("E5n", dict(h=h, eps=0.1, n_pairs=N)))
    for h, e in ((0.01, 0.1), (0.03, 0.2)):
        for extra in (dict(V=32), dict(V=128), dict(K=8), dict(K=32), dict(key_dist="uniform")):
            out.append(("E5d", dict(h=h, eps=e, **extra)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--B", type=int, default=512)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "surfaces"), exist_ok=True)
    jobs = []
    for grp, t in settings():
        if a.only and grp not in a.only.split(","):
            continue
        name = "_".join(f"{k}{v}" for k, v in t.items())
        jobs.append({"task": t, "name": name, "group": grp, "out": a.out, "B": a.B})
    with ProcessPoolExecutor(a.procs) as ex:
        rows = []
        for j, r in zip(jobs, ex.map(fit_one, jobs)):
            r["group"] = j["group"]
            rows.append(r)
            print(r["name"], {k: round(v, 4) for k, v in r.items() if k.endswith(("h_fit", "eps_fit", "b_fit"))}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "surrogate.csv"), index=False)


if __name__ == "__main__":
    main()
