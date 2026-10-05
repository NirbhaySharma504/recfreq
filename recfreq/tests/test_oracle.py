"""Oracle vs. brute-force enumeration over all hidden paths of all keys."""
import itertools
import math

import numpy as np
import pytest
import torch

from recfreq.data import TaskConfig, sample_batch
from recfreq.oracle import bayes_predictive, critical_gap


def brute_force(cfg: TaskConfig, keys, vals):
    """Enumerate hidden values z[k, t] for every key and step: exact P(y_t | y_<t)."""
    K, V, N, h, eps = cfg.K, cfg.V, len(keys), cfg.h, cfg.eps

    def trans(a, b):  # one-step transition z(t-1)=a -> z(t)=b
        return (1 - h) * (a == b) + h / V

    def emis(z, y):
        if cfg.typo == "structured":
            return (1 - eps) * (y == z) + eps * (y == (z + 1) % V)
        return (1 - eps) * (y == z) + eps * (y != z) / (V - 1)

    # forward over the joint state (z_0..z_{K-1}) at every step
    states = list(itertools.product(range(V), repeat=K))
    alpha = np.array([V ** -K] * len(states))
    T = np.array([[np.prod([trans(a[k], b[k]) for k in range(K)]) for b in states] for a in states])
    out = np.zeros((N, V))
    for t in range(N):
        if t > 0:
            alpha = alpha @ T
        k = keys[t]
        for y in range(V):
            out[t, y] = sum(alpha[i] * emis(s[k], y) for i, s in enumerate(states)) / alpha.sum()
        alpha = alpha * np.array([emis(s[k], vals[t]) for s in states])
    return out


@pytest.mark.parametrize("typo", ["structured", "uniform"])
@pytest.mark.parametrize("h,eps", [(0.0, 0.0), (0.0, 0.2), (0.15, 0.0), (0.1, 0.15), (0.3, 0.3)])
def test_oracle_matches_brute_force(typo, h, eps):
    cfg = TaskConfig(K=2, V=3, n_pairs=7, h=h, eps=eps, typo=typo, key_dist="uniform")
    g = torch.Generator().manual_seed(0)
    b = sample_batch(cfg, 4, g, "cpu")
    pred = bayes_predictive(cfg, b["keys"], b["vals"]).numpy()
    for i in range(4):
        bf = brute_force(cfg, b["keys"][i].tolist(), b["vals"][i].tolist())
        np.testing.assert_allclose(pred[i], bf, atol=1e-9)


def test_counting_limit():
    """h = 0, uniform typos: after n copies of the same value it dominates more and more."""
    cfg = TaskConfig(K=1, V=8, n_pairs=6, h=0.0, eps=0.2, typo="uniform")
    keys = torch.zeros(1, 6, dtype=torch.long)
    vals = torch.tensor([[3, 3, 3, 3, 3, 3]])
    p = bayes_predictive(cfg, keys, vals)[0, :, 3]
    assert torch.all(p[1:] > p[:-1])


def test_recency_limit():
    """eps = 0: the last observed value is the hidden value unless a switch happened."""
    cfg = TaskConfig(K=1, V=8, n_pairs=3, h=0.1, eps=0.0)
    keys = torch.zeros(1, 3, dtype=torch.long)
    vals = torch.tensor([[2, 2, 5]])
    p = bayes_predictive(cfg, keys, vals)
    # predictive for t=2 is concentrated on 2, but y_2=5 is still possible via a switch
    assert p[0, 2].argmax() == 2
    assert p[0, 2, 5] > 0


def test_critical_gap_matches_plan_table():
    # plan §4.3 table, V = 64, eps = 0.1 (formula column)
    assert round(critical_gap(64, 1e-3, 0.1, 2)) == 582
    assert round(critical_gap(64, 1e-2, 0.1, 1)) == 208
    assert math.isinf(critical_gap(64, 0.0, 0.1, 1))


def test_layouts_match_fit_rules_and_oracle():
    """Lay(4) must equal fit_rules exactly; every layout's numpy Bayes must equal the torch oracle."""
    import torch
    from recfreq import fit_rules
    from recfreq.data import TaskConfig
    from recfreq.layouts import CELLS, Lay
    from recfreq.probes import key_predictive
    hs, es = [0.0, 0.003, 0.03], [0.05, 0.1, 0.3]
    assert np.allclose(Lay(4).bayes(hs, es), fit_rules.bayes_lo_grid(hs, es), atol=1e-12)
    for sp, swap in [(2, False), (8, False), (4, True)]:
        lay = Lay(sp, swap)
        for h, e in [(0.01, 0.1), (0.03, 0.2)]:
            ours = lay.bayes([h], [e])[0]
            task = TaskConfig(h=h, eps=e)
            for k in range(0, len(CELLS), 9):
                c_d, w_d = lay.dists[k]
                q = 255
                pos = sorted([int(q - d) for d in c_d] + [int(q - d) for d in w_d])
                c_val, w_val = (lay.yc, lay.yw)
                obs = torch.tensor([[c_val if (q - p) in set(c_d) else w_val for p in pos]])
                pred = key_predictive(task, pos, obs, q)[0]
                assert abs(float(pred[c_val].log() - pred[w_val].log()) - ours[k]) < 1e-9
