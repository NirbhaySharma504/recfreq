"""Exactness of the mechanism tools on real checkpoints (skipped if the checkpoints are not present):
logit-difference decomposition, the rebuilt path-ablation forward pass, and the score decomposition."""
import itertools
import os

import pytest
import torch

RUNS = ["results/runs/E1_h0.01_e0.1_L2_alibi_learn_attn_s0", "results/runs/E2a_h0.01_e0.1_L2_rope_attn_s0"]
pytestmark = pytest.mark.skipif(not all(os.path.exists(os.path.join(r, "ckpt.pt")) for r in RUNS),
                                reason="checkpoints not available")


@pytest.mark.parametrize("run", RUNS)
def test_decompositions_exact(run):
    from recfreq.mech import decompose, load_model
    from recfreq.paths import forward_paths
    from recfreq.probes import probe_batch
    from recfreq.run5 import decompose_scores
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, task = load_model(run, dev)
    g = torch.Generator(device=dev).manual_seed(0)
    tok, keys, vals, c, w = probe_batch(task, 4, 2, 16, 8, g, dev)
    qp = task.n_pairs - 1
    qi, lo_k = 1 + 2 * qp, 1 + task.K
    with torch.no_grad():
        logits = model(tok)
        du = model.unembed.weight[lo_k + c] - model.unembed.weight[lo_k + w]
        parts, attn_q, _, _ = decompose(model, tok, qi, du)
        ar = torch.arange(8, device=dev)
        L = logits[ar, qi, lo_k + c] - logits[ar, qi, lo_k + w]
        assert (sum(parts.values()) - L).abs().max() < 1e-3
        assert (forward_paths(model, tok) - logits).abs().max() < 1e-3
        w_pos = [qp - 4 * j for j in range(1, 3)]
        c_pos = [qp - 4 * 2 - 16 - 4 * i for i in range(4)]
        vpos = [2 + 2 * p for p in c_pos + w_pos]
        dec = decompose_scores(model, tok, [0, 1], qi, vpos)
        _, a1, _ = forward_paths(model, tok, return_attn=True)
        for h in (0, 1):
            sc = dec[h]["content"] + dec[h]["dist"]
            full = torch.softmax(sc, 1)  # relative weights among the copies
            A = a1[:, h, qi][:, vpos]
            assert torch.allclose(full, A / A.sum(1, keepdim=True), atol=1e-3)
            kside = sum(dec[h]["kside"].values())
            assert torch.allclose(kside, dec[h]["content"], atol=1e-3)
