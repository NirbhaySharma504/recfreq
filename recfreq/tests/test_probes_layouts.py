"""Probe layouts: tau never in the background, copies where and what they should be, Lay distances match."""
import itertools

import numpy as np
import pytest
import torch

from recfreq.data import TaskConfig
from recfreq.layouts import CELLS, Lay
from recfreq.probes import probe_batch


@pytest.mark.parametrize("sp,ws", [(2, 1), (4, 1), (8, 1), (4, -1)])
def test_probe_layout(sp, ws):
    task = TaskConfig(K=16, V=64, n_pairs=256)
    lay = Lay(sp, ws == -1)
    g = torch.Generator().manual_seed(0)
    for k, (n_c, n_w, G) in enumerate(CELLS):
        if k % 7:
            continue
        tok, keys, vals, c, w = probe_batch(task, n_c, n_w, G, 4, g, "cpu", sp=sp, w_shift=ws)
        q = task.n_pairs - 1
        for i in range(4):
            tau = int(keys[i, q])
            pos = (keys[i] == tau).nonzero().flatten().tolist()
            assert len(pos) == n_c + n_w + 1 and pos[-1] == q
            c_pos, w_pos = pos[:n_c], pos[n_c:n_c + n_w]
            assert (vals[i, c_pos] == c[i]).all() and (vals[i, w_pos] == w[i]).all()
            assert int(w[i]) == (int(c[i]) + ws) % task.V
            c_d, w_d = lay.dists[k]
            assert sorted(q - p for p in c_pos) == sorted(c_d.tolist())
            assert sorted(q - p for p in w_pos) == sorted(w_d.tolist())
            # token layout: key token at 1+2t, value token at 2+2t
            assert int(tok[i, 1 + 2 * q]) == tau + 1


def test_query_value_not_visible_to_prediction():
    """The logits that predict the query value come from the key position, before the value token."""
    task = TaskConfig()
    q = task.n_pairs - 1
    assert 1 + 2 * q < 2 + 2 * q == task.seq_len - 1
