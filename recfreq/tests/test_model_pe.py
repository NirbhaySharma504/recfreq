"""Positional encodings: ALiBi bias sign and distances; RoPE scores depend only on relative position."""
import math

import torch

from recfreq.model import Attention, ModelConfig


def test_alibi_bias_values():
    cfg = ModelConfig(vocab=10, max_len=8, n_heads=4, pe="alibi_learn")
    at = Attention(cfg, 0)
    b = at.bias(6, "cpu")[0]
    sl = at.slopes().detach()
    for h in range(4):
        for i in range(6):
            for j in range(6):
                if j > i:
                    assert b[h, i, j] == float("-inf")
                else:
                    assert torch.isclose(b[h, i, j], -sl[h] * (i - j))


def test_rope_relative_position_invariance():
    cfg = ModelConfig(vocab=10, max_len=64, n_heads=4, pe="rope")
    at = Attention(cfg, 0)
    torch.manual_seed(0)
    q = torch.randn(1, 4, 1, at.dh).expand(1, 4, 40, at.dh).clone()
    k = torch.randn(1, 4, 1, at.dh).expand(1, 4, 40, at.dh).clone()
    qr, kr = at.rope(q), at.rope(k)
    s = (qr @ kr.transpose(-1, -2))[0, 0]  # identical content at every position
    for d in (0, 3, 11):
        vals = torch.stack([s[i, i - d] for i in range(d, 40)])
        assert torch.allclose(vals, vals[0].expand_as(vals), atol=1e-4)
