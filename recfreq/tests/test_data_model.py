import math

import torch

from recfreq.data import TaskConfig, first_occurrence, sample_batch
from recfreq.model import ModelConfig, Transformer
from recfreq.probes import probe_batch
from recfreq.train import value_logits


def test_switch_and_typo_rates():
    cfg = TaskConfig(K=4, V=32, n_pairs=200, h=0.05, eps=0.2)
    g = torch.Generator().manual_seed(0)
    b = sample_batch(cfg, 256, g, "cpu")
    assert abs(b["typo"].float().mean().item() - 0.2) < 0.01
    # structured typo: y = z + 1 exactly where flagged
    assert torch.equal(b["vals"][b["typo"]], (b["z"][b["typo"]] + 1) % 32)
    assert torch.equal(b["vals"][~b["typo"]], b["z"][~b["typo"]])


def test_hidden_switch_rate():
    from recfreq.data import hidden_values
    cfg = TaskConfig(K=4, V=1000, n_pairs=400, h=0.05)
    z = hidden_values(cfg, 64, torch.Generator().manual_seed(1), "cpu")
    changed = (z[:, :, 1:] != z[:, :, :-1]).float().mean().item()
    assert abs(changed - 0.05 * (1 - 1 / 1000)) < 0.005


def test_tokens_layout():
    cfg = TaskConfig(K=3, V=5, n_pairs=4)
    b = sample_batch(cfg, 2, torch.Generator().manual_seed(0), "cpu")
    t = b["tokens"]
    assert t.shape == (2, 9) and (t[:, 0] == 0).all()
    assert ((t[:, 1::2] >= 1) & (t[:, 1::2] <= 3)).all()
    assert ((t[:, 2::2] >= 4) & (t[:, 2::2] <= 8)).all()


def test_first_occurrence():
    keys = torch.tensor([[0, 1, 0, 2, 1]])
    assert first_occurrence(keys, 3).tolist() == [[True, True, False, True, False]]


def test_model_init_loss_and_causality():
    cfg = TaskConfig(K=4, V=16, n_pairs=20)
    for pe in ["alibi_learn", "alibi_fixed", "learned_abs", "nope", "rope"]:
        m = Transformer(ModelConfig(vocab=cfg.vocab, max_len=cfg.seq_len, n_layers=2, pe=pe, mlp=True))
        b = sample_batch(cfg, 8, torch.Generator().manual_seed(0), "cpu")
        lg = value_logits(cfg, m(b["tokens"]))
        ce = torch.nn.functional.cross_entropy(lg.transpose(1, 2), b["vals"]).item()
        assert abs(ce - math.log(16)) < 0.3
        # causal: changing the last token must not change earlier logits
        t2 = b["tokens"].clone()
        t2[:, -1] = 1
        assert torch.allclose(m(b["tokens"])[:, :-1], m(t2)[:, :-1], atol=1e-5)
        # manual attention path == SDPA path
        lg2, _ = m(b["tokens"], return_attn=True)
        assert torch.allclose(m(b["tokens"]), lg2, atol=1e-5)


def test_probe_layout():
    cfg = TaskConfig(K=16, V=64, n_pairs=256)
    tok, keys, vals, c, w = probe_batch(cfg, 4, 3, 32, 8, torch.Generator().manual_seed(0), "cpu")
    for i in range(8):
        tau = keys[i, -1]
        pos = (keys[i] == tau).nonzero().flatten().tolist()
        assert len(pos) == 4 + 3 + 1
        assert vals[i, pos[:4]].eq(c[i]).all() and vals[i, pos[4:7]].eq(w[i]).all()
        assert pos[4] - pos[3] == 32


def test_conflict_batch_target_is_bayes():
    from recfreq.oracle import bayes_predictive
    from recfreq.probes import conflict_batch, key_predictive
    cfg = TaskConfig(K=16, V=64, n_pairs=256, h=0.01, eps=0.1)
    g = torch.Generator().manual_seed(3)
    tok, keys, vals, c, w = probe_batch(cfg, 3, 2, 16, 6, g, "cpu")
    pos = (keys[0] == keys[0, -1]).nonzero().flatten().tolist()[:-1]
    kp = key_predictive(cfg, pos, vals[:, pos], cfg.n_pairs - 1)
    full = bayes_predictive(cfg, keys, vals)[:, -1]
    assert torch.allclose(kp, full, atol=1e-10)
    cb = conflict_batch(cfg, 10, g, "cpu")
    assert cb["tokens"].shape == (10, cfg.seq_len)
