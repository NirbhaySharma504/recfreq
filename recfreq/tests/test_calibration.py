"""The generator and the oracle must agree: oracle probabilities are calibrated on generator samples,
and the mean log loss equals the oracle's expected entropy (both within Monte-Carlo error)."""
import numpy as np
import pytest
import torch

from recfreq.data import TaskConfig, sample_batch
from recfreq.oracle import bayes_predictive


def calibration(task: TaskConfig, B: int = 600, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    b = sample_batch(task, B, g, "cpu")
    pred = bayes_predictive(task, b["keys"], b["vals"]).numpy()  # (B, N, V)
    y = b["vals"].numpy()
    p_flat = pred.reshape(-1)
    hit = np.zeros_like(pred)
    np.put_along_axis(hit, y[..., None], 1.0, axis=2)
    h_flat = hit.reshape(-1)
    bins = np.minimum((p_flat * 20).astype(int), 19)
    cnt = np.bincount(bins, minlength=20)
    mean_p = np.bincount(bins, weights=p_flat, minlength=20) / np.maximum(cnt, 1)
    freq = np.bincount(bins, weights=h_flat, minlength=20) / np.maximum(cnt, 1)
    ece = float((cnt * np.abs(mean_p - freq)).sum() / cnt.sum())
    p_true = np.take_along_axis(pred, y[..., None], 2)[..., 0]
    nll = -np.log(p_true)
    ent = -(pred * np.log(np.clip(pred, 1e-300, None))).sum(-1)
    diff = nll - ent  # zero-mean if the oracle is the true conditional distribution
    z = diff.mean() / (diff.std() / np.sqrt(diff.size))
    return ece, float(z), float(nll.mean())


@pytest.mark.parametrize("h,eps,typo", [(0.01, 0.1, "structured"), (0.03, 0.2, "structured"),
                                       (0.003, 0.05, "structured"), (0.0, 0.2, "uniform")])
def test_oracle_calibrated_on_generator(h, eps, typo):
    task = TaskConfig(h=h, eps=eps, typo=typo)
    ece, z, _ = calibration(task)
    assert ece < 0.01, f"ECE {ece:.4f}"
    assert abs(z) < 4, f"mean NLL differs from expected entropy (z = {z:.2f})"
