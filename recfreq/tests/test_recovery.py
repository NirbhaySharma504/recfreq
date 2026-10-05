"""The fitting families must recover known parameters from noisy synthetic surfaces."""
import math

import numpy as np
import pytest

from recfreq.layouts import CELLS, LAY4, BayesFree, fit_counter, r2

NOISE = 0.04  # typical cell-mean noise of the stored probes (lo_model_sd / sqrt(256))


@pytest.fixture(scope="module")
def bf():
    return BayesFree(LAY4)


def bayes_in_range(h, e):
    """True if the Bayes log-odds change sign within the probed gaps for some (n_c >= 2, n_w)."""
    x = LAY4.bayes([h], [e])[0]
    cells = np.array(CELLS)
    for nw in (1, 2, 3, 4):
        m = (cells[:, 0] == 2) & (cells[:, 1] == nw)
        if x[m].max() > 0 > x[m].min():
            return True
    return False


def test_bayes_free_recovers_parameters(bf):
    rng = np.random.default_rng(0)
    rows = []
    for h in (0.003, 0.006, 0.01, 0.02, 0.03, 0.06, 0.1):
        for e in (0.05, 0.1, 0.2, 0.3):
            a, b = rng.uniform(-0.3, 0.3), rng.uniform(0.8, 1.6)
            clean = a + b * LAY4.bayes([h], [e])[0]
            y = clean + rng.normal(0, NOISE, len(CELLS))
            p = bf.fit(y, np.ones(len(y), bool))
            yh = BayesFree.predict(LAY4, p)
            rows.append((h, e, bayes_in_range(h, e), abs(math.log(p["h"] / h)), abs(math.log(p["eps"] / e)),
                         r2(yh, y) - r2(clean, y), np.sqrt(((yh - y) ** 2).mean())))
    rows = np.array(rows, dtype=float)
    inr = rows[rows[:, 2] == 1]
    # median relative error within 25% (|log ratio| < log 1.25) where the boundary is inside the probe range
    assert np.median(inr[:, 3]) < math.log(1.25), np.median(inr[:, 3])
    assert np.median(inr[:, 4]) < math.log(1.25), np.median(inr[:, 4])
    # the fit must reach the noise floor: as good as the true parameters, residual error close to the noise
    # (an absolute R² threshold is wrong for flat surfaces, where noise is a large share of the variance)
    assert rows[:, 5].min() > -0.005
    assert rows[:, 6].max() < 1.2 * NOISE


def test_counter_recovers_known_counter():
    rng = np.random.default_rng(1)
    for lam in (0.02, 0.1, 0.4):
        t = np.array([math.log(lam), 3.0, 0.3, math.log(0.5), 0.2])
        clean = LAY4.counter(t)
        y = clean + rng.normal(0, NOISE, len(CELLS))
        p = fit_counter(LAY4, y, np.ones(len(y), bool))
        assert r2(LAY4.counter(p), y) > r2(clean, y) - 0.005
        assert np.sqrt(((LAY4.counter(p) - y) ** 2).mean()) < 1.2 * NOISE
        assert abs(math.log(math.exp(p[0]) / lam)) < math.log(1.5)
