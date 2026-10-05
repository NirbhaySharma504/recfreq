"""Probe layouts and the candidate decision rules, for any copy spacing and for the swapped layout.

Lay(sp=4, swap=False) reproduces fit_rules exactly (same cells, distances and Bayes values).
V (number of values) defaults to 64; other V are used by the law-scope runs (plan §18).
swap=True: the recent value is w = c - 1, so the *old* copies are the typo-variant T(w) of the recent value.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import least_squares

from .probes import GAPS, NC, NW

V = 64
CELLS = list(itertools.product(NC, NW, GAPS))


class Lay:
    def __init__(self, sp: int = 4, swap: bool = False, V: int = V):
        self.sp, self.swap, self.V = sp, swap, V
        self.name = f"{'swap' if swap else 'sp'}{sp}"
        self.dists = []
        for n_c, n_w, G in CELLS:
            w_d = np.array([sp * j for j in range(1, n_w + 1)], float)
            c_d = np.array([sp * n_w + G + sp * i for i in range(n_c)], float)
            self.dists.append((c_d, w_d))
        self.DC = np.full((len(CELLS), max(NC)), np.inf)
        self.DW = np.full((len(CELLS), max(NW)), np.inf)
        for k, (c, w) in enumerate(self.dists):
            self.DC[k, : len(c)] = c
            self.DW[k, : len(w)] = w
        # value labels: normal c=0, w=1=T(c); swapped c=1, w=0 so that c = T(w)
        self.yc, self.yw = (1, 0) if swap else (0, 1)

    # ---- exact Bayes -----------------------------------------------------------------------------------------
    def bayes(self, hs, es, typo: str = "structured") -> np.ndarray:
        hs, es = np.asarray(hs, float)[:, None], np.asarray(es, float)[:, None]
        P, V = hs.shape[0], self.V

        def em(y):
            col = np.zeros(V); col[y] = 1.0
            if typo == "structured":
                sh = np.zeros(V); sh[(y - 1) % V] = 1.0
                return (1 - es) * col[None] + es * sh[None]
            return (1 - es) * col[None] + es * (1 - col[None]) / (V - 1)

        Ec, Ew = em(self.yc), em(self.yw)
        out = np.empty((P, len(CELLS)))
        for k, (c_d, w_d) in enumerate(self.dists):
            obs = sorted([(d, 0) for d in c_d] + [(d, 1) for d in w_d], key=lambda t: -t[0])
            b = np.full((P, V), 1.0 / V)
            prev = None
            for d, which in obs:
                if prev is not None:
                    s = (1 - hs) ** (prev - d)
                    b = s * b + (1 - s) / V
                b = b * (Ec if which == 0 else Ew)
                b /= b.sum(1, keepdims=True)
                prev = d
            s = (1 - hs) ** prev
            prior = s * b + (1 - s) / V
            out[:, k] = np.log(np.clip((prior * Ec).sum(1), 1e-300, None)) - \
                np.log(np.clip((prior * Ew).sum(1), 1e-300, None))
        return out

    # ---- counters -------------------------------------------------------------------------------------------
    def _norm_counter(self, lam, gamma, kappa, z0):
        sc, sw = np.exp(-lam * self.DC).sum(1), np.exp(-lam * self.DW).sum(1)
        # kappa: a copied typo-variant also votes for the value it is a variant of
        num = (1 - kappa) * sc - sw if self.swap else sc - (1 - kappa) * sw
        return gamma * num / (sc + sw + z0)

    def counter(self, t):
        lp, gamma, kappa, lz, delta = t
        return delta + self._norm_counter(math.exp(lp), gamma, kappa, math.exp(lz))

    def counter2(self, t):
        """Sum of two normalized exponential counters with their own decay, gain and background mass."""
        l1, l2, g1, g2, z1, z2, kappa, delta = t
        return delta + self._norm_counter(math.exp(l1), g1, kappa, math.exp(z1)) + \
            self._norm_counter(math.exp(l2), g2, kappa, math.exp(z2))

    def additive(self, t):
        lc, lw, gc, gw, delta = t
        return delta + gc * np.exp(-math.exp(lc) * self.DC).sum(1) - gw * np.exp(-math.exp(lw) * self.DW).sum(1)


LAY4 = Lay(4, False)


def _ls(fun, y, mask, inits, lo, hi):
    best = None
    for x0 in inits:
        try:
            r = least_squares(lambda t: (fun(t) - y)[mask], x0, bounds=(lo, hi), max_nfev=600)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def fit_counter(lay: Lay, y, mask):
    inits = [np.array([lp, 5.0, k, lz, 0.0]) for lp, lz, k in
             itertools.product(np.log([0.005, 0.02, 0.06, 0.2, 0.6]), np.log([0.01, 0.3, 3.0]), [0.0, 0.7])]
    return _ls(lay.counter, y, mask, inits, [-9, -60, -2, -12, -20], [3, 60, 2, 8, 20])


def fit_counter2(lay: Lay, y, mask):
    inits = [np.array([l1, l2, 3.0, 3.0, z, z, k, 0.0]) for l1, l2, z, k in
             itertools.product(np.log([0.01, 0.05]), np.log([0.2, 0.8]), np.log([0.05, 1.0]), [0.0, 0.7])]
    return _ls(lay.counter2, y, mask, inits, [-9, -9, -60, -60, -12, -12, -2, -20], [3, 3, 60, 60, 8, 8, 2, 20])


def fit_additive(lay: Lay, y, mask):
    inits = [np.array([a, b, 1.0, 1.0, 0.0]) for a, b in itertools.product(np.log([0.005, 0.03, 0.15]), repeat=2)]
    return _ls(lay.additive, y, mask, inits, [-9, -9, -50, -50, -30], [3, 3, 50, 50, 30])


def fit_affine(x, y, mask):
    A = np.stack([np.ones(mask.sum()), x[mask]], 1)
    coef, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
    return coef


class BayesFree:
    """Bayes with fitted (h', eps') plus affine calibration; grid search then local refinement."""

    def __init__(self, lay: Lay, typo: str = "structured"):
        self.lay, self.typo = lay, typo
        hg = np.concatenate([[0.0], np.exp(np.linspace(np.log(1e-4), np.log(0.5), 36))])
        eg = np.exp(np.linspace(np.log(1e-3), np.log(0.49), 36))
        H, E = np.meshgrid(hg, eg, indexing="ij")
        self.H, self.E = H.ravel(), E.ravel()
        self.table = lay.bayes(self.H, self.E, typo)

    def fit(self, y, mask):
        X, Y = self.table[:, mask], y[mask]
        xm, ym = X.mean(1, keepdims=True), Y.mean()
        b = ((X - xm) * (Y - ym)).sum(1) / np.clip(((X - xm) ** 2).sum(1), 1e-12, None)
        a = ym - b * xm[:, 0]
        sse = ((a[:, None] + b[:, None] * X - Y) ** 2).sum(1)
        g = int(np.argmin(sse))
        h0, e0, a0, b0 = self.H[g], self.E[g], a[g], b[g]
        fix0 = h0 == 0.0

        def hv(t):
            return 0.0 if fix0 else math.exp(t[0])

        def resid(t):
            return (t[2] + t[3] * self.lay.bayes([hv(t)], [1 / (1 + math.exp(-t[1]))], self.typo)[0] - y)[mask]

        t0 = np.array([math.log(1e-4) if fix0 else math.log(h0), math.log(e0 / (1 - e0)), a0, b0])
        lo = [t0[0] - 1e-9 if fix0 else math.log(1e-5), -9, -30, -30]
        hi = [t0[0] + 1e-9 if fix0 else math.log(0.7), 0, 30, 30]
        try:
            r = least_squares(resid, t0, bounds=(lo, hi), max_nfev=200)
            t = r.x if 2 * r.cost <= sse[g] else t0
        except Exception:
            t = t0
        return {"h": hv(t), "eps": 1 / (1 + math.exp(-t[1])), "a": t[2], "b": t[3]}

    @staticmethod
    def predict(lay: Lay, p: dict, typo: str = "structured"):
        return p["a"] + p["b"] * lay.bayes([p["h"]], [p["eps"]], typo)[0]


def r2(yhat, y):
    return float(1 - ((yhat - y) ** 2).mean() / y.var())
