"""Calibration metrics and small statistics helpers (numpy / scipy only, so they are unit-testable).

Conventions
-----------
* ``logits``: array (n, 4) of answer-letter logits; ``y``: gold index array (n,).
* Confidence = max softmax probability; correctness = (argmax == gold).
* ECE uses equal-width bins; temperature is fitted by minimising NLL.
* Cross-fitting: ``folds`` is a per-ITEM fold id (0/1) assigned once, BEFORE any bootstrap resampling, so a
  resampled duplicate of an item can never appear in both the fit and the evaluation fold.
"""
import numpy as np
from scipy.special import softmax


def ece(conf, corr, n_bins=15):
    """Expected calibration error with equal-width bins: sum_k (n_k/n) * |acc_k - conf_k|."""
    conf = np.asarray(conf, dtype=float); corr = np.asarray(corr, dtype=float)
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    sc = np.bincount(b, weights=corr, minlength=n_bins)
    sf = np.bincount(b, weights=conf, minlength=n_bins)
    return float(np.abs(sc - sf).sum() / len(conf))


def _nll_grid(logits, y, temps):
    """NLL of the gold label for every temperature in ``temps`` (vectorised)."""
    z = logits[None, :, :] / temps[:, None, None]
    z = z - z.max(axis=2, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=2))
    gold = z[:, np.arange(len(y)), y]
    return -(gold - lse).mean(axis=1)


def nll_at_T(logits, y, T=1.0):
    return float(_nll_grid(np.asarray(logits, float), np.asarray(y), np.array([float(T)]))[0])


_COARSE = np.exp(np.linspace(np.log(0.3), np.log(60.0), 60))


def fit_temperature(logits, y):
    """Temperature minimising NLL: coarse log-spaced grid, then a fine grid around the best coarse value."""
    logits = np.asarray(logits, float); y = np.asarray(y)
    i = int(np.argmin(_nll_grid(logits, y, _COARSE)))
    lo, hi = _COARSE[max(i - 1, 0)], _COARSE[min(i + 1, len(_COARSE) - 1)]
    fine = np.linspace(lo, hi, 25)
    return float(fine[int(np.argmin(_nll_grid(logits, y, fine)))])


def conf_correct(logits, y, T=1.0):
    p = softmax(np.asarray(logits, float) / T, axis=1)
    return p.max(axis=1), (p.argmax(axis=1) == np.asarray(y)).astype(float)


def raw_ece(logits, y, n_bins=15):
    c, o = conf_correct(logits, y)
    return ece(c, o, n_bins)


def crossfit_ece(logits, y, folds, n_bins=15):
    """2-fold cross-fitted ECE (fit T on one fold, evaluate on the other, average both directions)."""
    logits = np.asarray(logits, float); y = np.asarray(y); folds = np.asarray(folds)
    vals = []
    for k in (0, 1):
        tr, te = folds != k, folds == k
        if tr.sum() < 20 or te.sum() < 20:
            return float("nan")
        T = fit_temperature(logits[tr], y[tr])
        c, o = conf_correct(logits[te], y[te], T)
        vals.append(ece(c, o, n_bins))
    return float(np.mean(vals))


def margin(logits, t):
    """Logit margin of class t over the best other class."""
    logits = np.asarray(logits, float)
    return float(logits[t] - np.delete(logits, t).max())


def bootstrap_p(d):
    """Two-sided bootstrap p-value for a paired difference (floor 1/B)."""
    d = np.asarray(d); d = d[~np.isnan(d)]
    return float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / len(d)))


def holm(ps):
    """Holm step-down adjusted p-values."""
    ps = np.asarray(ps, float); order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(run, 1.0)
    return adj


def percentile_ci(x, lo=2.5, hi=97.5):
    return np.percentile(np.asarray(x)[~np.isnan(x)], [lo, hi])
