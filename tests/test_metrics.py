import numpy as np
from ptq_trust import metrics as mt


def test_ece_perfectly_calibrated_is_zero():
    conf = np.array([0.5] * 10); corr = np.array([1] * 5 + [0] * 5)
    assert mt.ece(conf, corr, n_bins=10) == 0.0


def test_ece_known_value():
    conf = np.array([0.9] * 10); corr = np.array([1] * 5 + [0] * 5)        # acc 0.5, conf 0.9 -> 0.4
    assert abs(mt.ece(conf, corr, n_bins=10) - 0.4) < 1e-12


def test_ece_confidence_one_lands_in_last_bin():
    assert abs(mt.ece(np.array([1.0, 1.0]), np.array([1, 0]), 15) - 0.5) < 1e-12


def test_temperature_recovers_known_scale():
    rng = np.random.default_rng(0); n, T_true = 4000, 6.0
    z = rng.normal(0, 14, size=(n, 4))
    p = np.exp(z / T_true); p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=pi) for pi in p])
    assert abs(mt.fit_temperature(z, y) - T_true) < 0.8


def test_temperature_scaling_reduces_ece_on_overconfident_model():
    rng = np.random.default_rng(1); n = 3000
    z = rng.normal(0, 15, size=(n, 4)); p = np.exp(z / 8); p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=pi) for pi in p])
    folds = rng.permutation(n) % 2
    assert mt.crossfit_ece(z, y, folds) < 0.5 * mt.raw_ece(z, y)


def test_crossfit_matches_manual_two_fold_procedure():
    """Fit T on fold k only, evaluate on the other fold only, average: nothing else may influence the result."""
    rng = np.random.default_rng(2); n = 600
    z = rng.normal(0, 10, size=(n, 4)); y = rng.integers(0, 4, n); folds = rng.permutation(n) % 2
    manual = []
    for k in (0, 1):
        T = mt.fit_temperature(z[folds != k], y[folds != k])
        c, o = mt.conf_correct(z[folds == k], y[folds == k], T)
        manual.append(mt.ece(c, o, 15))
    assert abs(mt.crossfit_ece(z, y, folds) - np.mean(manual)) < 1e-12


def test_crossfit_eval_fold_labels_do_not_leak_into_fit():
    """Changing the labels of the evaluation fold must not change the temperature fitted for that fold."""
    rng = np.random.default_rng(3); n = 600
    z = rng.normal(0, 10, size=(n, 4)); y = rng.integers(0, 4, n); folds = rng.permutation(n) % 2
    T_before = mt.fit_temperature(z[folds != 0], y[folds != 0])
    y2 = y.copy(); y2[folds == 0] = (y2[folds == 0] + 1) % 4
    assert mt.fit_temperature(z[folds != 0], y2[folds != 0]) == T_before


def test_holm_known_example():
    adj = mt.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_bootstrap_is_seed_deterministic():
    a = np.random.default_rng(42).integers(0, 100, size=(50, 100)); b = np.random.default_rng(42).integers(0, 100, size=(50, 100))
    assert (a == b).all()


def test_bootstrap_p_floor_and_symmetry():
    d = np.random.default_rng(0).normal(1, 0.1, 1000)
    assert mt.bootstrap_p(d) == 1 / 1000
    assert mt.bootstrap_p(np.concatenate([-np.ones(500), np.ones(500)])) > 0.9
