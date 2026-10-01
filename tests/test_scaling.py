"""Tests for DD-02, the heterogeneous-block scaling decision."""
import numpy as np
import pytest

import recurra as rc
from recurra.preprocess.scaling import (
    CIRCLE_RMS_PAIRWISE,
    Scaler,
    rms_pairwise_distance,
    theoretical_rms_pairwise,
)


def _circle(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([np.cos(phi), np.sin(phi)])


def test_unit_circle_rms_is_sqrt_two():
    """The analytic result the default policy is built on."""
    assert theoretical_rms_pairwise(_circle()) == pytest.approx(CIRCLE_RMS_PAIRWISE, abs=0.02)


def test_scalar_rms_is_sigma_sqrt_two():
    rng = np.random.default_rng(1)
    for sigma in (0.1, 1.0, 25.0):
        x = (rng.normal(0, sigma, 5000)).reshape(-1, 1)
        assert theoretical_rms_pairwise(x) == pytest.approx(sigma * np.sqrt(2), rel=0.05)


def test_sampled_and_theoretical_rms_agree():
    X = _circle(3000)
    assert rms_pairwise_distance(X, rng=0) == pytest.approx(theoretical_rms_pairwise(X), rel=0.05)


@pytest.mark.parametrize("sigma", [0.001, 0.1, 1.0, 10.0, 1000.0])
def test_rms_balanced_equalises_regardless_of_amplitude_scale(sigma):
    rng = np.random.default_rng(2)
    C = _circle(4000)
    A = (rng.normal(0, sigma, 4000)).reshape(-1, 1)
    rep = Scaler("rms_balanced").fit([("phase", C), ("amplitude", A)])
    assert rep.variance_share[0] == pytest.approx(0.5, abs=1e-6)
    assert rep.dominant is None


def test_unscaled_blocks_are_dominated_and_flagged():
    rng = np.random.default_rng(3)
    C = _circle(3000)
    A = (rng.normal(0, 200.0, 3000)).reshape(-1, 1)
    rep = Scaler("none").fit([("phase", C), ("amplitude", A)])
    assert rep.variance_share[1] > 0.99
    assert rep.dominant == "amplitude"
    assert "distance variance" in rep.warning


@pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_lambda_controls_share_exactly(lam):
    rng = np.random.default_rng(4)
    C = _circle(3000)
    A = (rng.normal(0, 13.7, 3000)).reshape(-1, 1)   # arbitrary scale
    rep = Scaler("weighted", lambda_=lam).fit([("phase", C), ("amplitude", A)])
    assert rep.variance_share[1] == pytest.approx(lam, abs=1e-6)


def test_lambda_rejects_bad_values():
    C, A = _circle(500), np.zeros((500, 1))
    with pytest.raises(rc.ParameterError):
        Scaler("weighted", lambda_=1.5).fit([("p", C), ("a", A)])


def test_gain_invariance_of_the_default():
    """A global gain change must not alter the geometry."""
    rng = np.random.default_rng(5)
    C = _circle(3000)
    A = rng.gamma(2.0, 1.0, 3000).reshape(-1, 1)
    r1 = Scaler("rms_balanced").fit([("p", C), ("a", A)])
    r2 = Scaler("rms_balanced").fit([("p", C), ("a", A * 1000.0)])
    assert r1.variance_share == pytest.approx(r2.variance_share, abs=1e-9)
