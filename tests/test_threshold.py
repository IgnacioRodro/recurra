"""Tests for F4: threshold selection."""
import numpy as np
import pytest

import recurra as rc
from recurra import statespace as sm
from recurra.threshold import MODES, sample_pair_distances, threshold
from recurra.threshold.estimate import cross_threshold


@pytest.fixture(scope="module")
def ss_pac():
    s = rc.generate_cfc(modality="pac", duration=12.0, fs=500.0, alpha=0.9,
                        preferred_phase=np.pi / 2, snr_db=20.0, seed=17)
    r = rc.analytic(rc.filterbank(s.to_recording("t"), {"theta": (4, 8),
                                                        "gamma": (50, 70)}),
                    sources=["theta", "gamma"])
    return sm.pac_space(r, smooth=8.0).subsample(max_points=1500)


# --------------------------------------------------------------- sampling
def test_sampler_honours_theiler():
    """DD-32: sampled pairs must come from the region RR is defined over."""
    X = np.cumsum(np.random.default_rng(0).standard_normal((2000, 2)), axis=0)
    near = sample_pair_distances(X, n_samples=20000, theiler=0, rng=0)
    far = sample_pair_distances(X, n_samples=20000, theiler=50, rng=0)
    # Excluding near-diagonal pairs removes the closest ones, so the mean rises
    assert far.mean() > near.mean()


def test_sampler_refuses_impossible_theiler():
    X = np.random.default_rng(0).standard_normal((20, 2))
    with pytest.raises(rc.ParameterError, match="excludes every pair"):
        sample_pair_distances(X, theiler=25, rng=0)


# ------------------------------------------------------------------ modes
def test_target_rr_hits_the_target(ss_pac):
    for target in (0.01, 0.05, 0.15):
        th = threshold(ss_pac, mode="target_rr", target_rr=target, theiler=1, rng=0)
        assert th.achieved_rr == pytest.approx(target, abs=0.005)


def test_bisect_converges_and_reports_iterations(ss_pac):
    th = threshold(ss_pac, mode="target_rr", target_rr=0.05, method="bisect",
                   tol=1e-3, theiler=1, rng=0)
    assert th.diagnostics["converged"] is True
    assert th.achieved_rr == pytest.approx(0.05, abs=2e-3)


def test_fixed_reports_what_it_achieves(ss_pac):
    th = threshold(ss_pac, mode="fixed", value=0.4, theiler=1, rng=0)
    assert th.value == 0.4 and 0.0 < th.achieved_rr < 1.0


def test_percentile_rejects_out_of_range(ss_pac):
    with pytest.raises(rc.ParameterError, match="fraction in"):
        threshold(ss_pac, mode="percentile", percentile=5.0, rng=0)


def test_fan_is_pointwise(ss_pac):
    th = threshold(ss_pac, mode="fan", n_neighbors=10, theiler=1, rng=0)
    assert th.is_pointwise
    assert th.value.shape == (ss_pac.n_points,)
    assert "not symmetric" in th.diagnostics["warning"]


def test_fan_neighbour_count_sets_the_rate(ss_pac):
    th = threshold(ss_pac, mode="fan", n_neighbors=15, theiler=1, rng=0)
    expected = 15 / (ss_pac.n_points - 1)          # theiler=1: only i = j out
    assert th.achieved_rr == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("theiler", [0, 1])
def test_fan_never_counts_a_point_as_its_own_neighbour(ss_pac, theiler):
    """DD-115: neighbours need |i - j| >= max(theiler, 1), so with theiler=0
    the nearest neighbour is still another point, never the point itself."""
    th = threshold(ss_pac, mode="fan", n_neighbors=1, theiler=theiler, rng=0)
    assert np.all(np.asarray(th.value) > 0)


def test_per_block_gives_one_epsilon_per_block(ss_pac):
    th = threshold(ss_pac, mode="per_block", target_rr=0.05, rng=0)
    assert th.is_per_block and len(th.per_block) == len(ss_pac.groups)
    assert th.block_labels == tuple(ss_pac.block_labels)


def test_per_block_needs_a_statespace():
    with pytest.raises(rc.ParameterError, match="needs a StateSpace"):
        threshold(np.random.default_rng(0).standard_normal((200, 3)),
                  mode="per_block", rng=0)


def test_std_fraction_scales_with_the_attractor(ss_pac):
    a = threshold(ss_pac, mode="std_fraction", factor=0.1, rng=0)
    b = threshold(ss_pac, mode="std_fraction", factor=0.2, rng=0)
    assert b.value == pytest.approx(2 * a.value, rel=1e-9)


def test_adaptive_dim_warns_when_epsilon_swallows_the_attractor(ss_pac):
    th = threshold(ss_pac, mode="adaptive_dim", target_rr=0.7, rng=0)
    assert "epsilon_over_rms_radius" in th.diagnostics


def test_unknown_mode_is_rejected(ss_pac):
    with pytest.raises(rc.ParameterError, match="unknown threshold mode"):
        threshold(ss_pac, mode="magic")


def test_all_documented_modes_run(ss_pac):
    kwargs = {"fixed": {"value": 0.4}, "percentile": {"percentile": 0.05},
              "fan": {"n_neighbors": 8}, "std_fraction": {"factor": 0.1},
              "maxdist_fraction": {"factor": 0.1}, "per_block": {}}
    for mode in MODES:
        th = threshold(ss_pac, mode=mode, theiler=1, rng=0, **kwargs.get(mode, {}))
        assert th.mode == mode


def test_threshold_frame_is_exportable(ss_pac):
    df = threshold(ss_pac, target_rr=0.05, rng=0).to_frame()
    assert {"mode", "epsilon", "target_rr", "achieved_rr"} <= set(df.columns)


# ------------------------------------------------------------------ cross
def test_cross_threshold_requires_matching_dimensions(ss_pac):
    other = np.random.default_rng(0).standard_normal((ss_pac.n_points, 5))
    with pytest.raises(rc.ParameterError, match="equal dimensionality"):
        cross_threshold(ss_pac, other, rng=0)


def test_cross_threshold_hits_target(ss_pac):
    other = sm.channels.__wrapped__ if False else np.asarray(ss_pac.weighted_coords)
    th = cross_threshold(ss_pac, other + 0.05, target_rr=0.05, rng=0)
    assert th.achieved_rr == pytest.approx(0.05, abs=0.01)


# --------------------------------------------------------- DD-122: ties
# The reference is scipy's cdist: the matrix must hold exactly the pairs whose
# true distance is <= epsilon, with no pair left for rounding to decide.

def _reference(X, eps, theiler):
    from scipy.spatial.distance import cdist

    n = X.shape[0]
    D = cdist(X, X)
    band = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= theiler
    return D, band, (D <= eps) & band


@pytest.mark.parametrize("n,theiler", [(12, 2), (40, 0), (300, 1), (1157, 1)])
def test_epsilon_is_never_an_observed_distance(n, theiler):
    """DD-122: epsilon is kept off every distance the data contain, so the
    plot equals the reference whatever formula computes the distances. Before,
    it differed from cdist in 5 to 20% of the draws at every size."""
    from recurra.threshold.estimate import TIE_RTOL

    for seed in range(4):
        X = np.random.default_rng(seed).standard_normal((n, 3)) + 3.0
        th = threshold(X, target_rr=0.05, theiler=theiler, rng=seed)
        eps = float(th.value)
        D, band, ref = _reference(X, eps, theiler)
        if n * n <= 200_000:
            # every pair was in the sample, so none may sit at epsilon
            assert np.min(np.abs(D[band] - eps)) > TIE_RTOL * eps
        M = rc.recurrence_plot(X, th, store="memory").matrix
        assert np.array_equal(M, ref), f"n={n}, seed={seed}"


def test_small_matrices_take_every_pair_once():
    """DD-122: with fewer admissible pairs than samples the quantile is the
    exact one, the same for every seed. Sampled with replacement, a 12-point
    plot asked for 0.20 got anything from 0.15 to 0.35."""
    X = np.random.default_rng(3).standard_normal((12, 3))
    rates = set()
    for seed in range(10):
        th = threshold(X, target_rr=0.2, theiler=2, rng=seed)
        _, band, ref = _reference(X, float(th.value), 2)
        assert th.achieved_rr == pytest.approx(ref[band].mean(), abs=1e-15)
        rates.add(round(th.achieved_rr, 12))
    assert len(rates) == 1
    assert rates.pop() == pytest.approx(0.2, abs=1 / 110)   # one pair of 55


def test_ties_are_inside():
    """DD-122: on quantised data many pairs share the distance the quantile
    lands on; all of them are recurrent, none is left to rounding."""
    X = np.random.default_rng(5).integers(0, 4, size=(60, 3)).astype(float)
    th = threshold(X, target_rr=0.1, theiler=1, rng=0)
    eps = float(th.value)
    D, band, ref = _reference(X, eps, 1)
    tied = D[band][np.abs(D[band] - np.max(D[band][D[band] <= eps])) == 0]
    assert tied.size > 2 and np.all(tied < eps)
    M = rc.recurrence_plot(X, th, store="memory").matrix
    assert np.array_equal(M, ref)


def test_fan_rows_hold_exactly_n_neighbours():
    """DD-122: the FAN epsilon of a point was the distance to its n-th
    neighbour, which the matrix then recomputed with another formula; 47% of
    the rows held n - 1 recurrences. It now lies between the n-th neighbour
    and the next one."""
    X = np.random.default_rng(1).standard_normal((400, 3)) + 3.0
    th = threshold(X, mode="fan", n_neighbors=10, theiler=1, rng=0)
    M = rc.recurrence_plot(X, th, store="memory").matrix
    assert np.all(M.sum(axis=1) == 10)


def test_small_cross_thresholds_are_exact():
    """DD-122 for cross plots: a small one takes every pair once, so epsilon
    does not depend on the seed, and it stays off every cross distance."""
    from scipy.spatial.distance import cdist

    from recurra.threshold.estimate import TIE_RTOL

    g = np.random.default_rng(2)
    X, Y = g.standard_normal((10, 2)), g.standard_normal((14, 2))
    eps = {float(cross_threshold(X, Y, target_rr=0.2, rng=s).value) for s in range(5)}
    assert len(eps) == 1
    e = eps.pop()
    assert np.min(np.abs(cdist(X, Y) - e)) > TIE_RTOL * e
    crp = rc.cross_recurrence_plot(X, Y, target_rr=0.2, rng=0).matrix
    assert np.array_equal(crp, cdist(X, Y) <= e)


def test_per_block_rates_do_not_depend_on_the_weights(ss_pac):
    """DD-35, revised in 0.26: each block's epsilon is estimated on the
    coordinates the matrix compares, so every block sees the rate it was
    given. Estimated on the unweighted coordinates, the block weights leaked
    back in: 0.221 for the phase block and 0.181 for the amplitude block, for
    a target of 0.2."""
    from scipy.spatial.distance import cdist

    ss = ss_pac.subsample(max_points=300)
    th = threshold(ss, mode="per_block", target_rr=0.2, rng=0)
    W = np.asarray(ss.weighted_coords)
    n = W.shape[0]
    band = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= 1
    joint = band.copy()
    for g, eps in zip(ss.groups, th.per_block, strict=True):
        block = (cdist(W[:, g.slice], W[:, g.slice]) <= eps) & band
        assert block[band].mean() == pytest.approx(0.2, abs=1e-3), g.label
        joint &= block
    assert np.array_equal(rc.recurrence_plot(ss, th, store="memory").matrix, joint)
