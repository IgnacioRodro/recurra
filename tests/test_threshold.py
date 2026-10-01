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
