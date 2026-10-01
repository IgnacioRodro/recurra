"""Tests for F5: recurrence structures.

The central test here is equivalence: the streaming path must give exactly
the same matrix as the dense path, bit for bit. Everything the library does
for long signals rests on that.
"""
import numpy as np
import pytest

import recurra as rc
from recurra import statespace as sm
from recurra.recurrence import (
    cross_recurrence_plot,
    joint_recurrence_plot,
    plan,
    recurrence_plot,
)
from recurra.recurrence.distance import sq_distance_block
from recurra.threshold import threshold


@pytest.fixture(scope="module")
def ss_pac():
    s = rc.generate_cfc(modality="pac", duration=10.0, fs=500.0, alpha=0.9,
                        preferred_phase=np.pi / 2, snr_db=20.0, seed=17)
    r = rc.analytic(rc.filterbank(s.to_recording("t"), {"theta": (4, 8),
                                                        "gamma": (50, 70)}),
                    sources=["theta", "gamma"])
    return sm.pac_space(r, smooth=8.0).subsample(max_points=800)


# ------------------------------------------------------------- distances
def test_expanded_norm_matches_naive_broadcasting():
    """DD-33: the BLAS expansion must be numerically identical."""
    rng = np.random.default_rng(0)
    A, B = rng.standard_normal((60, 7)), rng.standard_normal((45, 7))
    fast = sq_distance_block(A, B, "euclidean")
    naive = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
    np.testing.assert_allclose(fast, naive, atol=1e-9)


def test_squared_distances_are_non_negative():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((200, 3)) * 1e6
    assert (sq_distance_block(A, A, "euclidean") >= 0).all()


@pytest.mark.parametrize("metric", ["euclidean", "chebyshev", "manhattan", "cosine"])
def test_all_metrics_build_a_plot(ss_pac, metric):
    rm = recurrence_plot(ss_pac, "target_rr", metric=metric, target_rr=0.05,
                         theiler=1, rng=0)
    assert 0.0 < rm.recurrence_rate() < 1.0


# --------------------------------------------------- streaming equivalence
def test_tiled_equals_dense_bit_for_bit(ss_pac):
    """The whole long-signal strategy depends on this being exact."""
    th = threshold(ss_pac, target_rr=0.05, theiler=1, rng=0)
    dense = recurrence_plot(ss_pac, th, theiler=1, store="memory").matrix
    streamed = recurrence_plot(ss_pac, th, theiler=1, store="none", tile_size=97)
    rebuilt = np.zeros_like(dense)
    for tile in streamed.tiles():
        rebuilt[tile.i0:tile.i1, tile.j0:tile.j1] = tile.data
    np.testing.assert_array_equal(dense, rebuilt)


@pytest.mark.parametrize("tile_size", [31, 97, 256, 4096])
def test_recurrence_rate_is_tile_size_independent(ss_pac, tile_size):
    th = threshold(ss_pac, target_rr=0.05, theiler=1, rng=0)
    rm = recurrence_plot(ss_pac, th, theiler=1, store="none", tile_size=tile_size)
    reference = recurrence_plot(ss_pac, th, theiler=1, store="memory")
    assert rm.recurrence_rate() == pytest.approx(reference.recurrence_rate(), rel=1e-12)


def test_sparse_matches_dense(ss_pac):
    th = threshold(ss_pac, target_rr=0.03, theiler=1, rng=0)
    rm = recurrence_plot(ss_pac, th, theiler=1, store="memory")
    np.testing.assert_array_equal(rm.to_sparse().toarray().astype(np.uint8), rm.matrix)


# ------------------------------------------------------------- properties
def test_plot_is_symmetric(ss_pac):
    rm = recurrence_plot(ss_pac, target_rr=0.05, theiler=1, rng=0)
    assert rm.is_symmetric
    np.testing.assert_array_equal(rm.matrix, rm.matrix.T)


def test_theiler_zero_keeps_the_line_of_identity(ss_pac):
    """Corrected convention [DD-19]: theiler=0 excludes nothing."""
    rm = recurrence_plot(ss_pac, 0.5, theiler=0)
    assert np.all(np.diag(rm.matrix) == 1)


def test_theiler_one_removes_the_line_of_identity(ss_pac):
    rm = recurrence_plot(ss_pac, 0.5, theiler=1)
    assert np.all(np.diag(rm.matrix) == 0)


def test_theiler_one_removes_nothing_but_the_line_of_identity(ss_pac):
    """Standard convention [DD-115]: the first off-diagonals survive at
    theiler=1, as in the CRP Toolbox, pyunicorn and PyRQA. Before 0.25 they
    were removed too, so a window of 1 here meant a window of 2 elsewhere."""
    M = recurrence_plot(ss_pac, 0.5, theiler=1).matrix
    assert np.any(np.diag(M, 1) == 1) and np.any(np.diag(M, -1) == 1)


@pytest.mark.parametrize("theiler", [0, 1, 5, 20])
@pytest.mark.parametrize("tile_size", [64, 1024])
def test_theiler_band_is_exactly_excluded(ss_pac, theiler, tile_size):
    """Exactly the band |i - j| < theiler is removed [DD-115]: not one cell
    more, not one less. An epsilon larger than the attractor makes every pair
    recurrent, so the matrix must be 0 inside the band and 1 everywhere else;
    with a realistic epsilon an empty diagonal could be the data, not the band."""
    rm = recurrence_plot(ss_pac, 1e6, theiler=theiler, tile_size=tile_size)
    M = rm.matrix
    i, j = np.indices(M.shape)
    expected = (np.abs(i - j) >= theiler).astype(M.dtype)
    np.testing.assert_array_equal(M, expected)
    assert rm.recurrence_rate() == 1.0


def test_recurrence_rate_matches_the_target(ss_pac):
    for target in (0.02, 0.05, 0.10):
        rm = recurrence_plot(ss_pac, target_rr=target, theiler=1, rng=0)
        assert rm.recurrence_rate() == pytest.approx(target, abs=0.01)


def test_rate_denominator_excludes_theiler(ss_pac):
    """The reported rate must be comparable with the threshold's estimate."""
    rm = recurrence_plot(ss_pac, target_rr=0.05, theiler=10, rng=0)
    M = rm.matrix
    n = M.shape[0]
    excluded = sum(n - abs(k) for k in range(-9, 10))      # |i - j| < 10
    manual = M.sum() / (n * n - excluded)
    assert rm.recurrence_rate() == pytest.approx(manual, rel=1e-12)


def test_fixed_threshold_shortcut(ss_pac):
    rm = recurrence_plot(ss_pac, 0.42, theiler=1)
    assert rm.threshold.mode == "fixed" and rm.threshold.value == 0.42


def test_fan_makes_an_asymmetric_plot(ss_pac):
    th = threshold(ss_pac, mode="fan", n_neighbors=10, theiler=1, rng=0)
    rm = recurrence_plot(ss_pac, th, theiler=1)
    assert not rm.is_symmetric
    assert not np.array_equal(rm.matrix, rm.matrix.T)


def test_per_block_threshold_uses_chebyshev_across_blocks(ss_pac):
    th = threshold(ss_pac, mode="per_block", target_rr=0.2, rng=0)
    rm = recurrence_plot(ss_pac, th, theiler=1)
    assert 0.0 < rm.recurrence_rate() < 1.0


def test_sine_gives_diagonals_at_multiples_of_its_period():
    """Ground truth. A circle traced at frequency f0 recurs with itself every
    period, so the recurrence plot must be dense on the diagonals at integer
    multiples of fs/f0 and empty between them."""
    fs, f0, n = 200.0, 5.0, 4000
    t = np.arange(n) / fs
    X = np.column_stack([np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t)])
    # The rate must exceed the share of exact repeats (1/40 of all pairs), or
    # the threshold lands at rounding level (4.7e-14 at a rate of 0.02) and
    # which repeats count is decided by how the platform's sin() rounds: the
    # densities were 0.90 with numpy 2.4 and 0.86 with numpy 1.24.
    M = recurrence_plot(X, target_rr=0.05, theiler=1, rng=0).matrix
    density = np.array([np.diag(M, k).mean() for k in range(1, 181)])
    period = int(round(fs / f0))                     # 40 samples

    for mult in (1, 2, 3, 4):
        assert density[mult * period - 1] > 0.85, f"diagonal {mult * period} is sparse"
    for offset in (period // 2, period + period // 2, 2 * period + period // 2):
        assert density[offset - 1] < 0.05, f"diagonal {offset} should be empty"


# -------------------------------------------------------------------- CRP
def test_crp_is_rectangular_and_asymmetric(ss_pac):
    a = ss_pac.window(0, 400)
    b = ss_pac.window(100, 650)
    rm = cross_recurrence_plot(a, b, target_rr=0.05, rng=0)
    assert rm.shape == (400, 550) and rm.kind == "crp"
    assert not rm.is_symmetric


def test_crp_rejects_mismatched_dimensions(ss_pac):
    other = np.random.default_rng(0).standard_normal((ss_pac.n_points, 5))
    with pytest.raises(rc.ParameterError, match="same phase space"):
        cross_recurrence_plot(ss_pac, other, rng=0)


# -------------------------------------------------------------------- JRP
def test_jrp_allows_different_dimensions(ss_pac):
    """DD-20: this is the whole point of the joint plot."""
    a = ss_pac                              # 3-D
    b = ss_pac.split(["phase_theta"])       # 2-D
    rm = joint_recurrence_plot([a, b], target_rr=0.1, theiler=1, rng=0)
    assert rm.kind == "jrp"
    assert [p.dim for p in rm.parts] == [3, 2]


def test_jrp_uses_independent_thresholds(ss_pac):
    a = ss_pac
    b = ss_pac.split(["amp_gamma"])
    rm = joint_recurrence_plot([a, b], target_rr=0.1, theiler=1, rng=0)
    eps = [t.scalar for t in rm.thresholds]
    assert eps[0] != eps[1]
    for r in rm.subsystem_rates():
        assert r == pytest.approx(0.1, abs=0.02)


def test_jrp_is_the_and_of_its_parts(ss_pac):
    a, b = ss_pac, ss_pac.split(["phase_theta"])
    rm = joint_recurrence_plot([a, b], target_rr=0.1, theiler=1, rng=0)
    expected = rm.parts[0].matrix.astype(bool) & rm.parts[1].matrix.astype(bool)
    np.testing.assert_array_equal(rm.matrix.astype(bool), expected)


def test_jrp_rate_never_exceeds_its_parts(ss_pac):
    rm = joint_recurrence_plot([ss_pac, ss_pac.split(["phase_theta"])],
                               target_rr=0.1, theiler=1, rng=0)
    assert rm.recurrence_rate() <= min(rm.subsystem_rates()) + 1e-12


def test_jrp_independence_ratio_is_one_for_independent_systems():
    rng = np.random.default_rng(3)
    n = 900
    a = rng.standard_normal((n, 2))
    b = rng.standard_normal((n, 2))
    rm = joint_recurrence_plot([a, b], target_rr=0.1, theiler=1, rng=0)
    assert rm.independence_ratio() == pytest.approx(1.0, abs=0.25)


def test_jrp_rejects_misaligned_records(ss_pac):
    with pytest.raises(rc.ParameterError, match="common time base"):
        joint_recurrence_plot([ss_pac, ss_pac.window(0, 300)], rng=0)


def test_jrp_needs_two_subsystems(ss_pac):
    with pytest.raises(rc.ParameterError, match="at least two"):
        joint_recurrence_plot([ss_pac], rng=0)


# ----------------------------------------------------------------- memory
def test_plan_refuses_an_impossible_dense_matrix():
    with pytest.raises(rc.MemoryBudgetError, match="budget"):
        plan(2_000_000, 2_000_000, 3, dtype="uint8", store="memory", budget_gb=1.0)


def test_plan_falls_back_to_streaming():
    p = plan(500_000, 500_000, 3, dtype="uint8", budget_gb=1.0, estimated_rr=0.05)
    assert p.store in ("sparse", "stream") and p.tile_size >= 128


def test_plan_uses_memory_when_it_fits():
    assert plan(1000, 1000, 3, dtype="uint8", budget_gb=4.0).store == "memory"


def test_materialise_refuses_over_budget(ss_pac):
    """Built happily while the budget allowed it; refuses once it does not."""
    rm = recurrence_plot(ss_pac, 0.5, theiler=1, store="none", tile_size=128)
    rc.set_config(memory_budget_gb=1e-7)
    try:
        with pytest.raises(rc.MemoryBudgetError, match="budget"):
            rm.materialize(force=True)
    finally:
        rc.set_config(memory_budget_gb=4.0)


# --------------------------------------------------------------- reporting
def test_describe_and_summary(ss_pac):
    rm = recurrence_plot(ss_pac, target_rr=0.05, theiler=1, rng=0)
    df = rm.describe()
    assert {"kind", "epsilon", "actual_rr", "target_rr"} <= set(df.columns)
    assert "RecurrenceMatrix" in rm.summary()


def test_density_profile_streams(ss_pac):
    rm = recurrence_plot(ss_pac, target_rr=0.05, theiler=1, store="none", rng=0)
    df = rm.density_profile(n_bins=16)
    assert len(df) == 16 and df["density"].notna().any()


def test_provenance_carries_through(ss_pac):
    rm = recurrence_plot(ss_pac, target_rr=0.05, theiler=1, rng=0)
    steps = list(rm.provenance_frame()["step"])
    assert "statespace" in steps and "recurrence_plot" in steps



# ------------------------------------------------------------- DD-110
def test_single_precision_is_translation_invariant():
    """Centring before the norm expansion: an offset must not change the plot.

    Before DD-110 an offset of 1000 in every coordinate moved the recurrence
    rate of a single-precision plot from 0.051 to 0.082.
    """
    import recurra as rc

    rng = np.random.default_rng(7)
    X = np.cumsum(rng.normal(size=(1500, 3)), axis=0)
    th = rc.threshold(X, target_rr=0.05, theiler=1, rng=0)
    base = rc.recurrence_plot(X, th, theiler=1, precision="single",
                              store="memory").matrix
    for offset in (100.0, 1000.0, 10000.0):
        shifted = rc.recurrence_plot(X + offset, th, theiler=1,
                                     precision="single", store="memory").matrix
        assert np.array_equal(base, shifted), f"offset {offset} changed the plot"
    # and the double path agrees with the single one on centred data
    dbl = rc.recurrence_plot(X, th, theiler=1, precision="double",
                             store="memory").matrix
    assert np.mean(dbl != base) < 1e-4


@pytest.mark.parametrize("metric,scipy_name", [("euclidean", "euclidean"),
                                               ("chebyshev", "chebyshev"),
                                               ("manhattan", "cityblock"),
                                               ("cosine", "cosine")])
def test_every_metric_matches_scipy_on_an_offset_cloud(metric, scipy_name):
    """An independent reference for each distance. Coordinates far from the
    origin matter: centring them is exact for the three translation-invariant
    metrics and wrong for cosine, which was centred anyway before 0.25 and
    came out at a rate of 0.0013 for a target of 0.05 [DD-110]."""
    from scipy.spatial.distance import cdist

    X = np.random.default_rng(0).standard_normal((600, 3)) * 0.5 + [3.0, 1.0, 0.5]
    th = rc.threshold(X, target_rr=0.05, theiler=1, metric=metric, rng=0)
    rm = recurrence_plot(X, th, theiler=1, metric=metric, store="memory")
    i, j = np.indices((600, 600))
    expected = (cdist(X, X, scipy_name) <= float(th.scalar)) & (np.abs(i - j) >= 1)
    np.testing.assert_array_equal(rm.matrix.astype(bool), expected)
    assert rm.recurrence_rate() == pytest.approx(0.05, abs=0.01)


# ------------------------------------------------------ usability [DD-118]
def test_a_plot_inherits_the_theiler_window_of_its_threshold(ss_pac):
    th = threshold(ss_pac, target_rr=0.05, theiler=7, rng=0)
    rm = recurrence_plot(ss_pac, th)
    assert rm.theiler == 7
    assert rm.recurrence_rate() == pytest.approx(0.05, abs=0.01)


def test_a_mismatched_theiler_window_or_metric_warns(ss_pac):
    """Measured before 0.25: estimated at 1, used at 50, the rate came out at
    0.024 for a target of 0.05 with no warning."""
    th = threshold(ss_pac, target_rr=0.05, theiler=1, rng=0)
    with pytest.warns(rc.ParameterWarning, match="theiler=1"):
        recurrence_plot(ss_pac, th, theiler=50)
    with pytest.warns(rc.ParameterWarning, match="metric"):
        recurrence_plot(ss_pac, th, metric="chebyshev")


def test_threshold_and_plot_defaults_agree():
    import inspect

    assert (inspect.signature(threshold).parameters["theiler"].default
            == rc.recurrence.builders.DEFAULT_THEILER == 1)


def test_streaming_is_called_stream_and_the_old_name_still_works(ss_pac):
    a = recurrence_plot(ss_pac, target_rr=0.05, store="stream", rng=0)
    b = recurrence_plot(ss_pac, target_rr=0.05, store="none", rng=0)
    assert a.store == b.store == "stream"
    assert a.recurrence_rate() == b.recurrence_rate()


def test_epsilon_at_rounding_level_is_reported():
    """A sine with 2.5% exactly repeated pairs and a target of 0.02."""
    t = np.arange(4000) / 200.0
    X = np.column_stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 5 * t)])
    with pytest.warns(rc.GeometryWarning, match="rounding level"):
        threshold(X, target_rr=0.02, rng=0)


def test_single_precision_reaches_every_structure(ss_pac):
    X = np.asarray(ss_pac.coords)
    assert rc.cross_recurrence_plot(X, X[::-1], target_rr=0.05, precision="single",
                                    rng=0).precision == "single"
    wr = rc.windowed_recurrence(ss_pac, 3, target_rr=0.05, precision="single", rng=0)
    assert wr[0].precision == "single"
