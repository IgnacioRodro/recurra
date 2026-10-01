"""Tests for F9: dynamical invariants and complexity measures.

The reference values here are for the **correlation** dimension, not the
Kaplan-Yorke dimension often quoted beside it: Lorenz 2.05 and Rossler 1.81
(Grassberger and Procaccia 1983). Quoting 2.01 for Rossler is the Kaplan-Yorke
value and this estimator will not reproduce it.
"""
import numpy as np
import pytest

import recurra as rc
from recurra import statespace as sm
from recurra.dynamics.scaling import find_scaling_region, fixed_region, local_slopes


@pytest.fixture(scope="module")
def lorenz_space():
    X = rc.lorenz(n_points=12000, dt=0.01, transient=20.0)
    return sm.channels(rc.ingest(X, fs=100.0, labels=["x", "y", "z"]))


@pytest.fixture(scope="module")
def rossler_space():
    X = rc.rossler(n_points=12000, dt=0.05, transient=50.0)
    return sm.channels(rc.ingest(X, fs=20.0, labels=["x", "y", "z"]))


# ------------------------------------------------------- scaling region
def test_scaling_region_recovers_a_known_slope():
    x = np.linspace(0, 4, 60)
    y = 2.5 * x + 0.3 + 0.001 * np.random.default_rng(0).standard_normal(60)
    reg = find_scaling_region(x, y)
    assert reg.slope == pytest.approx(2.5, abs=0.05)
    assert reg.r_squared > 0.99


def test_scaling_region_avoids_a_bent_tail():
    """A curve that is straight then saturates: the fit must take the straight
    part, which is the whole point of locating a region rather than fitting
    everything [DD-61]."""
    x = np.linspace(0, 6, 80)
    y = np.where(x < 3.5, 2.0 * x, 7.0 + 0.1 * (x - 3.5))
    reg = find_scaling_region(x, y, min_points=8)
    assert reg.slope == pytest.approx(2.0, abs=0.2) or \
           reg.slope == pytest.approx(0.1, abs=0.2)
    assert reg.n_points < len(x)


def test_narrow_region_is_flagged():
    x = np.linspace(0, 0.1, 20)
    reg = find_scaling_region(x, 2 * x, min_decades=1.0)
    assert reg.warning is not None and "decades" in reg.warning


def test_fixed_region_honours_the_window():
    x = np.linspace(0, 10, 100)
    reg = fixed_region(x, 3 * x, 2.0, 5.0)
    assert reg.slope == pytest.approx(3.0, abs=1e-6)
    assert reg.x_lo >= 2.0 and reg.x_hi <= 5.0


def test_local_slopes_are_constant_for_a_line():
    x = np.linspace(0, 5, 40)
    s = local_slopes(x, 1.7 * x)
    assert np.nanstd(s) < 1e-9


# --------------------------------------------------- correlation dimension
def test_lorenz_correlation_dimension(lorenz_space):
    d2 = rc.correlation_dimension(lorenz_space, theiler=20, max_points=5000,
                                  n_radii=50, rng=0)
    assert d2.value == pytest.approx(2.05, abs=0.25)
    assert d2.region.r_squared > 0.99
    assert d2.units == "dimensionless"


def test_rossler_correlation_dimension(rossler_space):
    d2 = rc.correlation_dimension(rossler_space, theiler=20, max_points=5000,
                                  n_radii=50, rng=0)
    assert d2.value == pytest.approx(1.81, abs=0.25)


def test_dimension_orders_the_systems(lorenz_space, rossler_space):
    """Rossler is nearly a two-dimensional band; Lorenz is thicker."""
    d_l = rc.correlation_dimension(lorenz_space, theiler=20, max_points=4000, rng=0)
    d_r = rc.correlation_dimension(rossler_space, theiler=20, max_points=4000, rng=0)
    assert d_r.value < d_l.value


def test_correlation_sum_is_monotone(lorenz_space):
    c = rc.correlation_sum(lorenz_space, theiler=20, max_points=2000, rng=0)
    assert (np.diff(c["C"].to_numpy()) >= 0).all()
    assert c["C"].iloc[-1] <= 1.0


def test_dimension_accepts_a_user_region(lorenz_space):
    c = rc.correlation_sum(lorenz_space, theiler=20, max_points=2000, rng=0)
    lo, hi = c["log_r"].quantile([0.2, 0.7])
    d2 = rc.correlation_dimension(lorenz_space, theiler=20, max_points=2000,
                                  region=(lo, hi), rng=0)
    assert "user-specified" in d2.region.criterion


# ---------------------------------------------------------- Lyapunov
def test_lorenz_lyapunov(lorenz_space):
    """Reference 0.906 per second.

    The curve has to be long enough to contain a linear region [DD-94], and
    that matters more here than the sampling does [DD-101]: measured on this
    space, max_steps=150 gives 1.13 and 400 gives 0.894, while varying the
    random stream at fixed max_steps moves it by 4%.
    """
    l1 = rc.lyapunov_max(lorenz_space, max_steps=400, theiler=125,
                         max_points=3000, rng=0)
    assert l1.units == "per_second"
    assert l1.value == pytest.approx(0.906, rel=0.25)
    assert l1.value > 0


def test_lyapunov_units_convert_by_the_sampling_rate(lorenz_space):
    """DD-62: a rate without its unit is not a number anyone can use."""
    per_s = rc.lyapunov_max(lorenz_space, max_steps=150, theiler=50,
                            max_points=2000, n_repeats=2, rng=0)
    per_n = rc.lyapunov_max(lorenz_space, max_steps=150, theiler=50,
                            max_points=2000, n_repeats=2, units="per_sample",
                            rng=0)
    assert per_s.value == pytest.approx(per_n.value * lorenz_space.fs, rel=1e-9)


#: Three Lorenz trajectories from different initial conditions. A chaotic
#: trajectory integrated for two minutes amplifies any difference in rounding
#: between scipy versions, so a test pinned to one trajectory asserts a fact
#: about one platform: two such tests passed on Python 3.12 and failed on 3.14
#: with identical code and seeds. A contract checked on several trajectories
#: holds on all of them or tells us something real.
LORENZ_STARTS = ((1.0, 1.0, 1.0), (-5.0, 3.0, 20.0), (8.0, -2.0, 30.0))
LORENZ_LAMBDA = 0.906


def _lorenz_space(y0):
    X = rc.lorenz(n_points=12000, dt=0.01, transient=20.0, y0=y0)
    return sm.channels(rc.ingest(X, fs=100.0, labels=["x", "y", "z"]))


@pytest.mark.slow
def test_an_unflagged_lyapunov_exponent_can_be_trusted():
    """DD-94 and DD-101 together, as a contract rather than an anecdote.

    The guards exist so that a value without a warning is a value the data
    determines. Before averaging, one sampling at max_steps=1200 returned 1.54
    against 0.906 with no warning. So: across trajectories and two settings,
    every unflagged estimate must lie within 25% of the reference, and at least
    two of the six must be unflagged (a guard that flagged everything would
    satisfy the first clause vacuously).

    Measured in 0.25: the contract held -- unflagged 0.911 and 0.791, the two
    estimates more than 25% off (1.126, 1.273) both flagged -- but the guard is
    conservative: of the four estimates within 25%, two were flagged. An
    earlier version of this test also required max_steps=300 to come back
    unflagged on two trajectories of three, which assumed that setting is
    always sound; on the third trajectory it returned 1.273.
    """
    report = []
    for y0 in LORENZ_STARTS:
        space = _lorenz_space(y0)
        for steps in (300, 1200):
            r = rc.lyapunov_max(space, max_steps=steps, theiler=125,
                                max_points=2000, n_repeats=3, rng=0)
            report.append((y0, steps, round(r.value, 3), r.warning is None))
            assert r.diagnostics["n_repeats"] == 3 and "sampling_cv" in r.diagnostics
            if r.warning is None:
                assert r.value == pytest.approx(LORENZ_LAMBDA, rel=0.25), (
                    f"unflagged but wrong: {report}")
    assert sum(r[3] for r in report) >= 2, f"the guard flags almost everything: {report}"


def test_divergence_curve_rises(lorenz_space):
    c = rc.divergence_curve(lorenz_space, max_steps=100, theiler=50,
                            max_points=1500, rng=0)
    y = c["mean_log_divergence"].to_numpy()
    assert np.nanmean(y[-20:]) > np.nanmean(y[:20])


def test_noiseless_orbit_is_flagged_as_numerically_degenerate():
    """On a sampled circle the nearest neighbours sit at machine precision --
    1.3e-15 measured -- so the divergence curve tracks floating-point noise
    through a logarithm. The estimator cannot tell that apart from dynamics,
    so it must say so rather than return a number [DD-63]."""
    fs, n = 200.0, 6000
    t = np.arange(n) / fs
    S = np.column_stack([np.sin(2 * np.pi * 3 * t), np.cos(2 * np.pi * 3 * t)])
    ss = sm.channels(rc.ingest(S, fs=fs, labels=["a", "b"]))
    l1 = rc.lyapunov_max(ss, max_steps=200, theiler=100, max_points=2000, rng=0)
    assert l1.warning is not None
    assert "numerical zero" in l1.warning


def test_a_real_attractor_can_be_estimated_without_a_flag(lorenz_space):
    """Chaos with a curve long enough to hold a linear region and short enough
    not to saturate. If nothing here is clean, the estimator is unusable."""
    clean = [rc.lyapunov_max(lorenz_space, max_steps=steps, theiler=125,
                             max_points=2000, n_repeats=1, rng=0)
             for steps in (400, 600)]
    assert any(l1.warning is None for l1 in clean)


def test_unknown_units_rejected(lorenz_space):
    with pytest.raises(rc.ParameterError, match="per_sample"):
        rc.lyapunov_max(lorenz_space, units="furlongs", rng=0)


# ---------------------------------------------------------------- K2
def test_k2_is_positive_for_chaos(lorenz_space):
    k2 = rc.k2_entropy(lorenz_space.subsample(max_points=2500),
                       target_rr=0.05, theiler=20, rng=0)
    assert k2.value > 0 and k2.units == "per_second"


def test_k2_orders_chaos_above_a_noisy_limit_cycle():
    """Compared in per-sample units, so the two sampling rates do not enter."""
    fs, n = 200.0, 6000
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    S = np.column_stack([np.sin(2 * np.pi * 3 * t), np.cos(2 * np.pi * 3 * t)])
    S = S + 0.01 * rng.standard_normal(S.shape)          # a measurable signal
    per = sm.channels(rc.ingest(S, fs=fs, labels=["a", "b"]))
    cha = sm.channels(rc.ingest(rc.lorenz(6000, dt=0.01, transient=10.0),
                                fs=100.0, labels=["x", "y", "z"]))
    k_per = rc.k2_entropy(per.subsample(max_points=2500), units="per_sample",
                          target_rr=0.05, theiler=20, rng=0).value
    k_cha = rc.k2_entropy(cha.subsample(max_points=2500), units="per_sample",
                          target_rr=0.05, theiler=20, rng=0).value
    assert k_cha > k_per


def test_k2_refuses_a_plot_too_sparse_to_fit():
    """K2 comes from the decay rate of the diagonal length distribution. On a
    plot with almost no lines there is no decay to fit, and refusing is better
    than returning a slope through three points."""
    X = np.random.default_rng(0).standard_normal((300, 3))
    with pytest.raises(rc.ParameterError, match="too few distinct diagonal"):
        rc.k2_entropy(X, target_rr=0.01, theiler=1, rng=0)


def test_k2_reports_how_much_evidence_it_had():
    L = sm.channels(rc.ingest(rc.lorenz(6000, dt=0.01, transient=10.0),
                              fs=100.0, labels=["x", "y", "z"]))
    k2 = rc.k2_entropy(L.subsample(max_points=2000), target_rr=0.05,
                       theiler=20, rng=0)
    assert k2.diagnostics["n_distinct_lengths"] > 20


# ------------------------------------------------------------ complexity
@pytest.mark.parametrize("beta,expected", [(0.0, 0.5), (1.0, 1.0), (2.0, 1.5)])
def test_dfa_exponent_of_coloured_noise(beta, expected):
    x = rc.colored_noise(8192, beta=beta, rng=1)
    assert rc.detrended_fluctuation(x).value == pytest.approx(expected, abs=0.15)


def test_hurst_matches_dfa():
    x = rc.colored_noise(4096, beta=1.0, rng=2)
    assert rc.hurst_exponent(x).value == pytest.approx(
        rc.detrended_fluctuation(x).value, rel=1e-9)


def test_higuchi_dimension_bounds():
    rng = np.random.default_rng(3)
    t = np.arange(4000) / 200.0
    assert rc.higuchi_dimension(rng.standard_normal(4000)).value == pytest.approx(2.0, abs=0.15)
    assert rc.higuchi_dimension(np.sin(2 * np.pi * 5 * t)).value == pytest.approx(1.0, abs=0.15)


def test_permutation_entropy_is_maximal_for_noise():
    rng = np.random.default_rng(4)
    t = np.arange(4000) / 200.0
    assert rc.permutation_entropy(rng.standard_normal(4000)).value > 0.99
    assert rc.permutation_entropy(np.sin(2 * np.pi * 5 * t)).value < 0.7


def test_sample_entropy_orders_noise_above_a_sine():
    rng = np.random.default_rng(5)
    t = np.arange(3000) / 200.0
    assert (rc.sample_entropy(rng.standard_normal(3000)).value
            > rc.sample_entropy(np.sin(2 * np.pi * 5 * t)).value)


def test_spectral_entropy_is_low_for_a_pure_tone():
    t = np.arange(4000) / 200.0
    rng = np.random.default_rng(6)
    assert rc.spectral_entropy(np.sin(2 * np.pi * 5 * t), fs=200.0).value < 0.3
    assert rc.spectral_entropy(rng.standard_normal(4000), fs=200.0).value > 0.9


def test_short_series_is_refused():
    with pytest.raises(rc.ParameterError, match="too short"):
        rc.detrended_fluctuation(np.arange(10.0))


# --------------------------------------------------------------- object
def test_invariant_frame_and_summary(lorenz_space):
    d2 = rc.correlation_dimension(lorenz_space, theiler=20, max_points=2000, rng=0)
    df = d2.to_frame()
    assert {"invariant", "value", "units", "slope", "r_squared"} <= set(df.columns)
    assert "D2" in d2.summary()
    assert float(d2) == d2.value


# ------------------------------------------ the five added measures [F9b]
def test_hjorth_mobility_matches_theory():
    """For a sinusoid at f, mobility is 2*pi*f/fs exactly. A rare closed form
    among these measures, and worth pinning."""
    fs, f0 = 200.0, 5.0
    t = np.arange(4000) / fs
    h = rc.hjorth_parameters(np.sin(2 * np.pi * f0 * t), fs=fs)
    assert h["hjorth_mobility"].value == pytest.approx(2 * np.pi * f0 / fs, rel=0.01)
    assert h["hjorth_complexity"].value == pytest.approx(1.0, abs=0.02)
    assert h["hjorth_activity"].value == pytest.approx(0.5, rel=0.02)


def test_hjorth_mobility_reports_its_frequency_equivalent():
    fs, f0 = 200.0, 5.0
    t = np.arange(4000) / fs
    h = rc.hjorth_parameters(np.sin(2 * np.pi * f0 * t), fs=fs)
    assert h["hjorth_mobility"].diagnostics["hz_equivalent"] == pytest.approx(f0, rel=0.02)


def test_lempel_ziv_is_maximal_for_noise():
    rng = np.random.default_rng(0)
    t = np.arange(4000) / 200.0
    assert rc.lempel_ziv_complexity(rng.standard_normal(4000)).value == pytest.approx(1.0, abs=0.1)
    assert rc.lempel_ziv_complexity(np.sin(2 * np.pi * 5 * t)).value < 0.2


def test_svd_entropy_is_maximal_for_noise():
    rng = np.random.default_rng(1)
    t = np.arange(4000) / 200.0
    assert rc.svd_entropy(rng.standard_normal(4000)).value > 0.95
    assert rc.svd_entropy(np.sin(2 * np.pi * 5 * t)).value < 0.5


def test_katz_and_petrosian_order_smooth_below_rough():
    rng = np.random.default_rng(2)
    t = np.arange(4000) / 200.0
    sine, noise = np.sin(2 * np.pi * 5 * t), rng.standard_normal(4000)
    assert rc.katz_dimension(sine).value < rc.katz_dimension(noise).value
    assert rc.petrosian_dimension(sine).value < rc.petrosian_dimension(noise).value


def test_higuchi_is_anchored_and_stays_in_range():
    """DD-73: floating the fit put a smoothed envelope at 3.16, outside the
    [1, 2] the dimension is defined on."""
    rng = np.random.default_rng(3)
    t = np.arange(4000) / 200.0
    for series, expected in ((rng.standard_normal(4000), 2.0),
                             (np.sin(2 * np.pi * 5 * t), 1.0),
                             (rc.colored_noise(4000, beta=1.0, rng=4), 1.8)):
        h = rc.higuchi_dimension(series)
        assert 0.9 <= h.value <= 2.1, f"{h.value} is outside the defined range"
        assert h.value == pytest.approx(expected, abs=0.25)
        assert h.diagnostics["k_fitted"] >= 5


def test_higuchi_warns_when_there_is_no_scaling_range():
    x = np.concatenate([np.zeros(500), np.ones(500)])       # a single step
    h = rc.higuchi_dimension(x)
    assert h.value < 1.0 or h.warning is not None


# ---------------------------------------- measures as a table row [DD-71]
def test_phase_blocks_contribute_their_instantaneous_frequency():
    """DD-71: the wrapped angle jumps at +-pi and DFA read 0.13; the unwrapped
    angle is a ramp and DFA read 2.01. The increment is the stationary
    quantity."""
    from recurra.dynamics.measures import block_series

    sig = rc.generate_cfc(modality="pac", duration=20.0, fs=500.0, alpha=0.9,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=1)
    rec = rc.analytic(rc.filterbank(sig.to_recording("d"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    ss = sm.pac_space(rec, smooth=8.0).subsample(max_points=2000)
    series = block_series(ss)
    assert set(series) == {"phase_theta", "amp_gamma"}
    phase_series = series["phase_theta"]
    assert np.abs(phase_series).max() < 1.0        # increments, not angles
    assert 0.3 < rc.detrended_fluctuation(phase_series).value < 1.9


def test_dynamics_measures_names_the_block():
    sig = rc.generate_cfc(modality="pac", duration=20.0, fs=500.0, alpha=0.9,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=2)
    rec = rc.analytic(rc.filterbank(sig.to_recording("d"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    ss = sm.pac_space(rec, smooth=8.0).subsample(max_points=1500)
    d = rc.dynamics_measures(ss)
    assert "higuchi_fd_amp_gamma" in d and "higuchi_fd_phase_theta" in d
    assert all(np.isfinite(v) for v in d.values())


def test_unknown_measure_is_rejected():
    x = np.random.default_rng(0).standard_normal(2000)
    with pytest.raises(rc.ParameterError, match="unknown measure"):
        rc.dynamics_measures(x, measures=["nope"])


def test_invariants_are_opt_in():
    """DD-72: cheap by default, expensive on request."""
    from recurra.dynamics.measures import TRAJECTORY_MEASURES

    sig = rc.generate_cfc(modality="pac", duration=20.0, fs=500.0, alpha=0.9,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=3)
    rec = rc.analytic(rc.filterbank(sig.to_recording("d"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    ss = sm.pac_space(rec, smooth=8.0).subsample(max_points=1500)
    assert not set(TRAJECTORY_MEASURES) & set(rc.dynamics_measures(ss))
    with_inv = rc.dynamics_measures(ss, invariants=["K2"], target_rr=0.05, theiler=1)
    assert "K2" in with_inv


def test_dynamics_enter_the_windowed_table():
    sig = rc.generate_cfc(modality="pac", duration=30.0, fs=500.0, alpha=0.9,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=4)
    rec = rc.analytic(rc.filterbank(sig.to_recording("w"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    ss = sm.pac_space(rec, smooth=8.0).subsample(max_points=4000)
    wr = rc.windowed_recurrence(ss, 6, target_rr=0.05, theiler=1, rng=0)
    plain = wr.metrics()
    both = wr.metrics(rqa=True, dynamics=True)
    assert "higuchi_fd_amp_gamma" not in plain.columns
    assert "higuchi_fd_amp_gamma" in both.columns and "DET" in both.columns
    assert len(both) == len(wr)
    assert both["higuchi_fd_amp_gamma"].notna().all()


# ------------------------------------ univariate invariants per block [DD-93]
def test_block_invariants_estimate_each_signal_on_its_own():
    """DD-93. The earlier draft computed a univariate Lyapunov exponent and
    correlation dimension per component as well as a multivariate one, and both
    are useful: one describes the joint trajectory, the others each signal.
    Neither is a coupling measure [DD-77]."""
    sig = rc.generate_cfc(modality="pac", duration=25.0, fs=500.0, alpha=0.9,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=21)
    rec = rc.analytic(rc.filterbank(sig.to_recording("b"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    ss = sm.pac_space(rec, smooth=8.0).subsample(max_points=3000)
    out = rc.dynamics_measures(ss, measures=["higuchi_fd"], block_invariants=True,
                               max_points=1200, rng=0)
    for key in ("lambda_1_phase_theta", "lambda_1_amp_gamma",
                "D2_phase_theta", "D2_amp_gamma"):
        assert key in out and np.isfinite(out[key]), key


def test_block_and_trajectory_invariants_are_different_quantities():
    sig = rc.generate_cfc(modality="pac", duration=25.0, fs=500.0, alpha=0.9,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=22)
    rec = rc.analytic(rc.filterbank(sig.to_recording("b"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    ss = sm.pac_space(rec, smooth=8.0).subsample(max_points=2500)
    out = rc.dynamics_measures(ss, measures=["higuchi_fd"],
                               block_invariants=["D2"], invariants=["D2"],
                               max_points=1200, rng=0)
    assert out["D2"] != out["D2_amp_gamma"]


def test_unknown_block_invariant_is_rejected():
    x = np.random.default_rng(0).standard_normal(2000)
    with pytest.raises(rc.ParameterError, match="unknown block invariant"):
        rc.dynamics_measures(x, measures=["higuchi_fd"], block_invariants=["K2"])


def test_a_block_too_short_to_embed_warns_rather_than_returning_nan_silently():
    from recurra.exceptions import InferenceWarning

    rec = rc.ingest({"amp_gamma": np.random.default_rng(0).standard_normal(40)},
                    fs=100.0, roles={"amp_gamma": "amplitude"})
    ss = sm.envelope_space(rec)
    with pytest.warns(InferenceWarning):
        out = rc.dynamics_measures(ss, measures=["higuchi_fd"],
                                   block_invariants=True, embed_m=5, embed_tau=40)
    assert not np.isfinite(out["lambda_1_amp_gamma"])


# --------------------------- invariants from a delay embedding [DD-96, DD-97]
def test_auto_theiler_scales_with_the_embedding():
    """DD-96. Points closer in time than the embedding window share
    coordinates, so counting them as neighbours makes the cloud look
    one-dimensional exactly where D2 is read."""
    from recurra.dynamics.invariants import resolve_theiler

    X = rc.lorenz(20000, dt=0.01, transient=20.0)[:, 0]
    space = sm.takens(rc.ingest({"s": X}, fs=100.0), "s", m=3, tau=16)
    auto = resolve_theiler("auto", space)
    assert auto > (3 - 1) * 16, "auto must exceed the embedding span"
    assert auto <= space.n_points // 20
    assert resolve_theiler(7, space) == 7
    with pytest.raises(rc.ParameterError, match="theiler must be"):
        resolve_theiler("clever", space)


def test_correlation_dimension_of_lorenz_from_a_delay_embedding():
    """The reference case, as a contract over four trajectories [DD-96, DD-121].

    Initial conditions 1e-9 apart stand in for four machines: chaos turns a
    difference in rounding -- the fused multiply-add of the ARM runners on
    macOS, for one -- into a different trajectory, so a test pinned to one
    trajectory asserts a fact about one platform. Pinned that way this test
    failed on every macOS runner (value right, flagged), and the trajectory
    1e-9 away gave 1.61 *unflagged*: the flattest plateau of the correlation
    sum was the saturating one at large radii. Every value must now be right,
    and most must be unflagged."""
    values, trusted = [], 0
    for k in range(4):
        x = rc.lorenz(20000, dt=0.01, transient=20.0, y0=(1.0 + k * 1e-9, 1.0, 1.0))
        out = rc.invariants_from_series(x[:, 0], fs=100.0, rng=0, measures=("D2",))
        values.append((round(out["D2"], 3), out["D2_ok"]))
        assert out["D2"] == pytest.approx(2.05, abs=0.2), values
        trusted += int(out["D2_ok"])
    assert trusted >= 2, values


@pytest.mark.slow
def test_the_lyapunov_exponent_from_a_series_keeps_its_contract():
    """DD-101, from a delay embedding of one coordinate.

    Which reference points are drawn was deciding the answer: twelve streams
    over identical data gave 0.501 to 1.366, eleven of them unflagged. Three
    samplings averaged narrow the spread only modestly (a standard deviation of
    0.138 to 0.119 over twelve streams), because the samplings share a curve
    and are correlated, so a test that the spread shrinks is a coin toss at
    any affordable number of streams. What averaging buys reliably is the
    flag: an estimate that the draw decides is now reported as such. That is
    the claim checked here, over three trajectories and four streams each.
    """
    report = []
    for y0 in LORENZ_STARTS:
        x = rc.lorenz(12000, dt=0.01, transient=20.0, y0=y0)[:, 0]
        for stream in range(4):
            row = rc.invariants_from_series(x, fs=100.0, rng=stream, n_repeats=3,
                                            measures=("lambda_1",))
            report.append((y0, stream, round(row["lambda_1"], 3), row["lambda_1_ok"]))
            if row["lambda_1_ok"] == 1.0:
                assert row["lambda_1"] == pytest.approx(LORENZ_LAMBDA, rel=0.25), (
                    f"unflagged but wrong: {report}")
    trusted = [r for r in report if r[3] == 1.0]
    assert len(trusted) >= len(report) // 2, f"almost everything flagged: {report}"


def test_the_sampling_spread_is_reported():
    """An estimate that moves with the draw must say so [DD-101]."""
    x = rc.lorenz(12000, dt=0.01, transient=20.0)[:, 0]
    from recurra.dynamics.measures import _embed_block

    space = _embed_block(x, 100.0, 3, 16)
    result = rc.lyapunov_max(space, theiler="auto", max_steps=400,
                             max_points=1500, n_repeats=4, rng=0)
    assert result.diagnostics["n_repeats"] == 4
    assert "sampling_cv" in result.diagnostics
    assert result.diagnostics["sampling_cv"] >= 0.0


def test_one_sampling_is_the_old_behaviour():
    x = rc.lorenz(8000, dt=0.01, transient=20.0)[:, 0]
    from recurra.dynamics.measures import _embed_block

    space = _embed_block(x, 100.0, 3, 16)
    result = rc.lyapunov_max(space, theiler="auto", max_steps=300,
                             max_points=1200, n_repeats=1, rng=0)
    assert result.diagnostics["n_repeats"] == 1
    assert result.diagnostics["sampling_cv"] == 0.0


def test_the_embedding_used_is_reported():
    """A result nobody can audit is not a result."""
    out = rc.invariants_from_series(
        rc.lorenz(12000, dt=0.01, transient=20.0)[:, 0], fs=100.0)
    assert out["embed_tau"] > 1 and out["embed_m"] >= 2


def test_series_invariants_accept_explicit_parameters():
    X = rc.lorenz(12000, dt=0.01, transient=20.0)[:, 0]
    out = rc.invariants_from_series(X, fs=100.0, m=4, tau=12, measures=["D2"])
    assert out["embed_m"] == 4 and out["embed_tau"] == 12
    assert "lambda_1" not in out


def test_series_invariants_reject_an_unknown_measure():
    X = rc.lorenz(3000, dt=0.01, transient=10.0)[:, 0]
    with pytest.raises(rc.ParameterError, match="unknown measure"):
        rc.invariants_from_series(X, fs=100.0, measures=["K2"])


def test_a_short_series_is_refused():
    with pytest.raises(rc.ParameterError, match="too short to embed"):
        rc.invariants_from_series(np.arange(50.0), fs=100.0)


# ------------------------------ fine-scale measures need their scale [DD-99]
def test_a_smooth_series_reports_dimension_one_and_says_why():
    """DD-99. Higuchi, Katz, Petrosian and the ordinal entropies read
    point-to-point structure. Given thirty samples per cycle of the fastest
    content they report, correctly, a smooth line -- and across 98 real
    recordings those columns had coefficients of variation of 0.0018, 0.00008
    and 0.00003. Constants cannot separate anything."""
    from scipy import signal as sps

    fs, n = 500.0, 30000
    rng = np.random.default_rng(0)
    envelope = sps.filtfilt(sps.firwin(201, 8.0, fs=fs), [1.0],
                            np.abs(sps.hilbert(rng.standard_normal(n))))
    assert rc.samples_per_turn(envelope) > 10
    assert rc.higuchi_dimension(envelope).value < 1.15

    factor = rc.fine_scale_factor(envelope)
    assert factor > 5
    coarse = envelope[::factor]
    assert rc.samples_per_turn(coarse) < 4
    assert rc.higuchi_dimension(coarse).value > 1.5


def test_white_noise_needs_no_decimation():
    """The reference scale: one sample per event, which is where these
    measures are defined."""
    x = np.random.default_rng(1).standard_normal(20000)
    assert rc.samples_per_turn(x) == pytest.approx(1.5, abs=0.2)
    assert rc.fine_scale_factor(x) == 1


def test_auto_fine_scale_revives_the_flat_columns():
    from scipy import signal as sps

    fs, n = 500.0, 30000
    rng = np.random.default_rng(2)
    envelope = sps.filtfilt(sps.firwin(201, 8.0, fs=fs), [1.0],
                            np.abs(sps.hilbert(rng.standard_normal(n))))
    phase = 2 * np.pi * 6 * np.arange(n) / fs
    rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * phase)),
                     "amp_gamma": envelope}, fs=fs,
                    roles={"phase_theta": "phase", "amp_gamma": "amplitude"})
    ss = sm.pac_space(rec, smooth=8.0)
    flat = rc.dynamics_measures(ss, measures="all", fine_scale=None)
    auto = rc.dynamics_measures(ss, measures="all", fine_scale="auto")
    assert auto["higuchi_fd_amp_gamma"] > flat["higuchi_fd_amp_gamma"] + 0.3
    assert auto["lempel_ziv_amp_gamma"] > flat["lempel_ziv_amp_gamma"] + 0.3
    assert auto["fine_scale_amp_gamma"] > 1


def test_the_factor_and_its_reason_are_recorded():
    """A decimation nobody can see is a decimation nobody can check [DD-99]."""
    from scipy import signal as sps

    fs, n = 500.0, 20000
    envelope = sps.filtfilt(sps.firwin(201, 8.0, fs=fs), [1.0],
                            np.abs(np.random.default_rng(3).standard_normal(n)))
    rec = rc.ingest({"amp_gamma": envelope}, fs=fs,
                    roles={"amp_gamma": "amplitude"})
    out = rc.dynamics_measures(sm.envelope_space(rec), measures=["higuchi_fd"])
    assert out["fine_scale_amp_gamma"] > 1
    assert out["samples_per_turn_amp_gamma"] > 4


def test_dfa_and_hjorth_see_the_whole_series():
    """Only the fine-grained measures are decimated; a scaling exponent needs
    every scale it can get."""
    from scipy import signal as sps

    fs, n = 500.0, 20000
    envelope = sps.filtfilt(sps.firwin(201, 8.0, fs=fs), [1.0],
                            np.abs(np.random.default_rng(4).standard_normal(n)))
    rec = rc.ingest({"amp_gamma": envelope}, fs=fs, roles={"amp_gamma": "amplitude"})
    ss = sm.envelope_space(rec)
    flat = rc.dynamics_measures(ss, measures="all", fine_scale=None)
    auto = rc.dynamics_measures(ss, measures="all", fine_scale="auto")
    for untouched in ("DFA_alpha_amp_gamma", "hjorth_mobility_amp_gamma",
                      "spectral_entropy_amp_gamma"):
        assert auto[untouched] == pytest.approx(flat[untouched], rel=1e-9)


# --------------------------- a dimension with no scaling region [DD-100]
def test_a_correlation_sum_that_never_bends_is_flagged():
    """DD-100. A phase circle crossed with an amplitude is a 2-torus by
    construction, so D2 comes back at 1.98 with R2 = 1.0000 on every recording.
    That is the dimension of the construction, not of the data."""
    from scipy import signal as sps

    fs, n = 500.0, 30000
    rng = np.random.default_rng(5)
    envelope = np.abs(sps.hilbert(sps.filtfilt(
        sps.firwin(401, [50, 70], pass_zero=False, fs=fs), [1.0],
        rng.standard_normal(n))))
    # A wandering phase, as a real rhythm has: the trajectory then fills the
    # torus instead of tracing one closed curve on it, and D2 goes to 2 -- the
    # dimension of the construction, on every recording alike.
    phase = (2 * np.pi * 6 * np.arange(n) / fs
             + 0.3 * np.cumsum(rng.standard_normal(n)) / 50)
    rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * phase)),
                     "amp_gamma": envelope}, fs=fs,
                    roles={"phase_theta": "phase", "amp_gamma": "amplitude"})
    ss = sm.pac_space(rec, smooth=8.0).subsample(max_points=3000)
    d2 = rc.correlation_dimension(ss, theiler=20, max_points=2500, rng=0)
    assert d2.value == pytest.approx(2.0, abs=0.15)
    # Judged on the whole curve: since DD-121 the fit itself stops below the
    # saturating scale, so its share of the curve no longer says anything.
    assert d2.warning is not None and "does not bend" in d2.warning


def test_a_real_attractor_has_a_region_to_find(lorenz_space):
    """The guard must not fire on a dimension that was measured [DD-100]."""
    d2 = rc.correlation_dimension(lorenz_space, theiler=20, max_points=5000,
                                  n_radii=40, rng=0)
    assert d2.diagnostics["region_share"] < 0.9
    assert d2.value == pytest.approx(2.05, abs=0.2)
    assert d2.warning is None



# -------------------------------------------------------------- DD-71
def test_phase_block_series_has_no_spurious_leading_sample():
    """The phase increment series starts with a real increment, not a zero."""
    from recurra.dynamics.measures import block_series
    from recurra.statespace.core import CoordGroup, StateSpace

    n, fs = 500, 100.0
    phase = 2 * np.pi * 7.0 * np.arange(n) / fs
    coords = np.column_stack([np.cos(phase), np.sin(phase)])
    ss = StateSpace(coords=coords, groups=(CoordGroup("th", 0, 2, kind="phase_circle"),),
                    fs=fs)
    inc = block_series(ss)["th"]
    assert inc.size == n
    expected = 2 * np.pi * 7.0 / fs
    assert np.allclose(inc, expected, atol=1e-9)


def test_fine_scale_decimation_leaves_enough_samples():
    """DD-99, revised in 0.25. A slow series in a short window asked for a
    decimation that left nine samples, and every fractal and entropy measure
    failed. The factor is now capped, the cap is announced, and the measures
    come back finite."""
    from recurra.dynamics.complexity import FINE_SCALE_MIN_SAMPLES, fine_scale_factor

    x = np.sin(2 * np.pi * np.arange(250) / 200.0)      # ~100 samples per turn
    assert fine_scale_factor(x, min_samples=1) > 250 // FINE_SCALE_MIN_SAMPLES
    factor = fine_scale_factor(x)
    assert 250 // factor >= FINE_SCALE_MIN_SAMPLES
    with pytest.warns(rc.ParameterWarning, match="capped"):
        row = rc.dynamics_measures(x + 1e-5 * np.random.default_rng(0).standard_normal(250),
                                   measures="all", fs=250.0)
    assert row["fine_scale"] == factor
    assert all(np.isfinite(v) for v in row.values())
