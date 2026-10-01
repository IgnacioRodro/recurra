"""Tests for F3: state space construction, all routes."""
import numpy as np
import pytest

import recurra as rc
from recurra import statespace as sm


@pytest.fixture(scope="module")
def rec_pac():
    s = rc.generate_cfc(modality="pac", duration=20.0, fs=500.0, alpha=0.9,
                        preferred_phase=np.pi / 2, snr_db=20.0, seed=17)
    r = rc.filterbank(s.to_recording("t01"), {"theta": (4, 8), "gamma": (50, 70)})
    return rc.analytic(r, sources=["theta", "gamma"])


@pytest.fixture(scope="module")
def rec_lorenz():
    X = rc.lorenz(n_points=6000, dt=0.01, transient=10.0)
    return rc.ingest(X, fs=100.0, labels=["x", "y", "z"], subject="lorenz")


# ------------------------------------------------------------------ core
def test_weighted_coords_reproduce_grouped_distance(rec_pac):
    """DD-21: Euclidean distance on weighted coords IS the grouped distance."""
    ss = sm.pac_space(rec_pac)
    W = ss.weighted_coords
    rng = np.random.default_rng(0)
    for _ in range(20):
        i, j = rng.integers(0, ss.n_points, 2)
        grouped = sum(g.weight * np.sum((ss.coords[i, g.slice] - ss.coords[j, g.slice]) ** 2)
                      for g in ss.groups)
        assert grouped == pytest.approx(float(np.sum((W[i] - W[j]) ** 2)), rel=1e-10)


def test_coords_are_read_only(rec_pac):
    ss = sm.pac_space(rec_pac)
    with pytest.raises(ValueError):
        ss.coords[0, 0] = 1.0


def test_groups_partition_the_columns(rec_pac):
    ss = sm.pac_space(rec_pac)
    assert sum(g.dim for g in ss.groups) == ss.dim
    assert [g.start for g in ss.groups] == [0, 2]


def test_phase_circle_lies_on_unit_circle(rec_pac):
    ss = sm.phase_circle(rec_pac, "phase_theta", scaling="none")
    r = np.linalg.norm(np.asarray(ss.coords), axis=1)
    np.testing.assert_allclose(r, 1.0, atol=1e-12)


# -------------------------------------------------------- observable route
def test_pac_space_shape_and_blocks(rec_pac):
    ss = sm.pac_space(rec_pac)
    assert ss.dim == 3 and ss.route == "observable"
    assert [g.kind for g in ss.groups] == ["phase_circle", "amplitude"]


def test_pac_space_picks_slow_phase_and_fast_amplitude(rec_pac):
    ss = sm.pac_space(rec_pac)
    assert ss.groups[0].source == "phase_theta"    # slowest phase
    assert ss.groups[1].source == "amp_gamma"      # fastest amplitude


def test_pac_space_refuses_when_bands_unknown():
    T = 4000
    r = rc.ingest({"phase_a": np.random.default_rng(6900).random(T), "phase_b": np.random.default_rng(6901).random(T),
                   "amp_a": np.random.default_rng(7000).random(T), "amp_b": np.random.default_rng(7001).random(T)},
                  fs=500.0,
                  roles={"phase_a": "phase", "phase_b": "phase",
                         "amp_a": "amplitude", "amp_b": "amplitude"})
    with pytest.raises(rc.ParameterError, match="cannot be identified"):
        sm.pac_space(r)


@pytest.mark.parametrize("builder,kwargs,expected_dim", [
    ("phase_circle", {"phase": "phase_theta"}, 2),
    ("pac_space", {}, 3),
    ("envelope_space", {}, 2),
    ("freq_amp_space", {"inst_freq": "ifreq_theta", "amplitude": "amp_gamma"}, 2),
])
def test_observable_builders(rec_pac, builder, kwargs, expected_dim):
    ss = getattr(sm, builder)(rec_pac, **kwargs)
    assert ss.dim == expected_dim and ss.route == "observable"


def test_ppc_space_has_two_circles(rec_pac):
    ss = sm.ppc_space(rec_pac, "phase_theta", "phase_gamma")
    assert ss.dim == 4
    assert all(g.kind == "phase_circle" for g in ss.groups)


def test_ppa_space_is_five_dimensional(rec_pac):
    ss = sm.ppa_space(rec_pac, "phase_theta", "phase_gamma", "amp_gamma")
    assert ss.dim == 5


def test_channels_needs_no_coupling(rec_lorenz):
    """Nothing forces the CFC reading."""
    ss = sm.channels(rec_lorenz, ["x", "y", "z"])
    assert ss.dim == 3 and ss.n_points == rec_lorenz.n_samples


def test_custom_callable(rec_pac):
    ss = sm.custom(rec_pac, lambda r: np.column_stack(
        [r.get("amp_gamma"), r.get("amp_theta")]))
    assert ss.dim == 2 and ss.route == "external"


# ------------------------------------------------------------- delay route
def test_takens_shape(rec_lorenz):
    ss = sm.takens(rec_lorenz, "x", m=3, tau=16)
    assert ss.dim == 3
    assert ss.n_points == rec_lorenz.n_samples - 2 * 16


def test_takens_auto_recovers_lorenz_dimension(rec_lorenz):
    """AMI + normalised FNN should find m=3 for Lorenz [DD-23]."""
    p = sm.estimate_embedding(np.asarray(rec_lorenz.get("x")), tau_method="ami",
                              m_method="fnn", tau_range=(1, 200), m_range=(1, 10), rng=0)
    assert p.m == 3


def test_takens_multivariate(rec_lorenz):
    ss = sm.takens_multivariate(rec_lorenz, ["x", "y"], m=2, tau=10)
    assert ss.dim == 4 and ss.route == "delay"


def test_impossible_embedding_is_rejected(rec_lorenz):
    with pytest.raises(rc.ParameterError, match="leaves"):
        sm.takens(rec_lorenz, "x", m=50, tau=1000)


# ------------------------------------------------------------ hybrid route
def test_hybrid_keeps_circle_and_embeds_envelope(rec_pac):
    ss = sm.pac_space(rec_pac, embed_amplitude=3, tau=8)
    assert ss.route == "hybrid" and ss.dim == 5
    assert ss.groups[0].kind == "phase_circle" and ss.groups[0].dim == 2
    assert ss.groups[1].kind == "delay" and ss.groups[1].dim == 3


def test_blocks_of_different_length_are_aligned_and_flagged(rec_pac):
    ss = sm.pac_space(rec_pac, embed_amplitude=3, tau=8)
    step = [s for s in ss.provenance if s.name == "statespace"][0]
    assert any("aligned to the shortest" in c for c in step.caveats)


def test_from_bands_end_to_end():
    s = rc.generate_cfc(modality="pac", duration=20.0, fs=500.0, alpha=0.8, seed=3)
    ss = sm.from_bands(s.to_recording(), [
        {"band": (4, 8), "as": "phase", "label": "theta"},
        {"band": (50, 70), "as": "amplitude", "label": "gamma"}])
    assert ss.dim == 3 and [g.kind for g in ss.groups] == ["phase_circle", "amplitude"]


# ---------------------------------------------------------------- scaling
def test_default_scaling_balances_blocks(rec_pac):
    ss = sm.pac_space(rec_pac)
    shares = ss.scale_frame()["variance_share"].to_numpy()
    assert shares == pytest.approx([0.5, 0.5], abs=1e-6)


def test_scaling_none_leaves_imbalance_visible(rec_pac):
    ss = sm.pac_space(rec_pac, scaling="none")
    assert ss.scaling == "none"
    assert all(g.weight == 1.0 for g in ss.groups)


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.5, 0.8, 1.0])
def test_reweight_sets_share_to_lambda(rec_pac, lam):
    ss = sm.pac_space(rec_pac, scaling="none").reweight(lam)
    assert ss.scale_frame()["variance_share"].iloc[1] == pytest.approx(lam, abs=1e-6)


def test_tail_ratio_of_unit_circle_is_sqrt_two(rec_pac):
    """DD-28: reference value for the shape diagnostic."""
    ss = sm.phase_circle(rec_pac, "phase_theta")
    assert ss.scale_frame()["tail_ratio"].iloc[0] == pytest.approx(np.sqrt(2), abs=0.05)


def test_heavy_tailed_block_is_flagged(rec_pac):
    ss = sm.pac_space(rec_pac)
    assert ss.scale_report.shape_warning is not None
    assert "tail" in ss.scale_report.shape_warning


def test_smoothing_is_recorded(rec_pac):
    ss = sm.pac_space(rec_pac, smooth=10.0)
    step = [s for s in ss.provenance if s.name == "statespace"][0]
    assert any("low-pass" in c for c in step.caveats)


# ------------------------------------------------------------- transforms
def test_subsample_records_caveat(rec_pac):
    ss = sm.pac_space(rec_pac)
    out = ss.subsample(4)
    assert out.n_points == int(np.ceil(ss.n_points / 4))
    assert any("decimated" in c for s in out.provenance for c in s.caveats)


def test_reduce_pca(rec_pac):
    ss = sm.pac_space(rec_pac).reduce("pca", 2)
    assert ss.dim == 2 and ss.groups[0].kind == "derived"


def test_split_extracts_blocks(rec_pac):
    ss = sm.ppc_space(rec_pac, "phase_theta", "phase_gamma")
    part = ss.split(["phase_theta"])
    assert part.dim == 2 and part.block_labels == ["phase_theta"]


def test_window_shifts_t0(rec_pac):
    ss = sm.pac_space(rec_pac)
    w = ss.window(1000, 3000)
    assert w.n_points == 2000
    assert w.t0 == pytest.approx(ss.t0 + 1000 / ss.fs)


# --------------------------------------------------------------- compare
def test_compare_routes_returns_row_per_route(rec_pac):
    df, spaces = sm.compare_routes(rec_pac, rng=0)
    assert len(df) >= 3
    assert "nn_mean" in df.columns
    ok = df[df.status == "ok"]
    assert len(ok) >= 2


def test_lambda_sweep_needs_two_blocks(rec_pac):
    ss = sm.ppa_space(rec_pac, "phase_theta", "phase_gamma", "amp_gamma")
    with pytest.raises(rc.ParameterError, match="exactly 2 blocks"):
        sm.lambda_sweep(ss)


def test_lambda_sweep_endpoints_are_pure(rec_pac):
    df = sm.lambda_sweep(sm.pac_space(rec_pac, scaling="none"), grid=[0.0, 0.5, 1.0], rng=0)
    assert df["nn_agreement_vs_block0"].iloc[0] == pytest.approx(1.0)
    assert df["nn_agreement_vs_block1"].iloc[-1] == pytest.approx(1.0)


def test_neighbour_agreement_is_one_for_identical(rec_pac):
    ss = sm.pac_space(rec_pac)
    assert sm.neighbour_agreement(ss, ss, rng=0) == pytest.approx(1.0)


# ------------------------------------------------------------------- viz
@pytest.mark.requires("matplotlib")
def test_plot_modes_render(rec_pac):
    ss = sm.pac_space(rec_pac)
    import matplotlib.pyplot as plt

    for mode in ("2d", "3d", "pairs", "time", "torus"):
        fig = rc.plot_attractor(ss, mode=mode, max_points=400)
        assert fig is not None
        plt.close(fig)


@pytest.mark.requires("matplotlib")
def test_torus_mode_needs_a_phase_circle(rec_lorenz):
    ss = sm.channels(rec_lorenz, ["x", "y", "z"])
    with pytest.raises(rc.ParameterError, match="phase_circle"):
        rc.plot_attractor(ss, mode="torus")


@pytest.mark.requires("matplotlib")
def test_save_attractor_writes(tmp_path, rec_pac):
    rc.set_config(output_dir=str(tmp_path))
    p = rc.save_attractor(sm.pac_space(rec_pac), "a.png", mode="3d", max_points=300)
    import os

    assert os.path.exists(p)


# ------------------------------------------- the corrected DD-02 hypothesis
def test_lambda_sweep_endpoints_are_uninformative_about_coupling():
    """Both ends of the sweep are single-block projections, so a coupled and
    an uncoupled signal must agree there [DD-02]."""
    slopes = {}
    for alpha, label in [(0.0, "uncoupled"), (0.95, "coupled")]:
        s = rc.generate_cfc(modality="pac", duration=30.0, fs=500.0, alpha=alpha,
                            preferred_phase=np.pi / 2, snr_db=20.0, seed=17)
        r = rc.analytic(rc.filterbank(s.to_recording("p"),
                                      {"theta": (4, 8), "gamma": (50, 70)}),
                        sources=["theta", "gamma"])
        base = sm.pac_space(r, scaling="none").subsample(max_points=2000)
        slopes[label] = [
            sm.geometry_descriptors(base.reweight(lam), rng=0,
                                    max_points=1200)["corr_slope"]
            for lam in (0.0, 1.0)
        ]
    for k in (0, 1):
        assert slopes["uncoupled"][k] == pytest.approx(slopes["coupled"][k], abs=0.15)


def test_coupling_lowers_the_joint_dimension():
    """The corrected sign of DD-02: coupling is a constraint, and a constraint
    removes dimension. The uncoupled signal, whose blocks form a cartesian
    product, must show the *larger* interior slope."""
    interior = {}
    for alpha, label in [(0.0, "uncoupled"), (0.95, "coupled")]:
        s = rc.generate_cfc(modality="pac", duration=30.0, fs=500.0, alpha=alpha,
                            preferred_phase=np.pi / 2, snr_db=20.0, seed=17)
        r = rc.analytic(rc.filterbank(s.to_recording("p"),
                                      {"theta": (4, 8), "gamma": (50, 70)}),
                        sources=["theta", "gamma"])
        base = sm.pac_space(r, scaling="none").subsample(max_points=2000)
        interior[label] = float(np.mean([
            sm.geometry_descriptors(base.reweight(lam), rng=0,
                                    max_points=1200)["corr_slope"]
            for lam in (0.5, 0.7, 0.9)
        ]))
    assert interior["uncoupled"] > interior["coupled"] + 0.3


def test_lambda_sweep_reports_a_deficit_not_a_synergy(rec_pac):
    df = sm.lambda_sweep(sm.pac_space(rec_pac, scaling="none"),
                         grid=[0.0, 0.5, 1.0], rng=0, max_points=800)
    assert "corr_slope_deficit" in df.columns
    assert not any(c.endswith("_synergy") for c in df.columns)


# ---------------------------------------- the tail as a column [DD-102]
def test_shape_descriptors_report_the_tail_per_block():
    """DD-102. The warning existed from the start and nothing recorded how
    heavy the tail was, so nobody could check whether it differed between the
    groups being compared."""
    rng = np.random.default_rng(0)
    n, fs = 8000, 250.0
    t = np.arange(n) / fs
    phase = 2 * np.pi * 6 * t + 0.3 * np.cumsum(rng.standard_normal(n)) / 50

    ratios = {}
    for label, burst in (("mild", 0.0), ("heavy", 12.0)):
        env = np.abs(1 + 0.6 * np.cos(phase) + 0.3 * rng.standard_normal(n))
        env = env * (1 + burst * np.abs(rng.standard_normal(n)) ** 3
                     * (rng.random(n) < 0.02))
        rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * phase)),
                         "amp_gamma": env}, fs=fs,
                        roles={"phase_theta": "phase", "amp_gamma": "amplitude"})
        shape = rc.shape_descriptors(sm.pac_space(rec, smooth=8.0))
        ratios[label] = shape["tail_ratio_amp_gamma"]
        # the phase circle is a circle whatever the amplitude does
        assert shape["tail_ratio_phase_theta"] == pytest.approx(1.4, abs=0.2)
    assert ratios["heavy"] > 3 * ratios["mild"]


def test_the_tail_comes_free_with_the_geometry():
    """It is the quantity most likely to confound a group comparison, so it
    should not need asking for [DD-102]."""
    rng = np.random.default_rng(1)
    rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * np.linspace(0, 200, 6000))),
                     "amp_gamma": np.abs(rng.standard_normal(6000))}, fs=250.0,
                    roles={"phase_theta": "phase", "amp_gamma": "amplitude"})
    descriptors = sm.geometry_descriptors(sm.pac_space(rec, smooth=8.0), rng=0)
    assert "tail_ratio_amp_gamma" in descriptors
    assert "skew_amp_gamma" in descriptors
    assert "nn_cv" in descriptors            # the old ones are still there



# ------------------------------------------------------------- DD-113
def test_pac_space_warns_when_the_amplitude_band_cannot_hold_the_sidebands():
    """A 10 Hz band around 40 Hz cannot carry an 8 Hz modulation."""
    import warnings

    import recurra as rc
    from recurra import statespace as sm
    from recurra.exceptions import GeometryWarning

    n, fs = 4000, 250.0
    t = np.arange(n) / fs
    rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * 2 * np.pi * 6 * t)),
                     "amp_assr": 1.0 + 0.3 * np.cos(2 * np.pi * 6 * t)},
                    fs=fs, roles={"phase_theta": "phase", "amp_assr": "amplitude"},
                    meta={"bands": {"theta": (4.0, 8.0), "assr": (35.0, 45.0)}})
    with pytest.warns(GeometryWarning, match="sidebands"):
        sm.pac_space(rec, "phase_theta", "amp_assr", smooth=8.0)
    wide = rc.ingest({"phase_theta": np.angle(np.exp(1j * 2 * np.pi * 6 * t)),
                      "amp_assr": 1.0 + 0.3 * np.cos(2 * np.pi * 6 * t)},
                     fs=fs, roles={"phase_theta": "phase", "amp_assr": "amplitude"},
                     meta={"bands": {"theta": (4.0, 8.0), "assr": (30.0, 50.0)}})
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeometryWarning)
        sm.pac_space(wide, "phase_theta", "amp_assr", smooth=8.0)


# ------------------------------------------------------------- DD-112
def test_pac_space_warns_when_smoothing_cuts_the_phase_band():
    """An envelope low-pass below the phase band removes the coupling."""
    import warnings

    import recurra as rc
    from recurra import statespace as sm
    from recurra.exceptions import GeometryWarning

    n, fs = 4000, 250.0
    t = np.arange(n) / fs
    rec = rc.ingest({"phase_beta": np.angle(np.exp(1j * 2 * np.pi * 20 * t)),
                     "amp_gamma": 1.0 + 0.3 * np.cos(2 * np.pi * 20 * t)},
                    fs=fs, roles={"phase_beta": "phase", "amp_gamma": "amplitude"},
                    # wide enough for DD-113, so only the smoothing can warn here
                    meta={"bands": {"beta": (12.0, 30.0), "gamma": (30.0, 100.0)}})
    with pytest.warns(GeometryWarning, match="below the upper edge"):
        sm.pac_space(rec, "phase_beta", "amp_gamma", smooth=8.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeometryWarning)
        sm.pac_space(rec, "phase_beta", "amp_gamma", smooth=30.0)
        sm.pac_space(rec, "phase_beta", "amp_gamma", smooth=None)


def test_gaussian_smoothing_keeps_the_envelope_non_negative():
    """DD-117. The Butterworth smoother rang below zero on a strongly modulated
    envelope (4.7% of samples at alpha 0.9). A non-negative kernel cannot."""
    sig = rc.generate_cfc(modality="pac", duration=20.0, fs=250.0, alpha=0.9,
                          preferred_phase=np.pi / 2, snr_db=15.0, seed=0)
    r = rc.analytic(rc.filterbank(sig.to_recording("s"), {"theta": (4, 8),
                                                          "gamma": (50, 70)}),
                    sources=["theta", "gamma"])
    gauss = sm.pac_space(r, smooth=8.0)
    butter = sm.pac_space(r, smooth=8.0, smooth_method="butterworth")
    assert gauss.groups[-1].params["smooth_method"] == "gaussian"
    assert gauss.coords[:, -1].min() >= 0.0
    assert butter.coords[:, -1].min() < 0.0      # the failure the change removes


@pytest.mark.parametrize("method", ["gaussian", "butterworth"])
def test_both_smoothers_halve_the_amplitude_at_the_cut_off(method):
    """The cut-off means the same thing for both methods: gain 1/2."""
    from recurra.statespace.builders import _smooth

    fs, fc = 250.0, 8.0
    t = np.arange(int(40 * fs)) / fs
    x = (2.0 + np.sin(2 * np.pi * fc * t)).reshape(-1, 1)
    y = _smooth(x, fc, fs, method=method)[:, 0]
    mid = slice(len(t) // 4, 3 * len(t) // 4)          # away from the edges
    gain = (y[mid].max() - y[mid].min()) / 2.0
    assert gain == pytest.approx(0.5, abs=0.02)


def test_unknown_smoothing_method_is_refused():
    from recurra.statespace.builders import _smooth

    with pytest.raises(rc.ParameterError, match="smooth_method"):
        _smooth(np.ones((100, 1)), 5.0, 100.0, method="boxcar")


# ------------------------------------------------ embedding estimation, checked
def test_ami_matches_an_independent_histogram_estimate():
    """Average mutual information against a plain 2-D histogram computation."""
    from recurra.statespace.embedding import mutual_information

    x = rc.lorenz(n_points=6000, dt=0.01)[:, 0]
    edges = np.histogram_bin_edges(x, 32)
    for lag in (1, 8, 16, 30):
        h, _, _ = np.histogram2d(x[:-lag], x[lag:], bins=[edges, edges])
        p = h / h.sum()
        px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        nz = p > 0
        ref = float(np.sum(p[nz] * np.log(p[nz] / (px @ py)[nz])))
        assert mutual_information(x, lag, 32) == pytest.approx(ref, rel=1e-9)


def test_lorenz_embedding_is_recovered():
    """Ground truth: tau about 16 samples at dt=0.01 by AMI, and m=3 by FNN."""
    x = rc.lorenz(n_points=8000, dt=0.01)[:, 0]
    tau = sm.estimate_tau(x, method="ami", tau_range=(1, 100)).tau
    assert 13 <= tau <= 19
    assert sm.estimate_m(x, tau, method="fnn", rng=0).m == 3


def test_fnn_handles_an_exactly_periodic_series():
    """A sine sampled at an integer number of samples per period repeats its
    points to within rounding. FNN used to count those duplicates as false
    neighbours and return the top of m_range (12); a circle needs 2."""
    x = np.sin(2 * np.pi * np.arange(4000) / 50.0)
    tau = sm.estimate_tau(x, method="ami", tau_range=(1, 40)).tau
    assert sm.estimate_m(x, tau, method="fnn", rng=0).m == 2


def test_an_estimate_at_the_edge_of_its_range_warns():
    """P3: a value forced by the search range, not by the data, says so."""
    x = rc.lorenz(n_points=8000, dt=0.01)[:, 0]
    with pytest.warns(rc.ParameterWarning, match="edge of the range"):
        sm.estimate_tau(x, method="first_zero", tau_range=(1, 30))
    noise = np.random.default_rng(0).standard_normal(3000)
    with pytest.warns(rc.ParameterWarning, match="never fell"):
        sm.estimate_m(noise, 1, method="fnn", m_range=(1, 3), rng=0)
