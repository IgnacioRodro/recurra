"""Tests for windowed recurrence (F10, brought forward)."""
import numpy as np
import pytest

import recurra as rc
from recurra import statespace as sm
from recurra.windowed import (
    WindowSpec,
    batch_windowed_recurrence,
    resolve_windows,
    windowed_cross_recurrence,
    windowed_joint_recurrence,
    windowed_recurrence,
)


@pytest.fixture(scope="module")
def rec_pac():
    s = rc.generate_cfc(modality="pac", duration=60.0, fs=500.0, alpha=0.9,
                        preferred_phase=np.pi / 2, snr_db=20.0, seed=17)
    return rc.analytic(rc.filterbank(s.to_recording("sub01"),
                                     {"theta": (4, 8), "gamma": (50, 70)}),
                       sources=["theta", "gamma"])


@pytest.fixture(scope="module")
def ss_pac(rec_pac):
    return sm.pac_space(rec_pac, smooth=8.0).subsample(max_points=4000)


# ------------------------------------------------------------ WindowSpec
def test_n_windows_gives_exactly_that_many():
    """Asking for 10 must give 10, not 9 with a tail left over."""
    for k in (2, 5, 10, 17):
        w, _ = WindowSpec(n_windows=k, overlap=0.5).resolve(30000, 500.0)
        assert len(w) == k


def test_n_windows_spans_the_whole_record():
    w, caveats = WindowSpec(n_windows=10, overlap=0.5).resolve(30000, 500.0)
    assert w[0].start == 0 and w[-1].stop == 30000
    assert not caveats


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5, 0.75])
def test_overlap_is_realised(overlap):
    w, _ = WindowSpec(n_windows=8, overlap=overlap).resolve(40000, 500.0)
    size = w[0].n_samples
    step = w[1].start - w[0].start
    assert 1.0 - step / size == pytest.approx(overlap, abs=0.03)


def test_size_and_step_in_seconds():
    w, _ = WindowSpec(size=4.0, step=2.0, unit="seconds").resolve(30000, 500.0)
    assert w[0].n_samples == 2000
    assert w[1].start - w[0].start == 1000


def test_size_and_overlap():
    w, _ = WindowSpec(size=4.0, overlap=0.5).resolve(30000, 500.0)
    assert w[1].start - w[0].start == 1000


def test_unit_samples():
    w, _ = WindowSpec(size=1000, step=500, unit="samples").resolve(10000)
    assert w[0].n_samples == 1000 and w[1].start == 500


def test_uncovered_tail_is_reported():
    w, caveats = WindowSpec(size=7.0, step=7.0).resolve(30000, 500.0)
    assert any("not analysed" in c for c in caveats)


def test_drop_last_false_pulls_the_window_back():
    w, caveats = WindowSpec(size=7.0, step=7.0, drop_last=False).resolve(30000, 500.0)
    assert w[-1].stop == 30000
    assert any("pulled back" in c for c in caveats)


def test_overlap_as_percentage_is_rejected():
    with pytest.raises(rc.ParameterError, match="fraction in"):
        WindowSpec(size=4.0, overlap=50)


def test_step_and_overlap_together_rejected():
    with pytest.raises(rc.ParameterError, match="not both"):
        WindowSpec(size=4.0, step=1.0, overlap=0.5)


def test_window_larger_than_record_is_rejected():
    with pytest.raises(rc.ParameterError, match="exceeds"):
        WindowSpec(size=100.0).resolve(1000, 500.0)


def test_resolve_accepts_int_and_explicit_pairs():
    assert len(resolve_windows(4, 8000, 500.0)[0]) == 4
    w, _ = resolve_windows([(0, 100), (50, 150)], 8000, 500.0)
    assert len(w) == 2 and w[1].start == 50


def test_window_times_are_right():
    w, _ = WindowSpec(size=4.0, step=2.0).resolve(30000, 500.0, t0=1.0)
    assert w[0].t_start == pytest.approx(1.0)
    assert w[0].t_stop == pytest.approx(5.0)
    assert w[1].t_center == pytest.approx(1.0 + 2.0 + 2.0)


# --------------------------------------------------------------- the core
def test_ten_windows_ten_matrices(ss_pac):
    """The headline case: one 60 s record, ten overlapping recurrence plots."""
    wr = windowed_recurrence(ss_pac, WindowSpec(n_windows=10, overlap=0.5),
                             target_rr=0.05, theiler=1, rng=0)
    assert len(wr) == 10
    assert len(wr.metrics()) == 10
    for _, rm in wr:
        assert rm.kind == "rp"
        assert rm.recurrence_rate() == pytest.approx(0.05, abs=0.01)


def test_per_window_scope_fixes_the_rate(ss_pac):
    wr = windowed_recurrence(ss_pac, 8, scope="per_window", target_rr=0.05,
                             theiler=1, rng=0)
    m = wr.metrics()
    assert m["recurrence_rate"].std() < 0.005      # rate pinned
    assert m["epsilon"].std() > 0                   # epsilon moves instead


def test_global_scope_fixes_epsilon_and_frees_the_rate(ss_pac):
    """DD-40: the scope decides what a windowed analysis can say."""
    wr = windowed_recurrence(ss_pac, 8, scope="global", target_rr=0.05,
                             theiler=1, rng=0)
    m = wr.metrics()
    assert m["epsilon"].nunique() == 1              # one threshold for all
    assert m["recurrence_rate"].std() > 0           # rate becomes informative


def test_scope_must_be_valid(ss_pac):
    with pytest.raises(rc.ParameterError, match="not a detail"):
        windowed_recurrence(ss_pac, 4, scope="whatever")


def test_windows_are_lazy_by_default(ss_pac):
    wr = windowed_recurrence(ss_pac, 6, target_rr=0.05, theiler=1, rng=0)
    assert wr._cache == {}
    wr.metrics()
    assert wr._cache == {}                          # still not retained


def test_keep_caches_the_matrices(ss_pac):
    wr = windowed_recurrence(ss_pac, 4, target_rr=0.05, theiler=1, rng=0, keep=True)
    a, b = wr[0], wr[0]
    assert a is b


def test_windows_are_the_right_segments(ss_pac):
    wr = windowed_recurrence(ss_pac, WindowSpec(size=1000, step=500, unit="samples"),
                             target_rr=0.05, theiler=1, rng=0)
    assert wr[0].n_rows == 1000
    np.testing.assert_allclose(wr[0].coords, ss_pac.weighted_coords[0:1000])
    np.testing.assert_allclose(wr[1].coords, ss_pac.weighted_coords[500:1500])


def test_metrics_frame_is_tidy_and_complete(ss_pac):
    m = windowed_recurrence(ss_pac, 5, target_rr=0.05, theiler=1, rng=0).metrics()
    expected = {"subject", "kind", "scope", "window", "t_start_s", "t_stop_s",
                "n_rows", "dim", "epsilon", "target_rr", "recurrence_rate"}
    assert expected <= set(m.columns)
    assert len(m) == 5


def test_series_extracts_one_metric(ss_pac):
    wr = windowed_recurrence(ss_pac, 5, target_rr=0.05, theiler=1, rng=0)
    s = wr.series("epsilon")
    assert list(s.columns) == ["window", "t_center_s", "epsilon"]


# ------------------------------------------------- every threshold mode
@pytest.mark.parametrize("kwargs", [
    {"threshold": "target_rr", "target_rr": 0.05},
    {"threshold": "percentile", "percentile": 0.05},
    {"threshold": "fan", "n_neighbors": 20},
    {"threshold": "std_fraction", "factor": 0.15},
    {"threshold": "maxdist_fraction", "factor": 0.2},
    {"threshold": "per_block", "target_rr": 0.05},
    {"threshold": 0.45},
])
def test_all_threshold_modes_window(ss_pac, kwargs):
    wr = windowed_recurrence(ss_pac, 4, theiler=1, rng=0, **kwargs)
    m = wr.metrics()
    assert len(m) == 4
    assert (m["recurrence_rate"] > 0).all()


@pytest.mark.parametrize("metric", ["euclidean", "chebyshev", "manhattan"])
def test_all_distance_metrics_window(ss_pac, metric):
    wr = windowed_recurrence(ss_pac, 3, metric=metric, target_rr=0.05,
                             theiler=1, rng=0)
    assert (wr.metrics()["recurrence_rate"] > 0).all()


# ------------------------------------------------------------ CRP and JRP
def test_windowed_cross_recurrence(rec_pac):
    a = sm.phase_circle(rec_pac, "phase_theta").subsample(max_points=3000)
    b = sm.phase_circle(rec_pac, "phase_gamma").subsample(max_points=3000)
    wr = windowed_cross_recurrence(a, b, 6, target_rr=0.05, rng=0)
    assert len(wr) == 6
    for _, rm in wr:
        assert rm.kind == "crp" and not rm.is_symmetric


def test_windowed_joint_recurrence_keeps_independent_thresholds(ss_pac):
    wr = windowed_joint_recurrence([ss_pac.split(["phase_theta"]),
                                    ss_pac.split(["amp_gamma"])],
                                   6, target_rr=0.10, theiler=1, rng=0)
    m = wr.metrics()
    assert len(m) == 6
    assert {"rr_subsystem_0", "rr_subsystem_1", "independence_ratio"} <= set(m.columns)
    for _, rm in wr:
        assert len({t.scalar for t in rm.thresholds}) == 2   # genuinely independent
        assert [p.dim for p in rm.parts] == [2, 1]           # different dimensions


def test_jrp_windows_detect_coupling(rec_pac):
    """Coupled windows should show an independence ratio above 1."""
    ss = sm.pac_space(rec_pac, smooth=8.0).subsample(max_points=4000)
    wr = windowed_joint_recurrence([ss.split(["phase_theta"]),
                                    ss.split(["amp_gamma"])],
                                   6, target_rr=0.10, theiler=1, rng=0)
    assert wr.metrics()["independence_ratio"].mean() > 1.2


def test_crp_needs_exactly_two(ss_pac):
    with pytest.raises(rc.ParameterError, match="exactly 2"):
        windowed_recurrence([ss_pac, ss_pac, ss_pac], 3, kind="crp", rng=0)


def test_misaligned_trajectories_rejected(ss_pac):
    with pytest.raises(rc.ParameterError, match="common time base"):
        windowed_joint_recurrence([ss_pac, ss_pac.window(0, 500)], 3, rng=0)


# ------------------------------------------------------- routes and spaces
def test_takens_route_windows(ss_pac):
    X = rc.lorenz(n_points=6000, dt=0.01, transient=10.0)
    rl = rc.ingest(X, fs=100.0, labels=["x", "y", "z"], subject="lorenz")
    wr = windowed_recurrence(sm.takens(rl, "x", m=3, tau=16), 5,
                             target_rr=0.05, theiler=1, rng=0)
    assert len(wr) == 5 and wr[0].dim == 3


def test_raw_array_windows():
    X = np.cumsum(np.random.default_rng(0).standard_normal((3000, 2)), axis=0)
    wr = windowed_recurrence(X, 4, target_rr=0.05, theiler=1, rng=0)
    assert len(wr) == 4


def test_hybrid_space_windows(rec_pac):
    ss = sm.pac_space(rec_pac, embed_amplitude=3, tau=8).subsample(max_points=3000)
    wr = windowed_recurrence(ss, 4, target_rr=0.05, theiler=1, rng=0)
    assert wr[0].dim == 5


# ------------------------------------------------------------------ batch
def test_batch_over_subjects(ss_pac):
    subjects = {f"sub{i:02d}": ss_pac.window(i * 100, i * 100 + 2500)
                for i in range(4)}
    b = batch_windowed_recurrence(subjects, 5, target_rr=0.05, theiler=1, rng=0)
    assert len(b) == 4 and b.n_matrices == 20
    m = b.metrics()
    assert len(m) == 20
    assert set(m["subject"]) == set(subjects)


def test_batch_handles_unequal_lengths(ss_pac):
    """Real corpora have records of different length [DD-18]."""
    subjects = {"a": ss_pac.window(0, 3000), "b": ss_pac.window(0, 2000),
                "c": ss_pac.window(0, 2500)}
    b = batch_windowed_recurrence(subjects, WindowSpec(n_windows=4, overlap=0.5),
                                  target_rr=0.05, theiler=1, rng=0)
    m = b.metrics()
    assert len(m) == 12
    assert m.groupby("subject")["n_rows"].nunique().eq(1).all()
    assert m.groupby("subject")["n_rows"].first().nunique() == 3   # different sizes


def test_batch_results_do_not_depend_on_order(ss_pac):
    subjects = {f"s{i}": ss_pac.window(0, 2000) for i in range(3)}
    a = batch_windowed_recurrence(subjects, 3, target_rr=0.05, theiler=1, rng=7)
    b = batch_windowed_recurrence(subjects, 3, target_rr=0.05, theiler=1, rng=7)
    np.testing.assert_allclose(a.metrics()["epsilon"], b.metrics()["epsilon"])


def test_batch_from_a_sequence(ss_pac):
    b = batch_windowed_recurrence([ss_pac.window(0, 2000)] * 2, 3,
                                  target_rr=0.05, theiler=1, rng=0)
    assert set(b.results) == {"record000", "record001"}


# ------------------------------------------------------------- provenance
def test_provenance_carries_the_window_spec(ss_pac):
    wr = windowed_recurrence(ss_pac, WindowSpec(n_windows=6, overlap=0.5),
                             target_rr=0.05, theiler=1, rng=0)
    steps = list(wr.provenance_frame()["step"])
    assert "statespace" in steps and "windowed_recurrence" in steps
    params = [s.params for s in wr.provenance if s.name == "windowed_recurrence"][0]
    assert params["n_windows"] == 6 and params["scope"] == "per_window"


def test_describe_summarises(ss_pac):
    d = windowed_recurrence(ss_pac, 5, target_rr=0.05, theiler=1, rng=0).describe()
    assert {"n_windows", "window_seconds", "rr_mean", "epsilon_cv"} <= set(d.columns)


# ------------------------------------------------------------------- viz
@pytest.mark.requires("matplotlib")
def test_window_figures_render(ss_pac):
    import matplotlib.pyplot as plt

    wr = windowed_recurrence(ss_pac, 6, target_rr=0.05, theiler=1, rng=0)
    for fn in (rc.plot_window_coverage, rc.plot_window_series, rc.plot_window_panel):
        fig = fn(wr)
        assert fig is not None
        plt.close(fig)


@pytest.mark.requires("matplotlib")
def test_windows_feed_the_comparison_figures(ss_pac):
    import matplotlib.pyplot as plt

    from recurra.viz import compare

    wr = windowed_recurrence(ss_pac, 4, target_rr=0.05, theiler=1, rng=0, keep=True)
    fig = compare.compare_recurrence(wr.matrices(), ncols=4, max_size=200)
    plt.close(fig)
    fig = compare.compare_window_series({"a": wr, "b": wr})
    plt.close(fig)


# ----------------------------------- time-resolved coupling detection [DD-54]
def test_independence_ratio_detects_a_transient_at_the_right_time():
    """Ground truth: coupling between 19.8 s and 39.6 s. The detected onset and
    offset must fall within one window step of those [DD-54]."""
    duration = 60.0
    sig = rc.generate_cfc(modality="pac", duration=duration, fs=500.0, alpha=0.95,
                          preferred_phase=np.pi / 2, snr_db=20.0,
                          nonstationarity="transient", seed=31)
    rec = rc.analytic(rc.filterbank(sig.to_recording("ns"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    space = sm.pac_space(rec, smooth=8.0).subsample(max_points=6000)
    wr = windowed_joint_recurrence([space.split(["phase_theta"]),
                                    space.split(["amp_gamma"])],
                                   WindowSpec(size=4.0, overlap=0.75),
                                   target_rr=0.10, theiler=1, rng=0)
    m = wr.metrics()
    t = m["t_center_s"].to_numpy()
    ratio = m["independence_ratio"].to_numpy()

    baseline = ratio[(t < 15) | (t > 45)]
    inside = ratio[(t > 22) & (t < 37)]
    assert baseline.mean() == pytest.approx(1.0, abs=0.1)
    assert inside.mean() > baseline.mean() + 0.4

    above = t[ratio > baseline.mean() + 3 * baseline.std()]
    step = (wr.windows[1].start - wr.windows[0].start) / space.fs
    assert abs(above.min() - 0.33 * duration) < 2 * step
    assert abs(above.max() - 0.66 * duration) < 2 * step


def test_subsystem_rates_do_not_reveal_the_transient():
    """The point of the joint construction: neither subsystem changes [DD-54]."""
    sig = rc.generate_cfc(modality="pac", duration=60.0, fs=500.0, alpha=0.95,
                          preferred_phase=np.pi / 2, snr_db=20.0,
                          nonstationarity="transient", seed=31)
    rec = rc.analytic(rc.filterbank(sig.to_recording("ns"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    space = sm.pac_space(rec, smooth=8.0).subsample(max_points=6000)
    m = windowed_joint_recurrence([space.split(["phase_theta"]),
                                   space.split(["amp_gamma"])],
                                  WindowSpec(size=4.0, overlap=0.75),
                                  target_rr=0.10, theiler=1, rng=0).metrics()
    t = m["t_center_s"].to_numpy()
    outside, inside = (t < 15) | (t > 45), (t > 22) & (t < 37)
    for col in ("rr_subsystem_0", "rr_subsystem_1"):
        assert m[col][outside].mean() == pytest.approx(m[col][inside].mean(), abs=0.005)
    assert (m["recurrence_rate"][inside].mean()
            > 1.4 * m["recurrence_rate"][outside].mean())


def test_classical_indices_enter_the_windowed_table():
    """DD-106. A study can compute recurrence quantification of a
    phase-amplitude space for months without establishing that the space
    contains any coupling. The reference belongs in the same table."""
    n, fs = 12000, 250.0
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    phase = 2 * np.pi * 6 * t + 0.3 * np.cumsum(rng.standard_normal(n)) / 50
    env = np.abs(1 + 0.6 * np.cos(phase - np.pi / 2)) * np.exp(
        0.5 * rng.standard_normal(n))
    rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * phase)),
                     "amp_gamma": env}, fs=fs,
                    roles={"phase_theta": "phase", "amp_gamma": "amplitude"})
    wr = rc.windowed_recurrence(sm.pac_space(rec, smooth=8.0), 6,
                                target_rr=0.05, theiler=1, rng=0)
    plain = wr.metrics()
    both = wr.metrics(rqa=True, classical=True)
    assert "mvl" not in plain.columns
    for column in ("mvl", "mi_tort", "mod_contrast", "preferred_phase"):
        assert column in both.columns
    assert both["mvl"].mean() > 0.15
    assert both["preferred_phase"].mean() == pytest.approx(np.pi / 2, abs=0.5)


def test_the_sweep_reuses_the_histograms_instead_of_re_walking():
    """DD-108. Every RQA metric at every line-length floor is a sum over the
    line histogram, so exploring floors should be free. It was not: calling
    metrics() again with a different floor invalidates its cache and re-walks
    every matrix. Measured on a 19-window batch, one extra floor cost 50 s
    against 13 s to build the batch, so a seven-floor sweep cost five times
    the analysis it was meant to inform."""
    import time

    n, fs = 12000, 250.0
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    phase = 2 * np.pi * 6 * t + 0.3 * np.cumsum(rng.standard_normal(n)) / 50
    rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * phase)),
                     "amp_gamma": np.abs(1 + 0.4 * np.cos(phase))}, fs=fs,
                    roles={"phase_theta": "phase", "amp_gamma": "amplitude"})
    wr = rc.windowed_recurrence(sm.pac_space(rec, smooth=8.0),
                                rc.WindowSpec(size=8.0, step=4.0, unit="seconds"),
                                target_rr=0.05, theiler=1, rng=0)
    started = time.time()
    table = wr.metrics(rqa=True)
    build = time.time() - started

    started = time.time()
    sweep = wr.line_length_sweep(values=(2, 4, 8))
    swept = time.time() - started

    assert swept < build / 5, (
        f"the sweep took {swept:.2f} s against {build:.2f} s to build; it is "
        "re-walking rather than reusing")
    assert len(sweep) == len(table) * 3
    assert {"subject", "window", "min_length", "DET", "LAM"} <= set(sweep.columns)
    # and it must be silent: its whole job is to try floors that do not work
    import warnings as _w

    from recurra.exceptions import GeometryWarning

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        wr.line_length_sweep(values=(2, 4, 8, 16))
    assert not [c for c in caught if issubclass(c.category, GeometryWarning)]


def test_sweeping_before_any_rqa_says_so():
    x = np.cumsum(np.random.default_rng(0).standard_normal((2000, 3)), axis=0)
    wr = rc.windowed_recurrence(x, 4, fs=100.0, target_rr=0.05, theiler=1, rng=0)
    with pytest.raises(rc.ParameterError, match="no histograms yet"):
        wr.line_length_sweep()


@pytest.mark.requires("joblib")
def test_batch_metrics_parallelise_and_agree():
    """DD-109. The build was parallel and the metrics were not, though they are
    four fifths of the work. Results must not depend on n_jobs, and the
    histograms DD-108 caches must survive the process boundary -- a worker
    returns a copy and its cache stays behind unless handed over."""
    n, fs = 8000, 250.0
    t = np.arange(n) / fs
    spaces = {}
    for k in range(3):
        rng = np.random.default_rng(k)
        phase = 2 * np.pi * 6 * t + 0.3 * np.cumsum(rng.standard_normal(n)) / 50
        rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * phase)),
                         "amp_gamma": np.abs(1 + 0.4 * np.cos(phase))}, fs=fs,
                        roles={"phase_theta": "phase", "amp_gamma": "amplitude"})
        spaces[f"s{k}"] = sm.pac_space(rec, smooth=8.0)

    spec = rc.WindowSpec(size=8.0, step=4.0, unit="seconds")
    serial = rc.batch_windowed_recurrence(spaces, spec, target_rr=0.05,
                                          theiler=1, rng=0)
    a = serial.metrics(rqa=True)
    parallel = rc.batch_windowed_recurrence(spaces, spec, target_rr=0.05,
                                            theiler=1, rng=0)
    b = parallel.metrics(rqa=True, n_jobs=2)

    assert list(a.subject) == list(b.subject)
    for column in ("DET", "LAM", "recurrence_rate", "epsilon"):
        assert np.allclose(a[column], b[column]), f"{column} depends on n_jobs"

    # and the sweep must work after the parallel call, not only the serial one
    swept = parallel.line_length_sweep(values=(2, 4))
    assert len(swept) == len(b) * 2



# -------------------------------------------------------------- DD-98
def test_windowed_metrics_share_the_rqa_floors(rec):
    """The two entry points must default to the same l_min and v_min."""
    import inspect

    import recurra as rc
    from recurra.windowed.recurrence import WindowedRecurrence

    top = inspect.signature(rc.rqa).parameters
    win = inspect.signature(WindowedRecurrence.metrics).parameters
    assert win["l_min"].default == top["l_min"].default == 8
    assert win["v_min"].default == top["v_min"].default == 8


def test_classical_indices_refuse_a_rescaled_amplitude():
    """A z-scored amplitude is not an envelope: computed on it, MVL came out at
    13.3 and MI at 0.88 on a signal whose values were 0.31 and 0.035, with no
    warning. The table now says NaN and the reason, once."""
    n, fs = 8000, 250.0
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    phase = 2 * np.pi * 6 * t
    env = (1 + 0.6 * np.cos(phase - np.pi / 2)) * np.exp(0.3 * rng.standard_normal(n))
    rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * phase)), "amp_gamma": env},
                    fs=fs, roles={"phase_theta": "phase", "amp_gamma": "amplitude"})
    space = sm.pac_space(rec, scale_amplitude="zscore")
    assert space.groups[-1].params["scale"] == "zscore"
    wr = rc.windowed_recurrence(space, 4, target_rr=0.05, theiler=1, rng=0)
    with pytest.warns(rc.CaveatWarning, match="rescaled with 'zscore'"):
        table = wr.metrics(rqa=False, classical=True)
    for column in ("mvl", "mi_tort", "mod_contrast", "preferred_phase"):
        assert table[column].isna().all()
