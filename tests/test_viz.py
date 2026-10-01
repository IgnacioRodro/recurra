"""Tests for the figure layer, including the policies of DD-30 and DD-31."""
import re

import numpy as np
import pytest

import recurra as rc
from recurra import statespace as sm
from recurra.viz import COMPARISON_FIGURES, SINGLE_FIGURES, compare

pytestmark = pytest.mark.requires("matplotlib")


@pytest.fixture(scope="module")
def rec_pac():
    s = rc.generate_cfc(modality="pac", duration=16.0, fs=500.0, alpha=0.9,
                        preferred_phase=np.pi / 2, snr_db=20.0, seed=17)
    r = rc.filterbank(s.to_recording("t01"), {"theta": (4, 8), "gamma": (50, 70)})
    return rc.analytic(r, sources=["theta", "gamma"])


@pytest.fixture(scope="module")
def ss_pac(rec_pac):
    return sm.pac_space(rec_pac)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


# ------------------------------------------------- single figures [DD-31]
@pytest.mark.parametrize("mode", ["auto", "2d", "3d", "pairs", "time", "torus"])
def test_attractor_modes(ss_pac, mode):
    fig = rc.plot_attractor(ss_pac, mode=mode, max_points=300)
    assert fig is not None


def test_every_single_figure_takes_one_object(rec_pac, ss_pac):
    """Each entry of the registry must render from a single object."""
    params = sm.estimate_embedding(np.asarray(rec_pac.get("signal")),
                                   tau_range=(1, 60), m_range=(1, 5), rng=0)
    sweep = sm.lambda_sweep(sm.pac_space(rec_pac, scaling="none"),
                            grid=[0.0, 0.5, 1.0], rng=0, max_points=500)
    rm = rc.recurrence_plot(ss_pac.subsample(max_points=400), target_rr=0.05,
                            theiler=1, rng=0)
    wr_small = rc.windowed_recurrence(ss_pac.subsample(max_points=1200), 4,
                                      target_rr=0.05, theiler=1, rng=0)
    small = ss_pac.subsample(max_points=1200)
    batch = rc.batch_windowed_recurrence(
        {f"s{i}": small for i in range(6)},
        rc.WindowSpec(size=400, step=200, unit="samples"),
        target_rr=0.05, theiler=1, rng=0, verify=False)
    grouping = {f"s{i}": ("a" if i < 3 else "b") for i in range(6)}
    gc_small = batch.group_comparison(grouping, values=["recurrence_rate"],
                                      min_per_group=2)
    fm_small = batch.metrics_by_window("recurrence_rate")
    rep_small = batch.check_comparability()
    objects = {
        "attractor": (ss_pac, {"max_points": 300}),
        "recurrence": (rm, {"max_size": 200}),
        "recurrence_panel": (rm, {"rqa": True, "max_size": 200}),
        "joint_recurrence": (rc.joint_recurrence_plot(
            [small.coords[:400, :2], small.coords[:400, 2:]], target_rr=0.1, rng=0), {}),
        "rqa_summary": (rm, {"l_min": 2, "v_min": 2}),
        "density_profile": (rm, {"n_bins": 8}),
        "threshold_diagnostics": (rm.threshold, {}),
        "window_coverage": (wr_small, {}),
        "window_series": (wr_small, {}),
        "window_panel": (wr_small, {}),
        "group_comparison": (gc_small, {}),
        "feature_matrix": (fm_small, {}),
        "comparability": (rep_small, {}),
        "line_histogram": (rm.line_histogram(), {}),
        "diagonal_profile": (rc.diagonal_profile(rm), {"in_seconds": False}),
        "embedding_diagnostics": (params, {}),
        "scale_report": (ss_pac, {}),
        "lambda_sweep": (sweep, {}),
        "quality": (rec_pac, {}),
        "phase_amplitude": (rec_pac, {}),
        "signal": (rec_pac, {"seconds": 2.0}),
        "envelope_pair": (rec_pac, {"amplitude_a": "amp_theta",
                                    "amplitude_b": "amp_gamma"}),
    }
    assert set(objects) == set(SINGLE_FIGURES)
    for name, (obj, kw) in objects.items():
        fn, _ = SINGLE_FIGURES[name]
        assert fn(obj, **kw) is not None


def test_single_figure_does_not_mutate_its_input(ss_pac):
    before = np.array(ss_pac.coords, copy=True)
    rc.plot_attractor(ss_pac, mode="3d", max_points=200)
    np.testing.assert_array_equal(np.asarray(ss_pac.coords), before)


def test_plot_functions_return_without_saving(ss_pac, tmp_path):
    """Plot functions return a Figure; saving is a separate, explicit step."""
    fig = rc.plot_attractor(ss_pac, mode="2d", max_points=200)
    assert not list(tmp_path.iterdir())
    rc.set_config(output_dir=str(tmp_path))
    path = rc.save_figure(fig, "x.png")
    assert path.endswith("x.png")
    assert list(tmp_path.iterdir())


# --------------------------------------------- comparison figures [DD-31]
def test_comparison_takes_a_mapping(rec_pac, ss_pac):
    spaces = {"a": ss_pac, "b": sm.pac_space(rec_pac, scaling="none")}
    assert compare.compare_attractors(spaces, max_points=300) is not None
    assert compare.compare_scale_policies(spaces) is not None
    assert compare.compare_phase_amplitude({"one": rec_pac}) is not None


def test_comparison_rejects_empty_mapping():
    with pytest.raises(rc.ParameterError):
        compare.compare_attractors({})


def test_comparison_shares_limits(rec_pac):
    """Shared axes are what makes a comparison honest."""
    spaces = {"scaled": sm.pac_space(rec_pac),
              "unscaled": sm.pac_space(rec_pac, scaling="none")}
    fig = compare.compare_attractors(spaces, mode="3d", max_points=300,
                                     share_limits=True)
    axes = fig.get_axes()
    assert axes[0].get_xlim() == axes[1].get_xlim()


def test_registries_are_complete():
    assert len(SINGLE_FIGURES) >= 8 and len(COMPARISON_FIGURES) >= 5
    listing = rc.list_figures()
    for name in SINGLE_FIGURES:
        assert name in listing


# ------------------------------------------------- English-only [DD-30]
ACCENTED = re.compile(r"[áéíóúñÁÉÍÓÚÑ¿¡]")


def test_figure_text_is_english(ss_pac, rec_pac):
    """Titles, axis labels and legends carry no accented (Spanish) text."""
    figs = [rc.plot_attractor(ss_pac, mode="3d", max_points=200),
            rc.plot_scale_report(ss_pac),
            rc.plot_phase_amplitude(rec_pac),
            rc.plot_signal(rec_pac, seconds=1.0),
            compare.compare_attractors({"a": ss_pac}, max_points=200)]
    for fig in figs:
        texts = [fig._suptitle.get_text() if fig._suptitle else ""]
        for ax in fig.get_axes():
            texts += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
            legend = ax.get_legend()
            if legend:
                texts += [t.get_text() for t in legend.get_texts()]
        for t in texts:
            assert not ACCENTED.search(t), f"non-English text in a figure: {t!r}"


def test_dataframe_columns_are_english(ss_pac, rec_pac):
    frames = [ss_pac.scale_frame(), ss_pac.describe(), ss_pac.provenance_frame(),
              rec_pac.channel_frame(), rec_pac.provenance_frame()]
    for df in frames:
        for col in df.columns:
            assert not ACCENTED.search(str(col)), f"non-English column: {col!r}"


def test_source_has_no_spanish_identifiers():
    """The package source is English-only [DD-30]."""
    import pathlib

    import recurra

    root = pathlib.Path(recurra.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if ACCENTED.search(line):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"non-English text in source: {offenders[:10]}"


# ------------------------------------------------------ pooling [DD-38]
def test_max_pooling_saturation_is_predicted():
    from recurra.viz.recurrence import expected_saturation

    assert expected_saturation(1, 0.05) == pytest.approx(0.05)
    assert expected_saturation(5, 0.05) == pytest.approx(0.723, abs=0.01)
    assert expected_saturation(13, 0.05) > 0.99


def test_auto_pooling_avoids_saturation():
    from recurra.viz.recurrence import choose_pooling

    assert choose_pooling(1, 0.05) == "none"
    assert choose_pooling(2, 0.05) == "max"       # 18% lit, lines survive
    assert choose_pooling(5, 0.05) == "mean"      # 72% lit, would saturate


def test_pooled_image_is_not_saturated(ss_pac):
    """The bug this rule exists for: a 5x reduction turning the plot solid."""
    from recurra.viz.recurrence import render_recurrence

    rm = rc.recurrence_plot(ss_pac.subsample(max_points=1500), target_rr=0.05,
                            theiler=1, rng=0)
    naive, _, _ = render_recurrence(rm, max_size=300, pooling="max")
    auto, _, method = render_recurrence(rm, max_size=300, pooling="auto")
    assert method == "mean"
    # Max pooling inflates apparent density several-fold; mean pooling
    # reproduces the true recurrence rate.
    assert naive.mean() > 3 * auto.mean()
    assert auto.mean() == pytest.approx(rm.recurrence_rate(), abs=0.01)


@pytest.mark.parametrize("tile_size", [97, 128, 251])
def test_render_streams_identically_to_dense(ss_pac, tile_size):
    """Bin edges must be anchored to the global grid, not to tile boundaries."""
    from recurra.viz.recurrence import render_recurrence

    small = ss_pac.subsample(max_points=600)
    th = rc.threshold(small, target_rr=0.05, theiler=1, rng=0)
    dense = rc.recurrence_plot(small, th, theiler=1, store="memory")
    streamed = rc.recurrence_plot(small, th, theiler=1, store="none",
                                  tile_size=tile_size)
    for method in ("max", "mean"):
        a, fa, _ = render_recurrence(dense, max_size=200, pooling=method)
        b, fb, _ = render_recurrence(streamed, max_size=200, pooling=method)
        assert fa == fb
        np.testing.assert_allclose(a, b, atol=1e-12)


# ------------------------------------------------- figure fixes [DD-49..52]
def test_shared_limits_refused_for_incomparable_spaces(rec_pac):
    """DD-49: a Lorenz attractor shares axes with a PAC space only by
    shrinking the latter to a dot."""
    from recurra.viz.compare import _should_share

    lorenz = sm.channels(rc.ingest(rc.lorenz(2000), fs=100.0,
                                   labels=["x", "y", "z"]))
    pac = sm.pac_space(rec_pac)
    ok, reason = _should_share({"pac": pac, "lorenz": lorenz}, ["pac", "lorenz"])
    assert not ok and "different coordinate blocks" in reason

    ok, _ = _should_share({"a": pac, "b": sm.pac_space(rec_pac, scaling="none")},
                          ["a", "b"])
    assert ok


def test_auto_share_limits_leaves_panels_independent(rec_pac):
    lorenz = sm.channels(rc.ingest(rc.lorenz(2000), fs=100.0,
                                   labels=["x", "y", "z"]))
    fig = compare.compare_attractors({"pac": sm.pac_space(rec_pac),
                                      "lorenz": lorenz},
                                     mode="3d", max_points=300)
    axes = fig.get_axes()
    assert axes[0].get_xlim() != axes[1].get_xlim()
    assert "axes not shared" in fig._suptitle.get_text()


def test_scale_report_has_shape_stats_without_a_policy(rec_pac):
    """DD-28 diagnostics are wanted most where no policy was applied."""
    df = sm.pac_space(rec_pac, scaling="none").scale_frame()
    assert {"tail_ratio", "skewness"} <= set(df.columns)
    assert df["tail_ratio"].notna().all()


def test_scale_report_figure_draws_both_panels(rec_pac):
    for space in (sm.pac_space(rec_pac), sm.pac_space(rec_pac, scaling="none")):
        fig = rc.plot_scale_report(space)
        right = fig.get_axes()[1]
        assert right.patches, "the shape panel is empty"


def test_robust_limits_ignore_rare_extremes():
    """DD-50: instantaneous frequency spikes must not squash everything else."""
    from recurra.viz.attractor import _robust_limits

    bulk = np.random.default_rng(0).normal(6.0, 0.5, 5000)
    spiky = np.concatenate([bulk, [-12.0, 24.0, 80.0]])
    lim = _robust_limits(spiky, 0.995)
    assert lim is not None and lim[0] > -5 and lim[1] < 15
    assert _robust_limits(bulk, 0.995) is None       # nothing to clip


def test_quality_panel_lists_the_checks(rec_pac):
    """DD-52: a panel saying only 'no issues' reads as one that failed."""
    fig = rc.plot_quality(rec_pac)
    right = fig.get_axes()[1]
    assert len(right.patches) >= 6
    assert "all passed" in right.get_title()


def test_quality_panel_marks_failures():
    rec = rc.ingest({"good": np.random.default_rng(29300).standard_normal(2000), "flat": np.ones(2000)},
                    fs=500.0)
    fig = rc.plot_quality(rec)
    assert "not passed" in fig.get_axes()[1].get_title()


def test_pairs_diagonal_shows_phase_for_circle_blocks(rec_pac):
    """DD-51: the histogram of cos is an arcsine U-shape that says nothing."""
    ss = sm.ppc_space(rec_pac, "phase_theta", "phase_gamma")
    fig = rc.plot_attractor(ss, mode="pairs", max_points=800)
    diagonals = [fig.get_axes()[i * 4 + i] for i in range(4)]
    for ax in diagonals:
        assert ax.get_title() == "phase"
        edges = [p.get_x() for p in ax.patches]
        assert min(edges) < -3.0 and max(edges) < 3.2   # spans [-pi, pi]


# ------------------------------------------------- route comparison [DD-53]
def test_route_figure_excludes_scale_dependent_metrics_by_default(rec_pac):
    """nn_mean is in the units of each space, so routes with different
    scaling policies cannot be compared on it [DD-53]."""
    df, _ = sm.compare_routes(rec_pac, rng=0)
    fig = compare.compare_routes(df)
    labels = [ax.get_xlabel() for ax in fig.get_axes()]
    assert not any(lbl.startswith("nn_mean") for lbl in labels)
    assert any("nn_cv" in lbl for lbl in labels)
    assert any("nearest neighbours shared" in lbl for lbl in labels)


def test_route_figure_flags_a_scale_dependent_metric_when_asked(rec_pac):
    df, _ = sm.compare_routes(rec_pac, rng=0)
    fig = compare.compare_routes(df, metrics=["nn_mean", "nn_cv"])
    labels = " ".join(ax.get_xlabel() for ax in fig.get_axes())
    assert "mix scaling policies" in labels


def test_route_figure_reports_failed_routes(rec_pac):
    """A route that did not apply must not vanish from the figure."""
    df, _ = sm.compare_routes(rec_pac, rng=0)
    assert (df["status"] != "ok").any(), "expected at least one route to fail here"
    fig = compare.compare_routes(df)
    caption = fig._suptitle.get_text()
    assert "did not apply" in caption


def test_routes_do_not_see_the_same_geometry(rec_pac):
    """DD-53: the choice of route is not an implementation detail."""
    df, _ = sm.compare_routes(rec_pac, rng=0)
    col = [c for c in df.columns if c.startswith("nn_agreement")][0]
    ok = df[df["status"] == "ok"]
    assert ok[col].max() == pytest.approx(1.0)          # the reference itself
    others = ok[ok[col] < 1.0][col]
    assert others.max() < 0.5, "routes agree far more than expected"


# --------------------------- axes shared across separate files [DD-104]
def test_shared_limits_cover_every_space():
    """DD-104. Two attractors on their own axes are two pictures of two
    shapes: the reader cannot tell a wider trajectory from a different scale."""
    spaces = {}
    for i, gain in enumerate((1.0, 3.0)):
        rng = np.random.default_rng(i)
        n, fs = 4000, 250.0
        t = np.arange(n) / fs
        phase = 2 * np.pi * 6 * t
        rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * phase)),
                         "amp_gamma": gain * np.abs(1 + 0.5 * np.cos(phase)
                                                    + 0.2 * rng.standard_normal(n))},
                        fs=fs, roles={"phase_theta": "phase",
                                      "amp_gamma": "amplitude"})
        spaces[f"s{i}"] = sm.pac_space(rec, smooth=8.0)

    from recurra.viz.compare import scale_factor

    limits = rc.shared_limits(spaces)
    assert len(limits) == 3
    for space in spaces.values():
        # The limits are normalised by default [DD-105], so the coordinates
        # must be compared in the same units. Checking raw ones against them
        # is the very confusion the normalisation exists to remove.
        coords = np.asarray(space.weighted_coords) / scale_factor(space)
        for i, (lo, hi) in enumerate(limits):
            inside = ((coords[:, i] >= lo) & (coords[:, i] <= hi)).mean()
            assert inside > 0.9, f"coordinate {i} mostly outside the shared limits"


def test_plot_attractor_honours_explicit_limits():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    space = sm.channels(rc.ingest(rng.standard_normal((2000, 3)), fs=100.0,
                                  labels=list("xyz")))
    limits = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    fig = rc.plot_attractor(space, mode="2d", axis_limits=limits)
    ax = fig.get_axes()[0]
    assert ax.get_xlim() == pytest.approx((-5.0, 5.0))
    plt.close(fig)


def test_every_comparison_figure_is_reachable_from_the_top_level():
    """DD-104: they existed only as rc.viz.compare.something while every
    single-object figure was rc.something. Nothing intended that."""
    for name in ("compare_attractors", "compare_recurrence", "compare_routes",
                 "compare_thresholds", "compare_series", "compare_window_series",
                 "compare_scale_policies", "compare_phase_amplitude"):
        assert hasattr(rc, name), f"{name} is not exported"


# ------------------- the absolute scale is not information [DD-105]
def _gain_spaces(gains=(0.1, 1.0, 100.0)):
    n, fs = 6000, 250.0
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    phase = 2 * np.pi * 6 * t + 0.3 * np.cumsum(rng.standard_normal(n)) / 50
    spaces = {}
    for gain in gains:
        env = gain * np.abs(1 + 0.6 * np.cos(phase - np.pi / 2)
                            + 0.3 * rng.standard_normal(n))
        rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * phase)),
                         "amp_gamma": env}, fs=fs,
                        roles={"phase_theta": "phase", "amp_gamma": "amplitude"})
        spaces[f"gain{gain:g}"] = sm.pac_space(rec, smooth=8.0)
    return spaces


def test_the_weighted_scale_follows_the_recordings_gain():
    """DD-105. rms_balanced fixes the *ratio* between blocks and leaves the
    absolute scale at n / sum(1/rms^2), which follows the raw amplitude. On a
    real corpus epsilon ranged over a factor of eight between subjects of the
    same group, and their attractors looked like a blob, a ring and a full box
    while having the same shape."""
    from recurra.viz.compare import scale_factor

    spaces = _gain_spaces()
    scales = [scale_factor(s) for s in spaces.values()]
    assert max(scales) / min(scales) > 10, (
        "the gain no longer moves the absolute scale; DD-105 rests on it doing so")


def test_normalising_makes_the_same_shape_look_the_same():
    from recurra.viz.compare import scale_factor

    radii = []
    for space in _gain_spaces().values():
        X = np.asarray(space.weighted_coords) / scale_factor(space)
        radii.append(float(np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2).mean()))
    assert max(radii) - min(radii) < 0.01, (
        f"normalised radii still differ: {radii}")


def test_shared_limits_normalise_by_default():
    spaces = _gain_spaces()
    raw = rc.shared_limits(spaces, normalise=False)
    normalised = rc.shared_limits(spaces)
    # without normalising, the limits are set by the loudest recording
    assert abs(raw[0][1]) > 3 * abs(normalised[0][1]) or \
           abs(normalised[0][1] - 1.22) < 0.2


def test_the_figure_says_when_it_has_normalised():
    import matplotlib.pyplot as plt

    space = next(iter(_gain_spaces().values()))
    limits = [(-2.0, 2.0)] * 3
    fig = rc.plot_attractor(space, mode="2d", axis_limits=limits)
    label = fig.get_axes()[0].get_xlabel()
    assert "normalised" in label, f"the axis does not say so: {label!r}"
    plt.close(fig)
    plain = rc.plot_attractor(space, mode="2d")
    assert "normalised" not in plain.get_axes()[0].get_xlabel()
    plt.close(plain)


def test_the_recurrence_plot_never_used_the_absolute_scale():
    """The normalisation is a display decision only: epsilon follows the scale
    and the recurrence rate is pinned, so nothing computed changes [DD-105]."""
    results = []
    for space in _gain_spaces().values():
        rm = rc.recurrence_plot(space, target_rr=0.05, theiler=1, rng=0)
        results.append((rm.recurrence_rate(),
                        rc.rqa(rm, metrics=["DET"])["DET"]))
    rates = [r for r, _ in results]
    dets = [d for _, d in results]
    assert max(rates) - min(rates) < 0.005
    assert max(dets) - min(dets) < 0.03, (
        f"DET moved with the gain: {dets}")


def test_the_polar_view_puts_phase_on_the_angle():
    """The view that makes coupling legible by eye: a coupled recording is an
    eccentric ring, an uncoupled one a round band."""
    import matplotlib.pyplot as plt

    space = next(iter(_gain_spaces((1.0,)).values()))
    fig = rc.plot_attractor(space, mode="torus")
    ax = fig.get_axes()[0]
    assert "phase" in ax.get_ylabel().lower()
    assert "amp" in ax.get_xlabel().lower()
    plt.close(fig)


# ------------------------------------------------ composite figures [DD-120]
def test_the_panel_draws_every_kind_of_plot(ss_pac):
    """RP, CRP, JRP and meta-RP all render with their signals in the margins."""
    small = ss_pac.subsample(max_points=300)
    X = np.asarray(small.coords)
    plots = {
        "rp": rc.recurrence_plot(small, target_rr=0.05, rng=0),
        "crp": rc.cross_recurrence_plot(X, X[::-1], target_rr=0.05, rng=0),
        "jrp": rc.joint_recurrence_plot([X[:, :2], X[:, 2:]], target_rr=0.1, rng=0),
    }
    for kind, rm in plots.items():
        fig = rc.plot_recurrence_panel(rm, rqa=(kind == "rp"), l_min=2, v_min=2)
        texts = " ".join(t.get_text() for t in fig.texts)
        assert {"rp": "Recurrence plot", "crp": "Cross", "jrp": "Joint"}[kind] in texts
        assert len(fig.axes) == (4 if kind == "rp" else 3)


def test_the_panel_accepts_raw_signals_for_its_margins(ss_pac):
    small = ss_pac.subsample(max_points=300)
    rm = rc.recurrence_plot(small, target_rr=0.05, rng=0)
    raw = np.sin(np.linspace(0, 20, small.n_points))
    fig = rc.plot_recurrence_panel(rm, signals=(raw, raw), names=("raw", "raw"))
    assert fig.axes[1].get_title(loc="left").startswith("raw")


def test_the_joint_figure_refuses_a_plain_plot(ss_pac):
    rm = rc.recurrence_plot(ss_pac.subsample(max_points=200), target_rr=0.05, rng=0)
    with pytest.raises(rc.ParameterError, match="joint"):
        rc.plot_joint_recurrence(rm)
