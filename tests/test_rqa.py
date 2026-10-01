"""Tests for F6: line histograms and RQA metrics."""
import numpy as np
import pytest

import recurra as rc
from recurra import statespace as sm
from recurra.rqa.histogram import _accumulate, line_histogram, line_histogram_dense
from recurra.rqa.metrics import METRIC_NAMES, diagonal_profile, rqa_from_histogram


# ------------------------------------------------------------ carry logic
@pytest.mark.parametrize("segment,open_in,expected_out,expected_open", [
    ([1, 1, 0, 1, 1, 1, 0], 0, [2, 3], 0),
    ([1, 1, 0, 1, 1, 1], 0, [2], 3),
    ([1, 1, 0], 5, [7], 0),
    ([1, 1, 1], 5, [], 8),
    ([0, 0, 0], 5, [5], 0),
    ([0, 1, 0], 5, [5, 1], 0),
    ([], 4, [], 4),
])
def test_run_carry(segment, open_in, expected_out, expected_open):
    """The carry across a tile boundary, on cases checkable by hand [DD-56]."""
    out: list = []
    left = _accumulate(np.array(segment, dtype=bool), open_in, out)
    assert out == expected_out
    assert left == expected_open


# --------------------------------------------------- streaming equivalence
@pytest.mark.parametrize("n", [37, 64, 211])
@pytest.mark.parametrize("theiler", [0, 1, 3])
@pytest.mark.parametrize("tile", [7, 16, 1024])
def test_streaming_histogram_equals_dense(n, theiler, tile):
    """The whole long-signal strategy rests on this being exact [DD-56]."""
    X = np.random.default_rng(0).standard_normal((n, 3))
    rm = rc.recurrence_plot(X, target_rr=0.15, theiler=theiler, rng=0)
    ref = line_histogram_dense(rm.matrix, theiler=theiler)
    got = line_histogram(rm, tile_size=tile)
    for kind in ("diagonal", "vertical", "white_vertical"):
        np.testing.assert_array_equal(getattr(got, kind), getattr(ref, kind),
                                      err_msg=f"{kind} differs at tile={tile}")


def test_histogram_rate_matches_the_matrix():
    X = np.random.default_rng(1).standard_normal((300, 3))
    rm = rc.recurrence_plot(X, target_rr=0.08, theiler=1, rng=0)
    assert line_histogram(rm).recurrence_rate == pytest.approx(rm.recurrence_rate())


def test_every_point_is_on_a_line_of_length_one_or_more():
    """Conservation: the diagonal histogram must account for every recurrent
    point, since every point lies on a diagonal run of at least one."""
    X = np.random.default_rng(2).standard_normal((200, 2))
    rm = rc.recurrence_plot(X, target_rr=0.10, theiler=0, rng=0)
    h = line_histogram(rm)
    lengths = np.arange(h.diagonal.size)
    assert int(np.sum(h.diagonal * lengths)) == h.recurrent_points


# --------------------------------------------------------- ground truth
def test_white_noise_determinism_is_twice_the_rate():
    """For an independent recurrence matrix of density p, the fraction of
    points on diagonal lines of length >= 2 is 1 - (1 - p)^2, about 2p.

    **At l_min=2, which is where that identity is defined.** The library's
    default is 8 because smooth trajectories saturate DET at 2 [DD-96], but
    the analytic check belongs at the convention the mathematics assumes, and
    it is the strongest ground truth this module has. It is passed explicitly
    so that a change of default can never silently invalidate it.
    """
    X = np.random.default_rng(3).standard_normal((1500, 3))
    for target in (0.02, 0.05, 0.10):
        m = rc.rqa(X, target_rr=target, theiler=1, l_min=2, v_min=2, rng=0)
        expected = 1 - (1 - m["RR"]) ** 2
        assert m["DET"] == pytest.approx(expected, rel=0.15)


def test_periodic_signal_is_deterministic_and_spans_the_record():
    fs, f0, n = 200.0, 5.0, 2000
    t = np.arange(n) / fs
    S = np.column_stack([np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t)])
    m = rc.rqa(S, target_rr=0.05, theiler=1, l_min=2, v_min=2, rng=0)
    assert m["DET"] > 0.85
    assert m["Lmax"] > 0.9 * n            # a line running the whole record


def test_lorenz_is_strongly_deterministic():
    L = rc.lorenz(n_points=3000, dt=0.01, transient=10.0)
    m = rc.rqa(L, target_rr=0.05, theiler=1, l_min=2, v_min=2, rng=0)
    assert m["DET"] > 0.99
    assert m["LAM"] > 0.95
    assert m["L"] > 10 and m["ENTR"] > 3


def test_determinism_orders_the_three_reference_systems():
    """Noise, a chaotic attractor and a periodic orbit must order correctly at
    a matched recurrence rate."""
    rng = np.random.default_rng(4)
    n = 2000
    t = np.arange(n) / 200.0
    signals = {
        "noise": rng.standard_normal((n, 3)),
        "lorenz": rc.lorenz(n, dt=0.01, transient=10.0),
        "sine": np.column_stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 5 * t)]),
    }
    det = {k: rc.rqa(v, target_rr=0.05, theiler=1, l_min=2, v_min=2, rng=0)["DET"]
           for k, v in signals.items()}
    ent = {k: rc.rqa(v, target_rr=0.05, theiler=1, l_min=2, v_min=2, rng=0)["ENTR"]
           for k, v in signals.items()}
    assert det["noise"] < det["sine"] < det["lorenz"]
    assert ent["noise"] < ent["sine"] < ent["lorenz"]


# ---------------------------------------------------------- conventions
def test_denominator_convention_changes_det(caplog):
    """DD-58: the choice is real, systematic, and previously silent."""
    L = rc.lorenz(2000, dt=0.01, transient=10.0)
    corrected = rc.rqa(L, target_rr=0.05, theiler=1, rng=0,
                       denominator="theiler_corrected")["DET"]
    old = rc.rqa(L, target_rr=0.05, theiler=1, rng=0, denominator="all")["DET"]
    assert corrected > old


def test_unknown_denominator_is_rejected():
    X = np.random.default_rng(5).standard_normal((200, 2))
    with pytest.raises(rc.ParameterError, match="denominator must be"):
        rc.rqa(X, target_rr=0.05, denominator="whatever", rng=0)


def test_l_min_filters_short_lines():
    X = np.random.default_rng(6).standard_normal((400, 2))
    rm = rc.recurrence_plot(X, target_rr=0.10, theiler=1, rng=0)
    a = rc.rqa(rm, l_min=2)["DET"]
    b = rc.rqa(rm, l_min=5)["DET"]
    assert b < a


def test_normalised_entropy_is_bounded():
    L = rc.lorenz(1500, dt=0.01, transient=10.0)
    m = rc.rqa(L, target_rr=0.05, theiler=1, rng=0)
    assert 0.0 <= m["ENTR_norm"] <= 1.0


def test_metric_selection():
    X = np.random.default_rng(7).standard_normal((300, 2))
    m = rc.rqa(X, target_rr=0.05, theiler=1, rng=0, metrics=["RR", "DET"])
    assert set(m) == {"RR", "DET"}
    with pytest.raises(rc.ParameterError, match="unknown metric"):
        rc.rqa(X, target_rr=0.05, rng=0, metrics=["NOPE"])


def test_all_declared_metrics_are_produced():
    X = np.random.default_rng(8).standard_normal((300, 2))
    m = rc.rqa(X, target_rr=0.05, theiler=1, rng=0)
    assert set(m) == set(METRIC_NAMES)


def test_as_frame_carries_the_conventions():
    X = np.random.default_rng(9).standard_normal((300, 2))
    df = rc.rqa(X, target_rr=0.05, theiler=1, rng=0, as_frame=True)
    assert {"l_min", "denominator", "theiler", "epsilon"} <= set(df.columns)


# ------------------------------------------------------------------ CRQA
def test_diagonal_profile_finds_a_known_lag():
    """A trajectory against a shifted copy of itself must peak at the shift."""
    rng = np.random.default_rng(10)
    X = np.cumsum(rng.standard_normal((900, 2)), axis=0)
    shift = 40
    a, b = X[shift:], X[:-shift]
    crp = rc.cross_recurrence_plot(a, b, target_rr=0.05, rng=0)
    prof = diagonal_profile(crp)
    peak = int(prof.loc[prof["density"].idxmax(), "offset"])
    assert abs(peak - shift) <= 2


def test_crqa_reports_lag_and_asymmetry():
    rng = np.random.default_rng(11)
    X = np.cumsum(rng.standard_normal((700, 2)), axis=0)
    crp = rc.cross_recurrence_plot(X[30:], X[:-30], target_rr=0.05, rng=0)
    out = rc.crqa(crp)
    assert {"peak_offset", "peak_lag_s", "profile_asymmetry"} <= set(out)
    assert abs(out["peak_offset"] - 30) <= 3


# ------------------------------------------------------------- integration
def test_rqa_enters_the_windowed_table():
    s = rc.generate_cfc(modality="pac", duration=20.0, fs=500.0, alpha=0.9,
                        preferred_phase=np.pi / 2, snr_db=20.0, seed=17)
    r = rc.analytic(rc.filterbank(s.to_recording("p"),
                                  {"theta": (4, 8), "gamma": (50, 70)}),
                    sources=["theta", "gamma"])
    ss = sm.pac_space(r, smooth=8.0).subsample(max_points=2000)
    wr = rc.windowed_recurrence(ss, 4, target_rr=0.05, theiler=1, rng=0)
    plain = wr.metrics()
    with_rqa = wr.metrics(rqa=True)
    assert not set(METRIC_NAMES) <= set(plain.columns)
    assert set(METRIC_NAMES) <= set(with_rqa.columns)
    assert len(with_rqa) == len(wr)


def test_matrix_convenience_methods():
    X = np.random.default_rng(12).standard_normal((300, 2))
    rm = rc.recurrence_plot(X, target_rr=0.05, theiler=1, rng=0)
    assert rm.rqa()["RR"] == pytest.approx(rm.recurrence_rate())
    assert rm.line_histogram().n_rows == 300


def test_rqa_never_needs_the_matrix():
    """Streaming all the way: a structure that is never materialised."""
    X = np.random.default_rng(13).standard_normal((600, 3))
    rm = rc.recurrence_plot(X, target_rr=0.05, theiler=1, store="none",
                            tile_size=64, rng=0)
    assert rm._matrix is None
    m = rc.rqa(rm, l_min=2, v_min=2)
    assert rm._matrix is None
    assert 0 < m["DET"] < 1


def test_saturated_determinism_is_flagged():
    """DD-98. DET pinned against its ceiling cannot separate anything, and on a
    smooth trajectory that is where it sits with the conventional floor of 2.

    Both floors now default to 8. Which failure mode a given space falls into
    is a property of the space: on this synthetic one the default gives DET
    room and crushes LAM, and both ends are warned about.
    """
    from recurra import statespace as sm
    from recurra.exceptions import GeometryWarning

    n, fs = 8000, 250.0
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    phase = 2 * np.pi * 6 * t + 0.3 * np.cumsum(rng.standard_normal(n)) / 50
    envelope = 1 + 0.7 * np.cos(phase - np.pi / 2) + 0.4 * rng.standard_normal(n)
    rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * phase)),
                     "amp_gamma": np.abs(envelope)}, fs=fs,
                    roles={"phase_theta": "phase", "amp_gamma": "amplitude"})
    space = sm.pac_space(rec, smooth=8.0)

    with pytest.warns(GeometryWarning, match="against its ceiling"):
        saturated = rc.rqa(space, target_rr=0.05, theiler=1, l_min=2, v_min=2,
                           rng=0)
    assert saturated["DET"] > 0.99

    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", GeometryWarning)
        roomy = rc.rqa(space, target_rr=0.05, theiler=1, rng=0)
    assert roomy["DET"] < saturated["DET"], (
        "the default floor gives DET no more room than the classical one")
    assert 0.01 < roomy["DET"] < 0.99, (
        f"DET has no room at the default floor: {roomy['DET']:.4f}")

def test_the_defaults_departed_from_the_classical_two():
    """DD-98. Both floors default to 8 rather than the classical 2, because on
    a real corpus 2 saturates both metrics: DET at 0.9965 with a spread of
    0.0005, LAM at 0.9983.

    The first version of this decision kept v_min at 2 on the strength of a
    synthetic measurement -- LAM's range went 0.71 at 2 to zero at 4 there --
    and a real corpus said the opposite. The synthetic envelope was too rough
    to stand in for a recorded one. This test pins the departure, not the
    number: the right floor is a property of the data and
    line_length_sweep finds it.
    """
    import inspect

    from recurra.rqa.metrics import rqa as _rqa

    params = inspect.signature(_rqa).parameters
    assert params["l_min"].default != 2
    assert params["v_min"].default != 2


def test_a_crushed_metric_is_flagged_as_loudly_as_a_saturated_one():
    from recurra.exceptions import GeometryWarning

    X = np.random.default_rng(0).standard_normal((1200, 3))
    with pytest.warns(GeometryWarning, match="crushed to zero"):
        rc.rqa(X, target_rr=0.05, theiler=1, v_min=30, rng=0)


def test_the_sweep_answers_the_floor_question_from_the_data():
    """DD-98: the histogram already holds every threshold, so choosing well
    costs milliseconds against the minutes the plot took."""
    X = np.cumsum(np.random.default_rng(1).standard_normal((1500, 3)), axis=0)
    rm = rc.recurrence_plot(X, target_rr=0.05, theiler=1, rng=0)
    frame = rc.line_length_sweep(rm.line_histogram())
    assert {"min_length", "DET", "LAM", "DET_headroom"} <= set(frame.columns)
    assert frame["DET"].is_monotonic_decreasing
    assert frame["n_diagonal_lines"].is_monotonic_decreasing


def test_the_sweep_does_not_warn_about_the_floors_it_is_testing():
    """DD-98. The sweep exists to find the floor at which a metric has room,
    so its whole job is to evaluate floors that do not work. Warning about
    each one produced as many warnings as floors, per subject, per channel --
    noise about the very thing being measured. The headroom column reports the
    same fact once, quantitatively."""
    import warnings as w

    from recurra.exceptions import GeometryWarning

    n, fs = 8000, 250.0
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    phase = 2 * np.pi * 2 * t + 0.3 * np.cumsum(rng.standard_normal(n)) / 50
    rec = rc.ingest({"phase_delta": np.angle(np.exp(1j * phase)),
                     "amp_gamma": np.abs(1 + 0.4 * np.cos(phase))}, fs=fs,
                    roles={"phase_delta": "phase", "amp_gamma": "amplitude"})
    from recurra import statespace as sm

    rm = rc.recurrence_plot(sm.pac_space(rec, smooth=8.0), target_rr=0.05,
                            theiler=1, rng=0)
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        frame = rc.line_length_sweep(rm.line_histogram())
    assert not [x for x in caught if issubclass(x.category, GeometryWarning)]
    # and the information is still there, as a number
    assert "DET_headroom" in frame.columns
    assert frame["DET_headroom"].max() > frame["DET_headroom"].min()



# ------------------------------------------------------------- DD-111
def test_normalised_entropy_uses_admissible_lengths():
    """One rare long line must not move ENTR_norm by itself."""
    from recurra.rqa.histogram import LineHistogram

    size = 300
    h = np.zeros(size + 1, dtype=np.int64)
    h[8] = 50
    h[9] = 50
    base = LineHistogram(diagonal=h.copy(), vertical=h.copy(), white_vertical=h.copy(),
                         recurrent_points=int((h * np.arange(h.size)).sum()),
                         considered_points=10_000, n_rows=100, n_cols=100)
    with_tail = h.copy()
    with_tail[250] = 1
    tail = LineHistogram(diagonal=with_tail, vertical=with_tail.copy(),
                         white_vertical=with_tail.copy(),
                         recurrent_points=int((with_tail * np.arange(h.size)).sum()),
                         considered_points=10_000, n_rows=100, n_cols=100)
    a = rqa_from_histogram(base, l_min=8, v_min=8, w_min=8)
    b = rqa_from_histogram(tail, l_min=8, v_min=8, w_min=8)
    # two equiprobable lengths out of two admissible: fully spread
    assert abs(a["ENTR_norm"] - 1.0) < 1e-12
    # the same two lengths out of 243 admissible: far from the ceiling, and
    # the old convention (log of lengths present) would have given ~0.63
    assert b["ENTR_norm"] < 0.2
    assert 0.0 <= b["RTE"] <= 1.0
