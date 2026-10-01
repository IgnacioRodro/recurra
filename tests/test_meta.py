"""Meta-recurrence: recurrence of the per-window measurements [DD-119]."""
import numpy as np
import pandas as pd
import pytest

import recurra as rc
from recurra import statespace as sm


def _table(n=60, period=6, step_s=5.0, size_s=10.0, seed=0):
    """A per-window table that alternates between two regimes every half period."""
    rng = np.random.default_rng(seed)
    regime = (np.arange(n) % period) < period // 2
    return pd.DataFrame({
        "window": np.arange(n),
        "start_sample": (np.arange(n) * step_s * 100).astype(int),
        "stop_sample": (np.arange(n) * step_s * 100 + size_s * 100).astype(int),
        "t_center_s": np.arange(n) * step_s + size_s / 2,
        "target_rr": 0.05, "recurrence_rate": 0.05, "theiler": 1, "l_min": 8,
        "DET": np.where(regime, 0.8, 0.3) + 0.01 * rng.standard_normal(n),
        "LAM": np.where(regime, 0.2, 0.6) + 0.01 * rng.standard_normal(n),
        "qc_burst_fraction": rng.random(n),
    })


def test_the_meta_space_keeps_measurements_and_drops_bookkeeping():
    space = rc.meta_space(_table())
    assert [g.label for g in space.groups] == ["DET", "LAM"]
    assert space.n_points == 60 and space.fs == pytest.approx(1 / 5.0)
    assert space.t0 == pytest.approx(5.0)
    np.testing.assert_allclose(space.coords.mean(axis=0), 0, atol=1e-12)


@pytest.mark.parametrize("size,step,expected", [(10, 5, 2), (10, 10, 1), (8, 2, 4)])
def test_the_automatic_theiler_window_excludes_overlapping_windows(size, step, expected):
    """Windows fewer than size/step positions apart share samples."""
    assert rc.meta_theiler(_table(size_s=size, step_s=step)) == expected


def test_a_regime_alternation_recurs_at_its_period():
    """Ground truth: two regimes alternating with a period of 6 windows give
    a meta-RP that is full on diagonals at multiples of 6 and empty halfway."""
    # A fixed epsilon in z-units: within a regime points are ~0.04 apart,
    # between regimes ~2.8. (A target rate would have to exceed the 50% of
    # pairs that share a regime to mark them all.)
    rm = rc.meta_recurrence(_table(), threshold=0.5, rng=0)
    assert rm.theiler == 2                       # 50% overlap
    M = rm.matrix
    for k in (6, 12, 18):
        assert np.diag(M, k).mean() == 1.0
    assert np.diag(M, 3).mean() == 0.0
    values = rc.rqa(rm, l_min=2, v_min=2)
    assert 0 < values["DET"] <= 1


def test_meta_recurrence_from_a_windowed_analysis():
    sig = rc.generate_cfc(modality="pac", duration=40.0, fs=250.0, alpha=0.8, rng=1)
    rec = rc.analytic(rc.filterbank(sig.to_recording("s"), {"theta": (4, 8),
                                                           "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    wr = rc.windowed_recurrence(sm.pac_space(rec, smooth=12.0),
                                rc.WindowSpec(size=4.0, overlap=0.5, unit="seconds"),
                                target_rr=0.05, rng=0)
    wr.metrics(rqa=True, l_min=2, v_min=2)
    rm = wr.meta_recurrence(["DET", "LAM", "L", "TT"], target_rr=0.2, rng=0)
    assert rm.n_rows == len(wr) and rm.theiler == 2
    assert rm.fs == pytest.approx(1 / 2.0)


def test_missing_windows_are_dropped_with_a_warning():
    t = _table()
    t.loc[[3, 10], "DET"] = np.nan
    with pytest.warns(rc.CaveatWarning, match="2 window"):
        space = rc.meta_space(t)
    assert space.n_points == 58
    with pytest.raises(rc.ParameterError, match="missing"):
        rc.meta_space(t, missing="error")


def test_cross_and_joint_meta_recurrence():
    a, b = _table(seed=0), _table(seed=1)
    crp = rc.meta_recurrence(a, kind="crp", other=b, target_rr=0.3, rng=0)
    assert crp.kind == "crp" and crp.shape == (60, 60)
    jrp = rc.meta_recurrence([a, b], kind="jrp", target_rr=0.3, rng=0)
    assert jrp.theiler == 2 and jrp.shape == (60, 60)
