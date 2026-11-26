"""Tests for F7: parallelism that does not change the answer.

The container these were written on has a single core, so they verify
**correctness** and not speed. A test that asserted a speedup would fail on a
laptop with one free core and pass on a cluster, which is not a useful test.
"""
import numpy as np
import pytest

import recurra as rc
from recurra import statespace as sm
from recurra.parallel.runner import parallel_map, resolve_jobs


def _space(alpha, seed):
    sig = rc.generate_cfc(modality="pac", duration=20.0, fs=500.0, alpha=alpha,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=seed)
    rec = rc.analytic(rc.filterbank(sig.to_recording(f"s{seed}"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    return sm.pac_space(rec, smooth=8.0).subsample(max_points=2000)


# ------------------------------------------------------------- resolution
def test_job_count_resolution():
    import os

    assert resolve_jobs(1) == 1
    assert resolve_jobs(4) == 4
    assert resolve_jobs(0) == 1
    assert resolve_jobs(-1) == max(1, os.cpu_count() or 1)


def test_unknown_backend_is_rejected():
    with pytest.raises(rc.ParameterError, match="backend must be"):
        parallel_map(lambda x, rng=None: x, [1, 2], n_jobs=2, backend="magic")


# -------------------------------------------------- the promise of DD-05
@pytest.mark.requires("joblib")
def test_parallel_map_preserves_input_order():
    out = parallel_map(lambda x, rng=None: x * 2, list(range(20)),
                       n_jobs=4, pass_rng=False)
    assert out == [x * 2 for x in range(20)]


@pytest.mark.requires("joblib")
def test_random_streams_depend_on_position_not_scheduling():
    """DD-81. Each item's stream is derived from its position before any work
    starts, so two runs at different job counts must draw the same numbers."""
    def draw(x, rng=None):
        return float(rng.random())

    serial = parallel_map(draw, list(range(12)), n_jobs=1, rng=42)
    parallel = parallel_map(draw, list(range(12)), n_jobs=4, rng=42)
    assert serial == parallel


@pytest.mark.requires("joblib")
def test_batch_results_do_not_depend_on_n_jobs():
    """The promise of DD-05, checked on the real pipeline rather than argued."""
    subjects = {f"sub{i:02d}": _space(0.3 + 0.1 * i, 3000 + i) for i in range(5)}
    spec = rc.WindowSpec(size=4.0, overlap=0.5)
    frames = {}
    for n_jobs in (1, 3):
        batch = rc.batch_windowed_recurrence(subjects, spec, target_rr=0.05,
                                             theiler=1, rng=11, n_jobs=n_jobs)
        frames[n_jobs] = batch.metrics(rqa=True)
    a, b = frames[1], frames[3]
    assert list(a["subject"]) == list(b["subject"])
    for column in ("epsilon", "recurrence_rate", "DET", "L", "LAM", "ENTR"):
        np.testing.assert_allclose(a[column].to_numpy(), b[column].to_numpy(),
                                   err_msg=f"{column} changed with n_jobs")


def test_batch_worker_is_picklable():
    """DD-81: the process backend pickles the callable, so it cannot be a
    lambda. This is the mistake the first implementation made."""
    import pickle

    from recurra.windowed.recurrence import _one_record

    assert pickle.loads(pickle.dumps(_one_record)) is _one_record


def test_serial_backend_is_always_available():
    """joblib is an optional dependency; asking for one worker must never
    reach it."""
    out = parallel_map(lambda x, rng=None: x + 1, [1, 2, 3], n_jobs=1,
                       backend="serial", pass_rng=False)
    assert out == [2, 3, 4]


# ------------------------------------------------------- default workers
def _fresh_auto(monkeypatch, cores=8):
    """auto_jobs with the config default restored and a fixed core count."""
    import recurra.engine as engine
    import recurra.parallel.runner as runner

    rc.set_config(n_jobs=None)
    monkeypatch.setattr(runner, "_ANNOUNCED_JOBS", set())
    monkeypatch.setattr(engine, "physical_cores", lambda: cores)
    return runner


def test_default_takes_every_physical_core_when_the_work_is_worth_it(monkeypatch):
    """DD-126: by default a batch computes as fast as the machine allows."""
    pytest.importorskip("joblib")
    runner = _fresh_auto(monkeypatch, cores=8)
    big = 10 * runner.AUTO_PARALLEL_CELLS
    with pytest.warns(rc.CaveatWarning, match="8 worker processes"):
        assert runner.auto_jobs(None, n_items=100, work_cells=big) == 8
    # Never more workers than items, and the announcement is not repeated.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert runner.auto_jobs(None, n_items=100, work_cells=big) == 8
    assert runner.auto_jobs(None, n_items=3, work_cells=big) == 3


def test_default_stays_on_one_core_for_small_work(monkeypatch):
    runner = _fresh_auto(monkeypatch, cores=8)
    assert runner.auto_jobs(None, n_items=100, work_cells=1e6) == 1
    assert runner.auto_jobs(None, n_items=1, work_cells=1e12) == 1


def test_an_explicit_count_is_used_as_given(monkeypatch):
    runner = _fresh_auto(monkeypatch, cores=8)
    assert runner.auto_jobs(1, n_items=100, work_cells=1e12) == 1
    assert runner.auto_jobs(3, n_items=100, work_cells=1e3) == 3
    rc.set_config(n_jobs=2)
    assert runner.auto_jobs(None, n_items=100, work_cells=1e12) == 2


def test_default_without_joblib_runs_on_one_core_and_says_how(monkeypatch):
    import importlib.util

    runner = _fresh_auto(monkeypatch, cores=8)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "joblib" else real(name, *a))
    with pytest.warns(rc.CaveatWarning, match="joblib is not installed"):
        assert runner.auto_jobs(None, n_items=100, work_cells=1e12) == 1


def test_batch_rqa_default_matches_one_core(monkeypatch):
    """The default spreads the batch; the numbers do not move [DD-81]."""
    pytest.importorskip("joblib")
    import recurra.parallel.runner as runner

    g = np.random.default_rng(3)
    W = g.standard_normal((4, 300, 2))
    one = rc.batch_rqa(W, rng=0, engine="numpy", n_jobs=1)
    _fresh_auto(monkeypatch, cores=2)
    monkeypatch.setattr(runner, "AUTO_PARALLEL_CELLS", 1)
    with pytest.warns(rc.CaveatWarning, match="2 worker processes"):
        auto = rc.batch_rqa(W, rng=0, engine="numpy")
    assert one.equals(auto)


def test_default_inside_a_worker_process_stays_on_one_core(monkeypatch):
    """A batch called from the user's own parallel loop does not start one
    process per core in every worker [DD-126]."""
    import multiprocessing

    runner = _fresh_auto(monkeypatch, cores=8)
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    assert runner.auto_jobs(None, n_items=100, work_cells=1e12) == 1
    assert runner.auto_jobs(4, n_items=100, work_cells=1e12) == 4
