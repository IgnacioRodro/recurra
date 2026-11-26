"""The compute engines [DD-125]: choice, and the same answer on every one.

Three groups:

* engine choice, and ``batch_rqa`` on the reference engine -- core only;
* the torch engine on the CPU (``requires("torch")``), which checks its logic
  on any machine with PyTorch, GPU or not;
* the torch engine on CUDA (``cuda``), which only a machine with a GPU runs.
  ``docs/TESTS.md`` says how.

The reference is the single-plot path, ``rqa(recurrence_plot(X, threshold(X,
...)))``, and the comparisons are exact: thresholds bit for bit, histograms
count for count, every column of the table equal.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import recurra as rc
from recurra.engine import select as engine_select

FLOORS = [(2, 2, 2), (8, 8, 2), (5, 3, 4)]


def _signals(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    g = np.random.default_rng(seed)
    t = np.arange(n) / 200.0
    return {
        "noise": g.standard_normal((n, 3)),
        "sine": np.column_stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 5 * t),
                                 0.1 * g.standard_normal(n)]),
        "lorenz": rc.lorenz(n, dt=0.01, transient=10.0),
    }


def _windows(n: int, count: int = 4, seed: int = 0) -> np.ndarray:
    g = np.random.default_rng(seed)
    base = rc.lorenz(n * count, dt=0.01, transient=10.0)
    return base.reshape(count, n, 3) + 0.05 * g.standard_normal((count, n, 3))


def _same_tables(a: pd.DataFrame, b: pd.DataFrame, where: str) -> None:
    assert list(a.columns) == list(b.columns), where
    for c in a.columns:
        assert a[c].equals(b[c]), f"{c} differs: {where}"


# ------------------------------------------------------------ engine choice
def test_auto_without_a_device_uses_the_reference_quietly(monkeypatch):
    monkeypatch.setattr(engine_select, "cuda_devices", lambda: ())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        choice = rc.engine.resolve_engine("auto", n_points=5000)
    assert (choice.engine, choice.device) == ("numpy", "cpu")


def test_auto_announces_what_it_chose_when_a_device_exists(monkeypatch):
    """P3: with a GPU present the choice is announced, whichever way it went."""
    monkeypatch.setattr(engine_select, "cuda_devices", lambda: ("cuda:0",))
    monkeypatch.setattr(engine_select, "_torch_details", lambda dev: {})
    monkeypatch.setattr(engine_select, "_ANNOUNCED", set())
    with pytest.warns(rc.CaveatWarning, match="torch engine on cuda:0"):
        big = rc.engine.resolve_engine("auto", n_points=5000)
    assert (big.engine, big.device) == ("torch", "cuda:0")
    with pytest.warns(rc.CaveatWarning, match="numpy engine"):
        small = rc.engine.resolve_engine("auto", n_points=40)
    assert small.engine == "numpy"


def test_torch_never_falls_back_to_the_cpu(monkeypatch):
    """G3 of the request that motivated the engine: asking for the device and
    not getting it is an error, not a slower run."""
    monkeypatch.setattr(engine_select, "torch_available", lambda: True)
    monkeypatch.setattr(engine_select, "cuda_devices", lambda: ())
    with pytest.raises(rc.ParameterError, match="does\\s+not fall back"):
        rc.engine.resolve_engine("torch")


def test_missing_torch_names_the_extra(monkeypatch):
    monkeypatch.setattr(engine_select, "torch_available", lambda: False)
    with pytest.raises(rc.BackendUnavailable, match=r"recurra\[gpu\]"):
        rc.engine.resolve_engine("torch")


def test_engine_arguments_are_checked():
    with pytest.raises(rc.ParameterError, match="numpy engine runs on the CPU"):
        rc.engine.resolve_engine("numpy", "cuda:0")
    with pytest.raises(rc.ParameterError, match="engine must be one of"):
        rc.engine.resolve_engine("cupy")
    with pytest.raises(rc.ParameterError, match="device must be"):
        rc.engine.resolve_engine("auto", "gpu0")


# ------------------------------------------------- batch_rqa, reference engine
@pytest.mark.parametrize("theiler", [0, 1, 2, 5])
def test_each_row_is_the_single_plot_path(theiler):
    """DD-125: a row of batch_rqa is rqa(recurrence_plot(X, threshold(X)))
    with the integer seed, whatever else is in the batch."""
    X = _windows(301, count=3)
    table = rc.batch_rqa(X, target_rr=0.06, theiler=theiler, l_min=2, v_min=2,
                         rng=5, engine="numpy")
    for b in range(3):
        th = rc.threshold(X[b], target_rr=0.06, theiler=theiler, rng=5)
        ref = rc.rqa(rc.recurrence_plot(X[b], th, store="stream"), l_min=2, v_min=2)
        assert table.loc[b, "epsilon"] == float(th.value)
        assert table.loc[b, "achieved_rr"] == th.achieved_rr
        for name, value in ref.items():
            assert table.loc[b, name] == value or (np.isnan(value)
                                                   and np.isnan(table.loc[b, name])), name


@pytest.mark.requires("joblib")
def test_reference_engine_gives_the_same_table_with_processes():
    """DD-81, DD-82: worker processes change the time, not the answer."""
    X = _windows(257, count=6)
    one = rc.batch_rqa(X, rng=1, engine="numpy", n_jobs=1)
    two = rc.batch_rqa(X, rng=1, engine="numpy", n_jobs=2)
    _same_tables(one, two, "n_jobs 1 vs 2")


def test_windows_of_different_lengths_keep_their_order():
    g = np.random.default_rng(2)
    ws = [g.standard_normal((300, 3)), g.standard_normal((180, 3)),
          g.standard_normal((300, 3))]
    table = rc.batch_rqa(ws, rng=0, engine="numpy")
    assert list(table["n_points"]) == [300, 180, 300]
    single = rc.batch_rqa([ws[1]], rng=0, engine="numpy")
    assert table.loc[1, "DET"] == single.loc[0, "DET"]
    assert table.attrs["provenance"]["engine"] == "numpy"


# ------------------------------------------------- torch engine, on the CPU
@pytest.mark.requires("torch")
@pytest.mark.parametrize("metric", ["euclidean", "chebyshev", "manhattan"])
@pytest.mark.parametrize("n", [40, 300, 1157])
def test_torch_thresholds_are_numpys_to_the_last_bit(metric, n):
    """The device sorts and numpy does the arithmetic that decides epsilon, so
    epsilon and the achieved rate are identical, whether every pair is taken
    (40, 300 points) or a sample is drawn (1157)."""
    import torch

    from recurra.engine.torch_rqa import quantile_thresholds, sampled_values
    from recurra.threshold.sampling import sample_pair_indices

    for theiler in (0, 1, 5):
        X = np.random.default_rng(n + theiler).standard_normal((4, n, 3)) * 2 + 1
        i, j = sample_pair_indices(n, theiler=theiler, rng=3)
        V = sampled_values(torch.as_tensor(X), torch.as_tensor(i), torch.as_tensor(j),
                           metric=metric)
        eps, achieved, _ = quantile_thresholds(V, 0.07, metric=metric)
        for b in range(4):
            ref = rc.threshold(X[b], target_rr=0.07, theiler=theiler, metric=metric, rng=3)
            assert eps[b] == float(ref.value)
            assert achieved[b] == ref.achieved_rr


def _compare_engines(X, *, theiler, device, precision="double",
                     memories=(8.0, 0.002)) -> None:
    """Histograms count for count and every column of the table, against the
    reference engine. Equal histograms make the metrics equal at any floors,
    since both engines compute them with the same function; one set of floors
    is compared anyway."""
    windows = np.stack([X, X[::-1].copy(), X + 0.3])
    l_min, v_min, w_min = FLOORS[2]
    ref, ref_h = rc.batch_rqa(windows, theiler=theiler, l_min=l_min, v_min=v_min,
                              w_min=w_min, rng=7, engine="numpy", return_histograms=True)
    for memory in memories:                  # whole plots; bands of a few rows
        got, got_h = rc.batch_rqa(windows, theiler=theiler, l_min=l_min, v_min=v_min,
                                  w_min=w_min, rng=7, engine="torch", device=device,
                                  precision=precision, device_memory_gb=memory,
                                  return_histograms=True)
        where = f"theiler={theiler}, {precision}, memory={memory}"
        _same_tables(ref, got, where)
        for a, b in zip(ref_h, got_h, strict=True):
            for kind in ("diagonal", "vertical", "white_vertical"):
                assert np.array_equal(getattr(a, kind), getattr(b, kind)), (kind, where)
            assert a.recurrent_points == b.recurrent_points, where


@pytest.mark.requires("torch")
@pytest.mark.parametrize("signal", ["noise", "sine", "lorenz"])
@pytest.mark.parametrize("theiler", [0, 1, 2, 5])
def test_torch_engine_equals_the_reference(signal, theiler):
    """The battery that verified the CPU path -- three systems, four Theiler
    windows -- against the torch engine on the CPU, on an odd size, in one
    band and in bands of a few rows."""
    _compare_engines(_signals(601)[signal], theiler=theiler, device="cpu")


@pytest.mark.slow
@pytest.mark.requires("torch")
@pytest.mark.parametrize("signal", ["noise", "sine", "lorenz"])
@pytest.mark.parametrize("theiler", [0, 1, 2, 5])
def test_torch_engine_equals_the_reference_at_1157_points(signal, theiler):
    """The same at 1157 points, the size that exposed the figure bug of 0.25,
    where the thresholds are sampled rather than exact."""
    _compare_engines(_signals(1157)[signal], theiler=theiler, device="cpu")


@pytest.mark.requires("torch")
@pytest.mark.parametrize("metric", ["chebyshev", "manhattan"])
def test_torch_engine_equals_the_reference_for_other_metrics(metric):
    X = _windows(211, count=3)
    ref = rc.batch_rqa(X, metric=metric, rng=2, engine="numpy")
    got = rc.batch_rqa(X, metric=metric, rng=2, engine="torch", device="cpu")
    _same_tables(ref, got, metric)


def _bytes(precision):
    from recurra.engine.torch_rqa import BYTES_PER_CELL

    return BYTES_PER_CELL[precision]


@pytest.mark.requires("torch")
def test_batch_size_and_band_height_change_nothing():
    """G2, G5: one window per batch or all of them, whole plots or bands of
    three rows -- identical histograms."""
    X = _windows(389, count=5)
    runs = {
        "all at once": rc.batch_rqa(X, rng=0, engine="torch", device="cpu",
                                    return_histograms=True),
        "one by one": rc.batch_rqa(X, rng=0, engine="torch", device="cpu",
                                   batch_size=1, return_histograms=True),
        "bands of 3": rc.batch_rqa(X, rng=0, engine="torch", device="cpu",
                                   batch_size=2,
                                   device_memory_gb=389 * 3 * 2 * _bytes("mixed") / 1e9,
                                   return_histograms=True),
    }
    ref_table, ref_hists = runs.pop("all at once")
    for name, (table, hists) in runs.items():
        _same_tables(ref_table, table, name)
        for a, b in zip(ref_hists, hists, strict=True):
            for kind in ("diagonal", "vertical", "white_vertical"):
                assert np.array_equal(getattr(a, kind), getattr(b, kind)), (name, kind)
    assert runs["bands of 3"][0].attrs["provenance"]["band_rows"] == {389: 3}


@pytest.mark.requires("torch")
def test_mixed_precision_restores_the_double_decision():
    """Single precision decides in float32 and counts the cells within float32
    rounding of epsilon; mixed recomputes exactly those in float64."""
    X = _windows(600, count=3) + 50.0           # far from the origin on purpose
    double = rc.batch_rqa(X, rng=0, engine="torch", device="cpu", precision="double")
    mixed = rc.batch_rqa(X, rng=0, engine="torch", device="cpu", precision="mixed")
    single = rc.batch_rqa(X, rng=0, engine="torch", device="cpu", precision="single")
    _same_tables(double, mixed, "mixed vs double")
    prov = mixed.attrs["provenance"]
    assert prov["cells_rechecked_in_double"] == prov["cells_near_epsilon"]
    assert single.attrs["provenance"]["cells_rechecked_in_double"] == 0
    assert np.allclose(single["RR"], double["RR"], atol=1e-4)


@pytest.mark.requires("torch")
def test_torch_engine_refuses_what_it_cannot_do():
    with pytest.raises(rc.ParameterError, match="supports the metrics"):
        rc.batch_rqa(_windows(100, 2), metric="cosine", engine="torch", device="cpu")
    with pytest.raises(rc.MemoryBudgetError, match="budget"):
        rc.batch_rqa(_windows(100, 2), engine="torch", device="cpu",
                     device_memory_gb=1e-9)


# ----------------------------------------------------- torch engine, on CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("signal", ["noise", "sine", "lorenz"])
@pytest.mark.parametrize("theiler", [0, 1, 2, 5])
@pytest.mark.parametrize("precision", ["double", "mixed"])
def test_cuda_engine_equals_the_reference(signal, theiler, precision):
    """The same battery on the GPU, at 1157 points, in double and in mixed
    precision."""
    _compare_engines(_signals(1157)[signal], theiler=theiler, device="cuda:0",
                     precision=precision)


@pytest.mark.cuda
def test_cuda_engine_is_deterministic():
    """G5: run twice, with PyTorch's deterministic mode on, and with batch
    sizes 1 and all: identical."""
    import torch

    X = _windows(1280, count=6)
    torch.use_deterministic_algorithms(True)
    try:
        a = rc.batch_rqa(X, rng=0, engine="torch", device="cuda:0")
        b = rc.batch_rqa(X, rng=0, engine="torch", device="cuda:0")
        c = rc.batch_rqa(X, rng=0, engine="torch", device="cuda:0", batch_size=1)
    finally:
        torch.use_deterministic_algorithms(False)
    _same_tables(a, b, "run 1 vs run 2")
    _same_tables(a, c, "batch 6 vs 1")
    assert a.attrs["provenance"]["tf32_matmul"] is False


@pytest.mark.cuda
def test_both_gpus_give_the_same_answer():
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    X = _windows(1280, count=4)
    a = rc.batch_rqa(X, rng=0, engine="torch", device="cuda:0")
    b = rc.batch_rqa(X, rng=0, engine="torch", device="cuda:1")
    _same_tables(a, b, "cuda:0 vs cuda:1")


# ------------------------------------------- the windowed path on the engine
def _pac_space():
    from recurra import statespace as sm

    sig = rc.generate_cfc(modality="pac", duration=24.0, fs=250.0, alpha=0.8,
                          preferred_phase=np.pi / 2, snr_db=15.0, rng=42)
    rec = rc.analytic(rc.filterbank(sig.to_recording("s"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    return sm.pac_space(rec, smooth=12.0)


@pytest.mark.requires("torch")
def test_windowed_metrics_are_the_same_on_the_torch_engine():
    """The call a study makes, windowed_recurrence(...).metrics(rqa=True),
    with the histograms computed by the torch engine: the same table, and the
    engine recorded."""
    ss = _pac_space()
    spec = rc.WindowSpec(size=4.0, overlap=0.5, unit="seconds")
    ref = rc.windowed_recurrence(ss, spec, target_rr=0.05, rng=0).metrics(
        rqa=True, engine="numpy")
    got = rc.windowed_recurrence(ss, spec, target_rr=0.05, rng=0).metrics(
        rqa=True, engine="torch", device="cpu")
    _same_tables(ref, got, "windowed metrics")
    assert got.attrs["engine"]["windows_on_device"] == len(got)
    assert ref.attrs["engine"]["engine"] == "numpy"


@pytest.mark.cuda
def test_windowed_metrics_are_the_same_on_cuda():
    ss = _pac_space()
    spec = rc.WindowSpec(size=4.0, overlap=0.5, unit="seconds")
    ref = rc.windowed_recurrence(ss, spec, target_rr=0.05, rng=0).metrics(
        rqa=True, engine="numpy")
    got = rc.windowed_recurrence(ss, spec, target_rr=0.05, rng=0).metrics(
        rqa=True, engine="torch", device="cuda:0")
    _same_tables(ref, got, "windowed metrics on cuda")


def test_mixed_equals_double_on_coordinates_far_from_epsilon():
    """The float32 band widens with the window's reach [DD-125]: two clusters
    4000 apart with epsilon of order one would put float32 rounding (about
    2.4e-4 of a coordinate) far outside a fixed 1e-4 band."""
    pytest.importorskip("torch")
    g = np.random.default_rng(0)
    X = g.standard_normal((400, 3))
    X[:200] += 2000.0
    X[200:] -= 2000.0
    W = np.stack([X, X[::-1].copy()])
    kw = dict(rng=0, engine="torch", device="cpu", return_histograms=True)
    d, hd = rc.batch_rqa(W, precision="double", **kw)
    m, hm = rc.batch_rqa(W, precision="mixed", **kw)
    assert d.equals(m)
    for a, b in zip(hd, hm, strict=True):
        for kind in ("diagonal", "vertical", "white_vertical"):
            assert np.array_equal(getattr(a, kind), getattr(b, kind))


def test_stage_profile_is_off_by_default_and_complete_when_on(monkeypatch):
    """The benchmark's --profile switch: seconds per stage in the provenance,
    nothing when it is off, and the same table either way."""
    pytest.importorskip("torch")
    import recurra.engine.torch_rqa as torch_rqa

    X = _windows(200, count=2)
    plain = rc.batch_rqa(X, rng=0, engine="torch", device="cpu")
    assert "stage_seconds" not in plain.attrs["provenance"]
    monkeypatch.setattr(torch_rqa, "PROFILE", True)
    timed = rc.batch_rqa(X, rng=0, engine="torch", device="cpu")
    stages = timed.attrs["provenance"]["stage_seconds"]
    assert {"threshold", "decide", "vertical", "diagonal", "white_vertical"} <= set(stages)
    assert plain.equals(timed)


def test_mixed_is_the_default_precision_on_the_device():
    """DD-125: the torch engine defaults to mixed, which decides as double."""
    pytest.importorskip("torch")
    X = _windows(200, count=2)
    default = rc.batch_rqa(X, rng=0, engine="torch", device="cpu")
    assert default.attrs["provenance"]["precision"] == "mixed"
    reference = rc.batch_rqa(X, rng=0, engine="numpy", n_jobs=1)
    _same_tables(reference, default, "default precision")
