"""RQA of many windows in one call, on the CPU or on a GPU.

Design note (DD-125) -- a batch is a list of independent plots.

``batch_rqa`` takes windows that are already trajectories -- a ``(B, N, d)``
array, or a list of arrays or state spaces of any lengths -- and returns one
row of RQA per window. Each row is exactly what the single-plot path gives
for that window::

    rqa(recurrence_plot(X, threshold(X, target_rr=q, theiler=w, rng=seed)))

with ``seed`` the integer ``rng`` (or, for a generator, one seed drawn from
it per window length). Windows of the same length therefore share the same
sampled pairs for their thresholds, as they already do when one integer seed
is passed to every window. That is what lets a device estimate a batch of
thresholds at once: the pairs are drawn once.

Neither the engine nor the batch size can change a result: the windows of a
batch never interact, every count is an integer, and the torch engine
reproduces the reference's threshold arithmetic (see
:mod:`recurra.engine.torch_rqa`). The provenance of the table records which
engine and device ran it.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..config import SeedLike, get_config, resolve_rng
from ..core.points import as_points
from ..exceptions import ParameterError
from .histogram import LineHistogram, line_histogram
from .metrics import rqa_from_histogram


def _windows(windows) -> list[np.ndarray]:
    if isinstance(windows, np.ndarray) and windows.ndim == 3:
        return [np.asarray(windows[b], dtype=float) for b in range(windows.shape[0])]
    if hasattr(windows, "weighted_coords"):
        return [np.asarray(windows.weighted_coords, dtype=float)]
    if isinstance(windows, np.ndarray):
        return [as_points(windows, name="batch_rqa")]
    out = []
    for w in windows:
        if hasattr(w, "weighted_coords"):
            out.append(np.asarray(w.weighted_coords, dtype=float))
        else:
            out.append(as_points(w, name="batch_rqa"))
    return out


def _seeds(rng: SeedLike, lengths) -> dict[int, int]:
    """One seed per window length: the integer itself, or drawn once."""
    if isinstance(rng, (int, np.integer)) and not isinstance(rng, bool):
        return {n: int(rng) for n in lengths}
    gen = resolve_rng(rng)
    return {n: int(gen.integers(0, 2**63 - 1)) for n in sorted(lengths)}


def _numpy_window(X, *, seed, target_rr, epsilon, theiler, metric, n_samples):
    from ..recurrence.builders import recurrence_plot
    from ..threshold.estimate import threshold

    if epsilon is None:
        th = threshold(X, target_rr=target_rr, theiler=theiler, metric=metric,
                       n_samples=n_samples, rng=seed)
    else:
        th = threshold(X, mode="fixed", value=float(epsilon), theiler=theiler,
                       metric=metric, n_samples=n_samples, rng=seed)
    rm = recurrence_plot(X, th, metric=metric, store="stream")
    return th, line_histogram(rm)


def _numpy_chunk(chunk, *, rng=None, **kw):
    return [_numpy_window(X, seed=seed, **kw) for X, seed in chunk]


def batch_rqa(windows, *, target_rr: float = 0.05, epsilon: float | None = None,
              theiler: int = 1, metric: str = "euclidean", l_min: int = 8,
              v_min: int = 8, w_min: int = 2, denominator: str = "theiler_corrected",
              metrics: Sequence[str] | str = "all", n_samples: int = 200_000,
              rng: SeedLike = None, engine: str = "auto", device: str | None = None,
              precision: str = "mixed", batch_size: int | None = None,
              device_memory_gb: float | None = None, n_jobs: int | None = None,
              return_histograms: bool = False):
    """Recurrence quantification of many windows, one row per window [DD-125].

    Parameters
    ----------
    windows
        A ``(B, N, d)`` array, or a sequence of ``(N_i, d)`` arrays or state
        spaces (their weighted coordinates are used). 1-D arrays are scalar
        series [DD-123].
    target_rr, epsilon
        Each window's threshold: the quantile ``target_rr`` of its sampled
        distances (default), or the fixed ``epsilon`` for every window.
    theiler, metric, l_min, v_min, w_min, denominator, metrics
        As for :func:`recurra.threshold`, :func:`recurra.recurrence_plot` and
        :func:`recurra.rqa`; the line floors default to 8 [DD-98].
    rng
        An integer seed is used as is for every window, so each row equals the
        single-plot path called with that seed. A generator, or ``None``, gives
        one seed per window length.
    engine, device
        ``"auto"`` (default) runs on the first CUDA device when PyTorch sees
        one and the windows have at least ``AUTO_MIN_POINTS`` points, and on
        the CPU otherwise, saying which it chose. ``"numpy"`` is the reference;
        ``"torch"`` asks for the device and fails if it is not there, never
        falling back to the CPU [DD-125].
    precision
        Torch engine: ``"mixed"`` (default: float32, with every cell near
        epsilon recomputed in float64, so the decisions of ``"double"`` and of
        the reference, faster), ``"double"`` or ``"single"`` (float32 only,
        which can differ from the reference next to epsilon) [DD-125].
    batch_size, device_memory_gb
        Torch engine: windows per batch and the device memory it may use
        (default ``Config.gpu_memory_budget_gb``). Neither changes a result.
    n_jobs
        Numpy engine: worker processes [DD-82]. ``None`` (default) uses every
        physical core when the batch is large enough to gain from it, and says
        so once [DD-126]. Torch engine: unused.
    return_histograms
        Also return the list of line histograms, in window order.

    Returns
    -------
    pandas.DataFrame
        ``window``, ``n_points``, ``epsilon``, ``achieved_rr`` (on the
        threshold's sample), then the RQA metrics, ``RR`` being the plot's own
        rate. ``table.attrs["provenance"]`` records the engine, the device and,
        for torch, its versions, precision, batch plan and how many cells lay
        within rounding of epsilon.
    """
    from ..engine import resolve_engine

    if epsilon is None and not 0.0 < target_rr < 1.0:
        raise ParameterError(f"target_rr must be in (0, 1), got {target_rr}")
    if epsilon is not None and epsilon <= 0:
        raise ParameterError(f"epsilon must be positive, got {epsilon}")
    ws = _windows(windows)
    if not ws:
        raise ParameterError("batch_rqa needs at least one window")
    lengths = sorted({w.shape[0] for w in ws})
    choice = resolve_engine(engine, device, n_points=min(lengths))
    seeds = _seeds(rng, lengths)
    common = dict(target_rr=target_rr, epsilon=epsilon, theiler=int(theiler),
                  metric=metric, n_samples=n_samples)

    thresholds: list[Any] = [None] * len(ws)
    hists: list[LineHistogram | None] = [None] * len(ws)
    prov: dict[str, Any] = choice.provenance()

    if choice.engine == "numpy":
        from ..parallel.runner import auto_jobs, parallel_map

        jobs = auto_jobs(n_jobs, n_items=len(ws),
                         work_cells=float(sum(w.shape[0] ** 2 for w in ws)))
        items = [(w, seeds[w.shape[0]]) for w in ws]
        if jobs <= 1:
            results = _numpy_chunk(items, **common)
        else:
            size = max(1, -(-len(items) // (4 * jobs)))     # a few chunks per worker
            chunks = [items[k:k + size] for k in range(0, len(items), size)]
            results = [r for part in parallel_map(_numpy_chunk, chunks, n_jobs=jobs,
                                                  backend="process", pass_rng=False,
                                                  **common)
                       for r in part]
        for k, (th, h) in enumerate(results):
            thresholds[k], hists[k] = th, h
        prov.update({"n_jobs": jobs})
    else:
        from ..engine.torch_rqa import run_batches

        budget = (device_memory_gb if device_memory_gb is not None
                  else get_config().gpu_memory_budget_gb)
        info = run_batches(ws, seeds, device=choice.device, precision=precision,
                           batch_size=batch_size, device_memory_gb=budget,
                           thresholds=thresholds, hists=hists, **common)
        prov.update(info)

    rows = []
    for k, (X, th, h) in enumerate(zip(ws, thresholds, hists, strict=True)):
        values = rqa_from_histogram(h, l_min=l_min, v_min=v_min, w_min=w_min,
                                    denominator=denominator, metrics=metrics)
        rows.append({"window": k, "n_points": X.shape[0],
                     "epsilon": float(th.scalar), "achieved_rr": th.achieved_rr,
                     **values})
    table = pd.DataFrame(rows)
    prov.update({"theiler": int(theiler), "metric": metric, "l_min": l_min,
                 "v_min": v_min, "w_min": w_min, "n_samples": n_samples,
                 "target_rr": None if epsilon is not None else target_rr,
                 "epsilon": epsilon})
    table.attrs["provenance"] = prov
    return (table, hists) if return_histograms else table
