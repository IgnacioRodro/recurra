"""Meta-recurrence: recurrence of the measurements taken window by window.

Design note (DD-119) -- the second level of the analysis.

A windowed analysis turns one record into a sequence of measurements, one row
per window: RQA metrics of a phase-amplitude space, a coupling index, a
dynamical invariant. That sequence is itself a multivariate time series, with
one sample per window, and the questions asked of a signal can be asked of it:
does the state of the measurements recur, are there laminar stretches, does
the record pass through distinct regimes? The project this library serves
states it directly -- treat coupling "not as a static index but as a dynamic
object whose temporal evolution can be analysed with RQA".

So :func:`meta_space` turns the table into a :class:`StateSpace` whose time
axis is the window centres (``fs`` is windows per second), and every tool of
the library applies to it unchanged: :func:`recurrence_plot`,
:func:`cross_recurrence_plot`, :func:`joint_recurrence_plot`, :func:`rqa`,
the dynamical measures, the figures. :func:`meta_recurrence` is the shortcut
for the common case.

Three things differ from a signal, and each has a default:

* **Overlapping windows share samples**, so neighbouring rows are correlated
  by construction, not by dynamics. ``theiler="auto"`` excludes every pair of
  windows that share a sample: ``ceil(size / step)``, which is 2 at 50%
  overlap, 4 at 75% and 1 without overlap. The first study did this by hand.
* **The columns have different units**, so each is rescaled on its own
  (``scaling="zscore"`` by default; ``"rank"``, ``"robust"``, ``"none"``).
* **Some columns are bookkeeping, not measurements**: window indices, times,
  thresholds, floors, quality flags. They are left out, and so is the
  recurrence rate when the threshold fixed it -- a constant by design.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import CaveatWarning, ParameterError
from ..provenance import Step
from ..statespace.core import CoordGroup, StateSpace

#: Columns of a windowed metrics table that describe the analysis rather than
#: measure the signal. Never part of a meta-space unless asked for by name.
BOOKKEEPING = frozenset({
    "window", "window_label", "start_sample", "stop_sample", "n_samples",
    "t_start_s", "t_stop_s", "t_center_s", "subject", "group", "channel",
    "n_rows", "n_cols", "n_points", "dim", "fs", "theiler", "metric",
    "epsilon", "target_rr", "estimated_rr", "threshold_mode", "scope",
    "l_min", "v_min", "w_min", "denominator", "store", "tile_size", "dtype",
    "precision", "kept", "n_masks", "seconds", "n_windows",
})
BOOKKEEPING_PREFIXES = ("qc_", "fine_scale", "samples_per_turn", "sweep_")
BOOKKEEPING_SUFFIXES = ("_ok",)
#: The recurrence rate is fixed by a target-rate threshold, so it carries no
#: information then [ROADMAP guard: RR is not a feature under target_rr].
RATE_COLUMNS = ("RR", "recurrence_rate")
SCALINGS = ("zscore", "rank", "robust", "none")


def _is_bookkeeping(name: str) -> bool:
    return (name in BOOKKEEPING or name.startswith(BOOKKEEPING_PREFIXES)
            or name.endswith(BOOKKEEPING_SUFFIXES))


def _table_of(source) -> pd.DataFrame:
    """The per-window table behind any accepted source."""
    if isinstance(source, pd.DataFrame):
        return source
    if hasattr(source, "windows") and hasattr(source, "metrics"):
        table = getattr(source, "_metrics", None)
        return table if table is not None else source.metrics(rqa=True)
    raise ParameterError(
        "meta_space needs a WindowedRecurrence or a DataFrame with one row per "
        f"window; got {type(source).__name__}. For a plain array, wrap it in a "
        "DataFrame with a t_center_s column.")


def meta_theiler(source) -> int:
    """Theiler window that excludes every pair of windows sharing a sample.

    Windows of ``size`` samples taken every ``step`` samples overlap when they
    are fewer than ``size / step`` positions apart, so the window is
    ``ceil(size / step)``: 2 at 50% overlap, 1 without overlap [DD-119].
    """
    if hasattr(source, "windows"):
        ws = list(source.windows)
        if len(ws) < 2:
            return 1
        size, step = ws[0].stop - ws[0].start, ws[1].start - ws[0].start
    else:
        table = _table_of(source)
        if not {"start_sample", "stop_sample"} <= set(table.columns) or len(table) < 2:
            return 1
        starts = table["start_sample"].to_numpy()
        size = int(table["stop_sample"].iloc[0] - table["start_sample"].iloc[0])
        step = int(np.median(np.diff(np.sort(starts))))
    if step <= 0:
        return 1
    return max(1, math.ceil(size / step))


def _scale(column: np.ndarray, how: str) -> np.ndarray:
    x = np.asarray(column, dtype=float)
    if how == "none":
        return x
    if how == "zscore":
        sd = x.std(ddof=0)
        return (x - x.mean()) / sd if sd > 0 else x - x.mean()
    if how == "robust":
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        iqr = q3 - q1
        return (x - med) / iqr if iqr > 0 else x - med
    from scipy.stats import rankdata

    return (rankdata(x) - 1) / max(x.size - 1, 1)


def meta_space(source, metrics: Sequence[str] | None = None, *,
               scaling: str = "zscore", missing: str = "drop",
               subject: str | None = None) -> StateSpace:
    """The sequence of per-window measurements as a state space [DD-119].

    Parameters
    ----------
    source
        A :class:`WindowedRecurrence` (its last metrics table is used, or
        ``metrics(rqa=True)`` if none was computed yet) or a DataFrame with one
        row per window, such as ``wr.metrics(...)`` or any table of your own.
        Rows must be in time order; a ``t_center_s`` column gives the time axis.
    metrics
        Columns to use. ``None`` (default) takes every numeric column that is
        a measurement: bookkeeping columns are left out, and so are the
        recurrence-rate columns when the table's threshold fixed the rate, and
        columns that are constant over the windows.
    scaling
        Per-column rescaling, because the columns have different units:
        ``"zscore"`` (default), ``"rank"`` (to [0, 1]), ``"robust"`` (median
        and IQR) or ``"none"``.
    missing
        ``"drop"`` (default) removes windows with a missing value in a chosen
        column and warns; ``"error"`` refuses them.

    Returns
    -------
    StateSpace
        One point per window and one coordinate per measurement, ``fs`` in
        windows per second and ``t0`` at the first window centre.
    """
    if scaling not in SCALINGS:
        raise ParameterError(f"scaling must be one of {SCALINGS}, got {scaling!r}")
    if missing not in ("drop", "error"):
        raise ParameterError(f"missing must be 'drop' or 'error', got {missing!r}")
    table = _table_of(source)
    if len(table) < 3:
        raise ParameterError(f"a meta-space needs at least 3 windows, got {len(table)}")

    notes: list[str] = []
    if metrics is None:
        numeric = [c for c in table.columns
                   if pd.api.types.is_numeric_dtype(table[c])
                   and not pd.api.types.is_bool_dtype(table[c])
                   and not _is_bookkeeping(str(c))]
        fixed_rate = ("target_rr" in table.columns
                      and np.isfinite(pd.to_numeric(table["target_rr"],
                                                    errors="coerce")).any())
        if fixed_rate:
            numeric = [c for c in numeric if c not in RATE_COLUMNS]
            notes.append("recurrence rate left out: the threshold fixed it")
        constant = [c for c in numeric if table[c].nunique(dropna=True) <= 1]
        if constant:
            notes.append(f"constant columns left out: {constant}")
        metrics = [c for c in numeric if c not in constant]
        if not metrics:
            raise ParameterError(
                "no measurement columns found; compute some first, e.g. "
                "wr.metrics(rqa=True), or name them with metrics=[...]")
    else:
        unknown = [c for c in metrics if c not in table.columns]
        if unknown:
            raise ParameterError(f"columns not in the table: {unknown}; have "
                                 f"{list(table.columns)}")
        metrics = list(metrics)

    values = table[metrics].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    if bad.any():
        if missing == "error":
            raise ParameterError(
                f"{int(bad.sum())} window(s) have missing values in {metrics}; "
                "drop them or pass missing='drop'")
        notes.append(f"{int(bad.sum())} window(s) with missing values dropped; "
                     "Theiler and line lengths count positions in what remains")
        warnings.warn(f"meta_space: {notes[-1]}", CaveatWarning, stacklevel=2)
    kept = table.loc[~bad]
    values = values.loc[~bad]
    if len(kept) < 3:
        raise ParameterError("fewer than 3 windows left after dropping missing values")

    X = np.column_stack([_scale(values[c].to_numpy(), scaling) for c in metrics])
    if "t_center_s" in kept.columns:
        t = kept["t_center_s"].to_numpy(dtype=float)
        spacing = float(np.median(np.diff(t))) if t.size > 1 else 1.0
        fs, t0 = (1.0 / spacing if spacing > 0 else 1.0), float(t[0])
    else:
        fs, t0 = 1.0, 0.0
        notes.append("no t_center_s column: time is the window index")

    groups = tuple(CoordGroup(label=str(c), start=k, stop=k + 1, kind="raw",
                              source=str(c), params={"scale": scaling})
                   for k, c in enumerate(metrics))
    parent = tuple(getattr(source, "provenance", ()) or ())
    step = Step("meta_space", {"n_windows": int(len(kept)), "metrics": list(metrics),
                               "scaling": scaling, "fs_windows_per_s": fs},
                caveats=tuple(notes))
    who = subject if subject is not None else (
        str(kept["subject"].iloc[0]) if "subject" in kept.columns
        else str(getattr(source, "subject", "")))
    return StateSpace(coords=X, groups=groups, fs=fs, t0=t0, route="external",
                      provenance=parent + (step,),
                      meta={"subject": who, "builder": "meta_space",
                            "meta_theiler": meta_theiler(source),
                            "windows": kept["window"].to_list()
                            if "window" in kept.columns else list(range(len(kept))),
                            "notes": notes})


def meta_recurrence(source, metrics: Sequence[str] | None = None, *,
                    kind: str = "rp", other=None, scaling: str = "zscore",
                    theiler: int | str = "auto", threshold=None,
                    target_rr: float = 0.05, metric: str = "euclidean",
                    missing: str = "drop", rng=None, **kwargs: Any):
    """Recurrence plot of the per-window measurements: the meta-RP [DD-119].

    Parameters
    ----------
    source
        A :class:`WindowedRecurrence` or a per-window DataFrame (see
        :func:`meta_space`). For ``kind="jrp"`` a list of them.
    kind
        ``"rp"`` (default); ``"crp"`` against ``other`` (another windowed
        analysis with the same columns, e.g. a second channel); ``"jrp"`` across
        a list of sources, each with its own threshold.
    theiler
        ``"auto"`` (default) excludes every pair of windows that share a
        sample, :func:`meta_theiler`; an integer overrides it. Not used by
        ``kind="crp"``.
    threshold, target_rr, metric
        As in :func:`recurrence_plot`. A meta-series is short -- tens of
        points -- so consider a higher ``target_rr`` than for a signal.

    Every other keyword goes to the recurrence builder. The result is an
    ordinary :class:`RecurrenceMatrix`: ``rqa()``, the figures and the
    exports all apply, with line lengths in windows and times in seconds.
    """
    from ..recurrence.builders import cross_recurrence_plot, joint_recurrence_plot, recurrence_plot

    if kind not in ("rp", "crp", "jrp"):
        raise ParameterError(f"kind must be 'rp', 'crp' or 'jrp', got {kind!r}")
    opts = dict(scaling=scaling, missing=missing)
    if kind == "crp":
        if other is None:
            raise ParameterError("kind='crp' needs other=<a second windowed analysis>")
        a = meta_space(source, metrics, **opts)
        b = meta_space(other, [g.label for g in a.groups], **opts)
        if a.n_points != b.n_points:
            n = min(a.n_points, b.n_points)
            a, b = a.window(0, n), b.window(0, n)
        return cross_recurrence_plot(a, b, threshold, metric=metric,
                                     target_rr=target_rr, rng=rng, **kwargs)
    if kind == "jrp":
        spaces = [meta_space(s, metrics, **opts) for s in source]
        w = (max(meta_theiler(s) for s in source) if theiler == "auto" else int(theiler))
        ths = None if threshold is None else [threshold] * len(spaces)
        return joint_recurrence_plot(spaces, ths, theiler=w, metric=metric,
                                     target_rr=target_rr, rng=rng, **kwargs)
    space = meta_space(source, metrics, **opts)
    w = space.meta["meta_theiler"] if theiler == "auto" else int(theiler)
    if space.n_points < 20:
        warnings.warn(
            f"meta_recurrence: {space.n_points} windows. Line-based measures of a "
            "plot this small rest on a handful of lines; read them as descriptive.",
            CaveatWarning, stacklevel=2)
    return recurrence_plot(space, threshold, theiler=w, metric=metric,
                           target_rr=target_rr, rng=rng, **kwargs)
