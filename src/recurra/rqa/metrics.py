"""RQA metrics, computed from line-length histograms.

Design note (DD-58) -- the denominator is a choice, and it must be stated.

DET is the fraction of recurrent points that lie on diagonal lines. Under a
Theiler window the numerator naturally excludes the band, because the lines
are counted there; the denominator is a decision. The old draft counted all
recurrent points including the excluded band, which biases DET and LAM
downwards by a fixed and unreported amount.

``denominator="theiler_corrected"`` (the default) counts only recurrent points
outside the band, so numerator and denominator describe the same region.
``denominator="all"`` reproduces the older convention for comparison with
published values that used it. The choice is recorded on every result.

Design note (DD-59) -- entropy needs its normalisation stated too.

ENTR is the Shannon entropy of the diagonal line-length distribution. Its
maximum depends on how many lengths the distribution *could* take, which grows
with the record, so raw ENTR is not comparable between recordings of different
size. ``ENTR_norm`` divides by log of the number of admissible lengths,
``Lmax - l_min + 1`` [DD-111]. An earlier version divided by log of the number
of lengths *present*, a random quantity that a single rare long line moves.
Both are reported; the comparability layer knows that the raw one needs a
matched point count.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import GeometryWarning, ParameterError
from .histogram import LineHistogram, line_histogram

DENOMINATORS = ("theiler_corrected", "all")

#: Everything :func:`rqa` can return, grouped as the comparability layer groups it.
METRIC_NAMES = (
    "RR",
    "DET", "L", "Lmax", "DIV", "ENTR", "ENTR_norm",
    "LAM", "TT", "Vmax",
    "W", "Wmax", "ENTW", "RTE",
)


def _entropy(counts: np.ndarray, n_possible: int) -> tuple[float, float]:
    """Shannon entropy of a length distribution, raw and normalised.

    Design note (DD-111). The normalisation divides by ``log(n_possible)``,
    the number of lengths the distribution may take between the floor and the
    longest line found. Dividing by the number of lengths *present* -- the
    previous convention -- made the normalised value depend on a random
    count: one rare long line adds a term to the denominator and moves the
    number without any change in the shape of the distribution.
    """
    total = counts.sum()
    if total <= 0:
        return 0.0, 0.0
    p = counts[counts > 0] / total
    h = float(-np.sum(p * np.log(p)))
    return h, (h / np.log(n_possible) if n_possible > 1 else 0.0)


def _line_stats(h: np.ndarray, l_min: int) -> dict[str, float]:
    counts = h.copy()
    counts[:l_min] = 0
    lengths = np.arange(counts.size)
    n_lines = float(counts.sum())
    points = float(np.sum(counts * lengths))
    longest = float(np.max(np.flatnonzero(counts), initial=0))
    mean = points / n_lines if n_lines else 0.0
    n_possible = int(longest) - int(l_min) + 1 if longest >= l_min else 0
    ent, ent_norm = _entropy(counts, n_possible)
    return {"points": points, "n_lines": n_lines, "mean": mean,
            "max": longest, "entropy": ent, "entropy_norm": ent_norm}


def line_length_sweep(h: LineHistogram, *, values: Sequence[int] = (2, 3, 4, 6, 8,
                      12, 16, 20), metrics: Sequence[str] = ("DET", "LAM"),
                      **kwargs) -> pd.DataFrame:
    """DET and LAM at a range of minimum line lengths, from one histogram.

    Design note (DD-98) -- the minimum line length has to be chosen from the
    data, and it is nearly free to choose it well.

    ``l_min`` and ``v_min`` decide what counts as a line, and the right value
    depends on how smooth the trajectory is. On a PAC space built from a
    smoothed envelope almost every recurrent point lies on a diagonal, so DET
    sits at 0.996 with a spread of 0.0005 across subjects: a constant, unable
    to separate anything. The same space can give DET a range of 0.43 at
    ``l_min=8``.

    But the opposite failure is just as easy: raise it too far and the metric
    is crushed to zero. Measured on one synthetic space, LAM went from a range
    of 0.47 at ``v_min=2`` to exactly zero at ``v_min=4``.

    There is no default that is right for every space, and the histogram
    already holds the answer: every metric at every threshold is a sum over
    counts already accumulated. Sweeping costs milliseconds against the
    minutes the plot took.
    """
    rows = []
    # The sweep exists to find the floor at which a metric has room. Its whole
    # job is to evaluate floors that do not work, so warning about each one is
    # noise about the very thing being measured -- and there are as many
    # warnings as floors, per subject, per channel. The headroom column
    # reports the same fact once, quantitatively.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeometryWarning)
        computed_rows = [rqa_from_histogram(h, l_min=v, v_min=v, **kwargs)
                         for v in values]
    for value, computed in zip(values, computed_rows, strict=False):
        row = {"min_length": int(value)}
        row.update({k: computed[k] for k in metrics if k in computed})
        row["n_diagonal_lines"] = h.n_lines("diagonal", l_min=value)
        row["n_vertical_lines"] = h.n_lines("vertical", l_min=value)
        rows.append(row)
    frame = pd.DataFrame(rows)
    # The useful column: how much room the metric has left at each choice.
    for name in metrics:
        if name in frame:
            frame[f"{name}_headroom"] = np.minimum(frame[name], 1.0 - frame[name])
    return frame


def rqa_from_histogram(h: LineHistogram, *, l_min: int = 8, v_min: int = 8,
                       w_min: int = 2, denominator: str = "theiler_corrected",
                       metrics: Sequence[str] | str = "all") -> dict[str, float]:
    """Compute the metrics from an already-accumulated histogram."""
    if denominator not in DENOMINATORS:
        raise ParameterError(
            f"denominator must be one of {DENOMINATORS}, got {denominator!r}")

    diag = _line_stats(h.diagonal, l_min)
    vert = _line_stats(h.vertical, v_min)
    white = _line_stats(h.white_vertical, w_min)

    if denominator == "theiler_corrected":
        # every recurrent point counted is outside the band, because the band
        # is stored as zero and never contributes to a line
        denom = float(h.recurrent_points)
    else:
        # The classical convention counted the line of identity as recurrent.
        # Only that line can be restored exactly: its cells are recurrent by
        # definition. Cells of a wider Theiler band were never evaluated, so
        # they are not added back, and "all" with theiler > 1 means "plus the
        # LOI", not "plus the band" [DD-58].
        denom = float(h.recurrent_points + (h.n_rows if h.theiler else 0))

    out = {
        "RR": h.recurrence_rate,
        "DET": diag["points"] / denom if denom else 0.0,
        "L": diag["mean"],
        "Lmax": diag["max"],
        "DIV": 1.0 / diag["max"] if diag["max"] else np.inf,
        "ENTR": diag["entropy"],
        "ENTR_norm": diag["entropy_norm"],
        "LAM": vert["points"] / denom if denom else 0.0,
        "TT": vert["mean"],
        "Vmax": vert["max"],
        "W": white["mean"],
        "Wmax": white["max"],
        "ENTW": white["entropy"],
        "RTE": white["entropy_norm"],
    }
    # Design note (DD-98). DET and LAM are bounded above by 1 and a smooth
    # trajectory pushes them against it: measured on a phase-amplitude space,
    # DET across coupling strengths from 0.1 to 0.9 spanned 0.804 to 0.952 with
    # l_min = 2, and 0.177 to 0.608 with l_min = 8. Pinned near 1 the metric
    # cannot separate anything, however real the difference.
    #
    # The floors default to 8 for both diagonals and verticals [DD-98; v_min
    # moved from 2 to 8 when a real corpus saturated LAM at 0.998]. Both ends
    # are failure modes -- a metric against its ceiling and one crushed to zero
    # are equally unable to separate anything -- so both are said out loud, and
    # line_length_sweep() answers the question from the data.
    for name, floor, bound in (("DET", l_min, "diagonal"), ("LAM", v_min, "vertical")):
        value = out.get(name)
        if value is None or not np.isfinite(value):
            continue
        knob = "l_min" if name == "DET" else "v_min"
        if value > 0.99:
            warnings.warn(
                f"rqa: {name} = {value:.4f} is against its ceiling of 1 at "
                f"{knob}={floor}. On a smooth trajectory nearly every recurrent "
                f"point falls on a {bound} line, so the metric has no room left "
                f"to vary. Raise {knob}; line_length_sweep() shows how far, "
                "from your own data [DD-98].",
                GeometryWarning, stacklevel=2)
        elif value < 0.01:
            warnings.warn(
                f"rqa: {name} = {value:.4f} is crushed to zero at {knob}={floor}. "
                f"There are almost no {bound} lines that long, so the metric is "
                f"as unable to separate anything as a saturated one. Lower "
                f"{knob}; line_length_sweep() shows how far [DD-98].",
                GeometryWarning, stacklevel=2)

    if metrics != "all":
        unknown = set(metrics) - set(METRIC_NAMES)
        if unknown:
            raise ParameterError(f"unknown metric(s) {sorted(unknown)}; "
                                 f"available: {list(METRIC_NAMES)}")
        out = {k: out[k] for k in metrics}
    return out


def rqa(source, *, l_min: int = 8, v_min: int = 8, w_min: int = 2,
        theiler: int | None = None, denominator: str = "theiler_corrected",
        metrics: Sequence[str] | str = "all", tile_size: int | None = None,
        return_histogram: bool = False, as_frame: bool = False, **kwargs):
    """Recurrence quantification of a recurrence structure.

    Parameters
    ----------
    source
        A :class:`~recurra.RecurrenceMatrix`, a :class:`LineHistogram`, or
        anything :func:`recurra.recurrence_plot` accepts, in which case the
        plot is built first with the remaining keyword arguments.
    l_min, v_min, w_min
        Shortest diagonal, vertical and white vertical line that counts.

        **Both default to 8, not the classical 2** [DD-98]. On smooth
        trajectories -- a phase circle with a low-passed envelope is the case
        that forced this -- ``l_min=2`` puts DET at 0.996 with a spread of
        0.0005 across a real corpus, a constant that cannot separate anything;
        8 gives it a range of 0.43. ``v_min=2`` saturates LAM the same way:
        0.9983 on a real delta-gamma corpus.

        Neither number is right for every space, and the right one is cheap to
        find: :func:`line_length_sweep` reports both metrics at every floor
        from the histogram already accumulated.

        ``l_min=8`` is a departure from the literature, so it is stated on
        every result: the conventions are recorded in the frame, and a DET
        compared against a published one computed at 2 is a comparison of two
        different quantities. Use :func:`line_length_sweep` to choose from your
        own data rather than trusting any default.
    denominator
        ``"theiler_corrected"`` (default) or ``"all"`` [DD-58]. ``"all"`` adds
        the line of identity back to the denominator, the classical convention;
        with ``theiler > 1`` the rest of the band is not recoverable and is
        not added.

    Returns
    -------
    dict, or a one-row DataFrame when ``as_frame=True``.
    """
    from ..recurrence.core import RecurrenceMatrix

    hist = None
    rm = None
    if isinstance(source, LineHistogram):
        hist = source
    elif isinstance(source, RecurrenceMatrix):
        rm = source
    else:
        from ..recurrence.builders import recurrence_plot

        rm = recurrence_plot(source, theiler=1 if theiler is None else theiler,
                             **kwargs)
    if hist is None:
        hist = line_histogram(rm, tile_size=tile_size)

    values = rqa_from_histogram(hist, l_min=l_min, v_min=v_min, w_min=w_min,
                                denominator=denominator, metrics=metrics)
    if as_frame:
        row: dict[str, Any] = {
            "n_rows": hist.n_rows, "n_cols": hist.n_cols,
            "theiler": hist.theiler, "l_min": l_min, "v_min": v_min,
            "w_min": w_min, "denominator": denominator, **values,
        }
        if rm is not None:
            row = {"kind": rm.kind, "dim": rm.dim, "metric": rm.metric,
                   "epsilon": rm.threshold.scalar,
                   "target_rr": rm.threshold.target, **row}
        values = pd.DataFrame([row])
    return (values, hist) if return_histogram else values


def diagonal_profile(rm, *, max_offset: int | None = None,
                     tile_size: int | None = None) -> pd.DataFrame:
    """Recurrence density along each diagonal offset.

    For a cross recurrence plot this is the quantity that carries lag and
    direction: the offset of the maximum is the delay at which one trajectory
    best matches the other, and asymmetry about zero says which leads.
    """
    n, m = rm.shape
    k_max = int(max_offset if max_offset is not None else max(n, m) - 1)
    counts = np.zeros(2 * k_max + 1, dtype=np.int64)
    totals = np.zeros(2 * k_max + 1, dtype=np.int64)

    ts = int(tile_size or rm.tile_size)
    for i0 in range(0, n, ts):
        i1 = min(i0 + ts, n)
        for j0 in range(0, m, ts):
            j1 = min(j0 + ts, m)
            b = rm._block(i0, i1, j0, j1).astype(bool)
            h, w = b.shape
            for d in range(-(h - 1), w):
                k = (j0 + max(d, 0)) - (i0 + max(-d, 0))
                if abs(k) > k_max:
                    continue
                seg = np.diagonal(b, offset=d)
                counts[k + k_max] += int(seg.sum())
                totals[k + k_max] += seg.size

    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(totals > 0, counts / totals, np.nan)
    offsets = np.arange(-k_max, k_max + 1)
    return pd.DataFrame({
        "offset": offsets,
        "lag_s": offsets / rm.fs if rm.fs else np.nan,
        "density": density, "n_cells": totals,
    })


def crqa(rm, *, l_min: int = 8, v_min: int = 8, w_min: int = 2,
         denominator: str = "theiler_corrected", max_offset: int | None = None,
         as_frame: bool = False, **kwargs):
    """RQA of a cross recurrence plot, plus the quantities specific to one.

    Adds the diagonal profile's peak -- the lag at which the two trajectories
    match best -- and the asymmetry of the profile about zero lag, which is
    zero for a symmetric relation and signed when one trajectory leads.
    """
    values, hist = rqa(rm, l_min=l_min, v_min=v_min, w_min=w_min,
                       denominator=denominator, return_histogram=True, **kwargs)
    prof = diagonal_profile(rm, max_offset=max_offset)
    d = prof["density"].to_numpy(dtype=float)
    offsets = prof["offset"].to_numpy()
    ok = np.isfinite(d) & (prof["n_cells"].to_numpy() > 10)

    if ok.any():
        peak = int(offsets[ok][np.argmax(d[ok])])
        pos = float(np.nansum(d[ok & (offsets > 0)]))
        neg = float(np.nansum(d[ok & (offsets < 0)]))
        asym = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0
    else:
        peak, asym = 0, float("nan")

    values = dict(values)
    values.update({
        "peak_offset": peak,
        "peak_lag_s": peak / rm.fs if rm.fs else np.nan,
        "peak_density": float(np.nanmax(d[ok])) if ok.any() else np.nan,
        "profile_asymmetry": asym,
    })
    return pd.DataFrame([values]) if as_frame else values
