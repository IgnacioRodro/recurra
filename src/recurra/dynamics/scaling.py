"""Finding the region where a log-log curve is straight.

Design note (DD-61) -- the scaling region is the estimate.

D2 and the Lyapunov exponent are both slopes of a curve that is straight only
over part of its range. Below the straight part the curve is dominated by
noise and by the finite number of pairs; above it, by the finite size of the
attractor. The old draft took a fixed percentile band -- the 5th to 50th
percentile of the distance distribution -- which is a guess that happens to be
reasonable for some attractors and silently wrong for others.

Here the region is found: the widest window of the curve whose local slope is
most nearly constant, subject to a minimum width. The window that was used is
returned with the estimate, so a reader can see what was fitted rather than
trusting that something sensible was.

Nothing about this makes the estimate correct. It makes it *auditable*, which
is the most a library can offer for a quantity whose textbook definition is a
limit that finite data cannot reach.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import ParameterError


@dataclass(frozen=True)
class ScalingRegion:
    """The window of a log-log curve that was fitted, and how good it was."""

    start: int
    stop: int
    slope: float
    intercept: float
    r_squared: float
    slope_std: float
    x_lo: float
    x_hi: float
    n_points: int
    criterion: str = ""
    warning: str | None = None

    @property
    def width_decades(self) -> float:
        return float((self.x_hi - self.x_lo) / np.log(10.0))

    def to_row(self) -> dict[str, Any]:
        return {"slope": self.slope, "r_squared": self.r_squared,
                "slope_std": self.slope_std, "region_start": self.x_lo,
                "region_stop": self.x_hi, "region_decades": self.width_decades,
                "region_points": self.n_points, "criterion": self.criterion,
                "warning": self.warning or ""}


def local_slopes(x: np.ndarray, y: np.ndarray, half_width: int = 2) -> np.ndarray:
    """Slope of y against x estimated locally, by a small centred fit."""
    n = x.size
    out = np.full(n, np.nan)
    for i in range(n):
        a, b = max(0, i - half_width), min(n, i + half_width + 1)
        if b - a >= 3:
            out[i] = np.polyfit(x[a:b], y[a:b], 1)[0]
    return out


def find_scaling_region(x: np.ndarray, y: np.ndarray, *, min_points: int = 8,
                        min_decades: float = 0.5,
                        max_slope_cv: float = 0.25) -> ScalingRegion:
    """Locate the straightest sufficiently wide window of a log-log curve.

    Every window of at least ``min_points`` is scored by the standard
    deviation of its local slopes divided by their mean; the flattest wins,
    with wider windows preferred among near-ties. A window narrower than
    ``min_decades`` or with slope variation above ``max_slope_cv`` still
    produces an estimate, and a warning saying it should not be trusted.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < min_points:
        raise ParameterError(
            f"only {x.size} usable points on the curve, need at least "
            f"{min_points}. The record is too short, or the range of radii too "
            "narrow."
        )

    slopes = local_slopes(x, y)
    span = float(x[-1] - x[0]) or 1.0
    best = None
    for start in range(0, x.size - min_points + 1):
        for stop in range(start + min_points, x.size + 1):
            seg = slopes[start:stop]
            seg = seg[np.isfinite(seg)]
            if seg.size < 3:
                continue
            mean = float(np.mean(seg))
            if abs(mean) < 1e-12:
                continue
            cv = float(np.std(seg) / abs(mean))
            width = float(x[stop - 1] - x[start])
            # Flatness decides; width breaks ties. The bonus is *relative* to
            # the whole curve, because x is log-radius for a correlation sum
            # and plain sample counts for a divergence curve: an absolute bonus
            # would dwarf the flatness term on the second and swallow the
            # saturated tail into the fit.
            score = cv - 0.05 * (width / span)
            if best is None or score < best[0]:
                best = (score, start, stop, cv, width)

    if best is None:
        raise ParameterError("no window of the curve could be scored")
    _, start, stop, cv, width = best

    xs, ys = x[start:stop], y[start:stop]
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    seg = slopes[start:stop]
    slope_std = float(np.nanstd(seg))

    warning = None
    decades = width / np.log(10.0)
    if decades < min_decades:
        warning = (f"the fitted region spans only {decades:.2f} decades, below "
                   f"the {min_decades} asked for: the estimate is not supported "
                   "by enough range to be a scaling law")
    elif cv > max_slope_cv:
        warning = (f"local slopes vary by {cv:.0%} within the fitted region, "
                   f"above the {max_slope_cv:.0%} tolerance: the curve is not "
                   "straight there")

    return ScalingRegion(
        start=start, stop=stop, slope=float(slope), intercept=float(intercept),
        r_squared=float(r2), slope_std=slope_std,
        x_lo=float(x[start]), x_hi=float(x[stop - 1]), n_points=int(stop - start),
        criterion=f"flattest window of >= {min_points} points "
                  f"(slope variation {cv:.1%})",
        warning=warning,
    )


def fixed_region(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> ScalingRegion:
    """Fit a window the caller chooses, for replicating published analyses."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if sel.sum() < 3:
        raise ParameterError(f"only {int(sel.sum())} points inside [{lo}, {hi}]")
    xs, ys = x[sel], y[sel]
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    idx = np.flatnonzero(sel)
    return ScalingRegion(
        start=int(idx[0]), stop=int(idx[-1] + 1), slope=float(slope),
        intercept=float(intercept),
        r_squared=1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan,
        slope_std=float(np.nanstd(local_slopes(xs, ys))),
        x_lo=float(xs[0]), x_hi=float(xs[-1]), n_points=int(sel.sum()),
        criterion=f"user-specified window [{lo:g}, {hi:g}]",
    )
