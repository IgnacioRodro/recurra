"""Dynamical invariants: correlation dimension, Lyapunov exponent, K2.

Design note (DD-62) -- rates carry their units, always.

The Lyapunov exponent and K2 are rates: nats per unit time. Whether that unit
is a sample or a second depends on the sampling rate, and the old draft
returned a number with no unit attached while silently decimating the data
inside the estimator -- so the exponent it returned was in nats per decimated
sample, and two windows of different length were not comparable.

Every rate here is returned as an :class:`Invariant` carrying ``units``, the
sampling rate it was computed at, and the scaling region that was fitted.
Nothing is decimated without saying so.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..config import SeedLike, resolve_rng, spawn_rngs
from ..exceptions import InferenceWarning, ParameterError
from .scaling import ScalingRegion, find_scaling_region, fixed_region

UNITS = ("per_sample", "per_second", "dimensionless")


@dataclass(frozen=True)
class Invariant:
    """An estimated invariant, with the evidence behind it."""

    name: str
    value: float
    units: str = "dimensionless"
    fs: float | None = None
    method: str = ""
    region: ScalingRegion | None = None
    curve: pd.DataFrame | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def warning(self) -> str | None:
        w = self.region.warning if self.region else None
        return self.diagnostics.get("warning", w)

    def to_frame(self) -> pd.DataFrame:
        row = {"invariant": self.name, "value": self.value, "units": self.units,
               "fs_hz": self.fs, "method": self.method}
        if self.region:
            row.update(self.region.to_row())
        row.update({k: v for k, v in self.diagnostics.items()
                    if isinstance(v, (int, float, str, bool))})
        return pd.DataFrame([row])

    def summary(self) -> str:
        lines = [f"{self.name} = {self.value:.5g} {self.units}"
                 f"   ({self.method})"]
        if self.region:
            lines.append(f"  fitted over {self.region.width_decades:.2f} decades, "
                         f"{self.region.n_points} points, R2 = "
                         f"{self.region.r_squared:.4f}")
            lines.append(f"  criterion : {self.region.criterion}")
        if self.warning:
            lines.append(f"  WARNING   : {self.warning}")
        return "\n".join(lines)

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Invariant {self.name}={self.value:.5g} {self.units}>"


def resolve_theiler(theiler: int | str, source) -> int:
    """How many samples apart two points must be to count as neighbours.

    Design note (DD-96) -- a delay embedding needs a large Theiler window.

    Two points of a delay embedding whose indices differ by less than the
    embedding window ``(m-1)*tau`` **share coordinates**: they are the same
    numbers in a different order. Counting them as neighbours makes the cloud
    look one-dimensional at small radii, which is exactly where the correlation
    dimension is read.

    Measured on a Lorenz delay embedding with tau = 16 and m = 3: a Theiler
    window of 1, 16, 32 or 64 all give D2 near 1.60, and 200 gives 1.97 against
    a reference of 2.05. The embedding span is 32, so the span is not enough
    either -- what is needed is the orbital scale.

    ``"auto"`` therefore takes the larger of the embedding window and a tenth
    of the record, capped so that enough pairs survive. It is a heuristic and
    it is stated; pass a number to override it.
    """
    if isinstance(theiler, (int, np.integer)):
        return int(theiler)
    if theiler != "auto":
        raise ParameterError(f"theiler must be an int or 'auto', got {theiler!r}")

    span = 1
    for group in getattr(source, "groups", ()) or ():
        params = getattr(group, "params", None) or {}
        m, tau = params.get("m"), params.get("tau")
        if m and tau:
            span = max(span, int((int(m) - 1) * int(tau)))
    n = getattr(source, "n_points", None)
    if n is None:
        n = np.atleast_2d(np.asarray(source)).shape[0]
    return int(min(max(span, n // 100), n // 20))


def _as_points(source) -> tuple[np.ndarray, float]:
    """Accept a StateSpace or an array; return coordinates and sampling rate."""
    if hasattr(source, "weighted_coords"):
        return np.asarray(source.weighted_coords, dtype=float), float(source.fs)
    arr = np.atleast_2d(np.asarray(source, dtype=float))
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return arr, 1.0


# ============================================================ dimension
def correlation_sum(source, *, n_radii: int = 40, theiler: int | str = "auto",
                    max_points: int = 4000, rng: SeedLike = None,
                    lo_quantile: float = 0.0002, hi_quantile: float = 0.5,
                    radii: np.ndarray | None = None) -> pd.DataFrame:
    """Grassberger-Procaccia correlation sum C(r), Theiler-excluded.

    Returns a tidy frame with the radius, the count, C(r) and its logarithms,
    so the fit can be inspected or redone.
    """
    X, _ = _as_points(source)
    theiler = resolve_theiler(theiler, source)
    rng = resolve_rng(rng)
    n = X.shape[0]
    idx = (np.sort(rng.choice(n, size=max_points, replace=False))
           if n > max_points else np.arange(n))
    Y = X[idx]
    m = Y.shape[0]

    i, j = np.triu_indices(m, k=1)
    keep = np.abs(idx[i] - idx[j]) >= max(theiler, 1)      # [DD-115]
    i, j = i[keep], j[keep]
    if i.size < 100:
        raise ParameterError(
            f"only {i.size} admissible pairs after the Theiler exclusion; "
            "use a longer record or a smaller Theiler window")
    d = np.linalg.norm(Y[i] - Y[j], axis=1)
    d = d[d > 0]

    if radii is None:
        # Design note (DD-96). The lower end decides whether there is a scaling
        # region at all. At the 0.1st percentile the straight part of a Lorenz
        # correlation sum spans 0.39 decades and is rejected for being too thin;
        # at the 0.02nd it spans 0.81 and D2 comes out 2.009 against a reference
        # of 2.05. The upper end is pulled in because the largest radii are
        # saturated and only distract the region finder.
        lo, hi = np.quantile(d, [lo_quantile, hi_quantile])
        radii = np.exp(np.linspace(np.log(lo), np.log(hi), n_radii))
    counts = np.searchsorted(np.sort(d), radii, side="right")
    C = counts / d.size

    with np.errstate(divide="ignore"):
        return pd.DataFrame({"radius": radii, "count": counts, "C": C,
                             "log_r": np.log(radii),
                             "log_C": np.where(C > 0, np.log(C), np.nan)})


def correlation_dimension(source, *, n_radii: int = 40, theiler: int | str = "auto",
                          max_points: int = 4000, region=None,
                          min_points: int = 8, rng: SeedLike = None,
                          **kwargs) -> Invariant:
    """D2 by Grassberger-Procaccia, over an automatically located region [DD-61].

    Pass ``region=(lo, hi)`` in units of log radius to fit a window of your
    own instead.

    Note this is the **correlation** dimension, which is not the Kaplan-Yorke
    dimension often quoted alongside it and is generally a little smaller.
    Reference values for the correlation dimension: Lorenz 2.05, Rossler 1.81
    (Grassberger and Procaccia 1983). Quoting 2.01 for Rossler, as many
    sources do, is the Kaplan-Yorke value and this estimator will not
    reproduce it.
    """
    theiler = resolve_theiler(theiler, source)
    curve = correlation_sum(source, n_radii=n_radii, theiler=theiler,
                            max_points=max_points, rng=rng)
    x, y = curve["log_r"].to_numpy(), curve["log_C"].to_numpy()
    if region is None:
        reg = find_scaling_region(x, y, min_points=min_points, **kwargs)
    else:
        reg = fixed_region(x, y, *region)

    # Design note (DD-100). A correlation sum that is straight over its whole
    # range has no scaling *region*: there is nothing to locate, because
    # nothing bends. That happens when the geometry is imposed rather than
    # measured -- a phase circle crossed with an amplitude is a 2-torus by
    # construction, and D2 comes back at 1.98 with R2 = 1.0000 on every
    # recording, identical across 98 real ones. The number is a property of
    # how the space was built, not of the data in it.
    warning = reg.warning
    if warning is None and region is None:
        span = reg.n_points / max(int(np.isfinite(y).sum()), 1)
        if span > 0.9 and reg.r_squared > 0.9999:
            warning = (
                f"the fit covers {span:.0%} of the correlation sum with "
                f"R2 = {reg.r_squared:.6f}: the curve does not bend, so there "
                "is no scaling region to find. A dimension this well determined "
                "is usually the dimension of the construction -- a phase circle "
                "crossed with an amplitude gives 2 whatever the signal does -- "
                "rather than a measured property. Check it varies across "
                "recordings before using it."
            )

    _, fs = _as_points(source)
    return Invariant(
        name="D2", value=reg.slope, units="dimensionless", fs=fs,
        method="Grassberger-Procaccia", region=reg, curve=curve,
        diagnostics={"theiler": theiler, "n_radii": n_radii,
                     "max_points": max_points,
                     "region_share": float(reg.n_points
                                           / max(int(np.isfinite(y).sum()), 1)),
                     **({"warning": warning} if warning else {})},
    )


# ============================================================= Lyapunov
def divergence_curve(source, *, max_steps: int = 100,
                     theiler: int | str | None = None,
                     max_points: int = 3000, method: str = "rosenstein",
                     radius: float | None = None, min_neighbours: int = 5,
                     rng: SeedLike = None) -> pd.DataFrame:
    """Mean logarithmic divergence of initially nearby trajectories.

    ``rosenstein`` follows each point's single nearest neighbour;
    ``kantz`` averages over every neighbour inside a ball of ``radius``,
    which is steadier on noisy data at the cost of choosing a radius.
    """
    X, fs = _as_points(source)
    rng = resolve_rng(rng)
    n = X.shape[0]
    theiler = (max(1, n // 100) if theiler is None
               else resolve_theiler(theiler, source))
    horizon = int(min(max_steps, n // 4))
    if horizon < 5:
        raise ParameterError(f"record too short for a divergence curve ({n} points)")

    idx = (np.sort(rng.choice(n - horizon, size=max_points, replace=False))
           if (n - horizon) > max_points else np.arange(n - horizon))
    tree = cKDTree(X)

    sums = np.zeros(horizon + 1)
    counts = np.zeros(horizon + 1, dtype=np.int64)

    if method == "rosenstein":
        k = min(2 + 2 * theiler, n)
        dist, nb = tree.query(X[idx], k=k)
        for row, i in enumerate(idx):
            valid = nb[row][(np.abs(nb[row] - i) >= max(theiler, 1))
                            & (nb[row] < n - horizon)]
            if valid.size == 0:
                continue
            j = int(valid[0])
            seg = np.linalg.norm(X[i:i + horizon + 1] - X[j:j + horizon + 1], axis=1)
            good = seg > 0
            sums[:horizon + 1][good] += np.log(seg[good])
            counts[:horizon + 1][good] += 1
    elif method == "kantz":
        if radius is None:
            sample = np.linalg.norm(X[rng.integers(0, n, 5000)]
                                    - X[rng.integers(0, n, 5000)], axis=1)
            radius = float(np.quantile(sample[sample > 0], 0.05))
        for i in idx:
            nb = np.asarray(tree.query_ball_point(X[i], radius), dtype=int)
            nb = nb[(np.abs(nb - i) >= max(theiler, 1)) & (nb < n - horizon)]
            if nb.size < min_neighbours:
                continue
            seg = np.linalg.norm(X[i:i + horizon + 1][None, :, :]
                                 - np.stack([X[j:j + horizon + 1] for j in nb]),
                                 axis=2)
            mean_sep = seg.mean(axis=0)
            good = mean_sep > 0
            sums[good] += np.log(mean_sep[good])
            counts[good] += 1
    else:
        raise ParameterError(f"method must be 'rosenstein' or 'kantz', got {method!r}")

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_log = np.where(counts > 0, sums / counts, np.nan)
    steps = np.arange(horizon + 1)
    return pd.DataFrame({"step": steps, "time_s": steps / fs,
                         "mean_log_divergence": mean_log, "n_pairs": counts})


def lyapunov_max(source, *, method: str = "rosenstein", max_steps: int = 100,
                 theiler: int | None = None, max_points: int = 3000,
                 units: str = "per_second", region=None, min_points: int = 15,
                 stability_tolerance: float = 0.25, n_repeats: int = 3,
                 sampling_tolerance: float = 0.20, rng: SeedLike = None,
                 **kwargs) -> Invariant:
    """Largest Lyapunov exponent, from the slope of the divergence curve.

    The slope is in nats per sample; ``units="per_second"`` multiplies by the
    sampling rate. A state space carries its own rate, so the conversion is
    automatic and correct even after decimation [DD-47, DD-62].

    ``min_points`` defaults to 15 rather than the 8 used for a correlation
    sum, because a short window can sit entirely inside the initial transient.

    **The estimate is an average over samplings, not a single draw** [DD-101].
    The divergence curve is built from a few thousand reference points chosen
    at random out of the record, and which ones are chosen matters more than it
    should: measured on Lorenz, twelve streams over identical data gave 0.501
    to 1.366 against a reference of 0.906, a coefficient of variation of 0.25,
    and eleven of the twelve came back unflagged. ``n_repeats`` draws several
    independent samplings, returns their mean, and reports how far they
    disagreed as ``sampling_cv``. A spread above ``sampling_tolerance`` is
    flagged.

    **The fit window is the estimate.** Measured on Lorenz, the exponent comes
    out 1.06, 0.73, 1.09 or 0.30 per second depending only on ``max_steps``,
    against a reference of 0.906. No automatic criterion reliably picks the
    linear region, so this reports **how much the answer depends on the
    choice**: the exponent is refitted on truncations of the same curve and
    flagged when it moves by more than ``stability_tolerance`` [DD-94].

    An unflagged value is one the data determines. A flagged one is a number
    that depends on a parameter, and should be treated as a descriptor or
    replaced by a window chosen from ``divergence_curve()`` by eye.
    """
    if units not in ("per_sample", "per_second"):
        raise ParameterError("units must be 'per_sample' or 'per_second'")
    # Several independent samplings of the reference points, not one [DD-101].
    streams = spawn_rngs(rng, max(1, int(n_repeats)))
    curve_kw = {k: v for k, v in kwargs.items()
                if k in ("radius", "min_neighbours")}
    curves = [divergence_curve(source, max_steps=max_steps, theiler=theiler,
                               max_points=max_points, method=method,
                               rng=stream, **curve_kw)
              for stream in streams]
    curve = curves[0]
    x = curve["step"].to_numpy(dtype=float)
    y = curve["mean_log_divergence"].to_numpy(dtype=float)
    fit = {k: v for k, v in kwargs.items() if k not in ("radius", "min_neighbours")}
    _, fs_hint = _as_points(source)

    # The most specific diagnosis available goes first: on noiseless synthetic
    # data the nearest neighbours can sit at machine precision -- measured
    # 1.3e-15 on a sampled circle -- and the curve then tracks floating-point
    # error through a logarithm rather than any dynamics [DD-63]. Left until
    # after the generic checks, it was masked by "the curve is not straight",
    # which is true but says nothing about why.
    degenerate = None
    X_, _ = _as_points(source)
    extent = float(np.max(X_.max(axis=0) - X_.min(axis=0))) or 1.0
    finite_y = np.isfinite(y)
    if finite_y.any():
        initial = float(np.exp(np.nanmin(y[finite_y])))
        if initial / extent < 1e-8:
            degenerate = (
                f"initial neighbour separation is {initial:.2e} against an "
                f"attractor extent of {extent:.3g}, a ratio of "
                f"{initial / extent:.1e}. The neighbours are at numerical zero, "
                "so this curve measures floating-point noise, not divergence. "
                "Add measurement noise or use a longer Theiler window."
            )

    # A divergence curve has three parts: a steep transient while the initial
    # neighbours align with the most unstable direction, the linear region
    # whose slope is the exponent, and a saturated plateau once the separation
    # reaches the size of the attractor.
    #
    # The flattest window is not enough to find the middle one. Measured on
    # Lorenz, whose exponent is 0.906 per second, the same estimator over the
    # same data returns 1.06, 0.73, 1.09 and 0.30 as max_steps goes 150, 300,
    # 600, 1200 -- and the slope variation inside the fitted window is 0.01 in
    # every case, because the plateau is flat too. An earlier guard checked
    # whether the fit began in the first quarter of the curve, which fired on
    # a correct Lorenz fit and is a statement about max_steps rather than
    # about the data.
    #
    # So the answer is not a better automatic criterion. It is to measure how
    # much the answer depends on the choice and to say so [DD-94].
    def _fit(curve_frame):
        xs = curve_frame["step"].to_numpy(dtype=float)
        ys = curve_frame["mean_log_divergence"].to_numpy(dtype=float)
        return (find_scaling_region(xs, ys, min_points=min_points,
                                    min_decades=0.0, **fit)
                if region is None else fixed_region(xs, ys, *region))

    regions = []
    for frame in curves:
        try:
            regions.append(_fit(frame))
        except ParameterError:
            pass
    if not regions:
        raise ParameterError("no sampling produced a fittable divergence curve")
    reg = regions[0]
    slopes = np.asarray([r.slope for r in regions], dtype=float)
    mean_slope = float(np.mean(slopes))
    sampling_cv = (float(np.std(slopes, ddof=1) / abs(mean_slope))
                   if slopes.size > 1 and abs(mean_slope) > 1e-12 else 0.0)

    warning = degenerate or reg.warning
    if warning is None and slopes.size > 1 and sampling_cv > sampling_tolerance:
        warning = (
            f"the exponent moves by {sampling_cv:.0%} between {slopes.size} "
            f"samplings of the same record: "
            f"{', '.join(f'{v * (fs_hint or 1):.3f}' for v in slopes)}. "
            "Which reference points are drawn is deciding the answer. Raise "
            "max_points or n_repeats, or treat this as a descriptor rather "
            "than an exponent [DD-101]."
        )
    stability: dict[str, Any] = {}
    if degenerate is None and region is None and x.size >= 4 * min_points:
        # Refit on truncations of the same curve. Free -- the curve is already
        # computed -- and it asks the question that matters: would a different
        # max_steps have given a different exponent?
        # A separate list from the per-sampling slopes above: this varies the
        # fit window, that varied which points were sampled. Two different
        # questions, and reusing the name silently answered neither.
        truncated = [reg.slope]
        for share in (0.5, 0.75):
            cut = int(x.size * share)
            if cut >= 2 * min_points:
                try:
                    truncated.append(find_scaling_region(
                        x[:cut], y[:cut], min_points=min_points,
                        min_decades=0.0, **fit).slope)
                except ParameterError:
                    pass
        values = np.asarray(truncated, dtype=float)
        spread = float(np.std(values) / abs(np.mean(values))) if values.size > 1 \
            and abs(np.mean(values)) > 1e-12 else np.nan
        stability = {"slopes_over_truncations": [float(v) for v in values],
                     "stability_cv": spread}
        if np.isfinite(spread) and spread > stability_tolerance:
            warning = (
                f"the exponent moves by {spread:.0%} when the divergence curve "
                f"is truncated: {', '.join(f'{v * (fs_hint or 1):.3f}' for v in values)}. "
                "The fit window is not determined by the data. Look at "
                "divergence_curve() and choose region= by eye, or treat this as "
                "a descriptor rather than an exponent."
            )

    # Saturation: a fit sitting in the plateau returns a slope near zero and
    # looks perfectly straight while measuring nothing.
    finite = np.isfinite(y)
    if warning is None and finite.sum() > 5:
        lo, hi = float(np.nanmin(y[finite])), float(np.nanmax(y[finite]))
        if hi > lo:
            level = (np.nanmean(y[reg.start:reg.stop]) - lo) / (hi - lo)
            if level > 0.85:
                warning = (
                    f"the fitted window sits at {level:.0%} of the curve's "
                    "range, in the saturated tail where the separation has "
                    "reached the size of the attractor. Lower max_steps."
                )
    _, fs = _as_points(source)
    value = mean_slope * (fs if units == "per_second" else 1.0)
    return Invariant(
        name="lambda_1", value=float(value), units=units, fs=fs,
        method=f"{method} divergence", region=reg, curve=curve,
        diagnostics={"max_steps": max_steps, "theiler": theiler,
                     "slope_per_sample": mean_slope,
                     "fit_starts_at_step": int(reg.start),
                     "n_repeats": int(slopes.size),
                     "sampling_cv": sampling_cv,
                     "sampling_sd": float(np.std(slopes, ddof=1) * (fs_hint or 1))
                     if slopes.size > 1 else 0.0,
                     **stability,
                     **({"warning": warning} if warning else {})},
    )


# =================================================================== K2
def k2_entropy(source, *, l_min: int = 2, units: str = "per_second",
               fs: float | None = None, **kwargs) -> Invariant:
    """K2 from the diagonal line-length distribution.

    The cumulative distribution of diagonal lines decays exponentially at a
    rate set by the second-order Renyi entropy, so K2 is minus the slope of
    log P(l >= l) against l, converted to the requested unit. Accepts a
    :class:`LineHistogram`, a recurrence matrix, or anything a recurrence plot
    can be built from.
    """
    from ..rqa.histogram import LineHistogram, line_histogram

    if isinstance(source, LineHistogram):
        hist, rate = source, fs
    else:
        from ..recurrence.core import RecurrenceMatrix

        if isinstance(source, RecurrenceMatrix):
            rm = source
        else:
            from ..recurrence.builders import recurrence_plot

            rm = recurrence_plot(source, **kwargs)
        hist, rate = line_histogram(rm), (fs if fs is not None else rm.fs)
    rate = rate or 1.0

    counts = hist.diagonal.astype(float)
    lengths = np.arange(counts.size)
    cumulative = np.cumsum(counts[::-1])[::-1]      # number of lines >= l
    sel = (lengths >= l_min) & (cumulative > 0)
    if sel.sum() < 5:
        raise ParameterError(
            "too few distinct diagonal line lengths to estimate K2; the plot is "
            "too sparse or the record too short")
    x = lengths[sel].astype(float)
    y = np.log(cumulative[sel] / cumulative[sel][0])
    reg = find_scaling_region(x, y, min_points=min(8, sel.sum()), min_decades=0.0)
    warn = reg.warning
    if int(sel.sum()) < 15:
        warn = (f"only {int(sel.sum())} distinct diagonal lengths carry the fit. "
                "K2 from a decay rate needs a well-populated distribution; on a "
                "near-periodic or very sparse plot this number is unreliable.")

    per_sample = -reg.slope
    value = per_sample * (rate if units == "per_second" else 1.0)
    return Invariant(
        name="K2", value=float(value), units=units, fs=rate,
        method="diagonal line-length decay", region=reg,
        curve=pd.DataFrame({"length": x, "log_cumulative": y}),
        diagnostics={"l_min": l_min, "rate_per_sample": per_sample,
                     "n_distinct_lengths": int(sel.sum()),
                     **({"warning": warn} if warn else {})},
    )


def invariants_from_series(x, *, fs: float = 1.0, m: int | None = None,
                           tau: int | None = None, theiler: int | str = "auto",
                           measures: Sequence[str] = ("D2", "lambda_1"),
                           max_steps: int | None = None, prefix: str = "",
                           rng: SeedLike = None, **kwargs) -> dict[str, float]:
    """D2 and the Lyapunov exponent of a scalar series, via a delay embedding.

    Design note (DD-97) -- embed the oscillation, not its description.

    A phase-amplitude space is a poor object for these two [DD-94], and so are
    the series a phase or amplitude block contributes: an instantaneous
    frequency and a smoothed envelope are stochastic, and neither has an
    exponential divergence regime to measure. Measured on a synthetic
    recording, every combination of delay and dimension left both quantities
    rejected by their own diagnostics.

    What the method was built for is the **oscillation**: a band-limited signal
    ``A(t) cos(phi(t))``, embedded with a delay the data chooses. This function
    exists so that the invariants can be taken from that object while the
    recurrence analysis stays on the coupling space. The embedding is built,
    used, and discarded; nothing else in the pipeline sees it.

    ``m`` and ``tau`` default to estimates from the series itself, and the
    values used are returned alongside the invariants so a result can be
    audited.
    """
    from ..io.ingest import ingest
    from ..statespace.builders import takens
    from ..statespace.embedding import estimate_m, estimate_tau

    series = np.asarray(x, dtype=float).ravel()
    series = series[np.isfinite(series)]
    if series.size < 100:
        raise ParameterError(f"series too short to embed: {series.size} samples")

    tau_used = int(tau) if tau else int(estimate_tau(series).tau)
    m_used = int(m) if m else int(estimate_m(series, tau_used).m)
    space = takens(ingest({"series": series}, fs=fs), "series", m=m_used,
                   tau=tau_used)

    out: dict[str, float] = {f"{prefix}embed_tau": float(tau_used),
                             f"{prefix}embed_m": float(m_used)}
    for name in measures:
        if name not in ("D2", "lambda_1"):
            raise ParameterError(
                f"unknown measure {name!r}; this computes 'D2' and 'lambda_1'")
        try:
            if name == "D2":
                result = correlation_dimension(space, theiler=theiler, rng=rng,
                                               **kwargs)
            else:
                steps = max_steps or max(300, 20 * (m_used - 1) * tau_used)
                result = lyapunov_max(space, theiler=theiler, max_steps=steps,
                                      rng=rng, **kwargs)
            out[f"{prefix}{name}"] = float(result.value)
            out[f"{prefix}{name}_ok"] = 0.0 if result.warning else 1.0
        except Exception as exc:
            warnings.warn(f"dynamics: {name} from the series failed: {exc}",
                          InferenceWarning, stacklevel=2)
            out[f"{prefix}{name}"] = float("nan")
            out[f"{prefix}{name}_ok"] = 0.0
    return out
