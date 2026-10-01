"""Threshold estimation, every mode.

Design note (DD-35) -- the default is never a fixed epsilon.

The project specification records the problem plainly: recurrence density falls
exponentially with dimension at fixed epsilon, so a threshold that gives a
sensible plot in 3 dimensions gives an empty one in 12. The default mode is
therefore ``target_rr``: fix the recurrence rate, let epsilon follow. What a
paper should report is the *achieved* rate, which is what
``Threshold.achieved_rr`` carries.

``method="bisect"`` refines the sampled quantile against fresh, independent
samples until the achieved rate is within tolerance of the target. The plain
quantile is exact for the sample it saw but biased for the population; the
refinement removes most of that bias at the cost of a few more samples.
"""
from __future__ import annotations

import numpy as np

from ..config import SeedLike, resolve_rng
from ..exceptions import ParameterError
from .core import Threshold
from .sampling import sample_cross_distances, sample_pair_distances

MODES = ("fixed", "target_rr", "percentile", "fan", "std_fraction",
         "maxdist_fraction", "per_block", "adaptive_dim")
SCOPES = ("global", "per_record", "per_window", "per_block")
BISECT_METHODS = ("quantile", "bisect")


def _coords(source):
    """Accept a StateSpace or a raw array; always use the weighted geometry."""
    if hasattr(source, "weighted_coords"):
        return np.asarray(source.weighted_coords, dtype=float), source
    return np.atleast_2d(np.asarray(source, dtype=float)), None


def _rate_at(dists: np.ndarray, eps: float) -> float:
    return float(np.mean(dists <= eps)) if dists.size else float("nan")


def threshold(source, *, mode: str = "target_rr", target_rr: float = 0.05,
              percentile: float | None = None, value: float | None = None,
              n_neighbors: int = 20, factor: float = 0.1, metric: str = "euclidean",
              theiler: int = 1, method: str = "quantile", tol: float = 1e-3,
              max_iter: int = 12, n_samples: int = 200_000,
              scope: str = "global", rng: SeedLike = None,
              block_targets=None) -> Threshold:
    """Fix the recurrence threshold epsilon.

    Parameters
    ----------
    mode
        ``target_rr`` (default) sets epsilon so the recurrence rate hits
        ``target_rr``. ``fixed`` takes ``value`` verbatim. ``percentile``
        takes a quantile of the distance distribution. ``fan`` gives every
        point its own epsilon, the distance to its ``n_neighbors``-th
        neighbour, which keeps the neighbour count constant instead of the
        radius. ``std_fraction`` and ``maxdist_fraction`` scale ``factor`` by
        the spread of the data. ``per_block`` sets one epsilon per coordinate
        block, so no relative weighting is needed at all. ``adaptive_dim``
        targets a rate and reports how far the geometry is from achieving it.
    method
        For ``target_rr``: ``quantile`` (one sample) or ``bisect`` (refine
        against fresh samples until within ``tol`` of the target).
    theiler
        Pairs with ``|i - j| < theiler`` are excluded from the estimate,
        matching the region the recurrence rate will be computed over [DD-32].
        ``0`` excludes nothing and ``1`` (default, as in ``recurrence_plot``)
        the line of identity [DD-115, DD-118]. The window is stored on the
        returned :class:`Threshold`, and a plot built from it inherits it.
    """
    th = _threshold(source, mode=mode, target_rr=target_rr, percentile=percentile,
                    value=value, n_neighbors=n_neighbors, factor=factor,
                    metric=metric, theiler=theiler, method=method, tol=tol,
                    max_iter=max_iter, n_samples=n_samples, scope=scope, rng=rng,
                    block_targets=block_targets)
    # Guard [DD-118]: when more than the target share of pairs coincide --
    # an exactly periodic or a quantised series -- the quantile lands on those
    # repeats and epsilon sits at rounding level. Which repeats then count as
    # recurrent is decided by floating point: on a sine with 2.5% exact
    # repeats and a target of 0.02, epsilon was 4.7e-14 and the diagonal
    # densities differed between numpy 1.24 and 2.4.
    if mode != "fixed" and not th.is_pointwise and not th.is_per_block:
        X, _ = _coords(source)
        extent = float(np.max(np.ptp(X, axis=0))) if X.size else 0.0
        eps = float(th.scalar)
        if extent > 0 and np.isfinite(eps) and eps <= 1e-9 * extent:
            import warnings

            from ..exceptions import GeometryWarning

            warnings.warn(
                f"threshold: epsilon = {eps:.2e} is at rounding level for an "
                f"attractor of size {extent:.3g}. More than the target share of "
                "pairs coincide exactly (a periodic or quantised series), so which "
                "of them count as recurrent is decided by floating point. Raise "
                "target_rr above the share of exact repeats, or add measurement "
                "noise [DD-118].", GeometryWarning, stacklevel=2)
    return th


def _threshold(source, *, mode: str = "target_rr", target_rr: float = 0.05,
              percentile: float | None = None, value: float | None = None,
              n_neighbors: int = 20, factor: float = 0.1, metric: str = "euclidean",
              theiler: int = 0, method: str = "quantile", tol: float = 1e-3,
              max_iter: int = 12, n_samples: int = 200_000,
              scope: str = "global", rng: SeedLike = None,
              block_targets=None) -> Threshold:
    """Fix the recurrence threshold epsilon.

    Parameters
    ----------
    mode
        ``target_rr`` (default) sets epsilon so the recurrence rate hits
        ``target_rr``. ``fixed`` takes ``value`` verbatim. ``percentile``
        takes a quantile of the distance distribution. ``fan`` gives every
        point its own epsilon, the distance to its ``n_neighbors``-th
        neighbour, which keeps the neighbour count constant instead of the
        radius. ``std_fraction`` and ``maxdist_fraction`` scale ``factor`` by
        the spread of the data. ``per_block`` sets one epsilon per coordinate
        block, so no relative weighting is needed at all. ``adaptive_dim``
        targets a rate and reports how far the geometry is from achieving it.
    method
        For ``target_rr``: ``quantile`` (one sample) or ``bisect`` (refine
        against fresh samples until within ``tol`` of the target).
    theiler
        Pairs with ``|i - j| < theiler`` are excluded from the estimate,
        matching the region the recurrence rate will be computed over [DD-32].
        ``0`` excludes nothing and ``1`` the line of identity [DD-115].
    """
    if mode not in MODES:
        raise ParameterError(f"unknown threshold mode {mode!r}; use one of {MODES}")
    if scope not in SCOPES:
        raise ParameterError(f"unknown scope {scope!r}; use one of {SCOPES}")
    rng = resolve_rng(rng)
    X, ss = _coords(source)
    n = X.shape[0]
    common = dict(mode=mode, metric=metric, scope=scope, theiler=theiler)

    # ------------------------------------------------------------- fixed
    if mode == "fixed":
        if value is None or value <= 0:
            raise ParameterError("mode='fixed' needs value > 0")
        d = sample_pair_distances(X, n_samples=min(n_samples, 50_000),
                                  theiler=theiler, rng=rng, metric=metric)
        return Threshold(value=float(value), target=None,
                         achieved_rr=_rate_at(d, float(value)), n_samples=d.size,
                         diagnostics={"criterion": "user-specified epsilon"}, **common)

    # ------------------------------------------------------ scale-derived
    if mode in ("std_fraction", "maxdist_fraction"):
        if mode == "std_fraction":
            eps = float(factor * np.sqrt(np.sum(np.var(X, axis=0))))
            crit = f"epsilon = {factor} x attractor RMS radius"
        else:
            d = sample_pair_distances(X, n_samples=min(n_samples, 100_000),
                                      theiler=theiler, rng=rng, metric=metric)
            eps = float(factor * (d.max() if d.size else 1.0))
            crit = f"epsilon = {factor} x maximum sampled distance"
        d = sample_pair_distances(X, n_samples=min(n_samples, 50_000),
                                  theiler=theiler, rng=rng, metric=metric)
        return Threshold(value=eps, target=None, achieved_rr=_rate_at(d, eps),
                         n_samples=d.size, diagnostics={"criterion": crit}, **common)

    # --------------------------------------------------------------- FAN
    if mode == "fan":
        from scipy.spatial import cKDTree

        if n_neighbors < 1:
            raise ParameterError("n_neighbors must be >= 1")
        if metric != "euclidean":
            raise ParameterError("mode='fan' currently requires metric='euclidean'")
        # A point is never its own neighbour, whatever the window [DD-115]:
        # neighbours need |i - j| >= max(theiler, 1).
        gap = max(int(theiler), 1)
        k = int(min(n_neighbors + 2 * gap, n))
        dist, idx = cKDTree(X).query(X, k=k)
        eps = np.empty(n)
        for i in range(n):
            valid = dist[i][np.abs(idx[i] - i) >= gap]
            eps[i] = valid[min(n_neighbors, valid.size) - 1] if valid.size else 0.0
        achieved = float(n_neighbors / max(n - (2 * gap - 1), 1))
        return Threshold(
            value=eps, target=achieved, achieved_rr=achieved, n_samples=n,
            diagnostics={"criterion": f"distance to neighbour #{n_neighbors}",
                         "n_neighbors": n_neighbors,
                         "epsilon_median": float(np.median(eps)),
                         "epsilon_iqr": float(np.subtract(*np.percentile(eps, [75, 25]))),
                         "warning": "FAN thresholds are per point, so the recurrence "
                                    "matrix is not symmetric"},
            **common)

    # -------------------------------------------------------- per block
    if mode == "per_block":
        if ss is None or not hasattr(ss, "groups"):
            raise ParameterError("mode='per_block' needs a StateSpace, not a raw array")
        targets = (block_targets if block_targets is not None
                   else [target_rr] * len(ss.groups))
        if len(targets) != len(ss.groups):
            raise ParameterError(
                f"block_targets has {len(targets)} entries for {len(ss.groups)} blocks")
        eps_blocks, labels = [], []
        for g, t in zip(ss.groups, targets, strict=False):
            Xb = np.asarray(ss.coords)[:, g.slice]
            d = sample_pair_distances(Xb, n_samples=n_samples, theiler=theiler,
                                      rng=rng, metric=metric)
            eps_blocks.append(float(np.quantile(d, np.clip(t, 0, 1))) if d.size else 1.0)
            labels.append(g.label)
        return Threshold(
            value=float(max(eps_blocks)), per_block=tuple(eps_blocks),
            block_labels=tuple(labels), target=float(np.mean(targets)),
            achieved_rr=None, n_samples=n_samples,
            diagnostics={"criterion": "one epsilon per block, combined with Chebyshev; "
                                      "no relative weighting needed"},
            **common)

    # ------------------------------------------- percentile / target rate
    q = None
    if mode == "percentile":
        if percentile is None:
            raise ParameterError("mode='percentile' needs percentile in [0, 1]")
        if not 0.0 <= percentile <= 1.0:
            raise ParameterError(
                f"percentile must be a fraction in [0, 1], got {percentile}. "
                "Pass 0.05 for the 5th percentile, not 5."
            )
        q = float(percentile)
    else:
        if not 0.0 < target_rr < 1.0:
            raise ParameterError(f"target_rr must be in (0, 1), got {target_rr}")
        q = float(target_rr)

    d = sample_pair_distances(X, n_samples=n_samples, theiler=theiler, rng=rng,
                              metric=metric)
    if d.size == 0:
        raise ParameterError("could not sample any admissible pair of points")
    eps = float(np.quantile(d, q))
    achieved = _rate_at(d, eps)
    diag = {"criterion": f"quantile {q:g} of the sampled distance distribution",
            "iterations": 0}

    if mode in ("target_rr", "adaptive_dim") and method == "bisect":
        if method not in BISECT_METHODS:
            raise ParameterError(f"method must be one of {BISECT_METHODS}")
        lo, hi = 0.0, float(d.max() * 1.5)
        it = 0
        for it in range(1, max_iter + 1):  # noqa: B007 -- read after the loop
            check = sample_pair_distances(X, n_samples=max(20_000, n_samples // 4),
                                          theiler=theiler, rng=rng, metric=metric)
            achieved = _rate_at(check, eps)
            if abs(achieved - q) <= tol:
                break
            if achieved < q:
                lo = eps
            else:
                hi = eps
            eps = 0.5 * (lo + hi)
        diag = {"criterion": f"bisection to RR = {q:g} within tol {tol:g}",
                "iterations": it,
                "converged": bool(abs(achieved - q) <= tol)}
        if not diag["converged"]:
            diag["warning"] = (f"did not reach the target rate in {max_iter} iterations; "
                               f"achieved {achieved:.4%} against {q:.4%}")

    if mode == "adaptive_dim":
        dim = X.shape[1]
        diag["dimension"] = dim
        diag["epsilon_over_rms_radius"] = float(
            eps / (np.sqrt(np.sum(np.var(X, axis=0))) or 1.0))
        if diag["epsilon_over_rms_radius"] > 0.5:
            diag["warning"] = (
                f"epsilon is {diag['epsilon_over_rms_radius']:.2f} of the attractor "
                f"radius in {dim} dimensions: at this density the recurrence plot "
                "reflects the size of the cloud more than its structure. Reduce the "
                "target rate or the dimension.")

    return Threshold(value=eps, target=q, achieved_rr=achieved, n_samples=d.size,
                     diagnostics=diag, **common)


def cross_threshold(source1, source2, *, mode: str = "target_rr",
                    target_rr: float = 0.05, percentile: float | None = None,
                    value: float | None = None, metric: str = "euclidean",
                    n_samples: int = 200_000, rng: SeedLike = None) -> Threshold:
    """Threshold for a cross recurrence plot between two trajectories."""
    rng = resolve_rng(rng)
    X1, _ = _coords(source1)
    X2, _ = _coords(source2)
    common = dict(mode=mode, metric=metric, scope="global", theiler=0)

    if mode == "fixed":
        if value is None or value <= 0:
            raise ParameterError("mode='fixed' needs value > 0")
        d = sample_cross_distances(X1, X2, n_samples=min(n_samples, 50_000),
                                   rng=rng, metric=metric)
        return Threshold(value=float(value), target=None,
                         achieved_rr=_rate_at(d, float(value)), n_samples=d.size,
                         diagnostics={"criterion": "user-specified epsilon"}, **common)

    q = float(percentile if mode == "percentile" and percentile is not None else target_rr)
    if not 0.0 < q < 1.0:
        raise ParameterError(f"target rate must be in (0, 1), got {q}")
    d = sample_cross_distances(X1, X2, n_samples=n_samples, rng=rng, metric=metric)
    if d.size == 0:
        raise ParameterError("could not sample any cross pair")
    eps = float(np.quantile(d, q))
    return Threshold(value=eps, target=q, achieved_rr=_rate_at(d, eps),
                     n_samples=d.size,
                     diagnostics={"criterion": f"quantile {q:g} of cross distances",
                                  "note": "no Theiler exclusion: the two trajectories "
                                          "are different systems"},
                     **common)
