"""Public constructors: RP, CRP and JRP.

Design note (DD-20, now implemented) -- the joint plot uses independent
thresholds.

Romano et al. (2004) define the joint recurrence plot as the elementwise
product of recurrence plots computed in *independent* phase spaces, each with
its own threshold fixed to its own recurrence rate. The old draft forced a
single shared epsilon and required both subsystems to have equal
dimensionality. Both were wrong: the shared epsilon makes the joint rate
depend on the arbitrary relative scaling of two different systems, and the
equal-dimension requirement belongs to the *cross* recurrence plot, where the
two trajectories must live in one common space to be compared at all.

The distinction matters here more than usual, because Hypothesis 2 of the
project rests on the joint plot working between attractors of different
dimension.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from ..config import SeedLike, get_config
from ..exceptions import CaveatWarning, ParameterError, ParameterWarning
from ..provenance import Step
from ..threshold.core import Threshold
from ..threshold.estimate import cross_threshold as _cross_threshold
from ..threshold.estimate import threshold as _threshold
from .core import RecurrenceMatrix
from .distance import METRICS
from .memory import plan


def _as_coords(source, name: str):
    if hasattr(source, "weighted_coords"):
        return (np.asarray(source.weighted_coords, dtype=float), source)
    arr = np.atleast_2d(np.asarray(source, dtype=float))
    if arr.shape[0] < arr.shape[1]:
        raise ParameterError(
            f"{name}: expected (n_points, n_dimensions); got {arr.shape}, which has "
            "more dimensions than points. Transpose it if that is wrong."
        )
    return arr, None


#: Theiler window used when neither the call nor a Threshold object gives one:
#: the line of identity only [DD-115].
DEFAULT_THEILER = 1


def _reconcile(th, theiler, metric: str, where: str) -> int:
    """The Theiler window to use, made consistent with a precomputed threshold.

    A :class:`Threshold` remembers the Theiler window and the metric it was
    estimated with, and its recurrence rate holds only for the same region and
    the same distance. ``theiler=None`` therefore inherits the threshold's
    window. An explicit, different window or metric is honoured but warned
    about [DD-118]: measured before 0.25, a threshold estimated at theiler=1
    and used at theiler=50 gave a rate of 0.024 for a target of 0.05, silently.
    """
    own = getattr(th, "theiler", None) if isinstance(th, Threshold) else None
    if theiler is None:
        theiler = DEFAULT_THEILER if own is None else int(own)
    elif own is not None and int(own) != int(theiler):
        warnings.warn(
            f"{where}: the threshold was estimated with theiler={own} but the plot "
            f"uses theiler={theiler}, so its recurrence rate will not match the "
            "threshold's target. Leave theiler unset to inherit it [DD-118].",
            ParameterWarning, stacklevel=3)
    if isinstance(th, Threshold) and getattr(th, "metric", metric) != metric:
        warnings.warn(
            f"{where}: the threshold was estimated with metric={th.metric!r} but the "
            f"plot uses metric={metric!r}; epsilon is a distance in the first metric "
            "and means something else in the second [DD-118].",
            ParameterWarning, stacklevel=3)
    if theiler < 0:
        raise ParameterError("theiler must be >= 0")
    return int(theiler)


def _resolve_threshold(th, coords, *, metric, theiler, rng, kw) -> Threshold:
    if isinstance(th, Threshold):
        return th
    if isinstance(th, (int, float)) and not isinstance(th, bool):
        return _threshold(coords, mode="fixed", value=float(th), metric=metric,
                          theiler=theiler, rng=rng)
    if th is None or isinstance(th, str):
        return _threshold(coords, mode=th or "target_rr", metric=metric,
                          theiler=theiler, rng=rng, **kw)
    raise ParameterError(
        f"threshold must be a Threshold, a positive number, a mode name or None; "
        f"got {type(th).__name__}"
    )


def recurrence_plot(source, threshold=None, *, metric: str = "euclidean",
                    theiler: int | None = None, dtype: str = "uint8", store: str = "auto",
                    backend: str = "auto", tile_size: int | None = None,
                    budget_gb: float | None = None, precision: str = "double",
                    rng: SeedLike = None, **threshold_kwargs) -> RecurrenceMatrix:
    """Build a recurrence plot from a state space or a point array.

    Parameters
    ----------
    threshold
        A :class:`Threshold`, a positive number (a fixed epsilon), a mode name
        such as ``"target_rr"`` or ``"fan"``, or ``None`` for the default
        target-rate mode. Extra keyword arguments are forwarded to
        :func:`recurra.threshold`.
    theiler
        Theiler window: every pair with ``|i - j| < theiler`` is excluded.
        ``0`` excludes nothing, ``1`` the line of identity only, ``w`` the
        ``2w - 1`` central diagonals. This is the convention of the CRP
        Toolbox, pyunicorn and PyRQA [DD-19, DD-115]. ``None`` (default)
        takes the window a :class:`Threshold` was estimated with, or 1.
    store
        ``"memory"`` materialises, ``"sparse"`` keeps recurrent pairs only,
        ``"stream"`` streams and never allocates the matrix (``"none"`` is accepted as an old alias), ``"auto"`` decides
        from the memory budget.
    precision
        ``"double"`` (default) or ``"single"``. Single precision makes the
        distance kernel 3.2 times faster and is more than accurate enough for a
        threshold comparison, but it is opt-in [DD-79].
    """
    if precision not in ("double", "single"):
        raise ParameterError(
            f"precision must be 'double' or 'single', got {precision!r}")
    if metric not in METRICS:
        raise ParameterError(f"unknown metric {metric!r}; use one of {METRICS}")
    theiler = _reconcile(threshold, theiler, metric, "recurrence_plot")

    X, ss = _as_coords(source, "recurrence_plot")
    th = _resolve_threshold(threshold, ss if ss is not None else X,
                            metric=metric, theiler=theiler, rng=rng,
                            kw=threshold_kwargs)

    p = plan(X.shape[0], X.shape[0], X.shape[1], dtype=dtype, budget_gb=budget_gb,
             tile_size=tile_size, backend=backend, store=store,
             estimated_rr=th.achieved_rr)

    step = Step("recurrence_plot",
                {"kind": "rp", "n_points": X.shape[0], "dim": X.shape[1],
                 "metric": metric, "theiler": theiler,
                 "threshold_mode": th.mode, "epsilon": th.scalar,
                 "target_rr": th.target, "store": p.store, "tile_size": p.tile_size,
                 "precision": precision},
                caveats=(p.note,) if p.note else ())

    rm = RecurrenceMatrix(
        coords=X, threshold=th, kind="rp", metric=metric, theiler=theiler,
        dtype=dtype, tile_size=p.tile_size, store=p.store, precision=precision,
        groups=getattr(ss, "groups", ()) if ss is not None else (),
        fs=getattr(ss, "fs", 1.0) if ss is not None else 1.0,
        t0=getattr(ss, "t0", 0.0) if ss is not None else 0.0,
        meta={"subject": getattr(ss, "meta", {}).get("subject", "") if ss else "",
              "builder": "recurrence_plot"},
        provenance=(getattr(ss, "provenance", ()) if ss is not None else ()) + (step,),
        memory_plan=p,
    )
    if p.store == "memory":
        rm.materialize()
    if p.note and get_config().warn_on_caveat:
        import warnings

        warnings.warn(f"recurrence_plot: {p.note}", CaveatWarning, stacklevel=2)
    return rm


def cross_recurrence_plot(source1, source2, threshold=None, *,
                          metric: str = "euclidean", dtype: str = "uint8",
                          store: str = "auto", tile_size: int | None = None,
                          budget_gb: float | None = None, precision: str = "double",
                          rng: SeedLike = None,
                          **threshold_kwargs) -> RecurrenceMatrix:
    """Cross recurrence plot between two trajectories in one common space.

    The matrix is rectangular and not symmetric; its asymmetry about the line
    of identity is what encodes direction and lag of the coupling. Both
    trajectories must have the same dimensionality, since a cross recurrence
    compares them in the same phase space.
    """
    X1, ss1 = _as_coords(source1, "cross_recurrence_plot(source1)")
    X2, ss2 = _as_coords(source2, "cross_recurrence_plot(source2)")
    if X1.shape[1] != X2.shape[1]:
        raise ParameterError(
            f"a cross recurrence plot compares two trajectories in the same phase "
            f"space, but they have {X1.shape[1]} and {X2.shape[1]} dimensions. "
            "Use a joint recurrence plot for subsystems with different dimensions."
        )

    if isinstance(threshold, Threshold):
        th = threshold
    elif isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        th = _cross_threshold(X1, X2, mode="fixed", value=float(threshold),
                              metric=metric, rng=rng)
    else:
        th = _cross_threshold(X1, X2, mode=threshold or "target_rr", metric=metric,
                              rng=rng, **threshold_kwargs)

    p = plan(X1.shape[0], X2.shape[0], X1.shape[1], dtype=dtype, budget_gb=budget_gb,
             tile_size=tile_size, store=store, estimated_rr=th.achieved_rr)

    step = Step("cross_recurrence_plot",
                {"kind": "crp", "n_rows": X1.shape[0], "n_cols": X2.shape[0],
                 "dim": X1.shape[1], "metric": metric, "threshold_mode": th.mode,
                 "epsilon": th.scalar, "target_rr": th.target, "store": p.store},
                caveats=(p.note,) if p.note else ())

    rm = RecurrenceMatrix(
        coords=X1, coords2=X2, threshold=th, kind="crp", metric=metric, theiler=0,
        precision=precision,
        dtype=dtype, tile_size=p.tile_size, store=p.store,
        fs=getattr(ss1, "fs", 1.0) if ss1 is not None else 1.0,
        meta={"builder": "cross_recurrence_plot"},
        provenance=(getattr(ss1, "provenance", ()) if ss1 is not None else ()) + (step,),
        memory_plan=p,
    )
    if p.store == "memory":
        rm.materialize()
    return rm


def joint_recurrence_plot(sources: Sequence, thresholds=None, *,
                          combine: str = "and", metric: str = "euclidean",
                          theiler: int | None = None, dtype: str = "uint8",
                          store: str = "auto", tile_size: int | None = None,
                          target_rr: float = 0.05, rng: SeedLike = None,
                          **threshold_kwargs) -> RecurrenceMatrix:
    """Joint recurrence plot of several subsystems [DD-20].

    Each subsystem gets its **own** threshold, fixed to its own recurrence
    rate in its own phase space; the joint plot is then the elementwise AND.
    Subsystems need the same number of points but **not** the same
    dimensionality -- that is the whole point, and what makes the joint plot
    able to detect generalised synchronisation between attractors of
    different dimension.
    """
    if combine not in ("and", "product"):
        raise ParameterError(f"combine must be 'and' or 'product', got {combine!r}")
    if len(sources) < 2:
        raise ParameterError("a joint recurrence plot needs at least two subsystems")

    coords, spaces = [], []
    for k, s in enumerate(sources):
        X, ss = _as_coords(s, f"joint_recurrence_plot(sources[{k}])")
        coords.append(X)
        spaces.append(ss)
    n_points = {X.shape[0] for X in coords}
    if len(n_points) > 1:
        raise ParameterError(
            f"subsystems have different numbers of points {sorted(n_points)}. "
            "A joint recurrence plot needs a common time base; align them first "
            "(same window, same tau and m, or trim to the shortest)."
        )
    n = coords[0].shape[0]

    if thresholds is None:
        thresholds = [None] * len(coords)
    if len(thresholds) != len(coords):
        raise ParameterError(
            f"{len(thresholds)} thresholds for {len(coords)} subsystems")
    given = theiler
    for th in thresholds:
        theiler = _reconcile(th, given, metric, "joint_recurrence_plot")
        given = theiler if given is None else given

    ths: list[Threshold] = []
    for X, ss, th in zip(coords, spaces, thresholds, strict=False):
        ths.append(_resolve_threshold(
            th, ss if ss is not None else X, metric=metric, theiler=theiler,
            rng=rng, kw={"target_rr": target_rr, **threshold_kwargs}))

    parts = [RecurrenceMatrix(coords=X, threshold=t, kind="rp", metric=metric,
                              theiler=theiler, dtype=dtype, store="stream",
                              groups=getattr(ss, "groups", ()) if ss else ())
             for X, ss, t in zip(coords, spaces, ths, strict=False)]

    p = plan(n, n, sum(X.shape[1] for X in coords), dtype=dtype,
             tile_size=tile_size, store=store,
             estimated_rr=float(np.prod([t.achieved_rr or target_rr for t in ths])))

    joint = _JointRecurrenceMatrix(
        coords=coords[0], threshold=ths[0], kind="jrp", metric=metric,
        theiler=theiler, dtype=dtype, tile_size=p.tile_size, store=p.store,
        fs=getattr(spaces[0], "fs", 1.0) if spaces[0] is not None else 1.0,
        meta={"builder": "joint_recurrence_plot", "n_subsystems": len(parts)},
        provenance=((getattr(spaces[0], "provenance", ()) if spaces[0] else ()) + (
            Step("joint_recurrence_plot",
                 {"kind": "jrp", "n_points": n, "n_subsystems": len(parts),
                  "combine": combine, "metric": metric, "theiler": theiler,
                  "epsilons": [round(float(t.scalar), 6) for t in ths],
                  "target_rr_each": target_rr,
                  "dims": [int(X.shape[1]) for X in coords]},
                 notes="each subsystem carries its own threshold [DD-20]"),)),
        memory_plan=p,
    )
    joint.parts = parts
    joint.combine = combine
    joint.thresholds = tuple(ths)
    if p.store == "memory":
        joint.materialize()
    return joint


class _JointRecurrenceMatrix(RecurrenceMatrix):
    """A RecurrenceMatrix whose tiles are the AND of its subsystems' tiles."""

    parts: list = []
    combine: str = "and"
    thresholds: tuple = ()

    def _block(self, i0: int, i1: int, j0: int, j1: int) -> np.ndarray:
        out = None
        for part in self.parts:
            blk = part._block(i0, i1, j0, j1).astype(bool)
            out = blk if out is None else (out & blk)
        return out.astype(np.uint8) if self.dtype == "uint8" else out

    def subsystem_rates(self) -> list[float]:
        """Recurrence rate of each subsystem on its own."""
        return [p.recurrence_rate() for p in self.parts]

    def independence_ratio(self) -> float:
        """Joint rate divided by the product of the individual rates.

        Equal to 1 when the subsystems recur independently. Above 1 means
        they tend to recur at the same times, which is the signature the
        joint plot exists to detect.
        """
        prod = float(np.prod(self.subsystem_rates()))
        return self.recurrence_rate() / prod if prod > 0 else float("nan")
