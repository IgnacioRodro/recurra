"""Windowed recurrence: many plots from one record.

Design note (DD-40) -- the threshold scope decides what the windows can say.

Fixing the recurrence rate *per window* makes every window comparable in
structure but constant in density: RR carries no information, by construction,
and DET, LAM and the rest describe geometry at matched density. Fixing one
threshold *globally* instead turns RR(t) into an informative time series --
density itself becomes the signal -- at the cost that a window whose
amplitude drifts may come out nearly empty.

Both are right for different questions, neither is right for both, and the
choice silently changes what a windowed analysis means. So ``scope`` is an
explicit argument with no safe default to hide behind: ``per_window`` is the
default because comparing structure is the more common intent, and the
scope used is recorded on every row of the metrics table.

Design note (DD-41) -- windows are lazy.

A hundred subjects with ten windows each is a thousand recurrence matrices.
Materialising them all is pointless when the goal is a metrics table, so
``WindowedRecurrence`` holds the trajectory, the window list and the
thresholds, and builds a matrix only when one is asked for. ``keep`` caches
them when they will be reused; the default computes metrics and lets each
matrix go.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import SeedLike, resolve_rng
from ..exceptions import CaveatWarning, ComparabilityWarning, ParameterError
from ..provenance import Step, provenance_frame
from ..recurrence.builders import (
    cross_recurrence_plot,
    joint_recurrence_plot,
    recurrence_plot,
)
from ..recurrence.core import RecurrenceMatrix
from ..threshold.core import Threshold
from .spec import Window, WindowSpec, resolve_windows

#: Columns of coupling_indices() that read the amplitude, and so are void when
#: the amplitude coordinate has been rescaled.
CLASSICAL_AMPLITUDE_KEYS = ("mvl", "mi_tort", "mod_contrast", "preferred_phase")

SCOPES = ("per_window", "global")
KINDS = ("rp", "crp", "jrp")


def _window_source(source, w: Window):
    """Restrict a StateSpace or an array to a window."""
    if hasattr(source, "window"):
        return source.window(w.start, w.stop)
    return np.asarray(source)[w.start:w.stop]


def _n_points(source) -> int:
    return (source.n_points if hasattr(source, "n_points")
            else np.atleast_2d(np.asarray(source)).shape[0])


def _fs(source, default: float = 1.0) -> float:
    return float(getattr(source, "fs", default) or default)


def _t0(source) -> float:
    return float(getattr(source, "t0", 0.0))


@dataclass
class WindowedRecurrence:
    """A sequence of recurrence structures over windows of one record.

    Iterating yields ``(Window, RecurrenceMatrix)`` pairs; indexing by window
    number returns the matrix. Matrices are built on demand [DD-41].
    """

    sources: tuple
    windows: tuple[Window, ...]
    thresholds: tuple
    kind: str = "rp"
    scope: str = "per_window"
    build_kwargs: dict = field(default_factory=dict)
    subject: str = ""
    caveats: tuple[str, ...] = ()
    provenance: tuple[Step, ...] = ()
    keep: bool = False
    _cache: dict = field(default_factory=dict, repr=False)
    _metrics: pd.DataFrame | None = field(default=None, repr=False)
    _metrics_key: tuple = field(default=(), repr=False)
    #: Line histograms from the last RQA pass, keyed by (subject, window).
    #: Every metric at every floor is a sum over these, so keeping them makes
    #: exploring floors free [DD-108].
    _histograms: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> RecurrenceMatrix:
        if i in self._cache:
            return self._cache[i]
        w = self.windows[i]
        th = self.thresholds[i]
        parts = [_window_source(s, w) for s in self.sources]
        kw = dict(self.build_kwargs)
        if self.kind == "rp":
            rm = recurrence_plot(parts[0], th, **kw)
        elif self.kind == "crp":
            kw.pop("theiler", None)          # a cross plot has no Theiler band
            rm = cross_recurrence_plot(parts[0], parts[1], th, **kw)
        else:
            kw.pop("theiler", None)
            rm = joint_recurrence_plot(parts, thresholds=th,
                                       theiler=self.build_kwargs.get("theiler", 1), **kw)
        rm.meta = {**rm.meta, "subject": self.subject, "window": w.index,
                   "window_label": w.label}
        if self.keep:
            self._cache[i] = rm
        return rm

    def __iter__(self) -> Iterator[tuple[Window, RecurrenceMatrix]]:
        for i in range(len(self)):
            yield self.windows[i], self[i]

    def matrices(self) -> dict[str, RecurrenceMatrix]:
        """All matrices, keyed by window label. Materialises every window."""
        return {w.label: self[i] for i, w in enumerate(self.windows)}

    def window_frame(self) -> pd.DataFrame:
        return pd.DataFrame([w.to_row() for w in self.windows])

    # ----------------------------------------------------------- metrics
    def line_length_sweep(self, values=(2, 3, 4, 6, 8, 12, 16),
                          metrics=("DET", "LAM"), **kwargs) -> pd.DataFrame:
        """DET and LAM at several floors, per window, from the histograms.

        The floor that suits a whole record does not suit a window: measured on
        13.6-second windows, LAM read 0.857 at a floor of 2, 0.442 at 3, 0.098
        at 4 and zero from 6 -- while the same recording whole wanted 8. This
        answers it from counts already accumulated, so it costs milliseconds
        where calling :meth:`metrics` again per floor costs a full re-walk
        [DD-108].

        Call :meth:`metrics` with ``rqa`` first; the histograms come from there.
        """
        from ..rqa.metrics import line_length_sweep as _sweep

        if not self._histograms:
            raise ParameterError(
                "no histograms yet: call metrics(rqa=True) before sweeping")
        frames = []
        for (subject, window), hist in sorted(self._histograms.items()):
            frame = _sweep(hist, values=values, metrics=metrics, **kwargs)
            frame.insert(0, "window", window)
            frame.insert(0, "subject", subject or self.subject)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def metrics(self, *, recompute: bool = False, extra: Sequence[str] = (),
                rqa: bool | Sequence[str] = False, l_min: int = 8,
                v_min: int = 8, w_min: int = 2,
                denominator: str = "theiler_corrected",
                dynamics: bool | Sequence[str] = False,
                invariants: bool | Sequence[str] = False,
                classical: bool = False,
                dynamics_kwargs: dict | None = None) -> pd.DataFrame:
        """One tidy row per window: geometry, threshold, rate and optionally RQA.

        ``l_min`` and ``v_min`` default to 8, the same floors as
        :func:`recurra.rqa` [DD-98]. They defaulted to 2 here after the
        top-level default had moved, so the two entry points returned
        different DET for the same matrix and the recipe that used this one
        came back saturated.

        This is the table that goes into a paper. With ``rqa=True`` every
        metric of :data:`recurra.METRIC_NAMES` is added as a column; pass a
        sequence to select. The line histograms are accumulated by streaming,
        so asking for RQA does not force any matrix into memory [DD-56].
        """
        key = (bool(rqa) if isinstance(rqa, bool) else tuple(rqa), l_min, v_min,
               w_min, denominator,
               bool(dynamics) if isinstance(dynamics, bool) else tuple(dynamics),
               bool(invariants) if isinstance(invariants, bool) else tuple(invariants),
               bool(classical))
        if self._metrics is not None and not recompute and self._metrics_key == key:
            return self._metrics
        self._metrics_key = key
        rows = []
        warned_rescaled = False
        for i, w in enumerate(self.windows):
            rm = self[i]
            th = rm.threshold
            row = {
                "subject": self.subject, "kind": self.kind, "scope": self.scope,
                **w.to_row(),
                "n_rows": rm.n_rows, "n_cols": rm.n_cols, "dim": rm.dim,
                "metric": rm.metric, "theiler": rm.theiler,
                "threshold_mode": th.mode, "epsilon": th.scalar,
                "target_rr": th.target, "recurrence_rate": rm.recurrence_rate(),
            }
            if self.kind == "jrp" and hasattr(rm, "subsystem_rates"):
                rates = rm.subsystem_rates()
                for k, r in enumerate(rates):
                    row[f"rr_subsystem_{k}"] = r
                row["independence_ratio"] = rm.independence_ratio()
            if rqa:
                from ..rqa.metrics import rqa as _rqa

                # Keep the histogram. Every RQA metric at every line-length
                # floor is a sum over these counts, so a caller exploring
                # floors can do it for free -- and without them each floor
                # costs a full re-walk of every matrix. Measured on a
                # 19-window batch of two subjects, one extra floor cost 50 s
                # against 13 s to build the batch, so a seven-floor sweep was
                # five times the analysis it was meant to inform [DD-108].
                values, hist = _rqa(rm, l_min=l_min, v_min=v_min, w_min=w_min,
                                    denominator=denominator,
                                    metrics="all" if rqa is True else list(rqa),
                                    return_histogram=True)
                row.update(values)
                self._histograms[(row.get("subject", ""), int(row["window"]))] = hist
                row["l_min"] = l_min
                row["denominator"] = denominator
            if classical:
                from ..metrics.classic import coupling_indices

                window = _window_source(self.sources[0], w)
                blocks = {g.kind: g for g in window.groups}
                if "phase_circle" in blocks and "amplitude" in blocks:
                    coords = np.asarray(window.coords, dtype=float)
                    pg, ag = blocks["phase_circle"], blocks["amplitude"]
                    angle = np.arctan2(coords[:, pg.start + 1], coords[:, pg.start])
                    rescaled = ag.params.get("scale")
                    if rescaled:
                        # The coordinate is no longer the envelope: a z-scored
                        # one gave MVL 13.3 and MI 0.88 on a signal whose true
                        # values were 0.31 and 0.035. NaN is the honest entry
                        # [DD-116].
                        row.update({k: np.nan for k in CLASSICAL_AMPLITUDE_KEYS})
                        if not warned_rescaled:
                            warnings.warn(
                                f"classical indices skipped: the amplitude block "
                                f"{ag.label!r} was rescaled with {rescaled!r}, so it "
                                "is not an envelope any more. Compute them with "
                                "coupling_indices() on the recording's envelope.",
                                CaveatWarning, stacklevel=2)
                            warned_rescaled = True
                    else:
                        row.update(coupling_indices(angle, coords[:, ag.start]))

            if dynamics or invariants:
                from ..dynamics.measures import dynamics_measures

                window = _window_source(self.sources[0], w)
                row.update(dynamics_measures(
                    window,
                    measures="default" if dynamics is True else (
                        list(dynamics) if dynamics else []),
                    invariants=invariants, **(dynamics_kwargs or {})))
            for name in extra:
                row[name] = getattr(rm, name)() if callable(getattr(rm, name, None)) else np.nan
            rows.append(row)
        self._metrics = pd.DataFrame(rows)
        return self._metrics

    def meta_recurrence(self, metrics: Sequence[str] | None = None, **kwargs):
        """Recurrence plot of this analysis' per-window measurements.

        The meta-RP [DD-119]: see :func:`recurra.meta_recurrence` for the
        options. Uses the last table computed with :meth:`metrics`, or
        ``metrics(rqa=True)`` if none was.
        """
        from .meta import meta_recurrence

        return meta_recurrence(self, metrics, **kwargs)

    def series(self, column: str = "recurrence_rate") -> pd.DataFrame:
        """A time series of one metric, indexed by window centre time."""
        df = self.metrics()
        if column not in df.columns:
            raise ParameterError(f"no column {column!r}; have {list(df.columns)}")
        return df[["window", "t_center_s", column]].copy()

    # --------------------------------------------------------- reporting
    def describe(self) -> pd.DataFrame:
        m = self.metrics()
        return pd.DataFrame([{
            "subject": self.subject, "kind": self.kind, "scope": self.scope,
            "n_windows": len(self), "window_samples": self.windows[0].n_samples,
            "window_seconds": self.windows[0].n_samples / self.windows[0].fs,
            "overlap_samples": (self.windows[1].start - self.windows[0].start
                                if len(self) > 1 else 0),
            "dim": int(m["dim"].iloc[0]),
            "epsilon_mean": float(m["epsilon"].mean()),
            "epsilon_cv": float(m["epsilon"].std() / m["epsilon"].mean())
            if m["epsilon"].mean() else np.nan,
            "rr_mean": float(m["recurrence_rate"].mean()),
            "rr_std": float(m["recurrence_rate"].std()),
            "rr_min": float(m["recurrence_rate"].min()),
            "rr_max": float(m["recurrence_rate"].max()),
        }])

    def provenance_frame(self) -> pd.DataFrame:
        return provenance_frame(self.provenance)

    def summary(self) -> str:
        m = self.metrics()
        w0 = self.windows[0]
        lines = [
            f"WindowedRecurrence({self.kind.upper()}) {len(self)} windows of "
            f"{w0.n_samples} samples ({w0.n_samples / w0.fs:.2f} s)",
            f"  subject     : {self.subject or '(unnamed)'}",
            f"  scope       : {self.scope}"
            + ("  (one epsilon per window: RR is constant by construction, "
               "structure is comparable)" if self.scope == "per_window"
               else "  (one epsilon for all windows: RR(t) is informative)"),
            f"  epsilon     : mean {m['epsilon'].mean():.5g}"
            + (f", cv {m['epsilon'].std() / m['epsilon'].mean():.3f}"
               if m['epsilon'].mean() else ""),
            f"  recurrence  : mean {m['recurrence_rate'].mean():.3%}, "
            f"range {m['recurrence_rate'].min():.3%}-{m['recurrence_rate'].max():.3%}",
        ]
        for c in self.caveats:
            lines.append(f"  caveat      : {c}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<WindowedRecurrence {self.kind} {len(self)} windows "
                f"subject={self.subject!r}>")


# ===================================================================== API
def _resolve_thresholds(sources, windows, kind, scope, threshold, rng, kw):
    """One threshold per window, or one shared by all [DD-40]."""
    from ..threshold.estimate import cross_threshold as _cross
    from ..threshold.estimate import threshold as _thr

    theiler = kw.get("theiler", 1)
    metric = kw.get("metric", "euclidean")
    tkw = {k: v for k, v in kw.items()
           if k in ("target_rr", "percentile", "value", "n_neighbors", "factor",
                    "method", "tol", "max_iter", "n_samples", "block_targets")}

    def one(parts, mode_or_th):
        if isinstance(mode_or_th, Threshold):
            return mode_or_th
        if isinstance(mode_or_th, (int, float)) and not isinstance(mode_or_th, bool):
            return _thr(parts[0], mode="fixed", value=float(mode_or_th),
                        metric=metric, theiler=theiler, rng=rng)
        mode = mode_or_th or "target_rr"
        if kind == "crp":
            return _cross(parts[0], parts[1], mode=mode, metric=metric, rng=rng,
                          **{k: v for k, v in tkw.items()
                             if k in ("target_rr", "percentile", "value", "n_samples")})
        if kind == "jrp":
            return [_thr(p, mode=mode, metric=metric, theiler=theiler, rng=rng, **tkw)
                    for p in parts]
        return _thr(parts[0], mode=mode, metric=metric, theiler=theiler, rng=rng, **tkw)

    if scope == "global":
        th = one(list(sources), threshold)
        return tuple(th for _ in windows)
    return tuple(one([_window_source(s, w) for s in sources], threshold)
                 for w in windows)


def windowed_recurrence(source, window, *, kind: str = "rp", threshold=None, target_rr: float = 0.05,
                        theiler: int = 1, metric: str = "euclidean",
                        store: str | None = None, precision: str = "double",
                        scope: str = "per_window", subject: str = "",
                        keep: bool = False, rng: SeedLike = None,
                        sources: Sequence | None = None,
                        **kwargs) -> WindowedRecurrence:
    """Cut a record into windows and build one recurrence structure per window.

    Parameters
    ----------
    source
        A `StateSpace` or a point array. For ``kind="crp"`` or ``"jrp"``, pass
        the additional trajectories in ``sources`` (or pass a sequence as
        ``source``).
    window
        A :class:`WindowSpec`, an integer number of windows, or explicit
        ``(start, stop)`` pairs.
    scope
        ``"per_window"`` estimates a threshold inside each window, so every
        window sits at the same recurrence rate and structure is comparable.
        ``"global"`` estimates one threshold on the whole record, so the
        recurrence rate becomes an informative time series [DD-40].
    keep
        Cache the matrices. Off by default: a thousand matrices is a lot of
        memory when the goal is a metrics table [DD-41].

    Any other keyword goes to the recurrence builder and the threshold
    estimator, so every option of the unwindowed path is available here.
    """
    if kind not in KINDS:
        raise ParameterError(f"kind must be one of {KINDS}, got {kind!r}")
    if scope not in SCOPES:
        raise ParameterError(
            f"scope must be one of {SCOPES}, got {scope!r}. This is not a detail: "
            "per_window fixes the rate and compares structure, global fixes epsilon "
            "and makes the rate itself the signal."
        )
    rng = resolve_rng(rng)

    if sources is not None:
        srcs = [source, *sources] if source is not None else list(sources)
    elif isinstance(source, (list, tuple)):
        srcs = list(source)
    else:
        srcs = [source]

    if kind == "crp" and len(srcs) != 2:
        raise ParameterError(f"kind='crp' needs exactly 2 trajectories, got {len(srcs)}")
    if kind == "jrp" and len(srcs) < 2:
        raise ParameterError(f"kind='jrp' needs at least 2 subsystems, got {len(srcs)}")

    lengths = {_n_points(s) for s in srcs}
    if len(lengths) > 1:
        raise ParameterError(
            f"windowed analysis needs a common time base; the trajectories have "
            f"{sorted(lengths)} points. Trim them to the shortest first."
        )
    n = lengths.pop()
    fs = _fs(srcs[0])
    windows, caveats = resolve_windows(window, n, fs, _t0(srcs[0]))

    thresholds = _resolve_thresholds(srcs, windows, kind, scope, threshold, rng, kwargs)

    # The common options are explicit parameters [DD-118]; they travel to the
    # threshold estimator and the builder like any other keyword.
    kwargs.update(target_rr=target_rr, theiler=theiler, metric=metric,
                  precision=precision)
    if store is not None:
        kwargs["store"] = store
    build_kwargs = {k: v for k, v in kwargs.items()
                    if k in ("metric", "theiler", "dtype", "store", "tile_size",
                             "budget_gb", "combine", "precision")}
    build_kwargs.setdefault("store", "memory" if keep else "stream")

    step = Step("windowed_recurrence",
                {"kind": kind, "scope": scope, "n_windows": len(windows),
                 "window_samples": windows[0].n_samples,
                 "window_seconds": round(windows[0].n_samples / fs, 4),
                 "step_samples": (windows[1].start - windows[0].start
                                  if len(windows) > 1 else 0),
                 "n_trajectories": len(srcs), **build_kwargs},
                caveats=tuple(caveats))

    parent_prov = tuple(getattr(srcs[0], "provenance", ()) or ())
    subject = subject or str(getattr(srcs[0], "meta", {}).get("subject", ""))

    wr = WindowedRecurrence(
        sources=tuple(srcs), windows=tuple(windows), thresholds=thresholds,
        kind=kind, scope=scope, build_kwargs=build_kwargs, subject=subject,
        caveats=tuple(caveats), provenance=parent_prov + (step,), keep=keep,
    )
    from ..config import get_config

    if caveats and get_config().warn_on_caveat:
        import warnings

        for c in caveats:
            warnings.warn(f"windowed_recurrence: {c}", CaveatWarning, stacklevel=2)
    return wr


def windowed_cross_recurrence(source1, source2, window, *, target_rr: float = 0.05,
                              metric: str = "euclidean", store: str | None = None,
                              precision: str = "double", **kw) -> WindowedRecurrence:
    """Windowed cross recurrence between two trajectories.

    Same options as :func:`windowed_recurrence`, without ``theiler``: the two
    trajectories are different systems, so index proximity carries no
    autocorrelation to exclude [DD-32].
    """
    return windowed_recurrence(source1, window, kind="crp", sources=[source2],
                               target_rr=target_rr, metric=metric, store=store,
                               precision=precision, **kw)


def windowed_joint_recurrence(sources: Sequence, window, *, target_rr: float = 0.05,
                              theiler: int = 1, metric: str = "euclidean",
                              store: str | None = None, precision: str = "double",
                              **kw) -> WindowedRecurrence:
    """Windowed joint recurrence across subsystems, independent thresholds.

    Same options as :func:`windowed_recurrence`; each subsystem gets its own
    threshold at ``target_rr`` in every window [DD-20].
    """
    return windowed_recurrence(list(sources), window, kind="jrp",
                               target_rr=target_rr, theiler=theiler, metric=metric,
                               store=store, precision=precision, **kw)


# ================================================================== batch
def _one_record(pair, *, window, kw, rng=None) -> WindowedRecurrence:
    """One record of a batch. At module level so it survives pickling."""
    name, source = pair
    return windowed_recurrence(source, window, subject=name, rng=rng, **kw)


@dataclass
class BatchWindowedRecurrence:  # noqa: D101 -- documented below
    """Windowed analysis across several records.

    Real corpora are subjects times windows: a hundred subjects with ten
    windows each is a thousand structures, each small. This holds one
    :class:`WindowedRecurrence` per record and concatenates their metrics
    into one tidy table.
    """

    results: dict[str, WindowedRecurrence]
    spec: Any = None
    contract: Any = None
    report: Any = None

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, key: str) -> WindowedRecurrence:
        return self.results[key]

    def __iter__(self):
        return iter(self.results.items())

    @property
    def n_matrices(self) -> int:
        return sum(len(w) for w in self.results.values())

    def line_length_sweep(self, **kw) -> pd.DataFrame:
        """The sweep for every record in the batch, from their histograms."""
        frames = []
        for name, wr in self.results.items():
            frame = wr.line_length_sweep(**kw)
            frame["subject"] = frame["subject"].replace("", name)
            frame.loc[frame["subject"].isna() | (frame["subject"] == ""),
                      "subject"] = name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def metrics(self, *, n_jobs: int | None = None, backend: str = "auto",
                **kw) -> pd.DataFrame:
        """One tidy table for the whole batch.

        Design note (DD-109) -- the metrics are the work, so they run in
        parallel too.

        ``batch_windowed_recurrence`` takes ``n_jobs`` and parallelises the
        *build*, which resolves windows and estimates thresholds. It does not
        keep the matrices, so the first call to this method rebuilds every one
        of them -- and that is where the time is. Measured on a 34 000-point
        record split into 19 windows: 4.8 s to build and resolve, 20 s to walk
        the matrices for RQA. The build was spread over sixteen cores and the
        expensive part ran on one.

        On a corpus of 49 subjects that is sixteen minutes of one core per
        channel while fifteen sit idle. With ``n_jobs`` here the same work
        takes as long as its slowest subject.

        Results do not depend on ``n_jobs``: each record has its own stream
        fixed at build time [DD-81].
        """
        names = list(self.results)
        if n_jobs in (None, 1, 0) or len(names) < 2:
            frames = [self._one(name, **kw) for name in names]
        else:
            from ..parallel.runner import parallel_map

            # A process worker returns a copy, so anything the call cached --
            # the line histograms above all -- stays in the worker and is lost.
            # The histograms come back explicitly and are stored here, or a
            # later line_length_sweep would find nothing and say so.
            pairs = parallel_map(self._one_with_histograms, names,
                                 n_jobs=n_jobs, backend=backend,
                                 pass_rng=False, **kw)
            frames = []
            for name, (frame, hists) in zip(names, pairs, strict=False):
                frames.append(frame)
                self.results[name]._histograms.update(hists)
        return pd.concat(frames, ignore_index=True)

    def _one_with_histograms(self, name: str, **kw):
        frame = self._one(name, **kw)
        return frame, dict(self.results[name]._histograms)

    def _one(self, name: str, **kw) -> pd.DataFrame:
        """One record's table, with its name filled in. At method level so a
        worker process can pickle it."""
        df = self.results[name].metrics(**kw).copy()
        df["subject"] = df["subject"].replace("", name)
        df.loc[df["subject"].isna() | (df["subject"] == ""), "subject"] = name
        return df

    def describe(self) -> pd.DataFrame:
        return pd.concat([wr.describe() for wr in self.results.values()],
                         ignore_index=True)

    def check_comparability(self, contract=None, **kw):
        """Verify the records were built the same way [DD-42]."""
        from ..comparability import check_comparability as _check

        return _check(self, contract or self.contract, **kw)

    def metrics_by_window(self, value: str = "recurrence_rate") -> pd.DataFrame:
        """Subject-by-window matrix of one metric: the shape a classifier wants."""
        from ..comparability import feature_matrix

        return feature_matrix(self.metrics(), value)

    def group_comparison(self, groups: Mapping[str, str], **kw) -> pd.DataFrame:
        """Compare two groups window by window [DD-44].

        ``groups`` maps record name to group label.
        """
        from ..comparability import group_comparison as _gc

        m = self.metrics()
        m = m.assign(group=m["subject"].map(groups))
        return _gc(m, "group", **kw)

    def summary(self) -> str:
        m = self.metrics()
        return (f"BatchWindowedRecurrence: {len(self)} records, "
                f"{self.n_matrices} windows total\n"
                f"  windows per record : "
                f"{sorted({len(w) for w in self.results.values()})}\n"
                f"  recurrence rate    : mean {m['recurrence_rate'].mean():.3%}, "
                f"range {m['recurrence_rate'].min():.3%}-"
                f"{m['recurrence_rate'].max():.3%}")


def batch_windowed_recurrence(sources: Mapping[str, Any] | Sequence, window, *,
                              rng: SeedLike = None, progress: bool = False,
                              contract=None, verify: bool = True,
                              n_jobs: int | None = None, backend: str = "auto",
                              **kw) -> BatchWindowedRecurrence:
    """Run a windowed analysis over many records.

    ``sources`` maps a record name to a trajectory (or to a sequence of
    trajectories for CRP/JRP). Records may have different lengths: the window
    spec is resolved against each one [DD-18].

    Each record gets an independent random stream derived from ``rng``, so
    results do not depend on the order or on how the work is distributed
    [DD-05].

    Parameters
    ----------
    contract
        A :class:`~recurra.comparability.ComparabilityContract` the results
        must satisfy. With ``strict=True`` on the contract, a violation
        raises; otherwise it is reported and warned about [DD-42].
    verify
        Check comparability even without a contract, by asking whether every
        axis is constant across records. A window spec that resolves to
        different window lengths per record is caught here.
    n_jobs
        Records are independent, so they run in parallel when asked. Each gets
        its own random stream derived by position, so **the result does not
        depend on n_jobs** [DD-81]. ``-1`` uses every core.
    """
    from ..comparability import ComparabilityContract, check_comparability

    if isinstance(window, WindowSpec) and window.n_windows is not None and verify:
        import warnings

        warnings.warn(
            "batch_windowed_recurrence: n_windows resolves against each record's "
            "own length, so records of different length get windows of different "
            "duration and 'window 0' stops meaning the same thing. For "
            "cross-record comparison use WindowSpec(size=..., step=...) in "
            "seconds or samples [DD-42].", ComparabilityWarning, stacklevel=2)

    from ..config import spawn_rngs

    items = (list(sources.items()) if isinstance(sources, Mapping)
             else [(f"record{i:03d}", s) for i, s in enumerate(sources)])
    streams = spawn_rngs(rng, max(1, len(items)))

    from ..parallel.runner import parallel_map, resolve_jobs

    if resolve_jobs(n_jobs) > 1 and len(items) > 1:
        # A module-level callable, not a lambda: the process backend pickles it.
        computed = parallel_map(_one_record, items, n_jobs=n_jobs,
                                backend=backend, rng=rng, progress=progress,
                                window=window, kw=kw)
        results = {name: wr for (name, _), wr in zip(items, computed, strict=False)}
    else:
        results = {}
        for (name, src), stream in zip(items, streams, strict=False):
            if progress:
                print(f"  [{len(results) + 1}/{len(items)}] {name}", flush=True)
            results[name] = windowed_recurrence(src, window, subject=name,
                                                rng=stream, **kw)

    batch = BatchWindowedRecurrence(results=results, spec=window, contract=contract)
    if verify or contract is not None:
        report = check_comparability(batch, contract or ComparabilityContract(
            metric=None, theiler=None))
        batch.report = report
        if not report.ok:
            import warnings

            from ..config import get_config

            if get_config().warn_on_caveat:
                warnings.warn(
                    "batch_windowed_recurrence: the records are not fully "
                    "comparable -- "
                    + "; ".join(f"{a.axis} {a.message}" for a in report.violations)
                    + ". Call .check_comparability() for the detail and for which "
                      "metric families remain valid.", ComparabilityWarning,
                    stacklevel=2)
    return batch
