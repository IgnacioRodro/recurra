"""Comparability: making recurrence structures answerable side by side.

Design note (DD-42) -- comparability is a contract, not an accident.

The project's aim is to relate RQA and nonlinear metrics to classical
coupling indices, and eventually to separate groups by them. That only means
anything if the structures being compared were built the same way. Three
defaults break this silently, and all three were reachable from the API as it
stood:

1. ``m="auto"``, ``tau="auto"`` estimate per record, so subjects end up in
   phase spaces of different dimension. Distances, rates and line statistics
   are then not on a common footing at all.
2. ``n_windows=k`` resolves against each record's own length, so window 0 of
   a 45 s record covers 7.9 s while window 0 of a 72 s record covers 12.8 s.
   "Window 0" stops meaning the same thing.
3. ``subsample(max_points=N)`` decimates by a different factor per record,
   giving effective sampling rates of 85, 115 and 71 Hz for three subjects
   asked for the same point count. Every time-based quantity then lives in
   different units per subject.

So a :class:`ComparabilityContract` states the parameters that must be shared,
and :func:`check_comparability` verifies that the structures actually produced
satisfy it, naming the offenders. The contract is data: it is exported next to
the results, so a reader can see what was held fixed.

Design note (DD-43) -- what matched, and therefore what may be compared.

Matching is not all-or-nothing. Holding the recurrence rate fixed makes DET,
LAM and the line statistics comparable while making RR itself uninformative;
holding epsilon fixed does the opposite. Matching point counts is required
before any length-valued metric (L, Lmax, trapping time) can be compared at
all. The report therefore does not merely pass or fail: it states which
families of metrics the observed matching licenses, and which it does not.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .config import SeedLike, resolve_rng
from .exceptions import ParameterError

#: Metric families and what each needs matched before it can be compared.
METRIC_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "recurrence_rate": ("dim", "metric", "theiler", "epsilon"),
    "line_length": ("dim", "metric", "theiler", "n_points", "fs", "target_rr"),
    "line_ratio": ("dim", "metric", "theiler", "target_rr"),
    "entropy": ("dim", "metric", "theiler", "n_points", "target_rr"),
    "timescale": ("dim", "metric", "theiler", "fs", "target_rr"),
    "invariant": ("dim", "metric", "fs"),
}

#: Which concrete metrics belong to each family.
METRIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "recurrence_rate": ("RR", "recurrence_rate", "independence_ratio"),
    "line_length": ("L", "Lmax", "TT", "Vmax", "W", "Wmax", "DIV"),
    "line_ratio": ("DET", "LAM", "transitivity", "clustering"),
    "entropy": ("ENTR", "ENTR_norm", "ENTW", "RTE"),
    "timescale": ("T1", "T2", "TREND"),
    "invariant": ("D2", "lambda_1", "K2"),
}

AXES = ("dim", "route", "scaling", "blocks", "m", "tau", "fs", "n_points",
        "metric", "theiler", "threshold_mode", "target_rr", "epsilon",
        "window_samples", "window_step")


@dataclass(frozen=True)
class ComparabilityContract:
    """The construction parameters that must be shared to allow comparison.

    Any field left ``None`` is not checked. ``strict`` turns violations into
    exceptions instead of report entries.
    """

    dim: int | None = None
    route: str | None = None
    scaling: str | None = None
    blocks: tuple[str, ...] | None = None
    m: int | None = None
    tau: int | None = None
    fs: float | None = None
    n_points: int | None = None
    metric: str | None = "euclidean"
    theiler: int | None = 1
    threshold_mode: str | None = None
    target_rr: float | None = None
    epsilon: float | None = None
    window_samples: int | None = None
    window_step: int | None = None
    #: Relative tolerance allowed on numeric axes.
    tolerance: float = 0.0
    strict: bool = False
    notes: str = ""

    def with_(self, **kw) -> ComparabilityContract:
        return replace(self, **kw)

    def declared(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()
                if k in AXES and v is not None}

    def to_frame(self) -> pd.DataFrame:
        rows = [{"axis": k, "required_value": v} for k, v in self.declared().items()]
        rows.append({"axis": "tolerance", "required_value": self.tolerance})
        if self.notes:
            rows.append({"axis": "notes", "required_value": self.notes})
        return pd.DataFrame(rows)

    def summary(self) -> str:
        d = self.declared()
        if not d:
            return "ComparabilityContract: nothing declared (no axis is checked)"
        width = max(len(k) for k in d)
        lines = ["ComparabilityContract"]
        lines += [f"  {k:<{width}} = {v}" for k, v in d.items()]
        if self.tolerance:
            lines.append(f"  tolerance{'':<{max(0, width - 9)}} = {self.tolerance:g}")
        return "\n".join(lines)


@dataclass
class AxisResult:
    axis: str
    required: Any
    observed: tuple
    status: str          # "ok" | "violated" | "unchecked" | "absent"
    offenders: tuple[str, ...] = ()
    message: str = ""


@dataclass
class ComparabilityReport:
    """What matched, what did not, and what may therefore be compared."""

    axes: list[AxisResult] = field(default_factory=list)
    n_units: int = 0
    labels: tuple[str, ...] = ()
    contract: ComparabilityContract | None = None

    @property
    def ok(self) -> bool:
        return not any(a.status == "violated" for a in self.axes)

    @property
    def violations(self) -> list[AxisResult]:
        return [a for a in self.axes if a.status == "violated"]

    def matched(self) -> set[str]:
        """Axes that are constant across the units, whether required or not."""
        return {a.axis for a in self.axes if a.status == "ok"}

    def metric_validity(self) -> pd.DataFrame:
        """Which metric families the observed matching licenses [DD-43]."""
        matched = self.matched()
        rows = []
        for family, needs in METRIC_REQUIREMENTS.items():
            missing = [n for n in needs if n not in matched]
            rows.append({
                "metric_family": family,
                "metrics": ", ".join(METRIC_FAMILIES.get(family, ())),
                "requires": ", ".join(needs),
                "comparable": not missing,
                "missing": ", ".join(missing),
            })
        return pd.DataFrame(rows)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "axis": a.axis, "status": a.status, "required": a.required,
            "n_distinct_observed": len(a.observed),
            "observed": "; ".join(str(o) for o in a.observed[:6])
            + ("; ..." if len(a.observed) > 6 else ""),
            "offenders": "; ".join(a.offenders[:6]),
            "message": a.message,
        } for a in self.axes])

    def raise_if_violated(self) -> None:
        if self.ok:
            return
        detail = "; ".join(f"{a.axis}: {a.message}" for a in self.violations)
        raise ParameterError(
            f"comparability contract violated across {self.n_units} units -- {detail}"
        )

    def summary(self) -> str:
        head = (f"ComparabilityReport: {self.n_units} units, "
                + ("all declared axes match" if self.ok
                   else f"{len(self.violations)} violation(s)"))
        lines = [head]
        for a in self.axes:
            mark = {"ok": "  OK  ", "violated": "  FAIL", "unchecked": "  --  ",
                    "absent": "  ?   "}[a.status]
            lines.append(f"{mark} {a.axis:<16s} {a.message}")
        valid = self.metric_validity()
        good = valid[valid.comparable]["metric_family"].tolist()
        bad = valid[~valid.comparable]
        lines.append("")
        lines.append("  comparable metric families : "
                     + (", ".join(good) if good else "none"))
        for _, r in bad.iterrows():
            lines.append(f"  NOT comparable: {r.metric_family:<16s} "
                         f"(unmatched: {r.missing})")
        return "\n".join(lines)


# ------------------------------------------------------------- extraction
def _describe_unit(obj, label: str) -> dict[str, Any]:
    """Reduce any analysis object to the axes that decide comparability."""
    from .recurrence.core import RecurrenceMatrix
    from .statespace.core import StateSpace
    from .windowed.recurrence import WindowedRecurrence

    d: dict[str, Any] = {"label": label}

    if isinstance(obj, WindowedRecurrence):
        m = obj.metrics()
        w0 = obj.windows[0]
        d.update(dim=int(m["dim"].iloc[0]), fs=float(w0.fs),
                 n_points=int(m["n_rows"].iloc[0]),
                 metric=str(m["metric"].iloc[0]), theiler=int(m["theiler"].iloc[0]),
                 threshold_mode=str(m["threshold_mode"].iloc[0]),
                 target_rr=(None if pd.isna(m["target_rr"].iloc[0])
                            else float(m["target_rr"].iloc[0])),
                 epsilon=(float(m["epsilon"].iloc[0])
                          if m["epsilon"].nunique() == 1 else None),
                 window_samples=int(w0.n_samples),
                 window_step=(int(obj.windows[1].start - w0.start)
                              if len(obj.windows) > 1 else int(w0.n_samples)),
                 n_windows=len(obj.windows), scope=obj.scope, kind=obj.kind)
        src = obj.sources[0]
        if isinstance(src, StateSpace):
            d.update(route=src.route, scaling=src.scaling,
                     blocks=tuple(src.block_labels))
            emb = (src.meta or {}).get("embedding") or {}
            if emb:
                first = next(iter(emb.values()))
                d.update(m=first.get("m"), tau=first.get("tau"))
        for g in getattr(src, "groups", ()):
            if g.kind == "delay":
                d.setdefault("m", g.params.get("m"))
                d.setdefault("tau", g.params.get("tau"))
        return d

    if isinstance(obj, RecurrenceMatrix):
        th = obj.threshold
        d.update(dim=obj.dim, fs=float(obj.fs), n_points=obj.n_rows,
                 metric=obj.metric, theiler=obj.theiler,
                 threshold_mode=th.mode, target_rr=th.target,
                 epsilon=float(th.scalar), kind=obj.kind)
        return d

    if isinstance(obj, StateSpace):
        d.update(dim=obj.dim, fs=float(obj.fs), n_points=obj.n_points,
                 route=obj.route, scaling=obj.scaling,
                 blocks=tuple(obj.block_labels))
        for g in obj.groups:
            if g.kind == "delay":
                d.setdefault("m", g.params.get("m"))
                d.setdefault("tau", g.params.get("tau"))
        return d

    raise ParameterError(
        f"cannot assess comparability of {type(obj).__name__}; pass StateSpace, "
        "RecurrenceMatrix, WindowedRecurrence objects or a BatchWindowedRecurrence"
    )


def describe_units(units) -> pd.DataFrame:
    """Tabulate the comparability-relevant parameters of a set of units."""
    from .windowed.recurrence import BatchWindowedRecurrence

    if isinstance(units, BatchWindowedRecurrence):
        items = list(units.results.items())
    elif isinstance(units, Mapping):
        items = list(units.items())
    elif isinstance(units, Sequence):
        items = [(getattr(u, "subject", "") or f"unit{i:03d}", u)
                 for i, u in enumerate(units)]
    else:
        raise ParameterError("units must be a mapping, a sequence or a batch")
    if not items:
        raise ParameterError("no units to compare")
    return pd.DataFrame([_describe_unit(u, str(k)) for k, u in items])


# ------------------------------------------------------------------ check
def _values_agree(values, tolerance: float) -> bool:
    vals = [v for v in values if v is not None and not (
        isinstance(v, float) and np.isnan(v))]
    if not vals:
        return True
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        lo, hi = min(vals), max(vals)
        if lo == hi:
            return True
        scale = abs(hi) or 1.0
        return (hi - lo) / scale <= tolerance
    return len(set(map(str, vals))) == 1


def check_comparability(units, contract: ComparabilityContract | None = None,
                        *, axes: Iterable[str] | None = None) -> ComparabilityReport:
    """Verify that a set of analyses can legitimately be compared.

    ``units`` may be a mapping of label to object, a sequence, or a
    :class:`BatchWindowedRecurrence`. Without a contract, every axis present is
    checked for constancy, which answers "are these comparable at all?"; with
    one, the declared axes are checked against their required values.
    """
    df = describe_units(units)
    contract = contract or ComparabilityContract(metric=None, theiler=None)
    tol = contract.tolerance
    required = contract.declared()
    to_check = list(axes) if axes is not None else [a for a in AXES if a in df.columns]

    results: list[AxisResult] = []
    for axis in to_check:
        if axis not in df.columns:
            results.append(AxisResult(axis, required.get(axis), (), "absent",
                                      message="not recorded by these units"))
            continue
        col = df[axis]
        observed = tuple(sorted({str(v) for v in col if v is not None
                                 and not (isinstance(v, float) and np.isnan(v))}))
        if not observed:
            results.append(AxisResult(axis, required.get(axis), (), "absent",
                                      message="no unit reports this axis"))
            continue

        if axis in required:
            want = required[axis]
            bad = []
            for lbl, val in zip(df["label"], col, strict=False):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                if isinstance(want, (int, float)) and not isinstance(want, bool) \
                        and isinstance(val, (int, float)):
                    scale = abs(want) or 1.0
                    if abs(val - want) / scale > tol:
                        bad.append(f"{lbl}={val}")
                elif str(val) != str(want):
                    bad.append(f"{lbl}={val}")
            if bad:
                results.append(AxisResult(
                    axis, want, observed, "violated", tuple(bad),
                    f"required {want}, found {len(observed)} distinct value(s)"))
            else:
                results.append(AxisResult(axis, want, observed, "ok",
                                          message=f"all units at {want}"))
        else:
            if _values_agree(list(col), tol):
                results.append(AxisResult(axis, None, observed, "ok",
                                          message=f"constant at {observed[0]}"))
            else:
                counts = col.astype(str).value_counts()
                minority = counts.index[1:].tolist()
                bad = tuple(f"{lab}={v}" for lab, v in zip(df["label"], col, strict=False)
                            if str(v) in minority)
                results.append(AxisResult(
                    axis, None, observed, "violated", bad[:12],
                    f"varies across units: {len(observed)} distinct values"))

    report = ComparabilityReport(axes=results, n_units=len(df),
                                 labels=tuple(df["label"]), contract=contract)
    if contract.strict:
        report.raise_if_violated()
    return report


def contract_from(unit, *, axes: Iterable[str] | None = None,
                  **overrides) -> ComparabilityContract:
    """Derive a contract from a reference analysis, to apply to the rest."""
    d = _describe_unit(unit, "reference")
    keep = set(axes) if axes is not None else set(AXES)
    fields = {k: v for k, v in d.items()
              if k in keep and k in AXES and v is not None}
    fields.update(overrides)
    return ComparabilityContract(**fields)


# ------------------------------------------------------------ group tests
def feature_matrix(metrics: pd.DataFrame, value: str = "recurrence_rate", *,
                   index: str = "subject", columns: str = "window") -> pd.DataFrame:
    """Pivot a tidy metrics table into subject-by-window form.

    This is the shape a classifier wants: one row per subject, one column per
    window, ready to be concatenated across metrics.
    """
    for col in (index, columns, value):
        if col not in metrics.columns:
            raise ParameterError(f"no column {col!r}; have {sorted(metrics.columns)}")
    return metrics.pivot_table(index=index, columns=columns, values=value)


def group_comparison(metrics: pd.DataFrame, group: str, *,
                     values: Sequence[str] | None = None, by: str = "window",
                     min_per_group: int = 2, test: str = "mannwhitney"
                     ) -> pd.DataFrame:
    """Compare two groups within each window index [DD-44].

    Windows are compared like with like: every subject's window 0 against every
    other subject's window 0, and so on, because window *k* only means the same
    thing across subjects when the window grid is shared. Pass ``by=None`` to
    pool all windows instead.

    Reports, per window and metric: group sizes, means, the difference,
    Cohen's d, the rank-based AUC (probability a random member of group B
    exceeds one of group A) and an uncorrected two-sided p-value. Multiple
    comparisons across windows are the caller's responsibility -- the number of
    tests is in the table.
    """
    if group not in metrics.columns:
        raise ParameterError(f"no group column {group!r}")
    groups = [g for g in metrics[group].dropna().unique()]
    if len(groups) != 2:
        raise ParameterError(
            f"group_comparison needs exactly 2 groups, found {len(groups)}: {groups}"
        )
    a_name, b_name = sorted(groups, key=str)

    skip = {group, "subject", "window", "window_label", "kind", "scope", "metric",
            "threshold_mode", "start_sample", "stop_sample", "n_samples",
            "t_start_s", "t_stop_s", "t_center_s", "n_rows", "n_cols", "dim",
            "theiler"}
    if values is None:
        values = [c for c in metrics.columns
                  if c not in skip and pd.api.types.is_numeric_dtype(metrics[c])]
    if not values:
        raise ParameterError("no numeric metric columns to compare")

    rows = []
    groupings = (metrics.groupby(by) if by else [(None, metrics)])
    for key, chunk in groupings:
        for value in values:
            a = chunk.loc[chunk[group] == a_name, value].dropna().to_numpy(float)
            b = chunk.loc[chunk[group] == b_name, value].dropna().to_numpy(float)
            if a.size < min_per_group or b.size < min_per_group:
                continue
            pooled = np.sqrt(((a.size - 1) * a.var(ddof=1)
                              + (b.size - 1) * b.var(ddof=1))
                             / max(a.size + b.size - 2, 1))
            d = (b.mean() - a.mean()) / pooled if pooled > 0 else np.nan
            auc, p = np.nan, np.nan
            if test == "mannwhitney":
                from scipy.stats import mannwhitneyu

                try:
                    u, p = mannwhitneyu(b, a, alternative="two-sided")
                    auc = float(u / (a.size * b.size))
                except ValueError:
                    pass
            rows.append({
                (by or "scope"): key, "metric": value,
                "group_a": a_name, "group_b": b_name,
                "n_a": int(a.size), "n_b": int(b.size),
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "std_a": float(a.std(ddof=1)), "std_b": float(b.std(ddof=1)),
                "difference": float(b.mean() - a.mean()),
                "cohens_d": float(d), "auc": auc, "p_value": p,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["n_tests"] = len(out)
        out["p_bonferroni"] = (out["p_value"] * len(out)).clip(upper=1.0)
    return out


def rank_metrics(comparison: pd.DataFrame, *, by: str = "auc") -> pd.DataFrame:
    """Rank metrics by how well they separate the groups, averaged over windows."""
    if comparison.empty:
        return comparison
    col = "auc" if by == "auc" else "cohens_d"
    agg = comparison.groupby("metric").agg(
        mean_effect=(col, lambda s: float(np.nanmean(np.abs(s - 0.5) + 0.5))
                     if col == "auc" else float(np.nanmean(np.abs(s)))),
        max_effect=(col, lambda s: float(np.nanmax(np.abs(s - 0.5) + 0.5))
                    if col == "auc" else float(np.nanmax(np.abs(s)))),
        min_p=("p_value", "min"), n_windows=("metric", "size"),
    ).reset_index()
    return agg.sort_values("mean_effect", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time-resolved separation
# ---------------------------------------------------------------------------
def _auc_curves(values: np.ndarray, is_b: np.ndarray) -> tuple[np.ndarray, ...]:
    """AUC per column of a (units x times) matrix with NaN for missing.

    AUC is P(B > A) from the rank sum, ties shared, computed per column over
    the units present in that column.
    """
    from scipy.stats import rankdata

    n_units, n_times = values.shape
    auc = np.full(n_times, np.nan)
    n_a = np.zeros(n_times, dtype=int)
    n_b = np.zeros(n_times, dtype=int)
    for t in range(n_times):
        col = values[:, t]
        ok = np.isfinite(col)
        a, b = ok & ~is_b, ok & is_b
        n_a[t], n_b[t] = int(a.sum()), int(b.sum())
        if n_a[t] and n_b[t]:
            ranks = rankdata(col[ok])
            rb = ranks[is_b[ok]].sum()
            u = rb - n_b[t] * (n_b[t] + 1) / 2.0
            auc[t] = u / (n_a[t] * n_b[t])
    return auc, n_a, n_b


def _trend_stat(auc: np.ndarray) -> float:
    """Slope of the AUC curve over time, per time step."""
    ok = np.isfinite(auc)
    if ok.sum() < 3:
        return np.nan
    x = np.arange(auc.size, dtype=float)[ok]
    return float(np.polyfit(x, auc[ok], 1)[0])


def _timecourse_null_hits(chunk, *, grid, is_b_perms, obs_peak, obs_trend,
                          min_per_group):
    """Peak and trend hits over one chunk of precomputed permutations.

    Module level and picklable; the permutations were drawn once before any
    split, so the counts do not depend on how many workers run them [DD-81].
    """
    hit_peak = hit_trend = 0
    for k in chunk:
        a_p, na_p, nb_p = _auc_curves(grid, is_b_perms[k])
        a_p[(na_p < min_per_group) | (nb_p < min_per_group)] = np.nan
        if np.any(np.abs(a_p - 0.5) >= obs_peak - 1e-12):
            hit_peak += 1
        tr = _trend_stat(a_p)
        if np.isfinite(tr) and abs(tr) >= obs_trend - 1e-12:
            hit_trend += 1
    return hit_peak, hit_trend


def group_timecourse(metrics: pd.DataFrame, group: str, *, values: Sequence[str],
                     time: str = "window", unit: str = "subject",
                     by: str | None = None, keep: str | None = None,
                     min_per_group: int = 5, n_permutations: int = 2000,
                     rng: SeedLike = 0,
                     n_jobs: int | None = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does the separation between two groups evolve along the recording?

    Design note (DD-114) -- the unit of inference stays the subject, and the
    peak of a curve is a max-statistic question.

    The windows of a corpus are aligned in time, so at every window index the
    two groups can be compared across subjects: an AUC per window, a curve per
    metric. Two temptations come with it. Treating windows as independent
    samples multiplies the sample size by the window count with rows that are
    strongly correlated within a subject -- pseudo-replication, and the reason
    the aggregated comparison collapses each subject to one number first. And
    reading the best window of a 24-point curve against an uncorrected 0.05
    forgets that 24 correlated looks were taken.

    Both are answered by permuting **subject labels** and recomputing the whole
    curve: the distribution of the curve's extreme |AUC - 1/2| under the null
    gives ``p_peak``, already corrected for taking the best window; the
    distribution of the fitted slope gives ``p_trend`` for a drift of the
    separation along the recording. Rows excluded by a quality mask stay
    excluded under every permutation, so uneven window loss cannot fake an
    effect on its own.

    Parameters
    ----------
    metrics
        One row per unit and time (and ``by`` level), e.g. a windowed metrics
        table.
    group
        Column with the two group labels. AUC is P(second group > first),
        groups ordered alphabetically.
    values
        Metric columns to test.
    time, unit, by, keep
        Time column; unit (subject) column; optional split column such as the
        channel; optional boolean column marking rows to use.
    min_per_group
        Windows with fewer units than this in either group get no AUC.
    n_permutations
        Label shuffles for ``p_peak`` and ``p_trend``. Zero skips inference.
    n_jobs
        Workers for the permutation counting [DD-109]. The default 1 keeps it
        serial, which is right when the caller already parallelises across
        channels; the shuffles are drawn once up front, so the p-values are
        identical at any worker count.

    Returns
    -------
    (timecourse, summary)
        ``timecourse``: one row per (by, metric, time) with ``auc``, ``n_a``,
        ``n_b``. ``summary``: one row per (by, metric) with the extreme window
        and AUC, ``p_peak``, the slope and ``p_trend``.
    """
    for col in ([group, time, unit] + list(values) + ([by] if by else [])
                + ([keep] if keep else [])):
        if col not in metrics.columns:
            raise ParameterError(f"no column {col!r} in the table")
    frame = metrics[metrics[keep].astype(bool)] if keep else metrics
    labels = sorted(frame[group].astype(str).unique())
    if len(labels) != 2:
        raise ParameterError(f"exactly two groups are needed, found {labels}")
    rng_ = resolve_rng(rng)
    times = np.array(sorted(frame[time].unique()))
    t_index = {t: k for k, t in enumerate(times)}

    course_rows, summary_rows = [], []
    for level, part in (frame.groupby(by) if by else [(None, frame)]):
        units = np.array(sorted(part[unit].unique()))
        u_index = {u: k for k, u in enumerate(units)}
        is_b = np.array([str(part[part[unit] == u][group].iloc[0]) == labels[1]
                         for u in units])
        perms = None
        if n_permutations:
            perms = [rng_.permutation(is_b) for _ in range(n_permutations)]
        for metric in values:
            grid = np.full((units.size, times.size), np.nan)
            rows = part[[unit, time, metric]].dropna()
            grid[[u_index[u] for u in rows[unit]],
                 [t_index[t] for t in rows[time]]] = rows[metric].to_numpy(float)
            auc, n_a, n_b = _auc_curves(grid, is_b)
            thin = (n_a < min_per_group) | (n_b < min_per_group)
            auc[thin] = np.nan
            for k, t in enumerate(times):
                course_rows.append({**({by: level} if by else {}),
                                    "metric": metric, time: t,
                                    "auc": auc[k], "n_a": n_a[k], "n_b": n_b[k]})
            finite = np.isfinite(auc)
            summary = {**({by: level} if by else {}), "metric": metric,
                       "groups": f"{labels[0]}|{labels[1]}",
                       "n_windows_tested": int(finite.sum())}
            if finite.any():
                peak_k = int(np.nanargmax(np.abs(auc - 0.5)))
                summary.update({"auc_peak": float(auc[peak_k]),
                                "peak_" + time: times[peak_k],
                                "trend_per_window": _trend_stat(auc)})
                if perms is not None:
                    obs_peak = abs(auc[peak_k] - 0.5)
                    obs_trend = abs(summary["trend_per_window"])
                    from .parallel.runner import parallel_map, resolve_jobs
                    workers = max(1, min(resolve_jobs(n_jobs), len(perms)))
                    chunks = [list(range(i, len(perms), workers))
                              for i in range(workers)]
                    counts = parallel_map(
                        _timecourse_null_hits, chunks, n_jobs=workers,
                        backend="process", pass_rng=False, grid=grid,
                        is_b_perms=perms, obs_peak=obs_peak,
                        obs_trend=obs_trend, min_per_group=min_per_group)
                    hit_peak = 1 + sum(c[0] for c in counts)
                    hit_trend = 1 + sum(c[1] for c in counts)
                    summary["p_peak"] = hit_peak / (n_permutations + 1)
                    summary["p_trend"] = hit_trend / (n_permutations + 1)
            summary_rows.append(summary)
    return pd.DataFrame(course_rows), pd.DataFrame(summary_rows)
