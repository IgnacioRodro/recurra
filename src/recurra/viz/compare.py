"""Comparison-mode figures.

Design note (DD-31). Everything here takes a *mapping* of already-built
objects and arranges them into panels. None of these functions generates
data, filters a signal or builds a state space: the caller does that, so a
comparison of synthetic regimes and a comparison of real subjects use the
same code path.

Every panel a comparison figure draws is also available as a standalone
single figure elsewhere in ``recurra.viz``, so nothing is only reachable
through a multi-panel view.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..exceptions import ParameterError
from .attractor import plot_attractor
from .style import DEFAULT_DPI, PALETTE, apply_style, lazy_mpl


def _grid_shape(n: int, ncols: int | None) -> tuple[int, int]:
    ncols = min(ncols or min(3, n), n)
    return int(np.ceil(n / ncols)), ncols


#: Extent ratio above which spaces are considered too dissimilar to share axes.
EXTENT_RATIO_LIMIT = 4.0


def _extents(spaces, names) -> list[float]:
    out = []
    for n in names:
        X = np.asarray(spaces[n].weighted_coords)
        out.append(float(np.max(X.max(axis=0) - X.min(axis=0))))
    return out


def _should_share(spaces, names) -> tuple[bool, str]:
    """Decide whether these spaces may honestly share axes [DD-49]."""
    if len(names) < 2:
        return True, ""
    blocks = {tuple(spaces[n].block_labels) for n in names}
    if len(blocks) > 1:
        return False, ("the spaces have different coordinate blocks "
                       f"({sorted(blocks)}), so their axes are not the same "
                       "quantities")
    ext = _extents(spaces, names)
    lo, hi = min(ext), max(ext)
    if lo > 0 and hi / lo > EXTENT_RATIO_LIMIT:
        return False, (f"the spaces differ in extent by a factor of {hi / lo:.1f}; "
                       "shared axes would shrink the smaller ones to a dot")
    return True, ""


def scale_factor(space) -> float:
    """The overall size of a weighted space, as its root-mean-square extent.

    Design note (DD-105) -- the absolute scale of a weighted space is not
    information, and sharing axes in it shows the recording's gain.

    ``rms_balanced`` sets the weights so that ``w_g * rms_g**2`` is equal
    across blocks, which fixes their *ratio* correctly. The weights are then
    normalised to sum to the number of blocks, so the common value works out
    at ``n / sum(1/rms_j**2)`` -- and that follows the raw amplitude of the
    blocks. A recording measured in a smaller unit produces a smaller space.

    Measured on one signal with only the microvolt gain changed:

    | gain | phase-circle radius | epsilon | RR | DET |
    |---|---|---|---|---|
    | 0.1 | 0.054 | 0.016 | 0.0500 | 0.9072 |
    | 1 | 0.509 | 0.144 | 0.0498 | 0.9060 |
    | 10 | 1.370 | 0.393 | 0.0493 | 0.8993 |
    | 100 | 1.414 | 0.417 | 0.0500 | 0.9033 |

    **Every metric is unaffected** -- epsilon follows the scale and the
    recurrence rate is pinned to its target -- so nothing computed from the
    space depends on this. A *picture* does. On a real corpus epsilon ranged
    over a factor of eight between subjects of the same group, and figures
    drawn on shared absolute axes showed a tiny blob, a ring and a full box
    for three controls whose attractors have the same shape.
    """
    X = np.asarray(space.weighted_coords if hasattr(space, "weighted_coords")
                   else space, dtype=float)
    rms = float(np.sqrt(np.mean(np.var(X, axis=0))))
    return rms if rms > 0 else 1.0


def shared_limits(spaces: Mapping[str, object], names=None,
                  quantiles=(0.002, 0.998), normalise: bool = True
                  ) -> list[tuple[float, float]]:
    """Axis limits that hold for every space in a set.

    Design note (DD-104) -- figures in separate files still have to share their
    axes.

    Two attractors drawn on their own axes are two pictures of two shapes; the
    reader cannot tell whether one trajectory is wider than the other or
    whether matplotlib chose different limits. Within a single comparison
    figure this was already handled [DD-49], and that was where it stopped: a
    caller writing one file per subject had no way to get the same treatment.

    The limits come from the pooled coordinates at a robust quantile, so one
    outlying excursion does not squeeze every panel [DD-50]. Whether sharing
    is *legitimate* is a separate question -- axes should only be shared
    between spaces that are comparable -- and :func:`compare_attractors`
    answers it with ``share_limits="auto"``.

    ``normalise`` divides each space by its own overall scale first [DD-105].
    Without it the limits are in units that depend on the recording's gain,
    and the picture shows how many microvolts the electrode measured rather
    than what the trajectory does. It is on by default because the absolute
    scale is not information: every metric the library computes is invariant
    to it.
    """
    names = list(names if names is not None else spaces)
    if not names:
        raise ParameterError("no space to take limits from")
    arrays = [np.asarray(spaces[n].weighted_coords)
              / (scale_factor(spaces[n]) if normalise else 1.0)
              for n in names]
    k = min(c.shape[1] for c in arrays)
    stacked = np.vstack([c[:, :k] for c in arrays])
    return [tuple(np.quantile(stacked[:, i], list(quantiles))) for i in range(k)]


def compare_attractors(spaces: Mapping[str, object], *, mode: str = "auto",
                       ncols: int | None = None, dpi: int = DEFAULT_DPI,
                       max_points: int = 4000, share_limits: str | bool = "auto",
                       title: str | None = None, panel_size=(4.4, 4.0), **kw):
    """One attractor panel per state space.

    Parameters
    ----------
    spaces
        Mapping ``panel label -> StateSpace``. Build them yourself; this
        function never constructs a state space.
    share_limits
        ``"auto"`` (default) shares axes only when the spaces are built from
        the same coordinate blocks and are of comparable extent [DD-49].
        ``True`` forces sharing, ``False`` lets each panel scale itself.

        Sharing is what makes a comparison honest when the spaces are alike:
        matplotlib's independent autoscaling makes different point clouds look
        the same. But forcing it across spaces in different units does the
        opposite damage -- a Lorenz attractor twenty times larger than a PAC
        space shrinks the latter to a dot.
    """
    plt = lazy_mpl()
    if not spaces:
        raise ParameterError("compare_attractors needs at least one state space")
    names = list(spaces)
    nrows, ncols = _grid_shape(len(names), ncols)
    fig = plt.figure(figsize=(panel_size[0] * ncols, panel_size[1] * nrows), dpi=dpi)

    note = ""
    if share_limits == "auto":
        share_limits, note = _should_share(spaces, names)
        if note:
            note = f"axes not shared: {note}"

    lims = shared_limits(spaces, names) if share_limits else None

    for i, name in enumerate(names):
        ss = spaces[name]
        panel_mode = mode
        if panel_mode == "auto":
            panel_mode = "3d" if ss.dim >= 3 else "2d"
        ax = (fig.add_subplot(nrows, ncols, i + 1, projection="3d")
              if panel_mode == "3d" else fig.add_subplot(nrows, ncols, i + 1))
        # The limits are normalised [DD-105], so the coordinates must be too.
        # Setting normalised limits on raw data was the same mistake in a new
        # place: the panel looked shared and compared different units.
        plot_attractor(ss, mode=panel_mode, ax=ax, max_points=max_points,
                       axis_limits=lims if lims else None,
                       title=f"{name}\n{ss.route} · dim {ss.dim} · {ss.scaling}", **kw)
        if lims and panel_mode == "3d":
            ax.set_box_aspect((1, 1, 0.85))
            ax.view_init(elev=16, azim=-62)
    full_title = title or ""
    if note:
        full_title = (full_title + "\n" if full_title else "") + note
    if full_title:
        fig.suptitle(full_title, fontsize=10)
    fig.tight_layout()
    return fig


def compare_phase_amplitude(recordings: Mapping[str, object], *, phase=None,
                            amplitude=None, n_bins: int = 18, ncols: int | None = None,
                            dpi: int = DEFAULT_DPI, title: str | None = None,
                            share_y: bool = True):
    """One phase-amplitude distribution per recording, on shared axes."""
    plt = lazy_mpl()
    from ..metrics.classic import modulation_index, mvl

    names = list(recordings)
    nrows, ncols = _grid_shape(len(names), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.2 * nrows),
                             dpi=dpi, squeeze=False, sharey=share_y)
    flat = axes.ravel()

    for ax, name in zip(flat, names, strict=False):
        rec = recordings[name]
        ph_name = phase or (rec.by_role("phase") or [None])[0]
        am_name = amplitude or (rec.by_role("amplitude") or [None])[-1]
        ph = np.asarray(rec.get(ph_name), dtype=float)
        am = np.asarray(rec.get(am_name), dtype=float)
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        idx = np.clip(np.digitize(np.angle(np.exp(1j * ph)), edges) - 1, 0, n_bins - 1)
        means = np.array([am[idx == k].mean() if (idx == k).any() else 0.0
                          for k in range(n_bins)])
        p = means / (means.sum() or 1.0)
        ax.bar(edges[:-1] + np.pi / n_bins, p, width=2 * np.pi / n_bins * 0.9,
               color=PALETTE["line"])
        ax.axhline(1.0 / n_bins, color=PALETTE["accent"], ls="--", lw=1.1)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels(["$-\\pi$", "0", "$\\pi$"])
        ax.set_title(f"{name}\nMI = {modulation_index(ph, am, n_bins=n_bins):.4f} · "
                     f"MVL = {mvl(ph, am):.3f}", fontsize=8)
        ax.set_xlabel("phase (rad)")
        apply_style(ax)
    for ax in flat[len(names):]:
        ax.set_visible(False)
    flat[0].set_ylabel("normalised amplitude")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def compare_scale_policies(spaces: Mapping[str, object], *, dpi: int = DEFAULT_DPI,
                           figsize=None, title: str | None = None):
    """Variance share per block across scaling policies, as stacked bars.

    Makes the point of [DD-02] in one picture: without a policy the share
    depends on the amplifier gain; with ``rms_balanced`` it does not.
    """
    plt = lazy_mpl()
    names = list(spaces)
    frames = {n: spaces[n].scale_frame() for n in names}
    blocks = list(frames[names[0]]["block"])

    fig, ax = plt.subplots(figsize=figsize or (max(6.0, 1.3 * len(names)), 3.8), dpi=dpi)
    bottom = np.zeros(len(names))
    cmap = plt.get_cmap("tab20")
    for b, block in enumerate(blocks):
        vals = np.array([float(frames[n].loc[frames[n]["block"] == block,
                                             "variance_share"].iloc[0])
                         if (frames[n]["block"] == block).any() else 0.0
                         for n in names])
        ax.bar(names, vals, bottom=bottom, label=block, color=cmap(b * 2 % 20))
        bottom += vals
    ax.axhline(1.0 / max(len(blocks), 1), color=PALETTE["accent"], ls="--", lw=1.2,
               label="balanced")
    ax.set_ylabel("share of distance variance")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False, ncol=min(4, len(blocks) + 1))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    apply_style(ax)
    ax.set_title(title or "Scaling policies", fontsize=10)
    fig.tight_layout()
    return fig


#: Descriptors that are dimensionless, and so comparable between routes.
SCALE_FREE_METRICS = ("nn_cv", "corr_slope")

#: Descriptors carried in the units of the space, and so only comparable
#: between routes built with the same scaling policy [DD-53].
SCALE_DEPENDENT_METRICS = ("nn_mean", "extent")


def compare_routes(df, *, metrics=None, dpi: int = DEFAULT_DPI, figsize=None,
                   title: str | None = None, show_failures: bool = True):
    """Geometric descriptors per route, from the DataFrame of ``compare_routes``.

    Design note (DD-53). Three things this figure gets wrong if left naive.

    *Scale-dependent descriptors are not comparable across routes.* The mean
    nearest-neighbour distance is expressed in the units of the weighted
    space, so a route built with ``scaling="none"`` sits twenty times higher
    than one built with ``rms_balanced`` for reasons that have nothing to do
    with its geometry. Such descriptors are excluded by default, and when
    requested explicitly across mixed scalings their panel says so.

    *A route that failed must not vanish.* ``compare_routes`` reports failures
    as rows with a status message rather than raising, so a route that does
    not apply to a recording is recorded. Plotting only the successes hides
    the fact that anything was attempted.

    *The most interpretable descriptor was missing.* The fraction of nearest
    neighbours a route shares with the reference answers the question a user
    actually has -- do these constructions see the same geometry? -- and is
    included whenever ``compare_routes`` computed it.
    """
    plt = lazy_mpl()
    ok = df[df.get("status", "ok") == "ok"] if "status" in df else df
    failed = df[df.get("status", "ok") != "ok"] if "status" in df else df.iloc[:0]

    if metrics is None:
        metrics = [m for m in SCALE_FREE_METRICS if m in ok.columns]
        metrics += [c for c in ok.columns if c.startswith("nn_agreement")]
    metrics = [m for m in metrics if m in ok.columns]
    if ok.empty or not metrics:
        raise ParameterError("nothing to plot: no successful routes or no metrics")

    mixed_scaling = ("scaling" in ok.columns and ok["scaling"].nunique() > 1)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=figsize or (4.2 * len(metrics), 3.6),
                             dpi=dpi, squeeze=False)
    y = np.arange(len(ok))
    labels = ok["route_name"] if "route_name" in ok else ok.index.astype(str)
    for ax, metric in zip(axes[0], metrics, strict=False):
        suspect = mixed_scaling and metric in SCALE_DEPENDENT_METRICS
        ax.barh(y, ok[metric].to_numpy(dtype=float),
                color=PALETTE["muted"] if suspect else PALETTE["line"], height=0.55)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(metric + ("\nnot comparable: the routes mix scaling policies"
                                if suspect else ""))
        if metric.startswith("nn_agreement"):
            ax.set_xlim(0, 1)
            ax.set_xlabel("fraction of nearest neighbours shared\nwith "
                          + metric.replace("nn_agreement_vs_", ""))
        apply_style(ax)

    caption = title or "Route geometry"
    if show_failures and not failed.empty:
        names = ", ".join(str(n) for n in failed.get("route_name", []))
        caption += f"\n{len(failed)} route(s) did not apply and are not shown: {names}"
        first = failed.iloc[0]
        reason = str(first.get("status", ""))
        if reason:
            caption += f"\n{reason[:150]}"
    fig.suptitle(caption, fontsize=9)
    fig.tight_layout()
    return fig


def compare_recurrence(matrices: Mapping[str, object], *, ncols: int | None = None,
                       dpi: int = DEFAULT_DPI, max_size: int = 700,
                       pooling: str = "auto", cmap: str = "binary",
                       panel_size=(4.0, 4.2), title: str | None = None, **kw):
    """One recurrence plot per panel, all rendered at the same scale."""
    plt = lazy_mpl()
    from .recurrence import plot_recurrence

    if not matrices:
        raise ParameterError("compare_recurrence needs at least one matrix")
    names = list(matrices)
    nrows, ncols = _grid_shape(len(names), ncols)
    fig, axes = plt.subplots(nrows, ncols, dpi=dpi, squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    flat = axes.ravel()
    for ax, name in zip(flat, names, strict=False):
        rm = matrices[name]
        plot_recurrence(rm, ax=ax, max_size=max_size, pooling=pooling, cmap=cmap,
                        title=f"{name}\nRR = {rm.recurrence_rate():.3%} · "
                              f"$\\varepsilon$ = {rm.threshold.scalar:.3g}", **kw)
    for ax in flat[len(names):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def compare_thresholds(thresholds: Mapping[str, object], *, dpi: int = DEFAULT_DPI,
                       figsize=None, title: str | None = None):
    """Epsilon and achieved recurrence rate across threshold modes."""
    plt = lazy_mpl()
    names = list(thresholds)
    eps = [thresholds[n].scalar for n in names]
    rr = [thresholds[n].achieved_rr for n in names]

    fig, axes = plt.subplots(1, 2, figsize=figsize or (10.0, 3.6), dpi=dpi)
    y = np.arange(len(names))
    axes[0].barh(y, eps, color=PALETTE["line"], height=0.55)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel("$\\varepsilon$")
    axes[0].set_title("threshold value", fontsize=9)
    apply_style(axes[0])

    valid = [(i, v) for i, v in enumerate(rr) if v is not None and np.isfinite(v)]
    if valid:
        idx = [i for i, _ in valid]
        axes[1].barh(idx, [v for _, v in valid], color=PALETTE["line"], height=0.55)
        targets = [thresholds[names[i]].target for i in idx]
        for i, t in zip(idx, targets, strict=False):
            if t is not None:
                axes[1].plot([t], [i], marker="|", ms=16, color=PALETTE["accent"])
    axes[1].set_yticks(y); axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlabel("achieved recurrence rate  (bar = achieved, tick = target)")
    axes[1].set_title("recurrence rate", fontsize=9)
    apply_style(axes[1])

    fig.suptitle(title or "Threshold modes", fontsize=10)
    fig.tight_layout()
    return fig


def compare_window_series(results: Mapping[str, object], column: str = "recurrence_rate",
                          *, dpi: int = DEFAULT_DPI, figsize=None,
                          title: str | None = None):
    """One window series per record, overlaid on a shared time axis.

    Takes a mapping of already-built WindowedRecurrence objects (or their
    metrics frames), so a cohort of subjects and a set of conditions are the
    same call.
    """
    plt = lazy_mpl()
    if not results:
        raise ParameterError("compare_window_series needs at least one result")
    fig, ax = plt.subplots(figsize=figsize or (8.8, 3.8), dpi=dpi)
    cmap = plt.get_cmap("viridis")
    names = list(results)
    for i, name in enumerate(names):
        obj = results[name]
        df = obj.metrics() if hasattr(obj, "metrics") else obj
        if column not in df.columns:
            raise ParameterError(f"{name!r} has no column {column!r}")
        ax.plot(df["t_center_s"], df[column], "o-", lw=1.2, ms=4,
                color=cmap(0.12 + 0.76 * i / max(1, len(names) - 1)), label=name)
    ax.set_xlabel("window centre (s)")
    ax.set_ylabel(column)
    ax.legend(fontsize=8, frameon=False, ncol=min(4, len(names)))
    apply_style(ax, grid=True)
    ax.set_title(title or f"{column} across windows", fontsize=10)
    fig.tight_layout()
    return fig


def compare_series(frames: Mapping[str, object], *, x: str, y: str,
                   dpi: int = DEFAULT_DPI, figsize=None, title: str | None = None,
                   xlabel: str | None = None, ylabel: str | None = None):
    """Overlay one curve per DataFrame. Generic; used for sweeps and profiles."""
    plt = lazy_mpl()
    fig, ax = plt.subplots(figsize=figsize or (5.8, 3.8), dpi=dpi)
    cmap = plt.get_cmap("viridis")
    names = list(frames)
    for i, name in enumerate(names):
        df = frames[name]
        ax.plot(df[x], df[y], "o-", lw=1.2, ms=4,
                color=cmap(0.15 + 0.7 * i / max(1, len(names) - 1)), label=name)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.legend(fontsize=8, frameon=False)
    apply_style(ax, grid=True)
    ax.set_title(title or f"{y} vs {x}", fontsize=10)
    fig.tight_layout()
    return fig
