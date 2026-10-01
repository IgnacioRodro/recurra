"""Attractor plots.

Design note (DD-27) -- plot the *weighted* coordinates by default.

The geometry that matters is the one the recurrence engine will see, and
that is the weighted space [DD-21]. Plotting raw coordinates would show a
picture the analysis never uses -- typically one where the amplitude axis
dwarfs the phase circle, exactly the artefact the scaling policy removes.
``weighted=False`` is available for diagnosing the problem.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import SeedLike, output_path
from ..exceptions import ParameterError
from .style import PALETTE, lazy_mpl

_DEF_CMAP = PALETTE["sequential"]
_lazy_mpl = lazy_mpl


def _subsample(X: np.ndarray, max_points: int, rng: SeedLike = None):
    n = X.shape[0]
    if n <= max_points:
        return X, np.arange(n)
    idx = np.linspace(0, n - 1, max_points).astype(int)   # keep temporal order
    return X[idx], idx


#: Clip only when the extremes stretch the range by more than this factor.
CLIP_RATIO = 3.0


def _robust_limits(values: np.ndarray, quantile: float = 0.995,
                   pad: float = 0.05) -> tuple[float, float] | None:
    """Axis limits that show the bulk instead of chasing outliers [DD-50].

    Returns None -- meaning "leave the axis alone" -- unless the extremes
    stretch the range by more than :data:`CLIP_RATIO` beyond the quantile
    span. A Gaussian coordinate spans about 1.4 times its 0.5-99.5% range and
    is left exactly as it is; an instantaneous frequency with phase slips can
    span thirty times its own bulk, and is clipped.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 10:
        return None
    lo, hi = np.quantile(v, [1.0 - quantile, quantile])
    span = float(hi - lo)
    full = float(v.max() - v.min())
    if span <= 0 or full <= 0 or full / span < CLIP_RATIO:
        return None
    margin = pad * span
    return float(lo - margin), float(hi + margin)


def plot_attractor(ss, *, mode: str = "auto", dims: Sequence[int] | None = None,
                   weighted: bool = True, color_by: str = "time",
                   max_points: int = 6000, point_size: float = 1.2,
                   alpha: float = 0.45, line: bool = False, cmap: str = _DEF_CMAP,
                   figsize=None, dpi: int = 130, title: str | None = None,
                   ax=None, rng: SeedLike = None, show_labels: bool = True,
                   clip: float | None = 0.995, axis_limits=None,
                   normalise: str | bool = "auto"):
    """Plot a state space trajectory.

    Parameters
    ----------
    mode
        ``"auto"`` picks 3-D when the space has 3+ dimensions, 2-D otherwise.
        ``"2d"``, ``"3d"``, ``"pairs"`` (scatter-matrix of every dimension
        pair) and ``"time"`` (each coordinate against time) are explicit.
    dims
        Which coordinate columns to use. Defaults to the first 2 or 3.
    color_by
        ``"time"``, ``"none"``, a block label, or an array of length N.
    line
        Draw the trajectory as a connected path instead of a point cloud.
        Useful for smooth, short attractors; unreadable for long noisy ones.
    """
    plt = _lazy_mpl()
    X = np.asarray(ss.weighted_coords if weighted else ss.coords, dtype=float)

    # The absolute scale of a weighted space follows the recording's gain and
    # nothing computed from the space depends on it, so a picture meant to be
    # compared with another must not [DD-105].
    do_normalise = (axis_limits is not None) if normalise == "auto" else bool(normalise)
    scale = 1.0
    if do_normalise:
        from .compare import scale_factor

        scale = scale_factor(ss) if weighted else float(
            np.sqrt(np.mean(np.var(X, axis=0))) or 1.0)
        X = X / scale
    K = X.shape[1]

    if mode == "auto":
        mode = "3d" if K >= 3 else ("2d" if K == 2 else "time")
    if mode not in ("2d", "3d", "pairs", "time", "torus"):
        raise ParameterError(f"unknown mode {mode!r}; use auto/2d/3d/pairs/time/torus")
    if mode == "3d" and K < 3:
        raise ParameterError(f"mode='3d' needs at least 3 dimensions, space has {K}")
    if mode == "2d" and K < 2:
        raise ParameterError(f"mode='2d' needs at least 2 dimensions, space has {K}")

    Xs, idx = _subsample(X, max_points, rng)
    labels = ss.labels

    if isinstance(color_by, np.ndarray):
        c = np.asarray(color_by)[idx]
        clabel = "custom"
    elif color_by == "time":
        c = ss.times()[idx]
        clabel = "time (s)"
    elif color_by in ss.block_labels:
        blk = ss.block(color_by)
        c = blk[idx, 0] if blk.shape[1] == 1 else np.linalg.norm(blk[idx], axis=1)
        clabel = color_by
    else:
        c, clabel = None, ""

    subtitle = (f"{ss.route} | dim={K} | N={ss.n_points} | "
                f"scaling={ss.scaling}" + (" (weighted)" if weighted else " (raw)"))

    # Coordinates such as instantaneous frequency carry rare extreme values --
    # a phase slip sends it to tens of Hz -- and autoscaling to the full range
    # then squashes the other 99% into a smear. Limits default to a high
    # quantile, and the figure says when it has clipped [DD-50].
    # Given explicitly, several figures in separate files share their axes and
    # are comparable at a glance [DD-104]; otherwise each picks its own.
    if axis_limits is not None:
        limits = list(axis_limits)[:K] + [None] * max(0, K - len(axis_limits))
    elif clip is not None:
        limits = [_robust_limits(X[:, k], clip) for k in range(K)]
    else:
        limits = [None] * K
    n_clipped = sum(1 for lim in limits if lim is not None)
    if n_clipped:
        subtitle += f" | axes clipped to {clip:.1%} of points"

    # ---------------------------------------------------------- torus
    if mode == "torus":
        # The natural view of a (phase circle, amplitude) space: radius
        # modulated by amplitude. Under PAC the ring thickens at the
        # preferred phase, which is the toroidal geometry the joint state
        # space was chosen for. Requires one phase_circle block and at least
        # one scalar block.
        circ = [g for g in ss.groups if g.kind == "phase_circle"]
        other = [g for g in ss.groups if g.kind != "phase_circle" and g.dim >= 1]
        if not circ or not other:
            raise ParameterError(
                "mode='torus' needs one phase_circle block and one scalar block; "
                f"this space has blocks {[(g.label, g.kind) for g in ss.groups]}"
            )
        g_c, g_a = circ[0], other[0]
        C = np.asarray(ss.coords)[:, g_c.slice]
        phi = np.arctan2(C[:, 1], C[:, 0])
        a = np.asarray(ss.coords)[:, g_a.start]
        lo, hi = np.quantile(a, [0.01, 0.99])
        a_n = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)
        r = 1.0 + float(kw_radius) * a_n if (kw_radius := 0.85) else 1.0
        xs, ys = r * np.cos(phi), r * np.sin(phi)
        sel = np.linspace(0, xs.size - 1, min(xs.size, max_points)).astype(int)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (5.6, 5.4), dpi=dpi)
            created = True
        else:
            fig, created = ax.figure, False
        sc = ax.scatter(xs[sel], ys[sel], s=point_size, alpha=alpha, c=a[sel],
                        cmap=cmap, linewidths=0)
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), color="0.35", lw=0.8, ls="--", zorder=0)
        ax.set_aspect("equal")
        ax.set_xlabel(f"radius = 1 + {g_a.label} (normalised)", fontsize=8)
        ax.set_ylabel(f"angle = {g_c.label}", fontsize=8)
        ax.set_title(title or subtitle, fontsize=9)
        for ang, lab in [(0, "0"), (np.pi / 2, "pi/2"), (np.pi, "pi"),
                         (-np.pi / 2, "-pi/2")]:
            ax.annotate(lab, (2.05 * np.cos(ang), 2.05 * np.sin(ang)),
                        ha="center", va="center", fontsize=7, color="0.4")
        ax.set_xlim(-2.25, 2.25); ax.set_ylim(-2.25, 2.25)
        ax.set_xticks([]); ax.set_yticks([])
        if created:
            cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.03)
            cb.set_label(g_a.label, fontsize=8)
            fig.tight_layout()
        return fig

    # ---------------------------------------------------------- pairs
    if mode == "pairs":
        k = min(K, 5)
        fig, axes = plt.subplots(k, k, figsize=figsize or (2.1 * k, 2.1 * k), dpi=dpi)
        axes = np.atleast_2d(axes)
        # For a phase_circle block, the histogram of cos or sin is an arcsine
        # U-shape that says nothing; the histogram of the phase itself does
        # [DD-51]. Map each column back to its block to decide.
        col_phase = []
        for g in ss.groups:
            for _ in range(g.dim):
                col_phase.append((g.start, g.kind == "phase_circle"))

        for i in range(k):
            for j in range(k):
                a = axes[i, j]
                if i == j:
                    start, is_circle = col_phase[i]
                    if is_circle:
                        raw = np.asarray(ss.coords)[:, start:start + 2]
                        ang = np.arctan2(raw[:, 1], raw[:, 0])
                        a.hist(ang, bins=48, range=(-np.pi, np.pi), color="0.4")
                        if show_labels:
                            a.set_title("phase", fontsize=6, pad=1)
                    else:
                        a.hist(Xs[:, i], bins=48, color="0.4")
                else:
                    a.scatter(Xs[:, j], Xs[:, i], s=point_size * 0.6, alpha=alpha,
                              c=c, cmap=cmap if c is not None else None, linewidths=0)
                a.set_xticks([]); a.set_yticks([])
                if i == k - 1 and show_labels:
                    a.set_xlabel(labels[j], fontsize=7)
                if j == 0 and show_labels:
                    a.set_ylabel(labels[i], fontsize=7)
        fig.suptitle(title or subtitle, fontsize=9)
        fig.tight_layout()
        return fig

    # ----------------------------------------------------------- time
    if mode == "time":
        fig, axes = plt.subplots(K, 1, figsize=figsize or (8, 1.5 * K), dpi=dpi,
                                 sharex=True)
        axes = np.atleast_1d(axes)
        t = ss.times()[idx]
        for i in range(K):
            axes[i].plot(t, Xs[:, i], lw=0.6, color="0.2")
            axes[i].set_ylabel(labels[i], fontsize=8)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(title or subtitle, fontsize=9)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------ 2d and 3d
    d = list(dims) if dims else list(range(3 if mode == "3d" else 2))
    if max(d) >= K:
        raise ParameterError(f"dims={d} out of range for a {K}-dimensional space")

    created = ax is None
    if mode == "3d":
        if created:
            fig = plt.figure(figsize=figsize or (6.2, 5.4), dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
        if line:
            ax.plot(Xs[:, d[0]], Xs[:, d[1]], Xs[:, d[2]], lw=0.35, color="0.25",
                    alpha=0.8)
        sc = ax.scatter(Xs[:, d[0]], Xs[:, d[1]], Xs[:, d[2]], s=point_size,
                        alpha=alpha, c=c, cmap=cmap if c is not None else None,
                        linewidths=0)
        ax.set_zlabel(labels[d[2]], fontsize=9)
        if limits[d[2]]:
            ax.set_zlim(*limits[d[2]])
        ax.view_init(elev=22, azim=-58)
    else:
        if created:
            fig, ax = plt.subplots(figsize=figsize or (5.8, 5.2), dpi=dpi)
        else:
            fig = ax.figure
        if line:
            ax.plot(Xs[:, d[0]], Xs[:, d[1]], lw=0.35, color="0.25", alpha=0.8)
        sc = ax.scatter(Xs[:, d[0]], Xs[:, d[1]], s=point_size, alpha=alpha, c=c,
                        cmap=cmap if c is not None else None, linewidths=0)
        # 'datalim' lets matplotlib move the limits to keep the aspect square,
        # which silently undoes limits the caller asked for [DD-104]. With
        # explicit limits the box is adjusted instead and the limits stand.
        ax.set_aspect("equal",
                      adjustable="box" if axis_limits is not None else "datalim")

    suffix = " (normalised)" if do_normalise else ""
    ax.set_xlabel(labels[d[0]] + suffix, fontsize=9)
    ax.set_ylabel(labels[d[1]] + suffix, fontsize=9)
    if limits[d[0]]:
        ax.set_xlim(*limits[d[0]])
    if limits[d[1]]:
        ax.set_ylim(*limits[d[1]])
    ax.set_title(title or subtitle, fontsize=9)
    if c is not None and clabel and created:
        cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.10)
        cb.set_label(clabel, fontsize=8)
    if created:
        fig.tight_layout()
    return fig


def save_attractor(ss, filename: str, *, subdir: str | None = None, **kw) -> str:
    """Render and write an attractor plot; returns the path."""
    plt = _lazy_mpl()
    fig = plot_attractor(ss, **kw)
    path = output_path(filename, subdir)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
