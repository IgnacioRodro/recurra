"""Single-figure views of recurrence structures [DD-31].

Design note (DD-38) -- downsample by pooling, and watch for saturation.

A recurrence plot is a texture of thin diagonal and vertical lines. Sampling
one pixel in eight to fit a display drops most of those lines and keeps a
speckle that looks like noise, which is precisely the structure the analysis
is about. So a plot too large to draw is reduced by *pooling*: each output
pixel summarises the block it covers.

Maximum pooling preserves lines, but it saturates: at reduction factor f and
recurrence rate r, an output pixel is lit with probability 1 - (1 - r)^(f^2)
if recurrences were independent. For f = 5 and r = 5% that is 72%, and the
picture turns almost solid -- the structure disappears exactly as thoroughly
as with naive subsampling. This was found by looking at the output, not by
reasoning about it.

Real recurrence plots are clustered rather than independent, so the formula
is an upper bound; the measured inflation is smaller but still large (a 5x
reduction of a 5% plot reads as 25% density under maximum pooling).

``pooling="auto"`` therefore estimates that saturation and switches to mean
pooling, which preserves density instead of lines, whenever maximum pooling
would light more than ``SATURATION_LIMIT`` of the pixels. Mean-pooled images
are rescaled to their own upper quantile so contrast survives. Whichever was
used is written into the caption.
"""
from __future__ import annotations

import numpy as np

from ..exceptions import ParameterError
from .style import DEFAULT_DPI, PALETTE, apply_style, lazy_mpl

POOLING = ("auto", "max", "mean", "none")

#: Fraction of lit pixels above which maximum pooling is considered saturated.
SATURATION_LIMIT = 0.35


def expected_saturation(factor: int, rate: float) -> float:
    """Fraction of pixels maximum pooling would light, at this factor and rate."""
    if factor <= 1:
        return float(rate)
    return float(1.0 - (1.0 - rate) ** (factor * factor))


def choose_pooling(factor: int, rate: float | None) -> str:
    """Pick a pooling method that will not destroy the structure."""
    if factor <= 1:
        return "none"
    if rate is None or not np.isfinite(rate):
        return "max"
    return "mean" if expected_saturation(factor, rate) > SATURATION_LIMIT else "max"


def pool_matrix(M: np.ndarray, max_size: int = 1200, method: str = "max"):
    """Reduce a binary matrix for display. Returns (reduced, factor, method)."""
    if method not in POOLING:
        raise ParameterError(f"pooling must be one of {POOLING}")
    n, m = M.shape
    factor = int(max(1, np.ceil(max(n, m) / max_size)))
    if method == "auto":
        method = choose_pooling(factor, float(M.mean()))
    if factor == 1 or method == "none":
        return M.astype(float), 1, "none"
    ny, mx = n // factor * factor, m // factor * factor
    blocks = M[:ny, :mx].astype(float).reshape(ny // factor, factor,
                                               mx // factor, factor)
    reduced = blocks.max(axis=(1, 3)) if method == "max" else blocks.mean(axis=(1, 3))
    return reduced, factor, method


def render_recurrence(rm, *, max_size: int = 1200, pooling: str = "auto"):
    """Materialise a displayable image of a recurrence matrix.

    Streams tiles when the matrix is not stored, so a plot can be drawn even
    for a structure that never exists in memory as an array.

    Returns ``(image, factor, method)``.
    """
    n, m = rm.shape
    factor = int(max(1, np.ceil(max(n, m) / max_size)))
    if pooling == "auto":
        pooling = choose_pooling(factor, rm.recurrence_rate())
    if rm._matrix is not None or rm.store == "memory":
        return pool_matrix(rm.matrix, max_size=max_size, method=pooling)

    # Bin edges are anchored to the *global* grid, not to tile boundaries:
    # aligning them per tile silently shifts the blocks whenever the tile
    # size is not a multiple of the reduction factor.
    out = np.zeros((int(np.ceil(n / factor)), int(np.ceil(m / factor))), dtype=float)
    counts = np.zeros_like(out)
    for tile in rm.tiles():
        data = tile.data.astype(float)
        row_bins = np.arange(tile.i0, tile.i1) // factor
        col_bins = np.arange(tile.j0, tile.j1) // factor
        idx = (row_bins[:, None], col_bins[None, :])
        if pooling == "mean":
            np.add.at(out, idx, data)
            np.add.at(counts, idx, np.ones_like(data))
        else:
            np.maximum.at(out, idx, data)
    if pooling == "mean":
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(counts > 0, out / counts, 0.0)
    return out, factor, ("none" if factor == 1 else pooling)


def plot_recurrence(rm, *, max_size: int = 1200, pooling: str = "auto",
                    cmap: str = "binary", time_axis: bool = True,
                    figsize=None, dpi: int = DEFAULT_DPI, ax=None,
                    title: str | None = None, show_stats: bool = True):
    """Draw one recurrence, cross-recurrence or joint recurrence plot.

    ``pooling="auto"`` (default) reduces oversized matrices with the method
    that will not saturate at this recurrence rate [DD-38].
    """
    plt = lazy_mpl()
    img, factor, used = render_recurrence(rm, max_size=max_size, pooling=pooling)
    vmax = 1.0
    if used == "mean" and img.size:
        vmax = float(max(np.quantile(img, 0.995), 1e-6))

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or (5.6, 5.4), dpi=dpi)
    else:
        fig = ax.figure

    n, m = rm.shape
    if time_axis and rm.fs and rm.fs != 1.0:
        extent = (rm.t0, rm.t0 + m / rm.fs, rm.t0 + n / rm.fs, rm.t0)
        xlabel = ylabel = "time (s)"
    else:
        extent = (0, m, n, 0)
        xlabel = ylabel = "index"

    ax.imshow(img, cmap=cmap, extent=extent, interpolation="nearest",
              aspect="equal" if n == m else "auto", vmin=0, vmax=vmax)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

    if title is None:
        label = {"rp": "Recurrence plot", "crp": "Cross recurrence plot",
                 "jrp": "Joint recurrence plot"}.get(rm.kind, "Recurrence plot")
        title = label
        if show_stats:
            title += (f"\nRR = {rm.recurrence_rate():.3%} · "
                      f"$\\varepsilon$ = {rm.threshold.scalar:.4g} "
                      f"({rm.threshold.mode}) · Theiler {rm.theiler}")
            if factor > 1:
                title += (f"\ndisplayed at 1/{factor} scale, {used} pooling"
                          + (f" (density, rescaled to {vmax:.2f})"
                             if used == "mean" else ""))
    ax.set_title(title, fontsize=9)
    apply_style(ax)
    if created:
        fig.tight_layout()
    return fig


def plot_density_profile(rm, *, n_bins: int = 64, figsize=None,
                         dpi: int = DEFAULT_DPI, title: str | None = None):
    """Recurrence density along the trajectory: where the plot is dense or empty.

    Streams the matrix, so it works for structures too large to materialise.
    A collapse in density marks a stretch the threshold does not suit -- a
    regime change, an artefact, or an amplitude excursion [DD-28].
    """
    plt = lazy_mpl()
    df = rm.density_profile(n_bins=n_bins)
    fig, ax = plt.subplots(figsize=figsize or (8.2, 3.0), dpi=dpi)
    ax.fill_between(df["start_time_s"], df["density"], color=PALETTE["line"],
                    alpha=0.75, step="post")
    rate = rm.recurrence_rate()
    ax.axhline(rate, color=PALETTE["accent"], ls="--", lw=1.2,
               label=f"overall RR = {rate:.3%}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("local recurrence density")
    ax.legend(fontsize=8, frameon=False)
    apply_style(ax, grid=True)
    ax.set_title(title or "Recurrence density profile", fontsize=9)
    fig.tight_layout()
    return fig


def plot_threshold_diagnostics(th, distances=None, *, figsize=None,
                               dpi: int = DEFAULT_DPI, title: str | None = None):
    """Where the threshold sits in the distance distribution.

    Pass the sampled distances to see the histogram; without them the figure
    still reports the mode, the target and the achieved rate.
    """
    plt = lazy_mpl()
    fig, ax = plt.subplots(figsize=figsize or (6.0, 3.6), dpi=dpi)

    if distances is not None and np.size(distances):
        d = np.asarray(distances, dtype=float)
        ax.hist(d, bins=120, color=PALETTE["muted"], edgecolor="none",
                density=True, label="sampled pairwise distances")
        ax.set_xlabel("distance")
        ax.set_ylabel("density")
    else:
        ax.set_xlabel("distance")

    if th.is_pointwise:
        ax.axvline(float(np.median(th.value)), color=PALETTE["accent"], lw=1.4,
                   label=f"median $\\varepsilon$ = {np.median(th.value):.4g}")
        ax.axvspan(float(np.min(th.value)), float(np.max(th.value)),
                   color=PALETTE["accent"], alpha=0.12, label="per-point range")
    else:
        ax.axvline(th.scalar, color=PALETTE["accent"], lw=1.4,
                   label=f"$\\varepsilon$ = {th.scalar:.4g}")
    ax.legend(fontsize=8, frameon=False)
    apply_style(ax, grid=True)

    if title is None:
        title = f"Threshold · mode {th.mode}"
        if th.target is not None:
            title += f" · target RR {th.target:.2%}"
        if th.achieved_rr is not None:
            title += f" · achieved {th.achieved_rr:.2%}"
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig
