"""Single figures for recurrence quantification [DD-31]."""
from __future__ import annotations

import numpy as np

from ..exceptions import ParameterError
from .style import DEFAULT_DPI, PALETTE, apply_style, lazy_mpl


def plot_line_histogram(hist, *, kinds=("diagonal", "vertical", "white_vertical"),
                        l_min: int = 2, log: bool = True, figsize=None,
                        dpi: int = DEFAULT_DPI, title: str | None = None):
    """The three line-length distributions every RQA metric is derived from.

    Drawn on log axes by default, because the distributions span orders of
    magnitude and it is their tail that carries Lmax, ENTR and the recurrence
    times. The cut at ``l_min`` is marked: everything to its left is discarded
    before DET, LAM and the entropies are computed.
    """
    plt = lazy_mpl()
    kinds = [k for k in kinds if getattr(hist, k, None) is not None]
    if not kinds:
        raise ParameterError("no line-length distribution to plot")

    fig, axes = plt.subplots(1, len(kinds), dpi=dpi, squeeze=False,
                             figsize=figsize or (4.2 * len(kinds), 3.4))
    labels = {"diagonal": "diagonal lines", "vertical": "vertical lines",
              "white_vertical": "white vertical lines (recurrence times)"}
    for ax, kind in zip(axes[0], kinds, strict=False):
        h = getattr(hist, kind)
        nz = np.flatnonzero(h)
        if nz.size:
            ax.bar(nz, h[nz], width=1.0, color=PALETTE["line"])
        ax.axvline(l_min - 0.5, color=PALETTE["accent"], ls="--", lw=1.1,
                   label=f"cut at {l_min}")
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("line length")
        ax.set_ylabel("count")
        ax.set_title(labels.get(kind, kind), fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        apply_style(ax)
    fig.suptitle(title or (f"Line-length distributions · "
                           f"RR = {hist.recurrence_rate:.3%} · "
                           f"Theiler {hist.theiler}"), fontsize=10)
    fig.tight_layout()
    return fig


def plot_diagonal_profile(profile, *, in_seconds: bool = True, figsize=None,
                          dpi: int = DEFAULT_DPI, title: str | None = None,
                          min_cells: int = 10):
    """Recurrence density along each diagonal offset.

    For a cross recurrence plot this is where lag and direction live: the peak
    marks the delay at which one trajectory best matches the other, and any
    asymmetry about zero says which of the two leads.
    """
    plt = lazy_mpl()
    df = profile[profile["n_cells"] >= min_cells]
    if df.empty:
        raise ParameterError("no diagonal has enough cells to plot")
    x = df["lag_s"] if (in_seconds and df["lag_s"].notna().all()) else df["offset"]

    fig, ax = plt.subplots(figsize=figsize or (8.2, 3.2), dpi=dpi)
    ax.plot(x, df["density"], lw=1.0, color=PALETTE["line"])
    ax.axvline(0, color=PALETTE["muted"], ls=":", lw=1.0)
    peak = df.loc[df["density"].idxmax()]
    px = peak["lag_s"] if (in_seconds and np.isfinite(peak["lag_s"])) else peak["offset"]
    ax.axvline(px, color=PALETTE["accent"], ls="--", lw=1.2,
               label=f"peak at {px:.4g}"
                     + (" s" if in_seconds and np.isfinite(peak['lag_s']) else ""))
    ax.set_xlabel("lag (s)" if in_seconds and df["lag_s"].notna().all() else "offset")
    ax.set_ylabel("recurrence density")
    ax.legend(fontsize=8, frameon=False)
    apply_style(ax, grid=True)
    ax.set_title(title or "Diagonal profile", fontsize=9)
    fig.tight_layout()
    return fig
