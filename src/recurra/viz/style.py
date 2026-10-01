"""Shared figure style and saving.

Design note (DD-31) -- single figures vs comparison figures.

Every plotting function in ``recurra.viz`` falls into exactly one of two
kinds, and the distinction is enforced by the signature:

* **Single figures** take *one* object (a Recording, a StateSpace, an
  EmbeddingParams, a DataFrame) and return one Figure. They work identically
  on synthetic and on real data, because they never generate anything.
* **Comparison figures** live in ``recurra.viz.compare`` and take a *mapping*
  of already-built objects. They arrange panels; they still generate nothing.

No plotting function ever creates data, filters a signal or builds a state
space. That keeps every figure reproducible from an object the user already
has, and stops multi-panel demo figures from becoming the only way to get a
particular view.

Design note (DD-30) -- language. All code, docstrings, figure titles, axis
labels, exported file names and CSV column names are in English, regardless
of the language of the surrounding discussion, because the repository is
public and the outputs go into English-language papers.
"""
from __future__ import annotations

from typing import Any

from ..config import output_path

#: Default figure sizes, in inches.
FIGSIZE = {
    "square": (5.8, 5.2),
    "square3d": (6.2, 5.4),
    "wide": (8.4, 3.6),
    "tall": (6.0, 7.0),
}

#: Palette used consistently across the library.
PALETTE = {
    "line": "#2b2b2b",
    "accent": "#c1121f",
    "muted": "#8d99ae",
    "grid": "#d8d8d8",
    "sequential": "viridis",
    "amplitude": "plasma",
    "diverging": "coolwarm",
}

DEFAULT_DPI = 130


def lazy_mpl():
    """Import matplotlib on demand, with a useful error if it is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:  # pragma: no cover
        from ..exceptions import BackendUnavailable

        raise BackendUnavailable("matplotlib", "viz") from e


def apply_style(ax, *, grid: bool = False) -> None:
    """Minimal consistent styling for 2-D axes."""
    for side in ("top", "right"):
        if side in ax.spines:
            ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)
    if grid:
        ax.grid(True, color=PALETTE["grid"], lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)


def save_figure(fig, filename: str, *, subdir: str | None = None,
                close: bool = True, dpi: int | None = None, **kw: Any) -> str:
    """Write a figure to the configured output directory; return the path."""
    plt = lazy_mpl()
    path = output_path(filename, subdir)
    fig.savefig(path, bbox_inches="tight", dpi=dpi, **kw)
    if close:
        plt.close(fig)
    return path


def subtitle_for(ss) -> str:
    """One-line provenance caption used by state-space figures."""
    return (f"route: {ss.route} | dim {ss.dim} | N={ss.n_points} | "
            f"scaling: {ss.scaling}")
