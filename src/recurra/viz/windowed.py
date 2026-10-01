"""Single-figure views of windowed analyses [DD-31].

Each function takes one WindowedRecurrence (or a metrics DataFrame) and
returns one figure.
"""
from __future__ import annotations

from ..exceptions import ParameterError
from .style import DEFAULT_DPI, PALETTE, apply_style, lazy_mpl


def plot_window_coverage(wr, *, figsize=None, dpi: int = DEFAULT_DPI,
                         title: str | None = None, max_windows: int = 60):
    """How the windows tile the record: extent, overlap and any gaps.

    Worth drawing before trusting a windowed result. It shows immediately
    whether the last stretch of the record is analysed, and how much
    consecutive windows share.
    """
    plt = lazy_mpl()
    windows = wr.windows if hasattr(wr, "windows") else list(wr)
    if len(windows) > max_windows:
        raise ParameterError(
            f"{len(windows)} windows is too many to draw legibly; pass a subset "
            f"or raise max_windows (currently {max_windows})"
        )
    fig, ax = plt.subplots(figsize=figsize or (8.6, max(2.4, 0.24 * len(windows))),
                           dpi=dpi)
    for w in windows:
        ax.barh(w.index, w.t_stop - w.t_start, left=w.t_start, height=0.68,
                color=PALETTE["line"], alpha=0.85)
    if len(windows) > 1:
        overlap = windows[0].t_stop - windows[1].t_start
        if overlap > 0:
            for w in windows[1:]:
                ax.barh(w.index, overlap, left=w.t_start, height=0.68,
                        color=PALETTE["accent"], alpha=0.55)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("window")
    ax.invert_yaxis()
    apply_style(ax, grid=True)

    if title is None:
        w0 = windows[0]
        step = windows[1].start - w0.start if len(windows) > 1 else w0.n_samples
        ov = max(0.0, 1.0 - step / w0.n_samples)
        title = (f"Window coverage: {len(windows)} windows of "
                 f"{w0.n_samples / w0.fs:.2f} s, {ov:.0%} overlap"
                 "  (red = shared with the previous window)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_window_series(wr, column: str = "recurrence_rate", *, figsize=None,
                       dpi: int = DEFAULT_DPI, title: str | None = None,
                       show_extent: bool = True, ax=None):
    """One metric across windows, plotted at the window centres.

    With ``scope="global"`` the recurrence rate is an informative series;
    with ``scope="per_window"`` it is flat by construction and the
    interesting series is epsilon [DD-40]. The subtitle says which.
    """
    plt = lazy_mpl()
    df = wr.metrics() if hasattr(wr, "metrics") else wr
    if column not in df.columns:
        raise ParameterError(f"no column {column!r}; have {sorted(df.columns)}")

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or (8.6, 3.2), dpi=dpi)
    else:
        fig = ax.figure

    t = df["t_center_s"].to_numpy(dtype=float)
    y = df[column].to_numpy(dtype=float)
    if show_extent and {"t_start_s", "t_stop_s"} <= set(df.columns):
        for a, b, v in zip(df["t_start_s"], df["t_stop_s"], y, strict=False):
            ax.plot([a, b], [v, v], color=PALETTE["muted"], lw=1.0, alpha=0.55)
    ax.plot(t, y, "o-", lw=1.3, ms=5, color=PALETTE["line"])
    ax.set_xlabel("window centre (s)")
    ax.set_ylabel(column)
    apply_style(ax, grid=True)

    if title is None:
        scope = df["scope"].iloc[0] if "scope" in df.columns else ""
        note = {"per_window": "one threshold per window: the rate is fixed, "
                              "structure is comparable",
                "global": "one threshold for all windows: the rate is the signal"}
        title = f"{column} across windows"
        if scope:
            title += f"\nscope: {scope} -- {note.get(scope, '')}"
    ax.set_title(title, fontsize=9)
    if created:
        fig.tight_layout()
    return fig


def plot_window_panel(wr, columns=("epsilon", "recurrence_rate"), *,
                      figsize=None, dpi: int = DEFAULT_DPI,
                      title: str | None = None):
    """Several window series stacked on a shared time axis."""
    plt = lazy_mpl()
    df = wr.metrics() if hasattr(wr, "metrics") else wr
    columns = [c for c in columns if c in df.columns]
    if not columns:
        raise ParameterError("none of the requested columns exist in the metrics")
    fig, axes = plt.subplots(len(columns), 1, sharex=True, dpi=dpi, squeeze=False,
                             figsize=figsize or (8.6, 2.2 * len(columns)))
    for ax, col in zip(axes[:, 0], columns, strict=False):
        plot_window_series(df, col, ax=ax, title="")
        ax.set_title("")
        ax.set_ylabel(col, fontsize=8)
        ax.set_xlabel("")
    axes[-1, 0].set_xlabel("window centre (s)")
    subject = df["subject"].iloc[0] if "subject" in df.columns else ""
    fig.suptitle(title or f"Windowed metrics{' · ' + subject if subject else ''}",
                 fontsize=10)
    fig.tight_layout()
    return fig
