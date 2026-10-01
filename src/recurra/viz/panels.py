"""Composite recurrence figures [DD-120].

Design note (DD-120) -- a recurrence plot is read against its signals.

A bare matrix says where recurrences are; it does not say what the trajectory
was doing there. The classical presentation, from Eckmann, Kamphorst and
Ruelle (1987) to the CRP Toolbox, puts the signals along the axes so a block,
a gap or a drift in the plot can be traced to the stretch of signal that made
it. The library's logo is that layout: the *x* component along the top in
blue, the *y* component along the left in orange.

Three figures:

* :func:`plot_recurrence_panel` -- the plot with its trajectories in the
  margins, for an RP, a CRP (rows and columns are different trajectories), a
  JRP (one subsystem per margin) or a meta-RP (the metric series in the
  margins), optionally with the RQA measures beside it.
* :func:`plot_joint_recurrence` -- a joint plot decomposed: each subsystem's
  plot, their conjunction, and an overlay coloured by which subsystem recurs.
* :func:`plot_rqa_summary` -- the plot with the two line-length distributions
  that every line-based measure is computed from, the floors marked, and the
  measures themselves.

Like every figure of the library, these draw and return; saving is
:func:`save_figure` [DD-31].
"""
from __future__ import annotations

import numpy as np

from .recurrence import render_recurrence
from .style import DEFAULT_DPI, apply_style, lazy_mpl

#: The logo's colours: x component, y component, recurrence ink, light ground.
BRAND = {"x": "#2E5BFF", "y": "#FF8A3D", "ink": "#141A26", "light": "#F2F4F7"}

KIND_LABEL = {"rp": "Recurrence plot", "crp": "Cross recurrence plot",
              "jrp": "Joint recurrence plot"}


# ----------------------------------------------------------------- helpers
def _ink_cmap(colour: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("recurra_ink", ["#FFFFFF", colour])


def _coordinate_labels(groups, dim: int) -> list[str]:
    labels: list[str] = []
    for g in groups or ():
        for k in range(g.dim):
            if g.kind == "phase_circle":
                labels.append(f"{g.label} {'cos' if k == 0 else 'sin'}")
            elif g.dim > 1:
                labels.append(f"{g.label}[{k}]")
            else:
                labels.append(g.label)
    return labels if len(labels) == dim else [f"x{k}" for k in range(dim)]


def _traces(coords, groups, max_traces: int):
    """Up to ``max_traces`` coordinates, each z-scored, with their labels."""
    X = np.atleast_2d(np.asarray(coords, dtype=float))
    if X.shape[0] < X.shape[1]:
        X = X.T
    labels = _coordinate_labels(groups, X.shape[1])
    k = min(max_traces, X.shape[1])
    Z = X[:, :k]
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    return (Z - Z.mean(axis=0)) / sd, labels[:k], X.shape[1] - k


def _time(n: int, fs: float, t0: float, time_axis: bool):
    # fs = 1 is also the default of a plot built from a bare array, so on its
    # own it does not mean "seconds"; a non-zero origin does (a meta-RP at one
    # window per second starts at the first window centre).
    if time_axis and fs and (fs != 1.0 or t0 != 0.0):
        return t0 + np.arange(n) / fs, "time (s)", (t0, t0 + n / fs)
    return np.arange(n, dtype=float), "index", (0.0, float(n))


def _margin_sources(rm):
    """(row trajectory, row groups, column trajectory, column groups, names)."""
    if rm.kind == "jrp" and getattr(rm, "parts", None):
        a, b = rm.parts[0], rm.parts[1]
        return (a.coords, a.groups, b.coords, b.groups,
                ("subsystem 1", "subsystem 2"))
    if rm.kind == "crp":
        return (rm.coords, rm.groups, rm.coords2, (), ("trajectory 1", "trajectory 2"))
    return rm.coords, rm.groups, rm.coords, rm.groups, ("trajectory", "trajectory")


def _decimate(t, Z, limit: int = 4000):
    step = max(1, int(np.ceil(t.size / limit)))
    return t[::step], Z[::step]


# --------------------------------------------------------------- the panel
def plot_recurrence_panel(rm, *, signals=None, names=None, rqa: bool = False,
                          l_min: int = 8, v_min: int = 8, max_traces: int = 3,
                          max_size: int = 1200, pooling: str = "auto",
                          time_axis: bool = True, title: str | None = None,
                          figsize=None, dpi: int = DEFAULT_DPI):
    """A recurrence, cross or joint recurrence plot with its signals alongside.

    The plot is drawn with the line of identity from bottom left to top right;
    the trajectory of the columns runs along the top (blue) and that of the
    rows along the left (orange), each coordinate z-scored and stacked.

    Parameters
    ----------
    rm
        Any :class:`RecurrenceMatrix`: RP, CRP, JRP or a meta-RP.
    signals
        Optional ``(left, top)`` arrays to draw in the margins instead of the
        state-space coordinates -- the raw signal, say. Each is 1-D or
        (n_points, k) and must match the plot's rows and columns.
    names
        Optional ``(left, top)`` labels for the two margins.
    rqa
        Add a column with the RQA measures at floors ``l_min`` and ``v_min``.
    max_traces
        Coordinates drawn per margin; the rest are counted in the label.
    max_size, pooling
        Display reduction of large matrices, as in :func:`plot_recurrence`.
    """
    plt = lazy_mpl()
    n, m = rm.shape
    rows, rgroups, cols, cgroups, default_names = _margin_sources(rm)
    if signals is not None:
        left, top = signals
        rows, rgroups, cols, cgroups = left, (), top, ()
    names = names or default_names
    t_rows, xlabel, (r0, r1) = _time(n, rm.fs, rm.t0, time_axis)
    t_cols, _, (c0, c1) = _time(m, rm.fs, rm.t0, time_axis)

    img, factor, used = render_recurrence(rm, max_size=max_size, pooling=pooling)
    vmax = float(max(np.quantile(img, 0.995), 1e-6)) if used == "mean" and img.size else 1.0

    widths = [1.15, 4.6] + ([1.9] if rqa else [])
    fig = plt.figure(figsize=figsize or (sum(widths) + 0.6, 6.3), dpi=dpi)
    gs = fig.add_gridspec(2, len(widths), width_ratios=widths, height_ratios=[1.15, 4.6],
                          wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax)
    ax_left = fig.add_subplot(gs[1, 0], sharey=ax)

    ax.imshow(img, origin="lower", cmap=_ink_cmap(BRAND["ink"]),
              extent=(c0, c1, r0, r1), interpolation="nearest",
              aspect="equal" if n == m else "auto", vmin=0, vmax=vmax)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(labelleft=False, labelsize=8)

    Zc, lab_c, more_c = _traces(cols, cgroups, max_traces)
    Zr, lab_r, more_r = _traces(rows, rgroups, max_traces)
    tc, Zc = _decimate(t_cols[: Zc.shape[0]], Zc)
    tr, Zr = _decimate(t_rows[: Zr.shape[0]], Zr)
    # Stacked in the order of their label: the first coordinate on the outside
    # (top of the upper margin, far left of the left one), all equally visible.
    for k in range(Zc.shape[1]):
        ax_top.plot(tc, Zc[:, k] - 3.2 * k, color=BRAND["x"], lw=0.7)
    for k in range(Zr.shape[1]):
        ax_left.plot(Zr[:, k] - 3.2 * (Zr.shape[1] - 1 - k), tr, color=BRAND["y"],
                     lw=0.7)
    ax_top.set_title(f"{names[1]}: " + ", ".join(lab_c)
                     + (f" (+{more_c} more)" if more_c else ""),
                     fontsize=8, color=BRAND["x"], loc="left")
    ax_left.set_ylabel(f"{names[0]}: " + ", ".join(lab_r)
                       + (f" (+{more_r} more)" if more_r else ""),
                       fontsize=8, color=BRAND["y"])
    for a in (ax_top, ax_left):
        a.set_xticks([]) if a is ax_left else a.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            a.spines[side].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_left.tick_params(labelsize=8)

    heading = title or KIND_LABEL.get(rm.kind, "Recurrence plot")
    sub = (f"RR = {rm.recurrence_rate():.2%} · $\\varepsilon$ = {rm.threshold.scalar:.4g} "
           f"({rm.threshold.mode}) · Theiler {rm.theiler}")
    if factor > 1:
        sub += f" · shown at 1/{factor} ({used} pooling)"
    fig.suptitle(f"{heading}\n{sub}", fontsize=10, color=BRAND["ink"])

    if rqa:
        from ..rqa.metrics import rqa as _rqa

        values = _rqa(rm, l_min=l_min, v_min=v_min)
        ax_tab = fig.add_subplot(gs[1, 2])
        ax_tab.axis("off")
        lines = [f"RQA  (l_min={l_min}, v_min={v_min})", ""]
        for key in ("RR", "DET", "L", "Lmax", "ENTR", "LAM", "TT", "Vmax", "RTE"):
            v = values.get(key)
            if v is None:
                continue
            lines.append(f"{key:<6s}{v:>10.3%}" if key in ("RR", "DET", "LAM")
                         else f"{key:<6s}{float(v):>10.3f}")
        ax_tab.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                    family="monospace", fontsize=9, color=BRAND["ink"],
                    transform=ax_tab.transAxes)
    return fig


# --------------------------------------------------------- the joint plot
def plot_joint_recurrence(rm, *, names=None, max_size: int = 800,
                          time_axis: bool = True, figsize=None,
                          dpi: int = DEFAULT_DPI):
    """A joint recurrence plot decomposed into what it is made of.

    Four panels: each subsystem's own recurrence plot, the joint plot (their
    conjunction, each with its own threshold [DD-20]), and an overlay in which
    a cell is orange when only subsystem 1 recurs, blue when only subsystem 2
    does, and dark when both do -- the joint recurrences. The title gives the
    joint rate against the product of the two rates: 1 when the subsystems
    recur independently [DD-54].
    """
    if rm.kind != "jrp" or not getattr(rm, "parts", None):
        from ..exceptions import ParameterError

        raise ParameterError("plot_joint_recurrence needs a joint recurrence plot "
                             "(joint_recurrence_plot or windowed_joint_recurrence)")
    plt = lazy_mpl()
    a, b = rm.parts[0], rm.parts[1]
    names = names or ("subsystem 1", "subsystem 2")
    ia, fa, _ = render_recurrence(a, max_size=max_size, pooling="max")
    ib, _, _ = render_recurrence(b, max_size=max_size, pooling="max")
    ij, _, _ = render_recurrence(rm, max_size=max_size, pooling="max")
    n, m = rm.shape
    _, xlabel, (r0, r1) = _time(n, rm.fs, rm.t0, time_axis)
    _, _, (c0, c1) = _time(m, rm.fs, rm.t0, time_axis)
    extent = (c0, c1, r0, r1)

    from matplotlib.colors import to_rgb

    overlay = np.ones(ia.shape + (3,))
    only_a, only_b, both = (ia > 0) & ~(ib > 0), (ib > 0) & ~(ia > 0), (ij > 0)
    overlay[only_a] = to_rgb(BRAND["y"])
    overlay[only_b] = to_rgb(BRAND["x"])
    overlay[both] = to_rgb(BRAND["ink"])

    fig, axes = plt.subplots(1, 4, figsize=figsize or (17, 4.9), dpi=dpi)
    panels = [(ia, BRAND["y"], f"{names[0]}\nRR = {a.recurrence_rate():.2%}"),
              (ib, BRAND["x"], f"{names[1]}\nRR = {b.recurrence_rate():.2%}"),
              (ij, BRAND["ink"], f"joint (AND)\nJRR = {rm.recurrence_rate():.2%}")]
    for ax, (im, colour, label) in zip(axes[:3], panels, strict=True):
        ax.imshow(im, origin="lower", cmap=_ink_cmap(colour), extent=extent,
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(label, fontsize=9, color=colour if colour != BRAND["ink"]
                     else BRAND["ink"])
    axes[3].imshow(overlay, origin="lower", extent=extent, interpolation="nearest")
    axes[3].set_title("overlay: orange 1 only · blue 2 only · dark both",
                      fontsize=9, color=BRAND["ink"])
    for ax in axes:
        ax.set_xlabel(xlabel, fontsize=9)
        ax.tick_params(labelsize=8)
        apply_style(ax)
    ratio = rm.independence_ratio() if hasattr(rm, "independence_ratio") else np.nan
    fig.suptitle(f"Joint recurrence · JRR / (RR1 x RR2) = {ratio:.2f} "
                 "(1 = independent)" + (f" · shown at 1/{fa}" if fa > 1 else ""),
                 fontsize=10, color=BRAND["ink"])
    fig.tight_layout()
    return fig


# ------------------------------------------------------------ RQA summary
def plot_rqa_summary(rm, *, l_min: int = 8, v_min: int = 8, max_size: int = 1000,
                     time_axis: bool = True, figsize=None, dpi: int = DEFAULT_DPI):
    """The plot, the two line-length distributions and the measures from them.

    Every line-based measure is a sum over these histograms, so they show what
    DET and LAM are made of and where the floors cut them [DD-98]: lines left
    of the dashed floor are not counted.
    """
    from ..rqa.metrics import rqa_from_histogram

    plt = lazy_mpl()
    hist = rm.line_histogram()
    values = rqa_from_histogram(hist, l_min=l_min, v_min=v_min)
    n, m = rm.shape
    img, factor, used = render_recurrence(rm, max_size=max_size, pooling="auto")
    vmax = float(max(np.quantile(img, 0.995), 1e-6)) if used == "mean" and img.size else 1.0
    _, xlabel, (r0, r1) = _time(n, rm.fs, rm.t0, time_axis)
    _, _, (c0, c1) = _time(m, rm.fs, rm.t0, time_axis)

    fig = plt.figure(figsize=figsize or (12.5, 6.2), dpi=dpi)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], height_ratios=[1, 1, 0.9],
                          wspace=0.25, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img, origin="lower", cmap=_ink_cmap(BRAND["ink"]), extent=(c0, c1, r0, r1),
              interpolation="nearest", aspect="equal" if n == m else "auto",
              vmin=0, vmax=vmax)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(xlabel, fontsize=9)
    ax.set_title(f"{KIND_LABEL.get(rm.kind, 'Recurrence plot')} · RR = "
                 f"{rm.recurrence_rate():.2%} · Theiler {rm.theiler}", fontsize=9)
    apply_style(ax)

    for row, (counts, floor, colour, label) in enumerate((
            (hist.diagonal, l_min, BRAND["x"], "diagonal lines (determinism)"),
            (hist.vertical, v_min, BRAND["y"], "vertical lines (laminarity)"))):
        a = fig.add_subplot(gs[row, 1])
        lengths = np.flatnonzero(counts)
        if lengths.size:
            a.bar(lengths, counts[lengths], width=1.0, color=colour, alpha=0.85)
            a.set_yscale("log")
        a.axvline(floor - 0.5, color=BRAND["ink"], ls="--", lw=1)
        a.set_title(f"{label} · floor {floor}", fontsize=9, color=colour)
        a.set_xlabel("line length (samples)", fontsize=8)
        a.set_ylabel("count", fontsize=8)
        a.tick_params(labelsize=8)
        apply_style(a)

    t = fig.add_subplot(gs[2, 1])
    t.axis("off")
    cells = [("RR", "{:.2%}"), ("DET", "{:.2%}"), ("L", "{:.2f}"), ("Lmax", "{:.0f}"),
             ("ENTR", "{:.3f}"), ("LAM", "{:.2%}"), ("TT", "{:.2f}"), ("Vmax", "{:.0f}")]
    text = "    ".join(f"{k} {fmt.format(float(values[k]))}" for k, fmt in cells[:4])
    text += "\n" + "    ".join(f"{k} {fmt.format(float(values[k]))}" for k, fmt in cells[4:])
    t.text(0.0, 0.8, text, family="monospace", fontsize=9, color=BRAND["ink"],
           va="top", transform=t.transAxes)
    return fig
