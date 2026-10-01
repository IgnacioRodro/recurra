"""Single-figure coupling views: phase-amplitude distribution, envelopes.

Each function takes exactly one object and returns one Figure [DD-31].
"""
from __future__ import annotations

import numpy as np

from ..exceptions import ParameterError
from .style import DEFAULT_DPI, PALETTE, apply_style, lazy_mpl


def plot_phase_amplitude(rec, phase: str | None = None, amplitude: str | None = None,
                         *, n_bins: int = 18, figsize=None, dpi: int = DEFAULT_DPI,
                         title: str | None = None, show_indices: bool = True):
    """Mean amplitude per phase bin: the distribution behind Tort's MI.

    A flat profile at 1/n_bins means no coupling; a peak marks the preferred
    phase. This is the classical view, drawn from a Recording so it can be
    put next to the recurrence-based ones on identical data.
    """
    plt = lazy_mpl()
    from ..metrics.classic import modulation_index, mvl

    ph_name = phase or (rec.by_role("phase") or [None])[0]
    am_name = amplitude or (rec.by_role("amplitude") or [None])[-1]
    if ph_name is None or am_name is None:
        raise ParameterError(
            "plot_phase_amplitude needs a phase and an amplitude channel; "
            f"available roles: {sorted(set(rec.roles.values()))}"
        )

    ph = np.asarray(rec.get(ph_name), dtype=float)
    am = np.asarray(rec.get(am_name), dtype=float)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(np.angle(np.exp(1j * ph)), edges) - 1, 0, n_bins - 1)
    means = np.array([am[idx == k].mean() if (idx == k).any() else 0.0
                      for k in range(n_bins)])
    total = means.sum() or 1.0
    p = means / total

    fig, ax = plt.subplots(figsize=figsize or (5.6, 3.6), dpi=dpi)
    centres = edges[:-1] + np.pi / n_bins
    ax.bar(centres, p, width=2 * np.pi / n_bins * 0.9, color=PALETTE["line"])
    ax.axhline(1.0 / n_bins, color=PALETTE["accent"], ls="--", lw=1.2,
               label="uniform (no coupling)")
    ax.set_xlabel(f"{ph_name} (rad)")
    ax.set_ylabel(f"normalised {am_name}")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(["$-\\pi$", "$-\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])
    ax.legend(fontsize=8, frameon=False)
    apply_style(ax)

    if title is None and show_indices:
        title = (f"Phase-amplitude distribution · "
                 f"MI = {modulation_index(ph, am, n_bins=n_bins):.4f} · "
                 f"MVL = {mvl(ph, am):.3f}")
    ax.set_title(title or "Phase-amplitude distribution", fontsize=9)
    fig.tight_layout()
    return fig


def plot_signal(rec, channels=None, *, seconds: float | None = None,
                start: float = 0.0, figsize=None, dpi: int = DEFAULT_DPI,
                title: str | None = None):
    """Stacked time series for selected channels of one Recording."""
    plt = lazy_mpl()
    names = list(channels) if channels else rec.names[: min(6, rec.n_channels)]
    t = rec.times()
    i0 = int(max(0, (start - rec.t0) * rec.fs))
    i1 = rec.n_samples if seconds is None else min(rec.n_samples,
                                                   i0 + int(seconds * rec.fs))

    fig, axes = plt.subplots(len(names), 1, sharex=True, dpi=dpi,
                             figsize=figsize or (9.0, 1.15 * len(names)),
                             squeeze=False)
    for ax, name in zip(axes[:, 0], names, strict=False):
        ax.plot(t[i0:i1], np.asarray(rec.get(name))[i0:i1], lw=0.7,
                color=PALETTE["line"])
        ax.set_ylabel(f"{name}\n({rec.roles.get(name, 'raw')})", fontsize=7)
        apply_style(ax)
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle(title or f"Signal · {rec.subject} · {rec.fs:g} Hz", fontsize=10)
    fig.tight_layout()
    return fig


def plot_envelope_pair(rec, amplitude_a: str, amplitude_b: str, *,
                       figsize=None, dpi: int = DEFAULT_DPI, max_points: int = 6000,
                       title: str | None = None):
    """Scatter of two envelopes against each other: the AAC view."""
    plt = lazy_mpl()
    from ..metrics.classic import aac_index

    a = np.asarray(rec.get(amplitude_a), dtype=float)
    b = np.asarray(rec.get(amplitude_b), dtype=float)
    sel = np.linspace(0, a.size - 1, min(a.size, max_points)).astype(int)

    fig, ax = plt.subplots(figsize=figsize or (4.8, 4.4), dpi=dpi)
    ax.scatter(a[sel], b[sel], s=2.0, alpha=0.3, color=PALETTE["line"], linewidths=0)
    ax.set_xlabel(amplitude_a)
    ax.set_ylabel(amplitude_b)
    r = aac_index(a, b)
    ax.set_title(title or f"Envelope coupling · Spearman r = {r:.3f}", fontsize=9)
    apply_style(ax, grid=True)
    fig.tight_layout()
    return fig


def plot_modulogram(data, *, phase=None, figsize=None, dpi: int = DEFAULT_DPI,
                    title: str | None = None, n_bins: int = 18, **kwargs):
    """Mean amplitude per phase bin, with the error bars that decide it.

    Accepts the frame :func:`recurra.modulogram` returns, or a phase and an
    amplitude to compute it from. A flat bar chart means no coupling; a peak
    means the envelope is systematically larger at that phase [DD-106].

    The error bars are not decoration. A bin with few samples wanders on its
    own, and a peak inside the bars is not a peak.
    """
    import pandas as pd

    plt = lazy_mpl()
    if phase is not None:
        from ..metrics.classic import modulogram

        data = modulogram(data, phase, n_bins=n_bins) if False else \
            modulogram(data, phase, n_bins=n_bins)
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=figsize or (7.2, 3.4), dpi=dpi)
    width = 2 * np.pi / len(frame)
    ax.bar(frame["phase"], frame["amplitude"], width=width * 0.9,
           yerr=frame["sem"], color=PALETTE["line"], ecolor=PALETTE["muted"],
           capsize=2)
    flat = float(np.nanmean(frame["amplitude"]))
    ax.axhline(flat, color=PALETTE["accent"], ls="--", lw=1.1,
               label="uniform: no coupling")
    ax.set_xlabel("phase (radians)")
    ax.set_ylabel("mean amplitude")
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
    ax.legend(fontsize=8, frameon=False)
    apply_style(ax, grid=True)
    contrast = float(np.nanmax(frame["amplitude"]) - np.nanmin(frame["amplitude"]))
    peak = float(frame.loc[frame["amplitude"].idxmax(), "phase"])
    fig.suptitle(title or f"Modulogram · contrast {contrast:.4f} · "
                          f"peak at {peak:+.2f} rad", fontsize=10)
    fig.tight_layout()
    return fig


def plot_comodulogram(frame, *, figsize=None, dpi: int = DEFAULT_DPI,
                      title: str | None = None, cmap: str = "magma"):
    """Coupling index for every phase band against every amplitude band.

    The map that says which pair couples rather than assuming one.

    **Row and column order is taken from the frame**, not sorted. Band names
    sort alphabetically into alpha, beta, delta, gamma, theta, which is not a
    frequency order and makes the map say nothing about where in the spectrum
    the coupling sits. Pass the rows in the order you want them read.
    """
    plt = lazy_mpl()
    # pivot sorts alphabetically, which on band names puts gamma between delta
    # and theta and makes a frequency map unreadable. The order the caller
    # supplied is kept instead: it is the only one that carries meaning here.
    rows = list(dict.fromkeys(frame["phase_band"]))
    cols = list(dict.fromkeys(frame["amplitude_band"]))
    grid = (frame.pivot(index="phase_band", columns="amplitude_band",
                        values="value")
            .reindex(index=rows, columns=cols))
    fig, ax = plt.subplots(figsize=figsize or (5.6, 4.6), dpi=dpi)
    im = ax.imshow(grid.to_numpy(dtype=float), cmap=cmap, aspect="auto",
                   origin="lower")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(grid.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(grid.index, fontsize=8)
    ax.set_xlabel("amplitude taken from")
    ax.set_ylabel("phase taken from")
    fig.colorbar(im, ax=ax, label=str(frame["index"].iloc[0]))
    fig.suptitle(title or "Comodulogram", fontsize=10)
    fig.tight_layout()
    return fig
