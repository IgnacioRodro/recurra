"""Single-figure diagnostics: embedding curves, scale reports, data quality.

Each function takes exactly one object and returns one Figure [DD-31].
The quality panel reports what it checked, not merely that nothing failed
[DD-52].
"""
from __future__ import annotations

import numpy as np

from .style import DEFAULT_DPI, PALETTE, apply_style, lazy_mpl


def plot_embedding_diagnostics(params, *, figsize=None, dpi: int = DEFAULT_DPI,
                               title: str | None = None):
    """The tau and m curves behind an :class:`EmbeddingParams`.

    This is the figure that justifies the embedding choice in a paper: the
    criterion is drawn, not merely asserted.
    """
    plt = lazy_mpl()
    fig, axes = plt.subplots(1, 2, figsize=figsize or (9.4, 3.4), dpi=dpi)

    if params.tau_lags is not None:
        axes[0].plot(params.tau_lags, params.tau_curve, lw=1.1, color=PALETTE["line"])
        axes[0].axvline(params.tau, color=PALETTE["accent"], ls="--", lw=1.3,
                        label=f"$\\tau$ = {params.tau}")
        axes[0].set_xlabel("lag (samples)")
        axes[0].set_ylabel({"ami": "average mutual information",
                            "acf": "autocorrelation",
                            "first_zero": "autocorrelation"}.get(
                               params.tau_method, params.tau_method))
        axes[0].legend(fontsize=8, frameon=False)
        axes[0].set_title(f"delay selection · {params.tau_method}", fontsize=9)
    else:
        axes[0].text(0.5, 0.5, "no delay curve", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=9, color=PALETTE["muted"])
    apply_style(axes[0], grid=True)

    if params.m_values is not None:
        axes[1].plot(params.m_values, params.m_curve, "o-", lw=1.1, ms=4.5,
                     color=PALETTE["line"])
        axes[1].axvline(params.m, color=PALETTE["accent"], ls="--", lw=1.3,
                        label=f"m = {params.m}")
        axes[1].set_xlabel("embedding dimension $m$")
        axes[1].set_ylabel({"fnn": "false nearest neighbours (fraction)",
                            "cao": "Cao $E_1(m)$"}.get(params.m_method, params.m_method))
        axes[1].legend(fontsize=8, frameon=False)
        axes[1].set_title(f"dimension selection · {params.m_method}", fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "no dimension curve", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=9, color=PALETTE["muted"])
    apply_style(axes[1], grid=True)

    fig.suptitle(title or "Embedding parameter diagnostics", fontsize=10)
    fig.tight_layout()
    return fig


def plot_scale_report(ss, *, figsize=None, dpi: int = DEFAULT_DPI,
                      title: str | None = None):
    """Per-block variance share and tail ratio for one state space [DD-02, DD-28].

    The left panel answers 'is one block dominating the distance?'. The right
    panel answers 'do the blocks have comparable distribution shape?' -- a
    question balancing variance does not settle.
    """
    plt = lazy_mpl()
    df = ss.scale_frame()
    fig, axes = plt.subplots(1, 2, figsize=figsize or (9.4, 3.4), dpi=dpi)

    y = np.arange(len(df))
    axes[0].barh(y, df["variance_share"], color=PALETTE["line"], height=0.55)
    axes[0].axvline(1.0 / len(df), color=PALETTE["accent"], ls="--", lw=1.2,
                    label="balanced")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["block"], fontsize=8)
    axes[0].set_xlabel("share of distance variance")
    axes[0].set_xlim(0, 1)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title(f"variance balance · policy: {ss.scaling}", fontsize=9)
    apply_style(axes[0])

    if "tail_ratio" in df:
        colours = [PALETTE["accent"] if t > 1.6 * df["tail_ratio"].min()
                   else PALETTE["line"] for t in df["tail_ratio"]]
        axes[1].barh(y, df["tail_ratio"], color=colours, height=0.55)
        axes[1].axvline(np.sqrt(2), color=PALETTE["muted"], ls=":", lw=1.2,
                        label="unit circle ($\\sqrt{2}$)")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(df["block"], fontsize=8)
        axes[1].set_xlabel("tail ratio  (p99.9 pairwise distance / RMS)")
        axes[1].legend(fontsize=8, frameon=False)
        axes[1].set_title("distribution shape", fontsize=9)
        apply_style(axes[1])

    fig.suptitle(title or "Scale report", fontsize=10)
    fig.tight_layout()
    return fig


def plot_lambda_sweep(df, *, metrics=("nn_cv", "corr_slope"), figsize=None,
                      dpi: int = DEFAULT_DPI, title: str | None = None):
    """Geometry as the two-block weighting lambda goes from 0 to 1 [DD-02].

    Takes the DataFrame returned by :func:`recurra.statespace.lambda_sweep`.
    Both endpoints are single-block projections and carry no information about
    the relation between the blocks; the interior is where a joint space has
    something to say. The dashed line joins the endpoints, so the interior
    departure from it can be read off. Under coupling the correlation-sum
    slope sits *below* that line: coupling constrains the joint space rather
    than enriching it.
    """
    plt = lazy_mpl()
    metrics = [m for m in metrics if m in df.columns]
    n = max(1, len(metrics))
    fig, axes = plt.subplots(1, n, figsize=figsize or (4.6 * n, 3.4), dpi=dpi,
                             squeeze=False)
    lam = df["lambda"].to_numpy(dtype=float)

    for ax, metric in zip(axes[0], metrics, strict=False):
        y = df[metric].to_numpy(dtype=float)
        ax.plot(lam, y, "o-", lw=1.2, ms=4.5, color=PALETTE["line"], label=metric)
        if np.isfinite(y[0]) and np.isfinite(y[-1]):
            ax.plot([lam[0], lam[-1]], [y[0], y[-1]], ls="--", lw=1.0,
                    color=PALETTE["muted"], label="linear interpolation")
        ax.set_xlabel("$\\lambda$   (0 = first block, 1 = second block)")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8, frameon=False)
        apply_style(ax, grid=True)

    fig.suptitle(title or "Weighting sweep", fontsize=10)
    fig.tight_layout()
    return fig


def plot_quality(rec, *, figsize=None, dpi: int = DEFAULT_DPI,
                 title: str | None = None):
    """Per-channel statistics and flagged issues for one Recording."""
    plt = lazy_mpl()
    df = rec.channel_frame()
    fig, axes = plt.subplots(1, 2, figsize=figsize or (9.6, 3.4), dpi=dpi)

    y = np.arange(len(df))
    axes[0].barh(y, df["std"], color=PALETTE["line"], height=0.55)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["channel"], fontsize=7)
    axes[0].set_xlabel("standard deviation")
    axes[0].set_title("channel scale", fontsize=9)
    apply_style(axes[0])

    # A panel that says only "no issues" reads as one that failed to draw.
    # List the checks and their verdicts instead, so it carries information
    # either way [DD-52].
    CHECKS = {
        "FS_INVALID": "sampling rate",
        "LENGTH_MISMATCH": "equal channel lengths",
        "EMPTY": "non-empty channels",
        "NONFINITE": "finite samples",
        "CONSTANT": "non-constant channels",
        "SATURATION": "no converter clipping",
    }
    found = {}
    if rec.quality is not None:
        for issue in rec.quality.issues:
            prev = found.get(issue.code)
            if prev is None or issue.severity == "error":
                found[issue.code] = issue.severity

    y = np.arange(len(CHECKS))
    colours, marks = [], []
    for code in CHECKS:
        sev = found.get(code)
        colours.append({"error": PALETTE["accent"], "warning": "#e09f3e"}.get(
            sev, "#2a9d8f"))
        marks.append({"error": "failed", "warning": "warning"}.get(sev, "passed"))
    axes[1].barh(y, [1] * len(CHECKS), color=colours, height=0.62)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(list(CHECKS.values()), fontsize=7)
    for i, mark in enumerate(marks):
        axes[1].text(1.02, i, mark, va="center", fontsize=7,
                     color=PALETTE["muted"])
    axes[1].set_xlim(0, 1.35)
    axes[1].set_xticks([])
    axes[1].invert_yaxis()
    n_bad = sum(1 for m in marks if m != "passed")
    axes[1].set_title(
        "quality checks" + (f" -- {n_bad} not passed" if n_bad else " -- all passed"),
        fontsize=9)
    for spine in axes[1].spines.values():
        spine.set_visible(False)

    fig.suptitle(title or f"Data quality · {rec.subject}", fontsize=10)
    fig.tight_layout()
    return fig
