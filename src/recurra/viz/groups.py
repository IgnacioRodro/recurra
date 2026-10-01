"""Group separation and comparability figures [DD-31].

Single figures, each taking one DataFrame or report.
"""
from __future__ import annotations

import numpy as np

from ..exceptions import ParameterError
from .style import DEFAULT_DPI, PALETTE, apply_style, lazy_mpl

GROUP_COLOURS = ("#4361a4", "#c1121f")


def plot_group_comparison(comparison, metric: str | None = None, *,
                          effect: str = "auc", alpha: float = 0.05,
                          figsize=None, dpi: int = DEFAULT_DPI,
                          title: str | None = None):
    """Group means per window, with the effect size underneath.

    Takes the DataFrame from :func:`recurra.group_comparison`. Windows are
    compared like with like [DD-44], so the x axis is the window index and
    every point compares the same stretch of every subject's record.
    """
    plt = lazy_mpl()
    df = comparison
    if metric is not None:
        df = df[df["metric"] == metric]
    elif df["metric"].nunique() > 1:
        raise ParameterError(
            f"several metrics present ({sorted(df['metric'].unique())}); name one"
        )
    if df.empty:
        raise ParameterError("nothing to plot")
    metric = df["metric"].iloc[0]
    x = df["window"].to_numpy() if "window" in df.columns else np.arange(len(df))

    fig, axes = plt.subplots(2, 1, sharex=True, dpi=dpi,
                             figsize=figsize or (8.6, 5.2),
                             gridspec_kw={"height_ratios": [2, 1]})

    a_name, b_name = df["group_a"].iloc[0], df["group_b"].iloc[0]
    for col_mean, col_std, name, colour in (
            ("mean_a", "std_a", a_name, GROUP_COLOURS[0]),
            ("mean_b", "std_b", b_name, GROUP_COLOURS[1])):
        mu = df[col_mean].to_numpy(float)
        sd = df[col_std].to_numpy(float)
        axes[0].plot(x, mu, "o-", lw=1.4, ms=4.5, color=colour, label=str(name))
        axes[0].fill_between(x, mu - sd, mu + sd, color=colour, alpha=0.16)
    axes[0].set_ylabel(metric)
    axes[0].legend(fontsize=8, frameon=False, title="group", title_fontsize=8)
    axes[0].set_title(title or f"{metric} by group, window by window", fontsize=10)
    apply_style(axes[0], grid=True)

    if effect == "auc" and "auc" in df.columns:
        y = df["auc"].to_numpy(float)
        axes[1].axhline(0.5, color=PALETTE["muted"], ls="--", lw=1.0)
        axes[1].set_ylabel("AUC")
        axes[1].set_ylim(0, 1)
    else:
        y = df["cohens_d"].to_numpy(float)
        axes[1].axhline(0.0, color=PALETTE["muted"], ls="--", lw=1.0)
        axes[1].set_ylabel("Cohen's d")

    sig = (df["p_bonferroni"] < alpha).to_numpy() if "p_bonferroni" in df else \
          np.zeros(len(df), dtype=bool)
    axes[1].bar(x[~sig], y[~sig], color=PALETTE["muted"], width=0.7,
                label="not significant")
    axes[1].bar(x[sig], y[sig], color=PALETTE["line"], width=0.7,
                label=f"p < {alpha} (Bonferroni)")
    axes[1].set_xlabel("window")
    axes[1].legend(fontsize=7, frameon=False)
    apply_style(axes[1], grid=True)

    fig.tight_layout()
    return fig


def plot_feature_matrix(matrix, *, groups=None, cmap: str = "viridis",
                        figsize=None, dpi: int = DEFAULT_DPI,
                        title: str | None = None):
    """Subject-by-window heat map: the shape a classifier consumes.

    ``groups`` optionally maps subject to group label; subjects are then
    ordered by group and the boundary is drawn.
    """
    plt = lazy_mpl()
    M = matrix
    if groups is not None:
        order = sorted(M.index, key=lambda s: (str(groups.get(s, "")), str(s)))
        M = M.loc[order]
    fig, ax = plt.subplots(figsize=figsize or (8.2, 4.6), dpi=dpi)
    im = ax.imshow(M.to_numpy(float), aspect="auto", cmap=cmap,
                   interpolation="nearest")
    ax.set_xlabel("window")
    ax.set_ylabel("subject")
    ax.set_yticks(range(len(M.index)))
    ax.set_yticklabels(M.index, fontsize=6)
    if groups is not None:
        labels = [str(groups.get(s, "")) for s in M.index]
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                ax.axhline(i - 0.5, color=GROUP_COLOURS[1], lw=1.6)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title or "Feature matrix", fontsize=10)
    fig.tight_layout()
    return fig


def plot_comparability(report, *, figsize=(11.4, 3.8), dpi: int = DEFAULT_DPI,
                       title: str | None = None):
    """Which construction axes matched, and which metric families that licenses."""
    plt = lazy_mpl()
    axes_res = report.axes
    fig, axs = plt.subplots(1, 2, dpi=dpi, figsize=figsize or (11.4, 3.8),
                            gridspec_kw={"width_ratios": [1, 1]})

    # Four statuses encoded in colour alone is not readable. The legend names
    # them, and an axis that nobody reports is annotated with why, because
    # "absent" and "violated" mean quite different things: the first is a
    # quantity that does not exist for these units, the second one that exists
    # and disagrees [DD-55].
    colours = {"ok": "#2a9d8f", "violated": "#c1121f",
               "unchecked": "#adb5bd", "absent": "#dee2e6"}
    meanings = {"ok": "matches across units",
                "violated": "varies across units",
                "absent": "not reported by these units",
                "unchecked": "not checked"}
    y = np.arange(len(axes_res))
    axs[0].barh(y, [1] * len(axes_res),
                color=[colours[a.status] for a in axes_res], height=0.7)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([a.axis for a in axes_res], fontsize=7)
    axs[0].set_xticks([])
    axs[0].set_xlim(0, 1.75)
    axs[0].invert_yaxis()
    for i, a in enumerate(axes_res):
        if a.status in ("absent", "unchecked"):
            axs[0].text(1.03, i, meanings[a.status], va="center", fontsize=6,
                        color=PALETTE["muted"], style="italic")
    axs[0].set_title(f"construction axes ({report.n_units} units)", fontsize=9)
    present = [s for s in ("ok", "violated", "absent", "unchecked")
               if any(a.status == s for a in axes_res)]
    axs[0].legend(handles=[plt.matplotlib.patches.Patch(facecolor=colours[s],
                                                        label=f"{s}: {meanings[s]}")
                           for s in present],
                  fontsize=6, frameon=False, loc="lower right",
                  bbox_to_anchor=(1.0, -0.02))
    for spine in axs[0].spines.values():
        spine.set_visible(False)

    valid = report.metric_validity()
    y2 = np.arange(len(valid))
    axs[1].barh(y2, [1] * len(valid), height=0.7,
                color=[colours["ok"] if c else colours["violated"]
                       for c in valid["comparable"]])
    axs[1].set_yticks(y2)
    axs[1].set_yticklabels(valid["metric_family"], fontsize=7)
    axs[1].set_xticks([])
    axs[1].set_xlim(0, 1.9)
    axs[1].invert_yaxis()
    # Naming the unmatched requirement turns a red bar into an instruction.
    for i, (_, row) in enumerate(valid.iterrows()):
        if not row["comparable"]:
            axs[1].text(1.03, i, f"needs {row['missing']}", va="center",
                        fontsize=6, color=PALETTE["muted"], style="italic")
    axs[1].legend(handles=[
        plt.matplotlib.patches.Patch(facecolor=colours["ok"], label="comparable"),
        plt.matplotlib.patches.Patch(facecolor=colours["violated"],
                                     label="withheld: a requirement is unmatched")],
        fontsize=6, frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.02))
    axs[1].set_title("metric families licensed", fontsize=9)
    for spine in axs[1].spines.values():
        spine.set_visible(False)

    fig.suptitle(title or ("Comparability: "
                           + ("all declared axes match" if report.ok
                              else f"{len(report.violations)} violation(s)")),
                 fontsize=10)
    fig.tight_layout()
    return fig
