"""Figures.

Two kinds, kept apart on purpose [DD-31]:

* single figures take one object and return one Figure;
* :mod:`recurra.viz.compare` figures take a mapping of already-built objects.

Neither kind generates data, so every figure works the same on synthetic and
on real recordings. All titles, labels and file names are in English [DD-30].
"""
from . import compare
from .attractor import plot_attractor, save_attractor
from .compare import (
    compare_attractors,
    compare_phase_amplitude,
    compare_recurrence,
    compare_routes,
    compare_scale_policies,
    compare_series,
    compare_thresholds,
    compare_window_series,
    shared_limits,
)
from .coupling import (
    plot_comodulogram,
    plot_envelope_pair,
    plot_modulogram,
    plot_phase_amplitude,
    plot_signal,
)
from .diagnostics import (
    plot_embedding_diagnostics,
    plot_lambda_sweep,
    plot_quality,
    plot_scale_report,
)
from .groups import plot_comparability, plot_feature_matrix, plot_group_comparison
from .panels import plot_joint_recurrence, plot_recurrence_panel, plot_rqa_summary
from .recurrence import (
    plot_density_profile,
    plot_recurrence,
    plot_threshold_diagnostics,
    render_recurrence,
)
from .rqa import plot_diagonal_profile, plot_line_histogram
from .style import FIGSIZE, PALETTE, save_figure
from .windowed import plot_window_coverage, plot_window_panel, plot_window_series

#: Registry of single-figure functions: name -> (callable, what it takes).
SINGLE_FIGURES = {
    "attractor": (plot_attractor, "StateSpace"),
    "embedding_diagnostics": (plot_embedding_diagnostics, "EmbeddingParams"),
    "scale_report": (plot_scale_report, "StateSpace"),
    "lambda_sweep": (plot_lambda_sweep, "DataFrame from lambda_sweep()"),
    "quality": (plot_quality, "Recording"),
    "phase_amplitude": (plot_phase_amplitude, "Recording"),
    "signal": (plot_signal, "Recording"),
    "envelope_pair": (plot_envelope_pair, "Recording"),
    "recurrence": (plot_recurrence, "RecurrenceMatrix"),
    "recurrence_panel": (plot_recurrence_panel, "RecurrenceMatrix"),
    "joint_recurrence": (plot_joint_recurrence, "RecurrenceMatrix (joint)"),
    "rqa_summary": (plot_rqa_summary, "RecurrenceMatrix"),
    "density_profile": (plot_density_profile, "RecurrenceMatrix"),
    "threshold_diagnostics": (plot_threshold_diagnostics, "Threshold"),
    "window_coverage": (plot_window_coverage, "WindowedRecurrence"),
    "window_series": (plot_window_series, "WindowedRecurrence"),
    "window_panel": (plot_window_panel, "WindowedRecurrence"),
    "group_comparison": (plot_group_comparison, "DataFrame from group_comparison()"),
    "feature_matrix": (plot_feature_matrix, "DataFrame from feature_matrix()"),
    "comparability": (plot_comparability, "ComparabilityReport"),
    "line_histogram": (plot_line_histogram, "LineHistogram"),
    "diagonal_profile": (plot_diagonal_profile, "DataFrame from diagonal_profile()"),
}

#: Registry of comparison-mode figures: name -> (callable, what it takes).
COMPARISON_FIGURES = {
    "attractors": (compare.compare_attractors, "Mapping[str, StateSpace]"),
    "phase_amplitude": (compare.compare_phase_amplitude, "Mapping[str, Recording]"),
    "scale_policies": (compare.compare_scale_policies, "Mapping[str, StateSpace]"),
    "routes": (compare.compare_routes, "DataFrame from compare_routes()"),
    "series": (compare.compare_series, "Mapping[str, DataFrame]"),
    "recurrence": (compare.compare_recurrence, "Mapping[str, RecurrenceMatrix]"),
    "thresholds": (compare.compare_thresholds, "Mapping[str, Threshold]"),
    "window_series": (compare.compare_window_series,
                      "Mapping[str, WindowedRecurrence]"),
}


def list_figures() -> str:
    """Human-readable catalogue of every figure the library can draw."""
    lines = ["single figures  (one object -> one figure):"]
    for name, (_, takes) in SINGLE_FIGURES.items():
        lines.append(f"  recurra.viz.plot_{name:<24s} takes {takes}")
    lines.append("")
    lines.append("comparison figures  (mapping of built objects -> panels):")
    for name, (_, takes) in COMPARISON_FIGURES.items():
        lines.append(f"  recurra.viz.compare.compare_{name:<16s} takes {takes}")
    return "\n".join(lines)


__all__ = [
    "plot_attractor", "save_attractor", "save_figure",
    "plot_embedding_diagnostics", "plot_scale_report", "plot_lambda_sweep",
    "plot_quality", "plot_phase_amplitude", "plot_signal", "plot_envelope_pair",
    "plot_recurrence", "plot_density_profile", "plot_threshold_diagnostics",
    "plot_recurrence_panel", "plot_joint_recurrence", "plot_rqa_summary",
    "render_recurrence",
    "plot_window_coverage", "plot_window_series", "plot_window_panel",
    "plot_group_comparison", "plot_feature_matrix", "plot_comparability",
    "plot_line_histogram", "plot_diagonal_profile", "shared_limits",
    "plot_modulogram", "plot_comodulogram",
    "compare_attractors", "compare_recurrence", "compare_routes",
    "compare_thresholds", "compare_series", "compare_window_series",
    "compare_scale_policies", "compare_phase_amplitude",
    "compare", "SINGLE_FIGURES", "COMPARISON_FIGURES", "list_figures",
    "PALETTE", "FIGSIZE",
]
