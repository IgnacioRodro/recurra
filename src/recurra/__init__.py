"""recurra -- recurrence analysis over arbitrary state spaces.

Built for multivariate signals and cross-frequency coupling, but the state
space is user-defined: nothing forces the coupling interpretation.

Typical entry points
--------------------
>>> import recurra as rc
>>> rec = rc.ingest(x, fs=500)                      # any of 11 input shapes
>>> rec = rc.filterbank(rec, {"theta": (4, 8), "gamma": (30, 80)})
>>> rec = rc.analytic(rec)                          # phase + envelope
>>> rec.summary()
"""
from . import statespace, viz
from ._version import __version__
from .capabilities import Capability
from .classify import (
    ClassificationReport,
    build_features,
    classify_groups,
    classify_timecourse,
    select_features,
)
from .comparability import (
    ComparabilityContract,
    ComparabilityReport,
    check_comparability,
    contract_from,
    describe_units,
    feature_matrix,
    group_comparison,
    group_timecourse,
    rank_metrics,
)
from .config import available_backends, get_config, resolve_rng, set_config, spawn_rngs
from .core import Recording
from .diagnostics import QualityReport
from .dynamics import (
    BLOCK_INVARIANTS,
    FINE_SCALE_MEASURES,
    SCALAR_MEASURES,
    TRAJECTORY_MEASURES,
    Invariant,
    correlation_dimension,
    correlation_sum,
    detrended_fluctuation,
    divergence_curve,
    dynamics_measures,
    fine_scale_factor,
    higuchi_dimension,
    hjorth_parameters,
    hurst_exponent,
    invariants_from_series,
    k2_entropy,
    katz_dimension,
    lempel_ziv_complexity,
    lyapunov_max,
    permutation_entropy,
    petrosian_dimension,
    sample_entropy,
    samples_per_turn,
    spectral_entropy,
    svd_entropy,
)
from .exceptions import (
    BackendUnavailable,
    CapabilityError,
    CaveatWarning,
    ComparabilityWarning,
    GeometryWarning,
    InferenceWarning,
    IngestError,
    MemoryBudgetError,
    ParameterError,
    ParameterWarning,
    QualityError,
    RecurraError,
    RecurraWarning,
)
from .images import (
    attractor_image,
    export_attractor_images,
    export_recurrence_images,
    recurrence_image,
)
from .io import (
    CorpusIndex,
    CsvExporter,
    export_frame,
    export_recording,
    ingest,
    read_corpus,
)
from .metrics import (
    aac_index,
    comodulogram,
    coupling_indices,
    modulation_index,
    modulogram,
    mvl,
    plv,
    surrogate_significance,
)
from .parallel import parallel_map, resolve_jobs
from .preprocess import (
    Scaler,
    analytic,
    bandpass,
    filterbank,
    hilbert_components,
    notch,
    resample,
    rms_pairwise_distance,
    scale_array,
)
from .provenance import Step, environment_stamp, provenance_frame
from .recurrence import (
    RecurrenceMatrix,
    cross_recurrence_plot,
    joint_recurrence_plot,
    recurrence_plot,
)
from .report import RunReport
from .rqa import (
    METRIC_NAMES,
    LineHistogram,
    crqa,
    diagonal_profile,
    line_histogram,
    line_length_sweep,
    rqa,
)
from .signals import colored_noise, generate_cfc, generate_corpus, lorenz, rossler
from .statespace import (
    StateSpace,
    estimate_embedding,
    estimate_m,
    estimate_tau,
    shape_descriptors,
)
from .threshold import Threshold, cross_threshold, threshold
from .threshold import sampling as threshold_sampling
from .viz import (
    compare_attractors,
    compare_phase_amplitude,
    compare_recurrence,
    compare_routes,
    compare_scale_policies,
    compare_series,
    compare_thresholds,
    compare_window_series,
    list_figures,
    plot_attractor,
    plot_comodulogram,
    plot_comparability,
    plot_density_profile,
    plot_diagonal_profile,
    plot_embedding_diagnostics,
    plot_envelope_pair,
    plot_feature_matrix,
    plot_group_comparison,
    plot_joint_recurrence,
    plot_lambda_sweep,
    plot_line_histogram,
    plot_modulogram,
    plot_phase_amplitude,
    plot_quality,
    plot_recurrence,
    plot_recurrence_panel,
    plot_rqa_summary,
    plot_scale_report,
    plot_signal,
    plot_threshold_diagnostics,
    plot_window_coverage,
    plot_window_panel,
    plot_window_series,
    save_attractor,
    save_figure,
    shared_limits,
)
from .windowed import (
    BatchWindowedRecurrence,
    Window,
    WindowedRecurrence,
    WindowSpec,
    batch_windowed_recurrence,
    meta_recurrence,
    meta_space,
    meta_theiler,
    windowed_cross_recurrence,
    windowed_joint_recurrence,
    windowed_recurrence,
)

__all__ = [
    "__version__",
    # core
    "Recording", "Capability", "QualityReport", "Step",
    # config
    "get_config", "set_config", "available_backends", "resolve_rng", "spawn_rngs",
    # io
    "ingest", "read_corpus", "CorpusIndex",
    # recurrence plots as images
    "recurrence_image", "export_recurrence_images",
    "attractor_image", "export_attractor_images", "export_frame", "export_recording", "CsvExporter",
    "provenance_frame", "environment_stamp",
    # preprocess
    "bandpass", "filterbank", "notch", "resample", "analytic", "hilbert_components",
    "Scaler", "scale_array", "rms_pairwise_distance",
    # signals
    "generate_cfc", "generate_corpus", "colored_noise", "lorenz", "rossler",
    # statespace
    "statespace", "StateSpace", "estimate_tau", "shape_descriptors", "estimate_m", "estimate_embedding",
    # viz
    "viz", "plot_attractor", "save_attractor", "save_figure", "list_figures",
    "plot_embedding_diagnostics", "plot_scale_report", "plot_lambda_sweep",
    "plot_quality", "plot_phase_amplitude", "plot_signal", "plot_envelope_pair",
    "plot_recurrence", "plot_density_profile", "plot_threshold_diagnostics",
    "plot_recurrence_panel", "plot_joint_recurrence", "plot_rqa_summary",
    "plot_window_coverage", "plot_window_series", "plot_window_panel",
    "plot_group_comparison", "plot_feature_matrix", "plot_comparability",
    "plot_line_histogram", "plot_diagonal_profile", "shared_limits",
    "plot_modulogram", "plot_comodulogram",
    # comparison figures: several objects side by side [DD-31]
    "compare_attractors", "compare_recurrence", "compare_routes",
    "compare_thresholds", "compare_series", "compare_window_series",
    "compare_scale_policies", "compare_phase_amplitude",
    # threshold and recurrence
    "threshold", "cross_threshold", "Threshold", "threshold_sampling",
    "recurrence_plot", "cross_recurrence_plot", "joint_recurrence_plot",
    "RecurrenceMatrix",
    # rqa
    "rqa", "crqa", "line_histogram", "LineHistogram", "diagonal_profile",
    "line_length_sweep",
    "METRIC_NAMES",
    # dynamics
    "Invariant", "correlation_dimension", "correlation_sum", "lyapunov_max",
    "divergence_curve", "k2_entropy", "invariants_from_series", "detrended_fluctuation", "hurst_exponent",
    "higuchi_dimension", "permutation_entropy", "sample_entropy",
    "spectral_entropy", "katz_dimension", "petrosian_dimension", "svd_entropy",
    "hjorth_parameters", "lempel_ziv_complexity",
    "dynamics_measures", "SCALAR_MEASURES", "TRAJECTORY_MEASURES",
    "BLOCK_INVARIANTS", "FINE_SCALE_MEASURES", "fine_scale_factor",
    "samples_per_turn",
    # windowed
    "WindowSpec", "Window", "WindowedRecurrence", "windowed_recurrence",
    "meta_space", "meta_recurrence", "meta_theiler",
    "windowed_cross_recurrence", "windowed_joint_recurrence",
    "BatchWindowedRecurrence", "batch_windowed_recurrence",
    # comparability
    "ComparabilityContract", "ComparabilityReport", "check_comparability",
    "contract_from", "describe_units", "feature_matrix", "group_comparison",
    "group_timecourse",
    "rank_metrics",
    # classification
    "classify_groups",
    "classify_timecourse", "ClassificationReport", "build_features",
    "select_features",
    # reporting
    "RunReport",
    # parallelism
    "parallel_map", "resolve_jobs",
    # metrics
    "mvl", "modulogram", "comodulogram", "modulation_index", "plv", "aac_index", "coupling_indices", "surrogate_significance", "surrogate_significance",
    # errors
    "RecurraError", "CapabilityError", "IngestError", "ParameterError",
    # warnings
    "RecurraWarning", "CaveatWarning", "InferenceWarning", "GeometryWarning",
    "ComparabilityWarning", "ParameterWarning",
    "QualityError", "BackendUnavailable", "MemoryBudgetError",
]


def doctor() -> str:
    """Report which optional backends are available in this environment."""
    b = available_backends()
    lines = [f"recurra {__version__}"]
    for k, v in b.items():
        lines.append(f"  {'OK ' if v else '-- '} {k}")
    return "\n".join(lines)
