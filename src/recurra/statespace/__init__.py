"""State space construction -- the three routes to a phase space."""
from .builders import (
    build,
    channels,
    custom,
    envelope_space,
    freq_amp_space,
    from_bands,
    nested_space,
    pac_space,
    phase_circle,
    ppa_space,
    ppc_space,
    takens,
    takens_multivariate,
)
from .compare import (
    compare_routes,
    geometry_descriptors,
    lambda_sweep,
    neighbour_agreement,
    shape_descriptors,
)
from .core import CoordGroup, StateSpace
from .embedding import (
    EmbeddingParams,
    autocorrelation,
    estimate_embedding,
    estimate_embedding_multi,
    estimate_m,
    estimate_tau,
    fnn_fraction,
    mutual_information,
)

__all__ = [
    "StateSpace", "CoordGroup",
    "build", "from_bands",
    "phase_circle", "pac_space", "ppc_space", "envelope_space", "ppa_space",
    "nested_space", "freq_amp_space", "channels", "custom",
    "takens", "takens_multivariate",
    "estimate_tau", "estimate_m", "estimate_embedding", "estimate_embedding_multi",
    "EmbeddingParams", "autocorrelation", "mutual_information", "fnn_fraction",
    "compare_routes", "lambda_sweep", "geometry_descriptors", "shape_descriptors", "neighbour_agreement",
]
