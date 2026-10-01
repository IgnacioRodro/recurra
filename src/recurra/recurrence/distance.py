"""Distance kernels.

Design note (DD-33) -- squared-norm expansion, not broadcasting.

The old draft built ``Ei[:, None, :] - Ej[None, :, :]``, a temporary of
``block^2 * dim`` floats: 96 MB for a 1000-point block in 12 dimensions,
allocated and discarded for every tile. Euclidean distance is instead

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b

where the cross term is a single matrix product routed to BLAS. That is one
temporary of ``block^2`` floats -- a factor of ``dim`` less memory and
roughly an order of magnitude faster.

Squared distances are carried throughout and compared against a squared
threshold, so no square root is taken on the hot path.

Design note (DD-79) -- single precision is a supported choice, not a silent one.

The distance block exists only to be compared against a threshold. Single
precision resolves about seven significant digits, which is far more than any
recurrence decision needs, and it makes the block **3.2 times faster** --
435 ms against 1373 ms for an 8192-square tile, measured. The memory halves
too, so a tile of a given budget can be twice as wide.

It is not the default. A recurrence matrix is the substrate of every metric
downstream, and a user who has not asked for reduced precision should not get
it. ``precision="single"`` opts in, and the choice is recorded in the
provenance like any other.

The one case where it can change a result is a pair sitting within one part in
10^7 of the threshold, which flips membership. At a 5% recurrence rate on
100 000 points that is a handful of cells out of 10^10, and their effect on any
metric is far below the variation between seeds.
"""
from __future__ import annotations

import numpy as np

from ..exceptions import ParameterError

METRICS = ("euclidean", "chebyshev", "manhattan", "cosine")


def _check_metric(metric: str) -> str:
    if metric not in METRICS:
        raise ParameterError(f"unknown metric {metric!r}; use one of {METRICS}")
    return metric


def sq_distance_block(A: np.ndarray, B: np.ndarray, metric: str = "euclidean",
                      A_sq: np.ndarray | None = None,
                      B_sq: np.ndarray | None = None) -> np.ndarray:
    """Block of pairwise distances between rows of A and rows of B.

    Returns *squared* distances for ``euclidean`` and plain distances for the
    other metrics; :func:`compare_threshold` knows which is which.
    """
    _check_metric(metric)
    if metric == "euclidean":
        if A_sq is None:
            A_sq = np.einsum("ij,ij->i", A, A)
        if B_sq is None:
            B_sq = np.einsum("ij,ij->i", B, B)
        D = A_sq[:, None] + B_sq[None, :]
        D -= 2.0 * (A @ B.T)
        np.maximum(D, 0.0, out=D)          # numerical noise can go negative
        return D
    if metric == "chebyshev":
        return np.max(np.abs(A[:, None, :] - B[None, :, :]), axis=2)
    if metric == "manhattan":
        return np.sum(np.abs(A[:, None, :] - B[None, :, :]), axis=2)
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    return 1.0 - (A @ B.T) / np.outer(na, nb)


def threshold_operand(epsilon, metric: str = "euclidean"):
    """Convert a threshold into the units :func:`sq_distance_block` returns."""
    eps = np.asarray(epsilon, dtype=float)
    return eps * eps if metric == "euclidean" else eps


def to_distance(D: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Convert a block back to true distances (for reporting, not for tests)."""
    return np.sqrt(D) if metric == "euclidean" else D


def pairwise_rows(X: np.ndarray, i: np.ndarray, j: np.ndarray,
                  metric: str = "euclidean") -> np.ndarray:
    """Distances for an explicit list of (i, j) pairs within one trajectory."""
    _check_metric(metric)
    d = X[i] - X[j]
    if metric == "euclidean":
        return np.sqrt(np.einsum("ij,ij->i", d, d))
    if metric == "chebyshev":
        return np.max(np.abs(d), axis=1)
    if metric == "manhattan":
        return np.sum(np.abs(d), axis=1)
    a, b = X[i], X[j]
    na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
    na[na == 0] = 1.0; nb[nb == 0] = 1.0
    return 1.0 - np.einsum("ij,ij->i", a, b) / (na * nb)


def pairwise_cross_rows(X1: np.ndarray, X2: np.ndarray, i: np.ndarray,
                        j: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Distances for explicit pairs taken one from each trajectory."""
    _check_metric(metric)
    d = X1[i] - X2[j]
    if metric == "euclidean":
        return np.sqrt(np.einsum("ij,ij->i", d, d))
    if metric == "chebyshev":
        return np.max(np.abs(d), axis=1)
    if metric == "manhattan":
        return np.sum(np.abs(d), axis=1)
    a, b = X1[i], X2[j]
    na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
    na[na == 0] = 1.0; nb[nb == 0] = 1.0
    return 1.0 - np.einsum("ij,ij->i", a, b) / (na * nb)
