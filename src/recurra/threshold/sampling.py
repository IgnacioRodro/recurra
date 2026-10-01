"""Pairwise-distance sampling for threshold estimation.

Design note (DD-32) -- sample pairs from the region the recurrence rate is
actually defined over.

The recurrence rate is the fraction of recurrent pairs *within the region
considered*, which excludes the Theiler band when one is used. Estimating a
threshold from pairs drawn uniformly over all (i, j) -- including the
near-diagonal pairs that will later be discarded -- biases the threshold
downwards, because near-diagonal pairs are systematically closer than
average. So the sampler honours the Theiler window, and refuses to pretend
it is sampling something it is not.

The old draft also concatenated subjects before sampling, which mixes
within-subject and between-subject distances and inflates the threshold.
Cross-record pairs are excluded here by construction: each record is
sampled separately and the samples are pooled.
"""
from __future__ import annotations

import numpy as np

from ..config import SeedLike, resolve_rng
from ..exceptions import ParameterError


def sample_pair_distances(X: np.ndarray, *, n_samples: int = 200_000,
                          theiler: int = 0, rng: SeedLike = None,
                          metric: str = "euclidean",
                          max_attempts: int = 8) -> np.ndarray:
    """Distances between random pairs of rows of X, excluding the Theiler band.

    The full N x N matrix is never formed. Pairs with |i - j| < theiler are
    rejected and redrawn, so the sample matches the region over which the
    recurrence rate is defined [DD-32, DD-115]. With ``theiler=0`` nothing is
    rejected, self-pairs included, exactly as the matrix keeps its line of
    identity.
    """
    from ..recurrence.distance import pairwise_rows

    rng = resolve_rng(rng)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    if n < 2:
        return np.array([], dtype=float)

    # Ordered pairs (i, j) with |i - j| >= theiler. Ordered, because i and j
    # are drawn independently below.
    t = min(theiler - 1, n - 1)
    excluded = (n + 2 * sum(n - k for k in range(1, t + 1))) if theiler > 0 else 0
    n_valid = n * n - excluded
    if n_valid <= 0:
        raise ParameterError(
            f"theiler={theiler} excludes every pair of a {n}-point trajectory"
        )
    k = int(min(n_samples, max(1, n_valid)))

    out = []
    got = 0
    for _ in range(max_attempts):
        need = k - got
        if need <= 0:
            break
        draw = int(need * 1.6) + 32
        i = rng.integers(0, n, size=draw)
        j = rng.integers(0, n, size=draw)
        ok = np.abs(i.astype(np.int64) - j.astype(np.int64)) >= theiler
        i, j = i[ok][:need], j[ok][:need]
        if i.size:
            out.append(pairwise_rows(X, i, j, metric=metric))
            got += i.size
    if not out:
        return np.array([], dtype=float)
    return np.concatenate(out)


def sample_cross_distances(X1: np.ndarray, X2: np.ndarray, *,
                           n_samples: int = 200_000, rng: SeedLike = None,
                           metric: str = "euclidean") -> np.ndarray:
    """Distances between random pairs drawn one from each trajectory.

    No Theiler exclusion: in a cross recurrence plot the two trajectories are
    different systems, so index proximity carries no auto-correlation
    artefact to remove.
    """
    from ..recurrence.distance import pairwise_cross_rows

    rng = resolve_rng(rng)
    X1 = np.atleast_2d(np.asarray(X1, dtype=float))
    X2 = np.atleast_2d(np.asarray(X2, dtype=float))
    if X1.shape[1] != X2.shape[1]:
        raise ParameterError(
            f"cross sampling needs equal dimensionality, got {X1.shape[1]} and "
            f"{X2.shape[1]}. Both trajectories must live in the same space."
        )
    n1, n2 = X1.shape[0], X2.shape[0]
    if n1 < 1 or n2 < 1:
        return np.array([], dtype=float)
    k = int(min(n_samples, n1 * n2))
    i = rng.integers(0, n1, size=k)
    j = rng.integers(0, n2, size=k)
    return pairwise_cross_rows(X1, X2, i, j, metric=metric)


def subsample_points(n: int, max_points: int, rng: SeedLike = None) -> np.ndarray:
    """Sorted random subset of point indices, preserving temporal order."""
    rng = resolve_rng(rng)
    if n <= max_points:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_points, replace=False))
