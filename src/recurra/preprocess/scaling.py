"""Scaling policies -- the heterogeneous-block problem.

Design note (DD-02) -- THE central design decision. See docs/DESIGN_DECISIONS.md.

In a state space such as s(t) = (cos phi, sin phi, A), the phase block has a
*geometrically fixed* extent (chord distance in [0, 2]) while the amplitude
block has arbitrary units. Under the Euclidean metric

    d^2 = 4 sin^2(dphi/2) + dA^2

the block with larger spread dominates. If sigma_A >> 1 the recurrence plot
is the amplitude's; if sigma_A << 1 it is the phase's, which yields high DET
whether or not there is any coupling -- a false positive that looks like a
result.

The default resolves it by equalising mean-square pairwise distance:

    scalar with std sigma : E[(A_i - A_j)^2] = 2 sigma^2   -> RMS = sigma*sqrt(2)
    unit circle, uniform  : E[||s_i - s_j||^2] = 2         -> RMS = sqrt(2)

so dividing the amplitude by its standard deviation matches the two blocks
exactly. It also makes the joint space invariant to a global gain change,
which is required to compare across subjects and electrodes.

Design note (DD-28) -- balancing variance is necessary, not sufficient.

Matching RMS pairwise distance equalises *second moments*. It does not
equalise distribution *shape*. A gamma envelope is right-skewed (skewness
around 1.7, max/sigma around 6.6) while the phase circle is bounded with
max/sigma = 1.41. After balancing, the rare high-amplitude excursions still
sit far from everything else, and under a fixed threshold they become
non-recurrent -- producing empty bands in the recurrence plot that reflect
the tail of the amplitude distribution, not the coupling.

The library therefore reports ``tail_ratio`` and ``skewness`` per block
alongside the variance share, and warns when they differ sharply. The fix,
when it matters, is a shape-normalising per-block transform: ``scale="rank"``
removes skew entirely, ``scale="robust"`` limits the influence of outliers.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..config import SeedLike, resolve_rng
from ..exceptions import ParameterError

ScalingPolicy = Literal[
    "rms_balanced", "empirical_balanced", "weighted",
    "per_block_threshold", "rank", "zscore", "robust", "minmax", "none",
]

#: RMS pairwise distance of a unit circle under a uniform phase distribution.
CIRCLE_RMS_PAIRWISE = np.sqrt(2.0)


def rms_pairwise_distance(X: np.ndarray, *, max_samples: int = 20000,
                          rng: SeedLike = None) -> float:
    """Root-mean-square distance between random pairs of rows of X.

    Estimated by sampling; the full N^2 matrix is never built. For a block
    with independent columns this equals sqrt(2 * sum_k var_k), which is the
    quantity the balancing policies equalise.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] < 2:
        return 0.0
    rng = resolve_rng(rng)
    n = X.shape[0]
    k = int(min(max_samples, n * (n - 1) // 2))
    i = rng.integers(0, n, size=k)
    j = rng.integers(0, n, size=k)
    same = i == j
    j[same] = (j[same] + 1) % n
    d2 = np.sum((X[i] - X[j]) ** 2, axis=1)
    return float(np.sqrt(np.mean(d2)))


def theoretical_rms_pairwise(X: np.ndarray) -> float:
    """Closed form: sqrt(2 * sum of column variances). No sampling needed."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return float(np.sqrt(2.0 * np.sum(np.var(X, axis=0))))


def shape_stats(X: np.ndarray, *, max_samples: int = 20000,
                rng: SeedLike = None) -> dict[str, float]:
    """Distribution shape of a block, beyond its second moment [DD-28].

    ``tail_ratio`` is the 99.9th percentile of *pairwise* distance divided by
    the RMS pairwise distance. Pairwise, not distance-to-centroid: a phase
    circle is a shell whose radius barely varies, so a centroid-based
    dispersion is degenerate for it, whereas pairwise distance is exactly the
    quantity a recurrence threshold is compared against.

    Reference values: a uniform unit circle gives 1.41 (its maximum chord is
    2 and its RMS pairwise distance sqrt(2)); a Gaussian coordinate gives
    about 2.3; a right-skewed envelope can exceed 4.
    """
    from scipy.stats import skew as _skew

    rng = resolve_rng(rng)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    if n < 3:
        return {"tail_ratio": np.nan, "max_ratio": np.nan, "skew": np.nan}
    k = int(min(max_samples, n * (n - 1) // 2))
    i = rng.integers(0, n, size=k)
    j = rng.integers(0, n, size=k)
    same = i == j
    j[same] = (j[same] + 1) % n
    d = np.linalg.norm(X[i] - X[j], axis=1)
    rms = float(np.sqrt(np.mean(d**2))) or 1.0
    return {
        "tail_ratio": float(np.quantile(d, 0.999) / rms),
        "max_ratio": float(d.max() / rms),
        "skew": float(np.mean([_skew(X[:, c]) for c in range(X.shape[1])])),
    }


@dataclass(frozen=True)
class ScaleReport:
    """Per-block contribution to total squared distance. Always exported."""

    labels: tuple[str, ...]
    rms: tuple[float, ...]
    weights: tuple[float, ...]
    variance_share: tuple[float, ...]
    policy: str
    dominant: str | None
    warning: str | None
    tail_ratio: tuple[float, ...] = ()
    skewness: tuple[float, ...] = ()
    shape_warning: str | None = None

    def to_frame(self):
        import pandas as pd

        d = {
            "block": self.labels,
            "rms_pairwise": self.rms,
            "weight": self.weights,
            "variance_share": self.variance_share,
            "policy": self.policy,
        }
        if self.tail_ratio:
            d["tail_ratio"] = self.tail_ratio
            d["skewness"] = self.skewness
        return pd.DataFrame(d)


def scale_array(X, *, policy: str = "zscore", eps: float = 1e-12) -> np.ndarray:
    """Scale a single block of coordinates."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if policy in ("none", None):
        return X
    if policy == "zscore":
        s = X.std(axis=0)
        s = np.where(s < eps, 1.0, s)
        return (X - X.mean(axis=0)) / s
    if policy == "robust":
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) * 1.4826
        mad = np.where(mad < eps, 1.0, mad)
        return (X - med) / mad
    if policy == "minmax":
        lo, hi = X.min(axis=0), X.max(axis=0)
        rng_ = np.where((hi - lo) < eps, 1.0, hi - lo)
        return (X - lo) / rng_
    if policy == "rank":
        from scipy.stats import rankdata

        R = np.apply_along_axis(rankdata, 0, X)
        return (R - 0.5) / R.shape[0]
    raise ParameterError(f"unknown block scaling policy {policy!r}")


class Scaler:
    """Applies a scaling policy across heterogeneous coordinate blocks.

    Blocks are (label, array) pairs. The scaler returns per-block weights
    such that the combined distance behaves as the policy intends, plus a
    report of how much each block actually contributes.
    """

    def __init__(self, policy: str = "rms_balanced", *, lambda_: float | None = None,
                 weights: Sequence[float] | None = None,
                 reference_rms: float = CIRCLE_RMS_PAIRWISE,
                 estimator: str = "theoretical", rng: SeedLike = None,
                 dominance_threshold: float = 0.90, tail_ratio_threshold: float = 1.6):
        self.policy = policy
        self.lambda_ = lambda_
        self.weights_in = weights
        self.reference_rms = reference_rms
        self.estimator = estimator
        self.rng = rng
        self.dominance_threshold = dominance_threshold
        self.tail_ratio_threshold = tail_ratio_threshold

    def _rms(self, X) -> float:
        if self.estimator == "sampled":
            return rms_pairwise_distance(X, rng=self.rng)
        return theoretical_rms_pairwise(X)

    def fit(self, blocks: Sequence[tuple[str, np.ndarray]]) -> ScaleReport:
        labels = tuple(lbl for lbl, _ in blocks)
        rms = tuple(self._rms(X) for _, X in blocks)
        n = len(blocks)

        if self.policy in ("rms_balanced", "empirical_balanced"):
            # weight w_g so that w_g * rms_g^2 is equal across blocks
            w = []
            for r in rms:
                w.append(1.0 / (r**2) if r > 0 else 0.0)
            total = sum(w) or 1.0
            weights = tuple(wi / total * n for wi in w)
        elif self.policy == "weighted":
            if self.lambda_ is None:
                raise ParameterError("policy='weighted' requires lambda_")
            if n != 2:
                raise ParameterError(
                    f"lambda_ weighting is defined for exactly 2 blocks, got {n}. "
                    "Use policy='custom' weights for more."
                )
            lam = float(self.lambda_)
            if not 0.0 <= lam <= 1.0:
                raise ParameterError(f"lambda_ must be in [0,1], got {lam}")
            base = [1.0 / (r**2) if r > 0 else 0.0 for r in rms]
            weights = (2 * (1 - lam) * base[0], 2 * lam * base[1])
        elif self.policy == "custom":
            if self.weights_in is None or len(self.weights_in) != n:
                raise ParameterError(f"custom policy needs {n} weights")
            weights = tuple(float(w) for w in self.weights_in)
        elif self.policy in ("none", "per_block_threshold"):
            weights = tuple(1.0 for _ in blocks)
        else:
            raise ParameterError(f"unknown scaling policy {self.policy!r}")

        contrib = np.array([w * r**2 for w, r in zip(weights, rms, strict=False)])
        total = contrib.sum()
        share = tuple(float(c / total) if total > 0 else 0.0 for c in contrib)

        dominant = None
        warning = None
        if share:
            k = int(np.argmax(share))
            if share[k] > self.dominance_threshold and n > 1:
                dominant = labels[k]
                warning = (
                    f"block {labels[k]!r} contributes {100 * share[k]:.1f}% of the "
                    "distance variance; the recurrence structure will mostly "
                    "reflect that block alone"
                )

        shapes = [shape_stats(X, rng=self.rng) for _, X in blocks]
        tails = tuple(float(s_["tail_ratio"]) for s_ in shapes)
        skews = tuple(float(s_["skew"]) for s_ in shapes)

        shape_warning = None
        if len(tails) > 1 and min(tails) > 0:
            spread = max(tails) / min(tails)
            if spread > self.tail_ratio_threshold:
                k = int(np.argmax(tails))
                shape_warning = (
                    f"block {labels[k]!r} has a much heavier tail than the others "
                    f"(tail ratio {max(tails):.1f} vs {min(tails):.1f}). Variance is "
                    "balanced but distribution shape is not: outlying points will be "
                    "isolated in the recurrence plot. Consider scale='rank' or "
                    "'robust' on that block."
                )

        return ScaleReport(labels, rms, weights, share, self.policy, dominant, warning,
                           tail_ratio=tails, skewness=skews, shape_warning=shape_warning)
