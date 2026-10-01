"""Comparing routes and sweeping lambda.

Design note (DD-26). Objective O2.1 of the project specification asks to justify the
choice of state space. The only honest justification is empirical, so
comparing routes is a first-class operation, not something a user must
improvise.

Until the RQA layer exists (F6), the geometric descriptors here are
threshold-free and cheap: nearest-neighbour statistics and a
correlation-sum slope. They already discriminate the routes, and they will
be joined -- not replaced -- by DET/LAM once F6 lands.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..config import SeedLike, resolve_rng
from ..exceptions import ParameterError
from .core import StateSpace


def shape_descriptors(ss: StateSpace) -> dict[str, float]:
    """Per-block distribution shape: how heavy-tailed each block is.

    Design note (DD-102) -- the warning was a warning and not a column.

    ``rms_balanced`` equalises the *variance* of the blocks and not their
    *shape*, so a block with a heavy tail contributes points that sit far from
    everything else [DD-28]. The library warned about it from the start and
    that was the whole of it: nothing recorded how heavy the tail was, so
    nobody could check whether it differed between the groups being compared.

    Measured with coupling held constant across sixteen recordings, the tail
    ratio of the amplitude block drove twenty-two of thirty-nine metrics --
    spectral entropy at Spearman +0.93, Katz at -0.91, DET at +0.83 -- and
    moved the isolated-point fraction from 0.3% to 12.9%. A study whose groups
    differ in envelope burstiness will show differences in most of the table
    for that reason alone, and without this column there is no way to tell.

    ``tail_ratio`` is the 99.9th percentile of pairwise distance within the
    block over its RMS: about 1.4 for a phase circle, 3 to 12 for a real gamma
    envelope. ``skew`` and ``max_ratio`` come with it.
    """
    from ..preprocess.scaling import shape_stats

    coords = np.asarray(ss.coords, dtype=float)
    out: dict[str, float] = {}
    for group in ss.groups:
        stats = shape_stats(coords[:, group.slice])
        for key, value in stats.items():
            out[f"{key}_{group.label}"] = float(value)
    return out


def geometry_descriptors(ss: StateSpace, *, max_points: int = 4000,
                         theiler: int = 1, rng: SeedLike = None) -> dict[str, float]:
    """Threshold-free descriptors of the trajectory's geometry.

    ``nn_mean``      mean nearest-neighbour distance (Theiler-excluded)
    ``nn_cv``        its coefficient of variation: homogeneity of the attractor
    ``corr_slope``   local slope of the correlation sum -> effective dimension
    ``extent``       RMS radius of the point cloud
    """
    rng = resolve_rng(rng)
    X = ss.weighted_coords
    n = X.shape[0]
    idx = np.sort(rng.choice(n, size=max_points, replace=False)) if n > max_points else np.arange(n)
    Y = X[idx]

    tree = cKDTree(Y)
    k = min(max(4, theiler + 2), Y.shape[0])
    dist, nb = tree.query(Y, k=k)
    nn = np.full(Y.shape[0], np.nan)
    for i in range(Y.shape[0]):
        for j in range(1, k):
            if abs(idx[nb[i, j]] - idx[i]) >= max(theiler, 1):   # [DD-115]
                nn[i] = dist[i, j]
                break
    nn = nn[np.isfinite(nn) & (nn > 0)]

    m = min(2000, Y.shape[0])
    sub = Y[rng.choice(Y.shape[0], size=m, replace=False)] if Y.shape[0] > m else Y
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    d = d[np.triu_indices(sub.shape[0], k=1)]
    d = d[d > 0]
    slope = np.nan
    if d.size > 50:
        r = np.quantile(d, [0.02, 0.05, 0.10, 0.20, 0.35])
        C = np.array([np.mean(d < ri) for ri in r])
        ok = (C > 0) & (C < 1)
        if ok.sum() >= 3:
            slope = float(np.polyfit(np.log(r[ok]), np.log(C[ok]), 1)[0])

    return {
        "nn_mean": float(nn.mean()) if nn.size else np.nan,
        "nn_cv": float(nn.std() / nn.mean()) if nn.size and nn.mean() > 0 else np.nan,
        "corr_slope": slope,
        "extent": float(np.sqrt(np.sum(np.var(X, axis=0)))),
        "n_used": int(Y.shape[0]),
        # Cheap, and the thing most likely to be confounding a group
        # comparison without anyone being able to check [DD-102].
        **shape_descriptors(ss),
    }


def neighbour_agreement(a: StateSpace, b: StateSpace, *, k: int = 8,
                        max_points: int = 3000, rng: SeedLike = None) -> float:
    """Fraction of k-nearest neighbours shared between two state spaces.

    Both must have the same number of points and the same time base. This is
    the cheapest honest way to ask 'do these two constructions see the same
    geometry?' before any threshold is chosen.
    """
    n = min(a.n_points, b.n_points)
    rng = resolve_rng(rng)
    idx = np.sort(rng.choice(n, size=max_points, replace=False)) if n > max_points else np.arange(n)
    A, B = a.weighted_coords[-n:][idx], b.weighted_coords[-n:][idx]
    ka = min(k + 1, A.shape[0])
    na = cKDTree(A).query(A, k=ka)[1][:, 1:]
    nb = cKDTree(B).query(B, k=ka)[1][:, 1:]
    return float(np.mean([len(set(x) & set(y)) / (ka - 1) for x, y in zip(na, nb, strict=False)]))


def compare_routes(rec, routes: Mapping[str, Callable[[Any], StateSpace]] | None = None,
                   *, reference: str | None = None, rng: SeedLike = None,
                   **kw) -> tuple[pd.DataFrame, dict[str, StateSpace]]:
    """Build the same recording several ways and tabulate the differences.

    ``routes`` maps a name to a callable taking the Recording. If omitted, a
    default set covering observable / delay / hybrid is used where the
    recording's capabilities allow it.
    """
    from ..capabilities import Capability
    from . import builders as B

    if routes is None:
        routes = {}
        caps = rec.caps
        if Capability.PHASE in caps and Capability.AMPLITUDE in caps:
            routes["observable_pac"] = lambda r: B.pac_space(r)
            routes["hybrid_pac_m3"] = lambda r: B.pac_space(r, embed_amplitude=3)
        if Capability.PHASE in caps:
            routes["observable_phase_only"] = lambda r: B.phase_circle(r)
        if Capability.AMPLITUDE in caps:
            routes["observable_envelope"] = lambda r: B.envelope_space(r)
        if Capability.RAW in caps:
            routes["delay_takens"] = lambda r: B.takens(r, m="auto", tau="auto", rng=rng)
        if not routes:
            raise ParameterError("no default route applies; pass routes= explicitly")

    spaces: dict[str, StateSpace] = {}
    rows = []
    for name, fn in routes.items():
        try:
            ss = fn(rec)
        except Exception as e:  # a route may not apply; report, do not crash
            rows.append({"route_name": name, "status": f"failed: {type(e).__name__}: {e}"})
            continue
        spaces[name] = ss
        d = ss.describe().iloc[0].to_dict()
        d.update(geometry_descriptors(ss, rng=rng, **kw))
        d = {"route_name": name, "status": "ok", **d}
        rows.append(d)

    df = pd.DataFrame(rows)
    ref = reference or (next(iter(spaces)) if spaces else None)
    if ref and ref in spaces:
        agree = []
        for name in df["route_name"]:
            if name in spaces and name != ref:
                agree.append(neighbour_agreement(spaces[ref], spaces[name], rng=rng))
            elif name == ref:
                agree.append(1.0)
            else:
                agree.append(np.nan)
        df[f"nn_agreement_vs_{ref}"] = agree
    return df, spaces


def lambda_sweep(ss: StateSpace, *, grid: Sequence[float] | None = None,
                 rng: SeedLike = None, max_points: int = 3000) -> pd.DataFrame:
    """Sweep the two-block weighting lambda from pure-A to pure-B geometry.

    lambda=0 is the first block alone, lambda=1 the second, and both endpoints
    are one-dimensional projections that carry no information about the
    relation between the blocks. Whatever a joint space has to say lies in the
    interior of the sweep [DD-02].

    The reported ``*_deficit`` columns are the descriptor minus the straight
    line joining the endpoints. For the correlation-sum slope this is
    *negative* under coupling, because a coupling is a constraint and a
    constraint removes dimension: independent blocks give a cartesian product
    whose dimension is the sum of the parts, while a deterministic dependence
    collapses the joint space onto a curve. Measured on synthetic PAC, the
    deficit grows monotonically with coupling strength, Spearman -0.94.

    Note the endpoints are degenerate: one block has weight exactly zero, so
    its coordinates collapse and the space loses a dimension. Descriptors
    there are measuring a different object from the ones in the interior, and
    a deficit dominated by an endpoint says nothing.
    """
    if len(ss.groups) != 2:
        raise ParameterError(
            f"lambda_sweep needs exactly 2 blocks, this space has {len(ss.groups)}: "
            f"{ss.block_labels}"
        )
    grid = np.linspace(0, 1, 11) if grid is None else np.asarray(grid, dtype=float)
    rows = []
    ends = {}
    for lam in grid:
        s = ss.rescale("weighted", lambda_=float(lam))
        g = geometry_descriptors(s, rng=rng, max_points=max_points)
        share = s.scale_frame()["variance_share"].to_numpy()
        rows.append({"lambda": float(lam),
                     f"share_{ss.block_labels[0]}": float(share[0]),
                     f"share_{ss.block_labels[1]}": float(share[1]), **g})
        if lam in (0.0, 1.0):
            ends[lam] = s

    df = pd.DataFrame(rows)
    if 0.0 in ends and 1.0 in ends:
        df["nn_agreement_vs_block0"] = [
            neighbour_agreement(ends[0.0], ss.rescale("weighted", lambda_=float(lam)),
                                max_points=max_points, rng=rng) for lam in df["lambda"]
        ]
        df["nn_agreement_vs_block1"] = [
            neighbour_agreement(ends[1.0], ss.rescale("weighted", lambda_=float(lam)),
                                max_points=max_points, rng=rng) for lam in df["lambda"]
        ]
        # Departure from the straight line joining the endpoints. Named a
        # deficit rather than a synergy: under coupling the correlation-sum
        # slope falls *below* the interpolation, because coupling constrains
        # the joint space rather than enriching it [DD-02].
        for col in ("nn_cv", "corr_slope"):
            y = df[col].to_numpy(dtype=float)
            if np.isfinite(y).sum() >= 3:
                lin = y[0] + (y[-1] - y[0]) * df["lambda"].to_numpy()
                df[f"{col}_deficit"] = y - lin
    return df
