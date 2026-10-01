"""The recurrence matrix: an engine, not necessarily an array.

Design note (DD-36) -- the matrix is a detail of implementation.

Principle P2. For an hour of EEG at 500 Hz the recurrence matrix has 3.2e12
entries; it cannot be an array, and an API built around one would be unusable
exactly where the project needs it most. So :class:`RecurrenceMatrix` is a
*lazy engine* that knows how to produce any tile of itself on demand. Storing
the whole thing is an option (``store="memory"``), not the premise.

Every quantity that matters -- recurrence rate now, line-length histograms in
F6 -- is computed by streaming tiles and accumulating. Streaming results are
exact, not approximate: the same numbers the dense path gives, verified bit
for bit in the tests.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import get_config
from ..exceptions import ParameterError
from ..provenance import Step, provenance_frame
from ..threshold.core import Threshold
from .distance import sq_distance_block, threshold_operand
from .memory import MemoryPlan, plan

KINDS = ("rp", "crp", "jrp")
#: Metrics whose distances do not change when both points are shifted alike.
#: Only these may be centred before the kernel runs [DD-110].
TRANSLATION_INVARIANT = ("euclidean", "chebyshev", "manhattan")
STORES = ("memory", "sparse", "stream")
#: Accepted spellings of a store. "none" was the only name for streaming before
#: 0.25 and read as "no storage at all"; it is kept as an alias [DD-118].
STORE_ALIASES = {"none": "stream"}


def canonical_store(store: str) -> str:
    """The canonical name of a store choice, accepting the old alias."""
    store = STORE_ALIASES.get(store, store)
    if store not in STORES + ("auto",):
        raise ParameterError(
            f"unknown store {store!r}; use one of {STORES + ('auto',)}")
    return store
BACKENDS = ("dense", "tiled", "auto")


@dataclass(frozen=True)
class Tile:
    """One rectangular block of the recurrence matrix."""

    i0: int
    i1: int
    j0: int
    j1: int
    data: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape


@dataclass
class RecurrenceMatrix:
    """A recurrence structure, materialised or streamed.

    Attributes
    ----------
    coords, coords2
        The trajectories. ``coords2 is coords`` for an ordinary recurrence
        plot; a different array for a cross recurrence plot.
    threshold
        The :class:`~recurra.threshold.Threshold` that defines recurrence.
        May be scalar, per point (FAN) or per block.
    """

    coords: np.ndarray
    threshold: Threshold
    coords2: np.ndarray | None = None
    kind: str = "rp"
    metric: str = "euclidean"
    theiler: int = 0
    dtype: str = "uint8"
    tile_size: int = 1024
    store: str = "memory"
    precision: str = "double"
    groups: tuple = ()
    fs: float = 1.0
    t0: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    provenance: tuple[Step, ...] = ()
    memory_plan: MemoryPlan | None = None
    _matrix: np.ndarray | None = field(default=None, repr=False)
    _rr: float | None = field(default=None, repr=False)

    # ----------------------------------------------------------- shape
    @property
    def n_rows(self) -> int:
        return self.coords.shape[0]

    @property
    def n_cols(self) -> int:
        return (self.coords if self.coords2 is None else self.coords2).shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_cols)

    @property
    def is_symmetric(self) -> bool:
        return (self.coords2 is None and not self.threshold.is_pointwise)

    @property
    def dim(self) -> int:
        return self.coords.shape[1]

    # -------------------------------------------------------- tile work
    def _working(self, which: int) -> np.ndarray:
        """The coordinate array the kernels see, centred, in the requested precision.

        Design note (DD-110) -- centre before expanding the norm.

        The kernel computes ``||a||^2 + ||b||^2 - 2 a.b`` [DD-33]. That is exact
        algebra and poor arithmetic: when the coordinates sit far from the
        origin the two norms are large and nearly equal, and the small distance
        that matters is what is left after they cancel. In single precision the
        error is a few parts in 10^7 of the *norm*, not of the distance, so it
        grows with the offset. Measured on a phase-amplitude space at a 5%
        rate: with no offset, no cell differs between single and double; with
        an offset of 100 in every coordinate, 21 000 cells of 3.2e7 differ; at
        1000, the recurrence rate itself moves from 0.051 to 0.082.

        Euclidean, Chebyshev and Manhattan distances are invariant to
        translation, so the column mean of the first trajectory is subtracted
        from both trajectories once. The same vector is used for both, which
        keeps a cross recurrence plot exact. Applied in double as well, where
        the error is far smaller, so the two precisions follow one code path.

        Cosine distance is *not* translation invariant: it measures the angle
        seen from the origin, and centring moves the origin. Before 0.25 it was
        centred anyway, while the threshold had been estimated on the raw
        coordinates, so the matrix and its threshold described two different
        geometries -- measured on an offset point cloud at a target rate of
        0.05, the matrix came out at 0.0013. Cosine coordinates are now left
        where they are; the cosine kernel has no norm expansion to protect.
        """
        cache = self.__dict__.setdefault("_work_cache", {})
        if which not in cache:
            if "offset" not in cache:
                cache["offset"] = (
                    np.asarray(self.coords, dtype=float).mean(axis=0)
                    if self.metric in TRANSLATION_INVARIANT
                    else np.zeros(self.dim))
            X = self.coords if which == 0 else (
                self.coords if self.coords2 is None else self.coords2)
            centred = np.asarray(X, dtype=float) - cache["offset"]
            cache[which] = (np.ascontiguousarray(centred, dtype=np.float32)
                            if self.precision == "single"
                            else np.ascontiguousarray(centred))
        return cache[which]

    def _norms(self, which: int) -> np.ndarray:
        """Squared row norms, computed once and reused across every tile.

        Every tile of a row band needs the same ``||a||^2``; recomputing it per
        tile costs O(N x dim) per tile and nothing else.
        """
        cache = self.__dict__.setdefault("_norm_cache", {})
        if which not in cache:
            X = self._working(which)
            cache[which] = np.einsum("ij,ij->i", X, X)
        return cache[which]

    def _block(self, i0: int, i1: int, j0: int, j1: int) -> np.ndarray:
        """Recurrence for one tile. The single place recurrence is decided.

        Design note (DD-78) -- most tiles never touch the Theiler band.

        Building the exclusion mask costs an (h x w) index comparison, and for
        a long record almost every tile lies far from the diagonal where the
        band cannot reach. At 240 000 points with 8192-sample tiles, 30 of 900
        tiles intersect the band and 870 do not. The mask is therefore built
        only where it can matter.
        """
        A = self._working(0)[i0:i1]
        B = self._working(1)[j0:j1]

        if self.threshold.is_per_block:
            # Chebyshev across blocks: recurrent only if close in every block.
            out = np.ones((i1 - i0, j1 - j0), dtype=bool)
            for g, eps in zip(self.groups, self.threshold.per_block, strict=False):
                D = sq_distance_block(A[:, g.slice], B[:, g.slice], self.metric)
                out &= D <= threshold_operand(eps, self.metric)
        else:
            D = sq_distance_block(
                A, B, self.metric,
                A_sq=(self._norms(0)[i0:i1] if self.metric == "euclidean" else None),
                B_sq=(self._norms(1)[j0:j1] if self.metric == "euclidean" else None))
            eps = self.threshold.value
            if self.threshold.is_pointwise:
                op = threshold_operand(np.asarray(eps)[i0:i1], self.metric)[:, None]
            else:
                op = threshold_operand(float(eps), self.metric)
            out = D <= op

        # The band is every pair with |i - j| < theiler, so theiler=1 removes
        # the line of identity and nothing else [DD-115]. Only tiles whose
        # index ranges come within the band can contain excluded cells [DD-78].
        reach = self.theiler - 1
        if (self.theiler > 0 and self.coords2 is None
                and i0 - reach <= j1 - 1 and j0 - reach <= i1 - 1):
            ii = np.arange(i0, i1)[:, None]
            jj = np.arange(j0, j1)[None, :]
            out &= np.abs(ii - jj) >= self.theiler
        return out

    def tiles(self, tile_size: int | None = None) -> Iterator[Tile]:
        """Stream the matrix tile by tile. Never allocates the whole thing."""
        ts = int(tile_size or self.tile_size)
        for i0 in range(0, self.n_rows, ts):
            i1 = min(i0 + ts, self.n_rows)
            for j0 in range(0, self.n_cols, ts):
                j1 = min(j0 + ts, self.n_cols)
                yield Tile(i0, i1, j0, j1, self._block(i0, i1, j0, j1))

    # -------------------------------------------------------- materialise
    def materialize(self, force: bool = False) -> np.ndarray:
        """Build the full matrix in memory. Checks the budget first."""
        if self._matrix is not None and not force:
            return self._matrix
        p = self.memory_plan or plan(self.n_rows, self.n_cols, self.dim,
                                     dtype=self.dtype, store="memory")
        if p.matrix_gb > get_config().memory_budget_gb:
            from ..exceptions import MemoryBudgetError

            raise MemoryBudgetError(
                f"materialising needs {p.matrix_gb:.2f} GiB, budget is "
                f"{get_config().memory_budget_gb:.2f} GiB. Use the streaming API "
                "(.tiles(), .recurrence_rate()) or raise the budget."
            )
        M = np.zeros(self.shape, dtype=np.uint8 if self.dtype == "uint8" else bool)
        for t in self.tiles():
            M[t.i0:t.i1, t.j0:t.j1] = t.data      # bool -> uint8 on assignment
        self._matrix = M
        return M

    @property
    def matrix(self) -> np.ndarray:
        return self.materialize()

    def to_sparse(self):
        """CSR of the recurrent pairs. Cheap when the rate is low."""
        from scipy import sparse

        rows, cols = [], []
        for t in self.tiles():
            r, c = np.nonzero(t.data)
            rows.append(r + t.i0)
            cols.append(c + t.j0)
        r = np.concatenate(rows) if rows else np.array([], dtype=int)
        c = np.concatenate(cols) if cols else np.array([], dtype=int)
        return sparse.csr_array((np.ones(r.size, dtype=np.uint8), (r, c)),
                                shape=self.shape)

    # ------------------------------------------------------------ metrics
    def recurrence_rate(self, *, recompute: bool = False) -> float:
        """Fraction of recurrent pairs, streamed and exact.

        The denominator excludes the Theiler band, matching the region the
        threshold was estimated over [DD-32], so the number is directly
        comparable with ``Threshold.achieved_rr``.
        """
        if self._rr is not None and not recompute:
            return self._rr
        count = 0
        for t in self.tiles():
            count += int(np.count_nonzero(t.data))
        n, m = self.shape
        if self.coords2 is None:
            total = n * m - theiler_band_cells(n, m, self.theiler)
        else:
            total = n * m
        self._rr = count / total if total > 0 else float("nan")
        return self._rr

    def rqa(self, **kwargs):
        """Recurrence quantification of this structure. See :func:`recurra.rqa`."""
        from ..rqa.metrics import rqa as _rqa

        return _rqa(self, **kwargs)

    def line_histogram(self, **kwargs):
        """The three line-length distributions, accumulated by streaming."""
        from ..rqa.histogram import line_histogram as _lh

        return _lh(self, **kwargs)

    def density_profile(self, n_bins: int = 64) -> pd.DataFrame:
        """Recurrence density per row band: where the plot is dense or empty."""
        n = self.n_rows
        edges = np.linspace(0, n, n_bins + 1).astype(int)
        counts = np.zeros(n_bins)
        totals = np.zeros(n_bins)
        for t in self.tiles():
            per_row = t.data.sum(axis=1)
            for b in range(n_bins):
                lo, hi = max(edges[b], t.i0), min(edges[b + 1], t.i1)
                if hi > lo:
                    counts[b] += per_row[lo - t.i0:hi - t.i0].sum()
                    totals[b] += (hi - lo) * (t.j1 - t.j0)
        with np.errstate(invalid="ignore", divide="ignore"):
            dens = np.where(totals > 0, counts / totals, np.nan)
        return pd.DataFrame({
            "bin": np.arange(n_bins),
            "start_sample": edges[:-1], "stop_sample": edges[1:],
            "start_time_s": self.t0 + edges[:-1] / self.fs,
            "density": dens,
        })

    # ---------------------------------------------------------- reporting
    def describe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "kind": self.kind, "n_rows": self.n_rows, "n_cols": self.n_cols,
            "dim": self.dim, "metric": self.metric, "theiler": self.theiler,
            "symmetric": self.is_symmetric,
            "threshold_mode": self.threshold.mode,
            "epsilon": self.threshold.scalar,
            "target_rr": self.threshold.target,
            "estimated_rr": self.threshold.achieved_rr,
            "actual_rr": self.recurrence_rate(),
            "store": self.store, "tile_size": self.tile_size, "dtype": self.dtype,
            "precision": self.precision,
            "dense_gb": (self.memory_plan.matrix_gb if self.memory_plan
                         else np.nan),
            "subject": self.meta.get("subject", ""),
            "builder": self.meta.get("builder", ""),
        }])

    def provenance_frame(self) -> pd.DataFrame:
        return provenance_frame(self.provenance)

    def summary(self) -> str:
        rr = self.recurrence_rate()
        lines = [
            f"RecurrenceMatrix({self.kind.upper()}) {self.n_rows} x {self.n_cols}, "
            f"dim {self.dim}",
            f"  threshold  : {self.threshold.mode} -> epsilon = "
            f"{self.threshold.scalar:.6g} ({self.metric})",
            f"  Theiler    : {self.theiler}",
            f"  recurrence : {rr:.4%} actual"
            + (f", {self.threshold.achieved_rr:.4%} estimated at threshold time"
               if self.threshold.achieved_rr is not None else ""),
            f"  storage    : {self.store}, tiles of {self.tile_size}, dtype {self.dtype}",
        ]
        if self.threshold.target is not None:
            drift = abs(rr - self.threshold.target)
            lines.append(f"  target     : {self.threshold.target:.4%} "
                         f"(deviation {drift:.4%})")
        if self.memory_plan and self.memory_plan.note:
            lines.append(f"  note       : {self.memory_plan.note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<RecurrenceMatrix {self.kind} {self.n_rows}x{self.n_cols} "
                f"eps={self.threshold.scalar:.4g}>")


def theiler_band_cells(n: int, m: int, theiler: int) -> int:
    """Cells of an auto-recurrence matrix inside the Theiler band.

    Design note (DD-115) -- the standard convention.

    ``theiler=w`` excludes every pair with ``|i - j| < w``: ``0`` excludes
    nothing, ``1`` the line of identity, ``w`` the ``2w - 1`` central
    diagonals. This is the convention of the CRP Toolbox, pyunicorn and PyRQA,
    so a window passed to any of them means the same thing here.
    """
    if theiler <= 0:
        return 0
    return int(sum(max(0, min(n, m) - abs(k)) for k in range(-(theiler - 1), theiler)))


def _validate_store(store: str) -> str:
    return canonical_store(store)
