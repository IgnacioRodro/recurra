"""Line-length histograms, accumulated by streaming tiles.

Design note (DD-56) -- the whole long-signal strategy rests on this file.

Every RQA metric is a function of three distributions: the lengths of the
diagonal lines, of the vertical lines, and of the white vertical lines. Those
can be accumulated tile by tile, so the recurrence matrix never has to exist
[DD-36]. The price is that a run crossing a tile boundary must be carried:
its length so far is held open, and completed only when a zero closes it or
the matrix ends.

That carry is where off-by-one errors live, so the implementation is checked
against a naive dense reference for deliberately awkward tile sizes, and the
histograms must agree exactly -- not approximately.

Tile order matters and is not free. Tiles are visited row band by row band,
left to right within a band, because that is the order in which both a
diagonal and a column encounter their own cells contiguously:

* a column ``j`` is continued from the band above, so one open length per
  column is carried, an array of ``n_cols``;
* a diagonal ``k = j - i`` runs down-right, so within a band it moves left to
  right across tiles and then continues in the next band. One open length per
  diagonal is carried, an array of ``n_rows + n_cols``.

Design note (DD-57) -- white lines and the Theiler band.

A white vertical line is a run of *non*-recurrent cells, and its length is a
recurrence time. But cells excluded by a Theiler window are stored as zero,
so they read as white and would inflate every recurrence time by the width of
the band. Excluded cells therefore break the column into segments, and any
white run touching the band is discarded rather than counted: it has been
truncated by an artefact of the analysis rather than by a recurrence.

Runs reaching the *start or end of the record* are kept. They are truncated
too, but by the data rather than by a choice, and dropping them would bias the
distribution against long recurrence times, which are exactly the ones a
recurrence-time entropy is sensitive to. The asymmetry is deliberate and worth
knowing when reading RTE.

Design note (DD-124) -- whole-array operations, not a loop per line.

Up to 0.25 each tile was scanned with one Python call per diagonal and per
column. At 2560 points that was 12 767 calls and 0.54 s per plot, of which the
distances took 0.048 s: the interpreter, not the arithmetic, set the cost.
``line_histogram`` now visits the matrix in *row bands of full width* and finds
every run of a band at once, with array operations:

* a band is transposed so that each column is a row of the array, padded with
  zeros, and differenced: +1 marks where a run starts and -1 where it ends;
* diagonals are sheared into columns first (row ``r`` shifted by ``h - 1 - r``),
  and then treated exactly like columns;
* the carry of DD-56 becomes one array per kind of line: the run each column
  or diagonal leaves open at the bottom of a band is added to the run that
  starts at the top of the next, or emitted if none does;
* the Theiler band is a third cell state for white lines. A white run is
  dropped when the cell just before or just after it is excluded, which is the
  rule of DD-57, and since the band is known analytically (``|i - j| <
  theiler``) that cell can be tested even when it lies in another band. A run
  open at a band edge carries its flag with it.

Lengths are counted with ``np.bincount`` into integer histograms, so the
result is exact and identical, count for count, to the tile-by-tile version
(kept as ``_line_histogram_tiles``) and to the naive ``line_histogram_dense``,
for any band height.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..recurrence.core import band_rows, theiler_band_cells


def _runs(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end indices of every run of True in a 1-D boolean array."""
    d = np.diff(np.concatenate(([0], a.view(np.int8) if a.dtype == bool
                                else (a != 0).astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _accumulate(segment: np.ndarray, open_len: int, out: list) -> int:
    """Add the completed runs of one segment; return the length left open.

    ``open_len`` is the length of a run continuing from the previous segment.
    A run touching the start of this segment extends it; a run touching the
    end is left open for the next.
    """
    n = segment.size
    if n == 0:
        return open_len
    starts, ends = _runs(segment)
    if starts.size == 0:                       # nothing recurrent here
        if open_len:
            out.append(open_len)
        return 0

    lengths = ends - starts
    first_touches = starts[0] == 0
    last_touches = ends[-1] == n

    if first_touches and last_touches and starts.size == 1:
        return open_len + int(lengths[0])       # the whole segment is one run

    if first_touches:
        out.append(open_len + int(lengths[0]))
        body = lengths[1:]
    else:
        if open_len:
            out.append(open_len)
        body = lengths

    if last_touches:
        out.extend(int(v) for v in body[:-1])
        return int(body[-1])
    out.extend(int(v) for v in body)
    return 0


@dataclass
class LineHistogram:
    """Counts of line lengths, plus what was needed to compute them.

    ``diagonal[l]`` is the number of diagonal lines of length ``l``; index 0
    is unused. The same for ``vertical`` and ``white_vertical``.
    """

    diagonal: np.ndarray
    vertical: np.ndarray
    white_vertical: np.ndarray
    recurrent_points: int = 0
    considered_points: int = 0
    n_rows: int = 0
    n_cols: int = 0
    theiler: int = 0
    exact: bool = True
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def recurrence_rate(self) -> float:
        return (self.recurrent_points / self.considered_points
                if self.considered_points else float("nan"))

    def counts(self, kind: str = "diagonal", l_min: int = 1) -> np.ndarray:
        h = getattr(self, kind)
        out = h.copy()
        out[:l_min] = 0
        return out

    def points_in_lines(self, kind: str = "diagonal", l_min: int = 2) -> int:
        h = self.counts(kind, l_min)
        return int(np.sum(h * np.arange(h.size)))

    def n_lines(self, kind: str = "diagonal", l_min: int = 2) -> int:
        return int(self.counts(kind, l_min).sum())

    def to_frame(self, kind: str = "diagonal") -> pd.DataFrame:
        h = getattr(self, kind)
        nz = np.flatnonzero(h)
        return pd.DataFrame({"kind": kind, "length": nz, "count": h[nz]})

    def frames(self) -> pd.DataFrame:
        return pd.concat([self.to_frame(k) for k in
                          ("diagonal", "vertical", "white_vertical")],
                         ignore_index=True)

    def summary(self) -> str:
        return (
            f"LineHistogram {self.n_rows}x{self.n_cols}, Theiler {self.theiler}\n"
            f"  recurrence   : {self.recurrence_rate:.4%} "
            f"({self.recurrent_points} of {self.considered_points} pairs)\n"
            f"  diagonal     : {int(self.diagonal.sum())} lines, "
            f"longest {int(np.max(np.flatnonzero(self.diagonal), initial=0))}\n"
            f"  vertical     : {int(self.vertical.sum())} lines, "
            f"longest {int(np.max(np.flatnonzero(self.vertical), initial=0))}\n"
            f"  white vert.  : {int(self.white_vertical.sum())} lines, "
            f"longest {int(np.max(np.flatnonzero(self.white_vertical), initial=0))}"
        )


def _to_histogram(lengths: list, size: int) -> np.ndarray:
    h = np.zeros(size + 1, dtype=np.int64)
    if lengths:
        vals, counts = np.unique(np.asarray(lengths, dtype=np.int64),
                                 return_counts=True)
        h[vals] = counts
    return h


def _line_histogram_tiles(rm, *, tile_size: int | None = None) -> LineHistogram:
    """The 0.25 implementation: square tiles, one Python call per line.

    Kept as a second reference for :func:`line_histogram`, which must agree
    with it count for count [DD-124]. Exact for any tile size [DD-56].
    """
    n, m = rm.shape
    theiler = rm.theiler
    symmetric_source = rm.coords2 is None

    diag_open = np.zeros(n + m + 1, dtype=np.int64)      # keyed by k + n
    vert_open = np.zeros(m, dtype=np.int64)
    white_open = np.zeros(m, dtype=np.int64)
    # A white run reaching the Theiler band is discarded, and it may extend
    # beyond the tile where the band was met, so the decision has to persist.
    white_dropping = np.zeros(m, dtype=bool)

    diag_lengths: list = []
    vert_lengths: list = []
    white_lengths: list = []
    recurrent = 0

    ts = int(tile_size or rm.tile_size)
    row_bands = range(0, n, ts)
    for i0 in row_bands:
        i1 = min(i0 + ts, n)
        for j0 in range(0, m, ts):
            j1 = min(j0 + ts, m)
            block = rm._block(i0, i1, j0, j1)
            b = block.astype(bool)
            recurrent += int(b.sum())

            # ---- diagonals: offsets local to the block, mapped to global k
            h, w = b.shape
            for d in range(-(h - 1), w):
                seg = np.diagonal(b, offset=d)
                if seg.size == 0:
                    continue
                k = (j0 + max(d, 0)) - (i0 + max(-d, 0))
                diag_open[k + n] = _accumulate(seg, int(diag_open[k + n]),
                                               diag_lengths)

            # ---- verticals: one carry per column, continued from the band above
            for c in range(w):
                col = b[:, c]
                j = j0 + c
                vert_open[j] = _accumulate(col, int(vert_open[j]), vert_lengths)

                # ---- white verticals, with the Theiler band excised [DD-57]
                white = ~col
                if theiler > 0 and symmetric_source:
                    if white_dropping[j]:
                        # still inside a run that reached the band
                        if white.all():
                            continue
                        white = white[int(np.argmin(white)):]
                        white_dropping[j] = False
                        white_open[j] = _accumulate(white, 0, white_lengths)
                        continue
                    # the band in column j is rows j-(w-1) .. j+(w-1) [DD-115]
                    lo, hi = j - (theiler - 1), j + (theiler - 1)
                    a, z = max(lo, i0), min(hi, i1 - 1)
                    if a <= z:
                        before, after = white[:a - i0], white[z - i0 + 1:]
                        pending: list = []
                        # whatever _accumulate leaves open reached the band and
                        # is dropped; the runs it completed are legitimate
                        _accumulate(before, int(white_open[j]), pending)
                        white_lengths.extend(pending)
                        if after.size == 0:
                            # the band runs to the tile edge; whatever follows
                            # in the next tile still touches it
                            white_open[j] = 0
                            white_dropping[j] = True
                            continue
                        if after[0]:
                            if after.all():
                                white_open[j] = 0
                                white_dropping[j] = True
                                continue
                            after = after[int(np.argmin(after)):]
                        white_open[j] = _accumulate(after, 0, white_lengths)
                        continue
                white_open[j] = _accumulate(white, int(white_open[j]), white_lengths)

    # flush whatever is still open at the end of the matrix
    for arr, sink in ((diag_open, diag_lengths), (vert_open, vert_lengths),
                      (white_open, white_lengths)):
        sink.extend(int(v) for v in arr[arr > 0])

    considered = n * m - (theiler_band_cells(n, m, theiler) if symmetric_source else 0)

    size = max(n, m)
    return LineHistogram(
        diagonal=_to_histogram(diag_lengths, size),
        vertical=_to_histogram(vert_lengths, size),
        white_vertical=_to_histogram(white_lengths, size),
        recurrent_points=recurrent, considered_points=considered,
        n_rows=n, n_cols=m, theiler=theiler,
        diagnostics={"tile_size": ts, "symmetric": bool(rm.is_symmetric)},
    )


def _runs_by_row(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every run of True along the rows of a 2-D boolean array, at once.

    Returns ``(row, start, end)``, sorted by row and then by start, with
    ``end`` exclusive. Each row is padded with a zero at both ends, so every
    start has its end in the same row and the two lists pair up in order.
    """
    n_lines, h = X.shape
    P = np.zeros((n_lines, h + 2), dtype=np.int8)
    P[:, 1:-1] = X
    d = np.diff(P, axis=1).ravel()
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    row = s // (h + 1)
    return row, s - row * (h + 1), e - row * (h + 1)


def _carry_runs(X: np.ndarray, keys: np.ndarray, open_len: np.ndarray,
                counts: np.ndarray) -> None:
    """Runs of one band, joined to the runs the previous band left open.

    ``X[k]`` holds the cells of line ``keys[k]`` in this band, in order.
    ``open_len`` is indexed by key and updated in place; completed lengths are
    added to ``counts`` [DD-56, DD-124].
    """
    h = X.shape[1]
    row, start, end = _runs_by_row(X)
    length = end - start
    carried = open_len[keys]
    top = start == 0
    length[top] += carried[row[top]]
    continued = np.zeros(X.shape[0], dtype=bool)
    continued[row[top]] = True
    bottom = end == h
    _count(counts, length[~bottom])
    _count(counts, carried[(carried > 0) & ~continued])     # closed at the edge
    still_open = np.zeros(X.shape[0], dtype=np.int64)
    still_open[row[bottom]] = length[bottom]
    open_len[keys] = still_open


def _count(counts: np.ndarray, lengths: np.ndarray) -> None:
    if lengths.size:
        counts += np.bincount(lengths, minlength=counts.size)[:counts.size]




def line_histogram(rm, *, tile_size: int | None = None) -> LineHistogram:
    """Accumulate the three line-length distributions, band by band.

    Exact: identical, count for count, to what the dense matrix would give,
    for any band height [DD-56, DD-124]. A band holds as many cells as one
    square tile of ``tile_size`` (default: the matrix's own), so the memory
    used is the same as before. A materialised matrix is read rather than
    recomputed.
    """
    n, m = rm.shape
    theiler = int(rm.theiler)
    banded = theiler > 0 and rm.coords2 is None      # the band exists [DD-57]
    size = max(n, m)
    ts = int(tile_size or rm.tile_size)
    h_max = band_rows(n, m, ts)
    stored = getattr(rm, "_matrix", None)

    diag = np.zeros(size + 1, dtype=np.int64)
    vert = np.zeros(size + 1, dtype=np.int64)
    white = np.zeros(size + 1, dtype=np.int64)
    diag_open = np.zeros(n + m + 1, dtype=np.int64)          # keyed by k + n
    vert_open = np.zeros(m, dtype=np.int64)
    white_open = np.zeros(m, dtype=np.int64)
    white_flag = np.zeros(m, dtype=bool)     # the open white run touched the band
    columns = np.arange(m, dtype=np.int64)
    recurrent = 0

    def excluded(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Is cell (row, col) in the Theiler band? Outside the matrix: no."""
        if not banded:
            return np.zeros(rows.shape, dtype=bool)
        return (rows >= 0) & (rows < n) & (np.abs(rows - cols) < theiler)

    for i0 in range(0, n, h_max):
        i1 = min(i0 + h_max, n)
        h = i1 - i0
        B = (np.asarray(stored[i0:i1]) if stored is not None
             else rm._block(i0, i1, 0, m)).astype(bool, copy=False)
        recurrent += int(np.count_nonzero(B))

        # ---- verticals: each column is a line
        BT = np.ascontiguousarray(B.T)
        _carry_runs(BT, columns, vert_open, vert)

        # ---- diagonals: shear so that diagonal k becomes column k + h-1 - i0
        S = np.zeros((h, m + h - 1), dtype=bool)
        for r in range(h):
            S[r, h - 1 - r:h - 1 - r + m] = B[r]
        keys = np.arange(m + h - 1, dtype=np.int64) - (h - 1) - i0 + n
        _carry_runs(np.ascontiguousarray(S.T), keys, diag_open, diag)

        # ---- white verticals, with the Theiler band as a third state [DD-57]
        W = ~BT                                        # (m, h): column, row
        if banded:
            for c in range(max(0, i0 - theiler + 1), min(m, i1 + theiler - 1)):
                lo, hi = max(c - theiler + 1, i0), min(c + theiler, i1)
                if lo < hi:
                    W[c, lo - i0:hi - i0] = False
        col, start, end = _runs_by_row(W)
        length = end - start
        carried = white_open[col]
        top = start == 0
        joined = top & (carried > 0)
        length[joined] += carried[joined]
        flag = np.where(joined, white_flag[col],
                        excluded(i0 + start - 1, col))
        bottom = end == h
        ends_here = ~bottom
        flag_end = flag | excluded(i0 + end, col)
        _count(white, length[ends_here & ~flag_end])
        # open runs from above that stop at this band's first row
        continued = np.zeros(m, dtype=bool)
        continued[col[joined]] = True
        stopped = (white_open > 0) & ~continued
        if stopped.any():
            c_stop = columns[stopped]
            bad = white_flag[stopped] | excluded(np.full(c_stop.size, i0), c_stop)
            _count(white, white_open[stopped][~bad])
        white_open[:] = 0
        white_flag[:] = False
        white_open[col[bottom]] = length[bottom]
        white_flag[col[bottom]] = flag[bottom]

    # flush what is still open at the end of the matrix: runs reaching the end
    # of the record are kept [DD-57]
    _count(diag, diag_open[diag_open > 0])
    _count(vert, vert_open[vert_open > 0])
    _count(white, white_open[(white_open > 0) & ~white_flag])

    considered = n * m - (theiler_band_cells(n, m, theiler) if rm.coords2 is None else 0)
    return LineHistogram(
        diagonal=diag, vertical=vert, white_vertical=white,
        recurrent_points=recurrent, considered_points=considered,
        n_rows=n, n_cols=m, theiler=theiler,
        diagnostics={"tile_size": ts, "band_rows": h_max,
                     "symmetric": bool(rm.is_symmetric)},
    )


def line_histogram_dense(matrix: np.ndarray, *, theiler: int = 0,
                         symmetric_source: bool = True) -> LineHistogram:
    """Reference implementation on a materialised matrix.

    Deliberately naive. It exists so the streaming version can be checked
    against something whose correctness is obvious by inspection.
    """
    b = np.asarray(matrix).astype(bool)
    n, m = b.shape

    diag: list = []
    for k in range(-(n - 1), m):
        seg = np.diagonal(b, offset=k)
        s, e = _runs(seg)
        diag.extend((e - s).tolist())

    vert: list = []
    white: list = []
    for j in range(m):
        col = b[:, j]
        s, e = _runs(col)
        vert.extend((e - s).tolist())

        w = ~col
        if theiler > 0 and symmetric_source:
            # The band splits the column. A white run reaching it was cut short
            # by the analysis rather than by a recurrence, so it is dropped;
            # runs reaching the ends of the record are kept [DD-57].
            before, after = w[:max(j - theiler + 1, 0)], w[min(j + theiler, n):]
            s, e = _runs(before)
            if s.size and e[-1] == before.size:
                s, e = s[:-1], e[:-1]                  # reaches the band
            white.extend((e - s).tolist())
            s, e = _runs(after)
            if s.size and s[0] == 0:
                s, e = s[1:], e[1:]                    # reaches the band
            white.extend((e - s).tolist())
        else:
            s, e = _runs(w)
            white.extend((e - s).tolist())

    considered = n * m - (theiler_band_cells(n, m, theiler) if symmetric_source else 0)

    size = max(n, m)
    return LineHistogram(
        diagonal=_to_histogram(diag, size),
        vertical=_to_histogram(vert, size),
        white_vertical=_to_histogram(white, size),
        recurrent_points=int(b.sum()), considered_points=considered,
        n_rows=n, n_cols=m, theiler=theiler,
        diagnostics={"reference": True},
    )
