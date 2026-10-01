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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..recurrence.core import theiler_band_cells


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


def line_histogram(rm, *, tile_size: int | None = None) -> LineHistogram:
    """Accumulate the three line-length distributions by streaming tiles.

    Exact: the result is identical to what the dense matrix would give, for
    any tile size [DD-56].
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
