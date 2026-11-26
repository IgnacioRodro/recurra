"""Thresholds, recurrence plots and line histograms on a PyTorch device.

Design note (DD-125) -- the same answer as the reference, by construction.

The torch engine runs the band algorithm of DD-124 on a batch of windows of
equal length at once: a band of every window of the batch is decided,
differenced and counted in one set of array operations, the lines found
through the symmetry of the plot and the float32 decisions fused on CUDA
(DD-127). Three rules make its
answer the reference's answer rather than an approximation of it:

* **Thresholds.** The pairs are the ones ``sample_pair_indices`` draws, with
  the same generator, so the sample is the same. The device computes their
  distances, coordinate by coordinate, in double precision, sorts them and
  finds the two order statistics and the neighbours of epsilon; the last
  arithmetic -- numpy's linear interpolation, the move off a sampled distance
  of DD-122 and the achieved rate -- is done on the CPU with numpy's own
  operations on those few numbers. Epsilon is the reference's epsilon, bit for
  bit, whenever the distances are.
* **Decisions.** Distances are computed coordinate by coordinate, never as a
  matrix product, so neither cuBLAS's choice of algorithm nor TF32 can enter
  and the value of a cell does not depend on the shape of the block it was
  computed in. In double precision a cell can disagree with the reference
  (which expands the norm, DD-33) only if its distance is within rounding of
  epsilon; DD-122 keeps every sampled pair at least 1e-9 away, and the cells
  within ``NEAR_RTOL`` are counted and reported.
* **Counting.** Line lengths are found with differences and counted with
  integer ``bincount``: no floating-point reduction, no atomic float addition,
  so the counts cannot depend on scheduling, batch size, band height or
  device.

``precision="single"`` decides in float32 and reports how many cells were
within float32 rounding of epsilon; ``"mixed"`` decides in float32 and
recomputes those cells in float64, which restores the double-precision
decision at a fraction of its cost on cards whose float64 is slow.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..exceptions import MemoryBudgetError, ParameterError

PRECISIONS = ("double", "single", "mixed")
METRICS = ("euclidean", "chebyshev", "manhattan")

#: Cells within this relative distance of epsilon (in the compared units) are
#: counted: there, another formula for the distance could decide otherwise.
NEAR_RTOL = 1e-12

#: In single precision, cells within this relative distance of epsilon are
#: counted, and with ``precision="mixed"`` recomputed in double. Float32
#: rounding of a squared distance between centred coordinates is a few parts
#: in 10^6 of epsilon squared at the rates used in practice; this is 25 times
#: that. It is the floor: a window whose coordinates reach far from its centre
#: compared with epsilon gets a wider band (``_single_rtol``), so that no cell
#: outside the band can be decided differently in float32 and float64.
SINGLE_RTOL = 1e-4

#: Bytes per cell of a band, all temporaries included, used to fit batches in
#: the declared device memory. Measured on an RTX 3090 with an 8 GB budget
#: and the earlier 48 and 36 bytes: the peak was about 5.3 GB in double and
#: 3.5 GB in mixed, that is about 32 and 16 bytes per cell. Single keeps the
#: same float32 temporaries as mixed. The values keep a margin over those.
BYTES_PER_CELL = {"double": 36, "single": 24, "mixed": 24}


#: Set by ``tools/benchmark_engine.py --profile``: time each stage of the
#: engine, waiting for the device at every boundary (which costs a little
#: speed), and record the seconds in the provenance. Not part of the API.
PROFILE = False


class _Stages:
    """Seconds per stage when ``PROFILE`` is on; nothing at all otherwise."""

    def __init__(self, dev):
        self.on = PROFILE
        self.dev = dev
        self.seconds: dict[str, float] = {}
        self._last = 0.0
        if self.on:
            self._sync()
            self._last = time.perf_counter()

    def _sync(self):
        if self.dev.type == "cuda":
            _torch().cuda.synchronize(self.dev)

    def mark(self, name: str) -> None:
        if not self.on:
            return
        self._sync()
        now = time.perf_counter()
        self.seconds[name] = self.seconds.get(name, 0.0) + now - self._last
        self._last = now


def _torch():
    import torch

    return torch


def prepare_torch():
    """Switch off what could make the device disagree with itself [DD-125].

    TF32 only affects matrix products and convolutions, which this engine does
    not use, but it is switched off anyway and recorded, so that no later
    change can bring it in unnoticed.
    """
    torch = _torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


@dataclass
class BatchPlan:
    """How a batch of windows is cut to fit the device."""

    batch_size: int
    band_rows: int
    cells_budget: int


def plan_batches(n_windows: int, n_points: int, *, precision: str,
                 device_memory_gb: float, batch_size: int | None = None) -> BatchPlan:
    """Windows per batch and rows per band, from the declared memory (P7)."""
    budget = int(device_memory_gb * 1e9 / BYTES_PER_CELL[precision])
    row_cells = n_points                      # one row of one window
    if row_cells > budget:
        raise MemoryBudgetError(
            f"one row of a {n_points}-point window needs about "
            f"{row_cells * BYTES_PER_CELL[precision] / 1e9:.2f} GB on the device; "
            f"the budget is {device_memory_gb} GB [DD-34, DD-125]")
    whole = n_points * n_points
    if batch_size is None:
        batch_size = max(1, min(n_windows, budget // whole)) if whole <= budget else 1
    batch_size = max(1, min(int(batch_size), n_windows))
    rows = max(1, min(n_points, budget // (batch_size * n_points)))
    return BatchPlan(batch_size=batch_size, band_rows=rows, cells_budget=budget)


# ---------------------------------------------------------------- thresholds
def sampled_values(X, i, j, *, metric: str):
    """What the threshold compares, for pairs ``(i, j)`` of every window.

    ``X`` is (B, N, d). Coordinate by coordinate, in order, with the same
    elementwise operations as ``pairwise_rows``. For the euclidean metric the
    *squared* distances are returned: the square root is taken on the CPU, by
    numpy, on the few values that decide epsilon, because a device's square
    root need not be correctly rounded (PyTorch's on the CPU is not), and the
    order of the values is the same either way.
    """
    torch = _torch()
    A = X[:, i, :]
    C = X[:, j, :]
    acc = None
    for k in range(X.shape[2]):
        diff = A[:, :, k] - C[:, :, k]
        term = diff * diff if metric == "euclidean" else torch.abs(diff)
        if acc is None:
            acc = term
        elif metric == "chebyshev":
            acc = torch.maximum(acc, term)
        else:
            acc = acc + term
    return acc


def _to_distance(values: np.ndarray, metric: str) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return np.sqrt(v) if metric == "euclidean" else v


def _exact_counts(S, eps: np.ndarray, metric: str) -> np.ndarray:
    """How many sampled distances are ``<= eps``, per window, exactly.

    ``S`` holds the sorted compared values (squared distances for euclidean).
    The device narrows the boundary to the values within 1e-12 of epsilon in
    the compared units; those few are converted by numpy and compared there.
    """
    torch = _torch()
    B, k = S.shape
    e = np.asarray(eps, dtype=float)
    e_op = e * e if metric == "euclidean" else e
    lo_v = torch.as_tensor(e_op * (1 - 1e-12), dtype=S.dtype, device=S.device).reshape(B, 1)
    hi_v = torch.as_tensor(e_op * (1 + 1e-12), dtype=S.dtype, device=S.device).reshape(B, 1)
    lo = torch.searchsorted(S, lo_v, right=False).reshape(B).cpu().numpy()
    hi = torch.searchsorted(S, hi_v, right=True).reshape(B).cpu().numpy()
    out = lo.astype(np.int64)
    for w in np.flatnonzero(hi > lo):
        seg = S[w, int(lo[w]):int(hi[w])].cpu().numpy()
        out[w] += int(np.count_nonzero(_to_distance(seg, metric) <= e[w]))
    return out


def _at(S, pos: np.ndarray, metric: str) -> np.ndarray:
    """Sorted distances at positions ``pos`` (one per window), via numpy."""
    torch = _torch()
    B, k = S.shape
    idx = torch.as_tensor(np.clip(pos, 0, k - 1), dtype=torch.int64,
                          device=S.device).reshape(B, 1)
    return _to_distance(torch.gather(S, 1, idx).reshape(B).cpu().numpy(), metric)


def quantile_thresholds(V, q: float, *, metric: str = "euclidean"):
    """Epsilon per window from sampled values ``V`` (B, k), as numpy would.

    The device sorts; numpy does the arithmetic on the few values that decide
    epsilon -- the two order statistics of its linear quantile, the
    neighbours of epsilon for the move of DD-122 and the count for the
    achieved rate -- so the result is numpy's to the last bit.
    """
    from ..threshold.estimate import TIE_RTOL, _gap_midpoint

    torch = _torch()
    B, k = V.shape
    S, _ = torch.sort(V, dim=1)
    # numpy's 'linear' quantile: virtual index (n - 1) q, floor, lerp
    virtual = (k - 1) * np.asarray(q, dtype=float)
    if virtual >= k - 1:
        prev = nxt = k - 1
    else:
        prev = int(np.floor(virtual))
        nxt = prev + 1
    gamma = np.asarray(virtual - prev, dtype=float)
    a = _at(S, np.full(B, prev), metric)
    b = _at(S, np.full(B, nxt), metric)
    diff = b - a
    eps0 = a + diff * gamma
    if gamma >= 0.5:
        eps0 = b - diff * (1 - gamma)
    eps0 = np.asarray(eps0, dtype=float)

    pos = _exact_counts(S, eps0, metric)                 # sorted[:pos] <= eps0
    lower = _at(S, pos - 1, metric)
    upper = _at(S, pos, metric)
    eps = eps0.copy()
    moved = np.zeros(B, dtype=bool)
    for w in range(B):
        e = float(eps0[w])
        if not math.isfinite(e):
            continue
        kk = int(pos[w])
        tol = TIE_RTOL * abs(e)
        near_below = kk > 0 and e - float(lower[w]) <= tol
        near_above = kk < k and float(upper[w]) - e <= tol
        if not (near_below or near_above):
            continue
        moved[w] = True
        if kk == 0:
            eps[w] = 0.5 * e
        else:
            eps[w] = _gap_midpoint(float(lower[w]), float(upper[w]) if kk < k else None)
    counts = np.where(moved, _exact_counts(S, eps, metric) if moved.any() else pos, pos)
    achieved = counts / k
    return eps, achieved, moved


def fixed_rates(V, eps: float, *, metric: str = "euclidean") -> np.ndarray:
    """Achieved rate of a fixed epsilon on each window's sampled values."""
    torch = _torch()
    S, _ = torch.sort(V, dim=1)
    B, k = S.shape
    return _exact_counts(S, np.full(B, float(eps)), metric) / k


def _single_rtol(Xw: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Per window, a relative band around epsilon that contains every cell
    whose float32 decision could differ from its float64 one.

    Rounding the coordinates to float32 moves each by at most ``u |x|``
    (``u = 2**-24``), so a difference moves by at most ``u (|x| + |y| + d)``
    and a distance near epsilon, summed over ``dim`` coordinates, by about
    ``2 u dim (2 R + eps)``, with ``R`` the largest centred coordinate. The
    band is eight times that, relative to epsilon, and never narrower than
    ``SINGLE_RTOL``.
    """
    u = 2.0 ** -24
    dim = Xw.shape[2]
    R = np.max(np.abs(Xw.reshape(Xw.shape[0], -1)), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(eps > 0, R / eps, np.inf)
    return np.maximum(SINGLE_RTOL, 16.0 * u * dim * (2.0 * ratio + 1.0))


# ----------------------------------------------------------------- decisions
def _decide(A, C, eps_op, *, metric: str, i0: int, theiler: int, banded: bool):
    """Recurrent cells of one band: rows ``A`` (B, h, d) against ``C`` (B, m, d).

    ``eps_op`` is epsilon in the compared units, shape (B, 1, 1). Returns the
    boolean band and the raw compared values (for the near-epsilon counts).
    """
    torch = _torch()
    acc = None
    for k in range(A.shape[2]):
        diff = A[:, :, None, k] - C[:, None, :, k]
        if metric == "euclidean":
            term = diff * diff
        else:
            term = torch.abs(diff)
        if acc is None:
            acc = term
        elif metric == "chebyshev":
            acc = torch.maximum(acc, term)
        else:
            acc = acc + term
    R = acc <= eps_op
    if banded:
        h, m = R.shape[1], R.shape[2]
        rows = torch.arange(i0, i0 + h, device=R.device).reshape(h, 1)
        cols = torch.arange(m, device=R.device).reshape(1, m)
        R &= (rows - cols).abs() >= theiler
    return R, acc


# ------------------------------------------------------- fused decisions
#: Tests set this to try the fused decisions on the CPU, where torch.compile
#: generates C++ instead of GPU kernels; normal runs fuse only on CUDA.
_FUSE_ON_CPU = False
_DECIDERS: dict = {}
_FUSE_WARNED: set = set()


def _fused_kernel(metric: str, dim: int):
    """The float32 pass as one function for torch.compile: distance, decision
    and the cells near epsilon, without the intermediate arrays [DD-127]."""
    torch = _torch()

    def decide(A, C, eps, rtol, i0: int, theiler: int):
        acc = None
        for k in range(dim):
            diff = A[:, :, None, k] - C[:, None, :, k]
            term = diff * diff if metric == "euclidean" else torch.abs(diff)
            if acc is None:
                acc = term
            elif metric == "chebyshev":
                acc = torch.maximum(acc, term)
            else:
                acc = acc + term
        R = acc <= eps
        amb = (acc - eps).abs() <= rtol * eps
        if theiler > 0:
            h, m = A.shape[1], C.shape[1]
            rows = torch.arange(h, device=A.device).reshape(h, 1) + i0
            cols = torch.arange(m, device=A.device).reshape(1, m)
            keep = (rows - cols).abs() >= theiler
            R = R & keep
            amb = amb & keep
        return R, amb

    return decide


class _FusedDecider:
    """The float32 decisions of one metric and dimension, compiled once.

    If compiling or running the compiled function fails (no Triton for this
    card or platform, an old driver), it says so once and decides as before,
    with the same numbers: fusing changes the speed, never a decision outside
    the band of cells that are recomputed in float64 anyway.
    """

    def __init__(self, metric: str, dim: int):
        self.metric = metric
        self.plain = _fused_kernel(metric, dim)
        self.ok = True
        self.reason = ""
        try:
            self.compiled = _torch().compile(self.plain, dynamic=True, fullgraph=True)
        except Exception as exc:                      # pragma: no cover - platform
            self._fail(exc)

    def _fail(self, exc) -> None:
        import warnings

        from ..exceptions import CaveatWarning

        self.ok = False
        self.reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        if self.reason not in _FUSE_WARNED:
            _FUSE_WARNED.add(self.reason)
            warnings.warn(
                "torch engine: the fused float32 decisions could not be compiled "
                f"({self.reason}); deciding without fusion, which is slower and gives "
                "the same numbers. Set recurra.set_config(gpu_compile=False) to skip "
                "the attempt [DD-127].", CaveatWarning, stacklevel=4)

    def decide(self, A, C, eps, rtol, *, i0: int, theiler: int):
        if self.ok:
            try:
                return self.compiled(A, C, eps, rtol, i0, theiler)
            except Exception as exc:                  # pragma: no cover - platform
                self._fail(exc)
        return self.plain(A, C, eps, rtol, i0, theiler)


def _fused_decider(metric: str, dim: int, dev):
    """The compiled float32 decider for this job, or None to decide unfused."""
    from ..config import get_config

    if not getattr(get_config(), "gpu_compile", True):
        return None
    if dev.type != "cuda" and not _FUSE_ON_CPU:
        return None
    key = (metric, dim)
    if key not in _DECIDERS:
        _DECIDERS[key] = _FusedDecider(metric, dim)
    return _DECIDERS[key]


# ---------------------------------------------------------------- run lengths
def _runs_by_row(X):
    """Every run of True along the rows of a 2-D boolean tensor.

    Each row is padded with False at both ends, so its changes alternate
    start, end, start, end; one search for the changes finds both.
    """
    torch = _torch()
    n_lines, h = X.shape
    P = torch.zeros((n_lines, h + 2), dtype=torch.int8, device=X.device)
    P[:, 1:-1] = X
    changes = torch.nonzero(torch.diff(P, dim=1).reshape(-1)).reshape(-1)
    s = changes[0::2]
    e = changes[1::2]
    row = torch.div(s, h + 1, rounding_mode="floor")
    return row, s - row * (h + 1), e - row * (h + 1)


def _count(counts, lengths, batch_of, size1: int) -> None:
    torch = _torch()
    if lengths.numel():
        idx = lengths + batch_of * size1
        counts += torch.bincount(idx, minlength=counts.numel())[:counts.numel()]


def _carry_runs(X, keys, line_batch, open_len, counts, size1: int) -> None:
    """Join the runs of a band to those the previous band left open [DD-56]."""
    torch = _torch()
    n_lines, h = X.shape
    row, start, end = _runs_by_row(X)
    length = end - start
    carried = open_len[keys]
    top = start == 0
    length[top] += carried[row[top]]
    continued = torch.zeros(n_lines, dtype=torch.bool, device=X.device)
    continued[row[top]] = True
    bottom = end == h
    done = ~bottom
    _count(counts, length[done], line_batch[row[done]], size1)
    closed = (carried > 0) & ~continued
    _count(counts, carried[closed], line_batch[closed], size1)
    still = torch.zeros(n_lines, dtype=torch.int64, device=X.device)
    still[row[bottom]] = length[bottom]
    open_len[keys] = still


def _shear_upper(R, j0: int):
    """The diagonals of a band that start in its first row at column ``j0`` or
    to its right, as columns: (B, h, m) -> (B, h, m - j0).

    Column ``c`` holds the cells ``(r, j0 + r + c)`` for ``r = 0 .. h-1``, and
    False past the right edge of the plot; it is a strided view of one padded
    copy of the right part of the band.
    """
    torch = _torch()
    B, h, m = R.shape
    w = m - j0
    L = w + h - 1
    P = torch.zeros((B, h, L), dtype=R.dtype, device=R.device)
    P[:, :, :w] = R[:, :, j0:]
    return torch.as_strided(P, size=(B, h, w), stride=(h * L, L + 1, 1))


#: States of a cell in the row search: recurrent, not recurrent, inside the
#: Theiler band, and the padding past either end of a row.
_WHITE, _BLACK, _BAND, _EDGE = 0, 1, 2, 3


def _row_runs(R, i0: int, theiler: int):
    """The vertical and the white vertical lines of a band, in one search.

    Each cell of each row gets one of three states -- recurrent, not
    recurrent, in the Theiler band -- and each row is padded with a fourth
    state at both ends, so every row starts and ends with a change. One
    search for the changes then gives every run of every state: consecutive
    changes in the same row bound one run, whose state is that of its first
    cell. Runs of recurrent cells are the vertical lines (column ``j`` is row
    ``j``, DD-127); runs of non-recurrent cells are white vertical lines unless
    the cell before or after them lies in the Theiler band [DD-57].

    Returns ``(lengths, window)`` for the vertical lines and for the white
    ones.
    """
    torch = _torch()
    B, h, m = R.shape
    width = m + 2
    P = torch.full((B, h, width), _EDGE, dtype=torch.int8, device=R.device)
    P[:, :, 1:-1] = R
    if theiler > 0:
        rows = torch.arange(i0, i0 + h, device=R.device).reshape(h, 1)
        cols = torch.arange(m, device=R.device).reshape(1, m)
        P[:, :, 1:-1] += (_BAND * ((rows - cols).abs() < theiler)).to(torch.int8)
    flat = P.reshape(B * h, width)
    changes = torch.nonzero(torch.diff(flat, dim=1).reshape(-1)).reshape(-1)
    line = torch.div(changes, width - 1, rounding_mode="floor")
    pos = changes - line * (width - 1)
    run = line[:-1] == line[1:]                    # not the step between two rows
    line, start, end = line[:-1][run], pos[:-1][run], pos[1:][run]
    cells = P.reshape(-1)
    base = line * width
    state = cells[base + start + 1]
    window = torch.div(line, h, rounding_mode="floor")
    black = state == _BLACK
    white = state == _WHITE
    if theiler > 0:
        white &= (cells[base + start] != _BAND) & (cells[base + end + 1] != _BAND)
    length = end - start
    return (length[black], window[black]), (length[white], window[white])


def batch_line_histograms(Xw, eps, *, theiler: int, metric: str, precision: str,
                          device: str, band_rows: int, stages: _Stages | None = None):
    """Line histograms of one batch of recurrence plots on the device.

    ``Xw`` is (B, N, d), float64, already centred as the reference centres it;
    ``eps`` the B thresholds. Returns the counts as numpy arrays and the
    diagnostics (cells near epsilon, cells recomputed).

    Design note (DD-127) -- the lines are found using the symmetry of the plot.

    Cell ``(i, j)`` and cell ``(j, i)`` are decided from the same numbers by
    the same operations (a difference changes sign exactly, its square or
    absolute value does not, and the coordinates are added in the same order),
    so every plot here is symmetric to the last bit, and so is the Theiler
    band. Two consequences remove most of the copying the line search did:

    * column ``j`` is row ``j``, so the vertical lines and the white vertical
      lines are the runs along the *rows*, each of which lies whole inside one
      band: no transposed copy and nothing carried from band to band;
    * the diagonal at offset ``+k`` is the one at ``-k``, so only the upper
      half is searched and its counts are doubled; the main diagonal, when the
      Theiler window is 0, is searched once.
    """
    torch = prepare_torch()
    if metric not in METRICS:
        raise ParameterError(
            f"the torch engine supports the metrics {METRICS}; {metric!r} needs "
            "engine='numpy'")
    if precision not in PRECISIONS:
        raise ParameterError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    dev = torch.device(device)
    stages = stages or _Stages(dev)
    B, n, d = Xw.shape
    m = n
    size1 = n + 1
    X64 = torch.as_tensor(np.ascontiguousarray(Xw), dtype=torch.float64, device=dev)
    Xc = X64 if precision == "double" else X64.to(torch.float32)
    eps = np.asarray(eps, dtype=float)
    eps_np = eps * eps if metric == "euclidean" else eps
    eps64 = torch.as_tensor(eps_np, dtype=torch.float64, device=dev).reshape(B, 1, 1)
    eps_c = eps64 if precision == "double" else eps64.to(torch.float32)
    rtol32 = torch.as_tensor(_single_rtol(np.asarray(Xw, dtype=float), eps),
                             dtype=torch.float32, device=dev).reshape(B, 1, 1)
    banded = theiler > 0
    fused = _fused_decider(metric, d, dev) if precision != "double" else None
    stages.mark("centre_and_copy_to_device")

    diag_upper = torch.zeros(B * size1, dtype=torch.int64, device=dev)
    diag_main = torch.zeros(B * size1, dtype=torch.int64, device=dev)
    vert = torch.zeros(B * size1, dtype=torch.int64, device=dev)
    white = torch.zeros(B * size1, dtype=torch.int64, device=dev)
    diag_open = torch.zeros(B * n, dtype=torch.int64, device=dev)
    main_open = torch.zeros(B, dtype=torch.int64, device=dev)
    window_ids = torch.arange(B, dtype=torch.int64, device=dev)
    recurrent = torch.zeros(B, dtype=torch.int64, device=dev)
    kmin = max(1, theiler)                # first diagonal of the upper half
    near = 0
    rechecked = 0

    for i0 in range(0, n, band_rows):
        i1 = min(i0 + band_rows, n)
        h = i1 - i0
        if fused is not None:
            R, amb = fused.decide(Xc[:, i0:i1, :], Xc, eps_c, rtol32, i0=i0,
                                  theiler=theiler)
            acc = None
        else:
            R, acc = _decide(Xc[:, i0:i1, :], Xc, eps_c, metric=metric, i0=i0,
                             theiler=theiler, banded=banded)
        if precision == "double":
            near += int(((acc - eps64).abs() <= NEAR_RTOL * eps64).sum())
        else:
            if acc is not None:
                amb = (acc - eps_c).abs() <= rtol32 * eps_c
                if banded:
                    rows = torch.arange(i0, i1, device=dev).reshape(h, 1)
                    cols = torch.arange(m, device=dev).reshape(1, m)
                    amb &= (rows - cols).abs() >= theiler
            n_amb = int(amb.sum())
            near += n_amb
            if precision == "mixed" and n_amb:
                b_i, r_i, c_i = torch.nonzero(amb, as_tuple=True)
                A64 = X64[b_i, i0 + r_i, :]
                C64 = X64[b_i, c_i, :]
                acc64 = None
                for k in range(d):
                    diff = A64[:, k] - C64[:, k]
                    term = diff * diff if metric == "euclidean" else diff.abs()
                    if acc64 is None:
                        acc64 = term
                    elif metric == "chebyshev":
                        acc64 = torch.maximum(acc64, term)
                    else:
                        acc64 = acc64 + term
                R[b_i, r_i, c_i] = acc64 <= eps64.reshape(B)[b_i]
                rechecked += n_amb
            del amb
        del acc
        stages.mark("decide")
        recurrent += R.reshape(B, -1).sum(dim=1)

        # vertical and white vertical lines: column j is row j, so they are
        # the runs along the rows, all found in one search
        (v_len, v_of), (w_len, w_of) = _row_runs(R, i0, theiler)
        _count(vert, v_len, v_of, size1)
        _count(white, w_len, w_of, size1)
        stages.mark("vertical_and_white")

        # diagonals k >= kmin from the upper half, carried between bands; the
        # main diagonal on its own when the Theiler window is 0
        j0 = i0 + kmin
        if j0 < m:
            w = m - j0
            ST = _shear_upper(R, j0).transpose(1, 2).reshape(B * w, h)
            b_of = window_ids.repeat_interleave(w)
            keys = b_of * n + (kmin + torch.arange(w, dtype=torch.int64,
                                                   device=dev)).repeat(B)
            _carry_runs(ST, keys, b_of, diag_open, diag_upper, size1)
        if theiler == 0:
            M = torch.diagonal(R, offset=i0, dim1=1, dim2=2).contiguous()
            _carry_runs(M, window_ids, window_ids, main_open, diag_main, size1)
        stages.mark("diagonal")

        del R

    # flush: diagonal runs reaching the end of the plot are kept [DD-57]
    open_d = diag_open > 0
    _count(diag_upper, diag_open[open_d],
           torch.div(torch.arange(B * n, device=dev), n, rounding_mode="floor")[open_d],
           size1)
    open_m = main_open > 0
    _count(diag_main, main_open[open_m], window_ids[open_m], size1)
    diag = 2 * diag_upper + diag_main

    out = {
        "diagonal": diag.reshape(B, size1).cpu().numpy(),
        "vertical": vert.reshape(B, size1).cpu().numpy(),
        "white_vertical": white.reshape(B, size1).cpu().numpy(),
        "recurrent": recurrent.cpu().numpy(),
        "near_epsilon": near,
        "rechecked": rechecked,
        "fused_decisions": fused is not None and fused.ok,
    }
    stages.mark("flush_and_copy_back")
    return out


# ---------------------------------------------------------------- the driver
def run_batches(ws, seeds, *, device, precision, batch_size, device_memory_gb,
                thresholds, hists, target_rr, epsilon, theiler, metric, n_samples):
    """Thresholds and histograms of every window, in batches of equal length.

    Fills ``thresholds`` and ``hists`` in window order; returns what the
    provenance should record.
    """
    import warnings

    from ..exceptions import GeometryWarning
    from ..recurrence.core import theiler_band_cells
    from ..rqa.histogram import LineHistogram
    from ..threshold.core import Threshold
    from ..threshold.sampling import sample_pair_indices

    torch = prepare_torch()
    if metric not in METRICS:
        raise ParameterError(
            f"the torch engine supports the metrics {METRICS}; {metric!r} needs "
            "engine='numpy'")
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    groups: dict[tuple[int, int], list[int]] = {}
    for k, w in enumerate(ws):
        groups.setdefault(w.shape, []).append(k)

    near = rechecked = 0
    fused: list[bool] = []
    plans = {}
    rounding = 0
    stages = _Stages(dev)
    for (n, d), members in sorted(groups.items()):
        plan = plan_batches(len(members), n, precision=precision,
                            device_memory_gb=device_memory_gb, batch_size=batch_size)
        plans[n] = plan
        if epsilon is None:
            i, j = sample_pair_indices(n, n_samples=n_samples, theiler=theiler,
                                       rng=seeds[n])
        else:
            i, j = sample_pair_indices(n, n_samples=min(n_samples, 50_000),
                                       theiler=theiler, rng=seeds[n])
        if not i.size:
            raise ParameterError("could not sample any admissible pair of points")
        i_dev = torch.as_tensor(i, device=dev)
        j_dev = torch.as_tensor(j, device=dev)
        considered = n * n - theiler_band_cells(n, n, theiler)
        # sampled distances take B x k x d values twice; keep them near 1 GB
        sub = max(1, int(1.2e8 // max(1, i.size * d)))

        for b0 in range(0, len(members), plan.batch_size):
            chunk = members[b0:b0 + plan.batch_size]
            raw = np.stack([ws[k] for k in chunk])
            eps = np.empty(len(chunk))
            achieved = np.empty(len(chunk))
            moved = np.zeros(len(chunk), dtype=bool)
            for s0 in range(0, len(chunk), sub):
                part = torch.as_tensor(raw[s0:s0 + sub], dtype=torch.float64, device=dev)
                V = sampled_values(part, i_dev, j_dev, metric=metric)
                if epsilon is None:
                    e, a, mv = quantile_thresholds(V, float(target_rr), metric=metric)
                else:
                    e = np.full(V.shape[0], float(epsilon))
                    a = fixed_rates(V, float(epsilon), metric=metric)
                    mv = np.zeros(V.shape[0], dtype=bool)
                eps[s0:s0 + sub], achieved[s0:s0 + sub], moved[s0:s0 + sub] = e, a, mv
                del V, part
            stages.mark("threshold")
            # centred as the reference centres each plot [DD-110]
            centred = np.stack([w - w.mean(axis=0) for w in raw])
            out = batch_line_histograms(centred, eps, theiler=theiler, metric=metric,
                                        precision=precision, device=device,
                                        band_rows=plan.band_rows, stages=stages)
            near += out["near_epsilon"]
            rechecked += out["rechecked"]
            fused.append(out["fused_decisions"])
            for b, k in enumerate(chunk):
                extent = float(np.max(np.ptp(raw[b], axis=0))) if raw[b].size else 0.0
                if epsilon is None and extent > 0 and eps[b] <= 1e-9 * extent:
                    rounding += 1
                diag = ({"criterion": "user-specified epsilon"} if epsilon is not None
                        else {"criterion": f"quantile {float(target_rr):g} of the "
                                           "sampled distance distribution",
                              "iterations": 0})
                if moved[b]:
                    diag["moved_off_a_sampled_distance"] = True
                thresholds[k] = Threshold(
                    value=float(eps[b]), mode="target_rr" if epsilon is None else "fixed",
                    metric=metric, scope="global", theiler=theiler,
                    target=None if epsilon is not None else float(target_rr),
                    achieved_rr=float(achieved[b]), n_samples=int(i.size),
                    diagnostics=diag)
                hists[k] = LineHistogram(
                    diagonal=out["diagonal"][b], vertical=out["vertical"][b],
                    white_vertical=out["white_vertical"][b],
                    recurrent_points=int(out["recurrent"][b]),
                    considered_points=int(considered), n_rows=n, n_cols=n,
                    theiler=theiler,
                    diagnostics={"engine": "torch", "device": device,
                                 "band_rows": plan.band_rows})
            stages.mark("results")
    if rounding:
        warnings.warn(
            f"batch_rqa: in {rounding} window(s) epsilon is at rounding level for the "
            "size of the attractor: more than the target share of pairs coincide "
            "exactly (a periodic or quantised series) [DD-118].",
            GeometryWarning, stacklevel=3)
    return {"precision": precision,
            "batch_size": {n: p.batch_size for n, p in plans.items()},
            "band_rows": {n: p.band_rows for n, p in plans.items()},
            "device_memory_gb": device_memory_gb,
            "peak_device_memory_gb": _peak_gb(dev),
            "cells_near_epsilon": int(near),
            "cells_rechecked_in_double": int(rechecked),
            **_fused_provenance(fused),
            "deterministic_counts": True,
            **({"stage_seconds": {k: round(v, 4) for k, v in stages.seconds.items()}}
               if stages.on else {})}


def _fused_provenance(fused: list[bool]) -> dict:
    """Whether the float32 decisions ran fused, and if not why [DD-127]."""
    reasons = sorted({d.reason for d in _DECIDERS.values() if not d.ok})
    info = {"fused_decisions": bool(fused) and all(fused)}
    if reasons:
        info["fused_decisions_failed"] = reasons
    return info


def _peak_gb(dev) -> float | None:
    """Peak memory the device allocated during the call, to check the plan."""
    torch = _torch()
    if dev.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(dev) / 1e9, 3)


def device_histograms(items, *, device, precision, device_memory_gb,
                      batch_size=None):
    """Line histograms of recurrence plots whose thresholds are already fixed.

    ``items`` are ``(key, centred coordinates, epsilon, theiler, metric)``;
    plots of the same shape, window and metric are batched together. Returns
    ``({key: LineHistogram}, provenance)``.
    """
    from ..recurrence.core import theiler_band_cells
    from ..rqa.histogram import LineHistogram

    torch = prepare_torch()
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    groups: dict[tuple, list] = {}
    for key, X, eps, theiler, metric in items:
        groups.setdefault((X.shape, theiler, metric), []).append((key, X, eps))
    out = {}
    near = rechecked = 0
    fused: list[bool] = []
    plans = {}
    for ((n, _d), theiler, metric), members in sorted(groups.items(),
                                                      key=lambda g: (g[0][0], g[0][1])):
        plan = plan_batches(len(members), n, precision=precision,
                            device_memory_gb=device_memory_gb, batch_size=batch_size)
        plans[n] = plan
        considered = n * n - theiler_band_cells(n, n, theiler)
        for b0 in range(0, len(members), plan.batch_size):
            chunk = members[b0:b0 + plan.batch_size]
            res = batch_line_histograms(
                np.stack([X for _, X, _ in chunk]),
                np.array([e for _, _, e in chunk]), theiler=theiler, metric=metric,
                precision=precision, device=device, band_rows=plan.band_rows)
            near += res["near_epsilon"]
            rechecked += res["rechecked"]
            fused.append(res["fused_decisions"])
            for b, (key, _, _) in enumerate(chunk):
                out[key] = LineHistogram(
                    diagonal=res["diagonal"][b], vertical=res["vertical"][b],
                    white_vertical=res["white_vertical"][b],
                    recurrent_points=int(res["recurrent"][b]),
                    considered_points=int(considered), n_rows=n, n_cols=n,
                    theiler=theiler,
                    diagnostics={"engine": "torch", "device": device,
                                 "band_rows": plan.band_rows})
    return out, {"precision": precision,
                 "batch_size": {n: p.batch_size for n, p in plans.items()},
                 "band_rows": {n: p.band_rows for n, p in plans.items()},
                 "device_memory_gb": device_memory_gb,
                 "peak_device_memory_gb": _peak_gb(dev),
                 "cells_near_epsilon": int(near),
                 "cells_rechecked_in_double": int(rechecked),
                 **_fused_provenance(fused)}
