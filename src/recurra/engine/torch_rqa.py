"""Thresholds, recurrence plots and line histograms on a PyTorch device.

Design note (DD-125) -- the same answer as the reference, by construction.

The torch engine runs the band algorithm of DD-124 on a batch of windows of
equal length at once: a band of every window of the batch is decided,
differenced and counted in one set of array operations. Three rules make its
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
#: that.
SINGLE_RTOL = 1e-4

#: Bytes per cell of a band, all temporaries included, used to fit batches in
#: the declared device memory. Counted from the temporaries of one band (in
#: double: the difference, its square and the running sum, 8 bytes each, the
#: near-epsilon check, the boolean band, the padded and sheared copies of the
#: run search), with a margin; to be confirmed against the device's own
#: memory statistics.
BYTES_PER_CELL = {"double": 48, "single": 32, "mixed": 36}


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


# ---------------------------------------------------------------- run lengths
def _runs_by_row(X):
    """Every run of True along the rows of a 2-D boolean tensor."""
    torch = _torch()
    n_lines, h = X.shape
    P = torch.zeros((n_lines, h + 2), dtype=torch.int8, device=X.device)
    P[:, 1:-1] = X
    d = torch.diff(P, dim=1).reshape(-1)
    s = torch.nonzero(d == 1).reshape(-1)
    e = torch.nonzero(d == -1).reshape(-1)
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


def _shear(R):
    """(B, h, m) -> (B, h, m + h - 1): diagonal k of the band becomes column
    ``k + h - 1 - i0``, through a strided view of a padded copy."""
    torch = _torch()
    B, h, m = R.shape
    L = m + 2 * (h - 1)
    P = torch.zeros((B, h, L), dtype=R.dtype, device=R.device)
    P[:, :, h - 1:h - 1 + m] = R
    return torch.as_strided(P, size=(B, h, m + h - 1), stride=(h * L, L + 1, 1))


def batch_line_histograms(Xw, eps, *, theiler: int, metric: str, precision: str,
                          device: str, band_rows: int):
    """Line histograms of one batch of recurrence plots on the device.

    ``Xw`` is (B, N, d), float64, already centred as the reference centres it;
    ``eps`` the B thresholds. Returns the counts as numpy arrays and the
    diagnostics (cells near epsilon, cells recomputed).
    """
    torch = prepare_torch()
    if metric not in METRICS:
        raise ParameterError(
            f"the torch engine supports the metrics {METRICS}; {metric!r} needs "
            "engine='numpy'")
    if precision not in PRECISIONS:
        raise ParameterError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    dev = torch.device(device)
    B, n, d = Xw.shape
    m = n
    size1 = n + 1
    X64 = torch.as_tensor(np.ascontiguousarray(Xw), dtype=torch.float64, device=dev)
    Xc = X64 if precision == "double" else X64.to(torch.float32)
    eps = np.asarray(eps, dtype=float)
    eps_np = eps * eps if metric == "euclidean" else eps
    eps64 = torch.as_tensor(eps_np, dtype=torch.float64, device=dev).reshape(B, 1, 1)
    eps_c = eps64 if precision == "double" else eps64.to(torch.float32)
    banded = theiler > 0

    diag = torch.zeros(B * size1, dtype=torch.int64, device=dev)
    vert = torch.zeros(B * size1, dtype=torch.int64, device=dev)
    white = torch.zeros(B * size1, dtype=torch.int64, device=dev)
    n_keys = n + m + 1
    diag_open = torch.zeros(B * n_keys, dtype=torch.int64, device=dev)
    vert_open = torch.zeros(B * m, dtype=torch.int64, device=dev)
    white_open = torch.zeros(B * m, dtype=torch.int64, device=dev)
    white_flag = torch.zeros(B * m, dtype=torch.bool, device=dev)
    col_keys = torch.arange(B * m, dtype=torch.int64, device=dev)
    col_batch = torch.div(col_keys, m, rounding_mode="floor")
    col_index = col_keys - col_batch * m
    recurrent = torch.zeros(B, dtype=torch.int64, device=dev)
    near = 0
    rechecked = 0

    def excluded(rows, cols):
        if not banded:
            return torch.zeros(rows.shape, dtype=torch.bool, device=dev)
        return (rows >= 0) & (rows < n) & ((rows - cols).abs() < theiler)

    for i0 in range(0, n, band_rows):
        i1 = min(i0 + band_rows, n)
        h = i1 - i0
        R, acc = _decide(Xc[:, i0:i1, :], Xc, eps_c, metric=metric, i0=i0,
                         theiler=theiler, banded=banded)
        if precision == "double":
            near += int(((acc - eps64).abs() <= NEAR_RTOL * eps64).sum())
        else:
            amb = (acc - eps_c).abs() <= SINGLE_RTOL * eps_c
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
        del acc
        recurrent += R.reshape(B, -1).sum(dim=1)

        # verticals: each column of each window is a line
        RT = R.transpose(1, 2).reshape(B * m, h)
        _carry_runs(RT, col_keys, col_batch, vert_open, vert, size1)

        # diagonals, sheared into columns
        S = _shear(R)
        q = m + h - 1
        ST = S.transpose(1, 2).reshape(B * q, h)
        k_local = torch.arange(q, dtype=torch.int64, device=dev) - (h - 1) - i0 + n
        b_of = torch.arange(B, dtype=torch.int64, device=dev).repeat_interleave(q)
        keys = b_of * n_keys + k_local.repeat(B)
        _carry_runs(ST, keys, b_of, diag_open, diag, size1)

        # white verticals, the Theiler band as a third state [DD-57]
        W = ~R
        if banded:
            rows = torch.arange(i0, i1, device=dev).reshape(h, 1)
            cols = torch.arange(m, device=dev).reshape(1, m)
            W &= (rows - cols).abs() >= theiler
        WT = W.transpose(1, 2).reshape(B * m, h)
        line, start, end = _runs_by_row(WT)
        length = end - start
        carried = white_open[line]
        top = start == 0
        joined = top & (carried > 0)
        length[joined] += carried[joined]
        c_of = col_index[line]
        flag = torch.where(joined, white_flag[line], excluded(i0 + start - 1, c_of))
        bottom = end == h
        flag_end = flag | excluded(i0 + end, c_of)
        keep = ~bottom & ~flag_end
        _count(white, length[keep], col_batch[line[keep]], size1)
        continued = torch.zeros(B * m, dtype=torch.bool, device=dev)
        continued[line[joined]] = True
        stopped = (white_open > 0) & ~continued
        if bool(stopped.any()):
            bad = white_flag[stopped] | excluded(
                torch.full((int(stopped.sum()),), i0, dtype=torch.int64, device=dev),
                col_index[stopped])
            _count(white, white_open[stopped][~bad], col_batch[stopped][~bad], size1)
        white_open.zero_()
        white_flag.zero_()
        white_open[line[bottom]] = length[bottom]
        white_flag[line[bottom]] = flag[bottom]

    # flush: runs reaching the end of the record are kept [DD-57]
    open_d = diag_open > 0
    _count(diag, diag_open[open_d],
           torch.div(torch.arange(B * n_keys, device=dev), n_keys,
                     rounding_mode="floor")[open_d], size1)
    open_v = vert_open > 0
    _count(vert, vert_open[open_v], col_batch[open_v], size1)
    open_w = (white_open > 0) & ~white_flag
    _count(white, white_open[open_w], col_batch[open_w], size1)

    return {
        "diagonal": diag.reshape(B, size1).cpu().numpy(),
        "vertical": vert.reshape(B, size1).cpu().numpy(),
        "white_vertical": white.reshape(B, size1).cpu().numpy(),
        "recurrent": recurrent.cpu().numpy(),
        "near_epsilon": near,
        "rechecked": rechecked,
    }


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
    plans = {}
    rounding = 0
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
            # centred as the reference centres each plot [DD-110]
            centred = np.stack([w - w.mean(axis=0) for w in raw])
            out = batch_line_histograms(centred, eps, theiler=theiler, metric=metric,
                                        precision=precision, device=device,
                                        band_rows=plan.band_rows)
            near += out["near_epsilon"]
            rechecked += out["rechecked"]
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
            "deterministic_counts": True}


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
                 "cells_rechecked_in_double": int(rechecked)}
