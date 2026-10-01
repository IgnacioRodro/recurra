"""Estimation of the delay tau and the embedding dimension m.

Design notes
------------
DD-22 -- ACF by FFT. The old draft recomputed the correlation from scratch
for every candidate lag, giving O(tau_max * N * D). Here the autocorrelation
comes from one FFT, O(N log N), which makes tau_max=2000 as cheap as 20.

DD-23 -- FNN criterion A must be normalised. The standard false-nearest-
neighbour test has two criteria: the relative jump R = |d_{m+1} - d_m| / d_m
against Rtol, and the absolute one d_{m+1} / R_A against Atol, where R_A is
the size of the attractor. The old draft compared d_{m+1} against Atol
directly. Since distances grow like sqrt(m*D), that criterion becomes
progressively harder to satisfy as m rises, so the FNN fraction never falls
below threshold and m saturates at m_max. Normalising by R_A fixes it.

DD-24 -- aggregate across subjects, never concatenate. The old draft stacked
subjects along the time axis before computing the ACF/AMI, which inserts an
artificial discontinuity at every join. Here each subject is estimated
separately and the results are aggregated (median by default).
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..config import SeedLike, resolve_rng
from ..exceptions import ParameterError, ParameterWarning

TauMethod = Literal["ami", "acf", "first_zero"]
MMethod = Literal["fnn", "cao"]


@dataclass(frozen=True)
class EmbeddingParams:
    """Estimated (tau, m) plus the curves that justify them."""

    tau: int
    m: int
    tau_method: str
    m_method: str
    tau_curve: np.ndarray | None = None
    tau_lags: np.ndarray | None = None
    m_curve: np.ndarray | None = None
    m_values: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "tau": self.tau, "m": self.m,
            "tau_method": self.tau_method, "m_method": self.m_method,
            **{k: v for k, v in self.diagnostics.items()
               if isinstance(v, (int, float, str, bool))},
        }])

    def curve_frame(self) -> pd.DataFrame:
        rows = []
        if self.tau_lags is not None:
            for lag, val in zip(self.tau_lags, self.tau_curve, strict=False):
                rows.append({"quantity": self.tau_method, "x": int(lag), "y": float(val)})
        if self.m_values is not None:
            for mv, val in zip(self.m_values, self.m_curve, strict=False):
                rows.append({"quantity": self.m_method, "x": int(mv), "y": float(val)})
        return pd.DataFrame(rows)

    def __repr__(self) -> str:  # pragma: no cover
        return f"EmbeddingParams(tau={self.tau}, m={self.m}, {self.tau_method}/{self.m_method})"


# --------------------------------------------------------------------- tau
def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation up to max_lag, via FFT [DD-22]."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = x.size
    if n < 4:
        return np.zeros(max_lag + 1)
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    F = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(F * np.conj(F), nfft)[: max_lag + 1]
    return acf / acf[0] if acf[0] != 0 else acf


def mutual_information(x: np.ndarray, lag: int, n_bins: int = 32) -> float:
    """Histogram-based AMI between x(t) and x(t+lag)."""
    x = np.asarray(x, dtype=float)
    a, b = x[:-lag], x[lag:]
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < n_bins * 4:
        return 0.0
    c, _, _ = np.histogram2d(a, b, bins=n_bins)
    p = c / c.sum()
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / (px @ py)[nz])))


def estimate_tau(X, *, method: TauMethod = "ami", tau_range=(1, 200),
                 n_bins: int = 32, threshold: float | None = None,
                 ami_step: int = 1) -> EmbeddingParams:
    """Estimate the embedding delay.

    ``ami`` takes the first local minimum of average mutual information;
    ``acf`` the first lag where autocorrelation drops below 1/e;
    ``first_zero`` the first zero crossing of the autocorrelation.
    Multivariate input is averaged across columns.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] < X.shape[1]:
        X = X.T
    n, d = X.shape
    lo, hi = max(1, int(tau_range[0])), int(min(tau_range[1], n // 3))
    if hi < lo:
        hi = lo

    if method == "ami":
        lags = np.arange(lo, hi + 1, max(1, ami_step))
        curve = np.array([np.mean([mutual_information(X[:, k], int(t), n_bins)
                                   for k in range(d)]) for t in lags])
        tau = None
        for i in range(1, curve.size - 1):
            if curve[i] < curve[i - 1] and curve[i] <= curve[i + 1]:
                tau = int(lags[i])
                break
        crit = "first_local_minimum"
        if tau is None:
            tau = int(lags[int(np.argmin(curve))])
            crit = "global_minimum(no local min found)"

    elif method in ("acf", "first_zero"):
        acf = np.mean([autocorrelation(X[:, k], hi) for k in range(d)], axis=0)
        lags = np.arange(lo, hi + 1)
        curve = acf[lo:hi + 1]
        if method == "acf":
            thr = np.exp(-1.0) if threshold is None else float(threshold)
            below = np.where(curve <= thr)[0]
            tau = int(lags[below[0]]) if below.size else int(lags[int(np.argmin(np.abs(curve)))])
            crit = f"first lag with acf <= {thr:.4f}"
        else:
            sign = np.where(np.diff(np.sign(curve)) != 0)[0]
            tau = int(lags[sign[0]]) if sign.size else int(lags[int(np.argmin(np.abs(curve)))])
            crit = "first zero crossing"
    else:
        raise ParameterError(f"unknown tau method {method!r}; use ami/acf/first_zero")

    fallback = crit.startswith("global_minimum") or (
        method == "first_zero" and not sign.size) or (
        method == "acf" and not below.size)
    if fallback or tau >= hi:
        warnings.warn(
            f"estimate_tau: method {method!r} found no {crit.split('(')[0].strip()} "
            f"within tau_range={lo}-{hi}"
            + (f" (the estimate {tau} is the edge of the range)" if tau >= hi else "")
            + ". Widen tau_range, or look at the curve with "
            "plot_embedding_diagnostics() before trusting this tau.",
            ParameterWarning, stacklevel=2)
    return EmbeddingParams(
        tau=int(max(1, tau)), m=0, tau_method=method, m_method="",
        tau_curve=curve, tau_lags=lags,
        diagnostics={"criterion": crit, "tau_range": f"{lo}-{hi}", "n_components": d},
    )


# ----------------------------------------------------------------------- m
def _embed(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Multivariate delay embedding: (N, D) -> (N_eff, m*D)."""
    X = np.atleast_2d(X)
    if X.shape[0] < X.shape[1]:
        X = X.T
    n, d = X.shape
    n_eff = n - (m - 1) * tau
    if n_eff <= 2:
        raise ParameterError(
            f"m={m}, tau={tau} leaves {n_eff} points out of {n}; reduce m or tau"
        )
    out = np.empty((n_eff, m * d))
    for i in range(m):
        out[:, i * d:(i + 1) * d] = X[i * tau: i * tau + n_eff]
    return out


def _attractor_size(Y: np.ndarray) -> float:
    """R_A: RMS deviation from the centroid. The FNN absolute-criterion scale."""
    return float(np.sqrt(np.sum(np.var(Y, axis=0))))


def _nearest_distinct(Y: np.ndarray, idx: np.ndarray):
    """Distance to, and index of, the nearest neighbour at a non-zero distance.

    Kennel's and Cao's criteria compare a point with its nearest *distinct*
    neighbour. Taking the second-nearest point blindly picks an exact
    duplicate whenever the data repeat -- a sine sampled at an integer number
    of samples per period, or an EEG quantised by its converter -- and the
    ratio of distances is then undefined. Before 0.25 FNN counted every such
    point as a false neighbour, so on an exactly periodic sine it never
    dropped below threshold and returned the top of ``m_range``.

    Points closer than 1e-9 of the attractor's size are one point: a period
    repeated in floating point lands 1e-16 away rather than exactly on top,
    and a ratio of two such distances is rounding noise, not geometry. The
    duplicates are collapsed first, so a point with many copies (80 periods of
    a sine give 79) still finds its nearest distinct neighbour. Returns
    ``(distance, neighbour_index, valid)``; a point is invalid only when every
    point of the trajectory coincides with it.
    """
    tol = 1e-9 * (_attractor_size(Y) or 1.0)
    keys = np.round(Y / tol)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    reps = Y[first]
    n_idx = idx.size
    if reps.shape[0] < 2:
        return np.zeros(n_idx), np.zeros(n_idx, dtype=int), np.zeros(n_idx, dtype=bool)
    tree = cKDTree(reps)
    dist, nb = tree.query(Y[idx], k=min(3, reps.shape[0]))
    own = inverse[idx]
    # the nearest representative that is not the point's own group
    pick = np.where(nb[:, 0] == own, 1, 0)
    rows = np.arange(n_idx)
    d, g = dist[rows, pick], nb[rows, pick]
    ok = d > tol
    return d, first[g], ok


def fnn_fraction(X, m: int, tau: int, *, rtol: float = 10.0, atol: float = 2.0,
                 max_points: int = 5000, rng: SeedLike = None) -> float:
    """Fraction of false nearest neighbours, KD-tree based [DD-23]."""
    rng = resolve_rng(rng)
    Ym = _embed(X, m, tau)
    Ym1 = _embed(X, m + 1, tau)
    n = min(Ym.shape[0], Ym1.shape[0])
    if n <= 3:
        return 1.0
    Ym, Ym1 = Ym[:n], Ym1[:n]

    idx = (rng.choice(n, size=max_points, replace=False)
           if n > max_points else np.arange(n))

    d_m, j, ok = _nearest_distinct(Ym, idx)
    if not ok.any():
        return 1.0
    d_m1 = np.linalg.norm(Ym1[j[ok]] - Ym1[idx[ok]], axis=1)
    ra = _attractor_size(Ym1) or 1.0

    crit_rel = np.abs(d_m1 - d_m[ok]) / d_m[ok] > rtol
    crit_abs = d_m1 / ra > atol                   # normalised, unlike the old draft
    false = crit_rel | crit_abs
    return float(false.sum() / ok.sum())


def cao_e_values(X, tau: int, m_range=(1, 12), *, max_points: int = 3000,
                 rng: SeedLike = None):
    """Cao's E(m) and E*(m). E1 = E(m+1)/E(m) saturates at the right m;
    E2 distinguishes deterministic signals (E2 != 1 somewhere) from noise."""
    rng = resolve_rng(rng)
    ms = np.arange(max(1, m_range[0]), m_range[1] + 2)
    E, Estar = [], []
    for m in ms:
        try:
            Ym, Ym1 = _embed(X, m, tau), _embed(X, m + 1, tau)
        except ParameterError:
            E.append(np.nan); Estar.append(np.nan); continue
        n = min(Ym.shape[0], Ym1.shape[0])
        Ym, Ym1 = Ym[:n], Ym1[:n]
        idx = rng.choice(n, size=max_points, replace=False) if n > max_points else np.arange(n)
        d_m, j, ok = _nearest_distinct(Ym, idx)
        if not ok.any():
            E.append(np.nan); Estar.append(np.nan); continue
        d_m1 = np.linalg.norm(Ym1[j[ok]] - Ym1[idx[ok]], axis=1)
        E.append(float(np.mean(d_m1 / d_m[ok])))
        last = Ym1[:, -1]
        Estar.append(float(np.mean(np.abs(last[j[ok]] - last[idx[ok]]))))
    E, Estar = np.array(E), np.array(Estar)
    with np.errstate(invalid="ignore", divide="ignore"):
        E1 = E[1:] / E[:-1]
        E2 = Estar[1:] / Estar[:-1]
    return ms[:-1], E1, E2


def estimate_m(X, tau: int, *, method: MMethod = "fnn", m_range=(1, 12),
               fnn_threshold: float = 0.01, rtol: float = 10.0, atol: float = 2.0,
               stabilization_tol: float = 0.05, max_points: int = 5000,
               rng: SeedLike = None) -> EmbeddingParams:
    """Estimate the embedding dimension for a given tau."""
    rng = resolve_rng(rng)
    if method == "fnn":
        ms = np.arange(max(1, m_range[0]), m_range[1] + 1)
        curve, chosen, crit = [], None, ""
        for m in ms:
            try:
                f = fnn_fraction(X, int(m), tau, rtol=rtol, atol=atol,
                                 max_points=max_points, rng=rng)
            except ParameterError:
                f = np.nan
            curve.append(f)
            if chosen is None and np.isfinite(f) and f <= fnn_threshold:
                chosen, crit = int(m), f"first m with FNN <= {fnn_threshold}"
        curve = np.array(curve)
        if chosen is None:
            valid = np.isfinite(curve)
            chosen = int(ms[np.nanargmin(np.where(valid, curve, np.inf))])
            crit = "minimum FNN (threshold never reached)"
            warnings.warn(
                f"estimate_m: the FNN fraction never fell to {fnn_threshold} within "
                f"m_range={tuple(m_range)}; returning the m where it was lowest "
                f"({chosen}). Noise or a too-small tau keep FNN high; "
                "method='cao' reports whether the series looks stochastic.",
                ParameterWarning, stacklevel=2)
        return EmbeddingParams(tau=tau, m=chosen, tau_method="", m_method="fnn",
                               m_curve=curve, m_values=ms,
                               diagnostics={"criterion": crit,
                                            "fnn_at_chosen": float(curve[list(ms).index(chosen)])})

    if method == "cao":
        ms, E1, E2 = cao_e_values(X, tau, m_range, max_points=min(max_points, 3000), rng=rng)
        chosen, crit = None, ""
        for i in range(len(E1) - 1):
            if np.isfinite(E1[i]) and abs(E1[i + 1] - E1[i]) < stabilization_tol:
                chosen, crit = int(ms[i + 1]), f"E1 saturates (|dE1| < {stabilization_tol})"
                break
        if chosen is None:
            finite = np.isfinite(E1)
            chosen = int(ms[int(np.argmax(np.where(finite, E1, -np.inf)))])
            crit = "max E1 (no saturation found)"
            warnings.warn(
                f"estimate_m: Cao's E1 did not saturate within m_range="
                f"{tuple(m_range)}; returning m={chosen}, where E1 was largest.",
                ParameterWarning, stacklevel=2)
        stochastic = bool(np.nanmax(np.abs(E2 - 1.0)) < 0.05) if np.isfinite(E2).any() else False
        return EmbeddingParams(tau=tau, m=chosen, tau_method="", m_method="cao",
                               m_curve=E1, m_values=ms,
                               diagnostics={"criterion": crit,
                                            "looks_stochastic": stochastic})
    raise ParameterError(f"unknown m method {method!r}; use 'fnn' or 'cao'")


def estimate_embedding(X, *, tau_method: TauMethod = "ami", m_method: MMethod = "fnn",
                       tau_range=(1, 200), m_range=(1, 12), rng: SeedLike = None,
                       **kw) -> EmbeddingParams:
    """Estimate tau then m. Returns both plus the diagnostic curves."""
    t = estimate_tau(X, method=tau_method, tau_range=tau_range,
                     n_bins=kw.pop("n_bins", 32), ami_step=kw.pop("ami_step", 1))
    mm = estimate_m(X, t.tau, method=m_method, m_range=m_range, rng=rng, **kw)
    return EmbeddingParams(
        tau=t.tau, m=mm.m, tau_method=tau_method, m_method=m_method,
        tau_curve=t.tau_curve, tau_lags=t.tau_lags,
        m_curve=mm.m_curve, m_values=mm.m_values,
        diagnostics={"tau_criterion": t.diagnostics.get("criterion", ""),
                     "m_criterion": mm.diagnostics.get("criterion", ""),
                     **{k: v for k, v in mm.diagnostics.items() if k != "criterion"}},
    )


def estimate_embedding_multi(X_list: Sequence, *, aggregate: str = "median",
                             **kw) -> tuple[EmbeddingParams, pd.DataFrame]:
    """Estimate per subject and aggregate. Never concatenates records [DD-24]."""
    if aggregate not in ("median", "max", "mean", "mode"):
        raise ParameterError(f"unknown aggregate {aggregate!r}")
    per = [estimate_embedding(X, **kw) for X in X_list]
    taus = np.array([p.tau for p in per])
    ms = np.array([p.m for p in per])

    def agg(a):
        if aggregate == "median":
            return int(np.median(a))
        if aggregate == "max":
            return int(a.max())
        if aggregate == "mean":
            return int(round(a.mean()))
        vals, counts = np.unique(a, return_counts=True)
        return int(vals[np.argmax(counts)])

    df = pd.DataFrame({"record": range(len(per)), "tau": taus, "m": ms})
    combined = EmbeddingParams(
        tau=agg(taus), m=agg(ms),
        tau_method=per[0].tau_method, m_method=per[0].m_method,
        diagnostics={"aggregate": aggregate, "n_records": len(per),
                     "tau_spread": f"{taus.min()}-{taus.max()}",
                     "m_spread": f"{ms.min()}-{ms.max()}"},
    )
    return combined, df
