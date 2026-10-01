"""Complexity and scaling measures of a scalar series.

These are cheaper than the invariants and answer different questions: how the
fluctuations scale with window size, how irregular the ordering is, how
predictable the next value. They take a 1-D series, not a state space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..exceptions import ParameterError
from .invariants import Invariant
from .scaling import find_scaling_region, fixed_region, local_slopes


def samples_per_turn(x) -> float:
    """How many samples the series spends between changes of direction.

    Sign changes in the first difference, which is what Petrosian's dimension
    already counts. White noise sits at about 1.5 -- one sample per event, the
    scale the fine-grained measures are defined at. A smooth series sampled far
    above its content sits much higher: a gamma envelope low-passed at 8 Hz and
    sampled at 250 Hz measures 24.
    """
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size < 3:
        return float("nan")
    d = np.diff(a)
    turns = int(np.sum(d[1:] * d[:-1] < 0))
    return float(a.size / max(turns, 1))


#: Fewest samples the fine-scale measures may be left with after decimation.
#: Below this the ordinal and fractal estimators are reading their own
#: small-sample bias, not the series [DD-99].
FINE_SCALE_MIN_SAMPLES = 64


def fine_scale_factor(x, target: float = 2.0,
                      min_samples: int = FINE_SCALE_MIN_SAMPLES) -> int:
    """The decimation that brings a series to the scale its structure lives at.

    Design note (DD-99) -- a fractal dimension of 1 can mean a smooth curve or
    a curve looked at from too close.

    Higuchi, Katz, Petrosian, permutation entropy, sample entropy and
    Lempel-Ziv all read point-to-point structure. Given a series with thirty
    samples per cycle of its own fastest content they report, correctly, that
    it is a smooth line: Higuchi 1.04, Katz 1.0004, Petrosian 1.0016. Across
    98 real recordings those columns had coefficients of variation of 0.0018,
    0.00008 and 0.00003 -- constants, unable to separate anything.

    Decimated to about two samples per turning point the same envelope gives
    Higuchi 1.97 and permutation entropy 0.99. The measures were never wrong;
    they were being asked about a scale where nothing happens.

    Revision (0.25). The factor is capped so that at least ``min_samples``
    remain. Uncapped, a 250-point window of a slow envelope was decimated to
    nine samples and every fractal and entropy measure failed as too short;
    the slow test that checks every measure on a short window had been
    failing since the decimation was introduced. A capped factor reads a
    finer scale than the target, which is the lesser error: a number from a
    scale slightly too fine against no number at all.
    """
    ratio = samples_per_turn(x)
    if not np.isfinite(ratio) or ratio <= target:
        return 1
    factor = max(1, int(round(ratio / target)))
    n = int(np.isfinite(np.asarray(x, dtype=float)).sum())
    return max(1, min(factor, n // max(1, int(min_samples))))


def _series(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size < 32:
        raise ParameterError(f"series too short: {a.size} finite samples")
    return a


def detrended_fluctuation(x, *, scales=None, order: int = 1,
                          min_points: int = 6) -> Invariant:
    """DFA. The exponent is 0.5 for white noise, 1 for 1/f, 1.5 for Brownian."""
    a = _series(x)
    n = a.size
    if scales is None:
        scales = np.unique(np.logspace(np.log10(8), np.log10(n // 4), 20).astype(int))
    y = np.cumsum(a - a.mean())

    fluct = []
    for s in scales:
        n_win = n // s
        if n_win < 2:
            fluct.append(np.nan)
            continue
        seg = y[:n_win * s].reshape(n_win, s)
        t = np.arange(s)
        resid = np.array([w - np.polyval(np.polyfit(t, w, order), t) for w in seg])
        fluct.append(float(np.sqrt(np.mean(resid ** 2))))

    fl = np.asarray(fluct)
    ok = np.isfinite(fl) & (fl > 0)
    reg = find_scaling_region(np.log(scales[ok]), np.log(fl[ok]),
                              min_points=min(min_points, int(ok.sum())),
                              min_decades=0.0)
    return Invariant(
        name="DFA_alpha", value=reg.slope, units="dimensionless",
        method=f"detrended fluctuation, order {order}", region=reg,
        curve=pd.DataFrame({"scale": scales[ok], "fluctuation": fl[ok]}),
        diagnostics={"order": order, "n_scales": int(ok.sum())},
    )


def hurst_exponent(x, **kwargs) -> Invariant:
    """Hurst exponent, taken as the DFA exponent. 0.5 means no memory."""
    out = detrended_fluctuation(x, **kwargs)
    return Invariant(name="hurst", value=out.value, units="dimensionless",
                     method="DFA exponent", region=out.region, curve=out.curve,
                     diagnostics=out.diagnostics)


def higuchi_dimension(x, *, k_max: int = 20, tolerance: float = 0.25) -> Invariant:
    """Higuchi fractal dimension. 1 for a smooth curve, 2 for a very rough one.

    Design note (DD-73) -- the fit is anchored at k = 1, not floated.

    Higuchi's construction is a power law only for small k; past that the
    curve bends and then falls. Letting the general scaling-region finder pick
    the flattest window lands it on the bend: on a smoothed envelope the local
    slopes ran 1.04, 1.28, 2.03, 3.27, 0.79 across k, and the finder returned
    **3.16** -- outside the [1, 2] the dimension is defined on.

    The fit is therefore anchored at k = 1 and extended while the local slope
    stays within ``tolerance`` of its initial value. A result still outside
    [1, 2] is returned with a warning rather than silently clipped, because it
    means the series does not have a Higuchi scaling range at all.
    """
    a = _series(x)
    n = a.size
    ks = np.arange(1, min(k_max, n // 4) + 1)
    lengths = []
    for k in ks:
        lk = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            lk.append(np.abs(np.diff(a[idx])).sum()
                      * (n - 1) / (((idx.size - 1) or 1) * k) / k)
        lengths.append(np.mean(lk) if lk else np.nan)
    lg = np.asarray(lengths)
    ok = np.isfinite(lg) & (lg > 0)
    xs, ys = np.log(1.0 / ks[ok]), np.log(lg[ok])
    if xs.size < 5:
        raise ParameterError("too few usable scales for a Higuchi estimate")

    slopes = local_slopes(xs, ys, half_width=1)
    initial = float(np.nanmean(slopes[:3]))
    stop = xs.size
    for i in range(3, xs.size):
        if np.isfinite(slopes[i]) and abs(slopes[i] - initial) > tolerance * abs(initial):
            stop = max(i, 5)
            break
    reg = fixed_region(xs[:stop], ys[:stop], float(xs[:stop].min()),
                       float(xs[:stop].max()))
    warning = None
    if not 0.95 <= reg.slope <= 2.05:      # a little slack for estimation noise
        warning = (f"Higuchi dimension {reg.slope:.3f} lies outside [1, 2], where "
                   "the quantity is defined. The series has no power-law range "
                   "at these scales; treat the number as absent, not as a "
                   "dimension.")
    return Invariant(
        name="higuchi_fd", value=reg.slope, units="dimensionless",
        method=f"Higuchi, fitted over k = 1..{int(ks[ok][stop - 1])}", region=reg,
        curve=pd.DataFrame({"k": ks[ok], "length": lg[ok]}),
        diagnostics={"k_max": int(ks[-1]), "k_fitted": int(ks[ok][stop - 1]),
                     **({"warning": warning} if warning else {})},
    )


def permutation_entropy(x, *, order: int = 3, delay: int = 1,
                        normalise: bool = True) -> Invariant:
    """Bandt-Pompe entropy of the ordinal patterns. 1 when normalised is random."""
    a = _series(x)
    n = a.size - (order - 1) * delay
    if n < 10:
        raise ParameterError("series too short for this order and delay")
    patterns = np.argsort(np.column_stack([a[i * delay:i * delay + n]
                                           for i in range(order)]), axis=1)
    codes = np.ravel_multi_index(patterns.T, (order,) * order)
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    h = float(-np.sum(p * np.log(p)))
    from math import factorial

    value = h / np.log(factorial(order)) if normalise else h
    return Invariant(
        name="permutation_entropy", value=float(value),
        units="dimensionless" if normalise else "nats",
        method=f"Bandt-Pompe, order {order}, delay {delay}",
        diagnostics={"order": order, "delay": delay, "n_patterns": int(p.size),
                     "normalised": normalise},
    )


def sample_entropy(x, *, m: int = 2, tolerance: float | None = None,
                   max_points: int = 5000) -> Invariant:
    """Sample entropy: the negative log chance that close pairs stay close."""
    a = _series(x)
    n_total = int(a.size)
    if a.size > max_points:
        a = a[:max_points]
    r = tolerance if tolerance is not None else 0.2 * a.std()
    if r <= 0:
        raise ParameterError("tolerance must be positive; the series is constant")

    # Richman and Moorman count N - m templates for both m and m + 1, so the
    # two counts are over the same set of starting points. An earlier version
    # used N - m + 1 for the shorter one, a small but systematic bias.
    n_templates = a.size - m

    def _count(dim: int) -> int:
        n = n_templates
        emb = np.column_stack([a[i:i + n] for i in range(dim)])
        tree = cKDTree(emb, compact_nodes=False, balanced_tree=False)
        return int(sum(len(nb) for nb in tree.query_ball_point(emb, r, p=np.inf))
                   - emb.shape[0])

    b, a_ = _count(m), _count(m + 1)
    if b == 0 or a_ == 0:
        return Invariant(name="sample_entropy", value=float("inf"), units="nats",
                         method=f"m={m}", diagnostics={
                             "warning": "no matching templates at this tolerance; "
                                        "raise it or lengthen the series"})
    return Invariant(name="sample_entropy", value=float(-np.log(a_ / b)),
                     units="nats", method=f"m={m}, r={r:.4g}",
                     diagnostics={"m": m, "tolerance": float(r),
                                  "matches_m": b, "matches_m1": a_,
                                  "n_used": int(a.size), "n_total": n_total,
                                  "truncated": bool(n_total > a.size)})


def spectral_entropy(x, *, fs: float = 1.0, normalise: bool = True) -> Invariant:
    """Shannon entropy of the normalised power spectrum."""
    a = _series(x)
    power = np.abs(np.fft.rfft(a - a.mean())) ** 2
    power = power[1:]
    total = power.sum()
    if total <= 0:
        raise ParameterError("the series has no spectral power")
    p = power / total
    p = p[p > 0]
    h = float(-np.sum(p * np.log(p)))
    value = h / np.log(p.size) if normalise else h
    return Invariant(name="spectral_entropy", value=float(value),
                     units="dimensionless" if normalise else "nats",
                     fs=fs, method="Shannon entropy of the power spectrum",
                     diagnostics={"n_bins": int(p.size), "normalised": normalise})


def katz_dimension(x) -> Invariant:
    """Katz fractal dimension of a curve. 1 for a straight line, higher when
    the path wanders relative to its overall extent.

    Sensitive to the amplitude scale of the series, unlike Higuchi, because it
    compares path length against the distance spanned. Series are therefore
    normalised to unit standard deviation first, so the value describes shape
    rather than gain.
    """
    a = _series(x)
    a = (a - a.mean()) / (a.std() or 1.0)
    steps = np.sqrt(1.0 + np.diff(a) ** 2)          # unit spacing on the time axis
    length = float(steps.sum())
    mean_step = length / max(a.size - 1, 1)
    extent = float(np.max(np.sqrt(np.arange(a.size) ** 2 + (a - a[0]) ** 2)))
    if extent <= 0 or mean_step <= 0:
        raise ParameterError("the series has no extent; Katz dimension is undefined")
    n = length / mean_step
    denom = np.log(n) + np.log(extent / length)
    return Invariant(
        name="katz_fd", value=float(np.log(n) / denom) if denom else np.nan,
        units="dimensionless", method="Katz",
        diagnostics={"path_length": length, "extent": extent, "n_steps": float(n)},
    )


def petrosian_dimension(x) -> Invariant:
    """Petrosian fractal dimension, from the number of sign changes in the
    first difference. Cheap, and the only one here that needs no embedding."""
    a = _series(x)
    d = np.diff(a)
    n_delta = int(np.sum(d[1:] * d[:-1] < 0))
    n = a.size
    denom = np.log10(n) + np.log10(n / (n + 0.4 * n_delta))
    return Invariant(
        name="petrosian_fd",
        value=float(np.log10(n) / denom) if denom else np.nan,
        units="dimensionless", method="Petrosian",
        diagnostics={"sign_changes": n_delta, "n_samples": n},
    )


def svd_entropy(x, *, m: int = 5, tau: int = 1, normalise: bool = True) -> Invariant:
    """Entropy of the singular spectrum of a delay embedding.

    Low when the trajectory lies close to a low-dimensional subspace, high when
    the singular values are spread evenly. A complementary view to the
    correlation dimension, and far cheaper.
    """
    a = _series(x)
    n = a.size - (m - 1) * tau
    if n < m:
        raise ParameterError("series too short for this embedding")
    emb = np.column_stack([a[i * tau:i * tau + n] for i in range(m)])
    sv = np.linalg.svd(emb, compute_uv=False)
    total = sv.sum()
    if total <= 0:
        raise ParameterError("the embedding is degenerate")
    p = sv / total
    p = p[p > 0]
    h = float(-np.sum(p * np.log(p)))
    value = h / np.log(m) if normalise else h
    return Invariant(
        name="svd_entropy", value=float(value),
        units="dimensionless" if normalise else "nats",
        method=f"singular spectrum, m={m}, tau={tau}",
        diagnostics={"m": m, "tau": tau, "n_singular_values": int(sv.size),
                     "normalised": normalise},
    )


def hjorth_parameters(x, *, fs: float = 1.0) -> dict[str, Invariant]:
    """Activity, mobility and complexity.

    Activity is the variance. Mobility is the standard deviation of the first
    difference over that of the signal, an estimate of mean frequency in
    radians per sample -- for a sinusoid at ``f`` it is ``2*pi*f/fs``, which
    the tests check exactly. Complexity is the mobility of the first difference
    over the mobility of the signal, 1 for a pure sinusoid and larger as the
    waveform departs from one.
    """
    a = _series(x)
    d1 = np.diff(a)
    d2 = np.diff(d1)
    var0, var1, var2 = a.var(), d1.var(), d2.var()
    mobility = float(np.sqrt(var1 / var0)) if var0 > 0 else np.nan
    mob_d1 = float(np.sqrt(var2 / var1)) if var1 > 0 else np.nan
    complexity = mob_d1 / mobility if mobility else np.nan
    common = {"method": "Hjorth", "fs": fs}
    return {
        "hjorth_activity": Invariant(name="hjorth_activity", value=float(var0),
                                     units="signal units squared", **common),
        "hjorth_mobility": Invariant(name="hjorth_mobility", value=mobility,
                                     units="radians per sample", **common,
                                     diagnostics={"hz_equivalent":
                                                  mobility * fs / (2 * np.pi)}),
        "hjorth_complexity": Invariant(name="hjorth_complexity",
                                       value=float(complexity),
                                       units="dimensionless", **common),
    }


def lempel_ziv_complexity(x, *, threshold: str = "median",
                          normalise: bool = True) -> Invariant:
    """Lempel-Ziv complexity of the binarised series (LZ76).

    Counts the distinct patterns needed to build the sequence left to right.
    Normalised by ``n / log2(n)``, the asymptotic count for a random binary
    sequence, so 1 means incompressible and lower means structured.

    Binarisation discards amplitude entirely; that is the point, and it makes
    the measure robust to gain and to monotone distortion, but blind to
    anything the sign of the deviation does not carry.
    """
    a = _series(x)
    if threshold == "median":
        cut = float(np.median(a))
    elif threshold == "mean":
        cut = float(a.mean())
    else:
        raise ParameterError("threshold must be 'median' or 'mean'")
    s = (a > cut).astype(np.uint8)
    n = s.size

    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    norm = n / np.log2(n) if n > 1 else 1.0
    return Invariant(
        name="lempel_ziv", value=float(c / norm) if normalise else float(c),
        units="dimensionless" if normalise else "patterns",
        method=f"LZ76, binarised at the {threshold}",
        diagnostics={"raw_count": int(c), "n_samples": int(n),
                     "threshold_value": cut, "normalised": normalise},
    )
