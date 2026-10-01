"""Canonical cross-frequency coupling indices.

These are not the contribution -- they are the yardstick. Without MVL, MI
and PLV there is no way to validate the synthetic generator, no quality
control on the corpus, and nothing to benchmark recurrence-based measures
against.

References
----------
MVL : Canolty et al. (2006), Science 313:1626
MI  : Tort et al. (2008, 2010), J Neurophysiol
PLV : Lachaux et al. (1999), Hum Brain Mapp
"""
from __future__ import annotations

import warnings

import numpy as np

from ..config import SeedLike, resolve_rng
from ..exceptions import ParameterError, ParameterWarning


def _check_pair(a, b, names=("phase", "amplitude")):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ParameterError(f"{names[0]} has {a.size} samples, {names[1]} has {b.size}")
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def _check_envelope(am: np.ndarray, index: str) -> bool:
    """Warn when an 'amplitude' has negative values. Returns True if it does.

    MVL, MI and the modulogram are defined on an envelope, which is never
    negative. A rescaled coordinate (z-scored, robust-scaled) is not one, and
    neither is an envelope low-passed by a filter that rings below zero. The
    message is fixed text, so a loop over windows warns once, not per window
    [DD-116].
    """
    if am.size and np.any(am < 0):
        warnings.warn(
            f"{index}: the amplitude has negative values, so it is not an "
            "envelope. Canonical coupling indices need the Hilbert envelope "
            "itself, not a rescaled or ringing copy of it.",
            ParameterWarning, stacklevel=3)
        return True
    return False


def mvl(phase, amplitude, *, normalise: bool = True, normalize: bool | None = None) -> float:
    """Mean Vector Length (Canolty).

    ``normalize`` is accepted as the old spelling; the library spells it
    ``normalise`` everywhere else [DD-118].

    ``normalise=True`` divides by the mean amplitude, making the index
    dimensionless and insensitive to signal gain. Without it, MVL scales
    with amplitude units and cannot be compared across subjects.
    """
    ph, am = _check_pair(phase, amplitude)
    if ph.size == 0:
        return np.nan
    _check_envelope(am, "mvl")
    z = np.mean(am * np.exp(1j * ph))
    if normalize is not None:
        normalise = bool(normalize)
    if not normalise:
        return float(np.abs(z))
    denom = np.mean(am)
    return float(np.abs(z) / denom) if denom > 0 else np.nan


def modulation_index(phase, amplitude, *, n_bins: int = 18) -> float:
    """Tort's Modulation Index: KL divergence of the phase-amplitude
    distribution from uniform, normalised by log(n_bins) to lie in [0, 1]."""
    ph, am = _check_pair(phase, amplitude)
    if ph.size < n_bins * 4:
        return np.nan
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(np.angle(np.exp(1j * ph)), edges) - 1, 0, n_bins - 1)
    means = np.zeros(n_bins)
    for k in range(n_bins):
        sel = idx == k
        means[k] = am[sel].mean() if sel.any() else 0.0
    # The index is a divergence between probability distributions; a phase
    # bin with a negative mean amplitude makes the "distribution" undefined.
    if _check_envelope(am, "modulation_index") and np.any(means < 0):
        return np.nan
    total = means.sum()
    if total <= 0:
        return np.nan
    p = means / total
    nz = p > 0
    h = -np.sum(p[nz] * np.log(p[nz]))
    return float((np.log(n_bins) - h) / np.log(n_bins))


def plv(phase1, phase2, *, n: int = 1, m: int = 1) -> float:
    """n:m Phase Locking Value: ``|mean(exp(i (n phi_1 - m phi_2)))|``.

    ``n`` multiplies the first phase and ``m`` the second, so for a slow
    rhythm locked to a fast one at one slow cycle per five fast cycles, pass
    the slow phase first with ``n=5, m=1``.
    """
    p1, p2 = _check_pair(phase1, phase2, ("phase1", "phase2"))
    if p1.size == 0:
        return np.nan
    d = n * np.unwrap(p1) - m * np.unwrap(p2)
    return float(np.abs(np.mean(np.exp(1j * d))))


def aac_index(amp1, amp2, *, method: str = "spearman") -> float:
    """Amplitude-amplitude coupling: correlation between envelopes."""
    a1, a2 = _check_pair(amp1, amp2, ("amp1", "amp2"))
    if a1.size < 3:
        return np.nan
    if method == "spearman":
        from scipy.stats import spearmanr

        return float(spearmanr(a1, a2).statistic)
    return float(np.corrcoef(a1, a2)[0, 1])


def surrogate_significance(index_fn, phase, amplitude, *, n_surrogates: int = 200,
                           method: str = "circular_shift", rng: SeedLike = None):
    """p-value and z-score of an index against phase-scrambled surrogates."""
    rng = resolve_rng(rng)
    ph, am = _check_pair(phase, amplitude)
    observed = index_fn(ph, am)
    n = ph.size
    null = np.empty(n_surrogates)
    for i in range(n_surrogates):
        if method == "circular_shift":
            k = int(rng.integers(n // 10, n - n // 10))
            null[i] = index_fn(ph, np.roll(am, k))
        elif method == "shuffle":
            null[i] = index_fn(ph, rng.permutation(am))
        else:
            raise ParameterError(f"unknown surrogate method {method!r}")
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {"observed": observed, "p_value": np.nan, "z_score": np.nan}
    p = float((np.sum(null >= observed) + 1) / (null.size + 1))
    s = null.std()
    z = float((observed - null.mean()) / s) if s > 0 else np.nan
    return {"observed": float(observed), "p_value": p, "z_score": z,
            "null_mean": float(null.mean()), "null_std": float(s),
            "n_surrogates": int(null.size)}


def coupling_indices(phase, amplitude, *, phase2=None, amp2=None,
                     n_bins: int = 18, nm: tuple[int, int] = (1, 1),
                     shape: bool = True) -> dict:
    """All applicable canonical indices in one call, for CSV export.

    These belong in every table beside the recurrence metrics, and for a long
    time were in none of ours. A study can spend months computing recurrence
    quantification of a phase-amplitude space without ever establishing that
    the space contains any coupling, and then has no way to tell a null result
    from a well-measured absence [DD-106].

    ``shape`` adds two summaries of the modulogram: the contrast between its
    largest and smallest bin, and the phase at which the envelope peaks. The
    contrast is what the picture shows; the preferred phase is the quantity a
    reader will ask about next.
    """
    out = {
        "mvl": mvl(phase, amplitude),
        "mi_tort": modulation_index(phase, amplitude, n_bins=n_bins),
    }
    if shape:
        prof = modulogram(phase, amplitude, n_bins=n_bins)
        finite = prof["amplitude"].to_numpy(dtype=float)
        out["mod_contrast"] = float(np.nanmax(finite) - np.nanmin(finite))
        out["preferred_phase"] = float(
            prof.loc[prof["amplitude"].idxmax(), "phase"])
    if phase2 is not None:
        out["plv"] = plv(phase, phase2, n=nm[0], m=nm[1])
    if amp2 is not None:
        out["aac_spearman"] = aac_index(amplitude, amp2)
    return out


def modulogram(phase, amplitude, *, n_bins: int = 18, normalise: bool = True):
    """Mean amplitude in each phase bin: the picture that shows coupling.

    Design note (DD-106) -- a scatter of the joint space cannot show
    phase-amplitude coupling, and the mean can.

    The obvious way to look for coupling is to plot the trajectory and see
    whether the envelope is higher at some phases. It does not work, and the
    reason is quantitative. On a realistic heavy-tailed envelope the spread of
    amplitude *within* one phase bin is several times the movement of the mean
    *across* bins:

    | coupling | within a bin | movement across bins | ratio |
    |---|---|---|---|
    | alpha 0.3 (MVL 0.16, clearly detectable) | 1.317 | 0.298 | 4.4 : 1 |
    | alpha 0.9 (MVL 0.46, strong) | 1.324 | 0.873 | 1.5 : 1 |

    So the individual points drown the effect even when the coupling is strong,
    and a scatter of a real recording shows nothing whatever the coupling. The
    conditional mean is where the signal is, and averaging is what recovers it:
    measured on the same signals, the contrast between the highest and lowest
    bin went 0.118 with no coupling, 0.888 at alpha 0.3 and 2.546 at 0.9.

    Returns a frame with the bin centres, the mean amplitude, its standard
    error, and the count -- the error bars matter, because a bin with few
    samples will wander on its own.
    """
    import pandas as pd

    ph, am = _check_pair(phase, amplitude)
    _check_envelope(am, "modulogram")
    # Wrap first, as modulation_index does: a phase given in [0, 2*pi) would
    # otherwise put every value above pi into the last bin.
    ph = np.angle(np.exp(1j * ph))
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(n_bins, np.nan)
    sem = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        sel = am[idx == k]
        count[k] = sel.size
        if sel.size:
            mean[k] = sel.mean()
            sem[k] = sel.std(ddof=1) / np.sqrt(sel.size) if sel.size > 1 else np.nan
    if normalise:
        total = np.nansum(mean)
        if total > 0:
            mean, sem = mean / total, sem / total
    return pd.DataFrame({"phase": centres, "amplitude": mean, "sem": sem,
                         "count": count})


def comodulogram(recording, phase_bands=None, amplitude_bands=None, *,
                 index: str = "mi", n_bins: int = 18):
    """Coupling index for every pair of a phase band and an amplitude band.

    The map that says *which* pair couples, rather than assuming one. Rows are
    the band whose phase is taken, columns the band whose envelope is taken,
    and the value is the modulation index by default.

    Bands are named channels of the recording; the defaults are every channel
    with a phase role against every channel with an amplitude role, which for
    a Hilbert-decomposed recording is the whole grid at no extra cost.
    """
    import pandas as pd

    fns = {"mi": modulation_index, "mvl": mvl}
    if index not in fns:
        raise ParameterError(f"index must be one of {list(fns)}, got {index!r}")
    roles = getattr(recording, "roles", {}) or {}
    if phase_bands is None:
        phase_bands = [c for c in recording.names if roles.get(c) == "phase"]
    if amplitude_bands is None:
        amplitude_bands = [c for c in recording.names if roles.get(c) == "amplitude"]
    if not phase_bands or not amplitude_bands:
        raise ParameterError(
            "a comodulogram needs at least one phase channel and one amplitude "
            f"channel; found {phase_bands} and {amplitude_bands}")

    rows = []
    for p in phase_bands:
        for a in amplitude_bands:
            kw = {"n_bins": n_bins} if index == "mi" else {}
            rows.append({"phase_band": p, "amplitude_band": a,
                         "index": index,
                         "value": float(fns[index](recording.get(p),
                                                   recording.get(a), **kw))})
    return pd.DataFrame(rows)
