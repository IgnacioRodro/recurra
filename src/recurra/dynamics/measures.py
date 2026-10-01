"""Dynamical and complexity measures as a tidy row, per window and per subject.

Design note (DD-71) -- the scalar measures need a series, so say which one.

Every measure in :mod:`recurra.dynamics.complexity` takes a one-dimensional
series, while a state space has several coordinates. Applying them to a
coordinate chosen by position would be arbitrary and would silently change
meaning between routes.

Each **block** therefore contributes its natural scalar: a phase circle gives
back its angle, an amplitude block its amplitude, a delay block its first
coordinate, and anything else its first column. Columns are named
``measure_block``, so ``higuchi_fd_amp_gamma`` says what was measured and on
what. The invariants that genuinely need the whole trajectory -- D2, the
Lyapunov exponent, K2 -- are computed on the state space itself and keep their
bare names.

Design note (DD-77) -- these are not coupling measures.

Every scalar measure here is computed on **one block**. A quantity that never
sees the other block cannot be measuring the relation between them, whatever
its discriminative power.

Measured: on phase-phase coupling, nine of these separate coupled from control
at AUC 1.000 where the phase-locking value reaches 0.71. Pair the gamma phase
of a coupled recording with the theta phase of an *independent* one, and they
still separate at 1.000 -- while PLV falls to 0.60 and the line length of the
joint space to 0.70. They detect that a phase is regular, not that two are
related.

They remain useful features, and under a given generator regularity and
coupling may coincide. But a claim about coupling needs a measure defined on
the joint object: RQA of a joint space or cross plot, the joint independence
ratio, or a classical index.

Design note (DD-72) -- cheap by default, expensive on request.

Higuchi, Katz, Petrosian, the entropies, DFA, Hjorth and Lempel-Ziv are all
O(N log N) or better and cost milliseconds on a window. D2 and the Lyapunov
exponent build neighbour structures and cost seconds. Asking for
``dynamics=True`` over a thousand windows should not silently start an
overnight job, so the default set is the cheap one and the invariants are
opt-in through ``invariants=True``.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from ..exceptions import InferenceWarning, ParameterError, ParameterWarning
from . import complexity as cx
from .invariants import correlation_dimension, k2_entropy, lyapunov_max

#: Measures defined on a scalar series, computed per coordinate block.
SCALAR_MEASURES: tuple[str, ...] = (
    "higuchi_fd", "katz_fd", "petrosian_fd",
    "svd_entropy", "permutation_entropy", "sample_entropy", "spectral_entropy",
    "DFA_alpha", "hurst", "lempel_ziv",
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
)

#: Measures defined on the whole trajectory. Expensive [DD-72].
TRAJECTORY_MEASURES: tuple[str, ...] = ("D2", "lambda_1", "K2")

#: Invariants that can also be estimated from a single block, by embedding
#: that one series. The earlier draft computed both a univariate and a
#: multivariate version of each, and both are useful: the multivariate one
#: describes the joint trajectory, the univariate ones describe each signal on
#: its own. Neither is a coupling measure [DD-77].
BLOCK_INVARIANTS: tuple[str, ...] = ("D2", "lambda_1")

#: Measures that read point-to-point structure, and are therefore blind on a
#: series sampled far above its own content [DD-99].
FINE_SCALE_MEASURES: tuple[str, ...] = (
    "higuchi_fd", "katz_fd", "petrosian_fd", "permutation_entropy",
    "sample_entropy", "lempel_ziv",
)

#: A compact default: one fractal, one entropy, one scaling exponent, one
#: complexity, and the Hjorth triple, which between them span the families
#: without producing forty columns per block.
DEFAULT_SCALAR: tuple[str, ...] = (
    "higuchi_fd", "permutation_entropy", "DFA_alpha", "lempel_ziv",
    "hjorth_mobility", "hjorth_complexity",
)

_SCALAR_FN = {
    "higuchi_fd": lambda x, fs: cx.higuchi_dimension(x).value,
    "katz_fd": lambda x, fs: cx.katz_dimension(x).value,
    "petrosian_fd": lambda x, fs: cx.petrosian_dimension(x).value,
    "svd_entropy": lambda x, fs: cx.svd_entropy(x).value,
    "permutation_entropy": lambda x, fs: cx.permutation_entropy(x).value,
    "sample_entropy": lambda x, fs: cx.sample_entropy(x).value,
    "spectral_entropy": lambda x, fs: cx.spectral_entropy(x, fs=fs).value,
    "DFA_alpha": lambda x, fs: cx.detrended_fluctuation(x).value,
    "hurst": lambda x, fs: cx.hurst_exponent(x).value,
    "lempel_ziv": lambda x, fs: cx.lempel_ziv_complexity(x).value,
}


def block_series(ss) -> dict[str, np.ndarray]:
    """The natural scalar of each coordinate block [DD-71]."""
    coords = np.asarray(ss.coords, dtype=float)
    out: dict[str, np.ndarray] = {}
    for g in ss.groups:
        block = coords[:, g.slice]
        if g.kind == "phase_circle" and block.shape[1] >= 2:
            # Not the wrapped angle: its jumps at +-pi are discontinuities the
            # scalar measures read as structure, and DFA on it returned 0.13.
            # Not the unwrapped angle either: that is a ramp, and DFA returned
            # 2.01, the value of a pure trend. The stationary quantity carrying
            # the phase dynamics is its increment, the instantaneous frequency
            # in radians per sample [DD-71].
            # Central differences keep the length and put no artificial zero
            # at the first sample, which ``diff`` with ``prepend`` did.
            phase = np.unwrap(np.arctan2(block[:, 1], block[:, 0]))
            out[g.label] = np.gradient(phase)
        else:
            out[g.label] = block[:, 0]
    if not out:
        out["coord0"] = coords[:, 0]
    return out


def _embed_block(series: np.ndarray, fs: float, m: int, tau: int):
    """A delay embedding of one block, as a state space the invariants accept."""
    from ..statespace.core import CoordGroup, StateSpace

    n = series.size - (m - 1) * tau
    if n < m + 2:
        raise ParameterError("series too short for this embedding")
    coords = np.column_stack([series[i * tau:i * tau + n] for i in range(m)])
    group = CoordGroup(label="delay", start=0, stop=m, kind="delay", weight=1.0)
    return StateSpace(coords=coords, groups=(group,), fs=fs)


def dynamics_measures(source, *, measures: Sequence[str] | str = "default",
                      invariants: bool | Sequence[str] = False,
                      block_invariants: bool | Sequence[str] = False,
                      embed_m: int = 5, embed_tau: int = 1,
                      per_block: bool = True, fs: float | None = None,
                      fine_scale: str | int | None = "auto",
                      prefix: str = "", **invariant_kwargs) -> dict[str, float]:
    """A flat dict of dynamical measures, ready to be a row of a table.

    Parameters
    ----------
    measures
        ``"default"`` for the compact set, ``"all"`` for every scalar measure,
        or an explicit sequence. Applied per coordinate block [DD-71].
    invariants
        ``False`` (default), ``True`` for D2, the Lyapunov exponent and K2, or
        a subset. These are expensive and computed on the whole trajectory
        [DD-72].
    block_invariants
        D2 and the Lyapunov exponent estimated from each block on its own, by
        delay-embedding that single series. Named ``lambda_1_<block>``, each
        with a companion ``_ok`` column that is 0 when the estimator's own
        diagnostics reject the value [DD-63]. Useful features; not coupling
        measures [DD-77, DD-93].
    per_block
        When False, the scalar measures are applied to the first block only and
        the column names carry no block suffix.
    fine_scale
        What to do when a series is sampled far above its own content, where
        the fractal dimensions and ordinal entropies read a smooth line and
        return constants [DD-99]. ``"auto"`` decimates each block to about two
        samples per turning point for those measures only, and records the
        factor as ``fine_scale_<block>``. An integer forces a factor; ``None``
        keeps the old behaviour. Nothing else is affected: DFA, Hjorth and the
        invariants see the full series.
    """
    if measures == "default":
        names = list(DEFAULT_SCALAR)
    elif measures == "all":
        names = list(SCALAR_MEASURES)
    else:
        names = list(measures)
    unknown = set(names) - set(SCALAR_MEASURES)
    if unknown:
        raise ParameterError(
            f"unknown measure(s) {sorted(unknown)}; available: "
            f"{list(SCALAR_MEASURES)}")

    rate = float(fs if fs is not None else getattr(source, "fs", 1.0) or 1.0)
    if hasattr(source, "groups"):
        series = block_series(source)
    else:
        # A plain array: the series itself, or the first column of a matrix.
        # atleast_2d(x).T[0] did the second and, for a 1-D x, returned its
        # first *sample*, so every measure failed on a one-point series.
        arr = np.asarray(source, dtype=float)
        series = {"": arr if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)[:, 0]}
    if not per_block:
        first = next(iter(series))
        series = {"": series[first]}

    out: dict[str, float] = {}
    for label, x in series.items():
        suffix = f"_{label}" if label else ""
        hj = None

        # The fine-grained measures get the series at the scale its structure
        # lives at; everything else gets it whole [DD-99].
        if fine_scale is None:
            factor = 1
        elif fine_scale == "auto":
            factor = cx.fine_scale_factor(x)
            wanted = cx.fine_scale_factor(x, min_samples=1)
            if factor < wanted:
                # Fixed text, so a loop over windows warns once [DD-46].
                warnings.warn(
                    "dynamics: the fine-scale decimation was capped so that at "
                    f"least {cx.FINE_SCALE_MIN_SAMPLES} samples remain; the "
                    "window is too short for the scale its structure lives at. "
                    "Compare fine_scale_<block> with samples_per_turn_<block> "
                    "[DD-99].", ParameterWarning, stacklevel=2)
        else:
            factor = max(1, int(fine_scale))
        fine = x[::factor] if factor > 1 else x
        if any(n in FINE_SCALE_MEASURES for n in names):
            out[f"{prefix}fine_scale{suffix}"] = float(factor)
            out[f"{prefix}samples_per_turn{suffix}"] = cx.samples_per_turn(x)

        for name in names:
            key = f"{prefix}{name}{suffix}"
            source_series = fine if name in FINE_SCALE_MEASURES else x
            source_rate = (rate / factor) if name in FINE_SCALE_MEASURES else rate
            try:
                if name.startswith("hjorth_"):
                    if hj is None:
                        hj = cx.hjorth_parameters(x, fs=rate)
                    out[key] = float(hj[name].value)
                else:
                    out[key] = float(_SCALAR_FN[name](source_series, source_rate))
            except Exception as exc:
                # A silent nan looks like a measurement that came out empty.
                # The invariants say what happened [DD-95]; so do these.
                warnings.warn(f"dynamics: {name} on block {label or 'series'!r} "
                              f"failed: {exc}", InferenceWarning, stacklevel=2)
                out[key] = float("nan")

    # D2 and lambda_1 do not take the same options: max_steps belongs to the
    # divergence curve and n_radii to the correlation sum, and passing one to
    # the other reaches find_scaling_region as an unexpected keyword. Sharing
    # **invariant_kwargs between them made every block D2 fail silently into a
    # nan column [DD-95].
    _LYAP_ONLY = {"max_steps", "method", "radius", "min_neighbours", "units",
                  "stability_tolerance"}
    _D2_ONLY = {"n_radii", "radii"}
    def _kw_for(name):
        drop = _D2_ONLY if name == "lambda_1" else _LYAP_ONLY
        return {k: v for k, v in invariant_kwargs.items() if k not in drop}

    if block_invariants:
        wanted_block = (list(BLOCK_INVARIANTS) if block_invariants is True
                        else list(block_invariants))
        bad = set(wanted_block) - set(BLOCK_INVARIANTS)
        if bad:
            raise ParameterError(
                f"unknown block invariant(s) {sorted(bad)}; available: "
                f"{list(BLOCK_INVARIANTS)}")
        for label, x in series.items():
            suffix = f"_{label}" if label else ""
            try:
                embedded = _embed_block(np.asarray(x, dtype=float), rate,
                                        embed_m, embed_tau)
            except Exception as exc:
                # A silent nan looks like a measurement that came out empty.
                # Say what happened, once per block.
                warnings.warn(f"dynamics: could not embed block {label!r}: {exc}",
                              InferenceWarning, stacklevel=2)
                for name in wanted_block:
                    out[f"{prefix}{name}{suffix}"] = float("nan")
                    out[f"{prefix}{name}{suffix}_ok"] = 0.0
                continue
            for name in wanted_block:
                try:
                    result = (correlation_dimension(embedded, **_kw_for("D2"))
                              if name == "D2"
                              else lyapunov_max(embedded, **_kw_for("lambda_1")))
                    value, flagged = result.value, bool(result.warning)
                    if flagged:
                        warnings.warn(
                            f"dynamics: {name} on block {label!r} is not "
                            f"trustworthy -- {result.warning}",
                            InferenceWarning, stacklevel=2)
                except Exception as exc:
                    warnings.warn(f"dynamics: {name} on block {label!r} "
                                  f"failed: {exc}", InferenceWarning,
                                  stacklevel=2)
                    value, flagged = float("nan"), True
                out[f"{prefix}{name}{suffix}"] = float(value)
                # An estimate whose own diagnostics reject it must not sit in a
                # table looking like a measurement. A companion column says so,
                # because a warning scrolls past and a column does not [DD-63].
                out[f"{prefix}{name}{suffix}_ok"] = 0.0 if flagged else 1.0

    if invariants:
        wanted = (list(TRAJECTORY_MEASURES) if invariants is True
                  else list(invariants))
        bad = set(wanted) - set(TRAJECTORY_MEASURES)
        if bad:
            raise ParameterError(
                f"unknown invariant(s) {sorted(bad)}; available: "
                f"{list(TRAJECTORY_MEASURES)}")
        for name in wanted:
            try:
                if name == "D2":
                    v = correlation_dimension(source, **_kw_for("D2")).value
                elif name == "lambda_1":
                    v = lyapunov_max(source, **_kw_for("lambda_1")).value
                else:
                    v = k2_entropy(source, **_kw_for("K2")).value
                out[f"{prefix}{name}"] = float(v)
            except Exception:
                out[f"{prefix}{name}"] = float("nan")
    return out
