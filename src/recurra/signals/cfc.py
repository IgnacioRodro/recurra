"""Synthetic signals with controlled cross-frequency coupling.

Kramer & Eden style generator, generalised. Start from coloured noise,
extract a slow phase component and a fast amplitude component by band-pass
filtering, then impose the coupling:

* PAC -- the fast amplitude is modulated by the slow phase through a window
  whose height scales with the coupling strength alpha in [0, 1].
* PPC -- the fast phase is locked to n:m multiples of the slow phase.
* AAC -- the two envelopes are correlated by a controlled factor.

Design note (DD-13). alpha is defined so that alpha=0 gives no coupling and
alpha=1 gives maximal modulation, and the *marginal spectra are preserved*
across alpha. Otherwise a coupling index would rise with alpha simply
because the spectrum changed, which is the pseudo-coupling artefact the
generator is supposed to let us study, not produce by accident.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import signal as sps

from ..config import SeedLike, resolve_rng, spawn_rngs
from ..exceptions import ParameterError
from .noise import add_noise_at_snr, colored_noise

Modality = Literal["pac", "ppc", "aac", "ppa", "none"]
MODALITIES = ("pac", "ppc", "aac", "ppa", "none")
SHAPES = ("hanning", "vonmises", "sine", "square")


@dataclass
class SyntheticSignal:
    """A generated signal plus the ground truth that produced it."""

    data: np.ndarray                    # (T, D)
    fs: float
    labels: list[str]
    truth: dict[str, Any] = field(default_factory=dict)
    components: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    def to_recording(self, subject: str = "synthetic"):
        from ..io.ingest import ingest

        rec = ingest(self.data, fs=self.fs, labels=self.labels,
                     subject=subject, meta={"truth": dict(self.truth)})
        return rec

    def truth_row(self) -> dict[str, Any]:
        """Flat dict for CSV export."""
        row = {"subject": self.truth.get("subject", "synthetic"),
               "n_samples": self.n_samples, "fs_hz": self.fs}
        for k, v in self.truth.items():
            row[k] = v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
        return row


def _warn_generator(message: str) -> None:
    from ..config import get_config

    if get_config().warn_on_caveat:
        import warnings

        from ..exceptions import ParameterWarning

        warnings.warn(f"generate_cfc: {message}", ParameterWarning, stacklevel=3)


def _modulation_window(phase: np.ndarray, *, shape: str, preferred_phase: float | None,
                       concentration: float) -> np.ndarray:
    """Modulating function of phase, normalised to [0, 1] with unit maximum."""
    if preferred_phase is None:
        # Distributed coupling: broad, smooth, one maximum per cycle
        w = 0.5 * (1.0 + np.cos(phase))
        return w

    d = np.angle(np.exp(1j * (phase - preferred_phase)))  # wrapped to [-pi, pi]
    if shape == "vonmises":
        w = np.exp(concentration * (np.cos(d) - 1.0))
    elif shape == "hanning":
        half = np.pi / max(concentration, 1e-6)
        inside = np.abs(d) < half
        w = np.zeros_like(d)
        w[inside] = 0.5 * (1.0 + np.cos(np.pi * d[inside] / half))
    elif shape == "sine":
        w = 0.5 * (1.0 + np.cos(d))
    elif shape == "square":
        half = np.pi / max(concentration, 1e-6)
        w = (np.abs(d) < half).astype(float)
    else:
        raise ParameterError(f"unknown modulation_shape {shape!r}; use one of {SHAPES}")
    m = w.max()
    return w / m if m > 0 else w


def _alpha_profile(alpha, n: int, nonstationarity: str | None,
                   onset: float | None, offset: float | None,
                   fs: float) -> np.ndarray:
    """Turn a scalar / array / callable alpha into a per-sample profile."""
    if callable(alpha):
        return np.clip(np.asarray(alpha(np.arange(n) / fs), dtype=float), 0.0, 1.0)
    if isinstance(alpha, np.ndarray):
        if alpha.size != n:
            raise ParameterError(f"alpha array has {alpha.size} samples, need {n}")
        return np.clip(alpha.astype(float), 0.0, 1.0)

    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ParameterError(f"alpha must be in [0,1], got {a}")

    prof = np.full(n, a)
    if nonstationarity in (None, "none"):
        return prof

    t = np.arange(n) / fs
    dur = n / fs
    i0 = int((onset if onset is not None else 0.33 * dur) * fs)
    i1 = int((offset if offset is not None else 0.66 * dur) * fs)
    i0, i1 = max(0, min(i0, n)), max(0, min(i1, n))

    if nonstationarity == "transient":
        prof = np.zeros(n)
        prof[i0:i1] = a
    elif nonstationarity == "ramp":
        prof = a * np.clip((t - t[i0]) / max(t[min(i1, n - 1)] - t[i0], 1e-9), 0, 1)
    elif nonstationarity == "intermittent":
        period = max((i1 - i0), 1)
        prof = a * ((np.arange(n) // period) % 2 == 1).astype(float)
    elif nonstationarity == "switching":
        prof = np.where(np.arange(n) < i0, a, a * 0.2)
    else:
        raise ParameterError(
            f"unknown nonstationarity {nonstationarity!r}; use transient/ramp/"
            "intermittent/switching"
        )
    return np.clip(prof, 0.0, 1.0)


def _ppc_deviation_scale(alpha) -> np.ndarray:
    """Phase-deviation amplitude that makes PLV rise monotonically with alpha.

    For a phase perturbed by a zero-mean deviation of standard deviation s,
    the phase-locking value is approximately exp(-s^2 / 2). Inverting that at
    alpha gives s = sqrt(-2 ln alpha), which diverges as alpha goes to zero, so
    the deviation is capped at a value that already destroys any locking.
    """
    a = np.clip(np.asarray(alpha, dtype=float), 1e-6, 1.0)
    return np.minimum(np.sqrt(-2.0 * np.log(a)), 12.0)


def _band_component(n, fs, band, rng, beta=1.0):
    """Coloured noise restricted to a band, unit variance."""
    x = colored_noise(n, beta=beta, fs=fs, rng=rng)
    lo, hi = band
    n_taps = int(np.clip(np.ceil(4 * fs / lo), 15, max(15, n // 3)))
    n_taps += 1 - n_taps % 2
    b = sps.firwin(n_taps, [lo, hi], pass_zero=False, fs=fs, window="hamming")
    y = sps.filtfilt(b, [1.0], x, padlen=min(3 * n_taps, n - 1))
    s = y.std()
    return y / s if s > 0 else y


def _add_artifacts(x, kinds, fs, rng, strength=5.0):
    x = x.copy()
    n = x.size
    for kind in kinds or ():
        if kind == "spike":
            idx = rng.integers(0, n, size=max(1, n // 5000))
            x[idx] += strength * x.std() * rng.choice([-1, 1], size=idx.size)
        elif kind == "drift":
            t = np.linspace(0, 1, n)
            x += strength * 0.2 * x.std() * np.sin(2 * np.pi * 0.05 * t * n / fs)
        elif kind == "saturation":
            lim = np.percentile(np.abs(x), 98)
            x = np.clip(x, -lim, lim)
        elif kind == "blink":
            for _ in range(max(1, n // 20000)):
                i = int(rng.integers(0, max(1, n - int(fs))))
                w = int(0.3 * fs)
                x[i:i + w] += strength * x.std() * np.hanning(w)[: max(0, min(w, n - i))]
        else:
            raise ParameterError(f"unknown artifact {kind!r}")
    return x


def generate_cfc(
    *,
    modality: Modality = "pac",
    duration: float = 20.0,
    fs: float = 500.0,
    f_low: float | Sequence[float] = 6.0,
    f_high: float | Sequence[float] = 60.0,
    bandwidth_low: float = 4.0,
    bandwidth_high: float = 20.0,
    alpha: float | np.ndarray | Callable = 0.6,
    preferred_phase: float | None = None,
    concentration: float = 2.0,
    modulation_shape: str = "hanning",
    nm_ratio: tuple[int, int] = (1, 1),
    aac_correlation: float | None = None,
    nonstationarity: str | None = None,
    onset: float | None = None,
    offset: float | None = None,
    noise_beta: float = 1.0,
    snr_db: float | None = 10.0,
    harmonic_contamination: float = 0.0,
    harmonic_sharpness: float = 30.0,
    artifacts: Sequence[str] | None = None,
    n_extra_channels: int = 0,
    return_components: bool = True,
    subject: str = "synthetic",
    seed: SeedLike = None,
    rng: SeedLike = None,
) -> SyntheticSignal:
    """Generate one signal with controlled cross-frequency coupling.

    Returns a :class:`SyntheticSignal` carrying the mixture *and* the
    underlying components (slow phase, fast envelope), so that any estimator
    can be checked against the values that actually generated the data.

    ``rng`` and ``seed`` are the same thing under two names: every other
    function of the library calls it ``rng`` [DD-118], and generators have
    traditionally called it ``seed``. Pass either, not both.
    """
    if modality not in MODALITIES:
        raise ParameterError(f"modality must be one of {MODALITIES}, got {modality!r}")
    seed = _one_seed(seed, rng, "generate_cfc")
    rng = resolve_rng(seed)
    n = int(round(duration * fs))
    if n < 64:
        raise ParameterError(f"duration too short: {n} samples")

    fl = float(f_low if np.isscalar(f_low) else np.mean(f_low))
    fh = float(f_high if np.isscalar(f_high) else np.mean(f_high))
    if fh <= fl:
        raise ParameterError(f"f_high ({fh}) must exceed f_low ({fl})")
    if fh + bandwidth_high / 2 >= fs / 2:
        raise ParameterError(
            f"f_high band reaches {fh + bandwidth_high / 2:g} Hz, at or above "
            f"Nyquist ({fs / 2:g} Hz). Raise fs or lower f_high."
        )

    band_low = (max(fl - bandwidth_low / 2, 0.5), fl + bandwidth_low / 2)
    band_high = (fh - bandwidth_high / 2, fh + bandwidth_high / 2)

    if modality == "ppc":
        # n:m locking means n cycles of the slow rhythm per m of the fast one,
        # so n/m must be close to f_high / f_low. Asking for 1:1 between 6 and
        # 60 Hz produces a "fast" component oscillating at 6 Hz, which any
        # band-pass at the high band then removes entirely [DD-48].
        implied = (nm_ratio[0] / nm_ratio[1])
        expected = fh / fl
        if not 0.5 <= implied / expected <= 2.0:
            _warn_generator(
                f"nm_ratio {nm_ratio[0]}:{nm_ratio[1]} implies a frequency ratio "
                f"of {implied:g}, but f_high/f_low is {expected:.3g}. The locked "
                f"component will oscillate near {fl * implied:.3g} Hz, outside "
                f"the high band {band_high[0]:g}-{band_high[1]:g} Hz, so the "
                f"coupling will not survive band-pass filtering. Use "
                f"nm_ratio=({round(expected)}, 1) or adjust the frequencies.")

    r_low, r_high, r_noise, r_art, r_extra = spawn_rngs(rng, 5)

    slow = _band_component(n, fs, band_low, r_low, beta=noise_beta)
    fast = _band_component(n, fs, band_high, r_high, beta=noise_beta)

    phi_low = np.angle(sps.hilbert(slow))
    amp_high_free = np.abs(sps.hilbert(fast))
    prof = _alpha_profile(alpha, n, nonstationarity, onset, offset, fs)

    components: dict[str, np.ndarray] = {}
    truth: dict[str, Any] = {
        "subject": subject, "modality": modality, "fs_hz": fs, "duration_s": duration,
        "f_low_hz": fl, "f_high_hz": fh,
        "band_low_hz": f"{band_low[0]:g}-{band_low[1]:g}",
        "band_high_hz": f"{band_high[0]:g}-{band_high[1]:g}",
        "alpha_mean": float(prof.mean()), "alpha_max": float(prof.max()),
        "preferred_phase_rad": preferred_phase,
        "concentration": concentration, "modulation_shape": modulation_shape,
        "nonstationarity": nonstationarity or "none",
        "snr_db": snr_db, "noise_beta": noise_beta,
        "harmonic_contamination": harmonic_contamination,
        "harmonic_sharpness": harmonic_sharpness,
        "nm_ratio": f"{nm_ratio[0]}:{nm_ratio[1]}",
        "seed": None if seed is None else str(seed),
    }

    if modality == "pac":
        w = _modulation_window(phi_low, shape=modulation_shape,
                               preferred_phase=preferred_phase,
                               concentration=concentration)
        # Modulation preserving mean envelope power across alpha
        gain = (1.0 - prof) + prof * w
        gain = gain / (gain.mean() if gain.mean() > 0 else 1.0)
        fast_mod = fast * gain
        x = slow + fast_mod
        components["slow"] = slow
        components["fast"] = fast_mod
        components["phase_low"] = phi_low
        components["amp_high"] = np.abs(sps.hilbert(fast_mod))

    elif modality == "ppc":
        nn, mm = nm_ratio
        # Locking is n:m plus a bounded phase deviation, not a convex mixture
        # of two unwrapped phases [DD-48]. The deviation is a random walk whose
        # spread grows as alpha falls, so PLV moves monotonically from 0 to 1.
        locked = (nn / mm) * np.unwrap(phi_low)
        walk = np.cumsum(r_high.standard_normal(n)) / np.sqrt(fs)
        walk = walk - walk.mean()
        spread = _ppc_deviation_scale(prof)
        phi_high = np.angle(np.exp(1j * (locked + spread * walk)))
        fast_c = amp_high_free * np.cos(phi_high)
        x = slow + fast_c
        components["slow"] = slow
        components["fast"] = fast_c
        components["phase_low"] = phi_low
        components["phase_high"] = phi_high
        components["phase_deviation"] = spread * walk

    elif modality == "aac":
        env_slow = np.abs(sps.hilbert(slow))
        env_slow_n = (env_slow - env_slow.mean()) / (env_slow.std() or 1.0)
        env_free = (amp_high_free - amp_high_free.mean()) / (amp_high_free.std() or 1.0)
        r = prof if aac_correlation is None else np.full(n, float(aac_correlation)) * prof
        env_mix = r * env_slow_n + np.sqrt(np.clip(1 - r**2, 0, 1)) * env_free
        env_mix = env_mix - env_mix.min() + 1e-3
        fast_c = fast / (amp_high_free + 1e-12) * env_mix
        x = slow + fast_c
        components["slow"] = slow
        components["fast"] = fast_c
        components["amp_low"] = env_slow
        components["amp_high"] = env_mix

    elif modality == "ppa":
        f_mid = np.sqrt(fl * fh)
        band_mid = (max(f_mid - bandwidth_low, 1.0), f_mid + bandwidth_low)
        mid = _band_component(n, fs, band_mid, r_extra, beta=noise_beta)
        phi_mid = np.angle(sps.hilbert(mid))
        w = _modulation_window(phi_low + phi_mid, shape=modulation_shape,
                               preferred_phase=preferred_phase,
                               concentration=concentration)
        gain = (1.0 - prof) + prof * w
        gain = gain / (gain.mean() or 1.0)
        fast_mod = fast * gain
        x = slow + mid + fast_mod
        components["slow"] = slow
        components["mid"] = mid
        components["fast"] = fast_mod
        components["phase_low"] = phi_low
        components["phase_mid"] = phi_mid
        components["amp_high"] = np.abs(sps.hilbert(fast_mod))

    else:  # "none" -- negative control, same spectrum, zero coupling
        x = slow + fast
        components["slow"] = slow
        components["fast"] = fast
        components["phase_low"] = phi_low
        components["amp_high"] = amp_high_free

    if harmonic_contamination > 0:
        # A non-sinusoidal slow rhythm fakes PAC: its harmonics reach the high
        # band, and their envelope is locked to the slow phase, so every
        # phase-amplitude index reports coupling where none exists [DD-64].
        #
        # The distortion is applied to the *phase*, not to the waveform. The
        # slow rhythm here is band-limited noise with a fluctuating amplitude,
        # and distorting that amplitude smears the harmonics instead of placing
        # them at multiples of the slow frequency -- which is why an earlier
        # version of this line produced no artefact at all.
        pulse = np.exp(harmonic_sharpness * np.cos(phi_low))
        pulse = (pulse - pulse.mean()) / (pulse.std() or 1.0)
        env_slow = np.abs(sps.hilbert(slow))
        x = x + harmonic_contamination * env_slow * pulse / (env_slow.std() or 1.0)

    if artifacts:
        x = _add_artifacts(x, artifacts, fs, r_art)
    if snr_db is not None:
        x = add_noise_at_snr(x, snr_db, beta=noise_beta, fs=fs, rng=r_noise)

    cols = [x]
    labels = ["signal"]
    for i in range(n_extra_channels):
        cols.append(colored_noise(n, beta=noise_beta, fs=fs, rng=r_extra))
        labels.append(f"extra{i:02d}")

    data = np.column_stack(cols)
    truth["alpha_profile_constant"] = bool(np.ptp(prof) < 1e-12)
    return SyntheticSignal(
        data=data, fs=fs, labels=labels, truth=truth,
        components=components if return_components else {},
    )


def _one_seed(seed, rng, where: str):
    """Accept ``seed`` or ``rng`` (one name, two spellings), never both."""
    if seed is not None and rng is not None:
        raise ParameterError(f"{where}: pass seed or rng, not both; they are the same thing")
    return rng if rng is not None else seed


def generate_corpus(grid: dict[str, Sequence], *, n_per_cell: int = 1,
                    seed: SeedLike = 0, rng: SeedLike = None,
                    **fixed) -> list[SyntheticSignal]:
    """Cartesian sweep over parameters. Each cell gets an independent stream.

    ``rng`` is accepted as another name for ``seed`` [DD-118].
    """
    import itertools

    seed = _one_seed(None if rng is not None and seed == 0 else seed, rng,
                     "generate_corpus")

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    total = len(combos) * n_per_cell
    rngs = spawn_rngs(seed, total)

    out: list[SyntheticSignal] = []
    idx = 0
    for combo in combos:
        params = dict(zip(keys, combo, strict=False))
        for rep in range(n_per_cell):
            sig = generate_cfc(**{**fixed, **params}, seed=rngs[idx],
                               subject=f"syn{idx:05d}")
            sig.truth["replicate"] = rep
            sig.truth["cell"] = "|".join(f"{k}={v}" for k, v in params.items())
            out.append(sig)
            idx += 1
    return out
