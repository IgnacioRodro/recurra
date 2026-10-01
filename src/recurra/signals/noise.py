"""Coloured noise generators.

The Kramer & Eden style CFC generator starts from coloured noise rather than
pure sinusoids, because real neural signals are broadband and a sinusoidal
carrier produces recurrence structure that has nothing to do with coupling.
"""
from __future__ import annotations

import numpy as np

from ..config import SeedLike, resolve_rng


def colored_noise(n: int, beta: float = 1.0, *, fs: float = 1.0,
                  rng: SeedLike = None) -> np.ndarray:
    """Noise with power spectral density ~ 1/f**beta.

    beta = 0 white, 1 pink, 2 brown/red. Returned with zero mean, unit variance.
    """
    rng = resolve_rng(rng)
    n_fft = int(2 ** np.ceil(np.log2(max(n, 2)))) * 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    scale = np.ones_like(freqs)
    nz = freqs > 0
    scale[nz] = freqs[nz] ** (-beta / 2.0)
    scale[0] = 0.0  # kill DC

    phases = rng.uniform(0, 2 * np.pi, size=freqs.size)
    spectrum = scale * np.exp(1j * phases)
    spectrum[0] = 0.0
    if n_fft % 2 == 0:
        spectrum[-1] = np.abs(spectrum[-1])  # Nyquist must be real

    x = np.fft.irfft(spectrum, n=n_fft)[:n]
    std = x.std()
    return (x - x.mean()) / std if std > 0 else x


def white_noise(n: int, *, rng: SeedLike = None) -> np.ndarray:
    return colored_noise(n, beta=0.0, rng=rng)


def pink_noise(n: int, *, rng: SeedLike = None) -> np.ndarray:
    return colored_noise(n, beta=1.0, rng=rng)


def add_noise_at_snr(signal: np.ndarray, snr_db: float, *, beta: float = 0.0,
                     fs: float = 1.0, rng: SeedLike = None) -> np.ndarray:
    """Add coloured noise so the result has the requested SNR in dB."""
    if snr_db is None or not np.isfinite(snr_db):
        return signal
    rng = resolve_rng(rng)
    noise = colored_noise(signal.size, beta=beta, fs=fs, rng=rng)
    p_sig = float(np.mean(signal**2))
    p_noise = float(np.mean(noise**2))
    if p_noise == 0 or p_sig == 0:
        return signal
    target = p_sig / (10 ** (snr_db / 10.0))
    return signal + noise * np.sqrt(target / p_noise)
