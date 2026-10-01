"""Band-pass filtering.

Design note (DD-11) -- filter order and edge effects.

Cross-frequency coupling is notoriously sensitive to filter choice: too
narrow a band manufactures a sinusoid where none exists, and a filter with
too few cycles smears the phase. Aru et al. (2015) list filtering as a main
source of spurious PAC. So:

* ``order="auto"`` sets the FIR length from the *low* edge of the band,
  guaranteeing a minimum number of cycles in the impulse response (default
  3), which is the standard EEG recommendation.
* Zero-phase filtering by default (``filtfilt``): a phase-distorting filter
  invalidates any phase-based coupling measure.
* The number of samples that are unreliable at each edge is computed and
  reported, not ignored.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from ..capabilities import Capability, require
from ..exceptions import ParameterError
from ..provenance import Step


@dataclass(frozen=True)
class FilterDesign:
    method: str
    band: tuple[float, float]
    fs: float
    order: int
    coeffs: tuple
    edge_samples: int
    cycles: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.method == "fir":
            b = self.coeffs[0]
            return sps.filtfilt(b, [1.0], x, padlen=min(3 * len(b), x.size - 1))
        sos = self.coeffs[0]
        return sps.sosfiltfilt(sos, x)


def design_bandpass(band, fs, *, method="fir", order="auto", min_cycles=3.0,
                    transition=None) -> FilterDesign:
    """Design a band-pass filter and report its edge cost."""
    lo, hi = float(band[0]), float(band[1])
    nyq = fs / 2.0
    if not (0 < lo < hi < nyq):
        raise ParameterError(
            f"band {band} invalid for fs={fs} Hz (need 0 < low < high < {nyq})"
        )

    if order == "auto":
        # FIR length covering min_cycles of the slowest component
        n_taps = int(np.ceil(min_cycles * fs / lo))
        n_taps += 1 - (n_taps % 2)  # force odd -> exact linear phase
        cap = int(4 * fs)
        if n_taps > cap:
            # filtfilt needs about three filter lengths of data, so the
            # length is capped at four seconds; below ~1.5 Hz at five cycles
            # the cap binds, and the impulse response then holds fewer
            # cycles than asked. That is reported, not absorbed.
            import warnings

            from ..exceptions import CaveatWarning

            warnings.warn(
                f"bandpass: {min_cycles:g} cycles of {lo:g} Hz need {n_taps} "
                f"taps; capped at {cap} ({cap * lo / fs:.2f} cycles). Lower "
                "min_cycles, or pass order= explicitly to override.",
                CaveatWarning, stacklevel=3)
            n_taps = cap
        n_taps = max(15, n_taps)
        order_ = n_taps
    else:
        order_ = int(order)

    if method == "fir":
        n_taps = order_ if order_ % 2 == 1 else order_ + 1
        max_taps = 2 * 10**5
        if n_taps > max_taps:
            raise ParameterError(f"FIR would need {n_taps} taps; lower min_cycles or fs")
        b = sps.firwin(n_taps, [lo, hi], pass_zero=False, fs=fs, window="hamming")
        edge = n_taps
        cycles = n_taps * lo / fs
        return FilterDesign("fir", (lo, hi), fs, n_taps, (b,), edge, cycles)

    if method in ("iir", "butter"):
        n = 4 if order == "auto" else int(order)
        sos = sps.butter(n, [lo, hi], btype="bandpass", fs=fs, output="sos")
        edge = int(np.ceil(3 * fs / lo))
        return FilterDesign("iir", (lo, hi), fs, n, (sos,), edge, 3.0)

    raise ParameterError(f"unknown filter method {method!r}; use 'fir' or 'iir'")


def _filter_array(x, band, fs, **kw) -> tuple[np.ndarray, FilterDesign]:
    d = design_bandpass(band, fs, **kw)
    return d.apply(np.asarray(x, dtype=float)), d


def bandpass(rec, band, *, source=None, name=None, method="fir", order="auto",
             min_cycles=3.0, role="band"):
    """Band-pass one channel of a Recording, returning a new Recording."""
    require("bandpass", rec.caps, Capability.RAW,
            hint="bandpass needs a raw time series; ingest with kind='raw'.")
    source = source or rec.by_role("raw")[0]
    name = name or f"{source}_{band[0]:g}-{band[1]:g}Hz"

    y, design = _filter_array(rec.get(source), band, rec.fs,
                              method=method, order=order, min_cycles=min_cycles)
    step = Step(
        "bandpass",
        {"band": list(band), "method": design.method, "order": design.order,
         "min_cycles": min_cycles, "edge_samples": design.edge_samples,
         "effective_cycles": round(design.cycles, 2)},
        inputs=(source,), outputs=(name,),
        caveats=(
            f"first/last ~{design.edge_samples} samples "
            f"({design.edge_samples / rec.fs:.3f} s) are filter-transient",
        ),
    )
    out = rec.with_channels({name: y}, roles={name: role}, step=step)
    # Register the band so downstream builders can pick the slow/fast component
    # by frequency instead of guessing from channel names.
    registry = dict(out.meta.get("bands", {}))
    registry[name] = (float(band[0]), float(band[1]))
    return out.with_meta(bands=registry)


def filterbank(rec, bands: Mapping[str, Sequence[float]], *, source=None, **kw):
    """Apply several band-passes at once. Returns one Recording with all bands."""
    out = rec
    for label, band in bands.items():
        out = bandpass(out, band, source=source, name=label, **kw)
    return out


def notch(rec, freq, *, source=None, name=None, q=30.0):
    source = source or rec.by_role("raw")[0]
    name = name or source
    b, a = sps.iirnotch(freq, q, rec.fs)
    y = sps.filtfilt(b, a, np.asarray(rec.get(source), dtype=float))
    return rec.with_channels(
        {name: y}, roles={name: rec.roles.get(source, "raw")},
        step=Step("notch", {"freq": freq, "q": q}, inputs=(source,), outputs=(name,)),
    )


def resample(rec, fs_new, *, method="poly"):
    """Resample every channel. Changes sample count, so channels are replaced."""
    from fractions import Fraction

    ratio = Fraction(fs_new / rec.fs).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    new = {}
    for n, a in rec.channels.items():
        x = np.asarray(a, dtype=float)
        new[n] = sps.resample_poly(x, up, down) if method == "poly" else sps.resample(
            x, int(round(x.size * fs_new / rec.fs))
        )
    from dataclasses import replace as _replace

    out = rec.with_channels(
        new, roles=dict(rec.roles), replace_all=True,
        step=Step("resample", {"fs_old": rec.fs, "fs_new": fs_new,
                               "up": up, "down": down, "method": method}),
    )
    return _replace(out, fs=float(fs_new))
