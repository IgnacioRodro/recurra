"""Hilbert decomposition into phase, amplitude and instantaneous frequency.

Design note (DD-12) -- edge policy.

The Hilbert transform is non-local: the analytic signal near the edges of a
finite record is contaminated. Combined with the filter transient this is a
classic source of spurious coupling at the start and end of every epoch. The
policy is explicit and defaults to trimming, with the number of discarded
samples recorded in provenance rather than quietly absorbed.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from ..capabilities import Capability, require
from ..exceptions import ParameterError
from ..provenance import Step

EDGE_POLICIES = ("trim", "mirror", "taper", "none")


def hilbert_components(x, *, edge_policy="trim", edge_samples=0, unwrap=False,
                       fs=1.0):
    """Return (phase, amplitude, inst_freq, n_trimmed) for a 1-D signal."""
    x = np.asarray(x, dtype=float)
    if edge_policy not in EDGE_POLICIES:
        raise ParameterError(f"edge_policy must be one of {EDGE_POLICIES}")

    n = x.size
    if edge_policy == "mirror":
        pad = min(edge_samples or n // 10, n - 1)
        xp = np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]]) if pad > 0 else x
        z = sps.hilbert(xp)
        z = z[pad:pad + n] if pad > 0 else z
        n_trim = 0
    elif edge_policy == "taper":
        w = sps.windows.tukey(n, alpha=0.1)
        z = sps.hilbert(x * w)
        n_trim = 0
    else:
        z = sps.hilbert(x)
        n_trim = int(edge_samples) if edge_policy == "trim" else 0

    phase = np.angle(z)
    amplitude = np.abs(z)
    unwrapped = np.unwrap(phase)
    inst_freq = np.gradient(unwrapped) * fs / (2 * np.pi)

    if unwrap:
        phase = unwrapped
    return phase, amplitude, inst_freq, n_trim


def analytic(rec, sources=None, *, edge_policy="trim", edge_samples=None,
             keep=("phase", "amplitude", "inst_freq"), unwrap=False,
             prefix=None):
    """Add phase/amplitude/inst-freq channels for the given band channels.

    ``edge_samples=None`` reuses the filter transient length recorded by the
    preceding bandpass step, so the two edge effects are handled together
    instead of being counted twice or forgotten.
    """
    require("analytic", rec.caps, Capability.RAW | Capability.BANDS,
            hint="run bandpass()/filterbank() first, or ingest kind='bands'.")

    if sources is None:
        sources = rec.by_role("band") or rec.by_role("raw")
    if isinstance(sources, str):
        sources = [sources]

    if edge_samples is None:
        edge_samples = 0
        for st in rec.provenance:
            if st.name == "bandpass":
                edge_samples = max(edge_samples, int(st.params.get("edge_samples", 0)))

    new: dict[str, np.ndarray] = {}
    roles: dict[str, str] = {}
    units: dict[str, str] = {}
    max_trim = 0

    for src in sources:
        base = prefix + src if prefix else src
        ph, am, fr, n_trim = hilbert_components(
            rec.get(src), edge_policy=edge_policy, edge_samples=edge_samples,
            unwrap=unwrap, fs=rec.fs,
        )
        max_trim = max(max_trim, n_trim)
        if "phase" in keep:
            new[f"phase_{base}"] = ph
            roles[f"phase_{base}"] = "phase"
            units[f"phase_{base}"] = "rad"
        if "amplitude" in keep:
            new[f"amp_{base}"] = am
            roles[f"amp_{base}"] = "amplitude"
            units[f"amp_{base}"] = rec.units.get(src, "a.u.")
        if "inst_freq" in keep:
            new[f"ifreq_{base}"] = fr
            roles[f"ifreq_{base}"] = "inst_freq"
            units[f"ifreq_{base}"] = "Hz"

    caveats = ()
    if edge_policy == "trim" and max_trim > 0:
        caveats = (
            f"{max_trim} samples trimmed from each edge "
            f"({2 * max_trim / rec.fs:.3f} s total) to remove filter and "
            "Hilbert transients",
        )

    step = Step(
        "analytic",
        {"edge_policy": edge_policy, "edge_samples": int(edge_samples),
         "keep": list(keep), "unwrap": unwrap},
        inputs=tuple(sources), outputs=tuple(new), caveats=caveats,
    )

    out = rec.with_channels(new, roles=roles, units=units, step=step)

    if edge_policy == "trim" and max_trim > 0:
        n = out.n_samples
        if 2 * max_trim >= n:
            raise ParameterError(
                f"edge trim ({max_trim} per side) would consume the whole "
                f"{n}-sample record. Use a longer segment or edge_policy='mirror'."
            )
        sl = slice(max_trim, n - max_trim)
        trimmed = {k: np.asarray(v)[sl] for k, v in out.channels.items()}
        from dataclasses import replace as _replace

        out = out.with_channels(
            trimmed, roles=dict(out.roles), units=dict(out.units), replace_all=True,
            t0=out.t0 + max_trim / out.fs,
            step=Step("edge_trim", {"per_side": int(max_trim),
                                    "n_before": n, "n_after": n - 2 * max_trim}),
        )
        out = _replace(out, fs=rec.fs)
    return out
