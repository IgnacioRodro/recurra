"""ingest(): the single entry point for every kind of input.

Design note (DD-08) -- one funnel, eleven input shapes.

The library must accept: a scalar array, a multichannel matrix, a dict of
bands, a dict of already-computed Hilbert components, a complex analytic
signal, a prebuilt state space, a file, or an existing Recording. Rather
than a constructor per case, there is one function that recognises the
shape, assigns roles, derives capabilities and records how it interpreted
the input in the provenance trail.

The ``roles`` argument is what removes the ambiguity that would otherwise
be fatal: a (T, 3) array could be three channels, a (cos, sin, A) state
space, or three bands, and only the caller knows which.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..capabilities import VALID_ROLES, Capability
from ..core.recording import Recording
from ..diagnostics import check_channels
from ..exceptions import InferenceWarning, IngestError, ParameterError
from ..provenance import Step
from .readers import read_file

#: Heuristics used only when the caller gives no explicit role.
_ROLE_PATTERNS = [
    (re.compile(r"(^|_)(phase|phi|ph)(_|$)", re.I), "phase"),
    (re.compile(r"(^|_)(amp|amplitude|env|envelope)(_|$)", re.I), "amplitude"),
    (re.compile(r"(^|_)(freq|ifreq|inst_?freq)(_|$)", re.I), "inst_freq"),
    (re.compile(r"(^|_)(cos|sin)(_|$)", re.I), "state"),
]


def _guess_role(name: str, default: str = "raw") -> tuple[str, bool]:
    """Return (role, was_guessed)."""
    for pattern, role in _ROLE_PATTERNS:
        if pattern.search(name):
            return role, True
    return default, False


def _as_channel_dict(
    data, labels: Sequence[str] | None, prefix: str
) -> dict[str, np.ndarray]:
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        raise IngestError(
            "complex input: pass kind='analytic' so the real/imaginary parts "
            "are decomposed into phase and amplitude explicitly."
        )
    if arr.ndim == 1:
        names = list(labels) if labels else [prefix]
        if len(names) != 1:
            raise ParameterError(f"1-D input needs exactly one label, got {names}")
        return {str(names[0]): arr.astype(float)}
    if arr.ndim == 2:
        n_t, n_c = arr.shape
        if n_c > n_t:
            raise IngestError(
                f"input shape {arr.shape} has more columns than rows. recurra "
                "expects (n_samples, n_channels); transpose it if that is wrong."
            )
        names = list(labels) if labels else [f"{prefix}{i:02d}" for i in range(n_c)]
        if len(names) != n_c:
            raise ParameterError(f"got {len(names)} labels for {n_c} columns")
        return {str(names[i]): np.asarray(arr[:, i], dtype=float) for i in range(n_c)}
    raise IngestError(f"array input must be 1-D or 2-D, got shape {arr.shape}")


def ingest(
    source: Any,
    *,
    fs: float | None = None,
    labels: Sequence[str] | None = None,
    kind: str = "auto",
    roles: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    t0: float = 0.0,
    meta: Mapping[str, Any] | None = None,
    subject: str | None = None,
    validate: bool = True,
    strict: bool | None = None,
    **reader_kwargs,
) -> Recording:
    """Turn anything into a :class:`Recording`.

    ``meta["bands"]`` may carry ``{name: (low_hz, high_hz)}`` for channels
    decomposed elsewhere; the state-space builders use it to pick the slow
    and fast components and to check that an envelope low-pass does not cut
    into the phase band [DD-112]. :func:`bandpass` fills it automatically.

    Parameters
    ----------
    source
        One of: 1-D array, 2-D array ``(n_samples, n_channels)``, mapping of
        name -> 1-D array, a tuple ``(phase, amplitude)``, an existing
        Recording, or a path to a file.
    fs
        Sampling rate in Hz. Required unless the file format carries it.
    kind
        ``"auto"`` (default) or one of ``raw``, ``bands``, ``analytic``,
        ``statespace``. Fixes the default role for channels whose role is
        not given explicitly.
    roles
        Explicit ``name -> role`` mapping. This is the unambiguous way to say
        what each channel is; without it, names are pattern-matched and any
        guess is recorded as a caveat.

    Returns
    -------
    Recording
    """
    if isinstance(source, Recording):
        rec = source
        if subject:
            rec = rec.with_meta(subject=subject)
        return rec.with_step(Step("ingest", {"kind": "recording", "passthrough": True}))

    meta = dict(meta or {})
    roles_in = dict(roles or {})
    file_roles: dict[str, str] = {}
    caveats: list[str] = []

    # ---------------------------------------------------------- from file
    if isinstance(source, (str, os.PathLike)):
        fmt = None if kind == "auto" else (kind if kind in {"csv", "npy", "npz", "hdf5", "edf"} else None)
        channels, fs_file, file_roles, file_meta = read_file(
            source, fmt=fmt, fs=fs, **reader_kwargs
        )
        fs = fs if fs is not None else fs_file
        # Units the file declares are physical facts about the recording, so
        # they take effect unless the caller states otherwise [DD-88].
        file_units = file_meta.pop("units", None)
        if file_units and not units:
            units = file_units
        meta = {**file_meta, **meta}
        if "t0" in file_meta and t0 == 0.0:
            t0 = float(file_meta["t0"])
        kind = "auto" if kind in {"csv", "npy", "npz", "hdf5", "edf"} else kind

    # ------------------------------------------------ tuple (phase, ampl)
    elif isinstance(source, tuple) and len(source) == 2 and kind in ("auto", "analytic"):
        phase, amplitude = source
        channels = {"phase": np.asarray(phase, dtype=float).ravel(),
                    "amplitude": np.asarray(amplitude, dtype=float).ravel()}
        roles_in.setdefault("phase", "phase")
        roles_in.setdefault("amplitude", "amplitude")
        kind = "analytic"

    # --------------------------------------------------------- from dict
    elif isinstance(source, Mapping):
        channels = {}
        for name, value in source.items():
            a = np.asarray(value)
            if np.iscomplexobj(a):
                channels[f"{name}_phase"] = np.angle(a).astype(float)
                channels[f"{name}_amplitude"] = np.abs(a).astype(float)
                roles_in.setdefault(f"{name}_phase", "phase")
                roles_in.setdefault(f"{name}_amplitude", "amplitude")
                caveats.append(f"complex channel {name!r} split into phase and amplitude")
            elif a.ndim == 1:
                channels[str(name)] = a.astype(float)
            elif a.ndim == 2:
                for j in range(a.shape[1]):
                    channels[f"{name}_{j:02d}"] = np.asarray(a[:, j], dtype=float)
            else:
                raise IngestError(f"channel {name!r} has unsupported shape {a.shape}")

    # -------------------------------------------------------- from array
    else:
        prefix = "state" if kind == "statespace" else "ch"
        channels = _as_channel_dict(source, labels, prefix)

    if fs is None:
        raise IngestError(
            "fs is required: recurra cannot infer the sampling rate from this input. "
            "Pass fs=<Hz>."
        )

    # ------------------------------------------------------ assign roles
    default_role = {
        "auto": "raw", "raw": "raw", "bands": "band",
        "analytic": "amplitude", "statespace": "state",
    }.get(kind, "raw")

    final_roles: dict[str, str] = {}
    for name in channels:
        if name in roles_in:
            final_roles[name] = roles_in[name]
        elif name in file_roles:
            final_roles[name] = file_roles[name]
        elif kind == "statespace":
            final_roles[name] = "state"
        else:
            guessed, was_guessed = _guess_role(name, default_role)
            final_roles[name] = guessed
            if was_guessed and guessed != default_role:
                caveats.append(
                    f"role of channel {name!r} inferred as {guessed!r} from its name; "
                    "pass roles= to be explicit"
                )

    bad = set(final_roles.values()) - VALID_ROLES
    if bad:
        raise ParameterError(f"unknown role(s) {bad}; valid: {sorted(VALID_ROLES)}")

    if subject:
        meta["subject"] = subject
    meta.setdefault("subject", "unknown")

    quality = check_channels(channels, fs, strict=strict) if validate else None

    step = Step(
        name="ingest",
        params={
            "kind": kind, "fs": fs, "n_channels": len(channels),
            "n_samples": int(next(iter(channels.values())).size),
            "validate": validate,
        },
        outputs=tuple(channels),
        caveats=tuple(caveats),
        notes=f"source={type(source).__name__}",
    )

    rec = Recording(
        channels=channels, fs=float(fs), roles=final_roles, t0=float(t0),
        units=dict(units or {}), meta=meta, provenance=(step,),
        quality=quality, caps=Capability.NONE,
    )

    from ..config import get_config

    if caveats and get_config().warn_on_caveat:
        import warnings

        for c in caveats:
            warnings.warn(f"ingest: {c}", InferenceWarning, stacklevel=2)
    return rec
