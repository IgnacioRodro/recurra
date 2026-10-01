"""The Recording: the object that travels through the whole pipeline.

Design note (DD-07) -- immutability with copy-on-write.

Every operation returns a *new* Recording; the input is never modified. That
gives predictable semantics and exact provenance. The cost would normally be
memory duplication, which is unacceptable for hour-long EEG. So the new
Recording *shares* the arrays that did not change (references, not copies)
and allocates only what is genuinely new. Arrays are marked read-only so a
shared buffer cannot be mutated behind another object's back.

Design note (DD-14) -- channels are stored as a mapping name -> 1-D array
rather than a single (T, D) matrix. Reason: a Recording routinely holds heterogeneous derived
components -- ``raw``, ``theta``, ``phase_theta``, ``amp_gamma`` -- each with
its own role and units. A matrix would force them into a single anonymous
axis and lose exactly the information the state-space builders need.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ..capabilities import VALID_ROLES, Capability, capabilities_from_roles
from ..diagnostics import QualityReport
from ..exceptions import ParameterError
from ..provenance import Step, array_digest, provenance_frame


def _freeze(arr: np.ndarray) -> np.ndarray:
    """Return a read-only view so shared buffers cannot be mutated."""
    a = np.asarray(arr)
    if a.flags.writeable:
        a = a.view()
        a.flags.writeable = False
    return a


@dataclass(frozen=True)
class Recording:
    """A set of time-aligned channels plus everything known about them.

    Parameters
    ----------
    channels
        Mapping ``name -> 1-D array``. All arrays share the same length.
    fs
        Sampling rate in Hz.
    roles
        Mapping ``name -> role``; role in
        ``{raw, band, phase, amplitude, inst_freq, state}``.
    """

    channels: Mapping[str, np.ndarray]
    fs: float
    roles: Mapping[str, str] = field(default_factory=dict)
    t0: float = 0.0
    units: Mapping[str, str] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[Step, ...] = ()
    quality: QualityReport | None = None
    caps: Capability = Capability.NONE

    # ---------------------------------------------------------------- init
    def __post_init__(self) -> None:
        if not self.channels:
            raise ParameterError("a Recording needs at least one channel")

        frozen = {str(k): _freeze(np.asarray(v)) for k, v in self.channels.items()}
        for name, arr in frozen.items():
            if arr.ndim != 1:
                raise ParameterError(
                    f"channel {name!r} must be 1-D, got shape {arr.shape}. "
                    "Pass a 2-D array to ingest() to have it split into channels."
                )
        lengths = {a.size for a in frozen.values()}
        if len(lengths) > 1:
            raise ParameterError(
                f"all channels of one Recording must share length; got {lengths}"
            )

        roles = dict(self.roles)
        for name in frozen:
            roles.setdefault(name, "raw")
        bad = {r for r in roles.values()} - VALID_ROLES
        if bad:
            raise ParameterError(f"unknown channel role(s) {bad}; valid: {sorted(VALID_ROLES)}")

        object.__setattr__(self, "channels", MappingProxyType(frozen))
        object.__setattr__(self, "roles", MappingProxyType(roles))
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))
        if self.caps is Capability.NONE:
            object.__setattr__(self, "caps", capabilities_from_roles(roles))

    # ---------------------------------------------------------- properties
    @property
    def names(self) -> list[str]:
        return list(self.channels)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_samples(self) -> int:
        return next(iter(self.channels.values())).size

    @property
    def duration(self) -> float:
        return self.n_samples / self.fs

    @property
    def subject(self) -> str:
        return str(self.meta.get("subject", "unknown"))

    def times(self) -> np.ndarray:
        return self.t0 + np.arange(self.n_samples) / self.fs

    # ------------------------------------------------------------- access
    def get(self, name: str) -> np.ndarray:
        try:
            return self.channels[name]
        except KeyError:
            raise ParameterError(
                f"no channel named {name!r}; available: {self.names}"
            ) from None

    def by_role(self, role: str) -> list[str]:
        """Channel names carrying a given role, in insertion order."""
        return [n for n in self.channels if self.roles.get(n) == role]

    def select(self, names: Sequence[str]) -> Recording:
        """A Recording restricted to some channels (shares their buffers)."""
        missing = [n for n in names if n not in self.channels]
        if missing:
            raise ParameterError(f"unknown channel(s) {missing}; available: {self.names}")
        return replace(
            self,
            channels={n: self.channels[n] for n in names},
            roles={n: self.roles[n] for n in names},
            units={n: self.units[n] for n in names if n in self.units},
            caps=Capability.NONE,  # recomputed in __post_init__
            provenance=self.provenance
            + (Step("select", {"names": list(names)}, inputs=tuple(names)),),
        )

    def as_array(self, names: Sequence[str] | None = None) -> np.ndarray:
        """Stack channels into a ``(T, D)`` array, in the given order."""
        names = list(self.channels) if names is None else list(names)
        return np.column_stack([self.get(n) for n in names])

    # ------------------------------------------------------- construction
    def with_channels(
        self,
        new: Mapping[str, np.ndarray],
        *,
        roles: Mapping[str, str] | None = None,
        units: Mapping[str, str] | None = None,
        step: Step | None = None,
        replace_all: bool = False,
        t0: float | None = None,
        keep: Iterable[str] | None = None,
    ) -> Recording:
        """Return a new Recording with channels added or replaced.

        Copy-on-write: unchanged channels are shared, not duplicated.
        ``replace_all=True`` drops existing channels (used when an operation
        changes the sample count, e.g. edge trimming or resampling).
        """
        if replace_all:
            base_ch: dict[str, np.ndarray] = {}
            base_roles: dict[str, str] = {}
            base_units: dict[str, str] = {}
            if keep:
                for n in keep:
                    base_ch[n] = self.channels[n]
                    base_roles[n] = self.roles[n]
                    if n in self.units:
                        base_units[n] = self.units[n]
        else:
            base_ch = dict(self.channels)
            base_roles = dict(self.roles)
            base_units = dict(self.units)

        base_ch.update({str(k): np.asarray(v) for k, v in new.items()})
        if roles:
            base_roles.update(roles)
        if units:
            base_units.update(units)
        base_roles = {k: v for k, v in base_roles.items() if k in base_ch}
        base_units = {k: v for k, v in base_units.items() if k in base_ch}

        return replace(
            self,
            channels=base_ch,
            roles=base_roles,
            units=base_units,
            t0=self.t0 if t0 is None else t0,
            caps=Capability.NONE,
            provenance=self.provenance + ((step,) if step else ()),
        )

    def with_meta(self, **kwargs) -> Recording:
        merged = dict(self.meta)
        merged.update(kwargs)
        return replace(self, meta=merged)

    def with_step(self, step: Step) -> Recording:
        return replace(self, provenance=self.provenance + (step,))

    # ---------------------------------------------------------- reporting
    def provenance_frame(self) -> pd.DataFrame:
        return provenance_frame(self.provenance)

    def channel_frame(self) -> pd.DataFrame:
        """One row per channel: role, units, basic statistics, digest.

        This is the table that gets exported to CSV so that any figure can
        be traced back to the exact data that produced it.
        """
        rows = []
        for name, arr in self.channels.items():
            a = np.asarray(arr, dtype=float)
            finite = a[np.isfinite(a)]
            rows.append(
                {
                    "subject": self.subject,
                    "channel": name,
                    "role": self.roles.get(name, "raw"),
                    "units": self.units.get(name, ""),
                    "n_samples": a.size,
                    "fs_hz": self.fs,
                    "duration_s": a.size / self.fs,
                    "mean": float(finite.mean()) if finite.size else np.nan,
                    "std": float(finite.std()) if finite.size else np.nan,
                    "min": float(finite.min()) if finite.size else np.nan,
                    "max": float(finite.max()) if finite.size else np.nan,
                    "n_nonfinite": int(a.size - finite.size),
                    "digest": array_digest(a),
                }
            )
        return pd.DataFrame(rows)

    def to_frame(self, names: Sequence[str] | None = None) -> pd.DataFrame:
        """Wide time-series DataFrame indexed by time. Use for small segments."""
        names = list(self.channels) if names is None else list(names)
        return pd.DataFrame({n: np.asarray(self.get(n)) for n in names}, index=self.times())

    def summary(self) -> str:
        role_counts: dict[str, int] = {}
        for r in self.roles.values():
            role_counts[r] = role_counts.get(r, 0) + 1
        roles = ", ".join(f"{k}:{v}" for k, v in sorted(role_counts.items()))
        q = self.quality.summary() if self.quality else "quality: not checked"
        return (
            f"Recording(subject={self.subject!r}, {self.n_channels} channels, "
            f"{self.n_samples} samples @ {self.fs:g} Hz = {self.duration:.2f} s)\n"
            f"  roles       : {roles}\n"
            f"  capabilities: {self.caps}\n"
            f"  steps       : {len(self.provenance)}\n"
            f"  {q}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<Recording {self.subject!r} {self.n_channels}ch "
            f"{self.n_samples}@{self.fs:g}Hz caps={self.caps}>"
        )
