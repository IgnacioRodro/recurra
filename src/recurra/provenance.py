"""Provenance: an auditable record of everything done to the data.

Design note (DD-04). Principle P3 of the design plan says no simplification
may be silent. That is only enforceable if every operation leaves a trace
that travels with the data and can be exported next to the results. Any
number that ends up in a paper must be reconstructible from its provenance.
"""
from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ._version import __version__


def _jsonable(value: Any) -> Any:
    """Reduce a parameter value to something safe to store and serialise."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return {"__array__": list(value.shape), "dtype": str(value.dtype)}
    return repr(value)


@dataclass(frozen=True)
class Step:
    """One operation applied to the data."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    notes: str = ""
    #: Anything the step silently did that the user should know about
    caveats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", {k: _jsonable(v) for k, v in self.params.items()})

    def with_caveat(self, text: str) -> Step:
        return replace(self, caveats=self.caveats + (text,))

    def to_row(self, index: int = 0) -> dict[str, Any]:
        return {
            "order": index,
            "step": self.name,
            "timestamp": self.timestamp,
            "inputs": ";".join(self.inputs),
            "outputs": ";".join(self.outputs),
            "params": "; ".join(f"{k}={v}" for k, v in self.params.items()),
            "caveats": " | ".join(self.caveats),
            "notes": self.notes,
        }


def provenance_frame(steps) -> pd.DataFrame:
    """Render a sequence of Steps as a tidy DataFrame, ready for CSV export."""
    if not steps:
        return pd.DataFrame(
            columns=["order", "step", "timestamp", "inputs", "outputs", "params",
                     "caveats", "notes"]
        )
    return pd.DataFrame([s.to_row(i) for i, s in enumerate(steps)])


def environment_stamp() -> dict[str, str]:
    """Environment fingerprint, attached to every exported artefact."""
    import scipy

    return {
        "recurra_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def array_digest(arr: np.ndarray, n_bytes: int = 8) -> str:
    """Short stable digest of an array, to detect silent data changes."""
    a = np.ascontiguousarray(arr)
    h = hashlib.blake2b(a.tobytes(), digest_size=n_bytes)
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    return h.hexdigest()
