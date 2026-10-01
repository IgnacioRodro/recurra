"""Data-quality checks and diagnostic reports.

Design note (DD-06). Validation reports, it does not silence. A constant
channel, a saturated ADC or a signal too short for the requested embedding
are all things that produce plausible-looking numbers downstream. They are
detected here, recorded, and carried alongside the data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import get_config
from .exceptions import QualityError

SEVERITIES = ("info", "warning", "error")


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str
    channel: str
    message: str
    value: Any = None

    def to_row(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "channel": self.channel,
            "message": self.message,
            "value": self.value,
        }


@dataclass
class QualityReport:
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def add(self, code, severity, channel, message, value=None) -> None:
        self.issues.append(Issue(code, severity, channel, message, value))

    def by_severity(self, severity: str) -> list[Issue]:
        return [i for i in self.issues if i.severity == severity]

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["code", "severity", "channel", "message", "value"])
        return pd.DataFrame([i.to_row() for i in self.issues])

    def stats_frame(self) -> pd.DataFrame:
        if not self.stats:
            return pd.DataFrame()
        return pd.DataFrame(self.stats).T.rename_axis("channel").reset_index()

    def raise_if_error(self) -> None:
        errs = self.by_severity("error")
        if errs:
            raise QualityError("; ".join(f"[{e.code}] {e.channel}: {e.message}" for e in errs))

    def summary(self) -> str:
        if not self.issues:
            return "quality: clean"
        counts = {s: len(self.by_severity(s)) for s in SEVERITIES}
        return "quality: " + ", ".join(f"{n} {s}" for s, n in counts.items() if n)


def check_channels(
    channels: dict[str, np.ndarray],
    fs: float,
    *,
    saturation_tol: float = 1e-9,
    flat_run_fraction: float = 0.05,
    strict: bool | None = None,
) -> QualityReport:
    """Run the standard battery of input checks over a channel mapping."""
    rep = QualityReport()
    strict = get_config().strict_quality if strict is None else strict

    if fs is None or not np.isfinite(fs) or fs <= 0:
        rep.add("FS_INVALID", "error", "-", f"sampling rate must be positive and finite, got {fs}")

    lengths = {name: len(a) for name, a in channels.items()}
    if len(set(lengths.values())) > 1:
        rep.add("LENGTH_MISMATCH", "error", "-",
                f"channels have differing lengths: {lengths}")

    for name, arr in channels.items():
        a = np.asarray(arr, dtype=float)
        n = a.size
        finite = np.isfinite(a)
        n_bad = int((~finite).sum())

        if n == 0:
            rep.add("EMPTY", "error", name, "channel has no samples")
            continue

        if n_bad:
            sev = "error" if n_bad > 0.01 * n else "warning"
            rep.add("NONFINITE", sev, name,
                    f"{n_bad} non-finite samples ({100 * n_bad / n:.2f}%)", n_bad)

        clean = a[finite]
        if clean.size == 0:
            rep.add("ALL_NONFINITE", "error", name, "no finite samples")
            continue

        std = float(clean.std())
        if std == 0.0:
            rep.add("CONSTANT", "error", name, "channel is constant (zero variance)", 0.0)

        # Saturation: repeated extreme values suggest ADC clipping.
        lo, hi = float(clean.min()), float(clean.max())
        n_hi = int(np.sum(np.abs(clean - hi) <= saturation_tol * max(abs(hi), 1.0)))
        n_lo = int(np.sum(np.abs(clean - lo) <= saturation_tol * max(abs(lo), 1.0)))
        if max(n_hi, n_lo) > max(3, flat_run_fraction * clean.size):
            rep.add("SATURATION", "warning", name,
                    f"{max(n_hi, n_lo)} samples pinned at an extreme value "
                    "(possible clipping)", max(n_hi, n_lo))

        rep.stats[name] = {
            "n_samples": float(n),
            "duration_s": float(n / fs) if fs else float("nan"),
            "mean": float(clean.mean()),
            "std": std,
            "min": lo,
            "max": hi,
            "n_nonfinite": float(n_bad),
        }

    if strict:
        rep.raise_if_error()
    return rep


def check_length_for(n_samples: int, m: int, tau: int, label: str = "embedding") -> None:
    """Guard against embeddings that would leave too few points."""
    n_eff = n_samples - (m - 1) * tau
    if n_eff <= 2:
        raise QualityError(
            f"{label}: m={m}, tau={tau} leaves {n_eff} points from {n_samples} samples. "
            f"Reduce m or tau, or use a longer segment."
        )
