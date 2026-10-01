"""The Threshold object."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Threshold:
    """A recurrence threshold, with the evidence that it is the right one.

    ``value`` is a scalar for global modes, a per-point array for ``fan``
    (fixed amount of neighbours), or a per-block tuple for ``per_block``.
    ``achieved_rr`` is the recurrence rate actually obtained on the sample,
    which is what a paper should report -- not the target that was asked for.
    """

    value: float | np.ndarray
    mode: str
    metric: str = "euclidean"
    scope: str = "global"
    theiler: int = 0
    target: float | None = None
    achieved_rr: float | None = None
    n_samples: int = 0
    per_block: tuple[float, ...] | None = None
    block_labels: tuple[str, ...] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_pointwise(self) -> bool:
        """True when every point carries its own threshold (FAN)."""
        return isinstance(self.value, np.ndarray) and self.value.ndim == 1

    @property
    def is_per_block(self) -> bool:
        return self.per_block is not None

    @property
    def scalar(self) -> float:
        """A single representative number, for reporting."""
        if self.is_pointwise:
            return float(np.median(self.value))
        if self.is_per_block:
            return float(np.max(self.per_block))
        return float(self.value)

    def to_frame(self) -> pd.DataFrame:
        row = {
            "mode": self.mode, "metric": self.metric, "scope": self.scope,
            "epsilon": self.scalar, "theiler": self.theiler,
            "target_rr": self.target, "achieved_rr": self.achieved_rr,
            "n_samples": self.n_samples,
            "pointwise": self.is_pointwise, "per_block": self.is_per_block,
        }
        if self.is_per_block and self.block_labels:
            for lbl, v in zip(self.block_labels, self.per_block, strict=False):
                row[f"epsilon_{lbl}"] = float(v)
        row.update({k: v for k, v in self.diagnostics.items()
                    if isinstance(v, (int, float, str, bool))})
        return pd.DataFrame([row])

    def summary(self) -> str:
        if self.is_pointwise:
            v = (f"per-point (median {np.median(self.value):.4g}, "
                 f"range {np.min(self.value):.4g}-{np.max(self.value):.4g})")
        elif self.is_per_block:
            v = "per-block " + ", ".join(
                f"{lbl}={e:.4g}" for lbl, e in zip(self.block_labels or (),
                                                   self.per_block or (), strict=False))
        else:
            v = f"{float(self.value):.6g}"
        lines = [f"Threshold(mode={self.mode!r}, scope={self.scope!r}) epsilon = {v}",
                 f"  metric      : {self.metric}, Theiler window {self.theiler}"]
        if self.target is not None:
            lines.append(f"  target RR   : {self.target:.4%}")
        if self.achieved_rr is not None:
            lines.append(f"  achieved RR : {self.achieved_rr:.4%} "
                         f"(estimated from {self.n_samples} sampled pairs)")
        for key in ("criterion", "iterations", "warning"):
            if key in self.diagnostics:
                lines.append(f"  {key:<12s}: {self.diagnostics[key]}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Threshold {self.mode} eps={self.scalar:.4g} rr={self.achieved_rr}>"
