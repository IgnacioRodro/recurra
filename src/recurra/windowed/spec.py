"""Window definitions.

Design note (DD-39) -- one window spec, three ways to state it.

Users describe windows in whichever way the problem gives them: "four-second
windows every half second", "ten windows covering the record with 50%
overlap", or explicit sample indices. All three produce the same object, so
everything downstream sees one representation.

Windows are resolved against a record length at the moment they are used, not
when they are declared, so the same spec applies to subjects of different
length -- which is the normal case with real data [DD-18].
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..exceptions import ParameterError

UNITS = ("samples", "seconds")
ALIGNMENTS = ("left", "center", "right")


@dataclass(frozen=True)
class Window:
    """One resolved window: half-open sample interval [start, stop)."""

    index: int
    start: int
    stop: int
    fs: float
    t0: float = 0.0

    @property
    def n_samples(self) -> int:
        return self.stop - self.start

    @property
    def t_start(self) -> float:
        return self.t0 + self.start / self.fs

    @property
    def t_stop(self) -> float:
        return self.t0 + self.stop / self.fs

    @property
    def t_center(self) -> float:
        return 0.5 * (self.t_start + self.t_stop)

    @property
    def label(self) -> str:
        return f"w{self.index:03d}"

    def to_row(self) -> dict:
        return {"window": self.index, "window_label": self.label,
                "start_sample": self.start, "stop_sample": self.stop,
                "n_samples": self.n_samples,
                "t_start_s": self.t_start, "t_stop_s": self.t_stop,
                "t_center_s": self.t_center}


@dataclass(frozen=True)
class WindowSpec:
    """How to cut a record into windows.

    Give either ``size`` (with ``step`` or ``overlap``) or ``n_windows``
    (with ``overlap``). ``unit`` applies to ``size`` and ``step``.

    Parameters
    ----------
    size
        Window length, in ``unit``.
    step
        Advance between consecutive window starts, in ``unit``. Mutually
        exclusive with ``overlap``.
    overlap
        Fraction of a window shared with the next, in [0, 1).
        ``step = size * (1 - overlap)``.
    n_windows
        Number of windows to fit over the record. The size is derived so
        that the windows, at the requested overlap, span the whole record.
    drop_last
        Discard a final window shorter than ``size``. When False, the last
        window is pulled back so that it ends at the record end and keeps
        full length -- which introduces extra overlap there, recorded as a
        caveat rather than hidden.
    min_samples
        Reject windows shorter than this after resolution.
    """

    size: float | None = None
    step: float | None = None
    overlap: float | None = None
    n_windows: int | None = None
    unit: str = "seconds"
    align: str = "left"
    drop_last: bool = True
    min_samples: int = 16

    def __post_init__(self) -> None:
        if self.unit not in UNITS:
            raise ParameterError(f"unit must be one of {UNITS}, got {self.unit!r}")
        if self.align not in ALIGNMENTS:
            raise ParameterError(f"align must be one of {ALIGNMENTS}")
        if self.size is None and self.n_windows is None:
            raise ParameterError(
                "a WindowSpec needs either size= (with step= or overlap=) or "
                "n_windows= (with overlap=)"
            )
        if self.step is not None and self.overlap is not None:
            raise ParameterError("give step= or overlap=, not both")
        if self.overlap is not None and not 0.0 <= self.overlap < 1.0:
            raise ParameterError(
                f"overlap is a fraction in [0, 1), got {self.overlap}. "
                "Pass 0.5 for 50% overlap, not 50."
            )
        if self.size is not None and self.size <= 0:
            raise ParameterError(f"size must be positive, got {self.size}")
        if self.n_windows is not None and self.n_windows < 1:
            raise ParameterError(f"n_windows must be >= 1, got {self.n_windows}")

    # ------------------------------------------------------------ builders
    @classmethod
    def from_count(cls, n_windows: int, *, overlap: float = 0.0, **kw) -> WindowSpec:
        """Fit ``n_windows`` over whatever record it is applied to."""
        return cls(n_windows=n_windows, overlap=overlap, **kw)

    @classmethod
    def from_size(cls, size: float, *, step=None, overlap=None,
                  unit: str = "seconds", **kw) -> WindowSpec:
        return cls(size=size, step=step, overlap=overlap, unit=unit, **kw)

    # ------------------------------------------------------------ resolve
    def _to_samples(self, value: float, fs: float) -> int:
        return int(round(value * fs)) if self.unit == "seconds" else int(round(value))

    def resolve(self, n_samples: int, fs: float = 1.0,
                t0: float = 0.0) -> tuple[list[Window], list[str]]:
        """Turn the spec into concrete windows. Returns (windows, caveats)."""
        caveats: list[str] = []
        if n_samples < self.min_samples:
            raise ParameterError(
                f"record has {n_samples} samples, below min_samples={self.min_samples}"
            )

        if self.n_windows is not None:
            # Asking for k windows must give exactly k, spanning the whole
            # record. Deriving a step and walking forward leaves a tail
            # unanalysed whenever the arithmetic does not divide evenly, so
            # the starts are placed evenly between 0 and n - size instead.
            k = int(self.n_windows)
            ov = float(self.overlap or 0.0)
            if k == 1:
                size_s = n_samples
            else:
                denom = 1.0 + (k - 1) * (1.0 - ov)
                size_s = int(round(n_samples / denom))
            size_s = int(np.clip(size_s, self.min_samples, n_samples))
            starts = (np.linspace(0, n_samples - size_s, k).round().astype(int)
                      if k > 1 else np.array([0]))
            windows = [Window(i, int(a), int(a) + size_s, fs, t0)
                       for i, a in enumerate(starts)]
            actual_ov = 0.0
            if k > 1 and size_s > 0:
                gap = starts[1] - starts[0]
                actual_ov = max(0.0, 1.0 - gap / size_s)
                if abs(actual_ov - ov) > 0.02:
                    caveats.append(
                        f"requested {ov:.0%} overlap, realised {actual_ov:.0%}: the "
                        f"{k} windows were spread to cover the record exactly"
                    )
            if starts[0] != 0 or windows[-1].stop != n_samples:
                caveats.append("window grid does not span the record exactly")
            return windows, caveats

        size_s = self._to_samples(self.size, fs)
        if self.step is not None:
            step_s = self._to_samples(self.step, fs)
        elif self.overlap is not None:
            step_s = max(1, int(round(size_s * (1.0 - self.overlap))))
        else:
            step_s = size_s
        if step_s < 1:
            raise ParameterError(f"step resolves to {step_s} samples; increase it")

        if size_s > n_samples:
            raise ParameterError(
                f"window size {size_s} samples exceeds the {n_samples}-sample record. "
                f"Shorten the window or use n_windows=."
            )

        windows = []
        start = 0
        idx = 0
        while start + size_s <= n_samples:
            windows.append(Window(idx, start, start + size_s, fs, t0))
            idx += 1
            start += step_s

        if not windows:
            raise ParameterError(
                f"no window survived: size {size_s}, step {step_s}, record {n_samples}"
            )

        covered = windows[-1].stop
        if covered < n_samples:
            missing = n_samples - covered
            if self.drop_last:
                caveats.append(
                    f"the last {missing} samples ({missing / fs:.3f} s) do not fill a "
                    "whole window and are not analysed; set drop_last=False to include "
                    "them in a final window pulled back to the record end"
                )
            else:
                start = n_samples - size_s
                extra = windows[-1].stop - start
                windows.append(Window(len(windows), start, n_samples, fs, t0))
                caveats.append(
                    f"a final window was pulled back to end at the record end; it "
                    f"overlaps the previous one by {extra} samples "
                    f"({extra / size_s:.0%}), more than the rest"
                )
        return windows, caveats

    def describe(self, n_samples: int, fs: float = 1.0) -> pd.DataFrame:
        windows, caveats = self.resolve(n_samples, fs)
        df = pd.DataFrame([w.to_row() for w in windows])
        df["overlap_with_previous"] = [0] + [
            max(0, windows[i - 1].stop - windows[i].start) for i in range(1, len(windows))
        ]
        df["caveats"] = " | ".join(caveats)
        return df

    def summary(self, n_samples: int | None = None, fs: float = 1.0) -> str:
        parts = []
        if self.n_windows is not None:
            parts.append(f"{self.n_windows} windows")
        if self.size is not None:
            parts.append(f"size {self.size:g} {self.unit}")
        if self.step is not None:
            parts.append(f"step {self.step:g} {self.unit}")
        if self.overlap is not None:
            parts.append(f"overlap {self.overlap:.0%}")
        text = "WindowSpec(" + ", ".join(parts) + ")"
        if n_samples is not None:
            windows, caveats = self.resolve(n_samples, fs)
            text += (f"\n  resolves to {len(windows)} windows of "
                     f"{windows[0].n_samples} samples "
                     f"({windows[0].n_samples / fs:.2f} s)")
            for c in caveats:
                text += f"\n  caveat: {c}"
        return text

    def with_(self, **kw) -> WindowSpec:
        return replace(self, **kw)


def resolve_windows(spec, n_samples: int, fs: float = 1.0,
                    t0: float = 0.0) -> tuple[list[Window], list[str]]:
    """Accept a WindowSpec, an int (that many windows) or explicit intervals."""
    if isinstance(spec, WindowSpec):
        return spec.resolve(n_samples, fs, t0)
    if isinstance(spec, int):
        return WindowSpec(n_windows=spec).resolve(n_samples, fs, t0)
    if isinstance(spec, Sequence):
        windows = [Window(i, int(a), int(b), fs, t0) for i, (a, b) in enumerate(spec)]
        for w in windows:
            if w.stop > n_samples or w.start < 0 or w.n_samples < 1:
                raise ParameterError(f"explicit window {w.start}:{w.stop} out of range")
        return windows, []
    raise ParameterError(
        f"cannot interpret {type(spec).__name__} as windows; pass a WindowSpec, "
        "an integer number of windows, or a sequence of (start, stop) pairs"
    )
