"""CSV export of every intermediate result.

Design note (DD-10). Requirement from the project: intermediate results --
metrics, parameters, thresholds, quality reports -- must land in CSV so they
can go straight into a paper. Two rules follow:

1. Every exported table carries a sidecar ``*_provenance.csv`` and an
   environment stamp. A number in a paper must be traceable to the code
   version, parameters and input digest that produced it.
2. Tables are *tidy* (long format, one observation per row). Wide tables are
   convenient to look at and painful to aggregate over subjects and windows.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import pandas as pd

from ..config import output_path
from ..provenance import environment_stamp, provenance_frame


def _stamp_frame(extra: Mapping[str, Any] | None = None) -> pd.DataFrame:
    stamp = environment_stamp()
    if extra:
        stamp.update({str(k): str(v) for k, v in extra.items()})
    return pd.DataFrame({"key": list(stamp), "value": [str(v) for v in stamp.values()]})


def export_frame(
    df: pd.DataFrame,
    name: str,
    *,
    subdir: str | None = None,
    provenance=None,
    stamp_extra: Mapping[str, Any] | None = None,
    index: bool = False,
) -> str:
    """Write a DataFrame to CSV plus its provenance and environment sidecars."""
    path = output_path(f"{name}.csv", subdir)
    df.to_csv(path, index=index)

    base = os.path.splitext(path)[0]
    _stamp_frame(stamp_extra).to_csv(f"{base}_environment.csv", index=False)
    if provenance is not None:
        provenance_frame(provenance).to_csv(f"{base}_provenance.csv", index=False)
    return path


def export_recording(rec, name: str | None = None, *, subdir: str | None = None,
                     include_timeseries: bool = False) -> dict[str, str]:
    """Export everything known about a Recording as CSV tables."""
    name = name or f"recording_{rec.subject}"
    written: dict[str, str] = {}

    written["channels"] = export_frame(
        rec.channel_frame(), f"{name}_channels", subdir=subdir,
        provenance=rec.provenance,
        stamp_extra={"subject": rec.subject, "fs_hz": rec.fs, "n_samples": rec.n_samples},
    )
    if rec.quality is not None:
        written["quality"] = export_frame(
            rec.quality.to_frame(), f"{name}_quality", subdir=subdir
        )
    if include_timeseries:
        written["timeseries"] = export_frame(
            rec.to_frame().rename_axis("time_s").reset_index(),
            f"{name}_timeseries", subdir=subdir,
        )
    return written


class CsvExporter:
    """Accumulates tidy rows across many subjects/windows, writes once.

    Used by corpus-level runs, where writing one file per subject would
    produce thousands of files nobody can aggregate.
    """

    def __init__(self, name: str, *, subdir: str | None = None):
        self.name = name
        self.subdir = subdir
        self._rows: list[dict[str, Any]] = []
        self._provenance: list = []

    def add(self, **row) -> CsvExporter:
        self._rows.append(dict(row))
        return self

    def add_many(self, rows) -> CsvExporter:
        self._rows.extend(dict(r) for r in rows)
        return self

    def add_frame(self, df: pd.DataFrame, **constant) -> CsvExporter:
        for _, r in df.iterrows():
            self._rows.append({**constant, **r.to_dict()})
        return self

    def track(self, provenance) -> CsvExporter:
        self._provenance.extend(provenance)
        return self

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def write(self, **stamp_extra) -> str:
        return export_frame(
            self.frame, self.name, subdir=self.subdir,
            provenance=self._provenance or None, stamp_extra=stamp_extra,
        )


def set_output_dir(path: str) -> None:
    """Convenience wrapper so users do not have to import config."""
    from ..config import set_config

    set_config(output_dir=path)
    os.makedirs(path, exist_ok=True)
