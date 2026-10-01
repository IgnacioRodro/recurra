"""Run reports: a plain-text account of what an execution did and produced.

Design note (DD-37) -- a run should explain itself.

CSV tables carry the numbers and provenance sidecars carry the operations,
but neither answers the question a user actually asks three weeks later:
*what did that run do, and which of these files is which?* A RunReport
accumulates sections, notes, timings, warnings and an index of every artefact
written, then writes one readable TXT next to the outputs.

It records rather than computes. Anything in the report was produced by the
analysis; the report itself never derives a result.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import get_config, output_path
from .provenance import environment_stamp


def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if x < 1024 or unit == "GiB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} GiB"


@dataclass
class Artifact:
    path: str
    kind: str                 # "table" | "figure" | "file"
    description: str = ""
    rows: int | None = None
    columns: int | None = None
    section: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0


@dataclass
class Section:
    title: str
    started: float
    notes: list[str] = field(default_factory=list)
    results: list[tuple[str, Any]] = field(default_factory=list)
    elapsed: float | None = None


@dataclass
class RunReport:
    """Accumulates what a run did; writes a TXT summary at the end."""

    name: str = "run"
    title: str = ""
    subdir: str | None = None
    sections: list[Section] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    _current: Section | None = field(default=None, repr=False)

    # ------------------------------------------------------------ record
    def section(self, title: str) -> RunReport:
        """Open a new section; closes the previous one and times it."""
        now = time.time()
        if self._current is not None:
            self._current.elapsed = now - self._current.started
        self._current = Section(title=title, started=now)
        self.sections.append(self._current)
        return self

    def note(self, text: str) -> RunReport:
        (self._current.notes if self._current else self.warnings).append(str(text))
        return self

    def result(self, label: str, value: Any) -> RunReport:
        if self._current is None:
            self.section("results")
        self._current.results.append((label, value))
        return self

    def results(self, **kwargs) -> RunReport:
        for k, v in kwargs.items():
            self.result(k, v)
        return self

    def warn(self, text: str) -> RunReport:
        self.warnings.append(str(text))
        return self

    def parameter(self, **kwargs) -> RunReport:
        self.parameters.update(kwargs)
        return self

    def table(self, path: str, description: str = "", frame=None) -> str:
        """Register a CSV table. Returns the path, so it can wrap an export."""
        rows = cols = None
        if frame is not None and hasattr(frame, "shape"):
            rows, cols = int(frame.shape[0]), int(frame.shape[1])
        self.artifacts.append(Artifact(path, "table", description, rows, cols,
                                       self._current.title if self._current else ""))
        return path

    def figure(self, path: str, description: str = "", kind: str = "single") -> str:
        self.artifacts.append(Artifact(path, f"figure ({kind})", description,
                                       section=self._current.title if self._current else ""))
        return path

    def file(self, path: str, description: str = "") -> str:
        self.artifacts.append(Artifact(path, "file", description,
                                       section=self._current.title if self._current else ""))
        return path

    # ------------------------------------------------- convenience wrappers
    def export(self, frame: pd.DataFrame, name: str, description: str = "",
               **kwargs) -> str:
        """Export a DataFrame to CSV and register it in one call."""
        from .io.export import export_frame

        path = export_frame(frame, name, subdir=kwargs.pop("subdir", self.subdir),
                            **kwargs)
        return self.table(path, description, frame)

    def save(self, fig, filename: str, description: str = "",
             kind: str = "single", **kwargs) -> str:
        """Save a figure and register it in one call."""
        from .viz.style import save_figure

        path = save_figure(fig, filename, subdir=kwargs.pop("subdir", self.subdir),
                           **kwargs)
        return self.figure(path, description, kind)

    # ------------------------------------------------------------- render
    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def artifact_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "section": a.section, "kind": a.kind, "name": a.name,
            "rows": a.rows, "columns": a.columns,
            "size_bytes": a.size, "description": a.description, "path": a.path,
        } for a in self.artifacts])

    def render(self) -> str:
        w = 78
        rule = "=" * w
        thin = "-" * w
        stamp = environment_stamp()
        out: list[str] = [
            rule,
            (self.title or f"recurra run report: {self.name}").upper(),
            rule,
            "",
            f"generated (UTC) : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"total runtime   : {self.elapsed:.1f} s",
            f"recurra version : {stamp['recurra_version']}",
            f"python          : {stamp['python']}",
            f"platform        : {stamp['platform']}",
            f"numpy / scipy   : {stamp['numpy']} / {stamp['scipy']}",
            f"output directory: {os.path.abspath(get_config().output_dir)}",
            "",
        ]

        if self.parameters:
            out += ["PARAMETERS", thin]
            width = max(len(k) for k in self.parameters)
            for k, v in self.parameters.items():
                out.append(f"  {k:<{width}} : {v}")
            out.append("")

        out += ["WHAT THIS RUN DID", thin]
        for i, sec in enumerate(self.sections, 1):
            elapsed = sec.elapsed if sec.elapsed is not None else (
                time.time() - sec.started)
            out.append(f"  {i}. {sec.title}   [{elapsed:.1f} s]")
            for note in sec.notes:
                out.append(f"       - {note}")
            for label, value in sec.results:
                out.append(f"       * {label}: {value}")
        out.append("")

        tables = [a for a in self.artifacts if a.kind == "table"]
        figures = [a for a in self.artifacts if a.kind.startswith("figure")]
        others = [a for a in self.artifacts if a.kind == "file"]

        if tables:
            out += [f"TABLES WRITTEN ({len(tables)})", thin]
            for a in tables:
                shape = (f"{a.rows} rows x {a.columns} cols"
                         if a.rows is not None else "")
                out.append(f"  {a.name}")
                detail = "  ".join(x for x in (shape, _fmt_bytes(a.size)) if x)
                out.append(f"      {detail}")
                if a.description:
                    out.append(f"      {a.description}")
            out += ["",
                    "  Each table has _environment.csv and, where applicable,",
                    "  _provenance.csv sidecars recording how it was produced.",
                    ""]

        if figures:
            out += [f"FIGURES WRITTEN ({len(figures)})", thin]
            for a in figures:
                out.append(f"  {a.name}   [{a.kind}]  {_fmt_bytes(a.size)}")
                if a.description:
                    out.append(f"      {a.description}")
            out.append("")

        if others:
            out += [f"OTHER FILES ({len(others)})", thin]
            for a in others:
                out.append(f"  {a.name}  {_fmt_bytes(a.size)}")
                if a.description:
                    out.append(f"      {a.description}")
            out.append("")

        if self.warnings:
            out += [f"WARNINGS AND CAVEATS ({len(self.warnings)})", thin]
            for warning in self.warnings:
                out.append(f"  ! {warning}")
            out.append("")

        total = sum(a.size for a in self.artifacts)
        out += [thin,
                f"{len(self.artifacts)} artefacts, {_fmt_bytes(total)} total",
                rule]
        return "\n".join(out)

    def write(self, filename: str | None = None, *, also_csv: bool = True) -> str:
        """Write the TXT summary (and optionally an artefact index CSV)."""
        if self._current is not None and self._current.elapsed is None:
            self._current.elapsed = time.time() - self._current.started
        path = output_path(filename or f"{self.name}_report.txt", self.subdir)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.render())
            fh.write("\n")
        if also_csv and self.artifacts:
            from .io.export import export_frame

            export_frame(self.artifact_frame(), f"{self.name}_artifact_index",
                         subdir=self.subdir)
        return path

    def __str__(self) -> str:  # pragma: no cover
        return self.render()
