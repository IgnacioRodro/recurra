"""Reading a directory of recordings, with the labels that go with them.

Design note (DD-89) -- a corpus is files plus a table, and the join must be
checked.

Everything downstream -- the batch, the comparability contract, the classifier
-- needs a mapping from subject to group. In practice that mapping lives in a
spreadsheet beside the recordings, and the commonest way a study goes wrong is
a silent mismatch between the two: a subject in the table with no file, a file
with no row, an identifier that differs by a prefix or by case.

``read_corpus`` therefore reports the join rather than performing it quietly.
Files without a row and rows without a file are both listed, and the caller
decides whether that is expected.

The BIDS convention -- ``sub-XXX`` directories and a ``participants.tsv`` keyed
by ``participant_id`` -- is recognised because it is common, not because the
reader validates a BIDS dataset. It does not: it reads files and joins a table.
"""
from __future__ import annotations

import pathlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..exceptions import IngestError, ParameterError

SUBJECT_PATTERN = re.compile(r"(sub-[A-Za-z0-9]+)")


@dataclass
class CorpusIndex:
    """What was found on disk, what the table said, and how they joined."""

    recordings: dict[str, pathlib.Path]
    labels: dict[str, Any] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    unmatched_files: list[str] = field(default_factory=list)
    unmatched_rows: list[str] = field(default_factory=list)
    root: str = ""
    label_column: str | None = None

    def __len__(self) -> int:
        return len(self.recordings)

    @property
    def subjects(self) -> list[str]:
        return sorted(self.recordings)

    @property
    def complete(self) -> bool:
        """True when every file has a label and every label has a file."""
        return not self.unmatched_files and not self.unmatched_rows

    def labelled(self) -> dict[str, Any]:
        """Only the subjects that have both a file and a label."""
        return {k: v for k, v in self.labels.items() if k in self.recordings}

    def to_frame(self) -> pd.DataFrame:
        rows = [{"subject": s, "path": str(self.recordings[s]),
                 "label": self.labels.get(s), "has_label": s in self.labels}
                for s in self.subjects]
        rows += [{"subject": s, "path": None, "label": self.labels.get(s),
                  "has_label": True} for s in self.unmatched_rows]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [f"CorpusIndex: {len(self.recordings)} recording(s) under "
                 f"{self.root or '(unset)'}"]
        if self.label_column:
            counts = pd.Series(list(self.labelled().values())).value_counts()
            lines.append("  labels     : " + ", ".join(
                f"{k}={v}" for k, v in counts.items())
                + f"   (column {self.label_column!r})")
        if self.unmatched_files:
            lines.append(f"  {len(self.unmatched_files)} file(s) with no row: "
                         + ", ".join(self.unmatched_files[:6])
                         + (" ..." if len(self.unmatched_files) > 6 else ""))
        if self.unmatched_rows:
            lines.append(f"  {len(self.unmatched_rows)} row(s) with no file: "
                         + ", ".join(self.unmatched_rows[:6])
                         + (" ..." if len(self.unmatched_rows) > 6 else ""))
        if self.complete and self.label_column:
            lines.append("  the join is complete")
        return "\n".join(lines)


def _subject_from(path: pathlib.Path, root: pathlib.Path,
                  subject_from: str | Callable[[pathlib.Path], str]) -> str:
    if callable(subject_from):
        return str(subject_from(path))
    if subject_from == "bids":
        for part in (*path.relative_to(root).parts, path.stem):
            m = SUBJECT_PATTERN.search(part)
            if m:
                return m.group(1)
        return path.stem
    if subject_from == "stem":
        return path.stem
    if subject_from == "parent":
        return path.parent.name
    raise ParameterError(
        "subject_from must be 'bids', 'stem', 'parent' or a callable")


def read_corpus(root, *, pattern: str = "*.edf", recursive: bool = True,
                participants=None, label_column: str | None = None,
                subject_column: str = "participant_id",
                subject_from: str | Callable = "bids") -> CorpusIndex:
    """Index a directory of recordings and join it to a participants table.

    Nothing is read into memory: the index holds paths. Use
    :meth:`CorpusIndex.recordings` with :func:`recurra.ingest` to load one at a
    time, which is what a corpus of eight-minute recordings needs.

    Parameters
    ----------
    participants
        Path to a ``.tsv`` or ``.csv``, or a DataFrame. BIDS puts this at
        ``participants.tsv`` with a ``participant_id`` column; it is found
        automatically when present.
    label_column
        The column holding the group. When not given and the table has exactly
        one plausible column -- ``group``, ``diagnosis``, ``condition`` or
        ``label`` -- that one is used and named in the report.
    """
    root = pathlib.Path(root)
    if not root.is_dir():
        raise IngestError(f"{root} is not a directory")
    files = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    if not files:
        raise IngestError(
            f"no file matching {pattern!r} under {root}. Recordings elsewhere in "
            "the tree need recursive=True, or a different pattern.")

    recordings: dict[str, pathlib.Path] = {}
    for path in files:
        subject = _subject_from(path, root, subject_from)
        if subject in recordings:
            raise IngestError(
                f"two files map to subject {subject!r}: {recordings[subject]} and "
                f"{path}. Give subject_from= a rule that separates them.")
        recordings[subject] = path

    if participants is None:
        for candidate in ("participants.tsv", "participants.csv"):
            if (root / candidate).is_file():
                participants = root / candidate
                break

    table = None
    labels: dict[str, Any] = {}
    unmatched_rows: list[str] = []
    if participants is not None:
        if isinstance(participants, pd.DataFrame):
            table = participants.copy()
        else:
            p = pathlib.Path(participants)
            table = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
        if subject_column not in table.columns:
            raise IngestError(
                f"the participants table has no {subject_column!r} column; it "
                f"has {list(table.columns)}")
        if label_column is None:
            plausible = [c for c in ("group", "diagnosis", "condition", "label",
                                     "cohort") if c in table.columns]
            if len(plausible) != 1:
                raise ParameterError(
                    f"name the label_column; candidates found: {plausible or 'none'} "
                    f"among {list(table.columns)}")
            label_column = plausible[0]
        for _, row in table.iterrows():
            subject = str(row[subject_column])
            labels[subject] = row[label_column]
            if subject not in recordings:
                unmatched_rows.append(subject)

    unmatched_files = [s for s in recordings if s not in labels] if labels else []
    return CorpusIndex(recordings=recordings, labels=labels, table=table,
                       unmatched_files=sorted(unmatched_files),
                       unmatched_rows=sorted(unmatched_rows),
                       root=str(root), label_column=label_column)
