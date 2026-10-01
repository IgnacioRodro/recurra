"""Reading real recording files, and the corpus that goes with them.

These write genuine EDF files and read them back, rather than mocking the
reader. Skipped where pyedflib is absent, because the reader is an optional
extra and a skipped test is honest where a mocked one is not.
"""
import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest

import recurra as rc

pyedflib = pytest.importorskip("pyedflib")


def _write_edf(path, labels, rates, data, *, patient="sub-001",
               units=None, annotations=()):
    units = units or ["uV"] * len(labels)
    writer = pyedflib.EdfWriter(str(path), len(labels),
                                file_type=pyedflib.FILETYPE_EDFPLUS)
    writer.setPatientCode(patient)
    writer.setStartdatetime(datetime.datetime(2026, 3, 14, 9, 30, 0))
    writer.setSignalHeaders([
        {"label": l, "dimension": u, "sample_frequency": r,
         "physical_max": 1000.0, "physical_min": -1000.0,
         "digital_max": 32767, "digital_min": -32768,
         "transducer": "", "prefilter": ""}
        for l, u, r in zip(labels, units, rates, strict=False)])
    writer.writeSamples([np.asarray(d) for d in data])
    for onset, duration, text in annotations:
        writer.writeAnnotation(onset, duration, text)
    writer.close()
    return path


@pytest.fixture
def simple_edf(tmp_path):
    fs, n = 500.0, 5000
    t = np.arange(n) / fs
    data = [np.sin(2 * np.pi * 6 * t) * 50, np.sin(2 * np.pi * 10 * t) * 30]
    path = _write_edf(tmp_path / "rest.edf", ["Fz", "Cz"], [fs, fs], data,
                      annotations=[(4.0, 2.0, "eyes closed")])
    return path, fs, dict(zip(["Fz", "Cz"], data, strict=False))


# ------------------------------------------------------------- reading
def test_edf_round_trip(simple_edf):
    path, fs, data = simple_edf
    rec = rc.ingest(path)
    assert rec.names == ["Fz", "Cz"]
    assert rec.fs == fs
    for name, original in data.items():
        # EDF stores 16-bit integers over the declared physical range, so a
        # round trip is exact only to the quantisation step.
        step = 2000.0 / 65535
        np.testing.assert_allclose(rec.get(name), original, atol=step)


def test_units_come_from_the_file(tmp_path):
    """DD-88: the physical unit is a fact about the recording."""
    n = 2000
    path = _write_edf(tmp_path / "u.edf", ["Fz", "ECG"], [250.0, 250.0],
                      [np.zeros(n), np.zeros(n)], units=["uV", "mV"])
    rec = rc.ingest(path)
    assert rec.units == {"Fz": "uV", "ECG": "mV"}


def test_metadata_and_annotations_are_kept(simple_edf):
    path, _, _ = simple_edf
    rec = rc.ingest(path)
    assert rec.meta["subject"] == "sub-001"
    assert rec.meta["start_datetime"].startswith("2026-03-14")
    assert rec.meta["format"] == "edf"
    labels = [a["label"] for a in rec.meta["annotations"]]
    assert any("eyes closed" in l for l in labels)


def test_mixed_rates_are_refused_by_name(tmp_path):
    """DD-88: a real montage mixes rates, and the message must say which."""
    path = _write_edf(tmp_path / "m.edf", ["Fz", "Cz", "ECG"],
                      [500.0, 500.0, 250.0],
                      [np.zeros(5000), np.zeros(5000), np.zeros(2500)])
    with pytest.raises(rc.IngestError) as excinfo:
        rc.ingest(path)
    message = str(excinfo.value)
    assert "ECG" in message and "250" in message and "500" in message
    assert "channels=" in message and "exclude=" in message


def test_channel_selection_resolves_mixed_rates(tmp_path):
    path = _write_edf(tmp_path / "m.edf", ["Fz", "Cz", "ECG"],
                      [500.0, 500.0, 250.0],
                      [np.zeros(5000), np.zeros(5000), np.zeros(2500)])
    assert rc.ingest(path, channels=["Fz", "Cz"]).names == ["Fz", "Cz"]
    assert rc.ingest(path, exclude=["ECG"]).names == ["Fz", "Cz"]


def test_unknown_channel_is_reported_with_what_is_there(tmp_path):
    path = _write_edf(tmp_path / "m.edf", ["Fz", "Cz"], [500.0, 500.0],
                      [np.zeros(2000), np.zeros(2000)])
    with pytest.raises(rc.IngestError, match="Pz"):
        rc.ingest(path, channels=["Fz", "Pz"])


def test_an_edf_reaches_a_state_space(simple_edf):
    """The point of a reader: that the rest of the library accepts what it gives."""
    from recurra import statespace as sm

    path, _, _ = simple_edf
    rec = rc.analytic(rc.filterbank(rc.ingest(path),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    space = sm.pac_space(rec, smooth=8.0)
    assert space.dim == 3
    assert "ingest" in [s.name for s in space.provenance]


# -------------------------------------------------------------- corpus
@pytest.fixture
def study(tmp_path):
    rows = []
    for i in range(6):
        subject = f"sub-{i:03d}"
        folder = tmp_path / subject / "eeg"
        folder.mkdir(parents=True)
        _write_edf(folder / f"{subject}_task-rest_eeg.edf", ["Cz"], [250.0],
                   [np.random.default_rng(i).standard_normal(2500) * 30],
                   patient=subject)
        rows.append({"participant_id": subject,
                     "group": "control" if i < 4 else "patient"})
    rows.append({"participant_id": "sub-099", "group": "patient"})   # no file
    pd.DataFrame(rows).to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    return tmp_path


def test_corpus_finds_subjects_and_joins_labels(study):
    index = rc.read_corpus(study)
    assert len(index) == 6
    assert index.subjects[0] == "sub-000"
    assert index.label_column == "group"
    assert index.labelled()["sub-005"] == "patient"


def test_corpus_reports_the_join_rather_than_hiding_it(study):
    """DD-89: a row with no file is the commonest way a study goes wrong."""
    index = rc.read_corpus(study)
    assert index.unmatched_rows == ["sub-099"]
    assert not index.unmatched_files
    assert not index.complete
    assert "sub-099" in index.summary()


def test_corpus_frame_lists_everything(study):
    frame = rc.read_corpus(study).to_frame()
    assert len(frame) == 7                       # six files plus the orphan row
    assert frame["path"].isna().sum() == 1


def test_corpus_holds_paths_not_data(study):
    """A corpus of eight-minute recordings cannot be loaded at once."""
    index = rc.read_corpus(study)
    assert all(isinstance(p, pathlib.Path) for p in index.recordings.values())
    assert all(p.is_file() for p in index.recordings.values())


def test_empty_directory_is_reported(tmp_path):
    with pytest.raises(rc.IngestError, match="no file matching"):
        rc.read_corpus(tmp_path)


def test_two_files_for_one_subject_is_refused(tmp_path):
    for name in ("sub-001_run-1_eeg.edf", "sub-001_run-2_eeg.edf"):
        _write_edf(tmp_path / name, ["Cz"], [250.0],
                   [np.zeros(1000)], patient="sub-001")
    with pytest.raises(rc.IngestError, match="two files map to subject"):
        rc.read_corpus(tmp_path)


def test_ambiguous_label_column_asks(study):
    table = pd.read_csv(study / "participants.tsv", sep="\t")
    table["condition"] = table["group"]
    with pytest.raises(rc.ParameterError, match="name the label_column"):
        rc.read_corpus(study, participants=table)


def test_corpus_labels_feed_the_classifier(study):
    """The chain the whole module exists for: files on disk to a group label."""
    index = rc.read_corpus(study)
    labels = index.labelled()
    assert set(labels) == set(index.subjects)
    assert set(labels.values()) == {"control", "patient"}
