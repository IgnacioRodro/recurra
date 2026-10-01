import os

import pandas as pd

import recurra as rc
from recurra.io.export import CsvExporter, export_frame, export_recording


def test_export_frame_writes_sidecars(tmp_path):
    rc.set_config(output_dir=str(tmp_path))
    p = export_frame(pd.DataFrame({"a": [1, 2]}), "demo", provenance=[rc.Step("x")])
    base = os.path.splitext(p)[0]
    assert os.path.exists(p)
    assert os.path.exists(f"{base}_environment.csv")
    assert os.path.exists(f"{base}_provenance.csv")


def test_environment_stamp_has_version(tmp_path):
    rc.set_config(output_dir=str(tmp_path))
    p = export_frame(pd.DataFrame({"a": [1]}), "d")
    env = pd.read_csv(f"{os.path.splitext(p)[0]}_environment.csv")
    assert "recurra_version" in set(env["key"])


def test_export_recording_tables(tmp_path, rec):
    rc.set_config(output_dir=str(tmp_path))
    written = export_recording(rec, "r")
    assert "channels" in written and "quality" in written
    df = pd.read_csv(written["channels"])
    assert "digest" in df.columns and len(df) == rec.n_channels


def test_csv_exporter_accumulates(tmp_path):
    rc.set_config(output_dir=str(tmp_path))
    ex = CsvExporter("acc")
    for i in range(5):
        ex.add(subject=f"s{i}", metric="mvl", value=i * 0.1)
    assert len(ex) == 5
    df = pd.read_csv(ex.write(note="unit test"))
    assert list(df.columns) == ["subject", "metric", "value"]
